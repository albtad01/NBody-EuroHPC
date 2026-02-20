#!/usr/bin/env python3
"""
parse_energy_log.py

Parse energy.log produced by node-conso and generate a report.
Designed for measure_energy.py which records a fixed 5-second window.

Usage:
    python3 parse_energy_log.py <energy_log_path> [report_output_path] [murb_log_path]
"""

import re
import sys
import math
import statistics
from pathlib import Path
from collections import OrderedDict


PARTITION_CONFIGS = {
    'az4-n4090': {
        'gpu_channels': ['1.0', '1.1', '1.2', '1.3'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
        'has_gpu': True,
    },
    'az4-a7900': {
        'gpu_channels': ['1.0', '1.1', '1.2'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
        'has_gpu': True,
    },
    'iml-ia770': {
        'gpu_channels': ['0.2', '0.3', '1.0', '1.1'],
        'cpu_channel': '2.0',
        'ram_channel': None,
        'mb_channel': None,
        'has_gpu': True,
    },
    'az5-a890m': {
        'gpu_channels': None,
        'cpu_channel': '0.0',
        'ram_channel': None,
        'mb_channel': None,
        'has_gpu': False,
    },
}


def _finite(vals):
    return [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mean(vals):
    vals = _finite(vals)
    return statistics.mean(vals) if vals else 0.0


def _pvar(vals):
    vals = _finite(vals)
    return statistics.pvariance(vals) if len(vals) > 1 else 0.0


def _min(vals):
    vals = _finite(vals)
    return min(vals) if vals else 0.0


def _max(vals):
    vals = _finite(vals)
    return max(vals) if vals else 0.0


def _fmt(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NaN"
    return f"{val:.3f}"


def parse_energy_log(energy_file, report_file=None, murb_log_file=None):
    energy_path = Path(energy_file)
    if not energy_path.exists():
        raise FileNotFoundError(f"Energy file {energy_file} not found")

    if report_file is None:
        report_file = energy_path.parent / "energy_report.txt"

    # FPS extraction (optional)
    fps = None
    if murb_log_file is None:
        cand = energy_path.parent / "murb.log"
        if cand.exists():
            murb_log_file = cand

    if murb_log_file:
        try:
            with open(murb_log_file, "r") as f:
                for line in f:
                    if "Entire simulation took" in line and "FPS" in line:
                        m = re.search(r"\(([0-9.]+)\s+FPS\)", line)
                        if m:
                            fps = float(m.group(1))
                            break
        except Exception:
            pass

    # Parse header + samples
    partition = None
    samples_by_ts = OrderedDict()  # ts -> {probe_chan -> energy_raw}

    pat = re.compile(
        r'^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+\S+\s+(?:[0-9.]+dC\s+)?([0-9.]+)\s*V\s+([\-0-9.]+)\s*A\s+([0-9.]+)\s*J'
    )

    with open(energy_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# PARTITION:"):
                partition = line.split(":", 1)[1].strip()
                continue

            m = pat.match(line)
            if not m:
                continue

            ts = int(m.group(1))
            probe = m.group(2)
            energy_raw = float(m.group(5))

            if ts not in samples_by_ts:
                samples_by_ts[ts] = {}
            samples_by_ts[ts][probe] = energy_raw

    if not samples_by_ts:
        raise ValueError(f"No valid samples parsed from {energy_file}")

    if partition is None:
        partition = "az4-n4090"
    if partition not in PARTITION_CONFIGS:
        raise ValueError(f"Unknown partition: {partition}")

    cfg = PARTITION_CONFIGS[partition]
    GPU_keys = cfg["gpu_channels"] or []
    CPU_key = cfg["cpu_channel"]
    RAM_key = cfg["ram_channel"]
    MB_key = cfg["mb_channel"]
    has_gpu = cfg["has_gpu"]

    # Fixed window: measure_energy does exactly one node-conso -t 5
    total_time = 5.0

    ts_list = list(samples_by_ts.keys())
    ts_start, ts_end = ts_list[0], ts_list[-1]
    if ts_end <= ts_start:
        raise ValueError("Non-increasing timestamps in energy log")

    # Convert ts-units -> seconds using the known 5s window
    sec_per_ts_unit = total_time / float(ts_end - ts_start)

    # Integrate between successive timestamps with per-step dt
    last_energy = {}
    last_ts = None

    gpuP_series, cpuP_series, ramP_series, mbP_series, totP_series = [], [], [], [], []
    stepE_series = []  # total energy per step (J)

    # total energies (J)
    E_gpu = 0.0
    E_cpu = 0.0
    E_ram = 0.0
    E_mb = 0.0

    for ts, probe_map in samples_by_ts.items():
        if last_ts is None:
            for p, e in probe_map.items():
                last_energy[p] = e
            last_ts = ts
            continue

        dt = (ts - last_ts) * sec_per_ts_unit
        last_ts = ts
        if dt <= 0:
            continue

        # deltas only for probes present now; missing probes => delta 0 for this step
        deltaE = {}
        for p, e_now in probe_map.items():
            e_prev = last_energy.get(p, e_now)
            d = e_now - e_prev
            if d < 0:
                d = 0.0  # clamp negative resets/glitches
            deltaE[p] = d
            last_energy[p] = e_now

        d_gpu = sum(deltaE.get(k, 0.0) for k in GPU_keys) if GPU_keys else math.nan
        d_cpu = deltaE.get(CPU_key, 0.0) if CPU_key else 0.0
        d_ram = deltaE.get(RAM_key, 0.0) if RAM_key else 0.0
        d_mb  = deltaE.get(MB_key, 0.0)  if MB_key  else 0.0

        if GPU_keys:
            gpuP = d_gpu / dt
        else:
            gpuP = math.nan
        cpuP = d_cpu / dt if CPU_key else 0.0
        ramP = d_ram / dt if RAM_key else 0.0
        mbP  = d_mb  / dt if MB_key  else 0.0

        totP = (cpuP + ramP + mbP) if (not has_gpu or (isinstance(gpuP, float) and math.isnan(gpuP))) else (gpuP + cpuP + ramP + mbP)

        gpuP_series.append(gpuP)
        cpuP_series.append(cpuP)
        ramP_series.append(ramP)
        mbP_series.append(mbP)
        totP_series.append(totP)

        stepE = (0.0 if (not has_gpu or (isinstance(d_gpu, float) and math.isnan(d_gpu))) else d_gpu) + d_cpu + d_ram + d_mb
        stepE_series.append(stepE)

        # integrate energy totals
        if GPU_keys and not (isinstance(d_gpu, float) and math.isnan(d_gpu)):
            E_gpu += d_gpu
        E_cpu += d_cpu
        E_ram += d_ram
        E_mb  += d_mb

    E_total = (E_cpu + E_ram + E_mb) if (not has_gpu or not GPU_keys) else (E_gpu + E_cpu + E_ram + E_mb)

    mean_gpu = _mean(gpuP_series)
    mean_cpu = _mean(cpuP_series)
    mean_ram = _mean(ramP_series)
    mean_mb  = _mean(mbP_series)
    mean_tot = _mean(totP_series)

    var_gpu = _pvar(gpuP_series)
    var_cpu = _pvar(cpuP_series)
    var_ram = _pvar(ramP_series)
    var_mb  = _pvar(mbP_series)
    var_tot = _pvar(totP_series)

    min_tot = _min(totP_series)
    max_tot = _max(totP_series)

    steps = len(totP_series)
    steps_per_sec = steps / total_time if total_time > 0 else 0.0

    # Efficiency metrics
    fps_per_watt = (fps / mean_tot) if (fps is not None and mean_tot > 0) else None
    j_per_frame = (mean_tot / fps) if (fps is not None and fps > 0) else None  # since W/FPS = J/frame

    # Report
    report_lines = []
    report_lines.append(f"SUMMARY (energy log: {energy_file})")
    report_lines.append(f"Partition: {partition}")
    report_lines.append(f"Total timestamps parsed: {len(samples_by_ts)}")
    report_lines.append(f"Total steps integrated: {steps}")
    report_lines.append(f"Measured window duration: {total_time:.3f} seconds")
    report_lines.append(f"Seconds per ts-unit (estimated): {sec_per_ts_unit:.9f} s")
    report_lines.append("")
    report_lines.append("MEAN WATTAGE (W):")
    report_lines.append(f"  GPU   : {_fmt(mean_gpu)} W")
    report_lines.append(f"  CPU   : {mean_cpu:.3f} W")
    report_lines.append(f"  RAM   : {mean_ram:.3f} W")
    report_lines.append(f"  MBOARD: {mean_mb:.3f} W")
    report_lines.append(f"  TOTAL : {mean_tot:.3f} W")
    report_lines.append("")
    report_lines.append("POWER VARIANCE (W^2):")
    report_lines.append(f"  GPU   : {var_gpu:.6f}")
    report_lines.append(f"  CPU   : {var_cpu:.6f}")
    report_lines.append(f"  RAM   : {var_ram:.6f}")
    report_lines.append(f"  MBOARD: {var_mb:.6f}")
    report_lines.append(f"  TOTAL : {var_tot:.6f}")
    report_lines.append("")
    report_lines.append(f"ENERGY CONSUMED over {total_time:.3f} s:")
    report_lines.append(f"  GPU   : {_fmt(E_gpu if GPU_keys else math.nan)} J")
    report_lines.append(f"  CPU   : {E_cpu:.6f} J")
    report_lines.append(f"  RAM   : {E_ram:.6f} J")
    report_lines.append(f"  MBOARD: {E_mb:.6f} J")
    report_lines.append(f"  TOTAL : {E_total:.6f} J")
    report_lines.append("")
    report_lines.append("ADDITIONAL METRICS:")
    report_lines.append(f"  Steps per second (approx): {steps_per_sec:.3f}")
    report_lines.append(f"  TOTAL POWER min/max: {min_tot:.3f}/{max_tot:.3f} W")
    report_lines.append(f"  Energy variance (per-step): {_pvar(stepE_series):.6f} (J^2)")
    if fps is not None:
        if fps_per_watt is not None:
            report_lines.append(f"  FPS/Watt (avg): {fps_per_watt:.6f} FPS/W")
        report_lines.append(f"  Joules per frame (avg): {j_per_frame:.6f} J/frame" if j_per_frame is not None else "  Joules per frame (avg): N/A")
        report_lines.append(f"  FPS (from murb.log): {fps:.5f}")

    Path(report_file).write_text("\n".join(report_lines))

    return {
        "partition": partition,
        "timestamps": len(samples_by_ts),
        "steps": steps,
        "total_time": total_time,
        "sec_per_ts_unit": sec_per_ts_unit,
        "mean_wattage": {"GPU": mean_gpu, "CPU": mean_cpu, "RAM": mean_ram, "MBOARD": mean_mb, "TOTAL": mean_tot},
        "power_variance": {"GPU": var_gpu, "CPU": var_cpu, "RAM": var_ram, "MBOARD": var_mb, "TOTAL": var_tot},
        "energy_consumed": {"GPU": (E_gpu if GPU_keys else math.nan), "CPU": E_cpu, "RAM": E_ram, "MBOARD": E_mb, "TOTAL": E_total},
        "fps": fps,
        "fps_per_watt": fps_per_watt,
        "joules_per_frame": j_per_frame,
        "report_file": str(report_file),
        "report_text": "\n".join(report_lines),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    energy_log = sys.argv[1]
    report_out = sys.argv[2] if len(sys.argv) > 2 else None
    murb_log = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        result = parse_energy_log(energy_log, report_out, murb_log)
        print(result["report_text"])
        print(f"\nReport written to {result['report_file']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
