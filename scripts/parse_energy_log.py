#!/usr/bin/env python3
"""
parse_energy_log.py

Parse energy.log file produced by node-conso and generate a human-readable report.
Can be used standalone or called from measure_energy.py.

Usage:
    python3 parse_energy_log.py <energy_log_path> [report_output_path]
    
    If report_output_path is not provided, writes to energy_report.txt in same directory.
"""

import re
import sys
import statistics
import math
from collections import OrderedDict
from pathlib import Path


# Partition configurations
PARTITION_CONFIGS = {
    'az4-n4090': {
        'name': 'az4-n4090',
        'gpu_channels': ['1.0', '1.1', '1.2', '1.3'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
        'has_gpu': True,
    },
    'az4-a7900': {
        'name': 'az4-a7900',
        'gpu_channels': ['1.0', '1.1', '1.2'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
        'has_gpu': True,
    },
    'iml-ia770': {
        'name': 'iml-ia770',
        'gpu_channels': ['0.2', '0.3', '1.0', '1.1'],
        'cpu_channel': '2.0',
        'ram_channel': None,
        'mb_channel': None,
        'has_gpu': True,
    },
    'az5-a890m': {
        'name': 'az5-a890m',
        'gpu_channels': None,
        'cpu_channel': '0.0',
        'ram_channel': None,
        'mb_channel': None,
        'has_gpu': False,
    },
}


def parse_energy_log(energy_file, report_file=None, murb_log_file=None):
    """
    Parse energy.log file and generate report.
    
    Args:
        energy_file: path to energy.log file
        report_file: path to output report file (if None, defaults to energy_report.txt)
        murb_log_file: path to murb.log file for extracting FPS (optional)
        
    Returns:
        dict with parsed data and computed statistics
    """
    energy_path = Path(energy_file)
    if not energy_path.exists():
        raise FileNotFoundError(f"Energy file {energy_file} not found")
    
    if report_file is None:
        report_file = energy_path.parent / "energy_report.txt"
    
    # Try to find and parse murb.log if not provided
    fps = None
    if murb_log_file is None:
        # Try to find murb.log in the same directory as energy.log
        potential_murb_log = energy_path.parent / "murb.log"
        if potential_murb_log.exists():
            murb_log_file = potential_murb_log
    
    # Extract FPS from murb.log
    if murb_log_file:
        murb_log_path = Path(murb_log_file)
        if murb_log_path.exists():
            try:
                with open(murb_log_path) as f:
                    for line in f:
                        # Line format: "Entire simulation took 14243.4 ms (7.02077 FPS)"
                        if "Entire simulation took" in line and "FPS" in line:
                            # Extract FPS value using regex
                            fps_match = re.search(r'\(([0-9.]+)\s+FPS\)', line)
                            if fps_match:
                                fps = float(fps_match.group(1))
                                break
            except Exception as e:
                # If we can't read murb.log, just continue without FPS
                pass
    
    # Parse header for wallclock times and partition
    start_wall = None
    end_wall = None
    partition = None
    samples_by_ts = OrderedDict()  # ts -> { probe_chan: energy_raw }
    
    # Regex patterns for parsing node-conso output
    # Formats:
    # 1) timestamp probe_chan hex voltage current energy
    #    e.g.: 5237  1.0 0xff  12.220V  0.6295A  16.480J
    # 2) with optional temperature (dC) prefix for some probes
    #    e.g.: 5235  2.0 0xff 27.6dC 18.936V 1.5445A 142.729J
    # The (?:[0-9.]+dC\s+)? makes temperature optional and non-capturing
    pat = re.compile(r'^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+\S+\s+(?:[0-9.]+dC\s+)?([0-9.]+)\s*V\s+([\-0-9.]+)\s*A\s+([0-9.]+)\s*J')
    pat2 = re.compile(r'^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+\S+\s*(?:[0-9.]+dC\s+)?([0-9.]+)\s*V\s*([\-0-9.]+)\s*A\s*([0-9.]+)\s*J')
    
    with open(energy_file) as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('# START_WALLCLOCK:'):
                start_wall = float(line.split(':', 1)[1].strip())
                continue
            if line.startswith('# END_WALLCLOCK:'):
                end_wall = float(line.split(':', 1)[1].strip())
                continue
            if line.startswith('# PARTITION:'):
                partition = line.split(':', 1)[1].strip()
                continue
            
            m = pat.match(line)
            if not m:
                m = pat2.match(line)
            if not m:
                continue
            
            ts = int(m.group(1))
            probe_chan = m.group(2)  # e.g., '0.4'
            volts = float(m.group(3))
            amps = float(m.group(4))
            energy_raw = float(m.group(5))  # raw accumulated energy in Joules
            
            if ts not in samples_by_ts:
                samples_by_ts[ts] = {}
            samples_by_ts[ts][probe_chan] = energy_raw
    
    if not samples_by_ts:
        raise ValueError(f"No valid samples parsed from {energy_file}")
    
    # Determine partition config
    if partition is None:
        partition = 'az4-n4090'  # Default fallback
    
    if partition not in PARTITION_CONFIGS:
        raise ValueError(f"Unknown partition: {partition}")
    
    config = PARTITION_CONFIGS[partition]
    GPU_keys = config['gpu_channels'] if config['gpu_channels'] else []
    CPU_key = config['cpu_channel']
    RAM_key = config['ram_channel']
    MB_key = config['mb_channel']
    has_gpu = config['has_gpu']
    
    n = len(samples_by_ts)
    
    # Compute dt: 5 seconds measurement interval divided by number of samples
    dt = 5.0 / n
    
    # First pass: collect energy values per probe_chan (as lists in timestamp order)
    energy_by_probe = {}  # probe_chan -> [energy_raw_0, energy_raw_1, ...]
    for ts, d in samples_by_ts.items():
        for probe_chan, energy_raw in d.items():
            if probe_chan not in energy_by_probe:
                energy_by_probe[probe_chan] = []
            energy_by_probe[probe_chan].append(energy_raw)
    
    # Convert raw energy to delta energy and then to power
    power_by_probe = {}  # probe_chan -> [power_0, power_1, ...]
    for probe_chan, energy_raws in energy_by_probe.items():
        delta_energies = [0.0]  # First timestamp has zero consumption
        for i in range(1, len(energy_raws)):
            delta_e = energy_raws[i] - energy_raws[i-1]
            delta_energies.append(delta_e)
        # Convert to power (energy / dt)
        power_by_probe[probe_chan] = [e / dt for e in delta_energies]
    
    # Helper to safely get power value by probe and index
    def get_power(probe_chan, idx):
        arr = power_by_probe.get(probe_chan, [])
        return arr[idx] if idx < len(arr) else 0.0
    
    # Aggregate by component
    gpu_vals = []
    cpu_vals = []
    ram_vals = []
    mb_vals = []
    total_vals = []
    
    for i in range(n):
        gpu = sum(get_power(k, i) for k in GPU_keys) if GPU_keys else math.nan
        cpu = get_power(CPU_key, i) if CPU_key else 0.0
        ram = get_power(RAM_key, i) if RAM_key else 0.0
        mb = get_power(MB_key, i) if MB_key else 0.0
        
        # Total only counts non-NaN values
        if not has_gpu or math.isnan(gpu):
            total = cpu + ram + mb
        else:
            total = gpu + cpu + ram + mb
        
        gpu_vals.append(gpu)
        cpu_vals.append(cpu)
        ram_vals.append(ram)
        mb_vals.append(mb)
        total_vals.append(total)
    
    # Helper functions
    def avg(arr):
        vals = [x for x in arr if not (isinstance(x, float) and math.isnan(x))]
        return statistics.mean(vals) if vals else 0.0
    
    def pvar(arr):
        vals = [x for x in arr if not (isinstance(x, float) and math.isnan(x))]
        return statistics.pvariance(vals) if len(vals) > 1 else 0.0
    
    def pmin(arr):
        vals = [x for x in arr if not (isinstance(x, float) and math.isnan(x))]
        return min(vals) if vals else 0.0
    
    def pmax(arr):
        vals = [x for x in arr if not (isinstance(x, float) and math.isnan(x))]
        return max(vals) if vals else 0.0
    
    def format_val(val):
        """Format value, handling NaN"""
        if isinstance(val, float) and math.isnan(val):
            return "NaN"
        return f"{val:.3f}"
    
    # Compute duration from wallclock times or estimate
    total_time = 5.0  # Standard measurement interval
    
    # Energy per timestamp (using delta energy already computed)
    energy_per_ts = [p * dt if not (isinstance(p, float) and math.isnan(p)) else 0.0 for p in total_vals]
    
    # Calculate total energy consumed per component
    total_energy_gpu = sum(get_power(k, i) * dt for k in GPU_keys for i in range(n)) if GPU_keys else math.nan
    total_energy_cpu = sum(get_power(CPU_key, i) * dt for i in range(n)) if CPU_key else 0.0
    total_energy_ram = sum(get_power(RAM_key, i) * dt for i in range(n)) if RAM_key else 0.0
    total_energy_mb = sum(get_power(MB_key, i) * dt for i in range(n)) if MB_key else 0.0
    total_energy_all = sum(energy_per_ts)
    
    # Build report
    report_lines = []
    report_lines.append(f'SUMMARY (energy log: {energy_file})')
    report_lines.append(f'Partition: {partition}')
    report_lines.append(f'Total samples (timestamps): {n}')
    report_lines.append(f'Measured wallclock duration: {total_time:.3f} seconds')
    report_lines.append(f'Time per sample (dt): {dt:.6f} seconds')
    report_lines.append('')
    report_lines.append('MEAN WATTAGE (W):')
    report_lines.append(f'  GPU   : {format_val(avg(gpu_vals))} W')
    report_lines.append(f'  CPU   : {avg(cpu_vals):.3f} W')
    report_lines.append(f'  RAM   : {avg(ram_vals):.3f} W')
    report_lines.append(f'  MBOARD: {avg(mb_vals):.3f} W')
    report_lines.append(f'  TOTAL : {avg(total_vals):.3f} W')
    report_lines.append('')
    report_lines.append('POWER VARIANCE (W^2):')
    report_lines.append(f'  GPU   : {pvar(gpu_vals):.6f}')
    report_lines.append(f'  CPU   : {pvar(cpu_vals):.6f}')
    report_lines.append(f'  RAM   : {pvar(ram_vals):.6f}')
    report_lines.append(f'  MBOARD: {pvar(mb_vals):.6f}')
    report_lines.append(f'  TOTAL : {pvar(total_vals):.6f}')
    report_lines.append('')
    report_lines.append(f'ENERGY CONSUMED over {total_time:.3f} s:')
    report_lines.append(f'  GPU   : {format_val(total_energy_gpu)} J')
    report_lines.append(f'  CPU   : {total_energy_cpu:.6f} J')
    report_lines.append(f'  RAM   : {total_energy_ram:.6f} J')
    report_lines.append(f'  MBOARD: {total_energy_mb:.6f} J')
    report_lines.append(f'  TOTAL : {total_energy_all:.6f} J')
    report_lines.append('')
    report_lines.append('ADDITIONAL METRICS:')
    report_lines.append(f'  Samples per second (approx): {n/total_time if total_time > 0 else 0.0:.3f}')
    report_lines.append(f'  TOTAL POWER min/max: {pmin(total_vals):.3f}/{pmax(total_vals):.3f} W')
    report_lines.append(f'  Energy variance (per-timestamp): {pvar(energy_per_ts):.6f} (J^2)')
    if fps is not None:
        mean_power = avg(total_vals)
        if mean_power > 0:
            fps_per_watt = fps / mean_power
            report_lines.append(f'  FPS/Watt (avg): {fps_per_watt:.6f} FPS/W')
            report_lines.append(f'  FPS (from murb.log): {fps:.5f}')
        else:
            report_lines.append(f'  FPS/Watt (avg): N/A (mean power = 0)')
            report_lines.append(f'  FPS (from murb.log): {fps:.5f}')
    
    # Write report
    with open(report_file, 'w') as out:
        out.write('\n'.join(report_lines))
    
    # Return statistics
    return {
        'n_samples': n,
        'total_time': total_time,
        'dt': dt,
        'partition': partition,
        'mean_wattage': {
            'GPU': avg(gpu_vals),
            'CPU': avg(cpu_vals),
            'RAM': avg(ram_vals),
            'MBOARD': avg(mb_vals),
            'TOTAL': avg(total_vals),
        },
        'power_variance': {
            'GPU': pvar(gpu_vals),
            'CPU': pvar(cpu_vals),
            'RAM': pvar(ram_vals),
            'MBOARD': pvar(mb_vals),
            'TOTAL': pvar(total_vals),
        },
        'energy_consumed': {
            'GPU': total_energy_gpu,
            'CPU': total_energy_cpu,
            'RAM': total_energy_ram,
            'MBOARD': total_energy_mb,
            'TOTAL': total_energy_all,
            'variance_per_ts': pvar(energy_per_ts),
        },
        'fps': fps,
        'fps_per_watt': (fps / avg(total_vals)) if (fps is not None and avg(total_vals) > 0) else None,
        'report_file': str(report_file),
        'report_text': '\n'.join(report_lines),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    energy_log = sys.argv[1]
    report_out = sys.argv[2] if len(sys.argv) > 2 else None
    murb_log = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result = parse_energy_log(energy_log, report_out, murb_log)
        print(result['report_text'])
        print(f"\nReport written to {result['report_file']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
