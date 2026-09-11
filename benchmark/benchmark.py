#!/usr/bin/env python3
"""Reproducible, additive benchmark runner and aggregator for NBody-EuroHPC."""


import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import resource
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "matrix.json"
DEFAULT_RESULTS_ROOT = Path("/leonardo_work/EUHPC_TDEMO_26/benchmark-results")
SUMMARY_RE = re.compile(r"^completed_iterations=(?P<completed>\d+)\s+(?P<rest>.*)$")
FIELD_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_output(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def git_info(repo):
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], stdout=subprocess.PIPE, universal_newlines=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        stdout=subprocess.PIPE, universal_newlines=True, check=True,
    ).stdout
    remote = command_output(["git", "-C", str(repo), "remote", "get-url", "origin"])
    return {"sha": sha, "dirty": bool(status.strip()), "status_porcelain": status.splitlines(), "origin": remote}


def is_within(child, parent):
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def init_campaign(args):
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    git = git_info(repo)
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to initialize a production campaign from a dirty worktree")
    results_root = args.results_root.resolve()
    if is_within(results_root, repo):
        raise RuntimeError("benchmark results must be outside the Git worktree")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = results_root / git["sha"] / stamp
    campaign.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "gpu_samples", "runs", "plots"):
        (campaign / name).mkdir()
    metadata = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repository": str(repo),
        "git": git,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.splitlines()[0],
        "config_source": str(config_path),
        "matrix": config,
        "measurement_policy": {
            "primary_statistic": "median of independent process invocations",
            "visualization": "off (--nv)",
            "trajectory_recording": "off (no --record argument)",
            "gpu_energy_timed_region": "not reported: external samples do not delimit compute_ms",
            "gpu_energy_process_window": "estimated from nvidia-smi board power over invocation wall time"
        }
    }
    atomic_json(campaign / "metadata.json", metadata)
    print(campaign)
    return 0


def parse_summary(stdout):
    matches = []
    for line in stdout.splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match:
            fields = dict(FIELD_RE.findall(match.group("rest")))
            fields["completed_iterations"] = match.group("completed")
            matches.append(fields)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one timing summary line, found {len(matches)}")
    fields = matches[0]
    required = (
        "completed_iterations", "compute_ms", "average_ms_per_iteration", "loop_wall_ms",
        "interactions_per_second", "estimated_GFLOP_per_second",
    )
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError("timing summary is missing: " + ", ".join(missing))
    parsed = {"completed_iterations": int(fields["completed_iterations"])}
    for name in required[1:]:
        value = float(fields[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}: {fields[name]}")
        parsed[name] = value
    parsed["simulation_steps_per_second"] = (
        1000.0 / float(parsed["average_ms_per_iteration"])
        if float(parsed["average_ms_per_iteration"]) > 0 else 0.0
    )
    return parsed


def start_gpu_sampler(path, devices, interval_ms):
    if not devices or not command_output(["nvidia-smi", "--help-query-gpu"]):
        return None
    query = "timestamp,uuid,name,memory.used,utilization.gpu,power.draw"
    handle = path.open("w", encoding="utf-8")
    handle.write("timestamp,gpu_uuid,gpu_name,memory_used_mib,utilization_percent,power_w\n")
    handle.flush()
    command = [
        "nvidia-smi", f"--id={devices}", f"--query-gpu={query}",
        "--format=csv,noheader,nounits", f"--loop-ms={interval_ms}",
    ]
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.DEVNULL, universal_newlines=True)
    except FileNotFoundError:
        handle.close()
        return None
    process._benchmark_handle = handle  # type: ignore[attr-defined]
    return process


def stop_gpu_sampler(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    process._benchmark_handle.close()  # type: ignore[attr-defined]


def parse_gpu_samples(path, wall_seconds, gpu_count):
    empty = {
        "gpu_sample_count": 0, "peak_gpu_memory_mib": None,
        "mean_gpu_utilization_percent": None, "peak_gpu_utilization_percent": None,
        "mean_gpu_power_w": None, "peak_gpu_power_w": None,
        "gpu_energy_process_window_j": None, "gpu_energy_timed_region_j": None,
        "gpu_energy_scope": "unavailable", "gpu_telemetry_trustworthy": False,
    }
    if not path.exists():
        return empty
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "uuid": row["gpu_uuid"].strip(),
                    "memory": float(row["memory_used_mib"]),
                    "util": float(row["utilization_percent"]),
                    "power": float(row["power_w"]),
                })
    except (KeyError, ValueError, csv.Error):
        return empty
    if not rows:
        return empty
    by_gpu = {}
    for row in rows:
        by_gpu.setdefault(row["uuid"], []).append(row)
    memory_delta_sum = 0.0
    mean_power_sum = 0.0
    peak_power_sum = 0.0
    for samples in by_gpu.values():
        baseline = samples[0]["memory"]
        memory_delta_sum += max(max(sample["memory"] - baseline, 0.0) for sample in samples)
        mean_power_sum += statistics.mean(sample["power"] for sample in samples)
        peak_power_sum += max(sample["power"] for sample in samples)
    enough = len(rows) >= max(3, gpu_count * 3) and len(by_gpu) == gpu_count
    return {
        "gpu_sample_count": len(rows),
        "peak_gpu_memory_mib": memory_delta_sum,
        "mean_gpu_utilization_percent": statistics.mean(row["util"] for row in rows),
        "peak_gpu_utilization_percent": max(row["util"] for row in rows),
        "mean_gpu_power_w": mean_power_sum,
        "peak_gpu_power_w": peak_power_sum,
        "gpu_energy_process_window_j": mean_power_sum * wall_seconds if enough else None,
        "gpu_energy_timed_region_j": None,
        "gpu_energy_scope": "whole_invocation_not_timed_region" if enough else "insufficient_samples",
        "gpu_telemetry_trustworthy": enough,
    }


def version_metadata():
    return {
        "compiler_version": command_output(["g++", "--version"]),
        "cuda_version": command_output(["nvcc", "--version"]),
        "mpi_version": command_output(["mpirun", "--version"]),
        "gpu_inventory": command_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]),
    }


def launcher_for(backend, cpu_count, local):
    if local:
        return [], ""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("non-smoke benchmark invocations must run inside a Slurm allocation")
    common = ["srun", "--exclusive", "--nodes=1", "--cpu-bind=cores"]
    if backend in ("cpu+naive", "cpu+omp"):
        return common + ["--ntasks=1", f"--cpus-per-task={cpu_count}"], ""
    if backend == "gpu+tile+full":
        return common + ["--ntasks=1", "--cpus-per-task=8", "--gpus-per-task=1", "--gpu-bind=map_gpu:0"], "0"
    if backend == "gpu+multinode":
        return common + ["--ntasks=4", "--ntasks-per-node=4", "--cpus-per-task=8", "--gpus-per-task=1", "--gpu-bind=map_gpu:0,1,2,3"], "0,1,2,3"
    raise ValueError(f"backend is not in the primary matrix: {backend}")


def validate_binary(binary, sha, backend, source_dirty):
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"missing executable: {binary}")
    if not (binary.parent.parent / "murb-build.ready").is_file():
        raise RuntimeError(f"missing successful build stamp beside {binary}")
    version = subprocess.run([str(binary), "--version"], stdout=subprocess.PIPE,
                             universal_newlines=True, check=True).stdout.strip()
    expected_dirty = "1" if source_dirty else "0"
    if not version.startswith(f"murb revision={sha} dirty={expected_dirty} "):
        raise RuntimeError(f"executable/source identity mismatch: {version}")
    if backend.startswith("gpu+") and " cuda=1 " not in f" {version} ":
        raise RuntimeError("CUDA-enabled executable required")
    if backend == "gpu+multinode" and " mpi=1" not in version:
        raise RuntimeError("MPI-enabled executable required")
    return version


def run_one(
    campaign: Path, repo: Path, binary: Path, backend: str, n: int, iterations: int,
    warmup: int, repetition: int, timeout: int, cpu_count: int, gpu_count: int,
    scheme: str, timestep: float, sampling_ms: int, local: bool, allow_dirty: bool,
):
    campaign_meta = load_json(campaign / "metadata.json")
    git = git_info(repo)
    if git["sha"] != campaign_meta["git"]["sha"]:
        raise RuntimeError("campaign Git SHA does not match the current worktree")
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("production benchmark requires a clean worktree")
    version = validate_binary(binary, git["sha"], backend, bool(git["dirty"]))
    launcher, devices = launcher_for(backend, cpu_count, local)
    time_prefix = ["/usr/bin/time", "-v"] if Path("/usr/bin/time").exists() else []
    simulation = [
        str(binary), "-n", str(n), "-i", str(iterations), "--warmup", str(warmup),
        "--im", backend, "--scheme", scheme, "--nv", "--gf", "--dt", str(timestep),
    ]
    command = launcher + time_prefix + simulation
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_id = f"{backend.replace('+', '-')}_n{n}_r{repetition}_j{job_id}_{uuid.uuid4().hex[:10]}"
    run_dir = campaign / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    gpu_path = campaign / "gpu_samples" / f"{run_id}.csv"
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(cpu_count if backend == "cpu+omp" else 1),
        "OMP_DYNAMIC": "FALSE", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close",
    })
    sampler = start_gpu_sampler(gpu_path, devices, sampling_ms) if gpu_count else None
    if sampler is not None:
        time.sleep(max(0.05, sampling_ms / 1000.0 * 1.25))
    before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            env=env, timeout=timeout, check=False, start_new_session=True,
        )
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        stderr += f"\nbenchmark timeout after {timeout} seconds\n"
    finally:
        wall_seconds = time.monotonic() - started
        stop_gpu_sampler(sampler)
    after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    atomic_text(stdout_path, stdout)
    atomic_text(stderr_path, stderr)
    rss_values = [int(value) for value in RSS_RE.findall(stderr)]
    record = {
        "schema_version": 1, "run_id": run_id, "timestamp_utc": utc_now(),
        "git_sha": git["sha"], "git_dirty": git["dirty"], "slurm_job_id": job_id,
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "hostname": socket.gethostname(),
        "backend": backend, "n": n, "iterations": iterations, "warmup_iterations": warmup,
        "repetition": repetition, "scheme": scheme, "dt": timestep,
        "cpu_count": cpu_count if backend != "cpu+naive" else 1,
        "allocated_cpu_count": cpu_count,
        "cpus_per_task": 8 if backend.startswith("gpu+") else cpu_count,
        "mpi_rank_count": 4 if backend == "gpu+multinode" else 1,
        "gpu_count": gpu_count, "executable": str(binary), "executable_version": version,
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"), "loaded_modules": os.environ.get("LOADEDMODULES"),
        "command": command, "command_shell_escaped": " ".join(shlex.quote(part) for part in command),
        "returncode": returncode, "timed_out": timed_out, "status": "failed",
        "wall_clock_seconds": wall_seconds,
        "peak_host_memory_kib": max(rss_values) if rss_values else None,
        "peak_host_memory_source": "GNU time per-task maximum" if rss_values else "unavailable",
        "wrapper_children_maxrss_kib_delta": max(0, after_rss - before_rss),
        "stdout_path": str(stdout_path.relative_to(campaign)),
        "stderr_path": str(stderr_path.relative_to(campaign)),
        "gpu_samples_path": str(gpu_path.relative_to(campaign)) if gpu_path.exists() else None,
        **version_metadata(),
        **(parse_gpu_samples(gpu_path, wall_seconds, gpu_count)
           if gpu_count else parse_gpu_samples(Path("/nonexistent"), wall_seconds, 0)),
    }
    try:
        timing = parse_summary(stdout)
        if int(timing["completed_iterations"]) != iterations:
            raise ValueError("completed iteration count does not match request")
        record.update(timing)
        if returncode == 0:
            record["status"] = "ok"
        else:
            record["parse_error"] = f"timing parsed but process returned {returncode}"
    except ValueError as error:
        record["parse_error"] = str(error)
    atomic_json(run_dir / "record.json", record)
    atomic_json(campaign / "raw" / f"{run_id}.json", record)
    print(f"{record['status']} backend={backend} n={n} repetition={repetition} run_id={run_id}")
    return record


def selected_n_values(spec, override):
    configured = [int(value) for value in spec["n_values"]]
    if not override:
        return configured
    requested = [int(value) for value in override.split(",")]
    unsupported = sorted(set(requested) - set(configured))
    if unsupported:
        raise ValueError(f"N values outside the backend-specific matrix: {unsupported}")
    return requested


def run_group(args):
    campaign = args.campaign.resolve()
    metadata = load_json(campaign / "metadata.json")
    matrix = metadata["matrix"]
    defaults = matrix["defaults"]
    repetitions = args.repetitions or int(os.environ.get("MURB_REPETITIONS", defaults["repetitions"]))
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    n_override = args.n_list or os.environ.get("MURB_N_LIST")
    backends = [name for name, spec in matrix["primary"].items() if spec["group"] == args.group]
    if args.backend:
        if args.backend not in backends:
            raise ValueError(f"{args.backend} is not a primary backend in group {args.group}")
        backends = [args.backend]
    if args.smoke:
        backends = [args.backend or ("cpu+naive" if args.group == "cpu" else "gpu+tile+full")]
        repetitions = 1
    plan = {}
    for backend in backends:
        spec = matrix["primary"][backend]
        plan[backend] = [32] if args.smoke else selected_n_values(spec, n_override)
    ordered_n = sorted({n for values in plan.values() for n in values})
    failures = 0
    for repetition in range(1, repetitions + 1):
        for n in ordered_n:
            for backend in backends:
                if n not in plan[backend]:
                    continue
                spec = matrix["primary"][backend]
                iterations = 2 if args.smoke else int(spec["iterations"][str(n)])
                warmup = 1 if args.smoke else int(defaults["warmup_iterations"])
                cpu_spec = spec["cpu_count"]
                cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) if cpu_spec == "SLURM_CPUS_PER_TASK" else int(cpu_spec)
                record = run_one(
                    campaign, args.repo.resolve(), args.binary.resolve(), backend, n, iterations,
                    warmup, repetition, int(spec["timeout_seconds"]), cpu_count,
                    int(spec["gpu_count"]), defaults["scheme"], float(defaults["dt"]),
                    int(defaults["gpu_sampling_ms"]), args.local, args.allow_dirty,
                )
    return 1 if failures else 0


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def linear_exponent(points):
    valid = [(math.log(n), math.log(value)) for n, value in points if n > 0 and value > 0]
    if len(valid) < 2:
        return None
    xs, ys = zip(*valid)
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in valid) / denominator if denominator else None


def read_records(campaign):
    return [load_json(path) for path in sorted((campaign / "raw").glob("*.json"))]


def write_csv(path, rows, fields):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate(args):
    campaign = args.campaign.resolve()
    records = read_records(campaign)
    if not records:
        raise RuntimeError("campaign contains no raw records")
    result_fields = [
        "run_id", "status", "timestamp_utc", "git_sha", "git_dirty", "slurm_job_id", "hostname",
        "backend", "n", "iterations", "warmup_iterations", "repetition", "scheme", "dt",
        "cpu_count", "allocated_cpu_count", "cpus_per_task", "mpi_rank_count", "gpu_count", "compute_ms", "average_ms_per_iteration",
        "simulation_steps_per_second", "interactions_per_second", "estimated_GFLOP_per_second",
        "loop_wall_ms", "wall_clock_seconds", "peak_host_memory_kib", "peak_gpu_memory_mib",
        "mean_gpu_utilization_percent", "peak_gpu_utilization_percent", "mean_gpu_power_w",
        "peak_gpu_power_w", "gpu_energy_process_window_j", "gpu_energy_timed_region_j",
        "gpu_energy_scope", "gpu_sample_count", "gpu_inventory", "compiler_version", "cuda_version", "mpi_version",
        "slurm_partition", "slurm_job_nodelist", "loaded_modules",
        "stdout_path", "stderr_path", "gpu_samples_path", "returncode", "timed_out", "parse_error",
    ]
    write_csv(campaign / "results.csv", records, result_fields)
    grouped = {}
    for record in records:
        if record.get("status") == "ok":
            grouped.setdefault((record["backend"], int(record["n"])), []).append(record)
    summaries = []
    metrics = [
        "average_ms_per_iteration", "simulation_steps_per_second", "interactions_per_second",
        "estimated_GFLOP_per_second", "wall_clock_seconds", "peak_host_memory_kib",
        "peak_gpu_memory_mib", "mean_gpu_utilization_percent", "peak_gpu_utilization_percent",
        "mean_gpu_power_w", "peak_gpu_power_w", "gpu_energy_process_window_j",
    ]
    for (backend, n), group in sorted(grouped.items()):
        row = {"backend": backend, "n": n, "successful_repetitions": len(group)}
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if values:
                row[f"median_{metric}"] = statistics.median(values)
                row[f"min_{metric}"] = min(values)
                row[f"max_{metric}"] = max(values)
                row[f"stddev_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"iqr_{metric}"] = percentile(values, 0.75) - percentile(values, 0.25)
        summaries.append(row)
    lookup = {(row["backend"], row["n"]): row for row in summaries}
    for row in summaries:
        n = row["n"]
        naive = lookup.get(("cpu+naive", n))
        single = lookup.get(("gpu+tile+full", n))
        multi = lookup.get(("gpu+multinode", n))
        current = row.get("median_average_ms_per_iteration")
        if naive and current:
            row["speedup_vs_cpu_naive"] = naive["median_average_ms_per_iteration"] / current
        if row["backend"] == "gpu+multinode" and single and multi:
            speedup = single["median_average_ms_per_iteration"] / multi["median_average_ms_per_iteration"]
            row["speedup_4gpu_vs_1gpu"] = speedup
            row["four_gpu_parallel_efficiency"] = speedup / 4.0
    summary_fields = sorted({key for row in summaries for key in row}, key=lambda key: (key not in ("backend", "n", "successful_repetitions"), key))
    write_csv(campaign / "summary.csv", summaries, summary_fields)
    exponents = {}
    for backend in sorted({row["backend"] for row in summaries}):
        points = [(row["n"], row["median_average_ms_per_iteration"]) for row in summaries if row["backend"] == backend and "median_average_ms_per_iteration" in row]
        exponents[backend] = linear_exponent(points)
    lines = ["# NBody-EuroHPC benchmark summary", "", f"Generated: {utc_now()}", "", "## Empirical scaling exponents", ""]
    for backend, exponent in exponents.items():
        lines.append(f"- `{backend}`: {exponent:.4f}" if exponent is not None else f"- `{backend}`: insufficient data")
    lines += ["", "The exponent is an ordinary least-squares fit of `log(median time/iteration)` against `log(N)`; it is measured, not fixed at 2.", "", "## Data quality", "", f"- Raw records: {len(records)}", f"- Successful records: {sum(item.get('status') == 'ok' for item in records)}", "- GPU energy for the timed compute region is intentionally blank because external sampling cannot delimit that region without changing the executable.", "- Reported GPU power and process-window energy include initialization and warm-up and are labeled accordingly.", ""]
    atomic_text(campaign / "summary.md", "\n".join(lines))
    make_plots(campaign, summaries)
    print(f"aggregated {len(records)} records into {campaign}")
    return 0


def make_plots(campaign, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        atomic_text(campaign / "plots" / "PLOTS_NOT_GENERATED.txt", "matplotlib is unavailable; summary.csv is ready for plotting.\n")
        return
    plots = campaign / "plots"
    backends = sorted({row["backend"] for row in rows})
    def series(backend, field):
        selected = sorted((int(row["n"]), float(row[field])) for row in rows if row["backend"] == backend and row.get(field) is not None)
        return [item[0] for item in selected], [item[1] for item in selected]
    def line_plot(filename, field, ylabel, logx=True, logy=False):
        fig, axis = plt.subplots(figsize=(8, 5))
        any_data = False
        for backend in backends:
            xs, ys = series(backend, field)
            if xs:
                axis.plot(xs, ys, marker="o", label=backend)
                any_data = True
        if not any_data:
            plt.close(fig)
            return
        if logx: axis.set_xscale("log")
        if logy: axis.set_yscale("log")
        axis.set_xlabel("Body count N")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)
    line_plot("01_time_per_iteration_vs_n.png", "median_average_ms_per_iteration", "Median time per iteration (ms)", logy=True)
    line_plot("02_simulation_steps_per_second_vs_n.png", "median_simulation_steps_per_second", "Simulation steps/s", logy=True)
    line_plot("03_speedup_vs_cpu_naive.png", "speedup_vs_cpu_naive", "Speedup vs cpu+naive")
    line_plot("04_estimated_gflops_vs_n.png", "median_estimated_GFLOP_per_second", "Estimated GFLOP/s", logy=True)
    line_plot("05_peak_gpu_memory_vs_n.png", "median_peak_gpu_memory_mib", "Peak GPU memory above baseline (MiB)")
    line_plot("06_gpu_utilization_vs_n.png", "median_mean_gpu_utilization_percent", "Mean GPU utilization (%)")
    line_plot("07_gpu_power_vs_n.png", "median_mean_gpu_power_w", "Mean GPU power over whole invocation (W)")
    multi = [row for row in rows if row["backend"] == "gpu+multinode" and row.get("speedup_4gpu_vs_1gpu") is not None]
    if multi:
        fig, axis = plt.subplots(figsize=(8, 5))
        xs = [row["n"] for row in multi]
        axis.plot(xs, [row["speedup_4gpu_vs_1gpu"] for row in multi], marker="o", label="4-GPU speedup")
        axis.plot(xs, [row["four_gpu_parallel_efficiency"] for row in multi], marker="s", label="4-GPU efficiency")
        axis.axhline(4, linestyle="--", color="gray", label="ideal speedup")
        axis.axhline(1, linestyle=":", color="gray", label="ideal efficiency")
        axis.set_xscale("log")
        axis.set_xlabel("Body count N")
        axis.set_ylabel("Ratio")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots / "08_one_vs_four_a100.png", dpi=180)
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    init = sub.add_parser("init", help="create a unique external campaign directory")
    init.add_argument("--repo", type=Path, default=SCRIPT_DIR.parent)
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    init.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    init.add_argument("--allow-dirty", action="store_true", help="development/smoke only")
    init.set_defaults(function=init_campaign)
    run = sub.add_parser("run-group", help="run one primary backend group")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--repo", type=Path, default=SCRIPT_DIR.parent)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--group", choices=("cpu", "gpu"), required=True)
    run.add_argument("--backend")
    run.add_argument("--n-list", help="comma-separated subset allowed by matrix")
    run.add_argument("--repetitions", type=int)
    run.add_argument("--local", action="store_true", help="development smoke without srun")
    run.add_argument("--smoke", action="store_true", help="N=32, 2 timed iterations, one repetition")
    run.add_argument("--allow-dirty", action="store_true", help="development/smoke only")
    run.set_defaults(function=run_group)
    agg = sub.add_parser("aggregate", help="atomically create CSV, Markdown, and plots")
    agg.add_argument("--campaign", type=Path, required=True)
    agg.set_defaults(function=aggregate)
    return parser


def main():
    try:
        args = build_parser().parse_args()
        if not hasattr(args, "function"):
            raise ValueError("an action is required")
        return args.function(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"benchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
