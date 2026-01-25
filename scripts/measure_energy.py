#!/usr/bin/env python3
"""
measure_energy.py

Submit a Slurm job that runs murb while collecting NCM power samples (node-conso -t 5).
Then parse energy_*.log and produce human-readable reports via parse_energy_log.py.

Supports:
- Standard single run on any partition
- On iml-ia770 only: optional core split (P / E / LPE) with CPU pinning + OMP_NUM_THREADS
- FAIR split by default: same workload (-n/-i) on P/E/LPE

Usage:
  python3 measure_energy.py [-n N] [-i I] [--im IM] [--partition PARTITION]
                            [--core-mode {auto,all,split,p,e,lpe}]
                            [--lpe-n N] [--lpe-i I] [--allow-unfair-lpe]
                            [--help]

Examples:
  # GPU partition (single run)
  python3 measure_energy.py --partition az4-n4090 --im gpu+tile -n 500000 -i 100

  # CPU partition (single run, no pinning)
  python3 measure_energy.py --partition iml-ia770 --im cpu+omp -n 30000 -i 200

  # CPU partition: FAIR split into P / E / LPE (same -n/-i for all)
  python3 measure_energy.py --partition iml-ia770 --im cpu+omp -n 30000 -i 200 --core-mode split

  # If you REALLY need to reduce LPE workload (unfair comparison), make it explicit:
  python3 measure_energy.py --partition iml-ia770 --im cpu+omp -n 30000 -i 200 \
    --core-mode split --lpe-n 5000 --lpe-i 200 --allow-unfair-lpe
"""

import sys
import argparse
import subprocess
import time
import os
import tempfile
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
PARSE_ENERGY_SCRIPT = SCRIPT_DIR / "parse_energy_log.py"

PARTITIONS = {
    "az4-n4090": {"node": "az4-n4090-1"},
    "az4-a7900": {"node": "az4-a7900-1"},
    "iml-ia770": {
        "node": "iml-ia770-1",
        # Your mapping:
        # P:   0-11 (SMT on P-cores, 12 threads)
        # E:  12-19 (8 threads)
        # LPE:20-21 (2 threads)
        "core_sets": {
            "P":   {"cpus": "0-11",   "threads": 12},
            "E":   {"cpus": "12-19",  "threads": 8},
            "LPE": {"cpus": "20-21",  "threads": 2},
            "ALL": {"cpus": "0-21",   "threads": 22},
        },
    },
    "az5-a890m": {"node": "az5-a890m-1"},
}

SLURM_BATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=murb_ncm
#SBATCH --partition=__PARTITION__
#SBATCH --nodelist=__NODELIST__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=murb_job.%j.out
#SBATCH --error=murb_job.%j.err

set -euo pipefail

module load ncm/gdcc873f || true

node-conso -P 1 || true
node-conso -P 2 || true
node-conso -M 1 || true
node-conso -M 2 || true

# Determine murb binary path based on current directory
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    MURB_BIN="../build/bin/murb"
else
    MURB_BIN="./build/bin/murb"
fi

run_one() {
  local TAG="$1"
  local CPU_LIST="$2"
  local OMP_THREADS="$3"
  local NVAL="$4"
  local IVAL="$5"
  local IMVAL="$6"

  local ENERGY_LOG="energy_${TAG}.log"
  local MURB_LOG="murb_${TAG}.log"

  echo "# NCM LOG" > "$ENERGY_LOG"
  echo "# HOST: $(hostname)" >> "$ENERGY_LOG"
  echo "# PARTITION: __PARTITION__" >> "$ENERGY_LOG"
  echo "# TAG: ${TAG}" >> "$ENERGY_LOG"
  echo "# CPU_LIST: ${CPU_LIST}" >> "$ENERGY_LOG"
  echo "# OMP_NUM_THREADS: ${OMP_THREADS}" >> "$ENERGY_LOG"
  echo "# WORKLOAD: -n ${NVAL} -i ${IVAL} --im ${IMVAL}" >> "$ENERGY_LOG"
  echo "# START_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"

  export OMP_NUM_THREADS="${OMP_THREADS}"
  export OMP_DYNAMIC=false
  export OMP_PROC_BIND=true
  export OMP_PLACES=cores

  local RUN_CMD="${MURB_BIN} -n ${NVAL} -i ${IVAL} --im ${IMVAL} --nv"
  if [[ -n "${CPU_LIST}" ]]; then
    RUN_CMD="taskset -c ${CPU_LIST} ${RUN_CMD}"
  fi

  local T_MURB_START T_MURB_END
  T_MURB_START=$(date +%s.%N)

  ${RUN_CMD} > "$MURB_LOG" 2>&1 &
  local MURB_PID=$!

  node-conso -t 5 > /dev/null || true
  node-conso -t 5 >> "$ENERGY_LOG" 2>&1 || true

  wait "$MURB_PID" || true

  T_MURB_END=$(date +%s.%N)

  echo "# END_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"
  echo "# MURB_WALLCLOCK_START: ${T_MURB_START}" >> "$ENERGY_LOG"
  echo "# MURB_WALLCLOCK_END: ${T_MURB_END}" >> "$ENERGY_LOG"
  echo "# ITERS: ${IVAL}" >> "$ENERGY_LOG"
}

__RUN_BLOCK__

node-conso -m 1 || true
node-conso -m 2 || true
"""


def show_help():
    print(__doc__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit and monitor NCM energy measurement job",
        add_help=False,
    )
    parser.add_argument("-n", type=int, default=500000)
    parser.add_argument("-i", type=int, default=100)
    parser.add_argument("--im", type=str, default="gpu+tile")
    parser.add_argument("--partition", type=str, default="az4-n4090",
                        choices=list(PARTITIONS.keys()))

    parser.add_argument("--core-mode", type=str, default="auto",
                        choices=["auto", "all", "split", "p", "e", "lpe"],
                        help=("iml-ia770 only. auto=single run no pinning; "
                              "split=P/E/LPE sequential; p/e/lpe only; all pins 0-21."))

    # Legacy knobs (disabled by default to keep comparisons fair)
    parser.add_argument("--lpe-n", type=int, default=None)
    parser.add_argument("--lpe-i", type=int, default=None)
    parser.add_argument("--allow-unfair-lpe", action="store_true",
                        help="Allow overriding LPE workload with --lpe-n/--lpe-i (breaks fairness).")

    parser.add_argument("--help", "-h", action="store_true")

    args = parser.parse_args()
    if args.help:
        show_help()
        sys.exit(0)

    # Enforce fairness by default
    if (args.lpe_n is not None or args.lpe_i is not None) and not args.allow_unfair_lpe:
        raise ValueError("You set --lpe-n/--lpe-i but did not pass --allow-unfair-lpe. "
                         "By default split mode enforces equal workload across P/E/LPE.")

    return args


def _build_run_block(args):
    part = args.partition
    im_val = args.im

    if part != "iml-ia770":
        if args.core_mode in ("split", "p", "e", "lpe", "all"):
            raise ValueError(f"--core-mode {args.core_mode} is only supported on iml-ia770.")
        return f'run_one "ALL" "" "1" "{args.n}" "{args.i}" "{im_val}"\n'

    core_sets = PARTITIONS["iml-ia770"]["core_sets"]
    mode = args.core_mode

    if mode == "auto":
        return f'run_one "ALL" "" "1" "{args.n}" "{args.i}" "{im_val}"\n'

    if mode == "split":
        p = core_sets["P"]
        e = core_sets["E"]
        lpe = core_sets["LPE"]

        # FAIR by default: same workload
        lpe_n = args.n
        lpe_i = args.i

        # Allow unfair override only if explicitly enabled
        if args.allow_unfair_lpe:
            if args.lpe_n is not None:
                lpe_n = args.lpe_n
            if args.lpe_i is not None:
                lpe_i = args.lpe_i

        lines = [
            f'run_one "P"   "{p["cpus"]}"   "{p["threads"]}"   "{args.n}"  "{args.i}"  "{im_val}"',
            f'run_one "E"   "{e["cpus"]}"   "{e["threads"]}"   "{args.n}"  "{args.i}"  "{im_val}"',
            f'run_one "LPE" "{lpe["cpus"]}" "{lpe["threads"]}" "{lpe_n}" "{lpe_i}" "{im_val}"',
        ]
        return "\n".join(lines) + "\n"

    if mode == "all":
        allc = core_sets["ALL"]
        return f'run_one "ALL" "{allc["cpus"]}" "{allc["threads"]}" "{args.n}" "{args.i}" "{im_val}"\n'

    if mode in ("p", "e", "lpe"):
        key = mode.upper()
        cfg = core_sets[key]
        nval, ival = args.n, args.i
        if key == "LPE" and args.allow_unfair_lpe:
            if args.lpe_n is not None:
                nval = args.lpe_n
            if args.lpe_i is not None:
                ival = args.lpe_i
        return f'run_one "{key}" "{cfg["cpus"]}" "{cfg["threads"]}" "{nval}" "{ival}" "{im_val}"\n'

    raise ValueError(f"Unknown core mode: {mode}")


def create_batch_script(args, temp_dir):
    node = PARTITIONS[args.partition]["node"]
    run_block = _build_run_block(args)

    batch_content = SLURM_BATCH_TEMPLATE
    batch_content = batch_content.replace("__PARTITION__", args.partition)
    batch_content = batch_content.replace("__NODELIST__", node)
    batch_content = batch_content.replace("__RUN_BLOCK__", run_block)

    batch_file = Path(temp_dir) / f"slurm_murb_{os.getpid()}.sh"
    batch_file.write_text(batch_content)
    batch_file.chmod(0o755)
    return batch_file


def submit_job(batch_file):
    result = subprocess.run(["sbatch", str(batch_file)],
                            capture_output=True, text=True, check=True)
    return int(result.stdout.split()[-1])


def wait_for_job(job_id):
    print(f"Job submitted (job id: {job_id})")
    print("Waiting for the job to start...")

    last_state = None
    while True:
        try:
            result = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                                    capture_output=True, text=True, timeout=5)
            state = result.stdout.strip()
            if not state:
                print("Job no longer in squeue. Waiting for finalization...")
                break
            if state != last_state:
                print(f"Job state: {state}")
                last_state = state
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)

    time.sleep(2)
    try:
        result = subprocess.run(["sacct", "-j", str(job_id), "-n", "-o", "State%20", "-P"],
                                capture_output=True, text=True, timeout=5)
        if result.stdout:
            job_exit = result.stdout.split("|")[0].strip()
            print(f"Job completed (sacct state: {job_exit})")
        else:
            print("Job completed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Job completed (sacct not available)")


def move_output_files(im_val):
    im_dir = f"power_tracking_{im_val.replace('+', '_')}"
    im_path = Path(im_dir)
    im_path.mkdir(parents=True, exist_ok=True)

    for src in Path(".").glob("energy_*.log"):
        shutil.move(str(src), str(im_path / src.name))
    for src in Path(".").glob("murb_*.log"):
        shutil.move(str(src), str(im_path / src.name))
    for src in Path(".").glob("murb_job*"):
        shutil.move(str(src), str(im_path / src.name))
    return im_path


def run_parser_on_outputs(im_path: Path):
    energy_logs = sorted(im_path.glob("energy_*.log"))
    if not energy_logs:
        raise FileNotFoundError(f"No energy_*.log found in {im_path}")

    for e_log in energy_logs:
        tag = e_log.stem.replace("energy_", "")
        report_file = im_path / f"energy_report_{tag}.txt"
        murb_log = im_path / f"murb_{tag}.log"

        cmd = ["python3", str(PARSE_ENERGY_SCRIPT), str(e_log), str(report_file)]
        if murb_log.exists():
            cmd.append(str(murb_log))

        print(f"\nAnalyzing: {e_log.name}  ->  {report_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"parse_energy_log.py failed for {e_log.name}")
        print(result.stdout)


def main():
    try:
        args = parse_args()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    im_val = args.im
    im_path = Path(f"power_tracking_{im_val.replace('+', '_')}")
    im_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {im_path}")
    print(f"Using partition: {args.partition}")
    print(f"Core mode: {args.core_mode}")

    with tempfile.TemporaryDirectory() as temp_dir:
        batch_file = create_batch_script(args, temp_dir)
        print(f"Batch script created: {batch_file}")
        job_id = submit_job(batch_file)
        wait_for_job(job_id)

    im_path = move_output_files(im_val)
    print(f"Output files moved to {im_path}")

    try:
        run_parser_on_outputs(im_path)
    except Exception as e:
        print(f"ERROR during parsing: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAnalysis complete.")
    print(f"Reports saved under: {im_path}")
    print("Done.")


if __name__ == "__main__":
    main()
