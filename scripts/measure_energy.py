#!/usr/bin/env python3
"""
measure_energy.py

Python translation of measure_ncm_murb.sh.

Submit a Slurm job on a selected partition that runs ./bin/murb
while continuously collecting NCM power samples (node-conso -t 5).
After the job finishes, the script computes mean/variance power per component
(GPU, CPU, RAM, MOTHERBOARD) and writes a human-readable report using parse_energy_log.py.

Usage:
    python3 measure_energy.py [-n N] [-i I] [--im IM] [--partition PARTITION] [--help]

Supported partitions: az4-n4090 (default), az4-a7900, iml-ia770, az5-a890m

Examples:
    python3 measure_energy.py -n 500000 -i 100 --im gpu+tile
    python3 measure_energy.py -n 500000 -i 100 --partition az4-a7900

Requirements & assumptions:
  * You have sbatch/squeue available and a shared filesystem
  * node-conso is available via module ncm/gdcc873f on the compute node
  * The murb binary exists at ./bin/murb relative to the job working directory
  * parse_energy_log.py is available in the same directory as this script
"""

import sys
import argparse
import subprocess
import time
import os
import tempfile
import shutil
from pathlib import Path


# Path to the parse_energy_log script
SCRIPT_DIR = Path(__file__).parent.absolute()
PARSE_ENERGY_SCRIPT = SCRIPT_DIR / "parse_energy_log.py"

# Partition configurations
PARTITIONS = {
    'az4-n4090': {
        'node': 'az4-n4090-1',
        'gpu_channels': ['1.0', '1.1', '1.2', '1.3'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
    },
    'az4-a7900': {
        'node': 'az4-a7900-1',
        'gpu_channels': ['1.0', '1.1', '1.2'],
        'cpu_channel': '0.4',
        'ram_channel': '0.2',
        'mb_channel': '0.3',
    },
    'iml-ia770': {
        'node': 'iml-ia770-1',
        'gpu_channels': ['0.2', '0.3', '1.0', '1.1'],
        'cpu_channel': '2.0',
        'ram_channel': None,
        'mb_channel': None,
    },
    'az5-a890m': {
        'node': 'az5-a890m-1',
        'gpu_channels': None,  # No GPU
        'cpu_channel': '0.0',
        'ram_channel': None,
        'mb_channel': None,
    },
}


SLURM_BATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=murb_ncm
#SBATCH --partition=__PARTITION__
#SBATCH --nodelist=__NODELIST__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=murb_job.%j.out
#SBATCH --error=murb_job.%j.err

# job body starts here
set -euo pipefail

# Load NCM module (required to have node-conso in PATH)
module load ncm/gdcc873f || true

# Ensure I2C chains powered and measurements started
node-conso -P 1 || true
node-conso -P 2 || true
node-conso -M 1 || true
node-conso -M 2 || true

ENERGY_LOG="energy.log"
MURB_LOG="murb.log"

# write header with wallclock times (to compute accurate duration later)
echo "# NCM LOG" > "$ENERGY_LOG"
echo "# HOST: $(hostname)" >> "$ENERGY_LOG"
echo "# PARTITION: __PARTITION__" >> "$ENERGY_LOG"
echo "# START_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"

# Determine murb binary path based on current directory
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    MURB_BIN="../build/bin/murb"
else
    MURB_BIN="./build/bin/murb"
fi

# start murb in background
$MURB_BIN -n __NVAL__ -i __IVAL__ --im __IMVAL__ --nv > "$MURB_LOG" 2>&1 &
MURB_PID=$!

# let murb start and reach steady state
node-conso -t 5 > /dev/null

# take exactly one 5-second power measurement
node-conso -t 5 >> "$ENERGY_LOG" 2>&1 || true

# final touch: stop measurements and write end time
node-conso -m 1 || true
node-conso -m 2 || true

echo "# END_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"

# exit with the same code as murb (wait and capture)
wait "$MURB_PID" || true
"""


def show_help():
    """Print help message."""
    print(__doc__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit and monitor NCM energy measurement job',
        add_help=False
    )
    parser.add_argument('-n', type=int, default=500000,
                        help='passes -n N to ./bin/murb (default: 500000)')
    parser.add_argument('-i', type=int, default=100,
                        help='passes -i I to ./bin/murb (default: 100)')
    parser.add_argument('--im', type=str, default='gpu+tile',
                        help='passes --im IM to ./bin/murb (default: gpu+tile)')
    parser.add_argument('--partition', type=str, default='az4-n4090',
                        choices=list(PARTITIONS.keys()),
                        help='Slurm partition to use (default: az4-n4090)')
    parser.add_argument('--help', '-h', action='store_true',
                        help='show this help and exit')
    
    args = parser.parse_args()
    
    if args.help:
        show_help()
        sys.exit(0)
    
    return args


def create_batch_script(n_val, i_val, im_val, partition, temp_dir):
    """
    Create a Slurm batch script with parameter substitution.
    
    Args:
        partition: partition name (e.g., 'az4-n4090')
    
    Returns:
        path to the batch script file
    """
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    
    partition_config = PARTITIONS[partition]
    node = partition_config['node']
    
    batch_content = SLURM_BATCH_TEMPLATE
    batch_content = batch_content.replace('__PARTITION__', partition)
    batch_content = batch_content.replace('__NODELIST__', node)
    batch_content = batch_content.replace('__NVAL__', str(n_val))
    batch_content = batch_content.replace('__IVAL__', str(i_val))
    batch_content = batch_content.replace('__IMVAL__', str(im_val))
    
    batch_file = Path(temp_dir) / f"slurm_murb_{os.getpid()}.sh"
    batch_file.write_text(batch_content)
    batch_file.chmod(0o755)
    
    return batch_file


def submit_job(batch_file):
    """
    Submit job using sbatch and return job ID.
    
    Returns:
        job ID (int)
    """
    result = subprocess.run(['sbatch', str(batch_file)], 
                          capture_output=True, text=True, check=True)
    # Output format: "Submitted batch job 12345"
    job_id = int(result.stdout.split()[-1])
    return job_id


def wait_for_job(job_id):
    """Wait for Slurm job to complete."""
    print(f"Job submitted (job id: {job_id})")
    print("Waiting for the job to start...")
    
    last_state = None
    
    while True:
        try:
            result = subprocess.run(['squeue', '-j', str(job_id), '-h', '-o', '%T'],
                                  capture_output=True, text=True, timeout=5)
            state = result.stdout.strip()
            
            if not state:
                # Job is no longer in squeue: finished
                print("Job no longer in squeue. Waiting for finalization...")
                break
            
            if state != last_state:
                if state == "PENDING":
                    print("Job pending")
                elif state in ["CONFIGURING", "COMPLETING"]:
                    print(f"Job {state}")
                elif state == "RUNNING":
                    print("Job running")
                else:
                    print(f"Job state: {state}")
                last_state = state
        
        except subprocess.TimeoutExpired:
            print("squeue timeout, continuing...")
        
        time.sleep(5)
    
    # Wait a bit for sacct records
    time.sleep(2)
    
    # Try to get job exit status via sacct
    try:
        result = subprocess.run(['sacct', '-j', str(job_id), '-n', '-o', 'State%20', '-P'],
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            job_exit = result.stdout.split('|')[0].strip()
            print(f"Job completed (sacct state: {job_exit})")
        else:
            print("Job completed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Job completed (sacct not available)")


def move_output_files(im_val):
    """Move output files to the measurement directory."""
    im_dir = f"power_tracking_{im_val.replace('+', '_')}"
    im_path = Path(im_dir)
    im_path.mkdir(parents=True, exist_ok=True)
    
    # Move log files
    for pattern in ['energy.log', 'murb.log', 'murb_job*']:
        if pattern == 'energy.log':
            src = Path('energy.log')
            if src.exists():
                shutil.move(str(src), str(im_path / 'energy.log'))
        elif pattern == 'murb.log':
            src = Path('murb.log')
            if src.exists():
                shutil.move(str(src), str(im_path / 'murb.log'))
        else:
            # murb_job* (glob pattern)
            for src in Path('.').glob('murb_job*'):
                shutil.move(str(src), str(im_path / src.name))
    
    return im_path


def main():
    """Main entry point."""
    args = parse_args()
    
    n_val = args.n
    i_val = args.i
    im_val = args.im
    partition = args.partition
    
    # Create output directory
    im_dir = f"power_tracking_{im_val.replace('+', '_')}"
    im_path = Path(im_dir)
    im_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {im_path}")
    print(f"Using partition: {partition}")
    
    # Create temporary batch script
    with tempfile.TemporaryDirectory() as temp_dir:
        batch_file = create_batch_script(n_val, i_val, im_val, partition, temp_dir)
        print(f"Batch script created: {batch_file}")
        
        # Submit job
        job_id = submit_job(batch_file)
        
        # Wait for job to complete
        wait_for_job(job_id)
    
    # Move output files
    im_path = move_output_files(im_val)
    print(f"Output files moved to {im_path}")
    
    # Check if energy log exists
    energy_file = im_path / 'energy.log'
    if not energy_file.exists():
        print(f"ERROR: energy file {energy_file} not found", file=sys.stderr)
        print("The job probably didn't run on a shared filesystem or failed before creating the log.",
              file=sys.stderr)
        sys.exit(1)
    
    # Parse energy log using parse_energy_log.py
    print(f"\nAnalyzing energy file: {energy_file}")
    report_file = im_path / 'energy_report.txt'
    
    try:
        result = subprocess.run(['python3', str(PARSE_ENERGY_SCRIPT), str(energy_file), str(report_file)],
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running parse_energy_log.py: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: parse_energy_log.py not found at {PARSE_ENERGY_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    
    print("\nAnalysis complete.")
    print(f"Report saved to {report_file}")
    print("Done.")


if __name__ == '__main__':
    main()
