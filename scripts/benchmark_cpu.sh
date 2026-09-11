#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --job-name=nbody_bench_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_cpu_%j.out
#SBATCH --error=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_cpu_%j.err

set -euo pipefail

[[ "${MURB_BENCHMARK_APPROVED:-}" == "YES" ]] || {
    echo "Set MURB_BENCHMARK_APPROVED=YES only after reviewing the benchmark plan." >&2
    exit 2
}
: "${MURB_CAMPAIGN_DIR:?Initialize a campaign and export its absolute path as MURB_CAMPAIGN_DIR}"
[[ "$MURB_CAMPAIGN_DIR" == /* ]] || { echo "MURB_CAMPAIGN_DIR must be absolute" >&2; exit 2; }

module purge
module load profile/base
module load gcc/12.2.0

ROOT="${MURB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the clean worktree root}}"
BUILD="${MURB_BUILD_DIR:-$ROOT/build-generic}"

python3 "$ROOT/benchmark/benchmark.py" run-group \
    --campaign "$MURB_CAMPAIGN_DIR" --repo "$ROOT" \
    --binary "$BUILD/bin/murb" --group cpu
