#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_bench_a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_gpu_%j.out
#SBATCH --error=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_gpu_%j.err

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
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

ROOT="${MURB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the clean worktree root}}"
BUILD="${MURB_BUILD_DIR:-$ROOT/build-leonardo-multi}"

# Both paths run on the same Booster node. The single-GPU step binds GPU 0;
# the four-GPU step preserves the validated four-rank map and host-staged MPI.
python3 "$ROOT/benchmark/benchmark.py" run-group \
    --campaign "$MURB_CAMPAIGN_DIR" --repo "$ROOT" \
    --binary "$BUILD/bin/murb" --group gpu
