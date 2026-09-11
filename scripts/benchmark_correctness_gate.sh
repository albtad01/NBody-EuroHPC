#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_bench_gate
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/correctness_%j.out
#SBATCH --error=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/correctness_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

ROOT="${MURB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the clean worktree root}}"
BUILD="${MURB_BUILD_DIR:-$ROOT/build-leonardo-multi}"
TEST_BIN="$BUILD/bin/murb-test"
[[ -x "$TEST_BIN" && -f "$BUILD/murb-build.ready" ]] || {
    echo "Build with the leonardo-multi preset before running the gate." >&2
    exit 1
}

revision="$(git -C "$ROOT" rev-parse HEAD)"
version="$("$BUILD/bin/murb" --version)"
[[ "$version" == "murb revision=$revision dirty=0 "* && "$version" == *" cuda=1 "* && "$version" == *" mpi=1"* ]] || {
    echo "Clean, current CUDA+MPI build required: $version" >&2
    exit 1
}
git -C "$ROOT" status --porcelain --untracked-files=normal | grep -q . && {
    echo "Correctness gate requires a clean worktree." >&2
    exit 1
}

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Existing tests cover 2048/2049 for these primary single-rank backends.
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    --gpu-bind=map_gpu:0 --cpu-bind=cores "$TEST_BIN" \
    '[cpu-naive],[cpu-omp],[gpu-tile-full]'

# Existing test additionally covers non-divisible N=2051. Exactly one node,
# four MPI ranks, and four A100s; no multi-node or CUDA-aware MPI claim.
export MURB_MPI_DIAGNOSTICS=1
srun --exclusive --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
    --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
    "$TEST_BIN" '[gpu-multinode]'
