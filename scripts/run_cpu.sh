#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --job-name=nbody_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/EUHPC_TDEMO_26/NBody-EuroHPC/log/nbody_cpu_%j.out
#SBATCH --error=/leonardo_work/EUHPC_TDEMO_26/NBody-EuroHPC/log/nbody_cpu_%j.err

# Prebuilt Phase 1 benchmark. Account may be overridden with sbatch --account.
set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

ROOT="${MURB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the worktree root}}"
BUILD="${MURB_BUILD_DIR:-$ROOT/build-generic}"
BIN="$BUILD/bin/murb"
[[ -x "$BIN" && -f "$BUILD/murb-build.ready" ]] || {
    echo "Missing executable or successful build stamp: $BUILD. Build before submitting." >&2
    exit 1
}
source "$ROOT/scripts/provenance.sh"
murb_check_provenance "$ROOT" "$BIN"

echo "job=${SLURM_JOB_ID:-unknown} node=$(hostname) executable=$BIN"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    "$BIN" -n "${MURB_N:-10000}" -i "${MURB_ITERS:-20}" \
    --im cpu+naive --nv --gf --dt "${MURB_DT:-3600}"
