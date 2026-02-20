#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_gpu_prof
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

set -euo pipefail

# -----------------------------
# Modules
# -----------------------------
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake

# (Opzionale ma utile) prova a vedere se esistono moduli nsys/ncu dedicati
# module avail nsight
# module load nsight-systems
# module load nsight-compute

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs profiles

# -----------------------------
# Build (solo se serve davvero)
# -----------------------------
# Se stai iterando sul kernel, ok rebuild.
# Se invece vuoi solo profilare, NON cancellare build ogni volta (perdi tempo).
# rm -rf build-leonardo

echo "--- Starting Build ---"
cmake --preset leonardo
cmake --build build-leonardo -j ${SLURM_CPUS_PER_TASK}

BIN="./build-leonardo/bin/murb"
ARGS="-n 200000 -i 200 --im gpu+tile+full --nv"

# -----------------------------
# Choose mode: RUN | NSYS | NCU
# -----------------------------
MODE="${1:-RUN}"   # usa: sbatch script.sh RUN  oppure NSYS oppure NCU

echo "--- MODE: ${MODE} ---"
echo "--- BIN:  ${BIN} ---"
echo "--- ARGS: ${ARGS} ---"

# Info GPU/allocazione (super utile nei log)
echo "--- GPU INFO ---"
nvidia-smi -L || true
nvidia-smi || true

# -----------------------------
# Execute
# -----------------------------
if [[ "${MODE}" == "RUN" ]]; then
  echo "--- Starting Execution (normal) ---"
  srun --cpu-bind=cores ${BIN} ${ARGS}

elif [[ "${MODE}" == "NSYS" ]]; then
  echo "--- Starting Nsight Systems ---"
  # Nota: profila poche iterazioni se vuoi ridurre la traccia
  # ARGS_NSYS="-n 200000 -i 20 --im gpu+tile+full --nv"
  ARGS_NSYS="${ARGS}"

  OUT="profiles/nsys_${SLURM_JOB_ID}"
  # -t cuda,nvtx : traccia CUDA e marker NVTX (se non usi NVTX va comunque)
  # --force-overwrite true : evita prompt
  # --sample=none : riduce overhead CPU sampling
  srun --cpu-bind=cores nsys profile \
    -t cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite true \
    -o "${OUT}" \
    ${BIN} ${ARGS_NSYS}

  echo "NSYS output: ${OUT}.qdrep (e .sqlite)"

elif [[ "${MODE}" == "NCU" ]]; then
  echo "--- Starting Nsight Compute ---"
  # IMPORTANTISSIMO: poche iterazioni, altrimenti ci mette una vita
  ARGS_NCU="-n 200000 -i 5 --im gpu+tile+full --nv"

  OUT="profiles/ncu_${SLURM_JOB_ID}"
  # --target-processes all : cattura kernel anche se ci sono processi figli (a volte utile)
  # --set full : molto pesante; alternativa: --set default + metriche mirate
  # --launch-skip/--launch-count: utile se vuoi skippare warmup
  srun --cpu-bind=cores ncu \
    --target-processes all \
    --set full \
    --force-overwrite \
    -o "${OUT}" \
    ${BIN} ${ARGS_NCU}

  echo "NCU output: ${OUT}.ncu-rep"

else
  echo "Unknown MODE='${MODE}'. Use RUN | NSYS | NCU"
  exit 1
fi

echo "--- Job Finished ---"
