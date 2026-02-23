#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # 4 Task = 4 MPI Ranks
#SBATCH --cpus-per-task=8         # 8 Core CPU per gestire MPI
#SBATCH --gres=gpu:4              # Richiama le 4 A100 del nodo
#SBATCH --time=00:20:00
#SBATCH --output=logs/run_4gpu_%j.out
#SBATCH --error=logs/run_4gpu_%j.err

set -e

# Load modules (i tuoi moduli standard funzionanti)
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

rm -rf build-leonardo

echo "--- Starting Build ---"
cmake --preset leonardo
cmake --build build-leonardo -j 32

BIN="./build-leonardo/bin/murb"

echo "=========================================================="
echo " RUNNING ON 4x NVIDIA A100 GPUs (CUDA-Aware MPI via NVLink)"
echo "=========================================================="
# srun lancia automaticamente 4 rank in base a --ntasks-per-node=4
srun $BIN -n 500000 -i 50 --im gpu+multinode --nv -v -gf

echo "--- Job Finished ---"