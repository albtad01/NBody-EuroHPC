#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

# Load modules
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake

# Navigate to project root
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# CRITICAL: Remove previous build to reset compiler cache
rm -rf build-leonardo

# Build process
echo "--- Starting Build ---"
cmake --preset leonardo
cmake --build build-leonardo -j 8

# Execution
echo "--- Starting Execution ---"
BIN="./build-leonardo/bin/murb"
# -v: verbose, -gf: show GFlop/s
srun $BIN -n 500000 -i 200 --im gpu+tile+full --nv
# Esegue il profiler Nsight Compute per una singola iterazione
# srun ncu --set full -o nbody_profile ./build-leonardo/bin/murb -n 200000 -i 2 --im gpu+tile+full --nv

echo "--- Job Finished ---"