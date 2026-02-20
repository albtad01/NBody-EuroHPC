#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err

# Stop on first error
set -e

# Load modules (Using Booster modules since we are on the GPU node)
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "======================================================"
echo "    PHASE 1: TESTING CPU (OpenMP) IMPLEMENTATION      "
echo "======================================================"
# Configure 32 threads (max on Booster node host CPU)
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

rm -rf build-generic
cmake --preset generic
cmake --build build-generic -j 32

echo "Running Catch2 tests for CPU..."
srun ./build-generic/bin/murb-test

echo ""
echo "======================================================"
echo "    PHASE 2: TESTING GPU (CUDA) IMPLEMENTATION        "
echo "======================================================"
rm -rf build-leonardo
cmake --preset leonardo
cmake --build build-leonardo -j 32

echo "Running Catch2 tests for GPU..."
srun ./build-leonardo/bin/murb-test

echo ""
echo "======================================================"
echo "                ALL TESTS PASSED!                     "
echo "======================================================"