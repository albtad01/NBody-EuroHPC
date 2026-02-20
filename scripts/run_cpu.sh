#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --job-name=nbody_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=00:30:00
#SBATCH --output=logs/run_cpu_%j.out
#SBATCH --error=logs/run_cpu_%j.err

# Load modules
module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0
module load cmake

# Navigate to project root
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Maximize OpenMP performance for Leonardo DCGP (112 cores)
export OMP_NUM_THREADS=112
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Use 'generic' preset for CPU-only build
echo "--- Starting Build ---"
cmake --preset generic
cmake --build build-generic -j 112

# Execution
echo "--- Starting Execution ---"
BIN="./build-generic/bin/murb"

# Use CPU OpenMP implementation
srun $BIN -n 30000 -i 200 --im cpu+omp -nv

echo "--- Job Finished ---"