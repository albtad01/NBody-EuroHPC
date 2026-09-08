# One-node four-A100 validation

The first supported `gpu+multinode` topology is deliberately limited to one
Leonardo Booster node, four MPI ranks, and four A100 GPUs with one Slurm-visible
GPU per rank. Two-node and eight-GPU runs are not supported in this phase.

## Build

Load the validated compiler/CUDA stack plus the CUDA-enabled OpenMPI module,
then build outside the batch job:

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
cmake --preset leonardo-multi
cmake --build build-leonardo-multi -j 32
```

The `leonardo-multi` preset is separate from `leonardo`. This keeps the
validated CPU, macOS, and single-A100 targets independent of MPI.

## Communication policy

This correctness-first implementation does **not** require CUDA-aware MPI.
Every rank computes and updates one contiguous body partition on its assigned
GPU. Each iteration copies that rank's six local position/velocity arrays to
host memory, uses host-buffer `MPI_Allgatherv` calls with explicit counts and
displacements, and copies the complete ordered state back to each GPU.

The OpenMPI module name indicates CUDA integration, but this phase does not
assume that GPU-direct collectives are enabled or correctly configured. There
is no automatic transport fallback: host staging is the only implemented MPI
transport. CUDA-aware collectives are deferred until they can be validated on
Leonardo.

## Correctness and smoke validation

Run the MPI correctness test only inside a one-node, four-GPU allocation:

```bash
export MURB_MPI_DIAGNOSTICS=1
srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
  --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
  ./build-leonardo-multi/bin/murb-test '[gpu-multinode]'
```

The test compares all positions and velocities with `cpu+naive` for random and
galaxy initial states, multiple iterations, and body counts 2048, 2049, and
2051. It reports the maximum normalized difference before applying the existing
backend tolerances.

After correctness passes, submit a small non-recording smoke run:

```bash
MURB_N=2049 MURB_ITERS=3 MURB_WARMUP=1 \
  sbatch scripts/run_gpu_multinode.sh
```

Set an absolute external `MURB_OUTPUT=...murbtraj` to enable optional rank-zero
recording. Without `MURB_OUTPUT`, no trajectory is created.

## Performance comparison (after correctness)

Use identical `N`, iteration count, warm-up, timestep, scheme, and softening,
with recording disabled. From a four-GPU allocation, run for each selected N:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
  --gpu-bind=map_gpu:0 --cpu-bind=cores \
  ./build-leonardo-multi/bin/murb -n N -i 20 --warmup 3 \
  --im gpu+tile+full --scheme galaxy --nv --gf --dt 3600

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
  --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
  ./build-leonardo-multi/bin/murb -n N -i 20 --warmup 3 \
  --im gpu+multinode --scheme galaxy --nv --gf --dt 3600
```

For each N, compute `speedup = T_1GPU / T_4GPU` from `compute_ms`, and parallel
efficiency as `speedup / 4`. Suggested initial sizes are 2048, 10000, 50000,
and 100000, subject to allocation time limits. Do not record trajectories in
these measurements.
