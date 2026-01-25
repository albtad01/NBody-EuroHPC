
# MUrB – PACC (UM5IN160) Project Submission

This repository contains our optimized implementations of the MUrB n-body simulator:
CPU (naive/optim/SIMD/OpenMP), MPI prototype, CUDA GPU kernels, and heterogeneous CPU+GPU execution.

## 1) Build 

We use out-of-source builds.

### Release build (fast-math + CUDA enabled)

This is the configuration used for both CPU benchmarking and GPU/heterogeneous runs.

```bash
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_FAST_MATH=ON -DENABLE_MURB_CUDA=ON
cmake --build . -j4
```

## 2) Validation tests (Catch2)

From the build directory:

```bash
cd build
./bin/murb-test
```

## 3) Run commands (copy-paste)

All runs below disable visualization (`--nv`) and enable FLOPs metric (`--gf`).

### Cluster prerequisites (modules)

To run on the cluster, load the following modules:

```bash
module load openmpi/5.0.8
module load easytools/g1d7d343
```

### A) CPU single-thread (iml-ia770)

#### cpu+naive

```bash
srun -p iml-ia770 -n 1 --cpus-per-task=1 --threads-per-core=1 --cpu-bind=cores \
	./build/bin/murb -n 30000 -i 200 --nv --im cpu+naive --gf
```

#### cpu+optim

```bash
srun -p iml-ia770 -n 1 --cpus-per-task=1 --threads-per-core=1 --cpu-bind=cores \
	./build/bin/murb -n 30000 -i 200 --nv --im cpu+optim --gf
```

#### cpu+simd

```bash
srun -p iml-ia770 -n 1 --cpus-per-task=1 --threads-per-core=1 --cpu-bind=cores \
	./build/bin/murb -n 30000 -i 200 --nv --im cpu+simd --gf
```

### B) OpenMP (iml-ia770)

```bash
export OMP_NUM_THREADS=12
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=ACTIVE

srun -p iml-ia770 -n 1 --cpus-per-task=12 --threads-per-core=1 --cpu-bind=cores --export=ALL \
	./build/bin/murb -n 30000 -i 200 --nv --im cpu+omp --gf
```

### C) MPI prototype (single node, iml-ia770)

```bash
srun -p iml-ia770 -n 4 --cpus-per-task=1 --threads-per-core=1 --cpu-bind=cores \
	./build/bin/murb -n 30000 -i 200 --nv --im mpi --gf
```

### D) CUDA GPU (az4-n4090)

```bash
srun -p az4-n4090 -n 1 --cpus-per-task=1 --cpu-bind=cores \
	./build/bin/murb -n 200000 -i 200 --nv --im gpu+tile+full --gf
```

Other available GPU tags:

- `gpu+tile`
- `gpu+tile+full`
- `gpu+tile+full200k`
- `gpu+tracking`
- `gpu+leapfrog`

### E) Heterogeneous CPU+GPU (az4-n4090)

```bash
export MURB_HETERO_GPU_FRACTION=0.75
export OMP_NUM_THREADS=12
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

srun -p az4-n4090 -n 1 --cpus-per-task=12 --threads-per-core=1 --cpu-bind=cores --export=ALL \
	./build/bin/murb -n 30000 -i 60 --nv --im hetero --gf
```
