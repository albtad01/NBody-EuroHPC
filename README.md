
# MUrB – PACC (UM5IN160) Project Submission

![MUrB demo](assets/demo.gif)

This repository contains our optimized implementations of the MUrB n-body simulator:
CPU (naive/optim/SIMD/OpenMP), an MPI prototype, CUDA GPU kernels, and heterogeneous CPU+GPU execution.

Our work is based on the open-source MUrB framework provided for the course and **extends/modifies it substantially**
(e.g., additional kernels/implementations, optimizations, build/run presets, and experimental variants).
This repository contains our optimized implementations of the MUrB n-body simulator:
CPU (naive/optim/SIMD/OpenMP), MPI prototype, CUDA GPU kernels, and heterogeneous CPU+GPU execution.

## 1) Build 

We use out-of-source builds.

### Release build (fast-math + CUDA enabled)

This is the configuration used for both CPU benchmarking and GPU/heterogeneous runs.

```bash
module load openmpi/5.0.8
module load easytools/g1d7d343
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

## 4) License & attribution (MIT)

This repository contains code derived from the **MUrB framework** developed at **Sorbonne University, LIP6**,
released under the **MIT License**.

Per the MIT License terms, the corresponding copyright notice and permission notice
must be included in all copies or substantial portions of the software.

- The full license text is provided in the `LICENSE` file.
- Copyright (c) 2023 Sorbonne University, LIP6.

If you redistribute or reuse substantial portions of this repository, keep the `LICENSE` file
and preserve the attribution above.

---

## 5) Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
