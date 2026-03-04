# MUrB – N-Body Simulation on CPU & GPU

![MUrB demo](assets/demo.gif)

**N-Body gravitational simulation with progressive optimisation from naive CPU to multi-GPU.**
Started as a project for *Parallel Programming (PACC – UM5IN160)* at Sorbonne University;
extended and scaled to the **Leonardo supercomputer** (CINECA) for the **EuroHPC Summit 2026**.

Our work builds on the open-source MUrB framework (Sorbonne / LIP6) and extends it substantially
with additional kernels, CUDA implementations, multi-node MPI+CUDA support, heterogeneous
CPU+GPU execution, a Barnes-Hut approximation, simulation history tracking, and a binary replay mode.

---

## Table of Contents

1. [Architecture & Implementations](#1-architecture--implementations)
2. [Repository Layout](#2-repository-layout)
3. [Build](#3-build)
4. [Run](#4-run)
5. [Simulation Replay (Binary Player)](#5-simulation-replay-binary-player)
6. [Validation Tests (Catch2)](#6-validation-tests-catch2)
7. [Scripts](#7-scripts)
8. [License & Attribution](#8-license--attribution-mit)

---

## 1) Architecture & Implementations

| Tag | Backend | Description |
|-----|---------|-------------|
| `cpu+naive` | CPU | Baseline O(n²) direct summation |
| `cpu+optim` | CPU | Loop optimisations, SoA layout, Newton's 3rd law |
| `cpu+simd` | CPU | Explicit SIMD via MIPP intrinsics |
| `cpu+omp` | CPU | OpenMP parallel-for with SIMD |
| `mpi` | CPU | MPI-distributed (single-node prototype) |
| `cpu+barneshut` | CPU | Barnes-Hut O(n log n) tree approximation |
| `gpu+tile` | CUDA | Tiled shared-memory kernel |
| `gpu+tile+full` | CUDA | Fully device-resident tiled kernel |
| `gpu+tile+full200k` | CUDA | Variant tuned for n ≥ 200 k bodies |
| `gpu+tracking` | CUDA | GPU kernel with simulation history tracking |
| `gpu+leapfrog` | CUDA | Leapfrog (symplectic) integrator on GPU |
| `gpu+multinode` | CUDA+MPI | Multi-GPU with CUDA-aware MPI (NVLink) |
| `hetero` | CPU+CUDA | Heterogeneous: OpenMP threads + GPU, configurable split |
| `bin+player` | — | Replay a previously saved `simulation_data.bin` |

---

## 2) Repository Layout

```
.
├── CMakeLists.txt            # Main build definition
├── CMakePresets.json          # mac / leonardo / generic presets
├── lib/
│   ├── Catch2/               # Unit-test framework (header-only)
│   └── MIPP/                 # Portable SIMD wrapper
├── src/
│   ├── common/
│   │   ├── core/             # Bodies, allocators, simulation interface, history tracking
│   │   ├── ogl/              # OpenGL real-time visualisation
│   │   └── utils/            # CLI parser, perf counters
│   ├── murb/
│   │   ├── main.cpp          # Entry point
│   │   └── implem/           # All simulation back-ends (see table above)
│   └── test/                 # Catch2 tests (CPU + CUDA)
├── scripts/                  # SLURM job scripts, profiling, plotting
└── assets/                   # demo.gif
```

---

## 3) Build

All builds are **out-of-source** and driven by CMake presets.

### Prerequisites

| Dependency | Required | Notes |
|------------|----------|-------|
| CMake ≥ 3.10 | Yes | |
| C++20 compiler | Yes | GCC 12+, AppleClang 17+ |
| MPI | Yes | OpenMPI 4.x / 5.x |
| CUDA Toolkit ≥ 12 | GPU builds | `sm_80` for A100 |
| OpenGL + GLEW + GLM + GLFW | Visualisation | macOS / desktop only |
| MIPP | Bundled | SIMD abstraction (included in `lib/`) |
| Catch2 v2 | Bundled | Testing (included in `lib/`) |

### macOS (visualisation, no CUDA)

```bash
cmake --preset mac
cmake --build build-mac -j $(sysctl -n hw.ncpu)
```

### Leonardo – GPU (CUDA + OpenMP + MPI)

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake

cmake --preset leonardo
cmake --build build-leonardo -j 32
```

### Leonardo – CPU only (OpenMP + MPI, no CUDA)

```bash
module purge
module load profile/base
module load openmpi/4.1.6--gcc--12.2.0
module load cmake

cmake --preset generic
cmake --build build-generic -j 112
```

### Preset summary

| Preset | CUDA | OpenMP | Visualisation | Target |
|--------|------|--------|---------------|--------|
| `mac` | OFF | OFF | ON | macOS desktop |
| `leonardo` | ON (`sm_80`) | ON | OFF | Leonardo Booster (A100) |
| `generic` | OFF | ON | OFF | Linux CPU cluster |

---

## 4) Run

Common flags: `--nv` disables visualisation, `--gf` prints GFlop/s, `-v` verbose output.

```
./murb -n <bodies> -i <iterations> --im <tag> [--nv] [--gf] [-v] [-dt <timestep>]
```

### A) CPU – single thread

```bash
srun ./build-generic/bin/murb -n 30000 -i 200 --nv --im cpu+naive --gf
srun ./build-generic/bin/murb -n 30000 -i 200 --nv --im cpu+optim --gf
srun ./build-generic/bin/murb -n 30000 -i 200 --nv --im cpu+simd  --gf
```

### B) CPU – OpenMP (112 cores, Leonardo DCGP)

```bash
export OMP_NUM_THREADS=112
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun ./build-generic/bin/murb -n 10000 -i 200 --nv --im cpu+omp --gf
```

<details>
<summary>Full SLURM job → <code>scripts/run_cpu.sh</code></summary>

```bash
sbatch scripts/run_cpu.sh
```

Partition: `dcgp_usr_prod` – 112 CPU cores, no GPU.
</details>

### C) Single GPU (1 × A100)

```bash
srun ./build-leonardo/bin/murb -n 30000 -i 50 --nv --im gpu+tile+full --gf
```

<details>
<summary>Full SLURM job → <code>scripts/run_gpu.sh</code></summary>

```bash
sbatch scripts/run_gpu.sh
```

Partition: `boost_usr_prod` – 1 node, 1 GPU, 8 CPU cores.
</details>

### D) Multi-GPU (4 × A100, CUDA-aware MPI)

```bash
srun ./build-leonardo/bin/murb -n 500000 -i 500 --im gpu+multinode -dt 1e11 --nv -v --gf
```

<details>
<summary>Full SLURM job → <code>scripts/run_gpu_multinode.sh</code></summary>

```bash
sbatch scripts/run_gpu_multinode.sh
```

Partition: `boost_usr_prod` – 1 node, 4 GPUs (NVLink), 4 MPI ranks.
</details>

### E) Heterogeneous CPU + GPU

```bash
export MURB_HETERO_GPU_FRACTION=0.75
export OMP_NUM_THREADS=12

srun ./build-leonardo/bin/murb -n 30000 -i 60 --nv --im hetero --gf
```

The environment variable `MURB_HETERO_GPU_FRACTION` controls the fraction of bodies
offloaded to the GPU (default 0.75).

---

## 5) Simulation Replay (Binary Player)

Any GPU/CPU run that enables history tracking writes a `simulation_data.bin` file.
You can replay it locally (e.g. on macOS with OpenGL) without re-running the simulation:

```bash
# Build with visualisation (mac preset)
cmake --preset mac
cmake --build build-mac -j $(sysctl -n hw.ncpu)

# Create a symlink so the binary finds the shaders
cd build-mac && ln -sf ../src src && cd bin

# Copy simulation_data.bin from the cluster into the bin/ directory, then:
./murb -n <same_n> -i <same_i> --im bin+player
```

This opens a real-time OpenGL window replaying the recorded trajectory.

---

## 6) Validation Tests (Catch2)

```bash
# From any build directory
./bin/murb-test
```

Tests cover CPU kernel correctness, CUDA body transfers, and simulation history integrity.

---

## 7) Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_cpu.sh` | SLURM job – CPU benchmarks (DCGP, 112 cores) |
| `scripts/run_gpu.sh` | SLURM job – single-GPU run (Booster, 1 × A100) |
| `scripts/run_gpu_multinode.sh` | SLURM job – multi-GPU run (Booster, 4 × A100) |
| `scripts/run_tests.sh` | Run Catch2 test suite on cluster |
| `scripts/nbody_profiling.sh` | Nsight Compute profiling wrapper |
| `scripts/make_plots.py` | Generate performance plots from CSV |
| `scripts/plot_history_metrics.py` | Plot simulation history metrics (energy, momentum) |
| `scripts/measure_energy.py` | Energy consumption measurement |
| `scripts/parse_energy_log.py` | Parse energy logs |

---

## 8) License & Attribution (MIT)

This repository contains code derived from the **MUrB framework** developed at
**Sorbonne University, LIP6**, released under the **MIT License**.

- The full license text is provided in the `LICENSE` file.
- Copyright (c) 2023 Sorbonne University, LIP6.

If you redistribute or reuse substantial portions of this repository, keep the `LICENSE` file
and preserve the attribution above.

---

**Disclaimer** — THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
