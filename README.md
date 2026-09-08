# MUrB: N-Body Simulation on CPU and GPU

MUrB is an all-pairs gravitational N-body simulation derived from the Sorbonne
University PACC project and extended for EuroHPC demonstrations. This branch is
a small stabilization baseline for CPU execution and one NVIDIA A100 on the
Leonardo supercomputer.

## Phase 1 support

The command-line demo supports these backends:

- `cpu+naive`: single-threaded CPU reference implementation.
- `cpu+omp`: OpenMP CPU implementation.
- `gpu+tile+full`: tiled CUDA implementation with all body state resident on
  one device.

This exploratory branch also makes `cpu+optim`, `cpu+simd`, `gpu+tile`, and
`gpu+tile+full200k` selectable for small correctness and smoke tests. They do
not replace the three Phase 1 demo backends. Their validation status and known
limits are recorded in [SINGLE_NODE_BACKENDS.md](SINGLE_NODE_BACKENDS.md).

Multi-GPU execution, trajectory recording, and `bin+player` replay remain
deferred. Their source files are preserved but excluded from the supported
single-node build.

Leonardo benchmark jobs are headless. A local build may still enable the
existing OpenGL visualization when OpenGL, GLEW, GLM, and GLFW are available.

## Requirements

- CMake 3.21 or newer.
- A C++20 compiler.
- OpenMP for `cpu+omp`.
- CUDA 12 and compute capability 8.0 for the Leonardo A100 build.
- OpenGL, GLEW, GLM, and GLFW only for an optional local visualization build.

MIPP and Catch2 are included under `lib/`.

## Build on Leonardo

Build once before submitting a job. The job scripts use the existing binary;
they never configure, rebuild, or remove a build directory.

For a headless CPU build:

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cmake/3.27.9
cmake --preset generic
cmake --build build-generic -j 32
```

For a headless A100 build:

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load cmake/3.27.9
cmake --preset leonardo
cmake --build build-leonardo -j 32
```

The `generic` preset enables OpenMP and disables CUDA and visualization. The
`leonardo` preset enables OpenMP and CUDA, targets `sm_80`, and disables
visualization.

A successful build creates `murb-build.ready`. The executable reports its
compiled source identity:

```bash
./build-generic/bin/murb --version
```

The SLURM scripts require the executable revision to match the checked-out
revision and reject binaries built from dirty tracked sources. Rebuild after
each source commit before submitting.

## Command line

The Phase 1 syntax is:

```text
murb -n BODIES -i ITERATIONS --im BACKEND [--warmup ITERATIONS] [--nv] [--gf] [--dt SECONDS]
```

Options used by the demo are:

- `-n`: positive number of bodies.
- `-i`: positive number of iterations.
- `--im`: a supported Phase 1 backend tag, or one of the four exploratory tags
  listed above on this branch.
- `--nv`: explicitly select headless operation.
- `--gf`: report an estimated GFLOP/s value using 20 operations per
  interaction.
- `--dt`: finite, positive time step in seconds.
- `--warmup`: optional positive number of untimed iterations performed in the
  same process before measurement.

`--scheme galaxy|random`, `-v`, `--help`, and `--version` are also accepted.
Unknown, duplicate, missing, and invalid options cause a clear nonzero exit.

The default mode is headless and does not record a trajectory. A normal run
does not create or overwrite `simulation_data.bin`. `--visu` is accepted only
by a build in which the OpenGL dependencies were found and visualization was
compiled.

## Run the Phase 1 jobs

From the root of a clean checkout with matching prebuilt binaries, submit the
CPU reference job with:

```bash
sbatch --account=EUHPC_TDEMO_26_0 scripts/run_cpu.sh
```

Submit the single-A100 job with:

```bash
sbatch --account=EUHPC_TDEMO_26 scripts/run_gpu.sh
```

The CPU script requests one node, one task, and one CPU. The GPU script
requests one Booster node, one task, eight CPUs, and one GPU. SLURM constrains
the task to one visible GPU; the program validates that a CUDA device exists
and selects visible device zero before allocating device memory.

The default workload is 10,000 bodies, 20 iterations, and a 3,600-second time
step. Override it at submission time, for example:

```bash
MURB_N=2048 MURB_ITERS=4 MURB_DT=3600 sbatch scripts/run_gpu.sh
```

The executable prints the backend, problem size, compiled revision, CUDA
device identity when applicable, elapsed compute time, loop wall time,
interactions per second, and optional estimated GFLOP/s.

## Validation

Run CPU correctness tests after building with tests enabled:

```bash
./build-generic/bin/murb-test "[correctness]"
```

On one allocated A100, the same filter compares `gpu+tile+full` with the
trusted `cpu+naive` implementation for random and galaxy inputs, including
body counts that are not exact CUDA block multiples:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
  ./build-leonardo/bin/murb-test "[correctness]"
```

That GPU validation must run in a scheduled allocation, not on a login node.

## Optional local visualization

Visualization is outside the Leonardo benchmark path but remains available.
Configure with `-DENABLE_VISU=ON` on a machine that provides OpenGL, GLEW, GLM,
and GLFW. If those dependencies are absent, CMake produces a headless build and
`--visu` reports that visualization was not compiled.

The existing `mac` preset requests visualization:

```bash
cmake --preset mac
cmake --build build-mac
```

Trajectory recording and `bin+player` are reserved for Phase 1b. They are not
selectable in this branch, and no file format compatibility is promised by the
Phase 1 benchmark workflow.

## Deferred implementations

The following paths are not supported or validated by this stabilization:

- MPI and `gpu+multinode`.
- `gpu+tracking` and `gpu+leapfrog`.
- Heterogeneous CPU/GPU execution.
- Barnes-Hut, OpenCL, CADNA, and other experimental kernels.
- Binary trajectory recording and `bin+player` replay.

Do not use `scripts/run_gpu_multinode.sh` for the Phase 1 demo.

## License and attribution

This repository contains code derived from the MUrB framework developed at
Sorbonne University, LIP6, and released under the MIT License. Keep the
repository `LICENSE` file and its attribution when redistributing the code.
