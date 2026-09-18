# MUrB – N-Body Simulation on CPU & GPU

![MUrB demo](assets/demo.gif)

N-body gravitational simulation with a progression from a CPU reference
implementation to a single NVIDIA A100 and a four-A100 MPI implementation,
validated on the Leonardo supercomputer.

The project derives from the Sorbonne University / LIP6 MUrB project and was
extended for EuroHPC demonstrations, including the EuroHPC User Days 2026 Demo
Lab.

## Implementations

| Tag | Backend | Description |
|-----|---------|-------------|
| `cpu+naive` | CPU | Reference O(N²) implementation |
| `cpu+omp` | CPU / OpenMP | Parallel CPU implementation |
| `gpu+tile+full` | CUDA | Fully device-resident tiled implementation |
| `gpu+multinode` | CUDA + MPI | Four A100s on one Leonardo Booster node |

The historical name `gpu+multinode` is retained, but the supported topology is
one Booster node, four MPI ranks, and four A100 GPUs. It does not currently
support multi-node execution.

Additional exploratory single-node implementations are `cpu+optim`,
`cpu+simd`, `gpu+tile`, and `gpu+tile+full200k`. See
[SINGLE_NODE_BACKENDS.md](SINGLE_NODE_BACKENDS.md) for their status and scope.

## Repository Layout

```text
.
├── CMakeLists.txt             # Main build definition
├── CMakePresets.json          # Build presets
├── assets/                    # Demo media
├── log/                       # Ignored SLURM stdout/stderr (except .gitkeep)
├── trajectories/              # Ignored generated .murbtraj files (except .gitkeep)
├── lib/                       # Bundled Catch2 and MIPP dependencies
├── scripts/                   # Leonardo SLURM jobs and utilities
├── src/
│   ├── common/
│   │   ├── core/              # Body data, simulation interfaces, trajectories
│   │   ├── ogl/               # OpenGL visualization
│   │   └── utils/             # CLI parsing and performance reporting
│   ├── murb/
│   │   ├── implem/            # CPU, CUDA, and MPI implementations
│   │   └── main.cpp           # Application entry point
│   └── test/                  # Correctness and format tests
├── SINGLE_NODE_BACKENDS.md
├── MULTI_GPU.md
├── TRAJECTORY_FORMAT.md
└── LEONARDO_NOTES.md
```

## Build

Build before submitting a job. Each path uses a separate CMake preset and build
directory. The `generic` preset builds the CPU/OpenMP executable without CUDA
or MPI. The `leonardo` preset adds CUDA for one A100 but keeps MPI disabled.
`leonardo-multi` inherits the CUDA settings and enables the four-rank MPI
backend, compiling additional MPI-specific sources. Keep its separate
`build-leonardo-multi` directory: the two CUDA executables are not equivalent.
Load OpenMPI only for the four-A100 build and job.

### Leonardo CPU

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cmake/3.27.9

cmake --preset generic
cmake --build build-generic -j 32
```

### Leonardo: one A100

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load cmake/3.27.9

cmake --preset leonardo
cmake --build build-leonardo -j 32
```

### Leonardo: four A100s with MPI

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load cmake/3.27.9

cmake --preset leonardo-multi
cmake --build build-leonardo-multi -j 32
```

### macOS visualization

The `mac` preset builds the OpenGL visualizer used for trajectory replay.

```bash
cmake --preset mac
cmake --build build-mac -j $(sysctl -n hw.ncpu)
```

## Run

After building the corresponding target, submit one of the three main Leonardo
jobs from the repository root:

```bash
sbatch scripts/run_cpu.sh
sbatch scripts/run_gpu.sh
sbatch scripts/run_gpu_multinode.sh
```

| Script | Execution path |
|--------|----------------|
| `scripts/run_cpu.sh` | `cpu+naive`, single-core CPU reference |
| `scripts/run_gpu.sh` | `gpu+tile+full`, 1 × A100 |
| `scripts/run_gpu_multinode.sh` | `gpu+multinode`, 4 × A100 with 4 MPI ranks |

These job scripts use the same minimal module sets as their matching build
commands above: GCC for CPU, GCC and CUDA for one A100, and GCC, CUDA, and
OpenMPI for four A100s.

Environment variables override script defaults. Common examples are `MURB_N`
and `MURB_ITERS`; the four-A100 script also accepts `MURB_WARMUP`. For example:

```bash
MURB_N=2049 MURB_ITERS=3 MURB_WARMUP=1 \
  sbatch scripts/run_gpu_multinode.sh
```

The scripts also accept `MURB_DT`; see the script headers and the linked
documentation for path-specific controls.

All five standard jobs write stdout and stderr under
`/leonardo_work/EUHPC_TDEMO_26/NBody-EuroHPC/log/`. The tracked `log/`
directory exists before Slurm opens those files.

Normal jobs may run from a dirty worktree. They report the current Git HEAD and
executable version and emit explicit warnings for dirty, stale, or mismatched
provenance. To make any such warning fatal, set
`MURB_STRICT_PROVENANCE=1` when submitting the job.

The normal visual demos use about 10,000 bodies so individual structures remain
clear. Performance benchmarks have also been run at much larger N; see
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the measured results and methods.

## Trajectory Recording & Replay

Trajectory generation and visualization are deliberately separated:

```text
Leonardo
  compute
    ↓
 .murbtraj
    ↓
   scp
    ↓
  macOS
    ↓
OpenGL replay
```

Generate a trajectory on one A100 with the dedicated recording job:

```bash
sbatch scripts/run_gpu_record.sh
```

By default it writes under
`/leonardo_work/EUHPC_TDEMO_26/NBody-EuroHPC/trajectories/`. The generated
filename records the backend, scheme, N, iteration count, timestep, recording
stride, UTC timestamp, and Slurm job ID, for example:
`gpu-galaxy-N10000-I720-W3-dt3600-every2-20260918T113000Z-job123456.murbtraj`.
Generated files under `trajectories/` are ignored by Git.
The job refuses to overwrite an existing trajectory; move or delete the old
file, or set `MURB_OUTPUT` to another absolute `.murbtraj` path.

For example, this records 10,000 bodies every two iterations over 720 simulated
hours (30 days):

```bash
MURB_N=10000 \
MURB_ITERS=720 \
MURB_WARMUP=3 \
MURB_DT=3600 \
MURB_RECORD_EVERY=2 \
sbatch scripts/run_gpu_record.sh
```

To record a four-A100 run, provide an absolute output path to the four-GPU job:

```bash
MURB_OUTPUT=/leonardo_work/EUHPC_TDEMO_26/NBody-EuroHPC/trajectories/gpu-multinode-galaxy-N10000-I720-20260918.murbtraj \
  sbatch scripts/run_gpu_multinode.sh
```

Copy the completed trajectory from Leonardo to the Mac with `scp`, then replay
it with the visualization build:

```bash
./build-mac/bin/murb \
  --replay /path/to/demo.murbtraj \
  --visu \
  --replay-fps 10 \
  --loop
```

Replay does not recompute the physics. It visualizes the trajectory generated
on Leonardo. See [TRAJECTORY_FORMAT.md](TRAJECTORY_FORMAT.md) for the versioned
binary format. The recording job defaults to about 10,000 bodies for a readable
demo; `MURB_N` and `MURB_OUTPUT` can select a different size and destination.

## Validation

Revision `e899599e429d94feb0f76f9329e393f7ba25b59b` has been validated on
Leonardo for:

- `cpu+naive` through the CPU reference build and job;
- `gpu+tile+full` on one A100;
- `gpu+multinode` on one Booster node with four A100s and four MPI ranks;
- non-even body partitions, including N=2049 and N=2051;
- numerical agreement with the CPU reference;
- four-A100 `.murbtraj` generation; and
- replay of that trajectory on macOS.

The current four-GPU implementation prioritizes correctness. Each rank uses
explicit host-staged `MPI_Allgatherv` synchronization; CUDA-aware or GPU-direct
MPI communication is future work. Performance depends on workload and
communication costs, so four GPUs are not assumed to be faster than one.

## Documentation

- [SINGLE_NODE_BACKENDS.md](SINGLE_NODE_BACKENDS.md) — single-node backend
  status and validation.
- [MULTI_GPU.md](MULTI_GPU.md) — supported four-A100 topology, communication,
  and correctness checks.
- [TRAJECTORY_FORMAT.md](TRAJECTORY_FORMAT.md) — `.murbtraj` format details.
- [LEONARDO_NOTES.md](LEONARDO_NOTES.md) — Leonardo environment and operational
  notes.
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) — validated performance results,
  campaign details, and measurement caveats.

## License & Attribution

This repository contains code derived from the MUrB framework developed at
Sorbonne University, LIP6, and released under the MIT License. The project was
extended for EuroHPC demonstrations; it was not written entirely from scratch.

See [LICENSE](LICENSE). Preserve the original license and attribution when
redistributing the code.
