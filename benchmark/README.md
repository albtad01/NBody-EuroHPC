# Reproducible scaling benchmark suite

This directory adds benchmark orchestration only. It does not change numerical
algorithms, CUDA kernels, MPI communication, trajectory data, build presets, or
the existing validated run scripts. Normal runs are headless (`--nv`) and never
pass `--record`.

## Scope and matrix

The primary matrix in `matrix.json` contains only:

- `cpu+naive` (one thread), capped at N=20,000;
- `cpu+omp` (all CPUs assigned to its Slurm task);
- `gpu+tile+full` (one A100); and
- `gpu+multinode` (the historical tag for exactly four ranks/four A100s on one
  Leonardo Booster node).

The four-GPU implementation is **not** a validated multi-node implementation.
It retains explicit host-staged `MPI_Allgatherv`. The exploratory backends are
listed separately in the configuration and are not runnable by the primary
batch jobs.

Each data point is a fresh process invocation. Defaults are three untimed
warm-up iterations and five independent repetitions. Iteration counts are
larger at small N to avoid clock-resolution-dominated measurements and smaller
at large N. Adjustments are possible through `MURB_N_LIST` (a subset of the
configured sizes) and `MURB_REPETITIONS`; record deviations with the campaign.

## Review-first workflow

Build outside batch jobs with the unchanged presets described in the main
README. Then create one immutable campaign directory from a clean committed
worktree:

```bash
CAMPAIGN=$(python3 benchmark/benchmark.py init)
export MURB_CAMPAIGN_DIR="$CAMPAIGN"
```

Run the lightweight correctness gate before performance jobs:

```bash
sbatch scripts/benchmark_correctness_gate.sh
```

It reuses the existing Catch2 tests: the representative single-rank tests cover
N=2048 and 2049, while the exact four-rank test also covers N=2051. Review its
external Slurm logs before continuing.

The production batch scripts deliberately refuse to run until approval is made
explicit. After review only:

```bash
export MURB_BENCHMARK_APPROVED=YES
sbatch --export=ALL scripts/benchmark_cpu.sh
sbatch --export=ALL scripts/benchmark_gpu.sh
```

The GPU job runs both the one-A100 and four-A100 measurements in the same
four-GPU allocation/node. The one-GPU steps bind only GPU 0; the four-GPU steps
use the validated `map_gpu:0,1,2,3` mapping. Once all jobs finish:

```bash
python3 benchmark/benchmark.py aggregate --campaign "$CAMPAIGN"
```

## Output and concurrency safety

The default path is
`/leonardo_work/EUHPC_TDEMO_26/benchmark-results/<git-sha>/<UTC-timestamp>/`.
Campaign creation refuses a path inside the repository; execution and
aggregation verify the canonical path against campaign metadata. Every
invocation gets a UUID-bearing run ID, its own directory, stdout, stderr, GPU
sample file, and JSON record. The one-A100 path also retains CUDA visibility
probe logs. Aggregate once after all jobs finish: an exclusive lock and
pre-existing-artifact check refuse concurrent or repeated aggregation rather
than overwriting CSV, Markdown, or plots. Use a new campaign for another run.

```text
metadata.json
raw/<run-id>.json
gpu_samples/<run-id>.csv
runs/<run-id>/{stdout.txt,stderr.txt,record.json}
results.csv
summary.csv
summary.md
plots/
```

`results.csv` is the per-invocation table. `summary.csv` reports median, min,
max, sample standard deviation, and IQR. Speedup versus `cpu+naive` is emitted
only at matching N. Four-versus-one A100 speedup and efficiency (`speedup/4`)
are emitted only for paired backend/N summaries. `summary.md` reports the
measured least-squares slope of log(median time/iteration) against log(N).
Simulation throughput is named `simulation_steps_per_second`, never rendering
FPS.

## Measurement interpretation

- `compute_ms` comes from the executable and excludes warm-up, initialization,
  recording, visualization, and aggregation. CUDA iterations synchronize before
  the timer stops. `wall_clock_seconds` covers the whole invocation.
- GNU `time` is placed around each rank by `srun`; the largest per-task MaxRSS is
  reported. It is not aggregate node memory. If GNU `time` is absent, RSS is
  left unavailable rather than guessed.
- `nvidia-smi` samples board utilization, memory, and power every 200 ms. Peak
  GPU memory is the sum of each selected GPU's peak increase above its first
  sample, so it remains an external approximation. Very short runs can have
  insufficient samples.
- Power samples span process initialization, warm-up, and timed work. The suite
  may report mean/peak invocation-window power and an explicitly scoped
  invocation-window energy estimate, but intentionally leaves timed-region
  energy and energy/iteration blank. Inferring those would be misleading
  without an in-process timing marker.
- Sampling and `/usr/bin/time` have observer overhead. Keep sampling settings
  identical across compared GPU runs and preserve the raw files.
- CPU frequency/turbo, NUMA placement, thermal state, Slurm placement, and node
  variation can affect results. The scripts bind cores and preserve host/module
  metadata, but repetitions and medians remain essential.
- The `N*N*iterations` interaction count and 20 FLOPs/interaction estimate are
  the repository's existing model, not hardware-counter measurements.
- The four-GPU summary is rank-zero timing of the complete synchronous iteration,
  including host-staged collectives. It must not be described as multi-node or
  GPU-direct performance.

## Tiny development smoke

For parser/output validation only, a locally built generic executable can run:

```bash
SMOKE=$(python3 benchmark/benchmark.py init --results-root /tmp/nbody-benchmark-smoke --allow-dirty)
python3 benchmark/benchmark.py run-group --campaign "$SMOKE" \
  --binary build-generic/bin/murb --group cpu --backend cpu+naive \
  --local --smoke --allow-dirty
python3 benchmark/benchmark.py aggregate --campaign "$SMOKE"
```

This executes N=32 for two timed iterations and is not benchmark evidence.
