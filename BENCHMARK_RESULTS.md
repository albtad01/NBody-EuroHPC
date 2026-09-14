# NBody-EuroHPC Benchmark Results on Leonardo

This document summarizes the reproducible CPU and GPU scaling measurements obtained on the Leonardo supercomputer for the NBody-EuroHPC project.

The benchmark harness is maintained on the `codex/benchmark-suite` branch. The numerical implementation itself is unchanged by the benchmark suite: the benchmark code only orchestrates builds, Slurm execution, repeated measurements, telemetry collection, and aggregation.

## Benchmark configuration

All reported measurements use:

- initial-condition scheme: `galaxy`;
- floating-point precision: FP32;
- time step: `dt = 3600 s`;
- three untimed warm-up iterations;
- five independent process repetitions per data point;
- headless execution (`--nv`);
- no trajectory recording during performance measurements;
- median time per iteration as the primary reported statistic.

The four-GPU backend is still named `gpu+multinode` for historical reasons. In the measurements reported here it runs as **four MPI ranks on four NVIDIA A100-SXM-64GB GPUs within one Leonardo Booster node**. It is not a multi-node result and does not use GPU-direct communication. The implementation uses explicit host staging and `MPI_Allgatherv`.

The estimated GFLOP/s values use the repository model of 20 floating-point operations per body-body interaction. They are algorithmic estimates rather than hardware-counter measurements.

## Physical simulation horizon

The executable defines the default time step as:

```text
dt = 3600 s = 1 simulated hour
```

There is no separate fixed final-time variable `T`. The simulation advances one `dt` at every call to `computeOneIteration()`.

For `I` timed iterations:

```text
T_timed = I * dt
```

Since the benchmark also executes three warm-up iterations before the timed section, the state at the end of a benchmark invocation has advanced by:

```text
T_total = (3 + I) * dt
```

from the initial condition.

The warm-up iterations are **not included in `compute_ms`**, but they do advance the physical simulation state.

Therefore, because the benchmark uses a different number of timed iterations at different problem sizes, there is no single common `T` across the benchmark matrix.

| Backend | N | Timed iterations | Timed horizon | Total horizon including 3 warm-ups |
|---|---:|---:|---:|---:|
| `cpu+naive` | 1,000 | 200 | 200 h | 203 h |
| `cpu+naive` | 2,000 | 100 | 100 h | 103 h |
| `cpu+naive` | 5,000 | 50 | 50 h | 53 h |
| `cpu+naive` | 10,000 | 20 | 20 h | 23 h |
| `cpu+naive` | 20,000 | 10 | 10 h | 13 h |
| `cpu+omp` | 1,000 | 500 | 500 h | 503 h |
| `cpu+omp` | 2,000 | 500 | 500 h | 503 h |
| `cpu+omp` | 5,000 | 200 | 200 h | 203 h |
| `cpu+omp` | 10,000 | 100 | 100 h | 103 h |
| `cpu+omp` | 20,000 | 50 | 50 h | 53 h |
| `cpu+omp` | 50,000 | 20 | 20 h | 23 h |
| `cpu+omp` | 100,000 | 10 | 10 h | 13 h |
| `cpu+omp` | 200,000 | 5 | 5 h | 8 h |
| `cpu+omp` | 500,000 | 3 | 3 h | 6 h |
| `gpu+tile+full` | 1,000 | 1,000 | 1,000 h | 1,003 h |
| `gpu+tile+full` | 2,000 | 1,000 | 1,000 h | 1,003 h |
| `gpu+tile+full` | 5,000 | 500 | 500 h | 503 h |
| `gpu+tile+full` | 10,000 | 200 | 200 h | 203 h |
| `gpu+tile+full` | 20,000 | 100 | 100 h | 103 h |
| `gpu+tile+full` | 50,000 | 50 | 50 h | 53 h |
| `gpu+tile+full` | 100,000 | 20 | 20 h | 23 h |
| `gpu+tile+full` | 200,000 | 10 | 10 h | 13 h |
| `gpu+tile+full` | 500,000 | 5 | 5 h | 8 h |
| `gpu+multinode` | 1,000 | 1,000 | 1,000 h | 1,003 h |
| `gpu+multinode` | 2,000 | 1,000 | 1,000 h | 1,003 h |
| `gpu+multinode` | 5,000 | 500 | 500 h | 503 h |
| `gpu+multinode` | 10,000 | 200 | 200 h | 203 h |
| `gpu+multinode` | 20,000 | 100 | 100 h | 103 h |
| `gpu+multinode` | 50,000 | 50 | 50 h | 53 h |
| `gpu+multinode` | 100,000 | 20 | 20 h | 23 h |
| `gpu+multinode` | 200,000 | 10 | 10 h | 13 h |
| `gpu+multinode` | 500,000 | 5 | 5 h | 8 h |

These horizons are a property of the benchmark configuration, not a statement that all backends were integrated to a common physical final time. The varying iteration counts were selected to obtain stable timing measurements without making large-N runs unnecessarily expensive.

## Validation

Before production measurements, the correctness gate was run on the same source revision used by the benchmark binaries.

The validation covered:

- `cpu+naive`;
- `cpu+omp`;
- `gpu+tile+full`;
- the four-rank/four-GPU `gpu+multinode` backend;
- representative body counts including 2,048, 2,049, and 2,051 bodies, so the MPI path was also tested for body counts that are not evenly divisible by four.

The final baseline campaign completed with:

```text
145 raw records
145 successful records
0 failed records
0 timed-out records
```

The subsequent large-N campaign completed with:

```text
45 raw records
45 successful records
0 failed records
0 timed-out records
```

Both Slurm jobs in the large-N campaign completed with exit code `0:0`.

## Campaign provenance

### Baseline scaling campaign

Benchmark source revision:

```text
4a44fe3ef468497020d7c62d6531542fb5ba1967
```

Campaign directory on Leonardo:

```text
/leonardo_work/EUHPC_TDEMO_26/benchmark-results/4a44fe3ef468497020d7c62d6531542fb5ba1967/20260914T085807Z
```

This campaign covered the original validated matrix through 200,000 bodies.

### Large-N extension

Benchmark source revision:

```text
54c9b355beab62de64d0520b7f6114415ce4db15
```

Campaign directory on Leonardo:

```text
/leonardo_work/EUHPC_TDEMO_26/benchmark-results/54c9b355beab62de64d0520b7f6114415ce4db15/20260914T093102Z
```

This campaign measured `N = 100,000`, `200,000`, and `500,000` for:

- `cpu+omp`;
- one A100 (`gpu+tile+full`);
- four A100s (`gpu+multinode`).

For the combined table below, the baseline campaign is used through 50,000 bodies and the large-N campaign is used from 100,000 bodies onward.

## Combined performance summary

Median timed milliseconds per simulation iteration are reported below.

| N | CPU naive [ms/iter] | CPU OpenMP [ms/iter] | 1x A100 [ms/iter] | 4x A100 [ms/iter] | 4-GPU vs 1-GPU speedup | 4-GPU parallel efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 10.904 | 0.0626 | 0.1720 | 0.2779 | 0.619x | 15.5% |
| 2,000 | 43.630 | 0.1461 | 0.3216 | 0.4388 | 0.733x | 18.3% |
| 5,000 | 273.067 | 0.6389 | 0.7870 | 0.9152 | 0.860x | 21.5% |
| 10,000 | 1,092.769 | 2.237 | 1.583 | 1.774 | 0.893x | 22.3% |
| 20,000 | 4,361.815 | 8.288 | 3.169 | 3.397 | 0.933x | 23.3% |
| 50,000 | — | 52.387 | 7.461 | 8.050 | 0.927x | 23.2% |
| 100,000 | — | 219.191 | 15.363 | 16.078 | 0.956x | 23.9% |
| 200,000 | — | 1,020.797 | 56.303 | 31.714 | **1.775x** | **44.4%** |
| 500,000 | — | 6,722.236 | 280.069 | 74.659 | **3.751x** | **93.8%** |

The `cpu+naive` backend was intentionally capped at 20,000 bodies because its direct reference implementation becomes prohibitively expensive at larger sizes.

## Large-N throughput

The large-N campaign produced the following median throughput estimates.

| N | CPU OpenMP [estimated GFLOP/s] | 1x A100 [estimated GFLOP/s] | 4x A100 [estimated GFLOP/s] |
|---:|---:|---:|---:|
| 100,000 | 912.45 | 13,018.49 | 12,439.31 |
| 200,000 | 783.70 | 14,208.79 | 25,225.15 |
| 500,000 | 743.80 | 17,852.74 | 66,971.16 |

At 500,000 bodies this corresponds to approximately:

```text
CPU OpenMP : 0.744 estimated TFLOP/s
1x A100    : 17.85 estimated TFLOP/s
4x A100    : 66.97 estimated TFLOP/s
```

Again, these are based on the repository's analytical FLOP model and must not be presented as hardware-counter measurements.

## Observed scaling

### Baseline campaign

The empirical log-log slopes of median time per iteration versus `N` were:

| Backend | Empirical exponent |
|---|---:|
| `cpu+naive` | 2.0003 |
| `cpu+omp` | 1.8513 |
| `gpu+tile+full` | 1.0447 |
| `gpu+multinode` | 0.9065 |

The `cpu+naive` result is effectively the expected quadratic scaling of a direct all-pairs N-body method.

The lower exponents measured on the GPU paths over the original range should not be interpreted as a change in algorithmic complexity. At small and medium problem sizes, GPU occupancy and fixed launch/communication overheads change substantially with `N`, so effective throughput increases while the problem is growing.

### Large-N campaign

Over the three-point `100k-500k` subset:

| Backend | Empirical exponent |
|---|---:|
| `cpu+omp` | 2.1233 |
| `gpu+tile+full` | 1.8010 |
| `gpu+multinode` | 0.9530 |

The single-A100 exponent moves much closer to the expected quadratic regime at large `N`.

The four-A100 exponent is strongly affected by the fact that this range crosses the multi-GPU efficiency transition: parallel efficiency increases from about 24% at 100,000 bodies to almost 94% at 500,000 bodies. The three-point exponent therefore describes the observed performance regime, not the asymptotic complexity of the underlying algorithm.

## Performance interpretation

### CPU reference scaling

`cpu+naive` follows the expected `O(N^2)` behavior almost exactly, with a fitted exponent of 2.0003. This provides a useful reference for the direct all-pairs workload.

`cpu+omp` is substantially faster than the naive backend. The ratio should not be interpreted as pure OpenMP strong scaling because these are distinct implementation backends rather than the same serial kernel executed with one and many threads.

### CPU-to-GPU crossover

At very small `N`, the OpenMP CPU backend is faster because there is not enough work to amortize GPU launch and device-management overhead.

The single A100 becomes faster than the OpenMP backend between 5,000 and 10,000 bodies.

At large `N`, the difference becomes substantial:

| N | 1-A100 speedup vs CPU OpenMP | 4-A100 speedup vs CPU OpenMP |
|---:|---:|---:|
| 100,000 | 14.27x | 13.63x |
| 200,000 | 18.13x | 32.19x |
| 500,000 | 24.00x | 90.04x |

At 500,000 bodies, one timed iteration takes approximately:

```text
CPU OpenMP : 6.72 s
1x A100    : 0.280 s
4x A100    : 0.0747 s
```

### One A100 versus four A100s

The most relevant result is the multi-GPU crossover.

Up to 100,000 bodies, the four-GPU path is slightly slower than the single-A100 implementation. This is expected for the current implementation: four ranks introduce MPI synchronization, explicit device-to-host/host-to-device staging, and `MPI_Allgatherv` communication every iteration.

The crossover occurs between 100,000 and 200,000 bodies:

```text
N = 100,000 : 0.956x  -> four GPUs are still slightly slower
N = 200,000 : 1.775x  -> four GPUs are clearly faster
N = 500,000 : 3.751x  -> close to the ideal 4x limit
```

At 500,000 bodies, the measured parallel efficiency is approximately:

```text
3.751 / 4 = 93.8%
```

This is consistent with the expected behavior of the implementation. At small `N`, communication and synchronization dominate. As `N` increases, the `O(N^2)` force calculation becomes dominant and there is enough computation per rank to amortize the host-staged collective communication.

The result should still be described as **single-node four-GPU scaling**, not multi-node scaling.

## Reproducing the benchmark on Leonardo

The benchmark harness is not currently on the main branch. Start by switching to the benchmark branch:

```bash
git fetch origin
git switch codex/benchmark-suite
git pull --ff-only origin codex/benchmark-suite
```

For exact provenance of the large-N measurements reported here:

```bash
git rev-parse HEAD
```

should resolve to, or include as an ancestor:

```text
54c9b355beab62de64d0520b7f6114415ce4db15
```

### 1. Build the three configurations

From the repository root:

```bash
export MURB_ROOT="$PWD"
```

Generic CPU build:

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cmake/3.27.9

cmake --preset generic
cmake --build build-generic -j 32
```

Single-A100 CUDA build:

```bash
module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load cmake/3.27.9

cmake --preset leonardo
cmake --build build-leonardo -j 32
```

Four-A100 CUDA+MPI build:

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

The binaries embed the Git revision, so rebuild after switching benchmark commits.

### 2. Run the correctness gate

Initialize a campaign directory:

```bash
export MURB_CAMPAIGN_DIR="$(python3 -B benchmark/benchmark.py init)"
```

Then submit the gate:

```bash
GATE_JOB_ID="$(sbatch --parsable --wait --export=ALL scripts/benchmark_correctness_gate.sh)"
GATE_JOB_ID="${GATE_JOB_ID%%;*}"

cat "/leonardo_work/EUHPC_TDEMO_26/benchmark-results/correctness_${GATE_JOB_ID}.out"
cat "/leonardo_work/EUHPC_TDEMO_26/benchmark-results/correctness_${GATE_JOB_ID}.err"
```

Do not start production measurements unless the correctness tests pass.

### 3. Reproduce the complete current matrix

The current benchmark matrix on `codex/benchmark-suite` includes the 500,000-body extension for `cpu+omp`, `gpu+tile+full`, and `gpu+multinode`; `cpu+naive` remains capped at 20,000.

Create a fresh production campaign:

```bash
export MURB_CAMPAIGN_DIR="$(python3 -B benchmark/benchmark.py init)"

unset MURB_N_LIST
unset MURB_REPETITIONS
export MURB_BENCHMARK_APPROVED=YES
```

Submit the standard CPU and GPU jobs:

```bash
CPU_JOB_ID="$(sbatch --parsable --export=ALL scripts/benchmark_cpu.sh)"
GPU_JOB_ID="$(sbatch --parsable --export=ALL scripts/benchmark_gpu.sh)"

echo "CPU job: $CPU_JOB_ID"
echo "GPU job: $GPU_JOB_ID"
```

With the current matrix this executes:

- 25 `cpu+naive` invocations;
- 45 `cpu+omp` invocations;
- 45 one-A100 invocations;
- 45 four-A100 invocations;

for **160 independent benchmark invocations** in total.

Monitor the jobs:

```bash
squeue --me

sacct -j "${CPU_JOB_ID},${GPU_JOB_ID}" \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

Aggregate only after both top-level jobs have completed with `COMPLETED 0:0`:

```bash
python3 -B benchmark/benchmark.py aggregate \
  --campaign "$MURB_CAMPAIGN_DIR"
```

Inspect the outputs:

```bash
cat "$MURB_CAMPAIGN_DIR/summary.md"
column -s, -t < "$MURB_CAMPAIGN_DIR/summary.csv" | less -S
```

The campaign directory contains:

```text
metadata.json
raw/
gpu_samples/
runs/
results.csv
summary.csv
summary.md
plots/
```

### 4. Reproduce only the large-N campaign

To repeat only the `100k`, `200k`, and `500k` measurements:

```bash
export MURB_CAMPAIGN_DIR="$(python3 -B benchmark/benchmark.py init)"
export MURB_N_LIST=100000,200000,500000
export MURB_REPETITIONS=5
export MURB_BENCHMARK_APPROVED=YES
```

The stock GPU benchmark script can be used directly:

```bash
GPU_JOB_ID="$(sbatch --parsable --export=ALL scripts/benchmark_gpu.sh)"
```

For the CPU side, the large-N campaign must select only `cpu+omp`, because `cpu+naive` is intentionally not configured for these sizes. The benchmark runner already supports backend selection, so no repository modification is required.

A minimal Slurm wrapper is:

```bash
CPU_JOB_ID="$(sbatch --parsable --export=ALL <<'EOF'
#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --job-name=nbody_large_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_large_cpu_%j.out
#SBATCH --error=/leonardo_work/EUHPC_TDEMO_26/benchmark-results/slurm_large_cpu_%j.err

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0

python3 -B "$MURB_ROOT/benchmark/benchmark.py" run-group \
    --campaign "$MURB_CAMPAIGN_DIR" \
    --repo "$MURB_ROOT" \
    --binary "$MURB_ROOT/build-generic/bin/murb" \
    --group cpu \
    --backend cpu+omp
EOF
)"
```

This produces 15 CPU OpenMP records and 30 GPU records, for 45 records in total.

Again, aggregate only after both jobs finish successfully:

```bash
sacct -j "${CPU_JOB_ID},${GPU_JOB_ID}" \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

python3 -B benchmark/benchmark.py aggregate \
  --campaign "$MURB_CAMPAIGN_DIR"
```

## Measurement caveats

The following points should be preserved when presenting or comparing the results:

1. `compute_ms` measures only the timed simulation iterations. Initialization, the three warm-up iterations, recording, visualization, and aggregation are excluded.
2. CUDA iterations explicitly synchronize before the timing interval is closed.
3. The GPU power sampler covers the process invocation rather than precisely the timed compute region. Timed-region energy is therefore intentionally not reported.
4. GPU memory reported by the benchmark is an external sampled approximation.
5. The four-A100 backend uses host-staged MPI collectives and is restricted here to one Booster node.
6. `estimated_GFLOP_per_second` comes from the analytical 20-FLOP/interaction model.
7. CPU and GPU measurements were run under Slurm on different Leonardo node types. Results should be interpreted as backend/system performance measurements, not as a pure instruction-level comparison.
8. Empirical scaling exponents depend on the fitted `N` interval. They summarize the measured regime and should not be confused with the algorithmic complexity of the direct N-body method.

## Conclusion

The measurements show three distinct performance regimes.

For small problems, CPU OpenMP is competitive because GPU launch and communication overheads dominate. The single-A100 backend becomes preferable between approximately 5,000 and 10,000 bodies. The four-A100 implementation requires a substantially larger workload before the communication cost is amortized: it remains slightly slower than one A100 at 100,000 bodies, crosses over by 200,000 bodies, and reaches a 3.75x speedup with 93.8% four-GPU parallel efficiency at 500,000 bodies.

The large-N results are particularly relevant for the current host-staged MPI design. They show that the multi-GPU implementation can approach ideal four-GPU scaling once the direct `O(N^2)` force calculation is large enough to dominate synchronization and collective-communication overhead. At the same time, the low- and medium-N measurements make clear why four GPUs should not be used indiscriminately for smaller problems.

Taken together, the benchmark provides a reproducible view of the transition from CPU-efficient small workloads, to single-GPU acceleration, to effective four-GPU scaling on a single Leonardo Booster node.
