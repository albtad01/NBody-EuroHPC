# Single-node backend exploration

This report records the scope and results of branch
`explore/single-node-backends`, based on the validated Phase 1 commit
`2549356c34f2826376cc0ec816149c8aef57b885`. The primary demo comparison
remains `cpu+omp` versus `gpu+tile+full`.

## Validation method

Each enabled exploratory backend was built headlessly and compared with an
independent `cpu+naive` simulation using identical initial conditions. Tests
cover 2,048 and 2,049 bodies so CUDA and SIMD boundary handling is exercised.
They compare every position and velocity component after multiple updates for
both random and galaxy initial conditions. The test uses single precision, a
3,600-second timestep, and the existing 200,000,000 softening value.

The random cases use a scaled tolerance of `1e-3`; the galaxy cases use
`1e-1` because the state magnitudes and accumulated floating-point ordering
differences are larger. Each backend completed 184,368 assertions.

Fresh headless generic CPU and Leonardo CUDA `sm_80` builds passed. The CUDA
tests ran on an NVIDIA A100-SXM-64GB. No CUDA errors or unexpected output files
were observed, and the worktree remained clean.

## Results

| Backend | Status | Build | Runtime | Numerical validation | Performance interest | Demo usefulness | Required fixes | Recommendation |
|---|---|---|---|---|---|---|---|---|
| `cpu+naive` | SUPPORTED | Pass | Pass | Trusted reference | Low | Essential reference | None | Keep |
| `cpu+omp` | SUPPORTED | Pass | Pass | 184,368 assertions pass | High | Primary CPU target | None found | Keep |
| `gpu+tile+full` | SUPPORTED | Pass | Pass on A100 | Validated in Phase 1 | High | Primary GPU target | None found | Keep |
| `cpu+optim` | PROMISING | Pass | Smoke pass | 184,368 assertions pass | Medium | Scalar optimization step | None found | Keep for explanation |
| `cpu+simd` | PROMISING | Pass | Smoke pass | 184,368 assertions pass | High | Vectorization step | Check compiler and ISA portability | Keep for explanation |
| `gpu+tile` | PROMISING | Pass | Smoke pass on A100 | 184,368 assertions pass | High | CUDA tiling step | Account for per-iteration transfers | Keep for explanation |
| `gpu+tile+full200k` | EXPERIMENTAL | Pass | Smoke pass on A100 | 184,368 assertions pass | Medium | Limited before large-N validation | Test near intended size | Explore later |
| `hetero` | EXPERIMENTAL | Not enabled | Not run | Not run | Medium | Weak before event | Enable OpenMP in CUDA host compile; validate split ratios | Defer |
| `gpu+tracking` | BROKEN | Not enabled | Not run | Not run | Different workload | Diagnostic feature only | Fix buffer sizing and lifecycle/error handling | Defer repair |
| `gpu+leapfrog` | BROKEN | Not enabled | Not run | Not run | Different integrator | Low before event | Fix constructor wiring and storage; add invariant tests | Defer repair |
| CPU MPI | DEFER | Not enabled | Not run | Not run | Low for this phase | Low for single-node demo | Centralize MPI lifecycle; add rank tests | Defer |
| `bin+player` | DEFER | Not enabled | Not run | Not run | None for benchmark | Useful in Phase 1b | Define format and correct EOF behavior | Defer |
| Barnes-Hut | DEFER | Absent | Not applicable | Not applicable | Algorithmic comparison | Low before event | No implementation in this commit | Do not reconstruct now |

## Small smoke measurements

These measurements only prove that the backends execute and emit plausible
timing data. They are too small and too short for performance ranking.

CPU node `lrdn4838`, 10,000 bodies, 3 iterations:

- `cpu+optim`: 607.203 ms compute time, 494.1 million interactions/s,
  estimated 9.88 GFLOP/s.
- `cpu+simd`: 144.453 ms compute time, 2.077 billion interactions/s,
  estimated 41.54 GFLOP/s.

A100 node `lrdn0641`, 10,000 bodies, 5 iterations:

- `gpu+tile`: 3.213 ms compute time, 155.6 billion interactions/s,
  estimated 3,112.02 GFLOP/s.
- `gpu+tile+full200k`: 8.967 ms compute time, 55.76 billion interactions/s,
  estimated 1,115.14 GFLOP/s.

`gpu+tile` copies positions to the device and accelerations back to the host on
every iteration, then updates bodies on the CPU. Its strong result in this tiny
case should not be extrapolated to longer runs. `gpu+tile+full200k` uses a
launch layout intended for a much larger body count, so 10,000 bodies do not
occupy the A100 well.

## Tier B audit details

### Heterogeneous CPU and GPU

`SimulationNBodyHetero.cu` protects its CPU parallel loops with `_OPENMP`, but
the CUDA compiler flags do not pass `-fopenmp` to the host compiler. The CPU
partition is therefore serial in the Leonardo build. Its default minimum size
also sends 2,048- and 2,049-body tests through the CPU-only fallback, which
would fail to exercise the heterogeneous path. A credible repair needs the
compile flag plus correctness and overlap measurements across split ratios.

### Property tracking

`SimulationNBodyCUDAPropertyTracking.cu` declares `bufferForEnergy` as `Q*`
but allocates `n * sizeof(T)`. The historical `<float, double>` instantiation
therefore underallocates the buffer. The backend also adds a second
O(N-squared) metrics kernel, a CUB reduction, and history transfers each
iteration. It must be presented as a diagnostic workload, not compared
directly with force-only implementations.

### Leapfrog

`SimulationNBodyCUDALeapfrog` expects constructor arguments in the order
`NIterations, softening`, while the historical factory passed
`softening, NIterations`. The values are silently reversed. It has the same
`Q*` allocated with `sizeof(T)` defect as property tracking, and its final
velocity update is explicitly approximate. This backend needs integrator
specific conservation tests rather than equality with the Euler reference.

### CPU MPI

`SimulationNBodyMultiNode.cpp` may call `MPI_Init`, but the repository has no
matching `MPI_Finalize`. The implementation also allocates fresh acceleration
gather buffers every iteration and does not check MPI errors. MPI ownership
belongs at the executable boundary, with rank-aware output and one- and
two-rank numerical tests.

### Binary replay

`SimulationNBodyBinaryPlayer.cpp` opens its input with `std::ifstream`, so it
does not truncate it. It does not validate header reads, format identity,
precision, endianness, or complete frames. An EOF during a frame can leave
partial or stale state before the stream is rewound. Recording remains disabled
by default, and replay should stay in Phase 1b.

## Recommendation

Keep `cpu+omp` and `gpu+tile+full` as the measured Demo Lab comparison. The
three most useful explanatory additions are `cpu+optim`, `cpu+simd`, and
`gpu+tile`, all of which pass the small numerical matrix. Leave
`gpu+tile+full200k` visible only for further experimentation at its intended
problem size. Do not enable any Tier B backend before the event.
