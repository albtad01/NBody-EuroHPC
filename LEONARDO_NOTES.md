# Leonardo Phase 1 validation notes

The supported baseline is `cpu+naive`, `cpu+omp`, and single-device
`gpu+tile+full`. All benchmark jobs are headless and recording is disabled.

Before submitting either script:

1. Check out a clean committed revision.
2. Load the module stack documented in `README.md`.
3. Configure from a fresh build directory with the matching preset.
4. Complete the build and verify that `murb-build.ready` exists.
5. Check `bin/murb --version` for the expected revision and feature flags.
6. Run CPU smoke and correctness tests.
7. Submit GPU correctness tests and the benchmark only from a scheduled A100
   allocation.

The initial A100 acceptance run should use a small body count and compare the
existing Catch2 `[correctness]` cases against `cpu+naive` before increasing the
benchmark workload. Multi-GPU execution and replay are deferred.
