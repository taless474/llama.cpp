# 2026-04-27 — TinyLlama Q4_K_M baseline decode tok/s (clean timing)

## Goal

Establish the baseline-tok/s floor that any HPX scheduling change has to beat,
following the fair-comparison protocol in `CLAUDE.local.md`. No HPX in the
binary, no instrumentation env vars, default thread count.

## Build

| | |
|---|---|
| binary | `build-baseline-no-hpx/bin/llama-bench` |
| build dir | `build-baseline-no-hpx/` |
| CMake flags | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `e0b332bb5` (working tree dirty: env-gated mulmat-path logger) |
| model | `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| host | Apple M4 (NEON + matmul-int8 + dotprod), Accelerate BLAS linked |

`GGML_CPU_LOG_MULMAT_PATH` is *not* set in this run — the env-gated logger
short-circuits at the first env check, so its overhead is one branch per
MUL_MAT compute call. Treated as effectively zero.

## Quiet-machine state at run time

`ps -A -o %cpu,comm` showed: WindowServer 29.3%, Code Helper Renderer 9.3%,
Chrome 8.6%, dasd 8.1%, claude 16.7%. WindowServer at ~30% is normal Mac
compositor load with displays active; nothing else is unexpected.
`pmset -g therm` reported no thermal warnings.

## Reproduce

```
./build-baseline-no-hpx/bin/llama-bench \
  -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 0 -p 0 -n 128 -r 5 -o csv
```

`-p 0 -n 128` → text generation only, no prefill timing. `-r 5` → 5 reps per
test. `-ngl 0` → CPU only (Metal is auto-attached at build time but not used).
Default `n_threads = 4` (M4 perf-core count).

## Results

| batch | avg tok/s | stdev tok/s | rel stdev |
|---|---|---|---|
| run1 | 95.52 | 4.94 | 5.17% |
| run2 | 97.57 | 2.50 | 2.56% |
| Δ between batches | +2.05 (+2.1%) | | |

Both batches under the 10% intra-batch stdev gate. Drift between batches is
2.1%, well inside the run-to-run noise floor for a quiet M4.

## Verdict

**Baseline is stable.** Median tok/s in the 95–98 range. Any HPX scheduling
change should be measured against this on the same machine state with the same
fair-comparison protocol (alternating order, single rebuild per batch, etc.).

## Files

- `bench_run1.csv` / `bench_run1.log` — first batch
- `bench_run2.csv` / `bench_run2.log` — second batch
- `README.md` — this file
