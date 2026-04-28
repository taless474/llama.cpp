# 2026-04-27 — HPX for_loop vs ggml CPU_REPACK threadpool, Q/K/attn_out shape, 3 workers

## Question

Same as the gate/up A/B but at the second TinyLlama Q4_K_M shape: 2048×2048×1
(Qcur, Kcur, attn_out — 3 invocations per layer × 22 layers = 66 ops per
decode step). Worker count = 3 because the prior thread sweep showed 3 was
ggml's best on this shape.

Companion to `2026-04-27-hpx-vs-ggml-4t-2048x5632/` (gate/up at 4 workers).
Two shapes establishes whether the HPX advantage is shape-uniform or
shape-dependent.

## Setup

| | |
|---|---|
| ggml binary | `build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv` |
| HPX binary  | `build-hpx-bench/bin/bench-hpx-route-q4k-gemv` |
| ggml build  | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| HPX build   | `-DGGML_HPX=ON  -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `dcf25c1d8` (working tree dirty: bench cpp + CMakeLists hook) |
| host | Apple M4 (4 P-cores + 6 E-cores; NEON + matmul-int8 + dotprod) |
| shape | cols=2048, out_cols=2048, rows=1 (TinyLlama Qcur/Kcur/attn_out) |
| workers / threads | 3 (matches the best ggml baseline at this shape) |
| iters | 1000 timed (after 10 warmup), per cell |
| HPX scheduler | `--hpx:queuing=static` |
| HPX bridge | `for_loop(par,...)` wrapped in `hpx::async([&]{...}).get()`, bridge cost included in per-iter timing |
| harness reality | both routes assert `buft=CPU_REPACK, trait=q4_K_8x8_q8_K` |
| correctness | HPX run prints `correctness: HPX matches ggml CPU reference at 8/8 sentinels` before timing |
| machine state | quiet, with one Chrome renderer at 8–13% (below the 20% protocol threshold) |

## Reproduce

```
GGML=build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv
HPX=build-hpx-bench/bin/bench-hpx-route-q4k-gemv

# Batch 1: ggml then HPX
$GGML 3 2048 2048
$HPX  3 2048 2048
# Batch 2: HPX then ggml
$HPX  3 2048 2048
$GGML 3 2048 2048
# Batch 3: ggml then HPX
$GGML 3 2048 2048
$HPX  3 2048 2048
```

## Results — 6 cells (3 batches × 2 routes, alternating order)

| batch | order | route | min us | median us | p90 us | mean us | stdev us |
|---|---|---|---|---|---|---|---|
| 1 | 1 | ggml | 39.38 | 48.71 | 61.92 | 50.51 | 8.45 |
| 1 | 2 | hpx  | 24.83 | 28.21 | 30.75 | 28.49 | 4.85 |
| 2 | 1 | hpx  | 27.13 | 28.42 | 35.13 | 30.46 | 5.41 |
| 2 | 2 | ggml | 41.79 | 49.58 | 62.38 | 52.04 | 7.19 |
| 3 | 1 | ggml | 42.96 | 49.75 | 61.71 | 51.99 | 7.73 |
| 3 | 2 | hpx  | 26.13 | 28.25 | 33.63 | 29.65 | 4.81 |

8/8 correctness sentinels matched on every HPX cell.

### Per-route summary (across 3 batches)

|     | min (avg of 3 mins) us | median (avg of 3 medians) us | mean stdev us |
|---|---|---|---|
| ggml CPU_REPACK 3t | **41.38** | **49.35** | 7.8 |
| HPX for_loop 3w    | **26.03** | **28.29** | 5.0 |
| HPX faster by      | **37%**   | **43%**   | — |
| HPX speedup        | **1.59x** | **1.74x** | — |

### Direction unanimous across batches

| | batch 1 | batch 2 | batch 3 | spread |
|---|---|---|---|---|
| ggml min (us)         | 39.38 | 41.79 | 42.96 | 3.58 (8.6%) |
| HPX min (us)          | 24.83 | 27.13 | 26.13 | 2.30 (8.8%) |
| HPX/ggml min ratio    | 0.631 | 0.649 | 0.608 | 0.041 |
| ggml/HPX min speedup  | 1.59x | 1.54x | 1.64x | 0.10 |

Order does not flip the result; HPX is faster whether it runs first or
second. Per-cell mins are within 4us of each other inside each route, well
under the 15us route-to-route gap.

## What this says

**HPX's win is *bigger* on the smaller shape.** Comparing absolute and
relative speedups across both shapes:

| shape | workers | ggml min us | HPX min us | abs gap | rel speedup |
|---|---|---|---|---|---|
| 2048×5632 (gate/up)        | 4 | 71.62 | 55.10 | 16.5 | 1.30x |
| 2048×2048 (Q/K/attn_out)   | 3 | 41.38 | 26.03 | 15.4 | **1.59x** |

The absolute dispatch overhead saved is essentially the same (15–17us per
iter) — HPX's `for_loop` consistently saves about that much vs ggml's
spin-barrier + atomic-fetch-add + final-barrier across both shapes. But
because the smaller shape has less kernel work per iter, the same fixed
saving is a bigger fraction of total time, so the relative speedup grows.

This matches the cold reading from the prior thread-sweep README: barrier
overhead is a fixed cost per iter, and the smaller shape's per-thread work
is 2.75× smaller for the same barrier — so dispatch is a relatively bigger
slice of the budget, and any dispatch improvement pays off harder.

**Variance gap also wider on the smaller shape.** mean stdev:
- ggml: 7.8us (about 16% of mean)
- HPX:  5.0us (about 17% of mean)

Both are tighter than gate/up because the runs are shorter (~30–50ms total
vs ~70–90ms), so fewer macOS background bursts get a chance to land. The
ggml/HPX stdev *ratio* is similar to the gate/up case in spirit, just at
lower absolute numbers.

## Implications for full TinyLlama Q4_K_M decode

Per decode step (22 layers):
- **66 ops at 2048×2048 (Q/K/attn_out)** — gap = 15.4us → **1.02 ms saved**
- **44 ops at 2048×5632 (gate/up)**     — gap = 16.5us → **0.73 ms saved**
- **Combined savings (Q4_K only):** ~1.75 ms / step

Standalone bench numbers, no integration, no Q6_K shapes considered. If
HPX can drive these gemv kernels through the live executor (currently gated
out of the selective path by `ggml-hpx-lower.cpp:249` for CPU_REPACK
weights), this is the upper bound it could capture for Q4_K MUL_MAT alone.

## Open

- Did not test 2048×2048 at *4* workers (where ggml itself regressed in the
  thread sweep, ~46us min). HPX's dispatch is cheaper, so it might tolerate
  the extra worker without hitting the same E-core spill penalty — or it
  might hit it harder. Worth one more cell.
- Did not run Q6_K shapes (5632×2048 ffn_out, 2048×32000 lm_head — both use
  the q*_K_8x8_q8_K trait family but a different concrete kernel
  `ggml_gemv_q6_K_8x8_q8_K`). The bench links the q4_K kernel by name; would
  need a small templating change to dispatch by trait. Holding for now.
- HPX bridge cost is currently always paid (every iter calls
  `hpx::async([&]{...}).get()` to enter HPX context). In production where
  the caller is already on an HPX worker, that cost would drop and the gap
  would widen. Did not measure with-bridge vs without-bridge yet.

## Files

- `bench.csv` — combined CSV (header + 6 rows), batch/order/route tagged
- `bench.log` — combined stderr (6 runs of harness output)
- `quiet-pre.txt` / `quiet-post.txt` — high-CPU process snapshot before / after
- `README.md` — this file
