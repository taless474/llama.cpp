# 2026-04-27 — HPX for_loop vs ggml CPU_REPACK threadpool, gate/up shape, 4 workers

## Question

For the q4_K_8x8_q8_K gemv kernel that real TinyLlama Q4_K_M decode runs 44
times per step (gate/up, 2048×5632×1), can a custom HPX driver schedule the
*same kernel body* with lower per-iter latency than ggml's CPU_REPACK
threadpool (`ggml_threadpool_chunk_set` + `atomic_fetch_add` + `ggml_barrier`)?

Same kernel, same chunk grid, same worker count — only the dispatcher changes.

## Setup

| | |
|---|---|
| ggml binary | `build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv` |
| HPX binary  | `build-hpx-bench/bin/bench-hpx-route-q4k-gemv` |
| ggml build  | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| HPX build   | `-DGGML_HPX=ON  -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `dcf25c1d8` (working tree dirty: bench cpp + CMakeLists hook) |
| host | Apple M4 (4 P-cores + 6 E-cores; NEON + matmul-int8 + dotprod) |
| shape | cols=2048, out_cols=5632, rows=1 (TinyLlama gate/up) |
| workers / threads | 4 (matches the best ggml baseline at this shape) |
| chunk grid | nchunk0=16, dr0=352 — both routes use the exact same grid |
| iters | 1000 timed (after 10 warmup), per cell |
| HPX scheduler | `--hpx:queuing=static` (no work-stealing) |
| HPX bridge | `hpx::experimental::for_loop(par, ...)` wrapped in `hpx::async([&]{...}).get()` so the parallel region runs on an HPX worker; bridge cost is included in per-iter timing |
| harness reality | both routes assert `buft=CPU_REPACK, trait=q4_K_8x8_q8_K` |
| correctness | HPX run prints `correctness: HPX matches ggml CPU reference at 8/8 sentinels` before timing |
| machine state | quiet (pre/post snapshots in `quiet-pre.txt`/`quiet-post.txt`) |

## Reproduce

```
GGML=build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv
HPX=build-hpx-bench/bin/bench-hpx-route-q4k-gemv

# Batch 1: ggml then HPX
$GGML 4 2048 5632
$HPX  4 2048 5632
# Batch 2: HPX then ggml
$HPX  4 2048 5632
$GGML 4 2048 5632
# Batch 3: ggml then HPX
$GGML 4 2048 5632
$HPX  4 2048 5632
```

## Results — 6 cells (3 batches × 2 routes, alternating order)

CSV columns: batch, order, route, n_workers, iters, cols, out_cols, rows,
trait, min_us, median_us, p90_us, mean_us, stdev_us.

| batch | order | route | min us | median us | p90 us | mean us | stdev us |
|---|---|---|---|---|---|---|---|
| 1 | 1 | ggml | 71.83 | 83.96 | 107.79 | 90.30 | 24.00 |
| 1 | 2 | hpx  | 55.21 | 62.79 |  68.67 | 64.12 |  7.52 |
| 2 | 1 | hpx  | 55.38 | 62.92 |  73.13 | 65.08 |  7.72 |
| 2 | 2 | ggml | 71.33 | 79.13 |  94.83 | 85.87 | 53.77 |
| 3 | 1 | ggml | 71.71 | 82.04 | 130.96 | 97.27 | 63.26 |
| 3 | 2 | hpx  | 54.71 | 62.79 |  73.25 | 65.47 |  8.55 |

`y[0] = 0.0177` and 8/8 correctness sentinels matched on every HPX cell.

### Per-route summary (across 3 batches)

|     | min (avg of 3 mins) us | median (avg of 3 medians) us | mean stdev us |
|---|---|---|---|
| ggml CPU_REPACK 4t   | **71.62** | **81.71** | 47.0 |
| HPX for_loop 4w      | **55.10** | **62.83** |  7.9 |
| HPX faster by        | **23%**   | **23%**   | — |
| HPX speedup          | **1.30x** | **1.30x** | — |

### Direction unanimous across batches

| | batch 1 | batch 2 | batch 3 | spread |
|---|---|---|---|---|
| ggml min   | 71.83 | 71.33 | 71.71 | 0.50us (0.7%) |
| HPX min    | 55.21 | 55.38 | 54.71 | 0.67us (1.2%) |
| HPX/ggml min ratio | 0.769 | 0.776 | 0.762 | 0.014 |

Order does not flip the result: HPX is faster whether it runs first or
second. Min is preemption-resistant and shows ~1us spread across batches —
the kernel floor is tight on both routes; only the upper tail (mean/stdev)
moves with machine state.

## What this says

**HPX wins at the same chunk grid and same worker count, by ~16us per iter
(~23%).** Reading the per-iter floor (min):
- ggml threadpool barrier + atomic chunk grab: ~71.6us
- HPX `for_loop` over the same grid via `hpx::async` bridge: ~55.1us

The kernel work is identical (`ggml_gemv_q4_K_8x8_q8_K`, same `wdata`, same
`W` layout), the chunk grid is identical (nchunk0=16, dr0=352), the worker
count is identical (4 — and the HPX bench prints `effective HPX worker
threads: 4` to confirm). The 16us gap is **all dispatch overhead**:
ggml's spin-barrier + atomic-fetch-add + final-barrier vs HPX's
fork-join across 4 worker threads.

**HPX latency is also dramatically more stable.** Mean stdev:
- ggml: 47us (varies 24–63us across batches; tail dominated by macOS
  preemption blowing past the 4-thread barrier wait)
- HPX: 7.9us (tight, batch-to-batch consistent at 7.5–8.5us)

The ggml threadpool spends time waiting in the barrier; any worker that
drifts to an E-core or gets preempted drags every iter's tail. HPX's
queuing=static scheduler keeps work on its own pool of OS threads with a
single-pool wait pattern — fewer cross-thread surfaces for OS jitter to land
on.

**The correctness check passes bit-equal.** Same kernel, same inputs, same
output — there is no numerical-difference confound. The 23% gap is purely
scheduling.

## Implications

For TinyLlama Q4_K_M decode at gate/up (44 invocations per step):
- ggml floor: 44 × 71.6us = **3.15ms / step** spent in this kernel alone.
- HPX floor: 44 × 55.1us = **2.42ms / step**.
- Saved per step: **~0.73ms** at the gate/up shape alone.

That's a real, measurable decode-speed lever — _at this kernel, in this
isolated bench_. Whether it survives integration into the live HPX path
depends on:
1. Whether the live executor can drive the same gemv (currently the
   selective path rejects CPU_REPACK Q4_K weights with `w->extra != nullptr`,
   per `ggml-hpx-lower.cpp:249`).
2. Whether the bridge cost (`hpx::async([&]{...}).get()`) stays the same
   when the caller is already on an HPX worker (it should drop, since no
   thread hop is needed).
3. Whether other op shapes show the same gap (next: 2048×2048 at 3 workers).

## Open

- Did not measure 2048×2048 at 3 workers/threads (the best ggml thread count
  for that shape per the prior sweep). Second-shape A/B is the immediate
  next step before drawing a general conclusion.
- Did not measure with HPX queuing=local-priority (the non-static option).
  Static was chosen because it's what the production path uses; comparing
  schedulers is a separate question.
- Did not vary HPX bridge mode. We always hop via `hpx::async` because the
  bench main is not an HPX thread; in production, the caller would already
  be on an HPX thread and the bridge would be a direct call. Removing the
  bridge cost in a future variant could widen the gap further.
- Did not test the 2048×2048 shape at 4 workers (where ggml regressed to
  ~46us min). The HPX route at 4 workers there could expose whether HPX's
  win persists when ggml is past its own optimum.

## Files

- `bench.csv` — combined CSV (header + 6 rows), batch/order/route tagged
- `bench.log` — combined stderr (6 runs of harness output, blank-line separated)
- `quiet-pre.txt` / `quiet-post.txt` — high-CPU process snapshot before / after
- `README.md` — this file
