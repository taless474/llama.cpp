# 2026-04-27 — HPX bridge-cost ablation

## Question

The previous A/B (`2026-04-27-hpx-vs-ggml-4t-2048x5632/` and
`2026-04-27-hpx-vs-ggml-3t-2048x2048/`) showed HPX's `for_loop` driver is
1.30x–1.59x faster than ggml's `CPU_REPACK` threadpool at the *same chunk
grid and worker count*.

Those HPX runs hop into HPX every iter via `hpx::async([&]{...}).get()`,
because the bench `main()` is a native OS thread and
`hpx::experimental::for_loop` requires HPX context.  In real integration
the caller is already on an HPX worker — there'd be no per-iter hop.

**This bench isolates the bridge cost.**  How much of the HPX win is the
`for_loop` driver itself, and how much is wasted on the hop we'd never pay
in production?

## Two modes

The bench (`tests/bench-hpx-route-q4k-gemv.cpp`) now takes a 4th positional
argument:

| mode | per-iter dispatch | rationale |
|---|---|---|
| `bridge` | `hpx::async([&]{ iter_body(); }).get()` per iter | matches all prior HPX runs in this repo; what a non-HPX caller pays today |
| `in_hpx` | one `hpx::async` wraps the whole warmup loop, then a second wraps the timed loop with `chrono` calls *inside* the worker | per-iter time captures only `for_loop` dispatch; no thread hop |

Both modes pass the same 8/8 sentinel correctness check against the ggml
CPU reference, and both abort if `hpx::get_num_worker_threads()` does not
match the requested count.

## Setup

| | |
|---|---|
| binary | `build-hpx-bench/bin/bench-hpx-route-q4k-gemv` |
| build  | `-DGGML_HPX=ON -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `dcf25c1d8` (working tree dirty: bench cpp + CMakeLists hook) |
| host | Apple M4 (4 P-cores + 6 E-cores) |
| iters | 1000 timed (after 10 warmup) per cell |
| HPX   | `--hpx:queuing=static --hpx:threads=N` |
| machine state | quiet (snapshots in `quiet-pre.txt`/`quiet-post.txt`) |

## Reproduce

```
HPX=build-hpx-bench/bin/bench-hpx-route-q4k-gemv

# gate-up, 4 workers, alternating bridge↔in_hpx across 3 batches
$HPX 4 2048 5632 bridge
$HPX 4 2048 5632 in_hpx
$HPX 4 2048 5632 in_hpx
$HPX 4 2048 5632 bridge
$HPX 4 2048 5632 bridge
$HPX 4 2048 5632 in_hpx

# qka, 3 workers, same alternating pattern
$HPX 3 2048 2048 bridge
$HPX 3 2048 2048 in_hpx
$HPX 3 2048 2048 in_hpx
$HPX 3 2048 2048 bridge
$HPX 3 2048 2048 bridge
$HPX 3 2048 2048 in_hpx
```

## Results — gate-up shape (2048×5632×1, 4 workers)

| batch | order | mode   | min us | median us | p90 us | mean us | stdev us |
|---|---|---|---|---|---|---|---|
| 1 | 1 | bridge | 54.38 | 67.29 | 73.79 | 67.92 |  8.08 |
| 1 | 2 | in_hpx | 47.46 | 51.67 | 63.21 | 54.94 | 10.94 |
| 2 | 1 | in_hpx | 47.42 | 48.17 | 61.42 | 52.79 | 11.60 |
| 2 | 2 | bridge | 54.29 | 62.08 | 71.75 | 65.34 | 31.91 |
| 3 | 1 | bridge | 54.58 | 62.42 | 73.21 | 65.28 |  7.75 |
| 3 | 2 | in_hpx | 47.46 | 50.54 | 61.29 | 53.10 |  8.07 |

### Per-mode summary (gate-up)

|     | min (avg) | median (avg) | p90 (avg) | mean stdev |
|---|---|---|---|---|
| bridge | **54.42** | **63.93** | 72.92 | 15.9 |
| in_hpx | **47.45** | **50.13** | 61.97 | 10.2 |
| **bridge cost** | **6.97** | **13.80** | 10.95 | — |

Per-mode min spread across 3 batches: bridge 0.29us, in_hpx 0.04us.

## Results — qka shape (2048×2048×1, 3 workers)

| batch | order | mode   | min us | median us | p90 us | mean us | stdev us |
|---|---|---|---|---|---|---|---|
| 1 | 1 | bridge | 25.79 | 28.33 | 35.92 | 30.26 | 4.91 |
| 1 | 2 | in_hpx | 22.75 | 23.04 | 27.58 | 24.60 | 4.60 |
| 2 | 1 | in_hpx | 22.71 | 23.00 | 27.71 | 24.48 | 4.24 |
| 2 | 2 | bridge | 25.58 | 28.25 | 33.71 | 29.55 | 3.52 |
| 3 | 1 | bridge | 26.46 | 28.38 | 35.08 | 30.46 | 5.50 |
| 3 | 2 | in_hpx | 22.75 | 23.00 | 27.08 | 24.01 | 3.48 |

### Per-mode summary (qka)

|     | min (avg) | median (avg) | p90 (avg) | mean stdev |
|---|---|---|---|---|
| bridge | **25.94** | **28.32** | 34.90 | 4.64 |
| in_hpx | **22.74** | **23.01** | 27.46 | 4.10 |
| **bridge cost** | **3.20** | **5.31** | 7.44 | — |

Per-mode min spread across 3 batches: bridge 0.88us, in_hpx 0.04us.

## What this says

**The bridge is real but small.**  Per-iter cost of one
`hpx::async([&]{...}).get()` round-trip:

| shape    | bridge cost (min) | bridge cost (median) |
|---|---|---|
| gate-up  | 6.97 us | 13.80 us |
| qka      | 3.20 us |  5.31 us |

The *minimum* bridge cost on the smaller shape is ~3us (think of this as
the fixed cost of HPX scheduling the lambda + signaling the future on
completion).  On the larger shape, the bridge minimum is ~7us — likely
because the iter takes longer (47us vs 23us) so the worker that runs the
lambda is the same one finishing kernel chunks, adding a small
contention/wait between get() and the next async dispatch.  Hard to be
sure without deeper instrumentation, but the absolute number is small.

The median bridge cost is larger than the min on both shapes — the bridge
itself has a tail (sometimes the future signaling lands behind a small
scheduler delay).  This shows up as a wider gap on median than on min.

**HPX `for_loop` itself is the dominant win.**  Comparing all three
condition tiers:

### gate-up summary (2048×5632×1, 4 threads/workers)

| condition                | min us | speedup vs ggml |
|---|---|---|
| ggml CPU_REPACK 4t       | 71.62 | 1.00x |
| HPX bridge mode 4w       | 54.42 | 1.32x |
| HPX **in_hpx** mode 4w   | **47.45** | **1.51x** |

### qka summary (2048×2048×1, 3 threads/workers)

| condition                | min us | speedup vs ggml |
|---|---|---|
| ggml CPU_REPACK 3t       | 41.38 | 1.00x |
| HPX bridge mode 3w       | 25.94 | 1.59x |
| HPX **in_hpx** mode 3w   | **22.74** | **1.82x** |

So if the live integration runs the bench's iter from inside an HPX worker
(no bridge), the win over ggml widens to:
- gate-up: **1.51x** (vs 1.32x with bridge → ~14% additional from removing the bridge)
- qka:     **1.82x** (vs 1.59x with bridge → ~14% additional)

The bridge is *not* free, but it is also *not* where the win comes from.
Most of the gap is the dispatcher itself (`for_loop` + `--hpx:queuing=static`
worker pool vs ggml's spin-barrier + atomic-fetch-add + final-barrier).

**HPX is also tighter on tail variance.**  Across all 12 cells:
- bridge stdev: 4–32 us per cell
- in_hpx stdev: 4–12 us per cell

In-HPX mode strips out one source of jitter (the bridge hop), so the tail
gets cleaner.  ggml at 4 threads on the larger shape was 24–63 us stdev —
HPX in-HPX mode at the same shape is 8–12 us, a 3–6× stdev reduction.

## Implications for full Q4_K_M decode

Re-running the per-step gate/up + Q/K/attn_out math under each condition:

| condition           | gate-up (44 ops) | qka (66 ops) | combined Q4_K savings vs ggml |
|---|---|---|---|
| ggml CPU_REPACK     | 44 × 71.6 = 3.15 ms | 66 × 41.4 = 2.73 ms | — |
| HPX bridge          | 44 × 54.4 = 2.39 ms | 66 × 25.9 = 1.71 ms | **1.78 ms / step** |
| HPX **in_hpx**      | 44 × 47.5 = 2.09 ms | 66 × 22.7 = 1.50 ms | **2.29 ms / step** |

**The bridge accounts for ~0.51ms of additional overhead per step that the
live path would not pay** (assuming the live HPX executor calls into HPX
once at region entry and stays inside HPX for the duration of the
sublayer, as it does today for the SWIGLU packet path).

## Implications for the live integration design

Two takeaways for designing a CPU_REPACK-aware HPX lowered path:

1. **Don't worry about adding a fast path that avoids HPX entry per op.**
   The bridge is small enough (3–7us min) that the existing per-region
   single-async pattern (used by `ggml_hpx_run_frozen_packet`) is fine.
   What matters is that the *loop body* runs entirely inside HPX, not that
   we eliminate the very first hop.
2. **The win is the `for_loop` over the same chunk grid** — no need to
   change the chunk shape, kernel, or worker count.  A live integration
   that simply replaces the ggml threadpool's chunk hand-out with
   `hpx::experimental::for_loop(par, 0, nchunk0, ...)` over the *exact same
   per-chunk pointer math* should capture most of the in_hpx-mode win.

The remaining open is the architectural one: today
`ggml-hpx-lower.cpp:249` rejects any tensor with `extra != nullptr`, which
includes every CPU_REPACK Q4_K weight.  A live A/B requires lifting that
gate (and adapting the lowered path to call the gemv kernel directly
instead of `vec_dot_q4_K_q8_K`, which can't read the repacked layout).

## Open

- Did not measure with `--hpx:queuing=local-priority` (the default; we
  pinned `static` to match production).  The bridge cost might differ.
- Did not measure quantize-step parallelization.  Today both modes serialize
  Q8_K quantization on the calling thread (matches ggml ith=0-only path
  for ne11=1).  An `in_hpx` mode that fuses the quantize into the same
  parallel region could shave more — orthogonal to bridge ablation.
- Did not test very small shapes where bridge cost would dominate.  For
  the two production-relevant Q4_K decode shapes here, the bridge is
  comfortably below the kernel time on both.

## Files

- `bench.csv` — combined CSV (header + 12 rows): shape, batch, order, route, mode, n_workers, iters, cols, out_cols, rows, trait, min, median, p90, mean, stdev
- `bench.log` — combined stderr (12 runs, blank-line separated)
- `quiet-pre.txt` / `quiet-post.txt` — high-CPU process snapshots before/after
- `README.md` — this file
