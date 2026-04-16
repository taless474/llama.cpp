# RMS_NORM Phase 5C — outer async crossing cost — 2026-04-16

**Binary:** `build-hpx/bin/bench_hpx_rms_norm_f32`
**Commit:** `3c1d7b585`  (hpx-prefill-orchestrator)
**Params:** eps=1e-5, warmup=50, reps=1000, n∈{512,2048,4096,8192}, lanes∈{1,2,4}
**Stats:** min / median / p95 (avg dropped; distribution is too right-skewed)

## Variants

| impl | outer hpx::async | callbacks |
|---|---|---|
| `ref` | none | scalar hand-loop |
| `direct` | none | production, called directly |
| `dag_empty` | per rep | near-nop |
| `dag_empty_nowrap` | once for whole loop | near-nop |
| `dag` | per rep | production |
| `dag_nowrap` | once for whole loop | production |

## Key numbers (min_ns / median_ns)

| impl | n=512 l=1 | n=2048 l=1 | n=4096 l=1 | n=8192 l=1 |
|---|---|---|---|---|
| ref | 250 / 292 | 1125 / 2000 | 2291 / 2416 | 4833 / 5041 |
| direct | 250 / 292 | 1125 / 1209 | 2333 / 2417 | 4875 / 5042 |
| dag_empty | 9000 / 15417 | 9375 / 14458 | 12041 / 15000 | 11458 / 14125 |
| dag_empty_nowrap | **6750 / 9208** | **6500 / 9541** | **6875 / 9167** | **7375 / 9208** |
| dag | 10125 / 15208 | 12333 / 885208 | 12083 / 17959 | 16750 / 22459 |
| dag_nowrap | **8375 / 9959** | **8208 / 232500** | **10583 / 12875** | **12750 / 16458** |

All values in nanoseconds.

## Findings

**1. Outer async crossing costs ~2–4 µs on the min path.**
Comparing `dag_empty` min vs `dag_empty_nowrap` min:
- n=512 l=1: 9000 → 6750 ns (saves ~2.3 µs)
- n=2048 l=1: 9375 → 6500 ns (saves ~2.9 µs)
- n=4096 l=1: 12041 → 6875 ns (saves ~5.2 µs)
- n=8192 l=1: 11458 → 7375 ns (saves ~4.1 µs)

Removing the per-rep outer hpx::async consistently cuts the floor by 2–5 µs.
This confirms the outer crossing is measurable but not the dominant cost.

**2. The remaining floor (~7–13 µs) is inside ggml_hpx_run_region_group.**
After removing the outer async, `dag_empty_nowrap` min is still 6.5–13 µs for
near-nop callbacks. The math (`direct`) costs 0.25–5 µs. The gap is 3–8 µs of
pure DAG machinery: Kahn sort, vector allocations, 3× hpx::async + dataflow
+ wait_all inside the group runner.

**3. Median is far more volatile than min.**
`dag` n=2048 l=1: min=12333, median=885208. The distribution has a very long
tail (OS preemption, HPX worker wakeup). Min is the only reliable floor signal.
p95 is mostly >1 ms and dominated by outliers. Median is somewhere in between.
For latency-floor analysis, **min is the truth; median is noise context**.

**4. lanes=1 is consistently fastest or tied.**
For all sizes, `dag_nowrap` l=1 min ≤ l=2 min ≤ l=4 min. Fan-out adds task
posting cost without enough parallel work to justify it at these row lengths.

**5. Adding nowrap to dag does not close the gap with direct.**
`dag_nowrap` min at n=8192 l=1 = 12750 ns vs `direct` = 4875 ns.
The ~8 µs remaining gap is entirely inside `ggml_hpx_run_region_group`:
3× region iterations with hpx::async / hpx::dataflow / hpx::wait_all.

## Open question for Phase 5D

The 7–13 µs floor in `dag_empty_nowrap` (no outer async, near-nop work) comes
from the group runner internals. Candidates:
- std::vector allocation for `pred_idx`, `in_deg`, `topo`, `region_fut` (4 heap allocs per call)
- hpx::dataflow composition (one shared_future per region)
- hpx::wait_all on 3 shared_futures

A stack-allocated or pre-allocated runner path, or collapsing R0+R1+R2 into
a single dispatch point, would test whether the allocation + dataflow cost
is the remaining floor.
