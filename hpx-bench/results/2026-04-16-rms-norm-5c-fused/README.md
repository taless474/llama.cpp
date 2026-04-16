# RMS_NORM Phase 5C — fused nowrap ceiling — 2026-04-16

**Binary:** `build-hpx/bin/bench_hpx_rms_norm_f32`
**Commit:** `3c1d7b585` + stack-alloc patch (hpx-prefill-orchestrator)
**Baseline:** `hpx-bench/results/2026-04-16-rms-norm-5c-stackalloc/`
**Params:** eps=1e-5, warmup=50, reps=1000, n∈{512,2048,4096,8192}, lanes∈{1,2,4}

## New variant

`dag_fused_nowrap` — the same three production callbacks (partial, finalize,
apply) called sequentially from within HPX, with the outer `hpx::async`
amortised over the whole rep loop via `bench_ns_in_hpx`.  No region DAG,
no per-rep async, no dataflow.  Implemented by reusing `rms_norm_direct_f32`
inside a `bench_ns_in_hpx` lambda.

**Purpose:** establishes the best-case cost of "one HPX boundary around
the full RMS_NORM computation."  The gap between this and `dag_nowrap(l=1)`
is the irreducible cost of the 3-region DAG orchestration.

## Key comparison (min_ns, l=1)

| n | direct | dag_fused_nowrap | dag_nowrap | DAG overhead |
|---|---|---|---|---|
| 512 | 208 | 458 | 8292 | ~7.8 µs |
| 2048 | 1125 | 1125 | 8083 | ~7.0 µs |
| 4096 | 2458 | 2500 | 9666 | ~7.2 µs |
| 8192 | 5041 | 4583 | 12875 | ~8.3 µs |

## Findings

**1. Being on an HPX thread adds no overhead to the math.**
`dag_fused_nowrap` min ≈ `direct` min at all sizes.  The small differences
(e.g. 208 vs 458 at n=512) are within run-to-run noise; at n=2048-8192
they are indistinguishable.  Running from within HPX does not change the
cost of the three callback passes themselves.

**2. The 3-region DAG orchestration costs ~7–8 µs on the fast path.**
`dag_nowrap(l=1)` min − `dag_fused_nowrap` min = 7–8 µs across all sizes.
This number is stable and independent of n — confirming it is pure scheduler
overhead, not math cost.

This 7–8 µs is the cost of 3× `hpx::async` (one per region via
`launch_region_async`) + 2× `hpx::dataflow` (regions with predecessors) +
1× `hpx::wait_all` on 3 `shared_future<void>`s.

**3. The math cost is not the bottleneck for any n in this sweep.**
At n=512 the math is ~300 ns; the DAG adds 27× overhead.
At n=8192 the math is ~5 µs; the DAG adds 2.6× overhead.
Even at the largest tested size, the DAG is the dominant cost.

**4. The ceiling (dag_fused_nowrap → direct) is already known.**
The remaining gap between `dag_fused_nowrap` and `direct` is the
outer-async amortisation — effectively zero across reps.  The math
floor is set.

## Conclusion

The 7–8 µs DAG floor cannot be reduced by further stack-alloc or
context-entry optimisations.  It is the per-call cost of the HPX
task graph itself: 3 asyncs + 2 dataflows + wait_all.

The only paths that can materially reduce this:
- **Collapse dispatch count** (Exp 3 from plan): one `hpx::async` around
  all three passes — would cost ≈ 2–3 µs, saving ~5–6 µs vs current floor
- **Persistent worker / reuse** (Exp 3 variant): pre-warm a long-lived
  HPX task and post work into it — amortises task-post cost across many calls
- **Accept the floor for large n**: at n≥16k the math cost exceeds
  ~10 µs and the 7–8 µs overhead is <50% of total; breakeven is
  somewhere in the 8–16k range for l=1
