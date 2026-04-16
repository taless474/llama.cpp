# RMS_NORM Phase 5C — stack-alloc group runner — 2026-04-16

**Binary:** `build-hpx/bin/bench_hpx_rms_norm_f32`
**Commit:** `3c1d7b585` + stack-alloc patch (hpx-prefill-orchestrator)
**Baseline:** `hpx-bench/results/2026-04-16-rms-norm-5c-nowrap/`
**Params:** eps=1e-5, warmup=50, reps=1000, n∈{512,2048,4096,8192}, lanes∈{1,2,4}

## Change

`ggml_hpx_run_region_group`: added a small-group fast path for
`n_regions <= 8 && n_deps <= 64`.  All four per-call heap allocations
replaced with stack-local arrays:

| was | now |
|---|---|
| `std::vector<std::vector<int>> pred_idx(n)` | `std::array<std::array<int,8>,8> pred_store` + `std::array<int,8> pred_count` |
| `std::vector<int> in_deg(n, 0)` | `std::array<int,8> in_deg_arr{}` |
| `std::vector<int> topo; topo.reserve(n)` | `std::array<int,8> topo_arr` + `int topo_count` |
| `std::vector<hpx::shared_future<void>> region_fut(n)` | `std::array<hpx::shared_future<void>,8> region_fut_arr` |

Kahn traversal uses a plain `int[8]` FIFO (head/tail indices) instead of
`std::queue`.  Execution model, dataflow composition, and `wait_all`
semantics are unchanged.  Heap fallback for n > 8 is the original code.

One residual heap allocation per region-with-predecessors: the `pred_futs`
local vector passed to `hpx::dataflow`.  For RMS_NORM (2 regions with 1
predecessor each) this is 2 single-element vectors per call.

## Floor comparison (min_ns, l=1)

| variant | n=512 | n=2048 | n=4096 | n=8192 |
|---|---|---|---|---|
| dag_empty_nowrap before | 6750 | 6500 | 6875 | 7375 |
| dag_empty_nowrap **after** | **6125** | 8208 | **6958** | **7208** |
| dag_nowrap before | 8375 | 8208 | 10583 | 12750 |
| dag_nowrap **after** | **8000** | 8375 | **10041** | 13416 |

## Finding

**Heap churn removal had no material effect on the floor.**

Changes are within run-to-run noise (±1–2 µs).  Some cells improved
slightly; others regressed slightly.  No clear directional trend.

The 6–13 µs floor in `dag_empty_nowrap` (near-nop callbacks, no outer
async, stack storage) is not caused by per-call vector allocations.

## What the floor actually is

After removing outer async (5C) and heap allocs (this run), the floor
for a 3-region near-nop DAG is still ~7–9 µs min.  The remaining cost
must come from HPX task scheduling itself:

- 1 `hpx::async` per region at `launch_region_async` entry (3 total)
- 2 `hpx::dataflow` compositions (R1 and R2 wait on predecessors)
- 1 `hpx::wait_all` on 3 `shared_future<void>`s

Each `hpx::async` + `hpx::dataflow` task post likely costs ~2–3 µs on
this system.  3–5 task posts × 2 µs = 6–10 µs, which matches the observed
floor.

## Next experiment (Exp 2 from plan)

Stack-alloc was not the bottleneck.  Task-post count is.

The question for Exp 2: does collapsing R0+R1+R2 into a single dispatch
point remove most of the floor?  If a one-shot path costs ~2 µs (one
`hpx::async`, no dataflow), and the 3-region path costs ~8 µs, then
reducing dispatch count to 1 would be a 4× floor improvement.

This does not require changing the DAG semantics — just benchmarking
a coarser dispatch as a measurement tool.
