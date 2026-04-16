# RMS_NORM DAG cost decomposition — 2026-04-16

**Binary:** `build-hpx/bin/bench_hpx_rms_norm_f32`
**Commit:** `3c1d7b585`  (hpx-prefill-orchestrator)
**Params:** eps=1e-5, warmup=50, reps=1000, n∈{512,2048,4096,8192}, lanes∈{1,2,4}

## Paths

| impl | description |
|---|---|
| `ref` | single-threaded scalar loop, lanes=1 |
| `direct` | production callbacks called directly, no HPX, no DAG, lanes=1 |
| `dag_empty` | R0→R1→R2 region group, near-nop callbacks (orchestration floor) |
| `dag` | R0→R1→R2 region group, production callbacks |

## Key numbers (avg_ns / min_ns)

| impl | n=512 | n=2048 | n=4096 | n=8192 |
|---|---|---|---|---|
| ref (lanes=1) | 299 / 250 | 1201 / 1125 | 2401 / 2292 | 4834 / 4666 |
| direct (lanes=1) | 307 / 250 | 1209 / 1125 | 2439 / 2333 | 5118 / 4833 |
| dag_empty l=1 | 422k / 9k | 149k / 10k | 58k / 10k | 96k / 10k |
| dag l=1 | 104k / 10k | 128k / 11k | 171k / 12k | 143k / 15k |
| dag_empty l=4 | 332k / 15k | 106k / 15k | 244k / 14k | 212k / 15k |
| dag l=4 | 243k / 15k | 262k / 16k | 1354k / 18k | 231k / 18k |

All values in nanoseconds.

## Findings

**1. direct ≈ ref.**
`direct` and `ref` agree within ~7% at all sizes and pass the correctness
check. The production callbacks add no measurable overhead over the hand-written
scalar loop. `ref` was already representative.

**2. The math cost is not the bottleneck.**
At n=8192, `direct` takes ~5 µs. `dag_empty` min is ~10 µs — 2× the math even
with near-nop callbacks. `dag` min is ~15 µs. Every tier of HPX dispatch adds
5–10 µs of irreducible floor.

**3. The avg is dominated by scheduler jitter, not math.**
stddev values are 200–1600× larger than min values across all rows. The
distribution is highly right-skewed: min represents the cache-warm fast path;
avg includes OS preemption, HPX worker wakeup latency, and contention from
the warmup and prior runs. The min column is the honest latency floor.

**4. dag_empty floor ≈ dag floor at n_lanes=1.**
At n=4096 lanes=1: dag_empty min=10k ns, dag min=12k ns. The extra math (two
full-n loops + a sqrt) costs only ~2 µs on the fast path. Orchestration
dominates even at n=4096.

**5. Lanes fan-out does not help at these sizes.**
All configurations with lanes>1 show the same or higher min latency than
lanes=1. For n≤4096 the fan-out overhead (extra hpx::async calls per region)
exceeds the parallelism gain. lanes=4 occasionally produces extreme outliers
(dag n=4096 l=4 avg=1354 µs).

## Open question for Phase 5C

The ~10 µs dispatch floor in `dag_empty` comes from somewhere in:
- outer `hpx::async([&]{...}).get()` wrapper
- `ggml_hpx_run_region_group` Kahn sort + vector allocation
- three `hpx::async` / `hpx::dataflow` posts inside the group runner

Profiling or a stripped-down single-region path could isolate which layer owns
that 10 µs.
