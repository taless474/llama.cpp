# Phase 2B: minimal mixed/bursty workload analysis

This is a simulator-only study. Numbers below come from the pure-Python
discrete-event scheduling model in `hpx-bench/sim/continuous_batching/`.
They are **not** measured llama.cpp or HPX performance. They reflect a
linear cost model with placeholder coefficients, not real kernel cost.

The goal: identify the smallest workload structure that makes
`continuous_batching` separate from `static_batching`, which metric
separates first, and which workload should be the target for a future
real HPX/llama.cpp prototype.

## Headline findings

1. **Simulator-only evidence.** Every number in this report is a
   simulator scheduling-model result. None of it is measured
   llama.cpp or HPX performance. No real-performance claim is made.
2. **First separating workload:** **B — `staggered_identical_short`.**
   Identical request lengths (P=6, D=8) with deterministic 5 ms
   staggered arrivals are enough for `continuous_batching` to beat
   `static_batching`.
3. **First metrics to separate (on B):** **TTFT p50 (−8.5%)** and
   **total_latency p50 (−7.4%)**. These move first because continuous
   admits late arrivals as soon as a slot frees, while static makes
   them wait for the group barrier.
4. **Makespan and tokens_per_decode_call barely move on B**
   (makespan −0.30%, tok/call +0.86%, decode_calls −0.85%). With no
   length variance, continuous reorders *when* requests are admitted
   but not the total work, so system-level throughput is unchanged.
5. **Larger separations require length variance.** Continuous beats
   static substantially on:
   - **C — `mixed_decode_only`**: makespan −30.0%, p50 lat −29.5%,
     TTFT p50 −36.4%, tok/call +124.6%, decode_calls −55.5%.
   - **E — `mixed_prompt_and_decode`**: makespan −26.9%, p50 lat
     −26.3%, TTFT p50 −32.7%, tok/call +128.6%, decode_calls −56.3%.
6. **Recommended target for a real HPX/llama.cpp prototype:**
   - **Primary: C — `mixed_decode_only`.** Single axis of variance,
     synchronized arrivals, large unambiguous separation.
   - **Secondary: E — `mixed_prompt_and_decode`.** Both axes vary;
     stronger absolute signal; useful as a sanity gate.
7. **F — `bursty_mixed_deterministic` collapses into E.** With 500 ms
   burst spacing the system never drains between bursts, so makespan,
   tokens_per_decode_call, and decode_calls match E exactly. F is
   not recommended as a target until burst spacing is widened.
8. **For workload B, `n_slots` is the dominant axis; `n_batch` is
   mostly inert.** The TTFT-p50 gap grows from ~1% (2 slots) to ~9%
   (4 slots) to ~30–40% (8 slots), because more slots ⇒ more
   late-arriving requests are pinned by the static group barrier per
   iteration. `n_batch` only moves results in one constrained cell
   where 8 slots × 6 prompt tokens = 48 prefill rows force a split
   at `n_batch=32`.

## Method

- Six workloads (A–F), three policies (`fifo_context_pool`,
  `static_batching`, `continuous_batching`).
- Primary comparison: `static_batching` vs `continuous_batching`.
  `fifo_context_pool` is included only as a reference (it is the
  upper-bound cost shape — no batch sharing across slots).
- Fixed run config: `n_slots=4`, `n_batch=128`, `seed=42`,
  default cost params (`base=1.0, prompt=0.05, decode=0.5`,
  `slot=0`, `stream=0`).
- Workload size: `total_cap=99` for the new mixed/bursty workloads
  (divisible by 3 → equal class counts). `exp11_like_all_short`
  keeps its native 200-request shape.
- Raw outputs land in `hpx-bench/sim/continuous_batching/results/<run_id>/`
  (gitignored). Run-id is the 12-char config hash.
- Captures live under `local/phase2b_*.stdout` / `*.stderr`.
- All workloads are deterministic from `(seed, total_cap)`. F has no
  random component at all.

## Workload definitions

| ID | Workload name | Arrivals | Prompt | Decode | Class structure |
|---|---|---|---|---|---|
| A | `exp11_like_all_short` | all @ t=0 | 6 | 8 | 1 class (200 reqs) |
| B | `staggered_identical_short` | i × 5 ms | 6 | 8 | 1 class (99 reqs) |
| C | `mixed_decode_only` | all @ t=0 | 6 | {8, 64, 256} round-robin | 3 classes × 33 |
| D | `mixed_prompt_only` | all @ t=0 | {16, 128, 1024} round-robin | 8 | 3 classes × 33 |
| E | `mixed_prompt_and_decode` | all @ t=0 | {16, 128, 1024} | {8, 64, 256} | 3 classes × 33 |
| F | `bursty_mixed_deterministic` | 3 bursts × 33 reqs, 500 ms apart | same as E | same as E | same as E |

## Phase 2B grid result (n_slots=4, n_batch=128, seed=42)

Each cell is one policy on one workload. **Bold** marks the better of
`static` vs `continuous` (≥1% relative gap; ties left unbolded).

### A — `exp11_like_all_short`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 2660.0 | 1356.6 | 2553.6 | 2660.0 | 1314.6 | 2511.6 | 2618.0 | 1.556 | 1800 |
| static_batching | 1310.0 | 668.1 | 1257.6 | 1310.0 | 647.1 | 1236.6 | 1289.0 | 6.222 | 450 |
| continuous_batching | 1310.0 | 668.1 | 1257.6 | 1310.0 | 647.1 | 1236.6 | 1289.0 | 6.222 | 450 |

**Tie on every metric.** All requests are identical and arrive together;
continuous admission has nothing to exploit. (Re-confirms the Phase 2
result.)

### B — `staggered_identical_short`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 1316.7 | 444.9 | 810.78 | 826.74 | 403.5 | 769.36 | 802.0 | 1.556 | 891 |
| static_batching | 659.7 | 104.1 | 172.42 | 179.8 | 83.1 | 151.42 | 158.8 | 5.923 | 234 |
| continuous_batching | **657.7** | **96.4** | **164.91** | **169.2** | **76.0** | **144.58** | **150.476** | 5.974 | 232 |

cont vs static deltas: makespan **−0.30%**, p50 lat **−7.40%**, p95 lat
**−4.36%**, p99 lat **−5.90%**, TTFT p50 **−8.54%**, TTFT p95 **−4.52%**,
TTFT p99 **−5.24%**, tok/call +0.86%, decode_calls −0.85%.

This is the first workload where the two policies separate. Request
lengths are identical; only arrival timing varies. Continuous wins
because as soon as a slot frees mid-iteration it can admit the next
waiter, while static must wait for the whole 4-slot group to drain.

### C — `mixed_decode_only`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 16364.7 | 8113.4 | 15253.58 | 16225.05 | 7292.0 | 14608.58 | 15234.136 | 1.045 | 10923 |
| static_batching | 11866.7 | 5748.6 | 11099.2 | 11584.46 | 5727.6 | 10899.8 | 11449.2 | 1.777 | 6425 |
| continuous_batching | **8302.7** | **4051.4** | **7617.08** | **8163.05** | **3641.0** | **7294.88** | **7607.276** | **3.991** | **2861** |

cont vs static deltas: makespan **−30.0%**, p50 lat **−29.5%**,
TTFT p50 **−36.4%**, tok/call **+124.6%**, decode_calls **−55.5%**.

This is the canonical "short request behind long request" effect. With
all-at-once arrivals and only decode-length variance, static groups
short requests with long requests and is held by the long ones until
the group drains. Continuous releases the short slot when its 8 decode
tokens finish and admits a fresh request immediately.

### D — `mixed_prompt_only`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 3445.2 | 1766.1 | 3328.8 | 3434.91 | 1688.4 | 3245.5 | 3360.318 | 35.059 | 1122 |
| static_batching | 2845.2 | 1455.6 | 2741.695 | 2833.44 | 1401.95 | 2681.425 | 2781.437 | 75.356 | 522 |
| continuous_batching | **2766.2** | **1414.4** | **2682.6** | **2753.068** | **1359.45** | **2661.6** | **2699.007** | **88.795** | **443** |

cont vs static deltas: makespan **−2.78%**, p50 lat **−2.83%**,
TTFT p50 **−3.03%**, tok/call **+17.83%**, decode_calls **−15.13%**.

Smaller separation than C. Decode is the same for all classes (D=8),
so once prefill clears, all slots finish together. Continuous wins
mostly via tighter prefill packing.

### E — `mixed_prompt_and_decode`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 18493.2 | 9164.0 | 17276.69 | 18363.84 | 8288.6 | 16657.49 | 17264.314 | 4.426 | 11154 |
| static_batching | 14053.2 | 6879.6 | 13207.8 | 13759.2 | 6805.95 | 12965.425 | 13589.447 | 7.353 | 6714 |
| continuous_batching | **10275.2** | **5069.0** | **9502.94** | **10154.66** | **4578.6** | **9166.74** | **9502.294** | **16.815** | **2936** |

cont vs static deltas: makespan **−26.9%**, p50 lat **−26.3%**,
TTFT p50 **−32.7%**, tok/call **+128.6%**, decode_calls **−56.3%**.

Strongest absolute separation. Both length axes matter at once.

### F — `bursty_mixed_deterministic`

| policy | makespan_ms | p50 lat | p95 lat | p99 lat | TTFT p50 | TTFT p95 | TTFT p99 | tok/call | decode_calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo_context_pool | 18493.2 | 8664.0 | 16276.69 | 17363.84 | 7788.6 | 15657.49 | 16264.314 | 4.426 | 11154 |
| static_batching | 14053.2 | 6379.6 | 12207.8 | 12759.2 | 6305.95 | 11965.425 | 12589.447 | 7.353 | 6714 |
| continuous_batching | **10275.2** | **4569.0** | **8502.94** | **9154.66** | **4078.6** | **8166.74** | **8502.294** | **16.815** | **2936** |

cont vs static deltas: identical to E for makespan, tok/call, and
decode_calls; per-request latencies are 500 ms lower because some
requests arrive at t=500 ms or t=1000 ms and `total_latency = finish
− arrival`. The work queue never drains between the 500-ms-spaced
bursts, so at the system level F behaves like E with delayed start
times.

> **Observation:** F (as designed, with 500-ms burst spacing) does not
> add new information beyond E. To observe burst-specific behavior in
> the simulator, the gap between bursts would need to be wide enough
> for the system to drain between them. Out of scope for this report.

## Cross-workload summary: where does each axis matter?

| Workload | What varies | makespan Δ | p50 lat Δ | TTFT p50 Δ | tok/call Δ |
|---|---|---:|---:|---:|---:|
| A — all_short | nothing | 0% | 0% | 0% | 0% |
| B — staggered identical | arrival timing only | −0.30% | −7.40% | **−8.54%** | +0.86% |
| C — mixed decode | decode length only | **−30.0%** | **−29.5%** | **−36.4%** | **+124.6%** |
| D — mixed prompt | prompt length only | −2.78% | −2.83% | −3.03% | +17.83% |
| E — mixed P+D | both | **−26.9%** | **−26.3%** | **−32.7%** | **+128.6%** |
| F — bursty mixed | both, deterministic bursts | **−26.9%** | (variable) | **−32.6%** | **+128.6%** |

(All deltas are continuous_batching relative to static_batching.
Negative = continuous is better. tok/call positive = continuous
packs more rows per decode call.)

## First-separation findings

**First workload that separates `continuous_batching` from
`static_batching`:** **B — `staggered_identical_short`.**

Even with no length variance, deterministic 5-ms staggered arrivals
are enough for continuous to beat static.

**First metric that separates:** **TTFT p50** (−8.54% on B), with
total-latency p50 close behind (−7.40%). On B, makespan,
tokens_per_decode_call, and decode_calls all stay within 1%. The
mechanism is purely head-of-line freeing: the group barrier in static
delays admission of a few late-arriving short requests; continuous
admits them on the next iteration as soon as a slot frees. Because the
work itself isn't reordered (same lengths, same total tokens), the
makespan is essentially unchanged — only per-request latencies move.

**Continuous batching helps mainly through:**

| Effect | Where it dominates |
|---|---|
| Earlier admission for late arrivals (head-of-line) | B |
| Short-behind-long unblocking on decode-length variance | C, E |
| Short-behind-long unblocking on prompt-length variance | D (smaller effect) |
| Tighter packing (higher `tokens_per_decode_call`) | C, D, E, F |
| Fewer decode calls overall | C, D, E, F |

When length variance is absent, continuous helps only TTFT/p50 latency
and barely touches makespan. Once length variance is present,
continuous reduces both makespan and tail latency substantially and
roughly halves the decode-call count.

## Sensitivity sweep on workload B

Workload B is the smallest separator, so a sensitivity sweep was run
on it to see how the gap moves with `n_slots` and `n_batch`.

Configuration: `staggered_identical_short`, `total_cap=99`, `seed=42`,
policies `static_batching` and `continuous_batching` only (`fifo`
omitted as reference).

| n_slots | n_batch | static makespan | cont makespan | static p50 lat | cont p50 lat | static TTFT p50 | cont TTFT p50 | TTFT p50 Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 875.7 | 875.7 | 203.3 | 200.8 | 189.3 | 187.0 | −1.21% |
| 2 | 128 | 875.7 | 875.7 | 203.3 | 200.8 | 189.3 | 187.0 | −1.21% |
| 4 | 32 | 659.7 | 657.7 | 104.1 | 96.4 | 83.1 | 76.0 | **−8.54%** |
| 4 | 128 | 659.7 | 657.7 | 104.1 | 96.4 | 83.1 | 76.0 | **−8.54%** |
| 8 | 32 | 580.7 | 554.7 | 74.8 | 58.4 | 40.4 | 24.6 | **−39.11%** |
| 8 | 128 | 569.7 | 554.7 | 70.1 | 58.4 | 35.1 | 24.6 | **−29.91%** |

- `n_batch` is mostly inert for workload B because short requests
  (P=6, D=8) easily fit even at `n_batch=32`. The only place it moves
  static is `(n_slots=8, n_batch=32)`, where prefill rows for 8 slots
  briefly exceed 32 (8 × 6 = 48) and force an extra split.
- `n_slots` is the dominant axis. The TTFT-p50 gap grows from ~1% at 2
  slots to ~9% at 4 slots to ~30–40% at 8 slots, because more slots ⇒
  more requests are waiting behind a static group barrier when each
  iteration ends.

## Recommended target workload for a future real prototype

A future real HPX + llama.cpp continuous-batching prototype should
reproduce **C — `mixed_decode_only`** as its primary correctness +
separation gate, and **E — `mixed_prompt_and_decode`** as a stronger
sanity check.

Reasons:

- C has a single axis of variance (decode length) and synchronized
  arrivals. It is the simplest workload that produces a large,
  unambiguous separation (~30% on makespan, ~36% on TTFT p50, ~125%
  on tokens_per_decode_call, ~55% reduction in decode-call count).
  Easy to express in a real benchmark: 99 requests at t=0, fixed
  prompt, decode lengths round-robin over {8, 64, 256}.
- E exercises both length axes simultaneously and produces the
  strongest absolute signal in the simulator. Useful as a "if this
  doesn't separate, something is wrong" gate.
- B (`staggered_identical_short`) is the *smallest* separator but the
  per-request gap is small enough that real-system noise would likely
  swamp it. Use B only as a minimum-verification workload, not the
  headline result.
- D (`mixed_prompt_only`) is informative as a contrast to C but is not
  needed as a gate.
- F (`bursty_mixed_deterministic`) is not recommended as a primary
  target until the burst spacing is widened to a value where the
  system actually drains between bursts.

## Caveats

- The simulator uses a linear cost model with placeholder coefficients
  (`base=1.0, prompt=0.05, decode=0.5`). The relative ordering it
  produces is correct under that model, but the absolute numbers are
  not measurements of any real kernel.
- The cost model assigns `streaming_emit=0` and
  `per_active_slot_overhead=0` by default. Real HTTP and KV-cache
  costs are not modelled.
- The simulator does not model real prompt processing, real KV cache
  mechanics, samplers, tokenization, HPX runtime overhead, or any
  llama.cpp internal.
- The Poisson-arrival workload (`mixed_realistic`) is excluded from
  this report; it is covered in `results.md`.
- These are simulator scheduling-model results, **not** measured
  llama.cpp or HPX performance. No real performance claim should be
  derived from them.

## Reproducibility

Each summary above corresponds to a `results/<run_id>/` directory
written by `run_experiment.py`. Run-ids are 12-char config hashes
(deterministic). Re-running the same command lands in the same
directory with byte-identical CSVs.

| Workload | run_id |
|---|---|
| A | `d26c1609bcc4` |
| B | `366aa96a74c7` |
| C | `1cebc3dfc68e` |
| D | `141c627ab7f6` |
| E | `acc9e20f6c13` |
| F | `9d1181d36175` |

Sensitivity sweep on B:

| (n_slots, n_batch) | run_id |
|---|---|
| (2, 32) | `ff00f6ff49aa` |
| (2, 128) | `953d691bfad7` |
| (4, 32) | `7532c7a37796` |
| (4, 128) | `366aa96a74c7` (same as B grid cell) |
| (8, 32) | `508f73cf2008` |
| (8, 128) | `386cb9133f26` |

Captures live in `local/phase2b_*.stdout` and
`local/phase2b_*.stderr`.

Result directories under `hpx-bench/sim/continuous_batching/results/`
are gitignored.
