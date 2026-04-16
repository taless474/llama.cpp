## Claude Guidance for HPX / ggml Integration

### Read in this order

1. README_HPX.md
2. docs/HPX_EXECUTOR_CONTRACT.md
3. docs/HPX_PROVENANCE.md (optional, historical only, only read if needed)

### Core principles

- Prefer current architecture over historical designs
- Follow explicit contracts, not inferred behavior
- Keep changes minimal and localized
- Preserve separation of concerns

### Critical invariants

- Two execution substrates:
  - pthread → small work / decode
  - HPX → large work / prefill

- Selection is based on:
  cplan->work_size

- Scheduler logic ONLY in:
  ggml-hpx-adapter.cpp

- Plans must be structural only

- BLAS is opaque and not decomposed

### Forbidden

- Do not merge pthread and HPX into one executor
- Do not route decode work to HPX casually
- Do not leak scheduler outside adapter
- Do not store runtime pointers in plans
- Do not use provenance as design truth

### Preferred direction

- Move toward run_range(...)
- Use fine-grained region DAG
- Reduce thread-centric execution

### One-line model

Small work → pthread
Large work → HPX
Future → region DAG

### Saving results

Never write benchmark output to `/tmp`.  All results stay inside the repo.

- `hpx-bench/results/<date>-<slug>/` — benchmark CSVs and logs
- `local/results/` — experiment summaries, environment notes, anything not suitable for git

Each run gets its own directory.  Minimum contents:
- `bench_<name>.csv` — raw CSV (stdout)
- `bench.log` — stderr (legend, done marker, any errors)
- `README.md` — short markdown summary: what was measured, key numbers, findings, open questions

Redirect output at the invocation step, not with a post-run `cp`.
Include the git commit hash and binary path in the summary so results are reproducible.


### Context entering Phase 5

By the end of Phase 4, the project had already established that the fine-region DAG was no longer only a scheduling abstraction:

- dependency-driven region-group execution was working
- `lane_scratch` and `reduction_buffer` had been proven by tests
- a real ggml-style CPU op, F32 RMS_NORM, had been lowered into a 3-region DAG:
  - `R0` partial sumsq per lane
  - `R1` single-thread finalize
  - `R2` lane-parallel apply
- correctness was validated end to end

So the central question changed from:

> Can this design work?

to:

> Can this design work at a useful performance granularity?

Phase 5 was therefore not an architecture-design phase in the abstract sense. It was a measurement and diagnosis phase.

---

### Phase 5 — performance diagnosis and granularity study

#### Goal

Determine whether the new HPX-native fine-region execution model is viable for real work at the tested granularity, and identify where the cost is coming from.

This phase was intentionally focused on:

- measurement
- overhead decomposition
- low-risk runner experiments
- deciding whether the current scheduling granularity is performance-viable

It was **not** about:

- adding more ops
- broad integration into llama / ggml graph execution
- rewriting the scheduler
- optimizing arithmetic kernels

---

## Phase 5A — baseline measurement

### Purpose

Establish a baseline comparison between:

- a direct scalar/reference RMS_NORM path
- the new 3-region HPX fine-region DAG path

### Work done

A new benchmark was created under:

- `hpx-bench/bench_hpx_rms_norm_f32.cpp`

with results saved under repo-local benchmark results directories rather than system temp space.

The benchmark measured:

- `ref`: scalar RMS_NORM baseline
- `dag`: real 3-region RMS_NORM DAG using production callbacks

Sweep dimensions included:
- multiple row sizes
- multiple lane counts

### Outcome

The benchmark showed that the new DAG path was much slower than the scalar reference at the tested sizes.

The first important signal was:

- correctness held
- but the path was **overhead-dominated**

This answered the first practical Phase 5 question:

> No, the current fine-region RMS_NORM execution is not yet a performance win.

---

## Phase 5B — overhead decomposition

### Purpose

Break the cost of the new path into understandable layers rather than treating “HPX overhead” as one undifferentiated number.

### Work done

The benchmark was extended with additional variants:

#### `direct`
A plain direct RMS_NORM path calling the production callbacks as ordinary function calls with no HPX and no region-group runner.

This answered:
> Are the production callbacks themselves expensive?

#### `dag_empty`
A synthetic near-no-op 3-region DAG with the same resource touch pattern but almost no math.

This answered:
> How much does the generic DAG machinery cost even when useful work is minimal?

### Main findings

The comparison ladder showed:

- `direct ≈ ref`
- `dag_empty >> direct`
- `dag` only modestly above `dag_empty`

This established:

1. the production callbacks and real RMS_NORM arithmetic were **not** the main problem
2. the dominant cost was the **generic per-call 3-region DAG machinery**

This was the key transition in understanding.

The project was no longer asking:
> Is HPX slow?

It was now asking:
> Is this scheduling granularity too fine for generic HPX DAG composition?

---

## Phase 5C — targeted overhead experiments

Once Phase 5B showed that the generic DAG path dominated cost, the project moved into small, targeted experiments to see which parts of that overhead were actually material.

### Remove the outer HPX crossing

A `nowrap` variant was introduced so the benchmark no longer paid an outer:

- `hpx::async([&]{ ... }).get()`

for each repetition.

This showed:

- removing the outer crossing saved a small but real amount of time
- the remaining floor stayed much larger than the pure math cost

Conclusion:
- the outer HPX boundary was **not** the main bottleneck

### Remove per-call heap churn inside the group runner

`ggml_hpx_run_region_group(...)` was given a small-buffer / stack-allocated fast path for tiny groups so that per-call heap allocations for runner bookkeeping could be avoided in the benchmarked RMS_NORM case.

The semantics remained unchanged:
- same Kahn topo logic
- same dependency-driven future composition
- same wait/join behavior

Re-running the benchmark showed that the min-time floor barely moved.

Conclusion:
- per-call heap allocation in the group runner was **not** the bottleneck

### Fused one-dispatch ceiling measurement

A fused benchmark path was added:

- `fused_async`

This performed the full RMS_NORM body inside one:

- `hpx::async(...).get()`

with no generic region DAG around the internal stages.

This created the decisive cost ladder:

- `direct`
- `fused_async`
- `dag_empty_nowrap`
- `dag_nowrap`

### Main findings

The fused result showed:

- `direct` = math only
- `fused_async` = math + one HPX post
- `dag_empty_nowrap` = 3-region DAG machinery with near-no-op callbacks
- `dag_nowrap` = real 3-region DAG

The comparison made the cost structure clear:

1. one HPX post was relatively cheap
2. the extra cost of the 3-region DAG itself was roughly a fixed several-microsecond tax
3. the real production callbacks added very little on top of the generic DAG machinery

This gave the cleanest Phase 5 conclusion:

> The dominant cost is not arithmetic, callback structure, outer HPX entry, or per-call heap churn.
> The dominant cost is the per-call generic 3-region HPX DAG dispatch/composition model at this granularity.

---

## What Phase 5 proved

Phase 5 proved something more precise than “fine-grained HPX is slow.”

It showed:

- the fine-region model is **correct**
- the production callbacks are **cheap**
- one HPX dispatch around coarser work is **plausible**
- the generic per-call tiny 3-stage DAG is **too expensive** at the tested sizes

This means the central issue is not merely “tuning HPX harder.”
It is the placement of the HPX runtime boundary.

The fine-region DAG remains valuable as:
- a semantic model
- a planning model
- a dependency/resource description

But Phase 5 showed that it is not necessarily the right **runtime scheduling granularity** for tiny per-op chains like this RMS_NORM example.

---

## Phase 5 conclusion

By the end of Phase 5, the project had a much sharper result:

> The current HPX-native fine-region representation is valid and expressive, but the generic per-call scheduling of tiny dependent region chains is too fine-grained to be performance-competitive at the tested sizes.

This was the important design insight.

Phase 5 therefore did not merely produce “bad benchmark numbers.”
It established where the HPX boundary likely needs to move:

- upward
- to coarser execution units
- so that HPX manages larger chunks of useful inference work rather than tiny per-op chains

That set up the next architectural question:

> What is the right coarse-grained HPX execution boundary for real inference work?