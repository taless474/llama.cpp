# Continuous Batching — Phase 2 Simulator Design

This document specifies the Phase 2 simulator. It does not implement it. The simulator's job is to model the serving-runtime behavior we learned from upstream `llama-server` (see `docs/hpx/continuous_batching_upstream_notes.md`) at a level high enough to compare scheduling policies, but low enough that the policies map 1:1 onto what we'd later build with HPX around llama.cpp.

The simulator is **not** an inference engine. There are no real tokens, no logits, no sampler, no KV cache. It is a discrete-event scheduler whose only job is to produce the same iteration shape, the same row-packing behavior, and the same lifecycle transitions that upstream produces, parameterized by a simple cost model.

---

## 0. Grounding (what we carry over from Phase 1)

Anchored in `docs/hpx/continuous_batching_upstream_notes.md`:

- `slot.id` is the `llama_seq_id` by construction; one slot owns its sequence id for the lifetime of the process.
- `update_slots()` owns the engine loop. Per iteration: ctx-shift → clear shared batch → append decode rows for GENERATING slots → (if `cont_batching`) append prompt rows for PROCESSING_PROMPT slots → one chunked `llama_decode` call → sample/process/emit per slot.
- The shared `llama_batch` is built per iteration. Each row is `(token, pos, seq_id={slot.id}, logits_bool)`. Total rows are capped at `n_batch`.
- Continuous batching is one boolean (`params.cont_batching`) that gates whether prompt rows are added when a decode row is already present.
- Cancellation only takes effect at iteration boundaries; the in-flight `llama_decode` is not interrupted.
- Streaming partials are emitted from inside the engine loop (`process_token` → `send_partial_response`).
- `n_cmpl > 1` parent/child KV fan-out (`llama_memory_seq_cp`) is **out of scope for v1**.

These properties are the simulator's behavioral contract. Anything that doesn't follow from them is intentionally not modeled.

---

## 1. Request

The unit of work submitted into the simulator. One request maps to one final response.

```
Request:
  request_id        : int           # monotonically increasing
  arrival_time_ms   : float         # absolute, simulator clock units
  prompt_tokens     : int           # how many prefill rows this request needs
  decode_tokens     : int           # how many decode rows this request will produce
  priority          : int           # OPTIONAL, ignored in v1 (reserved for v2)
  cancel_at_ms      : float | None  # OPTIONAL, ignored in v1 (reserved for v2)
```

v1 invariant: `prompt_tokens > 0` and `decode_tokens > 0`. Embeddings/rerank tasks (which are `decode_tokens == 0` in upstream) are excluded.

A request has no real text. It is a triple `(P, D, t_arrive)` plus an id. The simulator's job is to schedule it through a slot and report timings.

## 2. Slot

The simulator's analogue of `server_slot`. Reflects the Phase 1 finding that `slot.id` is also the `seq_id`.

```
Slot:
  slot_id              : int                 # also the seq_id
  seq_id               : int                 # == slot_id, kept for parity with upstream
  request_id           : int | None          # None when phase == waiting
  phase                : enum {
                            waiting,         # idle, no request assigned
                            prefill,         # PROCESSING_PROMPT analogue
                            decode,          # GENERATING analogue
                            done             # transient: scheduled to release this iteration
                         }
  prompt_remaining     : int                 # rows still to add to prefill batches
  decode_remaining     : int                 # rows still to add to decode batches
  generated_tokens     : int                 # ticks up by 1 per decode row consumed
  first_token_time_ms  : float | None        # time of the first decode row
  finish_time_ms       : float | None        # time when phase moved to done
  assigned_at_ms       : float | None        # time when request entered this slot (queue_wait end)
```

We deliberately collapse `STARTED → PROCESSING_PROMPT` into a single `prefill` phase. We collapse `DONE_PROMPT → GENERATING` into the `prefill→decode` transition. We omit `WAIT_OTHER` (n_cmpl>1 fan-out is out of scope for v1).

Slot count is fixed at startup (`n_slots`) and slots are reused — same as upstream.

## 3. Scheduler state

A single struct holding the simulator world.

```
SchedulerState:
  current_time_ms       : float
  waiting_queue         : deque[Request]      # FIFO; tail = newest
  active_slots          : list[Slot]          # length == n_slots
  completed_requests    : list[Completion]    # accumulated per-request results
  n_slots               : int
  n_batch               : int                 # max rows per shared decode call
  n_ubatch              : int | None          # OPTIONAL (used only by sanity checks in v1)
  cont_batching         : bool
  iteration_count       : int                 # number of update_slots() loops executed
  decode_call_count     : int                 # number of "llama_decode" calls (matters per-policy)
```

`active_slots` always has length `n_slots`. A slot in `phase == waiting` is "free."

`Completion` is the per-request output written when a slot reaches `done`:

```
Completion:
  request_id
  arrival_time_ms
  assigned_at_ms              # slot acquisition time
  first_token_time_ms
  finish_time_ms
  prompt_tokens
  decode_tokens
  queue_wait_ms               # = assigned_at_ms - arrival_time_ms
  prefill_ms                  # = first_token_time_ms - assigned_at_ms
  decode_ms                   # = finish_time_ms - first_token_time_ms
  total_latency_ms            # = finish_time_ms - arrival_time_ms
  ttft_ms                     # = first_token_time_ms - arrival_time_ms
```

## 4. Batch builder

One iteration of the engine produces one **logical batch** of rows. The batch is the simulator's analogue of the shared `llama_batch`. The builder follows upstream `update_slots`'s two-pass shape:

```
build_batch(state, policy):
  rows = []
  # PASS A: decode rows for every slot in phase == decode
  for slot in state.active_slots:
    if slot.phase == decode and slot.decode_remaining > 0:
      rows.append(BatchRow(slot_id=slot.slot_id, kind=decode))

  # PASS B: prompt rows for every slot in phase == prefill,
  #         gated by policy on whether and how many to add
  prefill_rows_allowed = policy.prefill_admission(rows, state)
  for slot in state.active_slots:
    if len(rows) >= state.n_batch:
      break
    if slot.phase == prefill and slot.prompt_remaining > 0:
      take = min(slot.prompt_remaining,
                 state.n_batch - len(rows),
                 prefill_rows_allowed)
      for _ in range(take):
        rows.append(BatchRow(slot_id=slot.slot_id, kind=prefill))
      prefill_rows_allowed -= take

  return rows
```

`policy.prefill_admission(...)` is what differentiates fifo / static / continuous (see §6).

We record `len(rows)` per iteration as `batch_size_over_time`.

A `BatchRow` carries the minimum needed for sanity checking: `slot_id`, `kind`. We don't model token positions, masks, or attention.

## 5. Cost model

Pluggable per-iteration cost function. v1 uses a small linear model.

```
CostParams:
  base_decode_step_cost_ms       : float    # fixed per llama_decode call
  per_prompt_token_cost_ms       : float    # per prefill row in a shared call
  per_decode_token_cost_ms       : float    # per decode row in a shared call
  per_active_slot_overhead_ms    : float    # optional, default 0.0
  streaming_emit_cost_ms         : float    # optional, default 0.0 (v1)
```

Per-iteration cost (for policies that share rows in one call — static and continuous):

```
iter_cost_ms =
    base_decode_step_cost_ms
  + per_prompt_token_cost_ms * (#prefill rows)
  + per_decode_token_cost_ms * (#decode rows)
  + per_active_slot_overhead_ms * (#active slots)
  + streaming_emit_cost_ms      * (#decode rows that emit a token)
```

For policies that do NOT share rows across slots (fifo_context_pool — see §6), each *active slot* contributes its own decode call inside the iteration:

```
iter_cost_ms =
    sum over active slots of (
      base_decode_step_cost_ms
      + per_prompt_token_cost_ms * (slot's prefill rows this iteration)
      + per_decode_token_cost_ms * (slot's decode rows this iteration)
      + streaming_emit_cost_ms   * (slot's decode rows this iteration)
    )
  + per_active_slot_overhead_ms * (#active slots)
```

This is the key qualitative difference: the `base_decode_step_cost_ms` overhead is paid **once per iteration** in static/continuous, but **once per active slot per iteration** in fifo. That's where continuous batching's throughput win shows up.

Calibration plan (out of scope for v1, called out for completeness): once we have a real `serving-bench` run on the same TinyLlama shape, fit `(base, per_prompt, per_decode)` to the per-iteration timings. Until then, sane defaults inferred from the M4/CPU repack baseline are loaded from a YAML/JSON config; the simulator must not hardcode them.

## 6. Policies to compare

Three policies. Each is a thin object that customizes (a) when a request leaves the waiting queue and enters a slot, (b) what `prefill_admission(...)` returns, (c) whether decode is shared across slots in one call.

### 6.1 `fifo_context_pool`

The current `llama-serving-bench` shape: every request owns its own context for its full lifetime; there is no shared decode step across slots.

- Admission: as soon as any slot is `waiting`, pop the head of `waiting_queue` and assign it.
- Prefill admission: each slot processes its own prefill rows independently. `prefill_admission` is unbounded (each slot fills up to `n_batch` of its own rows, but rows from different slots **never share a decode call**).
- Decode model: each active slot is its own decode call. The cost model uses the per-slot formula above.
- Iteration shape: per simulator iteration, each active slot advances by either a prefill chunk (up to `n_ubatch` rows) or one decode token, paid as its own decode call. `decode_call_count += #active_slots` per iteration that has any active slot.

This is included as the **floor**: under the simulator's cost model, `static_batching` and `continuous_batching` should amortize the per-call base cost across slots while `fifo_context_pool` does not. Whether this translates to a real-world speedup depends on calibration against measured llama.cpp behavior; the simulator does not assert it as a correctness gate.

### 6.2 `static_batching`

Fill the slot pool, run them as a group, do not admit new work until everyone in the group is done.

- Admission: only when ALL slots are `waiting` simultaneously (group barrier). At that point, pop up to `n_slots` requests from `waiting_queue` and assign them.
- Prefill admission: same as continuous (rows can share a call). `prefill_admission` is bounded only by remaining `n_batch`.
- Decode model: shared call across all slots in the group. `decode_call_count += 1` per iteration with at least one row.
- Iteration shape: from group-start to group-finish, the iteration looks identical to continuous batching. The difference is the admission rule — between groups, the engine sits idle even if work is queued and slots have just freed up.

### 6.3 `continuous_batching`

Upstream behavior with `--cont-batching`.

- Admission: as soon as any slot is `waiting`, pop the head of `waiting_queue` and assign it. Slots can finish and be refilled mid-iteration; the simulator handles that at iteration boundaries (a finished slot transitions to `waiting` at the *end* of its last iteration; the next iteration's admission step assigns it).
- Prefill admission: bounded only by `n_batch - #decode_rows`. Rows from different slots ARE batched in one call, both prefill and decode together.
- Decode model: one shared call per iteration. `decode_call_count += 1`.
- Iteration shape: matches the Phase 1 description of `update_slots`: PASS A (decode rows) + PASS B (prefill rows under cap) + one decode call.

### Policy contract

Each policy object exposes:

```
Policy:
  name                                  : str
  admit(state)                          : void        # may move requests from waiting_queue → slots
  build_iteration(state)                : Iteration   # produces rows + cost-model selector
  apply_iteration(state, iteration)     : void        # advances slot phases, decrements remainings,
                                                       #   marks done, records timings, releases slots
```

`Iteration` carries:
- `rows: list[BatchRow]`
- `cost_mode: enum {shared, per_slot}`
- `decode_calls: int` (1 for shared, k for per_slot where k = #slots that contributed rows)

This is small enough to keep policies side-by-side in one file without inheritance.

## 7. Workloads

Two initial workloads, both deterministic, both reproducible from a single seed.

### 7.1 Experiment-11-like

Mirrors `hpx-bench/experiments/11_perf_deep_queue_short_requests/` (the readme the user has open). The point is: a deep queue of identical short requests is exactly where continuous batching is supposed to dominate fifo_context_pool.

```
n_requests       = 200
arrival_time_ms  = 0   for all requests          # all-at-t=0 burst
prompt_tokens    = 6
decode_tokens    = 8
priority         = 0
cancel_at_ms     = None
```

Comparison: run `{fifo_context_pool, static_batching, continuous_batching}` × `n_slots ∈ {1, 2, 4, 8, 16}` × `n_batch ∈ {32, 128, 512}`. Cost params held constant across runs.

Expected qualitative result (to be validated, not asserted):
- On symmetric all-at-once workloads with identical request lengths, static batching and continuous batching may tie because there are no staggered arrivals or length differences for continuous admission to exploit. Continuous batching is expected to differ most on mixed or bursty workloads where slots free at different times.
- Both beat `fifo_context_pool` because the `base_decode_step_cost_ms` is amortized across slots in the cost model.
- The continuous-vs-static gap widens with workload heterogeneity (varied lengths, staggered arrivals), not with `n_slots` alone.

### 7.2 Mixed realistic

Bursty arrivals across three request classes. Used in v2 for fairness/priority studies.

```
short:    prompt = 16,   decode = 8
medium:   prompt = 128,  decode = 64
long:     prompt = 1024, decode = 256
```

Arrival pattern: configurable. v1 default is a Poisson process with three burst windows:
- t ∈ [0, 1000ms): λ = 50 req/s, mostly `short`.
- t ∈ [1000, 3000ms): λ = 20 req/s, mix of `short` and `medium`.
- t ∈ [3000, 5000ms): λ = 5 req/s, mostly `long`.

The mix per window is a tunable dict `{short: p_s, medium: p_m, long: p_l}`. Total request count is bounded (default 500) so runs are comparable.

The mixed workload is the place to study tail latency under contention and (in v2) priority/fairness policies.

## 8. Metrics

Per-request, per-iteration, and per-run.

### Per-request

- `total_latency_ms`           = `finish_time_ms - arrival_time_ms`
- `queue_wait_ms`              = `assigned_at_ms - arrival_time_ms`
- `time_to_first_token_ms`     = `first_token_time_ms - arrival_time_ms`
- `decode_time_ms`             = `finish_time_ms - first_token_time_ms`
- `prefill_time_ms`            = `first_token_time_ms - assigned_at_ms`

### Per-iteration

- `iteration_index`
- `t_iter_start_ms`, `t_iter_end_ms`
- `prefill_rows`, `decode_rows`, `batch_size`
- `active_slots`
- `decode_calls_in_iteration`
- `iter_cost_ms`

### Per-run (aggregate)

- `makespan_ms`               = `max finish_time_ms - min arrival_time_ms`
- `tokens_per_second`         = `sum(prompt_tokens + decode_tokens) / makespan`
- `decode_calls`              = `sum decode_calls_in_iteration`
- `tokens_per_decode_call`    = `sum batch_size / decode_calls`
- `active_slots_over_time`    = time series, sampled per iteration
- `batch_size_over_time`      = time series, sampled per iteration
- p50/p90/p95/p99 of `total_latency_ms`
- p50/p90/p95/p99 of `time_to_first_token_ms`
- fairness by request length: p50/p95 latency split by (`short`, `medium`, `long`) for the mixed workload, or by `decode_tokens` bucket for synthetic workloads

The aggregate report must be reproducible: same workload + same cost params + same policy + same seed → byte-identical CSV.

## 9. Correctness / sanity gates

Invariants that the simulator MUST hold and assert at every iteration boundary. A run with any violation aborts and writes the violation to the run summary. These gates are the simulator's own correctness contract — not a measurement of upstream.

1. **Every request completes exactly once.** Across the run, each `request_id` appears in `completed_requests` exactly once.
2. **No slot owns more than one request at a time.** For any slot, while `phase != waiting`, `request_id` does not change.
3. **`generated_tokens <= decode_tokens`** for every request, at every iteration.
4. **`completed_requests.size() == submitted_requests.size()`** at end of run.
5. **No negative remainings.** `prompt_remaining >= 0` and `decode_remaining >= 0` at every iteration.
6. **`active_slots <= n_slots`.** (Where "active" means `phase != waiting`.)
7. **`batch_size <= n_batch`** for every iteration.
8. **Continuous batching admits into free slots only after slots are released.** A request must not enter a slot while that slot still owns a previous request. In code terms: `slot.request_id` transitions only `(None → R) | (R → None)`, never `(R → S)`.
9. **fifo invariant.** In `fifo_context_pool`, no two slots' rows share a decode call. (Asserted by the per-iteration cost-mode flag.)
10. **static-batching barrier.** Between groups, all slots must reach `waiting` simultaneously before the next group is admitted. The simulator must record group boundaries.
11. **First-token monotonicity.** For each request, `first_token_time_ms >= assigned_at_ms`. For each request, `finish_time_ms >= first_token_time_ms`.
12. **Time monotonicity.** `current_time_ms` is non-decreasing across iterations.

A separate optional gate, useful for the experiment-11 workload:

13. **Tokens-per-decode-call sanity.** For continuous batching with `n_slots > 1` on the experiment-11 workload, `tokens_per_decode_call` must be `> 1.0` (i.e. continuous batching is actually packing rows, not degenerating to per-slot calls).

## 10. What v1 intentionally ignores

The simulator does not model:

- **Real token values.** Rows are scalars `(slot_id, kind)`; there are no llama_token ids.
- **Logits.** No sampling, no probabilities, no top-k.
- **Sampling.** `common_sampler_*` and its time cost; in v1 sampling is folded into `per_decode_token_cost_ms`.
- **KV memory fragmentation.** The simulator does not track per-slot KV cell positions, `pos_min/pos_max`, or memory occupancy.
- **Context shifting.** Long requests do not trigger shifts; we cap by sizing inputs to fit `n_ctx`.
- **Prefix cache reuse.** `slot_prompt_similarity`, `n_cache_reuse`, `server_prompt_cache` — none of these are modeled; every request prefills its full `prompt_tokens`.
- **`n_cmpl > 1` parent/child fan-out.** No `llama_memory_seq_cp` analogue; one request = one slot, one seq_id.
- **Real HTTP streaming.** `streaming_emit_cost_ms` exists in the cost model but is 0 by default; we do not simulate SSE, chunked encoding, or socket writes.
- **Real cancellation.** `cancel_at_ms` field exists in `Request` but is ignored in v1.
- **Real HPX runtime overhead.** No `hpx::async`, no future creation cost, no executor scheduling cost.
- **MTMD / multimodal / embeddings / rerank / infill / LoRA / alora / speculative decoding.** All upstream paths that aren't completion are out.
- **Sleeping mode, router mode, MCP proxy.** Server-lifecycle features not relevant to scheduling.

This list exists so that a reviewer can quickly map "is this thing intentionally absent or genuinely missing?"

## 11. What v2 may add

In rough priority order. None of these are mandatory for the first implementation.

- **Cancellation at iteration boundaries.** Honor `cancel_at_ms`; at the start of each iteration, scan active slots for requests whose `cancel_at_ms <= current_time_ms` and release the slot before building the batch. This mirrors upstream's iteration-boundary cancellation.
- **Priority scheduling.** Use `Request.priority` as a tie-breaker in `waiting_queue` ordering. Add a `priority_inversion_count` metric.
- **Streaming emit cost.** Set `streaming_emit_cost_ms > 0` and study how it perturbs throughput. Required to model the engine-thread serialization point we identified in Phase 1.
- **KV capacity pressure.** Add a `kv_used` counter per slot and a global `n_ctx_total`. When the queue exceeds capacity, defer admission. Maps onto upstream's `kv_unified` behavior.
- **Mixed prefill/decode batching limitations.** Model `n_ubatch` as a real cap on how much prefill can land in one decode call; today v1 lets prefill rows fill up to `n_batch`.
- **Arrival processes.** Multiple synthetic processes (Poisson, MMPP, replay-from-trace).
- **Fairness policies.** Token-based fair queuing across requests, age-based aging, deadline-aware scheduling.
- **n_cmpl>1 fan-out.** Add `parent/child` slot relationships and a `seq_cp` analogue. Probably belongs in v3.

## 12. Acceptance for Phase 2 implementation

The first implementation must produce:

1. **A runnable simulator** under `hpx-bench/sim/continuous_batching/`.
2. **Deterministic workloads.** Given the same `(workload_name, seed, cost_params, policy, n_slots, n_batch, cont_batching)`, the simulator must produce byte-identical output. No wallclock-dependent behavior, no thread races (single-threaded simulator).
3. **CSV outputs**, written under `hpx-bench/sim/continuous_batching/results/<date>-<slug>/`:
   - `requests.csv` — one row per request: id, arrival, assigned, ttft, finish, prompt_tokens, decode_tokens, queue_wait_ms, total_latency_ms, prefill_ms, decode_ms.
   - `iterations.csv` — one row per iteration: index, t_start, t_end, prefill_rows, decode_rows, batch_size, active_slots, decode_calls, iter_cost_ms.
   - `summary.csv` — one row per (policy, workload, n_slots, n_batch) configuration: makespan, tokens_per_second, decode_calls, tokens_per_decode_call, p50/p95/p99 of total_latency_ms, p50/p95/p99 of ttft_ms.
4. **A summary text** (`summary.md` per run) — human-readable per-run report, including: input config, sanity-gate verdict (`PASS` / `FAIL` with violation list), aggregate metrics, and a pointer to the CSVs.
5. **Comparison output** for the experiment-11 workload across the three policies. Must include side-by-side numbers for `makespan_ms`, `tokens_per_decode_call`, and p95 `total_latency_ms`. The output reports whether the policies differ; "continuous wins" is not a correctness gate. On symmetric workloads with identical lengths and arrivals, static and continuous may tie.
6. **No HPX dependency.** Pure Python, no `pyhpx`, no `hpx::*`. The simulator must run in any environment with a recent Python.
7. **No real model dependency.** No `llama.cpp` linkage, no GGUF file loading, no model weights. Pure scheduling.
8. **No `/tmp` writes.** Per CLAUDE.md, results go under `hpx-bench/sim/continuous_batching/results/<date>-<slug>/`. Local-only ad-hoc captures may go under `local/`. Never `/tmp`.
9. **Reproducibility line.** Each run must emit one line at the top of `summary.md` that contains the full command, seed, config hash, and simulator git revision. A reader should be able to reproduce the run from that single line.
10. **Sanity gates wired in.** All §9 invariants are checked at every iteration boundary. A violation aborts the run, writes the violating state to `summary.md`, and exits non-zero.

The first implementation does **not** need to:
- Implement v2 features.
- Calibrate the cost model from real serving-bench data (placeholder defaults are fine, but they must be visible in the config and overridable from the CLI).
- Render plots. CSV + text is enough; plotting can be a separate consumer.

## 13. Proposed file layout (Phase 2 implementation, not created yet)

```
hpx-bench/sim/continuous_batching/
  readme.md           # what this is, how to run, what the outputs mean
  facts.md            # state-of-the-simulator: cost params, last calibration, known caveats
  sim.py              # SchedulerState, Slot, the engine loop (build_batch, apply_iteration)
  workloads.py        # experiment_11_workload(), mixed_realistic_workload(), seedable
  policies.py         # FifoContextPool, StaticBatching, ContinuousBatching
  metrics.py          # per-request / per-iteration / per-run aggregation, CSV writers
  run_experiment.py   # CLI entry: parses config, runs one or many policies, writes results/<date>-<slug>/
  summarize.py        # consumes results/<date>-<slug>/ → summary.md, p50/p95/p99 tables, side-by-side comparisons
  results/            # gitignored except for evidence we explicitly commit
```

Per-file rough scope:

- `sim.py` — `~400 LOC`. The discrete-event loop, `Slot`, `SchedulerState`, `Iteration`, the cost-model evaluator, and the §9 sanity gates. Pure logic, no I/O.
- `workloads.py` — `~150 LOC`. Two named workloads + a small workload-builder API for future workloads. Returns deterministic `list[Request]` from a seed.
- `policies.py` — `~250 LOC`. Three small policy classes implementing the §6 contract. Pure logic, no I/O.
- `metrics.py` — `~200 LOC`. Aggregations + CSV writers. The only file allowed to write to disk other than `run_experiment.py`.
- `run_experiment.py` — `~150 LOC`. CLI parser, config loader, result-dir creation, calls into `sim.py` and `metrics.py`. The only file with `argparse`.
- `summarize.py` — `~150 LOC`. Reads `results/<date>-<slug>/` and produces `summary.md` plus the side-by-side comparison table.
- `readme.md` — short. How to run, what the outputs mean, how to add a workload, how to add a policy.
- `facts.md` — running log of what the simulator currently believes about cost parameters, calibration runs, and known limitations. Updated when calibration changes.

Implementation note (not approval to start): the simulator should be one process, single-threaded, deterministic. All policies, workloads, and cost params should be selectable from one CLI without touching code.

---

## Out of scope, again, for clarity

This document does not propose:

- Writing simulator code.
- Modifying `tools/server/` or `tools/serving-bench/`.
- Any HPX integration. The simulator is a stand-alone tool that runs without HPX, exists to validate scheduling, and will inform a later HPX prototype that wraps real `llama.cpp`.
- Any performance claim about HPX, llama.cpp, or the existing `serving-bench`.

When the simulator implementation is approved separately, the deliverables in §12 are the contract.
