# Phase 3 target: real llama.cpp / HPX continuous-batching prototype

This is a **design note**, not an implementation. It defines the
target workload and scope for a future Phase 3 real prototype that
replaces the Phase 2 simulator with actual `llama.cpp` calls and adds
HPX as the orchestration layer.

No HPX code, no `tools/server/` change, no `tools/serving-bench/`
change, and no real performance claim is made here.

## Top-line conclusion

The simulator suggests that the first real prototype should target
**mixed decode lengths**, not the symmetric all-short Experiment-11
workload.

### Primary target workload — C: `mixed_decode_only`

Shape:

- All requests arrive at `t = 0`.
- `prompt_tokens` fixed at `6` for all requests.
- Decode mix: `{8, 64, 256}`, class assignment round-robin by
  request id (`rid % 3`).
- Initial config: `n_slots = 4`, `n_batch = 128`.
- Suggested initial size: 99 requests (so each class has 33).

Reason: this isolates decode-length heterogeneity and avoids
long-prompt prefill complexity, while still producing a large
simulator separation between `static_batching` and `continuous_batching`
(makespan ~−30%, TTFT p50 ~−36%, tokens_per_decode_call ~+125%,
decode_calls ~−55%). It is the cleanest, smallest workload that
exercises the head-of-line / short-behind-long mechanism.

### Secondary target workload — E: `mixed_prompt_and_decode`

Shape:

- All requests arrive at `t = 0`.
- Three (prompt, decode) classes round-robin: `(16, 8)`, `(128, 64)`,
  `(1024, 256)`.

Reason: more realistic — exercises both prefill and decode
heterogeneity simultaneously and gives the strongest absolute signal
in the simulator. It should come *after* C because long prompts add
prefill complexity (multi-iteration prefill, n_batch interactions,
slot.i_batch mapping pressure) that should not be debugged in the
same step as the slot-lifecycle plumbing.

### Workloads explicitly **not** chosen as the first real target

- A (`exp11_like_all_short`) — symmetric / no separation in simulator.
- B (`staggered_identical_short`) — separates only by ~7–9% on
  per-request percentiles; real-system noise would likely swamp this.
- D (`mixed_prompt_only`) — small simulator separation (~3%); not
  load-bearing relative to C.
- F (`bursty_mixed_deterministic`) — collapses into E in the simulator
  because the system does not drain between 500-ms-spaced bursts.

## Q1. What is the real prototype trying to reproduce from the simulator?

The simulator-level claim that on workload C:

- `static_batching` and `continuous_batching` produce **the same
  generated tokens** for the same inputs (correctness must match
  before performance is compared).
- `continuous_batching` reduces `tokens_per_decode_call` divisor — i.e.
  packs more rows into a shared batch and runs fewer total decode
  calls.
- Tail (p95/p99) total latency and TTFT for short requests drop
  substantially because they no longer sit behind long requests
  inside a static group.

The real prototype is trying to confirm that those scheduling-level
effects survive contact with the real `llama_decode` cost curve. It
is **not** trying to match the simulator's absolute milliseconds.

## Q2. Why is C the primary target?

- **Smallest workload that produces a large, unambiguous separation.**
  ~30% on makespan, ~36% on TTFT p50, ~125% on tokens_per_decode_call.
  Real-system noise will not swamp deltas of this size.
- **Single axis of variance.** Only decode length differs across
  requests. Prompt length is fixed at 6 tokens. This isolates the
  short-behind-long decode effect and keeps prefill cost effectively
  constant.
- **Synchronized arrivals.** All requests at `t = 0` removes arrival
  timing as a confound. Any separation that survives must come from
  the policy, not from how arrivals overlap.
- **Cheap to run.** Six tokens of prompt × 99 requests is tractable
  on CPU-only TinyLlama setups already validated in this branch.
- **Easy to specify in a real benchmark.** `n=99`, prompt fixed,
  decode round-robin in `{8, 64, 256}`, no arrival schedule, no
  bursts.

## Q3. What correctness gates would a real prototype need?

Before any performance claim:

1. **Same-tokens equality.** For every request, the generated token
   sequence under `continuous_batching` must match the sequence under
   `static_batching` (and ideally the simpler FIFO context-pool
   baseline) for the same `(model, prompt, sampler, seed)` setup.
   Compare via per-request hashes over generated token ids — the same
   correctness shape used in the existing FIFO closeout work.
2. **Per-class hash consistency.** Group hashes by `(prompt_len,
   decode_len)` class and require equality across policies, since
   the harness reports completion order, not submission order
   (see existing harness caveat).
3. **Slot lifecycle invariants** (the existing simulator §9 gates
   carried over to real code):
   - At any moment, no more than `n_slots` slots are active.
   - No slot owns more than one request.
   - No `seq_id` is reused while still in flight.
   - Each `seq_id` KV state is reset deterministically between
     requests.
4. **Batch shape gate.** `llama_batch.n_tokens ≤ n_batch` and
   `n_tokens ≤ n_ubatch` per real `llama_decode` call.
5. **Engine-thread ownership.** Exactly one thread issues
   `llama_decode`. No concurrent `llama_decode` against the same
   `llama_context`.
6. **Determinism gate.** Re-running the same workload with the same
   seed produces identical generated-token hashes per request.
7. **Termination gate.** `n_completed == n_submitted` for every run.

Performance claims may only be made after all seven gates pass.

## Q4. What metrics should match the simulator's units?

Per request:

- `arrival_time_ms`
- `assigned_at_ms` (when the request first occupies a slot)
- `first_token_time_ms`
- `finish_time_ms`
- `prompt_tokens`, `decode_tokens`
- `queue_wait_ms = assigned_at_ms − arrival_time_ms`
- `time_to_first_token_ms = first_token_time_ms − arrival_time_ms`
- `total_latency_ms = finish_time_ms − arrival_time_ms`
- `class_label`

Per iteration (one `update_slots()` cycle):

- `iteration_index`, `t_iter_start_ms`, `t_iter_end_ms`
- `prefill_rows`, `decode_rows`, `batch_size`
- `active_slots`
- `decode_calls_in_iteration`
- `iter_cost_ms` (in real code: measured wall time of the iteration)
- `admitted_this_iteration`, `completed_this_iteration`

Aggregates:

- `makespan_ms`, `n_completed`
- `total_latency_ms` p50 / p95 / p99
- `time_to_first_token_ms` p50 / p95 / p99
- `tokens_per_decode_call` (`sum batch_size / decode_call_count`)
- `decode_calls`
- Per-class p50 / p95 latency and TTFT

These are the columns already produced by the simulator
(`metrics.py`). The real prototype should emit the same column names
in the same units (milliseconds) so simulator runs and real runs can
be compared side-by-side without translation.

## Q5. What should HPX own?

HPX should own **request orchestration** around an opaque
`llama_decode`. Specifically:

- Request lifecycle as an HPX-async state machine
  (`waiting → admitted → prefill → decoding → done`).
- The waiting queue and admission policy
  (`fifo_context_pool`, `static_batching`, `continuous_batching`).
- Slot leases as RAII / future-based ownership; a lease ends when the
  request completes or is cancelled.
- Continuations that fire on iteration boundaries (e.g. "this slot
  freed → next waiter gets admitted next iteration").
- Cancellation propagation at iteration boundaries (out of scope for
  the first real prototype; see Q8).
- Streaming wiring (also out of scope for the first real prototype).
- HPX runtime startup/shutdown at the program entry layer (one-shot,
  process-wide). Engines must not start or stop HPX. Existing
  invariant from `CLAUDE.md`.

HPX must **not** parallelize inside `llama_decode` and must not own
tensor kernels. That is upstream's territory.

## Q6. What should llama.cpp own?

- The `llama_model`.
- The single shared `llama_context`.
- The shared `llama_batch` and its row layout.
- KV-cache primitives, including per-`seq_id` reset and clear.
- Tokenization and sampler state.
- The synchronous `llama_decode` call itself.
- The `slot.i_batch → logits` mapping that lets the orchestrator
  attribute output rows back to the owning slot.

The real prototype should treat `llama.cpp` as the kernel layer and
HPX as the request layer. The boundary is `llama_decode` — HPX
schedules around it but never inside it.

## Q7. What is the smallest real llama.cpp technical gate before HPX?

**Prove a minimal multi-sequence `llama_batch` decode with multiple
`seq_id`s in one `llama_context`.**

Concretely, before any HPX code is written:

- Build one `llama_context` with `n_seq_max ≥ 2`.
- Submit two synthetic requests using two distinct `seq_id`s within
  one `llama_batch`.
- Run a small number of `llama_decode` iterations that interleave
  prefill and decode rows for both `seq_id`s in the same batch.
- Verify that the per-`seq_id` KV state is independent: clearing one
  `seq_id` after completion does not corrupt the other.
- Verify that generated tokens for each `seq_id` match what the same
  prompt would produce under a single-sequence baseline (modulo
  sampler seed).

This is the upstream `llama-server` shape stripped of HTTP, slots,
and policy. If this technical gate does not hold, no scheduling
policy on top of it can be correct. HPX integration must wait until
this gate passes.

## Q8. What is explicitly out of scope for the first real prototype?

The first real prototype is a single-axis correctness + separation
proof. Out of scope:

- HTTP / network transport
- Streaming partial responses
- Cancellation
- Priority classes
- Speculative decoding
- LoRA adapters
- Multimodal inputs
- `n_cmpl > 1` parent/child KV fan-out
- Distributed serving / multi-host

Each of these is a separate axis. None of them are required to test
the C / E target workloads above. They should be re-evaluated only
after the primary target (C) and secondary target (E) produce
correct, repeatable results in the real prototype.

## Status

This document is **design only**. Phase 3 implementation is **not**
started.

Source-of-truth signals when revisiting this note:

- Simulator evidence: `hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md`.
- Upstream reading note: `docs/hpx/continuous_batching_upstream_notes.md`.
- Simulator design: `docs/hpx/continuous_batching_simulator_design.md`.

If any of those change in a way that invalidates the C / E target
choice, this note must be updated before Phase 3 starts.
