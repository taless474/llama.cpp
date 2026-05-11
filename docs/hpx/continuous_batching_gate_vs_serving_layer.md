# HPX continuous-batching gate vs. an HPX serving layer

After Live Admission Slices 1–7 the final stdout label is
`HPX_CB_ADMIT_STEP7: PASS`. The
`tools/hpx-continuous-batch-gate/` binary now exercises a
complete end-to-end admission surface against a real
`llama_context` on a Metal-enabled build. This note is the
explicit boundary between **what that surface proves** and
**what a real HPX serving layer would still have to add on
top of it**.

The note is descriptive only. No performance claim is made
or implied.

## 1. Purpose

Two questions keep coming up around the HPX continuous-
batching work:

1. *Does the prototype already replace `llama-server`?*
2. *If not, what does it prove, and what is missing?*

This document gives a single short answer to both, framed
against the actual evidence captured in
`tools/hpx-continuous-batch-gate/results.md` and the design
in `docs/hpx/continuous_batching_live_admission_design.md`.

## 2. What the gate proves

Each slice introduces exactly one new behavior and lands a
strict pass/fail label on stdout. Slices 1–7 together
demonstrate that the HPX prototype:

| Slice | Surface proven | Label on PASS |
|---|---|---|
| 1 | live-admission data model only (`admission_source`, `request_id`, `admitted_at_iter`, `previous_request_id`, `reused_seq_id` plumbed; defaults always) | `HPX_CB_ADMIT_STEP1: PASS` |
| 2 | construction-time **waiting queue**, no admission yet (`--n-active`, `--n-waiting`, `--waiting-budget` introduced; queued/end-size sampled) | `HPX_CB_ADMIT_STEP2: PASS` |
| 3 | **cancel-freed-slot live admission** (waiting requests bound to slots freed by cooperative cancellation, prefilled in a mixed batch at `cancel_after+1`, run to completion alongside survivors; per-source promise/future plumbing) | `HPX_CB_ADMIT_STEP3: PASS` |
| 4 | admission-specific **traces and descriptive metrics** (`request_queued`, `seq_reused`, `request_admitted_live`, `admitted_prefilled`, `admitted_decode_row`, `admitted_complete`; descriptive metrics block extended) | `HPX_CB_ADMIT_STEP4: PASS` |
| 5 | **completion-freed-slot admission** under a demand gate (`--reuse-completed`; second engine deque `free_due_to_completion_`; per-source priority rule defined: cancel-freed first, completion-freed second) | `HPX_CB_ADMIT_STEP5: PASS` |
| 6 | **async external arrivals** (single scripted HPX submitter task, `engine::submit(arrival_msg)` under an `hpx::spinlock`-guarded inbox, deterministic **release + ack barrier** for visibility, `arrival_source=external` propagated end-to-end) | `HPX_CB_ADMIT_STEP6: PASS` |
| 7 | **mixed-source admission priority** end-to-end (`free_due_to_cancel_` and `free_due_to_completion_` both non-empty at the same admission boundary; cancel-freed drains first; completion-freed residual is unchanged) | `HPX_CB_ADMIT_STEP7: PASS` |

Carried-over correctness invariants gated on every Slice 1–7
run:

- exactly one HPX engine task per repeat;
- one `hpx::promise<request_result>` per seq, fulfilled only
  after KV clear + cross-talk check pass;
- every cancelled seq emits `status = cancelled` with
  `n_decoded == n_decoded_at_cancel` and **does not** appear
  in any later decode batch;
- residual KV is empty across all `n_seq_max` slots at engine
  end;
- `--repeat 2` is deterministic on the per-result tuple
  `(seq_id, request_id, n_decoded, generated_tokens, hash,
  done_iter, pos_max_at_clear, admitted_at_iter,
  reused_seq_id, previous_request_id, admission_src,
  arrival_src)`;
- with `LLAMA_HPX_CB_TRACE` unset, the binary emits zero
  `[hpx-cb-gate] event=` lines (the trace path is one atomic
  load + early return per call site);
- engine asserts `inbox_.empty()` and
  `external_promises_.empty()` before the residual-KV sweep.

## 3. What HPX owns

The HPX side of the prototype owns **orchestration only**:

- **Request lifecycle.** `request_id`, `decode_budget`,
  `arrival_source`, `admission_source`, `admitted_at_iter`,
  `previous_request_id`, `reused_seq_id`, `done_iter`,
  `pos_max_at_clear`, `status`, `cancel_observed_iter`,
  `n_decoded_at_cancel` — all live on `seq_state` /
  `request_result`, populated by HPX-side code, never by
  llama.cpp.
- **Slot / request state machine.** Waiting → admitted →
  prefilled → decoded → completed/cancelled. The admission
  loop, the demand-gated completion-freed pool, and the
  source-priority rule (cancel-freed before completion-
  freed) are all HPX-side decisions.
- **Exactly one engine task.** Scheduled via `hpx::async` per
  repeat. The engine task is the sole code path that touches
  llama.cpp execution state (see §4).
- **Futures and promises.**
  `hpx::promise<request_result>` per slot, one
  `hpx::future<void>` for the engine task. Main waits via
  `hpx::wait_all`. Admitted-request futures (preloaded
  waiters) are drained via `engine::take_admitted_futures()`
  after the engine task returns; external-arrival futures
  are owned by main and joined alongside the originals.
  Result snapshots are byte-copy `request_result` values —
  no future consumer ever touches llama.cpp objects.
- **Cooperative cancellation.** Observed only at iteration
  boundaries (top of decode iter, post-prefill). No
  `llama_decode` call is interrupted. The cancellation path
  is HPX-driven (predicate, KV clear, promise fulfillment);
  llama.cpp only executes the KV clear primitive when asked.
- **Async external arrivals.** A single scripted HPX
  submitter task pushes `arrival_msg` values into the engine
  inbox via `engine::submit()` under an `hpx::spinlock`.
  Timing is deterministic: the engine sets a release promise
  at end of iter K, suspends on the matching ack future, and
  resumes only after the submitter sets the ack. No
  `std::thread`, no `std::condition_variable`, no wall-clock
  sleep, no new `std::mutex`. The submitter helper body must
  not call any `llama_*` API; a grep-gate at review time
  enforces that.
- **Policy hooks.** Admission ordering, demand-gated pool
  push, source priority, cancel/admit one-iter delay,
  per-result determinism guarantees — all HPX-side. None of
  these decisions live inside llama.cpp.

## 4. What llama.cpp still owns

Everything that actually runs the model. The engine task is
the only owner; nothing outside the engine task touches any
of these:

- `llama_model`, `llama_context`, `llama_batch`.
- `llama_decode` (every call returns 0 is a gated invariant).
- `llama_memory_seq_*` (KV state per `seq_id`, including
  `_pos_min`, `_pos_max`, `_rm` for the per-seq clear).
- `llama_get_logits_ith`, `argmax` over the logit vector.
- Tokenizer (`common_tokenize`, `llama_vocab_is_eog`) and
  the vocab / EOG behavior.
- All compute kernels and backend behavior (Metal CPU
  fallback, `llama_synchronize`).
- The actual sequence of generated tokens, the model's
  outputs, and the per-shape hash anchors (e.g. the budget-8
  canonical anchor `0x0619d4d1900c2365` is an output of
  llama.cpp's execution on this prompt; the HPX side does
  not synthesize it).

This is a hard rule: every Slice 1–7 review gate audits that
no llama_* call escapes the engine task and that the
submitter helper body calls none at all.

## 5. What this is not yet

The gate is not a serving layer. Specifically, it does
**not** yet have:

- **No network arrival path.** No HTTP / gRPC / Unix-socket
  entry point. Requests in the gate come from `main()` (the
  preloaded waiting queue) and from a single scripted HPX
  submitter task running in the same process. There is no
  parser, no chat template, no /completions or /chat
  endpoint.
- **No multi-tenant / multi-context lifecycle.** Exactly one
  `llama_model` and one `llama_context`, created once at
  `main()` and torn down once at exit. No model hot-swap, no
  per-request model selection, no per-tenant isolation.
- **No streaming.** The gate fulfills `hpx::promise<...>`
  with a complete `request_result` snapshot **after** the
  request reaches its budget (or is cancelled). There is no
  per-token incremental emission to a downstream consumer.
- **No priority queue or scheduling fairness.** Admission
  within each source is FIFO by `request_id`. Source-
  priority is fixed at compile time (cancel-freed before
  completion-freed) and does not consider per-request
  priority, deadline, or fairness.
- **No multi-cycle slot reuse.** A `seq_id` is reused at
  most once per run (one cancel-then-admit chain, or one
  complete-then-admit chain). Chaining waiting → admitted →
  completed → second-waiting → second-admitted on the same
  slot is out of scope.
- **No per-request prompts.** Every request shares the same
  prompt string ("Hello, my name is" by default). There is
  no chat-template machinery, no system prompt, no per-
  request sampling configuration (temperature, top-k, etc.).
- **No persistence / disconnect-tolerance.** No prompt
  cache, no LoRA hot-swap, no save/restore of KV across
  process restarts. A submitter disconnect is not modelled
  (the submitter is a single HPX task scheduled by `main`).
- **No multi-K release schedule.** Slice 6 / 7 use exactly
  one release barrier per run. Multiple release iters in a
  single run is out of scope.
- **No wall-clock arrival schedules.** Timing is iter-
  deterministic via release+ack; nothing in the gate
  depends on `std::chrono::steady_clock::now()` for
  correctness (only for descriptive `ttc_ms` reporting).
- **No comparative performance evidence.** The descriptive
  metrics (`wall_ms`, `decode_calls`, `rows_per_batch`,
  `ttc_ms`) are observational; they are not framed as
  speed-up vs. an `llama-server` baseline, vs. an
  `std::async` baseline, or vs. a single-engine non-HPX
  baseline. The gate explicitly disclaims that comparison.

## 6. Why the gate matters

Even without server integration, the gate is the load-
bearing artifact for two upstream decisions:

1. **HPX/llama.cpp ownership boundary is exercised, not
   hypothetical.** Every Slice 1–7 PASS is signed off by
   ~99 concurrent `request_result` snapshots flowing through
   `hpx::future`, with all model execution funneled through
   one engine task. If the engine boundary were leaky, or if
   `engine::submit()` mutated llama.cpp state, or if any
   future consumer touched `llama_memory_seq_*`, no Slice
   would have hit PASS. The boundary is now testable
   evidence rather than design intent.
2. **The async-arrival and mixed-source priority surfaces
   are deterministic.** This is the prerequisite for any
   serving layer that has to multiplex external clients onto
   one `llama_context`: the layer needs to be able to admit
   late arrivals into slots freed by either cancellation or
   completion, in a defined order, without wall-clock races.
   Slices 6 and 7 prove that surface against canonical hash
   anchors and `--repeat 2` determinism, on a real Metal
   build.

If those two properties had not held, a real HPX serving
layer would have nothing to be built on top of. With them
held, a serving layer is a stack of well-scoped additions
rather than a rewrite.

## 7. Next bridge toward a serving layer

The smallest defensible step from "gate" to "serving layer"
is *not* "add HTTP and call it done". It is, in dependency
order:

1. **Multi-prompt support inside the engine.** Today every
   `request_result` is generated against the shared prompt.
   The first real serving-layer slice should let
   `arrival_msg` carry its own prompt token vector and have
   the admission step prefill that prompt on the freed slot.
   Per-result hash anchors stop being shape-canonical;
   determinism is gated within each `(prompt, budget,
   admission_src)` partition.
2. **Per-request sampling configuration.** Temperature,
   top-k / top-p / min-p, seed. Carried on `arrival_msg`,
   applied by the engine's post-decode argmax replacement.
   Determinism contract is then `(prompt, sampling_seed,
   budget)`-scoped.
3. **Streaming `hpx::channel` per request.** Replace the
   single-shot `hpx::promise<request_result>` with a
   per-request streaming channel that emits one token (or
   one chunk) per decode iter. The engine task remains the
   single producer; downstream consumers (network layer,
   logging, evals) attach as channel readers. The end-of-
   stream sentinel carries the existing `request_result`
   snapshot.
4. **External submission entry point.** A library API:
   `engine::submit(prompt, sampling, budget) ->
   hpx::future<stream_handle>`. The current scripted
   submitter becomes one of many submitters. The release+
   ack barrier shape stays — the binding is "external HPX
   task" rather than "scripted helper".
5. **Multi-request priority + fairness policy.** The
   source-priority rule (cancel-freed before completion-
   freed) is fixed. Per-request priority / deadline / token-
   budget fairness is a separate slice that adds an
   ordering function over the waiting queue, gated by a
   deterministic mapping for the regression smoke.
6. **Network adapter.** Only after 1–5 are in place. The
   adapter translates an HTTP / gRPC / Unix-socket request
   into a call into the library API from step 4. It owns
   no llama.cpp state. A serving binary is then this
   adapter + the engine + the existing HPX runtime startup.
   The engine and the adapter live in different translation
   units; the engine has no `#include` for any network
   library.
7. **Cross-process / multi-context surface.** Out of scope
   for v1. A future slice introduces a second
   `llama_context`, an admission-aware scheduler across
   contexts, and the corresponding ownership invariants.

Each step is a stop-and-check slice in the same shape as
Slices 1–7: smoke shape, expected mappings, strict gate,
final stdout label, closeout evidence in `results.md`.

## 8. Safe claim language

When summarizing this work in design docs, READMEs, status
posts, or talks, the following claims are **supported by
the gate** and safe to make verbatim:

- "The HPX prototype proves a deterministic continuous-
  batching admission surface — including cancel-freed,
  completion-freed, and async-external admission, with a
  fixed source-priority rule — under a single-engine, single-
  context, single-prompt smoke against a real Metal build of
  llama.cpp."
- "Every admission path is gated on per-request
  `hpx::promise<request_result>` fulfilment, per-seq KV
  clear, cross-talk checks against still-active siblings,
  and `--repeat 2` determinism on the per-result snapshot
  tuple."
- "Only the engine HPX task touches `llama_context`,
  `llama_batch`, `llama_decode`, `llama_memory_seq_*`, or
  `llama_get_logits_ith`. The scripted submitter calls no
  `llama_*` API. The engine inbox is guarded by an
  `hpx::spinlock`; coordination uses `hpx::promise<void>` /
  `hpx::future<void>` and a release+ack barrier. No
  `std::thread`, no `std::condition_variable`, no wall-clock
  sleep, no new `std::mutex` is introduced."

The following claims are **not** supported by the gate and
should be avoided:

- "The HPX prototype replaces `llama-server`." (No network
  adapter, no streaming, no per-request prompts, no chat
  template — see §5.)
- "The HPX prototype is faster than X." (No comparative
  benchmark is run; the gate is a correctness-and-lifecycle
  artifact, not a performance artifact.)
- "HPX manages the model." (HPX owns orchestration; the
  model and KV state are owned by llama.cpp — see §3
  vs. §4.)
- "Async external arrivals are wall-clock-driven."
  (Determinism is via release+ack; wall-clock is used only
  for descriptive `ttc_ms` reporting and is **not** part of
  the determinism contract.)

When in doubt: the gate is a correctness/lifecycle
checkpoint on the road to a serving layer, not the serving
layer.
