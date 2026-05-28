# HPX-Native Serving Control Plane — Design

Status: design only. No code in this slice.

This document defines what an HPX-native serving control plane for
`llama.cpp` looks like, what is already in place as a first scaffold,
and what remains target design. It is dataflow-first. It does not
implement features and it does not propose changes outside of HPX
control-plane ownership.

It complements, and does not restate, the existing milestone summary
and design notes:

- `docs/hpx/hpx_serving_layer_m0_m8_milestone_summary.md`
- `docs/hpx/continuous_batching_foundation.md`
- `docs/hpx/continuous_batching_lifecycle_design.md`
- `docs/hpx/continuous_batching_streaming_design.md`
- `docs/hpx/serving_overhead_diagnostics_roadmap.md`

Throughout this document, two phrases are kept strictly separate:

- **Current scaffold** — what is implemented today on this branch
  (M0–M8 plus the Phase 1 diagnostics and `/shutdown` precursor).
  Verifiable in `tools/hpx-continuous-batch-gate/` and
  `tools/hpx-server/`.
- **Target design** — what an HPX-native serving control plane is
  intended to become. Not implemented. Used to judge anti-patterns and
  to direct the next slices.

The current scaffold has the right shape at the boundary but is not a
fully realised target. Saying so explicitly is a design rule, not a
caveat.

---

## 1. Problem statement

The risk this design avoids is producing a project that is just
"llama-server rewritten with `hpx::future` sprinkled around". That
would be a thread-pool dressing on a request/response server and would
not justify HPX being in the architecture at all.

The intended thesis of the project is different: HPX owns the
*serving control plane*, while `llama.cpp` continues to own model
execution. The control plane is the set of decisions and state that
governs *which* sequence is decoded *when*, with *what* inputs, under
*what* policies (admission, cancellation, backpressure, streaming
delivery, sampling identity, diagnostics, later prefill/decode
scheduling and multi-engine routing).

Concretely, the failure mode the design rejects:

- one HPX task per request directly calling `llama_decode`;
- using `hpx::future` only as a return value from a synchronous
  request handler;
- duplicating `llama-server` slot logic without HPX-owned policy
  hooks;
- adding benchmarks, latency claims, or scheduling heuristics before
  ownership and dataflow are explicit;
- treating HPX as a replacement thread pool rather than a control
  plane.

The intended target shape:

- one long-lived, actor-like HPX engine task is the sole owner of
  `llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`, and per-seq sampler
  chains;
- producers (HTTP handler, cancel issuers, shutdown trigger) reach
  the engine *only* through HPX future / channel-style boundaries;
- per-request state lives in HPX-owned snapshots and per-request
  HPX promises / receive channels — never as a back-pointer into
  engine internals;
- admission, cancellation, streaming delivery, backpressure,
  diagnostics, and later prefill/decode scheduling and multi-engine
  routing are all expressed as control-plane decisions taken on
  iteration boundaries, not as ad-hoc handler logic.

The two-day goal is to land this design as a defensible HPX-native
shape, not to fully implement every policy hook it names.

---

## 2. Boundary

This boundary is the load-bearing invariant of the entire design and
matches the project-level rule in `CLAUDE.md` and the M0–M8 summary's
boundary table.

`llama.cpp` owns:

- `llama_model` and `llama_context` execution semantics
- `llama_decode`
- `ggml` graph and backend kernels (CPU / Metal / etc.)
- tokenizer and `llama_vocab` behavior
- sampler math (`llama_sampler_*`)
- KV memory implementation

HPX owns:

- request lifecycle (submission, admission, decode-loop scheduling,
  completion, cancellation, shutdown)
- admission policy
- active-sequence scheduling within the engine task
- prefill/decode policy
- cancellation policy and identity (`cancel_token{rid, epoch}`)
- backpressure (currently door-cap; target: HPX-side queueing)
- HPX futures / per-request promises
- streaming delivery and per-request stream channel lifetime
- metrics / diagnostics
- later: multi-engine routing, priority / deadline scheduling

Hard rules across the entire design:

- Only the engine task may touch mutable `llama.cpp` execution state.
- No parallel `llama_decode` on the same `llama_context`.
- No back-pointer into engine internals from outside the engine.
- No silent fallback to a non-HPX path.

The boundary is not a stylistic preference; it is the only thing that
keeps HPX from becoming a thread pool around `llama_decode`.

---

## 3. Current dataflow (scaffold)

This is what runs today. It is the M7 + M8 serving surface with the
Phase 1 diagnostics layer and the diagnostic-only `/shutdown` endpoint.

```text
                       client (HTTP)
                            │ POST /completion  { prompt, decode_budget,
                            │                     stream?, sampling? }
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ cpp-httplib accept thread → cpp-httplib worker thread     │
   │                                                           │
   │ hpx-server /completion handler (foreign thread to HPX):   │
   │   1. parse JSON                                           │
   │   2. validate sampling (shared validator)                 │
   │   3. capacity_lease acquire (--max-concurrent door)       │
   │      → 503 server_busy on overflow                        │
   │   4. common_tokenize(vocab, ...)                          │
   │   5. eng.submit_request(submit_request{...})              │
   │      → submit_handle { result, stream?, token }           │
   │   6a. non-streaming: h.result.get() → JSON response       │
   │   6b. streaming: set_chunked_content_provider lambda;     │
   │       on each invocation, h.stream->get(hpx::launch::sync)│
   │       drains one token_stream_event and sink.write's      │
   │       the SSE chunk. On sink.write false, set             │
   │       cancel_issued and eng.cancel_request(token).        │
   └───────────────────────────────────────────────────────────┘
                            │ engine API boundary
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ engine inbox_chan_  (HPX local channel)                   │
   │   inbox_msg_kind:                                         │
   │     arrival       (carries arrival_msg)                   │
   │     cancel_rid                                            │
   │     cancel_token                                          │
   │     shutdown                                              │
   └───────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ engine task (single long-lived HPX task, sole owner of    │
   │   llama_context / llama_batch / KV / sampler chains)      │
   │                                                           │
   │ run_body() per repeat:                                    │
   │   reset per-seq state, batch_init                         │
   │   prefill pass (one shared llama_decode over preloaded    │
   │     actives + per-seq sampler/argmax for first token)     │
   │                                                           │
   │   inner while (any_active                                 │
   │                || waiting_queue                           │
   │                || inbox_has_pending):                     │
   │     pump_inbox_nonblocking()    ── stage from channel     │
   │     drain_cancel_inbox()        ── cancels first          │
   │     drain_external_inbox(iter)  ── arrivals into queue    │
   │     apply_queued_cancellations()                          │
   │     snapshot admission_eligible_count                     │
   │     iter_observe_cancellations(iter)                      │
   │     iter_run_admissions(iter)  ── 3-source priority drain │
   │                                  (cancel_freed,           │
   │                                   completion_freed,       │
   │                                   initial_idle)           │
   │     iter_build_batch(iter)     ── prefill + decode rows   │
   │     iter_run_decode(batch)     ── ONE llama_decode call   │
   │     iter_sample_and_finalize() ── per-seq sampler/argmax, │
   │                                  publish_token,           │
   │                                  finalize_and_fulfill or  │
   │                                  cancel_and_fulfill       │
   │     iter_fire_release_ack_barrier(iter)  ── gate-test     │
   │                                                           │
   │   outer keep-alive loop:                                  │
   │     drain cancel + apply + pump                           │
   │     if shutdown && no work: drain_waiting_queue, break    │
   │     if staged work: continue                              │
   │     wait_inbox_blocking()  ── HPX-native suspend on chan  │
   └───────────────────────────────────────────────────────────┘
                            │ per-seq stream_channel.set(token_event)
                            │   when stream_enabled
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ per-request HPX local channel                             │
   │   token_stream_channel<token_stream_event>                │
   │   sole producer: engine task                              │
   │   sole consumer: hpx-server SSE provider lambda           │
   │                  (via foreign-thread get(hpx::launch::sync)) │
   └───────────────────────────────────────────────────────────┘
                            │ per-request promise.set_value(rr)
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ request_result snapshot                                   │
   │   no llama_context reference; no engine pointer;          │
   │   carries hash, generated_tokens, status, cancellation    │
   │   timestamps, admission source, arrival source, etc.      │
   └───────────────────────────────────────────────────────────┘
```

Per-stage annotation:

| Stage | Owner | Thread / task | Blocking point | Cancellation path | Diagnostic counters |
|---|---|---|---|---|---|
| HTTP accept | cpp-httplib | cpp-httplib accept thread | TCP accept | n/a | none |
| /completion handler | hpx-server | cpp-httplib worker (foreign to HPX) | `h.result.get()` (non-stream) | n/a | `request_metrics.submit_us` |
| Capacity door | hpx-server | cpp-httplib worker | none (atomic CAS) | n/a | `capacity.in_flight` |
| Tokenize | llama.cpp (const vocab) | cpp-httplib worker | none | n/a | none |
| `submit_request` | engine API | cpp-httplib worker (publish-only) | spinlock acquire on `submit_publish_mtx_` | none here | `next_epoch_` |
| Inbox channel | HPX | producer = any, consumer = engine task | producer never blocks; engine suspends via `wait_inbox_blocking` | n/a | n/a |
| `pump_inbox_nonblocking` | engine | engine task | none | stages cancel msgs | n/a |
| `drain_cancel_inbox` | engine | engine task | none | promotes staged → `cancelled_request_ids_` / `cancelled_tokens_` | `cancel_request_calls`, `cancel_request_duplicates` |
| `drain_external_inbox` | engine | engine task | none | short-circuits cancel-before-submit | `arrival_drained_count`, `first_external_drain_iter` |
| `apply_queued_cancellations` | engine | engine task | none | flips `seq.cancel_requested` for live actives; resolves queued cancels | `queued_cancelled`, `cancel_active_observed`, `cancel_unknown_request_id`, `cancel_stale_epoch` |
| `iter_observe_cancellations` | engine | engine task | none | `cancel_and_fulfill` runs KV clear + stream close + promise | `cancelled_per_iter` |
| `iter_run_admissions` | engine | engine task | none | admitted seq carries cancel flag if cancel-before-admit raced | `admitted_per_iter`, `admission_iter_set`, `waiting_queue_depth_after_admission_per_iter` |
| `iter_build_batch` | engine | engine task | none | n/a | `prefill_rows_per_iter`, `decode_rows_per_iter`, `rows_per_batch` |
| `iter_run_decode` | engine | engine task | `llama_decode` synchronous CPU/GPU call | n/a (cooperative; not preempted) | `llama_decode_wall_us_per_iter`, `decode_calls` |
| Sampling | engine | engine task | `llama_sampler_sample` or `argmax` | n/a | `tokens_emitted_per_iter` |
| `publish_token` | engine | engine task | local channel `set` (non-blocking) | n/a | `stream_tokens_emitted_total` |
| `finalize_and_fulfill` | engine | engine task | `promise.set_value` | n/a | `promises_fulfilled`, `completed_per_iter` |
| SSE provider | hpx-server | cpp-httplib worker | `h.stream->get(hpx::launch::sync)` (blocks foreign thread) | `sink.write` false → `cancel_issued` + `eng.cancel_request(token)` | `stream_channel_get_count`, `sse_write_count`, `sse_write_bytes`, `disconnect_observed` |
| `/shutdown` (diag) | hpx-server | cpp-httplib worker | none (`srv.stop()` is non-blocking-from-handler) | n/a | n/a |
| `wait_inbox_blocking` | engine | engine task | HPX-native suspend on channel shared state | woken by submit / cancel / shutdown | `engine_idle_waits` |

Key observations about the current scaffold:

- Producer side of the inbox is reached from foreign threads
  (cpp-httplib workers) calling `eng.submit_request` /
  `eng.cancel_request`. The producer call does not touch
  `llama.cpp` state; it copies a POD + a promise + an optional
  channel handle into a message and `set`s the inbox channel.
- The engine task is the sole consumer of the inbox channel and the
  sole mutator of all `llama.cpp` state. The single-mutator
  invariant is enforced structurally by the engine's API rather than
  by runtime checks.
- Cancellation is cooperative and iteration-boundary only: an
  in-flight `llama_decode` is never preempted. The engine observes
  cancels between iterations.
- Streaming delivery and result fulfillment both cross the
  engine→consumer boundary by HPX-native primitives
  (`hpx::lcos::local::channel` and `hpx::promise`); no shared mutable
  state and no foreign locks.

This is *the right shape at the boundary*. It is not the full target
design — see §11 / §12.

---

## 4. HPX-native target dataflow

The target dataflow keeps the same boundary as §3 but makes the
control-plane decisions explicit and adds the policy hooks the
scaffold does not yet have.

```text
                       client (HTTP/SSE today; optionally HPX-side
                       transport later, behind the same engine API)
                            │
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ adapter layer (today: cpp-httplib; tomorrow: any I/O      │
   │  source that can produce submit_request + consume         │
   │  request_result / token_stream_receiver)                  │
   │                                                           │
   │ HPX-OWNED ADMISSION GATE  ◀── target, not in scaffold     │
   │   - HPX-side bounded queue, not a foreign atomic counter  │
   │   - admission policy (FIFO / class-aware / deadline)      │
   │   - per-class concurrency caps                            │
   │   - explicit reject reason → adapter translates to wire   │
   └───────────────────────────────────────────────────────────┘
                            │ submit_request via engine API
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ engine inbox (HPX local channel, today's primitive is     │
   │  load-bearing here too)                                   │
   └───────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ engine task (single long-lived HPX task; same boundary)   │
   │                                                           │
   │ explicit per-iter phases, each a named control-plane      │
   │ decision rather than ad-hoc code:                         │
   │                                                           │
   │   PUMP        ── drain inbox into engine-task-only stage  │
   │   CANCEL      ── apply queued / token cancels             │
   │   ADMIT       ── three-source priority drain + policy     │
   │                  hook: PREFILL_VS_DECODE_INTERLEAVE       │
   │   BUILD       ── compose llama_batch; policy hook:        │
   │                  PREFILL_BUDGET_PER_ITER                  │
   │   DECODE      ── one llama_decode call (sole site)        │
   │   SAMPLE      ── per-seq sampler / argmax                 │
   │   PUBLISH     ── stream channel set / finalize promise    │
   │   IDLE        ── HPX-native suspend on inbox channel      │
   │                                                           │
   │ Each phase emits Phase 1 diagnostics rows; later phases   │
   │ may add structured decisions ("policy chose ADMIT 3 of    │
   │ 7 eligible: reason class-cap").                           │
   └───────────────────────────────────────────────────────────┘
                            │
            per-request   ──┴──   per-iter
            promise +              metrics / events
            stream channel
                            │
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ consumer side                                             │
   │   - per-request future / receive channel                  │
   │   - per-iter diagnostics stream                           │
   │   - target: optional HPX-side per-request policy reactor  │
   │     (e.g. backpressure feedback into the admission gate)  │
   └───────────────────────────────────────────────────────────┘
```

The target dataflow does not replace `llama_decode`, does not
parallelize inside the engine task, and does not change the boundary.
What it adds is:

- An HPX-owned admission gate that lives *upstream of* the engine
  inbox, replacing the cpp-httplib door-cap.
- Named policy hooks at ADMIT and BUILD that the engine task
  consults instead of hard-coding scheduling.
- A diagnostics surface that records what the policy *decided*, not
  only what happened.

The boundary stays identical between current and target. Only HPX-side
ownership and structure grow.

---

## 5. Engine actor model

Wording, on purpose:

- The engine is described as an *actor-like, long-lived HPX task*
  whose mailbox is an HPX local channel. The word "actor" is shorthand
  for the property "single owner of internal state, communicates only
  via async messages and futures". It is not a claim that this design
  uses a generic actor framework.
- Cross-boundary communication is described as *HPX future /
  channel-style boundaries with a single-mutator engine task*. The
  primitives are `hpx::future`, `hpx::promise`, and
  `hpx::lcos::local::channel`.
- We do not claim "HPX channels as actor mailboxes" as a general
  pattern. The engine inbox happens to be an HPX channel, but other
  per-request channels are stream consumers — different roles, both
  HPX-native.

Current scaffold:

- One `engine` instance owns one `llama_context`.
- `engine::run()` is launched on an HPX task via
  `hpx_runtime::async_on_engine`, which transparently routes onto a
  named single-PU engine pool when `--engine-pool` is enabled and
  falls through to a bare `hpx::async` otherwise.
- The engine task is the sole mutator of: `llama_context`,
  `llama_batch`, KV (`llama_memory_seq_*`), per-seq
  `llama_sampler_ptr`, per-seq `seq_state`, and all "engine-internal"
  state (`free_due_to_cancel_`, `free_due_to_completion_`,
  `free_idle_`, `external_promises_`, `live_epoch_by_rid_`,
  `cancelled_request_ids_`, etc.).
- Producers reach the engine through three public message kinds on
  `inbox_chan_`:
  - `inbox_msg_kind::arrival` (carries an `arrival_msg`),
  - `inbox_msg_kind::cancel_rid` and `cancel_token`,
  - `inbox_msg_kind::shutdown`.
  Each producer call is short, foreign-thread-safe, allocates only
  the message, and acquires no lock other than the brief spinlock in
  `submit_request` that pairs epoch issuance with the channel `set`.
- The engine suspends idle via `wait_inbox_blocking()`, which awaits
  on the channel's shared state — no `std::condition_variable`, no
  busy poll.

Target design adds:

- Explicit *per-iter phase* names in the engine, so future scheduling
  policies attach to phase boundaries rather than to ad-hoc decision
  sites. (Already partly in code: see `iter_observe_cancellations`,
  `iter_run_admissions`, `iter_build_batch`, `iter_run_decode`,
  `iter_sample_and_finalize`, `iter_fire_release_ack_barrier`.)
- Policy objects (admission, prefill/decode split, eventually
  priority) injected via `engine_options` instead of being implicit
  in the iter body.
- A symmetric per-engine *outbox* concept for structured policy
  decisions (e.g. "rejected 1 admission this iter, reason
  class-cap"). The current Phase 1 diagnostics already record some of
  this for completion / cancel events; the target makes it general.

Anti-pattern this section rules out:

- one HPX task per request directly calling `llama_decode`. The
  current scaffold already enforces this structurally; the design
  doc keeps it explicit so it survives future refactors.

---

## 6. Request lifecycle model

Request state machine (current scaffold and target are the same here):

```text
   client submits HTTP /completion
            │
            ▼
   ADAPTER_RECEIVED  (cpp-httplib worker has the body)
            │
            ▼
   ADAPTER_GATED     (capacity_lease acquired; current scaffold
                      is a foreign atomic counter; target is an
                      HPX-owned admission gate)
            │
            ▼
   ENGINE_QUEUED     (submit_request published to inbox_chan_;
                      engine task has not yet drained)
            │
            ▼
   ENGINE_WAITING    (drain_external_inbox pushed to
                      waiting_queue_consumable_; not yet bound)
            │
            ▼
   ENGINE_ADMITTED   (admit_one bound to a free slot at iter K;
                      seq_state populated, prompt rebound, sampler
                      built; seq is "active")
            │
            ▼
   ENGINE_PREFILLING (admitted-prefill rows emitted on iter K
                      batch; first token sampled post-iter)
            │
            ▼
   ENGINE_DECODING   (one decode row per iter; n_decoded grows;
                      publish_token per token if streaming)
            │
            ▼
   ENGINE_FINALIZING (EOG, budget reached, or cancel observed)
            │     │
            │     └─▶ KV clear + cross-talk check
            │     └─▶ close_stream(completed | cancelled | error)
            │     └─▶ fulfill_promise(status=...)
            │
            ▼
   CONSUMED          (caller's future.get() returns the snapshot;
                      stream receiver has seen its closed event)
```

Failure transitions:

- ADAPTER_GATED → REJECTED (`server_busy`) before ENGINE_QUEUED;
  no inbox entry, no engine work.
- ADAPTER_GATED → REJECTED (`invalid_sampling`,
  `payload_too_large`) before tokenize; same — no engine work.
- ENGINE_QUEUED → ENGINE_CANCELLED via cancel-before-drain
  shortcut in `drain_external_inbox` (epoch-aware).
- ENGINE_WAITING → ENGINE_CANCELLED via
  `apply_queued_cancellations` against the waiting queue.
- ENGINE_DECODING → ENGINE_CANCELLED via
  `cancel_and_fulfill` at the next iter boundary after the cancel
  flag is observed.
- Any active state → SHUTDOWN_ABORTED via
  `drain_waiting_queue_for_shutdown` and the run-tail drain in
  `engine::run()`.

The state machine is *engine-task-owned* end-to-end after
ENGINE_QUEUED. The adapter never sees any of those states directly;
it sees a future and an optional receive channel.

Target design extends this with:

- a small per-request observability struct (a subset of
  `request_result` fields available *before* completion), accessible
  to the diagnostics layer without changing the public boundary;
- structured `reject_reason` codes coming back from the admission
  gate to the adapter, so the wire-layer translation
  (HTTP 503 / 422 / etc.) does not encode policy.

---

## 7. Admission and scheduling model

Current scaffold:

- Admission has three sources, drained in a fixed priority order
  inside `iter_run_admissions`:
  1. `cancel_freed` — slot freed by a same-iter cancellation, eligible
     only at iter K+1 (one-iter delay snapshot).
  2. `completion_freed` — slot freed by natural completion, pooled
     only when `--reuse-completed` is on and the waiting queue is
     non-empty (demand-gated).
  3. `initial_idle` — slot from the engine's startup idle pool,
     reserved at engine construction for `--initial-idle-slots`.
- Slot reuse is governed by KV-empty invariants in `clear_and_check`,
  validated by cross-talk checks.
- Cancellation has identity (`cancel_token{rid, epoch}`) so a reused
  `request_id` cannot be hit by a stale cancel.
- Per-iter active scheduling is implicit: every active seq emits one
  decode row each iter, plus any admitted-prefill rows the same
  iter.
- Backpressure is at the adapter door (`--max-concurrent`), not in
  HPX. A 503 means "do not even tokenize, do not submit".

Target design adds *named policy hooks* that the engine task consults
rather than hard-coding:

- `AdmissionPolicy`:
  - decides *how many* eligible waiters to admit per iter
    (currently: as many as freed slots permit);
  - decides *which* eligible waiter (today: insertion order; target:
    deadline / priority / class-balance);
  - decides whether to *defer* admission to allow batch shape
    optimisation.
- `PrefillBudgetPolicy`:
  - caps admitted-prefill rows per iter so a long prompt cannot
    starve concurrent decodes (today implicit, bounded only by
    `batch_capacity`).
- `ActiveSchedulingPolicy`:
  - which subset of active seqs to advance this iter; relevant once
    "prefill vs decode" interleave becomes explicit.
- `BackpressurePolicy`:
  - HPX-side bounded queue with HPX-native overflow signal back to
    the adapter; the adapter translates this to the wire-level
    response.

These hooks are *control-plane* objects. They do not touch
`llama.cpp` state and they do not run on foreign threads. They are
consulted on the engine task at the relevant phase boundary and they
emit a structured decision record.

Critical invariants the hooks must preserve (target and current):

- one `llama_decode` per iter, in the engine task;
- KV-empty before reuse, validated;
- cancellation identity preserved (no stale cancels);
- canonical anchors held within shape — see `CLAUDE.md`'s anchor
  rules.

---

## 8. Prefill / decode split

Current scaffold:

- The engine task runs one shared `llama_decode` per iter. The batch
  composed by `iter_build_batch` mixes admitted-prefill rows (each
  admitted prompt emits as many rows as its prompt length) and
  decode rows (one row per active seq).
- There is no separate scheduling of prefill versus decode. A large
  admitted prefill in iter K simply produces a fatter batch in iter
  K and delays the per-iter decode budget for that iter.

Target design separates *policy* from *execution* while keeping
execution one `llama_decode` per iter:

- The engine task remains the sole mutator of `llama_context` and
  the sole site that calls `llama_decode`. The single-mutator
  invariant is not negotiable.
- A `PrefillBudgetPolicy` (see §7) bounds admitted-prefill rows per
  iter so that one long prompt does not absorb the whole batch
  capacity. The policy is consulted in BUILD phase, before
  `llama_decode`.
- An `ActiveSchedulingPolicy` may choose to advance only a subset
  of active seqs in a given iter, leaving others' decode rows for
  the next iter. This is *not* parallel decode; it is row
  composition.
- Diagnostics already separate `prefill_rows_per_iter` from
  `decode_rows_per_iter` (Phase 1). The policy decisions can be
  recorded next to them as structured fields.

What the split does **not** do:

- It does not introduce a second engine task or a second
  `llama_context`. Those are separate items in §11 ("multi-engine
  routing") and out of scope for this slice.
- It does not preempt an in-flight `llama_decode`. Decisions are
  iteration-boundary only.

---

## 9. Streaming and backpressure

### 9.1 Streaming

Current scaffold:

- Per-request opt-in via `submit_request.want_stream`.
- Engine constructs a `token_stream_channel` and returns the
  receiver half in `submit_handle.stream`.
- Engine task is the sole producer: `publish_token` sets a
  `kind=token` event on the channel; `close_stream` sets a
  `kind=closed{reason=...}` event and closes the channel.
- Consumer: in the gate, `main()`; in the server, the
  `set_chunked_content_provider` lambda on a cpp-httplib worker
  thread, which does `h.stream->get(hpx::launch::sync)` per chunk
  iteration and `sink.write`s the SSE payload.
- Close reasons partition: `completed`, `cancelled`, `error`. The
  `engine::run()` tail drains any open channels with reason `error`
  so the consumer never hangs on the chain walk.
- Streaming detokenization is cumulative: emitted text =
  `common_detokenize(vocab, emitted_tokens, false)`; the SSE payload
  is the new delta. Engine still emits token ids only.

Target design keeps the same primitives but:

- Treats the per-request channel lifetime explicitly as a
  control-plane responsibility ("stream rights"), not implicit in
  the slot.
- Documents the foreign-thread consumer path
  (`get(hpx::launch::sync)`) as the *only* foreign-thread blocking
  point on the consumer side, so future I/O substrate changes
  (e.g. an HPX-side SSE writer) have a clean replacement target.

### 9.2 Backpressure

Current scaffold:

- `--max-concurrent` capacity door at the cpp-httplib adapter
  boundary. Rejects with HTTP 503 + `Retry-After: 1` +
  `error.code == "server_busy"` *before* tokenize / submit.
- `capacity_lease` is move-only and released exactly once: on
  handler exit, on stream cleanup, on disconnect-finalize.
- No HPX-side queueing beyond the strict door-cap. A request either
  fits or is rejected synchronously.

Target design:

- Move backpressure inside HPX as a bounded HPX-side admission gate
  (target ordering rule: HPX-owned queue first, adapter
  door-translation only at the wire boundary).
- Slow-client / partial-disconnect isolation: a slow consumer of a
  per-request stream channel must not stall the engine task. Today
  this is provided by the channel's buffered shared state; the
  target design names this as a policy hook
  (`StreamBackpressurePolicy`) so a future slice can choose between
  "drop oldest token" / "close stream with error" / "block engine"
  rather than rely on whatever the channel default does.
- "Drop" is *not* the default. Engine-task starvation on a slow
  client is the failure mode to avoid; the design rule is "engine
  must never block on a per-request consumer".

---

## 10. Cancellation model

Current scaffold:

- Identity: `cancel_token{request_id, epoch}`. Issued by
  `engine::submit_request` under the same spinlock that publishes the
  arrival, so epoch order matches drain order.
- Producers: `engine::cancel_request(int32_t rid)` and
  `engine::cancel_request(const cancel_token &)`. Both publish an
  `inbox_msg` and return; no engine state mutated by the caller.
- Stage / drain: `pump_inbox_nonblocking` stages on the engine task;
  `drain_cancel_inbox` promotes to `cancelled_request_ids_` /
  `cancelled_tokens_`. Idempotent.
- Resolution sites:
  - `drain_external_inbox` short-circuits a cancel-before-submit for
    an arrival that has not yet entered the waiting queue;
  - `apply_queued_cancellations` resolves a cancel against a
    waiting-queue entry (fulfilled with status `cancelled` and
    default sentinels) and flips `seq.cancel_requested` on a live
    active seq;
  - `iter_observe_cancellations` runs `cancel_and_fulfill` at the
    next iter boundary, which clears KV, closes the stream with
    `reason=cancelled`, and fulfills the promise.
- Wire-side trigger: HTTP/SSE disconnect → `sink.write` returns false
  → `state->cancel_issued = true` → `eng.cancel_request(token)`.
  Fire-and-forget; the truth is observed via the future.

Engine task is the sole site that fulfills the promise with
`status=cancelled`. The adapter never touches engine internals or
slot state.

Target design adds:

- Structured *cancel reasons*: `client_disconnect`,
  `admission_timeout`, `deadline_exceeded`,
  `server_shutdown`, `policy_evict`. Today every cancel is
  effectively "client-initiated".
- A diagnostics row per cancel attempt: which site resolved it
  (inbox-short-circuit, waiting-queue, active iter-boundary), how
  many iters elapsed between submit and observation.
- (Optional, future) preemption-friendly cancels for very-long
  decode budgets. Today cancel observability is bounded by one iter
  duration, which is acceptable for current shapes; this is not a
  near-term hook.

What cancellation will not do, in target design:

- It will not interrupt an in-flight `llama_decode` call. The
  single-engine-task and single-mutator invariants forbid it.
- It will not be issued from the engine task to its own producers.
  The flow is producer → inbox → engine task → promise; never
  reversed.

---

## 11. Diagnostics as design support

Diagnostics are not the design; they are the substrate that proves
or refutes design decisions.

Current scaffold (Phase 1, behind `LLAMA_HPX_DIAG_METRICS=1`):

- *Engine* per-iter JSONL with one row per inner-while iter, plus an
  `engine_summary` row. Fields cover batch shape, active seq count,
  prefill/decode row split, admitted / completed / cancelled deltas,
  tokens emitted, queue depth after admission, `llama_decode` wall
  time, total iter wall time, and idle wait time.
- *Server* per-request JSONL with one row per request, covering
  submit / first-token / completion timestamps, stream channel get
  count, SSE write count and bytes, disconnect observed, cancel
  issued, final status, decoded count.
- Output paths: stderr by default, or
  `LLAMA_HPX_DIAG_METRICS_PATH=<file>` (`O_APPEND`, line-buffered).
- Zero allocation cost when the env switch is unset. The engine
  accumulators are per-engine (`result_.metrics.cur_iter_diag`), not
  file-scope, so concurrent engines never share state.
- Diagnostic-only `POST /shutdown`, behind
  `LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1`, triggers `srv.stop()` so the
  engine reaches its tail and flushes JSONL — required for W1-style
  diagnostics against the real server.

Target design extends Phase 1 with two additions, neither of which
changes the boundary:

- *Decision rows*: structured records of admission / scheduling /
  cancellation *decisions*, not only their counters. Example:
  "iter=K, admitted=3/7 eligible, reason class-cap balance".
- *Per-request lifecycle rows*: a join key (`request_id`) between
  server-side rows, engine-side admission iter, and engine-side
  completion iter. Today the cross-walk is implicit in the JSONL;
  the target makes it queryable without re-parsing.

The W1 mixed short+long diagnostic plan (Phase 1 §7 in the
diagnostics roadmap) is the first user of this surface and remains
the recommended next experiment after this design lands. It is
explicitly paused for the duration of this design slice.

---

## 12. Ranking of future improvements

Ordered by how clearly each one falls inside HPX-owned control plane
and how much it strengthens the HPX-native shape (highest first).

1. **Prefill / decode scheduling split** (§8). The single most
   load-bearing HPX-native control-plane decision. Already has
   diagnostic support (`prefill_rows_per_iter` /
   `decode_rows_per_iter`). Needs `PrefillBudgetPolicy` and
   `ActiveSchedulingPolicy` policy hooks on the engine boundary.
2. **Adaptive admission / batching policy** (§7). Replace the
   implicit FIFO with a small `AdmissionPolicy` that can express
   per-class caps or deadline ordering. Backed by existing
   `admission_iter_set` and `waiting_queue_depth_after_admission`
   diagnostics.
3. **Slow-client / backpressure isolation** (§9.2). Move
   `--max-concurrent` inside HPX as a bounded admission gate, with
   a `StreamBackpressurePolicy` for per-request stream channels.
   Pre-requisite for any "multi-tenant" framing.
4. **Low-overhead streaming delivery**, *only if diagnostics
   justify it*. Today the cpp-httplib chunked-content provider with
   `get(hpx::launch::sync)` is acceptable. If W1 / future diagnostics
   show this path is the bottleneck, an HPX-side SSE writer becomes
   a candidate. Not before evidence.
5. **Multi-engine routing later**. Multiple engines, each owning
   one `llama_context`, sit behind a router. The router is HPX-side;
   each engine retains its single-mutator invariant. Out of scope
   until a single engine is fully exploited.
6. **Priority / deadline scheduling later**. Wired on the existing
   `AdmissionPolicy` and `ActiveSchedulingPolicy` hooks once those
   exist. Cancellation reason `deadline_exceeded` becomes a
   first-class wire signal.
7. **NUMA / distributed as separate architecture track**. Not in
   this design. HPX's distributed surface is a different program
   and would need its own boundary doc.
8. **Dynamic slot / KV budget reshaping is not HPX-side for now.**
   That is a `llama.cpp` execution concern.

This ranking is not a roadmap commitment; it is a guide for which
slice is most defensibly "HPX-native control plane" if work were to
proceed.

---

## 13. What would make this design non-HPX-native?

Adopting any of the following would push the project back into the
"llama-server with HPX futures" failure mode. They are listed so the
design can be checked against them rather than implicitly drifted into.

Anti-patterns:

- **One HPX task per request directly calling `llama_decode`.**
  Loses the single-mutator invariant; produces parallel
  `llama_decode` corruption; collapses the control plane into ad-hoc
  task graphs.
- **Per-token tiny HPX synchronization everywhere** (e.g. a fresh
  `hpx::promise` per token, or a fork/join per emit). HPX overhead
  per token dwarfs the model decode work and gives nothing back.
- **Using HPX only as a thread pool.** If the design replaces
  `hpx::async` with `std::async` and nothing of substance changes,
  HPX is not in the architecture — it is a dependency.
- **Duplicating `llama-server` slot logic without HPX-owned policy.**
  Copying the slot table without naming an `AdmissionPolicy`,
  `PrefillBudgetPolicy`, etc., produces a re-implementation, not a
  control plane.
- **Touching `ggml` / tokenizer / KV internals prematurely.** Those
  are `llama.cpp` ownership. Any "HPX optimization" that crosses
  that line is by definition not HPX-side.
- **Adding benchmarks before defining ownership and dataflow.**
  Numbers from an under-specified architecture argue for arbitrary
  conclusions. Benchmarks come after policy hooks are named.
- **Producing performance claims from correctness gates.** Forbidden
  by `CLAUDE.md`'s correctness invariants; reiterated here because
  the temptation grows once diagnostics produce numeric outputs.
- **Foreign-thread mutation of engine state.** Producers may publish
  POD messages to the inbox; they may not mutate any engine state
  directly. The boundary today is structural — keep it that way.
- **`shared_ptr` / `weak_ptr` engine control blocks on the request
  handle.** Already rejected in M5. Keeps lifetime simple and
  prevents implicit engine resurrection.

---

## 14. Two-day roadmap

Day 1 (this slice):

- Land this design document. (One artifact, doc only.)
- Reconcile the M0–M8 milestone summary with the boundary statements
  here (no rewrite; cross-link only if needed).
- Run W1 *after* the design is in. The W1 diagnostic plan in
  `docs/hpx/serving_overhead_diagnostics_roadmap.md` §7 already
  exists; the driver is staged under
  `local/runs/w1-mixed-short-long-2026-05-26/driver.py`. Pair the
  W1 capture with this doc so it has explicit policy context.

Day 2:

- Optional: structural cleanup pass on the engine task's iter body
  to make the §5 "explicit per-iter phases" more legible as the
  policy hook sites (`PUMP / CANCEL / ADMIT / BUILD / DECODE /
  SAMPLE / PUBLISH / IDLE`). The functions
  `iter_observe_cancellations`, `iter_run_admissions`,
  `iter_build_batch`, `iter_run_decode`, `iter_sample_and_finalize`,
  `iter_fire_release_ack_barrier` already exist; the cleanup is a
  comment / naming pass plus one diagnostics row per phase, not a
  behavior change.
- Sketch the `AdmissionPolicy` interface (header only, no
  implementation) so future slices have a target to land into.
  Out of scope for this two-day window: any policy logic, any
  scheduling change, any benchmark change.

What Day 2 does *not* attempt:

- No prefill/decode scheduling change.
- No HPX-side backpressure gate implementation.
- No multi-engine routing scaffold.
- No new benchmark.
- No commit to performance numbers.

---

## 15. Final recommendation

The current implementation is **HPX-native enough as a first scaffold**.
It owns the right things and it crosses the boundary in the right
direction:

- one long-lived engine task is the sole mutator of `llama_context`,
  `llama_batch`, KV, and per-seq samplers;
- the inbox is an HPX local channel; producers publish POD messages
  and never touch engine internals;
- per-request promises and per-request stream channels are HPX-native
  primitives;
- idle-wait is HPX-native (`wait_inbox_blocking()`), with no
  `std::condition_variable`, no busy poll, no foreign suspension;
- cancellation identity is in place (`cancel_token{rid, epoch}`),
  with stale-token protection;
- diagnostics produce per-iter and per-request evidence without
  changing the boundary.

It is **not yet the target design**. The gaps are policy-shaped, not
boundary-shaped:

- admission, prefill/decode interleave, and active scheduling are
  *implicit in the iter body*, not named policy objects;
- backpressure lives at the cpp-httplib door, not inside HPX;
- diagnostics record what happened, not what was decided;
- there is only one engine instance — multi-engine routing is
  unbuilt;
- the HTTP / SSE adapter is cpp-httplib by design (documented), not
  an HPX-side I/O substrate.

Recommended one structural cleanup before any further feature work:
**name the per-iter phases explicitly** in the engine task (already
mostly factored into `iter_*` helpers — make the phase boundaries
the legible policy-hook sites). Behavior unchanged. This is the
minimum precondition for the §12 ranked improvements to land as
control-plane policy rather than as ad-hoc patches to a long
function.

Everything else in this document is target design and is not
required for the scaffold to be HPX-native.
