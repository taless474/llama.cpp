# HPX Serving-Overhead Diagnostics — Phase 1 Roadmap

Status: design only. No code in this slice.

This document captures the Phase 1 diagnostics design before any
implementation, so the next code slice is bounded, reviewable, and
boundary-safe.

## 1. Goal

Phase 1 *observes* the current HPX serving path. It does not change
behavior. It exists to answer one question, with data:

> When `c > 1`, is observed end-to-end latency on the HPX path
> attributable to poor batch filling, prefill/decode interference,
> service-layer overhead (channel/future/wakeup), cancellation
> cleanup, slow-client / backpressure effects, or something else?

The deliverable is a pair of JSONL streams (engine per-iteration,
server per-request) that, taken together, attribute observed wall
time to specific phases of the existing dataflow.

This is not a benchmark. It is a measurement substrate.

## 2. Non-goals

- No performance optimization in this slice.
- No scheduling policy change.
- No admission policy change.
- No change to `llama_decode`, `ggml`, tokenizer, sampler math, or
  KV book-keeping.
- No change to `token_stream_event`, `submit_handle`,
  `request_result`, `engine_options`, or any other shape that is
  part of the engine/server boundary.
- No change to `trace.{h,cpp}` or to the HPX runtime startup
  (`hpx_runtime.{cpp,h}`).
- No change to Exp14 or Exp15 configs, results, or
  `concurrent_bench.py`.
- No edit to the canonical-hash anchor table in
  `concurrent_bench.py`.
- No "HPX is faster than llama-server" claim. The diagnostics are
  intended to characterize the HPX path on its own terms.
- No new source files for the Phase 1 instrumentation itself. The
  three target files (§5) already exist.

## 3. Current dataflow

```text
                   client
                     │ POST /completion
                     ▼
   ┌──────────────────────────────────────────┐
   │ hpx-server handler (cpp-httplib worker)  │
   │  - validate body, sampling, prompt size  │
   │  - tokenize via common_tokenize          │
   │  ─▶ [SERVER METRIC: submit_us]           │
   │  - eng.submit_request(...)               │
   └──────────────────────────────────────────┘
                     │ submit_request
                     ▼
   ┌──────────────────────────────────────────┐
   │ engine inbox (HPX channel)               │
   └──────────────────────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────────────┐
   │ engine task — iter loop (engine-task-    │
   │   only owner of llama_context):          │
   │                                          │
   │  drain_external_inbox(iter)              │
   │  observe cancellations                   │
   │  iter_run_admissions     ─▶ [admitted_in_iter] │
   │  active sequence table   ─▶ [active_seq_count] │
   │  iter_build_batch        ─▶ [prefill_rows_in_iter] │
   │                          ─▶ [decode_rows_in_iter]  │
   │                          ─▶ [batch_n_tokens]       │
   │  llama_decode(ctx, batch)─▶ [llama_decode_wall_us] │
   │  sampling                                │
   │  publish_token           ─▶ [tokens_emitted_in_iter] │
   │  finalize / cancel       ─▶ [completed_in_iter]      │
   │                          ─▶ [cancelled_in_iter]      │
   │                                          │
   │  iter end                ─▶ [iter_wall_us]           │
   │                          ─▶ [idle_wait_us (opt)]     │
   └──────────────────────────────────────────┘
                     │ stream_event_kind::token / closed
                     ▼
   ┌──────────────────────────────────────────┐
   │ hpx-server SSE provider (cpp-httplib     │
   │   worker, foreign-thread `get(sync)`)    │
   │  - stream_channel get                    │
   │       ─▶ [stream_channel_get_count]      │
   │  - cumulative-detokenization delta       │
   │  - sink.write                            │
   │       ─▶ [sse_write_count, sse_write_bytes] │
   │       ─▶ [first_token_us on first write] │
   │  - terminal `done` SSE                   │
   │       ─▶ [completion_us, final_status,   │
   │           n_decoded]                     │
   │  - M7d cancel/drain finalize on          │
   │    disconnect ─▶ [disconnect_observed,   │
   │                   cancel_issued]         │
   └──────────────────────────────────────────┘
```

The non-streaming branch flows through the same submit/handle path
but exits via `h.result.get()` → `set_content(...)` rather than the
SSE provider; it emits the same `request_metrics` row with
`stream=false`.

## 4. Phase 1 diagnostics schema

### 4.1 Engine per-iteration JSONL

One JSON object per iteration of the engine's inner while loop,
plus one `engine_summary` row at engine-run end. Aggregated in
memory during the run; written once at the end so the per-iteration
I/O cost is zero.

Fields:

| Field | Type | Notes |
|---|---|---|
| `kind` | string | `"iter"` |
| `iter` | int | engine inner-while iteration ordinal (0-based) |
| `batch_n_tokens` | int | rows in the `llama_batch` passed to `llama_decode`; sources existing `engine_metrics.rows_per_batch` |
| `active_seq_count` | int | sources existing `engine_metrics.active_seqs_per_iter` |
| `prefill_rows_in_iter` | int | rows added for sequences still in initial-prefill |
| `decode_rows_in_iter` | int | rows added for sequences past initial-prefill |
| `admitted_in_iter` | int | `admit_one` successes in this iter |
| `completed_in_iter` | int | `finalize_and_fulfill` successes in this iter |
| `cancelled_in_iter` | int | `cancel_and_fulfill` + queued-cancel finalize successes |
| `tokens_emitted_in_iter` | int | `publish_token` calls in this iter |
| `waiting_queue_depth_after_admission` | int | sources existing `engine_metrics.waiting_queue_depth_after_admission_per_iter` |
| `llama_decode_wall_us` | int64 | `steady_clock` around `llama_decode(ctx_, batch)`; if a second `llama_decode` site fires in the same iter (admitted-prefill argmax path), its wall is summed in |
| `iter_wall_us` | int64 | top → bottom of the inner while body |
| `idle_wait_us` | int64 | time spent in the engine's wait-for-arrivals primitive, **only if** a single low-risk wait site exists; otherwise 0 (deferred to Phase 2) |

Example row:

```json
{"kind":"iter","iter":17,"batch_n_tokens":4,"active_seq_count":4,"prefill_rows_in_iter":0,"decode_rows_in_iter":4,"admitted_in_iter":0,"completed_in_iter":0,"cancelled_in_iter":0,"tokens_emitted_in_iter":4,"waiting_queue_depth_after_admission":0,"llama_decode_wall_us":24831,"iter_wall_us":25460,"idle_wait_us":0}
```

End-of-run summary row (one per engine run):

```json
{"kind":"engine_summary","decode_calls":33,"update_iterations":32,"wall_ms":812.40,"admitted_count":6,"completed_count":6,"cancelled_count":0,"streams_closed_completed":4,"streams_closed_cancelled":0,"streams_closed_error":0}
```

### 4.2 Server per-request JSONL

One JSON object per HTTP request handled by `llama-hpx-server`,
including 4xx/5xx error paths, 503 capacity rejections, and SSE
disconnect paths. Emitted exactly once per request via a
`metrics_emitted` flag on the per-request state so the SSE
happy-path and the M7d releaser never both fire.

Fields:

| Field | Type | Notes |
|---|---|---|
| `kind` | string | `"request"` |
| `request_id` | int | engine-assigned `rid`; `-1` if rejected before `submit_request` |
| `stream` | bool | true on the SSE branch |
| `submit_us` | int | always `0` (relative origin) |
| `first_token_us` | int | μs since submit at first `sink.write` (SSE) or at `h.result.get()` return (non-streaming); `-1` if no token was written |
| `completion_us` | int | μs since submit at terminal `done` write or `set_content` return or error response |
| `stream_channel_get_count` | int | per `state->h.stream->get(...)` |
| `sse_write_count` | int | per `sink.write(...)` |
| `sse_write_bytes` | int64 | sum of `payload.size()` written |
| `disconnect_observed` | bool | true if M7d in-provider disconnect or releaser non-clean exit |
| `cancel_issued` | bool | mirror of `state->cancel_issued` at emit time |
| `final_status` | string | `"completed"`, `"cancelled"`, `"failed_reserved"`, `"server_busy"`, `"bad_request"`, `"invalid_argument"`, `"engine_error"`, `"payload_too_large"`, or `"disconnect"` |
| `n_decoded` | int | from `r.n_decoded`; `0` on early-exit paths |

Example row (streaming, completed):

```json
{"kind":"request","request_id":3,"stream":true,"submit_us":0,"first_token_us":612,"completion_us":3811,"stream_channel_get_count":9,"sse_write_count":9,"sse_write_bytes":562,"disconnect_observed":false,"cancel_issued":false,"final_status":"completed","n_decoded":8}
```

Example row (capacity-rejected, non-streaming, 503):

```json
{"kind":"request","request_id":-1,"stream":false,"submit_us":0,"first_token_us":-1,"completion_us":4,"stream_channel_get_count":0,"sse_write_count":0,"sse_write_bytes":0,"disconnect_observed":false,"cancel_issued":false,"final_status":"server_busy","n_decoded":0}
```

### 4.3 Output sink

- `LLAMA_HPX_DIAG_METRICS=1` enables both streams.
- `LLAMA_HPX_DIAG_METRICS_PATH=<file>` is optional; when set, both
  engine and server lines are *appended* to that file with
  line-buffered I/O so concurrent lines stay atomic. JSONL line
  interleaving across processes is acceptable; analysis filters by
  `"kind"`.
- When the path is unset but `LLAMA_HPX_DIAG_METRICS=1`, output goes
  to stderr. The existing server log already uses stderr; diag
  lines are easy to grep by `"kind":` prefix.
- When `LLAMA_HPX_DIAG_METRICS` is unset, **no** lines are emitted.

## 5. File-scope plan

### 5.1 `tools/hpx-continuous-batch-gate/types.h`

Extend the existing `struct engine_metrics` substruct only. No new
field on `engine_result` itself; no public-API change.

Added fields:

```text
std::vector<int32_t> prefill_rows_per_iter;
std::vector<int32_t> decode_rows_per_iter;
std::vector<int32_t> admitted_per_iter;
std::vector<int32_t> completed_per_iter;
std::vector<int32_t> cancelled_per_iter;
std::vector<int32_t> tokens_emitted_per_iter;
std::vector<int64_t> llama_decode_wall_us_per_iter;
std::vector<int64_t> iter_wall_us_per_iter;
std::vector<int64_t> idle_wait_us_per_iter;
```

Existing `rows_per_batch` and `active_seqs_per_iter` are reused
verbatim; no duplication.

### 5.2 `tools/hpx-continuous-batch-gate/engine.cpp`

- Read `LLAMA_HPX_DIAG_METRICS` (and the optional path) once at
  `engine::run` startup, cache as a `const bool` and a
  `const char *`.
- Per-iter integer counters (`prefill_rows_in_iter`,
  `decode_rows_in_iter`, `admitted_in_iter`,
  `completed_in_iter`, `cancelled_in_iter`,
  `tokens_emitted_in_iter`) live as locals reset at iter top.
  Increments are guarded by `if (diag_enabled_)` so that the
  OFF path has zero observable cost.
- `steady_clock::now()` pairs are placed:
  - around the iter (top → bottom of the inner while body),
  - around both `llama_decode(ctx_, batch)` sites
    (`engine.cpp:2085` and `engine.cpp:2450`); both deltas
    accumulate into the same `llama_decode_wall_us_per_iter`
    entry for that iter.
- At iter bottom, one `push_back` per vector. All under the
  `diag_enabled_` guard.
- A new private `dump_metrics_jsonl_()` runs at the tail of
  `engine::run`, after the existing `engine_stop` trace
  (`engine.cpp:2932`), gated on `diag_enabled_`. Uses bare
  `fprintf`/`snprintf` — no `nlohmann::json` dependency added to
  this TU.

No change to: phase ordering, admission semantics, sampling, KV
operations, cancel semantics, the order or arguments of
`llama_decode` calls.

### 5.3 `tools/hpx-server/hpx-server.cpp`

- Read `LLAMA_HPX_DIAG_METRICS` once at server startup; cache as
  a file-scope `static const bool g_diag_enabled` and
  `static const char * const g_diag_path`.
- Per-request `struct request_metrics` POD declared at the top of
  the POST handler. Stack-local on the non-streaming branch;
  moved into `stream_state` on the SSE branch (mirrors the
  existing pattern for `submit_handle` and `capacity_lease`).
  Touched only from the single cpp-httplib worker that owns the
  response — no new synchronization primitive.
- Recording sites and the JSONL emit at every handler exit are
  guarded by `if (g_diag_enabled)`.
- A new `metrics_emitted` flag on `stream_state` ensures the SSE
  closed branch and the M7d resource releaser never both emit
  for the same request.
- Bare `fprintf`/`snprintf` for the diag emit; the SSE `data:`
  body continues to use `nlohmann::json` as today.

No change to: handler arg parsing, capacity-lease accounting,
M7d cancel/drain routine, cumulative-delta detokenization, the
SSE record shape, the non-streaming response shape, or error
responses.

### 5.4 Out of scope (Phase 1)

- `engine.h` public surface.
- `submit_handle`, `request_result`, `submit_request`,
  `token_stream_event`.
- `trace.{h,cpp}`, `hpx_runtime.{cpp,h}`.
- `common/`, sampler, tokenizer / model loader, ggml, KV APIs.
- Every `engine_*_smoke.cpp`.
- Exp14, Exp15, `concurrent_bench.py`, `CANONICAL_HPX_HASH` table.
- W1/W2/W3 workload driver scripts (see §8 — separate follow-up).

## 6. Default-OFF behavior

When `LLAMA_HPX_DIAG_METRICS` is unset:

- `getenv` is called once at startup; result cached as `false`.
- Every `steady_clock::now()` call sits inside
  `if (diag_enabled_)` — emits a single not-taken branch, no
  syscall.
- Every per-iter counter increment sits inside the same guard —
  no integer adds on the hot path.
- Every `push_back` sits inside the same guard. Vectors stay
  default-constructed (zero heap allocation).
- No JSON formatting, no `snprintf`, no `fopen`, no `fwrite`,
  no `fflush`.
- The new `request_metrics` POD on the handler stack is ~80 B
  whether the guard is on or off; its recording sites are all
  guarded. Effectively free.
- The engine-end JSONL dump function returns immediately when
  `diag_enabled_` is false.

The acceptance criterion is that, with the switch unset, the
binary's observable behavior — including token-ID sequences,
canonical hashes, smoke PASS lines, and per-request timing
character — is byte-identical to today.

## 7. Phase 1 acceptance gates

### 7.1 Env unset (default)

- All targeted hpx-server smokes still PASS:
  - `llama-hpx-server-smoke`,
  - `llama-hpx-server-sampling-smoke`,
  - `llama-hpx-server-invalid-sampling-smoke`,
  - `llama-hpx-server-oversize-smoke`,
  - `llama-hpx-server-backpressure-smoke`,
  - `llama-hpx-server-stream-smoke`,
  - `llama-hpx-server-stream-disconnect-smoke`,
  - `llama-hpx-server-backpressure-stream-disconnect-smoke`,
  - `llama-hpx-server-engine-pool-smoke`.
- All targeted engine smokes still PASS:
  - `llama-hpx-engine-smoke`,
  - `llama-hpx-engine-stream-smoke`,
  - `llama-hpx-engine-idle-smoke`,
  - `llama-hpx-engine-keepalive-smoke`,
  - `llama-hpx-engine-keepalive-multi-submit-smoke`,
  - `llama-hpx-engine-active-cancel-smoke`,
  - `llama-hpx-engine-queued-cancel-smoke`,
  - `llama-hpx-engine-queued-cancel-engine-pool-smoke`,
  - `llama-hpx-engine-handle-cancel-smoke`,
  - `llama-hpx-engine-stale-token-smoke`,
  - `llama-hpx-engine-cancel-freed-completion-admit-smoke`,
  - `llama-hpx-engine-shutdown-queued-unadmittable-smoke`,
  - `llama-hpx-engine-sampling-config-carry-smoke`,
  - `llama-hpx-engine-sampling-stochastic-smoke`,
  - `llama-hpx-engine-sampling-independence-smoke`.
- Canonical anchors hold:
  - TinyLlama greedy `p0_b8`: `0x0619d4d1900c2365`,
  - TinyLlama greedy `p0_b16`: `0x833045f1e2ebf49f`,
  - Stochastic seed=42: `0xa8e14acb4094aa3f`,
  - Every other anchor observed in the prior streaming-slice run.
- No JSONL line emitted anywhere (grep
  `"kind":"iter"|"kind":"engine_summary"|"kind":"request"`
  across all per-smoke stderr → zero hits).

### 7.2 Env set (`LLAMA_HPX_DIAG_METRICS=1`)

- Same smokes still PASS with identical PASS lines.
- Canonical anchors hold identically (instrumentation is observer
  only).
- For every smoke that decodes at least one row:
  - at least one `"kind":"iter"` row in the JSONL,
  - exactly one `"kind":"engine_summary"` row,
  - vector lengths consistent: each `*_per_iter` vector length
    matches `update_iterations` reported in the summary.
- For every server smoke that issues HTTP requests:
  - at least one `"kind":"request"` row per HTTP request, including
    capacity rejections and disconnects,
  - the SSE closed branch and the M7d releaser do not double-emit
    (one row per request, enforced by `metrics_emitted`).
- JSONL is parseable line-by-line by a short script.

## 8. Diagnostic workloads (planned follow-up)

The W1/W2/W3 workload drivers are **not** part of the Phase 1
implementation slice. They are planned as a separate follow-up
turn after the slice lands. Each driver is a small Python script
that posts to a running `llama-hpx-server` with diagnostics
enabled and a per-workload output path.

### W1 — Mixed short + long

- Eight requests, alternating `decode_budget=8` and
  `decode_budget=128`, posted at 100 ms stagger.
- Two passes: `--max-concurrent 2`, then `--max-concurrent 4`.
- Reveals: prefill/decode interference (B). Iters with
  `prefill_rows_in_iter > 0` aligned with stretches in long-request
  `first_token_us`; if `batch_n_tokens` lags `active_seq_count`
  consistently, that points instead at C (poor batch filling).

### W2 — Cancellation-heavy

- Ten streaming requests with `decode_budget=64`, posted at
  random 50–150 ms stagger; clients close after observing four SSE
  token events.
- Two passes: `c=2`, then `c=4`.
- Reveals: cancellation cleanup cost. Engine-side
  `cancelled_in_iter` spikes; `iter_wall_us − llama_decode_wall_us`
  at those iters quantifies cancel-finalize overhead.
  Cross-request: do *uncancelled* admissions after a wave of
  cancels see stretched `first_token_us`?

### W3 — Slow streaming / backpressure

- One streaming request with `decode_budget=64`. Driver sleeps
  100 ms between SSE reads.
- Two passes: `c=1` baseline; `c=2` with a second healthy client
  to test isolation.
- Reveals: backpressure isolation (E). If `idle_wait_us` correlates
  with the slow-client window, the engine is waiting on the
  consumer — a backpressure leak. In the `c=2` pass, the healthy
  client's `first_token_us` and per-token `iter_wall_us` should be
  independent of the slow client.

## 9. Improvement ordering

Hartmut-skeptical ranking, evidence-driven:

1. **Prefill/decode scheduling split.** Directly evaluable from
   Phase 1 data via `prefill_rows_in_iter` vs `iter_wall_us` and
   `first_token_us` stretches in W1.
2. **Adaptive admission / batching policy.** Directly evaluable
   from `active_seq_count`, `batch_n_tokens`, and
   `waiting_queue_depth_after_admission` in W1.
3. **Slow-client / backpressure isolation.** Directly evaluable
   from `idle_wait_us` (if captured), `sse_write_count` vs
   `tokens_emitted_in_iter`, and cross-stream `first_token_us` in
   W3.
4. **Low-overhead streaming delivery.** Phase 2 only, and only if
   Phase 1 data shows a residual non-decode gap that B/C/E do not
   explain. Phase 2 will require a separate plan for per-token
   channel timing that does *not* alter `token_stream_event`.
5. **Multi-engine / engine-actor routing.** Premature until single
   engine has a documented saturation story (B/C tried; remaining
   gap measured).
6. **Priority / deadline scheduling.** Defer until admission
   policy work establishes request classes worth scheduling.
7. **NUMA / distributed / multi-node HPX orchestration.** Separate
   architecture track; not a near-term local-server improvement.
8. **Dynamic slot / KV budget reshaping.** Not an HPX-side
   improvement; crosses the codebase's deliberate boundary with
   llama.cpp internals.

## 10. Decision rule

After Phase 1 data is collected from W1/W2/W3:

- **If** iters with `prefill_rows_in_iter > 0` consistently show
  `iter_wall_us` materially above neighboring decode-only iters,
  **and** other in-flight requests' `first_token_us` stretches
  align with those iter indices: **proceed with prefill/decode
  scheduling design (direction B).**
- **If** `active_seq_count < c` on a non-trivial fraction of
  iters while requests are queued, **or** `batch_n_tokens` lags
  `active_seq_count` materially: **proceed with adaptive
  admission / batching design (direction C).** This may combine
  with B.
- **If** under W3 the engine's `idle_wait_us` correlates with the
  slow-client read pattern, **or** the healthy concurrent client's
  per-token timing degrades in the slow-client pass: **proceed
  with backpressure isolation design (direction E).**
- **If** B / C / E do not account for the observed non-decode
  gap (`iter_wall_us − llama_decode_wall_us` remains a large
  fraction of `iter_wall_us` after the above signals are flat):
  **design Phase 2 channel-timing diagnostics**, with its own
  plan that addresses how to capture per-token engine→handler
  latency without altering `token_stream_event`.
- **If none of the above hold,** the honest answer is that Phase 1
  could not attribute the observed `c > 1` behavior to a specific
  HPX-side cause on this workload set. Record that conclusion as
  the result. Do not invent an improvement to justify the work.

## 11. Risks

- **Instrumentation perturbing timing.** Even default-OFF, every
  guarded branch is a conditional load+jump. With diagnostics ON,
  the chrono pairs around `llama_decode` and the iter add real
  syscalls. Expected overhead per iter on Apple Silicon is on the
  order of hundreds of nanoseconds against multi-millisecond
  decode times. The slice will spot-check ON vs OFF on
  `llama-hpx-server-stream-smoke` to confirm no measurable
  difference.
- **JSONL I/O during a run.** Per-iter rows are aggregated in
  memory; engine I/O happens only at run end. Per-request server
  rows are written from the handler thread at exit, not on the
  hot per-token path. Even so, `fprintf` to stderr can interleave
  with other stderr writes from the server log. The implementation
  uses `O_APPEND` + line buffering and a unique `"kind":` prefix
  so analysis filters cleanly.
- **Two `llama_decode` sites.** `engine.cpp:2085` and `:2450` both
  bump `result_.decode_calls`. The Phase 1 wall-time vector
  aggregates both into the per-iter entry; the design note in
  `engine.cpp` will state this so the analysis layer doesn't
  double-attribute.
- **`idle_wait_us` optionality.** If the engine's wait-for-arrival
  path turns out to be spread across multiple sites, the slice
  emits 0 and defers a proper measurement to Phase 2 rather than
  refactoring the inbox path inside a diagnostics slice.
- **Short-run interpretation.** Smokes decode 8–64 tokens at
  TinyLlama scale. Any conclusions from smoke-shaped runs are
  qualitative; quantitative claims need W1/W2/W3 at meaningful
  budgets and durations, and even there the wording stays
  descriptive (no "faster"/"slower"/"speedup" language).
- **Early API/boundary changes.** Phase 1 explicitly does not
  touch `token_stream_event`, `submit_handle`, or
  `request_result` exactly because we just preserved those
  through the streaming-detokenization slice. A boundary change
  driven by *suspicion* rather than data would be premature.
- **Measuring more than is useful.** Phase 1 stops short of
  per-token timing, KV-occupancy counters, and per-phase wall
  splits. Each of those is a potential Phase 2 addition, gated
  on Phase 1 results pointing at a specific cause.
- **Hash-anchor preservation.** Counters are observers; they
  cannot perturb token-ID sequence or `request_result.hash`.
  Verified by smoke runs in both modes producing identical
  canonical anchors.

## 12. Recommended next code slice

Implement Phase 1 diagnostics exactly as described above:

- three files edited (`types.h`, `engine.cpp`, `hpx-server.cpp`);
- no edit to `token_stream_event`, the engine/server API shapes,
  scheduling, sampling, detokenization, or batch composition;
- default-OFF behind `LLAMA_HPX_DIAG_METRICS=1`;
- JSONL output per §4 schemas;
- acceptance gates per §7.

Workload drivers (W1/W2/W3) and any Phase 2 channel-timing
diagnostics are out of scope for the next slice and will be planned
as separate, evidence-conditioned follow-up turns.

## 13. Phase 1 implementation result

Phase 1 diagnostics have been implemented and validated. This
section is the implementation checkpoint — no scheduling change,
no workload driver, no Exp14 / Exp15 / `concurrent_bench.py` run.

### 13.1 Files changed

- `tools/hpx-continuous-batch-gate/types.h` — appended
  `t_us_from_engine_start_per_iter` vector and the transient
  `phase1_iter_acc cur_iter_diag` accumulator inside `engine_metrics`.
  No new include in this header.
- `tools/hpx-continuous-batch-gate/engine.cpp` — env-switch caches
  (`diag_enabled` / `diag_path`), `dump_metrics_jsonl`, per-engine
  per-iter accumulation via `result_.metrics.cur_iter_diag.*`,
  iter-top reset, iter-bottom back-patch sharing the existing
  `t_start_` baseline. No file-scope mutable diagnostic state.
- `tools/hpx-server/hpx-server.cpp` — `server_diag_enabled` /
  `server_diag_path` env-switch caches, `server_now_us` absolute
  steady-clock helper, `struct request_metrics` POD,
  `dump_request_metrics_jsonl`, instrumented non-streaming and
  streaming paths, `metrics_emitted` dedupe flag inside
  `stream_state`.

No edit to `engine.h`, `token_stream_event`, `submit_handle`,
`request_result`, `submit_request`, `trace.{h,cpp}`,
`hpx_runtime.{cpp,h}`, `common/`, sampler, tokenizer / model loader,
ggml, KV APIs, Exp14, Exp15, `concurrent_bench.py`, hash-anchor
tables, or CMake. No new source files.

### 13.2 Validation

Captures: `local/runs/n6-phase1-diag/` (gitignored under `local/`).

- **Env unset (Pass 1): 24 / 24 PASS.**
  Set: 15 engine smokes + 9 hpx-server smokes from the streaming-
  detokenization slice. Zero `"kind"` JSONL emitted in any stdout
  or stderr. Disconnect / cancel smokes (06, 21, 24) PASS despite
  the new `request_result fr` capture sites in hpx-server.cpp's
  finalize paths — OFF-mode behavior is byte-equivalent to the
  prior `(void)`-discard.
- **`LLAMA_HPX_DIAG_METRICS=1` (Pass 2): 24 / 24 PASS.**
  Same PASS / `ALL_COMPLETED` markers as Pass 1.
- **538 valid JSONL rows total** across 26 JSONL files (24 smoke
  files + 2 real-server demo files). Every row parses through
  `json.loads` line-by-line.
- **Zero JSONL output with env unset** (Pass 1) — confirmed by
  `grep -c '"kind"' …` returning 0 across every Pass-1 stdout
  and stderr.
- **Canonical anchors held in both passes.** Greedy
  `0x0619d4d1900c2365` (TinyLlama p0_b8, "Hello, my name is",
  greedy) observed in both passes across `engine-smoke`,
  `server-smoke`, `keepalive-multi-submit`,
  `cancel-freed-completion-admit`, `sampling-independence`, and the
  real-server non-streaming response body. Budget-16 anchor
  `0x833045f1e2ebf49f` observed in `engine_stream_smoke` p0_b16 in
  both passes. Stochastic seed=42 anchor `0xa8e14acb4094aa3f`
  observed in sampling-independence / sampling-stochastic smokes
  in both passes.
- **`git diff --check`: clean (exit 0)** on all three changed files.
- **`pgrep -lf 'llama-server|llama-hpx-server'`: no servers
  running** after both passes.

### 13.3 JSONL coverage

- **Engine `iter` rows present.** 506 across Pass 2 + real-server
  demo. Schema matches §4: `iter`, `t_us_from_engine_start`,
  `batch_n_tokens`, `active_seq_count`, `prefill_rows_in_iter`,
  `decode_rows_in_iter`, `admitted_in_iter`, `completed_in_iter`,
  `cancelled_in_iter`, `tokens_emitted_in_iter`,
  `waiting_queue_depth_after_admission`, `llama_decode_wall_us`,
  `iter_wall_us`, `idle_wait_us`.
- **`engine_summary` rows present.** 29 across Pass 2 + real-server
  demo. One per `engine::run()` clean shutdown. Carries
  `preload_prefill_llama_decode_included:0` as an explicit
  exclusion marker (see §13.5).
- **Real-server `server_request` rows present.** 3 rows across
  the two real-server demos: one non-streaming completion, one
  streaming completion, one streaming client-disconnect. All
  populated through the diag-instrumented `hpx-server.cpp`
  handler. See §13.4 for why no `server_request` rows appear in
  the 9 in-process server smokes' Pass-2 JSONL files.
- **`metrics_emitted` dedupe verified on streaming disconnect.**
  A `curl --max-time 0.15` against the real server triggered the
  in-provider `sink.write`-returned-false finalize AND, after the
  response ended, the M7d resource releaser. Only the releaser
  emitted the JSONL row (per the centralized-dedupe design). The
  resulting row carried `disconnect_observed:true`,
  `cancel_issued:true`, `final_status:"cancelled"`, `n_decoded:4`,
  `sse_write_count:3`. **Exactly one row per submitted request,
  no duplicates.**

### 13.4 Explicit caveat — server smokes vs. real server

The nine in-process server smokes (`hpx_server_smoke.cpp`,
`hpx_server_stream_smoke.cpp`, `hpx_server_stream_disconnect_smoke.cpp`,
`hpx_server_sampling_smoke.cpp`, `hpx_server_invalid_sampling_smoke.cpp`,
`hpx_server_oversize_smoke.cpp`, `hpx_server_backpressure_smoke.cpp`,
`hpx_server_backpressure_stream_disconnect_smoke.cpp`,
`hpx_server_engine_pool_smoke.cpp`) each inline their own
`srv.Post("/completion", …)` cpp-httplib handler body. They do not
link or share `hpx-server.cpp`'s real handler. Phase 1 diag
instrumentation lives only in the real handler, so Pass-2 JSONL for
those smokes contains engine rows but **no `server_request` rows**.

Real `server_request` rows were verified by driving the real
`llama-hpx-server` binary directly with `curl`, captured under
`local/runs/n6-phase1-diag/pass2-real-server/`. The smoke handler
files are outside this slice's allowed-edit set; touching them is
a separate decision and should be deferred until either a Phase 2
diag-coverage goal requires it or those handlers can be
de-duplicated against `hpx-server.cpp`.

### 13.5 Deferred items

- **`idle_wait_us` is schema-present but emitted as 0 in Phase 1.**
  Every per-iter row carries the field so Phase 2 instrumentation
  can populate it without a row-shape break. The most useful
  variant is engine-inbox wake-up latency (push-to-iter-top
  delta), which requires touching `submit_request` — out of scope
  for Phase 1.
- **Preloaded-prefill `llama_decode` site is excluded by design
  (Option B from §13.1's engine.cpp comment block).** Every
  `engine_summary` row carries the explicit key
  `"preload_prefill_llama_decode_included":0` so analysis code can
  detect the omission unambiguously rather than infer it from row
  counts. The exclusion does not affect the hpx-server path
  (which constructs engines with `budgets_={}`) but is visible to
  gate/smoke runs with preloaded actives via the row offset logic
  already in `dump_metrics_jsonl`.
- **Cross-axis alignment between engine rows and server rows is
  not yet implemented.** Engine rows use an engine-relative axis
  (`t_us_from_engine_start_per_iter`, delta from each engine's
  `t_start_`); server rows use absolute `steady_clock` µs. A
  single `engine_t_start_us_absolute` key in the `engine_summary`
  row would let analysis code translate. Deferred — would touch
  engine.cpp again.
- **W1 / W2 / W3 workload drivers not run yet.** Phase 1 is the
  observation surface; W1+ is the experiment surface.
- **No scheduling policy change.** Phase 1 is observation-only
  per §10 (`Anti-goals — scheduling policy is unchanged`). No
  prefill/decode split, adaptive admission, backpressure tier,
  priority, or NUMA work has been started.

### 13.6 Recommended next step

**Design and run W1 (mixed short + long workload) only, using the
new JSONL diagnostics, before considering any HPX scheduling
change.** Specifically:

1. Use the existing real `llama-hpx-server` binary (no new server
   code).
2. Drive it with a workload that mixes short-budget and long-budget
   requests at c > 1, capturing both the engine JSONL (one path
   per `engine::run()`) and the server JSONL (one row per
   submitted request).
3. Use the per-iter axis (`t_us_from_engine_start`,
   `prefill_rows_in_iter`, `decode_rows_in_iter`,
   `tokens_emitted_in_iter`) to attribute observed c > 1 latency
   to: batch filling, prefill / decode interference, or service-
   layer overhead.
4. Use server rows (`submit_us` → `first_token_us` →
   `completion_us`, `stream_channel_get_count`,
   `disconnect_observed`, `cancel_issued`) to bound the
   server-side contribution and detect slow-client / backpressure
   effects.
5. Only after W1 evidence localizes a specific bottleneck (e.g.
   "decode iters are starving because prefill rows are dominating
   the shared batch at c > 1") should a scheduling-policy slice be
   proposed — and that slice should land its own W1 re-run on the
   same diagnostics axis to demonstrate the change.

Do not implement any scheduling change, Phase 2 diagnostic, or
cross-axis alignment until W1 evidence is on hand.
