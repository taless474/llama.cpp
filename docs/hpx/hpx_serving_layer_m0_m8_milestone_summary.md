# HPX Serving Layer for llama.cpp: M0–M8 Milestone Summary

## Purpose

The HPX Serving Layer for llama.cpp is an HPX-native serving control plane layered on top of `llama.cpp` model execution. The intent of the work is not to replace kernels, samplers, or KV memory; the intent is to study what HPX can own at the request, lifecycle, and run-coordination layer while leaving model execution untouched.

In the current state, HPX owns:

- request lifecycle (admission, decode loop, completion, cancellation)
- futures/promises for engine-fulfilled results
- streaming handoff from the engine task to per-request consumers
- keep-alive and shutdown of long-lived engines
- backpressure (capacity door, lease accounting)
- the server adapter control path
- the matched benchmark harness (external Python driver around two real servers)

`llama.cpp` owns:

- `llama_model` / `llama_context` execution semantics
- `llama_decode`
- KV memory implementation
- sampler math
- tokenizer/vocab behavior
- the `ggml` graph
- CPU/Metal/backend kernels

The hard rule across all milestones is that only the single engine task touches mutable `llama.cpp` execution state. The benchmark harness uses the HTTP/SSE surface of each server as a black box and is not part of the HPX runtime.

## Current state

As of M8, this branch contains:

- a reusable HPX continuous-batch engine library
- an `hpx-server` HTTP/SSE adapter exposing `POST /completion`
- per-request cancellation, sampling, streaming, and backpressure
- a matched-condition benchmark harness comparing `hpx-server` and `llama-server`
- validation anchors for greedy, stochastic, non-streaming, and streaming paths

## Boundary

| Layer | Owns | Touches `llama.cpp` mutable state? |
|---|---|---|
| HPX engine task | `llama_context`, `llama_batch`, `llama_decode`, `llama_get_logits_ith`, `llama_memory_seq_*`, `llama_sampler_*`, KV mutation, promise fulfillment for engine-owned seq state, stream close | yes (sole owner) |
| HPX control plane | request lifecycle, admission/queue state, cancellation tokens, capacity leases, keep-alive, shutdown, traces/counters | no |
| HPX result consumers | per-request `request_result` snapshots, stream receiver handles | no |
| `hpx-server` HTTP handlers | request parsing, response shaping, SSE writer, `const llama_vocab *` tokenize/detokenize | no (only `const` vocab) |
| `cpp-httplib` | TCP accept, HTTP wire format, content-provider callbacks | no (documented non-HPX adapter boundary) |
| Python benchmark harness | server launch, readiness, request issuing, response/SSE parsing, artifact capture, gate evaluation | no (external client) |

Only the engine task touches mutable `llama.cpp` execution state. Every other HPX component receives snapshots or operates on the request-control plane.

## How to read this document

M0–M3 describe the accepted foundation that existed before the formal M4–M8 milestone naming became consistent. M4–M8 describe the more recent, explicitly tracked slices. This document is a stable summary, not a full provenance log; detailed commands, captures, and run artifacts remain in the referenced result directories and provenance files.

## Milestone table M0–M8

The historical labels for the work that pre-dates M4 were not formally numbered as M0–M3 in the original notes. The labels below align the existing consolidated design documents with the M0–M8 framing used from M4 onward. Where the historical name is not exact, the caveat column says so.

| Milestone | Name | What changed | Why it mattered | Key validation / evidence | Remaining caveat |
|---|---|---|---|---|---|
| M0 | Direction & pure multi-seq gate | After the FIFO context-pool path closed out, the project picked an HPX-around-`llama_decode` direction. A pure-`llama.cpp` reference gate (`tools/multiseq-batch-gate/`) was built first to prove the multi-seq execution primitive end-to-end without any HPX code. | Established the `HPX owns orchestration / llama.cpp owns execution` boundary and a non-HPX correctness baseline before any HPX scheduling was added. | `docs/hpx/continuous_batching_foundation.md`; `tools/multiseq-batch-gate/results.md`. | Accepted pre-M4 state. Exact historical milestone numbering was informal; the name reflects the consolidated foundation doc. |
| M1 | HPX continuous-batch prototype | `tools/hpx-continuous-batch-gate/` introduced. One engine HPX task, one `llama_context`, one shared `llama_batch`, one `hpx::promise<request_result>` per active request, repeat-deterministic schedule. | First proof that HPX can drive a continuous-batching schedule around opaque `llama_decode` calls without breaking the execution boundary. | Repeat-determinism and KV-cleanliness gates in the gate sources; canonical greedy hash on `Hello, my name is` / budget=8 stabilised at `0x0619d4d1900c2365`. | Accepted pre-M4 state. Originally documented as the "continuous-batching prototype" rather than "M1". |
| M2 | Cancellation lifecycle & live admission | Cooperative iteration-boundary cancellation; admission of waiting requests into KV-empty slots at iteration boundaries; explicit residual-KV checks. | Established the leave path (cancel) and the enter path (admission) as engine-task-owned, with futures/promises receiving snapshots only. | `docs/hpx/continuous_batching_lifecycle_design.md`; cancellation and admission smokes under `tools/hpx-continuous-batch-gate/`. | Accepted pre-M4 state. The cancellation and live-admission notes were consolidated into one document after the fact. |
| M3 | Streaming inside the gate (Streaming Slices 1–7) | HPX-native per-request token streaming through `hpx::lcos::local::channel<token_stream_event>`; engine task is the sole producer, gate consumer is the sole receiver; close-reason carried explicitly. | Proved generated token ids can be published through an HPX-native channel and reconstructed by the consumer, while `llama.cpp` execution state stays inside the engine task. | `docs/hpx/continuous_batching_streaming_design.md`; `engine_stream_smoke.cpp`. | Accepted pre-M4 state. Consumer was the gate's `main()`, not yet HTTP/SSE. |
| M4 | Engine surface cleanup (library-vs-gate split) | `engine_options` split into `opts.lib` / `opts.preload` / `opts.gate_test`. `submit_request`-only users no longer need a fake `waiting_queue` or fake `prompt_tokens`. The `initial_idle` slot-reuse bug that later affected keep-alive multi-submit was fixed in this lineage before M6/M7 serving use. HPX-native cleanup preserved. | Made the engine usable as a serving library, not just as a gate harness. Canonical gate output stayed byte-identical after the split. | Canonical gate captures unchanged; submit-only smokes added under `tools/hpx-continuous-batch-gate/`. | Library-vs-gate split is an engine-surface change only; no HTTP yet. |
| M5 | Cancellation identity hardening | `cancel_token {request_id, epoch}` added. `submit_handle::token`. `engine::cancel_request(cancel_token const&)`. Stale-token protection for reused `request_id`. `submit_handle::cancel(engine &) const noexcept` as explicit-engine forwarder. No zero-arg handle cancel. No raw `engine *` observer. No `shared_ptr`/`weak_ptr` control block. | Closed a window where a reused `request_id` could let a stale cancel hit a new request. Kept cancellation engine-task-owned and HPX-native. | Stale-token and handle-cancel smokes under `tools/hpx-continuous-batch-gate/`. | Cancel is still cooperative iteration-boundary only; no preemption of an in-flight decode. |
| M6 | Per-request sampling | `sampling_config` plumbed end-to-end. Stochastic mode uses per-seq `llama_sampler` chains. `top_k` / `top_p` / `temperature` wired through `build_sampler_chain`. The default greedy path remains local `argmax` and byte-identical. `keepalive_multi_submit` bug fixed (initial-idle slots return to `free_idle_` on completion). | Allowed sampling diversity per request without breaking the engine-task-owned execution boundary or the greedy anchor. | Sampler isolation, stochastic, and config-carry smokes. Anchors: greedy `p0_b8` hash `0x0619d4d1900c2365`; default stochastic `seed=42` hash `0xa8e14acb4094aa3f`. | Sampler chain shapes are limited to the wired subset; full upstream sampler surface is not exposed. |
| M7 | `hpx-server` HTTP adapter | `tools/hpx-server/` introduced. `POST /completion` non-streaming (M7a). Optional nested sampling JSON with `HTTP 422` on invalid sampling (M7b). SSE streaming `"stream": true` with `event: token` / `event: done` (M7c). SSE client-disconnect maps to `submit_handle::cancel(engine &)` (M7d). `--max-concurrent` backpressure with `HTTP 503` + `Retry-After: 1` + `error.code == "server_busy"` (M7e). Stream-disconnect releases backpressure capacity (M7e2). | First HTTP surface for the serving layer. `cpp-httplib` is the explicit non-HPX network adapter boundary; HTTP handlers use `const llama_vocab *` only and never hold `llama_context *`. | hpx-server smokes under `tools/hpx-server/`: completion, sampling, oversize, stream, stream-disconnect, backpressure, backpressure-stream-disconnect. Greedy server path preserves `0x0619d4d1900c2365`; stochastic `seed=42` preserves `0xa8e14acb4094aa3f`. | Only `POST /completion`; no OpenAI-compatible endpoint; no auth/TLS/production shutdown; no multi-model; no HTTP-side queueing beyond strict door-cap; handler logic is duplicated across smokes. |
| M8 | Matched `hpx-server` vs `llama-server` benchmark harness | `hpx-bench/experiments/12_hpx_vs_llama_server_pair/` introduced. M8a precursor adds `--ctx-size` to `hpx-server` so `n_ctx` can be pinned symmetrically. M8b: minimal pair-run. M8c: repeat stability and descriptive `match_conditions.json` / `known_mismatches.txt` / forbidden-words guard. M8d: non-streaming 2×2 matrix (`workload_matrix` + `iterations_per_workload`). M8d-fix: `decode_budget` / `n_predict` is an upper bound; EOG-stop is accepted; cross-server `n_decoded` is recorded, not gated. M8e: canonical streaming TTFT (SSE parser, `stream_events.jsonl`, streaming row/anchor gates). M8e2: streaming 2×2 matrix + streaming detokenization and ok-field caveats. | First matched-condition measurement infrastructure for the serving layer. By construction it captures raw values and runs descriptive gates; it does not produce performance claims. | Passing runs: `20260521-231710-pair` (non-streaming matrix, harness `m8d.0`); `20260521-233521-pair` (canonical streaming, 10 repeats, harness `m8e.0`); `20260521-234615-pair` (streaming matrix, harness `m8e.0`). All `gates=PASS`, zero `GATE FAIL`, zero forbidden-word hits. | No aggregation, no percentiles, no performance claim. Cross-server `text` and `n_decoded` equality is recorded, not gated. `cpp-httplib` remains the non-HPX adapter boundary. |

## Detailed milestone notes

### M0 — Direction & pure multi-seq gate

After the FIFO context-pool serving experiment closed out (see `docs/hpx/serving_bench_fifo_closeout.md`), the project pivoted to HPX-around-`llama_decode` orchestration. The first concrete artifact in that direction was a pure-`llama.cpp` reference gate (`tools/multiseq-batch-gate/`) that demonstrated multiple active `seq_id` values, one shared `llama_batch`, per-seq logits readback, per-seq KV clear, and deterministic per-seq hashes. No HPX, no server, no streaming. This is the baseline that every later HPX milestone is compared against.

### M1 — HPX continuous-batch prototype

`tools/hpx-continuous-batch-gate/` introduced one engine HPX task that owns `llama_context` / `llama_batch` / `llama_decode`. Per-request lifecycle is driven by HPX through `hpx::promise<request_result>`. The gate's `main()` validates only snapshots. No parallel `llama_decode` on the same context. The canonical anchor `Hello, my name is` / greedy / budget=8 stabilised at `0x0619d4d1900c2365` and has remained stable across every subsequent milestone.

### M2 — Cancellation lifecycle & live admission

Cancellation is cooperative and iteration-boundary only; the engine never interrupts an in-flight `llama_decode`. KV is cleared by the engine task before the promise is fulfilled with `status=cancelled`. Live admission binds a waiting request to a KV-empty slot at an iteration boundary; `llama_memory_seq_pos_min/max(mem, seq_id) == -1` is required before reuse. Together the two surfaces establish the leave and enter paths inside HPX.

### M3 — Streaming inside the gate (Streaming Slices 1–7)

HPX-native per-request token streaming through `hpx::lcos::local::channel<token_stream_event>`. The engine task is the sole producer; the gate consumer is the sole receiver. Close reasons are explicit (`completed`, `cancelled`, `error`). The streaming substrate exists entirely inside the gate process; HTTP/SSE was deliberately out of scope at this point.

### M4 — Engine surface cleanup (library-vs-gate split)

`engine_options` was split into `opts.lib` / `opts.preload` / `opts.gate_test` so callers that only use `submit_request` no longer need to fabricate gate-test fields. `admitted_futures_mtx_` moved from `std::mutex` to `hpx::spinlock`. The canonical gate captures were byte-identical before and after the split. The `initial_idle` slot-reuse lineage that caused keep-alive multi-submit to hang was identified here and fixed in M6.

### M5 — Cancellation identity hardening

The cancel surface was tightened around a `cancel_token {request_id, epoch}`. `submit_handle::token` carries it. `submit_handle::cancel(engine &) const noexcept` is an explicit-engine forwarder; there is no zero-arg handle cancel and no raw `engine *` observer. Stale tokens (epoch mismatch when a `request_id` is reused) are silently dropped at the engine. Cancellation remains cooperative and iteration-boundary only.

### M6 — Per-request sampling

`sampling_config` is plumbed end-to-end. Stochastic mode builds a per-seq `llama_sampler` chain (`top_k` → `top_p` → `temperature`). The default greedy path stays as local `argmax` and remains byte-identical to M1's anchor. Two anchors were pinned: greedy `p0_b8` at `0x0619d4d1900c2365`, default stochastic `seed=42` at `0xa8e14acb4094aa3f`. The `keepalive_multi_submit` hang was fixed by routing `initial_idle`-sourced slots back to `free_idle_` on completion.

### M7 — `hpx-server` HTTP adapter

`tools/hpx-server/` exposes a `POST /completion` endpoint backed by one `hpx-continuous-batch-gate` engine.

- M7a: non-streaming greedy completion. `cpp-httplib` is documented as a non-HPX network adapter boundary; HTTP handlers use `const llama_vocab *` for tokenize/detokenize and never hold `llama_context *`.
- M7b: optional nested sampling. Invalid sampling returns `HTTP 422` with `error.code == "invalid_sampling"`. `validate_sampling_config` is shared between server and engine.
- M7c: SSE streaming. The wire format uses explicit `event: token` / `event: done` records. The terminal `done` record carries `n_decoded`, `hash`, and `status`.
- M7d: SSE client-disconnect maps to `submit_handle::cancel(engine &)` via the in-provider sink failure path and `ContentProviderResourceReleaser`. Engine cancellation stays fire-and-forget.
- M7e: `--max-concurrent` backpressure. Capacity door rejects before tokenize/submit with `HTTP 503` + `Retry-After: 1` + `error.code == "server_busy"`. `capacity_lease` is move-only and releases exactly once.
- M7e2: stream-disconnect releases backpressure capacity (composition of M7d and M7e).

Canonical hashes on the HTTP surface match the engine anchors: greedy `0x0619d4d1900c2365`, stochastic `seed=42` `0xa8e14acb4094aa3f`.

### M8 — Matched `hpx-server` vs `llama-server` benchmark harness

The harness lives at `hpx-bench/experiments/12_hpx_vs_llama_server_pair/`. It is a Python 3 stdlib driver (no third-party deps) that launches each server in turn (never concurrently), polls TCP + warm-up readiness, issues requests, captures artifacts, and evaluates descriptive gates. It is deliberately external to HPX and to `llama.cpp`; it treats both servers as HTTP/SSE black boxes.

Key M8 semantics that the harness encodes:

- **`decode_budget` / `n_predict` is an upper bound, not an exact-count guarantee.** Row-level gates require `0 ≤ n_decoded ≤ decode_budget`. Exact-budget equality is only required on the canonical anchor `p0_b8` for `hpx-server`.
- **EOG-stop behavior differs between the two servers and is accepted.** On a prompt that ends in sentence-final punctuation (`p1`):
  - `hpx-server`: `n_decoded = 0`, `hash = 0x0000000000000000`, empty `text`. The engine EOG branch finalises as `completed` without appending the EOG token.
  - `llama-server`: `tokens_predicted = 1`, empty `content`, terminal `stop_type = eos`. One EOS chunk is reported in the predicted bookkeeping, no content arrives.
  Both are stable across iterations and both are accepted as `completed`/`eos` terminations.
- **Cross-server `n_decoded` equality is recorded, not gated.** The harness logs the comparison per shape but does not fail on a difference.
- **Cross-server text equality is not gated.** Sampler chains, tokenizer/detokenizer conventions, and BOS handling differ.
- **Streaming text reconstruction differs by server.** `hpx-server` emits one `event: token` per generated token; per-token detokenization loses SentencePiece word-boundary spacing. `llama-server` streams `content` chunks produced by its own incremental detokenizer with spacing preserved. Cross-server streaming text equality is therefore not meaningful and is not gated.
- **Streaming `ok=True` means a clean terminal stream record was observed**, not necessarily non-empty text. On EOG-stop rows the text can be empty and the row is still a successful streaming completion.

Three independent passing runs cover the M8 surface:

| Run ID | Mode | Shapes | Iters | Harness | Result |
|---|---|---|---|---|---|
| `20260521-231710-pair` | non-streaming | `p0_b8`, `p0_b32`, `p1_b8`, `p1_b32` | 3 | `m8d.0` | gates=PASS |
| `20260521-233521-pair` | streaming, canonical | `p0_b8` | 10 | `m8e.0` | gates=PASS |
| `20260521-234615-pair` | streaming, matrix | `p0_b8`, `p0_b32`, `p1_b8`, `p1_b32` | 3 | `m8e.0` | gates=PASS |

All three: zero `GATE FAIL`, zero forbidden-word hits, `parse_error == ""` and `stream_parse_error == ""` on every row. `hpx-server` `p0_b8` hash stayed at `0x0619d4d1900c2365`; `p0_b32` stayed at `0x6794e47fe0f84af1`.

M8 did not change the engine, gate, or benchmarked execution semantics. The only server-source change in M8 was the default-preserving `--ctx-size` precursor in `tools/hpx-server/hpx-server.cpp` (default `2048`), needed for matched context size between `hpx-server` and `llama-server`. Engine, types, gate sources, and CMake were untouched in M8; doc, README, and `provenance.md` updates were also out of scope for the M8 slices themselves.

## HPX nativity assessment

Based on the most recent validation reports for the engine and server:

- No live `std::thread` / `std::mutex` / `std::condition_variable` / `std::this_thread::sleep_for` in authored HPX engine or server control-plane code. Any matches in grep audits are documenting-absence comments rather than live usages.
- `std::atomic` appears only in accepted adapter-boundary or counter contexts (for example, capacity-lease accounting and metrics counters) and not as a substitute for an HPX synchronization primitive on the control path.
- `cpp-httplib` is the explicit non-HPX HTTP adapter boundary. HTTP I/O runs on cpp-httplib's own threads; the boundary is documented in `tools/hpx-server/`.
- The Python benchmark harness is external measurement infrastructure. It runs as a separate process, talks to each server via TCP/HTTP/SSE, and is not part of the HPX runtime.

## Validation anchors

Stable, repeated-across-milestones anchors:

- HPX canonical greedy `p0_b8` hash:
  - `0x0619d4d1900c2365`
- HPX `p0_b32` hash (greedy, same prompt, longer budget):
  - `0x6794e47fe0f84af1`
- Default stochastic, `seed=42`, `p0_b8`:
  - `0xa8e14acb4094aa3f`
- M8 evidence run IDs:
  - `20260521-231710-pair` (non-streaming 2×2 matrix)
  - `20260521-233521-pair` (canonical streaming, 10 repeats)
  - `20260521-234615-pair` (streaming 2×2 matrix)

## Known limitations

- No OpenAI-compatible endpoint yet (`/v1/chat/completions` not implemented).
- `cpp-httplib` remains the non-HPX HTTP adapter boundary; HTTP I/O is not on HPX threads by design.
- No auth, no TLS, no production-grade shutdown or signal handling.
- No multi-model loading or model swap.
- No HTTP-side queueing beyond the strict door-cap rejection from `--max-concurrent`.
- `hpx-server` does not expose a prompt-token count in its `/completion` response; the harness records `prompt_tokens = -1` for `hpx-server` rows.
- The benchmark harness records raw values only — no aggregation, no percentiles, no averaged TTFT, no throughput. By construction it does not produce performance claims.
- Handler logic and SSE wiring are duplicated across `hpx-server` smokes for evidence purposes; the duplication is known and is not on the M8 critical path.

