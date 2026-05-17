# Continuous batching — Streaming Slice 1 design

## 1. Purpose

Streaming Slice 1 introduces the first HPX-native **per-request
token streaming boundary** into the HPX continuous-batching
prototype. It is layered on top of the closed admission gate
sequence at `HPX_CB_ADMIT_STEP7: PASS` and opens a new gate
sequence at:

```text
HPX_CB_STREAM_STEP1: PASS
```

Streaming Slice 1 is a **correctness/lifecycle gate**, not a
serving layer. The gate proves that:

- the engine task can publish every generated token id onto a
  per-request HPX-native channel before the existing
  `request_result` promise is fulfilled,
- main can consume the channel and reconstruct the same
  per-request token sequence (bit-equal hash, equal token
  count, single close, close reason `completed`),
- the new channel layer adds no new `std::thread`,
  `std::condition_variable`, wall-clock sleep, or `std::mutex`,
- only the engine task ever touches `llama_context`,
  `llama_batch`, `llama_decode`, `llama_get_logits_ith`, or
  `llama_memory_seq_*` — the stream carries only `int32_t`
  token ids and a close reason.

No performance claim is made.

## 2. Relation to Slices 1–7

Streaming Slice 1 is **additive**. It does not extend or
replace the admission gate sequence:

- `HPX_CB_ADMIT_STEP7: PASS` remains the closing label of the
  admission/lifecycle gate sequence.
- `HPX_CB_STREAM_STEP1: PASS` is the opening label of a new
  streaming gate sequence.
- Every Slice 1–7 admission/cancellation/lifecycle invariant
  is a carry-over invariant under
  `HPX_CB_STREAM_STEP1`. The `--stream-all` OFF regression on
  the Slice 7 smoke shape proves this semantically (only the
  final stdout label line changes between
  `HPX_CB_ADMIT_STEP7` and `HPX_CB_STREAM_STEP1`).

The new code path is opt-in via `--stream-all` (default OFF).
With `--stream-all` OFF the engine allocates no channel, emits
no `token_stream_*` events, and keeps every Slice 8 counter
at zero — preserving Slice 7 semantics for every existing
regression.

## 3. Why HPX local channel instead of future-chain

An earlier source-only revision implemented the same Slice 8
gates over a manual future-chain:

```cpp
struct token_stream_node {
    int32_t              token_id;
    bool                 closed;
    stream_close_reason  close_reason;
    hpx::shared_future<std::shared_ptr<const token_stream_node>> next;
};
using token_stream_promise = hpx::promise<std::shared_ptr<const token_stream_node>>;
```

The future-chain passed all four runs. We refactored to
`hpx::lcos::local::channel` for these reasons:

1. **HPX-native abstraction.** The channel is a first-class
   HPX LCO with built-in spinlock-guarded buffering and an
   explicit `close()` notion. The future-chain was a hand-
   rolled linked list of one-shot promises.
2. **Split producer / consumer handles.** `send_channel<T>`
   and `receive_channel<T>` are derivable from a `channel<T>`
   and share an intrusive_ptr to the same impl. The boundary
   between engine (producer) and main (consumer) is enforced
   at the type level, not by convention.
3. **Lifetime via intrusive_ptr.** No need for
   `std::shared_ptr<const node>` bookkeeping, no terminal-
   node convention to encode "stream over". The channel
   carries its own closed-state.
4. **No new HPX primitive.** `hpx::lcos::local::channel` was
   already available in the local HPX install
   (`/Users/Ashk/Desktop/HPX/hpx-install/include/hpx/lcos_local/channel.hpp`).
   The channel impl is in `libhpx_core`, already linked by
   the gate via `hpx::promise` / `hpx::future` /
   `hpx::async`. No CMake change, no new link target.

Bit-identical anchor preservation: the channel-based source
produces the same observed hash anchors and counts as the
future-chain build on the smoke shape (budget-8
`0x0619d4d1900c2365`, budget-64 `0x3b15a0474dfe11be`,
budget-256 `0x8790fbe5a60c9ae6`,
`stream_tokens_emitted_total = 328`). See
`tools/hpx-continuous-batch-gate/results.md` under
*Streaming Slice 1 results — HPX local-channel token
stream*.

## 4. Stream event model

```cpp
enum class stream_event_kind : uint8_t {
    token  = 0,
    closed = 1,
};

enum class stream_close_reason : uint8_t {
    completed = 0,
    cancelled = 1,
    error     = 2,
};

struct token_stream_event {
    stream_event_kind    kind;          // token or closed
    int32_t              token_id;      // valid iff kind == token
    stream_close_reason  close_reason;  // valid iff kind == closed
};

using token_stream_channel =
    hpx::lcos::local::channel<token_stream_event>;
using token_stream_sender =
    hpx::lcos::local::send_channel<token_stream_event>;
using token_stream_receiver =
    hpx::lcos::local::receive_channel<token_stream_event>;
```

Invariants:

- Each streaming seq emits exactly **one** `kind=closed`
  event in its lifetime, after some number (possibly zero)
  of `kind=token` events.
- The terminal `kind=closed` event is sent **before**
  `channel.close()` is called. Reversed order would make
  `set()` throw `invalid_status`.
- `close_reason` on a `kind=token` event is meaningless and
  must not be inspected by the consumer.
- `token_id` on a `kind=closed` event is meaningless and
  must not be inspected by the consumer.
- The consumer's stop condition is `kind == closed`.
- A consumer that performs an additional `get()` after the
  terminal event will observe `hpx::error::invalid_status`
  (a clear error, not a hang) because `channel.close()` was
  called.

## 5. Ownership rules

| Side | Owns | Touches |
|---|---|---|
| Engine task (single HPX task) | `seq_state.stream_channel` (the full `channel<token_stream_event>`); `llama_context`, `llama_batch`, `llama_decode`, `llama_memory_seq_*`, `llama_get_logits_ith` | Engine alone publishes via `set()` and closes via `close()`. No other HPX task or thread touches the channel or any llama.cpp state. |
| Main (consumer) | The matching `receive_channel<token_stream_event>` for each bound active seq (collected once per repeat via `engine::take_stream_receivers()`) | Main alone drains via `rx.get(launch::sync)`. Main never touches `llama_context`, `llama_batch`, `llama_decode`, `llama_get_logits_ith`, or `llama_memory_seq_*`. |

Engine ctor (when `--stream-all` is on):

1. Mark every bound active seq `stream_enabled = true`.
2. Default-construct `seq.stream_channel = token_stream_channel{}`.
3. Push a derived `token_stream_receiver{seq.stream_channel}`
   onto `stream_receivers_` for main to collect.

Main, once per repeat, before scheduling `engine::run()`:

```cpp
std::vector<token_stream_receiver> stream_receivers =
    eng.take_stream_receivers();
```

Engine task, once per generated token (at both argmax sites,
post-prefill and post-decode), in this order:

```cpp
const llama_token next_id = argmax(logits, n_vocab_);
// ... EOG / admitted_prefill_argmax checks ...
publish_token(seq, static_cast<int32_t>(next_id));   // channel.set(token event)
seq.generated_tokens.push_back(next_id);
seq.hash_state = fold_token_hash(seq.hash_state,
                                 static_cast<int32_t>(next_id));
seq.n_decoded++;
seq.last_token = next_id;
```

Engine task, once per seq, at completion (`finalize_and_fulfill`):

```cpp
clear_and_check(seq, iter, mem);        // KV clear + cross-talk
close_stream(seq, completed);           // channel.set(closed) + channel.close()
fulfill_promise(seq, request_status::completed, ttc_us);
```

Engine task, defensively at cancellation
(`cancel_and_fulfill`):

```cpp
clear_and_check(seq, iter, mem);
close_stream(seq, cancelled);
fulfill_promise(seq, request_status::cancelled, ttc_us);
```

Engine task, defensively at end-of-run drain (`engine::run()`
post-bailout):

```cpp
for (auto & seq : seqs_) {
    if (seq.stream_enabled && !seq.stream_closed) {
        close_stream(seq, error);
    }
}
```

The cancellation and error close paths are wired but
**not exercised** by the Slice 8 smoke. They are scaffolding
for later slices.

## 6. CLI flag

```text
--stream-all   default: OFF
               Streaming Slice 1: enable HPX-native per-
               request token streaming via
               hpx::lcos::local::channel<token_stream_event>.
               Each bound active seq gets an engine-owned
               channel; main holds the matching
               receive_channel and drains the chain after
               engine_fut.get(). OFF preserves Slice 7
               semantics (label-line excluded comparison).
```

OFF-mode regression gate (when `--stream-all` is OFF, every
counter must be zero):

```text
streams_opened              == 0
streams_closed_completed    == 0
streams_closed_cancelled    == 0
streams_closed_error        == 0
stream_tokens_emitted_total == 0
stream_receivers vector is empty
```

## 7. Smoke shape

The canonical first smoke is the simplest possible streaming
shape: one streaming request per budget class, no admission,
no cancellation, no external arrivals.

```text
--stream-all
--n-seqs 3
--decode-budget-mix 8,64,256
--cancel-plan none
--n-waiting 0
--n-external-arrivals 0
```

Implicit defaults from prior slices:

```text
--prompt "Hello, my name is"
--hpx-os-threads 1                  (required for deterministic order)
--reuse-completed OFF
```

Three streaming requests, mapped 1:1 to
`seq_id ∈ {0, 1, 2}` with budgets `{8, 64, 256}`:

| seq_id | budget | streamed token count | close reason |
|---|---|---|---|
| 0 | 8   | 8   | `completed` |
| 1 | 64  | 64  | `completed` |
| 2 | 256 | 256 | `completed` |

`stream_tokens_emitted_total = 8 + 64 + 256 = 328`.

## 8. Required gates

Per-request gates (every repeat, every streaming request):

- `stream_token_count == rr.n_decoded`
- `streamed_hash == rr.hash` (FNV-1a over `int32_t` token
  ids, same fold as `seq_state::hash_state`)
- stream close count == 1 per seq
- close reason == `completed`

Engine-side counter gates (`--stream-all` ON):

```text
streams_opened           == n_active
streams_closed_completed == n_active
streams_closed_cancelled == 0
streams_closed_error     == 0
streams_opened == streams_closed_completed
                + streams_closed_cancelled
                + streams_closed_error
stream_tokens_emitted_total
              == sum(rr.n_decoded for streamed completed requests)
```

Engine-side counter gates (`--stream-all` OFF — fail-closed):

```text
streams_opened              == 0
streams_closed_completed    == 0
streams_closed_cancelled    == 0
streams_closed_error        == 0
stream_tokens_emitted_total == 0
```

Carry-over invariants from Slices 1–7 (still gated unchanged):

- residual KV empty across all `n_seq_max` slots at engine
  end,
- `--repeat 2` deterministic on the per-result snapshot tuple,
- exactly one HPX engine task per repeat,
- one `hpx::promise<request_result>` per seq, fulfilled only
  after KV clear + cross-talk check,
- every `llama_decode` call returns 0,
- engine inbox + `external_promises_` empty at run end (when
  the external path is exercised).

Streaming-specific determinism gate (added under `--repeat 2`):

- per-seq streamed token vectors bit-equal across repeats,
- per-seq close reasons identical across repeats.

Trace-off quietness gate (unchanged from prior slices):

- with `LLAMA_HPX_CB_TRACE` unset, zero `[hpx-cb-gate]
  event=` lines on stderr, **including** the three new
  `token_stream_*` events.

## 9. Trace events

All gated on `LLAMA_HPX_CB_TRACE=1`. Trace-off path remains
one atomic load + early return per call site.

```text
[hpx-cb-gate] event=token_stream_opened request=<id> seq_id=<id>
[hpx-cb-gate] event=token_stream_token  request=<id> seq_id=<id> pos=<int> token=<int>
[hpx-cb-gate] event=token_stream_closed request=<id> seq_id=<id> n_tokens=<int> reason=completed|cancelled|error
```

`token_stream_token` is high-cardinality (one event per
emitted token); the 1:1 `token_stream_token`-per-streamed-
token invariant holds.

Slice 8 smoke counts (trace-on):

```text
token_stream_opened: 3
token_stream_token:  328
token_stream_closed: 3   (all reason=completed)
```

## 10. Close reasons

```cpp
enum class stream_close_reason : uint8_t {
    completed = 0,   // sole reason exercised by the Slice 8 smoke
    cancelled = 1,   // wired in cancel_and_fulfill; not exercised yet
    error     = 2,   // wired in engine::run() bailout drain; not exercised yet
};
```

Hard rule: an engine bail-out path must **never** report
`completed`. The engine drain closes with `error`. A future
slice will exercise the `cancelled` and `error` paths.

## 11. Risks and deferred work

| Risk / scope | Mitigation in Slice 8 | Deferred to |
|---|---|---|
| Channel lifetime (orphaned consumer) | intrusive_ptr-managed impl outlives both halves; main always drains | — |
| Consumer too slow (unbounded buffering) | `channel<T>` is unbounded; engine never suspends on consumer; smoke budgets ≤ 256 | Backpressure / bounded-channel slice |
| Double close | Single-producer invariant + `if (stream_closed) return;` guard in `close_stream` | — |
| Streamed token / final result mismatch | Same `int32_t next_id` feeds `publish_token` and `hash_state`, before `n_decoded++` | — |
| llama.cpp state crossing the channel | `token_stream_event` is POD-of-(enum, int32, enum); code review gate on the event type | — |
| Trace-off regression | All new events ride the existing `LLAMA_HPX_CB_TRACE` gate; structural counters bump unconditionally; OFF-mode regression gate fail-closed | — |
| `--repeat 2` determinism | `--hpx-os-threads 1`, FIFO channel `set()` order under `hpx::spinlock`, single producer | — |
| Streaming under cancellation | `cancel_and_fulfill` calls `close_stream(_, cancelled)` defensively; no smoke yet | Slice 8b / 9: cancellation+streaming smoke |
| Engine-failure close semantics | `engine::run()` bailout drain closes with `error` defensively; no smoke yet | Slice for error-path close gating |
| Streaming through admitted reuser slots | Wiring is admission-agnostic; smoke has no admission; gates target the original-active path | Streaming+admission slice |
| HTTP / network consumer | No network adapter; the consumer is `main()` | Network adapter milestone (see §7 of `continuous_batching_gate_vs_serving_layer.md`) |
| Per-request prompts / sampling | Smoke shares the same prompt; no per-request sampling | Per-request prompt + sampling slice |

## 12. Safe claim language

Supported by Streaming Slice 1 (verbatim-safe):

- "Streaming Slice 1 adds a first HPX-native per-request
  token streaming boundary on top of the closed admission
  gate sequence. Per-request streams use
  `hpx::lcos::local::channel<token_stream_event>` with split
  `send_channel` / `receive_channel` halves; the engine task
  is the sole producer and main is the sole consumer."
- "Stream events carry only `int32_t` token ids and a close
  reason. No `llama_context`, `llama_batch`,
  `llama_memory_seq_*`, or logits state crosses the channel
  boundary."
- "The engine emits the terminal `closed` event before
  calling `channel.close()`. Per-request gates enforce that
  `stream_token_count == rr.n_decoded`,
  `streamed_hash == rr.hash`, close count == 1, and close
  reason == `completed` on the Slice 8 smoke. `--repeat 2`
  determinism extends to bit-equal streamed token vectors
  and close reasons."
- "The channel adds no new `std::thread`, no
  `std::condition_variable`, no wall-clock sleep, and no new
  `std::mutex`. Synchronization uses
  `hpx::lcos::local::channel`'s internal `hpx::spinlock`."

Not supported and should be avoided:

- "Streaming Slice 1 is a serving layer." (No network
  adapter; consumer is `main()`. See
  `continuous_batching_gate_vs_serving_layer.md` §5.)
- "Streaming Slice 1 supports cancellation." (Wired
  defensively; not gated by a smoke.)
- "Streaming Slice 1 is faster than X." (No comparative
  benchmark; correctness-only gate.)
- "Streaming Slice 1 supports backpressure." (Unlimited
  channel; bounded-channel policy is a deferred slice.)
