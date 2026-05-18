// Shared POD types for the HPX continuous-batch gate.
//
// Everything in this header is either:
//   - a plain enum / `*_name()` formatter,
//   - a passive struct (request lifecycle, admission metadata,
//     arrival/release plumbing, per-seq state, engine metrics/result),
//   - or a `using`-alias around an HPX local channel template
//     specialization.
//
// The engine class itself, the CLI, the submitter, and the validation
// harness all #include this header. No llama.cpp execution state lives
// here — `seq_state` carries only IDs, counters, and an HPX channel
// handle; the actual `llama_context`, `llama_batch`, and KV are engine-
// owned and never cross this boundary.

#pragma once

#include "token_hash.h"

#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// ---- Request lifecycle status (Cancel Slice 1: data model only) --------
enum class request_status : uint8_t {
    completed       = 0,
    cancelled       = 1,
    failed_reserved = 2,  // reserved for future slices; not produced now
};

inline const char * status_name(request_status s) noexcept {
    switch (s) {
        case request_status::completed:        return "completed";
        case request_status::cancelled:        return "cancelled";
        case request_status::failed_reserved:  return "failed_reserved";
    }
    return "unknown";
}

// ---- Live admission source (Live Admission Slice 1: data model only) ----
// `none` is the default for every result. `cancel_freed` marks an admitted
// request that reused a slot popped from the engine's free_due_to_cancel
// queue. `completion_freed` (Slice 5) marks a slot popped from
// free_due_to_completion. `initial_idle` (M2f) marks a slot popped from
// the engine's startup idle pool — never-bound seq_ids reserved at engine
// construction so submit_request can be admitted without any prior
// active completion or cancellation.
enum class admission_source : uint8_t {
    none             = 0,
    cancel_freed     = 1,
    completion_freed = 2,
    initial_idle     = 3,
};

inline const char * admission_source_name(admission_source s) noexcept {
    switch (s) {
        case admission_source::none:             return "none";
        case admission_source::cancel_freed:     return "cancel_freed";
        case admission_source::completion_freed: return "completion_freed";
        case admission_source::initial_idle:     return "initial_idle";
    }
    return "unknown";
}

// ---- Arrival source (Live Admission Slice 6: async external arrivals) --
// Distinguishes a waiting request that came preloaded at engine
// construction (via --n-waiting, the Slice 2..5 path) from one that
// was submitted at runtime by an external HPX submitter task (via
// engine::submit, the Slice 6 path). Orthogonal to admission_source:
// an external arrival can be admitted via either cancel_freed or
// completion_freed; in the Slice 6 smoke it is cancel_freed.
enum class arrival_source : uint8_t {
    preloaded = 0,
    external  = 1,
};

inline const char * arrival_source_name(arrival_source s) noexcept {
    switch (s) {
        case arrival_source::preloaded: return "preloaded";
        case arrival_source::external:  return "external";
    }
    return "unknown";
}

// ---- Streaming token channel (Slice 8: HPX-native local channel) -------
// Engine is the sole producer of token_stream_event values; caller is the
// sole consumer. Events carry token ids (kind=token) or a close reason
// (kind=closed). No llama.cpp state crosses the channel. The full channel
// handle is engine-owned; a receive_channel handle is taken once per
// repeat by main via engine::take_stream_receivers() before run() is
// scheduled. The underlying impl is intrusive_ptr-managed so it outlives
// either half until both halves are released.
enum class stream_close_reason : uint8_t {
    completed = 0,
    cancelled = 1,
    error     = 2,
};

inline const char * stream_close_reason_name(stream_close_reason r) noexcept {
    switch (r) {
        case stream_close_reason::completed: return "completed";
        case stream_close_reason::cancelled: return "cancelled";
        case stream_close_reason::error:     return "error";
    }
    return "unknown";
}

enum class stream_event_kind : uint8_t {
    token  = 0,
    closed = 1,
};

struct token_stream_event {
    stream_event_kind    kind         = stream_event_kind::token;
    int32_t              token_id     = 0;     // valid iff kind == token
    stream_close_reason  close_reason = stream_close_reason::completed;
                                                // valid iff kind == closed
};

using token_stream_channel =
    hpx::lcos::local::channel<token_stream_event>;
using token_stream_sender =
    hpx::lcos::local::send_channel<token_stream_event>;
using token_stream_receiver =
    hpx::lcos::local::receive_channel<token_stream_event>;

// Streaming Slice 3: explicit per-admission stream handoff bundle.
// admit_one constructs one of these whenever it rebinds a completion-
// freed slot's stream channel for a streamed admitted request and pushes
// it onto engine::admitted_stream_handoffs_ under the existing
// admitted_futures_mtx_ critical section. Main drains the bundle vector
// after engine_fut.get() and keys the per-request streamed-token vector
// by request_id. The bundle never crosses the channel itself — it only
// carries the request_id and the (movable) receiver handle. No llama.cpp
// state lives on this struct.
struct admitted_stream_handoff {
    int32_t              request_id = -1;
    token_stream_receiver rx;
};

// ---- Per-request snapshot (the only thing crossing the engine boundary) -
struct request_result {
    int32_t                  request_id           = 0;
    int32_t                  seq_id               = 0;
    int32_t                  decode_budget        = 0;
    int32_t                  n_decoded            = 0;
    uint64_t                 hash                 = 0;
    int32_t                  done_iter            = -1;
    llama_pos                pos_max_at_clear     = -1;
    bool                     kv_cleared           = false;
    int64_t                  ttc_us               = 0;
    std::vector<llama_token> generated_tokens;

    // ---- cancellation snapshot (Cancel Slice 1: defaults always) -------
    request_status           status               = request_status::completed;
    int32_t                  cancel_observed_iter = -1;
    int32_t                  n_decoded_at_cancel  = -1;

    // ---- live admission snapshot (Live Admission Slice 1: defaults
    //      always). All four fields are populated on every completion
    //      and cancellation path; in Slice 1 the values are the
    //      defaults below for every result.
    int32_t                  admitted_at_iter     = -1;
    int32_t                  reused_seq_id        = -1;
    int32_t                  previous_request_id  = -1;
    admission_source         admission_src        = admission_source::none;

    // ---- arrival snapshot (Live Admission Slice 6) ----------------------
    // preloaded for every initially-bound active seq AND for preloaded
    // waiters admitted via the Slice 2..5 path. external only for
    // arrivals that entered through engine::submit(). ---------------------
    arrival_source           arrival_src          = arrival_source::preloaded;
};

// ---- Per-seq state (engine-internal) ------------------------------------
// Cancellation fields are present in Cancel Slice 1 but dormant: the
// plan is not propagated into seq_state and the engine does not
// observe cancellation. Slice 2 will wire those.
struct seq_state {
    int32_t                  seq_id                       = 0;
    int32_t                  decode_budget                = 0;
    int32_t                  n_decoded                    = 0;
    int32_t                  pos_next                     = 0;
    int32_t                  i_batch                      = -1;
    llama_token              last_token                   = 0;
    bool                     done                         = false;
    int32_t                  done_iter                    = -1;
    llama_pos                pos_max_at_clear             = -1;
    bool                     kv_cleared                   = false;
    bool                     promise_fulfilled            = false;
    uint64_t                 hash_state                   = token_hash::k_token_hash_init;
    std::vector<llama_token> generated_tokens;

    // ---- cancellation data model (Cancel Slice 1: not exercised) -------
    std::atomic<bool>        cancel_requested{false};
    int32_t                  cancel_after_decoded_tokens  = -1;  // -1 = never
    bool                     cancel_observed              = false;
    int32_t                  cancel_observed_iter         = -1;
    int32_t                  n_decoded_at_cancel          = -1;

    // ---- live admission data model (Live Admission Slice 1: not
    //      exercised). `request_id` is initialized to `seq_id` by the
    //      engine constructor; no admission rebinds it in Slice 1, so
    //      `request_id == seq_id` is an enforced invariant on every
    //      result. `reused_seq_id` lives only on `request_result`, not
    //      here. -----------------------------------------------------
    int32_t                  request_id                   = -1;
    int32_t                  admitted_at_iter             = -1;
    int32_t                  previous_request_id          = -1;
    admission_source         admission_src                = admission_source::none;

    // ---- arrival data model (Live Admission Slice 6) --------------------
    // Records whether this request entered the engine via the preloaded
    // construction-time waiting queue (--n-waiting) or via an external
    // submitter through engine::submit() at runtime. Default for all
    // initially-bound active seqs is preloaded; admission overwrites
    // this from the waiting_request that was bound. -----------------------
    arrival_source           arrival_src                  = arrival_source::preloaded;

    // ---- streaming token channel (Slice 8: HPX-native local channel) ----
    // stream_enabled is set in engine ctor when --stream-all is on; it is
    // never toggled mid-run. stream_channel is the engine-owned full
    // channel handle (intrusive_ptr to the impl); publish_token sends a
    // token event, close_stream sends a closed event then calls close().
    // stream_closed flips to true at close time. stream_tokens_emitted
    // counts every non-terminal publish for this seq.
    bool                     stream_enabled               = false;
    bool                     stream_closed                = false;
    int32_t                  stream_tokens_emitted        = 0;
    token_stream_channel     stream_channel;

    // ---- M1a per-request prompt data model (not exercised) ------------
    // Per-seq owned prompt token vector. Engine code in M1a does NOT
    // read or write this field — the existing shared-prompt path
    // (engine_options::prompt_tokens consumed by both the initial
    // prefill and admitted-prefill loops) is unchanged. The field is
    // intentionally left default-empty across the entire run, including
    // across --repeat iterations and admission-rebinds, so the engine
    // is behavior-identical to M0. M1b will populate this from the
    // existing shared prompt at engine construction time and switch the
    // initial-prefill loop to read it; M1c will move per-waiter prompt
    // vectors in at admit_one time and drop the shared-prompt borrow.
    std::vector<llama_token> prompt_tokens;

    uint64_t finalize_hash() const noexcept {
        return (n_decoded == 0) ? token_hash::k_token_hash_empty : hash_state;
    }
};

// ---- Waiting request (Live Admission Slice 2: queue-only) --------------
// Constructed once by main and passed to the engine as a const-ref handle
// (preloaded path), OR constructed inside drain_external_inbox() from an
// arrival_msg (external path). Slice 6: `src` records which path; the
// admission step uses it to decide promise ownership at bind time.
struct waiting_request {
    int32_t                  request_id    = -1;
    int32_t                  decode_budget = 0;
    arrival_source           src           = arrival_source::preloaded;
    // M1c: per-waiter owned prompt vector. Populated by main for
    // preloaded waiters, or moved out of arrival_msg by
    // drain_external_inbox for external arrivals. admit_one moves
    // this into the bound slot's seq_state::prompt_tokens at the
    // admission boundary, after which the admitted-prefill loop in
    // run_body() reads the per-seq vector. In M1c every waiter's
    // prompt is still a copy of the shared prompt source, so the
    // admitted-prefill batch is byte-identical to M1b / M0.
    std::vector<llama_token> prompt_tokens;
};

// ---- Async external arrival (Live Admission Slice 6) -------------------
// arrival_msg crosses from the external submitter HPX task to the
// engine task via engine::submit(), which acquires the inbox spinlock
// and pushes. The promise is created by the submitter; the submitter
// holds the corresponding future. At admission, the engine moves the
// promise into the slot's promise position and fulfills it via the
// existing fulfill_promise() path.
struct arrival_msg {
    int32_t                      request_id    = -1;
    int32_t                      decode_budget = 0;
    hpx::promise<request_result> promise;
    arrival_source               src           = arrival_source::external;
    // M1c: per-arrival owned prompt vector. Populated by the
    // scripted submitter task from the matching scripted_arrival
    // before eng.submit(). drain_external_inbox() moves it into
    // the new waiting_request and pushes it onto the consumable
    // waiting queue; admit_one then moves it into the bound
    // slot's seq_state::prompt_tokens.
    std::vector<llama_token>     prompt_tokens;
    // M2g: per-arrival opt-in streaming. `want_stream` is the bool the
    // caller set on `submit_request`; `stream_channel` carries the
    // engine-bound half of an HPX local channel constructed inside
    // `engine::submit_request` when `want_stream` is true. The receiver
    // half is returned to the caller via `submit_handle.stream` and the
    // channel is moved through `drain_external_inbox` into the engine's
    // private `external_stream_channels_` map keyed by request_id, then
    // moved a final time into the bound slot at `admit_one`.
    bool                                want_stream    = false;
    std::optional<token_stream_channel> stream_channel;
};

// ---- Public submit API (M2b) -------------------------------------------
// `submit_request` is the public counterpart to the engine-internal
// `arrival_msg`. Clients fill in the four fields below and call
// `engine::submit_request(...)` to enqueue work; the engine constructs
// the matching `arrival_msg` (with src=external and a fresh
// hpx::promise) and routes through the existing inbox/spinlock path.
// `arrival_source` is intentionally NOT exposed — runtime submissions
// always become arrival_source::external internally. Sampling
// configuration is intentionally NOT included; a later stage adds it.
struct submit_request {
    int32_t                  request_id    = -1;
    std::vector<llama_token> prompt_tokens;
    int32_t                  decode_budget = 0;
    bool                     want_stream   = false;
};

// `submit_handle` is the move-only result handle returned by
// `engine::submit_request(...)`. `result` is the per-request future
// (HPX-native, never std::future); awaiting it yields the same
// `request_result` snapshot the legacy path produces. `stream` is
// `std::nullopt` in M2b — stream support is deferred to M2c, where
// the engine's admit_one branch can be wired at the same time as the
// gate's submitter migration. Passing want_stream=true today causes
// `submit_request(...)` to throw with an explicit "not yet
// implemented" message so callers cannot silently miss the receiver.
struct submit_handle {
    hpx::future<request_result>          result;
    std::optional<token_stream_receiver> stream;
};

// ---- Pre-run release/ack handle (Live Admission Slice 6) ---------------
// Returned by engine::register_external_release_iter(K) before run().
// `release_future` is awaited by the submitter at the start of its
// per-K block. `ack_promise` is set_value()'d by the submitter after
// all K-arrivals have been pushed; the engine suspends on the matching
// future and only resumes once the ack fires (release+ack barrier).
struct external_release_handle {
    hpx::future<void>  release_future;
    hpx::promise<void> ack_promise;
};

// ---- Submitter script entry (Live Admission Slice 6) -------------------
// One per scripted arrival. The submitter walks the script in order;
// arrivals sharing the same release_iter are pushed in script order
// under a single release+ack barrier.
struct scripted_arrival {
    int32_t                  request_id    = -1;
    int32_t                  decode_budget = 0;
    int32_t                  release_iter  = -1;
    // M1c: per-arrival owned prompt vector populated by main from
    // the existing shared prompt source. The submitter task moves
    // it into the matching arrival_msg before calling eng.submit().
    std::vector<llama_token> prompt_tokens;
};

// ---- Engine metrics -----------------------------------------------------
struct engine_metrics {
    double               wall_ms              = 0.0;
    int32_t              decode_calls         = 0;
    int32_t              update_iterations    = 0;
    std::vector<int32_t> rows_per_batch;       // one per llama_decode call
    std::vector<int32_t> active_seqs_per_iter; // one per decode iteration
    // Live Admission Slice 4: queue depth sampled AFTER this iter's
    // admission loop runs. For the smoke shape, this stays at
    // n_waiting for iters 1..16 (cancel pushes free_due_to_cancel
    // entries during iter 16, but admission consumes them at iter 17),
    // then drops to 0 at iter 17 and stays 0 for the remaining decode
    // iters. One sample per decode iter; size == update_iterations.
    std::vector<int32_t> waiting_queue_depth_after_admission_per_iter;
    // Live Admission Slice 4: distinct decode-iter values at which
    // admission fired (one push per iter when the admission loop
    // bound at least one waiting request). Insertion order is
    // monotonically increasing, so it doubles as the iter set.
    std::vector<int32_t> admission_iter_set;
};

// ---- Engine result (engine-task-level diagnostics) ----------------------
struct engine_result {
    bool           ok                 = false;
    std::string    error;
    int32_t        decode_calls       = 0;   // mirror of metrics.decode_calls
    int32_t        decode_failures    = 0;
    int32_t        engine_task_count  = 0;
    int32_t        promises_fulfilled = 0;
    int32_t        cancelled_count    = 0;   // seqs fulfilled with status=cancelled
    bool           residual_kv_ok     = false;
    std::string    residual_kv_error;
    // Live Admission Slice 3: queue visibility + admission counters.
    // queued_count is sampled at engine start; waiting_queue_size_at_engine_end
    // is sampled at engine end. consumed = queued_count - end_size must
    // equal admitted_count. reused_seq_id_set lists the slot ids that
    // served an admitted request, in admission order (deterministic
    // ascending).
    int32_t              queued_count                     = 0;
    int32_t              waiting_queue_size_at_engine_end = 0;
    int32_t              admitted_count                   = 0;
    std::vector<int32_t> reused_seq_id_set;
    // Live Admission Slice 4: structural counter incremented next to
    // the admitted_prefilled trace site (unconditionally, regardless
    // of trace gating). Main asserts admitted_prefill_events ==
    // admitted_count at end of each repeat — catches "exactly one
    // admitted_prefilled per admitted request" without depending on
    // trace being enabled.
    int32_t              admitted_prefill_events          = 0;
    // Live Admission Slice 5: residual size of free_due_to_completion_
    // at engine end. Demand-gated push semantics mean unused freed
    // slots from the first wave remain pooled and KV-empty. For the
    // Slice 5 smoke shape (n_active=90, n_waiting=9, waiting_budget=8,
    // round-robin {8,64,256}) the expected residual is 21 (30 budget-8
    // slots completed at iter 7, 9 consumed by admission at iter 8).
    int32_t              completion_freed_pool_size_at_run_end = 0;

    // Live Admission Slice 6: async external arrival counters.
    // arrival_drained_count is incremented by the engine task in
    // drain_external_inbox() — engine::submit() never mutates result_
    // (Correction 1). external_admitted_count is incremented at the
    // admission step when a waiter with src=external is bound to a
    // freed slot. first_external_drain_iter records the iter of the
    // first arrival_drained event (-1 if none). iter_release_fired_set
    // and submitter_ack_set are pushed by the end-of-iter release+ack
    // barrier; their insertion order is monotonically ascending.
    int32_t              arrival_drained_count       = 0;
    int32_t              external_admitted_count     = 0;
    int32_t              first_external_drain_iter   = -1;
    std::vector<int32_t> iter_release_fired_set;
    std::vector<int32_t> submitter_ack_set;

    // Streaming Slice 8 counters. Maintained by the engine task only.
    // streams_opened bumps once per seq that had stream_enabled at engine
    // start. streams_closed_{completed,cancelled,error} partition the
    // close call sites — exactly one of the three fires per opened
    // stream. stream_tokens_emitted_total accumulates non-terminal
    // publishes across all streamed seqs in this run.
    int32_t              streams_opened              = 0;
    int32_t              streams_closed_completed    = 0;
    int32_t              streams_closed_cancelled    = 0;
    int32_t              streams_closed_error        = 0;
    int64_t              stream_tokens_emitted_total = 0;

    engine_metrics       metrics;
};
