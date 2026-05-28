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
// M6b: pulled in for `llama_sampler_ptr`, the std::unique_ptr alias
// with the correct llama_sampler_free deleter. The header is C++-only
// and itself includes <memory> plus llama.h, so adding it here is the
// minimal-surface way to give `seq_state` an owning sampler chain
// handle without forcing every TU that includes types.h to manage
// raw llama_sampler* lifetimes. The handle is engine-task-owned; no
// other TU touches it (mutating sampler state would violate the
// "engine task is the sole owner of llama.cpp mutable execution
// state" invariant).
#include "llama-cpp.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// ---- Forward declarations ----------------------------------------------
// M5b: `submit_handle::cancel(engine &)` is a thin forwarder onto
// `engine::cancel_request(const cancel_token &)`. The forwarder body
// lives in engine.cpp, where the full `engine` definition is in scope;
// only this forward declaration is needed in the header to declare the
// member. Keeping types.h independent of the full engine definition
// preserves the existing layering — types.h is included by engine.h,
// the gate driver, smokes, and library consumers, none of which should
// be forced to see the engine class definition just to use submit_handle.
class engine;

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

// ---- Per-request sampling configuration (M6a: data model only) ---------
// Carried alongside `prompt_tokens` / `decode_budget` from public
// `submit_request` through `arrival_msg`, `waiting_request`, and finally
// into `seq_state`. M6a is plumbing-only: every sampling site in
// `engine.cpp` (post-prefill argmax and per-iter decode argmax) still
// reads logits via `llama_get_logits_ith` and selects the next token
// through the local `argmax(logits, n_vocab_)` helper. The field is
// carried so M6b can branch on `seq_state::sampling.mode` without
// re-touching the plumbing surface.
//
// Greedy default (`sampling_mode::greedy`, all other fields irrelevant
// for the greedy branch) keeps the existing argmax path bit-identical,
// which is required to preserve the M3a/M4/M5 canonical hashes (e.g.
// budget-8: 0x0619d4d1900c2365). Stochastic mode is wired in M6b via a
// per-seq `llama_sampler_ptr`; in M6a, a stochastic-looking config is
// carried but ignored, so a stochastic submission still produces the
// greedy canonical hash. The carry smoke asserts both branches in M6a
// and the M6b smoke flips the stochastic assertion to "differs from
// greedy".
enum class sampling_mode : uint8_t {
    greedy     = 0,
    stochastic = 1,
};

inline const char * sampling_mode_name(sampling_mode m) noexcept {
    switch (m) {
        case sampling_mode::greedy:     return "greedy";
        case sampling_mode::stochastic: return "stochastic";
    }
    return "unknown";
}

struct sampling_config {
    sampling_mode mode             = sampling_mode::greedy;
    uint32_t      seed             = LLAMA_DEFAULT_SEED;
    float         temperature      = 1.0f;
    int32_t       top_k            = 0;
    float         top_p            = 1.0f;
    uint32_t      top_p_min_keep   = 1;
};

// M7b: shared validation helper. Single source of truth for the
// `sampling_config` invariants — used both by `engine.cpp`'s
// `build_sampler_chain` (engine-task fail-closed safety net) and by
// the hpx-server `/completion` handler (network-edge fast-fail before
// `submit_request`). Returns true on success; on failure returns false
// and populates `err` with a stable human-readable message. The error
// strings here are the contract — engine-side and server-side error
// reporting both surface them verbatim. Pure value-in / value-out:
// touches no llama.cpp state, takes no locks, allocates only the
// error string.
inline bool validate_sampling_config(const sampling_config & cfg,
                                     std::string &           err) {
    err.clear();
    if (!(cfg.temperature > 0.0f)) {
        err = "sampling_config.temperature must be > 0";
        return false;
    }
    if (cfg.top_k < 0) {
        err = "sampling_config.top_k must be >= 0";
        return false;
    }
    if (!(cfg.top_p > 0.0f && cfg.top_p <= 1.0f)) {
        err = "sampling_config.top_p must be in (0, 1]";
        return false;
    }
    if (cfg.top_p_min_keep == 0) {
        err = "sampling_config.top_p_min_keep must be >= 1";
        return false;
    }
    return true;
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

    // ---- N3.1: control-plane responsiveness timestamps (Exp 13) ---------
    // Absolute steady_clock::time_since_epoch() microseconds, written
    // by the engine task ONLY when
    // engine_options::lib.enable_responsiveness_timing == true.
    // Default -1 means "field not populated for this result". All four
    // fields default to -1 on every existing path (smokes, gate,
    // hpx-server, M8 configs) because they leave the option off.
    //
    // Semantics:
    //   t_admitted_us         set at admit_one bind site (live admission)
    //   t_first_publish_us    set at the first publish_token call for
    //                         this seq (streaming requests only)
    //   t_complete_us         set in finalize_and_fulfill, just before
    //                         the request_status::completed promise
    //   t_cancel_observed_us  set in cancel_and_fulfill (active cancel)
    //                         or fulfill_queued_cancelled (queued cancel)
    //
    // Preloaded actives keep t_admitted_us=-1: they are not admitted
    // through submit_request and are not part of the responsiveness
    // surface.
    int64_t                  t_admitted_us        = -1;
    int64_t                  t_first_publish_us   = -1;
    int64_t                  t_complete_us        = -1;
    int64_t                  t_cancel_observed_us = -1;
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
    // ---- Slice B (PrefillBudgetPolicy prep — no-behavior) --------------
    // prefill_cursor is the index of the next prompt row not yet placed in
    // a batch. Today's whole-prompt prefill jumps it from 0 to
    // prompt_tokens.size() in one iter; prefill_complete becomes true in
    // that same iter, before any sampling. Both fields are INERT in Slice
    // B: no chunking, no policy consult, no control-flow change — they are
    // set/reset alongside the existing per-occupant state and guarded by
    // debug assertions only. Slice C wires the per-iter budget that makes
    // prefill_cursor advance in steps. See
    // docs/hpx/prefill_budget_policy_design.md §4.
    int32_t                  prefill_cursor               = 0;
    bool                     prefill_complete             = false;
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

    // ---- M5a epoch ------------------------------------------------------
    // Per-seq engine-issued epoch propagated from `waiting_request::epoch`
    // by `admit_one` at admission time, or 0 for preloaded actives /
    // idle slots that never went through `engine::submit_request`. Used
    // by `cancel_and_fulfill` / `finalize_and_fulfill` / queued-cancel
    // fulfill to perform a guarded erase against `live_epoch_by_rid_`:
    // the entry is removed only if the map still points at this exact
    // (rid, epoch) pair, so a late completion of an older incarnation
    // cannot wipe the live entry of a newer same-rid submission.
    uint64_t                 epoch                        = 0;

    // ---- N3.1: responsiveness timestamps (Exp 13) ----------------------
    // Engine-task-only carriers; copied into the matching request_result
    // fields at fulfill time. Default -1; only written when
    // engine_options::lib.enable_responsiveness_timing == true.
    // t_admitted_us and t_first_publish_us are reset to -1 on every
    // admit_one rebind so a prior slot owner's stamps cannot leak. The
    // fulfill-time fields (t_complete_us, t_cancel_observed_us) are
    // written immediately before fulfill_promise; no rebind reset
    // needed for them because they are produced and consumed in the
    // same iteration's fulfill path.
    int64_t                  t_admitted_us                = -1;
    int64_t                  t_first_publish_us           = -1;
    int64_t                  t_complete_us                = -1;
    int64_t                  t_cancel_observed_us         = -1;

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

    // ---- M6a per-request sampling configuration (data model only) -----
    // Populated by `admit_one` by moving from the bound waiter's
    // `sampling` field at the same point `prompt_tokens` is rebound.
    // M6a does NOT read this in either sampling site — both still go
    // through the local argmax. M6b will branch here to drive a per-seq
    // llama_sampler chain when `sampling.mode == stochastic`. Default-
    // constructed value is greedy, so preloaded actives and any path
    // that does not explicitly set sampling stay on the existing argmax
    // path and keep canonical-hash byte equality.
    sampling_config          sampling                     = {};

    // ---- M6b per-seq sampler chain (engine-task-owned) ----------------
    // Default-null. `admit_one` first calls `.reset()` on this field
    // (clearing any chain left behind by a prior owner of the slot),
    // then — only when `sampling.mode == sampling_mode::stochastic` —
    // builds a fresh chain via:
    //   llama_sampler_chain_init(default_params{no_perf=true})
    //   + llama_sampler_chain_add(... llama_sampler_init_dist(seed))
    // and installs it with `.reset(chain)`. `finalize_and_fulfill` and
    // `cancel_and_fulfill` reset the chain at the end of the slot's
    // lifetime so the unique_ptr's deleter runs `llama_sampler_free`
    // promptly. Greedy/default admissions leave this null, so the
    // existing `llama_get_logits_ith` + local argmax path runs
    // bit-identically (a single null-pointer compare guards the
    // branch).
    //
    // Ownership rules (CRITICAL):
    //   - Only the engine task may construct, mutate, sample from,
    //     reset, or destroy this chain. The two sampling sites in
    //     `run_body` and the construction/reset sites in admit_one /
    //     finalize_and_fulfill / cancel_and_fulfill are all reached
    //     exclusively from the engine task.
    //   - `llama_sampler_chain_add` TRANSFERS ownership of the sub-
    //     sampler (e.g. the dist sampler from `llama_sampler_init_dist`)
    //     to the chain. The sub-sampler MUST NOT be wrapped in a
    //     separate llama_sampler_ptr after that call — the chain's
    //     deleter frees it.
    //   - `llama_sampler_sample(chain, ctx, idx)` internally calls
    //     `llama_sampler_accept`, so the engine MUST NOT call
    //     `llama_sampler_accept` separately — doing so would double-
    //     advance the dist RNG and break same-seed reproducibility.
    llama_sampler_ptr        sampler_chain;

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
    // M5a: epoch carried from `arrival_msg::epoch` when
    // `drain_external_inbox` constructs the waiter from an external
    // arrival. 0 for preloaded waiters (no engine-issued token).
    // `admit_one` copies this into `seq_state::epoch` at admission.
    uint64_t                 epoch         = 0;
    // M1c: per-waiter owned prompt vector. Populated by main for
    // preloaded waiters, or moved out of arrival_msg by
    // drain_external_inbox for external arrivals. admit_one moves
    // this into the bound slot's seq_state::prompt_tokens at the
    // admission boundary, after which the admitted-prefill loop in
    // run_body() reads the per-seq vector. In M1c every waiter's
    // prompt is still a copy of the shared prompt source, so the
    // admitted-prefill batch is byte-identical to M1b / M0.
    std::vector<llama_token> prompt_tokens;
    // M6a: per-request sampling configuration carried from
    // `arrival_msg::sampling` by `drain_external_inbox` and moved into
    // `seq_state::sampling` by `admit_one`. Default greedy for
    // preloaded waiters (`scripted_arrival` does not yet expose
    // sampling, so the gate's scripted-submitter path stays greedy).
    sampling_config          sampling      = {};
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
    // M5a: engine-issued epoch for this submission. Set by
    // `engine::submit_request` from the engine's monotonic
    // `next_epoch_` counter under `inbox_mtx_`; left at 0 for
    // arrivals that are constructed and pushed via the engine-
    // internal `engine::submit(arrival_msg)` path (the gate's
    // scripted submitter). The engine treats epoch 0 as "no
    // token-aware identity" and does not register such arrivals in
    // `live_epoch_by_rid_`, so they never participate in epoch-aware
    // cancel resolution.
    uint64_t                     epoch         = 0;
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
    // M6a: per-request sampling configuration. Copied from
    // `submit_request::sampling` in `engine::submit_request` before the
    // inbox push, and moved into `waiting_request::sampling` by
    // `drain_external_inbox`. Engine-internal `engine::submit(arrival_msg)`
    // callers that bypass `submit_request` (the gate's scripted
    // submitter) leave this default-constructed (greedy), so existing
    // gate shapes are unaffected.
    sampling_config                     sampling       = {};
};

// ---- Public submit API (M2b) -------------------------------------------
// `submit_request` is the public counterpart to the engine-internal
// `arrival_msg`. Clients fill in the four fields below and call
// `engine::submit_request(...)` to enqueue work; the engine constructs
// the matching `arrival_msg` (with src=external and a fresh
// hpx::promise) and routes through the existing inbox/spinlock path.
// `arrival_source` is intentionally NOT exposed — runtime submissions
// always become arrival_source::external internally. M6a: `sampling`
// is now exposed and carried end-to-end through `arrival_msg` →
// `waiting_request` → `seq_state`, but is NOT yet read at the sampling
// sites; both sampling sites still go through the local argmax. A
// default-constructed `sampling_config` (greedy) keeps the canonical
// hashes byte-identical. M6b activates `sampling.mode == stochastic`.
struct submit_request {
    int32_t                  request_id    = -1;
    std::vector<llama_token> prompt_tokens;
    int32_t                  decode_budget = 0;
    bool                     want_stream   = false;
    sampling_config          sampling      = {};
};

// ---- M5a: cancel_token ------------------------------------------------
// Opaque identity carried on `submit_handle` and accepted by the
// epoch-aware overload `engine::cancel_request(const cancel_token &)`.
// `request_id` is the caller-supplied rid (copied from
// submit_request.request_id at issuance time); `epoch` is engine-issued
// (monotonically increasing under `inbox_mtx_` inside
// `engine::submit_request`). Cancellation via this token cancels the
// specific request instance the token was issued for: if the rid is
// later reused for a new request, an old token's `(rid, epoch)` pair
// will NOT match the new live request's epoch, and the engine bumps
// `engine_result::cancel_stale_epoch` rather than cancelling the new
// request. Default-constructed tokens have `epoch = 0`, which is the
// "unset" sentinel — engine-issued tokens always have epoch >= 1.
// Trivially copyable; no engine back-pointer, no shared/weak handle to
// the engine. Caller-supplied lifetime; the token must not outlive the
// engine. M5b will add `submit_handle::cancel()` as a convenience
// forwarder; M5a leaves the token caller-driven via the engine API.
struct cancel_token {
    int32_t  request_id = -1;
    uint64_t epoch      = 0;
};

// `submit_handle` is the move-only result handle returned by
// `engine::submit_request(...)`. `result` is the per-request future
// (HPX-native, never std::future); awaiting it yields the same
// `request_result` snapshot the legacy path produces. `stream` is
// engaged when the caller set `want_stream=true` on `submit_request`,
// in which case it carries the receiver half of the per-request
// HPX local channel the engine populates with token_stream_event
// values as decoding progresses; it is `std::nullopt` otherwise.
// M5a: `token` carries the engine-issued `(rid, epoch)` identity for
// epoch-aware cancellation via
// `engine::cancel_request(const cancel_token &)`. The token is set by
// `engine::submit_request` before the handle is returned; the caller
// does not set or mutate it. Trivially copyable, so the caller may
// hold a copy of the token independently of the move-only handle.
struct submit_handle {
    hpx::future<request_result>          result;
    std::optional<token_stream_receiver> stream;
    cancel_token                         token;

    // M5b: explicit-engine convenience forwarder onto the M5a
    // token-aware cancel path. Equivalent to:
    //   eng.cancel_request(this->token);
    // Fire-and-forget; final truth is observed through `result`.
    // Does NOT introduce a new cancellation semantic, does NOT touch
    // llama.cpp state, and does NOT store any engine observer on the
    // handle — the caller asserts engine liveness by passing the
    // reference, exactly as for any C++ reference. Defined in
    // engine.cpp, where the full `engine` type is in scope.
    void cancel(engine & eng) const noexcept;
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

    // ---- Phase 1 serving-overhead diagnostics ---------------------------
    // Populated only when LLAMA_HPX_DIAG_METRICS=1. When the env switch
    // is unset, every vector stays default-constructed (no heap
    // allocation) and the engine code does not touch these fields. The
    // existing rows_per_batch (= batch_n_tokens per llama_decode call)
    // and active_seqs_per_iter are reused as-is — Phase 1 does not
    // duplicate them.
    //
    // When the switch is ON, each vector has length == update_iterations
    // at engine end (one row per inner-while iteration), and the JSONL
    // dump in engine.cpp walks them in lockstep.
    //
    // Two llama_decode call sites exist (the per-iter site and the
    // admitted-prefill argmax post-iter site). Per the Phase 1 roadmap,
    // both sites' wall times accumulate into the same per-iter entry
    // in llama_decode_wall_us_per_iter so the analysis layer does not
    // double-attribute a single iteration.
    //
    // idle_wait_us_per_iter is best-effort: if a single low-risk wait
    // site exists in the engine inbox path it carries real values;
    // otherwise the engine emits zeros (deferred to Phase 2).
    std::vector<int32_t> prefill_rows_per_iter;
    std::vector<int32_t> decode_rows_per_iter;
    std::vector<int32_t> admitted_per_iter;
    std::vector<int32_t> completed_per_iter;
    std::vector<int32_t> cancelled_per_iter;
    std::vector<int32_t> tokens_emitted_per_iter;
    std::vector<int64_t> llama_decode_wall_us_per_iter;
    std::vector<int64_t> iter_wall_us_per_iter;
    std::vector<int64_t> idle_wait_us_per_iter;

    // Phase 1 absolute per-iter timestamp, in microseconds since the
    // engine's run_body t_start_ baseline (the same baseline used for
    // wall_ms / Exp-13 responsiveness timestamps). Lets server-side
    // JSONL request events align to a common engine-relative axis for
    // c > 1 service-overhead analysis. One push per iter at the same
    // site as active_seqs_per_iter; default-empty (no allocation) when
    // LLAMA_HPX_DIAG_METRICS is unset.
    std::vector<int64_t> t_us_from_engine_start_per_iter;

    // Phase 1 transient per-iter accumulator. Lives inside
    // engine_metrics so the engine owns per-engine scratch state
    // without adding a new private member to class engine (engine.h
    // is out-of-scope for this slice). Each engine instance has its
    // own engine_metrics inside its own engine_result, so multiple
    // engines in one process never share this state — replacing the
    // earlier file-scope `s_iter_*` statics that would have aliased
    // across concurrent engines. Reset by engine::run_body at iter
    // top; read by engine::iter_run_decode at push time. Only touched
    // when LLAMA_HPX_DIAG_METRICS=1; otherwise stays default-init
    // (no chrono call, no allocation, no I/O).
    //
    // t0_us is microseconds since the engine's t_start_ baseline, so
    // this struct stays POD-friendly and types.h does NOT need to
    // pull in <chrono>.
    struct phase1_iter_acc {
        int32_t admitted     = 0;
        int32_t cancelled    = 0;
        int32_t prefill_rows = 0;
        int32_t decode_rows  = 0;
        int64_t t0_us        = 0;
    };
    phase1_iter_acc cur_iter_diag;

    // c=4 attribution: engine-task-only transient accumulator for time
    // spent HPX-suspended in wait_inbox_blocking() at the keep-alive
    // outer-loop tail. Lives OUTSIDE phase1_iter_acc so the per-iter
    // reset (run_body iter top) never clears it: the idle wait happens
    // between inner-loop passes and is credited to the NEXT iter that
    // produces a decode row. Bumped only when LLAMA_HPX_DIAG_METRICS=1;
    // drained into idle_wait_us_per_iter.back() and reset to 0 at the
    // iter_run_decode push site. Default 0; zero-cost when diag is off
    // (the wait site's timing is guarded). Non-keep-alive engines never
    // reach the wait site, so this stays 0 for every gate smoke.
    int64_t pending_idle_wait_us = 0;
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

    // M3a: long-running keep-alive counters. Default zero and stay
    // zero on every engine_options::keep_alive=false run, so the
    // seven canonical gate smokes remain byte-identical. With
    // keep_alive=true: engine_idle_waits bumps once per cv-suspend
    // (a single bump can cover any number of arrivals that drain in
    // the wake), and engine_shutdown_observed is set to 1 exactly
    // once when the decode loop exits because shutdown was
    // requested and all work drained.
    int32_t              engine_idle_waits           = 0;
    int32_t              engine_shutdown_observed    = 0;

    // M3b/M3c: cancellation counters. All default zero and stay
    // zero unless engine::cancel_request() is called, so every
    // existing engine/gate smoke remains byte-identical.
    //
    // queued_cancelled (M3b) — bumps once per request that was
    //   resolved with status=cancelled BEFORE admission (either
    //   intercepted at drain_external_inbox or pulled out of
    //   waiting_queue_consumable_ by apply_queued_cancellations).
    //
    // cancel_request_calls (M3b) — bumps once per drained
    //   cancel_inbox_ entry (duplicate cancel_request(rid) calls
    //   count twice).
    //
    // cancel_active_not_supported (legacy M3b counter) — was
    //   bumped by the pre-M3c stub when apply_queued_cancellations
    //   matched a live active seq but could not yet propagate
    //   cancellation. M3c replaced that stub with the real
    //   active-cancel bridge, so this counter MUST remain zero in
    //   M3c+ runs. Field retained so archived M3b evidence parses
    //   cleanly; do not bump from new code.
    //
    // cancel_active_observed (M3c) — the active-cancel bridge
    //   counter. Bumps once per rid that apply_queued_cancellations
    //   matches against a live non-done active seq. The match sets
    //   seq.cancel_requested under the engine task; the existing
    //   cancel_should_observe → cancel_and_fulfill iter-boundary
    //   pipeline performs the KV clear, stream close with
    //   reason=cancelled, and promise fulfillment.
    //
    // cancel_unknown_request_id (M3b) — bumps at engine end for any
    //   rid that was cancelled but never matched a queued/active
    //   request (cancel-before-submit that never received a
    //   submission, or cancel-after-completion).
    //
    // cancel_request_duplicates (M3b) — bumps when the engine
    //   drains a cancel_inbox_ entry whose rid was already present
    //   in cancelled_request_ids_.
    int32_t              queued_cancelled            = 0;
    int32_t              cancel_request_calls        = 0;
    int32_t              cancel_active_not_supported = 0;
    int32_t              cancel_active_observed      = 0;
    int32_t              cancel_unknown_request_id   = 0;
    int32_t              cancel_request_duplicates   = 0;

    // M5a: epoch-aware stale-token counter. Bumps when the token-aware
    // overload `engine::cancel_request(const cancel_token &)` resolves
    // a (rid, epoch) entry whose rid is currently live in
    // `live_epoch_by_rid_` but whose epoch does NOT match the live
    // epoch — i.e. the token was issued for a prior incarnation of the
    // rid that has since completed and been replaced by a fresh
    // submission. Stale tokens do NOT cancel the live request. Stays
    // 0 in every existing engine smoke and in every canonical gate
    // shape (no stale token is fired there), so adding the counter is
    // additive-only on the diff. Distinct from
    // `cancel_unknown_request_id`, which bumps when the rid is not
    // currently live at all (cancel-after-completion of an unrecycled
    // rid). Not printed by the gate driver; consumed only by the M5a
    // stale-token smoke via `eng.result().cancel_stale_epoch`.
    int32_t              cancel_stale_epoch          = 0;

    // c=4 attribution: absolute steady_clock microseconds of this
    // engine's run_body t_start_ baseline (t_start_.time_since_epoch()).
    // Lets server-side server_request JSONL rows (which use absolute
    // server_now_us) and engine per-iter rows (which use the relative
    // t_us_from_engine_start delta) be overlaid on one timeline:
    //   engine_iter_abs_us = engine_t_start_us_absolute
    //                        + t_us_from_engine_start
    // hpx-server embeds the engine in the same process, so both share
    // one steady_clock epoch. Written only when LLAMA_HPX_DIAG_METRICS=1;
    // emitted in the engine_summary JSONL row. Default -1 = not
    // populated (diag off).
    int64_t              engine_t_start_us_absolute  = -1;

    engine_metrics       metrics;
};
