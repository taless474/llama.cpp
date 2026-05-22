// engine_handle_cancel_smoke.cpp — M5b: explicit-engine handle-cancel
// convenience smoke for llama-hpx-engine. Proves that
// `submit_handle::cancel(engine &)` is a behavior-equivalent forwarder
// onto the existing M5a `engine::cancel_request(const cancel_token &)`
// path along three sub-cases, all driven through a SINGLE engine
// instance to keep wall-clock and resource use modest:
//
//   Round 1 — queued cancel via h.cancel(eng):
//     Submit rid=10, budget=8, want_stream=true with no available
//     admission source (initial_idle_slots starts at 2 but is consumed
//     ONLY when admission can find a slot; we drain to zero ahead of
//     this round so the request stays in waiting_queue_consumable_).
//     Actually, to avoid coupling rounds, this smoke uses a separate
//     "queued" engine instance for round 1 (initial_idle_slots=0,
//     budgets={}) just like engine_queued_cancel_smoke. Then a second
//     instance with initial_idle_slots=2 drives rounds 2 and 3.
//
//   Round 2 — active cancel via h.cancel(eng):
//     Submit rid=20, budget=256, want_stream=true. Wait for first
//     streamed token (proves admit + prefill + decode), then call
//     h.cancel(eng). Expect status=cancelled with 1 <= n_decoded <
//     256, kv_cleared=true, stream close=cancelled,
//     cancel_active_observed=1.
//
//   Round 3 — stale-token rid-reuse via h_A.cancel(eng):
//     Submit rid=30 (A), budget=8, no stream; await completion;
//     capture token_A on h_A. Submit rid=30 (B), budget=32,
//     want_stream=true; wait for B's first streamed token; call
//     h_A.cancel(eng). The forwarder pushes token_A (rid=30,
//     epoch=e_A) onto cancel_token_inbox_. The engine resolves it
//     against live_epoch_by_rid_[30]=e_B != e_A, bumps
//     cancel_stale_epoch, and B continues uncancelled. Expect B
//     status=completed, n_decoded=32, stream close=completed,
//     cancel_stale_epoch=1 (cumulative).
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution and KV mutation live inside eng.run() running on the
// engine task each round spawns via hpx::async. No llama_decode,
// llama_batch_*, llama_memory_seq_*, llama_get_logits_ith, or
// common_batch_add call sites exist here. Every cancellation in
// this smoke is issued through h.cancel(eng) — the new M5b path.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  M5b handle-cancel smoke for the llama-hpx-engine "
        "library.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, smoke_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --model requires a value\n");
                return false;
            }
            out.model_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return false;
    }
    return true;
}

void emit_fail(const std::string & reason) {
    fprintf(stdout, "HPX_ENGINE_HANDLE_CANCEL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_HANDLE_CANCEL_SMOKE: PASS\n");
    fflush(stdout);
}

const char * stream_close_label(stream_close_reason r) {
    switch (r) {
        case stream_close_reason::completed: return "completed";
        case stream_close_reason::cancelled: return "cancelled";
        case stream_close_reason::error:     return "error";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    if (!hpx_runtime::start_once(/*os_threads=*/1)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model *   model = nullptr;
    llama_context * ctx_queued = nullptr;
    llama_context * ctx_active = nullptr;

    auto cleanup_and_stop = [&](int rc) -> int {
        if (ctx_active) llama_free(ctx_active);
        if (ctx_queued) llama_free(ctx_queued);
        if (model)      llama_model_free(model);
        llama_backend_free();
        hpx_runtime::stop();
        return rc;
    };

    auto fail_and_cleanup = [&](const std::string & reason) -> int {
        emit_fail(reason);
        return cleanup_and_stop(1);
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(),
                                       model_params);
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize a shared prompt once; both engines reuse it.
    // The model is loaded but we need a context to tokenize. Use the
    // first (queued) context for tokenization, then build the active
    // context afterwards.
    llama_context_params ctx_queued_params = llama_context_default_params();
    ctx_queued_params.n_ctx           = 2048;
    ctx_queued_params.n_batch         = 64;
    ctx_queued_params.n_seq_max       = 1;
    ctx_queued_params.n_threads       = 2;
    ctx_queued_params.n_threads_batch = 2;
    ctx_queued = llama_init_from_model(model, ctx_queued_params);
    if (ctx_queued == nullptr) {
        return fail_and_cleanup("ctx_queued create failed");
    }

    std::vector<llama_token> shared_prompt = common_tokenize(
        ctx_queued, "Once upon a time", /*add_special=*/true,
        /*parse_special=*/true);
    if (shared_prompt.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }

    const int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(shared_prompt.size()), 1);

    std::vector<waiting_request> empty_waiting;

    // ----------------------------------------------------------------
    // Round 1 — queued cancel via h.cancel(eng).
    //
    // Mirrors engine_queued_cancel_smoke: no idle slots, empty budgets,
    // empty waiting queue. The request is drained from the inbox into
    // waiting_queue_consumable_ and never admitted. h.cancel(eng) goes
    // through the token-aware path; apply_queued_cancellations matches
    // the rid against the waiting queue entry and resolves it with
    // status=cancelled.
    // ----------------------------------------------------------------
    try {
        engine_options opts;
        opts.lib.ctx                  = ctx_queued;
        opts.lib.vocab                = vocab;
        opts.lib.n_vocab              = n_vocab;
        opts.lib.batch_capacity       = batch_capacity;
        opts.lib.n_seq_max            = 1;
        opts.lib.initial_idle_slots   = 0;
        opts.lib.keep_alive           = true;
        opts.preload.prompt_tokens    = &shared_prompt;
        opts.preload.budgets          = {};
        opts.preload.waiting_queue    = &empty_waiting;
        opts.preload.reuse_completed  = false;
        opts.preload.stream_all       = false;
        opts.gate_test.cancel_after     = -1;
        opts.gate_test.max_decode_iters = 0;

        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        submit_request req;
        req.request_id    = 10;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 8;
        req.want_stream   = true;
        submit_handle h = eng.submit_request(std::move(req));
        if (!h.stream.has_value()) {
            return fail_and_cleanup(
                "round1: submit_handle.stream must be engaged when "
                "want_stream=true");
        }
        if (h.token.epoch == 0 || h.token.request_id != 10) {
            return fail_and_cleanup(
                "round1: token not engine-issued");
        }

        // M5b cancel path: handle-rooted forwarder onto the M5a
        // token-aware engine API. Equivalent to
        // eng.cancel_request(h.token).
        h.cancel(eng);

        request_result r = h.result.get();

        stream_close_reason observed_close = stream_close_reason::completed;
        int32_t streamed_tokens = 0;
        bool    seen_close      = false;
        try {
            while (true) {
                hpx::future<token_stream_event> fev =
                    h.stream->get();
                token_stream_event ev = fev.get();
                if (ev.kind == stream_event_kind::closed) {
                    observed_close = ev.close_reason;
                    seen_close = true;
                    break;
                }
                streamed_tokens++;
            }
        } catch (...) {
            // Channel closed by engine; treat as terminal.
        }

        eng.request_shutdown();
        engine_fut.get();

        if (r.status != request_status::cancelled) {
            return fail_and_cleanup(
                "round1: status != cancelled");
        }
        if (r.n_decoded != 0) {
            return fail_and_cleanup("round1: n_decoded != 0");
        }
        if (r.admitted_at_iter != -1) {
            return fail_and_cleanup(
                "round1: admitted_at_iter != -1");
        }
        if (r.kv_cleared != false) {
            return fail_and_cleanup("round1: kv_cleared != false");
        }
        if (!seen_close) {
            return fail_and_cleanup(
                "round1: stream closed event missing");
        }
        if (observed_close != stream_close_reason::cancelled) {
            return fail_and_cleanup(
                "round1: stream close reason != cancelled");
        }
        if (streamed_tokens != 0) {
            return fail_and_cleanup(
                "round1: stream emitted tokens before close");
        }

        const engine_result & er = eng.result();
        // M5b: the forwarder pushes the (rid, epoch) token onto
        // cancel_token_inbox_, so the legacy rid-only counters
        // queued_cancelled / cancel_request_calls are bumped via
        // the token-aware resolution path that delegates to the
        // same fulfill_queued_cancelled body. Both counters must
        // be 1 for a queued-token resolution.
        if (er.queued_cancelled != 1) {
            return fail_and_cleanup(
                "round1: queued_cancelled != 1");
        }
        if (er.cancel_request_calls != 1) {
            return fail_and_cleanup(
                "round1: cancel_request_calls != 1");
        }
        if (er.cancel_active_not_supported != 0) {
            return fail_and_cleanup(
                "round1: cancel_active_not_supported != 0");
        }
        if (er.cancel_unknown_request_id != 0) {
            return fail_and_cleanup(
                "round1: cancel_unknown_request_id != 0");
        }
        if (er.cancel_stale_epoch != 0) {
            return fail_and_cleanup(
                "round1: cancel_stale_epoch != 0");
        }
        if (er.admitted_count != 0) {
            return fail_and_cleanup("round1: admitted_count != 0");
        }
        if (er.streams_opened != 1) {
            return fail_and_cleanup("round1: streams_opened != 1");
        }
        if (er.streams_closed_cancelled != 1) {
            return fail_and_cleanup(
                "round1: streams_closed_cancelled != 1");
        }
        if (er.streams_closed_completed != 0) {
            return fail_and_cleanup(
                "round1: streams_closed_completed != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                "round1: residual_kv_ok != true");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "round1: engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "round1 queued: request_id=%d status=%s n_decoded=%d "
            "stream_close=%s\n",
            r.request_id, status_name(r.status), r.n_decoded,
            stream_close_label(observed_close));
        fflush(stdout);
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("round1 exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("round1 unknown exception");
    }

    // ----------------------------------------------------------------
    // Build a fresh context for rounds 2 and 3 with n_seq_max=2 so a
    // completed request and a freshly submitted same-rid request can
    // coexist long enough to test stale-token resolution.
    // ----------------------------------------------------------------
    llama_context_params ctx_active_params = llama_context_default_params();
    ctx_active_params.n_ctx           = 2048;
    ctx_active_params.n_batch         = 64;
    ctx_active_params.n_seq_max       = 2;
    ctx_active_params.n_threads       = 2;
    ctx_active_params.n_threads_batch = 2;
    ctx_active = llama_init_from_model(model, ctx_active_params);
    if (ctx_active == nullptr) {
        return fail_and_cleanup("ctx_active create failed");
    }

    // ----------------------------------------------------------------
    // Round 2 — active cancel via h.cancel(eng).
    //
    // Mirrors engine_active_cancel_smoke: budgets={}, idle_slots=1,
    // keep_alive=true. Submit before starting the engine. Wait for
    // first streamed token (admit + prefill + decode visible). Then
    // h.cancel(eng). Expect status=cancelled, 1 <= n_decoded < 256.
    // Uses ctx_active with n_seq_max=2 (only one of the two slots is
    // exercised here; the second one is reused in round 3).
    // ----------------------------------------------------------------
    try {
        engine_options opts;
        opts.lib.ctx                  = ctx_active;
        opts.lib.vocab                = vocab;
        opts.lib.n_vocab              = n_vocab;
        opts.lib.batch_capacity       = batch_capacity;
        opts.lib.n_seq_max            = 2;
        opts.lib.initial_idle_slots   = 1;
        opts.lib.keep_alive           = true;
        opts.preload.prompt_tokens    = &shared_prompt;
        opts.preload.budgets          = {};
        opts.preload.waiting_queue    = &empty_waiting;
        opts.preload.reuse_completed  = false;
        opts.preload.stream_all       = false;
        opts.gate_test.cancel_after     = -1;
        opts.gate_test.max_decode_iters = 0;

        engine eng(std::move(opts));

        submit_request req;
        req.request_id    = 20;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 256;
        req.want_stream   = true;
        submit_handle h = eng.submit_request(std::move(req));
        if (!h.stream.has_value()) {
            return fail_and_cleanup(
                "round2: submit_handle.stream must be engaged");
        }
        if (h.token.epoch == 0 || h.token.request_id != 20) {
            return fail_and_cleanup(
                "round2: token not engine-issued");
        }

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        stream_close_reason observed_close = stream_close_reason::completed;
        int32_t streamed_tokens = 0;
        bool    seen_close      = false;
        bool    seen_first_token = false;
        try {
            hpx::future<token_stream_event> first_fev = h.stream->get();
            token_stream_event first_ev = first_fev.get();
            if (first_ev.kind == stream_event_kind::token) {
                streamed_tokens++;
                seen_first_token = true;
            } else {
                observed_close = first_ev.close_reason;
                seen_close = true;
            }
        } catch (...) {
            // Channel closed unexpectedly; treat as terminal.
        }

        if (!seen_first_token) {
            return fail_and_cleanup(
                std::string(
                    "round2: stream closed before first token; close=")
                + stream_close_label(observed_close));
        }

        // M5b cancel path: handle-rooted forwarder. Engine task
        // observes at the next iter boundary.
        h.cancel(eng);

        try {
            while (true) {
                hpx::future<token_stream_event> fev =
                    h.stream->get();
                token_stream_event ev = fev.get();
                if (ev.kind == stream_event_kind::closed) {
                    observed_close = ev.close_reason;
                    seen_close = true;
                    break;
                }
                streamed_tokens++;
            }
        } catch (...) {
            // Channel closed by engine; treat as terminal.
        }

        eng.request_shutdown();
        request_result r = h.result.get();
        engine_fut.get();

        if (r.status != request_status::cancelled) {
            return fail_and_cleanup("round2: status != cancelled");
        }
        if (r.n_decoded < 1) {
            return fail_and_cleanup("round2: n_decoded < 1");
        }
        if (r.n_decoded >= 256) {
            return fail_and_cleanup("round2: n_decoded >= 256");
        }
        if (r.cancel_observed_iter < 1) {
            return fail_and_cleanup(
                "round2: cancel_observed_iter < 1");
        }
        if (r.n_decoded_at_cancel != r.n_decoded) {
            return fail_and_cleanup(
                "round2: n_decoded_at_cancel != n_decoded");
        }
        if (streamed_tokens != r.n_decoded) {
            return fail_and_cleanup(
                "round2: streamed_tokens != n_decoded");
        }
        if (r.kv_cleared != true) {
            return fail_and_cleanup("round2: kv_cleared != true");
        }
        if (r.pos_max_at_clear < 0) {
            return fail_and_cleanup(
                "round2: pos_max_at_clear < 0");
        }
        if (!seen_close) {
            return fail_and_cleanup(
                "round2: stream closed event missing");
        }
        if (observed_close != stream_close_reason::cancelled) {
            return fail_and_cleanup(
                "round2: stream close reason != cancelled");
        }

        const engine_result & er = eng.result();
        if (er.cancel_active_observed != 1) {
            return fail_and_cleanup(
                "round2: cancel_active_observed != 1");
        }
        if (er.cancel_active_not_supported != 0) {
            return fail_and_cleanup(
                "round2: cancel_active_not_supported != 0");
        }
        if (er.cancelled_count != 1) {
            return fail_and_cleanup(
                "round2: cancelled_count != 1");
        }
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup(
                "round2: queued_cancelled != 0");
        }
        if (er.cancel_request_calls != 1) {
            return fail_and_cleanup(
                "round2: cancel_request_calls != 1");
        }
        if (er.cancel_stale_epoch != 0) {
            return fail_and_cleanup(
                "round2: cancel_stale_epoch != 0");
        }
        if (er.cancel_unknown_request_id != 0) {
            return fail_and_cleanup(
                "round2: cancel_unknown_request_id != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                "round2: residual_kv_ok != true");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "round2: engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "round2 active: request_id=%d status=%s n_decoded=%d "
            "stream_close=%s\n",
            r.request_id, status_name(r.status), r.n_decoded,
            stream_close_label(observed_close));
        fflush(stdout);
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("round2 exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("round2 unknown exception");
    }

    // ----------------------------------------------------------------
    // Round 3 — stale-token rid-reuse via h_A.cancel(eng).
    //
    // Mirrors engine_stale_token_smoke. Uses a fresh engine on the
    // shared ctx_active (note: the previous run's KV was cleared and
    // residual_kv_ok already verified above). n_seq_max=2 and
    // initial_idle_slots=2 so A completes, then B's same-rid
    // submission gets a fresh slot from the other idle entry. The
    // stale-token cancel is issued via h_A.cancel(eng), exercising
    // the M5b forwarder on a captured handle whose request has
    // already completed.
    //
    // Note on llama_context reuse across two engine instances:
    // engine_stale_token_smoke also reuses one ctx across rounds via
    // its repeat path. cancel_and_fulfill / finalize_and_fulfill
    // clear KV per seq, and residual_kv_ok is asserted at run end.
    // ----------------------------------------------------------------
    try {
        engine_options opts;
        opts.lib.ctx                  = ctx_active;
        opts.lib.vocab                = vocab;
        opts.lib.n_vocab              = n_vocab;
        opts.lib.batch_capacity       = batch_capacity;
        opts.lib.n_seq_max            = 2;
        opts.lib.initial_idle_slots   = 2;
        opts.lib.keep_alive           = true;
        opts.preload.prompt_tokens    = nullptr;
        opts.preload.budgets          = {};
        opts.preload.waiting_queue    = &empty_waiting;
        opts.preload.reuse_completed  = false;
        opts.preload.stream_all       = false;
        opts.gate_test.cancel_after     = -1;
        opts.gate_test.max_decode_iters = 0;

        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // Phase A — submit, complete, capture token_A.
        submit_request req_A;
        req_A.request_id    = 30;
        req_A.prompt_tokens = shared_prompt;
        req_A.decode_budget = 8;
        req_A.want_stream   = false;
        submit_handle h_A = eng.submit_request(std::move(req_A));
        const cancel_token token_A = h_A.token;
        if (token_A.request_id != 30 || token_A.epoch == 0) {
            return fail_and_cleanup(
                "round3: token_A not engine-issued");
        }

        request_result r_A = h_A.result.get();
        if (r_A.status != request_status::completed) {
            return fail_and_cleanup(
                std::string("round3: A status != completed (got ")
                + status_name(r_A.status) + ")");
        }
        if (r_A.n_decoded != 8) {
            return fail_and_cleanup("round3: A.n_decoded != 8");
        }

        // Phase B — same rid, fresh epoch, streamed.
        submit_request req_B;
        req_B.request_id    = 30;        // intentional reuse
        req_B.prompt_tokens = shared_prompt;
        req_B.decode_budget = 32;
        req_B.want_stream   = true;
        submit_handle h_B = eng.submit_request(std::move(req_B));
        const cancel_token token_B = h_B.token;
        if (token_B.request_id != 30 || token_B.epoch == 0) {
            return fail_and_cleanup(
                "round3: token_B not engine-issued");
        }
        if (token_B.epoch == token_A.epoch) {
            return fail_and_cleanup(
                "round3: token_A.epoch == token_B.epoch");
        }
        if (!h_B.stream.has_value()) {
            return fail_and_cleanup(
                "round3: h_B.stream must be engaged");
        }

        // Wait for B's first streamed token — proves
        // live_epoch_by_rid_[30] = token_B.epoch is set on the engine
        // task.
        int32_t streamed_tokens = 0;
        bool seen_first_token = false;
        try {
            hpx::future<token_stream_event> first_fev =
                h_B.stream->get();
            token_stream_event first_ev = first_fev.get();
            if (first_ev.kind == stream_event_kind::token) {
                streamed_tokens++;
                seen_first_token = true;
            } else {
                return fail_and_cleanup(
                    std::string("round3: B stream closed before "
                                "first token; close=")
                    + stream_close_label(first_ev.close_reason));
            }
        } catch (...) {
            return fail_and_cleanup(
                "round3: B stream first get() threw");
        }
        if (!seen_first_token) {
            return fail_and_cleanup(
                "round3: did not observe B's first streamed token");
        }

        // M5b cancel path: fire the stale cancel on the OLD handle's
        // captured token via the forwarder. h_A.cancel(eng) ≡
        // eng.cancel_request(token_A). The engine resolves
        // (rid=30, epoch=e_A) against live_epoch_by_rid_[30]=e_B,
        // observes the mismatch, bumps cancel_stale_epoch, and
        // leaves B running.
        h_A.cancel(eng);

        stream_close_reason observed_close = stream_close_reason::error;
        bool seen_close = false;
        try {
            while (true) {
                hpx::future<token_stream_event> fev =
                    h_B.stream->get();
                token_stream_event ev = fev.get();
                if (ev.kind == stream_event_kind::closed) {
                    observed_close = ev.close_reason;
                    seen_close = true;
                    break;
                }
                streamed_tokens++;
            }
        } catch (...) {
            // Channel closed by engine; treat as terminal.
        }
        if (!seen_close) {
            return fail_and_cleanup(
                "round3: B stream did not emit a terminal closed event");
        }

        request_result r_B = h_B.result.get();
        eng.request_shutdown();
        engine_fut.get();

        if (r_B.status != request_status::completed) {
            return fail_and_cleanup(
                std::string("round3: B status != completed (got ")
                + status_name(r_B.status)
                + ") — stale forwarder must NOT cancel a live "
                  "same-rid submission");
        }
        if (r_B.n_decoded != 32) {
            return fail_and_cleanup("round3: B.n_decoded != 32");
        }
        if (observed_close != stream_close_reason::completed) {
            return fail_and_cleanup(
                std::string("round3: B stream close != completed (got ")
                + stream_close_label(observed_close) + ")");
        }
        if (streamed_tokens != r_B.n_decoded) {
            return fail_and_cleanup(
                "round3: streamed_tokens != B.n_decoded");
        }

        const engine_result & er = eng.result();
        if (er.cancel_stale_epoch != 1) {
            return fail_and_cleanup(
                std::string("round3: cancel_stale_epoch != 1 (got ")
                + std::to_string(er.cancel_stale_epoch) + ")");
        }
        if (er.cancel_unknown_request_id != 0) {
            return fail_and_cleanup(
                "round3: cancel_unknown_request_id != 0");
        }
        if (er.cancel_active_observed != 0) {
            return fail_and_cleanup(
                "round3: cancel_active_observed != 0 (stale token must "
                "not trigger active cancellation)");
        }
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup(
                "round3: queued_cancelled != 0");
        }
        if (er.cancelled_count != 0) {
            return fail_and_cleanup(
                "round3: cancelled_count != 0");
        }
        if (er.cancel_request_calls != 1) {
            return fail_and_cleanup(
                std::string("round3: cancel_request_calls != 1 (got ")
                + std::to_string(er.cancel_request_calls) + ")");
        }
        if (er.cancel_request_duplicates != 0) {
            return fail_and_cleanup(
                "round3: cancel_request_duplicates != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                "round3: residual_kv_ok != true");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "round3: engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "round3 stale: A rid=%d epoch=%llu status=%s n_decoded=%d\n"
            "round3 stale: B rid=%d epoch=%llu status=%s n_decoded=%d "
            "stream_close=%s\n"
            "round3 stale: cancel_stale_epoch=%d "
            "cancel_request_calls=%d\n",
            token_A.request_id,
            static_cast<unsigned long long>(token_A.epoch),
            status_name(r_A.status), r_A.n_decoded,
            token_B.request_id,
            static_cast<unsigned long long>(token_B.epoch),
            status_name(r_B.status), r_B.n_decoded,
            stream_close_label(observed_close),
            er.cancel_stale_epoch,
            er.cancel_request_calls);
        fflush(stdout);
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("round3 exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("round3 unknown exception");
    }

    emit_pass();
    return cleanup_and_stop(0);
}
