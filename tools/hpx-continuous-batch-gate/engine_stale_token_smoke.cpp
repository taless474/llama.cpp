// engine_stale_token_smoke.cpp — M5a: epoch-aware identity smoke for
// llama-hpx-engine. Proves that a `cancel_token` captured for a
// completed request_id does NOT cancel a later submission that reuses
// the same request_id: the engine matches the token's (rid, epoch)
// pair against `live_epoch_by_rid_`, observes the epoch mismatch,
// bumps `engine_result::cancel_stale_epoch`, and leaves the live
// request untouched. The second submission completes normally.
//
// Shape (per the M5a plan, approved with the simpler smoke path):
//   1. Submit request A with request_id=42, no stream, decode_budget=8.
//      Capture token_A from h_A.token BEFORE awaiting the result.
//   2. Await h_A.result.get(); assert status=completed and n_decoded=8.
//   3. Submit request B with request_id=42 (REUSE), want_stream=true,
//      decode_budget=32. Capture token_B.
//   4. Assert token_A.epoch != token_B.epoch (engine-issued monotonic).
//   5. Wait for B's first streamed token. At this point B has been
//      drained, admitted, prefilled, and decoded at least one token —
//      `live_epoch_by_rid_[42] == token_B.epoch` is set on the engine
//      task.
//   6. Call eng.cancel_request(token_A). This is the stale-token
//      cancel: the engine drains the token onto cancel_token_inbox_,
//      folds it into cancelled_tokens_, and at the next iter the
//      apply_queued_cancellations() token-aware pass observes
//      `live[42]=token_B.epoch ≠ token_A.epoch` → bump cancel_stale_epoch
//      and erase. B is NOT cancelled.
//   7. Drain B's stream to terminal `closed`. Expect close_reason=
//      completed and exactly decode_budget tokens streamed.
//   8. Await h_B.result.get(); assert status=completed and
//      n_decoded=32.
//   9. eng.request_shutdown(); engine_fut.get();
//  10. Assert `eng.result().cancel_stale_epoch == 1`.
//  11. Assert `eng.result().cancel_unknown_request_id == 0` (the rid
//      WAS live at resolution time; the stale-epoch counter is the
//      one that should fire, not unknown_request).
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend
// / model / context setup, tokenization, and final cleanup. All
// llama execution and KV mutation live inside eng.run() running on
// the single hpx::async task this file spawns. No llama_decode,
// llama_batch_*, llama_memory_seq_*, llama_get_logits_ith, or
// common_batch_add call sites exist here.

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
        "  M5a stale-token smoke for the llama-hpx-engine "
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
    fprintf(stdout, "HPX_ENGINE_STALE_TOKEN_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_STALE_TOKEN_SMOKE: PASS\n");
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
    llama_context * ctx   = nullptr;

    auto cleanup_and_stop = [&](int rc) -> int {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
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

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> shared_prompt = common_tokenize(
        ctx, "Once upon a time", /*add_special=*/true,
        /*parse_special=*/true);
    if (shared_prompt.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }

    const int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(shared_prompt.size()), 1);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 2;
    // M5a: two idle slots so request A and request B (same rid=42,
    // serialized) each get a fresh slot from free_idle_ without
    // relying on completion-freed reuse semantics.
    opts.lib.initial_idle_slots   = 2;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        // Spawn the engine task BEFORE any submission. With no
        // preloaded actives and an empty inbox, the engine drives
        // straight into keep-alive idle-wait until the first
        // submit_request wakes it.
        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // ---- Phase A: submit, complete, capture token_A ------------
        submit_request req_A;
        req_A.request_id    = 42;
        req_A.prompt_tokens = shared_prompt;
        req_A.decode_budget = 8;
        req_A.want_stream   = false;
        submit_handle h_A     = eng.submit_request(std::move(req_A));
        const cancel_token token_A = h_A.token;
        if (token_A.request_id != 42) {
            return fail_and_cleanup(
                "token_A.request_id != 42");
        }
        if (token_A.epoch == 0) {
            return fail_and_cleanup(
                "token_A.epoch == 0 (engine-issued epochs start at 1)");
        }
        if (h_A.stream.has_value()) {
            return fail_and_cleanup(
                "h_A.stream must be nullopt when want_stream=false");
        }

        request_result r_A = h_A.result.get();
        if (r_A.status != request_status::completed) {
            return fail_and_cleanup(
                std::string("A status != completed (got ")
                + status_name(r_A.status) + ")");
        }
        if (r_A.n_decoded != 8) {
            return fail_and_cleanup(
                "A.n_decoded != 8");
        }

        // ---- Phase B: same rid, fresh epoch, streamed --------------
        submit_request req_B;
        req_B.request_id    = 42;        // intentional reuse
        req_B.prompt_tokens = shared_prompt;
        req_B.decode_budget = 32;
        req_B.want_stream   = true;
        submit_handle h_B     = eng.submit_request(std::move(req_B));
        const cancel_token token_B = h_B.token;
        if (token_B.request_id != 42) {
            return fail_and_cleanup(
                "token_B.request_id != 42");
        }
        if (token_B.epoch == 0) {
            return fail_and_cleanup(
                "token_B.epoch == 0");
        }
        if (token_B.epoch == token_A.epoch) {
            return fail_and_cleanup(
                "token_A.epoch == token_B.epoch "
                "(engine-issued epochs must differ across submissions)");
        }
        if (!h_B.stream.has_value()) {
            return fail_and_cleanup(
                "h_B.stream must be engaged when want_stream=true");
        }

        // Wait for B's first streamed token. The first token event
        // (kind=token, not kind=closed) confirms B has been drained,
        // admitted, prefilled, and produced at least one decoded
        // token — so the engine task has already set
        // `live_epoch_by_rid_[42] = token_B.epoch`.
        int32_t streamed_tokens = 0;
        bool    seen_first_token = false;
        try {
            hpx::future<token_stream_event> first_fev =
                h_B.stream->get();
            token_stream_event first_ev = first_fev.get();
            if (first_ev.kind == stream_event_kind::token) {
                streamed_tokens++;
                seen_first_token = true;
            } else {
                return fail_and_cleanup(
                    std::string("B stream closed before first token; "
                                "close=")
                    + stream_close_label(first_ev.close_reason));
            }
        } catch (...) {
            return fail_and_cleanup(
                "B stream first get() threw");
        }
        if (!seen_first_token) {
            return fail_and_cleanup(
                "did not observe B's first streamed token");
        }

        // ---- Fire the stale cancel on the OLD token ----------------
        // token_A's (rid=42, epoch=token_A.epoch) pair is stale: the
        // engine's live map currently points at (42, token_B.epoch).
        // Expected resolution path:
        //   drain_cancel_inbox -> cancelled_tokens_ += (42, e_A)
        //   apply_queued_cancellations (token-aware pass):
        //     live[42] = e_B != e_A -> cancel_stale_epoch++ + erase
        //   B continues to completion uncancelled.
        eng.cancel_request(token_A);

        // ---- Continue draining B's stream until terminal close -----
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
            // Channel closed by engine; treat as terminal but
            // mark the close reason as error if we never saw a
            // close event.
        }
        if (!seen_close) {
            return fail_and_cleanup(
                "B stream did not emit a terminal closed event");
        }

        request_result r_B = h_B.result.get();

        eng.request_shutdown();
        engine_fut.get();

        // ---- Assertions: B completes normally, stale-epoch fires ---
        if (r_B.status != request_status::completed) {
            return fail_and_cleanup(
                std::string("B status != completed (got ")
                + status_name(r_B.status)
                + ") — stale token must NOT cancel a live "
                  "same-rid submission");
        }
        if (r_B.n_decoded != 32) {
            return fail_and_cleanup(
                "B.n_decoded != 32");
        }
        if (observed_close != stream_close_reason::completed) {
            return fail_and_cleanup(
                std::string("B stream close != completed (got ")
                + stream_close_label(observed_close) + ")");
        }
        if (streamed_tokens != r_B.n_decoded) {
            return fail_and_cleanup(
                "streamed_tokens != B.n_decoded");
        }

        const engine_result & er = eng.result();
        if (er.cancel_stale_epoch != 1) {
            return fail_and_cleanup(
                std::string("cancel_stale_epoch != 1 (got ")
                + std::to_string(er.cancel_stale_epoch) + ")");
        }
        if (er.cancel_unknown_request_id != 0) {
            return fail_and_cleanup(
                "cancel_unknown_request_id != 0 (the rid WAS live "
                "at resolution time)");
        }
        if (er.cancel_active_observed != 0) {
            return fail_and_cleanup(
                "cancel_active_observed != 0 (stale token must not "
                "trigger active cancellation)");
        }
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup(
                "queued_cancelled != 0 (stale token must not "
                "trigger queued cancellation)");
        }
        if (er.cancelled_count != 0) {
            return fail_and_cleanup(
                "cancelled_count != 0 (no request should be in the "
                "cancelled outcome)");
        }
        if (er.cancel_request_calls != 1) {
            return fail_and_cleanup(
                std::string("cancel_request_calls != 1 (got ")
                + std::to_string(er.cancel_request_calls) + ")");
        }
        if (er.cancel_request_duplicates != 0) {
            return fail_and_cleanup(
                "cancel_request_duplicates != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "A: request_id=%d epoch=%llu status=%s n_decoded=%d\n"
            "B: request_id=%d epoch=%llu status=%s n_decoded=%d "
            "stream_close=%s\n"
            "cancel_stale_epoch=%d cancel_unknown_request_id=%d "
            "cancel_request_calls=%d\n",
            token_A.request_id,
            static_cast<unsigned long long>(token_A.epoch),
            status_name(r_A.status), r_A.n_decoded,
            token_B.request_id,
            static_cast<unsigned long long>(token_B.epoch),
            status_name(r_B.status), r_B.n_decoded,
            stream_close_label(observed_close),
            er.cancel_stale_epoch,
            er.cancel_unknown_request_id,
            er.cancel_request_calls);
        fflush(stdout);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
