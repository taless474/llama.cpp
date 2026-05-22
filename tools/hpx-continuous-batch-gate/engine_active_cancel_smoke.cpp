// engine_active_cancel_smoke.cpp — M3c: active-request cancellation
// smoke for llama-hpx-engine. Proves engine::cancel_request(rid)
// cooperatively cancels a request that has already been admitted to
// a seq and is actively decoding. The engine task observes the
// cancel at the next iter-boundary via apply_queued_cancellations,
// sets seq.cancel_requested under the engine task, and the existing
// cancel_should_observe → cancel_and_fulfill pipeline closes the
// stream with reason=cancelled, clears KV, frees the seq into
// free_due_to_cancel_, and fulfills the promise with
// status=cancelled. n_decoded at cancellation is strictly between
// 1 and the requested decode_budget.
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
        "  Minimal active-cancel smoke for the llama-hpx-engine "
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
    fprintf(stdout, "HPX_ENGINE_ACTIVE_CANCEL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_ACTIVE_CANCEL_SMOKE: PASS\n");
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
    ctx_params.n_seq_max       = 1;
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
    opts.lib.n_seq_max            = 1;
    opts.lib.initial_idle_slots   = 1;
    opts.lib.keep_alive           = true;
    // Borrowed reference required to be non-null by the engine ctor.
    // budgets={} + initial_idle_slots=1 means the only active slot
    // is rebound from the idle pool when the submitted request is
    // admitted; the per-request prompt vector carried on the
    // arrival_msg drives admitted prefill. The borrow is kept alive
    // solely to satisfy the existing engine_options contract.
    opts.preload.prompt_tokens    = &shared_prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        // M3c: submit the request BEFORE starting the engine task.
        // submit() pushes onto the inbox under inbox_mtx_ and then
        // notifies inbox_cv_; the engine task drains the inbox on
        // its first iter and admits the request via free_idle_.
        submit_request req;
        req.request_id    = 42;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 256;
        req.want_stream   = true;
        submit_handle h = eng.submit_request(std::move(req));
        if (!h.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be engaged when "
                "want_stream=true");
        }

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // Drain the stream until the first token event arrives.
        // This proves the engine has admitted the request, run
        // prefill, decoded at least one token, and published it via
        // the channel — guaranteeing n_decoded >= 1 when the
        // cancellation observation pass next fires.
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
            // The stream closed before any token was published; the
            // request never decoded — that violates the M3c smoke
            // shape (cancel before first-token belongs to a
            // different test). Surface as a failure with the
            // observed close reason for diagnostics.
            return fail_and_cleanup(
                std::string(
                    "stream closed before first token; close=") +
                stream_close_label(observed_close));
        }

        // M3c: cancel the active, mid-decode request. cancel_request
        // pushes the rid onto cancel_inbox_ under inbox_mtx_ and
        // notifies inbox_cv_. The engine task drains cancel_inbox_
        // at the next iter boundary and apply_queued_cancellations
        // matches the rid against the live non-done active seq,
        // setting seq.cancel_requested. The existing cancellation
        // observation pass then calls cancel_and_fulfill, which
        // closes the stream with reason=cancelled, clears KV,
        // pushes the seq onto free_due_to_cancel_, and fulfills the
        // promise with status=cancelled.
        eng.cancel_request(42);

        // Continue draining the stream until the terminal closed
        // event arrives. Any further token events count toward
        // n_decoded — they are tokens decoded by the engine
        // BETWEEN our cancel_request() call and the engine's next
        // iter-boundary observation.
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
            return fail_and_cleanup("request status != cancelled");
        }
        if (r.n_decoded < 1) {
            return fail_and_cleanup("r.n_decoded < 1");
        }
        if (r.n_decoded >= 256) {
            return fail_and_cleanup("r.n_decoded >= 256");
        }
        if (r.cancel_observed_iter < 1) {
            return fail_and_cleanup("r.cancel_observed_iter < 1");
        }
        if (r.n_decoded_at_cancel != r.n_decoded) {
            return fail_and_cleanup(
                "r.n_decoded_at_cancel != r.n_decoded");
        }
        // publish_token fires once per token BEFORE n_decoded is
        // incremented, so the consumer-side token count must equal
        // r.n_decoded at terminal close.
        if (streamed_tokens != r.n_decoded) {
            return fail_and_cleanup(
                "streamed_tokens != r.n_decoded");
        }
        if (r.kv_cleared != true) {
            return fail_and_cleanup("r.kv_cleared != true");
        }
        if (r.pos_max_at_clear < 0) {
            return fail_and_cleanup("r.pos_max_at_clear < 0");
        }
        if (!seen_close) {
            return fail_and_cleanup("stream closed event missing");
        }
        if (observed_close != stream_close_reason::cancelled) {
            return fail_and_cleanup(
                "stream close reason != cancelled");
        }

        const engine_result & er = eng.result();
        if (er.cancel_active_observed != 1) {
            return fail_and_cleanup(
                "cancel_active_observed != 1");
        }
        if (er.cancel_active_not_supported != 0) {
            return fail_and_cleanup(
                "cancel_active_not_supported != 0");
        }
        if (er.cancelled_count != 1) {
            return fail_and_cleanup("cancelled_count != 1");
        }
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup("queued_cancelled != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d stream_close=%s\n",
            r.request_id, status_name(r.status), r.n_decoded,
            stream_close_label(observed_close));
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
