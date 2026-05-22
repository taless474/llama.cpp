// engine_queued_cancel_smoke.cpp — M3b: queued-before-admission
// cancellation smoke for llama-hpx-engine. Proves
// engine::cancel_request(rid) resolves a request that has been
// drained into waiting_queue_consumable_ but never admitted — the
// engine wakes from its keep-alive idle-wait, matches the rid at
// the outer-loop tail or at the iter-top apply_queued_cancellations
// step, fulfills the promise with status=cancelled, closes the
// caller-supplied stream with reason=cancelled, and exits cleanly
// when request_shutdown() is observed on the drained engine.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution and KV mutation live inside eng.run() running on the
// single hpx::async task this file spawns. No llama_decode,
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
        "  Minimal queued-cancel smoke for the llama-hpx-engine "
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
    fprintf(stdout, "HPX_ENGINE_QUEUED_CANCEL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_QUEUED_CANCEL_SMOKE: PASS\n");
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
    opts.lib.initial_idle_slots   = 0;
    opts.lib.keep_alive           = true;
    // Borrowed reference required to be non-null by the engine ctor.
    // budgets={} + initial_idle_slots=0 means seqs_ is empty; the
    // prefill loop reads nothing and admit_one is never called. The
    // borrow is kept alive only to satisfy the existing contract.
    opts.preload.prompt_tokens    = &shared_prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        // M3b: start the engine BEFORE any submission. With no
        // initial actives, no idle slots, and an empty inbox/queue,
        // the engine drives straight into the keep-alive outer-loop
        // idle-wait. The submitted request will be drained into
        // waiting_queue_consumable_ at the next iter; no admission
        // source exists so the inner-while breaks with the request
        // still queued. The outer-loop tail's
        // drain_cancel_inbox + apply_queued_cancellations is the
        // only place where cancellation is observed.
        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        submit_request req;
        req.request_id    = 42;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 8;
        req.want_stream   = true;
        submit_handle h = eng.submit_request(std::move(req));
        if (!h.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be engaged when "
                "want_stream=true");
        }

        // Cancel the request. With initial_idle_slots=0 and
        // budgets={}, no admission source ever appears, so the
        // request is guaranteed to remain queued when this fires.
        eng.cancel_request(42);

        request_result r = h.result.get();

        // Drain the stream until the terminal closed event arrives.
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
            return fail_and_cleanup("request status != cancelled");
        }
        if (r.n_decoded != 0) {
            return fail_and_cleanup("r.n_decoded != 0");
        }
        if (r.admitted_at_iter != -1) {
            return fail_and_cleanup("r.admitted_at_iter != -1");
        }
        if (r.kv_cleared != false) {
            return fail_and_cleanup("r.kv_cleared != false");
        }
        if (!seen_close) {
            return fail_and_cleanup("stream closed event missing");
        }
        if (observed_close != stream_close_reason::cancelled) {
            return fail_and_cleanup(
                "stream close reason != cancelled");
        }
        if (streamed_tokens != 0) {
            return fail_and_cleanup(
                "stream emitted tokens before close");
        }

        const engine_result & er = eng.result();
        if (er.queued_cancelled != 1) {
            return fail_and_cleanup("queued_cancelled != 1");
        }
        if (er.cancel_request_calls != 1) {
            return fail_and_cleanup("cancel_request_calls != 1");
        }
        if (er.cancel_active_not_supported != 0) {
            return fail_and_cleanup(
                "cancel_active_not_supported != 0");
        }
        if (er.cancel_unknown_request_id != 0) {
            return fail_and_cleanup(
                "cancel_unknown_request_id != 0");
        }
        if (er.admitted_count != 0) {
            return fail_and_cleanup("admitted_count != 0");
        }
        if (er.external_admitted_count != 0) {
            return fail_and_cleanup(
                "external_admitted_count != 0");
        }
        if (er.streams_opened != 1) {
            return fail_and_cleanup("streams_opened != 1");
        }
        if (er.streams_closed_cancelled != 1) {
            return fail_and_cleanup(
                "streams_closed_cancelled != 1");
        }
        if (er.streams_closed_completed != 0) {
            return fail_and_cleanup(
                "streams_closed_completed != 0");
        }
        if (er.streams_closed_error != 0) {
            return fail_and_cleanup("streams_closed_error != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "engine_shutdown_observed != 1");
        }

        fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d "
            "admitted_at_iter=%d stream_close=%s\n",
            r.request_id, status_name(r.status), r.n_decoded,
            r.admitted_at_iter,
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
