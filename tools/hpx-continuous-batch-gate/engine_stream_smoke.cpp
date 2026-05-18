// engine_stream_smoke.cpp — M2g: per-request streaming smoke for
// llama-hpx-engine. Proves submit_request(want_stream=true) returns a
// real token_stream_receiver through submit_handle.stream, without any
// gate-style admitted_stream_handoffs_ drain.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution and KV mutation live inside eng.run() running on the single
// hpx::async task this file spawns. No llama_decode, llama_batch_*,
// llama_memory_seq_*, llama_get_logits_ith, or common_batch_add call
// sites exist here.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "token_hash.h"
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
        "  Per-request streaming smoke for the llama-hpx-engine "
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
    fprintf(stdout, "HPX_ENGINE_STREAM_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_STREAM_SMOKE: PASS\n");
    fflush(stdout);
}

void print_result(const request_result & r,
                  size_t                 streamed_tokens,
                  uint64_t               streamed_hash) {
    fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d hash=0x%016llx "
            "admission_source=%s streamed_tokens=%zu "
            "streamed_hash=0x%016llx\n",
            r.request_id, status_name(r.status), r.n_decoded,
            static_cast<unsigned long long>(r.hash),
            admission_source_name(r.admission_src),
            streamed_tokens,
            static_cast<unsigned long long>(streamed_hash));
    fflush(stdout);
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
    opts.ctx                = ctx;
    opts.vocab              = vocab;
    opts.n_vocab            = n_vocab;
    opts.prompt_tokens      = &shared_prompt;
    opts.budgets            = {};
    opts.batch_capacity     = batch_capacity;
    opts.cancel_after       = -1;
    opts.n_seq_max          = 1;
    opts.waiting_queue      = &empty_waiting;
    opts.reuse_completed    = false;
    opts.max_decode_iters   = 0;
    opts.stream_all         = false;
    opts.initial_idle_slots = 1;

    try {
        engine eng(std::move(opts));

        submit_request req;
        req.request_id    = 42;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 8;
        req.want_stream   = true;
        submit_handle h = eng.submit_request(std::move(req));
        if (!h.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be present when "
                "want_stream=true");
        }

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::vector<int32_t> streamed_tokens;
        stream_close_reason  observed_close =
            stream_close_reason::error;
        bool                 closed_seen = false;
        while (!closed_seen) {
            const token_stream_event ev =
                h.stream->get(hpx::launch::sync);
            switch (ev.kind) {
                case stream_event_kind::token:
                    streamed_tokens.push_back(ev.token_id);
                    break;
                case stream_event_kind::closed:
                    observed_close = ev.close_reason;
                    closed_seen    = true;
                    break;
            }
        }

        engine_fut.get();
        request_result r = h.result.get();

        uint64_t hs = token_hash::k_token_hash_init;
        for (auto t : streamed_tokens) {
            hs = token_hash::fold_token_hash(hs, t);
        }
        const uint64_t streamed_hash =
            streamed_tokens.empty()
                ? token_hash::k_token_hash_empty
                : hs;

        if (observed_close != stream_close_reason::completed) {
            return fail_and_cleanup(
                "stream close reason != completed");
        }
        if (streamed_tokens.size() != 8) {
            return fail_and_cleanup(
                "streamed_tokens.size() != 8");
        }
        if (streamed_hash != r.hash) {
            return fail_and_cleanup(
                "streamed_hash != request_result.hash");
        }
        if (r.status != request_status::completed) {
            return fail_and_cleanup("request status != completed");
        }
        if (r.n_decoded != 8) {
            return fail_and_cleanup("n_decoded != 8");
        }
        if (r.admission_src != admission_source::initial_idle) {
            return fail_and_cleanup(
                "admission_src != initial_idle");
        }

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.arrival_drained_count != 1) {
            return fail_and_cleanup("arrival_drained_count != 1");
        }
        if (er.external_admitted_count != 1) {
            return fail_and_cleanup("external_admitted_count != 1");
        }
        if (er.admitted_count != 1) {
            return fail_and_cleanup("admitted_count != 1");
        }
        if (er.streams_opened != 1) {
            return fail_and_cleanup("streams_opened != 1");
        }
        if (er.streams_closed_completed != 1) {
            return fail_and_cleanup(
                "streams_closed_completed != 1");
        }
        if (er.stream_tokens_emitted_total != 8) {
            return fail_and_cleanup(
                "stream_tokens_emitted_total != 8");
        }

        print_result(r, streamed_tokens.size(), streamed_hash);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
