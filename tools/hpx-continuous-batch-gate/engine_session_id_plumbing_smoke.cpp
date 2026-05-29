// engine_session_id_plumbing_smoke.cpp — B1 Slice 1 plumbing guard.
//
// Slice 1 adds an optional `session_id` string end-to-end
// (submit_request -> arrival_msg -> waiting_request ->
// seq_state::owning_session_id) but wires NO reuse, residency,
// admission, or KV behavior to it. This smoke proves the plumbing is
// inert:
//
//   r1: session_id = ""        (no-session path)
//   r2: session_id = "sess-A"  (session present, but Slice 1 has no reuse)
//
// Both requests use identical prompt / greedy / decode_budget=8 and must
// therefore produce the byte-identical canonical greedy hash
// 0x0619d4d1900c2365 with n_decoded == 8. If carrying a session_id
// changed any observable output, r2 would diverge.
//
// Engine shape matches engine_keepalive_multi_submit_smoke (greedy,
// n_seq_max=2, initial_idle_slots=2, keep_alive=true, reuse_completed=
// false, no streaming, no cancellation). Two sequential submit/await
// calls — well within the two idle slots, so both admit from
// free_idle_.

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
#include <cstdlib>
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
        "  B1 Slice 1 session_id plumbing guard for the\n"
        "  llama-hpx-engine library.\n",
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
    fprintf(stdout, "HPX_SESSION_ID_PLUMBING_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

submit_request build_greedy(int32_t rid,
                            const std::vector<llama_token> & prompt,
                            const std::string & session_id) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
    req.session_id    = session_id;
    // sampling default-constructed: greedy.
    return req;
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
        ctx, "Hello, my name is", /*add_special=*/true,
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
    opts.lib.initial_idle_slots   = 2;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    constexpr uint64_t k_canonical_greedy_hash = 0x0619d4d1900c2365ULL;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        // r1: no session (empty string) — today's byte-identical path.
        submit_handle h1 = eng.submit_request(
            build_greedy(/*rid=*/1, shared_prompt, /*session_id=*/""));
        request_result r1 = h1.result.get();

        // r2: session present, but Slice 1 wires no reuse — must be
        // byte-identical to r1.
        submit_handle h2 = eng.submit_request(
            build_greedy(/*rid=*/2, shared_prompt, /*session_id=*/"sess-A"));
        request_result r2 = h2.result.get();

        const request_result * rs[2]  = {&r1, &r2};
        const char *           lbl[2] = {"r1_no_session", "r2_session_present"};
        for (int i = 0; i < 2; i++) {
            if (rs[i]->status != request_status::completed) {
                return fail_and_cleanup(
                    std::string(lbl[i]) + " status != completed");
            }
            if (rs[i]->n_decoded != 8) {
                return fail_and_cleanup(
                    std::string(lbl[i]) + " n_decoded != 8");
            }
            if (rs[i]->hash != k_canonical_greedy_hash) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s hash=0x%016llx != canonical 0x%016llx",
                    lbl[i],
                    static_cast<unsigned long long>(rs[i]->hash),
                    static_cast<unsigned long long>(
                        k_canonical_greedy_hash));
                return fail_and_cleanup(buf);
            }
        }

        // Explicit cross-check: session-present output equals no-session
        // output (the core Slice 1 invariant).
        if (r1.hash != r2.hash) {
            return fail_and_cleanup(
                "session_id-present hash differs from no-session hash");
        }

        eng.request_shutdown();
        engine_fut.get();

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.admitted_count != 2) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "admitted_count=%d (expected 2)", er.admitted_count);
            return fail_and_cleanup(buf);
        }

        for (int i = 0; i < 2; i++) {
            fprintf(stdout, "%s hash=0x%016llx n_decoded=%d\n",
                lbl[i],
                static_cast<unsigned long long>(rs[i]->hash),
                rs[i]->n_decoded);
        }
        fprintf(stdout, "HPX_SESSION_ID_PLUMBING_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
