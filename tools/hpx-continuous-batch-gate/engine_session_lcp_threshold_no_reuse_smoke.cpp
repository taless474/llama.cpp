// engine_session_lcp_threshold_no_reuse_smoke.cpp — B+1 guard that a
// same-session common prefix BELOW LCP_MIN_TOKENS (=32) does not trigger
// partial LCP reuse and falls back to a correct full prefill.
//
// Shape (single engine, n_seq_max=2, keep_alive, greedy):
//   r1       : session "sB", "Hi there alpha"  -> resident (tiny prompt)
//   baseline : no session,   "Hi there beta"   -> full prefill reference
//   r2       : session "sB", "Hi there beta"   -> shares only a tiny
//              prefix (< 32 tokens) with r1's resident, so NO LCP reuse;
//              full prefill instead.
//
// Asserts: lcp_reuse_admitted_count == 0 and session_reuse_admitted_count
// == 0 (sub-threshold, not exact), r2 output == baseline, teardown clean.

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

struct smoke_args { std::string model_path; };

void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --model <path>\n"
        "  B+1 LCP threshold no-reuse guard.\n", argv0);
}

bool parse_args(int argc, char ** argv, smoke_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model") {
            if (i + 1 >= argc) { fprintf(stderr, "error: --model requires a value\n"); return false; }
            out.model_path = argv[++i];
        } else if (a == "-h" || a == "--help") { print_usage(argv[0]); return false; }
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return false; }
    }
    if (out.model_path.empty()) { fprintf(stderr, "error: --model is required\n"); return false; }
    return true;
}

void emit_fail(const std::string & reason) {
    fprintf(stdout, "HPX_SESSION_LCP_THRESHOLD_NO_REUSE_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

submit_request build(int32_t rid, const std::vector<llama_token> & prompt,
                     const std::string & session_id) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
    req.session_id    = session_id;
    return req;
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();
    if (!hpx_runtime::start_once(/*os_threads=*/1)) { emit_fail("hpx runtime start failed"); return 1; }
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
        emit_fail(reason); return cleanup_and_stop(1);
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(), model_params);
    if (model == nullptr) return fail_and_cleanup("model load failed");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 1024;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) return fail_and_cleanup("context create failed");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> p_old = common_tokenize(
        ctx, "Hi there alpha", true, true);
    std::vector<llama_token> p_new = common_tokenize(
        ctx, "Hi there beta", true, true);
    if (p_old.empty() || p_new.empty())
        return fail_and_cleanup("tokenization produced 0 tokens");

    std::vector<waiting_request> empty_waiting;
    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = 64;
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

    try {
        engine eng(std::move(opts));
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        request_result r1 = eng.submit_request(build(1, p_old, "sB")).result.get();
        request_result rb = eng.submit_request(build(2, p_new, "")).result.get();
        request_result r2 = eng.submit_request(build(3, p_new, "sB")).result.get();

        eng.request_shutdown();
        engine_fut.get();

        if (r1.status != request_status::completed
         || rb.status != request_status::completed
         || r2.status != request_status::completed)
            return fail_and_cleanup("a request did not complete");

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) return fail_and_cleanup("decode_failures != 0");
        if (er.lcp_reuse_admitted_count != 0) {
            char b[160]; std::snprintf(b, sizeof(b),
                "lcp_reuse_admitted_count=%d (expected 0; below threshold)",
                er.lcp_reuse_admitted_count);
            return fail_and_cleanup(b);
        }
        if (er.session_reuse_admitted_count != 0)
            return fail_and_cleanup("session_reuse_admitted_count != 0 (no exact extension expected)");
        if (r2.n_decoded != rb.n_decoded)
            return fail_and_cleanup("r2 n_decoded != baseline n_decoded");
        if (r2.hash != rb.hash) {
            char b[256]; std::snprintf(b, sizeof(b),
                "r2 hash=0x%016llx != baseline hash=0x%016llx",
                static_cast<unsigned long long>(r2.hash),
                static_cast<unsigned long long>(rb.hash));
            return fail_and_cleanup(b);
        }
        if (!er.residual_kv_ok)
            return fail_and_cleanup(std::string("residual_kv_ok != true: ") + er.residual_kv_error);

        fprintf(stdout,
            "baseline hash=0x%016llx r2 hash=0x%016llx "
            "lcp_reuse_admitted_count=%d session_reuse_admitted_count=%d "
            "residual_kv_ok=%d\n",
            static_cast<unsigned long long>(rb.hash),
            static_cast<unsigned long long>(r2.hash),
            er.lcp_reuse_admitted_count, er.session_reuse_admitted_count,
            er.residual_kv_ok ? 1 : 0);
        fprintf(stdout, "HPX_SESSION_LCP_THRESHOLD_NO_REUSE_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
