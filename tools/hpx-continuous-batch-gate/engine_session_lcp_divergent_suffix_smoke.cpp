// engine_session_lcp_divergent_suffix_smoke.cpp — B+1 guard for
// same-session longest-common-prefix reuse with a divergent suffix.
//
// B1 exact-session reuse only fires when the new prompt extends the WHOLE
// resident token sequence. B+1 generalizes to a useful common prefix:
// resident = [shared long prefix] + [old suffix]; a same-session request
// [shared long prefix] + [new suffix] reuses the shared prefix KV, trims
// the divergent resident tail (seq_rm), and prefills only the new suffix.
//
// Shape (single engine, n_seq_max=2, initial_idle_slots=2, keep_alive,
// greedy):
//   r1       : session "sA", SHARED + old suffix -> resident (slot A)
//   baseline : no session,   SHARED + new suffix -> FULL prefill (slot B);
//              the correctness reference for the extended prompt.
//   r2       : session "sA", SHARED + new suffix -> partial same-session
//              LCP reuse onto r1's slot (shared prefix kept, old tail
//              trimmed, new suffix prefilled).
//
// SHARED is > LCP_MIN_TOKENS (=32) tokens so the partial match clears the
// threshold. Asserts:
//   - exactly one LCP reuse (lcp_reuse_admitted_count == 1), matched >= 32,
//     trimmed > 0; no exact reuse (session_reuse_admitted_count == 0).
//   - r2 output == baseline output (hash + n_decoded): LCP reuse matches
//     full prefill.
//   - teardown residual-KV sweep passes.

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

// > 32 tokens of stable shared prefix.
const char * const SHARED =
    "You are a careful, concise assistant. Always answer truthfully and "
    "briefly. Do not speculate beyond the provided context. You are a "
    "careful, concise assistant. Always answer truthfully and briefly.";

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  B+1 same-session LCP divergent-suffix guard.\n", argv0);
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
    fprintf(stdout, "HPX_SESSION_LCP_DIVERGENT_SUFFIX_SMOKE: FAIL: %s\n",
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
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 256;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) return fail_and_cleanup("context create failed");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> p_old = common_tokenize(
        ctx, std::string(SHARED) + " The old topic is alpha.",
        /*add_special=*/true, /*parse_special=*/true);
    std::vector<llama_token> p_new = common_tokenize(
        ctx, std::string(SHARED) + " The new topic is beta.",
        /*add_special=*/true, /*parse_special=*/true);
    if (p_old.empty() || p_new.empty())
        return fail_and_cleanup("tokenization produced 0 tokens");

    const int32_t batch_capacity = 512;
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

    try {
        engine eng(std::move(opts));
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        request_result r1 = eng.submit_request(build(1, p_old, "sA")).result.get();
        if (r1.status != request_status::completed)
            return fail_and_cleanup("r1 status != completed");
        request_result rb = eng.submit_request(build(2, p_new, "")).result.get();
        if (rb.status != request_status::completed)
            return fail_and_cleanup("baseline status != completed");
        request_result r2 = eng.submit_request(build(3, p_new, "sA")).result.get();
        if (r2.status != request_status::completed)
            return fail_and_cleanup("r2 status != completed");

        eng.request_shutdown();
        engine_fut.get();

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) return fail_and_cleanup("decode_failures != 0");
        if (er.lcp_reuse_admitted_count != 1) {
            char b[160]; std::snprintf(b, sizeof(b),
                "lcp_reuse_admitted_count=%d (expected 1)",
                er.lcp_reuse_admitted_count);
            return fail_and_cleanup(b);
        }
        if (er.session_reuse_admitted_count != 0)
            return fail_and_cleanup("session_reuse_admitted_count != 0 (should be partial LCP, not exact)");
        if (er.lcp_reuse_matched_tokens < 32) {
            char b[160]; std::snprintf(b, sizeof(b),
                "lcp_reuse_matched_tokens=%lld (expected >= 32)",
                static_cast<long long>(er.lcp_reuse_matched_tokens));
            return fail_and_cleanup(b);
        }
        if (er.lcp_reuse_trimmed_tokens <= 0)
            return fail_and_cleanup("lcp_reuse_trimmed_tokens not > 0 (old tail not trimmed)");
        if (r2.n_decoded != rb.n_decoded)
            return fail_and_cleanup("r2 n_decoded != baseline n_decoded");
        if (r2.hash != rb.hash) {
            char b[256]; std::snprintf(b, sizeof(b),
                "r2 hash=0x%016llx != baseline hash=0x%016llx (LCP reuse diverged)",
                static_cast<unsigned long long>(r2.hash),
                static_cast<unsigned long long>(rb.hash));
            return fail_and_cleanup(b);
        }
        if (!er.residual_kv_ok)
            return fail_and_cleanup(std::string("residual_kv_ok != true: ") + er.residual_kv_error);

        fprintf(stdout,
            "baseline hash=0x%016llx n_decoded=%d\n"
            "r2       hash=0x%016llx n_decoded=%d\n"
            "lcp_reuse_admitted_count=%d matched_tokens=%lld trimmed_tokens=%lld "
            "session_reuse_admitted_count=%d residual_kv_ok=%d\n",
            static_cast<unsigned long long>(rb.hash), rb.n_decoded,
            static_cast<unsigned long long>(r2.hash), r2.n_decoded,
            er.lcp_reuse_admitted_count,
            static_cast<long long>(er.lcp_reuse_matched_tokens),
            static_cast<long long>(er.lcp_reuse_trimmed_tokens),
            er.session_reuse_admitted_count,
            er.residual_kv_ok ? 1 : 0);
        fprintf(stdout, "HPX_SESSION_LCP_DIVERGENT_SUFFIX_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
