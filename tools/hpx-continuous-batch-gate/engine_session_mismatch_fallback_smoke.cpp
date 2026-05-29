// engine_session_mismatch_fallback_smoke.cpp — B1 Slice 4 guard for the
// session-mismatch fallback path (no reuse -> evict -> full prefill).
//
// A request that carries the same session_id as a resident slot but
// whose prompt diverges before resident_tokens.size() must NOT reuse
// (Slice 3 supports exact-extension only). If no ordinary free slot
// exists, the resident slot is evicted (Slice 4) and the request is
// full-prefilled.
//
// Shape (single engine, n_seq_max=1, initial_idle_slots=1, keep_alive,
// greedy):
//   baseline : P_M (no session) on the single idle slot -> FULL prefill;
//              records the correctness reference for P_M. Slot returns
//              to free_idle_.
//   rA       : session_id="sA", P0="Hello, my name is" -> resident slot.
//   rM       : session_id="sA", prompt P_M which diverges from P0 (and so
//              from sA's resident prefix) almost immediately -> no reuse;
//              the only slot is resident, so sA is evicted and rM is
//              full-prefilled.
//
// Asserts:
//   - no reuse happened (session_reuse_admitted_count == 0).
//   - exactly one eviction (resident_slots_evicted_count == 1).
//   - rM output == baseline output (hash + n_decoded): fallback full
//     prefill is correct. Both are full prefills of P_M -> same batch
//     shape, byte-identical.
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

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  B1 Slice 4 session-mismatch fallback guard for the\n"
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
    fprintf(stdout, "HPX_SESSION_MISMATCH_FALLBACK_SMOKE: FAIL: %s\n",
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
    model = llama_model_load_from_file(args.model_path.c_str(), model_params);
    if (model == nullptr) return fail_and_cleanup("model load failed");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 1;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) return fail_and_cleanup("context create failed");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> p0 = common_tokenize(
        ctx, "Hello, my name is", /*add_special=*/true,
        /*parse_special=*/true);
    // P_M diverges from P0 right after BOS, so it is NOT an extension of
    // sA's resident prefix.
    std::vector<llama_token> pm = common_tokenize(
        ctx, "The quick brown fox jumps over", /*add_special=*/true,
        /*parse_special=*/true);
    if (p0.empty() || pm.empty())
        return fail_and_cleanup("tokenization produced 0 tokens");

    const int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(std::max(p0.size(), pm.size())), 1);

    std::vector<waiting_request> empty_waiting;
    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 1;
    opts.lib.initial_idle_slots   = 1;
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

        // baseline: full prefill of P_M, no session.
        submit_handle hb = eng.submit_request(build(1, pm, ""));
        request_result rb = hb.result.get();
        if (rb.status != request_status::completed)
            return fail_and_cleanup("baseline status != completed");

        // rA: session sA, P0 -> resident on the single slot.
        submit_handle hA = eng.submit_request(build(2, p0, "sA"));
        request_result rA = hA.result.get();
        if (rA.status != request_status::completed)
            return fail_and_cleanup("rA status != completed");

        // rM: same session sA but divergent prompt -> no reuse, evict, full prefill.
        submit_handle hM = eng.submit_request(build(3, pm, "sA"));
        request_result rM = hM.result.get();
        if (rM.status != request_status::completed)
            return fail_and_cleanup("rM status != completed");

        eng.request_shutdown();
        engine_fut.get();

        const engine_result & er = eng.result();
        if (er.decode_failures != 0)
            return fail_and_cleanup("decode_failures != 0");
        // mismatch must NOT reuse.
        if (er.session_reuse_admitted_count != 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "session_reuse_admitted_count=%d (expected 0; mismatch must "
                "not reuse)", er.session_reuse_admitted_count);
            return fail_and_cleanup(buf);
        }
        // the only slot was resident, so rM forced exactly one eviction.
        if (er.resident_slots_evicted_count != 1) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "resident_slots_evicted_count=%d (expected 1)",
                er.resident_slots_evicted_count);
            return fail_and_cleanup(buf);
        }
        // fallback full prefill must match the no-session full-prefill baseline.
        if (rM.n_decoded != rb.n_decoded) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "rM n_decoded=%d != baseline n_decoded=%d",
                rM.n_decoded, rb.n_decoded);
            return fail_and_cleanup(buf);
        }
        if (rM.hash != rb.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "rM hash=0x%016llx != baseline hash=0x%016llx "
                "(fallback full prefill diverged)",
                static_cast<unsigned long long>(rM.hash),
                static_cast<unsigned long long>(rb.hash));
            return fail_and_cleanup(buf);
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                std::string("residual_kv_ok != true: ")
                + er.residual_kv_error);
        }

        fprintf(stdout,
            "baseline hash=0x%016llx n_decoded=%d\n"
            "rM       hash=0x%016llx n_decoded=%d\n"
            "session_reuse_admitted_count=%d resident_slots_evicted_count=%d "
            "last_evicted_resident_lru=%llu residual_kv_ok=%d\n",
            static_cast<unsigned long long>(rb.hash), rb.n_decoded,
            static_cast<unsigned long long>(rM.hash), rM.n_decoded,
            er.session_reuse_admitted_count,
            er.resident_slots_evicted_count,
            static_cast<unsigned long long>(er.last_evicted_resident_lru),
            er.residual_kv_ok ? 1 : 0);
        fprintf(stdout,
            "HPX_SESSION_MISMATCH_FALLBACK_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
