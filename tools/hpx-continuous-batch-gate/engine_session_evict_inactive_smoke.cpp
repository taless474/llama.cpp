// engine_session_evict_inactive_smoke.cpp — B1 Slice 4 guard for
// inactive resident-slot eviction (LRU) under slot pressure.
//
// Slice 4: when no ordinary free slot is available and a waiter cannot
// exact-reuse a resident slot, the oldest (smallest resident_lru)
// INACTIVE resident slot is reclaimed (KV cleared, returned to
// free_idle_) and the waiter is admitted with a normal full prefill.
//
// Shape (single engine, n_seq_max=2, initial_idle_slots=2, keep_alive,
// greedy, all prompts "Hello, my name is" budget 8):
//   rA: session_id="sA" -> resident (resident_lru=1, the OLDEST)
//   rB: session_id="sB" -> resident (resident_lru=2)
//       both slots now resident; no free slot remains.
//   rC: session_id="sC" -> cannot reuse (new session) and no free slot,
//       so the LRU resident slot (sA, lru=1) is evicted and rC is
//       full-prefilled on it.
//
// Asserts:
//   - rA, rB, rC all complete with canonical b8 hash 0x0619d4d1900c2365
//     (full prefill of "Hello, my name is" greedy budget 8, including rC
//     after eviction — a same-shape full prefill, so byte-identical).
//   - exactly one eviction (resident_slots_evicted_count == 1) and it
//     reclaimed the OLDEST slot (last_evicted_resident_lru == 1) -> LRU.
//   - no reuse happened (session_reuse_admitted_count == 0): sA/sB/sC are
//     all distinct sessions and rC's prompt is not an extension.
//   - teardown clears remaining resident KV (residual_kv_ok == true).

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
        "  B1 Slice 4 LRU-eviction guard for the llama-hpx-engine library.\n",
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
    fprintf(stdout, "HPX_SESSION_EVICT_INACTIVE_SMOKE: FAIL: %s\n",
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
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) return fail_and_cleanup("context create failed");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> p0 = common_tokenize(
        ctx, "Hello, my name is", /*add_special=*/true,
        /*parse_special=*/true);
    if (p0.empty()) return fail_and_cleanup("tokenization produced 0 tokens");

    const int32_t batch_capacity =
        std::max<int32_t>(static_cast<int32_t>(p0.size()), 1);

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

    constexpr uint64_t k_b8 = 0x0619d4d1900c2365ULL;

    try {
        engine eng(std::move(opts));
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        submit_handle hA = eng.submit_request(build(1, p0, "sA"));
        request_result rA = hA.result.get();
        submit_handle hB = eng.submit_request(build(2, p0, "sB"));
        request_result rB = hB.result.get();
        // Both slots resident now; rC (new session) forces an eviction.
        submit_handle hC = eng.submit_request(build(3, p0, "sC"));
        request_result rC = hC.result.get();

        eng.request_shutdown();
        engine_fut.get();

        const request_result * rs[3]  = {&rA, &rB, &rC};
        const char *           lbl[3] = {"rA_sA", "rB_sB", "rC_sC"};
        for (int i = 0; i < 3; i++) {
            if (rs[i]->status != request_status::completed) {
                return fail_and_cleanup(
                    std::string(lbl[i]) + " status != completed");
            }
            if (rs[i]->hash != k_b8) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s hash=0x%016llx != b8 0x%016llx", lbl[i],
                    static_cast<unsigned long long>(rs[i]->hash),
                    static_cast<unsigned long long>(k_b8));
                return fail_and_cleanup(buf);
            }
        }

        const engine_result & er = eng.result();
        if (er.decode_failures != 0)
            return fail_and_cleanup("decode_failures != 0");
        if (er.session_reuse_admitted_count != 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "session_reuse_admitted_count=%d (expected 0)",
                er.session_reuse_admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.resident_slots_evicted_count != 1) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "resident_slots_evicted_count=%d (expected 1)",
                er.resident_slots_evicted_count);
            return fail_and_cleanup(buf);
        }
        // LRU: the OLDEST resident slot (rA, resident_lru==1) was evicted.
        if (er.last_evicted_resident_lru != 1) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "last_evicted_resident_lru=%llu (expected 1 == oldest)",
                static_cast<unsigned long long>(
                    er.last_evicted_resident_lru));
            return fail_and_cleanup(buf);
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                std::string("residual_kv_ok != true: ")
                + er.residual_kv_error);
        }

        fprintf(stdout,
            "rA/rB/rC hash=0x%016llx n_decoded=%d/%d/%d\n"
            "session_reuse_admitted_count=%d resident_slots_evicted_count=%d "
            "last_evicted_resident_lru=%llu residual_kv_ok=%d\n",
            static_cast<unsigned long long>(rA.hash),
            rA.n_decoded, rB.n_decoded, rC.n_decoded,
            er.session_reuse_admitted_count,
            er.resident_slots_evicted_count,
            static_cast<unsigned long long>(er.last_evicted_resident_lru),
            er.residual_kv_ok ? 1 : 0);
        fprintf(stdout, "HPX_SESSION_EVICT_INACTIVE_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
