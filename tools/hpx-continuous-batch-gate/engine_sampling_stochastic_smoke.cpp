// engine_sampling_stochastic_smoke.cpp — M6b: seeded stochastic
// sampling smoke for llama-hpx-engine.
//
// The carry smoke proves the M6b stochastic branch fires (Case B
// hash differs from greedy canonical). This smoke proves the
// stochastic path's correctness properties:
//
//   1. Same-seed reproducibility — two seed=42 requests on the same
//      prompt collapse to identical hashes.
//   2. Seed sensitivity — seed=42 vs seed=43 diverge.
//   3. Stochastic differs from the greedy canonical fingerprint.
//   4. Greedy after stochastic still reproduces the canonical hash
//      (proves no stochastic sampler state bleeds across admissions
//      in the supported ≤2-submit-per-engine shape).
//
// SMOKE STRUCTURE — multiple short-lived engines (path B):
//
//   This smoke uses four short-lived keep-alive engines, each handling
//   AT MOST TWO sequential `submit_request` calls before draining via
//   `request_shutdown()` + `engine_fut.get()`. Each engine borrows the
//   same shared model + llama_context; the engine's per-request KV
//   cleanup leaves the context empty between engine instances, and
//   only one engine is alive at a time so the "engine task is the
//   sole owner of llama.cpp mutable execution state" invariant is
//   preserved.
//
//     Engine A — property 1 (same-seed reproducibility)
//       submit_request(stochastic, seed=42)
//       submit_request(stochastic, seed=42)
//       assert r_a.hash == r_b.hash
//
//     Engine B — property 2 (seed sensitivity)
//       submit_request(stochastic, seed=42)
//       submit_request(stochastic, seed=43)
//       assert r_a.hash != r_b.hash
//
//     Engine C — property 3 (stochastic vs greedy canonical)
//       submit_request(greedy/default)
//       submit_request(stochastic, seed=42)
//       assert r_a.hash == 0x0619d4d1900c2365 (greedy canonical)
//       assert r_a.hash != r_b.hash
//
//     Engine D — property 4 (greedy after stochastic stays canonical)
//       submit_request(stochastic, seed=42)
//       submit_request(greedy/default)
//       assert r_b.hash == 0x0619d4d1900c2365
//
// WHY MULTIPLE ENGINES (FINDING TO PRESERVE):
//
//   The original M6b smoke design was a single keep-alive engine
//   handling four sequential `submit_request` calls. During M6b
//   validation that design HUNG on the third sequential submit —
//   r100 (stochastic seed=42) and r101 (stochastic seed=42) both
//   completed correctly (and produced identical hashes, proving
//   same-seed reproducibility), but r102 never received a result.
//
//   A diagnostic where r102 was temporarily swapped to GREEDY (zero
//   M6b code path involvement — the M6b `if (seq.sampler_chain)`
//   branch falls through to the existing argmax for greedy) ALSO
//   hung at exactly the same point. That rules out a stochastic-
//   specific bug: the hang is a pre-existing engine issue in the
//   keep-alive 3rd-or-later sequential `submit_request` path, never
//   exposed before because the existing `engine_keepalive_smoke`
//   only tests two sequential requests.
//
//   Fixing that engine bug is OUT OF SCOPE FOR M6b. This smoke is
//   structured around the bug (≤2 submits per engine) so that the
//   M6b sampling contracts can be validated independently. The
//   pre-existing hang is recommended as the next follow-up slice
//   after M6b closes — it deserves its own dedicated regression
//   smoke (a 3+ sequential greedy submit_request shape that does
//   not depend on M6b sampling at all).
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution, KV mutation, and sampler-chain construction/mutation
// live inside eng.run() running on the single hpx::async task each
// per-engine block spawns. No std::thread, no std::mutex, no
// std::condition_variable, no std::this_thread::sleep_for. No
// direct llama_sampler_* call sites in this TU — the engine task is
// the sole owner of every sampler chain it constructs, and only one
// engine instance is live at a time.

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
        "  M6b seeded stochastic sampling smoke for the "
        "llama-hpx-engine library.\n",
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
    fprintf(stdout,
            "HPX_ENGINE_SAMPLING_STOCHASTIC_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_SAMPLING_STOCHASTIC_SMOKE: PASS\n");
    fflush(stdout);
}

void print_pair(const char * engine_label,
                const char * label_a, const request_result & r_a,
                const char * label_b, const request_result & r_b) {
    fprintf(stdout,
            "%s %s rid=%d hash=0x%016llx | %s rid=%d hash=0x%016llx\n",
            engine_label,
            label_a, r_a.request_id,
            static_cast<unsigned long long>(r_a.hash),
            label_b, r_b.request_id,
            static_cast<unsigned long long>(r_b.hash));
    fflush(stdout);
}

submit_request build_stochastic(int32_t rid,
                                uint32_t seed,
                                const std::vector<llama_token> & prompt) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
    req.sampling.mode = sampling_mode::stochastic;
    req.sampling.seed = seed;
    // temperature / top_k / top_p / top_p_min_keep remain at default;
    // M6b ignores them (dist-only chain). M6c will wire them.
    return req;
}

submit_request build_greedy(int32_t rid,
                            const std::vector<llama_token> & prompt) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
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

    // Per-engine driver: construct a fresh keep-alive engine bound to
    // the shared llama_context, submit two requests sequentially,
    // await both, drain via request_shutdown(). Only one engine is
    // alive at any moment, so the engine task remains the sole owner
    // of llama_context mutable state across all four blocks.
    auto run_two_reqs = [&](const char * engine_label,
                            submit_request req_a,
                            submit_request req_b,
                            request_result & out_a,
                            request_result & out_b,
                            std::string & err) -> bool {
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

        engine eng(std::move(opts));
        hpx::future<void> fut = hpx::async([&] { eng.run(); });

        submit_handle h_a = eng.submit_request(std::move(req_a));
        out_a = h_a.result.get();

        submit_handle h_b = eng.submit_request(std::move(req_b));
        out_b = h_b.result.get();

        eng.request_shutdown();
        fut.get();

        if (out_a.status != request_status::completed) {
            err = std::string(engine_label) + " req_a status != completed";
            return false;
        }
        if (out_b.status != request_status::completed) {
            err = std::string(engine_label) + " req_b status != completed";
            return false;
        }
        if (out_a.n_decoded != 8) {
            err = std::string(engine_label) + " req_a n_decoded != 8";
            return false;
        }
        if (out_b.n_decoded != 8) {
            err = std::string(engine_label) + " req_b n_decoded != 8";
            return false;
        }
        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            err = std::string(engine_label) + " decode_failures != 0";
            return false;
        }
        if (!er.residual_kv_ok) {
            err = std::string(engine_label) + " residual_kv_ok != true";
            return false;
        }
        if (er.admitted_count != 2) {
            err = std::string(engine_label) + " admitted_count != 2";
            return false;
        }
        return true;
    };

    constexpr uint64_t k_canonical_greedy_b8 = 0x0619d4d1900c2365ULL;

    try {
        // ---- Engine A: property 1 — same-seed reproducibility --------
        request_result A_a, A_b;
        {
            std::string err;
            if (!run_two_reqs("EngineA",
                    build_stochastic(/*rid=*/100, /*seed=*/42, shared_prompt),
                    build_stochastic(/*rid=*/101, /*seed=*/42, shared_prompt),
                    A_a, A_b, err)) {
                return fail_and_cleanup(err);
            }
        }
        if (A_a.hash != A_b.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "EngineA property 1 (same-seed reproducibility) failed: "
                "seed=42 r100.hash=0x%016llx != r101.hash=0x%016llx",
                static_cast<unsigned long long>(A_a.hash),
                static_cast<unsigned long long>(A_b.hash));
            return fail_and_cleanup(buf);
        }
        if (A_a.generated_tokens != A_b.generated_tokens) {
            return fail_and_cleanup(
                "EngineA property 1: same-seed token sequences "
                "diverged despite hash equality");
        }

        // ---- Engine B: property 2 — seed sensitivity -----------------
        request_result B_a, B_b;
        {
            std::string err;
            if (!run_two_reqs("EngineB",
                    build_stochastic(/*rid=*/200, /*seed=*/42, shared_prompt),
                    build_stochastic(/*rid=*/201, /*seed=*/43, shared_prompt),
                    B_a, B_b, err)) {
                return fail_and_cleanup(err);
            }
        }
        if (B_a.hash == B_b.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "EngineB property 2 (seed sensitivity) failed: "
                "seed=42 and seed=43 both produced hash=0x%016llx — "
                "dist sub-sampler appears to ignore the seed",
                static_cast<unsigned long long>(B_a.hash));
            return fail_and_cleanup(buf);
        }

        // ---- Engine C: property 3 — stochastic vs greedy canonical ---
        request_result C_a, C_b;
        {
            std::string err;
            if (!run_two_reqs("EngineC",
                    build_greedy(/*rid=*/300, shared_prompt),
                    build_stochastic(/*rid=*/301, /*seed=*/42, shared_prompt),
                    C_a, C_b, err)) {
                return fail_and_cleanup(err);
            }
        }
        if (C_a.hash != k_canonical_greedy_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "EngineC property 3 part 1 (greedy unchanged) failed: "
                "r300.hash=0x%016llx != canonical 0x%016llx",
                static_cast<unsigned long long>(C_a.hash),
                static_cast<unsigned long long>(k_canonical_greedy_b8));
            return fail_and_cleanup(buf);
        }
        if (C_a.hash == C_b.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "EngineC property 3 part 2 (stochastic differs from "
                "greedy) failed: stochastic r301.hash=0x%016llx equals "
                "greedy canonical",
                static_cast<unsigned long long>(C_b.hash));
            return fail_and_cleanup(buf);
        }

        // ---- Engine D: property 4 — greedy after stochastic ---------
        request_result D_a, D_b;
        {
            std::string err;
            if (!run_two_reqs("EngineD",
                    build_stochastic(/*rid=*/400, /*seed=*/42, shared_prompt),
                    build_greedy(/*rid=*/401, shared_prompt),
                    D_a, D_b, err)) {
                return fail_and_cleanup(err);
            }
        }
        if (D_b.hash != k_canonical_greedy_b8) {
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "EngineD property 4 (greedy after stochastic) failed: "
                "r401.hash=0x%016llx != canonical 0x%016llx — slot "
                "reuse leaked stochastic state into a greedy admission "
                "(sampler_chain.reset() ordering in admit_one or "
                "finalize_and_fulfill is wrong)",
                static_cast<unsigned long long>(D_b.hash),
                static_cast<unsigned long long>(k_canonical_greedy_b8));
            return fail_and_cleanup(buf);
        }

        print_pair("EngineA", "stoc42", A_a, "stoc42", A_b);
        print_pair("EngineB", "stoc42", B_a, "stoc43", B_b);
        print_pair("EngineC", "greedy", C_a, "stoc42", C_b);
        print_pair("EngineD", "stoc42", D_a, "greedy", D_b);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
