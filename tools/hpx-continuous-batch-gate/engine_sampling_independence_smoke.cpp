// engine_sampling_independence_smoke.cpp — M6c: per-seq sampler
// isolation under concurrent admission and slot reuse.
//
// Submits two stochastic requests on a single keep_alive engine
// before awaiting either result, so both are bound concurrently to
// the two available slots. After both complete, a greedy submit
// exercises the post-keepalive-fix slot recycle: the new owner of
// either slot must take the existing local-argmax path (no
// stochastic sampler state may bleed across admissions). A final
// stochastic seed=42 default-knob submit confirms a clean per-seq
// chain is rebuilt on slot reuse and produces the M6b anchor.
//
// Anchors:
//   - greedy canonical:              0x0619d4d1900c2365
//   - default stochastic seed=42:    0xa8e14acb4094aa3f
//
// HPX-native boundary: this TU calls llama.cpp APIs only for
// backend / model / context setup, tokenization, and final cleanup.
// All llama execution, KV mutation, and sampler-chain construction
// /mutation live inside eng.run() running on the single hpx::async
// task this file spawns. No std::thread, no std::mutex, no
// std::condition_variable, no std::this_thread::sleep_for. No
// direct llama_sampler_* call sites in this TU — the engine task is
// the sole owner of every sampler chain it constructs.

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
        "  M6c per-seq sampler isolation smoke for the "
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
            "HPX_ENGINE_SAMPLING_INDEPENDENCE_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

submit_request build_stochastic(int32_t rid,
                                uint32_t seed,
                                const std::vector<llama_token> & prompt) {
    submit_request req;
    req.request_id      = rid;
    req.prompt_tokens   = prompt;
    req.decode_budget   = 8;
    req.want_stream     = false;
    req.sampling.mode   = sampling_mode::stochastic;
    req.sampling.seed   = seed;
    // temperature/top_k/top_p/top_p_min_keep left at defaults so the
    // chain reduces to init_dist(seed).
    return req;
}

submit_request build_greedy(int32_t rid,
                            const std::vector<llama_token> & prompt) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
    // default sampling_config: greedy.
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

    // Concurrent admission of two sequences shares a single prefill
    // batch, so capacity must cover n_seq_max prompts.
    const int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(shared_prompt.size()) * 2, 1);

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

    constexpr uint64_t k_canonical_greedy_b8 =
        0x0619d4d1900c2365ULL;
    constexpr uint64_t k_default_stochastic_seed42_b8 =
        0xa8e14acb4094aa3fULL;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // ---- Phase 1: concurrent stochastic admission -----------------
        // Submit both requests BEFORE awaiting either, so both are
        // bound by the same admission pass to the two available
        // initial-idle slots.
        submit_handle h1 = eng.submit_request(
            build_stochastic(/*rid=*/1, /*seed=*/42, shared_prompt));
        submit_handle h2 = eng.submit_request(
            build_stochastic(/*rid=*/2, /*seed=*/43, shared_prompt));

        request_result r1 = h1.result.get();
        request_result r2 = h2.result.get();

        if (r1.status != request_status::completed) {
            return fail_and_cleanup("r1 status != completed");
        }
        if (r2.status != request_status::completed) {
            return fail_and_cleanup("r2 status != completed");
        }
        if (r1.n_decoded != 8 || r2.n_decoded != 8) {
            return fail_and_cleanup("r1/r2 n_decoded != 8");
        }
        if (r1.hash == r2.hash) {
            return fail_and_cleanup(
                "r1.hash == r2.hash — seed=42 vs seed=43 must diverge "
                "(cross-seq sampler state bleed suspected)");
        }
        // r1 is default-knob stochastic seed=42 → must equal anchor.
        if (r1.hash != k_default_stochastic_seed42_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "r1 hash 0x%016llx != default-stochastic seed=42 "
                "anchor 0x%016llx — concurrent admission contaminated "
                "r1's per-seq sampler",
                static_cast<unsigned long long>(r1.hash),
                static_cast<unsigned long long>(
                    k_default_stochastic_seed42_b8));
            return fail_and_cleanup(buf);
        }

        // ---- Phase 2: greedy after stochastic — slot recycle ----------
        // r3 takes a slot previously bound to a stochastic admission.
        // The greedy path must run argmax (sampler_chain is null), so
        // the hash must equal the greedy canonical. Any leftover
        // sampler_chain on the recycled slot would surface here.
        submit_handle h3 = eng.submit_request(
            build_greedy(/*rid=*/3, shared_prompt));
        request_result r3 = h3.result.get();

        if (r3.status != request_status::completed) {
            return fail_and_cleanup("r3 status != completed");
        }
        if (r3.n_decoded != 8) {
            return fail_and_cleanup("r3 n_decoded != 8");
        }
        if (r3.hash != k_canonical_greedy_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "r3 hash 0x%016llx != greedy canonical 0x%016llx — "
                "slot recycled from stochastic to greedy did not "
                "clear sampler_chain",
                static_cast<unsigned long long>(r3.hash),
                static_cast<unsigned long long>(k_canonical_greedy_b8));
            return fail_and_cleanup(buf);
        }

        // ---- Phase 3: stochastic seed=42 after greedy on reused slot --
        // Final assertion: a clean chain is rebuilt for the new
        // stochastic owner, and seed=42/default knobs reproduces the
        // anchor that r1 also produced.
        submit_handle h4 = eng.submit_request(
            build_stochastic(/*rid=*/4, /*seed=*/42, shared_prompt));
        request_result r4 = h4.result.get();

        if (r4.status != request_status::completed) {
            return fail_and_cleanup("r4 status != completed");
        }
        if (r4.n_decoded != 8) {
            return fail_and_cleanup("r4 n_decoded != 8");
        }
        if (r4.hash != k_default_stochastic_seed42_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "r4 hash 0x%016llx != default-stochastic seed=42 "
                "anchor 0x%016llx — slot recycled from greedy back to "
                "stochastic did not rebuild a fresh chain",
                static_cast<unsigned long long>(r4.hash),
                static_cast<unsigned long long>(
                    k_default_stochastic_seed42_b8));
            return fail_and_cleanup(buf);
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
        if (er.admitted_count != 4) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "admitted_count=%d (expected 4)", er.admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.external_admitted_count != 4) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "external_admitted_count=%d (expected 4)",
                er.external_admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.arrival_drained_count != 4) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "arrival_drained_count=%d (expected 4)",
                er.arrival_drained_count);
            return fail_and_cleanup(buf);
        }

        fprintf(stdout,
            "r1 seed=42 stochastic hash=0x%016llx admission_src=%s\n",
            static_cast<unsigned long long>(r1.hash),
            admission_source_name(r1.admission_src));
        fprintf(stdout,
            "r2 seed=43 stochastic hash=0x%016llx admission_src=%s\n",
            static_cast<unsigned long long>(r2.hash),
            admission_source_name(r2.admission_src));
        fprintf(stdout,
            "r3 greedy            hash=0x%016llx admission_src=%s\n",
            static_cast<unsigned long long>(r3.hash),
            admission_source_name(r3.admission_src));
        fprintf(stdout,
            "r4 seed=42 stochastic hash=0x%016llx admission_src=%s\n",
            static_cast<unsigned long long>(r4.hash),
            admission_source_name(r4.admission_src));
        fprintf(stdout,
            "HPX_ENGINE_SAMPLING_INDEPENDENCE_SMOKE: PASS\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
