// engine_sampling_config_carry_smoke.cpp — per-request sampling_config
// carry + activation smoke for llama-hpx-engine.
//
// M6a established the data-model carry: `sampling_config` flows from
// public `submit_request` through `arrival_msg`, `waiting_request`,
// and into `seq_state` — but no sampling site read it, so a
// stochastic-looking config produced the same hash as greedy.
//
// M6b activates the stochastic path. The engine now branches in
// `admit_one` to build a per-seq `llama_sampler_ptr` chain when
// `seq.sampling.mode == sampling_mode::stochastic` (minimal chain:
// `llama_sampler_init_dist(seed)` only — temperature/top_k/top_p are
// deferred to M6c). Both token-selection sites in `engine.cpp`
// dispatch on `seq.sampler_chain`: non-null → `llama_sampler_sample`
// (which internally accepts), null → the existing
// `llama_get_logits_ith` + local `argmax` path bit-identically.
//
// Test shape: keep-alive engine with two sequential single-request
// submissions on the same prompt and decode_budget=8.
//
//   Case A — default sampling_config (greedy). Hash MUST equal the
//            canonical TinyLlama / "Hello, my name is" / greedy /
//            budget-8 fingerprint 0x0619d4d1900c2365. Proves the
//            greedy path is byte-identical to pre-M6b (a regression
//            in the argmax path or in any greedy-side wiring would
//            shift this literal).
//
//   Case B — explicit stochastic-looking config:
//              mode             = sampling_mode::stochastic
//              seed             = 42
//              temperature      = 0.7f
//              top_k            = 40
//              top_p            = 0.95f
//              top_p_min_keep   = 1
//            In M6b this MUST differ from Case A — the engine builds
//            a `dist(42)` chain at admission and routes token
//            selection through `llama_sampler_sample` instead of
//            argmax. temperature/top_k/top_p are still carried but
//            ignored (M6c). A regression that would silently re-
//            collapse Case B onto the greedy path (e.g. the
//            stochastic branch never built the chain, or the
//            sampling-site dispatch fell through) is caught here.
//
// Same-seed reproducibility, different-seed divergence, and
// greedy-after-stochastic canonical reproduction are covered by the
// companion `engine_sampling_stochastic_smoke`.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution, KV mutation, and sampler-chain construction/mutation
// live inside eng.run() running on the single hpx::async task this
// file spawns. No std::thread, no std::mutex, no
// std::condition_variable, no std::this_thread::sleep_for — the
// engine's keep-alive idle-wait runs on hpx::condition_variable_any
// under the engine task. No direct llama_sampler_* call sites in
// this TU — the engine task is the sole owner of every sampler
// chain it constructs.

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
        "  M6a per-request sampling_config carry smoke for the "
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
            "HPX_ENGINE_SAMPLING_CONFIG_CARRY_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_SAMPLING_CONFIG_CARRY_SMOKE: PASS\n");
    fflush(stdout);
}

void print_result(const char * label, const request_result & r,
                  sampling_mode mode) {
    fprintf(stdout,
            "%s request_id=%d sampling_mode=%s status=%s "
            "n_decoded=%d hash=0x%016llx\n",
            label, r.request_id, sampling_mode_name(mode),
            status_name(r.status), r.n_decoded,
            static_cast<unsigned long long>(r.hash));
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
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // Canonical TinyLlama / "Hello, my name is" / greedy reference
    // prompt. The two requests share this prompt token-for-token so
    // any post-prefill hash divergence between Case A and Case B can
    // only originate at token-selection — i.e. would prove the
    // engine read `seq.sampling.mode`, which is the M6a contract
    // violation this smoke catches.
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

    try {
        engine eng(std::move(opts));

        // Engine starts before any submission; both submits below
        // travel the same drain_external_inbox → admit_one path
        // exercised in M3a/M4/M5. Sampling carry is the only thing
        // new on this path in M6a.
        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // ---- Case A: default sampling_config (greedy) -----------------
        submit_request req_a;
        req_a.request_id    = 1001;
        req_a.prompt_tokens = shared_prompt;
        req_a.decode_budget = 8;
        req_a.want_stream   = false;
        // sampling is default-constructed: greedy, LLAMA_DEFAULT_SEED,
        // temperature=1.0, top_k=0, top_p=1.0, top_p_min_keep=1.
        submit_handle h_a = eng.submit_request(std::move(req_a));
        request_result r_a = h_a.result.get();

        // ---- Case B1: non-default stochastic config -------------------
        // M6c chain wires top_k / top_p / temperature alongside
        // dist(seed). The literal hash for this knob set is not pinned
        // in this slice; B1's value is captured for the reproducibility
        // and divergence assertions below.
        submit_request req_b1;
        req_b1.request_id    = 1002;
        req_b1.prompt_tokens = shared_prompt;
        req_b1.decode_budget = 8;
        req_b1.want_stream   = false;
        req_b1.sampling.mode           = sampling_mode::stochastic;
        req_b1.sampling.seed           = 42;
        req_b1.sampling.temperature    = 0.7f;
        req_b1.sampling.top_k          = 40;
        req_b1.sampling.top_p          = 0.95f;
        req_b1.sampling.top_p_min_keep = 1;
        submit_handle h_b1 = eng.submit_request(std::move(req_b1));
        request_result r_b1 = h_b1.result.get();

        // ---- Case B2: same non-default config — reproducibility ------
        submit_request req_b2;
        req_b2.request_id    = 1003;
        req_b2.prompt_tokens = shared_prompt;
        req_b2.decode_budget = 8;
        req_b2.want_stream   = false;
        req_b2.sampling.mode           = sampling_mode::stochastic;
        req_b2.sampling.seed           = 42;
        req_b2.sampling.temperature    = 0.7f;
        req_b2.sampling.top_k          = 40;
        req_b2.sampling.top_p          = 0.95f;
        req_b2.sampling.top_p_min_keep = 1;
        submit_handle h_b2 = eng.submit_request(std::move(req_b2));
        request_result r_b2 = h_b2.result.get();

        // ---- Case C: default-knob stochastic seed=42 ------------------
        // M6c contract: a stochastic config with all knobs at defaults
        // builds a chain containing exactly init_dist(seed) — byte-
        // equivalent to the M6b shape — so its hash MUST match the M6b
        // anchor recorded during the post-keepalive-fix validation.
        submit_request req_c;
        req_c.request_id    = 1004;
        req_c.prompt_tokens = shared_prompt;
        req_c.decode_budget = 8;
        req_c.want_stream   = false;
        req_c.sampling.mode = sampling_mode::stochastic;
        req_c.sampling.seed = 42;
        // temperature=1.0, top_k=0, top_p=1.0, top_p_min_keep=1 by
        // default — so the M6c helper adds only init_dist(seed).
        submit_handle h_c = eng.submit_request(std::move(req_c));
        request_result r_c = h_c.result.get();

        // ---- Drained shutdown -----------------------------------------
        eng.request_shutdown();
        engine_fut.get();

        // ---- Per-request invariants -----------------------------------
        // All four must complete successfully. Cases A and C use anchor
        // configs whose hashes are pinned; their decode lengths are
        // therefore implicitly pinned to 8 via the anchor literals.
        // Cases B1/B2 use a non-default stochastic config that may
        // terminate early on EOG under the chosen knobs — both still
        // "complete successfully" per the M6c contract, and B1/B2
        // reproducibility is checked on hash AND n_decoded equality.
        const request_result * rs[4]  = {&r_a, &r_b1, &r_b2, &r_c};
        const char *           lbl[4] = {"Case A", "Case B1", "Case B2", "Case C"};
        for (int i = 0; i < 4; i++) {
            if (rs[i]->status != request_status::completed) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s status != completed", lbl[i]);
                return fail_and_cleanup(buf);
            }
            if (rs[i]->n_decoded < 1) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s n_decoded < 1", lbl[i]);
                return fail_and_cleanup(buf);
            }
        }
        if (r_a.n_decoded != 8) {
            return fail_and_cleanup("Case A (greedy) n_decoded != 8");
        }
        if (r_c.n_decoded != 8) {
            return fail_and_cleanup(
                "Case C (default stochastic) n_decoded != 8");
        }
        if (r_b1.n_decoded != r_b2.n_decoded) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "Case B1 n_decoded=%d != Case B2 n_decoded=%d "
                "— same non-default config produced different lengths",
                r_b1.n_decoded, r_b2.n_decoded);
            return fail_and_cleanup(buf);
        }

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.admitted_count != 4) {
            return fail_and_cleanup("admitted_count != 4");
        }
        if (er.external_admitted_count != 4) {
            return fail_and_cleanup(
                "external_admitted_count != 4");
        }
        if (er.arrival_drained_count != 4) {
            return fail_and_cleanup(
                "arrival_drained_count != 4");
        }

        // ---- M6b contract: greedy/default produces the canonical
        //      TinyLlama / "Hello, my name is" / greedy / budget-8
        //      fingerprint. A regression in the greedy argmax path
        //      (or in the sampler-chain branch accidentally firing
        //      for greedy slots) would shift this literal. ------------
        constexpr uint64_t k_canonical_greedy_b8 =
            0x0619d4d1900c2365ULL;
        if (r_a.hash != k_canonical_greedy_b8) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "greedy regression: Case A hash 0x%016llx != "
                "canonical 0x%016llx — greedy default must keep the "
                "argmax path bit-identical",
                static_cast<unsigned long long>(r_a.hash),
                static_cast<unsigned long long>(k_canonical_greedy_b8));
            return fail_and_cleanup(buf);
        }

        // ---- M6c contract: default-knob stochastic seed=42 matches
        //      the M6b dist(seed)-only anchor. If this drifts, the
        //      helper's no-op-default skips are wrong, or default-
        //      constructed sampling_config changed. ------------------
        constexpr uint64_t k_default_stochastic_seed42_b8 =
            0xa8e14acb4094aa3fULL;
        if (r_c.hash != k_default_stochastic_seed42_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "M6b-equivalence regression: Case C hash 0x%016llx "
                "!= anchor 0x%016llx — default-knob stochastic must "
                "stay byte-equivalent to the M6b dist(seed)-only chain",
                static_cast<unsigned long long>(r_c.hash),
                static_cast<unsigned long long>(
                    k_default_stochastic_seed42_b8));
            return fail_and_cleanup(buf);
        }

        // ---- M6c contract: B1 and B2 use the same non-default
        //      config; hashes must match (per-seq sampler state
        //      reproduces on slot reuse). ----------------------------
        if (r_b1.hash != r_b2.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "M6c reproducibility regression: Case B1 hash "
                "0x%016llx != Case B2 hash 0x%016llx — same "
                "non-default stochastic config must reproduce",
                static_cast<unsigned long long>(r_b1.hash),
                static_cast<unsigned long long>(r_b2.hash));
            return fail_and_cleanup(buf);
        }

        // ---- M6c contract: non-default knobs must change output
        //      relative to greedy and to default-knob stochastic.
        //      Two distinct anchors guard against coincidence. -------
        if (r_b1.hash == k_canonical_greedy_b8) {
            return fail_and_cleanup(
                "Case B non-default stochastic equals greedy canonical");
        }
        if (r_b1.hash == k_default_stochastic_seed42_b8) {
            return fail_and_cleanup(
                "Case B non-default stochastic equals default-knob "
                "stochastic anchor — knob wiring did not take effect");
        }

        print_result("CaseA ", r_a,  sampling_mode::greedy);
        print_result("CaseB1", r_b1, sampling_mode::stochastic);
        print_result("CaseB2", r_b2, sampling_mode::stochastic);
        print_result("CaseC ", r_c,  sampling_mode::stochastic);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
