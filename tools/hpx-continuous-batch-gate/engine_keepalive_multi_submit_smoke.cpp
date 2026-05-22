// engine_keepalive_multi_submit_smoke.cpp — minimal greedy reproducer
// for the keep-alive 3rd-or-later sequential `submit_request` hang
// exposed during M6b validation.
//
// This smoke is M6b-independent: it uses default greedy sampling
// (no `sampling_config` mutation), `want_stream=false`, no
// streaming, no cancellation, no preloaded actives, no scripted
// arrivals — the bare keep-alive `submit_request` API on a fresh
// engine with `initial_idle_slots=2`.
//
// Shape — matches the M6b stochastic smoke's per-engine shape:
//   - n_seq_max         = 2
//   - initial_idle_slots= 2
//   - keep_alive        = true
//   - reuse_completed   = false   (matches M6b stochastic smoke + carry smoke)
//   - stream_all        = false
//   - decode_budget     = 8
//   - three sequential submit_request calls, awaited one at a time
//
// Hypothesis under test (from the M6b post-mortem):
//   With `reuse_completed=false`, completed slots are NEVER pushed to
//   any reuse pool, because `finalize_and_fulfill` gates the push on
//   `reuse_completed_ && !waiting_queue_consumable_.empty()`
//   (engine.cpp:688). On the strict sequential submit-await pattern,
//   `waiting_queue_consumable_` is always empty at completion time
//   (the next request has not been submitted yet), so even
//   `reuse_completed=true` would not help. `free_idle_` is populated
//   exactly once at engine construction and never repopulated. After
//   `initial_idle_slots` admissions consume `free_idle_`, the third
//   sequential `submit_request` reaches `waiting_queue_consumable_`
//   via `drain_external_inbox` but the admission step finds every
//   reuse pool empty — the inner loop's `if (!any_active())` short-
//   circuits, and the engine cv-waits forever for an inbox/cancel/
//   shutdown event that never arrives. The submitter's
//   `h3.result.get()` therefore blocks forever.
//
// Post-fix path: with the M6b-followup else-if in finalize_and_fulfill
// (engine.cpp:690-695), initial_idle-sourced slots return to free_idle_
// on normal completion. The strict sequential submit-await-submit-await
// pattern can therefore proceed beyond initial_idle_slots. This smoke
// submits three sequential requests and asserts all three complete with
// the canonical greedy hash.

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
        "  Greedy keep-alive 3-sequential-submit reproducer for the\n"
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
    fprintf(stdout,
            "HPX_KEEPALIVE_MULTI_SUBMIT_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
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

    constexpr uint64_t k_canonical_greedy_hash =
        0x0619d4d1900c2365ULL;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        submit_handle h1 = eng.submit_request(
            build_greedy(/*rid=*/1, shared_prompt));
        request_result r1 = h1.result.get();

        submit_handle h2 = eng.submit_request(
            build_greedy(/*rid=*/2, shared_prompt));
        request_result r2 = h2.result.get();

        submit_handle h3 = eng.submit_request(
            build_greedy(/*rid=*/3, shared_prompt));
        request_result r3 = h3.result.get();

        const request_result * rs[3]  = {&r1, &r2, &r3};
        const char *           lbl[3] = {"r1", "r2", "r3"};
        for (int i = 0; i < 3; i++) {
            if (rs[i]->status != request_status::completed) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s status != completed", lbl[i]);
                return fail_and_cleanup(buf);
            }
            if (rs[i]->n_decoded != 8) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s n_decoded != 8", lbl[i]);
                return fail_and_cleanup(buf);
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
            if (rs[i]->admission_src != admission_source::initial_idle) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s admission_src=%s != initial_idle",
                    lbl[i],
                    admission_source_name(rs[i]->admission_src));
                return fail_and_cleanup(buf);
            }
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
        if (er.admitted_count != 3) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "admitted_count=%d (expected 3)", er.admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.external_admitted_count != 3) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "external_admitted_count=%d (expected 3)",
                er.external_admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.arrival_drained_count != 3) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "arrival_drained_count=%d (expected 3)",
                er.arrival_drained_count);
            return fail_and_cleanup(buf);
        }

        for (int i = 0; i < 3; i++) {
            fprintf(stdout,
                "%s hash=0x%016llx n_decoded=%d admission_src=%s\n",
                lbl[i],
                static_cast<unsigned long long>(rs[i]->hash),
                rs[i]->n_decoded,
                admission_source_name(rs[i]->admission_src));
        }
        fprintf(stdout,
            "HPX_KEEPALIVE_MULTI_SUBMIT_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
