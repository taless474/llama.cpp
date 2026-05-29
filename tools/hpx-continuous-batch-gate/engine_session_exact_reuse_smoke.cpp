// engine_session_exact_reuse_smoke.cpp — B1 Slice 3 guard for
// exact-session prefix reuse on the same resident slot.
//
// Slice 3: a later request with the same session_id whose prompt EXACTLY
// extends the resident slot's KV is admitted back onto that slot; the
// matched prefix KV is kept and only the suffix is prefilled.
//
// Shape (single engine, n_seq_max=2, initial_idle_slots=2, keep_alive,
// greedy):
//   r1       : P1="Hello, my name is", session_id="sess-A", budget 8
//              -> completes, slot kept resident.
//   baseline : P2 (no session) on the OTHER idle slot -> FULL prefill.
//              This is the correctness reference for P2.
//   r2       : P2, session_id="sess-A" -> exact-reuse onto r1's resident
//              slot (prefix prefill skipped, suffix prefilled).
//
// P2 is constructed to exactly extend the resident KV. The engine's
// resident_tokens == (P1 ++ r1.generated)[0 : r1.pos_max_at_clear+1] —
// i.e. prompt plus all-but-the-last generated token (the final sampled
// token is never placed in KV). The smoke reconstructs that exact prefix
// from r1's returned generated_tokens + pos_max_at_clear, then appends a
// short new suffix.
//
// Asserts:
//   - r1, baseline, r2 all complete.
//   - exactly one reuse (session_reuse_admitted_count == 1) and the
//     skipped prefix length == the resident length (> 0): prefix prefill
//     was genuinely skipped, not re-done.
//   - r2 output == baseline output (hash + n_decoded): exact reuse
//     matches full prefill.
//   - teardown clears the resident KV (residual_kv_ok == true).
//
// n_seq_max=2 (not 1): in Slice 3 there is no eviction, so r1's resident
// slot is held; the no-session baseline needs a second slot. r2 reuses
// r1's slot directly (no free slot needed).

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
        "  B1 Slice 3 exact-session reuse guard for the\n"
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
    fprintf(stdout, "HPX_SESSION_EXACT_REUSE_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

submit_request build(int32_t rid,
                     const std::vector<llama_token> & prompt,
                     int32_t budget,
                     const std::string & session_id) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = budget;
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
    model = llama_model_load_from_file(args.model_path.c_str(),
                                       model_params);
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 256;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> p1 = common_tokenize(
        ctx, "Hello, my name is", /*add_special=*/true,
        /*parse_special=*/true);
    std::vector<llama_token> suffix = common_tokenize(
        ctx, " The weather today", /*add_special=*/false,
        /*parse_special=*/false);
    if (p1.empty() || suffix.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }

    // batch capacity must cover the largest single prefill (the no-session
    // baseline prefills the whole of P2 in one iter).
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

    constexpr int32_t k_budget = 8;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        // --- r1: session request, leaves a resident slot ---
        submit_handle h1 = eng.submit_request(
            build(/*rid=*/1, p1, k_budget, /*session_id=*/"sess-A"));
        request_result r1 = h1.result.get();
        if (r1.status != request_status::completed) {
            return fail_and_cleanup("r1 status != completed");
        }

        // --- reconstruct the resident prefix exactly ---
        // resident_tokens == (P1 ++ r1.generated)[0 : pos_max_at_clear+1]
        std::vector<llama_token> resident = p1;
        resident.insert(resident.end(),
                        r1.generated_tokens.begin(),
                        r1.generated_tokens.end());
        if (r1.pos_max_at_clear < 0) {
            return fail_and_cleanup("r1 pos_max_at_clear < 0");
        }
        const size_t resident_len =
            static_cast<size_t>(r1.pos_max_at_clear) + 1;
        if (resident_len > resident.size()) {
            return fail_and_cleanup(
                "reconstructed resident length exceeds prompt+generated");
        }
        resident.resize(resident_len);

        // P2 = resident prefix + new suffix (exact extension).
        std::vector<llama_token> p2 = resident;
        p2.insert(p2.end(), suffix.begin(), suffix.end());

        // --- baseline: P2 with NO session -> full prefill reference ---
        submit_handle hb = eng.submit_request(
            build(/*rid=*/2, p2, k_budget, /*session_id=*/""));
        request_result rb = hb.result.get();
        if (rb.status != request_status::completed) {
            return fail_and_cleanup("baseline status != completed");
        }

        // --- r2: P2 with same session -> exact reuse onto r1's slot ---
        submit_handle h2 = eng.submit_request(
            build(/*rid=*/3, p2, k_budget, /*session_id=*/"sess-A"));
        request_result r2 = h2.result.get();
        if (r2.status != request_status::completed) {
            return fail_and_cleanup("r2 status != completed");
        }

        eng.request_shutdown();
        engine_fut.get();

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }

        // exactly one reuse, and it skipped the whole resident prefix.
        if (er.session_reuse_admitted_count != 1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "session_reuse_admitted_count=%d (expected 1)",
                er.session_reuse_admitted_count);
            return fail_and_cleanup(buf);
        }
        if (er.session_reuse_prefill_skipped_tokens
                != static_cast<int64_t>(resident_len)) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "session_reuse_prefill_skipped_tokens=%lld (expected %zu)",
                static_cast<long long>(
                    er.session_reuse_prefill_skipped_tokens),
                resident_len);
            return fail_and_cleanup(buf);
        }
        if (er.session_reuse_prefill_skipped_tokens <= 0) {
            return fail_and_cleanup("skipped prefix tokens not > 0");
        }

        // exact reuse must match the full-prefill baseline output.
        if (r2.n_decoded != rb.n_decoded) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "r2 n_decoded=%d != baseline n_decoded=%d",
                r2.n_decoded, rb.n_decoded);
            return fail_and_cleanup(buf);
        }
        if (r2.hash != rb.hash) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "r2 hash=0x%016llx != baseline hash=0x%016llx "
                "(exact-reuse output diverged from full prefill)",
                static_cast<unsigned long long>(r2.hash),
                static_cast<unsigned long long>(rb.hash));
            return fail_and_cleanup(buf);
        }

        // teardown cleared the resident KV (r2 re-resident slot) and the
        // residual-KV sweep passes -> no leak on shutdown.
        if (er.resident_slots_cleared_at_teardown < 1) {
            return fail_and_cleanup(
                "resident_slots_cleared_at_teardown < 1");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                std::string("residual_kv_ok != true: ")
                + er.residual_kv_error);
        }

        fprintf(stdout,
            "r1 hash=0x%016llx n_decoded=%d pos_max_at_clear=%d\n"
            "resident_len=%zu p2_len=%zu suffix_len=%zu\n"
            "baseline hash=0x%016llx n_decoded=%d\n"
            "r2       hash=0x%016llx n_decoded=%d\n"
            "session_reuse_admitted_count=%d skipped_tokens=%lld\n"
            "resident_slots_created=%d cleared_at_teardown=%d "
            "residual_kv_ok=%d\n",
            static_cast<unsigned long long>(r1.hash), r1.n_decoded,
            r1.pos_max_at_clear,
            resident_len, p2.size(), suffix.size(),
            static_cast<unsigned long long>(rb.hash), rb.n_decoded,
            static_cast<unsigned long long>(r2.hash), r2.n_decoded,
            er.session_reuse_admitted_count,
            static_cast<long long>(
                er.session_reuse_prefill_skipped_tokens),
            er.resident_slots_created,
            er.resident_slots_cleared_at_teardown,
            er.residual_kv_ok ? 1 : 0);
        fprintf(stdout,
            "HPX_SESSION_EXACT_REUSE_SMOKE: ALL_COMPLETED\n");
        fflush(stdout);
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
