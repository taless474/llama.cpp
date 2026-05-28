// engine_chunked_prefill_smoke.cpp — Slice C: gated single-request
// chunked-prefill correctness smoke for the llama-hpx-engine library.
//
// Exercises the live-admission build path (Path A) with
// engine_options::lib.prefill_budget_rows set to several values, driving
// one greedy request per engine through prefill -> decode -> completion.
// The preloaded whole-prompt path (Path B) is NOT exercised here.
//
// Shape (single slot, per case): n_seq_max=1, initial_idle_slots=1,
// budgets={}, reuse_completed=false, keep_alive=true, one submit_request
// (greedy, "Hello, my name is", decode_budget=8). Each case builds its
// OWN llama_context so there is exactly one owner of a context at a time
// and zero KV carryover between cases.
//
// Cases:
//   1. B=0  (unbounded): completed, n_decoded==8, hash==canonical b8.
//   2. B>=n (collapse, B=64): completed, n_decoded==8, hash==canonical b8
//      — proves the one-chunk path matches the old whole-prompt path.
//   3. B=2  (multi-chunk, run twice): completed, n_decoded==8,
//      generated_tokens.size()==8, decode_failures==0, residual_kv_ok,
//      same-B repeat deterministic. Chunked hash MAY differ from b8 and
//      is recorded, not gated against the whole-prompt hash.
//   4. B=4  (multi-chunk, run twice): same gates as B=2.
//
// Chunking is observed via the engine's in-memory Phase-1 diagnostics
// vectors (prefill_rows_per_iter, tokens_emitted_per_iter), which are
// populated only when LLAMA_HPX_DIAG_METRICS=1. The smoke self-enables
// that switch in main() before any engine runs (diag-ON does not perturb
// token output — proven by the Slice B closure). No new diagnostic fields
// are added by this smoke. Chunking gates:
//   - B=2/B=4 split into more than one prefill iter (actual chunking);
//   - every positive prefill-row entry is <= B;
//   - positive prefill-row entries sum to the prompt length;
//   - no token is emitted before the prefill-completing iter;
//   - the prefill-completing iter is the first to emit a token.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and cleanup. All llama execution
// lives inside eng.run() on the single hpx::async task per case. No
// llama_decode / llama_batch_* / llama_memory_seq_* / llama_get_logits_ith
// / common_batch_add call sites exist here.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  Slice C gated single-request chunked-prefill correctness smoke "
        "for the llama-hpx-engine library.\n",
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
    fprintf(stdout, "HPX_ENGINE_CHUNKED_PREFILL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_CHUNKED_PREFILL_SMOKE: PASS\n");
    fflush(stdout);
}

// Bounded readiness poll on an HPX future from main() (foreign OS
// thread). is_ready() is safe from any thread (no suspension).
template <typename Fut>
bool poll_ready(Fut & f, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!f.is_ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

// Outcome of one engine run, snapshotted before the engine is destroyed.
struct run_outcome {
    bool                 ran = false;   // engine ran and joined cleanly
    std::string          err;
    request_status       status          = request_status::completed;
    int32_t              n_decoded       = 0;
    uint64_t             hash            = 0;
    size_t               generated_size  = 0;
    int32_t              decode_failures = 0;
    bool                 residual_kv_ok  = false;
    int32_t              admitted_count  = 0;
    std::vector<int32_t> prefill_rows_per_iter;
    std::vector<int32_t> tokens_emitted_per_iter;
};

constexpr auto k_wait_timeout = std::chrono::milliseconds(15000);
constexpr auto k_join_timeout = std::chrono::milliseconds(8000);

// Run a single greedy request through a fresh engine + context with the
// given per-seq prefill row budget B. Returns a snapshot; on infrastructure
// failure sets out.ran=false with out.err.
run_outcome run_one(llama_model *                    model,
                    const llama_context_params &     base_params,
                    const llama_vocab *              vocab,
                    int32_t                          n_vocab,
                    const std::vector<llama_token> & prompt,
                    int32_t                          B,
                    int32_t                          rid) {
    run_outcome out;

    llama_context * ctx = llama_init_from_model(model, base_params);
    if (ctx == nullptr) {
        out.err = "context create failed";
        return out;
    }

    // batch_capacity is the max single-iter row count, which is the
    // whole-prompt (one-chunk) case; chunked iters place <= B <= prompt
    // rows, so prompt.size() covers every case.
    const int32_t batch_capacity =
        std::max<int32_t>(static_cast<int32_t>(prompt.size()), 1);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 1;
    opts.lib.initial_idle_slots   = 1;
    opts.lib.keep_alive           = true;
    opts.lib.prefill_budget_rows  = B;            // Slice C knob under test
    opts.preload.prompt_tokens    = &prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        submit_request req;
        req.request_id    = rid;
        req.prompt_tokens = prompt;
        req.decode_budget = 8;
        req.want_stream   = false;
        submit_handle h = eng.submit_request(std::move(req));

        if (!poll_ready(h.result, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            llama_free(ctx);
            out.err = "result not ready within timeout";
            return out;
        }
        request_result rr = h.result.get();

        eng.request_shutdown();
        if (!poll_ready(engine_fut, k_join_timeout)) {
            // keep-alive engine failed to join — avoid a hang.
            llama_free(ctx);
            out.err = "engine did not join after request_shutdown";
            return out;
        }
        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            llama_free(ctx);
            out.err = std::string("engine task threw on join: ") + e.what();
            return out;
        }

        const engine_result & er = eng.result();
        out.status          = rr.status;
        out.n_decoded       = rr.n_decoded;
        out.hash            = rr.hash;
        out.generated_size  = rr.generated_tokens.size();
        out.decode_failures = er.decode_failures;
        out.residual_kv_ok  = er.residual_kv_ok;
        out.admitted_count  = er.admitted_count;
        out.prefill_rows_per_iter   = er.metrics.prefill_rows_per_iter;
        out.tokens_emitted_per_iter = er.metrics.tokens_emitted_per_iter;
        out.ran = true;
    } catch (const std::exception & e) {
        llama_free(ctx);
        out.err = std::string("exception: ") + e.what();
        return out;
    } catch (...) {
        llama_free(ctx);
        out.err = "unknown exception";
        return out;
    }

    llama_free(ctx);
    return out;
}

// ---- prefill-row vector helpers (over engine diagnostics) ---------------

int32_t count_positive(const std::vector<int32_t> & v) {
    int32_t c = 0;
    for (int32_t x : v) if (x > 0) c++;
    return c;
}

int64_t sum_all(const std::vector<int32_t> & v) {
    int64_t s = 0;
    for (int32_t x : v) s += x;
    return s;
}

int32_t last_positive_index(const std::vector<int32_t> & v) {
    int32_t idx = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(v.size()); i++) {
        if (v[i] > 0) idx = i;
    }
    return idx;
}

int32_t max_positive(const std::vector<int32_t> & v) {
    int32_t m = 0;
    for (int32_t x : v) if (x > m) m = x;
    return m;
}

std::string chunk_summary(const std::vector<int32_t> & v) {
    std::string s = "[";
    bool first = true;
    for (int32_t x : v) {
        if (x <= 0) continue;
        if (!first) s += ",";
        s += std::to_string(x);
        first = false;
    }
    s += "]";
    return s;
}

// Common structural checks for any completed run. Returns "" on success
// or a failure reason.
std::string check_completed(const run_outcome & o, const char * label) {
    if (!o.ran) return std::string(label) + ": infra: " + o.err;
    if (o.status != request_status::completed) {
        return std::string(label) + ": status != completed";
    }
    if (o.n_decoded != 8) return std::string(label) + ": n_decoded != 8";
    if (o.generated_size != 8) {
        return std::string(label) + ": generated_tokens.size() != 8";
    }
    if (o.decode_failures != 0) {
        return std::string(label) + ": decode_failures != 0";
    }
    if (!o.residual_kv_ok) {
        return std::string(label) + ": residual_kv_ok != true";
    }
    if (o.prefill_rows_per_iter.empty()) {
        return std::string(label) +
               ": prefill diagnostics empty (LLAMA_HPX_DIAG_METRICS off?)";
    }
    return "";
}

// Verify the prefill-row split matches expectations for budget B over a
// prompt of length n. multi==true requires more than one prefill iter and
// every chunk <= B; multi==false (collapse) requires exactly one prefill
// iter equal to n. Also verifies no token before the prefill-completing
// iter, and a token on that iter. Returns "" on success.
std::string check_prefill_split(const run_outcome & o, int32_t B,
                                int32_t n, bool multi, const char * label) {
    const int32_t pf_iters = count_positive(o.prefill_rows_per_iter);
    if (sum_all(o.prefill_rows_per_iter) != n) {
        return std::string(label) + ": prefill rows sum " +
               std::to_string(sum_all(o.prefill_rows_per_iter)) +
               " != prompt length " + std::to_string(n);
    }
    if (multi) {
        if (pf_iters <= 1) {
            return std::string(label) +
                   ": expected multi-chunk but prefill_iters=" +
                   std::to_string(pf_iters) +
                   " (prompt too short for B=" + std::to_string(B) + "?)";
        }
        if (max_positive(o.prefill_rows_per_iter) > B) {
            return std::string(label) + ": a prefill chunk exceeds B=" +
                   std::to_string(B);
        }
    } else {
        if (pf_iters != 1) {
            return std::string(label) +
                   ": expected single (collapsed) prefill chunk but "
                   "prefill_iters=" + std::to_string(pf_iters);
        }
        if (max_positive(o.prefill_rows_per_iter) != n) {
            return std::string(label) +
                   ": collapsed chunk != prompt length";
        }
    }
    // No token before prefill completes; first token on the completing iter.
    const int32_t last_pf = last_positive_index(o.prefill_rows_per_iter);
    for (int32_t i = 0; i < last_pf; i++) {
        if (i < static_cast<int32_t>(o.tokens_emitted_per_iter.size())
         && o.tokens_emitted_per_iter[i] != 0) {
            return std::string(label) +
                   ": token emitted at iter " + std::to_string(i) +
                   " before prefill completed (iter " +
                   std::to_string(last_pf) + ")";
        }
    }
    if (last_pf >= 0
     && last_pf < static_cast<int32_t>(o.tokens_emitted_per_iter.size())
     && o.tokens_emitted_per_iter[last_pf] < 1) {
        return std::string(label) +
               ": no token emitted on the prefill-completing iter";
    }
    return "";
}

void report(const char * label, int32_t B, const run_outcome & o) {
    fprintf(stdout,
        "%-10s B=%-3d status=%s n_decoded=%d gen=%zu hash=0x%016llx "
        "decode_failures=%d residual_kv_ok=%d prefill_chunks=%s\n",
        label, B,
        (o.status == request_status::completed) ? "completed" : "other",
        o.n_decoded, o.generated_size,
        static_cast<unsigned long long>(o.hash),
        o.decode_failures, o.residual_kv_ok ? 1 : 0,
        chunk_summary(o.prefill_rows_per_iter).c_str());
    fflush(stdout);
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    // Self-enable Phase-1 diagnostics so the engine populates the in-memory
    // per-iter vectors this smoke reads to verify chunking. Set BEFORE any
    // engine runs (diag_enabled() caches on first use). Diag-ON does not
    // change token output (Slice B closure). Per-iter JSONL is emitted to
    // stderr (harmless; redirected by the runner).
    setenv("LLAMA_HPX_DIAG_METRICS", "1", /*overwrite=*/1);

    trace::init();

    if (!hpx_runtime::start_once(/*os_threads=*/1)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model * model = nullptr;

    auto cleanup_and_stop = [&](int rc) -> int {
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
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 1;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;

    // Tokenize once with a throwaway context (tokens are ctx-independent).
    std::vector<llama_token> prompt;
    {
        llama_context * tok_ctx = llama_init_from_model(model, ctx_params);
        if (tok_ctx == nullptr) {
            return fail_and_cleanup("tokenization context create failed");
        }
        prompt = common_tokenize(tok_ctx, "Hello, my name is",
                                 /*add_special=*/true, /*parse_special=*/true);
        llama_free(tok_ctx);
    }
    if (prompt.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }
    const int32_t n = static_cast<int32_t>(prompt.size());
    fprintf(stdout, "prompt_len=%d\n", n);
    fflush(stdout);

    constexpr uint64_t k_b8 = 0x0619d4d1900c2365ULL;
    constexpr int32_t  k_collapse_B = 64;   // >= prompt length
    constexpr int32_t  k_chunk_B1   = 2;
    constexpr int32_t  k_chunk_B2   = 4;

    if (n <= k_chunk_B2) {
        return fail_and_cleanup(
            "prompt too short to exercise B=4 multi-chunk; "
            "expected canonical prompt length > 4");
    }
    if (k_collapse_B < n) {
        return fail_and_cleanup(
            "collapse B is not >= prompt length; adjust k_collapse_B");
    }

    // ---- Case 1: B=0 unbounded -> canonical b8 ----------------------------
    run_outcome u0 = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             /*B=*/0, /*rid=*/1);
    report("unbounded", 0, u0);
    {
        const std::string e = check_completed(u0, "B=0");
        if (!e.empty()) return fail_and_cleanup(e);
        if (u0.hash != k_b8) return fail_and_cleanup("B=0 hash != canonical b8");
        const std::string s = check_prefill_split(u0, /*B=*/0, n,
                                                  /*multi=*/false, "B=0");
        if (!s.empty()) return fail_and_cleanup(s);
    }

    // ---- Case 2: B>=n collapse -> canonical b8 ----------------------------
    run_outcome uc = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             /*B=*/k_collapse_B, /*rid=*/2);
    report("collapse", k_collapse_B, uc);
    {
        const std::string e = check_completed(uc, "B>=n");
        if (!e.empty()) return fail_and_cleanup(e);
        if (uc.hash != k_b8) {
            return fail_and_cleanup("B>=n collapse hash != canonical b8");
        }
        const std::string s = check_prefill_split(uc, k_collapse_B, n,
                                                  /*multi=*/false, "B>=n");
        if (!s.empty()) return fail_and_cleanup(s);
    }

    // ---- Case 3: B=2 multi-chunk, run twice (determinism) -----------------
    run_outcome a2 = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             k_chunk_B1, /*rid=*/3);
    run_outcome b2 = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             k_chunk_B1, /*rid=*/4);
    report("chunk", k_chunk_B1, a2);
    report("chunk", k_chunk_B1, b2);
    {
        std::string e = check_completed(a2, "B=2/run1");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_completed(b2, "B=2/run2");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_prefill_split(a2, k_chunk_B1, n, /*multi=*/true, "B=2/run1");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_prefill_split(b2, k_chunk_B1, n, /*multi=*/true, "B=2/run2");
        if (!e.empty()) return fail_and_cleanup(e);
        if (a2.hash != b2.hash) {
            return fail_and_cleanup("B=2 not deterministic across repeats");
        }
        fprintf(stdout,
            "B=2 deterministic hash=0x%016llx (differs_from_b8=%d, allowed)\n",
            static_cast<unsigned long long>(a2.hash),
            (a2.hash != k_b8) ? 1 : 0);
        fflush(stdout);
    }

    // ---- Case 4: B=4 multi-chunk, run twice (determinism) -----------------
    run_outcome a4 = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             k_chunk_B2, /*rid=*/5);
    run_outcome b4 = run_one(model, ctx_params, vocab, n_vocab, prompt,
                             k_chunk_B2, /*rid=*/6);
    report("chunk", k_chunk_B2, a4);
    report("chunk", k_chunk_B2, b4);
    {
        std::string e = check_completed(a4, "B=4/run1");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_completed(b4, "B=4/run2");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_prefill_split(a4, k_chunk_B2, n, /*multi=*/true, "B=4/run1");
        if (!e.empty()) return fail_and_cleanup(e);
        e = check_prefill_split(b4, k_chunk_B2, n, /*multi=*/true, "B=4/run2");
        if (!e.empty()) return fail_and_cleanup(e);
        if (a4.hash != b4.hash) {
            return fail_and_cleanup("B=4 not deterministic across repeats");
        }
        fprintf(stdout,
            "B=4 deterministic hash=0x%016llx (differs_from_b8=%d, allowed)\n",
            static_cast<unsigned long long>(a4.hash),
            (a4.hash != k_b8) ? 1 : 0);
        fflush(stdout);
    }

    emit_pass();
    return cleanup_and_stop(0);
}
