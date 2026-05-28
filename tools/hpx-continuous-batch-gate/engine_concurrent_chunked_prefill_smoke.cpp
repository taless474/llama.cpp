// engine_concurrent_chunked_prefill_smoke.cpp — Slice E: concurrent
// chunked prefill under n_seq_max=2 for the llama-hpx-engine library.
//
// Primary question: can one sequence (A) decode while another live-admitted
// sequence (B) does chunked prefill in the SAME llama_decode, without
// corrupting A? Correctness rests on per-seq causal masking: A's decode row
// attends only to A's KV, B's prefill rows only to B's KV.
//
// Deterministic shape (single engine, fresh context per scenario):
//   n_seq_max=2, initial_idle_slots=2, budgets={}, keep_alive=true,
//   prefill_budget_rows=B(=2). Submit A (budget 8) BEFORE run() so it is
//   admitted in iter 1; A chunk-prefills iters 1..K where K=ceil(n/B) (the
//   iter A completes prefill AND samples token 0). A gate barrier parks the
//   engine at end of iter K (A now decoding). The submitter then submits B
//   (budget 8) and releases the barrier; B admits in iter K+1 and
//   chunk-prefills iters K+1.. WHILE A decodes -> overlap.
//
// Baselines (avoid cross-shape hash gates — CLAUDE.md):
//   * within-shape repeat determinism: identical scenario twice -> A,B
//     hashes match;
//   * cross-seq content isolation (the masking test): SAME batch shape,
//     vary the OTHER seq's prompt content at the SAME token length ->
//     this seq's hash MUST be identical. Solo-vs-concurrent is NOT gated.
//
// Tightening 1: content variants are natural-language prompts of verified
// equal token length (no hand-crafted token IDs); fail early on mismatch.
// Tightening 2: A submitted before run(); barrier landing validated by
// metrics (A prefill sum == n AND tokens-emitted == 1) before B is sent.
// Tightening 3: prefill_rows_per_iter is AGGREGATE; A-prefill is snapshotted
// at the barrier and B-prefill is attributed as total-minus-A, with the
// B-range chunks asserted <= B; unexpected co-prefill shapes fail loudly.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and cleanup. All llama execution
// lives inside eng.run() on the single hpx::async task per scenario.

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

constexpr int32_t k_B      = 2;     // prefill row budget under test
constexpr int32_t k_budget = 8;     // decode budget for A and B

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  Slice E concurrent chunked-prefill correctness smoke for the "
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
    fprintf(stdout, "HPX_ENGINE_CONCURRENT_CHUNKED_PREFILL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_CONCURRENT_CHUNKED_PREFILL_SMOKE: PASS\n");
    fflush(stdout);
}

template <typename Fut>
bool poll_ready(Fut & f, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!f.is_ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

int64_t sum_vec(const std::vector<int32_t> & v) {
    int64_t s = 0;
    for (int32_t x : v) s += x;
    return s;
}

std::string vec_to_str(const std::vector<int32_t> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }

constexpr auto k_wait_timeout = std::chrono::milliseconds(20000);
constexpr auto k_join_timeout = std::chrono::milliseconds(8000);

// Per-scenario outcome, snapshotted before the engine is destroyed.
struct scenario {
    bool                 ok = false;
    std::string          err;
    // request A / B results
    request_status       a_status = request_status::completed;
    request_status       b_status = request_status::completed;
    int32_t              a_ndec = 0, b_ndec = 0;
    size_t               a_gen  = 0, b_gen  = 0;
    uint64_t             a_hash = 0, b_hash = 0;
    // engine invariants
    int32_t              decode_failures = 0;
    int32_t              cancelled_count = 0;
    bool                 residual_kv_ok  = false;
    // prefill attribution + overlap
    int32_t              barrier_len   = 0;
    int64_t              a_prefill_sum = 0;
    int64_t              total_prefill = 0;
    int64_t              b_prefill_sum = 0;
    bool                 overlap = false;
    int32_t              overlap_iter = -1;
    std::vector<int32_t> pf, dr, act, tok;   // final per-iter vectors
};

// Run one concurrent scenario on a fresh engine + context. pA/pB must have
// equal token length n; K=ceil(n/B). On infrastructure or shape failure
// sets s.ok=false with s.err and (where useful) the per-iter vectors.
scenario run_scenario(llama_model *                    model,
                      const llama_context_params &     base_params,
                      const llama_vocab *              vocab,
                      int32_t                          n_vocab,
                      const std::vector<llama_token> & pA,
                      const std::vector<llama_token> & pB,
                      int32_t                          n,
                      int32_t                          K) {
    scenario s;

    llama_context * ctx = llama_init_from_model(model, base_params);
    if (ctx == nullptr) { s.err = "context create failed"; return s; }

    // Max single-iter rows is A decode (1) + B prefill chunk (<=B); 2*n is
    // a generous upper bound that also covers any whole-batch shape.
    const int32_t batch_capacity = std::max<int32_t>(2 * n + 2, 1);
    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 2;
    opts.lib.initial_idle_slots   = 2;
    opts.lib.keep_alive           = true;
    opts.lib.prefill_budget_rows  = k_B;
    opts.preload.prompt_tokens    = &pA;       // unused (budgets empty)
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.release_iter_set = { K };
    opts.gate_test.max_decode_iters = K;       // sizes release-promise vectors

    auto free_ctx = [&]() { if (ctx) { llama_free(ctx); ctx = nullptr; } };

    try {
        engine eng(std::move(opts));

        // Tightening 2: submit A BEFORE the engine task is scheduled so it
        // is admitted in iter 1; register the barrier before run() too.
        submit_request reqA;
        reqA.request_id    = 1;
        reqA.prompt_tokens = pA;
        reqA.decode_budget = k_budget;
        reqA.want_stream   = false;
        submit_handle hA = eng.submit_request(std::move(reqA));

        external_release_handle rel = eng.register_external_release_iter(K);

        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        if (!poll_ready(rel.release_future, k_wait_timeout)) {
            free_ctx();
            s.err = "release barrier at K did not fire";
            return s;
        }
        rel.release_future.get();

        // Tightening 2 validation: at the barrier (end of iter K) A must
        // have completed prefill (prefill rows == n) and sampled exactly
        // one token (tokens emitted == 1). Race-safe: these iter-1..K
        // writes happen-before the release set_value.
        const std::vector<int32_t> pf_barrier =
            eng.result().metrics.prefill_rows_per_iter;
        const std::vector<int32_t> tok_barrier =
            eng.result().metrics.tokens_emitted_per_iter;
        const int64_t pf_b  = sum_vec(pf_barrier);
        const int64_t tok_b = sum_vec(tok_barrier);
        s.barrier_len   = static_cast<int32_t>(pf_barrier.size());
        s.a_prefill_sum = pf_b;
        if (pf_b != n || tok_b != 1) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            free_ctx();
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "barrier K=%d did not land on A-decoding state: "
                "prefill_sum=%lld (want %d) tokens_emitted=%lld (want 1) "
                "pf=%s tok=%s",
                K, static_cast<long long>(pf_b), n,
                static_cast<long long>(tok_b),
                vec_to_str(pf_barrier).c_str(),
                vec_to_str(tok_barrier).c_str());
            s.err = buf;
            return s;
        }

        // Submit B (overlaps A's decode), then release the engine.
        submit_request reqB;
        reqB.request_id    = 2;
        reqB.prompt_tokens = pB;
        reqB.decode_budget = k_budget;
        reqB.want_stream   = false;
        submit_handle hB = eng.submit_request(std::move(reqB));

        rel.ack_promise.set_value();

        if (!poll_ready(hA.result, k_wait_timeout)
         || !poll_ready(hB.result, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            free_ctx();
            s.err = "A/B result not ready within timeout";
            return s;
        }
        request_result rA = hA.result.get();
        request_result rB = hB.result.get();

        eng.request_shutdown();
        if (!poll_ready(engine_fut, k_join_timeout)) {
            emit_fail("engine did not join after request_shutdown");
            fflush(stdout); fflush(stderr);
            _Exit(2);
        }
        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            free_ctx();
            s.err = std::string("engine task threw on join: ") + e.what();
            return s;
        }

        const engine_result & er = eng.result();
        s.a_status = rA.status; s.b_status = rB.status;
        s.a_ndec = rA.n_decoded; s.b_ndec = rB.n_decoded;
        s.a_gen  = rA.generated_tokens.size();
        s.b_gen  = rB.generated_tokens.size();
        s.a_hash = rA.hash; s.b_hash = rB.hash;
        s.decode_failures = er.decode_failures;
        s.cancelled_count = er.cancelled_count;
        s.residual_kv_ok  = er.residual_kv_ok;
        s.pf  = er.metrics.prefill_rows_per_iter;
        s.dr  = er.metrics.decode_rows_per_iter;
        s.act = er.metrics.active_seqs_per_iter;
        s.tok = er.metrics.tokens_emitted_per_iter;
        s.total_prefill = sum_vec(s.pf);
        s.b_prefill_sum = s.total_prefill - s.a_prefill_sum;

        // Tightening 3: attribute B-prefill from the post-barrier range and
        // assert each B chunk <= B and the range sums to n. A prefill is in
        // [0, barrier_len); B prefill is in [barrier_len, end).
        int64_t b_range = 0;
        for (int32_t i = s.barrier_len;
             i < static_cast<int32_t>(s.pf.size()); i++) {
            if (s.pf[i] > 0) {
                if (s.pf[i] > k_B) {
                    free_ctx();
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B prefill chunk %d at iter-index %d exceeds B=%d; "
                        "pf=%s", s.pf[i], i, k_B, vec_to_str(s.pf).c_str());
                    s.err = buf;
                    return s;
                }
                b_range += s.pf[i];
            }
        }
        if (b_range != n || s.b_prefill_sum != n
         || s.total_prefill != 2 * static_cast<int64_t>(n)) {
            free_ctx();
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "unexpected prefill partition: a_prefill=%lld b_range=%lld "
                "b_attr=%lld total=%lld want a=%d b=%d total=%d pf=%s",
                static_cast<long long>(s.a_prefill_sum),
                static_cast<long long>(b_range),
                static_cast<long long>(s.b_prefill_sum),
                static_cast<long long>(s.total_prefill),
                n, n, 2 * n, vec_to_str(s.pf).c_str());
            s.err = buf;
            return s;
        }

        // Overlap: an iter with B prefill rows AND A decode rows AND two
        // active seqs.
        for (int32_t i = 0; i < static_cast<int32_t>(s.pf.size()); i++) {
            const bool has_pf  = s.pf[i] > 0;
            const bool has_dr  = (i < (int32_t)s.dr.size()) && s.dr[i] > 0;
            const bool two_act = (i < (int32_t)s.act.size()) && s.act[i] == 2;
            if (has_pf && has_dr && two_act) {
                s.overlap = true; s.overlap_iter = i; break;
            }
        }

        free_ctx();
        s.ok = true;
        return s;
    } catch (const std::exception & e) {
        free_ctx();
        s.err = std::string("exception: ") + e.what();
        return s;
    } catch (...) {
        free_ctx();
        s.err = "unknown exception";
        return s;
    }
}

// Per-scenario structural gates (independent of cross-scenario hashes).
std::string check_gates(const scenario & s, int32_t n, const char * label) {
    if (!s.ok) return std::string(label) + ": infra/shape: " + s.err;
    if (s.a_status != request_status::completed)
        return std::string(label) + ": A status != completed";
    if (s.b_status != request_status::completed)
        return std::string(label) + ": B status != completed";
    if (s.a_ndec != k_budget) return std::string(label) + ": A n_decoded != budget";
    if (s.b_ndec != k_budget) return std::string(label) + ": B n_decoded != budget";
    if (s.a_gen != (size_t)k_budget) return std::string(label) + ": A gen != budget";
    if (s.b_gen != (size_t)k_budget) return std::string(label) + ": B gen != budget";
    if (s.decode_failures != 0) return std::string(label) + ": decode_failures != 0";
    if (s.cancelled_count != 0) return std::string(label) + ": cancelled_count != 0";
    if (!s.residual_kv_ok) return std::string(label) + ": residual_kv_ok != true";
    if (!s.overlap)
        return std::string(label) +
               ": no overlap iter (prefill>0 & decode>0 & active==2); pf=" +
               vec_to_str(s.pf) + " dr=" + vec_to_str(s.dr) +
               " act=" + vec_to_str(s.act);
    (void)n;
    return "";
}

void report_scenario(const char * label, const scenario & s) {
    fprintf(stdout,
        "%-10s A{status=%s n=%d hash=0x%016llx} B{status=%s n=%d "
        "hash=0x%016llx} overlap_iter=%d a_prefill=%lld b_prefill=%lld "
        "decode_failures=%d residual_kv_ok=%d\n",
        label,
        (s.a_status == request_status::completed) ? "completed" : "other",
        s.a_ndec, static_cast<unsigned long long>(s.a_hash),
        (s.b_status == request_status::completed) ? "completed" : "other",
        s.b_ndec, static_cast<unsigned long long>(s.b_hash),
        s.overlap_iter,
        static_cast<long long>(s.a_prefill_sum),
        static_cast<long long>(s.b_prefill_sum),
        s.decode_failures, s.residual_kv_ok ? 1 : 0);
    fflush(stdout);
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

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
    if (model == nullptr) return fail_and_cleanup("model load failed");
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;

    // Tightening 1: P and Q are natural-language prompts of VERIFIED equal
    // token length (consistent BOS/special handling). Q is chosen from
    // single-word variants of P; we fail early if none matches P's length.
    auto tok = [&](const char * s) -> std::vector<llama_token> {
        llama_context * c = llama_init_from_model(model, ctx_params);
        std::vector<llama_token> t;
        if (c) {
            t = common_tokenize(c, s, /*add_special=*/true,
                                /*parse_special=*/true);
            llama_free(c);
        }
        return t;
    };

    const std::vector<llama_token> pP = tok("Hello, my name is");
    if (pP.empty()) return fail_and_cleanup("tokenization of P produced 0 tokens");
    const int32_t n = static_cast<int32_t>(pP.size());
    if (n <= k_B) return fail_and_cleanup("prompt length must exceed B");

    const char * q_candidates[] = {
        "Hello, your name is",
        "Hello, her name is",
        "Hello, our name is",
        "Hello, the name is",
    };
    std::vector<llama_token> pQ;
    std::string q_str;
    for (const char * cand : q_candidates) {
        std::vector<llama_token> t = tok(cand);
        if (static_cast<int32_t>(t.size()) == n) { pQ = std::move(t); q_str = cand; break; }
    }
    if (pQ.empty()) {
        return fail_and_cleanup(
            "no Q variant matched P's token length; adjust prompt strings");
    }

    const int32_t K = ceil_div(n, k_B);
    fprintf(stdout,
        "P=\"Hello, my name is\" (len=%d) Q=\"%s\" (len=%d) B=%d budget=%d K=%d\n",
        n, q_str.c_str(), static_cast<int32_t>(pQ.size()), k_B, k_budget, K);
    fflush(stdout);

    // ---- Scenarios -------------------------------------------------------
    // base/rep: A=P,B=P (determinism). Bvar: A=P,B=Q (vary B content ->
    // A hash unchanged). Avar: A=Q,B=P (vary A content -> B hash unchanged).
    scenario base = run_scenario(model, ctx_params, vocab, n_vocab, pP, pP, n, K);
    report_scenario("base", base);
    { const std::string e = check_gates(base, n, "base"); if (!e.empty()) return fail_and_cleanup(e); }

    scenario rep = run_scenario(model, ctx_params, vocab, n_vocab, pP, pP, n, K);
    report_scenario("repeat", rep);
    { const std::string e = check_gates(rep, n, "repeat"); if (!e.empty()) return fail_and_cleanup(e); }

    scenario bvar = run_scenario(model, ctx_params, vocab, n_vocab, pP, pQ, n, K);
    report_scenario("Bvar", bvar);
    { const std::string e = check_gates(bvar, n, "Bvar"); if (!e.empty()) return fail_and_cleanup(e); }

    scenario avar = run_scenario(model, ctx_params, vocab, n_vocab, pQ, pP, n, K);
    report_scenario("Avar", avar);
    { const std::string e = check_gates(avar, n, "Avar"); if (!e.empty()) return fail_and_cleanup(e); }

    // ---- Determinism (same shape, repeated) ------------------------------
    if (base.a_hash != rep.a_hash)
        return fail_and_cleanup("A hash not deterministic across same-shape repeats");
    if (base.b_hash != rep.b_hash)
        return fail_and_cleanup("B hash not deterministic across same-shape repeats");
    fprintf(stdout, "determinism: A=0x%016llx B=0x%016llx (base==repeat)\n",
        static_cast<unsigned long long>(base.a_hash),
        static_cast<unsigned long long>(base.b_hash));
    fflush(stdout);

    // ---- Content isolation (same shape, other seq's content varied) ------
    // Bvar: A=P unchanged, B content P->Q -> A hash MUST equal base A hash.
    if (bvar.a_hash != base.a_hash) {
        return fail_and_cleanup(
            "cross-seq contamination: A hash changed when only B content "
            "varied (per-seq masking violated)");
    }
    // Avar: B=P unchanged, A content P->Q -> B hash MUST equal base B hash.
    if (avar.b_hash != base.b_hash) {
        return fail_and_cleanup(
            "cross-seq contamination: B hash changed when only A content "
            "varied (per-seq masking violated)");
    }
    fprintf(stdout,
        "isolation: A invariant to B content (0x%016llx); B invariant to A "
        "content (0x%016llx); Bvar.B=0x%016llx Avar.A=0x%016llx (expected differ)\n",
        static_cast<unsigned long long>(base.a_hash),
        static_cast<unsigned long long>(base.b_hash),
        static_cast<unsigned long long>(bvar.b_hash),
        static_cast<unsigned long long>(avar.a_hash));
    fflush(stdout);

    emit_pass();
    return cleanup_and_stop(0);
}
