// engine_w1c_policy_smoke.cpp — Slice F (Option A): engine-only
// W1c-style PrefillBudgetPolicy comparison harness for the
// llama-hpx-engine library.
//
// NOT an apples-to-apples rerun of the original server-driven W1c.
// This is an engine-only harness that drives the same d2-staggered
// SHAPE — one anchor sequence decoding while a long-prefill probe is
// admitted — across `prefill_budget_rows` values B in {0,32,64,128}.
// The probe classes (L8/L64/L256/L1024) mirror the W1c driver's
// prompts. Per-B fresh `llama_context` per scenario, no cross-scenario
// KV carryover.
//
// Deterministic stagger via the existing release/ack barrier
// (gate_test_options::release_iter_set):
//   1. Submit anchor BEFORE the engine task is scheduled so it is
//      admitted in iter 1.
//   2. Register the release barrier at iter K=1 (the canonical anchor
//      "Hello, my name is" fits in one prefill chunk for every B
//      tested — its tokenized length is ~6 and min(B)=32 > 6 and B=0
//      collapses to whole-prompt).
//   3. After the release fires, validate at the barrier that the
//      anchor has prefilled (prefill_sum == anchor_len) and emitted
//      its first token (tokens_emitted_sum == 1). Fail loudly if not.
//   4. Submit the probe (long prompt), then ack the release. The
//      engine resumes at iter 2: admits the probe; with B=0 the
//      probe's whole prompt lands in one iter beside the anchor
//      decode, with B>0 the probe chunks across multiple iters.
//   5. Await both per-request completions; shutdown the engine
//      cleanly; snapshot metrics.
//
// Required interference cell (FAIL LOUDLY if missing):
//   prefill_rows_in_iter > 0 AND decode_rows_in_iter > 0 AND
//   active_seq_count == 2 for at least one post-barrier iter.
//
// Diag JSONL: this binary sets LLAMA_HPX_DIAG_METRICS=1 unconditionally.
// `--out-dir` is required and writes one JSONL per --B run:
//
//   <out-dir>/diag-B<n>.jsonl
//
// Schema (one JSON object per line):
//   - {"kind":"run", "B":...} once at top
//   - per scenario:
//       {"kind":"scenario_begin", "B":..., "class":"L1024",
//        "cycle":..., "anchor_len":..., "probe_len":...}
//       {"kind":"iter", ...}              (engine, one per engine iter)
//       {"kind":"engine_summary", ...}    (engine, one per scenario)
//       {"kind":"scenario_end", "B":..., "class":"L1024", "cycle":...,
//        "anchor_status":..., "anchor_n_decoded":...,
//        "anchor_hash":"0x...", "probe_status":..., "probe_n_decoded":...,
//        "probe_hash":"0x...", "decode_failures":...,
//        "cancelled_count":..., "residual_kv_ok":...,
//        "found_interference":1, "max_pf_post_barrier":..,
//        "interference_iter_count":..}
//
// The diag path is captured by the engine via a `static const` on
// first call inside `diag_path()`, so a single invocation of this
// binary uses one file for all scenarios; per-B invocations create
// separate files.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and cleanup. All llama
// execution lives inside `eng.run()` on the single hpx::async task per
// scenario. No `llama_decode` / `llama_batch_*` / `llama_memory_seq_*`
// / `llama_get_logits_ith` / `common_batch_add` call sites exist here.

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

constexpr int32_t k_anchor_budget = 8;
constexpr int32_t k_probe_budget  = 8;
constexpr int32_t k_default_cycles = 4;

constexpr auto k_wait_timeout = std::chrono::milliseconds(60000);
constexpr auto k_join_timeout = std::chrono::milliseconds(10000);

const char * const k_anchor_prompt = "Hello, my name is";

// W1c paragraphs (copied verbatim from
// local/runs/w1c-staggered-prefill-2026-05-26/driver.py so prompt-class
// token counts match the original baseline classes L8/L64/L256/L1024).
const char * const k_P1 =
    "The early morning sun rose slowly above the eastern hills "
    "casting long shadows across the quiet valley below where the "
    "river flowed gently toward the sea. A flock of birds called "
    "out in the cool air as the village began to wake. Smoke "
    "started to rise from chimneys and the smell of fresh bread "
    "drifted along the narrow cobblestone streets. Children ran "
    "to school with bags on their backs while older folk sat on "
    "wooden benches and watched the day begin. Far away, beyond "
    "the fields and the orchards, a single train whistle echoed.";
const char * const k_P2 =
    "By mid-morning the market square was full of voices, "
    "vendors calling out prices, customers haggling over baskets "
    "of fruit, vegetables, cheese, and dried fish. The square "
    "had been the heart of the town for generations, and every "
    "stone in its pavement had been worn smooth by countless "
    "footsteps. A traveling musician played a slow tune on a "
    "wooden flute under the shade of an old elm tree while two "
    "boys sat beside him listening with wide eyes. The smell of "
    "roasting chestnuts and warm cinnamon hung in the air.";
const char * const k_P3 =
    "The afternoon brought a soft warm wind from the south "
    "that rustled the leaves and turned the heat of the day "
    "into something gentler. Old fishermen mended nets on the "
    "stone quay while gulls wheeled and cried above the boats. "
    "A young woman in a blue dress walked along the harbor with "
    "a small dog at her heels and stopped now and then to look "
    "out at the calm water. The lighthouse on the far point "
    "stood tall and white against the deep green of the distant "
    "headland and the soft pale blue of the open sky.";
const char * const k_P4 =
    "Evening came slowly with a sky of gold and orange that "
    "faded to a deep dark blue as the first stars appeared one "
    "by one above the rooftops. Lamps were lit in every window "
    "and the streets filled with the low murmur of families "
    "gathering for the evening meal. The river caught the last "
    "of the light and shone for a moment like a ribbon of fire "
    "before turning quiet and dark again. From far away the "
    "single train whistle echoed once more across the valley "
    "and the long day at last drew gently to a close.";

const char * const k_L8 =
    "The early morning sun rose above the eastern hills.";

// L64 ~ first three sentences of P1 (~68 tokens). Same string as
// W1c driver.py PROMPT_L64.
const char * const k_L64 =
    "The early morning sun rose slowly above the eastern hills "
    "casting long shadows across the quiet valley below where the "
    "river flowed gently toward the sea. A flock of birds called "
    "out in the cool air as the village began to wake. Smoke "
    "started to rise from chimneys and the smell of fresh bread "
    "drifted along the narrow cobblestone streets.";

struct prompt_class {
    const char * name;
    std::string  text;
};

struct smoke_args {
    std::string model_path;
    std::string out_dir;
    int32_t     B      = -1;
    int32_t     cycles = k_default_cycles;
    std::string only_class;     // empty = all
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> --B <int> --out-dir <dir> "
        "[--class L8|L64|L256|L1024] [--cycles N]\n"
        "  Slice F (Option A): engine-only W1c-style policy comparison.\n"
        "  --B 0 enables whole-prompt baseline; --B >0 enables chunked.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, smoke_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need_value = [&](const char * flag) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return false;
            }
            return true;
        };
        if (a == "--model") {
            if (!need_value("--model")) return false;
            out.model_path = argv[++i];
        } else if (a == "--out-dir") {
            if (!need_value("--out-dir")) return false;
            out.out_dir = argv[++i];
        } else if (a == "--B") {
            if (!need_value("--B")) return false;
            out.B = std::atoi(argv[++i]);
        } else if (a == "--class") {
            if (!need_value("--class")) return false;
            out.only_class = argv[++i];
        } else if (a == "--cycles") {
            if (!need_value("--cycles")) return false;
            out.cycles = std::atoi(argv[++i]);
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
    if (out.out_dir.empty()) {
        fprintf(stderr, "error: --out-dir is required\n");
        return false;
    }
    if (out.B < 0) {
        fprintf(stderr, "error: --B (>=0) is required\n");
        return false;
    }
    if (out.cycles < 1) {
        fprintf(stderr, "error: --cycles must be >= 1\n");
        return false;
    }
    return true;
}

void emit_fail(const std::string & reason) {
    fprintf(stdout, "HPX_ENGINE_W1C_POLICY_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_W1C_POLICY_SMOKE: PASS\n");
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

std::string status_to_str(request_status st) {
    switch (st) {
        case request_status::completed:        return "completed";
        case request_status::cancelled:        return "cancelled";
        case request_status::failed_reserved:  return "failed";
        default:                               return "other";
    }
}

void append_jsonl_line(const std::string & path, const std::string & line) {
    FILE * f = std::fopen(path.c_str(), "a");
    if (f == nullptr) {
        fprintf(stderr, "warn: could not append to %s\n", path.c_str());
        return;
    }
    std::fwrite(line.data(), 1, line.size(), f);
    if (line.empty() || line.back() != '\n') std::fputc('\n', f);
    std::fclose(f);
}

std::vector<llama_token> tokenize_once(
    llama_model * model,
    const llama_context_params & ctx_params,
    const char * text)
{
    std::vector<llama_token> tokens;
    llama_context * c = llama_init_from_model(model, ctx_params);
    if (c == nullptr) return tokens;
    tokens = common_tokenize(c, text, /*add_special=*/true,
                             /*parse_special=*/true);
    llama_free(c);
    return tokens;
}

struct scenario_outcome {
    bool        ok = false;
    std::string err;

    // request results
    request_status anchor_status = request_status::completed;
    int32_t        anchor_ndec   = 0;
    size_t         anchor_gen    = 0;
    uint64_t       anchor_hash   = 0;
    request_status probe_status  = request_status::completed;
    int32_t        probe_ndec    = 0;
    size_t         probe_gen     = 0;
    uint64_t       probe_hash    = 0;

    // engine invariants
    int32_t decode_failures = 0;
    int32_t cancelled_count = 0;
    bool    residual_kv_ok  = false;

    // barrier evidence
    int32_t barrier_iter_count = 0;
    int64_t anchor_prefill_sum_at_barrier = 0;

    // interference cell evidence (post-barrier, in the snapshot)
    bool    found_interference        = false;
    int32_t max_pf_post_barrier       = 0;
    int32_t interference_iter_count   = 0;
};

// Run one (B, class, cycle) scenario on a fresh engine + context. The
// engine emits its own iter/engine_summary JSONL rows to the configured
// diag path; the caller writes scenario_begin/end markers around this
// call.
scenario_outcome run_scenario(
    llama_model *                    model,
    const llama_context_params &     base_params,
    const llama_vocab *              vocab,
    int32_t                          n_vocab,
    const std::vector<llama_token> & anchor,
    const std::vector<llama_token> & probe,
    int32_t                          B)
{
    scenario_outcome s;

    llama_context * ctx = llama_init_from_model(model, base_params);
    if (ctx == nullptr) {
        s.err = "context create failed";
        return s;
    }

    auto free_ctx = [&]() {
        if (ctx) { llama_free(ctx); ctx = nullptr; }
    };

    const int32_t anchor_len = static_cast<int32_t>(anchor.size());
    const int32_t probe_len  = static_cast<int32_t>(probe.size());

    // Worst-case per-iter rows: B=0 places the whole probe prompt in one
    // iter alongside the anchor decode row; B>0 places at most B+1.
    const int32_t worst = (B == 0)
        ? (probe_len + 2)
        : std::max<int32_t>(B + 2, anchor_len + 2);
    const int32_t batch_capacity = std::max<int32_t>(worst, 4);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 2;
    opts.lib.initial_idle_slots   = 2;
    opts.lib.keep_alive           = true;
    opts.lib.prefill_budget_rows  = B;
    opts.preload.prompt_tokens    = &anchor;     // unused (budgets empty)
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    // Release the engine after iter 1 (anchor has prefilled in one
    // chunk for every tested B, because anchor_len ~ 6 < min(B>0)=32
    // and B=0 is whole-prompt; the post-barrier validation below also
    // asserts this directly).
    opts.gate_test.release_iter_set = { 1 };
    opts.gate_test.max_decode_iters = 1;

    try {
        engine eng(std::move(opts));

        submit_request reqA;
        reqA.request_id    = 1;
        reqA.prompt_tokens = anchor;
        reqA.decode_budget = k_anchor_budget;
        reqA.want_stream   = false;
        submit_handle hA = eng.submit_request(std::move(reqA));

        external_release_handle rel = eng.register_external_release_iter(1);

        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        if (!poll_ready(rel.release_future, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            free_ctx();
            s.err = "release barrier at iter 1 did not fire";
            return s;
        }
        rel.release_future.get();

        // Barrier landing validation: anchor has prefilled and emitted
        // exactly one token. These vector writes happen-before
        // release set_value (engine task is the writer; barrier fires
        // at iter end via iter_fire_release_ack_barrier).
        const std::vector<int32_t> pf_barrier =
            eng.result().metrics.prefill_rows_per_iter;
        const std::vector<int32_t> tok_barrier =
            eng.result().metrics.tokens_emitted_per_iter;
        s.barrier_iter_count             = static_cast<int32_t>(pf_barrier.size());
        s.anchor_prefill_sum_at_barrier  = sum_vec(pf_barrier);
        const int64_t tok_b              = sum_vec(tok_barrier);
        if (s.anchor_prefill_sum_at_barrier != anchor_len || tok_b != 1) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            free_ctx();
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "barrier did not land on anchor-decoding state: "
                "prefill_sum=%lld (want %d) tokens_emitted=%lld (want 1) "
                "barrier_iter_count=%d",
                static_cast<long long>(s.anchor_prefill_sum_at_barrier),
                anchor_len, static_cast<long long>(tok_b),
                s.barrier_iter_count);
            s.err = buf;
            return s;
        }

        // Submit probe, then ack the release to resume the engine.
        submit_request reqB;
        reqB.request_id    = 2;
        reqB.prompt_tokens = probe;
        reqB.decode_budget = k_probe_budget;
        reqB.want_stream   = false;
        submit_handle hB = eng.submit_request(std::move(reqB));

        rel.ack_promise.set_value();

        if (!poll_ready(hA.result, k_wait_timeout)
         || !poll_ready(hB.result, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            free_ctx();
            s.err = "anchor/probe result not ready within timeout";
            return s;
        }
        request_result rA = hA.result.get();
        request_result rB = hB.result.get();

        eng.request_shutdown();
        if (!poll_ready(engine_fut, k_join_timeout)) {
            free_ctx();
            s.err = "engine did not join after request_shutdown";
            return s;
        }
        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            free_ctx();
            s.err = std::string("engine task threw on join: ") + e.what();
            return s;
        }

        const engine_result & er = eng.result();
        s.anchor_status   = rA.status; s.probe_status = rB.status;
        s.anchor_ndec     = rA.n_decoded; s.probe_ndec = rB.n_decoded;
        s.anchor_gen      = rA.generated_tokens.size();
        s.probe_gen       = rB.generated_tokens.size();
        s.anchor_hash     = rA.hash; s.probe_hash = rB.hash;
        s.decode_failures = er.decode_failures;
        s.cancelled_count = er.cancelled_count;
        s.residual_kv_ok  = er.residual_kv_ok;

        // Post-barrier scan over per-iter vectors. Barrier landed at end
        // of iter 1, so post-barrier range is index [barrier_iter_count, ..).
        const auto & pf  = er.metrics.prefill_rows_per_iter;
        const auto & dr  = er.metrics.decode_rows_per_iter;
        const auto & act = er.metrics.active_seqs_per_iter;
        const int32_t n_iters = static_cast<int32_t>(act.size());
        for (int32_t i = s.barrier_iter_count; i < n_iters; i++) {
            const int32_t p = (i < (int32_t)pf.size())  ? pf[i]  : 0;
            const int32_t d = (i < (int32_t)dr.size())  ? dr[i]  : 0;
            const int32_t a = (i < (int32_t)act.size()) ? act[i] : 0;
            if (p > s.max_pf_post_barrier) s.max_pf_post_barrier = p;
            if (p > 0 && d > 0 && a == 2) {
                s.found_interference = true;
                s.interference_iter_count++;
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

std::string scenario_begin_json(int32_t B, const std::string & klass,
                                int32_t cycle, int32_t anchor_len,
                                int32_t probe_len)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"kind\":\"scenario_begin\",\"B\":%d,\"class\":\"%s\","
        "\"cycle\":%d,\"anchor_len\":%d,\"probe_len\":%d}",
        B, klass.c_str(), cycle, anchor_len, probe_len);
    return buf;
}

std::string scenario_end_json(int32_t B, const std::string & klass,
                              int32_t cycle, const scenario_outcome & s)
{
    char buf[800];
    std::snprintf(buf, sizeof(buf),
        "{\"kind\":\"scenario_end\",\"B\":%d,\"class\":\"%s\",\"cycle\":%d,"
        "\"ok\":%d,\"err\":\"%s\","
        "\"anchor_status\":\"%s\",\"anchor_n_decoded\":%d,"
        "\"anchor_hash\":\"0x%016llx\","
        "\"probe_status\":\"%s\",\"probe_n_decoded\":%d,"
        "\"probe_hash\":\"0x%016llx\","
        "\"decode_failures\":%d,\"cancelled_count\":%d,"
        "\"residual_kv_ok\":%d,"
        "\"barrier_iter_count\":%d,\"anchor_prefill_sum_at_barrier\":%lld,"
        "\"found_interference\":%d,\"max_pf_post_barrier\":%d,"
        "\"interference_iter_count\":%d}",
        B, klass.c_str(), cycle,
        s.ok ? 1 : 0, s.err.c_str(),
        status_to_str(s.anchor_status).c_str(), s.anchor_ndec,
        static_cast<unsigned long long>(s.anchor_hash),
        status_to_str(s.probe_status).c_str(), s.probe_ndec,
        static_cast<unsigned long long>(s.probe_hash),
        s.decode_failures, s.cancelled_count,
        s.residual_kv_ok ? 1 : 0,
        s.barrier_iter_count,
        static_cast<long long>(s.anchor_prefill_sum_at_barrier),
        s.found_interference ? 1 : 0,
        s.max_pf_post_barrier,
        s.interference_iter_count);
    return buf;
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    // Phase 1 diagnostics ON unconditionally for this smoke. Set BEFORE
    // any engine runs (diag_enabled() / diag_path() cache on first
    // call).
    setenv("LLAMA_HPX_DIAG_METRICS", "1", /*overwrite=*/1);

    // Compose diag path: <out-dir>/diag-B<n>.jsonl
    char diag_path_buf[1024];
    std::snprintf(diag_path_buf, sizeof(diag_path_buf),
        "%s/diag-B%d.jsonl", args.out_dir.c_str(), args.B);
    setenv("LLAMA_HPX_DIAG_METRICS_PATH", diag_path_buf, /*overwrite=*/1);
    const std::string diag_path = diag_path_buf;

    // Truncate any pre-existing file at this path so re-runs are clean.
    {
        FILE * t = std::fopen(diag_path.c_str(), "w");
        if (t) std::fclose(t);
    }

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
    ctx_params.n_ctx           = 4096;
    ctx_params.n_batch         = 1280;   // covers L1024 + anchor + safety
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;

    // Tokenize anchor and the four probe classes once. Token IDs are
    // ctx-independent.
    const std::vector<llama_token> anchor =
        tokenize_once(model, ctx_params, k_anchor_prompt);
    if (anchor.empty()) return fail_and_cleanup("anchor tokenization failed");

    const std::string L256_text = std::string(k_P1) + " " + k_P2;
    const std::string L1024_text =
        std::string(k_P1) + " " + k_P2 + " " + k_P3 + " " + k_P4 +
        " " + k_P1 + " " + k_P2 + " " + k_P3 + " " + k_P4;

    std::vector<prompt_class> classes;
    classes.push_back({"L8",    k_L8});
    classes.push_back({"L64",   k_L64});
    classes.push_back({"L256",  L256_text});
    classes.push_back({"L1024", L1024_text});

    // Initial top-of-file run marker (also writes harness-side anchor
    // length so analyze.py does not need to re-tokenize).
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"kind\":\"run\",\"B\":%d,\"cycles\":%d,\"anchor_len\":%d}",
            args.B, args.cycles, static_cast<int32_t>(anchor.size()));
        append_jsonl_line(diag_path, buf);
    }

    bool any_fail = false;
    std::string first_fail;

    for (const auto & pc : classes) {
        if (!args.only_class.empty() && args.only_class != pc.name) continue;

        const std::vector<llama_token> probe =
            tokenize_once(model, ctx_params, pc.text.c_str());
        if (probe.empty()) {
            any_fail = true;
            first_fail = std::string("tokenize probe failed for ") + pc.name;
            break;
        }
        const int32_t probe_len = static_cast<int32_t>(probe.size());
        fprintf(stdout, "B=%d class=%s anchor_len=%d probe_len=%d cycles=%d\n",
            args.B, pc.name, static_cast<int32_t>(anchor.size()),
            probe_len, args.cycles);
        fflush(stdout);

        for (int32_t cycle = 0; cycle < args.cycles; cycle++) {
            append_jsonl_line(diag_path,
                scenario_begin_json(args.B, pc.name, cycle,
                                    static_cast<int32_t>(anchor.size()),
                                    probe_len));
            scenario_outcome s = run_scenario(
                model, ctx_params, vocab, n_vocab, anchor, probe, args.B);
            append_jsonl_line(diag_path,
                scenario_end_json(args.B, pc.name, cycle, s));

            fprintf(stdout,
                "  cycle=%d ok=%d anchor=%s/n=%d/hash=0x%016llx "
                "probe=%s/n=%d/hash=0x%016llx interference=%d max_pf=%d "
                "interference_iters=%d err=%s\n",
                cycle, s.ok ? 1 : 0,
                status_to_str(s.anchor_status).c_str(), s.anchor_ndec,
                static_cast<unsigned long long>(s.anchor_hash),
                status_to_str(s.probe_status).c_str(), s.probe_ndec,
                static_cast<unsigned long long>(s.probe_hash),
                s.found_interference ? 1 : 0,
                s.max_pf_post_barrier, s.interference_iter_count,
                s.err.c_str());
            fflush(stdout);

            if (!s.ok) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d failed: %s",
                        args.B, pc.name, cycle, s.err.c_str());
                    first_fail = buf;
                }
                continue;
            }
            // Always-on correctness gates
            if (s.anchor_status != request_status::completed
             || s.probe_status  != request_status::completed) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d: non-completed status",
                        args.B, pc.name, cycle);
                    first_fail = buf;
                }
                continue;
            }
            if (s.anchor_ndec != k_anchor_budget
             || s.probe_ndec  != k_probe_budget) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d: budget not exhausted "
                        "(anchor n=%d probe n=%d)",
                        args.B, pc.name, cycle, s.anchor_ndec, s.probe_ndec);
                    first_fail = buf;
                }
                continue;
            }
            if (s.decode_failures != 0 || s.cancelled_count != 0
             || !s.residual_kv_ok) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d: engine invariant failed "
                        "(decode_failures=%d cancelled=%d residual_kv_ok=%d)",
                        args.B, pc.name, cycle,
                        s.decode_failures, s.cancelled_count,
                        s.residual_kv_ok ? 1 : 0);
                    first_fail = buf;
                }
                continue;
            }
            // Required: at least one interference iter for this scenario.
            if (!s.found_interference) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d: no interference iter "
                        "(pf>0 AND dec>0 AND act==2) after the barrier",
                        args.B, pc.name, cycle);
                    first_fail = buf;
                }
                continue;
            }
            // If B>0, the max per-iter prefill rows must NOT exceed B.
            if (args.B > 0 && s.max_pf_post_barrier > args.B) {
                any_fail = true;
                if (first_fail.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "B=%d class=%s cycle=%d: max_pf_post_barrier=%d "
                        "exceeds B=%d (policy not bounding)",
                        args.B, pc.name, cycle,
                        s.max_pf_post_barrier, args.B);
                    first_fail = buf;
                }
                continue;
            }
        }
    }

    if (any_fail) {
        emit_fail(first_fail);
        return cleanup_and_stop(1);
    }

    emit_pass();
    return cleanup_and_stop(0);
}
