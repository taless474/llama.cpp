// engine_cancel_mid_prefill_smoke.cpp — Slice D: cancellation during
// partial (chunked) prefill for the llama-hpx-engine library.
//
// Slice C made a new state reachable: a live-admitted seq that is ACTIVE
// with partial KV (prefill_cursor in (0, n), prefill_complete=false) and
// zero sampled tokens. This smoke proves that cancelling such a seq:
//   - clears its partial KV,
//   - fulfills the per-request future as `cancelled` with no token,
//   - emits no token_stream_event before the terminal `closed`,
//   - and leaves the slot reusable by a later request (b8 anchor).
//
// Deterministic, flake-free timing via the gate release/ack barrier
// (engine::register_external_release_iter). The first work iter is iter 1
// (run_body sets `iter = 0` and does `iter++` at the top of each inner
// pass; the keep-alive idle-wait does not increment iter), so K0 = 1 is
// the iter that admits the request and places its first prefill chunk.
// The engine parks at the end of iter 1 (after chunk-0 decode, with the
// seq mid-prefill); the submitter issues the cancel, then releases the
// barrier; the cancel is observed as an ACTIVE cancel in iter 2.
//
// Metric scoping (Slice D tightening 1): the partial-prefill proof is
// snapshotted from the engine's in-memory diagnostics AFTER the cancelled
// result is observed but BEFORE the follow-up reuse request is submitted,
// so the follow-up's prefill rows / emitted tokens cannot contaminate it.
// This read is race-safe: the iter-1 prefill-row write happens-before the
// iter-2 cancel promise (so it is visible after h.result.get()), and no
// further per-iter append occurs while the keep-alive engine idles between
// the cancelled completion and the follow-up submit.
//
// K0 validation (Slice D tightening 2): if the snapshot does not prove
// "at least one partial prefill chunk and zero emitted tokens" before the
// cancel, the smoke FAILS LOUDLY, printing K0, both per-iter vectors,
// cancel_observed_iter, and the cancelled result fields.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and cleanup. All llama execution
// lives inside eng.run() on the single hpx::async task. No llama_decode /
// llama_batch_* / llama_memory_seq_* / llama_get_logits_ith /
// common_batch_add call sites exist here.

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

// The first work iter (admit + first prefill chunk). See file header.
constexpr int32_t  k_K0  = 1;
constexpr uint64_t k_b8  = 0x0619d4d1900c2365ULL;

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  Slice D cancel-during-partial-prefill correctness smoke for the "
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
    fprintf(stdout, "HPX_ENGINE_CANCEL_MID_PREFILL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_CANCEL_MID_PREFILL_SMOKE: PASS\n");
    fflush(stdout);
}

const char * stream_close_label(stream_close_reason r) {
    switch (r) {
        case stream_close_reason::completed: return "completed";
        case stream_close_reason::cancelled: return "cancelled";
        case stream_close_reason::error:     return "error";
    }
    return "unknown";
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

std::string vec_to_str(const std::vector<int32_t> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

int64_t sum_vec(const std::vector<int32_t> & v) {
    int64_t s = 0;
    for (int32_t x : v) s += x;
    return s;
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    // Self-enable Phase-1 diagnostics so the engine populates the in-memory
    // per-iter vectors this smoke snapshots. Set BEFORE any engine runs
    // (diag_enabled() caches on first use). Diag-ON does not change token
    // output (Slice B closure).
    setenv("LLAMA_HPX_DIAG_METRICS", "1", /*overwrite=*/1);

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
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 64;
    ctx_params.n_seq_max       = 1;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt = common_tokenize(
        ctx, "Hello, my name is", /*add_special=*/true,
        /*parse_special=*/true);
    if (prompt.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }
    const int32_t n = static_cast<int32_t>(prompt.size());
    constexpr int32_t k_B = 2;            // prefill row budget under test
    if (n <= k_B) {
        return fail_and_cleanup(
            "prompt length must exceed prefill_budget_rows to be partial");
    }
    fprintf(stdout, "prompt_len=%d prefill_budget_rows=%d K0=%d\n",
            n, k_B, k_K0);
    fflush(stdout);

    const int32_t batch_capacity = std::max<int32_t>(n, 1);
    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = 1;
    opts.lib.initial_idle_slots   = 1;
    opts.lib.keep_alive           = true;
    opts.lib.prefill_budget_rows  = k_B;
    opts.preload.prompt_tokens    = &prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    // Deterministic release barrier at the first work iter (K0).
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.release_iter_set = { k_K0 };
    opts.gate_test.max_decode_iters = k_K0;   // sizes release-promise vectors

    constexpr auto k_wait_timeout = std::chrono::milliseconds(15000);
    constexpr auto k_join_timeout = std::chrono::milliseconds(8000);

    try {
        engine eng(std::move(opts));

        // Register the release barrier BEFORE the engine task is scheduled.
        external_release_handle rel = eng.register_external_release_iter(k_K0);

        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        // ---- Streaming request, to be cancelled mid-prefill. ------------
        submit_request reqA;
        reqA.request_id    = 1;
        reqA.prompt_tokens = prompt;
        reqA.decode_budget = 8;
        reqA.want_stream   = true;
        submit_handle hA = eng.submit_request(std::move(reqA));
        if (!hA.stream.has_value()) {
            return fail_and_cleanup("stream not engaged for cancel request");
        }

        // Wait for the engine to finish iter K0 (first partial prefill
        // chunk) and park on the ack. Deterministic — no wall-clock guess.
        if (!poll_ready(rel.release_future, k_wait_timeout)) {
            return fail_and_cleanup("release barrier at K0 did not fire");
        }
        rel.release_future.get();

        // Cancel the ACTIVE mid-prefill seq, then release the engine.
        eng.cancel_request(1);
        rel.ack_promise.set_value();

        // Wait for the cancelled result and drain the stream to terminal.
        if (!poll_ready(hA.result, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            return fail_and_cleanup("cancelled result not ready in time");
        }
        request_result rA = hA.result.get();

        // Drain the stream: no token event must appear before the close.
        bool seen_token = false;
        bool seen_close = false;
        stream_close_reason close_reason = stream_close_reason::completed;
        try {
            while (true) {
                token_stream_event ev = hA.stream->get().get();
                if (ev.kind == stream_event_kind::token) {
                    seen_token = true;
                } else if (ev.kind == stream_event_kind::closed) {
                    close_reason = ev.close_reason;
                    seen_close   = true;
                    break;
                }
            }
        } catch (...) {
            // channel closed by engine; terminal
        }

        // ---- Partial-prefill proof, SCOPED to the cancelled request -----
        // Snapshot now (after the cancelled result, before any follow-up
        // submit) so the follow-up cannot contaminate the counts.
        const std::vector<int32_t> pf_rows =
            eng.result().metrics.prefill_rows_per_iter;
        const std::vector<int32_t> tok_emit =
            eng.result().metrics.tokens_emitted_per_iter;
        const int64_t pf_sum  = sum_vec(pf_rows);
        const int64_t tok_sum = sum_vec(tok_emit);

        auto loud_fail = [&](const std::string & why) -> int {
            fprintf(stdout,
                "DIAG K0=%d prefill_rows_per_iter=%s "
                "tokens_emitted_per_iter=%s cancel_observed_iter=%d "
                "status=%s n_decoded=%d n_decoded_at_cancel=%d "
                "prefill_sum=%lld tokens_sum=%lld\n",
                k_K0, vec_to_str(pf_rows).c_str(),
                vec_to_str(tok_emit).c_str(), rA.cancel_observed_iter,
                status_name(rA.status), rA.n_decoded, rA.n_decoded_at_cancel,
                static_cast<long long>(pf_sum),
                static_cast<long long>(tok_sum));
            fflush(stdout);
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            return fail_and_cleanup(why);
        };

        if (!(pf_sum > 0 && pf_sum < n)) {
            return loud_fail(
                "partial-prefill proof failed: expected 0 < prefill_sum < "
                "prompt_len before cancel (wrong K0 or not mid-prefill)");
        }
        if (tok_sum != 0) {
            return loud_fail(
                "partial-prefill proof failed: a token was emitted before "
                "prefill completed");
        }

        // ---- Cancelled-request result + stream gates --------------------
        if (rA.status != request_status::cancelled) {
            return loud_fail("cancelled request status != cancelled");
        }
        if (rA.n_decoded != 0) {
            return loud_fail("cancelled request n_decoded != 0");
        }
        if (rA.n_decoded_at_cancel != 0) {
            return loud_fail("cancelled request n_decoded_at_cancel != 0");
        }
        if (!rA.generated_tokens.empty()) {
            return loud_fail("cancelled request generated_tokens not empty");
        }
        if (seen_token) {
            return loud_fail("a token_stream_event arrived before close");
        }
        if (!seen_close) {
            return loud_fail("terminal stream close not observed");
        }
        if (close_reason != stream_close_reason::cancelled) {
            return loud_fail(
                std::string("terminal close reason != cancelled; close=") +
                stream_close_label(close_reason));
        }
        fprintf(stdout,
            "cancelled rid=1 status=%s n_decoded=%d n_decoded_at_cancel=%d "
            "gen=%zu cancel_observed_iter=%d close=%s "
            "prefill_rows=%s (sum=%lld < n=%d) tokens_emitted=%s\n",
            status_name(rA.status), rA.n_decoded, rA.n_decoded_at_cancel,
            rA.generated_tokens.size(), rA.cancel_observed_iter,
            stream_close_label(close_reason),
            vec_to_str(pf_rows).c_str(),
            static_cast<long long>(pf_sum), n,
            vec_to_str(tok_emit).c_str());
        fflush(stdout);

        // ---- Follow-up: reuse the freed slot, expect canonical b8 -------
        submit_request reqB;
        reqB.request_id    = 2;
        reqB.prompt_tokens = prompt;
        reqB.decode_budget = 8;
        reqB.want_stream   = false;
        submit_handle hB = eng.submit_request(std::move(reqB));
        if (!poll_ready(hB.result, k_wait_timeout)) {
            eng.request_shutdown();
            poll_ready(engine_fut, k_join_timeout);
            return fail_and_cleanup("follow-up reuse result not ready in time");
        }
        request_result rB = hB.result.get();

        // ---- Clean teardown, then read engine-level invariants ----------
        eng.request_shutdown();
        if (!poll_ready(engine_fut, k_join_timeout)) {
            emit_fail("engine did not join after request_shutdown");
            fflush(stdout);
            fflush(stderr);
            _Exit(2);
        }
        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            return fail_and_cleanup(
                std::string("engine task threw on join: ") + e.what());
        }

        if (rB.status != request_status::completed) {
            return fail_and_cleanup("follow-up status != completed");
        }
        if (rB.n_decoded != 8) {
            return fail_and_cleanup("follow-up n_decoded != 8");
        }
        if (rB.hash != k_b8) {
            return fail_and_cleanup("follow-up hash != canonical b8");
        }
        fprintf(stdout,
            "reuse rid=2 status=%s n_decoded=%d hash=0x%016llx "
            "admission_src=%s\n",
            status_name(rB.status), rB.n_decoded,
            static_cast<unsigned long long>(rB.hash),
            admission_source_name(rB.admission_src));
        fflush(stdout);

        const engine_result & er = eng.result();
        if (er.cancel_active_observed != 1) {
            return fail_and_cleanup("cancel_active_observed != 1");
        }
        if (er.cancelled_count != 1) {
            return fail_and_cleanup("cancelled_count != 1");
        }
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup("queued_cancelled != 0");
        }
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        fprintf(stdout,
            "counters cancel_active_observed=%d cancelled_count=%d "
            "queued_cancelled=%d decode_failures=%d residual_kv_ok=%d\n",
            er.cancel_active_observed, er.cancelled_count,
            er.queued_cancelled, er.decode_failures,
            er.residual_kv_ok ? 1 : 0);
        fflush(stdout);

        emit_pass();
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
