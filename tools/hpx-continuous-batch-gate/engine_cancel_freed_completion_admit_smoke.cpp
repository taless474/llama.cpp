// engine_cancel_freed_completion_admit_smoke.cpp — N5a: cancel_freed →
// natural-completion → later-admission slot-recovery reproducer for
// llama-hpx-engine.
//
// N5a bug: on an idle-pool engine (initial_idle_slots>0,
// reuse_completed=false), a slot bound via admission_source::cancel_freed
// that then completes NATURALLY is dropped from every admission pool by
// finalize_and_fulfill (the reuse_completed_ branch is gated off and the
// pre-fix branch only returned admission_source::initial_idle slots). A
// later submit_request then reaches arrival_drained + request_queued but
// is never admitted.
//
// Shape (single slot): n_seq_max=1, initial_idle_slots=1, budgets={},
// reuse_completed=false, keep_alive=true.
//
//   Phase A (rid=1, want_stream=true, budget=256): admitted from the
//     idle pool; drain to the first token (proves active + decoding),
//     then cancel. The active-cancel pipeline clears KV and pushes the
//     slot onto free_due_to_cancel_. Streaming is used ONLY here.
//   Phase B (rid=2, want_stream=false, budget=8): admitted from
//     free_due_to_cancel_ (admission_src=cancel_freed), completes
//     naturally. Pre-fix this is where the slot leaks.
//   Phase C (rid=3, want_stream=false, budget=8): never self-cancelled.
//     Pre-fix it cannot be admitted (leaked slot) and stays queued;
//     post-fix it admits from the recovered free_idle_ slot and
//     completes with the canonical b8 hash.
//
// Teardown is clean because N5b is fixed: after the bounded Phase C wait,
// request_shutdown() always joins the engine. A still-queued Phase C is
// resolved as request_status::failed_reserved during shutdown (N5b), so
// the PRE-FIX red is a controlled nonzero exit with a clean join — no
// _Exit on the expected path. The _Exit fallback fires only if the
// engine fails to join (an N5b regression), so the smoke can never hang.
//
// Pre-fix red:  A cancelled (active), B completed from cancel_freed,
//               C resolves failed_reserved (queued, never admitted),
//               engine joins cleanly, exit nonzero.
// Post-fix pass: C completes with canonical hash 0x0619d4d1900c2365,
//               admitted_count==3, engine joins cleanly.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution lives inside eng.run() on the single hpx::async task this
// file spawns. No llama_decode, llama_batch_*, llama_memory_seq_*,
// llama_get_logits_ith, or common_batch_add call sites exist here.

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
        "  N5a cancel_freed -> natural-completion -> later-admission "
        "slot-recovery reproducer for the llama-hpx-engine library.\n",
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
            "HPX_ENGINE_CANCEL_FREED_COMPLETION_ADMIT_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout,
            "HPX_ENGINE_CANCEL_FREED_COMPLETION_ADMIT_SMOKE: PASS\n");
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

submit_request build_greedy(int32_t rid,
                            const std::vector<llama_token> & prompt) {
    submit_request req;
    req.request_id    = rid;
    req.prompt_tokens = prompt;
    req.decode_budget = 8;
    req.want_stream   = false;
    return req;
}

// Bounded readiness poll on an HPX future from main() (foreign OS
// thread). is_ready() is safe from any thread (no suspension) — the
// gate_validation.cpp idiom. Returns true if ready before the deadline.
template <typename Fut>
bool poll_ready(Fut & f, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!f.is_ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
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
    ctx_params.n_seq_max       = 1;
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
    opts.lib.n_seq_max            = 1;
    opts.lib.initial_idle_slots   = 1;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = &shared_prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    constexpr uint64_t k_canonical_greedy_hash = 0x0619d4d1900c2365ULL;
    constexpr auto     k_wait_timeout = std::chrono::milliseconds(15000);
    constexpr auto     k_join_timeout = std::chrono::milliseconds(8000);

    try {
        engine eng(std::move(opts));
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        // ---- Phase A: streaming request, cancel while active. --------
        submit_request reqA;
        reqA.request_id    = 1;
        reqA.prompt_tokens = shared_prompt;
        reqA.decode_budget = 256;
        reqA.want_stream   = true;
        submit_handle hA = eng.submit_request(std::move(reqA));
        if (!hA.stream.has_value()) {
            return fail_and_cleanup("phaseA stream not engaged");
        }

        stream_close_reason a_close = stream_close_reason::completed;
        bool a_seen_first_token = false;
        try {
            token_stream_event ev = hA.stream->get().get();
            if (ev.kind == stream_event_kind::token) {
                a_seen_first_token = true;
            } else {
                a_close = ev.close_reason;
            }
        } catch (...) {
            // channel closed unexpectedly
        }
        if (!a_seen_first_token) {
            return fail_and_cleanup(
                std::string("phaseA closed before first token; close=") +
                stream_close_label(a_close));
        }

        eng.cancel_request(1);

        bool a_seen_close = false;
        try {
            while (true) {
                token_stream_event ev = hA.stream->get().get();
                if (ev.kind == stream_event_kind::closed) {
                    a_close = ev.close_reason;
                    a_seen_close = true;
                    break;
                }
            }
        } catch (...) {
            // channel closed by engine; terminal
        }
        if (!a_seen_close) {
            return fail_and_cleanup("phaseA terminal close not observed");
        }
        if (a_close != stream_close_reason::cancelled) {
            return fail_and_cleanup(
                std::string("phaseA close reason != cancelled; close=") +
                stream_close_label(a_close));
        }
        request_result rA = hA.result.get();
        if (rA.status != request_status::cancelled) {
            return fail_and_cleanup("phaseA status != cancelled");
        }
        if (rA.n_decoded <= 0) {
            return fail_and_cleanup("phaseA n_decoded <= 0");
        }
        fprintf(stdout, "phaseA rid=1 status=%s n_decoded=%d "
                "(active-cancel -> free_due_to_cancel_)\n",
                status_name(rA.status), rA.n_decoded);
        fflush(stdout);

        // ---- Phase B: non-streaming, admitted from cancel_freed. -----
        submit_handle hB =
            eng.submit_request(build_greedy(/*rid=*/2, shared_prompt));
        if (!poll_ready(hB.result, k_wait_timeout)) {
            return fail_and_cleanup(
                "phaseB result not ready within timeout "
                "(regression: cancel_freed admission broke)");
        }
        request_result rB = hB.result.get();
        if (rB.status != request_status::completed) {
            return fail_and_cleanup("phaseB status != completed");
        }
        if (rB.n_decoded != 8) {
            return fail_and_cleanup("phaseB n_decoded != 8");
        }
        if (rB.hash != k_canonical_greedy_hash) {
            return fail_and_cleanup("phaseB hash != canonical b8");
        }
        if (rB.admission_src != admission_source::cancel_freed) {
            return fail_and_cleanup(
                std::string("phaseB admission_src=") +
                admission_source_name(rB.admission_src) +
                " != cancel_freed (reproducer precondition unmet)");
        }
        fprintf(stdout, "phaseB rid=2 status=%s n_decoded=%d "
                "hash=0x%016llx admission_src=%s\n",
                status_name(rB.status), rB.n_decoded,
                static_cast<unsigned long long>(rB.hash),
                admission_source_name(rB.admission_src));
        fflush(stdout);

        // ---- Phase C: non-streaming probe, NEVER self-cancelled. -----
        submit_handle hC =
            eng.submit_request(build_greedy(/*rid=*/3, shared_prompt));
        const bool c_ready = poll_ready(hC.result, k_wait_timeout);
        fprintf(stdout, "phaseC rid=3 ready_within_%lldms=%d\n",
                static_cast<long long>(k_wait_timeout.count()),
                c_ready ? 1 : 0);
        fflush(stdout);

        // Clean teardown (N5b): shutdown always joins. A still-queued C
        // is drained as failed_reserved during shutdown.
        eng.request_shutdown();
        if (!poll_ready(engine_fut, k_join_timeout)) {
            // N5b regression — engine did not join. Avoid a hang.
            emit_fail(
                "engine did not join after request_shutdown "
                "(N5b shutdown-liveness regression)");
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

        request_result rC = hC.result.get();
        const engine_result & er = eng.result();

        // Invariants that hold in BOTH pre-fix and post-fix runs.
        if (er.queued_cancelled != 0) {
            return fail_and_cleanup(
                "queued_cancelled != 0 (phaseA contaminated the "
                "active-cancel path)");
        }
        if (er.cancel_active_observed != 1) {
            return fail_and_cleanup("cancel_active_observed != 1");
        }
        if (er.cancelled_count != 1) {
            return fail_and_cleanup("cancelled_count != 1");
        }
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }

        if (rC.status == request_status::completed) {
            // POST-FIX: the freed slot was recovered to free_idle_ and
            // C was admitted.
            if (rC.n_decoded != 8) {
                return fail_and_cleanup("phaseC n_decoded != 8");
            }
            if (rC.hash != k_canonical_greedy_hash) {
                return fail_and_cleanup("phaseC hash != canonical b8");
            }
            if (er.admitted_count != 3) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "admitted_count=%d (expected 3: A,B,C)",
                    er.admitted_count);
                return fail_and_cleanup(buf);
            }
            if (er.external_admitted_count != 3) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "external_admitted_count=%d (expected 3)",
                    er.external_admitted_count);
                return fail_and_cleanup(buf);
            }
            fprintf(stdout,
                "phaseC rid=3 status=%s n_decoded=%d hash=0x%016llx "
                "admission_src=%s\n",
                status_name(rC.status), rC.n_decoded,
                static_cast<unsigned long long>(rC.hash),
                admission_source_name(rC.admission_src));
            fprintf(stdout,
                "counters admitted_count=%d external_admitted_count=%d "
                "cancelled_count=%d queued_cancelled=%d\n",
                er.admitted_count, er.external_admitted_count,
                er.cancelled_count, er.queued_cancelled);
            fflush(stdout);
            emit_pass();
            return cleanup_and_stop(0);
        }

        if (rC.status == request_status::failed_reserved) {
            // PRE-FIX clean red: C was queued-but-unadmittable (the
            // cancel_freed slot leaked) and was resolved by the N5b
            // shutdown drain. Engine joined cleanly; controlled FAIL.
            fprintf(stdout,
                "phaseC not admitted pre-fix: status=%s admitted_count=%d "
                "external_admitted_count=%d (cancel_freed slot leaked; "
                "resolved as failed_reserved on shutdown)\n",
                status_name(rC.status), er.admitted_count,
                er.external_admitted_count);
            fflush(stdout);
            return fail_and_cleanup(
                "phaseC not admitted — cancel_freed natural-completion "
                "slot leak (N5a)");
        }

        return fail_and_cleanup(
            std::string("phaseC unexpected status=") +
            status_name(rC.status));
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
