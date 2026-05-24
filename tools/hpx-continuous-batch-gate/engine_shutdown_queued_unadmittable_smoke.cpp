// engine_shutdown_queued_unadmittable_smoke.cpp — N5b: shutdown-vs-
// queued-but-unadmittable liveness reproducer for llama-hpx-engine.
//
// N5b question: can request_shutdown() wake and join a keep-alive
// engine that is HPX-suspended in wait_inbox_blocking() while a
// submit_request sits queued in waiting_queue_consumable_ with no
// admission source?
//
// Shape (same as engine_queued_cancel_smoke, minus the cancel, and
// non-streaming): n_seq_max=1, initial_idle_slots=0, budgets={} -> seqs_
// is empty, so a submitted request is drained to
// waiting_queue_consumable_ (arrival_drained + request_queued) and can
// NEVER be admitted (no idle / cancel-freed / completion-freed pool).
// No cancellation, no streaming, no completion, no llama_decode.
//
// Pre-fix expectation (the bug): run_body's outer-tail shutdown-break
// is gated on waiting_queue_consumable_.empty(), so with a queued
// request the engine never observes shutdown and re-parks in
// wait_inbox_blocking() waiting for a message that will never come.
// request_shutdown() therefore does NOT let engine_fut join.
//
// This smoke must fail CLEANLY pre-fix: it bounded-waits on
// engine_fut readiness and, on timeout, prints a controlled FAIL and
// calls _Exit(nonzero) AFTER flushing stdout/stderr. _Exit deliberately
// skips destructors and hpx_runtime::stop() — the engine task is parked
// and unjoinable, and running stop() under a parked task is exactly
// what aborted an earlier draft. No SIGKILL, no hang, no abort.
//
// Post-fix expectation: request_shutdown() wakes the engine, it drains
// the queued request to a defined terminal status and exits; engine_fut
// joins, engine_shutdown_observed==1, and the queued request's promise
// is resolved (not broken_promise). The exact terminal status of a
// queued-at-shutdown request is the N5b fix's contract decision; this
// smoke accepts cancelled or failed_reserved and prints which.
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
        "  N5b shutdown-vs-queued-unadmittable liveness smoke for the "
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
            "HPX_ENGINE_SHUTDOWN_QUEUED_UNADMITTABLE_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout,
            "HPX_ENGINE_SHUTDOWN_QUEUED_UNADMITTABLE_SMOKE: PASS\n");
    fflush(stdout);
}

// Bounded readiness poll on an HPX future from main() (a foreign OS
// thread). is_ready() is safe to call from any thread (no suspension);
// this is the gate_validation.cpp idiom. Returns true if the future
// became ready before the deadline.
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
    opts.lib.initial_idle_slots   = 0;
    opts.lib.keep_alive           = true;
    // Borrowed reference required to be non-null by the engine ctor.
    // budgets={} + initial_idle_slots=0 means seqs_ is empty; the
    // prefill loop reads nothing and admit_one is never called. The
    // borrow is kept alive only to satisfy the existing contract.
    opts.preload.prompt_tokens    = &shared_prompt;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    // Bounded windows. The not-ready window only needs to let the engine
    // drain the arrival and re-park; the join window must be long enough
    // to be sure a clean shutdown would have joined (the pre-fix hang is
    // permanent, so any generous bound exposes it).
    constexpr auto k_not_ready_window = std::chrono::milliseconds(1500);
    constexpr auto k_join_timeout     = std::chrono::milliseconds(8000);

    try {
        engine eng(std::move(opts));

        // Start the engine before submitting. With no actives, no idle
        // slots, and an empty inbox, it drives straight to the keep-
        // alive idle-wait.
        hpx::future<void> engine_fut = hpx::async([&] { eng.run(); });

        // Non-streaming request; will be drained to
        // waiting_queue_consumable_ and never admitted.
        submit_request req;
        req.request_id    = 1;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 8;
        req.want_stream   = false;
        submit_handle h = eng.submit_request(std::move(req));

        // Precondition: the request must stay queued (not complete)
        // through a short bounded window. By construction it cannot be
        // admitted, so it must not become ready here.
        const bool ready_early = poll_ready(h.result, k_not_ready_window);
        const bool precondition_ok = !ready_early;
        if (ready_early) {
            fprintf(stdout,
                "note: request became ready before shutdown — "
                "queued-unadmittable precondition violated\n");
            fflush(stdout);
        } else {
            fprintf(stdout,
                "precondition ok: request queued, not admitted within "
                "%lld ms\n",
                static_cast<long long>(k_not_ready_window.count()));
            fflush(stdout);
        }

        // The single control action under test. Never cancel the
        // request — we are testing shutdown alone.
        eng.request_shutdown();

        // Bounded join. Pre-fix this never readies.
        const bool joined = poll_ready(engine_fut, k_join_timeout);

        if (!joined) {
            // PRE-FIX EXPECTED PATH. The engine is parked and
            // unjoinable; do NOT attempt normal cleanup (running
            // hpx_runtime::stop() under a parked task is what aborts).
            // Flush and exit with a controlled nonzero code.
            emit_fail(
                "engine did not join after request_shutdown with "
                "queued-unadmittable request");
            fflush(stdout);
            fflush(stderr);
            _Exit(1);
        }

        // POST-FIX PATH (engine joined). Surface any engine-task
        // exception, then assert the shutdown contract.
        engine_fut.get();

        if (!precondition_ok) {
            return fail_and_cleanup(
                "queued-unadmittable precondition violated "
                "(request resolved before shutdown)");
        }

        // The queued request's promise must be resolved (not
        // broken_promise) to the N5b shutdown-aborted terminal status:
        // failed_reserved (NOT completed, NOT user-cancelled).
        request_result r = h.result.get();
        if (r.status != request_status::failed_reserved) {
            return fail_and_cleanup(
                std::string("queued-at-shutdown request status=") +
                status_name(r.status) + " != failed_reserved");
        }

        const engine_result & er = eng.result();
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup("engine_shutdown_observed != 1");
        }
        if (er.admitted_count != 0) {
            return fail_and_cleanup("admitted_count != 0");
        }
        if (er.decode_calls != 0) {
            return fail_and_cleanup("decode_calls != 0");
        }
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }

        fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d admitted_count=%d "
            "decode_calls=%d engine_shutdown_observed=%d\n",
            r.request_id, status_name(r.status), r.n_decoded,
            er.admitted_count, er.decode_calls,
            er.engine_shutdown_observed);
        fflush(stdout);
        emit_pass();
        return cleanup_and_stop(0);
    } catch (const std::exception & e) {
        return fail_and_cleanup(std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }
}
