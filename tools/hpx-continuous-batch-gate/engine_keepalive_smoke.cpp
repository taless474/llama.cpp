// engine_keepalive_smoke.cpp — M3a: long-running keep-alive smoke for
// llama-hpx-engine. Proves engine.run() launched with no initial
// actives idle-waits on inbox_cv_, processes a request submitted at
// runtime, returns to idle-wait, processes a second request, and
// exits cleanly when request_shutdown() is observed on a drained
// engine.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution and KV mutation live inside eng.run() running on the single
// hpx::async task this file spawns. No llama_decode, llama_batch_*,
// llama_memory_seq_*, llama_get_logits_ith, or common_batch_add call
// sites exist here.

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
        "  Minimal keep-alive smoke for the llama-hpx-engine "
        "library.\n",
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
    fprintf(stdout, "HPX_ENGINE_KEEPALIVE_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_KEEPALIVE_SMOKE: PASS\n");
    fflush(stdout);
}

void print_result(const request_result & r) {
    fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d hash=0x%016llx\n",
            r.request_id, status_name(r.status), r.n_decoded,
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

    std::vector<llama_token> shared_prompt = common_tokenize(
        ctx, "Once upon a time", /*add_special=*/true,
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
    // M4c: preload.prompt_tokens is optional when preload.budgets is
    // empty. With budgets={} and initial_idle_slots=2 the engine reads
    // no rows from a shared prompt; each admission carries its own
    // per-request prompt vector via submit_request below.
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        // M3a: start the engine BEFORE any submission. With no
        // initial actives and an empty inbox the engine drives
        // straight into the keep-alive outer-loop idle-wait —
        // exercising the cv-wait path on a fresh run.
        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        // First submission: request 42, decode_budget=8. submit()
        // pushes onto the inbox under inbox_mtx_ and then calls
        // inbox_cv_.notify_one(), waking the engine task.
        submit_request req1;
        req1.request_id    = 42;
        req1.prompt_tokens = shared_prompt;
        req1.decode_budget = 8;
        req1.want_stream   = false;
        submit_handle h1 = eng.submit_request(std::move(req1));
        if (h1.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be nullopt when "
                "want_stream=false (req 42)");
        }
        request_result r1 = h1.result.get();

        // Second submission: request 43, decode_budget=8. With
        // keep_alive=true the engine is back in outer-loop idle-wait
        // (or transitioning to it) between r1 completion and this
        // submit; the next notify_one wakes it again.
        submit_request req2;
        req2.request_id    = 43;
        req2.prompt_tokens = shared_prompt;
        req2.decode_budget = 8;
        req2.want_stream   = false;
        submit_handle h2 = eng.submit_request(std::move(req2));
        if (h2.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be nullopt when "
                "want_stream=false (req 43)");
        }
        request_result r2 = h2.result.get();

        // M3a: drained-shutdown handshake. Both results are in, so
        // any_active() is false, the waiting queue is empty, and
        // the inbox is empty. request_shutdown() sets
        // shutdown_requested_ under inbox_mtx_ and notifies
        // inbox_cv_; the outer-loop tail observes the flag on a
        // drained engine, sets engine_shutdown_observed=1, and
        // breaks. engine_fut.get() then returns cleanly.
        eng.request_shutdown();
        engine_fut.get();

        if (r1.status != request_status::completed) {
            return fail_and_cleanup(
                "request 42 status != completed");
        }
        if (r2.status != request_status::completed) {
            return fail_and_cleanup(
                "request 43 status != completed");
        }
        if (r1.n_decoded != 8) {
            return fail_and_cleanup("request 42 n_decoded != 8");
        }
        if (r2.n_decoded != 8) {
            return fail_and_cleanup("request 43 n_decoded != 8");
        }

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.arrival_drained_count != 2) {
            return fail_and_cleanup(
                "arrival_drained_count != 2");
        }
        if (er.external_admitted_count != 2) {
            return fail_and_cleanup(
                "external_admitted_count != 2");
        }
        if (er.admitted_count != 2) {
            return fail_and_cleanup("admitted_count != 2");
        }
        if (er.engine_task_count != 1) {
            return fail_and_cleanup("engine_task_count != 1");
        }
        if (er.engine_idle_waits < 1) {
            return fail_and_cleanup("engine_idle_waits < 1");
        }
        if (er.engine_shutdown_observed != 1) {
            return fail_and_cleanup(
                "engine_shutdown_observed != 1");
        }

        print_result(r1);
        print_result(r2);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
