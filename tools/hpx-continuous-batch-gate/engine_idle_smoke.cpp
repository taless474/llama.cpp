// engine_idle_smoke.cpp — M2f: initial-idle-slot admission smoke for
// llama-hpx-engine. Proves submit_request can be served from an engine
// constructed with zero initial active requests, by consuming a slot
// from the new initial-idle pool (engine_options::initial_idle_slots).
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
        "  Minimal initial-idle-slot smoke for the llama-hpx-engine "
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
    fprintf(stdout, "HPX_ENGINE_IDLE_SMOKE: FAIL: %s\n", reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_ENGINE_IDLE_SMOKE: PASS\n");
    fflush(stdout);
}

void print_result(const request_result & r) {
    fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d hash=0x%016llx "
            "admission_source=%s\n",
            r.request_id, status_name(r.status), r.n_decoded,
            static_cast<unsigned long long>(r.hash),
            admission_source_name(r.admission_src));
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
        ctx, "Once upon a time", /*add_special=*/true,
        /*parse_special=*/true);
    if (shared_prompt.empty()) {
        return fail_and_cleanup("tokenization produced 0 tokens");
    }

    const int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(shared_prompt.size()), 1);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.ctx                = ctx;
    opts.vocab              = vocab;
    opts.n_vocab            = n_vocab;
    // Borrowed reference required to be non-null by the engine ctor.
    // The idle path never consumes prompt_tokens_ for prefill because
    // there are no initial actives; the borrow is kept alive solely to
    // satisfy the existing engine_options contract.
    opts.prompt_tokens      = &shared_prompt;
    opts.budgets            = {};
    opts.batch_capacity     = batch_capacity;
    opts.cancel_after       = -1;
    opts.n_seq_max          = 1;
    opts.waiting_queue      = &empty_waiting;
    opts.reuse_completed    = false;
    opts.max_decode_iters   = 0;
    opts.stream_all         = false;
    opts.initial_idle_slots = 1;

    try {
        engine eng(std::move(opts));

        // No take_futures() — there are no initial active futures.

        submit_request req;
        req.request_id    = 42;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = 8;
        req.want_stream   = false;
        submit_handle h = eng.submit_request(std::move(req));
        if (h.stream.has_value()) {
            return fail_and_cleanup(
                "submit_handle.stream must be nullopt when "
                "want_stream=false");
        }

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });
        engine_fut.get();

        request_result r = h.result.get();

        if (r.status != request_status::completed) {
            return fail_and_cleanup("request status != completed");
        }
        if (r.n_decoded != 8) {
            return fail_and_cleanup("n_decoded != 8");
        }
        if (r.admission_src != admission_source::initial_idle) {
            return fail_and_cleanup(
                "admission_src != initial_idle");
        }

        const engine_result & er = eng.result();
        if (er.decode_failures != 0) {
            return fail_and_cleanup("decode_failures != 0");
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup("residual_kv_ok != true");
        }
        if (er.arrival_drained_count != 1) {
            return fail_and_cleanup("arrival_drained_count != 1");
        }
        if (er.external_admitted_count != 1) {
            return fail_and_cleanup("external_admitted_count != 1");
        }
        if (er.admitted_count != 1) {
            return fail_and_cleanup("admitted_count != 1");
        }
        if (er.streams_opened != 0) {
            return fail_and_cleanup("streams_opened != 0");
        }

        print_result(r);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
