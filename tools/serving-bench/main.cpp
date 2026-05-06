#include "harness.h"
#include "backend.h"
#include "backend_std.h"
#include "backend_hpx.h"
#include "runtime_hpx.h"

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

// HPX orchestration carrier budget for v1.
//
// Mirrors the std backend's "one OS thread per worker" shape: at most one
// carrier per concurrent in-flight request, capped by the number of
// llama_context objects (no extra carriers can be put to useful work). At
// least one carrier so HPX can always make progress.
//
// Local to main.cpp on purpose: runtime_hpx.h must not see harness_config.
static int32_t os_threads_for(const serving_bench::harness_config & cfg) {
    return std::max<int32_t>(1, std::min(cfg.n_concurrent, cfg.n_contexts));
}

int main(int argc, char ** argv) {
    serving_bench::harness_config cfg;
    const auto pr = serving_bench::parse_cli(argc, argv, cfg);
    if (pr == serving_bench::parse_result::help) {
        return 0;
    }
    if (pr == serving_bench::parse_result::error) {
        return 1;
    }

    if (cfg.model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        return 1;
    }
    if (cfg.prompt.empty()) {
        std::fprintf(stderr, "error: --prompt is required\n");
        return 1;
    }
    if (cfg.n_contexts   < 1) cfg.n_contexts   = 1;
    if (cfg.n_concurrent < 1) cfg.n_concurrent = 1;
    if (cfg.n_requests   < 1) cfg.n_requests   = 1;

    if (!cfg.max_tokens_plan.empty()
        && static_cast<int32_t>(cfg.max_tokens_plan.size()) != cfg.n_requests) {
        std::fprintf(stderr,
                     "error: --max-tokens-plan has %zu entries but --n-requests is %d\n",
                     cfg.max_tokens_plan.size(), cfg.n_requests);
        return 1;
    }

    bool started_hpx = false;
    std::unique_ptr<serving_bench::engine> eng;
    if (cfg.backend == "std") {
        eng = serving_bench::make_engine_std();
    } else if (cfg.backend == "hpx") {
        // Order matters: HPX runtime must come up before any HPX-aware
        // engine construction, and must come down only after the engine
        // (and llama_backend) are torn down.
        if (!serving_bench::hpx_runtime_start_once(os_threads_for(cfg))) {
            // Either LLAMA_SERVING_BENCH_HPX=OFF (stub message printed by
            // runtime_hpx.cpp) or hpx::start failed (also printed there).
            // Exit before any model load or request submission.
            return 1;
        }
        started_hpx = true;
        eng = serving_bench::make_engine_hpx();
        if (!eng) {
            // HPX is up but the engine factory refused (currently always:
            // ON build, engine not yet implemented). Tear HPX down cleanly
            // before exiting.
            serving_bench::hpx_runtime_stop();
            return 1;
        }
    } else {
        // Parser already enforces the whitelist; this is a defensive guard.
        std::fprintf(stderr,
                     "[serving-bench] internal error: unknown backend \"%s\"\n",
                     cfg.backend.c_str());
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    serving_bench::print_config(stderr, cfg);

    serving_bench::engine_init_params eparams{};
    eparams.model_path        = cfg.model_path;
    eparams.n_contexts        = cfg.n_contexts;
    eparams.n_threads_per_ctx = cfg.n_threads_per_ctx;
    eparams.ctx_size          = cfg.ctx_size;
    eparams.batch_size        = cfg.batch_size;

    if (!eng->init(eparams)) {
        std::fprintf(stderr, "[serving-bench] engine init failed\n");
        eng.reset();
        llama_backend_free();
        if (started_hpx) {
            serving_bench::hpx_runtime_stop();
        }
        return 1;
    }

    const int32_t n_prompt_tokens = eng->prompt_token_count(cfg.prompt);
    if (n_prompt_tokens <= 0) {
        std::fprintf(stderr, "[serving-bench] error: failed to tokenize prompt\n");
        eng.reset();
        llama_backend_free();
        if (started_hpx) {
            serving_bench::hpx_runtime_stop();
        }
        return 1;
    }
    const int32_t fit_max_tokens = cfg.max_tokens_plan.empty()
        ? cfg.max_tokens
        : *std::max_element(cfg.max_tokens_plan.begin(), cfg.max_tokens_plan.end());
    if (n_prompt_tokens + fit_max_tokens > cfg.ctx_size) {
        std::fprintf(stderr,
                     "[serving-bench] error: prompt uses %d tokens, max_tokens=%d, "
                     "ctx_size=%d; request exceeds context size\n",
                     n_prompt_tokens, fit_max_tokens, cfg.ctx_size);
        eng.reset();
        llama_backend_free();
        if (started_hpx) {
            serving_bench::hpx_runtime_stop();
        }
        return 1;
    }
    std::fprintf(stderr,
                 "[serving-bench] prompt fits: %d prompt tokens + %d max_tokens <= %d ctx_size\n",
                 n_prompt_tokens, fit_max_tokens, cfg.ctx_size);

    const int32_t n_req = cfg.n_requests;
    const int32_t n_cc  = cfg.n_concurrent;

    std::vector<serving_bench::request_result> results;
    results.reserve(static_cast<size_t>(n_req));

    const auto t0 = std::chrono::steady_clock::now();

    int32_t next_idx = 0;
    auto submit_one = [&]() {
        serving_bench::request_params rp{};
        rp.prompt        = cfg.prompt;
        rp.max_tokens    = cfg.max_tokens_plan.empty()
                               ? cfg.max_tokens
                               : cfg.max_tokens_plan[next_idx];
        rp.seed          = cfg.seed_base + static_cast<uint32_t>(next_idx);
        rp.request_index = next_idx;
        next_idx++;
        return eng->submit(std::move(rp));
    };

    std::vector<std::future<serving_bench::request_result>> in_flight;
    in_flight.reserve(static_cast<size_t>(n_cc));

    while (next_idx < n_req && static_cast<int32_t>(in_flight.size()) < n_cc) {
        in_flight.emplace_back(submit_one());
    }

    while (!in_flight.empty()) {
        bool collected = false;
        for (size_t i = 0; i < in_flight.size(); i++) {
            if (in_flight[i].wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready) {
                results.push_back(in_flight[i].get());
                in_flight.erase(in_flight.begin() + static_cast<long>(i));
                if (next_idx < n_req) {
                    in_flight.emplace_back(submit_one());
                }
                collected = true;
                break;
            }
        }
        if (!collected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    const auto t1   = std::chrono::steady_clock::now();
    const auto wall = t1 - t0;

    for (size_t i = 0; i < results.size(); i++) {
        const auto & r = results[i];
        const char * status_str =
            r.status == serving_bench::request_status::ok        ? "ok"        :
            r.status == serving_bench::request_status::cancelled ? "cancelled" :
                                                                   "error";
        const double ttft_ms  = static_cast<double>(r.ttft .count()) / 1.0e6;
        const double total_ms = static_cast<double>(r.total.count()) / 1.0e6;
        std::fprintf(stdout,
                     "[serving-bench] req[%zu] status=%s n_tokens_generated=%d "
                     "generated_token_hash=0x%016llx ttft_ms=%.3f total_ms=%.3f\n",
                     i, status_str, r.n_tokens_generated,
                     static_cast<unsigned long long>(r.generated_token_hash),
                     ttft_ms, total_ms);
    }

    const auto agg = serving_bench::compute_aggregate(results, wall);
    serving_bench::print_aggregate(stdout, agg);

    eng.reset();
    llama_backend_free();
    if (started_hpx) {
        serving_bench::hpx_runtime_stop();
    }
    return 0;
}
