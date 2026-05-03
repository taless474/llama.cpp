#include "runtime_hpx.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef LLAMA_SERVING_BENCH_HPX
#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/init.hpp>
#endif

namespace serving_bench {

#ifdef LLAMA_SERVING_BENCH_HPX

namespace {

std::once_flag    g_start_flag;
std::atomic<bool> g_running{false};

// Lifecycle traces are noisy and only useful when diagnosing HPX startup or
// shutdown ordering. Errors and exception messages stay unconditional; per
// CLAUDE.md, only non-error lifecycle output is gated.
bool trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_SERVING_BENCH_HPX_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool do_start(int32_t os_threads) noexcept {
    try {
        hpx::init_params init_args;
        if (os_threads > 0) {
            init_args.cfg.push_back(
                std::string("hpx.os_threads=") + std::to_string(os_threads));
        }
        if (!hpx::start(nullptr, 0, nullptr, init_args)) {
            std::fprintf(stderr, "[serving-bench] hpx::start failed\n");
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[serving-bench] hpx::start threw: %s\n", e.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[serving-bench] hpx::start threw unknown\n");
        return false;
    }
}

} // namespace

bool hpx_runtime_start_once(int32_t os_threads) {
    std::call_once(g_start_flag, [&]() {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[serving-bench] hpx_runtime_start_once: starting (os_threads=%d)\n",
                         os_threads);
        }
        if (do_start(os_threads)) {
            g_running.store(true);
        }
    });
    return g_running.load();
}

void hpx_runtime_stop() {
    if (!g_running.exchange(false)) {
        return;
    }
    if (trace_enabled()) {
        std::fprintf(stderr, "[serving-bench] hpx_runtime_stop: stopping\n");
    }
    try {
        hpx::post([]() { hpx::finalize(); });
        hpx::stop();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[serving-bench] hpx::stop threw: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[serving-bench] hpx::stop threw unknown\n");
    }
}

#else

bool hpx_runtime_start_once(int32_t /*os_threads*/) {
    std::fprintf(stderr,
                 "[serving-bench] HPX backend not built "
                 "(LLAMA_SERVING_BENCH_HPX=OFF)\n");
    return false;
}

void hpx_runtime_stop() {
    // OFF: nothing to do.
}

#endif

} // namespace serving_bench
