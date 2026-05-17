#include "hpx_runtime.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/init.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>

namespace hpx_runtime {

namespace {

std::once_flag    g_start_flag;
std::atomic<bool> g_running{false};

bool trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_HPX_CB_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool do_start(int32_t os_threads) noexcept {
    try {
        hpx::init_params init_args;
        if (os_threads > 0) {
            init_args.cfg.push_back(
                std::string("hpx.os_threads=") +
                std::to_string(os_threads));
        }
        if (!hpx::start(nullptr, 0, nullptr, init_args)) {
            fprintf(stderr,
                "[hpx-cb-gate] hpx::start returned false\n");
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        fprintf(stderr,
            "[hpx-cb-gate] hpx::start threw: %s\n", e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[hpx-cb-gate] hpx::start threw unknown\n");
        return false;
    }
}

}  // namespace

bool start_once(int32_t os_threads) {
    std::call_once(g_start_flag, [&]() {
        if (trace_enabled()) {
            fprintf(stderr,
                "[hpx-cb-gate] hpx_runtime_start_once: starting "
                "(os_threads=%d)\n", os_threads);
        }
        if (do_start(os_threads)) {
            g_running.store(true);
        }
    });
    return g_running.load();
}

void stop() {
    if (!g_running.exchange(false)) {
        return;
    }
    if (trace_enabled()) {
        fprintf(stderr, "[hpx-cb-gate] hpx_runtime_stop\n");
    }
    try {
        hpx::post([]() { hpx::finalize(); });
        hpx::stop();
    } catch (const std::exception & e) {
        fprintf(stderr,
            "[hpx-cb-gate] hpx::stop threw: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "[hpx-cb-gate] hpx::stop threw unknown\n");
    }
}

}  // namespace hpx_runtime
