#include "hpx_runtime.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/init.hpp>

#include <hpx/threading_base/thread_helpers.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace hpx_runtime {

namespace {

constexpr const char * kEnginePoolName = "engine";

std::once_flag    g_start_flag;
std::atomic<bool> g_running{false};
// Latched in start_once BEFORE call_once when cfg.enable_engine_pool
// is true. Sticky across calls; the first successful start_once with
// cfg.enable_engine_pool=true binds this for the process lifetime.
std::atomic<bool> g_engine_pool_requested{false};
// Set from inside the rp_callback after the named pool was created
// and the PU was assigned. Cleared until proven created.
std::atomic<bool> g_engine_pool_created{false};

bool trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_HPX_CB_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool placement_trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_HPX_PLACEMENT_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

void install_rp_callback(hpx::init_params & init_args) {
    init_args.rp_callback =
        [](hpx::resource::partitioner & rp,
           hpx::program_options::variables_map const &) {
            // Defensive topology check: HPX-reported numa_domains /
            // cores / pus must be non-empty. If the partitioner sees
            // no usable PU we leave the engine pool uncreated; the
            // post-call_once code in start_once will surface this as
            // a startup failure rather than silently downgrade.
            const auto & numas = rp.numa_domains();
            if (numas.empty()
             || numas.front().cores().empty()
             || numas.front().cores().front().pus().empty()) {
                fprintf(stderr,
                    "[hpx-cb-gate] engine_pool: rp_callback found "
                    "no usable PU (numas=%zu); pool not created\n",
                    numas.size());
                return;
            }
            try {
                rp.create_thread_pool(kEnginePoolName);
                const auto & pu_ref =
                    numas.front().cores().front().pus().front();
                rp.add_resource(pu_ref, kEnginePoolName,
                                /*exclusive=*/true,
                                /*num_threads=*/1);
                g_engine_pool_created.store(true);
            } catch (const std::exception & e) {
                fprintf(stderr,
                    "[hpx-cb-gate] engine_pool: rp_callback threw: "
                    "%s\n", e.what());
            } catch (...) {
                fprintf(stderr,
                    "[hpx-cb-gate] engine_pool: rp_callback threw "
                    "unknown\n");
            }
        };
}

bool do_start(int32_t os_threads, runtime_config cfg) noexcept {
    try {
        hpx::init_params init_args;
        if (os_threads > 0) {
            init_args.cfg.push_back(
                std::string("hpx.os_threads=") +
                std::to_string(os_threads));
        }
        if (cfg.enable_engine_pool) {
            install_rp_callback(init_args);
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

// Emit a one-shot placement trace from inside the engine HPX task,
// gated on LLAMA_HPX_PLACEMENT_TRACE=1. Uses the public HPX API
// hpx::this_thread::get_pool(ec) to discover the pool that is
// running the current HPX worker. The lightweight error_code form
// keeps the trace path noexcept; the wrapped lambda in
// async_on_engine() is invoked from an HPX worker by construction,
// so a non-null pool is the expected outcome and the "<unknown>"
// fallback only fires if HPX returns an error.
void engine_placement_trace_if_enabled() {
    if (!placement_trace_enabled()) return;
    static std::atomic<bool> already_emitted{false};
    if (already_emitted.exchange(true)) return;
    hpx::error_code ec(hpx::throwmode::lightweight);
    auto * pool = hpx::this_thread::get_pool(ec);
    const char * pname = (!ec && pool != nullptr)
        ? pool->get_pool_name().c_str()
        : "<unknown>";
    fprintf(stderr,
        "[hpx-cb-gate] engine_task_placement pool=%s\n", pname);
}

}  // namespace

bool start_once(int32_t os_threads, runtime_config cfg) {
    // Preflight validation runs OUTSIDE call_once so a rejected cfg
    // does not consume g_start_flag. Only the
    // engine-pool-vs-os_threads relationship is checked here; real
    // HPX-topology checks (PU availability, etc.) live inside the
    // rp_callback because the partitioner is only available there.
    if (cfg.enable_engine_pool && os_threads < 2) {
        fprintf(stderr,
            "[hpx-cb-gate] start_once: enable_engine_pool requires "
            "os_threads >= 2 (got %d); not starting HPX\n",
            os_threads);
        return false;
    }
    if (cfg.enable_engine_pool) {
        g_engine_pool_requested.store(true);
    }

    std::call_once(g_start_flag, [&]() {
        if (trace_enabled()) {
            fprintf(stderr,
                "[hpx-cb-gate] hpx_runtime_start_once: starting "
                "(os_threads=%d enable_engine_pool=%d)\n",
                os_threads,
                static_cast<int>(cfg.enable_engine_pool));
        }
        if (do_start(os_threads, cfg)) {
            g_running.store(true);
        }
    });

    // If the engine pool was requested but the rp_callback failed to
    // create it (topology empty, internal HPX failure, etc.), surface
    // that as a startup failure. Never silently downgrade an explicit
    // placement request to default-pool behavior.
    if (g_engine_pool_requested.load() && g_running.load()) {
        if (!g_engine_pool_created.load()
         || !hpx::resource::pool_exists(kEnginePoolName)) {
            fprintf(stderr,
                "[hpx-cb-gate] start_once: engine pool requested "
                "but rp_callback did not create '%s'\n",
                kEnginePoolName);
            return false;
        }
        if (placement_trace_enabled()) {
            const std::size_t engine_pus =
                hpx::resource::get_num_threads(kEnginePoolName);
            const std::size_t total_pus =
                hpx::resource::get_num_threads();
            fprintf(stderr,
                "[hpx-cb-gate] engine_pool=created pool_name=%s "
                "engine_pus=%zu default_pus=%zu\n",
                kEnginePoolName,
                engine_pus,
                (total_pus >= engine_pus
                    ? total_pus - engine_pus : 0));
        }
    }

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

hpx::future<void> async_on_engine(std::function<void()> f) {
    if (!g_engine_pool_requested.load()) {
        // Default-off path: byte-identical to the legacy spawn form.
        return hpx::async(std::move(f));
    }
    // Engine pool was requested. start_once would have returned
    // false if creation failed, but recheck at spawn time anyway —
    // never silently downgrade an explicit placement request.
    if (!g_engine_pool_created.load()
     || !hpx::resource::pool_exists(kEnginePoolName)) {
        throw std::runtime_error(
            "hpx_runtime::async_on_engine: engine pool requested "
            "but not available");
    }
    auto wrapped = [g = std::move(f)]() mutable {
        engine_placement_trace_if_enabled();
        std::move(g)();
    };
    hpx::execution::parallel_executor exec(
        &hpx::resource::get_thread_pool(kEnginePoolName));
    return hpx::async(exec, std::move(wrapped));
}

}  // namespace hpx_runtime
