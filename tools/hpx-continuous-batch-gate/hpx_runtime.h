// Process-wide HPX runtime start/stop for the continuous-batch gate.
//
// `start_once(os_threads, cfg)` is idempotent — the first call brings the
// HPX runtime up with `hpx::start`; subsequent calls return the cached
// running state. It returns `false` if HPX failed to start, OR if a
// preflight check on `cfg` rejected the configuration BEFORE entering
// the call_once latch (so a rejected config does not poison future
// start_once calls). `stop()` finalizes and joins the runtime exactly
// once; further calls are no-ops.
//
// `runtime_config.enable_engine_pool` opts into a single-PU named HPX
// thread pool ("engine") created via the resource_partitioner at
// startup. When enabled, `async_on_engine(f)` spawns `f` on a
// parallel_executor bound to that pool. When disabled (default),
// `async_on_engine(f)` calls bare `hpx::async(f)` — byte-identical
// to the legacy spawn form. When enable_engine_pool was requested at
// startup but the named pool is unavailable at spawn time,
// `async_on_engine` throws `std::runtime_error` rather than silently
// downgrading to the default pool.
//
// Both functions log under `LLAMA_HPX_CB_TRACE=1` with the existing
// `[hpx-cb-gate] hpx_runtime_*` stderr lines. Engine-pool placement
// emits additional lines under `LLAMA_HPX_PLACEMENT_TRACE=1`
// (default off):
//   engine_pool=created pool_name=engine engine_pus=N default_pus=M
//   engine_task_placement pool=<actual current pool name>
//
// The runtime state itself (call_once latch, running atomic,
// engine-pool latches) is owned by `hpx_runtime.cpp`, so there is
// exactly one process-wide latch regardless of how many TUs include
// this header.

#pragma once

#include <cstdint>
#include <functional>

#include <hpx/hpx.hpp>

namespace hpx_runtime {

struct runtime_config {
    bool enable_engine_pool = false;
};

bool start_once(int32_t os_threads, runtime_config cfg = {});

void stop();

// Spawn `f` on the dedicated "engine" HPX thread pool when it was
// created by a prior successful start_once(..., {enable_engine_pool=
// true}); otherwise spawn on the default HPX scheduler via bare
// hpx::async (byte-identical to the legacy form). Throws
// std::runtime_error if the engine pool was requested at startup but
// is not available at spawn time. Engine-task only.
hpx::future<void> async_on_engine(std::function<void()> f);

}  // namespace hpx_runtime
