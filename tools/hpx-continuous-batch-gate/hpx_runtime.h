// Process-wide HPX runtime start/stop for the continuous-batch gate.
//
// `start_once(os_threads)` is idempotent — the first call brings the HPX
// runtime up with `hpx::start`; subsequent calls return the cached
// running state. It returns `false` if HPX failed to start. `stop()`
// finalizes and joins the runtime exactly once; further calls are no-ops.
//
// Both functions log under `LLAMA_HPX_CB_TRACE=1` with the existing
// `[hpx-cb-gate] hpx_runtime_*` stderr lines. The runtime state itself
// (the `std::once_flag` latch and the "running" atomic) is owned by
// `hpx_runtime.cpp`, so there is exactly one process-wide latch
// regardless of how many TUs include this header.

#pragma once

#include <cstdint>

namespace hpx_runtime {

bool start_once(int32_t os_threads);

void stop();

}  // namespace hpx_runtime
