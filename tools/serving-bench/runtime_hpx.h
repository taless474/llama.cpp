#pragma once

#include <cstdint>

namespace serving_bench {

// Process-wide, one-shot HPX runtime startup.
//
// The first successful call starts the HPX runtime with exactly `os_threads`
// orchestration carrier threads. Subsequent calls in the same process are
// no-ops and IGNORE `os_threads`. The runtime is never restarted within a
// process, per the project's HPX hard invariant
// ("HPX runtime startup must be process-wide and one-shot").
//
// `os_threads` budgets only the HPX orchestration carriers. llama.cpp still
// uses its own per-context compute threads via `n_threads_per_ctx`; HPX
// carrier threads do not displace them. Callers should pass `n_contexts` so
// that the HPX backend's orchestration thread budget mirrors the std
// backend's one-OS-thread-per-worker shape.
//
// Returns true if the runtime is up after this call (whether started here
// or previously). Returns false only when the first start attempt failed,
// or when LLAMA_SERVING_BENCH_HPX is OFF (in which case a clean error is
// logged to stderr).
//
// This function may only be called from non-HPX threads (typically `main`).
bool hpx_runtime_start_once(int32_t os_threads);

// Process-wide HPX runtime shutdown.
//
// Safe to call zero or one times per process, on a non-HPX thread, after
// all HPX engine instances have been destroyed and after every future
// returned by hpx-backend submit() has been collected.
//
// After this call no further HPX engine operations are valid in this
// process. The runtime is not re-startable; calling
// hpx_runtime_start_once() after hpx_runtime_stop() will not restart it.
void hpx_runtime_stop();

} // namespace serving_bench
