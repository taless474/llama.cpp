// ggml-hpx-runtime.cpp
//
// Decode dispatch: serial (always).
// Prefill dispatch: parallel via hpx::async on the default pool when
// n_chunks >= 2; serial fallback for n_chunks < 2.
// hpx::wait_all rethrows stored exceptions while waiting; no separate
// get() loop is needed after it.
//
// HPX runtime lifecycle: started once per process on first create() via
// std::call_once, and shut down at process exit via atexit. Runtime
// create/destroy manage only the ggml_hpx_runtime object and its scratch;
// they do not bracket HPX startup/shutdown.

#include "ggml-hpx-runtime.h"

#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/future.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/post.hpp>

#include <cstdlib>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// HPX global lifecycle
//
// HPX cannot be started more than once per process.  The first call to
// hpx_acquire() starts the runtime; all subsequent calls are no-ops.
// An atexit handler posts hpx::finalize() onto an HPX thread (it must run
// from within HPX) and then calls hpx::stop() to drain and shut down.
// hpx_release() is a no-op: there is nothing to undo between runtimes.
// ---------------------------------------------------------------------------

namespace
{

std::once_flag g_hpx_start_flag;

void hpx_acquire()
{
    std::call_once(g_hpx_start_flag, []() {
        hpx::start(nullptr, 0, nullptr);
        std::atexit([]() {
            hpx::post([]() { hpx::finalize(); });
            hpx::stop();
        });
    });
}

void hpx_release() noexcept {}

}    // namespace

// ---------------------------------------------------------------------------
// Runtime struct
// ---------------------------------------------------------------------------

struct ggml_hpx_runtime
{
    uint32_t n_prefill_threads = 0;
    void*    scratch_ptr       = nullptr;
    size_t   scratch_bytes     = 0;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_hpx_runtime* ggml_hpx_runtime_create(
    ggml_hpx_runtime_params const& params)
{
    hpx_acquire();

    auto* rt              = new ggml_hpx_runtime{};
    rt->n_prefill_threads = params.n_prefill_threads;

    if (params.initial_scratch_bytes > 0)
    {
        rt->scratch_ptr = std::malloc(params.initial_scratch_bytes);
        if (rt->scratch_ptr != nullptr)
        {
            rt->scratch_bytes = params.initial_scratch_bytes;
        }
    }

    return rt;
}

void ggml_hpx_runtime_destroy(ggml_hpx_runtime* rt)
{
    if (rt->scratch_ptr != nullptr)
    {
        std::free(rt->scratch_ptr);
    }
    delete rt;

    hpx_release();
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

void* ggml_hpx_runtime_scratch_ptr(ggml_hpx_runtime* rt)
{
    if (rt == nullptr)
    {
        return nullptr;
    }
    return rt->scratch_ptr;
}

void ggml_hpx_runtime_scratch_ensure(ggml_hpx_runtime* rt, size_t bytes)
{
    if (rt == nullptr)
    {
        return;
    }
    if (bytes <= rt->scratch_bytes)
    {
        return;
    }

    void* next = std::malloc(bytes);
    if (next == nullptr)
    {
        return;
    }

    std::free(rt->scratch_ptr);
    rt->scratch_ptr   = next;
    rt->scratch_bytes = bytes;
}

// ---------------------------------------------------------------------------
// Dispatch — decode: always serial
// ---------------------------------------------------------------------------

void ggml_hpx_runtime_dispatch_decode(ggml_hpx_runtime* /*rt*/,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data)
{
    if (fn == nullptr || n_chunks == 0)
    {
        return;
    }

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        fn(i, user_data);
    }
}

// ---------------------------------------------------------------------------
// Dispatch — prefill: parallel on default pool when n_chunks >= 2
// ---------------------------------------------------------------------------

void ggml_hpx_runtime_dispatch_prefill(ggml_hpx_runtime* /*rt*/,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data)
{
    if (fn == nullptr || n_chunks == 0)
    {
        return;
    }

    if (n_chunks < 2)
    {
        fn(0, user_data);
        return;
    }

    std::vector<hpx::future<void>> futures;
    futures.reserve(n_chunks);

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        futures.push_back(
            hpx::async(hpx::launch::async,
                [fn, i, user_data]() { fn(i, user_data); }));
    }

    // wait_all waits for all futures to become ready and rethrows any
    // stored exceptions. No separate get() loop is needed.
    hpx::wait_all(futures);
}
