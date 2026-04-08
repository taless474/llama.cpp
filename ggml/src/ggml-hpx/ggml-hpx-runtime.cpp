// ggml-hpx-runtime.cpp
//
// Stub implementation — HPX thread teams not yet wired.
// Scratch buffer support is real; dispatch runs serially on the caller thread.

#include "ggml-hpx-runtime.h"

#include <cstdlib>

struct ggml_hpx_runtime
{
    void*  scratch_ptr   = nullptr;
    size_t scratch_bytes = 0;
};

ggml_hpx_runtime* ggml_hpx_runtime_create(
    ggml_hpx_runtime_params const& params)
{
    auto* rt = new ggml_hpx_runtime{};

    if (params.initial_scratch_bytes > 0)
    {
        rt->scratch_ptr = std::malloc(params.initial_scratch_bytes);
        if (rt->scratch_ptr != nullptr)
            rt->scratch_bytes = params.initial_scratch_bytes;
    }

    return rt;
}

void ggml_hpx_runtime_destroy(ggml_hpx_runtime* rt)
{
    if (rt->scratch_ptr != nullptr)
        std::free(rt->scratch_ptr);
    delete rt;
}

void* ggml_hpx_runtime_scratch_ptr(ggml_hpx_runtime* rt)
{
    if (rt == nullptr)
        return nullptr;
    return rt->scratch_ptr;
}

void ggml_hpx_runtime_scratch_ensure(ggml_hpx_runtime* rt, size_t bytes)
{
    if (rt == nullptr)
        return;
    if (bytes <= rt->scratch_bytes)
        return;

    void* next = std::malloc(bytes);
    if (next == nullptr)
        return;

    std::free(rt->scratch_ptr);
    rt->scratch_ptr   = next;
    rt->scratch_bytes = bytes;
}

void ggml_hpx_runtime_dispatch_decode(ggml_hpx_runtime* /*rt*/,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data)
{
    for (uint32_t i = 0; i < n_chunks; ++i)
        fn(i, user_data);
}

void ggml_hpx_runtime_dispatch_prefill(ggml_hpx_runtime* /*rt*/,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data)
{
    for (uint32_t i = 0; i < n_chunks; ++i)
        fn(i, user_data);
}
