// ggml-hpx-cache.cpp
//
// Plan cache implementation for the HPX executor.

#include "ggml-hpx-cache.h"

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

ggml_hpx_decode_plan const* ggml_hpx_cache_lookup_decode(
    ggml_hpx_plan_cache& cache, ggml_hpx_plan_key const& candidate)
{
    if (!cache.decode.has_value())
        return nullptr;
    if (cache.decode->key != candidate)
        return nullptr;
    return &*cache.decode;
}

ggml_hpx_decode_plan const* ggml_hpx_cache_insert_decode(
    ggml_hpx_plan_cache& cache, ggml_hpx_decode_plan plan)
{
    cache.decode.emplace(std::move(plan));
    return &*cache.decode;
}

void ggml_hpx_cache_clear_decode(ggml_hpx_plan_cache& cache)
{
    cache.decode.reset();
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

ggml_hpx_prefill_plan const* ggml_hpx_cache_lookup_prefill(
    ggml_hpx_plan_cache& cache, ggml_hpx_plan_key const& candidate)
{
    if (!cache.prefill.has_value())
        return nullptr;
    if (cache.prefill->key != candidate)
        return nullptr;
    return &*cache.prefill;
}

ggml_hpx_prefill_plan const* ggml_hpx_cache_insert_prefill(
    ggml_hpx_plan_cache& cache, ggml_hpx_prefill_plan plan)
{
    cache.prefill.emplace(std::move(plan));
    return &*cache.prefill;
}

void ggml_hpx_cache_clear_prefill(ggml_hpx_plan_cache& cache)
{
    cache.prefill.reset();
}

// ---------------------------------------------------------------------------
// Combined clear
// ---------------------------------------------------------------------------

void ggml_hpx_cache_clear(ggml_hpx_plan_cache& cache)
{
    cache.decode.reset();
    cache.prefill.reset();
}
