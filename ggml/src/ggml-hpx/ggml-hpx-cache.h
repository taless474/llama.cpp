#pragma once

// ggml-hpx-cache.h
//
// Plan cache for the HPX executor.
//
// The cache stores at most one decode plan and one prefill plan at a time.
// That is sufficient because graph structure rarely changes between runs in
// the same session, and the two modes are mutually exclusive per run.
//
// Responsibilities:
//   - store a plan by value (owns it)
//   - return a const pointer to a stored plan when the key matches
//   - replace a stored plan on insert; return pointer to newly stored plan
//   - clear stored plans on request
//
// The cache does not diagnose mismatches. A null return from a lookup means
// miss; the exec layer calls ggml_hpx_decode_plan_check or
// ggml_hpx_prefill_plan_check directly when it needs the mismatch reason
// for diagnostics or counters.
//
// Thread safety: none. The cache is owned by the executor and accessed only
// from the exec dispatch path, which is single-entry per run.

#include "ggml-hpx-plan.h"

#include <optional>

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

struct ggml_hpx_plan_cache
{
    std::optional<ggml_hpx_decode_plan> decode;
    std::optional<ggml_hpx_prefill_plan> prefill;
};

// ---------------------------------------------------------------------------
// Decode plan: lookup, insert, clear
// ---------------------------------------------------------------------------

// Look up the stored decode plan against a candidate key.
//
// Returns a const pointer to the stored plan if the key matches (hit).
// Returns nullptr if no plan is stored or the key does not match (miss).
// The returned pointer is valid until the next insert or clear on this cache.
//
// Defined in ggml-hpx-cache.cpp.
ggml_hpx_decode_plan const* ggml_hpx_cache_lookup_decode(
    ggml_hpx_plan_cache& cache, ggml_hpx_plan_key const& candidate);

// Store a decode plan, replacing any previously stored plan. The plan is
// moved into the cache. Returns a const pointer to the stored plan so the
// exec layer can use it immediately without a second lookup.
// The pointer is valid until the next insert or clear on this cache.
//
// Defined in ggml-hpx-cache.cpp.
ggml_hpx_decode_plan const* ggml_hpx_cache_insert_decode(
    ggml_hpx_plan_cache& cache, ggml_hpx_decode_plan plan);

// Remove the stored decode plan, if any.
//
// Defined in ggml-hpx-cache.cpp.
void ggml_hpx_cache_clear_decode(ggml_hpx_plan_cache& cache);

// ---------------------------------------------------------------------------
// Prefill plan: lookup, insert, clear
// ---------------------------------------------------------------------------

// Look up the stored prefill plan against a candidate key.
//
// Returns a const pointer to the stored plan if the key matches (hit).
// Returns nullptr if no plan is stored or the key does not match (miss).
// The returned pointer is valid until the next insert or clear on this cache.
//
// Defined in ggml-hpx-cache.cpp.
ggml_hpx_prefill_plan const* ggml_hpx_cache_lookup_prefill(
    ggml_hpx_plan_cache& cache, ggml_hpx_plan_key const& candidate);

// Store a prefill plan, replacing any previously stored plan. The plan is
// moved into the cache. Returns a const pointer to the stored plan so the
// exec layer can use it immediately without a second lookup.
// The pointer is valid until the next insert or clear on this cache.
//
// Defined in ggml-hpx-cache.cpp.
ggml_hpx_prefill_plan const* ggml_hpx_cache_insert_prefill(
    ggml_hpx_plan_cache& cache, ggml_hpx_prefill_plan plan);

// Remove the stored prefill plan, if any.
//
// Defined in ggml-hpx-cache.cpp.
void ggml_hpx_cache_clear_prefill(ggml_hpx_plan_cache& cache);

// ---------------------------------------------------------------------------
// Combined clear
// ---------------------------------------------------------------------------

// Remove both stored plans. Use when the graph structure has changed in a
// way that invalidates both modes simultaneously.
//
// Defined in ggml-hpx-cache.cpp.
void ggml_hpx_cache_clear(ggml_hpx_plan_cache& cache);
