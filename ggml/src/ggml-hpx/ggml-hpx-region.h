#pragma once

// ggml-hpx-region.h
//
// Coarse execution unit for the HPX executor.
//
// A region is NOT a ggml op. It is a planner-defined coarse unit that maps
// to one of:
//   - a contiguous CPU-only subgraph (decode and prefill)
//   - a delegated BLAS-supported region (opaque to HPX)
//   - a boundary derived from a scheduler split (prefill only)
//
// Every region has exactly one synchronization boundary: nothing after the
// region may start until the region is complete. HPX may fan work inside a
// CPU region however it likes; the scheduler never sees that internal
// structure.
//
// Dependencies are expressed as a bitmask over region indices within the
// same plan. This limits a single plan to GGML_HPX_MAX_REGIONS regions,
// which is sufficient for both decode and realistic prefill workloads.

#include <cstddef>
#include <cstdint>

inline constexpr uint32_t GGML_HPX_MAX_REGIONS = 64;

// dep_mask is uint64_t: each bit represents one predecessor region.
// The constant above must never exceed the bit width of that field.
static_assert(GGML_HPX_MAX_REGIONS <= 64u,
    "dep_mask is uint64_t; GGML_HPX_MAX_REGIONS must not exceed 64");

// Execution strategy for a region.
enum class ggml_hpx_region_type : uint8_t
{
    cpu_contiguous,    // maximal contiguous CPU-only subgraph; HPX fans work inside
    blas_delegated,    // BLAS-capable region; treated as opaque, not planned internally
    sched_split,       // coarse boundary derived from a scheduler split (prefill path)
};

// A single coarse execution unit.
//
// node_begin / node_end are indices into the cgraph->nodes array at
// plan-build time. They are positional, not pointer-based. Whether they
// remain meaningful across runs depends on plan validation: the plan cache
// must confirm that the graph shape and node order are still compatible
// before these indices are trusted.
//
// dep_mask is a bitmask over region indices within the same plan. Bit k set
// means this region may not start until region k is complete.
struct ggml_hpx_region
{
    ggml_hpx_region_type type;
    uint32_t node_begin;    // inclusive
    uint32_t node_end;      // exclusive
    uint64_t dep_mask;      // bitmask of prerequisite region indices
};
