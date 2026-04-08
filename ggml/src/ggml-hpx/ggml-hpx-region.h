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
// Predecessor link: each region records the index of its immediate
// predecessor. UINT32_MAX means "no predecessor" (first region). The
// current execution model is a strictly sequential chain; the field exists
// so the runtime can reconstruct ordering without re-deriving it from node
// ranges. There is no upper bound on region count.

#include <cstdint>

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
// prev_idx is the index of the immediate predecessor region, or UINT32_MAX
// when this is the first region. Execution must not begin until the
// predecessor is complete.
struct ggml_hpx_region
{
    ggml_hpx_region_type type;
    uint32_t node_begin;    // inclusive
    uint32_t node_end;      // exclusive
    uint32_t prev_idx;      // predecessor region index, or UINT32_MAX if none
};
