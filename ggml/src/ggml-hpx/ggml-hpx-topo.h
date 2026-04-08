#pragma once

// ggml-hpx-topo.h
//
// Immutable region topology snapshot.
//
// This struct is the handoff point between the adapter and the plan builder:
//   - the adapter (ggml-hpx-adapter.cpp) produces it
//   - the plan builder (ggml-hpx-plan.cpp) consumes it
//   - neither the cache nor the executor reads it directly
//
// The snapshot must not contain:
//   - raw ggml_cgraph * pointers
//   - raw tensor pointers
//   - scheduler-owned allocations
//   - any pointer into transient scheduler state
//
// The region list is HPX-owned. It is safe to reuse across runs as long as
// the plan that holds it remains valid (same graph shape, same backend
// placement, same mode policy version). When the plan is invalidated, the
// topology goes with it.
//
// n_nodes is the total node count of the graph at snapshot time. The plan
// builder uses it to validate that node_begin/node_end in each region are
// within bounds.

#include "ggml-hpx-region.h"

#include <cstdint>
#include <vector>

struct ggml_hpx_region_topology
{
    std::vector<ggml_hpx_region> regions;
    uint32_t n_nodes;    // total node count at snapshot time
};
