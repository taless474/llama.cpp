#pragma once

// ggml-hpx-adapter.h
//
// Adapter interface: external graph/scheduler state → HPX-owned planning inputs.
//
// This header is the only place in the HPX layer that forward-declares types
// from ggml-backend.h. The full header is included only by
// ggml-hpx-adapter.cpp. No other .cpp in ggml-hpx/ may include
// ggml-backend.h.
//
// The adapter produces ggml_hpx_adapter_result, which bundles an immutable
// region topology snapshot and a fully populated plan key. The exec layer
// passes this to the cache and plan builder; neither reads scheduler state
// or retains graph pointers.
//
// Two strategies, one per mode:
//
//   ggml_hpx_adapt_decode   - independent graph walk; no scheduler state read
//   ggml_hpx_adapt_prefill  - scheduler-driven split translation
//
// Neither function retains the graph pointer or the scheduler pointer after
// returning. All retained data is HPX-owned (inside the result struct).

#include "ggml-hpx-fwd.h"     // ggml_cgraph, ggml_backend_sched, ggml_backend_sched_t
#include "ggml-hpx-plan.h"    // ggml_hpx_plan_key, ggml_hpx_mode
#include "ggml-hpx-topo.h"    // ggml_hpx_region_topology

#include <cstdint>

// ---------------------------------------------------------------------------
// Adapter result
// ---------------------------------------------------------------------------

// Bundled output of one adapter pass. Both fields are HPX-owned and remain
// valid across scheduler state changes and graph allocation resets.
//
// topo - immutable region topology snapshot derived from the graph or the
//        scheduler split structure; contains no external pointers
// key  - fully populated plan key; mode and policy_version are set by the
//        adapter from the function called and the policy_version argument;
//        the three hash fields are computed from the graph and, for prefill,
//        the scheduler-reported split structure
struct ggml_hpx_adapter_result
{
    ggml_hpx_region_topology topo;
    ggml_hpx_plan_key key;
};

// ---------------------------------------------------------------------------
// Decode adapter
// ---------------------------------------------------------------------------

// Produce a decode adapter result by walking the graph independently.
// Does not read scheduler state.
//
// graph           - the graph to analyse; used only during this call
// policy_version  - written into key.policy_version unchanged; the exec
//                   layer owns this constant and bumps it when planner
//                   logic changes
//
// The returned topology should be validated with
// ggml_hpx_validate_decode_topology before being passed to
// ggml_hpx_build_decode_plan.
//
// Defined in ggml-hpx-adapter.cpp.
ggml_hpx_adapter_result ggml_hpx_adapt_decode(
    ggml_cgraph const* graph, uint32_t policy_version);

// ---------------------------------------------------------------------------
// Prefill adapter
// ---------------------------------------------------------------------------

// Produce a prefill adapter result using a scheduler-driven split
// translation. Asks the scheduler to derive the split structure, then
// translates the result into an HPX-owned ggml_hpx_region_topology.
//
// graph           - the graph to analyse alongside the scheduler split
// sched           - caller-owned scheduler used only during this call.
//                   The adapter may invoke split-related scheduler functions
//                   to derive topology, but does not retain, reset, or own it.
// policy_version  - written into key.policy_version unchanged
//
// The returned topology should be validated with
// ggml_hpx_validate_prefill_topology before being passed to
// ggml_hpx_build_prefill_plan.
//
// Defined in ggml-hpx-adapter.cpp.
ggml_hpx_adapter_result ggml_hpx_adapt_prefill(
    ggml_cgraph const* graph, ggml_backend_sched_t sched,
    uint32_t policy_version);
