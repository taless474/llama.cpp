// ggml-hpx-plan.cpp
//
// Plan builder and validator for the HPX executor.
//
// This translation unit must not include ggml-backend.h.
// It consumes only ggml_hpx_region_topology produced by the adapter.

#include "ggml-hpx-plan.h"

#include "ggml.h"

// ---------------------------------------------------------------------------
// Decode topology validation
// ---------------------------------------------------------------------------

bool ggml_hpx_validate_decode_topology(
    ggml_hpx_region_topology const& topo)
{
    if (topo.regions.empty())
        return true;    // empty graph is valid for decode

    for (auto const& r : topo.regions)
    {
        if (r.node_begin >= r.node_end)
            return false;
        if (r.node_end > topo.n_nodes)
            return false;
        if (r.type == ggml_hpx_region_type::sched_split)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Decode plan: build and check
// ---------------------------------------------------------------------------

ggml_hpx_decode_plan ggml_hpx_build_decode_plan(
    ggml_hpx_region_topology const& topo, ggml_hpx_plan_key const& key)
{
    for (auto const& r : topo.regions)
    {
        GGML_ASSERT(r.node_begin < r.node_end);
        GGML_ASSERT(r.node_end <= topo.n_nodes);
        GGML_ASSERT(r.type != ggml_hpx_region_type::sched_split);
    }

    ggml_hpx_decode_plan plan{};
    plan.key             = key;
    plan.regions         = topo.regions;
    plan.chunk_specs.resize(
        topo.regions.size(), ggml_hpx_chunk_spec{});
    return plan;
}

ggml_hpx_plan_check_result ggml_hpx_decode_plan_check(
    ggml_hpx_decode_plan const& plan,
    ggml_hpx_plan_key const& candidate)
{
    if (plan.key.graph_shape_hash != candidate.graph_shape_hash)
        return ggml_hpx_plan_check_result::stale_graph_shape;
    if (plan.key.backend_assign_hash != candidate.backend_assign_hash)
        return ggml_hpx_plan_check_result::stale_backend_assign;
    if (plan.key.workspace_layout_sig != candidate.workspace_layout_sig)
        return ggml_hpx_plan_check_result::stale_workspace;
    if (plan.key.policy_version != candidate.policy_version)
        return ggml_hpx_plan_check_result::stale_policy;
    return ggml_hpx_plan_check_result::valid;
}

// ---------------------------------------------------------------------------
// Prefill topology validation
// ---------------------------------------------------------------------------

bool ggml_hpx_validate_prefill_topology(
    ggml_hpx_region_topology const& topo)
{
    for (auto const& r : topo.regions)
    {
        if (r.node_begin >= r.node_end)
            return false;
        if (r.node_end > topo.n_nodes)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prefill plan: build and check
// ---------------------------------------------------------------------------

ggml_hpx_prefill_plan ggml_hpx_build_prefill_plan(
    ggml_hpx_region_topology const& topo, ggml_hpx_plan_key const& key)
{
    for (auto const& r : topo.regions)
    {
        GGML_ASSERT(r.node_begin < r.node_end);
        GGML_ASSERT(r.node_end <= topo.n_nodes);
    }

    ggml_hpx_prefill_plan plan{};
    plan.key             = key;
    plan.topo            = topo;
    plan.region_policies.resize(
        topo.regions.size(), ggml_hpx_region_policy{});
    return plan;
}

ggml_hpx_plan_check_result ggml_hpx_prefill_plan_check(
    ggml_hpx_prefill_plan const& plan,
    ggml_hpx_plan_key const& candidate)
{
    if (plan.key.graph_shape_hash != candidate.graph_shape_hash)
        return ggml_hpx_plan_check_result::stale_graph_shape;
    if (plan.key.backend_assign_hash != candidate.backend_assign_hash)
        return ggml_hpx_plan_check_result::stale_backend_assign;
    if (plan.key.workspace_layout_sig != candidate.workspace_layout_sig)
        return ggml_hpx_plan_check_result::stale_workspace;
    if (plan.key.policy_version != candidate.policy_version)
        return ggml_hpx_plan_check_result::stale_policy;
    return ggml_hpx_plan_check_result::valid;
}
