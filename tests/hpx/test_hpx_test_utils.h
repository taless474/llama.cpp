#pragma once

#include "ggml-hpx-plan.h"
#include "ggml-hpx-region.h"
#include "ggml-hpx-topo.h"

#include <cstdint>

namespace hpx_test {

inline ggml_hpx_plan_key make_key(
    ggml_hpx_mode mode,
    uint64_t graph_shape_hash = 0x1111111111111111ULL,
    uint64_t backend_assign_hash = 0x2222222222222222ULL,
    uint64_t workspace_layout_sig = 0x3333333333333333ULL,
    uint32_t policy_version = 7)
{
    ggml_hpx_plan_key key{};
    key.mode = mode;
    key.graph_shape_hash = graph_shape_hash;
    key.backend_assign_hash = backend_assign_hash;
    key.workspace_layout_sig = workspace_layout_sig;
    key.policy_version = policy_version;
    return key;
}

inline ggml_hpx_region make_region(
    ggml_hpx_region_type type,
    uint32_t node_begin,
    uint32_t node_end,
    uint32_t prev_idx = UINT32_MAX)
{
    ggml_hpx_region r{};
    r.type = type;
    r.node_begin = node_begin;
    r.node_end = node_end;
    r.prev_idx = prev_idx;
    return r;
}

inline ggml_hpx_region_topology make_decode_topology()
{
    ggml_hpx_region_topology topo{};
    topo.n_nodes = 6;
    topo.regions = {
        make_region(ggml_hpx_region_type::cpu_contiguous, 0, 2),           // prev=none
        make_region(ggml_hpx_region_type::cpu_contiguous, 2, 5, 0u),       // prev=region 0
        make_region(ggml_hpx_region_type::blas_delegated, 5, 6, 1u),       // prev=region 1
    };
    return topo;
}

inline ggml_hpx_region_topology make_prefill_topology()
{
    ggml_hpx_region_topology topo{};
    topo.n_nodes = 10;
    topo.regions = {
        make_region(ggml_hpx_region_type::sched_split,    0,  3),          // prev=none
        make_region(ggml_hpx_region_type::cpu_contiguous, 3,  7, 0u),      // prev=region 0
        make_region(ggml_hpx_region_type::blas_delegated, 7, 10, 1u),      // prev=region 1
    };
    return topo;
}

inline ggml_hpx_region_topology make_bad_topology_out_of_bounds()
{
    ggml_hpx_region_topology topo{};
    topo.n_nodes = 4;
    topo.regions = {
        make_region(ggml_hpx_region_type::cpu_contiguous, 0, 2),
        // bad: end (5) > n_nodes (4)
        make_region(ggml_hpx_region_type::cpu_contiguous, 2, 5, 0u),
    };
    return topo;
}

inline ggml_hpx_region_topology make_bad_topology_sched_split()
{
    ggml_hpx_region_topology topo{};
    topo.n_nodes = 3;
    topo.regions = {
        make_region(ggml_hpx_region_type::sched_split, 0, 3),
    };
    return topo;
}

inline ggml_hpx_region_topology make_bad_topology_empty_span()
{
    ggml_hpx_region_topology topo{};
    topo.n_nodes = 4;
    topo.regions = {
        // bad: empty span (begin == end)
        make_region(ggml_hpx_region_type::cpu_contiguous, 0, 0),
    };
    return topo;
}

inline ggml_hpx_plan_key with_graph_hash(
    ggml_hpx_plan_key const& base, uint64_t v)
{
    auto k = base;
    k.graph_shape_hash = v;
    return k;
}

inline ggml_hpx_plan_key with_backend_hash(
    ggml_hpx_plan_key const& base, uint64_t v)
{
    auto k = base;
    k.backend_assign_hash = v;
    return k;
}

inline ggml_hpx_plan_key with_workspace_sig(
    ggml_hpx_plan_key const& base, uint64_t v)
{
    auto k = base;
    k.workspace_layout_sig = v;
    return k;
}

inline ggml_hpx_plan_key with_policy_version(
    ggml_hpx_plan_key const& base, uint32_t v)
{
    auto k = base;
    k.policy_version = v;
    return k;
}

}    // namespace hpx_test
