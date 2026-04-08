#pragma once

// ggml-hpx-plan.h
//
// Plan types and builder/validation declarations for the HPX executor.
//
// This header must not include ggml-backend.h. The plan builder (plan.cpp)
// consumes only ggml_hpx_region_topology from the adapter; it never reads
// scheduler state directly.
//
// Two plan types:
//   ggml_hpx_decode_plan  - low-latency single-token path
//                           topology from an independent graph walk
//                           optimised for minimal dispatch overhead
//
//   ggml_hpx_prefill_plan - throughput-oriented prompt/batch path
//                           topology from a scheduler-driven split
//                           translation done by the adapter
//                           supports optional region overlap
//
// Validation contract:
//   Call ggml_hpx_validate_*_topology() before building a plan. On success,
//   ggml_hpx_build_*_plan() may use GGML_ASSERT for programmer-error
//   invariants that should never fire once the adapter is correct.

#include "ggml-hpx-region.h"
#include "ggml-hpx-topo.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

enum class ggml_hpx_mode : uint8_t
{
    decode = 0,
    prefill = 1,
};

// ---------------------------------------------------------------------------
// Plan key
// ---------------------------------------------------------------------------

// A plan is reusable across runs only while its key matches the key the
// executor would compute for the current run. Keys must not contain
// ephemeral addresses.
//
// graph_shape_hash     - hash of op type sequence, node count, and src-index
//                        topology; computed at plan-build time
// backend_assign_hash  - hash of per-node backend assignment
// workspace_layout_sig - hash of workspace size requirements; a full layout
//                        descriptor is not yet stored in the plan, so this
//                        currently captures size only
// policy_version       - bumped when planner logic changes in a way that
//                        requires existing plans to be rebuilt
struct ggml_hpx_plan_key
{
    ggml_hpx_mode mode;
    uint64_t graph_shape_hash;
    uint64_t backend_assign_hash;
    uint64_t workspace_layout_sig;
    uint32_t policy_version;
};

inline bool operator==(
    ggml_hpx_plan_key const& a, ggml_hpx_plan_key const& b) noexcept
{
    return a.mode == b.mode && a.graph_shape_hash == b.graph_shape_hash &&
        a.backend_assign_hash == b.backend_assign_hash &&
        a.workspace_layout_sig == b.workspace_layout_sig &&
        a.policy_version == b.policy_version;
}

inline bool operator!=(
    ggml_hpx_plan_key const& a, ggml_hpx_plan_key const& b) noexcept
{
    return !(a == b);
}

// ---------------------------------------------------------------------------
// Plan check result
// ---------------------------------------------------------------------------

// Returned by the check functions below. Reports which key component caused
// a mismatch. This is a key mismatch reason, not a general plan health
// status: all values are key-based. The plan structure itself is assumed
// correct when the key is valid.
enum class ggml_hpx_plan_check_result : uint8_t
{
    valid,
    stale_graph_shape,
    stale_backend_assign,
    stale_workspace,
    stale_policy,
};

// ---------------------------------------------------------------------------
// Chunking policy
// ---------------------------------------------------------------------------

enum class ggml_hpx_chunk_policy : uint8_t
{
    auto_size,      // HPX decides based on available workers (preferred default)
    fixed_count,    // split into exactly param chunks
    fixed_nodes,    // each chunk covers at most param nodes
};

struct ggml_hpx_chunk_spec
{
    ggml_hpx_chunk_policy policy = ggml_hpx_chunk_policy::auto_size;
    uint32_t param = 0;    // chunk count or node limit; 0 = let executor decide
};

// ---------------------------------------------------------------------------
// Overlap policy (prefill only)
// ---------------------------------------------------------------------------

enum class ggml_hpx_overlap_policy : uint8_t
{
    none,           // strictly follow dependency order; no speculative overlap
    opportunistic,  // independent regions may overlap when HPX schedule permits
};

// ---------------------------------------------------------------------------
// Per-region execution policy (prefill only)
// ---------------------------------------------------------------------------

struct ggml_hpx_region_policy
{
    ggml_hpx_chunk_spec chunk;
};

// ---------------------------------------------------------------------------
// Decode plan
// ---------------------------------------------------------------------------

// A cached execution template for the low-latency single-token path.
//
// regions and chunk_specs are kept as parallel vectors for now.
// chunk_specs[i] applies to regions[i]. A combined struct
//   { ggml_hpx_region region; ggml_hpx_chunk_spec chunk; }
// would remove the parallel-vector invariant and is the preferred long-term
// form, but is deferred until the API stabilises.
//
// sched_split regions must not appear in a decode plan.
//
// workspace_bytes is the maximum scratch space required. The executor
// pre-allocates once and reuses across runs while the plan is valid.
struct ggml_hpx_decode_plan
{
    ggml_hpx_plan_key key;
    std::vector<ggml_hpx_region> regions;
    std::vector<ggml_hpx_chunk_spec> chunk_specs;    // parallel to regions
    size_t workspace_bytes = 0;
};

// Validate a topology snapshot before building a decode plan. Returns false
// if the topology is structurally incompatible with the decode path, for
// example: contains sched_split regions, has empty region list, or has node
// ranges that exceed n_nodes. Must be called before
// ggml_hpx_build_decode_plan.
//
// Defined in ggml-hpx-plan.cpp.
bool ggml_hpx_validate_decode_topology(ggml_hpx_region_topology const& topo);

// Build a decode plan from a validated topology snapshot and a pre-computed
// key. The topology must have been produced by an independent graph walk in
// the adapter. Behaviour is undefined if the topology has not passed
// ggml_hpx_validate_decode_topology; plan.cpp may assert on invariants that
// validation is expected to have caught.
//
// Defined in ggml-hpx-plan.cpp.
ggml_hpx_decode_plan ggml_hpx_build_decode_plan(
    ggml_hpx_region_topology const& topo, ggml_hpx_plan_key const& key);

// Check whether a cached decode plan is still compatible with a candidate
// key. Returns valid if all components match; returns the first stale
// component otherwise. The plan structure is not inspected; only the key
// is compared.
//
// Defined in ggml-hpx-plan.cpp.
ggml_hpx_plan_check_result ggml_hpx_decode_plan_check(
    ggml_hpx_decode_plan const& plan, ggml_hpx_plan_key const& candidate);

// ---------------------------------------------------------------------------
// Prefill plan
// ---------------------------------------------------------------------------

// A cached execution template for the throughput-oriented prompt/batch path.
//
// topo holds the scheduler-derived region topology as an immutable HPX-owned
// snapshot. region_policies is parallel to topo.regions; see the decode plan
// note on parallel vectors.
//
// A prefill plan may be reused only while backend placement, split topology,
// workspace size, and policy version all remain compatible with the key.
struct ggml_hpx_prefill_plan
{
    ggml_hpx_plan_key key;
    ggml_hpx_region_topology topo;
    std::vector<ggml_hpx_region_policy> region_policies;    // parallel to topo.regions
    ggml_hpx_overlap_policy overlap = ggml_hpx_overlap_policy::none;
    size_t workspace_bytes = 0;
};

// Validate a topology snapshot before building a prefill plan. Returns false
// if the topology is structurally invalid, for example: empty region list,
// node range out of bounds, or a dep_mask bit references a non-existent
// region index. Must be called before ggml_hpx_build_prefill_plan.
//
// Defined in ggml-hpx-plan.cpp.
bool ggml_hpx_validate_prefill_topology(ggml_hpx_region_topology const& topo);

// Build a prefill plan from a validated topology snapshot and a pre-computed
// key. The topology must have been produced by the scheduler-driven adapter
// path. Behaviour is undefined if the topology has not passed
// ggml_hpx_validate_prefill_topology; plan.cpp may assert on invariants that
// validation is expected to have caught.
//
// Defined in ggml-hpx-plan.cpp.
ggml_hpx_prefill_plan ggml_hpx_build_prefill_plan(
    ggml_hpx_region_topology const& topo, ggml_hpx_plan_key const& key);

// Check whether a cached prefill plan is still compatible with a candidate
// key. Returns valid if all components match; returns the first stale
// component otherwise. The topo snapshot inside the plan is trusted on a key
// hit and is not re-inspected.
//
// Defined in ggml-hpx-plan.cpp.
ggml_hpx_plan_check_result ggml_hpx_prefill_plan_check(
    ggml_hpx_prefill_plan const& plan, ggml_hpx_plan_key const& candidate);
