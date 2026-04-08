// ggml-hpx-adapter.cpp
//
// Adapter: external graph/scheduler state → HPX-owned planning inputs.
//
// This is the only translation unit in ggml-hpx that includes ggml-backend.h.
// All other HPX translation units use forward declarations from ggml-hpx-fwd.h.
//
// Two entry points, one per execution mode:
//
//   ggml_hpx_adapt_decode
//     Independent graph walk. Classifies each node by its buffer type.
//     Never reads scheduler state. Regions are delimited by changes in
//     buffer-type identity, so two GPU nodes on different backends produce
//     separate blas_delegated regions.
//
//   ggml_hpx_adapt_prefill
//     Scheduler-driven split translation. Calls ggml_backend_sched_split_graph
//     to materialise the split state, then maps each scheduler split to exactly
//     one ggml_hpx_region. Regions are delimited by changes in backend
//     identity, so two adjacent CPU splits from different ggml_backend_t
//     objects remain distinct. The region count is verified against
//     ggml_backend_sched_get_n_splits so the topology is authorised by the
//     scheduler, not re-derived independently.
//
// Both paths produce a fully populated ggml_hpx_adapter_result containing no
// external pointers — all data is HPX-owned.
//
// Pointer-based backend hashes
// ----------------------------
// backend_assign_hash uses backend/buft pointers as unique identifiers.
// These pointers are stable within one process lifetime (backends and buffer
// types are singletons). The hashes are NOT stable across process restarts
// and must not be persisted to disk or compared across sessions.

#include "ggml-hpx-adapter.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------

namespace {

// Map from tensor pointer to its index in the compute node array.
// Used to encode src connectivity as positional indices.
using NodeIndex = std::unordered_map<ggml_tensor*, uint32_t>;

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

static constexpr uint64_t k_fnv_offset = 0xcbf29ce484222325ULL;
static constexpr uint64_t k_fnv_prime  = 0x00000100000001B3ULL;

static uint64_t fnv1a_update(uint64_t h, uint64_t v) noexcept
{
    h ^= v;
    h *= k_fnv_prime;
    return h;
}

// ---------------------------------------------------------------------------
// Node index map
// ---------------------------------------------------------------------------

static NodeIndex build_node_index(ggml_tensor** nodes, uint32_t n)
{
    NodeIndex idx;
    idx.reserve(static_cast<size_t>(n));
    for (uint32_t i = 0; i < n; ++i)
        idx[nodes[i]] = i;
    return idx;
}

// ---------------------------------------------------------------------------
// Hash computation
// ---------------------------------------------------------------------------

// graph_shape_hash: op type sequence, node count, and src-index connectivity.
//
// Each src slot is encoded as:
//   0           -- null (unused slot)
//   1           -- external leaf (tensor not in the compute node array)
//   2 + index   -- compute node at the given positional index
static uint64_t compute_graph_shape_hash(
    ggml_tensor** nodes,
    uint32_t n,
    NodeIndex const& idx)
{
    uint64_t h = fnv1a_update(k_fnv_offset, static_cast<uint64_t>(n));

    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor* const node = nodes[i];
        h = fnv1a_update(h, static_cast<uint64_t>(node->op));

        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            ggml_tensor* const src = node->src[s];
            if (!src)
            {
                h = fnv1a_update(h, 0ULL);
                continue;
            }
            auto const it = idx.find(src);
            uint64_t const enc = (it != idx.end())
                ? (2ULL + static_cast<uint64_t>(it->second))
                : 1ULL;
            h = fnv1a_update(h, enc);
        }
    }

    return h;
}

// backend_assign_hash (decode path): hash of per-node buffer-type identity.
//
// ggml_backend_buffer_type_t is a singleton pointer per backend. Using it
// directly as a hash discriminant correctly distinguishes CPU from GPU0,
// GPU0 from GPU1, etc. Null (unallocated / no_alloc=true) hashes as 0,
// which is the correct identity for the host-default path.
static uint64_t compute_backend_assign_hash_decode(
    ggml_tensor** nodes, uint32_t n)
{
    uint64_t h = k_fnv_offset;
    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor* const node = nodes[i];
        ggml_backend_buffer_type_t const buft = node->buffer
            ? ggml_backend_buffer_get_type(node->buffer)
            : nullptr;
        uint64_t const id = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(buft));
        h = fnv1a_update(h, id);
    }
    return h;
}

// backend_assign_hash (prefill path): hash of per-node scheduler-assigned
// backend identity.
//
// ggml_backend_t is a singleton pointer per backend device. Using it directly
// as a hash discriminant gives full backend identity (null → 0 = host-default).
static uint64_t compute_backend_assign_hash_prefill(
    ggml_tensor** nodes, uint32_t n, ggml_backend_sched_t sched)
{
    uint64_t h = k_fnv_offset;
    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_backend_t const be =
            ggml_backend_sched_get_tensor_backend(sched, nodes[i]);
        uint64_t const id = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(be));
        h = fnv1a_update(h, id);
    }
    return h;
}

// workspace_layout_sig: per-node tensor type, shape (ne), strides (nb), and
// total byte count.
//
// Hashing ne + nb captures strided views, padding, and permuted layouts that
// ggml_nbytes alone cannot distinguish. Including type makes the hash
// independent of the shape/stride relationship assumption.
static uint64_t compute_workspace_sig(ggml_tensor** nodes, uint32_t n)
{
    uint64_t h = k_fnv_offset;
    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor const* const node = nodes[i];
        h = fnv1a_update(h, static_cast<uint64_t>(node->type));
        for (int d = 0; d < GGML_MAX_DIMS; ++d)
        {
            h = fnv1a_update(h, static_cast<uint64_t>(node->ne[d]));
            h = fnv1a_update(
                h, static_cast<uint64_t>(node->nb[d]));
        }
        h = fnv1a_update(h, static_cast<uint64_t>(ggml_nbytes(node)));
    }
    return h;
}

// ---------------------------------------------------------------------------
// Backend classification
// ---------------------------------------------------------------------------

// Decode path: CPU-bound if the buffer is absent or is a host buffer.
static bool decode_node_is_cpu(ggml_tensor const* node) noexcept
{
    if (!node->buffer)
        return true;
    return ggml_backend_buffer_is_host(node->buffer);
}

// Prefill path: CPU-bound if the scheduler assigned a host backend or none.
static bool prefill_backend_is_cpu(ggml_backend_t backend) noexcept
{
    if (!backend)
        return true;
    return ggml_backend_buft_is_host(
        ggml_backend_get_default_buffer_type(backend));
}

// ---------------------------------------------------------------------------
// Sequential dep_mask
// ---------------------------------------------------------------------------

// Returns the dep_mask for the next region to be appended: bit (size-1) set,
// expressing "depends only on its immediate predecessor."
//
// Precondition: regions.size() < GGML_HPX_MAX_REGIONS.
// The caller must assert this before calling. The static_assert in
// ggml-hpx-region.h guarantees GGML_HPX_MAX_REGIONS <= 64, so the shift
// is always within the uint64_t width.
static uint64_t sequential_dep(
    std::vector<ggml_hpx_region> const& regions) noexcept
{
    return regions.empty() ? 0ULL : (1ULL << (regions.size() - 1));
}

// ---------------------------------------------------------------------------
// Region topology builders
// ---------------------------------------------------------------------------

// Append one region to `regions`, checking the MAX_REGIONS bound first.
// This is the single point where sequential_dep is called; the guard here
// ensures the shift inside sequential_dep is always in-range.
static void push_region(
    std::vector<ggml_hpx_region>& regions,
    ggml_hpx_region_type type,
    uint32_t node_begin)
{
    GGML_ASSERT(regions.size() < GGML_HPX_MAX_REGIONS);
    ggml_hpx_region r{};
    r.type       = type;
    r.node_begin = node_begin;
    r.node_end   = node_begin + 1;
    r.dep_mask   = sequential_dep(regions);
    regions.push_back(r);
}

// Decode regions: cpu_contiguous or blas_delegated.
//
// Grouping key is the buffer-type pointer (buft identity), not the coarser
// CPU/non-CPU classification. This keeps two adjacent blas_delegated regions
// that run on different GPU backends as separate regions.
static std::vector<ggml_hpx_region> build_decode_regions(
    ggml_tensor** nodes, uint32_t n)
{
    std::vector<ggml_hpx_region> regions;
    // Sentinel: an address that can never be a valid buft pointer.
    // Using reinterpret_cast<ggml_backend_buffer_type_t>(1) would be
    // non-portable; instead we track a bool "have prev" alongside the key.
    bool have_prev          = false;
    ggml_backend_buffer_type_t prev_buft = nullptr;

    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor* const node = nodes[i];
        ggml_backend_buffer_type_t const buft = node->buffer
            ? ggml_backend_buffer_get_type(node->buffer)
            : nullptr;
        ggml_hpx_region_type const type =
            (!buft || ggml_backend_buft_is_host(buft))
            ? ggml_hpx_region_type::cpu_contiguous
            : ggml_hpx_region_type::blas_delegated;

        if (have_prev && buft == prev_buft)
        {
            regions.back().node_end = i + 1;
        }
        else
        {
            push_region(regions, type, i);
            prev_buft = buft;
            have_prev = true;
        }
    }

    return regions;
}

// Prefill regions: sched_split (CPU scheduler splits) or blas_delegated.
//
// Grouping key is the backend pointer (backend identity), not the coarser
// sched_split/blas_delegated classification. This keeps two adjacent CPU
// scheduler splits that run on different ggml_backend_t objects as separate
// regions, preserving the exact split structure produced by the scheduler.
//
// expected_splits is the authoritative split count from
// ggml_backend_sched_get_n_splits; the resulting region count is checked
// against it.
static std::vector<ggml_hpx_region> build_prefill_regions(
    ggml_tensor** nodes, uint32_t n,
    ggml_backend_sched_t sched, int expected_splits)
{
    std::vector<ggml_hpx_region> regions;
    bool have_prev       = false;
    ggml_backend_t prev_be = nullptr;

    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_backend_t const be =
            ggml_backend_sched_get_tensor_backend(sched, nodes[i]);
        ggml_hpx_region_type const type = prefill_backend_is_cpu(be)
            ? ggml_hpx_region_type::sched_split
            : ggml_hpx_region_type::blas_delegated;

        if (have_prev && be == prev_be)
        {
            regions.back().node_end = i + 1;
        }
        else
        {
            push_region(regions, type, i);
            prev_be   = be;
            have_prev = true;
        }
    }

    // Region count must match the scheduler's authoritative split count.
    // A mismatch means the adapter diverged from the split structure.
    GGML_ASSERT(n == 0 ||
        static_cast<int>(regions.size()) == expected_splits);

    return regions;
}

}    // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

ggml_hpx_adapter_result ggml_hpx_adapt_decode(
    ggml_cgraph const* graph, uint32_t policy_version)
{
    // The ggml graph API is not const-correct. All accesses here are
    // read-only: node count, node pointers, and tensor fields.
    ggml_cgraph* const mut = const_cast<ggml_cgraph*>(graph);

    int const n_int           = ggml_graph_n_nodes(mut);
    uint32_t const n          = (n_int > 0) ? static_cast<uint32_t>(n_int) : 0u;
    ggml_tensor** const nodes = ggml_graph_nodes(mut);

    NodeIndex const idx = build_node_index(nodes, n);

    ggml_hpx_adapter_result result{};
    result.topo.n_nodes = n;
    result.topo.regions = build_decode_regions(nodes, n);

    result.key.mode           = ggml_hpx_mode::decode;
    result.key.policy_version = policy_version;
    result.key.graph_shape_hash =
        compute_graph_shape_hash(nodes, n, idx);
    result.key.backend_assign_hash =
        compute_backend_assign_hash_decode(nodes, n);
    result.key.workspace_layout_sig =
        compute_workspace_sig(nodes, n);

    return result;
}

ggml_hpx_adapter_result ggml_hpx_adapt_prefill(
    ggml_cgraph const* graph, ggml_backend_sched_t sched,
    uint32_t policy_version)
{
    // The ggml graph API is not const-correct. See ggml_hpx_adapt_decode.
    ggml_cgraph* const mut = const_cast<ggml_cgraph*>(graph);

    // Materialise the scheduler's split state. This is the "split translation"
    // step: the scheduler assigns each node to a backend and records the
    // resulting split count. Subsequent get_tensor_backend queries reflect
    // this assignment. The call does not allocate buffers and does not require
    // a preceding ggml_backend_sched_reset.
    ggml_backend_sched_split_graph(sched, mut);
    int const n_splits = ggml_backend_sched_get_n_splits(sched);

    int const n_int           = ggml_graph_n_nodes(mut);
    uint32_t const n          = (n_int > 0) ? static_cast<uint32_t>(n_int) : 0u;
    ggml_tensor** const nodes = ggml_graph_nodes(mut);

    NodeIndex const idx = build_node_index(nodes, n);

    ggml_hpx_adapter_result result{};
    result.topo.n_nodes = n;
    result.topo.regions = build_prefill_regions(nodes, n, sched, n_splits);

    result.key.mode           = ggml_hpx_mode::prefill;
    result.key.policy_version = policy_version;
    result.key.graph_shape_hash =
        compute_graph_shape_hash(nodes, n, idx);
    result.key.backend_assign_hash =
        compute_backend_assign_hash_prefill(nodes, n, sched);
    result.key.workspace_layout_sig =
        compute_workspace_sig(nodes, n);

    return result;
}
