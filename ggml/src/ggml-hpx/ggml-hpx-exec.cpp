// ggml-hpx-exec.cpp
//
// Executor orchestration: adapt → cache → dispatch.
//
// Abort rule:
//   Entry:         observe-and-consume (check, reset, return aborted)
//   Chunk fn:      check only, return early — never resets the token
//   Post-dispatch: observe-and-consume (same as entry)
//
// Compute:
//   Each chunk fn uses ggml_graph_view (ggml-impl.h) to present a
//   region's node slice as a ggml_cgraph and then calls
//   ggml_backend_graph_compute.  Both graph parameters are ggml_cgraph*
//   (not const) because ggml_graph_view requires a non-const pointer.
//
// Prefill backend resolution:
//   run_prefill resolves one backend per region before dispatch using
//   ggml_backend_sched_get_tensor_backend on the region's first node.
//   Mixed-backend regions are unsupported (no cross-backend tensor copy
//   is implemented here).  A debug-build check asserts uniformity across
//   every node in each region; a mismatch means the adapter diverged
//   from the scheduler split structure.

#include "ggml-hpx-exec.h"
#include "ggml-hpx-abort.h"
#include "ggml-hpx-adapter.h"
#include "ggml-hpx-cache.h"
#include "ggml-hpx-plan.h"
#include "ggml-hpx-runtime.h"

#include "ggml-backend.h"    // ggml_backend_graph_compute,
                             //   ggml_backend_sched_get_tensor_backend
#include "ggml-cpu.h"        // ggml_backend_cpu_init/free/set_n_threads
#include "ggml-impl.h"       // ggml_graph_view
#include "ggml.h"            // GGML_ASSERT, ggml_status

#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/future.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Executor struct
// ---------------------------------------------------------------------------

struct ggml_hpx_exec
{
    ggml_hpx_runtime*    runtime        = nullptr;
    ggml_hpx_plan_cache  cache{};
    ggml_hpx_abort_token abort{};
    uint32_t             policy_version = 0;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_hpx_exec* ggml_hpx_exec_create(ggml_hpx_exec_params const& params)
{
    auto* exec           = new ggml_hpx_exec{};
    exec->runtime        = ggml_hpx_runtime_create(params.runtime);
    exec->policy_version = params.policy_version;
    return exec;
}

void ggml_hpx_exec_destroy(ggml_hpx_exec* exec)
{
    ggml_hpx_runtime_destroy(exec->runtime);
    delete exec;
}

// ---------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------

void ggml_hpx_exec_abort(ggml_hpx_exec* exec)
{
    exec->abort.request();
}

// ---------------------------------------------------------------------------
// Backend selection helper — decode
// ---------------------------------------------------------------------------

static ggml_backend_t backend_for_region(
    ggml_hpx_region const&            region,
    ggml_hpx_decode_backends const&   b) noexcept
{
    if (region.type == ggml_hpx_region_type::blas_delegated
        && b.blas != nullptr)
    {
        return b.blas;
    }
    return b.cpu;
}

// ---------------------------------------------------------------------------
// Chunk fn context types
// ---------------------------------------------------------------------------

struct decode_run_ctx
{
    ggml_hpx_decode_plan const*   plan     = nullptr;
    ggml_hpx_decode_backends      backends{};
    ggml_hpx_abort_token*         abort    = nullptr;
    ggml_cgraph*                  graph    = nullptr;
};

struct prefill_run_ctx
{
    ggml_hpx_prefill_plan const*  plan            = nullptr;
    ggml_hpx_abort_token*         abort           = nullptr;
    ggml_cgraph*                  graph           = nullptr;
    ggml_backend_t const*         region_backends = nullptr;
};

// ---------------------------------------------------------------------------
// Chunk fns — real ggml compute
// ---------------------------------------------------------------------------

static void decode_chunk_fn(uint32_t chunk_idx, void* user_data)
{
    auto* const ctx = static_cast<decode_run_ctx*>(user_data);

    if (ctx->abort->check())
    {
        return;
    }

    GGML_ASSERT(chunk_idx < ctx->plan->regions.size());

    ggml_hpx_region const& region = ctx->plan->regions[chunk_idx];
    GGML_ASSERT(region.node_begin <= region.node_end);

    ggml_backend_t const backend =
        backend_for_region(region, ctx->backends);

    ggml_cgraph view = ggml_graph_view(
        ctx->graph,
        static_cast<int>(region.node_begin),
        static_cast<int>(region.node_end));

    ggml_status const st = ggml_backend_graph_compute(backend, &view);
    if (st != GGML_STATUS_SUCCESS)
    {
        ctx->abort->request();
    }
}

static void prefill_chunk_fn(uint32_t chunk_idx, void* user_data)
{
    auto* const ctx = static_cast<prefill_run_ctx*>(user_data);

    if (ctx->abort->check())
    {
        return;
    }

    GGML_ASSERT(chunk_idx < ctx->plan->topo.regions.size());

    ggml_hpx_region const& region = ctx->plan->topo.regions[chunk_idx];
    GGML_ASSERT(region.node_begin <= region.node_end);
    GGML_ASSERT(region.node_end <= ctx->plan->topo.n_nodes);

    ggml_backend_t const backend = ctx->region_backends[chunk_idx];

    ggml_cgraph view = ggml_graph_view(
        ctx->graph,
        static_cast<int>(region.node_begin),
        static_cast<int>(region.node_end));

    ggml_status const st = ggml_backend_graph_compute(backend, &view);
    if (st != GGML_STATUS_SUCCESS)
    {
        ctx->abort->request();
    }
}

// ---------------------------------------------------------------------------
// Observe-and-consume helper
// ---------------------------------------------------------------------------

static bool consume_abort(ggml_hpx_abort_token& token) noexcept
{
    if (token.check())
    {
        token.reset();
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Run entry points
// ---------------------------------------------------------------------------

ggml_hpx_exec_status ggml_hpx_exec_run_decode(
    ggml_hpx_exec*                  exec,
    ggml_cgraph*                    graph,
    ggml_hpx_decode_backends const& backends)
{
    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    GGML_ASSERT(backends.cpu != nullptr);   // cpu is required; null makes the bundle meaningless

    ggml_hpx_adapter_result const ar =
        ggml_hpx_adapt_decode(graph, exec->policy_version);

    if (ar.topo.regions.empty())
    {
        return ggml_hpx_exec_status::ok;
    }

    ggml_hpx_decode_plan const* plan =
        ggml_hpx_cache_lookup_decode(exec->cache, ar.key);

    if (plan == nullptr)
    {
        if (!ggml_hpx_validate_decode_topology(ar.topo))
        {
            return ggml_hpx_exec_status::ok;
        }
        ggml_hpx_decode_plan built =
            ggml_hpx_build_decode_plan(ar.topo, ar.key);
        plan = ggml_hpx_cache_insert_decode(exec->cache, std::move(built));
    }

    auto const n_chunks = static_cast<uint32_t>(plan->regions.size());
    ggml_hpx_runtime_scratch_ensure(exec->runtime, plan->workspace_bytes);

    decode_run_ctx ctx{};
    ctx.plan     = plan;
    ctx.backends = backends;
    ctx.abort    = &exec->abort;
    ctx.graph    = graph;
    ggml_hpx_runtime_dispatch_decode(
        exec->runtime, n_chunks, decode_chunk_fn, &ctx);

    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    return ggml_hpx_exec_status::ok;
}

// ---------------------------------------------------------------------------
// Region-local dependency analyzer
// ---------------------------------------------------------------------------

// Prints topological levels for one CPU region of the prefill graph.
//
// Levels are computed with Kahn's BFS using only intra-region edges.
// Nodes at the same level are mutually independent within the region.
//
// Per level with width > 1, reports whether the ready nodes occupy a
// contiguous range in graph->nodes:
//   CONTIGUOUS  — a simple [begin, end) slice covers them; subregion
//                 dispatch is straightforward.
//   scattered   — nodes are not adjacent; an explicit index list would
//                 be required.
static void dump_region_local_deps(
    ggml_cgraph * graph,
    ggml_hpx_region const& region)
{
    uint32_t const begin = region.node_begin;
    uint32_t const end   = region.node_end;
    uint32_t const n     = end - begin;

    // Map tensor pointer → local index [0, n) for O(1) src lookup.
    std::unordered_map<ggml_tensor *, uint32_t> local_idx;
    local_idx.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        local_idx[graph->nodes[begin + i]] = i;
    }

    // Build forward adjacency list and in-degree array from intra-region
    // src[] edges only.
    std::vector<std::vector<uint32_t>> children(n);
    std::vector<uint32_t>              in_degree(n, 0u);

    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor * node = graph->nodes[begin + i];
        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            ggml_tensor * src = node->src[s];
            if (!src)
            {
                break;
            }
            auto const it = local_idx.find(src);
            if (it != local_idx.end())
            {
                uint32_t const parent = it->second;
                // Skip self-references: some ops (e.g. CPY) store their
                // own output tensor in src[1] as the destination buffer.
                if (parent != i)
                {
                    children[parent].push_back(i);
                    in_degree[i]++;
                }
            }
        }
    }

    // Kahn's BFS — compute level sets.
    std::vector<uint32_t> queue;
    queue.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        if (in_degree[i] == 0u)
        {
            queue.push_back(i);
        }
    }

    uint32_t level      = 0u;
    uint32_t max_width  = 0u;
    bool     any_wide   = false;

    std::fprintf(stderr,
        "[hpx-region] analyzing region nodes=[%u,%u)  n=%u"
        "  initial_roots=%zu\n",
        begin, end, n, queue.size());


    while (!queue.empty())
    {
        uint32_t const width = static_cast<uint32_t>(queue.size());
        max_width = std::max(max_width, width);
        if (width > 1u)
        {
            any_wide = true;
        }

        // Contiguity check: global indices must form a gapless run.
        std::vector<uint32_t> globals;
        globals.reserve(width);
        for (uint32_t li : queue)
        {
            globals.push_back(begin + li);
        }
        std::sort(globals.begin(), globals.end());
        bool const contiguous =
            (globals.back() - globals.front() + 1u == width);

        if (width > 1u)
        {
            std::fprintf(stderr,
                "[hpx-region]   level %u  width=%u  [%s]\n",
                level, width,
                contiguous ? "CONTIGUOUS" : "scattered");
        }
        else
        {
            std::fprintf(stderr,
                "[hpx-region]   level %u  width=%u\n",
                level, width);
        }

        for (uint32_t li : queue)
        {
            ggml_tensor * t = graph->nodes[begin + li];
            std::fprintf(stderr,
                "[hpx-region]     local[%2u] global=%3u"
                "  op=%-24s  name=%s\n",
                li, begin + li, ggml_op_name(t->op), t->name);
        }

        // Advance: release children whose in-degree reaches zero.
        std::vector<uint32_t> next;
        next.reserve(queue.size());
        for (uint32_t li : queue)
        {
            for (uint32_t child : children[li])
            {
                if (--in_degree[child] == 0u)
                {
                    next.push_back(child);
                }
            }
        }
        queue = std::move(next);
        ++level;
    }

    std::fprintf(stderr,
        "[hpx-region] result: max_width=%u  any_parallel_level=%s\n",
        max_width, any_wide ? "YES" : "NO");
}

// ---------------------------------------------------------------------------
// PROTOTYPE: parallel chain detection and dispatch
// ---------------------------------------------------------------------------
//
// Detects the first topological level in a CPU prefill region where ALL
// ready nodes are MUL_MAT and there are at least 2 of them
// (targets Q/K/V projection triplet or FFN gate+up pair).
//
// Labels every subsequent node with its source chain, then finds the first
// merge point where a node has ancestors from two or more distinct chains.
// The region is then decomposed into:
//   before  : [region.node_begin, first_parallel_global)   serial
//   chains  : N contiguous subgraphs, one per parallel root  parallel
//   after   : [merge_global, region.node_end)               serial
//
// Enable at runtime with LLAMA_HPX_PARALLEL_PROJ=1.
// Per-chain ggml thread count: LLAMA_HPX_PAR_CHAIN_THREADS (default 1).
//
// Each chain runs as an HPX async task with its own temporary CPU backend
// to avoid shared-state contention in ggml_backend_graph_compute.
// ---------------------------------------------------------------------------

struct hpx_parallel_chains
{
    uint32_t before_begin;
    uint32_t before_end;
    // chains[i] = {global_begin, global_end} for the i-th parallel chain.
    // Chains are contiguous node ranges derived from the sorted parallel
    // local indices: chain[c] = [p[c], p[c+1]) except the last, which
    // runs to the merge point.
    std::vector<std::pair<uint32_t, uint32_t>> chains;
    uint32_t after_begin;
    uint32_t after_end;
};

// Returns a populated hpx_parallel_chains when a valid parallel MUL_MAT
// level whose chains converge inside the region is found.
// Returns std::nullopt when:
//   - no all-MUL_MAT level with width >= 2 exists, or
//   - the chains do not converge within the region.
static std::optional<hpx_parallel_chains> detect_parallel_chains(
    ggml_cgraph *          graph,
    ggml_hpx_region const& region)
{
    uint32_t const begin = region.node_begin;
    uint32_t const end   = region.node_end;
    uint32_t const n     = end - begin;

    // Map tensor* → local index for O(1) parent lookup.
    std::unordered_map<ggml_tensor *, uint32_t> local_idx;
    local_idx.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        local_idx[graph->nodes[begin + i]] = i;
    }

    // Build intra-region adjacency list and in-degree array.
    std::vector<std::vector<uint32_t>> children(n);
    std::vector<uint32_t>              in_degree(n, 0u);

    for (uint32_t i = 0; i < n; ++i)
    {
        ggml_tensor * node = graph->nodes[begin + i];
        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            ggml_tensor * src = node->src[s];
            if (!src)
            {
                break;
            }
            auto const it = local_idx.find(src);
            if (it != local_idx.end())
            {
                uint32_t const parent = it->second;
                if (parent != i)    // skip CPY self-references
                {
                    children[parent].push_back(i);
                    in_degree[i]++;
                }
            }
        }
    }

    // Kahn's BFS: walk levels until we find one where every ready node is
    // MUL_MAT and there are at least 2 of them.
    std::vector<uint32_t> queue;
    queue.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        if (in_degree[i] == 0u)
        {
            queue.push_back(i);
        }
    }

    std::vector<uint32_t> parallel_locals;

    while (!queue.empty() && parallel_locals.empty())
    {
        bool const wide = (queue.size() >= 2u);
        bool all_mm     = wide;
        for (uint32_t li : queue)
        {
            if (graph->nodes[begin + li]->op != GGML_OP_MUL_MAT)
            {
                all_mm = false;
                break;
            }
        }
        if (all_mm)
        {
            parallel_locals = queue;
            std::sort(parallel_locals.begin(), parallel_locals.end());
        }

        std::vector<uint32_t> next;
        next.reserve(queue.size());
        for (uint32_t li : queue)
        {
            for (uint32_t child : children[li])
            {
                if (--in_degree[child] == 0u)
                {
                    next.push_back(child);
                }
            }
        }
        queue = std::move(next);
    }

    if (parallel_locals.empty())
    {
        return std::nullopt;
    }

    uint32_t const n_chains =
        static_cast<uint32_t>(parallel_locals.size());

    // Assign chain IDs (1-indexed).  0 = before-parallel; 255 = merged.
    std::vector<uint8_t> chain_id(n, 0u);
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        chain_id[parallel_locals[c]] = static_cast<uint8_t>(c + 1u);
    }

    // O(1) test for parallel-root positions (skip in propagation loop).
    std::unordered_set<uint32_t> is_parallel(
        parallel_locals.begin(), parallel_locals.end());

    // Propagate chain IDs forward through the topological order.
    // The node array is already topologically sorted, so all parents of
    // node i have local index < i (excluding self-references).
    for (uint32_t i = 0; i < n; ++i)
    {
        if (is_parallel.count(i))
        {
            continue;    // root of a chain: already assigned
        }

        ggml_tensor * node = graph->nodes[begin + i];
        uint8_t inherited  = 0u;
        bool    mixed      = false;

        for (int s = 0; s < GGML_MAX_SRC; ++s)
        {
            ggml_tensor * src = node->src[s];
            if (!src)
            {
                break;
            }
            auto const it = local_idx.find(src);
            if (it == local_idx.end())
            {
                continue;    // external (cross-region) tensor
            }
            uint32_t const parent = it->second;
            if (parent == i)
            {
                continue;    // self-reference (CPY pattern)
            }

            uint8_t const pc = chain_id[parent];
            if (pc == 255u)
            {
                mixed = true;    // already-merged parent → inherit merged
                break;
            }
            if (pc == 0u)
            {
                continue;    // before-parallel, not a chain source
            }
            if (inherited == 0u)
            {
                inherited = pc;
            }
            else if (inherited != pc)
            {
                mixed = true;
                break;
            }
        }

        chain_id[i] = mixed ? 255u : inherited;
    }

    // Merge point: first local index after the last parallel root where
    // chain_id == 255 (first node that depends on >= 2 chains).
    uint32_t merge_local = n;
    for (uint32_t i = parallel_locals.back() + 1u; i < n; ++i)
    {
        if (chain_id[i] == 255u)
        {
            merge_local = i;
            break;
        }
    }

    if (merge_local == n)
    {
        // Chains never converge within this region; skip parallel dispatch.
        return std::nullopt;
    }

    // Build per-chain global ranges.
    // Chain c covers [parallel_locals[c], parallel_locals[c+1]).
    // The last chain covers [parallel_locals.back(), merge_local).
    hpx_parallel_chains pc{};
    pc.before_begin = begin;
    pc.before_end   = begin + parallel_locals[0];
    pc.chains.reserve(n_chains);
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        uint32_t const c_begin = begin + parallel_locals[c];
        uint32_t const c_end   = (c + 1u < n_chains)
            ? begin + parallel_locals[c + 1u]
            : begin + merge_local;
        pc.chains.emplace_back(c_begin, c_end);
    }
    pc.after_begin = begin + merge_local;
    pc.after_end   = end;
    return pc;
}

// Execute a region with its parallel chains dispatched as concurrent HPX
// tasks, each on its own fresh CPU backend with n_threads internal ggml
// threads.  Before and after segments run serially on the supplied backend.
static ggml_status run_parallel_chains(
    ggml_cgraph *              graph,
    hpx_parallel_chains const& pc,
    ggml_backend_t             backend,
    int                        n_threads_per_chain)
{
    // --- before: serial ---
    if (pc.before_end > pc.before_begin)
    {
        ggml_cgraph before_view = ggml_graph_view(
            graph,
            static_cast<int>(pc.before_begin),
            static_cast<int>(pc.before_end));
        ggml_status const st =
            ggml_backend_graph_compute(backend, &before_view);
        if (st != GGML_STATUS_SUCCESS)
        {
            return st;
        }
    }

    // --- parallel chains ---
    uint32_t const n_chains =
        static_cast<uint32_t>(pc.chains.size());

    // Allocate one backend per chain.  These must outlive the futures.
    std::vector<ggml_backend_t> par_backends(n_chains, nullptr);
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        par_backends[c] = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(par_backends[c], n_threads_per_chain);
    }

    // Build graph views (small structs; nodes* stays valid because graph
    // itself outlives the futures).
    std::vector<ggml_cgraph> views;
    views.reserve(n_chains);
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        views.push_back(ggml_graph_view(
            graph,
            static_cast<int>(pc.chains[c].first),
            static_cast<int>(pc.chains[c].second)));
    }

    // Fire one HPX task per chain.
    std::vector<hpx::future<ggml_status>> futures;
    futures.reserve(n_chains);
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        ggml_backend_t be = par_backends[c];
        ggml_cgraph *  v  = &views[c];
        futures.push_back(
            hpx::async([be, v]() -> ggml_status
            {
                return ggml_backend_graph_compute(be, v);
            }));
    }

    hpx::wait_all(futures.begin(), futures.end());

    // Collect status and free temporary backends.
    ggml_status chain_st = GGML_STATUS_SUCCESS;
    for (uint32_t c = 0; c < n_chains; ++c)
    {
        if (futures[c].get() != GGML_STATUS_SUCCESS)
        {
            chain_st = GGML_STATUS_FAILED;
        }
        ggml_backend_free(par_backends[c]);
    }

    if (chain_st != GGML_STATUS_SUCCESS)
    {
        return chain_st;
    }

    // --- after: serial ---
    if (pc.after_end > pc.after_begin)
    {
        ggml_cgraph after_view = ggml_graph_view(
            graph,
            static_cast<int>(pc.after_begin),
            static_cast<int>(pc.after_end));
        ggml_status const st =
            ggml_backend_graph_compute(backend, &after_view);
        if (st != GGML_STATUS_SUCCESS)
        {
            return st;
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_hpx_exec_status ggml_hpx_exec_run_prefill(
    ggml_hpx_exec* exec, ggml_cgraph* graph, ggml_backend_sched_t sched)
{
    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    if (sched == nullptr)
    {
        return ggml_hpx_exec_status::ok;
    }

    ggml_hpx_adapter_result const ar =
        ggml_hpx_adapt_prefill(graph, sched, exec->policy_version);

    if (ar.topo.regions.empty())
    {
        return ggml_hpx_exec_status::ok;
    }

    ggml_hpx_prefill_plan const* plan =
        ggml_hpx_cache_lookup_prefill(exec->cache, ar.key);

    if (plan == nullptr)
    {
        if (!ggml_hpx_validate_prefill_topology(ar.topo))
        {
            return ggml_hpx_exec_status::ok;
        }
        ggml_hpx_prefill_plan built =
            ggml_hpx_build_prefill_plan(ar.topo, ar.key);
        plan = ggml_hpx_cache_insert_prefill(exec->cache, std::move(built));

        // Topology dump — fires once per unique plan (first build only).
        // Enable with LLAMA_HPX_DUMP_TOPO=1.
        {
            const char* env = std::getenv("LLAMA_HPX_DUMP_TOPO");
            if (env && env[0] == '1')
            {
                auto const& regions = plan->topo.regions;
                int const n_sched_splits =
                    ggml_backend_sched_get_n_splits(sched);

                std::fprintf(stderr,
                    "[hpx-topo] prefill plan:"
                    "  n_nodes=%u  n_regions=%zu  sched_splits=%d\n",
                    plan->topo.n_nodes,
                    regions.size(),
                    n_sched_splits);

                static const char* const k_type_name[] = {
                    "cpu_contiguous",
                    "blas_delegated ",    // padded to match longest name
                    "sched_split   ",
                };

                for (uint32_t ri = 0; ri < regions.size(); ++ri)
                {
                    ggml_hpx_region const& r = regions[ri];
                    uint32_t const n_nodes   = r.node_end - r.node_begin;
                    const char* const tname  =
                        (static_cast<int>(r.type) < 3)
                        ? k_type_name[static_cast<int>(r.type)]
                        : "unknown        ";

                    // Derive backend for this region from its first node.
                    ggml_backend_t const be = (r.node_begin < r.node_end)
                        ? ggml_backend_sched_get_tensor_backend(
                            sched, graph->nodes[r.node_begin])
                        : nullptr;
                    const char* const be_name =
                        be ? ggml_backend_name(be) : "none";

                    const char* const prev_str =
                        (r.prev_idx == UINT32_MAX) ? "none" : nullptr;
                    if (prev_str)
                    {
                        std::fprintf(stderr,
                            "[hpx-topo]   region[%2u]  type=%s"
                            "  nodes=[%u,%u)  n=%u"
                            "  prev=%-4s  backend=%s\n",
                            ri, tname,
                            r.node_begin, r.node_end, n_nodes,
                            prev_str, be_name);
                    }
                    else
                    {
                        std::fprintf(stderr,
                            "[hpx-topo]   region[%2u]  type=%s"
                            "  nodes=[%u,%u)  n=%u"
                            "  prev=%-4u  backend=%s\n",
                            ri, tname,
                            r.node_begin, r.node_end, n_nodes,
                            r.prev_idx, be_name);
                    }
                }

                // Summarise the dependency structure.
                bool all_linear = true;
                for (uint32_t ri = 1; ri < regions.size(); ++ri)
                {
                    if (regions[ri].prev_idx != ri - 1)
                    {
                        all_linear = false;
                        break;
                    }
                }
                const char* chain_desc;
                if (regions.size() <= 1)
                {
                    chain_desc = "trivial (single region)";
                }
                else if (all_linear)
                {
                    chain_desc = "fully linear — no independent regions";
                }
                else
                {
                    chain_desc = "non-linear — independent regions exist";
                }
                std::fprintf(stderr,
                    "[hpx-topo] dependency chain: %s\n", chain_desc);
            }
        }

        // Region-local dependency analysis.
        // Enable with LLAMA_HPX_DUMP_REGION=1.
        // Picks the first sched_split CPU region with >= 20 nodes —
        // a representative layer-sized CPU region.
        {
            const char* env = std::getenv("LLAMA_HPX_DUMP_REGION");
            if (env && env[0] == '1')
            {
                ggml_hpx_region const* target = nullptr;
                for (ggml_hpx_region const& r : plan->topo.regions)
                {
                    bool const is_cpu_split =
                        (r.type == ggml_hpx_region_type::sched_split);
                    bool const big_enough =
                        (r.node_end - r.node_begin >= 20u);
                    if (is_cpu_split && big_enough)
                    {
                        target = &r;
                        break;
                    }
                }
                if (target)
                {
                    dump_region_local_deps(graph, *target);
                }
                else
                {
                    std::fprintf(stderr,
                        "[hpx-region] no CPU sched_split region"
                        " with >= 20 nodes found\n");
                }
            }
        }
    }

    auto const n_chunks = static_cast<uint32_t>(plan->topo.regions.size());

    // Resolve one backend per region using the scheduler.
    //
    // Each region was built with backend identity as its grouping key, so
    // all nodes within a region are assigned to the same backend.  We use
    // node_begin to derive the representative backend for each region.
    //
    // Mixed-backend regions are not supported: no cross-backend tensor copy
    // is implemented here.  The #ifndef NDEBUG block below verifies the
    // uniformity assumption and fires a hard assertion if the adapter ever
    // produces a mixed-backend region.
    std::vector<ggml_backend_t> region_backends;
    region_backends.reserve(n_chunks);

    for (ggml_hpx_region const& region : plan->topo.regions)
    {
        ggml_backend_t const be = (region.node_begin < region.node_end)
            ? ggml_backend_sched_get_tensor_backend(
                sched, graph->nodes[region.node_begin])
            : nullptr;
        region_backends.push_back(be);
    }

#ifndef NDEBUG
    for (uint32_t ri = 0; ri < n_chunks; ++ri)
    {
        ggml_hpx_region const& region   = plan->topo.regions[ri];
        ggml_backend_t const   expected = region_backends[ri];
        for (uint32_t ni = region.node_begin + 1; ni < region.node_end; ++ni)
        {
            ggml_backend_t const actual =
                ggml_backend_sched_get_tensor_backend(sched,
                    graph->nodes[ni]);
            GGML_ASSERT(actual == expected &&
                "prefill region has mixed-backend nodes: "
                "cross-backend tensor copy is not implemented");
        }
    }
#endif

    // -----------------------------------------------------------------------
    // PROTOTYPE: parallel chain dispatch (LLAMA_HPX_PARALLEL_PROJ=1)
    //
    // For each sched_split CPU region with >= 20 nodes, attempt to detect
    // a Q/K/V-style parallel MUL_MAT level and run its chains concurrently.
    // Regions that don't match fall back to normal serial dispatch.
    // All other region types always run serially.
    //
    // Per-chain ggml thread count: LLAMA_HPX_PAR_CHAIN_THREADS (default 1).
    // Set higher to give each chain more ggml-internal parallelism, at the
    // cost of total-thread saturation across the three HPX tasks.
    // -----------------------------------------------------------------------
    {
        const char* par_env = std::getenv("LLAMA_HPX_PARALLEL_PROJ");
        if (par_env && par_env[0] == '1')
        {
            int n_threads_per_chain = 1;
            const char* nt_env = std::getenv("LLAMA_HPX_PAR_CHAIN_THREADS");
            if (nt_env)
            {
                n_threads_per_chain = std::max(1, std::atoi(nt_env));
            }

            for (uint32_t ri = 0; ri < n_chunks; ++ri)
            {
                if (exec->abort.check())
                {
                    exec->abort.request();
                    break;
                }

                ggml_hpx_region const& region = plan->topo.regions[ri];
                ggml_backend_t const   be     = region_backends[ri];

                bool const candidate =
                    (region.type == ggml_hpx_region_type::sched_split) &&
                    (region.node_end - region.node_begin >= 20u);

                if (candidate)
                {
                    auto pc = detect_parallel_chains(graph, region);
                    if (pc.has_value())
                    {
                        ggml_status const st =
                            run_parallel_chains(graph, *pc, be,
                                n_threads_per_chain);
                        if (st != GGML_STATUS_SUCCESS)
                        {
                            exec->abort.request();
                        }
                        continue;
                    }
                }

                // Normal serial dispatch for this region.
                if (region.node_begin < region.node_end)
                {
                    ggml_cgraph view = ggml_graph_view(
                        graph,
                        static_cast<int>(region.node_begin),
                        static_cast<int>(region.node_end));
                    ggml_status const st =
                        ggml_backend_graph_compute(be, &view);
                    if (st != GGML_STATUS_SUCCESS)
                    {
                        exec->abort.request();
                        break;
                    }
                }
            }

            if (consume_abort(exec->abort))
            {
                return ggml_hpx_exec_status::aborted;
            }
            return ggml_hpx_exec_status::ok;
        }
    }

    // Normal serial dispatch (LLAMA_HPX_PARALLEL_PROJ not set).
    ggml_hpx_runtime_scratch_ensure(exec->runtime, plan->workspace_bytes);

    prefill_run_ctx ctx{};
    ctx.plan            = plan;
    ctx.abort           = &exec->abort;
    ctx.graph           = graph;
    ctx.region_backends = region_backends.data();
    ggml_hpx_runtime_dispatch_prefill(
        exec->runtime, n_chunks, prefill_chunk_fn, &ctx);

    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    return ggml_hpx_exec_status::ok;
}
