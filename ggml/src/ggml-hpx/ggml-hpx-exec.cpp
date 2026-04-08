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
#include "ggml-impl.h"       // ggml_graph_view
#include "ggml.h"            // GGML_ASSERT, ggml_status

#include <cstdint>
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
