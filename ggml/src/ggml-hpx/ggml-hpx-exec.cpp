// ggml-hpx-exec.cpp
//
// Executor orchestration: adapt → cache → dispatch.
//
// Abort rule:
//   Entry:         observe-and-consume (check, reset, return aborted)
//   Chunk fn:      check only, return early — never resets the token
//   Post-dispatch: observe-and-consume (same as entry)
//
// The chunk fns are stubs: they validate bounds, touch one byte of scratch
// per chunk to prove execution was reached, and return without running ggml
// ops. The scratch write is not thread-safe across chunks, but each chunk
// writes only to its own offset (chunk_idx), so no two chunks alias.

#include "ggml-hpx-exec.h"
#include "ggml-hpx-abort.h"
#include "ggml-hpx-adapter.h"
#include "ggml-hpx-cache.h"
#include "ggml-hpx-plan.h"
#include "ggml-hpx-runtime.h"

#include "ggml.h"    // GGML_ASSERT

#include <cstdint>

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
    void*                         scratch  = nullptr;
};

struct prefill_run_ctx
{
    ggml_hpx_prefill_plan const* plan    = nullptr;
    ggml_hpx_abort_token*        abort   = nullptr;
    void*                        scratch = nullptr;
};

// ---------------------------------------------------------------------------
// Stub chunk fns
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

    // Resolve which backend owns this region. API plumbing only —
    // no subgraph construction or ggml_backend_graph_compute yet.
    ggml_backend_t const backend =
        backend_for_region(region, ctx->backends);
    (void)backend;

    // Touch scratch at this chunk's offset to prove the chunk was reached.
    if (ctx->scratch != nullptr)
    {
        static_cast<uint8_t*>(ctx->scratch)[chunk_idx] = 1;
    }

    // No real ggml ops yet.
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

    // Touch scratch at this chunk's offset to prove the chunk was reached.
    if (ctx->scratch != nullptr)
    {
        static_cast<uint8_t*>(ctx->scratch)[chunk_idx] = 1;
    }

    // No real ggml ops yet.
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
    ggml_cgraph const*              graph,
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
    ggml_hpx_runtime_scratch_ensure(exec->runtime, n_chunks);

    decode_run_ctx ctx{};
    ctx.plan     = plan;
    ctx.backends = backends;
    ctx.abort    = &exec->abort;
    ctx.scratch  = ggml_hpx_runtime_scratch_ptr(exec->runtime);
    ggml_hpx_runtime_dispatch_decode(
        exec->runtime, n_chunks, decode_chunk_fn, &ctx);

    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    return ggml_hpx_exec_status::ok;
}

ggml_hpx_exec_status ggml_hpx_exec_run_prefill(
    ggml_hpx_exec* exec, ggml_cgraph const* graph,
    ggml_backend_sched_t sched)
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
    ggml_hpx_runtime_scratch_ensure(exec->runtime, n_chunks);

    prefill_run_ctx ctx{plan, &exec->abort,
        ggml_hpx_runtime_scratch_ptr(exec->runtime)};
    ggml_hpx_runtime_dispatch_prefill(
        exec->runtime, n_chunks, prefill_chunk_fn, &ctx);

    if (consume_abort(exec->abort))
    {
        return ggml_hpx_exec_status::aborted;
    }

    return ggml_hpx_exec_status::ok;
}
