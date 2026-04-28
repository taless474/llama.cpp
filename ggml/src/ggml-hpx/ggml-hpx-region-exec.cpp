// ggml-hpx-region-exec.cpp
//
// Validator, F32 mul_mat kernel, single-region executor, and group runner
// for the ggml_hpx_cpu_region contract.
//
// None of the functions here call:
//   ggml_graph_compute_thread_run
//   ggml_graph_compute
//   ggml_backend_graph_compute
//
// Execution model (Phase 2: per-region dataflow)
// ──────────────────────────────────────────────
// launch_region_async (internal) is the single HPX boundary: all knowledge
// of hpx::async, hpx::dataflow, hpx::future, and hpx::wait_all lives in
// this translation unit.  Intra-region fan-out inside launch_region_async
// is unchanged from Phase 1.
//
// ggml_hpx_run_single_region:
//   Wraps launch_region_async(...).get().
//
// ggml_hpx_run_region_group:
//   One shared_future per region.  For each region i in topological order:
//     - gather shared_futures of i's predecessors;
//     - if none, region_fut[i] = launch_region_async(i).share();
//     - else,   region_fut[i] = dataflow(preds...).share(), whose body
//       calls launch_region_async(i) when every predecessor has completed.
//   The group barrier is a single hpx::wait_all on region_fut.
//
//   There is no explicit level loop: the true invariant ("a region is
//   runnable when its predecessors are done") is expressed directly by
//   the dataflow edges.

#ifndef GGML_HPX_REGION_DAG
#  error "ggml-hpx-region-exec.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-region-exec.h"

// HPX
#include <hpx/algorithm.hpp>
#include <hpx/async_base/async.hpp>
#include <hpx/async_base/dataflow.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>

// std
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <queue>
#include <vector>

// ---------------------------------------------------------------------------
// Forward-declare the F32 dot-product kernel used by mul_mat.
// Implemented in ggml-cpu; not reached through any ggml graph path.
// ---------------------------------------------------------------------------

extern "C" void ggml_vec_dot_f32(
    int           n,
    float *       s,  size_t bs,
    const float * x,  size_t bx,
    const float * y,  size_t by,
    int           nrc);

// Signatures match ggml/src/ggml-cpu/quants.h exactly.
// No header is included because quants.h is an internal header not reachable
// from ggml/include/. Same pattern as ggml_vec_dot_f32 above.
extern "C"
{
    void quantize_row_q8_K(const float * x, void * y, int64_t k);
    void ggml_vec_dot_q4_K_q8_K(
        int           n,
        float *       s,   size_t bs,
        const void *  vx,  size_t bx,
        const void *  vy,  size_t by,
        int           nrc);
    // CPU_REPACK NEON kernel; declared in ggml/src/ggml-cpu/repack.h:154.
    // The 8x8 trait variant — `nc` columns of repacked weight × 1 row of
    // Q8_K activation, written contiguously to `s`.  `bs` is element
    // stride between consecutive output rows when nr > 1; with nr == 1 it
    // is read but unused, and we pass `out_cols` to match the convention
    // ggml's own forward_mul_mat_one_chunk uses (repack.cpp:4247-4249).
    void ggml_gemv_q4_K_8x8_q8_K(
        int           n,
        float *       s,   size_t bs,
        const void *  vx,
        const void *  vy,
        int           nr,
        int           nc);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace
{

// Maximum n_regions for the stack-allocated small path in
// ggml_hpx_run_region_group.  Groups larger than this fall through to the
// original heap-vector path unchanged.
constexpr int kSmallN = 8;

// The single HPX boundary for the fine-region DAG path.
//
// All knowledge of hpx::async, hpx::future, and hpx::wait_all lives in
// this function's body.  Phase 2 will wrap calls to this seam in
// hpx::dataflow over predecessor shared_futures; the signature does not
// need to change to support that.
//
// Intra-region fan-out is implemented with one hpx::async per lane and a
// hpx::wait_all inside the outer task.  Using plain asyncs (rather than a
// pinned fork_join_executor) lets the default HPX scheduler schedule every
// lane of every concurrent region across all available workers without PU
// contention between regions.  Executor pooling / reuse is explicitly
// Phase 3.
hpx::future<void> launch_region_async(
    ggml_hpx_cpu_region const & r,
    int                         n_lanes,
    ggml_hpx_region_resources * res)
{
    // Capture r by value: launch_region_async returns before the async body
    // executes, so `r` (a reference parameter on the caller's stack) would be
    // dangling by the time an HPX worker picks up the task.  Copying the
    // ggml_hpx_cpu_region struct (~40 bytes) is cheap; the ctx pointer inside
    // the copy remains valid because the ggml_hpx_lowering object that owns
    // ctx_buf lives until the outer .get() returns.
    //
    // Inner lane lambdas use [&r] referencing the outer lambda's local copy;
    // that copy lives until hpx::wait_all(lane_futs) completes, so the inner
    // references are valid.
    return hpx::async([r, n_lanes, res]()
    {
        if (r.kind == GGML_HPX_CPU_REGION_KIND_REDUCTION || n_lanes <= 1)
        {
            r.run_range(r.ctx, 0, 1, r.begin, r.end, res);
            return;
        }

        int64_t const span = r.end - r.begin;

        std::vector<hpx::future<void>> lane_futs;
        lane_futs.reserve(static_cast<std::size_t>(n_lanes));

        for (int ith = 0; ith < n_lanes; ++ith)
        {
            int64_t const b = (span *  ith)      / n_lanes;
            int64_t const e = (span * (ith + 1)) / n_lanes;
            if (b >= e) continue;

            lane_futs.push_back(hpx::async(
                [&r, n_lanes, res, ith, b, e]()
                {
                    r.run_range(r.ctx, ith, n_lanes,
                                r.begin + b, r.begin + e, res);
                }));
        }

        hpx::wait_all(lane_futs);
    });
}

}    // namespace

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

const char * ggml_hpx_validate_region_group(
    ggml_hpx_cpu_region_group const * group,
    bool                              check_acyclic)
{
    if (!group)               return "group is null";
    if (group->n_regions < 0) return "n_regions is negative";
    if (group->n_deps    < 0) return "n_deps is negative";

    if (group->n_regions > 0 && !group->regions)
        return "regions is null but n_regions > 0";
    if (group->n_deps > 0 && !group->deps)
        return "deps is null but n_deps > 0";

    for (int i = 0; i < group->n_regions; ++i)
    {
        ggml_hpx_cpu_region const & r = group->regions[i];
        if (!r.run_range)     return "region has null run_range";
        if (r.begin >= r.end) return "region has begin >= end";
    }

    for (int e = 0; e < group->n_deps; ++e)
    {
        ggml_hpx_dep_edge const & d = group->deps[e];
        if (d.src < 0 || d.src >= group->n_regions)
            return "dep edge src out of range";
        if (d.dst < 0 || d.dst >= group->n_regions)
            return "dep edge dst out of range";
        if (d.src == d.dst)
            return "dep edge is a self-loop";
    }

    if (check_acyclic)
    {
        // Kahn's algorithm: topological sort via in-degree countdown.
        // If we cannot drain all nodes the graph has at least one cycle.
        int const               n = group->n_regions;
        std::vector<int> in_deg(static_cast<std::size_t>(n), 0);

        for (int e = 0; e < group->n_deps; ++e)
            ++in_deg[static_cast<std::size_t>(group->deps[e].dst)];

        std::queue<int> ready;
        for (int i = 0; i < n; ++i)
            if (in_deg[static_cast<std::size_t>(i)] == 0)
                ready.push(i);

        int visited = 0;
        while (!ready.empty())
        {
            int const u = ready.front();
            ready.pop();
            ++visited;

            for (int e = 0; e < group->n_deps; ++e)
            {
                if (group->deps[e].src != u) continue;
                int const dst = group->deps[e].dst;
                if (--in_deg[static_cast<std::size_t>(dst)] == 0)
                    ready.push(dst);
            }
        }

        if (visited != n)
            return "dep graph contains a cycle";
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// F32 mul_mat kernel
// ---------------------------------------------------------------------------

void ggml_hpx_mul_mat_f32_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<ggml_hpx_mul_mat_f32_ctx *>(ctx_void);
    int const n = static_cast<int>(ctx->cols);

    for (int64_t row = 0; row < ctx->rows; ++row)
    {
        const float * x_row = ctx->x + row * ctx->cols;
        float *       y_row = ctx->y + row * ctx->out_cols;

        for (int64_t col = begin; col < end; ++col)
        {
            ggml_vec_dot_f32(n,
                y_row + col,               /* s */ 0,
                ctx->w + col * ctx->cols,  /* x */ 0,
                x_row,                     /* y */ 0,
                1);
        }
    }
}

// ---------------------------------------------------------------------------
// F32 RMS_NORM kernels
// ---------------------------------------------------------------------------

void ggml_hpx_rms_norm_partial_f32_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    (void)nth;
    auto * ctx = static_cast<ggml_hpx_rms_norm_partial_f32_ctx *>(ctx_void);

    assert(resources->lane_scratch != nullptr);
    assert(ith >= 0 && ith < resources->n_lanes);
    assert(end <= ctx->n);

    float sumsq = 0.0f;
    for (int64_t i = begin; i < end; ++i)
    {
        sumsq += ctx->x[i] * ctx->x[i];
    }

    *static_cast<float *>(resources->lane_scratch[ith]) = sumsq;
}

void ggml_hpx_rms_norm_finalize_f32_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    auto * ctx = static_cast<ggml_hpx_rms_norm_finalize_f32_ctx *>(ctx_void);

    assert(ith == 0 && nth == 1);
    assert(begin == 0 && end == ctx->n);
    (void)ith; (void)nth; (void)begin; (void)end;

    float sumsq = 0.0f;
    for (int i = 0; i < resources->n_lanes; ++i)
    {
        sumsq += *static_cast<float *>(resources->lane_scratch[i]);
    }

    auto * buf = static_cast<ggml_hpx_rms_norm_f32_reduce_buffer *>(
        resources->reduction_buffer);

    buf->sumsq = sumsq;
    buf->scale = 1.0f / std::sqrt(sumsq / static_cast<float>(ctx->n) + ctx->eps);
}

void ggml_hpx_rms_norm_apply_f32_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    (void)ith;
    (void)nth;
    auto * ctx = static_cast<ggml_hpx_rms_norm_apply_f32_ctx *>(ctx_void);

    auto const * buf = static_cast<ggml_hpx_rms_norm_f32_reduce_buffer const *>(
        resources->reduction_buffer);

    float const scale = buf->scale;
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->dst[i] = ctx->x[i] * scale;
    }
}

// ---------------------------------------------------------------------------
// F32 SiLU kernel
// ---------------------------------------------------------------------------

void ggml_hpx_silu_f32_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<ggml_hpx_silu_f32_ctx *>(ctx_void);
    assert(end <= ctx->n);
    for (int64_t i = begin; i < end; ++i)
    {
        const float v = ctx->x[i];
        ctx->dst[i] = v / (1.0f + std::exp(-v));
    }
}

// ---------------------------------------------------------------------------
// F32 elementwise MUL kernel
// ---------------------------------------------------------------------------

void ggml_hpx_mul_f32_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<ggml_hpx_mul_f32_ctx *>(ctx_void);
    assert(end <= ctx->n);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->dst[i] = ctx->a[i] * ctx->b[i];
    }
}

// ---------------------------------------------------------------------------
// F32 SwiGLU kernel (fused)
// ---------------------------------------------------------------------------

void ggml_hpx_swiglu_f32_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<ggml_hpx_swiglu_f32_ctx *>(ctx_void);
    assert(end <= ctx->n);
    for (int64_t i = begin; i < end; ++i)
    {
        const float g = ctx->gate[i];
        ctx->dst[i]   = (g / (1.0f + std::exp(-g))) * ctx->up[i];
    }
}

// ---------------------------------------------------------------------------
// Q4_K decode pipeline — region 0: serial F32 → Q8_K quantization
// ---------------------------------------------------------------------------
//
// Runs with ith=0, nth=1 (enforced by REDUCTION kind).
// Writes one Q8_K-encoded row into x_q8 (points into lowering scratch arena).
// No heap allocation; no resources consumed.

void ggml_hpx_quantize_q8_k_f32_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * c = static_cast<ggml_hpx_quantize_q8_k_f32_ctx *>(ctx_void);
    assert(ith   == 0);
    assert(nth   == 1);
    assert(begin == 0);
    assert(end   == c->cols);
    assert(c->cols % 256 == 0);
    assert(c->x    != nullptr);
    assert(c->x_q8 != nullptr);
    (void)ith; (void)nth; (void)begin; (void)end;

    quantize_row_q8_K(c->x, c->x_q8, c->cols);
}

// ---------------------------------------------------------------------------
// Q4_K decode pipeline — region 1: parallel Q4_K × Q8_K dot products
// ---------------------------------------------------------------------------
//
// For each output column j in [begin, end):
//   y[j] = ggml_vec_dot_q4_K_q8_K(W_q4k[j], x_q8)
// Dep edge from region 0 ensures x_q8 is fully written before this runs.
// No heap allocation; no resources consumed.

void ggml_hpx_mul_mat_q4_k_q8_k_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * c = static_cast<ggml_hpx_mul_mat_q4_k_q8_k_ctx *>(ctx_void);
    assert(c->cols % 256 == 0);
    assert(c->w_q4k != nullptr);
    assert(c->x_q8  != nullptr);
    assert(c->y     != nullptr);
    assert(begin >= 0 && end <= c->out_cols);

    const auto * w_base = static_cast<const char *>(c->w_q4k);
          auto * y_base = reinterpret_cast<char *>(c->y);

    for (int64_t j = begin; j < end; ++j)
    {
        auto * dst = reinterpret_cast<float *>(y_base + static_cast<size_t>(j) * c->y_nb0);
        ggml_vec_dot_q4_K_q8_K(
            static_cast<int>(c->cols),
            dst,                                                      /* s  */ sizeof(float),
            w_base + static_cast<size_t>(j) * c->w_row_stride,       /* vx */ c->w_row_stride,
            c->x_q8,                                                  /* vy */ c->q8k_row_bytes,
            1);
    }
}

// Repacked variant.  Same R0/R1 pipeline shape as the non-repacked
// kernel above — only the kernel call body differs.  The chunk
// alignment (NB_COLS = 8) matches what the bench validates against the
// ggml CPU backend; mismatched alignment would silently produce
// out-of-band writes into the next chunk's output range.
void ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    constexpr int64_t kNBCols = 8;

    auto * c = static_cast<ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx *>(ctx_void);
    assert(c->cols % 256 == 0);
    assert(c->w_q4k_8x8 != nullptr);
    assert(c->x_q8      != nullptr);
    assert(c->y         != nullptr);
    assert(begin >= 0 && end <= c->out_cols);

    // Snap [begin, end) up to NB_COLS=8 boundaries so the kernel always
    // sees an 8-aligned column range.  Mirrors repack.cpp:4370-4372 and
    // bench-hpx-route-q4k-gemv.cpp's per-chunk math.  Out-of-range
    // chunks (start aligns past the end) collapse to a no-op.
    int64_t s = (begin % kNBCols)
        ? begin + kNBCols - (begin % kNBCols)
        : begin;
    int64_t e = (end % kNBCols)
        ? end + kNBCols - (end % kNBCols)
        : end;
    e = std::min(e, c->out_cols);
    if (s >= e) return;

    auto *       y_base = reinterpret_cast<char *>(c->y);
    const auto * w_base = static_cast<const char *>(c->w_q4k_8x8);

    auto * dst = reinterpret_cast<float *>(y_base + static_cast<size_t>(s) * c->y_nb0);

    ggml_gemv_q4_K_8x8_q8_K(
        static_cast<int>(c->cols),
        dst,
        static_cast<size_t>(c->out_cols),
        w_base + static_cast<size_t>(s) * c->w_row_stride,
        c->x_q8,
        /*nr=*/1,
        static_cast<int>(e - s));
}

// ---------------------------------------------------------------------------
// Single-region executor
// ---------------------------------------------------------------------------

void ggml_hpx_run_single_region(
    ggml_hpx_cpu_region const * region,
    ggml_hpx_region_resources * resources)
{
    assert(region && resources);
    launch_region_async(*region, resources->n_lanes, resources).get();
}

// ---------------------------------------------------------------------------
// Group runner
// ---------------------------------------------------------------------------

void ggml_hpx_run_region_group(
    ggml_hpx_cpu_region_group const * group,
    ggml_hpx_region_resources *       resources)
{
    assert(group && resources);
    int const n       = group->n_regions;
    int const n_lanes = resources->n_lanes;
    if (n == 0) return;

    // ── Small-group fast path ─────────────────────────────────────────────
    //
    // For groups with n_regions <= kSmallN and n_deps <= kSmallN*kSmallN,
    // all bookkeeping structures are stack-allocated.  The execution model,
    // Kahn traversal order, dataflow composition, and wait_all semantics are
    // identical to the heap path below.  Only storage changes.
    //
    // pred_store[i][0..pred_count[i]) holds the predecessor indices of
    // region i, exactly as pred_idx[i] does in the heap path.
    //
    // The ready queue is a plain array used as a FIFO (head/tail indices).
    // Each region is enqueued at most once, so q_tail <= n <= kSmallN.
    //
    // region_fut_arr[i] is assigned for every i in 0..n-1 before wait_all
    // because the topo loop visits every region exactly once (asserted below).
    // The iterator range [begin, begin+n) therefore contains only valid futures.
    if (n <= kSmallN && group->n_deps <= kSmallN * kSmallN)
    {
        std::array<std::array<int, kSmallN>, kSmallN> pred_store{};
        std::array<int, kSmallN>                       pred_count{};
        std::array<int, kSmallN>                       in_deg_arr{};

        for (int e = 0; e < group->n_deps; ++e)
        {
            int const src = group->deps[e].src;
            int const dst = group->deps[e].dst;
            assert(pred_count[static_cast<std::size_t>(dst)] < kSmallN);
            pred_store[static_cast<std::size_t>(dst)]
                      [static_cast<std::size_t>(pred_count[static_cast<std::size_t>(dst)]++)] = src;
            ++in_deg_arr[static_cast<std::size_t>(dst)];
        }

        std::array<int, kSmallN> topo_arr{};
        int                      topo_count = 0;
        {
            int ready_q[kSmallN];
            int q_head = 0, q_tail = 0;

            for (int i = 0; i < n; ++i)
            {
                if (in_deg_arr[static_cast<std::size_t>(i)] == 0)
                    ready_q[q_tail++] = i;
            }
            while (q_head != q_tail)
            {
                int const u          = ready_q[q_head++];
                topo_arr[static_cast<std::size_t>(topo_count++)] = u;
                for (int e = 0; e < group->n_deps; ++e)
                {
                    if (group->deps[e].src != u) continue;
                    int const dst = group->deps[e].dst;
                    if (--in_deg_arr[static_cast<std::size_t>(dst)] == 0)
                        ready_q[q_tail++] = dst;
                }
            }
            assert(topo_count == n);
        }

        std::array<hpx::shared_future<void>, kSmallN> region_fut_arr;

        for (int ii = 0; ii < topo_count; ++ii)
        {
            int const i   = topo_arr[static_cast<std::size_t>(ii)];
            int const cnt = pred_count[static_cast<std::size_t>(i)];

            if (cnt == 0)
            {
                region_fut_arr[static_cast<std::size_t>(i)] =
                    launch_region_async(
                        group->regions[i], n_lanes, resources).share();
                continue;
            }

            std::vector<hpx::shared_future<void>> pred_futs;
            pred_futs.reserve(static_cast<std::size_t>(cnt));
            for (int j = 0; j < cnt; ++j)
            {
                pred_futs.push_back(
                    region_fut_arr[static_cast<std::size_t>(
                        pred_store[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(j)])]);
            }

            region_fut_arr[static_cast<std::size_t>(i)] = hpx::dataflow(
                hpx::launch::async,
                [group, i, n_lanes, resources]
                (std::vector<hpx::shared_future<void>> && /*preds*/)
                {
                    launch_region_async(
                        group->regions[i], n_lanes, resources).get();
                },
                std::move(pred_futs)
            ).share();
        }

        // All n entries region_fut_arr[0..n-1] are now valid.
        hpx::wait_all(region_fut_arr.begin(),
                      region_fut_arr.begin() + static_cast<std::ptrdiff_t>(n));
        return;
    }

    // ── Heap path (n > kSmallN or n_deps > kSmallN*kSmallN) ──────────────
    //
    // Build per-region predecessor index lists from the dep edges.
    // pred_idx[i] holds the region indices whose futures i must wait on.
    std::vector<std::vector<int>> pred_idx(static_cast<std::size_t>(n));
    std::vector<int>              in_deg(static_cast<std::size_t>(n), 0);
    for (int e = 0; e < group->n_deps; ++e)
    {
        int const src = group->deps[e].src;
        int const dst = group->deps[e].dst;
        pred_idx[static_cast<std::size_t>(dst)].push_back(src);
        ++in_deg[static_cast<std::size_t>(dst)];
    }

    // Topological order via Kahn's algorithm.  The group is assumed acyclic
    // (callers are expected to have passed ggml_hpx_validate_region_group
    // with check_acyclic = true).  Visiting in topo order guarantees that
    // every predecessor's shared_future has already been stored in
    // region_fut by the time we compose the current region's dataflow.
    std::vector<int> topo;
    topo.reserve(static_cast<std::size_t>(n));
    {
        std::queue<int> ready;
        for (int i = 0; i < n; ++i)
            if (in_deg[static_cast<std::size_t>(i)] == 0)
                ready.push(i);
        while (!ready.empty())
        {
            int const u = ready.front();
            ready.pop();
            topo.push_back(u);
            for (int e = 0; e < group->n_deps; ++e)
            {
                if (group->deps[e].src != u) continue;
                int const dst = group->deps[e].dst;
                if (--in_deg[static_cast<std::size_t>(dst)] == 0)
                    ready.push(dst);
            }
        }
    }

    // One shared_future per region.  A region with no predecessors launches
    // immediately via launch_region_async; otherwise hpx::dataflow composes
    // the wait-for-preds-then-launch node.  The dataflow callable runs when
    // every predecessor future is ready; it invokes launch_region_async and
    // blocks (cooperatively on its HPX thread) on the resulting future so
    // the returned shared_future<void> becomes ready only once the region
    // itself has completed.
    std::vector<hpx::shared_future<void>> region_fut(
        static_cast<std::size_t>(n));

    for (int i : topo)
    {
        auto const & preds = pred_idx[static_cast<std::size_t>(i)];

        if (preds.empty())
        {
            region_fut[static_cast<std::size_t>(i)] =
                launch_region_async(
                    group->regions[i], n_lanes, resources).share();
            continue;
        }

        std::vector<hpx::shared_future<void>> pred_futs;
        pred_futs.reserve(preds.size());
        for (int p : preds)
            pred_futs.push_back(region_fut[static_cast<std::size_t>(p)]);

        region_fut[static_cast<std::size_t>(i)] = hpx::dataflow(
            hpx::launch::async,
            [group, i, n_lanes, resources]
            (std::vector<hpx::shared_future<void>> && /*preds*/)
            {
                launch_region_async(
                    group->regions[i], n_lanes, resources).get();
            },
            std::move(pred_futs)
        ).share();
    }

    // Group barrier.  wait_all rethrows any stored exception; partially
    // completed concurrent regions are not forcibly canceled (cooperative
    // abort per HPX_EXECUTOR_CONTRACT.md §6).
    hpx::wait_all(region_fut);
}
