// bench_hpx_mlp_gate_up_f32.cpp
//
// Microbenchmark: composed-group run vs frozen-packet bind+run
// for the MLP gate/up sublayer (gate MUL_MAT, up MUL_MAT, SiLU, elementwise MUL).
//
// Paths under test
// ─────────────────
//   dag_group     — composed 4-region group via ggml_hpx_compose_mlp_gate_up_group;
//                   outer hpx::async paid once per (shape) block;
//                   only ggml_hpx_run_region_group is in the timed hot loop.
//
//   direct_manual — 4 run_range callbacks called in fixed order on the caller
//                   thread (no HPX, no DAG).  ctx structs built once per shape
//                   (not per rep) since bind does not rebuild dims each iteration.
//                   n_lanes=1: ith=0, nth=1 — identical to the packet SERIAL path.
//                   Timed body: 4 run_range calls only.
//
//   frozen_packet — compiled 4-step packet; pinned decode Exec.
//                   n_lanes=1: all steps are SERIAL (no inner for_loop crossing).
//                   Timed body: bind (7 pointer patches) + run_frozen_packet.
//
// Shape sweep
// ────────────
//   Focused:        out_cols=3, cols=4, rows=2   (matches correctness test)
//   Sweep rows:     rows ∈ {1,2,4,8,16},   out_cols=64, cols=64
//   Sweep cols:     cols ∈ {32,64,128,256}, out_cols=64, rows=1
//   Sweep out_cols: out_cols ∈ {32,64,128,256}, cols=64, rows=1
//
// n_lanes=1 throughout (decode path).
//
// Output: CSV with columns
//   impl,out_cols,cols,rows,lanes,reps,min_ns,median_ns,p95_ns
// Ratios (stderr, one line per shape):
//   dag/pkt  direct/pkt  dag/direct  (median and p95)

#ifndef GGML_HPX_REGION_DAG
#  error "bench_hpx_mlp_gate_up_f32.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-compose.h"
#include "ggml-hpx-packet.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"
#include "ggml.h"

#include <hpx/future.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

// ---------------------------------------------------------------------------
// direct_manual helper
// ---------------------------------------------------------------------------
//
// Calls the 4 production run_range callbacks in topological order on the
// caller thread with ith=0, nth=1.  This matches the packet SERIAL step
// contract exactly (n_lanes=1): same callbacks, same work ranges, same
// resources struct — zero dispatch overhead on top.
//
// ctx structs are built once per shape (not per rep): bind patches only
// pointer fields, not structural dims, so there is nothing equivalent to
// rebuild in the direct path either.

static void run_mlp_gate_up_direct_manual(
    ggml_hpx_mul_mat_f32_ctx *  gate_ctx,
    ggml_hpx_mul_mat_f32_ctx *  up_ctx,
    ggml_hpx_silu_f32_ctx *     silu_ctx,
    ggml_hpx_mul_f32_ctx *      mul_ctx,
    ggml_hpx_region_resources * res)
{
    ggml_hpx_mul_mat_f32_run_range(gate_ctx, 0, 1, 0, gate_ctx->out_cols, res);
    ggml_hpx_mul_mat_f32_run_range(up_ctx,   0, 1, 0, up_ctx->out_cols,   res);
    ggml_hpx_silu_f32_run_range(  silu_ctx,  0, 1, 0, silu_ctx->n,        res);
    ggml_hpx_mul_f32_run_range(   mul_ctx,   0, 1, 0, mul_ctx->n,         res);
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

static float silu_f32(float v)
{
    return v / (1.0f + std::exp(-v));
}

// gate[r][j] = sum_k x[r][k] * W_gate[j][k]
// up  [r][j] = sum_k x[r][k] * W_up  [j][k]
// out [r][j] = silu(gate[r][j]) * up[r][j]
static void mlp_gate_up_ref_f32(
    const float * w_gate,
    const float * w_up,
    const float * x,
    float *       out_ref,
    int64_t       rows,
    int64_t       cols,
    int64_t       out_cols)
{
    for (int64_t r = 0; r < rows; ++r)
    {
        for (int64_t j = 0; j < out_cols; ++j)
        {
            float g = 0.f, u = 0.f;
            for (int64_t k = 0; k < cols; ++k)
            {
                const float xv = x[r * cols + k];
                g += xv * w_gate[j * cols + k];
                u += xv * w_up  [j * cols + k];
            }
            out_ref[r * out_cols + j] = silu_f32(g) * u;
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic input fill
// ---------------------------------------------------------------------------

static void fill_deterministic(float * buf, int64_t n, float scale = 1.0f)
{
    for (int64_t i = 0; i < n; ++i)
    {
        buf[i] = scale * ((i % 2 == 0) ? 1.0f : -1.0f)
               * (1.0f + static_cast<float>(i % 17));
    }
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------

// Relative tolerance: passes when |ref - got| <= tol * max(|ref|, 1).
static bool verify_match(
    const float * ref,
    const float * got,
    int64_t       n,
    float         rel_tol)
{
    for (int64_t i = 0; i < n; ++i)
    {
        const float abs_diff = std::abs(ref[i] - got[i]);
        const float scale    = std::max(1.0f, std::abs(ref[i]));
        if (abs_diff > rel_tol * scale)
        {
            std::fprintf(stderr,
                "  mismatch at i=%" PRId64 ": ref=%.8f got=%.8f"
                " abs_diff=%.3e rel_diff=%.3e\n",
                i,
                static_cast<double>(ref[i]),
                static_cast<double>(got[i]),
                static_cast<double>(abs_diff),
                static_cast<double>(abs_diff / scale));
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct BenchStats
{
    double min_ns;
    double median_ns;
    double p95_ns;
    int    reps;
};

template <typename Fn>
static BenchStats bench_ns(Fn fn, int n_warmup, int n_reps)
{
    for (int i = 0; i < n_warmup; ++i) { fn(); }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(n_reps));

    for (int i = 0; i < n_reps; ++i)
    {
        auto const t0 = std::chrono::steady_clock::now();
        fn();
        auto const t1 = std::chrono::steady_clock::now();
        using Nanos = std::chrono::duration<double, std::nano>;
        samples.push_back(Nanos{t1 - t0}.count());
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t ns = samples.size();
    return {
        samples[0],
        samples[ns / 2],
        samples[static_cast<std::size_t>(0.95 * static_cast<double>(ns - 1))],
        n_reps,
    };
}

// Enter HPX context once; run the entire warmup+reps loop from within that
// task.  No per-rep outer hpx::async crossing.
template <typename Fn>
static BenchStats bench_ns_in_hpx(Fn fn, int n_warmup, int n_reps)
{
    BenchStats result{};
    hpx::async([&]()
    {
        result = bench_ns(fn, n_warmup, n_reps);
    }).get();
    return result;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

static void print_csv_header()
{
    std::printf("impl,out_cols,cols,rows,lanes,reps,min_ns,median_ns,p95_ns\n");
}

static void print_csv_row(
    const char *       impl,
    int64_t            out_cols,
    int64_t            cols,
    int64_t            rows,
    int                lanes,
    BenchStats const & s)
{
    std::printf("%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%d,%d,%.1f,%.1f,%.1f\n",
        impl, out_cols, cols, rows, lanes, s.reps,
        s.min_ns, s.median_ns, s.p95_ns);
}

// ---------------------------------------------------------------------------
// ggml context RAII
// ---------------------------------------------------------------------------

struct GCtx
{
    ggml_context * ctx;

    explicit GCtx(size_t mem_bytes = 256 * 1024)
    {
        ggml_init_params p{};
        p.mem_size   = mem_bytes;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);
    }
    ~GCtx() { if (ctx) ggml_free(ctx); }

    GCtx(const GCtx &)             = delete;
    GCtx & operator=(const GCtx &) = delete;
};

// ---------------------------------------------------------------------------
// Shape descriptor
// ---------------------------------------------------------------------------

struct Shape
{
    int64_t out_cols;
    int64_t cols;
    int64_t rows;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    ggml_hpx_tpool_start();

    static constexpr int kWarmup = 50;
    static constexpr int kReps   = 1000;
    static constexpr int kLanes  = 1;

    // Focused point matching the correctness test, then 1D sweeps.
    static const Shape kShapes[] = {
        // focused: matches test_hpx_packet_mlp_gate_up_f32
        { 3, 4, 2 },
        // sweep rows (out_cols=64, cols=64)
        { 64, 64,  1 },
        { 64, 64,  2 },
        { 64, 64,  4 },
        { 64, 64,  8 },
        { 64, 64, 16 },
        // sweep cols (out_cols=64, rows=1)
        { 64,  32, 1 },
        { 64, 128, 1 },
        { 64, 256, 1 },
        // sweep out_cols (cols=64, rows=1)
        {  32, 64, 1 },
        { 128, 64, 1 },
        { 256, 64, 1 },
    };

    // Packet runtime shared across all shapes.
    auto * rt = ggml_hpx_packet_runtime_create(static_cast<uint32_t>(kLanes));
    if (rt == nullptr)
    {
        std::fprintf(stderr, "ABORT: packet_runtime_create failed\n");
        return 1;
    }

    std::fprintf(stderr,
        "# bench_hpx_mlp_gate_up_f32  warmup=%d  reps=%d  lanes=%d\n"
        "# dag_group     = composed 4-region group; only run_region_group timed\n"
        "# direct_manual = 4 run_range calls in fixed order; no HPX, no DAG\n"
        "# frozen_packet = compiled 4-step packet; bind+run per rep\n"
        "#\n"
        "# %-16s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
        kWarmup, kReps, kLanes,
        "shape(oc×c×r)",
        "dag_med", "dag_p95",
        "dm_med",  "dm_p95",
        "pkt_med", "pkt_p95",
        "dag/pkt", "dm/pkt",  "dag/dm");

    print_csv_header();

    for (const Shape & sh : kShapes)
    {
        const int64_t out_cols = sh.out_cols;
        const int64_t cols     = sh.cols;
        const int64_t rows     = sh.rows;

        const std::size_t n_wgt = static_cast<std::size_t>(out_cols * cols);
        const std::size_t n_x   = static_cast<std::size_t>(rows * cols);
        const std::size_t n_out = static_cast<std::size_t>(rows * out_cols);

        std::vector<float> wg(n_wgt);
        std::vector<float> wu(n_wgt);
        std::vector<float> xb(n_x);
        std::vector<float> gate_buf    (n_out, 0.f);
        std::vector<float> up_buf      (n_out, 0.f);
        std::vector<float> gate_act_buf(n_out, 0.f);
        std::vector<float> out_buf     (n_out, 0.f);
        std::vector<float> out_ref     (n_out, 0.f);

        fill_deterministic(wg.data(), static_cast<int64_t>(n_wgt), 0.1f);
        fill_deterministic(wu.data(), static_cast<int64_t>(n_wgt), 0.2f);
        fill_deterministic(xb.data(), static_cast<int64_t>(n_x));

        // Scalar reference (output only; intermediate bufs not compared)
        mlp_gate_up_ref_f32(
            wg.data(), wu.data(), xb.data(),
            out_ref.data(), rows, cols, out_cols);

        // ── ggml context + subgraph ───────────────────────────────────────────
        GCtx gc;
        ggml_tensor * t_wg  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
        ggml_tensor * t_wu  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
        ggml_tensor * t_x   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
        ggml_tensor * t_g   = ggml_mul_mat(gc.ctx, t_wg, t_x);
        ggml_tensor * t_u   = ggml_mul_mat(gc.ctx, t_wu, t_x);
        ggml_tensor * t_ga  = ggml_silu(gc.ctx, t_g);
        ggml_tensor * t_out = ggml_mul(gc.ctx, t_ga, t_u);

        t_wg->data  = wg.data();
        t_wu->data  = wu.data();
        t_x->data   = xb.data();
        t_g->data   = gate_buf.data();
        t_u->data   = up_buf.data();
        t_ga->data  = gate_act_buf.data();
        t_out->data = out_buf.data();

        // ── Compose ───────────────────────────────────────────────────────────
        ggml_hpx_mlp_gate_up_group grp;
        ggml_hpx_mlp_gate_up_group_init(&grp);

        if (!ggml_hpx_compose_mlp_gate_up_group(t_g, t_u, t_ga, t_out, &grp))
        {
            std::fprintf(stderr,
                "ABORT: compose failed out_cols=%" PRId64
                " cols=%" PRId64 " rows=%" PRId64 "\n",
                out_cols, cols, rows);
            ggml_hpx_packet_runtime_destroy(rt);
            return 1;
        }

        // Resources: MLP has 0-byte lane_scratch and reduction_buffer.
        // Provide a valid lane_ptrs[1] pointing to a dummy float so that
        // any run_range that does resources->lane_scratch[ith] safely bounces.
        float    lane_dummy = 0.f;
        void   * lane_ptr   = &lane_dummy;
        ggml_hpx_region_resources resources{};
        resources.n_lanes          = kLanes;
        resources.lane_scratch     = &lane_ptr;
        resources.reduction_buffer = nullptr;
        resources.shared_scratch   = nullptr;

        // ── dag_group correctness ─────────────────────────────────────────────
        std::fill(out_buf.begin(), out_buf.end(), 0.f);
        hpx::async([&]()
        {
            ggml_hpx_run_region_group(&grp.group, &resources);
        }).get();

        if (!verify_match(out_ref.data(), out_buf.data(),
                          static_cast<int64_t>(n_out), 1e-5f))
        {
            std::fprintf(stderr,
                "ABORT: dag_group correctness failed out_cols=%" PRId64
                " cols=%" PRId64 " rows=%" PRId64 "\n",
                out_cols, cols, rows);
            ggml_hpx_packet_runtime_destroy(rt);
            return 1;
        }

        // ── dag_group timing ──────────────────────────────────────────────────
        BenchStats dag_stats = bench_ns_in_hpx([&]()
        {
            ggml_hpx_run_region_group(&grp.group, &resources);
        }, kWarmup, kReps);

        print_csv_row("dag_group", out_cols, cols, rows, kLanes, dag_stats);

        // ── direct_manual ctxs (built once per shape, not per rep) ───────────
        //
        // Pointer fields match the binding struct; dims are fixed for this shape.
        // Gate MUL_MAT: x=activations, w=W_gate, y=gate output.
        // SiLU reads gate output, writes gate_act.
        // EW MUL: a=gate_act, b=up output, dst=final output.
        const int64_t n_elem = rows * out_cols;

        ggml_hpx_mul_mat_f32_ctx dm_gate_ctx{
            xb.data(), wg.data(), gate_buf.data(), rows, cols, out_cols };
        ggml_hpx_mul_mat_f32_ctx dm_up_ctx{
            xb.data(), wu.data(), up_buf.data(),   rows, cols, out_cols };
        ggml_hpx_silu_f32_ctx dm_silu_ctx{
            gate_buf.data(), gate_act_buf.data(), n_elem };
        ggml_hpx_mul_f32_ctx dm_mul_ctx{
            gate_act_buf.data(), up_buf.data(), out_buf.data(), n_elem };

        // ── direct_manual correctness ─────────────────────────────────────────
        std::fill(out_buf.begin(), out_buf.end(), 0.f);
        run_mlp_gate_up_direct_manual(
            &dm_gate_ctx, &dm_up_ctx, &dm_silu_ctx, &dm_mul_ctx, &resources);

        if (!verify_match(out_ref.data(), out_buf.data(),
                          static_cast<int64_t>(n_out), 1e-5f))
        {
            std::fprintf(stderr,
                "ABORT: direct_manual correctness failed out_cols=%" PRId64
                " cols=%" PRId64 " rows=%" PRId64 "\n",
                out_cols, cols, rows);
            ggml_hpx_packet_runtime_destroy(rt);
            return 1;
        }

        // ── direct_manual timing ──────────────────────────────────────────────
        // No HPX crossing needed: all 4 callbacks are synchronous on the caller
        // thread.  bench_ns_in_hpx is still used so the timing context matches
        // dag_group and frozen_packet (same scheduler state, same CPU frequency).
        BenchStats dm_stats = bench_ns_in_hpx([&]()
        {
            run_mlp_gate_up_direct_manual(
                &dm_gate_ctx, &dm_up_ctx, &dm_silu_ctx, &dm_mul_ctx, &resources);
        }, kWarmup, kReps);

        print_csv_row("direct_manual", out_cols, cols, rows, kLanes, dm_stats);

        // ── Compile packet ────────────────────────────────────────────────────
        ggml_hpx_packet_plan_key key{};
        key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32;
        key.team           = GGML_HPX_PACKET_TEAM_DECODE;
        key.n_lanes        = static_cast<uint32_t>(kLanes);
        key.dtype          = GGML_TYPE_F32;
        key.policy_version = 1;
        key.shape[0]       = out_cols;
        key.shape[1]       = cols;
        key.shape[2]       = rows;

        const char * compile_err = nullptr;
        auto * packet = ggml_hpx_compile_packet(&grp.group, &key, &compile_err);
        if (packet == nullptr)
        {
            std::fprintf(stderr,
                "ABORT: compile failed out_cols=%" PRId64
                " cols=%" PRId64 " rows=%" PRId64 ": %s\n",
                out_cols, cols, rows,
                compile_err != nullptr ? compile_err : "(null)");
            ggml_hpx_packet_runtime_destroy(rt);
            return 1;
        }

        // ── Frame ─────────────────────────────────────────────────────────────
        const size_t frame_sz = ggml_hpx_packet_frame_size(packet);
        const size_t frame_al = ggml_hpx_packet_frame_align(packet);
        void * frame_raw = ::operator new(frame_sz, std::align_val_t{frame_al});
        auto * frame = static_cast<ggml_hpx_packet_frame *>(frame_raw);
        ggml_hpx_packet_frame_init(frame, packet);

        // ── Binding ───────────────────────────────────────────────────────────
        ggml_hpx_mlp_gate_up_binding bind{};
        bind.w_gate   = wg.data();
        bind.w_up     = wu.data();
        bind.x        = xb.data();
        bind.gate     = gate_buf.data();
        bind.up       = up_buf.data();
        bind.gate_act = gate_act_buf.data();
        bind.out      = out_buf.data();

        // ── frozen_packet correctness ─────────────────────────────────────────
        std::fill(out_buf.begin(), out_buf.end(), 0.f);
        hpx::async([&]()
        {
            ggml_hpx_bind_mlp_gate_up_packet(frame, &bind);
            ggml_hpx_run_frozen_packet(rt, packet, frame, &resources);
        }).get();

        if (!verify_match(out_ref.data(), out_buf.data(),
                          static_cast<int64_t>(n_out), 1e-5f))
        {
            std::fprintf(stderr,
                "ABORT: frozen_packet correctness failed out_cols=%" PRId64
                " cols=%" PRId64 " rows=%" PRId64 "\n",
                out_cols, cols, rows);
            ::operator delete(frame_raw, std::align_val_t{frame_al});
            ggml_hpx_free_packet(packet);
            ggml_hpx_packet_runtime_destroy(rt);
            return 1;
        }

        // ── frozen_packet timing ──────────────────────────────────────────────
        BenchStats pkt_stats = bench_ns_in_hpx([&]()
        {
            ggml_hpx_bind_mlp_gate_up_packet(frame, &bind);
            ggml_hpx_run_frozen_packet(rt, packet, frame, &resources);
        }, kWarmup, kReps);

        print_csv_row("frozen_packet", out_cols, cols, rows, kLanes, pkt_stats);

        // ── Per-shape ratio summary (stderr) ──────────────────────────────────
        char shape_label[32];
        std::snprintf(shape_label, sizeof(shape_label),
            "%" PRId64 "×%" PRId64 "×%" PRId64, out_cols, cols, rows);

        std::fprintf(stderr,
            "# %-16s %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.1f %9.1f %9.1f\n",
            shape_label,
            dag_stats.median_ns, dag_stats.p95_ns,
            dm_stats.median_ns,  dm_stats.p95_ns,
            pkt_stats.median_ns, pkt_stats.p95_ns,
            dag_stats.median_ns / pkt_stats.median_ns,
            dm_stats.median_ns  / pkt_stats.median_ns,
            dag_stats.median_ns / dm_stats.median_ns);

        // ── Cleanup ───────────────────────────────────────────────────────────
        ::operator delete(frame_raw, std::align_val_t{frame_al});
        ggml_hpx_free_packet(packet);

        std::fflush(stdout);
    }

    ggml_hpx_packet_runtime_destroy(rt);
    std::fprintf(stderr, "# done.\n");
    return 0;
}
