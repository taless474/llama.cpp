// bench_hpx_rms_norm_f32.cpp
//
// Microbenchmark: decomposing DAG cost into layers.
//
// Paths under test
// ─────────────────
//   ref              — single-threaded scalar loop (models ggml CPU baseline)
//   direct           — three passes via production callbacks; no HPX, no DAG
//   fused_async      — one hpx::async(...).get() around the fused body;
//                      measures the cost of a single HPX boundary per call
//   dag_empty_nowrap — R0→R1→R2 group, near-nop callbacks, outer async once;
//                      orchestration floor of the 3-region DAG design
//   dag_nowrap       — R0→R1→R2 group, production callbacks, outer async once;
//                      full DAG path with HPX context amortised over reps
//   frozen_packet    — compiled 3-step packet, production callbacks, pinned
//                      decode Exec; only bind+run inside the timed loop.
//                      Two HPX entries per rep when n_lanes > 1 (R0 + R2
//                      lane fan-outs); zero when n_lanes == 1.
//
// Correctness checks (abort on mismatch):
//   ref vs direct        (single-threaded; must agree exactly)
//   ref vs fused_async   (one check before timing)
//   ref vs dag_nowrap    (one check per (n, lanes) before timing)
//
// Timed body:
//   direct           — plain function call, main thread
//   fused_async      — hpx::async(...).get() per rep, includes task-post cost
//   dag_empty_nowrap — bench_ns_in_hpx: outer async once, reps call runner directly
//   dag_nowrap       — bench_ns_in_hpx: outer async once, reps call runner directly
//
// Excluded from timing:
//   input generation, buffer allocation, lane_ptrs setup
//
// Output: CSV with columns impl,n,lanes,reps,min_ns,median_ns,p95_ns

#ifndef GGML_HPX_REGION_DAG
#  error "bench_hpx_rms_norm_f32.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-packet.h"
#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"

#include <hpx/async_base/async.hpp>
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
// Scalar reference
// ---------------------------------------------------------------------------

static void rms_norm_ref_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps)
{
    float sumsq = 0.0f;
    for (int64_t i = 0; i < n; ++i)
    {
        sumsq += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sumsq / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i)
    {
        dst[i] = x[i] * scale;
    }
}

// ---------------------------------------------------------------------------
// Fused body: three passes via production callbacks, no HPX, no DAG.
// Shared by the 'direct' and 'fused_async' benchmark paths.
// ---------------------------------------------------------------------------

static void rms_norm_fused_body_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps)
{
    float  lane_scalar = 0.0f;
    void * lane_ptr    = &lane_scalar;

    ggml_hpx_rms_norm_f32_reduce_buffer reduce_buf{};

    ggml_hpx_rms_norm_partial_f32_ctx  partial_ctx{x, n};
    ggml_hpx_rms_norm_finalize_f32_ctx finalize_ctx{n, eps};
    ggml_hpx_rms_norm_apply_f32_ctx    apply_ctx{x, dst, n};

    ggml_hpx_region_resources res{};
    res.lane_scratch     = &lane_ptr;
    res.reduction_buffer = &reduce_buf;
    res.n_lanes          = 1;

    ggml_hpx_rms_norm_partial_f32_run_range( &partial_ctx,  0, 1, 0, n, &res);
    ggml_hpx_rms_norm_finalize_f32_run_range(&finalize_ctx, 0, 1, 0, n, &res);
    ggml_hpx_rms_norm_apply_f32_run_range(   &apply_ctx,    0, 1, 0, n, &res);
}

// 'direct' benchmark path: fused body called on the main thread.
static void rms_norm_direct_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps)
{
    rms_norm_fused_body_f32(x, dst, n, eps);
}

// 'fused_async' benchmark path: one hpx::async(...).get() around the fused body.
// Each call pays one task-post; no region DAG, no dataflow chain.
static void rms_norm_fused_async_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps)
{
    hpx::async([&]()
    {
        rms_norm_fused_body_f32(x, dst, n, eps);
    }).get();
}

// ---------------------------------------------------------------------------
// Empty callbacks: same resource shapes, near-nop bodies
// ---------------------------------------------------------------------------

static void empty_partial_run_range(
    void *                      /*ctx*/,
    int                         ith,
    int                         /*nth*/,
    int64_t                     /*begin*/,
    int64_t                     /*end*/,
    ggml_hpx_region_resources * resources)
{
    *static_cast<float *>(resources->lane_scratch[ith]) = 1.0f;
}

static void empty_finalize_run_range(
    void *                      /*ctx*/,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     /*begin*/,
    int64_t                     /*end*/,
    ggml_hpx_region_resources * resources)
{
    float dummy = 0.0f;
    for (int i = 0; i < resources->n_lanes; ++i)
    {
        dummy += *static_cast<float *>(resources->lane_scratch[i]);
    }

    auto * buf = static_cast<ggml_hpx_rms_norm_f32_reduce_buffer *>(
        resources->reduction_buffer);
    buf->scale = dummy;
}

static void empty_apply_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    auto * ctx = static_cast<ggml_hpx_rms_norm_apply_f32_ctx *>(ctx_void);
    auto const * buf = static_cast<ggml_hpx_rms_norm_f32_reduce_buffer const *>(
        resources->reduction_buffer);
    if (begin < end && begin < ctx->n)
    {
        ctx->dst[begin] = buf->scale;
    }
}

// ---------------------------------------------------------------------------
// Empty DAG path: R0→R1→R2 structure with near-nop callbacks
// ---------------------------------------------------------------------------

static void rms_norm_empty_dag_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps,
    int           n_lanes,
    void **       lane_ptrs)
{
    ggml_hpx_rms_norm_f32_reduce_buffer reduce_buf{};

    ggml_hpx_rms_norm_partial_f32_ctx  partial_ctx{x, n};
    ggml_hpx_rms_norm_finalize_f32_ctx finalize_ctx{n, eps};
    ggml_hpx_rms_norm_apply_f32_ctx    apply_ctx{x, dst, n};

    ggml_hpx_cpu_region regions[3] = {
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            &partial_ctx,
            empty_partial_run_range,
        },
        {
            GGML_HPX_CPU_REGION_KIND_REDUCTION,
            0, n, 0,
            &finalize_ctx,
            empty_finalize_run_range,
        },
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            &apply_ctx,
            empty_apply_run_range,
        },
    };

    ggml_hpx_dep_edge deps[2] = {{0, 1}, {1, 2}};

    ggml_hpx_cpu_region_group group{regions, 3, deps, 2};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_ptrs;
    resources.reduction_buffer = &reduce_buf;
    resources.n_lanes          = n_lanes;

    ggml_hpx_run_region_group(&group, &resources);
}

// ---------------------------------------------------------------------------
// Region-DAG path: ctx/region/deps/resources on stack, then run
// ---------------------------------------------------------------------------

static void rms_norm_dag_f32(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps,
    int           n_lanes,
    void **       lane_ptrs)
{
    ggml_hpx_rms_norm_f32_reduce_buffer reduce_buf{};

    ggml_hpx_rms_norm_partial_f32_ctx partial_ctx{x, n};
    ggml_hpx_rms_norm_finalize_f32_ctx finalize_ctx{n, eps};
    ggml_hpx_rms_norm_apply_f32_ctx apply_ctx{x, dst, n};

    ggml_hpx_cpu_region regions[3] = {
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            &partial_ctx,
            ggml_hpx_rms_norm_partial_f32_run_range,
        },
        {
            GGML_HPX_CPU_REGION_KIND_REDUCTION,
            0, n, 0,
            &finalize_ctx,
            ggml_hpx_rms_norm_finalize_f32_run_range,
        },
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            &apply_ctx,
            ggml_hpx_rms_norm_apply_f32_run_range,
        },
    };

    ggml_hpx_dep_edge deps[2] = {{0, 1}, {1, 2}};

    ggml_hpx_cpu_region_group group{regions, 3, deps, 2};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_ptrs;
    resources.reduction_buffer = &reduce_buf;
    resources.n_lanes          = n_lanes;

    ggml_hpx_run_region_group(&group, &resources);
}

// ---------------------------------------------------------------------------
// Deterministic input: alternating sign, moderate magnitude
// ---------------------------------------------------------------------------

static void fill_deterministic(float * buf, int64_t n)
{
    for (int64_t i = 0; i < n; ++i)
    {
        buf[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * (1.0f + static_cast<float>(i % 17));
    }
}

// ---------------------------------------------------------------------------
// Timing: collect all samples in nanoseconds
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
    for (int i = 0; i < n_warmup; ++i)
    {
        fn();
    }

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

    auto const  ns      = samples.size();
    double const min_ns    = samples[0];
    double const median_ns = samples[ns / 2];
    double const p95_ns    = samples[static_cast<std::size_t>(
                                 0.95 * static_cast<double>(ns - 1))];

    return {min_ns, median_ns, p95_ns, n_reps};
}

// bench_ns_in_hpx: enter HPX context once, run the entire warmup+reps loop
// from within that task.  The timed fn() calls the group runner directly —
// no per-rep outer hpx::async crossing.
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
// Correctness check
// ---------------------------------------------------------------------------

static bool verify_match(
    const float * a,
    const float * b,
    int64_t       n,
    float         tol)
{
    for (int64_t i = 0; i < n; ++i)
    {
        if (std::abs(a[i] - b[i]) > tol)
        {
            std::fprintf(stderr,
                "  mismatch at i=%" PRId64 ": ref=%.8f dag=%.8f diff=%.3e\n",
                i,
                static_cast<double>(a[i]),
                static_cast<double>(b[i]),
                static_cast<double>(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

static void print_csv_header()
{
    std::printf("impl,n,lanes,reps,min_ns,median_ns,p95_ns\n");
}

static void print_csv_row(
    const char * impl,
    int64_t      n,
    int          lanes,
    BenchStats const & s)
{
    std::printf("%s,%" PRId64 ",%d,%d,%.1f,%.1f,%.1f\n",
        impl, n, lanes, s.reps,
        s.min_ns, s.median_ns, s.p95_ns);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    ggml_hpx_tpool_start();

    static constexpr float   kEps    = 1e-5f;
    static constexpr int     kWarmup = 50;
    static constexpr int     kReps   = 1000;
    static constexpr int     kNLanes[] = {1, 2, 4};
    static constexpr int64_t kSizes[]  = {512, 2048, 4096, 8192};

    static constexpr int kMaxLanes = 4;

    std::fprintf(stderr,
        "# bench_hpx_rms_norm_f32  eps=%.0e  warmup=%d  reps=%d\n"
        "# ref              = single-threaded scalar loop\n"
        "# direct           = fused body called on main thread, no HPX\n"
        "# fused_async      = one hpx::async(...).get() per rep around fused body\n"
        "# dag_empty_nowrap = 3-region group, near-nop callbacks, outer async once\n"
        "# dag_nowrap       = 3-region group, production callbacks, outer async once\n"
        "# frozen_packet    = compiled 3-step packet, pinned decode Exec; "
        "bind+run per rep\n",
        static_cast<double>(kEps), kWarmup, kReps);

    print_csv_header();

    for (int64_t n : kSizes)
    {
        auto const nu = static_cast<std::size_t>(n);
        std::vector<float> x(nu);
        std::vector<float> dst_ref(nu);
        std::vector<float> dst_direct(nu);
        std::vector<float> dst_dag(nu);

        fill_deterministic(x.data(), n);

        float  lane_scalars[kMaxLanes] = {};
        void * lane_ptrs[kMaxLanes]    = {};
        for (int i = 0; i < kMaxLanes; ++i)
        {
            lane_ptrs[i] = &lane_scalars[i];
        }

        // -- ref: bench once per n, lanes=1 --

        rms_norm_ref_f32(x.data(), dst_ref.data(), n, kEps);

        BenchStats ref_stats = bench_ns([&]()
        {
            rms_norm_ref_f32(x.data(), dst_ref.data(), n, kEps);
        }, kWarmup, kReps);

        print_csv_row("ref", n, 1, ref_stats);

        // -- direct: bench once per n, lanes=1; check against ref --

        rms_norm_direct_f32(x.data(), dst_direct.data(), n, kEps);

        if (!verify_match(dst_ref.data(), dst_direct.data(), n, 1e-6f))
        {
            std::fprintf(stderr,
                "ABORT: direct correctness failed n=%" PRId64 "\n", n);
            return 1;
        }

        BenchStats direct_stats = bench_ns([&]()
        {
            rms_norm_direct_f32(x.data(), dst_direct.data(), n, kEps);
        }, kWarmup, kReps);

        print_csv_row("direct", n, 1, direct_stats);

        // -- fused_async: one hpx::async(...).get() per rep around the fused body --

        rms_norm_fused_async_f32(x.data(), dst_direct.data(), n, kEps);

        if (!verify_match(dst_ref.data(), dst_direct.data(), n, 1e-6f))
        {
            std::fprintf(stderr,
                "ABORT: fused_async correctness failed n=%" PRId64 "\n", n);
            return 1;
        }

        BenchStats fused_async_stats = bench_ns([&]()
        {
            rms_norm_fused_async_f32(x.data(), dst_direct.data(), n, kEps);
        }, kWarmup, kReps);

        print_csv_row("fused_async", n, 1, fused_async_stats);

        // -- dag_empty_nowrap / dag_nowrap: one row each per (n, lanes) --

        for (int n_lanes : kNLanes)
        {
            // dag_empty_nowrap: outer hpx::async paid once for the whole loop
            for (int i = 0; i < n_lanes; ++i) { lane_scalars[i] = 0.0f; }
            std::fill(dst_dag.begin(), dst_dag.end(), 0.0f);

            BenchStats empty_nowrap_stats = bench_ns_in_hpx([&]()
            {
                rms_norm_empty_dag_f32(
                    x.data(), dst_dag.data(), n, kEps,
                    n_lanes, lane_ptrs);
            }, kWarmup, kReps);

            print_csv_row("dag_empty_nowrap", n, n_lanes, empty_nowrap_stats);

            // dag_nowrap: correctness check before timing
            std::fill(dst_dag.begin(), dst_dag.end(), 0.0f);
            for (int i = 0; i < n_lanes; ++i) { lane_scalars[i] = 0.0f; }

            hpx::async([&]()
            {
                rms_norm_dag_f32(
                    x.data(), dst_dag.data(), n, kEps,
                    n_lanes, lane_ptrs);
            }).get();

            if (!verify_match(dst_ref.data(), dst_dag.data(), n, 1e-6f))
            {
                std::fprintf(stderr,
                    "ABORT: dag_nowrap correctness failed n=%" PRId64 " lanes=%d\n",
                    n, n_lanes);
                return 1;
            }

            BenchStats nowrap_stats = bench_ns_in_hpx([&]()
            {
                rms_norm_dag_f32(
                    x.data(), dst_dag.data(), n, kEps,
                    n_lanes, lane_ptrs);
            }, kWarmup, kReps);

            print_csv_row("dag_nowrap", n, n_lanes, nowrap_stats);

            // frozen_packet: compile once per (n, lanes); bind+run in the timed loop.
            std::fill(dst_dag.begin(), dst_dag.end(), 0.0f);
            for (int i = 0; i < n_lanes; ++i) { lane_scalars[i] = 0.0f; }

            ggml_hpx_rms_norm_f32_reduce_buffer packet_reduce_buf{};
            ggml_hpx_rms_norm_partial_f32_ctx   packet_partial_ctx{x.data(), n};
            ggml_hpx_rms_norm_finalize_f32_ctx  packet_finalize_ctx{n, kEps};
            ggml_hpx_rms_norm_apply_f32_ctx     packet_apply_ctx{
                x.data(), dst_dag.data(), n};

            ggml_hpx_cpu_region packet_regions[3] = {
                {
                    GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
                    0, n, 0,
                    &packet_partial_ctx,
                    ggml_hpx_rms_norm_partial_f32_run_range,
                },
                {
                    GGML_HPX_CPU_REGION_KIND_REDUCTION,
                    0, n, 0,
                    &packet_finalize_ctx,
                    ggml_hpx_rms_norm_finalize_f32_run_range,
                },
                {
                    GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
                    0, n, 0,
                    &packet_apply_ctx,
                    ggml_hpx_rms_norm_apply_f32_run_range,
                },
            };
            ggml_hpx_dep_edge packet_deps[2] = {{0, 1}, {1, 2}};
            ggml_hpx_cpu_region_group packet_fine_group{
                packet_regions, 3, packet_deps, 2};

            ggml_hpx_packet_plan_key packet_key{};
            packet_key.sublayer       = GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32;
            packet_key.team           = GGML_HPX_PACKET_TEAM_DECODE;
            packet_key.n_lanes        = static_cast<uint32_t>(n_lanes);
            packet_key.dtype          = 0;
            packet_key.seq_regime     = 0;
            packet_key.policy_version = 0;
            packet_key.shape[0]       = n;

            const char * compile_err = nullptr;
            auto * packet = ggml_hpx_compile_packet(
                &packet_fine_group, &packet_key, &compile_err);
            if (packet == nullptr)
            {
                std::fprintf(stderr,
                    "ABORT: compile_packet failed n=%" PRId64 " lanes=%d: %s\n",
                    n, n_lanes,
                    compile_err != nullptr ? compile_err : "(null)");
                return 1;
            }

            auto * packet_runtime = ggml_hpx_packet_runtime_create(
                static_cast<uint32_t>(n_lanes));

            const size_t frame_sz = ggml_hpx_packet_frame_size(packet);
            const size_t frame_al = ggml_hpx_packet_frame_align(packet);
            void * frame_raw = ::operator new(
                frame_sz, std::align_val_t{frame_al});
            auto * frame = static_cast<ggml_hpx_packet_frame *>(frame_raw);
            ggml_hpx_packet_frame_init(frame, packet);

            ggml_hpx_region_resources packet_resources{};
            packet_resources.shared_scratch   = nullptr;
            packet_resources.lane_scratch     = lane_ptrs;
            packet_resources.reduction_buffer = &packet_reduce_buf;
            packet_resources.n_lanes          = n_lanes;

            ggml_hpx_rms_norm_binding packet_binding{
                x.data(), dst_dag.data(), n, kEps};

            hpx::async([&]()
            {
                ggml_hpx_bind_rms_norm_packet(frame, &packet_binding);
                ggml_hpx_run_frozen_packet(
                    packet_runtime, packet, frame, &packet_resources);
            }).get();

            if (!verify_match(dst_ref.data(), dst_dag.data(), n, 1e-6f))
            {
                std::fprintf(stderr,
                    "ABORT: frozen_packet correctness failed n=%" PRId64
                    " lanes=%d\n",
                    n, n_lanes);
                return 1;
            }

            BenchStats packet_stats = bench_ns_in_hpx([&]()
            {
                ggml_hpx_bind_rms_norm_packet(frame, &packet_binding);
                ggml_hpx_run_frozen_packet(
                    packet_runtime, packet, frame, &packet_resources);
            }, kWarmup, kReps);

            print_csv_row("frozen_packet", n, n_lanes, packet_stats);

            ::operator delete(frame_raw, std::align_val_t{frame_al});
            ggml_hpx_free_packet(packet);
            ggml_hpx_packet_runtime_destroy(packet_runtime);
        }

        std::fflush(stdout);
    }

    std::fprintf(stderr, "# done.\n");
    return 0;
}
