// bench_hpx_run_range_mul_mat.cpp
//
// Microbenchmark comparing three dispatch paths on a direct F32 kernel.
// No path calls ggml_graph_compute or ggml_backend_graph_compute.
//
// Paths under test
// ─────────────────
// spawn      — std::thread per-dispatch (worst-case legacy; measures
//              spawn+work rather than steady-state overhead)
//
// persistent — condvar-based persistent worker pool; models the real
//              ggml_threadpool wakeup/work/barrier cycle on the same
//              kernel, without graph machinery
//
// hpx        — hpx::experimental::for_loop on a reused fork_join_executor;
//              workers are persistent spinning HPX threads
//
// Kernels under test
// ───────────────────
// matmul     — ggml_vec_dot_f32, partitioned over output columns
// rms_partial — per-lane partial sum-of-squares into lane_scratch
// rms_reduce  — single-thread aggregation of lane_scratch into
//               reduction_buffer (exercises ggml_hpx_region_resources)
//
// Region chain
// ─────────────
// A 3-region linear group (matmul → rms_partial → rms_reduce) is executed
// on both the persistent-pool and HPX paths.  Dependency edges are used
// by both executors to enforce ordering.  The HPX path uses a level-based
// scheduler that can run independent regions at the same level concurrently
// (demonstrated structurally; the linear chain has no independent regions).
//
// Work shapes : decode (rows=1) and prefill (rows=32), cols=out=4096.
// Thread counts : 1, 2, 4.
//
// Debug invariant
// ─────────────────
// g_hpx_run_range_calls is incremented by every HPX-path run_range
// invocation.  Caller checks the expected count after each segment;
// aborts on mismatch.

#ifndef GGML_HPX_REGION_DAG
#  error "bench_hpx_run_range_mul_mat.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-runtime.h"    // ggml_hpx_tpool_start

// HPX
#include <hpx/algorithm.hpp>
#include <hpx/async_base/async.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/execution.hpp>
#include <hpx/executors/fork_join_executor.hpp>
#include <hpx/future.hpp>

// std
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Forward-declare ggml_vec_dot_f32.  restrict omitted; ABI is identical.
// ---------------------------------------------------------------------------

extern "C" void ggml_vec_dot_f32(
    int n,
    float *       s,  size_t bs,
    const float * x,  size_t bx,
    const float * y,  size_t by,
    int nrc);

// ---------------------------------------------------------------------------
// HPX executor type alias — fork_join_executor.
//
// fork_join_executor keeps persistent spinning worker threads for the
// lifetime of the executor object, unlike scheduler_executor which posts
// tasks on demand.  Its constructor must be called from an HPX thread
// (it calls this_thread::get_pool() internally), so make_exec() is only
// safe inside an hpx::async lambda.
//
// make_exec builds a pu_mask covering the first nth logical PUs using
// the resource partitioner, then constructs Exec{mask}.  With static_
// schedule and nth iterations in for_loop, each worker handles exactly
// one iteration — identical workload distribution to the previous
// scheduler_executor path.
// ---------------------------------------------------------------------------

using Exec = hpx::execution::experimental::fork_join_executor;

// Build a PU mask for the first nth logical threads and return a
// fork_join_executor pinned to those PUs.
// Precondition: called from an HPX thread.
static Exec make_exec(int nth)
{
    auto const &            rp = hpx::resource::get_partitioner();
    hpx::threads::mask_type mask(hpx::threads::hardware_concurrency());
    for (int i = 0; i < nth; ++i)
        hpx::threads::set(mask, rp.get_pu_num(static_cast<std::size_t>(i)));
    return Exec{mask};
}

// ---------------------------------------------------------------------------
// Debug counter — incremented by every HPX-path run_range invocation.
// ---------------------------------------------------------------------------

static std::atomic<int64_t> g_hpx_run_range_calls{0};

// ---------------------------------------------------------------------------
// Op-specific context
// ---------------------------------------------------------------------------

struct ggml_hpx_mul_mat_ctx
{
    const float * x;        // input  [n_rows × n_cols]
    const float * w;        // weight [n_out  × n_cols]
    float *       y;        // output [n_rows × n_out]

    int64_t n_rows;
    int64_t n_cols;
    int64_t n_out;

    void * shared_scratch;  // per contract; unused in most bench paths
};

// ---------------------------------------------------------------------------
// Work-range partition helper
// ---------------------------------------------------------------------------

static void col_range_for(
    int64_t n_work, int ith, int nth,
    int64_t & begin, int64_t & end) noexcept
{
    begin = (n_work *  ith)      / nth;
    end   = (n_work * (ith + 1)) / nth;
}

// ---------------------------------------------------------------------------
// Core matmul kernel: columns [col_begin, col_end) for all rows.
// Called by all three dispatch paths (no path branch inside).
// ---------------------------------------------------------------------------

static void matmul_col_range(
    ggml_hpx_mul_mat_ctx const & ctx,
    int64_t col_begin,
    int64_t col_end) noexcept
{
    int const n = static_cast<int>(ctx.n_cols);
    for (int64_t r = 0; r < ctx.n_rows; ++r)
    {
        const float * xr = ctx.x + r * ctx.n_cols;
        float *       yr = ctx.y + r * ctx.n_out;
        for (int64_t c = col_begin; c < col_end; ++c)
        {
            ggml_vec_dot_f32(n,
                yr + c,                  /* s */ 0,
                ctx.w + c * ctx.n_cols,  /* x */ 0,
                xr,                      /* y */ 0,
                1);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel callbacks satisfying ggml_hpx_run_range_fn
// ---------------------------------------------------------------------------

// matmul: begin/end are output column indices.
static void mul_mat_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*res*/)
{
    g_hpx_run_range_calls.fetch_add(1, std::memory_order_relaxed);
    matmul_col_range(*static_cast<ggml_hpx_mul_mat_ctx *>(ctx_void), begin, end);
}

// rms_partial: accumulate sum-of-squares of y[:,begin:end] into
// res->lane_scratch[ith].  Called once per lane; no synchronisation inside.
static void rms_partial_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * res)
{
    g_hpx_run_range_calls.fetch_add(1, std::memory_order_relaxed);

    auto * ctx = static_cast<ggml_hpx_mul_mat_ctx *>(ctx_void);
    float  sum = 0.0f;

    for (int64_t r = 0; r < ctx->n_rows; ++r)
    {
        const float * yr = ctx->y + r * ctx->n_out;
        for (int64_t c = begin; c < end; ++c)
            sum += yr[c] * yr[c];
    }

    *static_cast<float *>(res->lane_scratch[ith]) = sum;
}

// rms_reduce: collect lane_scratch[0..n_lanes-1], write RMS to
// res->reduction_buffer[0].  Only ith == 0 executes; all other lanes return
// immediately (the for_loop still fires nth tasks, nth-1 are no-ops).
static void rms_reduce_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         /*nth*/,
    int64_t                     /*begin*/,
    int64_t                     /*end*/,
    ggml_hpx_region_resources * res)
{
    if (ith != 0) return;

    g_hpx_run_range_calls.fetch_add(1, std::memory_order_relaxed);

    auto *       ctx   = static_cast<ggml_hpx_mul_mat_ctx *>(ctx_void);
    float        total = 0.0f;

    for (int lane = 0; lane < res->n_lanes; ++lane)
        total += *static_cast<float *>(res->lane_scratch[lane]);

    float const n = static_cast<float>(ctx->n_rows * ctx->n_out);
    *static_cast<float *>(res->reduction_buffer) = std::sqrt(total / n);
}

// ---------------------------------------------------------------------------
// PersistentPool
//
// Persistent condvar-based worker pool modelling the ggml_threadpool
// wakeup/work/barrier cycle.  Workers 1..nth-1 sleep on cv_work_ and
// wake when epoch_ is incremented.  Worker 0 always runs inline in
// dispatch().  done_count_ counts completions of workers 1..nth-1 only;
// dispatch() returns when done_count_ == nth_ - 1.
//
// WorkFn signature matches ggml_hpx_run_range_fn so region callbacks
// can be passed directly.
// ---------------------------------------------------------------------------

class PersistentPool
{
public:
    using WorkFn = ggml_hpx_run_range_fn;   // void(*)(void*,int,int,int64_t,int64_t,res*)

    explicit PersistentPool(int nth) noexcept
        : nth_{nth}
    {
        workers_.reserve(static_cast<std::size_t>(std::max(0, nth - 1)));

        for (int i = 1; i < nth; ++i)
        {
            workers_.emplace_back([this, i]() noexcept
            {
                // Must match epoch_'s initial value (0) so the worker does
                // not immediately fire on the first condvar check before any
                // dispatch() has been called.
                int local_epoch = 0;

                for (;;)
                {
                    // --- capture work descriptor under lock ---
                    WorkFn                     fn    = nullptr;
                    void *                     ctx   = nullptr;
                    int64_t                    nwork = 0;
                    ggml_hpx_region_resources* res   = nullptr;
                    int                        nth   = 0;

                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_work_.wait(lk, [&]() noexcept {
                            return stop_ || epoch_ != local_epoch;
                        });
                        if (stop_) return;

                        local_epoch = epoch_;
                        fn    = work_.fn;
                        ctx   = work_.ctx;
                        nwork = work_.n_work;
                        res   = work_.res;
                        nth   = nth_;
                    }

                    // --- execute outside lock ---
                    int64_t b = 0;
                    int64_t e = 0;
                    col_range_for(nwork, i, nth, b, e);
                    fn(ctx, i, nth, b, e, res);

                    // --- signal completion ---
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        if (++done_count_ == nth_ - 1)
                            cv_done_.notify_one();
                    }
                }
            });
        }
    }

    ~PersistentPool() noexcept
    {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
            cv_work_.notify_all();
        }
        for (auto & t : workers_)
            t.join();
    }

    PersistentPool(PersistentPool const &)             = delete;
    PersistentPool & operator=(PersistentPool const &) = delete;

    // Dispatch fn across [0, n_work) split over nth workers.
    // Blocks until all workers complete.
    void dispatch(WorkFn fn, void * ctx, int64_t n_work,
                  ggml_hpx_region_resources * res) noexcept
    {
        if (nth_ == 1)
        {
            fn(ctx, 0, 1, 0, n_work, res);
            return;
        }

        {
            std::unique_lock<std::mutex> lk(mu_);
            work_       = {fn, ctx, n_work, res};
            done_count_ = 0;
            ++epoch_;
            cv_work_.notify_all();
        }

        // Worker 0 runs inline (lock NOT held)
        int64_t b = 0;
        int64_t e = 0;
        col_range_for(n_work, 0, nth_, b, e);
        fn(ctx, 0, nth_, b, e, res);

        // Wait for workers 1..nth-1
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&]() noexcept { return done_count_ == nth_ - 1; });
    }

    int n_threads() const noexcept { return nth_; }

private:
    struct WorkDesc
    {
        WorkFn                     fn     = nullptr;
        void *                     ctx    = nullptr;
        int64_t                    n_work = 0;
        ggml_hpx_region_resources* res    = nullptr;
    };

    int                     nth_;
    std::vector<std::thread> workers_;

    alignas(64) WorkDesc    work_{};
    std::mutex              mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    int                     epoch_      = 0;
    int                     done_count_ = 0;
    bool                    stop_       = false;
};

// ---------------------------------------------------------------------------
// Timing helper — returns median wall-clock time in microseconds.
// ---------------------------------------------------------------------------

template <typename Fn>
static double median_us(Fn fn, int n_warmup, int n_bench)
{
    for (int i = 0; i < n_warmup; ++i)
        fn();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(n_bench));

    for (int i = 0; i < n_bench; ++i)
    {
        auto const t0 = std::chrono::steady_clock::now();
        fn();
        auto const t1 = std::chrono::steady_clock::now();
        using Micros = std::chrono::duration<double, std::micro>;
        samples.push_back(Micros{t1 - t0}.count());
    }

    std::sort(samples.begin(), samples.end());
    return samples[static_cast<std::size_t>(n_bench) / 2];
}

// ---------------------------------------------------------------------------
// Aligned float buffer
// ---------------------------------------------------------------------------

struct AlignedBuf
{
    explicit AlignedBuf(int64_t n_floats) noexcept
        : data_{nullptr}
        , n_{n_floats}
    {
        std::size_t const bytes = static_cast<std::size_t>(n_floats) * sizeof(float);
        std::size_t const abytes = (bytes + 63u) & ~std::size_t{63u};
        data_ = static_cast<float *>(std::aligned_alloc(64, abytes));
    }

    ~AlignedBuf() noexcept { std::free(data_); }

    AlignedBuf(AlignedBuf const &)             = delete;
    AlignedBuf & operator=(AlignedBuf const &) = delete;

    float *       data() noexcept       { return data_; }
    const float * data() const noexcept { return data_; }
    int64_t       size() const noexcept { return n_; }

private:
    float * data_;
    int64_t n_;
};

static void fill_random(float * buf, int64_t n, uint32_t seed = 0) noexcept
{
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    for (int64_t i = 0; i < n; ++i)
        buf[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// Shape descriptor
// ---------------------------------------------------------------------------

struct Shape
{
    int64_t      n_rows;
    int64_t      n_cols;
    int64_t      n_out;
    const char * label;
};

// ---------------------------------------------------------------------------
// Region group executors
//
// Both executors respect dependency edges via level-based topological
// ordering.  Regions at the same level have no mutual dependencies and may
// run concurrently.
//
//   run_region_group_persistent — runs each level sequentially through the
//     persistent pool; independent regions at the same level also run
//     sequentially (the pool has one dispatch channel).
//
//   run_region_group_hpx — runs each level via for_loop (single region) or
//     hpx::async + hpx::wait_all (multiple independent regions at the same
//     level, enabling genuine inter-region parallelism).
// ---------------------------------------------------------------------------

// Dispatch one region synchronously on the HPX executor.
// REDUCTION regions are single-threaded (only ith=0 is meaningful anyway).
static void dispatch_region_sync(
    ggml_hpx_cpu_region const &  r,
    int                          nth,
    Exec &                       exec,
    ggml_hpx_region_resources &  res)
{
    if (r.kind == GGML_HPX_CPU_REGION_KIND_REDUCTION || nth == 1)
    {
        r.run_range(r.ctx, 0, 1, r.begin, r.end, &res);
        return;
    }

    hpx::experimental::for_loop(
        hpx::execution::par.on(exec), 0, nth,
        [&r, nth, &res](int j)
        {
            int64_t b = 0;
            int64_t e = 0;
            col_range_for(r.end - r.begin, j, nth, b, e);
            r.run_range(r.ctx, j, nth, r.begin + b, r.begin + e, &res);
        });
}

// Compute topological levels for each region given the dep edges.
// level[i] = longest path from any root to region i.
static std::vector<int> compute_levels(
    ggml_hpx_cpu_region_group const & group)
{
    int const n = group.n_regions;
    std::vector<int> level(static_cast<std::size_t>(n), 0);

    for (int e = 0; e < group.n_deps; ++e)
    {
        int const src = group.deps[e].src;
        int const dst = group.deps[e].dst;
        if (level[dst] <= level[src])
            level[dst] = level[src] + 1;
    }

    return level;
}

static void run_region_group_persistent(
    ggml_hpx_cpu_region_group const & group,
    PersistentPool &                  pool,
    ggml_hpx_region_resources &       res)
{
    auto const level    = compute_levels(group);
    int const  n        = group.n_regions;
    int const  max_lv   = *std::max_element(level.begin(), level.end());
    int const  nth      = pool.n_threads();

    for (int lv = 0; lv <= max_lv; ++lv)
    {
        for (int i = 0; i < n; ++i)
        {
            if (level[static_cast<std::size_t>(i)] != lv) continue;

            auto & r = group.regions[i];

            // REDUCTION or single-threaded: run inline, bypass pool.
            if (r.kind == GGML_HPX_CPU_REGION_KIND_REDUCTION || nth == 1)
            {
                r.run_range(r.ctx, 0, 1, r.begin, r.end, &res);
            }
            else
            {
                // n_work = r.end - r.begin; begin offsets are all 0 in bench.
                pool.dispatch(r.run_range, r.ctx, r.end - r.begin, &res);
            }
        }
        // Pool dispatch() returns only after all workers complete:
        // the return IS the synchronisation boundary between levels.
    }
}

static void run_region_group_hpx(
    ggml_hpx_cpu_region_group const & group,
    int                               nth,
    Exec &                            exec,
    ggml_hpx_region_resources &       res)
{
    auto const level  = compute_levels(group);
    int const  n      = group.n_regions;
    int const  max_lv = *std::max_element(level.begin(), level.end());

    for (int lv = 0; lv <= max_lv; ++lv)
    {
        // Collect region indices at this level.
        std::vector<int> at_lv;
        for (int i = 0; i < n; ++i)
            if (level[static_cast<std::size_t>(i)] == lv)
                at_lv.push_back(i);

        if (at_lv.size() == 1)
        {
            // Single region: dispatch synchronously on the calling thread.
            dispatch_region_sync(group.regions[at_lv[0]], nth, exec, res);
        }
        else
        {
            // Multiple independent regions: launch concurrently via hpx::async.
            // Each async task calls dispatch_region_sync (which itself uses
            // for_loop internally), so HPX threads run intra-region work too.
            std::vector<hpx::future<void>> level_futs;
            level_futs.reserve(at_lv.size());

            for (int i : at_lv)
            {
                auto & ri = group.regions[i];
                level_futs.push_back(
                    hpx::async([&ri, nth, &exec, &res]()
                    {
                        dispatch_region_sync(ri, nth, exec, res);
                    }));
            }

            // hpx::wait_all blocks this thread until all level futures complete.
            // This is the dependency barrier between levels.
            hpx::wait_all(level_futs);
        }
    }
}

// ---------------------------------------------------------------------------
// bench 1: three-way dispatch comparison for matmul only.
// Compares: spawn / persistent-pool / HPX for_loop.
// ---------------------------------------------------------------------------

static void run_comparison(
    Shape const &    shape,
    int              nth,
    const float *    x,
    const float *    w,
    float *          y,
    PersistentPool & pool,
    Exec &           exec,
    int              n_warmup,
    int              n_bench)
{
    auto reset_y = [&]() noexcept
    {
        std::memset(y, 0,
            static_cast<std::size_t>(shape.n_rows * shape.n_out) * sizeof(float));
    };

    ggml_hpx_mul_mat_ctx ctx{};
    ctx.x      = x;
    ctx.w      = w;
    ctx.y      = y;
    ctx.n_rows = shape.n_rows;
    ctx.n_cols = shape.n_cols;
    ctx.n_out  = shape.n_out;

    ggml_hpx_region_resources resources{};
    resources.n_lanes = nth;
    // lane_scratch / reduction_buffer intentionally null: matmul does not use them.

    // ── spawn: one std::thread per worker ─────────────────────────────────

    reset_y();
    double const spawn_us = median_us([&]()
    {
        if (nth == 1)
        {
            matmul_col_range(ctx, 0, shape.n_out);
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(nth - 1));
        for (int j = 1; j < nth; ++j)
        {
            workers.emplace_back([&, j]()
            {
                int64_t b = 0; int64_t e = 0;
                col_range_for(shape.n_out, j, nth, b, e);
                matmul_col_range(ctx, b, e);
            });
        }
        int64_t b = 0; int64_t e = 0;
        col_range_for(shape.n_out, 0, nth, b, e);
        matmul_col_range(ctx, b, e);
        for (auto & t : workers) t.join();
    },
    n_warmup, n_bench);

    // ── persistent pool ────────────────────────────────────────────────────

    reset_y();
    double const pool_us = median_us([&]()
    {
        pool.dispatch(mul_mat_run_range, &ctx, shape.n_out, &resources);
    },
    n_warmup, n_bench);

    // ── HPX for_loop ───────────────────────────────────────────────────────

    reset_y();
    int64_t const hpx_calls_before =
        g_hpx_run_range_calls.load(std::memory_order_relaxed);

    double const hpx_us = median_us([&]()
    {
        if (nth == 1)
        {
            mul_mat_run_range(&ctx, 0, 1, 0, shape.n_out, &resources);
            return;
        }
        hpx::experimental::for_loop(
            hpx::execution::par.on(exec), 0, nth,
            [&](int j)
            {
                int64_t b = 0; int64_t e = 0;
                col_range_for(shape.n_out, j, nth, b, e);
                mul_mat_run_range(&ctx, j, nth, b, e, &resources);
            });
    },
    n_warmup, n_bench);

    // ── debug assertion ───────────────────────────────────────────────────

    int64_t const hpx_calls_after =
        g_hpx_run_range_calls.load(std::memory_order_relaxed);
    int64_t const expected =
        static_cast<int64_t>(n_warmup + n_bench) * (nth == 1 ? 1 : nth);
    int64_t const actual = hpx_calls_after - hpx_calls_before;

    if (actual != expected)
    {
        std::fprintf(stderr,
            "  ASSERTION FAILED [%s nth=%d]: "
            "expected %" PRId64 " run_range calls, got %" PRId64 "\n",
            shape.label, nth, expected, actual);
        std::abort();
    }

    // ── output ────────────────────────────────────────────────────────────

    std::printf(
        "  %-8s  nth=%d  spawn=%8.2f us  pool=%8.2f us  "
        "hpx=%8.2f us  [pool/spawn=%.2fx  hpx/pool=%.2fx]\n",
        shape.label, nth,
        spawn_us, pool_us, hpx_us,
        spawn_us / pool_us,
        pool_us  / hpx_us);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// bench 2: region chain exercising ggml_hpx_region_resources.
//
// 3-region linear DAG: matmul → rms_partial → rms_reduce
//   region 0: compute y = x @ W           (MATMUL,    parallel)
//   region 1: per-lane partial sum-sq(y)   (ELEMENTWISE, parallel, writes lane_scratch)
//   region 2: aggregate partials → RMS     (REDUCTION, single-threaded, writes reduction_buffer)
//
// Both persistent-pool and HPX paths run the same group and must produce
// the same RMS value.  A mismatch aborts.
// ---------------------------------------------------------------------------

static void bench_resources_and_chain(
    Shape const &    shape,
    int              nth,
    const float *    x,
    const float *    w,
    float *          y,
    PersistentPool & pool,
    Exec &           exec,
    int              n_warmup,
    int              n_bench)
{
    // Per-lane scratch: one float per worker.
    // Stack allocation is fine; bench nth ≤ 8.
    static constexpr int kMaxNth = 8;
    assert(nth <= kMaxNth);

    std::array<float, kMaxNth>  lane_scalars{};
    std::array<void *,  kMaxNth> lane_ptrs{};
    for (int i = 0; i < nth; ++i)
        lane_ptrs[static_cast<std::size_t>(i)] =
            &lane_scalars[static_cast<std::size_t>(i)];

    float rms_result = 0.0f;

    ggml_hpx_region_resources res{};
    res.lane_scratch     = lane_ptrs.data();
    res.reduction_buffer = &rms_result;
    res.n_lanes          = nth;

    ggml_hpx_mul_mat_ctx ctx{};
    ctx.x      = x;
    ctx.w      = w;
    ctx.y      = y;
    ctx.n_rows = shape.n_rows;
    ctx.n_cols = shape.n_cols;
    ctx.n_out  = shape.n_out;

    // 3-region chain: begin=0 for all regions (column-indexed).
    ggml_hpx_cpu_region regions[3]{};
    regions[0].kind      = GGML_HPX_CPU_REGION_KIND_MATMUL;
    regions[0].begin     = 0;
    regions[0].end       = shape.n_out;
    regions[0].ctx       = &ctx;
    regions[0].run_range = mul_mat_run_range;

    regions[1].kind      = GGML_HPX_CPU_REGION_KIND_ELEMENTWISE;
    regions[1].begin     = 0;
    regions[1].end       = shape.n_out;
    regions[1].ctx       = &ctx;
    regions[1].run_range = rms_partial_run_range;

    regions[2].kind      = GGML_HPX_CPU_REGION_KIND_REDUCTION;
    regions[2].begin     = 0;
    regions[2].end       = shape.n_out;
    regions[2].ctx       = &ctx;
    regions[2].run_range = rms_reduce_run_range;

    // deps: 0 → 1 → 2
    ggml_hpx_dep_edge deps[2] = {{0, 1}, {1, 2}};

    ggml_hpx_cpu_region_group group{};
    group.regions   = regions;
    group.n_regions = 3;
    group.deps      = deps;
    group.n_deps    = 2;

    auto reset_y = [&]() noexcept
    {
        std::memset(y, 0,
            static_cast<std::size_t>(shape.n_rows * shape.n_out) * sizeof(float));
    };

    // ── persistent pool ────────────────────────────────────────────────────

    reset_y();
    double const pool_us = median_us([&]()
    {
        run_region_group_persistent(group, pool, res);
    },
    n_warmup, n_bench);

    float const pool_rms = rms_result;

    // ── HPX ────────────────────────────────────────────────────────────────

    reset_y();
    double const hpx_us = median_us([&]()
    {
        run_region_group_hpx(group, nth, exec, res);
    },
    n_warmup, n_bench);

    float const hpx_rms = rms_result;

    // ── verify results agree (within 0.1% relative tolerance) ─────────────

    float const rms_diff = std::abs(pool_rms - hpx_rms);
    float const rms_tol  = 1e-3f * (pool_rms > 0.0f ? pool_rms : 1.0f);
    bool const  ok       = rms_diff <= rms_tol;

    std::printf(
        "  %-8s  nth=%d  [chain: matmul→rms_partial→rms_reduce]\n"
        "    pool=%8.2f us  hpx=%8.2f us  speedup=%.2fx\n"
        "    rms: pool=%.6f  hpx=%.6f  %s\n",
        shape.label, nth,
        pool_us, hpx_us, pool_us / hpx_us,
        static_cast<double>(pool_rms),
        static_cast<double>(hpx_rms),
        ok ? "OK" : "MISMATCH");
    std::fflush(stdout);

    if (!ok)
    {
        std::fprintf(stderr,
            "  RMS mismatch: pool=%.8f hpx=%.8f diff=%.3e\n",
            static_cast<double>(pool_rms),
            static_cast<double>(hpx_rms),
            static_cast<double>(rms_diff));
        std::abort();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    // Start the HPX runtime before allocating buffers.
    // fork_join_executor ctor calls this_thread::get_pool() and must run
    // from an HPX thread, so the bench body is wrapped in hpx::async().get().
    ggml_hpx_tpool_start();

    static constexpr int N_WARMUP = 20;
    static constexpr int N_BENCH  = 100;

    static constexpr Shape kShapes[] = {
        {  1, 4096, 4096, "decode"  },
        { 32, 4096, 4096, "prefill" },
    };
    static constexpr int kNths[] = {1, 2, 4};

    // Allocate matrices once; reused across all (shape, nth) combinations.
    AlignedBuf x_buf{32 * 4096};     //  32 × 4096 floats
    AlignedBuf w_buf{4096 * 4096};   // 4096 × 4096 floats (~64 MB)
    AlignedBuf y_buf{32 * 4096};     //  32 × 4096 floats

    if (!x_buf.data() || !w_buf.data() || !y_buf.data())
    {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_random(x_buf.data(), x_buf.size(), /*seed=*/1);
    fill_random(w_buf.data(), w_buf.size(), /*seed=*/2);

    std::printf(
        "bench_hpx_run_range_mul_mat\n"
        "  warmup=%d  bench=%d  kernel=ggml_vec_dot_f32 (F32)\n"
        "  spawn      = std::thread per-dispatch\n"
        "  persistent = condvar pool (models ggml_threadpool baseline)\n"
        "  hpx        = for_loop on persistent fork_join_executor\n\n",
        N_WARMUP, N_BENCH);

    // Run all benchmarks from inside an HPX thread so that
    // fork_join_executor constructors (which call this_thread::get_pool())
    // are satisfied.
    hpx::async([&]()
    {
        // ── bench 1: three-way dispatch comparison ────────────────────────

        std::printf("=== bench 1: matmul dispatch overhead ===\n");

        for (Shape const & shape : kShapes)
        {
            std::printf(
                "[%s  rows=%" PRId64 " cols=%" PRId64 " out=%" PRId64 "]\n",
                shape.label, shape.n_rows, shape.n_cols, shape.n_out);

            for (int nth : kNths)
            {
                Exec           exec = make_exec(nth);
                PersistentPool pool{nth};

                run_comparison(
                    shape, nth,
                    x_buf.data(), w_buf.data(), y_buf.data(),
                    pool, exec,
                    N_WARMUP, N_BENCH);
            }
            std::printf("\n");
        }

        // ── bench 2: region chain with resources ──────────────────────────

        std::printf(
            "=== bench 2: 3-region chain "
            "(matmul → rms_partial → rms_reduce) ===\n");

        for (Shape const & shape : kShapes)
        {
            std::printf(
                "[%s  rows=%" PRId64 " cols=%" PRId64 " out=%" PRId64 "]\n",
                shape.label, shape.n_rows, shape.n_cols, shape.n_out);

            for (int nth : kNths)
            {
                Exec           exec = make_exec(nth);
                PersistentPool pool{nth};

                bench_resources_and_chain(
                    shape, nth,
                    x_buf.data(), w_buf.data(), y_buf.data(),
                    pool, exec,
                    N_WARMUP, N_BENCH);
            }
            std::printf("\n");
        }

        std::printf("debug: total g_hpx_run_range_calls = %" PRId64 "\n",
            g_hpx_run_range_calls.load(std::memory_order_relaxed));

    }).get();

    return 0;
}
