// bench_run_job_overhead.cpp
//
// Microbenchmark: raw dispatch round-trip overhead for three variants:
//   pth  — pthread persistent-worker (N-1 threads + main as worker 0)
//   hpx  — scheduler_executor<thread_pool_scheduler> with
//           with_processing_units_count(N)  [production executor]
//   rpe  — restricted_thread_pool_executor(first=0, num=N)
//           [pinned OS-thread range, no topology delegation]
//
// at n_threads = {1, 2, 4}, bodies = {noop, tiny-touch}.
//
// ── Throughput section ──────────────────────────────────────────────────────
// Timed batches of BATCH_SIZE dispatches; one clock pair per batch.
// Stats (mean / median / p95 / p99 / max) are across BATCHES batch samples,
// each expressed as ns-per-dispatch = batch_ns / BATCH_SIZE.
//
// ── Breakdown section ───────────────────────────────────────────────────────
// For HPX (hpx + rpe) at n_threads = {2, 4}, BREAKDOWN_REPS individual
// dispatches are measured with per-worker entry timestamps.  From each
// dispatch:
//   submit  → first-start  (scheduler pickup latency for the first task)
//   submit  → last-start   (worst-case straggler delivery)
//   spread  = last-start − first-start
//   done    = for_loop return − submit  (total round-trip)
// Stats: median / p95 / p99 / max per component.
//
// ── Tail histogram ──────────────────────────────────────────────────────────
// Bucket distribution for total round-trip ns from the breakdown runs:
//   <5µs / <20µs / <50µs / <100µs / <300µs / <1ms / ≥1ms
//
// ── Bodies ──────────────────────────────────────────────────────────────────
//   noop       — relaxed atomic store per worker slot; compiler-safe, no
//                inter-thread memory traffic
//   tiny-touch — volatile write per worker slot + release fence; one dirty
//                cache line per worker, no contention between workers
//
// ── Isolated-pool note ──────────────────────────────────────────────────────
// rpe pins dispatch to OS threads [0, N) via restricted_policy_executor.
// It does not share the pool with HPX's own scheduler tasks during the bench.

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/executors/restricted_thread_pool_executor.hpp>
#include <hpx/executors/scheduler_executor.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/post.hpp>

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tuning knobs
// ---------------------------------------------------------------------------

static constexpr int WARM_DISPATCHES  = 500;
static constexpr int BATCHES          = 200;
static constexpr int BATCH_SIZE       = 500;
static constexpr int BREAKDOWN_REPS   = 10000;
static constexpr int BREAKDOWN_WARM   = 200;

// ---------------------------------------------------------------------------
// Scheduler label
// Set by --sched-label=X on the command line.  The flag is stripped from
// argv before hpx::start so HPX does not reject it as an unknown option.
// The actual scheduler HPX loaded is read back after start via
// hpx::get_config_entry.
// ---------------------------------------------------------------------------

static std::string g_sched_label{"(default)"};
static std::string g_hpx_queuing{"(default)"};

// Scan argv for our custom flags.
//   --sched-label=X  — human label printed in the header; stripped so HPX
//                      does not reject it as an unknown option.
//   --hpx:queuing=X  — left in argv for HPX; value is captured here so it
//                      appears verbatim in the output header.
static void extract_bench_args(int & argc, char ** argv)
{
    static char const label_prefix[]   = "--sched-label=";
    static char const queuing_prefix[] = "--hpx:queuing=";

    for (int i = 1; i < argc; )
    {
        if (std::strncmp(argv[i], label_prefix, sizeof(label_prefix) - 1) == 0)
        {
            g_sched_label = argv[i] + sizeof(label_prefix) - 1;
            for (int j = i; j < argc - 1; ++j)
                argv[j] = argv[j + 1];
            --argc;
            // Do not increment i; the next element has shifted into position i.
        }
        else
        {
            if (std::strncmp(
                    argv[i], queuing_prefix, sizeof(queuing_prefix) - 1) == 0)
                g_hpx_queuing = argv[i] + sizeof(queuing_prefix) - 1;
            ++i;
        }
    }
}

// ---------------------------------------------------------------------------
// Portable reusable barrier (pthread_barrier_t absent on macOS)
// ---------------------------------------------------------------------------

struct Barrier
{
    explicit Barrier(int count) noexcept
      : n_(count), waiting_(0), generation_(0)
    {
    }

    void wait()
    {
        std::unique_lock<std::mutex> lk(m_);
        int const gen = generation_;
        if (++waiting_ == n_)
        {
            waiting_ = 0;
            ++generation_;
            cv_.notify_all();
        }
        else
        {
            cv_.wait(lk, [&] { return generation_ != gen; });
        }
    }

private:
    std::mutex              m_;
    std::condition_variable cv_;
    int                     n_;
    int                     waiting_;
    int                     generation_;
};

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

struct alignas(64) AtomicSlot   { std::atomic<int>    val{0}; };
struct alignas(64) VolatileSlot { volatile std::int64_t val = 0; };

static AtomicSlot   g_atomic_slots[8];
static VolatileSlot g_volatile_slots[8];

static void body_noop(int j)
{
    g_atomic_slots[static_cast<std::size_t>(j)].val.store(
        j, std::memory_order_relaxed);
}

static void body_tiny_touch(int j)
{
    g_volatile_slots[static_cast<std::size_t>(j)].val = j;
    std::atomic_thread_fence(std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

static double mean_of(std::vector<double> const & v)
{
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
}

static double percentile_of(std::vector<double> v, double pct)
{
    std::sort(v.begin(), v.end());
    std::size_t idx =
        static_cast<std::size_t>(static_cast<double>(v.size()) * pct);
    if (idx >= v.size())
        idx = v.size() - 1;
    return v[idx];
}

static double max_of(std::vector<double> const & v)
{
    return *std::max_element(v.begin(), v.end());
}

// ---------------------------------------------------------------------------
// Pthread persistent-worker pool
// ---------------------------------------------------------------------------

struct PthreadPool
{
    int               n;
    Barrier           b_start;
    Barrier           b_done;
    std::atomic<bool> stop{false};
    void (* body)(int) = nullptr;

    std::vector<pthread_t> threads;

    static PthreadPool * g_pool;

    static void * worker_fn(void * arg)
    {
        int const     id = static_cast<int>(reinterpret_cast<intptr_t>(arg));
        PthreadPool * p  = g_pool;
        while (true)
        {
            p->b_start.wait();
            if (p->stop.load(std::memory_order_relaxed))
                break;
            p->body(id);
            p->b_done.wait();
        }
        return nullptr;
    }

    explicit PthreadPool(int n_threads)
      : n(n_threads)
      , b_start(n_threads)
      , b_done(n_threads)
      , threads(static_cast<std::size_t>(n_threads - 1))
    {
        g_pool = this;
        for (int i = 1; i < n_threads; ++i)
            pthread_create(
                &threads[static_cast<std::size_t>(i) - 1], nullptr,
                worker_fn,
                reinterpret_cast<void *>(static_cast<intptr_t>(i)));
    }

    ~PthreadPool()
    {
        stop.store(true, std::memory_order_relaxed);
        b_start.wait();
        for (auto & t : threads)
            pthread_join(t, nullptr);
    }

    PthreadPool(PthreadPool const &)             = delete;
    PthreadPool & operator=(PthreadPool const &) = delete;

    void dispatch(void (* fn)(int))
    {
        body = fn;
        b_start.wait();
        fn(0);
        b_done.wait();
    }
};

PthreadPool * PthreadPool::g_pool = nullptr;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
using NS    = std::chrono::nanoseconds;

static std::int64_t now_ns()
{
    return std::chrono::duration_cast<NS>(
        Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Throughput section
// ---------------------------------------------------------------------------

static void print_throughput_row(
    char const *        variant,
    int                 n_threads,
    char const *        body_name,
    std::vector<double> batch_ns)
{
    long const iters = static_cast<long>(BATCHES) * BATCH_SIZE;
    std::printf(
        "%-5s  t=%-2d  %-10s  iters=%7ld"
        "  mean=%8.1f  p50=%8.1f  p95=%8.1f  p99=%8.1f  max=%9.1f  ns\n",
        variant, n_threads, body_name, iters,
        mean_of(batch_ns),
        percentile_of(batch_ns, 0.50),
        percentile_of(batch_ns, 0.95),
        percentile_of(batch_ns, 0.99),
        max_of(batch_ns));
}

// Returns BATCHES values of ns-per-dispatch.
template <typename DispatchFn>
static std::vector<double> measure_throughput(
    int warm, DispatchFn dispatch)
{
    for (int i = 0; i < warm; ++i)
        dispatch();

    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(BATCHES));
    for (int b = 0; b < BATCHES; ++b)
    {
        std::int64_t const t0 = now_ns();
        for (int i = 0; i < BATCH_SIZE; ++i)
            dispatch();
        std::int64_t const t1 = now_ns();
        out.push_back(
            static_cast<double>(t1 - t0) /
            static_cast<double>(BATCH_SIZE));
    }
    return out;
}

static void bench_throughput_pthread(int n_threads, void (* body)(int))
{
    PthreadPool pool(n_threads);
    auto        samples =
        measure_throughput(WARM_DISPATCHES, [&] { pool.dispatch(body); });
    print_throughput_row("pth", n_threads,
        (body == body_noop) ? "noop" : "tiny-touch", samples);
}

// ---------------------------------------------------------------------------
// Breakdown section
// ---------------------------------------------------------------------------

// Per-dispatch breakdown record.
struct BreakdownSample
{
    double submit_to_first_ns;    // t_submit → min(t_enter[j])
    double submit_to_last_ns;     // t_submit → max(t_enter[j])
    double spread_ns;             // max(t_enter[j]) - min(t_enter[j])
    double total_ns;              // t_submit → t_done
};

// Shared per-dispatch entry-timestamp array.  Written by workers with
// relaxed store; read by main after for_loop returns (sequenced-after).
struct alignas(64) EntrySlot { std::atomic<std::int64_t> ts{0}; };
static EntrySlot g_enter[8];

static void body_noop_timed(int j)
{
    g_enter[static_cast<std::size_t>(j)].ts.store(
        now_ns(), std::memory_order_relaxed);
    g_atomic_slots[static_cast<std::size_t>(j)].val.store(
        j, std::memory_order_relaxed);
}

static void body_tiny_touch_timed(int j)
{
    g_enter[static_cast<std::size_t>(j)].ts.store(
        now_ns(), std::memory_order_relaxed);
    g_volatile_slots[static_cast<std::size_t>(j)].val = j;
    std::atomic_thread_fence(std::memory_order_release);
}

template <typename DispatchFn>
static std::vector<BreakdownSample> measure_breakdown(
    int n_threads, int warm, int reps, DispatchFn dispatch)
{
    for (int i = 0; i < warm; ++i)
        dispatch();

    std::vector<BreakdownSample> out;
    out.reserve(static_cast<std::size_t>(reps));

    for (int i = 0; i < reps; ++i)
    {
        // Reset entry timestamps.
        for (int j = 0; j < n_threads; ++j)
            g_enter[static_cast<std::size_t>(j)].ts.store(
                0, std::memory_order_relaxed);

        std::int64_t const t_submit = now_ns();
        dispatch();
        std::int64_t const t_done = now_ns();

        std::int64_t first = std::numeric_limits<std::int64_t>::max();
        std::int64_t last  = 0;
        for (int j = 0; j < n_threads; ++j)
        {
            std::int64_t const e =
                g_enter[static_cast<std::size_t>(j)].ts.load(
                    std::memory_order_relaxed);
            if (e < first) first = e;
            if (e > last)  last  = e;
        }

        BreakdownSample s{};
        s.submit_to_first_ns =
            static_cast<double>(first - t_submit);
        s.submit_to_last_ns  =
            static_cast<double>(last  - t_submit);
        s.spread_ns          =
            static_cast<double>(last  - first);
        s.total_ns           =
            static_cast<double>(t_done - t_submit);
        out.push_back(s);
    }
    return out;
}

static void print_breakdown(
    char const *                        variant,
    int                                 n_threads,
    char const *                        body_name,
    std::vector<BreakdownSample> const & v)
{
    auto col = [&](auto proj) {
        std::vector<double> tmp;
        tmp.reserve(v.size());
        for (auto const & s : v)
            tmp.push_back(proj(s));
        return tmp;
    };

    auto print_stat = [&](char const * comp, auto proj) {
        auto vals = col(proj);
        std::printf(
            "  %-24s  p50=%8.1f  p95=%8.1f  p99=%8.1f  max=%9.1f  ns\n",
            comp,
            percentile_of(vals, 0.50),
            percentile_of(vals, 0.95),
            percentile_of(vals, 0.99),
            max_of(vals));
    };

    std::printf("[breakdown] %-5s t=%d %-10s reps=%d  sched=%s\n",
        variant, n_threads, body_name, BREAKDOWN_REPS,
        g_sched_label.c_str());
    print_stat("submit→first-start",
        [](BreakdownSample const & s) { return s.submit_to_first_ns; });
    print_stat("submit→last-start",
        [](BreakdownSample const & s) { return s.submit_to_last_ns; });
    print_stat("spread (first→last)",
        [](BreakdownSample const & s) { return s.spread_ns; });
    print_stat("total (submit→done)",
        [](BreakdownSample const & s) { return s.total_ns; });

    // Tail histogram on total_ns.
    static constexpr double kBuckets[] = {
        5e3, 20e3, 50e3, 100e3, 300e3, 1000e3
    };
    static constexpr char const * kLabels[] = {
        "<5µs", "<20µs", "<50µs", "<100µs", "<300µs", "<1ms", "≥1ms"
    };
    constexpr int kNBuckets = 7;
    int counts[kNBuckets] = {};
    for (auto const & s : v)
    {
        int b = kNBuckets - 1;
        for (int k = 0; k < kNBuckets - 1; ++k)
        {
            if (s.total_ns < kBuckets[k]) { b = k; break; }
        }
        counts[b]++;
    }
    std::printf("  histogram (total):");
    for (int k = 0; k < kNBuckets; ++k)
        std::printf("  %s=%d", kLabels[k], counts[k]);
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// HPX executor types
// ---------------------------------------------------------------------------

using Scheduler =
    hpx::execution::experimental::thread_pool_scheduler;
using SchedExec =
    hpx::execution::experimental::scheduler_executor<Scheduler>;

using RestrictedExec =
    hpx::parallel::execution::restricted_thread_pool_executor;

// Build a scheduler_executor identical to HpxTpoolState in ggml-hpx-tpool.cpp.
static SchedExec make_sched_exec(int n_threads)
{
    return SchedExec{hpx::parallel::execution::with_processing_units_count(
        Scheduler{}, static_cast<std::size_t>(n_threads))};
}

// Build a restricted_thread_pool_executor pinned to OS threads [0, N).
static RestrictedExec make_restricted_exec(int n_threads)
{
    return RestrictedExec{
        /*first_thread=*/0,
        /*num_threads=*/static_cast<std::size_t>(n_threads)};
}

template <typename Exec>
static void bench_throughput_hpx_impl(
    char const * variant, int n_threads, void (* body)(int), Exec & exec)
{
    auto dispatch = [&] {
        if (n_threads == 1)
            body(0);
        else
            hpx::experimental::for_loop(
                hpx::execution::par.on(exec), 0, n_threads,
                [=](int j) { body(j); });
    };
    auto samples = measure_throughput(WARM_DISPATCHES, dispatch);
    print_throughput_row(variant, n_threads,
        (body == body_noop) ? "noop" : "tiny-touch", samples);
}

template <typename Exec>
static void bench_breakdown_hpx_impl(
    char const * variant, int n_threads, void (* timed_body)(int), Exec & exec)
{
    auto dispatch = [&] {
        if (n_threads == 1)
            timed_body(0);
        else
            hpx::experimental::for_loop(
                hpx::execution::par.on(exec), 0, n_threads,
                [=](int j) { timed_body(j); });
    };
    char const * bname =
        (timed_body == body_noop_timed) ? "noop" : "tiny-touch";
    auto samples =
        measure_breakdown(n_threads, BREAKDOWN_WARM, BREAKDOWN_REPS, dispatch);
    print_breakdown(variant, n_threads, bname, samples);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char ** argv)
{
    // Strip --sched-label=X and capture --hpx:queuing=X before HPX sees argv.
    extract_bench_args(argc, argv);

    // Pass the (now-clean) argv so --hpx:threads, --hpx:queuing, etc. work.
    hpx::start(nullptr, argc, argv);

    long const total = static_cast<long>(BATCHES) * BATCH_SIZE;
    std::printf(
        "bench_run_job_overhead  "
        "warm=%d  batches=%d  batch_size=%d  total_per_cell=%ld\n"
        "breakdown_warm=%d  breakdown_reps=%d\n"
        "sched_label=%s  hpx:queuing=%s\n\n",
        WARM_DISPATCHES, BATCHES, BATCH_SIZE, total,
        BREAKDOWN_WARM, BREAKDOWN_REPS,
        g_sched_label.c_str(), g_hpx_queuing.c_str());

    // ── Throughput ─────────────────────────────────────────────────────────
    std::printf("=== throughput (ns/dispatch) ===\n");
    std::printf("%-5s  %-4s  %-10s  %15s"
                "  %10s  %10s  %10s  %10s  %10s\n",
        "var", "t", "body", "iters",
        "mean", "p50", "p95", "p99", "max");
    std::printf("%s\n", std::string(90, '-').c_str());

    for (int t : {1, 2, 4})
    {
        bench_throughput_pthread(t, body_noop);
        bench_throughput_pthread(t, body_tiny_touch);
    }
    std::printf("\n");

    for (int t : {1, 2, 4})
    {
        auto exec = make_sched_exec(t);
        bench_throughput_hpx_impl("hpx", t, body_noop,       exec);
        bench_throughput_hpx_impl("hpx", t, body_tiny_touch, exec);
    }
    std::printf("\n");

    for (int t : {1, 2, 4})
    {
        auto exec = make_restricted_exec(t);
        bench_throughput_hpx_impl("rpe", t, body_noop,       exec);
        bench_throughput_hpx_impl("rpe", t, body_tiny_touch, exec);
    }

    // ── Breakdown ──────────────────────────────────────────────────────────
    std::printf("\n=== breakdown (ns per component, n_threads >= 2 only) ===\n\n");

    for (int t : {2, 4})
    {
        auto exec = make_sched_exec(t);
        bench_breakdown_hpx_impl("hpx", t, body_noop_timed,       exec);
        bench_breakdown_hpx_impl("hpx", t, body_tiny_touch_timed, exec);
        std::printf("\n");
        auto rexec = make_restricted_exec(t);
        bench_breakdown_hpx_impl("rpe", t, body_noop_timed,       rexec);
        bench_breakdown_hpx_impl("rpe", t, body_tiny_touch_timed, rexec);
        std::printf("\n");
    }

    hpx::post([]() { hpx::finalize(); });
    hpx::stop();
    return 0;
}
