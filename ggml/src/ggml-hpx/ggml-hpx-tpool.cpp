// ggml-hpx-tpool.cpp
//
// HPX executor ops — persistent-worker variant (Option B).
//
// init:        spawn n_threads-1 persistent HPX lightweight threads; each
//              sleeps on a counting semaphore until signalled by kickoff.
// kickoff:     signal all n_threads-1 semaphores; workers wake, call
//              ggml_graph_compute_thread_run, then go back to sleep.
// worker_wait: unreachable — ggml_graph_compute_secondary_thread is never
//              entered because hpx_init creates no pthreads.
// destroy:     set stop, signal semaphores to wake sleepers, wait for all
//              futures, free HpxTpoolState, delegate mutex/cond teardown.
//
// Synchronisation model
// ─────────────────────
// ggml_graph_compute_thread uses an atomic spin barrier (n_barrier /
// n_barrier_passed) that requires all n_active_threads to arrive before
// any proceed.  The caller (ggml_graph_compute) runs workers[0] on the
// main thread synchronously.  When ggml_graph_compute returns, the final
// barrier has already passed — all persistent tasks have finished accessing
// shared state and are back to sleeping on their semaphore.
//
// The seq_cst store of n_graph in ggml_graph_compute_kickoff acts as a
// full release fence before hpx_kickoff is called; the semaphore release/
// acquire pair provides at least acquire semantics on the worker side,
// establishing: job fields → n_graph store → sem.signal → sem.wait →
// task reads current_job.
//
// Coroutine guard
// ───────────────
// hpx::threads::get_self_ptr() returns non-null only inside an HPX
// lightweight thread (coroutine).  On any OS thread — including the main
// thread registered as HPX worker 0 — it returns nullptr.
//
// When kickoff is called from inside an HPX lightweight thread the spin
// barrier would block that OS thread while persistent workers cannot run
// on it → deadlock.  Fix: downgrade to single-thread execution.
//
// Serial-decode fallback
// ──────────────────────
// When GGML_HPX_SERIAL_DECODE=1, if the last graph node's ne[1] <= 1
// (single-token decode), override n_active_threads to 1 and skip the
// semaphore signals.
//
// Instrumentation
// ───────────────
// GGML_HPX_STATS=1 prints on destroy:
//   parallel-kickoffs  — dispatches where >1 thread was used
//   serial-fallbacks   — single-token decode fast-paths taken

#include "ggml-hpx-tpool.h"

#include "ggml.h"    // GGML_ABORT

#include <hpx/future.hpp>
#include <hpx/include/post.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/synchronization/mutex.hpp>    // hpx::threads::get_self_ptr

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// Runtime config — read once from environment.
// ---------------------------------------------------------------------------

namespace
{

struct HpxTpoolConfig
{
    bool serial_decode = false;    // GGML_HPX_SERIAL_DECODE=1
    bool print_stats   = false;    // GGML_HPX_STATS=1
};

HpxTpoolConfig g_cfg;
std::once_flag g_cfg_flag;

void load_config()
{
    std::call_once(g_cfg_flag, []() {
        g_cfg.serial_decode =
            (std::getenv("GGML_HPX_SERIAL_DECODE") != nullptr);
        g_cfg.print_stats =
            (std::getenv("GGML_HPX_STATS") != nullptr);
    });
}

// ---------------------------------------------------------------------------
// Global dispatch counters.
// ---------------------------------------------------------------------------

std::atomic<long long> g_kickoff_calls{0};
std::atomic<long long> g_serial_fallbacks{0};

// ---------------------------------------------------------------------------
// Persistent worker state.
//
// One HpxWorkerSlot per secondary worker (j = 1 .. n_threads-1).
// The slot is owned by HpxTpoolState; the worker task holds a raw pointer
// and must not outlive the state.  HpxTpoolState is stored as
// tp->executor_priv and freed in hpx_destroy after all futures complete.
// ---------------------------------------------------------------------------

struct HpxWorkerSlot
{
    // HPX-aware counting semaphore: worker sleeps here between dispatches.
    // Starts at 0 (blocked).  kickoff signals 1 per dispatch; destroy
    // signals 1 extra so the stop check runs and the loop exits.
    hpx::counting_semaphore_var<> sem{0};

    // Set by destroy before the final signal.  Checked after wakeup.
    std::atomic<bool> stop{false};

    // The future for this worker's persistent HPX task.  Awaited in destroy.
    hpx::future<void> task{};

    // Pointer into tp->workers[j].  Valid for the lifetime of the threadpool.
    struct ggml_compute_state* state = nullptr;

    // -----------------------------------------------------------------------
    // Wake-latency instrumentation (only used when GGML_HPX_STATS=1).
    //
    // t_signal_ns: written by the main thread with relaxed ordering just
    //   before sem.signal(); the semaphore release/acquire pair provides the
    //   full happens-before ordering so the worker can read it safely after
    //   sem.wait() returns.
    //
    // sum_wake_ns / sum_entry_ns / n_dispatches: written exclusively by the
    //   worker task; read by the main thread only after task.wait() completes
    //   (the future provides the necessary fence — no atomics needed).
    // -----------------------------------------------------------------------
    std::atomic<std::int64_t> t_signal_ns{0};

    long long sum_wake_ns  = 0;   // sum of (t_woke  - t_signal) per dispatch
    long long sum_entry_ns = 0;   // sum of (t_start - t_woke)   per dispatch
    long long n_dispatches = 0;
};

struct HpxTpoolState
{
    // Index 0 → secondary worker 1, index k → secondary worker k+1.
    std::vector<std::unique_ptr<HpxWorkerSlot>> slots;
};

// ---------------------------------------------------------------------------
// Persistent worker loop.
//
// Runs as an HPX lightweight thread.  The loop order is intentional:
//   1. wake (sem.wait)
//   2. check stop
//   3. exit if stopping    ← teardown can never trigger an extra dispatch
//   4. run one dispatch
// ---------------------------------------------------------------------------

void worker_loop(HpxWorkerSlot* slot)
{
    for (;;)
    {
        slot->sem.wait(1);
        auto t_woke = std::chrono::steady_clock::now();

        if (slot->stop.load(std::memory_order_acquire))
        {
            break;
        }

        auto t_start = std::chrono::steady_clock::now();
        ggml_graph_compute_thread_run(slot->state);

        if (g_cfg.print_stats)
        {
            auto sig_ns   = slot->t_signal_ns.load(std::memory_order_relaxed);
            auto woke_ns  = t_woke.time_since_epoch().count();
            auto start_ns = t_start.time_since_epoch().count();
            slot->sum_wake_ns  += woke_ns  - sig_ns;
            slot->sum_entry_ns += start_ns - woke_ns;
            ++slot->n_dispatches;
        }
    }
}

}    // namespace

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

static void hpx_init(
    struct ggml_threadpool*        tp,
    struct ggml_threadpool_params* /*tpp*/)
{
    load_config();

    int n_threads = ggml_threadpool_n_threads(tp);
    if (n_threads <= 1)
    {
        ggml_threadpool_set_priv(tp, nullptr);
        return;
    }

    auto* s = new HpxTpoolState{};
    s->slots.reserve(static_cast<std::size_t>(n_threads - 1));

    for (int j = 1; j < n_threads; ++j)
    {
        auto slot    = std::make_unique<HpxWorkerSlot>();
        slot->state  = ggml_threadpool_worker(tp, j);
        // Launch persistent HPX lightweight thread.  The raw pointer is safe
        // because HpxTpoolState (and its slots) outlive the future.
        slot->task   = hpx::async(worker_loop, slot.get());
        s->slots.push_back(std::move(slot));
    }

    ggml_threadpool_set_priv(tp, s);
}

static void hpx_kickoff(struct ggml_threadpool* tp, int n_threads)
{
    // Coroutine guard: spinning inside an HPX lightweight thread would
    // deadlock because persistent workers cannot run on that OS thread.
    if (hpx::threads::get_self_ptr() != nullptr)
    {
        ggml_threadpool_set_active_threads(tp, 1);
        return;
    }

    // Serial-decode fallback.
    if (g_cfg.serial_decode)
    {
        int64_t ne1 = ggml_threadpool_last_node_ne1(tp);
        if (ne1 >= 0 && ne1 <= 1)
        {
            ggml_threadpool_set_active_threads(tp, 1);
            g_serial_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }

    auto* s = static_cast<HpxTpoolState*>(ggml_threadpool_get_priv(tp));
    if (s == nullptr || s->slots.empty())
    {
        return;
    }

    g_kickoff_calls.fetch_add(1, std::memory_order_relaxed);

    // Signal n_threads-1 workers.  n_threads may be less than the pool size
    // on the last dispatch (capped by the job's n_active_threads); wake only
    // what will actually participate.
    int n_wake = n_threads - 1;
    if (n_wake > static_cast<int>(s->slots.size()))
    {
        n_wake = static_cast<int>(s->slots.size());
    }
    for (int j = 0; j < n_wake; ++j)
    {
        auto& slot = s->slots[static_cast<std::size_t>(j)];
        if (g_cfg.print_stats)
        {
            slot->t_signal_ns.store(
                std::chrono::steady_clock::now().time_since_epoch().count(),
                std::memory_order_relaxed);
        }
        slot->sem.signal(1);
    }
}

static void hpx_worker_wait(
    struct ggml_threadpool*    /*tp*/,
    struct ggml_compute_state* /*state*/)
{
    // Unreachable: hpx_init creates no pthreads; secondary workers run as
    // persistent HPX tasks and never enter ggml_graph_compute_secondary_thread.
    GGML_ABORT("hpx_worker_wait: unreachable in HPX persistent-worker substrate");
}

static void hpx_destroy(struct ggml_threadpool* tp)
{
    auto* s = static_cast<HpxTpoolState*>(ggml_threadpool_get_priv(tp));

    // Aggregate wake-latency stats before tearing down (s is freed below).
    long long total_dispatches = 0;
    long long total_wake_ns    = 0;
    long long total_entry_ns   = 0;

    if (s != nullptr)
    {
        // Signal stop before waking so the worker never starts a new dispatch
        // after seeing the signal.
        for (auto& slot : s->slots)
        {
            slot->stop.store(true, std::memory_order_release);
            slot->sem.signal(1);
        }
        // Block until all persistent tasks have exited their loops.
        // task.wait() provides the fence: slot accumulators are safe to read.
        for (auto& slot : s->slots)
        {
            slot->task.wait();
            total_dispatches += slot->n_dispatches;
            total_wake_ns    += slot->sum_wake_ns;
            total_entry_ns   += slot->sum_entry_ns;
        }
        delete s;
        ggml_threadpool_set_priv(tp, nullptr);
    }

    if (g_cfg.print_stats)
    {
        long long calls  = g_kickoff_calls.load(std::memory_order_relaxed);
        long long serial = g_serial_fallbacks.load(std::memory_order_relaxed);

        double avg_wake_us  = total_dispatches > 0
            ? double(total_wake_ns)  / double(total_dispatches) / 1000.0
            : 0.0;
        double avg_entry_us = total_dispatches > 0
            ? double(total_entry_ns) / double(total_dispatches) / 1000.0
            : 0.0;
        double avg_total_us = avg_wake_us + avg_entry_us;

        fprintf(stderr,
            "[hpx-tpool] parallel-kickoffs=%lld"
            " worker-dispatches=%lld"
            " avg-wake-latency-us=%.2f"
            " avg-entry-latency-us=%.2f"
            " avg-total-wake-to-run-us=%.2f"
            " serial-fallbacks=%lld\n",
            calls, total_dispatches,
            avg_wake_us, avg_entry_us, avg_total_us,
            serial);
    }

    ggml_threadpool_destroy_substrate(tp);
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

static const struct ggml_cpu_executor_ops ggml_hpx_executor_ops = {
    /* .init        = */ hpx_init,
    /* .kickoff     = */ hpx_kickoff,
    /* .worker_wait = */ hpx_worker_wait,
    /* .destroy     = */ hpx_destroy,
};

extern "C" const struct ggml_cpu_executor_ops* ggml_hpx_tpool_get_ops()
{
    return &ggml_hpx_executor_ops;
}
