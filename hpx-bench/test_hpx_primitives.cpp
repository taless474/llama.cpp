// hpx-bench/test_hpx_primitives.cpp
// Stage 1 smoke test: verify the HPX primitives we'll use in ggml-cpu-hpx.cpp
// work correctly against the installed HPX build.
//
// Build:
//   cmake -B /tmp/hpx-bench-build -S hpx-bench \
//     -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
//     -DCMAKE_BUILD_TYPE=Release
//   cmake --build /tmp/hpx-bench-build
//   /tmp/hpx-bench-build/test_hpx_primitives
//
// Expected output: all tests print PASS, exit 0.

#include <hpx/init.hpp>
#include <hpx/future.hpp>
#include <hpx/synchronization/barrier.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/condition_variable.hpp>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static void pass(const char* name) { std::printf("  PASS  %s\n", name); }
static void fail(const char* name) { std::printf("  FAIL  %s\n", name); std::abort(); }
#define CHECK(cond, name) do { if (cond) pass(name); else fail(name); } while(0)

// ── Test 1: hpx::async + hpx::future::get ────────────────────────────────────

static void test_async_future() {
    auto f = hpx::async([]{ return 42; });
    int result = f.get();
    CHECK(result == 42, "async/future: return value");

    // launch N tasks, sum results
    const int N = 8;
    std::vector<hpx::future<int>> futures;
    futures.reserve(N);
    for (int i = 0; i < N; ++i)
        futures.push_back(hpx::async([i]{ return i * i; }));

    int sum = 0;
    for (auto& fut : futures) sum += fut.get();
    // 0+1+4+9+16+25+36+49 = 140
    CHECK(sum == 140, "async/future: N tasks sum");
}

// ── Test 2: hpx::barrier ─────────────────────────────────────────────────────

static void test_barrier() {
    const int N = 4;
    hpx::barrier<> bar(N);
    std::atomic<int> counter{0};

    std::vector<hpx::future<void>> futures;
    futures.reserve(N - 1);

    // Spawn N-1 worker tasks; the main HPX thread acts as the Nth participant.
    for (int i = 0; i < N - 1; ++i) {
        futures.push_back(hpx::async([&bar, &counter]{
            counter.fetch_add(1, std::memory_order_relaxed);
            bar.arrive_and_wait();  // all N must arrive before any proceeds
            counter.fetch_add(10, std::memory_order_relaxed);
        }));
    }

    // main thread: wait until all workers have incremented once, then join
    bar.arrive_and_wait();
    for (auto& f : futures) f.get();

    // Each of the N-1 workers added 1 then 10; main thread did not add.
    CHECK(counter.load() == (N - 1) * 11, "barrier: all threads passed barrier");
}

// ── Test 3: hpx::mutex + hpx::condition_variable ────────────────────────────

static void test_mutex_condvar() {
    hpx::mutex mtx;
    hpx::condition_variable cv;
    bool ready = false;
    int  value = 0;

    // Producer task: set value, signal ready
    auto producer = hpx::async([&]{
        std::unique_lock<hpx::mutex> lk(mtx);
        value = 99;
        ready = true;
        cv.notify_all();
    });

    // Consumer (main thread): wait for ready
    {
        std::unique_lock<hpx::mutex> lk(mtx);
        cv.wait(lk, [&]{ return ready; });
    }
    producer.get();

    CHECK(value == 99, "mutex/condvar: producer value received");
    CHECK(ready == true, "mutex/condvar: ready flag set");
}

// ── Test 4: barrier simulating ggml_barrier usage ────────────────────────────
// Mirrors the pattern in ggml-cpu-hpx.cpp: multiple "graph" iterations,
// each with a barrier at the end (like ggml_barrier between node ops).

static void test_barrier_multi_phase() {
    const int N_THREADS = 4;
    const int N_PHASES  = 3;  // simulate 3 "graph" dispatches

    hpx::barrier<> bar(N_THREADS);
    std::atomic<int> phase_completions{0};

    std::vector<hpx::future<void>> futures;
    futures.reserve(N_THREADS - 1);

    for (int i = 0; i < N_THREADS - 1; ++i) {
        futures.push_back(hpx::async([&bar, &phase_completions]{
            for (int p = 0; p < N_PHASES; ++p) {
                // simulate work
                phase_completions.fetch_add(1, std::memory_order_relaxed);
                bar.arrive_and_wait();
            }
        }));
    }

    for (int p = 0; p < N_PHASES; ++p) {
        phase_completions.fetch_add(1, std::memory_order_relaxed);
        bar.arrive_and_wait();
    }

    for (auto& f : futures) f.get();

    CHECK(phase_completions.load() == N_THREADS * N_PHASES,
          "barrier multi-phase: all threads completed all phases");
}

// ── Test 5: worker loop pattern ──────────────────────────────────────────────
// Mirrors ggml_graph_compute_secondary_thread: workers wait on condvar,
// main thread signals work, workers process, then main waits for completion.

static void test_worker_loop_pattern() {
    const int N_WORKERS = 3;
    const int N_GRAPHS  = 4;

    hpx::mutex              mtx;
    hpx::condition_variable cv;
    std::atomic<int>        work_gen{0};   // incremented each "graph"
    std::atomic<bool>       stop_flag{false};
    std::atomic<int>        work_done{0};

    std::vector<hpx::future<void>> workers;
    workers.reserve(N_WORKERS);

    for (int w = 0; w < N_WORKERS; ++w) {
        workers.push_back(hpx::async([&, w]{
            int last_gen = 0;
            while (true) {
                // wait for new work or stop
                std::unique_lock<hpx::mutex> lk(mtx);
                cv.wait(lk, [&]{
                    return stop_flag.load() ||
                           work_gen.load(std::memory_order_relaxed) != last_gen;
                });
                if (stop_flag.load()) break;
                last_gen = work_gen.load(std::memory_order_relaxed);
                lk.unlock();

                work_done.fetch_add(1, std::memory_order_relaxed);
            }
        }));
    }

    // dispatch N_GRAPHS rounds of work
    for (int g = 0; g < N_GRAPHS; ++g) {
        {
            std::unique_lock<hpx::mutex> lk(mtx);
            work_gen.fetch_add(1, std::memory_order_relaxed);
            cv.notify_all();
        }
        // spin until all workers processed this graph
        while (work_done.load(std::memory_order_relaxed) < (g + 1) * N_WORKERS)
            hpx::this_thread::yield();
    }

    // stop workers
    {
        std::unique_lock<hpx::mutex> lk(mtx);
        stop_flag.store(true);
        cv.notify_all();
    }
    for (auto& f : workers) f.get();

    CHECK(work_done.load() == N_WORKERS * N_GRAPHS,
          "worker loop: all workers processed all graphs");
}

// ── HPX main ─────────────────────────────────────────────────────────────────

int hpx_main(int, char**) {
    std::printf("\n=== HPX primitives smoke test ===\n\n");

    test_async_future();
    test_barrier();
    test_mutex_condvar();
    test_barrier_multi_phase();
    test_worker_loop_pattern();

    std::printf("\nAll tests passed.\n\n");
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::init(hpx_main, argc, argv);
}
