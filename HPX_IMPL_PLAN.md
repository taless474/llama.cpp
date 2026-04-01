# Implementation Plan: ggml-cpu-hpx.cpp

Goal: replace the pthread thread pool in `ggml-cpu.c` with HPX synchronization
primitives, guarded by `GGML_USE_HPX`, so baseline and HPX builds can be
compared without modifying shared compute logic.

---

## C++ Standard

HPX 1.8+ (including 1.9.x and 1.10.x, the current stable series) requires **C++17**
as the minimum. The CMake target already enforces this:

```cmake
# ggml/src/ggml-cpu/CMakeLists.txt:57
target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
```

No change needed to the standard flag. HPX uses C++20 features internally on
compilers that support them, but its public API and the code we write targets C++17.

---

## What changes and what stays

### Unchanged (shared between pthread and HPX modes)

- `ggml_graph_compute_thread()` — the per-thread node-loop and dispatch logic
- `ggml_threadpool_chunk_set/add()` — `atomic_int current_chunk` operations
- All operator implementations (matmul, etc.) — they call `ggml_barrier` via
  `ggml-cpu-impl.h`, which gets a new body in HPX mode
- `ggml_graph_compute()` control flow — only the kickoff and thread dispatch differ
- `ggml_threadpool_get_n_threads()` — reads `n_threads`, no sync involved

### Replaced in HPX mode

| Current (pthread, `ggml-cpu.c`) | HPX replacement (`ggml-cpu-hpx.cpp`) |
|---|---|
| `ggml_mutex_t mutex` + `ggml_cond_t cond` | `hpx::mutex` + `hpx::condition_variable` |
| `atomic_int n_barrier` + `n_barrier_passed` (spin-wait) | `hpx::barrier<>` (phase barrier, cooperative) |
| `ggml_thread_t thrd` (per worker) | `hpx::future<void>` (per worker) |
| `ggml_graph_compute_secondary_thread()` infinite loop | same loop structure, HPX-aware waiting |
| `ggml_graph_compute_kickoff()` — cond_broadcast | `hpx::condition_variable::notify_all()` |
| `ggml_graph_compute_check_for_work()` — cond_wait | `hpx::condition_variable::wait()` |
| `ggml_barrier()` — spin on `n_barrier_passed` | `hpx_barrier->arrive_and_wait()` |
| `ggml_threadpool_new_impl()` — pthread_create loop | `hpx::async` loop |
| `ggml_threadpool_free()` — pthread_join loop | `future.get()` loop |
| `ggml_threadpool_pause/resume()` — cond_broadcast | `hpx::condition_variable::notify_all()` |

---

## Step 1: New shared header — `ggml-cpu-threads.h`

Create `ggml/src/ggml-cpu/ggml-cpu-threads.h`. Extract `struct ggml_threadpool`
and `struct ggml_compute_state` from `ggml-cpu.c` (currently at lines 461–497)
into this header, adding conditional fields for HPX mode.

```c
// ggml-cpu-threads.h
#pragma once
#include "ggml-cpu.h"
#include "ggml-impl.h"   // for GGML_CACHE_ALIGN, atomic_*, etc.

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

struct ggml_threadpool {
    // --- fields used by both C ops code and threading code ---
    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    atomic_int  GGML_CACHE_ALIGN current_chunk;  // work-stealing chunk index
    atomic_int  n_graph;    // packed: (graph_ctr << 16) | n_active_threads
    atomic_bool stop;
    atomic_bool pause;
    atomic_int  abort;

    struct ggml_compute_state * workers;
    int      n_threads;
    int32_t  prio;
    uint32_t poll;
    enum ggml_status ec;

    // --- pthread-specific fields (excluded in HPX mode) ---
#if !defined(GGML_USE_HPX)
    ggml_mutex_t mutex;
    ggml_cond_t  cond;
    atomic_int   GGML_CACHE_ALIGN n_barrier;
    atomic_int   GGML_CACHE_ALIGN n_barrier_passed;
#endif

    // --- HPX-specific fields (void* so this header stays valid C) ---
#if defined(GGML_USE_HPX)
    void * hpx_mutex;    // hpx::mutex*
    void * hpx_cond;     // hpx::condition_variable*
    void * hpx_barrier;  // hpx::barrier<>*
#endif
};

struct ggml_compute_state {
    // thread handle — pthread in default mode, hpx::future<void>* in HPX mode
#if !defined(GGML_USE_HPX) && !defined(GGML_USE_OPENMP)
    ggml_thread_t thrd;
#endif
#if defined(GGML_USE_HPX)
    void * hpx_future;   // hpx::future<void>*
#endif
#if !defined(GGML_USE_OPENMP)
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};
```

Then in `ggml-cpu.c`, remove the struct definitions at lines 461–497 and add:
```c
#include "ggml-cpu-threads.h"
```

Also move `GGML_THREADPOOL_N_THREADS_MASK` / `_BITS` defines (currently lines
198–199 in `ggml-cpu.c`) into this header and remove them from `ggml-cpu.c`.

---

## Step 2: Guards in `ggml-cpu.c`

Wrap the pthread-only functions so they are not compiled when `GGML_USE_HPX` is
defined. The rule: anything that touches `mutex`, `cond`, `n_barrier`,
`n_barrier_passed`, or `ggml_thread_t` gets guarded.

### 2a. `ggml_barrier()` — lines 556–592
```c
#if !defined(GGML_USE_HPX)
void ggml_barrier(struct ggml_threadpool * tp) { ... }
#endif
```

### 2b. `ggml_threadpool_free()` — lines 2657–2686
```c
#if !defined(GGML_USE_HPX)
void ggml_threadpool_free(...) { ... }
#endif
```

### 2c. `ggml_threadpool_pause/resume()` — lines 2703–2727
```c
#if !defined(GGML_USE_HPX)
void ggml_threadpool_pause(...) { ... }
void ggml_threadpool_resume(...) { ... }
#endif
```

### 2d. The four helper functions inside `#ifndef GGML_USE_OPENMP` block
`ggml_graph_compute_thread_ready`, `ggml_graph_compute_thread_sync`,
`ggml_graph_compute_poll_for_work`, `ggml_graph_compute_check_for_work`
(lines 3012–3076):
```c
#if !defined(GGML_USE_OPENMP) && !defined(GGML_USE_HPX)
static inline bool ggml_graph_compute_thread_ready(...) { ... }
...
#endif
```

### 2e. `ggml_graph_compute_secondary_thread()` — lines 3078–3113
```c
#if !defined(GGML_USE_OPENMP) && !defined(GGML_USE_HPX)
static thread_ret_t ggml_graph_compute_secondary_thread(...) { ... }
#endif
```

### 2f. `ggml_graph_compute_kickoff()` — lines 3116–3146
```c
#if !defined(GGML_USE_OPENMP) && !defined(GGML_USE_HPX)
static void ggml_graph_compute_kickoff(...) { ... }
#endif
```

### 2g. `ggml_threadpool_new_impl()` — lines 3150–3221
```c
#if !defined(GGML_USE_HPX)
static struct ggml_threadpool * ggml_threadpool_new_impl(...) { ... }

struct ggml_threadpool * ggml_threadpool_new(...) {
    return ggml_threadpool_new_impl(...);
}
#endif
```

### 2h. In `ggml_graph_compute()` — lines 3279–3290
Add an HPX branch alongside the existing OpenMP and pthread branches:
```c
#ifdef GGML_USE_OPENMP
    ...
#elif defined(GGML_USE_HPX)
    // defined in ggml-cpu-hpx.cpp, forward declared
    ggml_graph_compute_hpx(threadpool, n_threads);
#else
    ggml_graph_compute_kickoff(threadpool, n_threads);
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif
```

Where `ggml_graph_compute_hpx` does the kickoff + runs thread 0 work inline
(same role as the current `#else` branch, just using HPX kickoff).

---

## Step 3: HPX runtime initialization

HPX requires its runtime to be started before any HPX API call. We built HPX
with `HPX_WITH_NETWORKING=ON` (TCP parcelport, for future Pi cluster use), so
use the full distributed runtime headers:

```cpp
// in ggml-cpu-hpx.cpp
#include <hpx/init.hpp>

namespace {
    struct HpxRuntime {
        HpxRuntime() {
            hpx::init_params p;
            // let HPX use hardware_concurrency OS threads by default;
            // the barrier controls how many actually participate per graph
            hpx::start(nullptr, 0, p);
        }
        ~HpxRuntime() {
            hpx::finalize();
            hpx::stop();
        }
    };
    // constructed before main() in this TU, destroyed after
    HpxRuntime g_hpx_runtime;
}
```

**Important:** HPX OS thread count should be set to the maximum `n_threads` that
will ever be requested. For `llama-bench` this is `hardware_concurrency`. If a
finer control is needed, pass `--hpx:threads=N` on the command line (HPX reads
`argv` from the `start` call) or set `p.cfg = {"hpx.os_threads=N"}`.

The `hpx::barrier<>` is created per threadpool with `n_threads` as the expected
participant count, not the HPX OS thread count, so correctness is independent
of how many HPX threads are running overall.

---

## Step 4: `ggml-cpu-hpx.cpp` — full function inventory

All functions use `extern "C"` linkage so `ggml-cpu.c` can call them without
change to existing call sites.

### 4a. Includes
```cpp
#include "ggml-cpu-threads.h"
#include "ggml-cpu-impl.h"   // ggml_barrier decl, ggml_threadpool_chunk_*
#include "ggml-cpu.h"

#include <hpx/init.hpp>
#include <hpx/future.hpp>
#include <hpx/synchronization/barrier.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/condition_variable.hpp>

#include <memory>
#include <atomic>
```

### 4b. Accessor helpers (avoids casting everywhere)
```cpp
static hpx::mutex& tp_mutex(ggml_threadpool* tp) {
    return *static_cast<hpx::mutex*>(tp->hpx_mutex);
}
static hpx::condition_variable& tp_cond(ggml_threadpool* tp) {
    return *static_cast<hpx::condition_variable*>(tp->hpx_cond);
}
static hpx::barrier<>& tp_barrier(ggml_threadpool* tp) {
    return *static_cast<hpx::barrier<>*>(tp->hpx_barrier);
}
static hpx::future<void>& state_future(ggml_compute_state* s) {
    return *static_cast<hpx::future<void>*>(s->hpx_future);
}
```

### 4c. `ggml_barrier`
```cpp
extern "C" void ggml_barrier(ggml_threadpool* tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed)
                    & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) return;
    tp_barrier(tp).arrive_and_wait();
}
```

`hpx::barrier<>::arrive_and_wait()` suspends the calling HPX thread
cooperatively (not a busy-spin, not a blocking OS call) until all
`n_threads` participants have arrived. This is the key difference from the
current spin-wait on `n_barrier_passed`.

### 4d. Check-for-work and worker loop
```cpp
// HPX-aware equivalent of ggml_graph_compute_check_for_work
static bool hpx_check_for_work(ggml_compute_state* state) {
    auto* tp = state->threadpool;
    std::unique_lock<hpx::mutex> lk(tp_mutex(tp));
    tp_cond(tp).wait(lk, [&]{
        if (tp->stop || tp->pause) return true;
        int ng = atomic_load_explicit(&tp->n_graph, memory_order_relaxed);
        if (ng != state->last_graph) {
            int n_threads = ng & GGML_THREADPOOL_N_THREADS_MASK;
            state->pending    = (state->ith < n_threads);
            state->last_graph = ng;
            return true;
        }
        return false;
    });
    lk.unlock();
    return state->pending;
}

static void hpx_worker_thread(ggml_compute_state* state) {
    // priority and affinity can still be set — they affect the HPX OS thread
    // that runs this task (best-effort since HPX may steal)
    ggml_thread_apply_priority(state->threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // handle pause: wait on HPX cond without consuming work
        {
            std::unique_lock<hpx::mutex> lk(tp_mutex(state->threadpool));
            tp_cond(state->threadpool).wait(lk, [&]{
                return !state->threadpool->pause;
            });
        }

        if (state->threadpool->stop) break;

        hpx_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);  // unchanged C function
        }
    }
}
```

### 4e. Kickoff
```cpp
static void ggml_graph_compute_kickoff_hpx(ggml_threadpool* tp, int n_threads) {
    std::unique_lock<hpx::mutex> lk(tp_mutex(tp));

    int n_graph = atomic_load_explicit(&tp->n_graph, memory_order_relaxed)
                  >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS)
              | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    atomic_store_explicit(&tp->n_graph, n_graph, memory_order_seq_cst);

    // recreate barrier with correct participant count
    // (count changes between graphs if llama uses different thread counts)
    delete static_cast<hpx::barrier<>*>(tp->hpx_barrier);
    tp->hpx_barrier = new hpx::barrier<>(n_threads);

    tp_cond(tp).notify_all();
    // lk released on scope exit
}
```

Note on barrier recreation: `n_threads` can vary between graphs (see
`ggml_graph_compute:3280` where it clamps to `threadpool->n_threads`).
The `hpx::barrier` participant count is fixed at construction, so we
reconstruct it when the count changes. This is cheap — only happens at
graph dispatch, not at each node barrier within a graph.

Optimization: cache the last `n_threads` value and skip recreation if
unchanged. Most inference runs use the same thread count every graph.

### 4f. `ggml_graph_compute_hpx` (called from `ggml-cpu.c`)
```cpp
extern "C" void ggml_graph_compute_hpx(ggml_threadpool* tp, int n_threads) {
    ggml_graph_compute_kickoff_hpx(tp, n_threads);
    // thread 0 is the main thread — run its work inline
    ggml_graph_compute_thread(&tp->workers[0]);
}
```

### 4g. `ggml_threadpool_new_impl` (HPX version)
```cpp
static ggml_threadpool* ggml_threadpool_new_impl_hpx(
    ggml_threadpool_params* tpp,
    ggml_cgraph* cgraph,
    ggml_cplan* cplan)
{
    auto* tp = static_cast<ggml_threadpool*>(
        ggml_aligned_malloc(sizeof(ggml_threadpool)));

    // zero all fields, then set
    memset(tp, 0, sizeof(*tp));
    tp->cgraph    = cgraph;
    tp->cplan     = cplan;
    tp->n_threads = tpp->n_threads;
    tp->prio      = tpp->prio;
    tp->poll      = tpp->poll;   // kept for potential future use
    tp->ec        = GGML_STATUS_SUCCESS;
    atomic_store_explicit(&tp->abort, -1, memory_order_relaxed);
    atomic_store(&tp->pause, tpp->paused);

    tp->hpx_mutex   = new hpx::mutex();
    tp->hpx_cond    = new hpx::condition_variable();
    tp->hpx_barrier = new hpx::barrier<>(tpp->n_threads);

    const size_t wsz = sizeof(ggml_compute_state) * tpp->n_threads;
    auto* workers = static_cast<ggml_compute_state*>(ggml_aligned_malloc(wsz));
    memset(workers, 0, wsz);

    int32_t cpumask_iter = 0;
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = tp;
        workers[j].ith        = j;
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask,
                                 tpp->strict_cpu, &cpumask_iter);
    }
    tp->workers = workers;

    // spawn N-1 HPX tasks (worker 0 = main thread, same as pthread baseline)
    for (int j = 1; j < tpp->n_threads; j++) {
        workers[j].hpx_future = new hpx::future<void>(
            hpx::async(&hpx_worker_thread, &workers[j]));
    }

    if (!tpp->paused) {
        ggml_thread_apply_priority(tpp->prio);
        if (ggml_thread_cpumask_is_valid(workers[0].cpumask)) {
            ggml_thread_apply_affinity(workers[0].cpumask);
        }
    }

    return tp;
}

extern "C" ggml_threadpool* ggml_threadpool_new(ggml_threadpool_params* tpp) {
    return ggml_threadpool_new_impl_hpx(tpp, nullptr, nullptr);
}
```

### 4h. `ggml_threadpool_free`
```cpp
extern "C" void ggml_threadpool_free(ggml_threadpool* tp) {
    if (!tp) return;

    {
        std::unique_lock<hpx::mutex> lk(tp_mutex(tp));
        tp->stop  = true;
        tp->pause = false;
        tp_cond(tp).notify_all();
    }

    for (int j = 1; j < tp->n_threads; j++) {
        state_future(&tp->workers[j]).get();   // join
        delete static_cast<hpx::future<void>*>(tp->workers[j].hpx_future);
    }

    delete static_cast<hpx::barrier<>*>(tp->hpx_barrier);
    delete static_cast<hpx::condition_variable*>(tp->hpx_cond);
    delete static_cast<hpx::mutex*>(tp->hpx_mutex);

    const size_t wsz = sizeof(ggml_compute_state) * tp->n_threads;
    ggml_aligned_free(tp->workers, wsz);
    ggml_aligned_free(tp, sizeof(ggml_threadpool));
}
```

### 4i. `ggml_threadpool_pause` / `ggml_threadpool_resume`
```cpp
extern "C" void ggml_threadpool_pause(ggml_threadpool* tp) {
    std::unique_lock<hpx::mutex> lk(tp_mutex(tp));
    tp->pause = true;
    tp_cond(tp).notify_all();
}

extern "C" void ggml_threadpool_resume(ggml_threadpool* tp) {
    std::unique_lock<hpx::mutex> lk(tp_mutex(tp));
    tp->pause = false;
    tp_cond(tp).notify_all();
}
```

---

## Step 5: `ggml-cpu-impl.h` — forward-declare `ggml_graph_compute_hpx`

Add to `ggml-cpu-impl.h` (or just at the top of `ggml-cpu.c`):
```c
#ifdef GGML_USE_HPX
#ifdef __cplusplus
extern "C" {
#endif
void ggml_graph_compute_hpx(struct ggml_threadpool * tp, int n_threads);
#ifdef __cplusplus
}
#endif
#endif
```

---

## Step 6: CMake changes

### 6a. `ggml/CMakeLists.txt` — add option near the OpenMP option (line 240)
```cmake
option(GGML_OPENMP "ggml: use OpenMP"  ON)
option(GGML_HPX    "ggml: use HPX for threading" OFF)   # add this line
```

### 6b. `ggml/src/ggml-cpu/CMakeLists.txt` — add HPX block after the OpenMP block (after line 86)
```cmake
if (GGML_HPX)
    find_package(HPX REQUIRED)
    if (HPX_FOUND)
        message(STATUS "HPX found: ${HPX_VERSION}")
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_HPX)
        list(APPEND GGML_CPU_SOURCES ggml-cpu/ggml-cpu-hpx.cpp)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE HPX::hpx HPX::wrap_main HPX::hpx_init)
    else()
        message(FATAL_ERROR "HPX not found but GGML_HPX=ON")
    endif()
endif()
```

`HPX::wrap_main` rewrites `main()` to be HPX-aware (needed if we use the global
`HpxRuntime` approach; alternatively use `HPX::hpx_init` with explicit
`hpx::local::start`).

Note: `HPX::wrap_main` may conflict with llama.cpp's `main`. In that case, use
explicit `hpx::local::start`/`stop` in the `HpxRuntime` constructor/destructor
and link only `HPX::hpx HPX::hpx_init`.

---

## Step 7: File-by-file summary of all edits

| File | Action |
|---|---|
| `ggml-cpu.c` | Remove struct defs (lines 461–497); include `ggml-cpu-threads.h`; add `#ifndef GGML_USE_HPX` guards on 8 function bodies; add `#elif GGML_USE_HPX` branch in `ggml_graph_compute` |
| `ggml-cpu-impl.h` | Add `extern "C"` declaration for `ggml_graph_compute_hpx` under `GGML_USE_HPX` |
| `ggml-cpu-threads.h` | **NEW** — struct definitions with conditional fields |
| `ggml-cpu-hpx.cpp` | **NEW** — all HPX implementations (Steps 3–4 above) |
| `ggml/CMakeLists.txt` | Add `option(GGML_HPX ...)` |
| `ggml/src/ggml-cpu/CMakeLists.txt` | Add HPX `find_package` + source + link block |

---

## Step 8: Known risks and mitigations

### R1: CPU affinity on HPX tasks
`ggml_thread_apply_affinity` in `hpx_worker_thread` sets affinity on the
calling OS thread. HPX may run the same HPX task on different OS threads
across graphs (work-stealing). For the benchmark this is acceptable (and
mirrors what we want to measure). If affinity must be stable, use HPX thread
executors pinned to specific OS threads via `hpx::threads::executor`.

### R2: `hpx::barrier` participant count mismatch
If `ggml_barrier` is called by fewer threads than expected (e.g., a graph
where some ops use fewer threads), the barrier will deadlock. The existing
code handles this with the `n_threads == 1` early-return. We keep that check.
The reconstruction-on-count-change in kickoff handles the variable-count case.

### R3: HPX runtime not started before first `ggml_threadpool_new`
The `g_hpx_runtime` global object in `ggml-cpu-hpx.cpp` initializes before
`main()` only if the TU is linked and the linker doesn't dead-strip it. Add a
dummy `volatile` reference in `ggml_threadpool_new_impl_hpx` to force the
symbol live:
```cpp
(void)g_hpx_runtime;  // ensure TU is not dead-stripped
```

### R4: `GGML_USE_OPENMP` + `GGML_USE_HPX` both defined
These are mutually exclusive. Add to CMake:
```cmake
if (GGML_HPX AND GGML_OPENMP)
    message(FATAL_ERROR "GGML_HPX and GGML_OPENMP are mutually exclusive")
endif()
```

### R5: `disposable_threadpool` path
`ggml_graph_compute` creates a throwaway threadpool when `cplan->threadpool`
is NULL (line 3239). In HPX mode this creates and destroys HPX futures +
primitives on every call — expensive. For the benchmark, always pass a
persistent threadpool via `llama-bench`'s existing `-t` path. If needed,
optimize later by caching a singleton threadpool.

---

## Build commands (recap from HPX_BENCHMARK_PLAN.md)

```bash
# HPX must be installed, e.g.:
#   brew install hpx  (or build from source)

# Pair A — BLAS ON
cmake -B build-hpx-blas \
  -DGGML_METAL=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-blas -j --target llama-bench

# Pair B — BLAS OFF
cmake -B build-hpx-noblas \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-noblas -j --target llama-bench
```

---

## Implementation order

1. Create `ggml-cpu-threads.h` with the shared struct definitions (Step 1)
2. Add `#include "ggml-cpu-threads.h"` to `ggml-cpu.c`, verify it compiles
3. Add the guards in `ggml-cpu.c` (Step 2) — still compiles without HPX
4. Implement `ggml-cpu-hpx.cpp` skeleton with `extern "C"` stubs that abort,
   verify it links
5. Implement functions one-by-one in this order: runtime init → `ggml_barrier`
   → `ggml_threadpool_new/free` → kickoff → worker loop → pause/resume
6. Add CMake wiring (Step 6)
7. Build, run `./build-hpx/bin/llama-cli -m ... --n-predict 5` as a smoke test
8. Run `llama-bench` sweep
