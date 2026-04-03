# HPX Benchmarking Plan

**Goal**: evaluate an HPX-backed `ggml` thread pool against the existing pthread-based thread pool in `llama.cpp` on Apple Silicon M4, using Llama 3.1 8B Instruct Q4_K_M, across two BLAS conditions and understand where HPX helps, where it hurts, and why.

## Motivation

### Pair A: pthreads vs HPX with GGML_BLAS=ON (realistic baseline)

When Accelerate/BLAS is enabled (the default on macOS), large matrix multiplications are offloaded to Apple's Accelerate framework. The ggml thread pool handles only the non-GEMM ops (softmax, RoPE, layer norm, etc.) — a small fraction of total compute.
This pair answers: **does HPX help in real-world inference conditions?**

### Pair B: pthreads vs HPX with GGML_BLAS=OFF (pure threading comparison)

With BLAS disabled, ggml's thread pool is responsible for all compute including GEMM.
Note that disabling BLAS changes both the threading load *and* the compute path itself (ggml's own GEMM kernels replace Accelerate's), so any performance difference reflects a combination of scheduler quality and kernel differences — not scheduling alone.
This pair answers: **is HPX competitive when the thread pool is responsible for all compute?**

---

## Repository structure
```text
llama-hpx/
├── README.md                          ← main project README (modified)
├── .gitignore                         ← ignores local models, build dirs, and benchmark outputs
├── docs/
│   ├── HPX_BENCHMARK_PLAN.md          ← benchmark plan, current state, and results
│   └── HPX_IMPL_PROVENANCE.md         ← implementation notes
├── ggml/
│   ├── CMakeLists.txt                 ← added GGML_HPX option here
│   └── src/ggml-cpu/
│       ├── ggml-cpu.c                 ← pthread-based thread pool implementation
│       ├── ggml-cpu-hpx.cpp           ← HPX-backed thread pool implementation
│       └── CMakeLists.txt             ← added HPX conditional linking here
├── models/
│   ├── ggml-vocab-*.gguf
│   ├── templates/
│   ├── README.md                      ← download instructions for benchmark models
│   └── llama3.1-8b/                   ← gitignored, download manually
└── hpx-bench/
    ├── CMakeLists.txt                 ← benchmark/test build definitions
    ├── benchmark.sh                   ← benchmark runner
    ├── test_correctness.sh            ← correctness/stress test runner
    ├── bench_dispatch_overhead.c
    ├── test_ggml_threadpool.c
    ├── test_ggml_threadpool_common.h
    ├── test_hpx_primitives.cpp
    ├── ...                            ← additional targeted repro/stress tests
    ├── results/                       ← gitignored benchmark outputs
    ├── build-pthread/                 ← pthreads + BLAS ON
    ├── build-pthread-noblas/          ← pthreads + BLAS OFF
    ├── build-hpx/                     ← HPX + BLAS ON
    └── build-hpx-noblas/              ← HPX + BLAS OFF
```
---

## Threading model in ggml-cpu.c

The thread pool lives in `ggml/src/ggml-cpu/ggml-cpu.c`. It is **pthread-based**. Key components:

- `struct ggml_threadpool` — holds worker threads, sync primitives, atomic state
- `ggml_thread_create()` — wraps `pthread_create` (Unix) / `CreateThread` (Windows)
- `ggml_graph_compute_secondary_thread()` — worker loop (lines 3070–3107)
- `ggml_barrier()` — synchronization barrier
- Hybrid sync strategy: busy-wait polling + `pthread_mutex` + `pthread_cond_broadcast`

### pthread → HPX equivalents

| Current (pthread) | Current implementation |
|---|---|
| `ggml_thread_create()` | `hpx::async` task launch |
| `ggml_graph_compute_secondary_thread()` | persistent HPX worker task (`hpx_worker_thread`) |
| `ggml_barrier()` | `hpx::barrier<>` |
| `pthread_mutex_t` | `std::mutex` |
| `pthread_cond_t` + `pthread_cond_broadcast` | `std::condition_variable` |

---

## Integration plan

**New file:** `ggml/src/ggml-cpu/ggml-cpu-hpx.cpp`

Implements the same `ggml_threadpool` interface as `ggml-cpu.c` but using HPX primitives.
Compiled in place of the pthread implementation when `-DGGML_HPX=ON`.

**CMake changes:**
- `ggml/CMakeLists.txt` — add `option(GGML_HPX "use HPX for threading" OFF)`
- `ggml/src/ggml-cpu/CMakeLists.txt` — conditionally link HPX, swap source file

**Four build dirs** keep all variants independent (matching the names used throughout this document):
- `build-pthread-blas/` — pthreads + BLAS ON
- `build-hpx-blas/` — HPX + BLAS ON
- `build-pthread-noblas/` — pthreads + BLAS OFF
- `build-hpx-noblas/` — HPX + BLAS OFF

---
## Validation and debugging test suite

The HPX backend is not evaluated by benchmark numbers alone. Before timing anything, we run a correctness and stress suite that checks whether the `ggml` thread-pool contract still behaves correctly under both pthread and HPX implementations.

The main contract suite is `hpx-bench/test_ggml_threadpool.c`. Its purpose is to verify that the exact interface used by `ggml_graph_compute()` behaves correctly regardless of backend, including `ggml_threadpool_new/free/pause/resume`, `ggml_graph_plan`, and `ggml_graph_compute`. :contentReference[oaicite:0]{index=0}

### Main contract suite: `test_ggml_threadpool.c` (13-test contract suite)

1. **Lifecycle**
   Verifies that thread pools can be created and freed cleanly across several thread counts.

2. **Single-thread correctness**
   Runs a simple `ggml_add` graph with one thread and checks the numeric result.

3. **Multi-thread correctness**
   Runs a larger `ggml_add` graph with multiple threads and checks that parallel execution still produces the correct result.

4. **Consistency across thread counts**
   Confirms that changing `n_threads` changes performance characteristics only, not the output.

5. **Multiple sequential graphs on the same pool**
   Reuses one persistent pool across several dispatches and checks that each graph still computes the correct result.

6. **Pause / resume**
   Verifies that a pool created in the paused state can be resumed, used for compute, paused again, and freed without deadlock.

7. **Barrier correctness**
   Uses a chain of dependent add nodes so that each dispatch requires repeated barrier synchronization. This catches barrier races that would let one node start before the previous node fully completes. :contentReference[oaicite:1]{index=1}

8. **Variable `n_threads` on the same pool**
   Reuses one persistent pool while changing thread counts between dispatches. This is especially important for the HPX path because barrier state must match the active participant count. :contentReference[oaicite:2]{index=2}

9. **Sequential-dispatch stress**
   Repeats many chained dispatches on the same pool to expose intermittent races that do not appear in a single run. :contentReference[oaicite:3]{index=3}

10. **Thread-count edge cases**
    Checks both `n_threads = 1` and `n_threads = n_cores` to cover the barrier-bypass path and the maximum-contention path. :contentReference[oaicite:4]{index=4}

11. **Disposable threadpool path**
    Verifies the `cplan->threadpool = NULL` path, where `ggml_graph_compute()` creates and destroys a temporary pool internally. This is important because the HPX runtime lifecycle had to be made compatible with repeated disposable-pool use. :contentReference[oaicite:5]{index=5}

12. **`mul_mat` correctness across thread counts**
    Checks a more realistic `ggml_mul_mat` path against a scalar reference and verifies that numerical error stays within tolerance across several thread counts. :contentReference[oaicite:6]{index=6}

13. **`mul_mat` repeated reuse stress**
    Repeats the same `mul_mat` workload on the same pool to make sure correctness survives reuse, not just one isolated dispatch. :contentReference[oaicite:7]{index=7}

### Additional focused tests in `hpx-bench/`

Beyond the main 13-test contract suite, the benchmark harness includes several smaller focused tests and repros. These exist to isolate specific failure modes, especially the reusable-dispatch stalls that showed up in tiny workloads.

- `test_abort_then_reuse_same_pool.c`  
  Checks whether a pool remains usable after an abort path.

- `test_barrier_churn.c`  
  Stresses repeated barrier activity and repeated synchronization cycles.

- `test_disposable_threadpool_repeat.c`  
  Repeats the disposable-pool path many times to catch runtime lifecycle bugs.

- `test_hpx_primitives.cpp`  
  Focused checks for the HPX-side synchronization primitives and assumptions.

- `test_mixed_workload_reuse_same_pool.c`  
  Reuses the same pool across different workload shapes to catch stale state between dispatches.

- `test_mul_mat_variable_thread_counts_same_pool.c`  
  Combines `mul_mat` with changing thread counts on one persistent pool.

- `test_pause_resume_stress.c`  
  Repeats pause/resume cycles under load.

- `test_reuse_hang.c`  
  Targeted repro for the tiny reusable-dispatch stall / hang behavior.

- `test_startup_race_min.c`  
  Minimal repro for the original suspected startup-race theory.

- `test_threshold.c`  
  Probes whether behavior changes around a small-workload or dispatch-size threshold.

- `test_timing.c`  
  Lightweight timing-focused harness for quick experiments.

- `test_two_pools_isolation.c`  
  Verifies that two thread pools do not interfere with each other.

- `test_warmup_effect.c`  
  Measures or reproduces sensitivity to first-run versus warmed-up execution.

### Microbenchmark and runner files

Some files in `hpx-bench/` are not correctness tests, but part of the benchmarking harness:

- `bench_dispatch_overhead.c`  
  Tiny-workload microbenchmark used to magnify dispatch, wakeup, and barrier overhead.

- `benchmark.sh`  
  Scripted benchmark runner for build variants and sweeps.

- `test_correctness.sh`  
  Convenience script to run the correctness and stress checks.

- `test_ggml_threadpool_common.h`  
  Shared helper definitions used by the test harness.

### Why these tests matter

The point of this suite is to separate three different questions:

1. **Is the HPX backend correct?**
2. **Is it robust under reuse, pause/resume, varying thread counts, and disposable-pool paths?**
3. **Only after that: is it fast?**

That separation matters because the project already showed that a backend can be numerically correct yet still have pathological runtime behavior on tiny repeated dispatches. The correctness suite guards the interface contract; the focused repro tests guard against known failure modes; and the microbenchmarks then measure overhead on top of a backend that has already passed the robustness gate.
