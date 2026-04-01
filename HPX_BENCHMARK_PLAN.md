# HPX Benchmarking Plan

Goal: benchmark HPX threading vs the existing pthread-based thread pool in llama.cpp
on Apple M4, using Llama 3.1 8B Instruct Q4_K_M as the benchmark model, across two
BLAS conditions.

---

## Motivation

### Pair A: pthreads vs HPX with GGML_BLAS=ON (realistic baseline)

When Accelerate/BLAS is enabled (the default on macOS), large matrix multiplications
are offloaded to Apple's Accelerate framework. The ggml thread pool handles only the
non-GEMM ops (softmax, RoPE, layer norm, etc.) — a small fraction of total compute.
This pair answers: **does HPX help in real-world inference conditions?**

### Pair B: pthreads vs HPX with GGML_BLAS=OFF (pure threading comparison)

With BLAS disabled, ggml's thread pool is responsible for all compute including GEMM.
The thread pool overhead and scheduling quality become the dominant factor.
This pair answers: **how much does HPX's scheduler matter when threading is the bottleneck?**

Together the two pairs isolate whether any HPX gain is from better scheduling of
the ggml thread pool, or only visible when BLAS hides the threading cost.

---

## Repository structure

```
llama-hpx/
├── ggml/src/ggml-cpu/
│   ├── ggml-cpu.c              ← thread pool implementation (pthread-based)
│   ├── ggml-cpu-hpx.cpp        ← NEW: HPX thread pool (to be created, Step 4)
│   └── CMakeLists.txt          ← add HPX conditional linking here
├── ggml/CMakeLists.txt         ← add GGML_HPX option here
├── models/
│   ├── ggml-vocab-*.gguf       ← tracked (tokenizer test fixtures, part of repo)
│   ├── templates/              ← tracked
│   ├── README.md               ← tracked (download instructions for benchmark models)
│   └── llama3.1-8b/            ← gitignored, download manually (see models/README.md)
├── results/                    ← gitignored, benchmark CSV outputs
├── build-pthread-blas/         ← gitignored, pthreads + BLAS ON
├── build-hpx-blas/             ← gitignored, HPX + BLAS ON
├── build-pthread-noblas/       ← gitignored, pthreads + BLAS OFF
├── build-hpx-noblas/           ← gitignored, HPX + BLAS OFF
└── HPX_BENCHMARK_PLAN.md       ← this file
```

---

## Threading model in ggml-cpu.c

The thread pool lives in `ggml/src/ggml-cpu/ggml-cpu.c`. It is **pthread-based**, not
`std::thread`. Key components:

- `struct ggml_threadpool` — holds worker threads, sync primitives, atomic state
- `ggml_thread_create()` — wraps `pthread_create` (Unix) / `CreateThread` (Windows)
- `ggml_graph_compute_secondary_thread()` — worker loop (lines 3070–3107)
- `ggml_barrier()` — synchronization barrier
- Hybrid sync strategy: busy-wait polling + `pthread_mutex` + `pthread_cond_broadcast`

### pthread → HPX equivalents

| Current (pthread) | HPX equivalent |
|---|---|
| `ggml_thread_create()` | `hpx::async` / HPX thread pool |
| `ggml_graph_compute_secondary_thread()` | HPX task function |
| `ggml_barrier()` | `hpx::latch` or `hpx::barrier` |
| `pthread_mutex_t` | `hpx::mutex` |
| `pthread_cond_t` + `pthread_cond_broadcast` | `hpx::condition_variable` |

---

## Integration plan

**New file:** `ggml/src/ggml-cpu/ggml-cpu-hpx.cpp`

Implements the same `ggml_threadpool` interface as `ggml-cpu.c` but using HPX primitives.
Compiled in place of the pthread implementation when `-DGGML_HPX=ON`.

**CMake changes:**
- `ggml/CMakeLists.txt` — add `option(GGML_HPX "use HPX for threading" OFF)`
- `ggml/src/ggml-cpu/CMakeLists.txt` — conditionally link HPX, swap source file

**Two separate build dirs** keep baseline and HPX builds independent:
- `build-baseline/` — unmodified pthread build
- `build-hpx/` — HPX build

---

## Benchmark steps

### Step 1: Download the model

See `models/README.md` for download instructions.

### Step 2: Build all four variants

```bash
# Pair A — BLAS ON (realistic, Accelerate handles GEMM)
cmake -B build-pthread-blas \
  -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-pthread-blas -j --target llama-bench

cmake -B build-hpx-blas \
  -DGGML_METAL=OFF -DGGML_OPENMP=OFF -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-blas -j --target llama-bench

# Pair B — BLAS OFF (ggml thread pool handles all compute)
cmake -B build-pthread-noblas \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-pthread-noblas -j --target llama-bench

cmake -B build-hpx-noblas \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-noblas -j --target llama-bench
```

### Step 3: Implement HPX thread pool

Create `ggml/src/ggml-cpu/ggml-cpu-hpx.cpp` implementing the `ggml_threadpool`
interface with HPX primitives. Update CMake as described in `HPX_IMPL_PLAN.md`.

### Step 4: Run thread sweeps

Common flags for all runs:
- `-ngl 0` — CPU only, no Metal offload
- `-t 1,2,3,4,5,6,7,8,9,10` — full sweep across Apple M4's 10 cores
- `-p 512` — prompt processing (pp): batch matmul, scales well with threads
- `-n 128` — token generation (tg): memory-bandwidth bound, scales less

```bash
mkdir -p results

# Pair A
./build-pthread-blas/bin/llama-bench \
  -m models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 1,2,3,4,5,6,7,8,9,10 -p 512 -n 128 -r 3 \
  -o csv | tee results/pthread_blas_sweep.csv

./build-hpx-blas/bin/llama-bench \
  -m models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 1,2,3,4,5,6,7,8,9,10 -p 512 -n 128 -r 3 \
  -o csv | tee results/hpx_blas_sweep.csv

# Pair B
./build-pthread-noblas/bin/llama-bench \
  -m models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 1,2,3,4,5,6,7,8,9,10 -p 512 -n 128 -r 3 \
  -o csv | tee results/pthread_noblas_sweep.csv

./build-hpx-noblas/bin/llama-bench \
  -m models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 0 -t 1,2,3,4,5,6,7,8,9,10 -p 512 -n 128 -r 3 \
  -o csv | tee results/hpx_noblas_sweep.csv
```

### Step 5: Compare results

```bash
# Pair A
python scripts/compare-llama-bench.py \
  results/pthread_blas_sweep.csv \
  results/hpx_blas_sweep.csv

# Pair B
python scripts/compare-llama-bench.py \
  results/pthread_noblas_sweep.csv \
  results/hpx_noblas_sweep.csv
```

---

## Results table template

### Pair A — BLAS ON (avg_ts, t/s)

| threads | pthread pp | hpx pp | Δ pp | pthread tg | hpx tg | Δ tg |
|---------|-----------|--------|------|-----------|--------|------|
| 1       |           |        |      |           |        |      |
| 2       |           |        |      |           |        |      |
| 4       |           |        |      |           |        |      |
| 6       |           |        |      |           |        |      |
| 8       |           |        |      |           |        |      |
| 10      |           |        |      |           |        |      |

### Pair B — BLAS OFF (avg_ts, t/s)

| threads | pthread pp | hpx pp | Δ pp | pthread tg | hpx tg | Δ tg |
|---------|-----------|--------|------|-----------|--------|------|
| 1       |           |        |      |           |        |      |
| 2       |           |        |      |           |        |      |
| 4       |           |        |      |           |        |      |
| 6       |           |        |      |           |        |      |
| 8       |           |        |      |           |        |      |
| 10      |           |        |      |           |        |      |

---

## What to look for

| Metric | Expected behavior |
|--------|------------------|
| pp (prompt processing) | Scales up to ~4 perf cores, then diminishing returns |
| tg (token generation) | Memory-bandwidth bound, flattens early (~4 threads) |
| Pair A Δ | Small — BLAS dominates, HPX only affects non-GEMM ops (~10% of compute) |
| Pair B Δ | Larger — HPX scheduler directly affects all GEMM parallelism |
| Knee point | Thread count where adding more hurts or plateaus |

The key question: is there a meaningful HPX gain in Pair B that is absent in Pair A?
If yes, HPX helps the ggml thread pool but gets hidden by BLAS in practice.
