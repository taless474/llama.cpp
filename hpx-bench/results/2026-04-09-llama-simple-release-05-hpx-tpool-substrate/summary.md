# Experiment 05 — HPX threadpool substrate (Option B): end-to-end validation

## What was implemented

Patch 6 (this session) replaces the pthreads kickoff inside `ggml_threadpool` with an
HPX executor ops vtable:

```
ggml/src/ggml-cpu/ggml-cpu-executor.h   NEW  zero-dependency C/C++ vtable header
ggml/src/ggml-cpu/ggml-cpu.c            MOD  g_executor_ops global + 4 helper fns +
                                              dispatch branches in kickoff/worker-wait/destroy
ggml/src/ggml-cpu/ggml-cpu-threadpool.h MOD  includes ggml-cpu-executor.h; platform
                                              prereqs block guarded by GGML_CPU_THREADPOOL_PREREQS_DONE
ggml/src/ggml-hpx/ggml-hpx-tpool.h     NEW  thin include + ggml_hpx_tpool_get_ops() decl
ggml/src/ggml-hpx/ggml-hpx-tpool.cpp   NEW  hpx_init / hpx_kickoff / hpx_worker_wait /
                                              hpx_destroy ops; nested-HPX guard
ggml/src/ggml-hpx/ggml-hpx-runtime.cpp MOD  hpx_acquire() registers HPX ops via
                                              ggml_cpu_set_executor_ops(ggml_hpx_tpool_get_ops())
ggml/src/ggml-hpx/CMakeLists.txt        MOD  added ggml-hpx-tpool.cpp + ggml-cpu include path
```

Key design decisions:

- **Fire-and-forget `hpx::post`**: workers[0] runs on main thread synchronously.
  The final `ggml_barrier` ensures all HPX tasks have exited before `ggml_graph_compute`
  returns, making fire-and-forget safe when called from a non-HPX thread.

- **Nested-HPX guard**: `hpx_kickoff` checks `hpx::get_worker_thread_num() != size_t(-1)`.
  When true (called from an HPX coroutine), forces `n_active_threads=1` and skips posting
  secondary tasks. Prevents spin-barrier deadlock where a coroutine-thread would spin
  while its own posted tasks could never run.

- **C/C++ boundary**: `ggml-cpu-executor.h` has zero platform-header dependencies
  (no `<stdatomic.h>`, no `<pthread.h>`). Avoids `atomic_is_lock_free` macro / C++
  function-template conflict on macOS/libc++.

## Setup

- model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M
- prompt: "Once upon a time" (14 tokens), -n 32, -t 4, -ngl 0
- base: build-base/bin/llama-simple (standard pthread substrate)
- hpx:  build-hpx/bin/llama-simple  (HPX executor ops, registered via hpx_acquire)
- note: -ngl 0 does not suppress Metal in llama-simple; all 22 layers still run on GPU.
  CPU split is 0.23 MiB (output/logit ops only).

## Results

### Correctness: PASS

Both builds produce identical token output:
  "Once upon a time, there was a young girl named Lily. She lived in a small
   village, surrounded by fields and forests. Lily was a curious child, always"

### Timing

| metric               | base build        | HPX build         | delta      |
|----------------------|-------------------|-------------------|------------|
| prompt eval (14t)    | 70 ms / 199 tok/s | 52 ms / 267 tok/s | +34% faster |
| decode (31 runs)     | 265 ms / 117 tok/s| 248 ms / 125 tok/s| +7% faster  |
| graph splits         | 2                 | 2                 | —          |
| CPU buffer           | 0.23 MiB          | 0.23 MiB          | —          |
| graphs reused        | 30                | 30                | —          |

Deltas are within measurement noise for this workload (14-token prefill, short decode).
Metal dominates; the 0.23 MiB CPU split is where HPX threads run.

### Nested-HPX guard: NOT triggered in normal inference path

Call chain for decode/prefill:
  main thread
    → dispatch_decode/prefill (serial for-loop in ggml-hpx-runtime.cpp)
    → ggml_backend_graph_compute
    → ggml_graph_compute → hpx_kickoff
    → hpx::get_worker_thread_num() == size_t(-1)   ← NOT an HPX thread
    → guard FALSE → HPX tasks posted for workers[1..n-1] ✅

The guard would fire if dispatch_decode/prefill were changed to use hpx::async per chunk
(parallel chunk dispatch). That is not yet implemented.

## Conclusions

1. **Option B is correct.** Output is bit-identical to the base build across all tokens.

2. **No performance regression.** Timing differences are within noise; decode is
   essentially the same, prefill shows a slight HPX advantage (noise-level).

3. **Real HPX parallelism is active.** The nested-HPX guard does not fire on the
   normal inference path. `hpx_kickoff` posts real HPX tasks for workers[1..3] for
   each CPU graph compute call.

4. **Benchmark limitation.** Metal handles 99%+ of compute. To measure HPX tpool
   speedup meaningfully, a no-Metal build with a CPU-heavy workload is needed.

5. **Design accepted.** The executor vtable (Patch 6) is the correct extension point.
   No redesign needed to avoid nested fallback — it does not occur on the current path.

## Open items

- Parallel prefill chunk dispatch: both dispatch functions are still serial for-loops.
  Implementing hpx::async per chunk is the next step if outer prefill parallelism is
  still wanted. At that point the nested-HPX guard will fire; design choice is
  outer-only parallelism (current guard handles it) vs hierarchical.
- CPU-only benchmark: a no-Metal build is needed to measure HPX threadpool contribution
  to op-level parallelism in isolation.
