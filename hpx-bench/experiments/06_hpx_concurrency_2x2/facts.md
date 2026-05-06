# FACTS — HPX concurrency slice setup

This file records current-state facts before any concurrency-slice run. It is
not a benchmark report and not a design authority by itself.

## 1. HPX install state

HPX is built and installed:

```text
HPX install dir:  /Users/Ashk/Desktop/HPX/hpx-install/
HPX cmake config: /Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX/HPXConfig.cmake
HPX libs:         /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx.dylib
                  /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx_core.dylib
```

No HPX rebuild is required for this slice.

## 2. HPX-on llama binary

The HPX-on `llama-serving-bench` binary exists and was used for the
single-context structural-correctness run:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

It links to:

```text
@rpath/libhpx.2.dylib
@rpath/libhpx_core.dylib
```

No llama rebuild is required for this slice.

## 3. Previous (1, 1) HPX structural-correctness run — PASS

The prior single-context slice
(`local/baselines/comparison_hpx_vs_std/`) reported:

```text
hpx/summary.txt:                 OVERALL: PASS
hpx_off_regression/summary.txt:  OVERALL: PASS
comparison_summary.txt:          OVERALL: PASS
```

It established, for `(n_contexts=1, n_concurrent=1, n_requests=6, 16 tokens)`:

```text
canonical generated-token hash produced on every request
exactly 1 hpx_runtime_start_once: starting (os_threads=1)
exactly 1 engine_hpx ready: n_contexts=1 pool_size=1
exactly 1 hpx_runtime_stop: stopping
exactly 1 acquire ctx=0 and 1 release ctx=0 per request, acquire before release
HPX-capable binary at --backend std emitted zero HPX trace lines
```

The single-context run did not exercise multi-context capacity or any form of
in-flight concurrency.

## 4. Canonical hash

```text
0x833045f1e2ebf49f
```

This is the canonical generated-token hash for the fixed shape:

```text
model:   tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:  "Hello, my name is"
tokens:  16 generated tokens
decode:  greedy / argmax
CPU:     yes
```

It is a fingerprint of the generated token-ID sequence for this experiment
shape. It is not a model checksum, a file checksum, a universal llama.cpp
hash, or a performance number.

## 5. Current proposed shape — (2, 2)

```text
binary:        /HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:         /HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
n_contexts:    2
n_concurrent:  2
n_requests:    8
warmup:        none (all 8 requests are correctness-gated)
max_tokens:    16
ctx_size:      2048
batch_size:    512
n_threads:     4
seed_base:     1234
trace env:     LLAMA_SERVING_BENCH_HPX_TRACE=1   (set in BOTH runs)
```

## 6. Expected runtime invariants

From `tools/serving-bench/backend_hpx.cpp` and the carrier formula
`os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts))`:

```text
expected os_threads:        2
expected pool_size:         2
expected ctx ids in use:    {0, 1} (both must appear; alternation is NOT required)
expected filtered HPX trace count: 19
   = 1 hpx_runtime_start_once
   + 1 engine_hpx ready
   + 1 hpx_runtime_stop
   + 8 req[i] acquire
   + 8 req[i] release
```

Same-ctx pairing per request (release ctx == acquire ctx) is structurally
guaranteed by `context_guard` in `backend_hpx.cpp` and is gated as observable
evidence.

## 7. Scope of this slice

This slice tests:

```text
multi-context capacity at n_contexts=2
two-in-flight capacity at n_concurrent=2
canonical-hash determinism across 8 requests on independent contexts
RAII same-ctx acquire/release pairing
HPX-capable binary at --backend std staying trace-free at higher concurrency
```

This slice does NOT test:

```text
waiter pressure (n_concurrent > n_contexts) — that is Gate 9 / a later slice
true overlap of in-flight requests in time — trace is not timestamped
HPX scaling, batching, throughput parity
realistic serving-load behavior
```
