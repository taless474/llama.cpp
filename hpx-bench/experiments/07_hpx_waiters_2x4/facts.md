# FACTS — HPX waiter-pressure slice setup

This file records current-state facts before any waiter-pressure-slice
run. It is not a benchmark report and not a design authority by itself.

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

The HPX-on `llama-serving-bench` binary exists and was used for both
the single-context structural-correctness run and the (2, 2)
concurrency run:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

It links to:

```text
@rpath/libhpx.2.dylib
@rpath/libhpx_core.dylib
```

No llama rebuild is required for this slice. No source change is in
scope for this slice.

## 3. Previous (1, 1) and (2, 2) HPX structural-correctness runs — both PASS

The (1, 1) single-context slice
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

The (2, 2) concurrency slice
(`local/baselines/comparison_hpx_concurrency/`) reported:

```text
hpx/summary.txt:           OVERALL: PASS
std_control/summary.txt:   OVERALL: PASS
comparison_summary.txt:    OVERALL: PASS
```

It established, for `(n_contexts=2, n_concurrent=2, n_requests=8, 16 tokens)`:

```text
canonical generated-token hash on all 8 requests
exactly 1 hpx_runtime_start_once: starting (os_threads=2)
exactly 1 engine_hpx ready: n_contexts=2 pool_size=2
exactly 1 hpx_runtime_stop: stopping
exactly 1 acquire and 1 release per req[0..7] with same-ctx pairing,
acquire before release
both ctx=0 and ctx=1 used; total filtered HPX trace lines == 19
HPX-capable binary at --backend std emitted zero HPX trace lines
HPX total_ms - ttft_ms mean ≈198.306 ms vs std_control mean ≈193.528 ms
(Δ +4.779 ms, descriptive only; not a performance claim)
```

Neither prior slice exercised n_concurrent > n_contexts.

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

It is a fingerprint of the generated token-ID sequence for this
experiment shape. It is not a model checksum, a file checksum, a
universal llama.cpp hash, or a performance number.

The same hash is pinned in `docs/hpx/serving_bench_acceptance.md` for
`max_tokens=16`.

## 5. Current proposed shape — (2, 4, 12)

```text
binary:        /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:         /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
n_contexts:    2
n_concurrent:  4
n_requests:    12
warmup:        none (all 12 requests are correctness-gated)
max_tokens:    16
ctx_size:      2048
batch_size:    512
n_threads:     4
seed_base:     1234
trace env:     LLAMA_SERVING_BENCH_HPX_TRACE=1   (set in BOTH runs)
```

## 6. Expected runtime invariants

From `tools/serving-bench/backend_hpx.cpp`,
`tools/serving-bench/main.cpp`, and the carrier formula
`os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts))`:

```text
expected os_threads:        max(1, min(4, 2)) = 2
expected pool_size:         2
expected ctx ids in use:    {0, 1} (both must appear; alternation is NOT required)
expected filtered HPX trace count: 27
   = 1 hpx_runtime_start_once
   + 1 engine_hpx ready
   + 1 hpx_runtime_stop
   + 12 req[i] acquire
   + 12 req[i] release
```

Same-ctx pairing per request (release ctx == acquire ctx) is
structurally guaranteed by `context_guard` in `backend_hpx.cpp` and is
gated as observable evidence.

Implementation pointers (read-only, recorded for reference; no source
change in scope):

```text
hpx_context_pool::acquire()  enqueues a fresh hpx::promise on
                             std::deque waiters_ when no ctx is idle
                             (no trace event emitted at this point)
hpx_context_pool::release()  pops waiters_ from the front (FIFO) and
                             set_value's the held ctx onto that waiter
                             (no trace event emitted at this point)
hpx_context_pool::close()    drains waiters_ with set_exception; any
                             still-queued requests would route to
                             finish_error in their .then continuation
                             (not exercised in this slice)
context_guard dtor           always calls release() and traces
                             "[serving-bench] req[i] release ctx=N"
                             (this is the only release-side trace today)
```

## 7. Scope of this slice

This slice tests:

```text
no-starvation under n_concurrent > n_contexts: every submitted request
  reaches both an acquire and a release event
capacity correctness at saturation (4 in-flight, 2 contexts)
canonical-hash determinism across 12 requests on shared contexts
RAII same-ctx acquire/release pairing across 12 requests
HPX runtime lifecycle remains 1 start / 1 ready / 1 stop with
  os_threads=2 (no extra carriers when n_concurrent > n_contexts)
HPX-capable binary at --backend std staying trace-free under the same
  shape
```

This slice does NOT test:

```text
FIFO ordering of the waiter queue — backend_hpx.cpp does not emit
  'req[i] queued' or 'req[i] wake' trace events; the FIFO claim cannot
  be observed from the current trace and is explicitly NOT made here
true overlap of in-flight requests in time — trace is not timestamped
per-request wait time before acquire — no 'queued' event to anchor it
HPX scaling, batching, throughput parity, realistic serving load
n_concurrent >> n_contexts (e.g. 8x2, 16x2) — out of scope
cancellation under backpressure — out of scope
performance — no timing field is gated and no benchmark claim follows
```
