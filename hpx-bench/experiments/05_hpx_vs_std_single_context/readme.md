# HPX vs std comparison — `llama-serving-bench`

This experiment compares the HPX backend against the already-green std reference baseline for one narrow shape:

```text
single context
single concurrent request
1 warmup + 5 measured requests
16 generated tokens
TinyLlama Q4_K_M
greedy / argmax decoding
CPU-only
```

This is not a benchmark and not a speed claim.

## 1. Question

Can `llama-serving-bench --backend hpx` produce the canonical generated-token hash:

```text
0x833045f1e2ebf49f
```

and show a clean HPX lifecycle under the same single-context, single-concurrent shape as the std reference?

A second check verifies that the HPX-capable binary run with `--backend std` does not start HPX, even when `LLAMA_SERVING_BENCH_HPX_TRACE=1` is set.

## 2. Shape

Common workload:

```text
model:          /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:         "Hello, my name is"
n_contexts:     1
n_concurrent:   1
n_requests:     6
warmup:         req[0]
measured:       req[1..5]
max_tokens:     16
ctx_size:       2048
batch_size:     512
n_threads:      4
seed_base:      1234
```

HPX run:

```text
binary:   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
backend:  --backend hpx
env:      LLAMA_SERVING_BENCH_HPX_TRACE=1
```

HPX-off regression:

```text
binary:   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
backend:  --backend std
env:      LLAMA_SERVING_BENCH_HPX_TRACE=1
```

Std reference baseline:

```text
local/baselines/comparison_aligned/bench/
```

The std reference directory was read as an existing baseline and was not modified by this experiment.

## 3. Artifacts

Experiment root:

```text
local/baselines/comparison_hpx_vs_std/
```

Top-level files:

```text
FACTS.md
README.md
_run_hpx.py
_run_hpx_off_regression.py
_summarize_hpx_timing.py
_summarize_comparison.py
comparison_summary.txt
```

HPX run artifacts:

```text
hpx/bench.stdout
hpx/bench.stderr
hpx/bench.exit_code.txt
hpx/per_repeat.csv
hpx/request_summary.txt
hpx/summary.txt
hpx/process_wall_ms.txt
hpx/binary_used.txt
hpx/build_info.txt
hpx/git_head.txt
hpx/hpx_trace.txt
```

HPX-off regression artifacts:

```text
hpx_off_regression/bench.stdout
hpx_off_regression/bench.stderr
hpx_off_regression/bench.exit_code.txt
hpx_off_regression/per_repeat.csv
hpx_off_regression/request_summary.txt
hpx_off_regression/summary.txt
hpx_off_regression/process_wall_ms.txt
hpx_off_regression/binary_used.txt
hpx_off_regression/build_info.txt
hpx_off_regression/git_head.txt
hpx_off_regression/hpx_trace.txt
```

Build/setup logs also live in this directory:

```text
configure_hpx.stdout
configure_hpx.stderr
build_hpx.stdout
build_hpx.stderr
configure_llama_hpx_on.stdout
configure_llama_hpx_on.stderr
build_llama_hpx_on.stdout
build_llama_hpx_on.stderr
```

## 4. Acceptance gates

Per-backend gates:

```text
hpx/summary.txt contains OVERALL: PASS
hpx_off_regression/summary.txt contains OVERALL: PASS
comparison_summary.txt contains OVERALL: PASS
```

HPX lifecycle gates:

```text
exactly 1 hpx_runtime_start_once: starting (os_threads=1)
exactly 1 engine_hpx ready: n_contexts=1 pool_size=1
exactly 1 hpx_runtime_stop: stopping
for each req[0..5], exactly 1 acquire ctx=0
for each req[0..5], exactly 1 release ctx=0
for each req[0..5], acquire precedes release
```

HPX-off regression gates:

```text
hpx_off_regression/summary.txt contains OVERALL: PASS
hpx_off_regression/hpx_trace.txt is empty
canonical hash is produced for all 6 requests
```

Cross-backend structural-fidelity gates:

```text
HPX hash == HPX-off regression hash == std reference hash
all hashes equal 0x833045f1e2ebf49f
all n_tokens_generated == 16
```

Timing is descriptive only and is not a gate.

## 5. Result

All three summaries passed:

```text
hpx/summary.txt:                OVERALL: PASS
hpx_off_regression/summary.txt: OVERALL: PASS
comparison_summary.txt:         OVERALL: PASS
```

HPX lifecycle gates passed:

```text
trace_lines_total = 15
1 start
1 ready
1 stop
6 acquire
6 release
```

Expanded lifecycle result:

```text
exactly 1 hpx_runtime_start_once: starting (os_threads=1)
exactly 1 engine_hpx ready: n_contexts=1 pool_size=1
exactly 1 hpx_runtime_stop: stopping
req[0..5] each acquired ctx=0 once
req[0..5] each released ctx=0 once
acquire precedes release for each request
```

HPX-off regression trace result:

```text
hpx_off_regression/hpx_trace.txt: 0 bytes, 0 non-empty lines
```

So the HPX-capable binary does not start HPX when invoked with `--backend std`.

Canonical hash result:

```text
0x833045f1e2ebf49f
```

All 18 token-hash slots matched the canonical value:

```text
3 sources × 6 requests = 18 matching hashes
```

Sources:

```text
hpx
hpx_off_regression
std reference: local/baselines/comparison_aligned/bench/
```

## 6. Descriptive timing

Measured interval:

```text
total_ms - ttft_ms
```

Measured rows:

```text
req[1..5]
```

Timing table:

```text
source                                min      max      mean     median
hpx                                  145.141  148.804  146.288  145.805
std ref                              145.329  146.247  145.736  145.816
regression, HPX-on bin std backend   145.538  158.651  148.658  145.952
```

Delta of means:

```text
hpx - std_ref = +0.552 ms
```

This is not a gate. It is not a speed claim. With n=5, this is a smoke-level descriptive number only.

## 7. Interpretation

This experiment closes the first HPX-on structural-correctness and lifecycle gate for the narrow shape:

```text
n_contexts=1
n_concurrent=1
n_threads=4
n_requests=6
max_tokens=16
```

It establishes:

```text
--backend hpx produces the canonical generated-token hash
HPX starts once
engine_hpx becomes ready once
each request acquires and releases ctx=0 exactly once
HPX stops once
--backend std on the HPX-capable binary does not start HPX
```

It does not establish:

```text
HPX is faster
HPX scales
HPX improves throughput
HPX queueing behavior under contention
multi-context behavior
batching behavior
realistic serving-load behavior
```

The timing result is reassuring because HPX did not show obvious catastrophic overhead in this tiny smoke, but it is not benchmark-grade evidence.

## 8. Caveats

```text
n=5 measured per variant
single context
single concurrent request
one HPX runtime carrier
one model
one prompt
16 generated tokens
CPU-only
no contention
no batching
no pipelining
```

The canonical hash is specific to this experiment shape. It is not a universal correctness proof.

The std reference was verified indirectly by reading its still-green summary and hashes. Byte-level immutability of the directory was not asserted.
