# HPX waiter-pressure / no-starvation correctness

This experiment validates the next step after the `(2,2)` HPX multi-context capacity pass.

The previous `(2,2)` run proved:

```text
n_contexts=2
n_concurrent=2
both contexts reachable
HPX lifecycle clean
canonical hash matched
```

This run asks whether the HPX backend still preserves correctness and lifecycle invariants when there are more concurrent in-flight requests than available contexts.

This is a correctness and lifecycle smoke. It is not a performance benchmark.

## 1. Question

Can `llama-serving-bench --backend hpx` preserve canonical-hash correctness, same-context acquire/release pairing, and clean HPX lifecycle when configured with:

```text
n_contexts=2
n_concurrent=4
```

Specifically, this slice checks:

```text
all requests complete
no request starves
all requests acquire and release a context
every request releases the same context it acquired
both context slots are reused
HPX starts once
engine_hpx becomes ready once
HPX stops once
all generated token hashes match the canonical hash
```

This slice intentionally does **not** prove FIFO waiter ordering because the current trace does not emit `queued` or `wake` events.

## 2. Shape

Common workload:

```text
binary:         /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:          /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:         "Hello, my name is"
n_contexts:     2
n_concurrent:   4
n_requests:     12
max_tokens:     16
ctx_size:       2048
batch_size:     512
n_threads:      4
seed_base:      1234
trace env:      LLAMA_SERVING_BENCH_HPX_TRACE=1
canonical hash: 0x833045f1e2ebf49f
```

HPX run:

```text
--backend hpx
artifacts: local/baselines/comparison_hpx_waiters/hpx/
```

std-control run:

```text
--backend std
artifacts: local/baselines/comparison_hpx_waiters/std_control/
```

The std-control run uses the same HPX-capable binary with `LLAMA_SERVING_BENCH_HPX_TRACE=1`. It must remain trace-free to prove HPX is opt-in by backend selection.

## 3. Derived expectations

For this shape:

```text
os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts))
                    = max(1, min(4, 2))
                    = 2
```

Expected HPX lifecycle trace:

```text
hpx_runtime_start_once: starting (os_threads=2)
engine_hpx ready: n_contexts=2 pool_size=2
hpx_runtime_stop: stopping
12 acquire lines
12 release lines
```

Expected filtered trace count:

```text
1 start + 1 ready + 1 stop + 12 acquire + 12 release = 27
```

Expected context usage:

```text
ctx ids used = {0, 1}
```

No alternation is required. No exact request-to-context distribution is required.

## 4. Artifacts

Experiment root:

```text
local/baselines/comparison_hpx_waiters/
```

Top-level files:

```text
README.md
FACTS.md
_run_hpx_waiters.py
_run_std_control_waiters.py
_summarize_hpx_waiters.py
_summarize_waiters_comparison.py
comparison_summary.txt
```

HPX artifacts:

```text
hpx/bench.stdout
hpx/bench.stderr
hpx/bench.exit_code.txt
hpx/process_wall_ms.txt
hpx/per_repeat.csv
hpx/request_summary.txt
hpx/binary_used.txt
hpx/build_info.txt
hpx/git_head.txt
hpx/hpx_trace.txt
hpx/summary.txt
```

std-control artifacts:

```text
std_control/bench.stdout
std_control/bench.stderr
std_control/bench.exit_code.txt
std_control/process_wall_ms.txt
std_control/per_repeat.csv
std_control/request_summary.txt
std_control/binary_used.txt
std_control/build_info.txt
std_control/git_head.txt
std_control/hpx_trace.txt
std_control/summary.txt
```

Shell-side logs:

```text
local/run_hpx_waiters.{stdout,stderr}
local/run_std_control_waiters.{stdout,stderr}
local/summarize_hpx_waiters_hpx.{stdout,stderr}
local/summarize_hpx_waiters_std_control.{stdout,stderr}
local/summarize_waiters_comparison.{stdout,stderr}
```

## 5. Acceptance gates

### Per-backend gates

Applied to both `hpx/` and `std_control/`:

```text
harness exit code == 0
12 request rows parsed
n_ok=12 n_cancelled=0 n_error=0
all 12 status == ok
all 12 n_tokens_generated == 16
all 12 generated_token_hash == 0x833045f1e2ebf49f
prompt-fits line present
stderr has no [serving-bench] error
stderr has no failed
per_repeat.csv has 12 data rows + header
```

### HPX lifecycle / capacity gates

Applied to `hpx/`:

```text
exactly 1 hpx_runtime_start_once: starting (os_threads=2)
exactly 1 engine_hpx ready: n_contexts=2 pool_size=2
exactly 1 hpx_runtime_stop: stopping
for req[0..11], exactly 1 acquire line
for req[0..11], exactly 1 release line
for req[0..11], release ctx == acquire ctx
for req[0..11], acquire precedes release
both ctx=0 and ctx=1 are used
total filtered HPX trace lines == 27
all 12 requests reached both an acquire and release event
```

Explicitly not gated:

```text
FIFO waiter ordering
request wait time
queued/wake sequence
request ordering
ctx alternation
ctx usage balance
true timestamped overlap
performance delta
```

### std-control trace gate

Applied to `std_control/`:

```text
std_control/hpx_trace.txt has 0 non-empty lines
```

### Cross-backend gates

Applied by `comparison_summary.txt`:

```text
hpx/summary.txt contains OVERALL: PASS
std_control/summary.txt contains OVERALL: PASS
std_control/hpx_trace.txt is empty
HPX and std_control hashes match per request
all 24 hashes equal canonical 0x833045f1e2ebf49f
all 24 n_tokens_generated values are 16
HPX lifecycle / capacity gates passed
```

Timing is descriptive only and is not a gate.

## 6. Result

All summaries passed:

```text
hpx/summary.txt:          OVERALL: PASS
std_control/summary.txt:  OVERALL: PASS
comparison_summary.txt:   OVERALL: PASS
```

HPX lifecycle result:

```text
hpx_runtime_start_once (os_threads=2):              1
engine_hpx ready (n_contexts=2 pool_size=2):        1
hpx_runtime_stop: stopping:                         1
acquire lines:                                      12
release lines:                                      12
total filtered HPX trace lines:                     27
both ctx=0 and ctx=1 used:                          yes
same-ctx pairing per request:                       yes
acquire precedes release per request:               yes
```

Observed acquire to release context mapping:

```text
req[0]  = ctx 1
req[1]  = ctx 0
req[2]  = ctx 1
req[3]  = ctx 0
req[4]  = ctx 1
req[5]  = ctx 0
req[6]  = ctx 1
req[7]  = ctx 0
req[8]  = ctx 1
req[9]  = ctx 0
req[10] = ctx 1
req[11] = ctx 0
```

Each request released the same context it acquired.

The trace included interleaving such as:

```text
req[0] acquire ctx=1
req[1] acquire ctx=0
req[0] release ctx=1
req[2] acquire ctx=1
```

This is structural evidence that the context pool continued to service excess in-flight work with two reusable contexts. The trace is event-ordered, not timestamped, so this is not a true timing-overlap claim and not a FIFO-ordering proof.

std-control trace result:

```text
std_control/hpx_trace.txt non-empty lines: 0
```

The HPX-capable binary running with `--backend std` emitted zero matching HPX lifecycle or pool lines, even with `LLAMA_SERVING_BENCH_HPX_TRACE=1` set.

Hash result:

```text
HPX hashes:         all 0x833045f1e2ebf49f
std_control hashes: all 0x833045f1e2ebf49f
canonical hash:     0x833045f1e2ebf49f
```

All 24 generated-token hash slots matched the canonical value:

```text
12 HPX requests + 12 std_control requests = 24 matching hashes
```

All 24 requests generated 16 tokens.

## 7. No-starvation / capacity result

This slice passed the no-starvation / capacity-correctness gate.

Observed result:

```text
all 12 HPX requests reached acquire and release
aggregate line: n_ok=12 n_cancelled=0 n_error=0
both context slots stayed reachable
set of acquire ctx ids = {0, 1}
excess in-flight requests were serviced to completion through capacity 2
```

This supports the claim:

```text
HPX handles n_concurrent > n_contexts without starvation or correctness loss for this smoke shape.
```

It does not support the stronger claim:

```text
FIFO waiter ordering is proven.
```

FIFO cannot be proven with the current trace because the trace does not include `queued` or `wake` events.

## 8. Descriptive timing

Timing is **not gated**.

Measured interval:

```text
total_ms - ttft_ms
```

All rows:

```text
source        n    min        max        mean       median
HPX           12   184.641    198.378    191.438    191.284
std_control   12   182.052    202.316    190.083    189.358
```

Mean delta:

```text
HPX - std_control = +1.355 ms
```

Trim view, req[4..11]:

```text
source        n   min        max        mean       median
HPX           8   184.641    195.459    191.378    191.970
std_control   8   184.339    192.171    188.613    189.358
```

Trim mean delta:

```text
HPX - std_control = +2.765 ms
```

These deltas are descriptive only. They are not a performance claim.

Two observations from the run:

```text
total_ms - ttft_ms stayed relatively tight on both backends
ttft_ms jumped for queued requests, as expected for n_concurrent=4 and n_contexts=2
```

Those observations are useful sanity checks, but they are not measurement contracts.

## 9. Interpretation

This slice closes the waiter-pressure / no-starvation correctness gate for the narrow shape:

```text
n_contexts=2
n_concurrent=4
n_requests=12
```

It establishes:

```text
HPX correctness holds when n_concurrent > n_contexts for this smoke shape
HPX runtime carrier count follows the expected formula: os_threads=2
engine_hpx reports n_contexts=2 pool_size=2
both context slots are reused
each request releases the same context it acquired
all requests complete successfully
canonical generated-token hash is preserved across both backends
std_control remains HPX-trace-free
```

It does not establish:

```text
FIFO waiter ordering
exact per-request wait time
true timestamped overlap of decode calls
HPX is faster
HPX scales generally
HPX improves throughput
behavior at n_concurrent >> n_contexts
cancellation under backpressure
realistic serving-load behavior
multi-prompt behavior
KV eviction behavior
batching or pipelining behavior
```

## 10. Caveats

This is smoke-grade evidence:

```text
n=12 per backend
single model
single prompt
16 generated tokens
CPU-only
2 contexts
4 concurrent requests
n_threads=4 per context
no queued/wake trace events
no timestamped trace
```

Oversubscription is expected:

```text
2 contexts × 4 kernel threads = 8 kernel threads
```

on a 4-core machine. This may slow each decode and makes timing deltas noisy.

The canonical hash:

```text
0x833045f1e2ebf49f
```

is specific to this exact TinyLlama / `"Hello, my name is"` / 16-token greedy CPU shape. It is not a model checksum and not a universal correctness proof.

Reference integrity is a discipline note, not an automated byte-level gate. The helpers were designed not to read or write prior baseline directories.
