# HPX concurrency / multi-context correctness

This experiment validates the next step after the single-context HPX structural-correctness pass.

The single-context run proved:

```text
n_contexts=1
n_concurrent=1
HPX lifecycle clean
canonical hash matched
```

This run asks whether the HPX backend still preserves correctness and lifecycle invariants when two contexts are available and two requests may be in flight.

This is a correctness and lifecycle smoke. It is not a performance benchmark.

## 1. Question

Can `llama-serving-bench --backend hpx` preserve canonical-hash correctness and HPX lifecycle invariants when configured with:

```text
n_contexts=2
n_concurrent=2
```

Specifically, this slice checks:

```text
both context slots are reachable
every request acquires exactly one context
every request releases the same context it acquired
HPX starts once
engine_hpx becomes ready once
HPX stops once
all generated token hashes match the canonical hash
```

This slice does **not** test waiter pressure, because:

```text
n_concurrent == n_contexts
```

The FIFO waiter path belongs to the next Gate 9 slice with:

```text
n_concurrent > n_contexts
```

## 2. Shape

Common workload:

```text
binary:         /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:          /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:         "Hello, my name is"
n_contexts:     2
n_concurrent:   2
n_requests:     8
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
artifacts: local/baselines/comparison_hpx_concurrency/hpx/
```

std-control run:

```text
--backend std
artifacts: local/baselines/comparison_hpx_concurrency/std_control/
```

The std-control run uses the same HPX-capable binary with `LLAMA_SERVING_BENCH_HPX_TRACE=1`. It must remain trace-free to prove HPX is opt-in by backend selection.

## 3. Derived expectations

For this shape:

```text
os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts))
                    = max(1, min(2, 2))
                    = 2
```

Expected HPX lifecycle trace:

```text
hpx_runtime_start_once: starting (os_threads=2)
engine_hpx ready: n_contexts=2 pool_size=2
hpx_runtime_stop: stopping
8 acquire lines
8 release lines
```

Expected filtered trace count:

```text
1 start + 1 ready + 1 stop + 8 acquire + 8 release = 19
```

Expected context usage:

```text
ctx ids used = {0, 1}
```

No alternation is required. No exact request-to-context distribution is required.

## 4. Artifacts

Experiment root:

```text
local/baselines/comparison_hpx_concurrency/
```

Top-level files:

```text
README.md
FACTS.md
_run_hpx_concurrency.py
_run_std_control_concurrency.py
_summarize_hpx_concurrency.py
_summarize_concurrency_comparison.py
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
local/run_hpx_concurrency.{stdout,stderr}
local/run_std_control_concurrency.{stdout,stderr}
local/summarize_hpx_concurrency_hpx.{stdout,stderr}
local/summarize_hpx_concurrency_std_control.{stdout,stderr}
local/summarize_concurrency_comparison.{stdout,stderr}
```

## 5. Acceptance gates

### Per-backend gates

Applied to both `hpx/` and `std_control/`:

```text
harness exit code == 0
8 request rows parsed
n_ok=8 n_cancelled=0 n_error=0
all 8 status == ok
all 8 n_tokens_generated == 16
all 8 generated_token_hash == 0x833045f1e2ebf49f
prompt-fits line present
stderr has no [serving-bench] error
stderr has no failed
per_repeat.csv has 8 data rows + header
```

### HPX lifecycle gates

Applied to `hpx/`:

```text
exactly 1 hpx_runtime_start_once: starting (os_threads=2)
exactly 1 engine_hpx ready: n_contexts=2 pool_size=2
exactly 1 hpx_runtime_stop: stopping
for req[0..7], exactly 1 acquire line
for req[0..7], exactly 1 release line
for req[0..7], release ctx == acquire ctx
for req[0..7], acquire precedes release
both ctx=0 and ctx=1 are used
total filtered HPX trace lines == 19
```

Explicitly not gated:

```text
request ordering
ctx alternation
ctx usage balance
true timing overlap
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
all 16 hashes equal canonical 0x833045f1e2ebf49f
all 16 n_tokens_generated values are 16
HPX lifecycle gates passed
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
acquire lines:                                      8
release lines:                                      8
total filtered HPX trace lines:                     19
both ctx=0 and ctx=1 used:                          yes
same-ctx pairing per request:                       yes
acquire precedes release per request:               yes
```

Observed acquire to release context mapping:

```text
req[0] = ctx 1
req[1] = ctx 0
req[2] = ctx 0
req[3] = ctx 1
req[4] = ctx 0
req[5] = ctx 1
req[6] = ctx 0
req[7] = ctx 1
```

Each request released the same context it acquired.

The trace included interleaving such as:

```text
req[0] acquire ctx=1
req[1] acquire ctx=0
req[1] release ctx=0
req[0] release ctx=1
```

This is structural evidence that both context slots were reachable and live in the trace. The trace is event-ordered, not timestamped, so this is not a true timing-overlap claim.

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

All 16 generated-token hash slots matched the canonical value:

```text
8 HPX requests + 8 std_control requests = 16 matching hashes
```

All 16 requests generated 16 tokens.

## 7. Descriptive timing

Timing is **not gated**.

Measured interval:

```text
total_ms - ttft_ms
```

All rows:

```text
source        n   min       max       mean      median
HPX           8   191.949   215.221   198.306   195.983
std_control   8   186.795   199.430   193.528   194.050
```

Mean delta:

```text
HPX - std_control = +4.779 ms
```

Trim view, req[2..7]:

```text
source        n   mean       median
HPX           6   199.885    197.799
std_control   6   194.968    196.717
```

Trim mean delta:

```text
HPX - std_control = +4.917 ms
```

These deltas are descriptive only. They are not a performance claim.

## 8. Interpretation

This slice closes the `(2,2)` multi-context capacity correctness gate.

It establishes:

```text
HPX correctness holds for n_contexts=2, n_concurrent=2
HPX runtime carrier count follows the expected formula: os_threads=2
engine_hpx reports n_contexts=2 pool_size=2
both context slots are used
each request releases the same context it acquired
canonical generated-token hash is preserved across both backends
std_control remains HPX-trace-free
```

It does not establish:

```text
HPX is faster
HPX scales
HPX improves throughput
true timestamped overlap of decode calls
queue waiter correctness
behavior when n_concurrent > n_contexts
realistic serving-load behavior
multi-prompt behavior
KV eviction behavior
batching or pipelining behavior
```

## 9. Caveats

This is smoke-grade evidence:

```text
n=8 per backend
single model
single prompt
16 generated tokens
CPU-only
2 contexts
2 concurrent requests
n_threads=4 per context
no waiter pressure
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

