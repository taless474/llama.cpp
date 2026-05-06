# FACTS — HPX-vs-std performance matrix protocol setup

This file records current-state facts before any matrix run. It is not
a benchmark report and not a design authority by itself.

## 1. HPX install state

HPX is built and installed:

```text
HPX install dir:  /Users/Ashk/Desktop/HPX/hpx-install/
HPX cmake config: /Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX/HPXConfig.cmake
HPX libs:         /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx.dylib
                  /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx_core.dylib
```

No HPX rebuild is required for this protocol.

## 2. HPX-on llama binary

The HPX-on `llama-serving-bench` binary exists and was used for the
prior single-context, (2, 2), and (2, 4, 12) correctness slices:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

It links to:

```text
@rpath/libhpx.2.dylib
@rpath/libhpx_core.dylib
```

No llama rebuild is required for this protocol. No source change is in
scope for this protocol.

## 3. Prior PASS evidence — context only, not reused as trials

The matrix protocol is independent of prior smoke results. Prior PASS
runs are listed here for context only; they will NOT be folded into
the matrix as trials. All four cells of the matrix are run from scratch
under this protocol, with K=31 trials per layer per backend.

```text
local/baselines/comparison_hpx_vs_std/
  shape: (n_contexts=1, n_concurrent=1, n_requests=6, max_tokens=16)
  result: hpx/, hpx_off_regression/, comparison_summary.txt all OVERALL: PASS
  scope: single-context structural correctness

local/baselines/comparison_hpx_concurrency/
  shape: (n_contexts=2, n_concurrent=2, n_requests=8, max_tokens=16)
  result: hpx/, std_control/, comparison_summary.txt all OVERALL: PASS
  scope: multi-context capacity correctness at n_concurrent == n_contexts

local/baselines/comparison_hpx_waiters/
  shape: (n_contexts=2, n_concurrent=4, n_requests=12, max_tokens=16)
  result: hpx/, std_control/, comparison_summary.txt all OVERALL: PASS
  scope: capacity correctness / no-starvation under n_concurrent > n_contexts;
         FIFO ordering NOT claimed (no queued/wake trace events emitted today)
```

These three slices establish that the four cells of the matrix are
known to be structurally correct at n=6/8/12 single-trial scale. The
matrix protocol re-tests correctness at K=31 per cell per backend per
layer with a different schedule (paired interleaving, fixed seed) and
adds a separate timing layer. None of the prior runs' artifacts are
read by the matrix.

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

## 5. Matrix shape

```text
A: n_contexts=1, n_concurrent=1, n_requests=6
B: n_contexts=2, n_concurrent=2, n_requests=8
C: n_contexts=2, n_concurrent=4, n_requests=12   (waiter-pressure cell)
D: n_contexts=4, n_concurrent=4, n_requests=16
```

Common shape, pinned across all cells:

```text
binary:        /HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:         /HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
max_tokens:    16
ctx_size:      2048
batch_size:    512
n_threads:     4
seed_base:     1234
canonical hash: 0x833045f1e2ebf49f
```

## 6. Protocol parameters

```text
trial granularity:        process-level (one binary invocation per trial)
trials per cell × backend × layer:  K = 31
trials retained for timing per cell × backend:  30 (trial 0 dropped)
schedule seed:            1234
schedule scope:           per cell × layer (fresh schedule per layer; same seed)
schedule recording:       _schedule.json written before any trial runs
layer ordering per cell:  correctness layer first; timing layer only if correctness PASS
cell ordering:            A → B → C → D (block by condition)
within-layer ordering:    randomized, std/hpx interleaved (paired)
total invocations:        4 cells × 2 backends × 2 layers × 31 trials = 496
estimated wall (warm):    ~20–30 minutes on an idle machine
```

## 7. Two-layer expectations

### Correctness layer (trace ON)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE=1
gates per trial: full common gates + lifecycle / pool / trace gates (hpx variant)
                 inverted-trace gate (std variant)
purpose:         prove cell × backend is structurally correct under our shape
output:          per-trial artifacts + correctness_summary.txt + layer_pass.txt
timing use:      none — this layer's per-request timings are NOT pooled
                 into the timing summary
```

Expected per-trial filtered HPX trace count, hpx variant only:

```text
A: 1 start + 1 ready + 1 stop + 6  acquire + 6  release = 15
B: 1 start + 1 ready + 1 stop + 8  acquire + 8  release = 19
C: 1 start + 1 ready + 1 stop + 12 acquire + 12 release = 27
D: 1 start + 1 ready + 1 stop + 16 acquire + 16 release = 35
```

Expected ctx-id usage, hpx variant only:

```text
A: ctx ids in {0}; only ctx=0 exists by definition
B: set of acquire ctx ids == {0, 1}
C: set of acquire ctx ids == {0, 1}
D: set of acquire ctx ids == {0, 1, 2, 3}
```

Expected `os_threads` and `pool_size`, hpx variant only:

```text
A: os_threads=1, pool_size=1
B: os_threads=2, pool_size=2
C: os_threads=2, pool_size=2  (no extra carriers under waiter pressure)
D: os_threads=4, pool_size=4
```

Expected std variant trace, all cells:

```text
0 matching HPX lifecycle / pool lines
```

### Timing layer (trace OFF)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE unset
gates per trial: reduced common gates (hash, status, n_tokens, aggregate counts,
                 prompt-fits line, no error/failed); NO lifecycle / pool / trace
                 gates (no trace lines emitted)
purpose:         measure per-request and per-trial timing free of HPX-only
                 fprintf trace overhead
output:          per-trial artifacts + per_trial_summary.csv + all_requests.csv
                 + timing_summary.txt + condition_comparison.txt
trial 0:         correctness-checked AND excluded from timing aggregation
trials measured: 30 (trials 1..30) per backend per cell
```

## 8. Scope of this protocol

This protocol measures:

```text
per-request latency (total_ms, ttft_ms, total_minus_ttft_ms) under
  paired std-vs-hpx interleaving, with median and tail (p95/p99) reported
process_wall_ms per trial
aggregate_tokens_per_sec per trial
within-cell std-vs-hpx timing delta (median, p95, p99, stdev, CV)
HPX correctness (lifecycle + pool + RAII pairing) at K=31 trials per cell
```

This protocol does NOT measure:

```text
general llama.cpp performance (this is a measurement of the
  serving-bench harness on this machine, this build, this model)
HPX scheduler quality in general
performance of any other prompt, generation length, model, quantization,
  or context size
performance of any deployment shape outside the four matrix cells
performance under realistic mixed workloads, multi-prompt traffic,
  KV eviction, or cancellation
behavior at K > 30; sub-1% deltas inside the CV band are not
  interpretable as "different" and require a follow-on protocol
```


