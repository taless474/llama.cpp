# FACTS — HPX-on serving-bench setup

This file records current-state facts after the first HPX-on structural-correctness run. It is not a benchmark report and not a design authority by itself.

## 1. Current llama build state

There are now two llama build directories:

```text
HPX-OFF baseline: /Users/Ashk/Desktop/HPX/builds/llama-base/
HPX-ON build:     /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/
```

The HPX-OFF baseline build has:

```text
LLAMA_BUILD_SERVING_BENCH=ON
LLAMA_SERVING_BENCH_HPX=OFF
```

Its current baseline binaries are:

```text
/Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-server
/Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-parallel
/Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-cli
/Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-serving-bench
```

The HPX-ON build has:

```text
LLAMA_BUILD_SERVING_BENCH=ON
LLAMA_SERVING_BENCH_HPX=ON
```

Its HPX-on serving-bench binary exists:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

The HPX-on binary links to HPX libraries:

```text
@rpath/libhpx.2.dylib
@rpath/libhpx_core.dylib
```

The HPX-OFF `llama-base/` build must remain untouched because existing PASS baseline artifacts depend on it.

## 2. HPX source and install state

HPX source is present:

```text
/Users/Ashk/Desktop/HPX/hpx-master/
```

HPX source git state from the setup probe:

```text
HEAD: 4d2663869b
status: clean
```

HPX is built and installed:

```text
HPX build dir:    /Users/Ashk/Desktop/HPX/hpx-master-build/
HPX install dir:  /Users/Ashk/Desktop/HPX/hpx-install/
```

The install contains:

```text
/Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX/HPXConfig.cmake
/Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx.dylib
/Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx_core.dylib
```

Install size:

```text
84M
```

HPX was configured with Homebrew/system dependencies:

```text
Boost: /opt/homebrew/opt/boost/lib/cmake/Boost-1.90.0
hwloc: /opt/homebrew/opt/hwloc/lib/libhwloc.dylib
Asio:  /opt/homebrew/include, version 1.36.0
malloc: system
```

The HPX-on llama build resolved HPX from:

```text
HPX_DIR=/Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX
```

So `find_package(HPX REQUIRED)` is no longer blocked.

## 3. Code-state facts

`tools/serving-bench/backend_hpx.cpp` is implemented in this branch. It is not a stub.

For the current single-request experiment shape, `engine_hpx` mirrors the std backend's important correctness path:

```text
tokenize
decode
argmax token selection
generated token hash path
```

The stale comment in `tools/serving-bench/main.cpp` saying the HPX engine is “not yet implemented” is not design truth anymore. It became stale after:

```text
63bae8a49 tools: add HPX-native serving backend and gates
```

Use current code and current result files over stale comments.

## 4. Existing acceptance-gate facts

HPX-on serving-bench acceptance gates are documented in:

```text
docs/hpx/serving_bench_acceptance.md
```

The key structural-fidelity gate is Gate 4:

```text
std hash == hpx hash == 0x833045f1e2ebf49f
```

The hash:

```text
0x833045f1e2ebf49f
```

is the canonical generated-token hash for this exact shape:

```text
model:     tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:    "Hello, my name is"
tokens:    16 generated tokens
decode:    greedy / argmax
CPU-only:  yes
shape:     single-context, single-concurrent
```

It is not a model checksum, file checksum, universal llama.cpp hash, or performance number.

It is a compact fingerprint of the generated token-ID sequence for this experiment shape.

## 5. Current baseline result facts

The following are already PASS and should not be modified by HPX setup:

```text
local/baselines/server/
local/baselines/server_repeat/
local/baselines/server_timing/
local/baselines/serving_bench_std_timing/
local/baselines/comparison_aligned/
```

The aligned comparison established:

```text
llama-server and llama-serving-bench std produced the same generated token IDs
canonical hash: 0x833045f1e2ebf49f
thread count: server = 4, bench = 4
```

The aligned inner-generation interval pair was:

```text
llama-server timings.predicted_ms
vs
llama-serving-bench std (total_ms - ttft_ms)
```

Measured mean values in that aligned smoke:

```text
llama-server predicted_ms mean:         146.076 ms
llama-serving-bench total_ms-ttft mean: 145.736 ms
```

This validates the comparison basis for the narrow std/upstream shape. It does not prove any HPX performance claim.

## 6. HPX-on structural-correctness run facts

The first HPX-on structural-correctness comparison completed and passed.

Result files:

```text
local/baselines/comparison_hpx_vs_std/hpx/summary.txt
local/baselines/comparison_hpx_vs_std/hpx_off_regression/summary.txt
local/baselines/comparison_hpx_vs_std/comparison_summary.txt
```

All three summaries contain:

```text
OVERALL: PASS
```

The HPX lifecycle gates passed:

```text
exactly 1 hpx_runtime_start_once: starting (os_threads=1)
exactly 1 engine_hpx ready: n_contexts=1 pool_size=1
exactly 1 hpx_runtime_stop: stopping
for each req[0..5]: exactly 1 acquire ctx=0
for each req[0..5]: exactly 1 release ctx=0
for each req[0..5]: acquire precedes release
```

Lifecycle trace count:

```text
trace_lines_total = 15 = 1 start + 1 ready + 1 stop + 6 acquire + 6 release
```

The HPX-off regression used the HPX-capable binary with `--backend std` and `LLAMA_SERVING_BENCH_HPX_TRACE=1`.

Regression trace result:

```text
local/baselines/comparison_hpx_vs_std/hpx_off_regression/hpx_trace.txt
0 bytes
0 non-empty lines
```

So HPX remained opt-in by backend selection.

All HPX, regression, and std-reference token hashes matched the canonical hash:

```text
0x833045f1e2ebf49f
```

This means all three sources produced the same generated token-ID sequence for this fixed experiment shape.

## 7. Runtime-thread facts

For the completed first HPX-on experiment:

```text
n_contexts = 1
n_concurrent = 1
```

The HPX runtime carrier budget was:

```text
os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts)) = 1
```

Observed lifecycle trace confirmed:

```text
hpx_runtime_start_once: starting (os_threads=1)
```

This is independent of llama inference threads.

The llama inference thread count was:

```text
--n-threads 4
```

So do not confuse:

```text
HPX runtime carriers:              1
llama inference threads/context:   4
```

The first HPX-on experiment was therefore a lifecycle and structural-fidelity test, not a scheduling-throughput test.

## 8. Descriptive timing facts

Timing was descriptive only and not a gate.

Measured interval:

```text
total_ms - ttft_ms
```

Measured rows:

```text
req[1..5]
```

Results:

```text
hpx mean:     146.288 ms
std ref mean: 145.736 ms
delta:        +0.552 ms
```

This does not support a speed claim. It only says the HPX backend did not show obvious catastrophic overhead in this narrow n=5 smoke.

The run did not test concurrency, batching, scheduling pressure, scaling, or realistic serving load.

## 9. Build decision already made

The HPX build used Homebrew/system dependencies, not fetched dependencies:

```text
HPX_WITH_FETCH_BOOST=OFF
HPX_WITH_FETCH_HWLOC=OFF
HPX_WITH_FETCH_ASIO=OFF
HPX_WITH_MALLOC=system
```

This completed successfully. No dependency flag change is currently needed.




