# Serving bench FIFO-pool closeout

This document consolidates the serving-bench acceptance checks, local benchmark results, and FIFO context-pool closeout for `llama-serving-bench`.

The scope is narrow:

```text
llama-serving-bench --backend std
vs.
llama-serving-bench --backend hpx
```

Both backends use the same core inference path:

```text
same model
same prompt
same llama_decode path
same greedy argmax
same request shape
same context-pool shape
same token-hash correctness checks
```

This is a closeout for the tested CPU-only TinyLlama serving harness. It does **not** claim that HPX is generally slower, that HPX cannot help LLM serving, or that HPX cannot help `llama.cpp` in a different design.

---

## 1. Purpose and acceptance philosophy

A serving-bench result is meaningful only after correctness and lifecycle checks are clean.

The acceptance checks prove that std and HPX backends are structurally comparable before any performance interpretation. Generated-token hashes are used as structural correctness signals. They are not semantic quality metrics and not performance metrics.

A valid run must preserve:

```text
backend selection correctness
prompt-fit validation
greedy decode determinism
generated-token hash stability
clean HPX-OFF behavior
HPX runtime lifecycle correctness
std-vs-HPX token-stream equality
context-pool correctness under contention
```

These checks are not benchmarks.

---

## 2. Canonical smoke input and pinned hashes

Canonical smoke input:

```text
Model:  TinyLlama 1.1B Chat Q4_K_M
Prompt: Hello, my name is
Decode: greedy
```

Pinned hash values:

```text
max_tokens=16:
  generated_token_hash = 0x833045f1e2ebf49f
  n_tokens_generated   = 16

max_tokens=0:
  generated_token_hash = 0x0000000000000000
  n_tokens_generated   = 0

max_tokens=32:
  generated_token_hash = 0x6794e47fe0f84af1
```

The 32-token hash is informational. It helps catch a degenerate constant-hash bug.

---

## 3. Architecture under test

The std backend is a FIFO context-pool orchestrator using standard C++ synchronization around a pool of prewarmed `llama_context` objects.

The HPX backend is also a FIFO context-pool orchestrator around the same pool shape. It uses HPX futures/promises and HPX lifecycle management, but still leases one `llama_context` per active request and runs the normal opaque `llama_decode` path inside that leased context.

The important architectural constraint:

```text
HPX is outside llama_decode.
HPX is not inside ggml graph execution.
HPX does not lower ggml nodes.
HPX does not rewrite kernels.
HPX does not change the matmul / quantized GEMV path.
```

The HPX backend changes the serving wake/lease mechanism. It does not add priority scheduling, cancellation, work stealing, distributed execution, or richer future-composition across a multi-stage request pipeline.

---

## 4. Acceptance gates

### 4.1 HPX-OFF / std baseline checks

These checks apply to the default build where HPX support is off.

| Test | Purpose | Expected result |
|---|---|---|
| Backend default is std | `--help` documents `--backend std|hpx` and shows `std` as default | backend selection is explicit |
| Unknown backend rejected | invalid backend exits nonzero before model load | config errors stay separate from model execution |
| std smoke | canonical prompt, `max_tokens=16` | `n_ok=1`, `n_error=0`, hash `0x833045f1e2ebf49f` |
| empty generation | `max_tokens=0` | zero tokens, zero hash |
| same prompt stable | same std command twice | same token count and hash |
| changed generation changes hash | e.g. `max_tokens=32` | hash `0x6794e47fe0f84af1` |
| HPX-OFF fails cleanly | `--backend hpx` in HPX-OFF build | exits before model load with clear HPX-not-built message |

HPX-OFF expected stderr:

```text
[serving-bench] HPX backend not built (LLAMA_SERVING_BENCH_HPX=OFF)
```

Source-truth checkpoint:

```text
Tests 1-7: PASS
std canonical hash: 0x833045f1e2ebf49f
empty hash: 0x0000000000000000
32-token informational hash: 0x6794e47fe0f84af1
HPX-OFF stub: clean exit before model load
```

### 4.2 HPX-ON gates

The HPX backend under test is HPX-native in the FIFO-pool sense:

```text
HPX async request state machine
future-based context pool
exclusive llama_context lease through context_guard
normal opaque llama_decode inside the leased context
no permanent worker-per-context loop
no central CV/mutex worker queue
```

| Gate | Purpose | Expected result |
|---|---|---|
| Build / link | HPX-ON configures, compiles, links, resolves HPX | build passes |
| std path regression | run std from HPX-capable binary | canonical hash, no HPX runtime start trace |
| HPX smoke | one canonical request through `--backend hpx` | `n_ok=1`, nonzero hash, HPX ready trace when tracing |
| std vs HPX hash equality | canonical request on both backends | both hash to `0x833045f1e2ebf49f` |
| empty generation | `--backend hpx --max-tokens 0` | zero tokens/hash; no acquire/release trace |
| HPX repeat stability | canonical HPX request twice | same token count/hash |
| two-context HPX smoke | multiple requests with two contexts | all hashes canonical, no corruption |
| lifecycle smoke | trace backend lifecycle | std stays cold; hpx starts/stops once |
| pool queueing under real decode | more concurrent requests than contexts | no deadlock; queued requests wake and complete |

Lifecycle trace evidence under `LLAMA_SERVING_BENCH_HPX_TRACE=1`:

```text
[serving-bench] hpx_runtime_start_once: starting (os_threads=1)
[serving-bench] engine_hpx ready: n_contexts=1 pool_size=1
[serving-bench] hpx_runtime_stop: stopping
```

Context-pool acquire/release trace from the one-context / four-request queueing gate:

```text
[serving-bench] req[0] acquire ctx=0
[serving-bench] req[0] release ctx=0
[serving-bench] req[1] acquire ctx=0
[serving-bench] req[1] release ctx=0
[serving-bench] req[2] acquire ctx=0
[serving-bench] req[2] release ctx=0
[serving-bench] req[3] acquire ctx=0
[serving-bench] req[3] release ctx=0
```

This supports the contract:

```text
requests lease exactly one context
the context is released after each request
queued requests reuse the same context without deadlock
acquire/release counts match
--backend std does not start HPX, even in an HPX-capable build
```

---

## 5. Local serving-bench result summary

Correctness status before benchmark interpretation:

```text
OFF Tests 1-7: PASS
HPX-ON Gates 1, 2, 3, 4, 5, 6, 7, 9: PASS
std hash == hpx hash == 0x833045f1e2ebf49f
```

These results are local engineering evidence, not final performance claims.

### 5.1 First local sanity benchmark

Single-run matrix after Slice 5 correctness passed:

```text
A: 1 context / 1 concurrent / 8 requests / 16 tokens
B: 2 contexts / 2 concurrent / 16 requests / 16 tokens
C: 1 context / 4 concurrent / 16 requests / 16 tokens
Backends: std, hpx
Repeats: 1 per cell
```

Result files:

```text
local/bench_small/{A,B,C}_{std,hpx}.{stdout,stderr}
```

All six runs had:

```text
n_error=0
n_cancelled=0
canonical hash counts matched expectations
```

First-pass aggregate throughput:

| Shape | std agg tok/s | HPX agg tok/s |
|---|---:|---:|
| A | 58.87 | 72.61 |
| B | 118.56 | 123.69 |
| C | 65.70 | 74.31 |

This single-shot matrix showed no obvious HPX overhead in that run. It is not final performance evidence.

### 5.2 Repeated local sanity benchmark

Repeated benchmark matrix:

| Shape | n_contexts | n_concurrent | n_requests | max_tokens | expected canonical-hash count |
|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | 8 | 16 | 8 |
| B | 2 | 2 | 16 | 16 | 16 |
| C | 1 | 4 | 16 | 16 | 16 |

Backends: `std`, `hpx`.

Repeats: five per `(shape, backend)` cell, 30 total runs.

Result files:

```text
local/bench_repeat/
```

Correctness summary:

```text
30 stdout files
30 stderr files
all runs: n_cancelled=0 and n_error=0
all A runs: canonical hash count = 8
all B runs: canonical hash count = 16
all C runs: canonical hash count = 16
git status before and after the matrix was identical
```

Five-repeat summary, mean ± sample standard deviation:

| Shape | Backend | wall s | agg tok/s | TTFT p50 | TTFT p95 | total p50 | total p95 | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | std | 1.31 ± 0.17 | 98.90 ± 12.24 | 22.63 ± 3.16 | 38.82 ± 8.24 | 141.70 ± 8.33 | 251.63 ± 92.95 | 0.1771 ± 0.0939 |
| A | hpx | 1.23 ± 0.11 | 104.64 ± 9.17 | 20.85 ± 1.05 | 66.87 ± 63.11 | 140.32 ± 4.74 | 205.95 ± 63.76 | 0.1293 ± 0.0746 |
| B | std | 1.83 ± 0.02 | 139.85 ± 1.84 | 34.17 ± 1.88 | 40.70 ± 3.69 | 224.33 ± 5.74 | 245.67 ± 11.48 | 0.0461 ± 0.0214 |
| B | hpx | 1.97 ± 0.06 | 130.29 ± 4.11 | 34.20 ± 1.22 | 57.08 ± 13.79 | 227.69 ± 2.72 | 340.28 ± 68.92 | 0.1247 ± 0.0330 |
| C | std | 2.50 ± 0.17 | 102.78 ± 7.44 | 441.42 ± 30.04 | 660.82 ± 134.34 | 562.33 ± 34.55 | 780.95 ± 135.65 | 0.6453 ± 0.0384 |
| C | hpx | 2.45 ± 0.27 | 105.57 ± 11.07 | 434.06 ± 18.53 | 614.39 ± 227.68 | 554.81 ± 22.79 | 737.63 ± 231.87 | 0.5736 ± 0.0248 |

Reading:

```text
Shape A: HPX had slightly higher mean aggregate throughput, but both paths had noisy outliers.
Shape B: std had higher mean aggregate throughput and tighter p95 total latency than HPX.
Shape C: HPX and std were close on aggregate throughput; HPX had lower mean TTFT p50 and total p50, but both paths showed high tail variability under queueing.
```

Supported limited claims:

```text
The HPX backend is structurally correct for the tested matrix.
The repeated benchmark completed without request errors or hash drift.
HPX does not show catastrophic overhead in these small local runs.
The performance picture is workload-dependent and needs broader testing.
```

Unsupported claims:

```text
HPX is faster than std.
HPX is better than upstream llama-server.
The current HPX thread-count formula is optimal.
The current small TinyLlama matrix predicts larger-model behavior.
```

---

## 6. FIFO-pool experiment closeout

The later FIFO-pool experiments are the actual closeout evidence for the HPX-as-context-pool direction.

| Experiment | Workload | Correctness result | Performance result | Conclusion |
|---|---|---|---|---|
| `08_perf_hpx_vs_std_matrix` | fixed-shape HPX-vs-std matrix | PASS | HPX indistinguishable or modestly slower in most cells; no reliable speedup | HPX correctness is solid, but fixed-shape CPU TinyLlama serving does not show an HPX advantage |
| `09_perf_thread_sweep` | C_2x4 thread sensitivity sweep | PASS | mixed trend; HPX overhead not explained cleanly by kernel-thread oversubscription alone | tuning `n_threads` does not turn FIFO HPX into a faster path |
| `10_perf_heterogeneous_budgets` | mixed request budgets under mild waiter pressure | PASS | `hpx_short_worse`; short median `total_ms` +1.9967%; makespan +1.00% | HPX handles heterogeneous budgets correctly but does not improve latency or makespan |
| `11_perf_deep_queue_short_requests` | deep queue of uniform 8-token requests under 16:1 waiter pressure | PASS | `hpx_worse`; short median `total_ms` +2.07%; p99 +1.20%; makespan -0.14% | wake/queue overhead amplified, but HPX still does not outperform std |

### 6.1 Experiment 08: HPX-vs-std matrix

Shape:

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

Protocol:

```text
correctness_trace_on
timing_trace_off
K=31 process-level trials per backend/cell/layer
trial 0 excluded from timing aggregation
```

Correctness:

```text
MATRIX_OVERALL_CORRECTNESS: PASS
16 / 16 layer summaries PASS
4 / 4 condition summaries PASS
496 / 496 raw trials completed
```

Median `total_ms` deltas:

```text
A_1x1: +0.08%
B_2x2: +4.33%
C_2x4: +7.68%
D_4x4: -1.43%, noisy/mixed; no speedup claim
```

Conclusion: HPX was correct across the tested matrix, but no reliable speedup was established.

### 6.2 Experiment 09: thread sensitivity sweep

Fixed shape:

```text
n_contexts   = 2
n_concurrent = 4
n_requests   = 12
```

Sweep:

```text
n_threads = 1, 2, 4
```

Protocol:

```text
2 backends: std, hpx
2 layers: correctness_trace_on, timing_trace_off
K=11 process-level trials per backend/thread/layer
trial 0 excluded from timing aggregation
total = 132 invocations
```

Correctness:

```text
SWEEP_OVERALL_CORRECTNESS: PASS
12 / 12 layer summaries PASS
3 / 3 thread-setting summaries PASS
132 / 132 raw trials completed
```

Median `total_ms` deltas:

```text
n_threads=1: +1.32%
n_threads=2: +7.14%
n_threads=4: +3.73%
```

Conclusion: trend was mixed. The result did not show that HPX overhead was only oversubscription, and it did not identify a tuning point where FIFO HPX reliably beat std.

### 6.3 Experiment 10: heterogeneous budgets

Path:

```text
hpx-bench/experiments/10_perf_heterogeneous_budgets/
```

Shape:

```text
n_contexts   = 2
n_concurrent = 4
n_requests   = 12
n_threads    = 2
plan         = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]
```

This creates 2:1 waiter pressure.

Completion-order-safe correctness gates:

```text
multiset(n_tokens_generated) == multiset(plan)
per-budget hash consistency
std-vs-hpx hash equality by budget class
```

Correctness:

```text
EXPERIMENT_OVERALL: PASS
44 / 44 invocations completed
0 n_tok multiset mismatches
std-vs-hpx per-budget hash equality: PASS
```

Canonical hashes:

```text
8  -> 0x0619d4d1900c2365
16 -> 0x833045f1e2ebf49f
32 -> 0x6794e47fe0f84af1
64 -> 0x7efcc69b4edadb70
```

Timing result:

```text
branch label: hpx_short_worse

short.total_ms median:
  std = 633.03 ms
  hpx = 645.67 ms
  delta = +12.64 ms / +1.9967%

short.total_ms p95:
  delta = -0.24%

medium.total_ms median:
  std = 979.89 ms
  hpx = 995.48 ms
  delta = +15.59 ms / +1.59%

long.total_ms median:
  std = 1566.96 ms
  hpx = 1561.85 ms
  delta = -5.11 ms / -0.33%

makespan median:
  std = 1598.37 ms
  hpx = 1614.35 ms
  delta = +15.98 ms / +1.00%

queue_drain_ms median:
  std = 16.99 ms
  hpx = 45.05 ms
  delta = +28.06 ms / +165.19%
```

The large relative queue-drain delta should be read carefully because the absolute difference was small and the metric was noisy.

Conclusion: HPX handled heterogeneous budgets correctly, but did not show latency or makespan advantage over std in this mixed-budget CPU TinyLlama workload.

### 6.4 Experiment 11: deep queue short requests

Path:

```text
hpx-bench/experiments/11_perf_deep_queue_short_requests/
```

Shape:

```text
n_contexts   = 2
n_concurrent = 32
n_requests   = 200
n_threads    = 2
plan         = [8] * 200
```

This creates 16:1 waiter pressure. During the bulk phase, two requests hold contexts while roughly 30 requests wait; each request generates only eight tokens, so wake/acquire/release overhead is more visible than in longer decode workloads.

Correctness:

```text
EXPERIMENT_OVERALL: PASS
44 / 44 invocations PASS
all four layers PASS
```

Budget-8 canonical hash:

```text
8 -> 0x0619d4d1900c2365
```

HPX trace correctness:

```text
expected trace lines = 403
11 / 11 HPX correctness trials matched this expectation
context pool ids observed: {0, 1}
```

Timing deltas, `hpx - std`:

```text
short.total_ms median: +2.07%
short.total_ms p90:    +0.13%
short.total_ms p95:    +0.43%
short.total_ms p99:    +1.20%
makespan median:       -0.14%
```

The `hpx_worse` branch label was triggered by the median threshold:

```text
short.total_ms median delta = +2.07% >= +1.0%
```

The p99 threshold did not trigger:

```text
short.total_ms p99 delta = +1.20% < +2.0%
```

Conclusion: under the cleanest tested workload for exposing per-request wake/queue overhead, HPX still showed about +2% median per-request overhead and no makespan advantage.

---

## 7. Interpretation and decision

The experiments consistently show:

```text
Correctness: strong
Performance: no reliable HPX advantage over std
```

The likely reason is structural. The std backend and HPX backend are too similar:

```text
Both are FIFO context-pool orchestrators.
Both lease one llama_context per active request.
Both run the same opaque llama_decode path.
Both preserve the same token stream.
Neither changes ggml execution.
Neither changes the matmul / quantized GEMV path.
Neither adds priority, cancellation, work stealing, distributed placement, or multi-stage future composition.
```

The HPX design mostly changes the mechanism used to wake a waiting request. That wake event occurs once per request, while the request still spends most time inside `llama.cpp` / ggml compute. Since the HPX backend does not add a new scheduling capability, there is little surface where it can outperform std.

Safe summary:

```text
HPX is correct here, but this HPX design is not adding a capability that std lacks.
```

Close this line of work as a performance path:

```text
Do not keep tuning HPX as a FIFO serving-level context-pool replacement.
```

Additional tuning over `n_threads`, `n_contexts`, or queue depth is likely to remeasure the same basic result: the HPX backend is correct, but it does not beat the simpler std backend when both are FIFO context pools around opaque `llama_decode`.

Closeout thesis:

```text
HPX as a FIFO serving-level context-pool replacement is correct and robust, but it does not outperform the simpler std backend across the tested CPU-only TinyLlama serving workloads.
```

---

## 8. What remains valuable

This line produced useful infrastructure and evidence:

```text
shared model + multiple prewarmed context pool
std and hpx serving backends with matched request semantics
HPX lifecycle correctness checks
trace-based acquire/release validation
context-pool pairing checks
per-request budget plans via --max-tokens-plan
completion-order-safe correctness gates
canonical token-hash checks
correctness-first benchmark protocol
trace-on and trace-off layers
trial-0 exclusion from timing aggregation
reusable experiment schedules, runners, and summarizers
```

Those pieces remain useful for future HPX-native serving experiments.

---

## 9. What invalidates benchmark interpretation

Do not interpret a benchmark result if any of the following is true:

```text
n_error > 0
n_cancelled > 0
generated-token hash count is wrong
std and HPX use different prompts, models, token counts, or context shapes
source changed between std and HPX runs
HPX lifecycle traces show std accidentally starting HPX
the run is a smoke test rather than a repeated benchmark
```

Passing the acceptance checks does not prove:

```text
HPX is faster than std
HPX is better than upstream llama-server
the pool-of-contexts design beats continuous batching
the HPX thread-count formula is optimal
```

---

## 10. Next viable HPX-native directions

The next HPX effort should add a capability that the std FIFO pool does not currently provide.

### 10.1 Cancellation

Recommended first.

Cancellation gives a concrete user-visible feature and a metric where HPX may have a natural advantage through future composition and cancellation propagation.

Possible metrics:

```text
cancel-to-stop latency
wasted tokens after cancellation
context-release correctness after cancellation
cancellation under waiter pressure
```

Caveat:

```text
Requires a decode-loop polling hook or cancellation point.
```

### 10.2 Priority scheduling

Priority scheduling could let short-budget or high-priority requests skip ahead of long-running requests.

Possible metrics:

```text
short-request latency improvement
priority inversion checks
fairness / starvation bounds for long requests
```

### 10.3 Future-composition serving API

A more HPX-native serving API could model requests as composed stages:

```text
admission
prefill
decode
postprocess
cancellation
metrics
```

### 10.4 Distributed serving

Distributed placement is closer to HPX's natural strengths than a single-machine FIFO context pool, but changes project scope and requires a different benchmark model.

---

## 11. Non-goals and safe closeout language

Do not use this closeout to claim:

```text
HPX is slower generally.
HPX cannot help LLM serving.
HPX cannot help llama.cpp.
HPX would not help with priority scheduling, cancellation, distributed serving, or richer future composition.
```

Do not overstate profiler-like claims unless measured. Prefer:

```text
decode is expected to be dominated by llama.cpp/ggml compute, and this design does not change that path.
```

Avoid:

```text
decode is 95% matmul
```

unless profiler evidence is added.

Final closeout statement:

```text
The FIFO serving-pool experiments are complete. The HPX backend is correct, robust, and well-instrumented, but the current design does not create a performance advantage over the simpler std backend. Future HPX work should move away from FIFO replacement and toward HPX-native capabilities such as cancellation, priority scheduling, richer future composition, or distributed serving.
```
