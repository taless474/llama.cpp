# Serving FIFO-pool closeout

## Scope

This note closes out the current HPX serving-bench direction:

```text
Use HPX as a serving-level FIFO context-pool replacement around opaque llama_decode.
```

The tested comparison is:

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

This closeout applies only to the tested CPU-only TinyLlama serving harness. It does not claim that HPX is generally slower, that HPX cannot help LLM serving, or that HPX cannot help llama.cpp in a different design.

## Architecture under test

The std backend and hpx backend are intentionally similar at the serving level.

The std backend is a FIFO context-pool orchestrator using standard C++ synchronization around a pool of prewarmed `llama_context` objects.

The hpx backend is also a FIFO context-pool orchestrator around the same pool shape. It uses HPX futures/promises and HPX lifecycle management, but it still leases one `llama_context` per active request and runs the normal opaque `llama_decode` path inside that leased context.

The important architectural constraint is:

```text
HPX is outside llama_decode.
HPX is not inside ggml graph execution.
HPX does not lower ggml nodes.
HPX does not rewrite kernels.
HPX does not change the matmul / quantized GEMV path.
```

So the current HPX backend changes the serving wake/lease mechanism, but does not add a new scheduling capability such as priority scheduling, cancellation, work stealing, distributed execution, or richer future-composition across multi-stage request pipelines.

## Evidence summary

| Experiment | Workload | Correctness result | Performance result | Conclusion |
|---|---|---|---|---|
| `08_perf_hpx_vs_std_matrix` | Fixed-shape HPX-vs-std matrix across context/concurrency cells | PASS | HPX was indistinguishable or modestly slower in most cells; no reliable speedup | HPX correctness is solid, but fixed-shape CPU TinyLlama serving does not show an HPX advantage |
| `09_perf_thread_sweep` | C_2x4 thread sensitivity sweep | PASS | Mixed trend; HPX overhead was not explained cleanly by kernel-thread oversubscription alone | Tuning `n_threads` does not turn the FIFO HPX backend into a faster path |
| `10_perf_heterogeneous_budgets` | Mixed request budgets under mild waiter pressure | PASS | Branch label `hpx_short_worse`; short median `total_ms` +1.9967%; makespan +1.00% | HPX honors heterogeneous budgets correctly but does not improve latency or makespan |
| `11_perf_deep_queue_short_requests` | Deep queue of uniform 8-token requests under 16:1 waiter pressure | PASS | Branch label `hpx_worse`; short median `total_ms` +2.07%; p99 +1.20%; makespan -0.14% | Even when wake/queue overhead is amplified, HPX does not outperform std |

## Experiment 08: HPX-vs-std matrix

Experiment 08 tested the std and hpx serving backends across a fixed-shape matrix:

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

The protocol used two layers:

```text
correctness_trace_on
timing_trace_off
```

and K=31 process-level trials per backend/cell/layer, with trial 0 excluded from timing aggregation.

Correctness passed:

```text
MATRIX_OVERALL_CORRECTNESS: PASS
16 / 16 layer summaries PASS
4 / 4 condition summaries PASS
496 / 496 raw trials completed
```

Top-line median `total_ms` deltas were:

```text
A_1x1: +0.08%
B_2x2: +4.33%
C_2x4: +7.68%
D_4x4: -1.43%, noisy/mixed; no speedup claim
```

The result established that the HPX backend was correct across the tested matrix, but it did not establish a reliable HPX speedup.

## Experiment 09: thread sensitivity sweep

Experiment 09 tested whether the C_2x4 HPX overhead was mostly caused by kernel-thread oversubscription.

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

Correctness passed:

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

The trend was mixed. The result did not support a simple explanation that HPX overhead was only an oversubscription artifact, and it did not identify a tuning point where the FIFO HPX backend reliably beat std.

## Experiment 10: heterogeneous budgets

Experiment 10 tested mixed generation budgets under mild waiter pressure.

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

This creates 2:1 waiter pressure:

```text
n_concurrent / n_contexts = 4 / 2 = 2
```

The harness output prints `req[i]` in completion order, not submission order. Because of that, Experiment 10 used completion-order-safe gates:

```text
multiset(n_tokens_generated) == multiset(plan)
per-budget hash consistency
std-vs-hpx hash equality by budget class
```

Correctness passed:

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

Conclusion from Experiment 10:

```text
The HPX backend handled heterogeneous per-request budgets correctly, but it did not show a latency or makespan advantage over std in this mixed-budget CPU TinyLlama workload.
```

## Experiment 11: deep queue short requests

Experiment 11 amplified waiter pressure and reduced per-request decode work.

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

This creates 16:1 waiter pressure:

```text
n_concurrent / n_contexts = 32 / 2 = 16
```

During the bulk phase, two requests hold contexts while roughly 30 requests wait in the queue. Each request generates only eight tokens, so wake/acquire/release overhead is more visible than in longer decode workloads.

Correctness passed:

```text
EXPERIMENT_OVERALL: PASS
44 / 44 invocations PASS
all four layers PASS
```

Budget-8 canonical hash:

```text
8 -> 0x0619d4d1900c2365
```

HPX trace correctness passed:

```text
expected trace lines = 403
11 / 11 HPX correctness trials matched this expectation
context pool ids observed: {0, 1}
```

Timing deltas, reported as `hpx - std`:

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

Makespan was effectively unchanged:

```text
makespan median delta = -0.14%
```

Conclusion from Experiment 11:

```text
Under the cleanest tested workload for exposing per-request wake/queue overhead, HPX still showed about +2% median per-request overhead and no makespan advantage.
```

This is not a broad HPX tail-latency failure. The p90, p95, and p99 deltas stayed close to std. The more precise interpretation is that the current HPX FIFO context-pool design has a small median per-request orchestration overhead without a compensating throughput or tail-latency benefit.

## Interpretation

The experiments consistently show:

```text
Correctness: strong
Performance: no reliable HPX advantage over std
```

The likely reason is structural.

The current HPX backend and std backend are too similar:

```text
Both are FIFO context-pool orchestrators.
Both lease one llama_context per active request.
Both run the same opaque llama_decode path.
Both preserve the same token stream.
Neither changes ggml execution.
Neither changes the matmul / quantized GEMV path.
Neither adds priority, cancellation, work stealing, distributed placement, or multi-stage future composition.
```

The current HPX design mostly changes the mechanism used to wake a waiting request. That wake event occurs once per request, while the request still spends most of its time inside llama.cpp/ggml compute. Since the HPX backend does not add a new scheduling capability, there is little surface area where it can outperform std.

A safe summary is:

```text
HPX is correct here, but this HPX design is not adding a capability that std lacks.
```

## Decision

Close this line of work as a performance path:

```text
Do not keep tuning HPX as a FIFO serving-level context-pool replacement.
```

The evidence does not support further sweeps of the same design. Additional tuning over `n_threads`, `n_contexts`, or queue depth is likely to remeasure the same basic result: the HPX backend is correct, but it does not beat the simpler std backend when both are FIFO context pools around opaque `llama_decode`.

This is the closeout thesis:

```text
HPX as a FIFO serving-level context-pool replacement is correct and robust, but it does not outperform the simpler std backend across the tested CPU-only TinyLlama serving workloads.
```

## What remains valuable

This line of work still produced useful infrastructure and evidence:

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

Those pieces remain valuable for future HPX-native serving experiments.

## Next viable HPX-native directions

The next HPX effort should add a capability that the std FIFO pool does not currently provide.

### 1. Cancellation

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

### 2. Priority scheduling

Priority scheduling could let short-budget or high-priority requests skip ahead of long-running requests.

Possible metrics:

```text
short-request latency improvement
priority inversion checks
fairness / starvation bounds for long requests
```

Caveat:

```text
Requires a fairness model so priority does not simply starve long requests.
```

### 3. Future-composition serving API

A more HPX-native serving API could model a request as a composition of stages rather than a single FIFO lease:

```text
admission
prefill
decode
postprocess
cancellation
metrics
```

Caveat:

```text
This is a larger design change and should be scoped separately.
```

### 4. Distributed serving

Distributed placement is closer to HPX's natural strengths than a single-machine FIFO context pool.

Caveat:

```text
This changes the project scope and requires a different benchmark model.
```

## Non-goals for the next phase

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

## Final closeout statement

The FIFO serving-pool experiments are complete. The HPX backend is correct, robust, and well-instrumented, but the current design does not create a performance advantage over the simpler std backend. Future HPX work should move away from FIFO replacement and toward HPX-native capabilities such as cancellation, priority scheduling, richer future composition, or distributed serving.
