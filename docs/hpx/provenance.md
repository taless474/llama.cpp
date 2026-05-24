# HPX serving-bench provenance

## 1. Add HPX serving-bench direction and docs

This branch starts fresh from upstream `llama.cpp`.

The previous HPX work tried to improve execution inside a single llama.cpp / ggml graph. That work produced useful evidence, but it also showed a hard limit: llama.cpp's existing scheduler and CPU kernels are already strong for single-request CPU decode, and the HPX paths we tried mostly added dispatch overhead.

The new direction is different.

Instead of asking:

```text
Can HPX make one ggml graph faster?
```

this branch asks:

```text
Can HPX improve serving-level orchestration for concurrent CPU inference requests?
```

For this branch, llama.cpp graph execution is treated as opaque. HPX should not enter individual ggml nodes, rewrite kernels, or replace the graph scheduler in the first prototype.

The initial experiment compares two orchestration strategies around the same llama.cpp workload:

```text
std::thread request scheduler
vs.
HPX request scheduler
```

Both should use the same model, the same number of contexts, the same prompts, the same thread allocation policy, and the same llama.cpp calls. The only intended difference is the orchestration runtime.


## What changed from the old HPX direction

The old direction operated inside one request:

```text
one llama_context
one decode stream
one ggml graph per token
HPX selective lowering / packets / fallback runs inside graph_compute
```

The new direction operates above llama.cpp execution:

```text
one shared llama_model
N llama_context objects
one exclusive context per active request
normal llama_decode inside each context
external scheduler chooses which request runs where
```

This avoids repeating the old anti-pattern:

```text
native llama thread -> tiny node -> hpx::async(...).get() -> next tiny node
```

The new v1 prototype does not use HPX inside ggml. It only builds the serving harness needed to ask whether HPX is useful at the request-orchestration layer.

## Design principles

1. Keep llama.cpp execution opaque.
   - Do not touch `graph_compute`.
   - Do not lower ops.
   - Do not add packets.
   - Do not add an HPX-backed ggml threadpool in v1.

2. Compare against a strong simple baseline.
   - Build the `std::thread` backend first.
   - The HPX backend must run the same workload through the same harness.
   - Do not claim HPX value without an apples-to-apples comparison.

3. Use real serving metrics.
   - Aggregate tokens/sec.
   - Time to first token.
   - Total request latency.
   - Fairness across concurrent requests.
   - Error/cancellation accounting when added.

4. Keep single-request behavior visible.
   - The `n_concurrent=1` case is the non-regression control.
   - The concurrent case is the actual HPX-serving question.

5. Use bounded context ownership.
   - A `llama_context` is mutable and single-threaded internally.
   - One active request owns one context exclusively.
   - Multiple contexts can share one model for CPU-only v1.

## Phase 0 findings

Before writing the serving harness, we inspected upstream llama.cpp to check whether the pool-of-contexts design is safe enough to prototype.

Findings:

- A `llama_context` is not safe for concurrent `llama_decode` calls.
- Multiple independent `llama_context` objects over one shared `llama_model` are appropriate for CPU-only v1.
- Each context owns its own scheduler, backend instance, threadpool, KV cache, compute buffers, and mutable decode state.
- Tokenization and detokenization APIs are documented as thread-safe.
- Context reset can use `llama_memory_clear(llama_get_memory(ctx), false)`.
- `llama_set_n_threads` writes plain integer fields, so v1 should not retune thread counts dynamically.
- v1 should set per-context thread counts statically at context construction.

Resulting v1 rule:

```text
Use static fair-share thread allocation:
n_threads_per_context = max(1, hardware_threads / n_contexts)
```

Dynamic retuning may be a later experiment, but it is not part of the first harness.


### 1.1: `tools: add serving benchmark stub`

Hash: `unknown`

Intent:

Create the smallest possible new tool target to prove that a serving benchmark can live under `tools/` and link against llama.cpp correctly.

What it added:

- A gated `LLAMA_BUILD_SERVING_BENCH=OFF` CMake option.
- A new `llama-serving-bench` executable.
- A stub `main.cpp` that initializes the llama backend, prints a banner and system info, then exits.

What it proved:

- The new tool directory is picked up only when explicitly enabled.
- `llama-common` linkage works.
- The binary lands in the expected build `bin/` location.
- No model loading, threading, HPX, or request scheduling was introduced yet.

Verification result:

```text
Build: clean.
Run: clean.
Output includes [serving-bench v0.0 stub] and llama_print_system_info().
Exit: 0.
```

Notes:

This subsection intentionally proves only build-system integration.

### 1.2: `tools: add serving benchmark harness types`

Hash: `unknown`

Intent:

Add the neutral harness surface before implementing any backend.

What it added:

- Request configuration types.
- Request result and aggregate metric types.
- Percentile and coefficient-of-variation helpers.
- Basic CLI parsing.
- A backend interface returning `std::future<request_result>`.
- Config-print behavior in the stub binary.

What it proved:

- Harness types and metric code compile cleanly.
- CLI parsing works.
- `--help` exits successfully.
- Unknown arguments fail as expected.
- Still no model loading, `llama_decode`, threading backend, or HPX.

Verification result:

```text
Normal config-print run: exit 0.
--help: prints usage and exits 0.
Unknown argument: exits 1.
```

Notes:

`std::future` is a neutral v1 interface for the harness. It is not intended to be a permanent statement about the best HPX-facing API.

### 1.3: `tools: add std serving benchmark backend`

Hash: `planned`

Intent:

Implement the baseline backend using `std::thread`, not HPX.

This subsection should load one model, create a fixed pool of prewarmed contexts, assign one worker to one context, process requests through a mutex/condition-variable queue, run normal llama.cpp decode, and collect request latency metrics.

Rules:

- No HPX headers.
- No HPX backend.
- No ggml changes.
- No `graph_compute` changes.
- CPU-only model execution.
- Greedy sampling only.
- No streaming.
- No cancellation.
- No performance claims from smoke runs.

Approved implementation details:

- Load one shared `llama_model`.
- Create `N` independent `llama_context` objects.
- Set `n_gpu_layers = 0`.
- Use static fair-share `n_threads_per_context`.
- Warm contexts before timing.
- Start request timing in `submit()`, not inside the worker.
- Include queue wait in TTFT and total latency.
- Retry tokenization when `llama_tokenize` returns a negative buffer-size result.
- Call `llama_synchronize(ctx)` after prefill and decode before reading logits.
- Use `llama_memory_clear(llama_get_memory(ctx), false)` to reset context state.

Planned smoke verification:

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench \
  -m /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello, my name is" \
  --n-contexts 1 \
  --n-concurrent 1 \
  --n-requests 1 \
  --max-tokens 16
```

and:

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench \
  -m /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Hello, my name is" \
  --n-contexts 2 \
  --n-concurrent 2 \
  --n-requests 4 \
  --max-tokens 16
```

Report only:

```text
build result
n_ok / n_error
aggregate tok/s
TTFT p50 / p95
total p50 / p95
```

Do not interpret these as performance results yet.

### 1.4: `planned HPX backend`

Hash: `planned`

Intent:

Add an HPX backend that runs the same request workload as the std backend.

This subsection should not change llama.cpp execution semantics. It should only replace the request-orchestration runtime.

Expected comparison:

```text
same model
same prompts
same n_contexts
same n_threads_per_context
same max_tokens
same request count
std backend vs HPX backend
```

Only after this subsection can we begin answering whether HPX helps the serving orchestration layer.

## Current results

Current real results are intentionally minimal.

- Base build succeeds.
- HPX is installed outside the repository.
- `llama-serving-bench` stub builds and runs.
- Harness types and CLI parsing build and run.
- No model execution result has been committed yet.
- No std-vs-HPX comparison exists yet.
- No performance claim exists yet.

## What would count as useful evidence

The first meaningful evidence will come only after both the std backend and HPX backend exist.

Useful evidence:

- `n_concurrent=1`: non-regression control.
- `n_concurrent=2/4`: concurrent request behavior.
- std backend vs HPX backend under the same workload.
- aggregate tokens/sec.
- TTFT p50/p95.
- total latency p50/p95.
- fairness across requests.
- error rate.
- CPU thread configuration.

Not useful yet:

- Smoke-run tok/s from one backend.
- Comparing different prompts.
- Comparing different thread counts without saying so.
- Comparing runs under different CPU contention or thermal state.
- Comparing against llama-server before the basic std-vs-HPX harness is ready.

## Relationship to llama-server

Upstream llama-server already implements a serious serving architecture based on one context, many slots, and continuous batching.

This branch is not claiming to be better than llama-server.

The immediate goal is narrower:

```text
Build a controlled harness to compare HPX request orchestration against std::thread request orchestration around a pool of independent llama_context objects.
```

A later experiment may compare the harness against llama-server, but that is not the first claim.

## Open questions

- Does the std backend behave correctly with real model decode?
- What is the cost of multiple prewarmed contexts for TinyLlama and Llama-3.1-8B?
- Does the static fair-share thread policy produce acceptable single-request behavior?
- Does HPX improve scheduling, fairness, or latency compared with the std backend?
- Does the pool-of-contexts approach lose too much throughput compared with upstream continuous batching?
- Is HPX valuable only above an upstream-style batching loop rather than as an alternative to it?

## Do not repeat old mistakes

Do not:

- use HPX per ggml node
- start with packets or lowering
- add a run-level analyzer before the serving question is tested
- touch `src/llama-context.cpp` for v1
- add an HPX-backed ggml threadpool for v1
- claim speedup from smoke tests
- compare against llama-server before the std-vs-HPX harness exists

## 2. Tools: add HPX-native serving backend and gates

This section turns the serving-bench direction from a stub and plan into a working controlled experiment.

The first section established the new question:

```text
Can HPX improve serving-level orchestration for concurrent CPU inference requests?
```

This section implements enough of the harness to ask that question honestly:

```text
std backend
vs.
HPX-native backend
```

Both backends run the same opaque llama.cpp decode path. HPX does not enter ggml, lower ops, rewrite kernels, or touch `src/llama-context.cpp`. The comparison stays at the serving-orchestration layer.

### What changed

The serving benchmark now has a real backend interface and two backend implementations:

```text
--backend std
--backend hpx
```

The std backend is the control group. It uses a simple worker-pool shape:

```text
one shared llama_model
N prewarmed llama_context objects
one std worker per context
mutex / condition-variable queue
normal llama_decode inside each context
```

The HPX backend is intentionally not a std worker-pool clone. After rejecting the conservative "one HPX worker per context" design, the HPX implementation uses an HPX-native orchestration shape:

```text
request as HPX async/continuation chain
future-based hpx_context_pool
exclusive llama_context lease
RAII context_guard release
normal llama_decode inside the leased context
```

This keeps the experiment focused:

```text
same model
same prompt
same context count
same llama_decode path
same greedy argmax
same generated-token hash
different orchestration runtime
```

### Backend selection and correctness surface

The harness now exposes backend selection explicitly:

```text
--backend std
--backend hpx
```

The default backend is `std`.

The harness also gained the pieces needed to prevent misleading runs:

- `--ctx-size`
- `--batch-size`
- prompt-token fit check before request submission
- clean unknown-backend error
- clean HPX-OFF stub error
- `generated_token_hash` for structural correctness
- request-index tracing for HPX acquire/release checks

The HPX-OFF path is deliberately safe. In a build without HPX support, asking for `--backend hpx` exits before model load or request submission:

```text
[serving-bench] HPX backend not built (LLAMA_SERVING_BENCH_HPX=OFF)
```

This means the benchmark can distinguish:

```text
invalid backend name
vs.
valid backend name unavailable in this build
vs.
real HPX backend
```

### HPX runtime lifecycle

HPX runtime ownership was placed at the program lifecycle layer, not inside the engine.

The narrow runtime API is:

```cpp
bool hpx_runtime_start_once(int32_t os_threads);
void hpx_runtime_stop();
```

Important invariants:

```text
--backend std does not start HPX
--backend hpx starts HPX before make_engine_hpx()
engine_hpx does not start or stop HPX
HPX stops after engine cleanup and llama_backend_free()
```

Lifecycle tracing is env-gated through:

```text
LLAMA_SERVING_BENCH_HPX_TRACE=1
```

This made it possible to verify that enabling HPX support does not perturb the std backend.

### HPX-native engine design

The HPX backend was built in slices.

First, `engine_hpx` became real but only supported the empty-generation path. That proved:

```text
HPX runtime starts
make_engine_hpx returns a real engine
engine_hpx loads model and creates contexts
max_tokens=0 completes through the HPX path
HPX runtime stops
```

Then the placeholder pool became a real HPX-native context pool:

```text
acquire() returns hpx::future<llama_context *>
available context -> ready future
no context available -> enqueue hpx::promise waiter
release(ctx) wakes one waiter or returns ctx to available pool
close() rejects future acquires and resolves queued waiters
```

A move-only `context_guard` releases the leased context on normal and error paths.

This is the key design decision of the section: the HPX backend does not use permanent context-bound workers. Waiting for a context is represented as future readiness, not a blocked OS thread.

### Real HPX decode

After the pool and guard were validated, the intentional placeholder:

```text
hpx decode not implemented yet
```

was replaced with real decode.

The HPX decode path matches the std decode behavior on the important structural points:

```text
llama_memory_clear(llama_get_memory(ctx), false)
same tokenization helper and retry behavior
same prompt prefill batch
same decode loop condition
same llama_decode handling
same llama_synchronize call
same logits access through llama_get_logits_ith(ctx, -1)
same greedy argmax
same EOG handling
same token-hash fold order
same empty-hash rule
same TTFT timing semantics
```

The result is that HPX changes request orchestration, not the token stream.

### Correctness gates

The project now has HPX-ON acceptance gates documented separately in:

```text
docs/hpx/gates.md
```

The relevant gates passed:

```text
Gate 1: HPX-ON build/link
Gate 2: std path regression in HPX-capable build
Gate 3: HPX smoke
Gate 4: std vs HPX hash equality
Gate 5: empty-generation path
Gate 6: HPX repeat stability
Gate 7: two-context HPX smoke
Gate 9: pool queueing under real decode
```

OFF Tests 1-7 also passed after the HPX implementation.

The most important correctness result is Gate 4:

```text
std hash == hpx hash == 0x833045f1e2ebf49f
```

That result says the HPX backend produced the same generated token stream as the std backend for the canonical TinyLlama smoke.

### Pinned correctness values

Canonical model:

```text
/Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Canonical prompt:

```text
Hello, my name is
```

Pinned hashes:

```text
max_tokens=16:
  generated_token_hash=0x833045f1e2ebf49f

max_tokens=0:
  generated_token_hash=0x0000000000000000

max_tokens=32:
  generated_token_hash=0x6794e47fe0f84af1
```

These hashes are correctness signals. They are not quality or performance metrics.

### First sanity benchmark

After correctness gates passed, a small local std-vs-HPX sanity benchmark was run.

Matrix:

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

All six runs completed with:

```text
n_error=0
n_cancelled=0
canonical hash counts matched expectations
```

First-pass aggregate tokens/sec:

```text
A_std: 58.87
A_hpx: 72.61

B_std: 118.56
B_hpx: 123.69

C_std: 65.70
C_hpx: 74.31
```

These numbers were useful only as a narrow sanity check: HPX showed no obvious overhead in that tiny single-shot matrix. They were not final performance evidence.

### Repeatable benchmark protocol and result

A repeatable local benchmark protocol was then prepared and run.

Protocol file:

```text
local/bench_protocol_repeat.md
```

Matrix:

```text
A: 1 context / 1 concurrent / 8 requests / 16 tokens
B: 2 contexts / 2 concurrent / 16 requests / 16 tokens
C: 1 context / 4 concurrent / 16 requests / 16 tokens
Backends: std, hpx
Repeats: 5 per cell
Total runs: 30
```

Run order alternated std and HPX within each shape:

```text
A_std_r1, A_hpx_r1, A_std_r2, A_hpx_r2, ...
then B
then C
```

Output directory:

```text
local/bench_repeat/
```

The repeated benchmark completed with:

```text
30 stdout files
30 stderr files
all runs: n_cancelled=0
all runs: n_error=0
all A runs: canonical hash count = 8
all B runs: canonical hash count = 16
all C runs: canonical hash count = 16
git_status_pre.txt == git_status_post.txt
```

Five-repeat summary, mean ± sample standard deviation:

| Shape | Backend | wall s | agg tok/s | TTFT p95 ms | total p95 ms |
|---|---|---:|---:|---:|---:|
| A | std | 1.311 ± 0.165 | 98.90 ± 12.24 | 38.82 ± 8.24 | 251.63 ± 92.95 |
| A | hpx | 1.231 ± 0.114 | 104.64 ± 9.17 | 66.87 ± 63.11 | 205.95 ± 63.76 |
| B | std | 1.831 ± 0.024 | 139.85 ± 1.84 | 40.70 ± 3.69 | 245.67 ± 11.48 |
| B | hpx | 1.967 ± 0.065 | 130.29 ± 4.11 | 57.08 ± 13.79 | 340.28 ± 68.92 |
| C | std | 2.501 ± 0.169 | 102.78 ± 7.44 | 660.82 ± 134.34 | 780.95 ± 135.65 |
| C | hpx | 2.448 ± 0.274 | 105.57 ± 11.07 | 614.39 ± 227.68 | 737.63 ± 231.87 |

The repeated benchmark changed the interpretation from the single-shot sanity run.

Observed in this local matrix:

```text
A: HPX had slightly higher mean aggregate throughput than std, but both paths had noisy outliers.
B: std had higher mean aggregate throughput and tighter p95 total latency than HPX.
C: HPX and std were close on mean aggregate throughput; both paths showed high queueing-tail variability.
```

So the result is useful but mixed. It supports correctness and plausibility. It does not support a broad speedup claim.

No final performance claim exists yet.

### What this section proves

This section proves that the branch has crossed the first real threshold:

```text
Both std and HPX backends exist.
Both run real llama.cpp decode.
HPX uses a native async context-pool design.
HPX preserves the std token stream for the canonical smoke.
HPX handles concurrent contexts and queued requests correctly in the tested matrix.
The repeated local benchmark completed without errors, cancellations, or hash drift.
```

This is enough to justify moving from implementation correctness into broader benchmark design.

It does not prove:

```text
HPX is faster than std.
HPX is better than llama-server.
The pool-of-contexts design beats upstream continuous batching.
The current thread-count formula is optimal.
TinyLlama short-decode behavior predicts larger-model or longer-decode behavior.
```

Those questions require broader workload matrices.

### What changed in the story

The branch no longer has only a plan for a std backend and a planned HPX backend.

It now has a working controlled harness:

```text
std backend: simple baseline
HPX backend: HPX-native async orchestration
correctness gates: green
first sanity benchmark: complete
repeat benchmark protocol: complete
repeat benchmark result: correct but mixed
```

The story has moved from:

```text
Can we build the harness?
```

to:

```text
The harness is correct. The first repeated benchmark is mixed. What workload shapes should we test next?
```

### Current useful evidence standard

Useful evidence now requires:

- correctness gates green
- same model and prompt
- same request shape
- same build
- canonical hash count verified for every run
- multiple repeats before interpreting performance
- working tree status stable during a benchmark matrix
- outputs saved under `local/` for exploratory work or under `hpx-bench/results/<date>-<slug>/` for shareable benchmark evidence

Still not useful as final evidence:

- one run per cell
- any run with `n_error > 0`
- any run with `n_cancelled > 0`
- any run with wrong hash count
- interpreting smoke tests as benchmark results
- comparing runs after source changes between std and HPX
- making broad speedup claims from this TinyLlama / short-decode matrix

## 3. Bench: package HPX serving-bench evidence

This section packages the benchmark evidence produced after the HPX serving backend became functional.

The previous section made the controlled serving harness real:

```text
std backend
vs.
HPX-native backend
```

with both backends running the same opaque llama.cpp decode path and preserving the same generated-token stream for the pinned TinyLlama smoke.

This section is about packaging and interpretation. It does not introduce a new speedup claim.

The central result is:

```text
HPX serving orchestration is correct and robust in the tested shapes.
HPX does not show a reliable performance win in the fixed-shape CPU TinyLlama serving-bench harness.
The measured overhead is small, real, and configuration-dependent.
```

### What changed

This section adds a curated benchmark evidence package under:

```text
hpx-bench/
```

It keeps:

```text
protocol docs
helper scripts
deterministic schedules
compact summaries
final result snapshots
```

The package is organized chronologically under:

```text
hpx-bench/experiments/
```

The project-level result reports live under:

```text
docs/hpx/
```

### Packaged experiment chain

The packaged experiment chain now records the evidence path from server baseline to HPX performance interpretation.

The main packaged steps are:

```text
server repeatability and timing baselines
serving-bench std timing baseline
aligned llama-server vs serving-bench comparison
HPX-vs-std single-context correctness
HPX 2-context concurrency correctness
HPX waiter-pressure correctness
full HPX-vs-std performance matrix
C_2x4 n_threads sensitivity sweep
heterogeneous-budget design checkpoint
```

The heterogeneous-budget workload is only a design checkpoint at this stage. It is not performance evidence yet.

### Full HPX-vs-std performance matrix

A benchmark-grade HPX-vs-std matrix was designed and summarized.

Matrix:

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

Common settings:

```text
model:      tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:     "Hello, my name is"
max_tokens: 16
ctx_size:   2048
batch_size: 512
n_threads:  4
seed_base:  1234
```

Protocol:

```text
two layers:
  correctness_trace_on
  timing_trace_off

two backends:
  std
  hpx

K = 31 process-level trials per backend per cell per layer
trial 0 correctness-checked but excluded from timing aggregation

total:
  4 cells × 2 layers × 2 backends × 31 trials = 496 invocations
```

Correctness result:

```text
MATRIX_OVERALL_CORRECTNESS: PASS
16 / 16 layer summaries: PASS
4 / 4 condition summaries: PASS
496 / 496 raw trials completed
```

The matrix correctness gates checked:

```text
all harness exit codes are 0
all expected request rows are parsed
all aggregate lines match n_ok=N n_cancelled=0 n_error=0
all request statuses are ok
all n_tokens_generated values are 16
all generated_token_hash values equal 0x833045f1e2ebf49f
prompt-fits line is present
no serving-bench error/failed diagnostics appear
HPX lifecycle counts match each cell
std remains HPX-trace-free where required
same-context acquire/release pairing holds
```

This allowed timing interpretation under the protocol.

The top-line timing result was:

```text
A_1x1: HPX and std are indistinguishable.
B_2x2: HPX shows modest overhead.
C_2x4: HPX shows larger overhead under waiter pressure.
D_4x4: signal is mixed and noisy.
```

Median `total_ms`:

| Cell | std median | hpx median | Delta | Delta % |
|---|---:|---:|---:|---:|
| A_1x1 | 174.075 ms | 174.210 ms | +0.136 ms | +0.08% |
| B_2x2 | 234.417 ms | 244.556 ms | +10.140 ms | +4.33% |
| C_2x4 | 475.062 ms | 511.560 ms | +36.498 ms | +7.68% |
| D_4x4 | 1217.710 ms | 1200.312 ms | -17.398 ms | -1.43% |

Median `total_minus_ttft_ms`:

| Cell | std median | hpx median | Delta | Delta % |
|---|---:|---:|---:|---:|
| A_1x1 | 146.084 ms | 146.170 ms | +0.087 ms | +0.06% |
| B_2x2 | 195.235 ms | 203.228 ms | +7.993 ms | +4.09% |
| C_2x4 | 200.026 ms | 213.831 ms | +13.804 ms | +6.90% |
| D_4x4 | 740.449 ms | 705.610 ms | -34.839 ms | -4.71% |

Median process wall time:

| Cell | std median | hpx median | Delta | Delta % |
|---|---:|---:|---:|---:|
| A_1x1 | 1333.215 ms | 1350.292 ms | +17.077 ms | +1.28% |
| B_2x2 | 1263.887 ms | 1335.703 ms | +71.817 ms | +5.68% |
| C_2x4 | 1794.560 ms | 1906.228 ms | +111.668 ms | +6.22% |
| D_4x4 | 5296.890 ms | 5487.855 ms | +190.965 ms | +3.61% |

The matrix does not support a reliable HPX speedup claim.

The useful result is:

```text
The HPX serving backend preserves correctness and lifecycle invariants across the tested matrix, but the current CPU-only TinyLlama harness does not show a performance advantage over the simpler std backend.
```

### C_2x4 n_threads sensitivity sweep

The C_2x4 cell had the clearest overhead in the matrix, so a targeted diagnostic sweep was run.

Fixed cell:

```text
n_contexts=2
n_concurrent=4
n_requests=12
```

Sweep variable:

```text
n_threads ∈ {1, 2, 4}
```

Protocol:

```text
two layers:
  correctness_trace_on
  timing_trace_off

two backends:
  std
  hpx

K = 11 process-level trials per backend per thread setting per layer
trial 0 correctness-checked but excluded from timing aggregation

total:
  3 thread settings × 2 layers × 2 backends × 11 trials = 132 invocations
```

Correctness result:

```text
SWEEP_OVERALL_CORRECTNESS: PASS
12 / 12 layer summaries: PASS
3 / 3 thread-setting summaries: PASS
132 / 132 raw trials completed
```

Median `total_ms`:

| n_threads | std median | hpx median | Delta | Delta % |
|---:|---:|---:|---:|---:|
| 1 | 741.918 ms | 751.691 ms | +9.773 ms | +1.32% |
| 2 | 541.571 ms | 580.258 ms | +38.687 ms | +7.14% |
| 4 | 488.700 ms | 506.925 ms | +18.226 ms | +3.73% |

Median `total_minus_ttft_ms`:

| n_threads | std median | hpx median | Delta | Delta % |
|---:|---:|---:|---:|---:|
| 1 | 296.796 ms | 300.538 ms | +3.743 ms | +1.26% |
| 2 | 220.489 ms | 236.490 ms | +16.001 ms | +7.26% |
| 4 | 204.109 ms | 211.719 ms | +7.609 ms | +3.73% |

All trend metrics were classified as:

```text
mixed
```

The sweep does not support a clean oversubscription-only explanation.

If kernel-thread oversubscription were the whole explanation, HPX overhead should grow monotonically with `n_threads`:

```text
n_threads=1 < n_threads=2 < n_threads=4
```

That did not happen.

The sweep also does not support a flat orchestration-only explanation because the overhead range is too large to call invariant.

The best interpretation is:

```text
C_2x4 HPX overhead appears to come from an interaction between HPX serving orchestration / waiter pressure and llama/ggml kernel-thread behavior. The interaction is worst at n_threads=2 in this run.
```

### Heterogeneous-budget design checkpoint

After the fixed-shape matrix and the n_threads sweep, continuing to add more fixed-shape dimensions was judged to have diminishing returns.

A future HPX-favoring workload was designed:

```text
heterogeneous request budgets
```

The design lives under:

```text
hpx-bench/experiments/10_perf_heterogeneous_budgets_design/
```

It asks:

```text
When requests have mixed generation lengths, does HPX serving orchestration behave differently from std in short-request latency, queue-drain behavior, or makespan?
```

The key inspection finding was:

```text
serving_bench::request_params already has per-request max_tokens and prompt
backend_std.cpp already reads req.max_tokens
backend_hpx.cpp already reads req.max_tokens
```

The limitation is the call site:

```text
main.cpp pins every request to cfg.max_tokens
```

So the expected source change is small and harness-level:

```text
add --max-tokens-plan CSV parsing
validate plan length equals n_requests
validate all entries are positive
use max(plan) for fit-check
set rp.max_tokens from plan[next_idx]
```

The heterogeneous-budget experiment is not completed in this section.

It still needs:

```text
source change for --max-tokens-plan
HPX-on rebuild
small CLI smoke
helper scripts
schedule generation
correctness and timing runs
summary docs
```

Until then, it is future work, not evidence.

### Current conclusion

The branch now has stronger evidence than the first repeated benchmark.

It proves:

```text
The std and HPX backends both run real llama.cpp decode.
The HPX backend preserves the std token stream for the pinned smoke.
HPX lifecycle ownership is clean.
HPX context leasing and release are correct in the tested shapes.
HPX handles multi-context and waiter-pressure workloads without starvation.
The benchmark-grade matrix and thread sweep both pass correctness.
```

It does not prove:

```text
HPX is faster than std.
HPX is better than llama-server.
HPX is useful for all serving workloads.
The pool-of-contexts design beats continuous batching.
The current HPX backend exposes HPX's full scheduling value.
```

The performance conclusion is:

```text
On this fixed-shape CPU TinyLlama serving-bench workload, HPX does not outperform the simpler std backend. Its overhead is small, measurable, and dependent on workload/thread configuration.
```

The project conclusion is:

```text
The HPX serving backend is correct and robust, but the current fixed-shape harness mostly measures the cost of an HPX wrapper around opaque llama.cpp work. To test HPX's actual strengths, the next workload must exercise richer orchestration: heterogeneous request lengths, cancellation/backpressure, priority scheduling, or pipelining.
```

### Evidence standard going forward

The evidence standard after this section is:

```text
correctness gates first
timing interpretation only after correctness passes
trace-on layer for lifecycle
trace-off layer for timing
trial 0 excluded from timing aggregation
no speedup claims from smoke tests
no broad performance claims from one model / one prompt / one machine
```

For future HPX-serving experiments, useful evidence should test at least one HPX-relevant serving behavior:

```text
heterogeneous request budgets
cancellation under load
backpressure
priority / fairness
prefill-decode overlap
pipeline scheduling
larger or more varied prompts
```

More fixed-shape TinyLlama sweeps are no longer likely to change the central conclusion.

## 4. Bench: add serving-bench closeout evidence

Added the final serving-bench evidence for the HPX-vs-std FIFO context-pool comparison.

This package adds two runnable experiment directories:

- `hpx-bench/experiments/10_perf_heterogeneous_budgets/`
- `hpx-bench/experiments/11_perf_deep_queue_short_requests/`

Experiment 10 tested mixed request budgets under mild waiter pressure:

- `EXPERIMENT_OVERALL: PASS`
- branch label: `hpx_short_worse`
- 44 / 44 invocations completed
- 0 token-count multiset mismatches
- std-vs-HPX per-budget hash equality passed
- short-request median `total_ms`: HPX +1.9967%
- trial makespan median: HPX +1.00%
- queue-drain median: HPX +28.06 ms, but noisy and small relative to total makespan

Experiment 11 tested a deep queue of uniform short requests to amplify request wake/queue overhead:

- `EXPERIMENT_OVERALL: PASS`
- branch label: `hpx_worse`
- 44 / 44 invocations passed
- budget-8 hash equality passed: `0x0619d4d1900c2365`
- HPX correctness traces matched the expected 403 lifecycle/pool lines in 11 / 11 correctness trials
- short-request median `total_ms`: HPX +2.07%
- short-request p99 `total_ms`: HPX +1.20%, inside the configured +2.0% p99 threshold
- trial makespan median: HPX -0.14%

Together with the earlier matrix and thread-sweep experiments, these results support a conservative closeout conclusion: the HPX serving backend is correct and robust, but the current FIFO context-pool design does not outperform the simpler std backend on the tested CPU-only TinyLlama workloads.

The old design-only heterogeneous-budget package was removed because the runnable experiment package supersedes it.

A closeout note was added at:

- `docs/hpx/serving_fifo_pool_closeout.md`

The closeout is intentionally scoped to the current FIFO context-pool design. It does not claim that HPX cannot help LLM serving in designs that use HPX-native capabilities such as cancellation, priority scheduling, richer future composition, or distributed execution.

## 5. Packaging HPX continuous-batching gate and cancellation evidence

This section packages the HPX continuous-batching direction and its evidence.

It records the move from the older FIFO serving-bench path to the current continuous-batching orchestration path, and it adds the prototype tools and documents that prove the current correctness/lifecycle gates.


### 5.1 What changed in this package

This package adds or updates the main documentation and prototype artifacts for the current HPX serving direction:

```text
docs/hpx/provenance.md
tools/CMakeLists.txt
docs/hpx/continuous_batching_gate_closeout.md
docs/hpx/continuous_batching_live_admission_design.md
docs/hpx/continuous_batching_phase3_target.md
docs/hpx/continuous_batching_simulator_design.md
docs/hpx/continuous_batching_upstream_notes.md
docs/hpx/continuous_batching_cancellation_design.md
docs/hpx/continuous_batching_prototype_design.md
docs/hpx/ggml_cgraph_dump_repro.md
docs/hpx/multiseq_llama_batch_gate.md
hpx-bench/sim/
tools/hpx-continuous-batch-gate/
tools/multiseq-batch-gate/
```

The package includes:

```text
continuous-batching simulator work
pure llama.cpp multi-seq shared-batch reference gate
HPX continuous-batch orchestration gate
cooperative cancellation design and results
closeout notes for the current HPX continuous-batching gate
design-only live-admission notes for later work
```

This is a correctness/lifecycle package, not a performance claim.

---

### 5.2 FIFO serving-bench path is closed out

The earlier serving-level HPX path used HPX as a FIFO context-pool replacement around opaque `llama_decode` calls.

The result was:

```text
Correct and robust, but no reliable HPX latency benefit.
```

Evidence came from the serving-bench experiments, especially:

```text
hpx-bench/experiments/10_perf_heterogeneous_budgets/
hpx-bench/experiments/11_perf_deep_queue_short_requests/
```

Experiment 10 result:

```text
EXPERIMENT_OVERALL: PASS
branch label: hpx_short_worse

short total_ms median: HPX +1.9967%
medium total_ms median: HPX +1.59%
long total_ms median: HPX -0.33%
trial makespan median: HPX +1.00%
```

Experiment 11 result:

```text
EXPERIMENT_OVERALL: PASS
branch label: hpx_worse

short.total_ms median: HPX +2.07%
short.total_ms p90:    HPX +0.13%
short.total_ms p95:    HPX +0.43%
short.total_ms p99:    HPX +1.20%
makespan median:       HPX -0.14%
```

Interpretation:

```text
The FIFO context-pool backend preserved correctness, but it did not show a useful latency advantage.
That path is closed out as a performance direction.
```

---

### 5.3 Continuous-batching simulator selected the next target

The simulator work lives under:

```text
hpx-bench/sim/continuous_batching/
```

Supporting design/results docs include:

```text
docs/hpx/continuous_batching_simulator_design.md
hpx-bench/sim/continuous_batching/results.md
hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md
```

The simulator is a scheduling model only:

```text
No HPX.
No llama.cpp.
No real logits, KV cache, HTTP, or GPU behavior.
```

Simulator result:

```text
Static and continuous batching mostly tie on symmetric all-short workloads.
Continuous batching separates on mixed or bursty workloads.
```

The first useful real target selected from the simulator was:

```text
mixed_decode_only
prompt_tokens = 6
decode budget mix = {8, 64, 256}
round-robin class assignment
all requests admitted at t = 0
```

Reason:

```text
This isolates decode-length heterogeneity without adding long-prompt prefill complexity.
```

---

### 5.4 Pure llama.cpp multi-seq gate proved the execution primitive

The pure llama.cpp reference gate lives under:

```text
tools/multiseq-batch-gate/
```

Supporting doc:

```text
docs/hpx/multiseq_llama_batch_gate.md
tools/multiseq-batch-gate/results.md
```

The purpose was to prove the real llama.cpp primitive before adding HPX orchestration.

Final Step-5 shape:

```text
n_seqs = 99
budget mix = {8, 64, 256}
33 seqs per budget class
prompt_tokens = 6
one llama_model
one llama_context
one shared llama_batch
shared prefill
shared decode loop
per-seq KV clear
```

Result:

```text
GATE_STEP5: PASS
```

Observed capacity:

```text
actual n_ctx = 50688
actual n_seq_max = 99
actual n_batch = 1024
```

Hashes:

```text
budget 8   -> 0x0619d4d1900c2365
budget 64  -> 0x88a4dc75a31d4325
budget 256 -> 0x8a1a3bd01360aada
```

Lifecycle anchors:

```text
done_iter:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

Other gates:

```text
1 unique hash per budget class
residual KV empty at end
no sibling cross-talk on KV clear
every llama_decode returned 0
--repeat 2 passed
```

Important caveat:

```text
Budget-64 and budget-256 hashes are batch-shape dependent.
Do not compare long-budget hashes across different batch shapes.
Use same-shape repeat determinism and within-run same-class hash equality.
```

---

### 5.5 HPX continuous-batch gate proved request lifecycle ownership

The HPX prototype lives under:

```text
tools/hpx-continuous-batch-gate/
```

Supporting docs:

```text
docs/hpx/hpx_continuous_batching_prototype_design.md
tools/hpx-continuous-batch-gate/results.md
docs/hpx/continuous_batching_gate_closeout.md
```

Question:

```text
Can HPX own request lifecycle around the proven llama.cpp multi-seq continuous-batching primitive?
```

Result:

```text
Yes, at correctness/lifecycle level.
No performance claim.
```

Slice results:

```text
HPX_CB_STEP1: PASS
HPX_CB_STEP2: PASS
HPX_CB_STEP3: PASS
HPX_CB_STEP4: PASS
HPX_CB_PROTO: PASS
```

What the slices proved:

```text
Slice 1:
  HPX runtime starts/stops cleanly with libllama.
  Model/context load works.
  99 metadata-only request objects can be built.

Slice 2:
  One HPX engine task can own the proven Step-5 loop.
  Only the engine task touches llama_context, llama_batch, llama_decode, llama_get_logits_ith, and llama_memory_seq_*.

Slice 3:
  Each request/seq can be represented by an HPX promise/future.
  The engine fulfills each promise after KV clear and cross-talk checks.
  Main waits on futures and validates request_result snapshots only.

Slice 4:
  Env-gated lifecycle traces and descriptive metrics were added.

Slice 5:
  Same-shape HPX-off vs HPX-on correctness equivalence passed.
```

Slice 3 counters:

```text
futures_created = 99
promises_fulfilled = 99
futures_completed = 99
engine_task_count = 1
decode_failures = 0
```

Same-shape HPX-off vs HPX-on equivalence:

```text
budget 8:   0x0619d4d1900c2365 == 0x0619d4d1900c2365
budget 64:  0x88a4dc75a31d4325 == 0x88a4dc75a31d4325
budget 256: 0x8a1a3bd01360aada == 0x8a1a3bd01360aada
```

Lifecycle equivalence:

```text
done_iter:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

Interpretation:

```text
HPX orchestration matched the pure llama.cpp same-shape reference on correctness fields.
This was not a performance comparison.
```

---

### 5.6 Cooperative cancellation proved the first real HPX serving capability

Cooperative cancellation is documented in:

```text
docs/hpx/hpx_continuous_batching_cancellation_design.md
tools/hpx-continuous-batch-gate/results.md
docs/hpx/continuous_batching_gate_closeout.md
```

Question:

```text
Can HPX add a real serving lifecycle feature beyond run-to-completion?
```

Result:

```text
Yes.
Cooperative cancellation passed through Cancel Slice 4.
```

Cancellation semantics:

```text
No interruption inside llama_decode.
Cancellation is observed at iteration boundaries.
Cancelled seqs leave future decode batches.
Cancelled seq KV is cleared by the engine task.
Cancelled futures complete with status = cancelled.
Non-cancelled seqs continue.
```

Cancellation smoke:

```text
n_seqs = 99
budget mix = {8,64,256}
cancel seqs {1,4,7} from budget 64
cancel seqs {2,5,8} from budget 256
cancel_after_decoded_tokens = 16
budget 8 is not cancelled
```

Final result:

```text
HPX_CB_CANCEL_STEP4: PASS
```

Outcome:

```text
completed_count = 93
cancelled_count = 6

budget 8:
  completed = 33
  cancelled = 0

budget 64:
  completed = 30
  cancelled = 3

budget 256:
  completed = 30
  cancelled = 3
```

Cancellation anchors:

```text
cancel_observed_iter_set = {16}
n_decoded_at_cancel_set = {16}
wasted_decode_rows_after_cancel = 0
residual_kv_empty = true
decode_failures = 0
```

Future/promise counters:

```text
futures_created = 99
promises_fulfilled = 99
futures_completed = 99
engine_task_count = 1
```

Cancellation trace counts:

```text
engine_start             = 1
engine_stop              = 1
request_admitted         = 99
seq_prefilled            = 99
decode_row               = 9861
seq_complete             = 93
kv_cleared               = 93
promise_fulfilled        = 93
cancel_requested         = 6
cancel_observed          = 6
cancel_kv_cleared        = 6
cancel_future_fulfilled  = 6
```

Interpretation:

```text
Cooperative cancellation is the first real HPX-native serving capability in this prototype.
It proves futures can complete with status = completed or status = cancelled, while the engine preserves KV safety and non-cancelled siblings continue.
This is still correctness/lifecycle evidence, not a performance claim.
```

---

### 5.7 Live admission is design-only next work

Live admission design lives at:

```text
docs/hpx/continuous_batching_live_admission_design.md
```

Status:

```text
Design only.
Not implemented.
```

Reason for next feature:

```text
Cancellation proved that an active request can leave the active batch early.
Live admission should prove that a waiting request can enter freed capacity while the engine loop is already running.
```

Important design caveat before implementation:

```text
The first live-admission smoke must reuse only cancellation-freed seq_ids {1,2,4,5,7,8}.
It must not use generic lowest-numbered free seq_id, because budget-8 seqs naturally finish early and would otherwise create earlier free slots such as 0,3,6,...
```

Required first-smoke policy:

```text
Maintain a free_due_to_cancel queue.
Append seq_id only after cancellation KV clear succeeds.
At the admission boundary, pop from free_due_to_cancel deterministically.
Bind waiting requests FIFO to those seq_ids.
Ignore naturally completed budget-8 free slots for this slice.
```

Expected first live-admission smoke after cleanup:

```text
n_active = 93
n_waiting = 6
waiting requests 93..98
waiting budget = 64
cancel seqs {1,4,7,2,5,8}
admit waiting requests into seq_ids {1,2,4,5,7,8}
```


---

### 5.8 Overall conclusion

This package records the current HPX continuous-batching evidence.

What was proven:

```text
The FIFO context-pool path was correct but did not show a latency benefit.
The simulator identified mixed decode lengths as a better target.
Pure llama.cpp can run the 99-seq shared-batch target shape.
HPX can own the engine-task lifecycle around that primitive.
HPX futures/promises can represent per-request completion.
HPX-on and HPX-off matched on same-shape correctness fields.
Cooperative cancellation works as a real serving lifecycle feature.
Cancelled and completed requests both resolve through request_result snapshots.
KV cleanup remains engine-owned and residual KV is empty.
```

## 6. hpx-cb-gate: componentize continuous-batch gate after live-admission and streaming closeout

After the cancellation closeout, the next question was no longer whether HPX could wrap one run-to-completion batch. Cancellation had already shown that HPX could own a real serving lifecycle event: a request could leave the active batch early, its future could resolve as `cancelled`, and the engine could clear its KV safely.

The next question became:

```text
Can the HPX engine continue running while request capacity changes underneath it?
```

This section records the evidence added after the earlier cancellation package:

```text
Live admission:
  waiting requests can enter freed capacity while the engine loop is already running

Streaming:
  token delivery can be represented as an HPX-native per-request stream

M0 componentization:
  the proven gate can be split into focused components without changing behavior
```

This section is still correctness and lifecycle provenance. It is not a performance claim.

---

### 6.1 Live admission changed the gate from static batch to dynamic serving lifecycle

Live admission was originally documented as design-only in Section 5.7.

That design has now been implemented and closed through the live-admission slices.

Question:

```text
Can an HPX-owned engine admit waiting work into freed seq_id capacity while the decode loop is already running?
```

Result:

```text
Yes.
Live admission passed through the later admission slices and is now part of the continuous-batch gate.
```

What live admission adds:

```text
The engine starts with active requests.
Other requests wait outside the active batch.
When capacity becomes free, the engine admits waiting requests.
The admitted request receives a reused seq_id.
KV for the reused seq_id is cleared before reuse.
The admitted request receives its own future/promise result.
The admitted request participates in later decode iterations.
```

Important serving meaning:

```text
Cancellation proved that requests can leave.
Live admission proves that new requests can enter.
Together, they turn the gate from a fixed batch experiment into a serving-lifecycle prototype.
```

Admission semantics:

```text
The engine task remains the only owner of llama.cpp mutable state.
External submitter code does not call llama_decode or mutate llama_context.
Admission happens at engine-controlled iteration boundaries.
Freed slots are reused only after the engine clears KV for the seq_id.
The waiting/admitted request is tracked through request_result.
```

Live-admission evidence includes:

```text
completion-freed reuse
cancel-freed reuse
external arrivals
release/ack synchronization for scripted external arrivals
multi-source admission accounting
source-priority validation
reused seq_id tracking
residual KV empty checks
```

Representative external-arrival smoke:

```text
--stream-all
--n-seqs 3
--n-active 2
--n-waiting 0
--n-external-arrivals 1
--decode-budget-mix 8,256
--external-arrival-budget 16
--external-release-iter 3
--reuse-completed
--cancel-plan none
--repeat 2
```

Observed external-arrival anchors:

```text
arrival_drained_count = 1
external_admitted_count = 1
first_external_drain_iter = 4
iter_release_fired_set = {3}
submitter_ack_set = {3}
```

Interpretation:

```text
The HPX gate now demonstrates dynamic request admission, not just static request execution.
A request can arrive after the engine has started, wait behind the engine boundary, and enter a reused seq_id slot when capacity becomes available.
```

---

### 6.2 Streaming slices proved HPX-native token delivery

After live admission, the next serving question was:

```text
Can HPX deliver tokens incrementally per request while preserving the same final request_result correctness?
```

Result:

```text
Yes.
Streaming closed through HPX_CB_STREAM_STEP7.
```

Streaming mechanism:

```text
Each active seq_id has an HPX local channel for token events.
The engine publishes token events while it owns llama.cpp execution state.
The consumer drains stream receivers outside the engine task.
Each stream ends with a closed event.
The close reason records completed, cancelled, or error.
```

Streaming event model:

```text
token_stream_opened
token_stream_token
token_stream_closed
```

Correctness contract:

```text
The streamed token sequence must match the request_result token sequence.
The streamed hash must match rr.hash.
Every opened stream must close.
Closed streams must report the correct close reason.
Trace-off mode must emit no event trace lines.
Trace-on mode must preserve per-event-name counts.
```

Streaming Slice 7 added the important reuse case:

```text
A seq_id can complete.
The engine clears its KV.
The same seq_id can be reused.
The newly admitted request gets a new stream.
The old stream and new stream remain distinguishable by request_id.
```

Representative Streaming Slice 7 smoke:

```text
--n-seqs 3
--n-active 1
--n-waiting 2
--waiting-budget 8
--decode-budget-mix 8
--reuse-completed
--cancel-plan none
--stream-all
--repeat 2
```

Representative Streaming Slice 7 anchors:

```text
HPX_CB_STREAM_STEP7: PASS

streams_opened = 3
streams_closed_completed = 3
streams_closed_cancelled = 0
streams_closed_error = 0
stream_tokens_emitted_total = 24
reused_seq_id_set = {0,0}
```

Trace-on representative anchors:

```text
token_stream_opened = 6
token_stream_closed = 6
token_stream_token = 48
seq_reused = 4
request_admitted_live = 4
engine_start = 2
engine_stop = 2
```

Interpretation:

```text
Streaming proves the gate can represent request progress before final completion.
This is a serving-control capability, not a llama.cpp execution change.
The engine still owns llama_decode and KV mutation; HPX owns the lifecycle, futures, and token stream handoff.
```

---

### 6.3 The gate now demonstrates the intended Project C serving-control boundary

After cancellation, live admission, seq_id reuse, external arrivals, and streaming, the HPX continuous-batch gate demonstrates:

```text
one llama_context
many seq_ids
one shared llama_batch
one HPX engine task owning llama.cpp mutable state
per-request futures/promises
cooperative cancellation
live admission
seq_id reuse
external arrivals
streaming token delivery
trace/counter validation
repeat determinism gates
residual KV cleanup checks
```

The boundary remains:

```text
HPX owns:
  request lifecycle
  admission
  cancellation observation
  future/promise completion
  stream handoff
  trace/counter validation

llama.cpp owns:
  llama_decode
  ggml graph execution
  kernels
  tokenizer behavior
  sampler math
  KV implementation
```

This is the right Project C boundary.

The gate is not trying to replace llama.cpp execution.
It is proving that HPX can own the serving-control plane around llama.cpp execution.

---

### 6.4 M0 componentization turned the proven gate into maintainable components

Once the live-admission and streaming slices passed, the main blocker was no longer missing lifecycle behavior.

The blocker became structure.

Before M0:

```text
tools/hpx-continuous-batch-gate/hpx-continuous-batch-gate.cpp
was a large monolithic translation unit.

It contained:
  shared helpers
  token hash helpers
  trace helpers
  POD types
  CLI parsing
  HPX runtime startup/shutdown
  scripted submitter logic
  engine implementation
  validation harness
  metrics printing
  repeat determinism checks
  main driver
```

Question:

```text
Can the proven HPX continuous-batch gate be split into focused files without changing behavior?
```

Result:

```text
Yes.
M0 componentization completed with no functional behavior change.
```

M0 produced this component layout:

```text
tools/hpx-continuous-batch-gate/
  token_hash.h
  trace.h
  trace.cpp
  gate_emit.h
  types.h
  cli.h
  cli.cpp
  hpx_runtime.h
  hpx_runtime.cpp
  engine.h
  engine.cpp
  submitter.h
  submitter.cpp
  gate_validation.h
  gate_validation.cpp
  hpx-continuous-batch-gate.cpp
```

What moved:

```text
token_hash.h:
  token hash constants and fold_token_hash

trace.h / trace.cpp:
  trace initialization, trace enablement, trace event emission

gate_emit.h:
  STEP label
  emit_pass
  emit_fail

types.h:
  request_status
  admission_source
  arrival_source
  stream close/event types
  token_stream_event
  channel aliases
  request_result
  waiting_request
  arrival_msg
  engine_metrics
  engine_result

cli.h / cli.cpp:
  cli_args
  usage printing
  argument parsing

hpx_runtime.h / hpx_runtime.cpp:
  HPX runtime start/stop helpers
  process-wide runtime state

engine.h / engine.cpp:
  engine_options
  class engine
  engine run/submit/result APIs
  admission/cancellation/streaming internals
  admit_one private method

submitter.h / submitter.cpp:
  submitter_release_block
  run_scripted_submitter

gate_validation.h / gate_validation.cpp:
  per-repeat validation harness
  per-result checks
  stream checks
  live-admission checks
  Slice 3/5/6/7 strict gates
  Slice 7 multi-cycle coverage
  metrics block
  residual KV checks
  repeat determinism checks

hpx-continuous-batch-gate.cpp:
  thin driver
```

M0 intentionally did not create a new library target.

```text
The target name stayed:
  llama-hpx-continuous-batch-gate

The engine is now in its own files, but still compiled into the same gate target.
Promoting the engine to a reusable library is later work.
```

---

### 6.5 M0 preserved the engine ownership model

The most important invariant was preserved:

```text
Only the HPX engine task touches llama.cpp mutable execution state.
```

M0 did not change:

```text
admission behavior
cancellation behavior
streaming behavior
seq_id reuse behavior
promise/future ownership
trace names
trace counts
CLI behavior
PASS/FAIL labels
validation rules
```

The engine construction was made explicit through `engine_options`:

```text
engine_options carries:
  borrowed llama_context / vocab / prompt tokens / waiting queue
  value-owned budgets
  value-owned release_iter_set
  cancel plan
  batch capacity
  n_seq_max
  stream_all
  reuse_completed
  max_decode_iters
```

The engine constructor now takes:

```text
engine(engine_options opts)
```

and the driver calls:

```text
engine eng(std::move(eng_opts));
```

Value-owned fields are moved into the engine.
Borrowed fields remain borrowed.

The `admit_one` logic was also lifted out of the decode-loop lambda into a private engine method:

```text
bool admit_one(int32_t reuse_seq,
               admission_source src,
               const char * src_label,
               int32_t iter,
               llama_memory_t mem);
```

This did not change admission semantics.
It only made the engine implementation more readable and less dependent on a large captured lambda.

---

### 6.6 M0 validation result

Build command:

```text
cmake --build /Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on \
  --target llama-hpx-continuous-batch-gate -j
```

Result:

```text
Built target llama-hpx-continuous-batch-gate
```

Six-smoke suite:

```text
1. Slice 7 multi-cycle streaming
2. Slice 7 trace-on
3. Streaming Slice 5 external arrival
4. Streaming Slice 6 external × cancel-freed
5. Default-mode regression
6. Stream-off Live Admission Slice 7
```

All passed:

```text
HPX_CB_STREAM_STEP7: PASS
```

Normalized diff result:

```text
0 lines of diff for all six smokes
excluding timing fields:
  wall_ms
  ttc_ms_*
  admitted_ttc_ms[*]
```

Trace-off result:

```text
0 [hpx-cb-gate] event= lines
```

Trace-on result:

```text
174 events
per-event-name totals matched the previous baseline
```

Representative trace-on totals:

```text
admitted_complete      = 4
admitted_decode_row    = 28
admitted_prefilled     = 4
decode_row             = 42
engine_start           = 2
engine_stop            = 2
kv_cleared             = 6
promise_fulfilled      = 6
request_admitted       = 2
request_admitted_live  = 4
request_queued         = 2
seq_complete           = 6
seq_prefilled          = 2
seq_reused             = 4
token_stream_closed    = 6
token_stream_opened    = 6
token_stream_token     = 48
```

Final source shape:

```text
hpx-continuous-batch-gate.cpp      692 lines
engine.cpp / engine.h              1372 / 244 lines
gate_validation.cpp / .h           2915 / 57 lines
cli.cpp / cli.h                    287 / 83 lines
submitter.cpp / submitter.h        61 / 37 lines
hpx_runtime.cpp / hpx_runtime.h    86 / 24 lines
trace.cpp / trace.h                41 / 25 lines
types.h                            370 lines
token_hash.h                       35 lines
gate_emit.h                        25 lines
```

Interpretation:

```text
M0 did not prove a new serving feature.
It made the already-proven serving-control gate maintainable.

The gate is now structurally ready for the next milestones:
  per-request prompts
  per-request sampling pass-through
  public engine API
  HTTP/server adapter
  matched comparison against llama-server
```

---

### 6.7 Overall conclusion

This section closes the provenance gap after the earlier cancellation package.

What was added after the old Section 5.7 design-only point:

```text
Live admission is now implemented.
Waiting requests can enter freed capacity.
External arrivals are represented.
Completion-freed and cancel-freed reuse paths are validated.
Streaming token delivery is implemented with HPX local channels.
Streams are opened, drained, hashed, and closed with checked close reasons.
Slice 7 validates multi-cycle seq_id reuse with streaming.
M0 splits the gate into maintainable components with no behavior change.
```

What this proves:

```text
The HPX continuous-batch gate is now a serving-control-plane prototype.

It can:
  complete requests
  cancel requests
  admit new requests
  reuse seq_ids
  stream tokens
  preserve KV safety
  preserve deterministic validation
  keep llama.cpp execution inside the engine task
```

What this does not claim:

```text
No performance advantage is claimed here.
No HTTP serving API exists yet.
No per-request prompt/sampling API exists yet.
No comparison against llama-server is claimed yet.
No distributed or multi-engine orchestration is claimed yet.
```

Next natural milestone:

```text
M1:
  per-request prompts
  per-request sampling pass-through
  request-shaped submit API inside the gate

M2:
  promote the engine into a reusable library target

M3:
  build the HPX serving binary

M4:
  compare hpx-server against llama-server under matched conditions
```

---

## 7. hpx-cb-gate: add reusable HPX engine submit API

After M0, the HPX continuous-batch gate was maintainable, but it was still mostly a gate-shaped prototype.
The engine lived in separate files, but the serving interface was not yet a reusable package interface.

The next question became:

```text
Can the HPX continuous-batch engine become a reusable HPX-native serving component,
with request-shaped submission, idle capacity, futures, and token streams,
while preserving the existing gate behavior byte-for-byte?
```

Result:

```text
Yes.
M1/M2 converted the componentized gate into a reusable HPX engine package.
```

This section records the work after M0:

```text
M1:
  add per-request prompt plumbing
  keep legacy shared-prompt behavior byte-identical

M2:
  add submit_request / submit_handle
  split llama-hpx-engine into a static library
  prove non-gate clients can consume the engine
  add initial idle-slot admission
  make submit_handle.stream real
```

This section is still correctness, lifecycle, and packaging provenance.
It is not a performance claim.

---

### 7.1 M1 made requests carry their own prompt data

Before M1, the gate used one shared prompt:

```text
cli_args::prompt
  -> tokenized once in main
  -> borrowed through engine_options::prompt_tokens
  -> replayed for every seq_id
```

That meant the gate could run many request lifecycles, but all requests still had the same prompt.
This was enough for lifecycle validation, but not enough for a real serving interface.

Question:

```text
Can each request carry its own prompt tokens through the HPX serving lifecycle?
```

Result:

```text
Yes.
Per-request prompt tokens now flow through initial active requests, waiting requests,
scripted external arrivals, live admission, streaming, and repeat-determinism checks.
```

M1 added request-specific prompt ownership in stages:

```text
M1a:
  add inert per-request prompt fields

M1b:
  initial active seqs own seq_state::prompt_tokens
  initial prefill reads seq.prompt_tokens

M1c:
  waiting_request, arrival_msg, and scripted_arrival carry prompt_tokens
  admitted prefill reads seq.prompt_tokens

M1d:
  add --request-prompts-file <path>
  one prompt per line
  line i maps to request_id i
```

The prompt-file mode assigns prompts as:

```text
request_id 0 .. n_active - 1:
  initial active requests

request_id n_active .. n_active + n_waiting - 1:
  preloaded waiting requests

request_id n_active + n_waiting .. n_active + n_waiting + n_external_arrivals - 1:
  scripted external arrivals
```

The legacy no-file path remains:

```text
--prompt <text>
  tokenized once
  copied into every request prompt slot
```

Compatibility result:

```text
Legacy no-file mode stayed byte-identical across the canonical gate smokes.
Prompt-file mode added a new behavior path without changing the old one.
```

Representative prompt-file smoke:

```text
n_seqs = 6
n_active = 3
n_waiting = 2
n_external_arrivals = 1
all prompts distinct
stream_all = true
reuse_completed = true
repeat = 2
```

Observed result:

```text
HPX_CB_STREAM_STEP7: PASS
```

Important evidence:

```text
variable prompt lengths reached the engine
pos_max_at_clear_set changed by prompt length
repeat determinism still passed
streamed hash matched request_result hash
residual_kv_empty remained true
```

Interpretation:

```text
The gate no longer only proves many lifecycles for one shared prompt.
It now proves request-shaped prompt ownership through the HPX control plane.
```

What M1 did not add:

```text
No sampling_config.
No llama_sampler integration.
No HTTP/server layer.
No public cancellation handle.
```

---

### 7.2 M2 introduced a public submit API

After M1, requests could carry their own prompt data, but the engine still exposed mostly gate-shaped entry points.
The next question was:

```text
Can a caller submit request-shaped work to the engine through an HPX-native public API?
```

Result:

```text
Yes.
M2 added submit_request, submit_handle, and engine::submit_request(...).
```

The public request shape is:

```cpp
struct submit_request {
    int32_t                  request_id    = -1;
    std::vector<llama_token> prompt_tokens;
    int32_t                  decode_budget = 0;
    bool                     want_stream   = false;
};
```

The public handle shape is:

```cpp
struct submit_handle {
    hpx::future<request_result>          result;
    std::optional<token_stream_receiver> stream;
};
```

Semantics:

```text
submit_request:
  caller-owned request metadata
  no arrival_source field
  no sampler field yet
  no cancellation handle yet

submit_handle.result:
  HPX future for final request_result

submit_handle.stream:
  optional HPX local token stream receiver
  present only when want_stream = true
```

The engine method:

```cpp
submit_handle engine::submit_request(submit_request req);
```

Important boundary:

```text
engine::submit_request does not call llama_decode.
engine::submit_request does not mutate KV.
engine::submit_request does not build llama_batch.
engine::submit_request only creates HPX-side request handles and enqueues request metadata.
```

The runtime submission is internally stamped as:

```text
arrival_source::external
```

Interpretation:

```text
The public API is HPX-native: callers receive hpx::future<request_result>
and, optionally, an HPX local receive_channel for token events.
```

---

### 7.3 Scripted external arrivals moved onto submit_request

Once the public submit API existed, the gate's scripted submitter was migrated to use it.

Before M2c:

```text
main pre-created per-arrival promises
submitter constructed arrival_msg directly
submitter moved the promise into arrival_msg
submitter called engine::submit(arrival_msg)
```

After M2c:

```text
submitter constructs submit_request
submitter calls engine::submit_request(...)
submitter stores submit_handle.result in external_futs
engine creates the internal promise and arrival_msg
engine routes through the existing inbox path
```

Important compatibility result:

```text
The external-arrival behavior stayed byte-identical.
```

The old gate-only path still exists:

```text
engine::submit(arrival_msg)
take_admitted_futures()
take_admitted_stream_handoffs()
take_stream_receivers()
```

But external result futures now flow through the new public submit API.

Interpretation:

```text
M2c proved submit_request is not just an unused public wrapper.
The existing scripted external-arrival gate path can drive the engine through it.
```

---

### 7.4 M2 split the engine into a reusable library target

After the public submit API existed, the next package question was:

```text
Can the engine build as a reusable library target while the gate remains a validation client?
```

Result:

```text
Yes.
M2d added llama-hpx-engine as a static library target.
```

Final target layout:

```text
llama-hpx-engine STATIC:
  engine.cpp
  hpx_runtime.cpp
  trace.cpp

llama-hpx-continuous-batch-gate executable:
  hpx-continuous-batch-gate.cpp
  cli.cpp
  gate_validation.cpp
  submitter.cpp

smoke clients:
  llama-hpx-engine-smoke
  llama-hpx-engine-idle-smoke
  llama-hpx-engine-stream-smoke
```

Engine/public/shared headers:

```text
engine.h
types.h
hpx_runtime.h
trace.h
token_hash.h
```

Gate-only files remain outside the library:

```text
cli.h / cli.cpp
submitter.h / submitter.cpp
gate_validation.h / gate_validation.cpp
gate_emit.h
hpx-continuous-batch-gate.cpp
```

The gate executable still has the same user-facing target name:

```text
llama-hpx-continuous-batch-gate
```

Validation result:

```text
All canonical gate smokes stayed byte-identical after the target split.
```

Interpretation:

```text
The engine is now a package boundary, not just a source-file boundary.
The gate is now one client of the engine library.
```

---

### 7.5 M2e proved a non-gate client can consume the engine library

After the library target existed, the next question was:

```text
Can another executable link against llama-hpx-engine and use the public API
without including gate CLI, submitter, validation, or gate_emit helpers?
```

Result:

```text
Yes.
llama-hpx-engine-smoke passed.
```

Smoke target:

```text
llama-hpx-engine-smoke
```

What it proves:

```text
a non-gate translation unit can link llama-hpx-engine
construct engine_options
construct engine
submit work through submit_request
await submit_handle.result
clean up without gate_validation or gate driver code
```

Known successful output:

```text
request_id=0 status=completed n_decoded=4 hash=0x6082b8ce0fec12ec
request_id=1 status=completed n_decoded=8 hash=0x7af78f8741363b89
HPX_ENGINE_SMOKE: PASS
```

The smoke intentionally used a small cooperating shape:

```text
one initial active request
one submit_request external request
reuse_completed = true
```

Interpretation:

```text
M2e proved the engine library is linkable and callable outside the gate,
but it also exposed a limitation: submit_request still needed an already-active
request to free a slot.
```

---

### 7.6 M2f added initial idle-slot admission

M2e exposed that the public submit API was usable, but not yet a natural serving shape.
A caller still needed a dummy initial active request to create reusable capacity.

Question:

```text
Can the engine start with real idle capacity and admit submit_request work directly?
```

Result:

```text
Yes.
M2f added initial idle-slot admission.
```

New engine option:

```cpp
int32_t initial_idle_slots = 0;
```

New admission source:

```cpp
admission_source::initial_idle
```

New engine state:

```text
free_idle_
```

New behavior:

```text
The engine can start with zero initial active requests.
Idle seq_ids are initialized as available capacity.
submit_request pushes work into the inbox.
inbox_has_pending() keeps the engine loop alive long enough to drain the inbox.
The engine admits queued work into an initial idle slot.
```

Admission priority became:

```text
1. cancel-freed slots
2. completion-freed slots
3. initial-idle slots
```

Important compatibility point:

```text
initial_idle_slots defaults to 0.
The existing gate never enters the initial_idle path.
Existing gate trace labels and counters stayed byte-identical.
```

Smoke target:

```text
llama-hpx-engine-idle-smoke
```

Known successful output:

```text
request_id=42 status=completed n_decoded=8 hash=0x7af78f8741363b89 admission_source=initial_idle
HPX_ENGINE_IDLE_SMOKE: PASS
```

Interpretation:

```text
M2f made submit_request usable in the shape a future server actually needs:
start the engine with capacity, submit work, and complete through the public handle.
No dummy active request is required.
```

---

### 7.7 M2g made submit_handle.stream real

After M2f, submit_request could complete through submit_handle.result.
The last public API gap was streaming.

Before M2g:

```text
submit_handle.stream existed as an optional token_stream_receiver,
but submit_request(want_stream=true) still threw before committing state.
```

Question:

```text
Can submit_request(want_stream=true) return a real token stream receiver
through submit_handle.stream?
```

Result:

```text
Yes.
M2g implemented public per-request streaming through submit_handle.stream.
```

Implementation model:

```text
submit_request(want_stream=true):
  creates token_stream_channel
  creates token_stream_receiver
  returns receiver through submit_handle.stream
  moves channel into arrival_msg.stream_channel

engine drain path:
  moves arrival_msg.stream_channel into external_stream_channels_[request_id]

admission path:
  admit_one checks external_stream_channels_ for request_id
  if present, moves the caller-supplied channel into seq_state::stream_channel
  opens the stream
  does not push admitted_stream_handoff for this request

publish/close path:
  existing publish_token and close_stream logic publishes token events
  stream close happens before result promise fulfillment
```

Precedence rule:

```text
Caller-supplied submit_handle stream wins over the gate's stream_all_ handoff path.
```

Gate compatibility:

```text
The gate submitter still calls submit_request with want_stream=false.
Therefore, the existing stream_all_ / admitted_stream_handoffs_ path remains unchanged.
```

Smoke target:

```text
llama-hpx-engine-stream-smoke
```

Known successful output:

```text
request_id=42 status=completed n_decoded=8 hash=0x7af78f8741363b89 admission_source=initial_idle streamed_tokens=8 streamed_hash=0x7af78f8741363b89
HPX_ENGINE_STREAM_SMOKE: PASS
```

Important checks:

```text
streamed_tokens = 8
streamed_hash == request_result.hash
stream close reason = completed
streams_opened = 1
streams_closed_completed = 1
stream_tokens_emitted_total = 8
residual_kv_ok = true
decode_failures = 0
```

Interpretation:

```text
A non-gate client can now submit a request and receive both:
  an HPX future for final completion
  an HPX token stream receiver for incremental token delivery
```

This is the public serving shape M3 can build on.

---

### 7.8 M2 package validation result

Build targets:

```text
llama-hpx-engine
llama-hpx-continuous-batch-gate
llama-hpx-engine-smoke
llama-hpx-engine-idle-smoke
llama-hpx-engine-stream-smoke
```

Canonical gate smokes:

```text
1. Slice 7 multi-cycle streaming
2. Slice 7 trace-on
3. Streaming Slice 5 external arrival
4. Streaming Slice 6 external × cancel-freed
5. Default-mode regression
6. Stream-off Live Admission Slice 7
7. M1d prompt-file smoke
```

Canonical command caveat:

```text
Slice 6 external × cancel-freed requires:
  --cancel-plan 0 --cancel-after 8

Stream-off Live Admission Slice 7 requires:
  --cancel-plan 1,4,7,2,5,8
```

Validation result:

```text
All seven canonical gate smokes passed.
Normalized stdout diff remained byte-identical except timing fields:
  wall_ms
  ttc_ms_*
  admitted_ttc_ms[*]
```

Trace result:

```text
Trace-off smokes:
  0 [hpx-cb-gate] event= lines

Trace-on smoke:
  174 events
  per-event-name totals matched baseline
```

Engine smoke result:

```text
HPX_ENGINE_SMOKE: PASS
HPX_ENGINE_IDLE_SMOKE: PASS
HPX_ENGINE_STREAM_SMOKE: PASS
```

HPX-native boundary checks:

```text
engine::submit_request:
  no llama_decode
  no common_batch_add
  no llama mutable execution calls

submitter.cpp:
  no llama mutable execution calls

engine smoke clients:
  no llama_decode
  no llama_batch_*
  no llama_memory_seq_*
  no llama_get_logits_ith
  no common_batch_add
```

The only allowed llama/common calls outside the engine task are:

```text
model/context setup
model/context cleanup
tokenization
```

All llama execution and KV mutation remain inside:

```text
engine::run()
and its private helpers
```

---

### 7.9 Public API as of M2

The reusable engine package now exposes the following serving-shaped API:

```text
engine_options:
  ctx
  vocab
  n_vocab
  batch_capacity
  n_seq_max
  initial_idle_slots
  stream_all
  plus legacy gate-shaped fields still used by the gate
```

```text
submit_request:
  request_id
  prompt_tokens
  decode_budget
  want_stream
```

```text
submit_handle:
  hpx::future<request_result> result
  optional HPX token stream receiver
```

```text
request_result:
  request_id
  status
  generated token count
  hash
  admission/source metadata
  KV/position accounting
```

```text
token_stream_event:
  token event
  closed event
  close reason
```

The simplest M2 serving shape is now:

```text
engine_options.initial_idle_slots = N
engine eng(std::move(opts));

submit_request req;
req.request_id = ...;
req.prompt_tokens = ...;
req.decode_budget = ...;
req.want_stream = true or false;

submit_handle h = eng.submit_request(std::move(req));

hpx::async([&] { eng.run(); });

if (h.stream) {
  drain token_stream_event values
}

request_result rr = h.result.get();
```

Interpretation:

```text
M2 turns the previous gate-internal engine into a reusable in-process HPX serving component.
```

---

### 7.10 What M2 intentionally does not claim

M2 does not claim:

```text
No HTTP/server transport yet.
No long-running engine loop yet.
No shutdown signal yet.
No public cancellation handle yet.
No sampling_config yet.
No llama_sampler integration yet.
No production backpressure yet.
No deadline or timeout API yet.
No multi-engine orchestration yet.
No metrics export from a running engine yet.
No performance advantage claim.
No comparison against llama-server yet.
```

Some public fields are still gate-shaped:

```text
engine_options::prompt_tokens
engine_options::budgets
engine_options::waiting_queue
engine_options::per_active_prompt_tokens
engine_options::cancel_plan
engine_options::release_iter_set
engine_options::stream_all
engine_options::max_decode_iters
```

These are acceptable for M2 because the gate still uses them.
They should be cleaned up later after M3 clarifies the server-facing surface.

---

### 7.11 Overall conclusion

What was proven in this package:

```text
Requests can carry their own prompt tokens.
A public submit_request can enqueue work.
A public submit_handle returns an HPX future.
A public submit_handle can return an HPX token stream receiver.
The engine can start with idle capacity and admit submitted work.
The engine can build as a reusable static library.
Non-gate smoke clients can consume the engine package.
The existing gate behavior remains byte-identical.
Only the engine task touches llama.cpp mutable execution state.
```

The reusable package now contains:

```text
llama-hpx-engine
llama-hpx-continuous-batch-gate
llama-hpx-engine-smoke
llama-hpx-engine-idle-smoke
llama-hpx-engine-stream-smoke
```

This closes the M2 package.

Next natural milestone:

```text
M3 planning.
```

M3 should decide the next serving surface before implementation:

```text
Option A:
  long-running engine loop
  shutdown signal
  public cancellation handle

Option B:
  sampling_config
  llama_sampler pass-through

Option C:
  HTTP/server wrapper around submit_request / submit_handle
```

Recommended caution:

```text
Do not jump directly into HTTP implementation without deciding engine lifetime,
shutdown/cancellation, and sampling scope.
```

---

## 8. hpx-cb-gate: long-running engine + cancellation control-plane

After M2 the engine package was reusable and `submit_request` worked end-to-end,
but the engine was still a one-shot loop:
it ran until every initial active and waiting request was done,
then it returned.

The next question became:

```text
Can the HPX continuous-batch engine become a long-running serving component
with a public shutdown signal and a public cancellation API,
while preserving the existing gate behavior byte-for-byte?
```

Result:

```text
Yes.
M3 added a keep-alive lifecycle, a public shutdown signal,
and a uniform rid-based public cancellation API that covers
both queued-before-admission and active/admitted requests.
```

This section records the work after M2:

```text
M3a:
  add engine_options::keep_alive
  add engine::request_shutdown()
  add HPX-native idle-wait on hpx::condition_variable_any
  keep inbox_mtx_ as hpx::spinlock

M3b:
  add engine::cancel_request(int32_t request_id)
  cancel queued-before-admission requests
  fulfill with status=cancelled, n_decoded=0, kv_cleared=false
  close stream with reason=cancelled
  no KV touched

M3c:
  extend engine::cancel_request to active/admitted requests
  set seq.cancel_requested from the engine task
  reuse cancel_should_observe / cancel_and_fulfill
  KV clear + stream close + promise fulfillment at iter boundary
  no interruption inside llama_decode
```

This section is still correctness, lifecycle, and packaging provenance.
It is not a performance claim.

---

### 8.1 M3a added keep-alive and a public shutdown signal

Before M3a, `eng.run()` returned as soon as every active and waiting request was done.
For a long-running server that would force the caller to recreate the engine for every request batch.

Question:

```text
Can the engine task stay alive between requests
without burning a CPU spin-waiting,
and exit cleanly only when the caller signals shutdown?
```

Result:

```text
Yes.
With engine_options::keep_alive = true the engine task suspends on
hpx::condition_variable_any when there is no work,
wakes on submit() or request_shutdown(),
and exits cleanly when request_shutdown() is observed on a fully-drained engine.
```

M3a wired:

```text
engine_options::keep_alive            (default false)
engine::request_shutdown()
hpx::condition_variable_any inbox_cv_
inbox_mtx_ remains hpx::spinlock      (BasicLockable; condition_variable_any accepts it)
engine_result::engine_idle_waits
engine_result::engine_shutdown_observed
```

Idle-wait predicate (observed under inbox_mtx_):

```text
!inbox_.empty() || !cancel_inbox_.empty() || shutdown_requested_
```

Wake sources (each publishes its state under inbox_mtx_ before notifying inbox_cv_):

```text
engine::submit()           inbox push, then notify
engine::cancel_request()   cancel_inbox_ push, then notify
engine::request_shutdown() shutdown_requested_ = true, then notify
```

With `keep_alive = false` the entire keep-alive outer loop runs once and exits,
matching the pre-M3 behavior byte-for-byte:

```text
engine_idle_waits          == 0
engine_shutdown_observed   == 0
```

Smoke evidence:

```text
llama-hpx-engine-keepalive-smoke
  request 42 (budget=8) -> completed
  request 43 (budget=8) -> completed
  engine_idle_waits >= 1
  engine_shutdown_observed == 1
```

Interpretation:

```text
The engine task can now host a long-running serving lifecycle
without changing how existing gate runs behave.
```

---

### 8.2 M3b added queued-before-admission cancellation

Before M3b, the only way to cancel a request was the scripted `cancel_plan` + `cancel_after` path,
which is gate-test plumbing.
A real serving caller needs a public way to cancel a request after submission but before admission.

Question:

```text
Can a public cancel_request(rid) cancel a request that has been submitted
but not yet admitted to a seq_id,
without ever touching llama mutable state from the caller?
```

Result:

```text
Yes.
engine::cancel_request(rid) is non-blocking, safe from any HPX task,
and fulfills the matching request_result with status=cancelled
without binding the request to a seq_id or touching KV.
```

M3b wired:

```text
engine::cancel_request(int32_t request_id)
cancel_inbox_                              (engine-internal deque)
cancelled_request_ids_                     (engine-task-only set)
drain_cancel_inbox()                       (engine task only)
apply_queued_cancellations()               (engine task only)
fulfill_queued_cancelled()                 (engine task only)
drain_external_inbox cancel-before-submit-drain shortcut
```

Queued-cancel snapshot:

```text
request_result.status                = cancelled
request_result.n_decoded             = 0
request_result.admitted_at_iter      = -1
request_result.kv_cleared            = false
stream terminal event.close_reason   = cancelled
```

Counters:

```text
queued_cancelled
cancel_request_calls
cancel_active_not_supported       (legacy stub; preserved by M3c, see 8.3)
cancel_unknown_request_id
cancel_request_duplicates
```

Smoke evidence:

```text
llama-hpx-engine-queued-cancel-smoke
  request_id=42 status=cancelled n_decoded=0 admitted_at_iter=-1 stream_close=cancelled
  queued_cancelled == 1
  streams_opened == 1, streams_closed_cancelled == 1
  residual_kv_ok == true
  engine_shutdown_observed == 1
```

Boundary:

```text
engine::cancel_request has no llama_* call sites.
drain_cancel_inbox has no llama_* call sites.
apply_queued_cancellations has no llama_* call sites.
fulfill_queued_cancelled has no llama_* call sites.
No KV mutation on the queued-cancel path.
```

Interpretation:

```text
A caller can now cancel a request between submit_request and admission
through a public API that never touches llama execution state.
```

---

### 8.3 M3c extended cancel_request to active/admitted requests

After M3b, a request that had already been admitted could not be cancelled through the public API.
The engine still finished the request to its full decode budget, ignoring any incoming cancel.

Question:

```text
Can engine::cancel_request(rid) cancel a request that has been admitted to a seq
and is actively decoding,
cooperatively at an iteration boundary,
without interrupting llama_decode?
```

Result:

```text
Yes.
M3c routes active cancellation through the same public API as M3b
(engine::cancel_request(rid)),
preserves the iter-boundary observation discipline,
and reuses the existing cancel_should_observe / cancel_and_fulfill pipeline.
```

M3c bridge (apply_queued_cancellations (b) branch):

```text
find first non-done seq with seq.request_id == rid
set seq.cancel_requested.store(true, std::memory_order_release)
bump cancel_active_observed
emit trace event cancel_active_observed
erase rid from cancelled_request_ids_
```

Downstream (unchanged from Cancel Slice 2):

```text
cancel_should_observe(seq)
  reads seq.cancel_requested.load(acquire)
  returns true at iter top

cancel_and_fulfill(seq, iter, mem)
  clear KV via clear_and_check
  close stream with reason=cancelled
  push seq_id onto free_due_to_cancel_
  fulfill request_result.status=cancelled exactly once
```

Active-cancel snapshot:

```text
request_result.status                = cancelled
request_result.n_decoded             >= 1
request_result.n_decoded             <  decode_budget
request_result.cancel_observed_iter  >= 1
request_result.n_decoded_at_cancel   == request_result.n_decoded
request_result.kv_cleared            = true
request_result.pos_max_at_clear      >= 0
stream terminal event.close_reason   = cancelled
```

New counter:

```text
cancel_active_observed
```

Legacy counter:

```text
cancel_active_not_supported
  retained for back-compat with archived M3b evidence
  must stay zero in M3c+ runs
  new code MUST NOT bump it
```

Smoke evidence:

```text
llama-hpx-engine-active-cancel-smoke
  request_id=42 status=cancelled n_decoded=1 stream_close=cancelled
  cancel_active_observed == 1
  cancel_active_not_supported == 0
  cancelled_count == 1
  queued_cancelled == 0
  residual_kv_ok == true
  engine_shutdown_observed == 1
```

Trace evidence:

```text
cancel_drained request=42
cancel_active_observed request=42 seq_id=0
cancel_observed seq=0 iter=2 n_decoded=1
token_stream_closed request=42 seq_id=0 n_tokens=1 reason=cancelled
cancel_kv_cleared seq=0 pos_max_at_clear=4 cross_talk_ok=1
cancel_future_fulfilled seq=0 status=cancelled
```

Boundary:

```text
cancel_request never touches llama_*.
The seq.cancel_requested.store(true) call happens on the engine task
inside apply_queued_cancellations.
KV clear / stream close / promise fulfillment all happen on the engine task
inside cancel_and_fulfill, at iter boundary, never inside llama_decode.
```

Interpretation:

```text
A caller can now cancel an admitted, mid-decode request
through the same rid-based public API as queued cancellation,
with no separate ergonomic surface.
```

---

### 8.4 HPX-native boundary preserved across M3

M3 added a long-running engine and a public cancellation API
without changing who owns what:

```text
HPX owns:
  request lifecycle
  admission
  futures/promises
  stream handoff
  keep-alive / shutdown
  cancellation control-plane
  orchestration
  traces/counters

llama.cpp owns:
  llama_decode
  KV implementation
  tokenizer behavior
  sampler math
  ggml graph execution
  kernels
```

Grep result on `engine.cpp` for `llama_` / `common_batch_` / `common_tokenize` call sites
in the M3 control-plane methods:

```text
engine::submit_request              0
engine::submit                      0
engine::request_shutdown            0
engine::cancel_request              0
engine::drain_cancel_inbox          0
engine::apply_queued_cancellations  0
engine::fulfill_queued_cancelled    0
```

Smoke clients (`engine_smoke`, `engine_idle_smoke`, `engine_stream_smoke`,
`engine_keepalive_smoke`, `engine_queued_cancel_smoke`,
`engine_active_cancel_smoke`) call llama APIs only for backend / model / context
setup, tokenization, and final cleanup. None of them call:

```text
llama_decode
llama_batch_*
llama_memory_seq_*
llama_get_logits_ith
common_batch_add
```

Only `engine::run` and the private execution helpers it calls touch llama mutable execution state.

---

### 8.5 M3 package validation result

New engine smokes:

```text
llama-hpx-engine-keepalive-smoke
llama-hpx-engine-queued-cancel-smoke
llama-hpx-engine-active-cancel-smoke
```

Existing engine smokes (still pass byte-identical to M2):

```text
llama-hpx-engine-smoke
llama-hpx-engine-idle-smoke
llama-hpx-engine-stream-smoke
```

Seven canonical gate smokes (still PASS, normalized stdout diff against
M3b baselines is empty modulo timing fields `wall_ms` / `ttc_ms_*` /
`admitted_ttc_ms[*]` and the `model_path` / `request_prompts_file` lines):

```text
Slice 7 multi-cycle streaming
Slice 7 trace-on
Streaming Slice 5 external arrival
Streaming Slice 6 external × cancel-freed
Default-mode regression
Stream-off Live Admission Slice 7
M1d prompt-file smoke
```

Trace evidence:

```text
trace-off gate smokes have 0 [hpx-cb-gate] event= lines
trace-on Slice 7 has the same per-event-name totals as the M3b baseline
no cancel_active_not_supported_in_m3b event in any M3c+ run
```

Interpretation:

```text
The M3 control-plane additions do not regress any existing engine smoke
or gate smoke. The new cancellation paths are covered end-to-end by
dedicated smoke clients.
```

---

### 8.6 Public API as of M3

The engine package now exposes the following serving-shaped API:

```text
engine_options:
  ctx
  vocab
  n_vocab
  batch_capacity
  n_seq_max
  initial_idle_slots
  stream_all
  keep_alive                              (M3a)
  plus legacy gate-shaped fields still used by the gate
```

```text
submit_request:
  request_id
  prompt_tokens
  decode_budget
  want_stream
```

```text
submit_handle:
  hpx::future<request_result> result
  optional HPX token stream receiver
```

```text
engine::submit_request(submit_request)            (M2b)
engine::request_shutdown()                        (M3a)
engine::cancel_request(int32_t request_id)        (M3b + M3c)
engine::run()
```

```text
request_result:
  request_id
  status                  completed | cancelled | failed_reserved
  n_decoded
  hash
  done_iter
  kv_cleared
  pos_max_at_clear
  cancel_observed_iter
  n_decoded_at_cancel
  admission/source metadata
  ttc_us
```

```text
token_stream_event:
  token event
  closed event with close_reason  completed | cancelled | error
```

The simplest M3 long-running serving shape is now:

```text
engine_options.initial_idle_slots = N
engine_options.keep_alive         = true
engine eng(std::move(opts));

hpx::async([&] { eng.run(); });

for each work item:
  submit_request req;
  req.request_id    = ...;
  req.prompt_tokens = ...;
  req.decode_budget = ...;
  req.want_stream   = true or false;
  submit_handle h = eng.submit_request(std::move(req));
  ...
  optionally eng.cancel_request(req.request_id);
  request_result rr = h.result.get();

eng.request_shutdown();
```

Interpretation:

```text
M3 turns the M2 reusable engine into a long-running HPX serving component
with a uniform public cancellation API.
```

---

### 8.7 What M3 intentionally does not claim

M3 does not claim:

```text
No submit_handle.cancel() ergonomic wrapper.
No cancellation outcome return value on cancel_request (outcome is on request_result).
No sampling_config.
No llama_sampler integration.
No HTTP / gRPC / Unix-socket / WebSocket server wrapper.
No deadline / timeout API.
No production backpressure on token streams.
No engine-internal metrics export during a long-running engine.
No engine_options cleanup separating library-public fields from gate-test fields.
No multi-engine orchestration / pool.
No reason=error stream-close semantics beyond the engine's bail-out path.
No performance advantage claim.
No comparison against llama-server.
```

Public fields and helpers still gate-shaped (acceptable for M3, candidate cleanup for the next milestone):

```text
engine_options::prompt_tokens
engine_options::budgets
engine_options::waiting_queue                  (must be non-null even when unused)
engine_options::per_active_prompt_tokens
engine_options::cancel_plan
engine_options::cancel_after
engine_options::release_iter_set
engine_options::max_decode_iters
engine_options::stream_all

engine::submit(arrival_msg)                    (internal POD entry; submit_request preferred)
engine::register_external_release_iter         (gate scripted release/ack barrier only)

engine_result::cancel_active_not_supported     (legacy M3b counter, must remain 0 in M3c+)
```

---

### 8.8 Overall conclusion

What was proven in this package:

```text
The engine task can host a long-running serving lifecycle gated by keep_alive.
The engine task can suspend on an HPX condition variable when idle and exit
  cleanly on a public request_shutdown signal.
A single public cancel_request(rid) API can resolve both
  queued-before-admission and active/admitted requests.
Queued cancellation never touches KV.
Active cancellation reuses the existing iter-boundary cancellation pipeline
  for KV clear, stream close, and promise fulfillment.
No interruption ever happens inside llama_decode.
Only the engine task touches llama.cpp mutable execution state.
The existing gate behavior and the existing engine smokes remain byte-identical.
```

The reusable package now contains:

```text
llama-hpx-engine
llama-hpx-continuous-batch-gate
llama-hpx-engine-smoke
llama-hpx-engine-idle-smoke
llama-hpx-engine-stream-smoke
llama-hpx-engine-keepalive-smoke
llama-hpx-engine-queued-cancel-smoke
llama-hpx-engine-active-cancel-smoke
```

This closes the M3 package.

Next natural milestone candidates (planning only):

```text
engine_options cleanup:
  split gate-flavored fields from library-public fields
  make engine::submit / engine::register_external_release_iter gate-only

submit_handle.cancel() ergonomic wrapper:
  thin wrapper over engine::cancel_request(rid)
  depends on engine_options cleanup

sampling_config / llama_sampler pass-through:
  per-request sampling on the engine task
  argmax stays as the default

HTTP / gRPC / Unix-socket server wrapper:
  depends on sampling and the cleaned-up engine_options
```

Recommended caution:

```text
Do not skip engine_options cleanup before adding HTTP transport.
Do not introduce backpressure or bounded channels without a real serving workload to measure.
Do not remove the legacy cancel_active_not_supported counter while archived
  M3b evidence is still consumed; deprecate first, remove later.
```

## 9. Add HPX serving layer M4-M8 milestones

This section records the documentation update that adds the stable milestone summary for the HPX Serving Layer for llama.cpp.


```text
docs/hpx/hpx_serving_layer_m0_m8_milestone_summary.md
```


### 9.1 Scope

The new milestone summary documents the serving-layer work from the engine cleanup milestones through the matched benchmark harness:

```text
M4  engine surface cleanup and library-vs-gate split
M5  cancellation identity hardening
M6  per-request sampling
M7  hpx-server HTTP/SSE adapter
M8  matched hpx-server vs llama-server benchmark harness
```

The document also includes a short orientation for the earlier foundation milestones M0-M3, but the commit is framed around the M4-M8 serving-layer progression.

### 9.2 Ownership boundary recorded

The summary records the current ownership boundary:

```text
HPX owns:
  request lifecycle
  admission
  futures/promises
  streaming handoff
  cancellation
  keep-alive/shutdown
  backpressure
  server adapter control path
  matched benchmark harness

llama.cpp owns:
  llama_model / llama_context execution semantics
  llama_decode
  KV memory implementation
  sampler math
  tokenizer/vocab behavior
  ggml graph
  CPU/Metal/backend kernels
```

The document preserves the hard invariant:

```text
Only the engine task touches mutable llama.cpp execution state.
```

### 9.3 M4-M8 outcomes recorded

M4 records the engine surface cleanup:

```text
engine_options split into opts.lib / opts.preload / opts.gate_test
submit_request-only users no longer need fake waiting_queue or fake prompt_tokens
HPX-native cleanup preserved
canonical gate output stayed byte-identical
```

M5 records cancellation identity hardening:

```text
cancel_token { request_id, epoch }
stale-token protection for reused request_id
submit_handle::cancel(engine &) explicit-engine forwarder
no zero-arg handle cancel
no raw engine pointer observer
```

M6 records per-request sampling:

```text
sampling_config plumbed end-to-end
stochastic mode uses per-seq llama_sampler chains
top_k / top_p / temperature wired
greedy/default path remains local argmax and byte-identical
sampler isolation smoke passes
```

M7 records the HTTP/SSE serving adapter:

```text
tools/hpx-server/ added
POST /completion non-streaming
optional sampling JSON
SSE streaming
client-disconnect cancellation
--max-concurrent backpressure
stream-disconnect capacity release proof
cpp-httplib documented as the explicit non-HPX HTTP adapter boundary
```

M8 records the matched benchmark harness:

```text
hpx-bench/experiments/12_hpx_vs_llama_server_pair/ added
hpx-server --ctx-size precursor added for matched n_ctx
minimal matched pair-run
repeat stability
non-streaming 2x2 matrix
EOG-stop semantics correction
canonical streaming TTFT capture
streaming 2x2 matrix
```

### 9.4 Validation anchors recorded

The milestone summary records these stable anchors:

```text
HPX canonical greedy p0_b8:
  0x0619d4d1900c2365

HPX greedy p0_b32:
  0x6794e47fe0f84af1

Default stochastic seed=42:
  0xa8e14acb4094aa3f
```

M8 evidence run IDs recorded:

```text
20260521-231710-pair  # non-streaming 2x2 matrix, gates=PASS
20260521-233521-pair  # canonical streaming, 10 repeats, gates=PASS
20260521-234615-pair  # streaming 2x2 matrix, gates=PASS
```

The document states that the benchmark harness records raw values only. It does not publish aggregation, percentiles, averaged TTFT, throughput claims, or cross-server performance conclusions.

### 9.5 EOG-stop and streaming semantics recorded

The summary records the corrected serving and harness semantics:

```text
decode_budget / n_predict is an upper bound, not an exact-count guarantee.
```

For the EOG-sensitive p1 prompt, the documented behavior is:

```text
hpx-server:
  n_decoded = 0
  hash = 0x0000000000000000
  empty text

llama-server:
  tokens_predicted = 1
  empty content
  stop_type = eos
```

The document records that:

```text
cross-server n_decoded equality is recorded, not gated
cross-server text equality is not gated
```

For streaming mode, it records that:

```text
hpx-server streams per-token detokenized text
llama-server streams chunk text
streaming text reconstruction differs between the two servers
streaming ok=True means a clean terminal stream record, not necessarily non-empty text
```

### 9.6 HPX nativity assessment recorded

The summary records the current HPX-nativity assessment:

```text
No live std::thread / std::mutex / std::condition_variable /
std::this_thread::sleep_for in authored HPX engine/server control-plane code,
based on current validation reports.
```

Accepted boundaries and caveats:

```text
std::atomic appears only in accepted adapter-boundary/counter contexts
cpp-httplib remains the explicit non-HPX HTTP adapter boundary
Python benchmark harness is external measurement infrastructure, not HPX runtime code
```

### 9.7 Known limitations recorded

The document records these limitations:

```text
No OpenAI-compatible endpoint yet.
cpp-httplib remains the non-HPX HTTP adapter boundary.
No auth, TLS, or production-grade shutdown.
No multi-model loading or model swap.
No HTTP-side queueing beyond strict door-cap rejection.
hpx-server does not expose prompt-token count in /completion.
Benchmark harness records raw values only.
No performance claims.
Handler/test duplication exists in server smokes.
```


## 10. Add HPX serving-control placement and responsiveness evidence

This section records the post-M8 HPX serving-control checkpoint. It covers the
HPX-native engine inbox, runtime placement, hpx-server placement support,
responsiveness experiments, and the shutdown-liveness fix for queued-but-unadmitted
requests.

### 10.1 Engine inbox and phase-structured actor loop recorded

This checkpoint records the N1 HPX-native inbox rewrite:

```text
producer-side mutex/cv/deque inbox removed
HPX local channel inbox added
typed inbox_msg used for arrivals, cancels, token cancels, and shutdown
engine-task-only staged queues added
submit_request preserves epoch assignment + publish ordering
```

The engine actor remains the only owner of llama.cpp mutable execution state:

```text
one llama_context
one llama_batch
one engine task
many request ids / seq ids
per-request futures/promises
per-request stream channels
```

The N3.0 phase extraction is recorded as a structural refactor, not a semantic change:

```text
iter_observe_cancellations
iter_run_admissions
iter_build_batch
iter_run_decode
iter_sample_and_finalize
iter_fire_release_ack_barrier
```

The documented phase-order invariants are preserved:

```text
cancel observation occurs outside llama_decode
admission priority is preserved
llama_decode remains between batch build and sampling
publish_token occurs before n_decoded++
close_stream occurs before set_value
```

### 10.2 Engine-pool runtime placement recorded

This checkpoint records opt-in HPX runtime placement support:

```text
runtime_config { enable_engine_pool }
hpx_runtime::start_once(os_threads, runtime_config)
named single-PU "engine" pool via HPX resource partitioner
hpx_runtime::async_on_engine(...)
LLAMA_HPX_PLACEMENT_TRACE=1 placement evidence
```

The default path remains unchanged:

```text
--engine-pool off:
  async_on_engine falls through to hpx::async

--engine-pool on:
  engine actor runs on the named "engine" pool
```

The gate exposes:

```text
--engine-pool
--hpx-os-threads
```

The placement path is fail-closed:

```text
--engine-pool requires enough HPX OS threads
engine-pool creation is checked after runtime start
async_on_engine rechecks placement availability at spawn time
no silent fallback to default pool when engine-pool was requested
```

### 10.3 Pump cooperativity and queued-cancel race fix recorded

This checkpoint records placement-aware pump cooperativity:

```text
engine_options::lib.cooperative_yield_on_pump
default-pool path keeps the cooperative yield
engine-pool path disables the pump yield
engine remains placement-agnostic except for ctor-time options
```

It also records the outer-tail staged-work fix for the queued-cancel race:

```text
after pump_inbox_nonblocking(), the engine must not park if staged work exists
staged arrivals or staged cancels force the loop to continue
request_shutdown remains governed by the shutdown predicate
```

The actor invariant recorded for this fix is:

```text
never suspend on the inbox while actionable staged work is already staged
```

The reproducer evidence came from queued-cancel and responsiveness-bench runs. The
fixed path was validated by existing queued-cancel smokes, the engine-pool queued-cancel
smoke, the default gate, and the placement-on gate.

### 10.4 hpx-server engine-pool support recorded

This checkpoint records N4 hpx-server placement support:

```text
tools/hpx-server/ accepts --engine-pool
tools/hpx-server/ accepts --hpx-os-threads
hpx-server uses hpx_runtime::start_once(..., runtime_config)
hpx-server spawns the engine via hpx_runtime::async_on_engine
cooperative_yield_on_pump is disabled when --engine-pool is active
```

Default behavior remains unchanged:

```text
--engine-pool is off by default
hpx-server default path remains on the HPX default pool
HTTP/SSE adapter semantics are unchanged
cpp-httplib remains the explicit non-HPX HTTP adapter boundary
```

A dedicated N4 smoke was added:

```text
llama-hpx-server-engine-pool-smoke
```

The smoke covers:

```text
engine-pool placement
canonical round-trip request
SSE client-disconnect path
post-disconnect sanity request
clean engine/server shutdown
```

The recorded placement evidence includes:

```text
engine_task_placement pool=engine
```

### 10.5 Experiment 13 recorded: internal control-plane responsiveness

Experiment 13 was added under:

```text
hpx-bench/experiments/13_control_plane_responsiveness/
```

It records internal HPX control-plane responsiveness, not decode speed or throughput.

Experiment 13 Phase 1 measured:

```text
W2 queued-cancel responsiveness
W3 multi-request streaming with serial stream drain
```

Experiment 13 Phase 2 corrected W3 measurement:

```text
W3 stream consumers changed to concurrent consumer-side draining
token_publish_us renamed / reframed as token_receive_us
inter-token gap now records adapter-observed receive cadence
```

Recorded interpretation:

```text
W2 queued-cancel:
  engine-pool improves control-plane tail latency

W3 streaming:
  decode-dominated
  engine-pool does not improve streaming completion or throughput
```

The valid performance claim is narrow:

```text
engine-pool placement improves queued-cancel / control-plane tail responsiveness
in Experiment 13
```

The document does not claim:

```text
HPX improves llama_decode speed
HPX improves token throughput
HPX improves decode-dominated streaming completion
HPX is faster than upstream llama.cpp
```

### 10.6 Experiment 14 recorded: end-to-end hpx-server responsiveness

Experiment 14 was added under:

```text
hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/
```

It records client-visible hpx-server behavior, not internal engine timing.

Experiment 14 Phase 1 compares hpx-server placement modes:

```text
default_os1
default_os2
engine_pool_os2
```

The Phase 1 scope intentionally excludes llama-server:

```text
no llama-server comparison in Phase 1
no cross-server conclusion
no upstream-vs-HPX performance claim
```

Recorded workloads include:

```text
W1 non-streaming round-trip
W2 full SSE streaming
W3 SSE disconnect / client-visible disconnect behavior
```

Recorded interpretation:

```text
default_os1 -> default_os2:
  improves W1/W2 client-visible latency

default_os2 -> engine_pool_os2:
  neutral-to-slightly-negative on end-to-end client-visible metrics
```

The recorded conclusion is:

```text
engine-pool remains opt-in
engine-pool is useful for placement/correctness diagnostics and internal control-plane
tail behavior
engine-pool is not an end-to-end latency win for this single-client decode-dominated
Exp14 workload
```

### 10.7 N5b shutdown-liveness fix recorded

This checkpoint records the N5b shutdown-liveness issue and fix.

The isolated N5b bug shape is:

```text
request is queued
no admission source exists
request_shutdown() is issued
engine must resolve queued work and join
```

A new smoke was added:

```text
llama-hpx-engine-shutdown-queued-unadmittable-smoke
```

The pre-fix red behavior was:

```text
request queued and not admitted
request_shutdown issued
engine did not join within bounded wait
controlled FAIL, no crash, no SIGKILL, no uncaught exception
```

The accepted N5b fix records a shutdown contract:

```text
when shutdown is requested and no active seqs exist,
drain queued-but-not-admitted requests,
resolve their promises with a defined terminal status,
close queued stream channels if present,
observe shutdown,
exit the engine loop,
allow engine_fut.get() to return
```

Queued-at-shutdown requests resolve as:

```text
request_status::failed_reserved
```

This status is used to distinguish shutdown-aborted queued work from user cancellation:

```text
cancelled:
  user/request cancellation

failed_reserved:
  shutdown-aborted queued work
```

The N5b fix does not change:

```text
llama_decode
admission priority
active request behavior
cancel_and_fulfill
finalize_and_fulfill
stream close ordering for admitted/active requests
hpx-server
hpx_runtime
```

### 10.8 N5a remains deferred

N5a remains documented but not fixed.

The deferred N5a issue is:

```text
active cancel
-> slot enters cancel_freed path
-> next request admits from cancel_freed
-> that request completes naturally
-> later request may not be admitted because slot recovery is incomplete
```

The suspected root cause is in the completion-side slot repush logic:

```text
reuse_completed=false
admission_src=cancel_freed
finalize_and_fulfill does not return the seq_id to an admission pool
```

The proposed future fix is documented separately and was not applied in this checkpoint.

The checkpoint explicitly records:

```text
N5b shutdown liveness is fixed
N5a slot recovery remains deferred
```

### 10.9 Validation anchors recorded

The checkpoint preserves the canonical HPX hash anchor:

```text
HPX canonical greedy p0_b8:
  0x0619d4d1900c2365
```

Validated paths recorded for this checkpoint include:

```text
engine queued-cancel smoke
engine active-cancel smoke
engine keepalive multi-submit smoke
engine idle smoke
engine-pool queued-cancel smoke
N5b shutdown queued-unadmittable smoke
default continuous-batch gate
placement-on continuous-batch gate
hpx-server engine-pool smoke
```

The placement-on gate records:

```text
engine_task_placement pool=engine
p0_b8 hash unchanged
```

The hpx-server engine-pool smoke records:

```text
canonical round-trip hash unchanged
SSE disconnect path succeeds
residual_kv_ok = 1
decode_failures = 0
```

### 10.10 Documentation and experiment index recorded

This checkpoint adds or updates documentation for discoverability:

```text
README.md
docs/hpx/hpx_serving_layer_architecture.md
docs/hpx/n5_deferred_slot_recovery_note.md
hpx-bench/experiments/README.md
hpx-bench/experiments/13_control_plane_responsiveness/
hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/
```

The experiment index records:

```text
Experiment 13:
  internal HPX control-plane responsiveness

Experiment 14:
  end-to-end hpx-server client-visible responsiveness
```

The architecture documentation records the layering distinction:

```text
Layer 1:
  adapter boundary

Layer 2:
  runtime placement / engine-pool

Layer 3:
  engine actor

Layer 4:
  phase contracts

Layer 5:
  future dataflow evolution
```

