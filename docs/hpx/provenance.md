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

## 3. bench: package HPX serving-bench evidence

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