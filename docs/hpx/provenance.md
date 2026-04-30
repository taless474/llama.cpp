# HPX serving-bench provenance

## 1. Initial commit, add HPX serving-bench direction and docs

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
- write result artifacts to `/tmp`
