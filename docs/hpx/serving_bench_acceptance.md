# Serving-bench acceptance checks

This document summarizes the correctness and lifecycle checks used for
`llama-serving-bench`.

The purpose of these checks is to prove that std and HPX backends are
structurally correct before any performance comparison is interpreted.

These checks are not performance benchmarks.

## Acceptance philosophy

A serving-bench result is only meaningful if correctness is clean first.

A valid run must preserve:

- backend selection correctness
- prompt-fit validation
- token-stream determinism for greedy decode
- generated-token hash stability
- clean HPX-OFF behavior
- HPX runtime lifecycle correctness
- std-vs-HPX token-stream equality
- context-pool correctness under contention

Generated-token hashes are structural correctness signals. They are not
semantic quality metrics and not performance metrics.

## Canonical smoke input

The canonical smoke tests use:

```text
Model:  TinyLlama 1.1B Chat Q4_K_M
Prompt: Hello, my name is
Decode: greedy
```

For public reproduction, replace the model path with your own local path
to the same GGUF model.

## Pinned canonical values

For the canonical prompt and greedy decode:

```text
max_tokens=16:
  generated_token_hash=0x833045f1e2ebf49f
  n_tokens_generated=16

max_tokens=0:
  generated_token_hash=0x0000000000000000
  n_tokens_generated=0

max_tokens=32:
  generated_token_hash=0x6794e47fe0f84af1
```

The 32-token hash is informational. It helped rule out a degenerate
constant-hash bug.

## OFF / std baseline checks

These checks apply to the default build where HPX support is off.

### Test 1 — backend default is std

Confirms `--help` documents `--backend std|hpx` and lists `std` as the
default.

This proves backend selection is explicit and discoverable.

### Test 2 — unknown backend is rejected

Confirms an invalid backend name exits nonzero before model load.

This proves parse/config errors are separated from model execution.

### Test 3 — std smoke still runs

Runs one std request with the canonical prompt and `max_tokens=16`.

Expected:

```text
n_ok=1
n_error=0
n_tokens_generated=16
generated_token_hash=0x833045f1e2ebf49f
```

This proves the std backend still performs real llama.cpp decode.

### Test 4 — max_tokens=0 hash is zero

Runs one std request with `max_tokens=0`.

Expected:

```text
n_ok=1
n_error=0
n_tokens_generated=0
generated_token_hash=0x0000000000000000
```

This pins the empty-generation hash contract.

### Test 5 — same prompt is stable

Runs the same canonical std command twice.

Expected:

```text
same token count
same generated-token hash
```

This proves greedy decode is deterministic in the baseline.

### Test 6 — changed generation changes hash

Runs the same prompt with a different generation length, such as
`max_tokens=32`.

Expected:

```text
generated_token_hash=0x6794e47fe0f84af1
```

This is informational, not a strict benchmark gate. It helps catch the
degenerate bug where the hash never changes.

### Test 7 — HPX-OFF fails cleanly

Runs `--backend hpx` in a build where HPX support is off.

Expected:

```text
exit nonzero
no model load
no request submission
no aggregate line
```

Expected stderr:

```text
[serving-bench] HPX backend not built (LLAMA_SERVING_BENCH_HPX=OFF)
```

This proves the HPX-OFF path fails before engine initialization.

## HPX-ON gates

These gates apply to a build configured with HPX support enabled.

The HPX backend under test is HPX-native:

```text
HPX async request state machine
future-based context pool
exclusive llama_context lease through context_guard
normal opaque llama_decode inside the leased context
no permanent worker-per-context loop
no central CV/mutex worker queue
```

### Gate 1 — build / link

Confirms the HPX-ON build configures, compiles, links, and resolves HPX
from the intended install.

This must pass before any runtime checks.

### Gate 2 — std path regression

Runs the std backend from the HPX-capable binary.

Expected:

```text
n_ok=1
n_error=0
generated_token_hash=0x833045f1e2ebf49f
no HPX runtime start trace
```

This proves enabling HPX support does not perturb the std backend.

### Gate 3 — HPX smoke

Runs one canonical request through `--backend hpx`.

Expected:

```text
n_ok=1
n_error=0
n_tokens_generated=16
generated_token_hash is nonzero
engine_hpx ready trace appears when tracing is enabled
```

This proves the HPX backend can execute a simple request end-to-end.

### Gate 4 — std vs HPX hash equality

Runs the canonical request with both std and HPX backends.

Expected:

```text
std generated_token_hash == hpx generated_token_hash
std hash == hpx hash == 0x833045f1e2ebf49f
```

This is the main structural-fidelity gate. A mismatch means the HPX
decode path is not equivalent to std.

### Gate 5 — empty generation

Runs `--backend hpx --max-tokens 0`.

Expected:

```text
n_ok=1
n_error=0
n_tokens_generated=0
generated_token_hash=0x0000000000000000
no req[0] acquire trace
no req[0] release trace
```

This proves the HPX empty path bypasses the context pool and decode loop.

### Gate 6 — HPX repeat stability

Runs the canonical HPX request twice.

Expected:

```text
same token count
same generated-token hash
hash == 0x833045f1e2ebf49f
```

This checks HPX determinism for the canonical greedy case.

### Gate 7 — two-context HPX smoke

Runs multiple requests with two contexts and two-way concurrency.

Expected:

```text
n_ok=4
n_error=0
all requests produce generated_token_hash=0x833045f1e2ebf49f
```

This proves multiple leased contexts can run concurrently without
corrupting output.

### Gate 8 — lifecycle smoke

Uses HPX trace output to confirm lifecycle ownership.

Expected for `--backend std`:

```text
no hpx_runtime_start_once trace
no engine_hpx ready trace
no hpx_runtime_stop trace
```

Expected for `--backend hpx`:

```text
exactly one hpx_runtime_start_once trace
exactly one engine_hpx ready trace
exactly one hpx_runtime_stop trace
```

This proves the program-level lifecycle owns HPX startup/shutdown, not
the engine or std backend.

### Gate 9 — pool queueing under real decode

Runs more concurrent requests than contexts.

Expected:

```text
n_ok=4
n_error=0
all requests produce generated_token_hash=0x833045f1e2ebf49f
one HPX runtime start
one HPX runtime stop
no deadlock
```

This is the key context-pool contention gate. It verifies that queued
requests are woken and completed correctly when contexts are reused.

## Source-truth OFF checkpoint

The OFF/std acceptance checkpoint passed all seven tests against a fresh
build.

Summary:

```text
Tests 1–7: PASS
std canonical hash: 0x833045f1e2ebf49f
empty hash: 0x0000000000000000
32-token informational hash: 0x6794e47fe0f84af1
HPX-OFF stub: clean exit before model load
```

No performance interpretation was made from these tests.

## What invalidates benchmark interpretation

A benchmark result should not be interpreted if any of the following is
true:

- `n_error > 0`
- `n_cancelled > 0`
- generated-token hash count is wrong
- std and HPX use different prompts, models, token counts, or context
  shapes
- source changed between std and HPX runs
- HPX lifecycle traces show std accidentally starting HPX
- the run is a smoke test rather than a repeated benchmark

## What these checks do not prove

Passing these checks does not prove:

- HPX is faster than std
- HPX is better than upstream llama-server
- the pool-of-contexts design beats continuous batching
- the current HPX thread-count formula is optimal

They only prove that the serving-bench comparison is structurally valid
enough for benchmark protocol work to begin.

## Minimal stderr trace evidence

The public docs do not keep full stderr logs. The useful stderr evidence is limited to short traces that prove backend isolation, HPX lifecycle ownership, and context-pool behavior.

### HPX-OFF stub

In an HPX-OFF build, `--backend hpx` fails before model load:

```text
[serving-bench] HPX backend not built (LLAMA_SERVING_BENCH_HPX=OFF)
```

This supports the HPX-OFF acceptance gate: the backend name is valid, but the backend is unavailable in that build.

### HPX lifecycle under `--backend hpx`

With `LLAMA_SERVING_BENCH_HPX_TRACE=1`, the HPX queueing gate emitted:

```text
[serving-bench] hpx_runtime_start_once: starting (os_threads=1)
[serving-bench] engine_hpx ready: n_contexts=1 pool_size=1
[serving-bench] hpx_runtime_stop: stopping
```

This supports the lifecycle contract:

```text
HPX starts once before engine use.
engine_hpx initializes with the expected context-pool size.
HPX stops once after cleanup.
```

### Context-pool acquire/release trace

The queueing gate used one context with four concurrent requests. The trace showed paired acquire/release events:

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

This supports the context-pool contract:

```text
requests lease exactly one context
the context is released after each request
queued requests reuse the same context without deadlock
acquire/release counts match
```

### std backend remains cold

For the std-backend regression gate, the same HPX trace grep produced no lifecycle lines. This supports the rule:

```text
--backend std does not start HPX, even in an HPX-capable build.
```
