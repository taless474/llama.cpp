# Claude Guidance for HPX / llama.cpp Serving Work

## Intent

This branch explores HPX-based orchestration for llama.cpp inference workloads.

The direction is: HPX serving-level orchestration around opaque llama.cpp work.

Focus on context ownership, request lifecycle, scheduling policy, cancellation, and later request-level concurrency / batching / pipelining.

Current active slice: HPX-on `llama-serving-bench` structural-correctness comparison against the already-green std baseline.

This is not a performance slice yet.

Current HPX-on shape:

```text
model: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt: "Hello, my name is"
n_contexts=1
n_concurrent=1
n_threads=4
expected HPX runtime carriers=1
canonical generated-token hash=0x833045f1e2ebf49f
```

The immediate goal is correctness and lifecycle:

```text
--backend hpx produces the canonical hash
HPX starts once
engine_hpx becomes ready
each request acquires/releases ctx=0 exactly once
HPX stops once
```

The old HPX branch is evidence and reference only. Do not continue the old design by default.

## Reading

Read first:

- `local/handoff.md`
- `local/baselines/comparison_hpx_vs_std/FACTS.md`

Use them as current session context. Do not treat them as permanent historical truth.

For the current HPX-on comparison slice, also read if needed:

- `local/baselines/comparison_hpx_vs_std/_run_hpx.py`
- `local/baselines/comparison_aligned/bench/`
- `docs/hpx/serving_bench_acceptance.md`

Do not read broad project history by default.

## Reference docs

Use these when relevant:

- `docs/hpx/serving_bench_acceptance.md`
- `docs/hpx/executor_contract.md`
- `docs/hpx/benchmark_protocol.md`
- `docs/hpx/cpu_repack_baseline_notes.md`
- `docs/hpx/prefill_branch_summary.md`
- `docs/hpx/selective_graph_map_reference.txt`

These are evidence and review context. They are not the current implementation plan.

## Source of truth

Use current code, current git state, and the current handoff as the source of truth.

Do not use provenance as design truth.

Do not treat old docs as implementation instructions unless the handoff explicitly points to them.

## Architecture principles

Prefer serving-level orchestration around opaque llama.cpp execution:

```text
one shared llama_model
prewarmed llama_context objects
exclusive llama_context ownership per active request
normal llama_decode inside each leased context
backend-owned work remains opaque
```

The validated baseline should be simple and understandable before HPX-specific behavior is added.

HPX designs should be HPX-native where appropriate:

```text
async request state machines
future-based resource pools
RAII ownership for leased resources
continuations for readiness
clear lifecycle boundaries
```

Avoid designs that merely rename a std::thread worker pool with HPX types unless that is explicitly the experiment being run.

## HPX runtime invariants

- HPX runtime startup must be process-wide and one-shot.
- The program entry/lifecycle layer owns HPX startup and shutdown.
- Individual engines must not start or stop HPX.
- Non-HPX backends must not start HPX.
- Fail closed when HPX is unavailable or not built.
- Keep the HPX runtime API narrow.
- Keep runtime diagnostics and lifecycle traces env-gated unless they are errors.

## Serving correctness invariants

- Preserve exclusive access to each `llama_context`.
- Do not run concurrent `llama_decode` calls on the same context.
- Reset per-request context state deliberately.
- Use the same tokenization path for fit checks and real request execution.
- Keep timing semantics explicit; queue wait should not be accidentally excluded.
- Treat generated-token hashes or equivalent fingerprints as correctness signals, not performance metrics.
- Do not make performance claims before correctness and lifecycle gates pass.

For generated-token hashes:

```text
0x833045f1e2ebf49f
```

is the canonical hash only for the current TinyLlama / `"Hello, my name is"` / 16-token greedy shape. It is not a model checksum or universal correctness proof.

## Preferred working style

- Define the problem first.
- Define contracts before implementation.
- Define acceptance gates before `.cpp` work.
- Implement in small slices.
- After each slice, run only the relevant checks.
- Stop on the first correctness failure and fix the narrowest cause.
- Keep build, test, and result claims separate.
- Do not interpret smoke tests as performance evidence.

## Layout

Builds stay outside this repo.

Current account paths:

```text
repo:        /Users/Ashk/Desktop/HPX/llama-hpx
models:      /Users/Ashk/Desktop/HPX/models
HPX source:  /Users/Ashk/Desktop/HPX/hpx-master
HPX build:   /Users/Ashk/Desktop/HPX/hpx-master-build
HPX install: /Users/Ashk/Desktop/HPX/hpx-install
```

Build directories:

```text
HPX-OFF baseline: /Users/Ashk/Desktop/HPX/builds/llama-base
HPX-ON build:     /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on
```

Current HPX-on binary:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

## Saving results

Do not write outputs to `/tmp`.

For committed or shareable benchmark evidence, use:

```text
hpx-bench/results/<date>-<slug>/
```

For local-only notes, use:

```text
local/
```

Correctness/lifecycle checks do not need benchmark result directories unless explicitly requested.
