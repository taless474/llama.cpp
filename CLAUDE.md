# Claude Guidance for HPX / llama.cpp Serving Work

## Intent

This branch explores HPX-based orchestration for llama.cpp inference workloads with real request-level concurrency.

The direction is: HPX serving-level orchestration around opaque llama.cpp work.

Focus on request-level concurrency, context ownership, scheduling policy, cancellation, and later batching / pipelining.

The old HPX branch is evidence and reference only. Do not continue the old design by default.

## Reading

Read first:

- `local/handoff.md`

Use it as current session context. Do not treat it as permanent historical truth.

Do not read broad project history by default.

## Reference docs

Use these when relevant:

- `docs/hpx/executor_contract.md`
- `docs/hpx/benchmark_protocol.md`
- `docs/hpx/cpu_repack_baseline_notes.md`
- `docs/hpx/prefill_branch_summary.md`
- `docs/hpx/selective_graph_map_reference.txt`
- `docs/hpx/gates.md`

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

Example build paths:

```text
/Users/unick/Desktop/hpx/builds/llama-base
/Users/unick/Desktop/hpx/builds/llama-hpx
/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on
```

External HPX paths:

```text
/Users/unick/Desktop/hpx/hpx-master
/Users/unick/Desktop/hpx/hpx-master-build
/Users/unick/Desktop/hpx/hpx-install
```

Models stay outside the repo:

```text
/Users/unick/Desktop/hpx/models
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
