# Claude Guidance for HPX / ggml Integration

## Readings

Read @local/HANDOFF.md first. Use it only as current handoff context, not as historical truth.

Do not read project docs by default.

If the task needs project context, skim:

- `README_HPX.md`

Only when needed, read:

- `docs/HPX_EXECUTOR_CONTRACT.md` for durable execution rules and review criteria
- latest relevant `local/results/*` or `hpx-bench/results/*`, only if the user names or asks about a result

When explicitly asked, read:

- `docs/HPX_NATIVE_REVIEWER.md`
- `docs/HPX_DECODE_BENCH_PROTOCOL.md`

Do not use old provenance as design truth. Use current code and current results first.

Do not read these historical/helper docs unless the user explicitly asks for them:

- `docs/HPX_LOWER_OP.md`
- `docs/HPX_BUILD.md`
- `docs/HPX_PROVENANCE.md`

## Active area

The main active area is HPX selective execution: lowering, region execution, packet/runtime experiments, and `graph_compute` integration.

Important files:

- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp`
- `ggml/src/ggml-hpx/ggml-hpx-lower.cpp`
- `ggml/src/ggml-hpx/ggml-hpx-region-exec.cpp`
- `src/llama-context.cpp`
- `tests/hpx/**`
- `hpx-bench/**`

Older coarse-path files such as `cache.cpp`, `plan.cpp`, and `adapter.cpp` may be irrelevant to the selective hot path unless the current change explicitly touches that path.

## Hard invariants

- HPX runtime startup must remain process-wide and one-shot.
- Selective execution must remain CPU-only and single-split guarded.
- Fail closed to normal ggml scheduler execution.
- Preserve ggml graph order, backend assignment, tensor lifetimes, and scratch ownership.
- Do not compute Metal/CUDA/offloaded nodes on CPU by mistake.
- BLAS remains opaque.
- Do not use historical provenance as design truth.

## Backend-layout rule

Do not assume `tensor->type` fully describes the physical layout.

Always check the actual backend path and metadata before lowering quantized tensors. Repacked tensors may have:

```cpp
tensor->extra != nullptr
```

A lowering branch must be explicit about which physical layout/trait it supports. Unsupported layouts must fall back.

## Preferred direction

- Baseline first: inspect the real ggml/backend path before lowering.
- Use physical backend paths, not only logical ggml op types.
- Reduce dispatch fragmentation.
- Prefer packets, lowered runs, or other larger execution units over per-node HPX entry.
- Keep diagnostics env-gated and cold when disabled.
- Do not add more kernels before understanding the selective graph structure, unless explicitly requested.

## Saving results

Never write outputs to `/tmp` or any temporary directory. Write run artifacts directly into the repo.

Use:

- `hpx-bench/results/<date>-<slug>/` for benchmark outputs, logs, and captures
- `local/results/` for local notes, summaries, environment notes, and non-git artifacts

## Decode benchmark protocol

For PACKET=0 vs PACKET=1, packet-vs-baseline, or HPX decode-speed A/B benchmarks, read:

- `docs/HPX_DECODE_BENCH_PROTOCOL.md`

Do not run or interpret decode A/B results unless that protocol is followed.
