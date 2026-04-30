# Claude Guidance for HPX / llama.cpp Run-Level Work

## Intent

This branch starts fresh from upstream `llama.cpp` and explores an HPX-based orchestration for inference workloads where there is actual concurrency:
multi-request serving, multiple decode streams, cancellation, scheduling policy, and later possible batching/pipelining.


The old HPX branch is evidence and reference only. Do not continue the old selective executor design by default.

Current priority order:

1. Keep the repo clean and reproducible.
2. Keep HPX installed outside this repo.
3. Preserve old-branch evidence as reference docs.
4. Do not implement execution before the design is reviewed.

## First reading

Read this first:

- `local/handoff.md`

Use it only as current session context. Do not treat it as historical truth.

Do not read broad project history by default.

## Reference docs

Use these when relevant:

- `docs/hpx/executor_contract.md`  
  Durable execution rules and review criteria.

- `docs/hpx/benchmark_protocol.md`  
  Required protocol before interpreting benchmark comparisons.

- `docs/hpx/cpu_repack_baseline_notes.md`  
  Baseline notes for CPU_REPACK, q4_K/q6_K physical layout, and real kernel paths.

- `docs/hpx/prefill_branch_summary.md`  
  Summary of what the previous HPX branch proved and why it was closed.

- `docs/hpx/selective_graph_map_reference.txt`  
  Raw old-branch graph-map reference.

Do not treat old reference docs as implementation instructions. They are evidence.


## Hard invariants

- HPX runtime startup must be process-wide and one-shot.
- Fail closed to normal llama.cpp / ggml execution.
- Preserve ggml graph order.
- Preserve backend assignment.
- Preserve tensor lifetimes and scratch ownership.
- Do not compute Metal/CUDA/offloaded nodes on CPU by mistake.
- Unified memory is not proof of CPU ownership.
- BLAS and delegated backend work remain opaque.
- Do not use HPX per tiny decode node.
- Do not use `hpx::async(...).get()` as a repeated per-node bridge.
- Do not use historical provenance as design truth.

## Backend-layout rule

Do not assume `tensor->type` fully describes the physical layout.

For quantized tensors, inspect the actual backend path and metadata. Repacked tensors may have:

```cpp
tensor->extra != nullptr
```

A lowering or classification rule must be explicit about the physical layout/trait it supports. Unsupported layouts must fall back.

## Preferred direction

- Build and baseline first.
- Inspect the real ggml/backend path before proposing lowering.
- Use physical backend paths, not only logical ggml op types.
- Reduce dispatch fragmentation.
- Prefer run-level planning over per-node HPX entry.
- Treat packets as one possible run kind, not the main architecture.
- Keep backend-owned work opaque.
- Keep diagnostics env-gated and cold when disabled.
- Do not add more kernels before proving the path engages and has a plausible route to beating the baseline.


## Layout

Builds should stay outside this repo. External build path examples:

```text
/Users/unick/Desktop/hpx/builds/llama-base
/Users/unick/Desktop/hpx/builds/llama-hpx
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

For committed or shareable benchmark evidence, use a clear repo path such as:

```text
hpx-bench/results/<date>-<slug>/
```

For local-only notes, use:

```text
local/
```

