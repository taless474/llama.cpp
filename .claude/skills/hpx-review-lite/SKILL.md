---
name: hpx-review-lite
description: Quick manual review of llama-hpx changes for the highest-risk HPX/ggml integration mistakes.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob
---

# HPX Review Lite

Use this skill only when the user explicitly asks for a quick HPX review, hpx-review-lite, or a lightweight pre-commit check.

You are doing a fast read-only review. Do not edit files. Do not run builds, tests, or benchmarks.

## Scope

Review only files the user names, changed files, or HPX-relevant files discovered from the current task.

Priority paths:

- `ggml/src/ggml-hpx/**`
- `ggml/src/ggml-hpx-*`
- HPX-related parts of `src/llama-context.cpp`
- `tests/hpx/**`
- `hpx-bench/**`

Skip unrelated llama.cpp internals unless they affect HPX graph execution, backend selection, threadpool binding, or tensor lifetime.

## Check only the top risks

Look for these issues first:

1. Per-node native-to-HPX bridge

   Suspicious shape:

   ```cpp
   hpx::async(...).get()
   ```

   inside graph-node, lowered-node, fallback-node, decode-token, selective, or region loops.

2. One-node fallback slicing

   Suspicious shape:

   ```cpp
   ggml_graph_view(gf, i, i + 1)
   ggml_backend_graph_compute(...)
   ```

   inside a hot selective loop.

   Prefer coalescing adjacent safe fallback nodes into one contiguous `[start, end)` view.

3. Broken CPU-only / single-split guard

   Selective execution must not run on mixed-backend graphs or offloaded Metal/CUDA nodes.

   It must fail closed to normal ggml scheduler execution.

4. HPX runtime lifecycle in a hot path

   Flag `hpx::init`, `hpx::start`, `hpx::local::init`, `hpx::finalize`, or runtime construction/destruction inside per-token, per-graph, or per-dispatch code.

5. Async lifetime bug

   If code became async/dataflow-based, check for unsafe captures:

   - stack-local lowering objects captured by reference
   - graph views/descriptors that die before futures complete
   - packet frames reused before completion
   - scratch/reduction buffers shared by independent in-flight work
   - loop variables captured by reference across async boundaries

## False positives to avoid

Do not flag:

- a single terminal `.get()` or `.wait()` after one batched HPX dispatch
- `std::call_once` for process-wide HPX startup
- `std::atomic` counters or cancellation tokens
- tests and benchmarks unless the user asked to review benchmark methodology
- cold diagnostics guarded by env vars
- raw sequential loops inside math kernels
- metadata scans over graph nodes

## Output format

Keep the answer short.

```md
## HPX review lite

Result: PASS | WARN | FAIL
Scope reviewed: <files/patterns>
Hot-path risk: low | medium | high

### Findings

1. [HIGH|MEDIUM|LOW] <title>
   - File: `path:line`
   - Problem: <one or two sentences>
   - Suggested fix: <small local fix>

## Summary

- Total findings:
- Highest-priority fix:
- Non-issues intentionally ignored:
```

If there are no findings, say so directly and mention what was checked.
