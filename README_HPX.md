# README_HPX.md — HPX CPU Execution Redesign for llama.cpp

## Overview

This project redesigns the CPU execution model of `llama.cpp` using HPX.

It began with HPX-based orchestration above ggml, then moved deeper into
`ggml-cpu` executor ownership, and now includes an HPX-native fine-region
execution path in `ggml/src/ggml-hpx/`.

The goal is not to replace ggml kernels or BLAS. The goal is to improve how
CPU work is:
- represented
- scheduled
- dispatched
- executed

The strongest current direction is no longer “HPX as orchestration.”
It is **HPX as the execution model for explicit CPU work regions**.

---

## Key idea

The problem is not “pthread vs HPX.”

The real question is:

> What is the right execution contract for inference?

This project evolved through three layers:

1. HPX above ggml as orchestration
2. HPX as a CPU executor substrate inside `ggml-cpu`
3. HPX as a direct executor of explicit fine-grained CPU work regions

The third layer is now the most important one.

---

## Architecture

### Existing llama.cpp CPU model

```text
graph (ggml)
  → scheduler
  → threadpool workers
  → ggml_graph_compute_thread_run(...)
```

This model ties execution to worker identity and a rigid barrier structure.

### Existing practical substrate policy in this project

```text
small work (decode-like)
  → pthread substrate

large work (prefill-like)
  → HPX substrate
```

Routing is based on `cplan->work_size`.

This remains the practical coarse-path policy.

### New fine-region execution layer

```text
coarse graph region
  → fine CPU region DAG
  → run_range(...)
  → HPX futures / dataflow
```

This path executes explicit CPU work units directly instead of routing through
the old ggml worker loop.

---

## What is implemented

### 1. HPX executor substrate inside `ggml-cpu`

- executor/job ownership split
- explicit executor seam:
  - `init`
  - `run_job`
  - `destroy`
- HPX-backed threadpool implementation
- work-size-based pthread / HPX substrate split

### 2. Fine-region DAG contract

Implemented under `GGML_HPX_REGION_DAG`:

- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`
- `ggml_hpx_dep_edge`
- `ggml_hpx_region_resources`
- `ggml_hpx_run_range_fn`

This defines a fine-grained CPU work DAG with:
- explicit work ranges
- explicit dependencies
- explicit resource bundle

### 3. Fine-region execution path

Implemented in `ggml-hpx-region-exec.*`:

- region-group validator
- direct F32 `mul_mat` kernel
- `ggml_hpx_run_single_region(...)`
- `ggml_hpx_run_region_group(...)`

Current properties:
- direct `run_range(...)` execution
- no `ggml_graph_compute_thread_run(...)`
- no graph re-entry
- dependency-driven inter-region scheduling with HPX futures / `dataflow`
- real same-level region overlap

### 4. Selective graph-level mixed execution

A narrow selective execution path now exists for graph-level experiments:

- supported lowered ops use the fine-region path
- unsupported ops fall back to the real CPU backend path

The first mixed execution target is:
- lowered `mul_mat`
- fallback for everything else

This is the current bridge toward real llama.cpp comparison.

---

## Current validated behavior

The following fine-region tests pass:

- `test_hpx_region_group_validate`
- `test_hpx_region_group_run`
- `test_hpx_region_group_parallel`
- `test_hpx_region_single_mul_mat`
- `test_hpx_region_mixed_mul_mat`

These prove:

- malformed region groups are rejected
- a small region DAG executes correctly
- same-level overlap is real
- direct single-region `mul_mat` works
- mixed lowered/fallback execution works
- fine-region execution does not re-enter:
  - `ggml_graph_compute`
  - `ggml_backend_graph_compute`
  - `ggml_graph_compute_thread_run`

---

## Performance highlights

### Coarse-path work-size routing result

| case  | t | base | hpx | delta |
|------|---:|-----:|----:|------:|
| tg128 | 2 | 87.2 | 83.1 | -5% |
| tg128 | 4 | 97.0 | 96.2 | -1% |
| pp32  | 2 | 170.7 | 162.4 | -5% |
| pp32  | 4 | 236.7 | 233.3 | -1% |
| pp512 | 2 | 155.5 | 156.7 | +1% |
| pp512 | 4 | 210.9 | 214.3 | +2% |

Interpretation:
- decode is protected from the HPX small-work penalty
- prefill stays competitive on the HPX path

### Fine-region DAG highlights

The fine-region path has already shown:
- direct `mul_mat` region execution works
- dependency-driven scheduling works
- same-level concurrency is real

At this stage, the fine-region story is primarily a correctness and
execution-model milestone, not yet a broad end-to-end performance claim inside
llama.cpp.

---

## Where to look

### Core implementation
- `ggml/src/ggml-hpx/`
  - HPX-specific lowering, region execution, selective execution, and runtime code
- `ggml/src/ggml-cpu/`
  - CPU executor seam and substrate work

### Tests
- `tests/hpx/`
  - fine-region DAG tests
  - selective mixed-execution tests
  - llama smoke tests

### Benchmarks
- `hpx-bench/`
  - standalone HPX microbenchmarks and region benchmarks

### Design / docs
- `README_HPX.md`
  - current HPX architecture and status
- `docs/HPX_EXECUTOR_CONTRACT.md`
  - strict design rules and execution boundaries
- `docs/HPX_PROVENANCE.md`
  - chronological project history and results

---

## Key insights

### 1. Executor replacement alone is not enough
Replacing pthread with HPX does not automatically improve inference.

### 2. Work granularity dominates performance
- large work amortizes HPX overhead well
- repeated short multithreaded dispatches do not

### 3. Graph shape is the wrong discriminator
Decode and prefill often use the same topology. `work_size` is the useful signal.

### 4. The right abstraction is explicit work, not worker identity

Wrong direction:
```text
HPX → logical worker ids → ggml worker loop
```

Better direction:
```text
HPX → explicit work units → dependency futures → direct execution
```

### 5. Fine-region execution is now the most important direction
The project’s most promising path is no longer “HPX everywhere.”
It is:
- pthread where small work wins
- HPX where large work wins
- explicit fine-region execution where the old worker-loop model is the wrong abstraction

---

## Current limitations

- only selected direct kernel paths exist so far
- `mul_mat` is the first real direct fine-region kernel path
- resource-heavy region cases (`reduction_buffer`, `lane_scratch`) still need stronger coverage
- selective graph-level lowering is still narrow
- full end-to-end llama.cpp comparison for the fine-region path is still in progress

---

## One-line summary

This project moves llama.cpp CPU execution from a thread-centric model toward a
structure-aware model where HPX executes explicit CPU work regions directly and
composes them through dependency futures.
