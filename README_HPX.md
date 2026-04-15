# README_HPX.md — HPX CPU Execution Redesign for llama.cpp

## Overview

This project redesigns the CPU execution model of `llama.cpp` using HPX.

Earlier phases used HPX primarily as an orchestration layer above ggml.
That approach established correctness and clarified the workload split between decode and prefill, but it also exposed a structural limit for intra-region CPU parallelism.

The current direction goes deeper:
- HPX inside the `ggml-cpu` execution layer
- work-size-based routing between pthread and HPX substrates
- an experimental HPX-native fine-grained CPU work DAG

The goal is not to rewrite ggml kernels or replace ggml itself. The goal is to
improve how CPU work is structured, dispatched, and executed.

---

## Key idea

The problem is not “pthread vs HPX.”

The real question is:

> What is the right execution contract for inference?

This project evolved through three layers:
1. HPX above ggml as orchestration
2. HPX as a CPU executor substrate inside `ggml-cpu`
3. HPX as a direct executor of explicit fine-grained CPU work regions

The strongest current direction is the third one.

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

### Current execution model in this project

```text
small work (decode-like)
  → pthread substrate

large work (prefill-like)
  → HPX substrate
```

Routing is based on `cplan->work_size`.

### Experimental next layer

```text
coarse graph region
  → fine CPU region DAG
  → run_range(...)
  → HPX block_fork_join_executor
```

This path executes real CPU work units directly instead of routing through the
old ggml worker loop.

---

## What is implemented

### 1. HPX executor substrate inside `ggml-cpu`

- executor/job ownership split
- explicit executor seam:
  - `init`
  - `run_job`
  - `destroy`
- HPX-backed threadpool implementation
- bulk-region execution
- later tuning toward lower-overhead executor choices

### 2. Work-size-based routing

Key discovery:

- graph topology does **not** distinguish decode from prefill
- `cplan->work_size` does

Policy:

```cpp
if (work_size < threshold)
    → pthread
else
    → HPX
```

Observed result:
- decode returns to near baseline
- prefill remains near parity or slightly better

### 3. HPX-native fine-region DAG (experimental)

New direct execution contract:

```cpp
run_range(ctx, ith, nth, begin, end, resources)
```

Characteristics:
- direct kernel entry
- no `ggml_graph_compute_thread_run(...)`
- no graph re-entry
- explicit region DAG
- explicit resource ownership

New abstractions:
- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`
- `ggml_hpx_dep_edge`
- `ggml_hpx_region_resources`

---

## Performance highlights

### Work-size routing result

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

### Fine-region DAG result

#### Bench 1 — matmul dispatch

| shape   | nth | scheduler_exec (prev) | fork_join_exec (now) |
|---------|-----|------------------------|----------------------|
| decode  | 1   | ~990 µs                | 993 µs (≈ same)      |
| decode  | 2   | ~880 µs                | 894 µs (≈ same)      |
| decode  | 4   | ~660 µs                | 642 µs (slightly faster) |
| prefill | 1   | ~31 ms                 | 31.2 ms (≈ same)     |
| prefill | 2   | ~28 ms                 | 27.8 ms (≈ same)     |
| prefill | 4   | ~21 ms                 | 20.5 ms (slightly faster) |

#### Bench 2 — 3-region chain

| shape   | nth | pool      | fork_join_exec | speedup |
|---------|-----|-----------|----------------|---------|
| decode  | 4   | 1145 µs   | 643 µs         | 1.78×   |
| prefill | 4   | 23.4 ms   | 21.9 ms        | 1.07×   |

This is the first clear positive performance result from the HPX-native
fine-region design.

---

## Key insights

### 1. Executor replacement alone is not enough
Replacing pthread with HPX does not automatically improve inference.

### 2. Work granularity dominates performance
- large work amortizes HPX overhead well
- repeated short multithreaded dispatches do not

### 3. Graph shape is the wrong discriminator
Decode and prefill often use the same topology. `work_size` is the useful
signal.

### 4. The right abstraction is explicit work, not worker identity

Wrong direction:
```text
HPX → logical worker ids → ggml worker loop
```

Better direction:
```text
HPX → explicit work units → direct execution
```

---

## Current limitations

- fine-region DAG is not yet integrated into the full llama.cpp execution path
- only selected kernels have direct `run_range` implementations
- same-level region parallelism is still conservative in the first prototype
- reduction and scratch-heavy fine-region cases are not yet generalized

---

## Recommended reading order

1. `README.md` — current architecture and results
2. `docs/HPX_EXECUTOR_CONTRACT.md` — design rules and execution boundaries
3. `docs/HPX_PROVENANCE.md` — full chronological provenance

---

## One-line summary

This project moves llama.cpp CPU execution from a thread-centric model toward a structure-aware model where HPX executes explicit CPU work units directly, and only where that pays off.
