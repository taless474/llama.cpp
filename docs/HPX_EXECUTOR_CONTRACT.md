# HPX Executor Contract for ggml/llama.cpp
**Tightened rulebook version**

## Scope

This document defines the current execution contract for the HPX-related CPU
execution paths in this `llama.cpp` fork.

This file is a **rulebook**, not a provenance document.

Use it for:
- execution invariants
- dependency boundaries
- allowed and forbidden design moves
- current substrate-selection rules

Do **not** use it for:
- chronological history
- superseded HPX designs
- benchmark storytelling

---

## 1. Execution model (current)

The system supports **two execution substrates**:

- **pthread substrate**
  - used for small work
  - optimized for low overhead and stable latency

- **HPX substrate**
  - used for large work
  - optimized for throughput and structured CPU parallelism

### Selection rule

Execution is selected **per dispatch** using:

```cpp
cplan->work_size
```

Policy:

```text
small work   → pthread
large work   → HPX
```

This rule is fundamental and must not be bypassed casually.

---

## 2. Decode vs prefill (operational meaning)

| mode    | meaning                         | preferred substrate |
|---------|----------------------------------|---------------------|
| decode  | small, latency-sensitive work    | pthread             |
| prefill | large, throughput-oriented work  | HPX                 |

Important:
- graph topology does **not** reliably distinguish decode from prefill
- `work_size` does

---

## 3. Execution units

### 3.1 Coarse regions (current production-oriented path)

The primary execution unit is a **coarse region**, such as:
- a scheduler split
- a contiguous CPU-only graph region
- a delegated BLAS-supported region
- another planner-defined unit with one synchronization boundary

The planner creates coarse regions.  
The executor consumes coarse regions.

### 3.2 Fine CPU regions (HPX-native path)

An optional finer execution unit may be used inside CPU regions.

Fine regions represent:
- explicit CPU work ranges
- explicit dependency edges
- explicit resource ownership

They are expressed by:
- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`
- `ggml_hpx_run_range_fn`

Fine-region execution is the preferred direction for HPX-native CPU work.

---

## 4. Fine-region DAG rule

The fine-region model is the HPX-native direction.

A fine region is executed through:

```cpp
run_range(ctx, ith, nth, begin, end, resources)
```

Properties:
- explicit work range
- no graph re-entry
- no dependence on ggml worker identity
- no direct use of ggml’s barrier model

A region group defines:
- region array
- dependency edges
- per-dispatch resources

The preferred executor for this layer is:

- `hpx::execution::experimental::block_fork_join_executor`

Reason:
- lower overhead on short fork/join CPU work
- better fit than the earlier scheduler-queue-based path

---

## 5. BLAS rule

BLAS remains:
- opaque
- coarse
- delegated

HPX may:
- schedule before BLAS regions
- schedule after BLAS regions
- attach dependencies around BLAS regions

HPX must not:
- decompose BLAS internally
- plan inside BLAS
- assume control of BLAS worker behavior

---

## 6. Abort rule

Abort is **cooperative**, not preemptive.

Abort means:
- no new region starts after abort is observed
- running CPU work stops only at safe checkpoints
- in-flight BLAS or delegated backend work is not forcibly interrupted

Abort checks must exist:
- before starting each region
- at safe checkpoints inside long CPU work
- before launching dependent follow-on work

---

## 7. Plan and cache invariants

Plans are:
- structural only
- reusable across compatible runs

Plans may include:
- mode
- structural topology
- dependency graph
- workspace requirement
- planner policy version
- execution policy metadata

Plans must not include:
- ephemeral addresses
- raw tensor pointers
- live backend instances
- scheduler-owned objects
- allocation-pass-owned transient state

---

## 8. Scheduler isolation rule (critical)

Scheduler coupling is isolated mechanically.

### Allowed
Only this translation unit may include scheduler-facing backend details:

```text
ggml-hpx-adapter.cpp
```

### Forbidden
Scheduler/backend internals must not appear in:
- `ggml-hpx-plan.cpp`
- `ggml-hpx-exec.cpp`
- cache headers
- abort headers
- topology headers
- fine-region headers
- fine-region executor code

This rule must not be violated.

---

## 9. Topology acquisition rule

The adapter may use two strategies.

### Decode
Default strategy:
- independent graph walk

Reason:
- lower overhead
- avoids scheduler-state dependency on the fast path

### Prefill
Default strategy:
- scheduler-driven split translation

Flow:
1. adapter asks scheduler for split structure
2. adapter translates it to immutable HPX-owned topology
3. planner consumes only that immutable snapshot

Fallback:
- if decode stops being simple or becomes mixed-backend, decode may also use scheduler-driven topology through the adapter

---

## 10. Execution substrate rules

### 10.1 Substrate ownership

Each threadpool owns exactly one substrate:

- pthread threadpool → pthread executor
- HPX threadpool → HPX executor

There is no mixed substrate inside one threadpool.

### 10.2 No hybrid execution inside one pool

This is forbidden:

```text
one threadpool object that actively contains and runs both:
- pthread workers
- HPX workers
```

Reason:
- causes contention
- breaks decode behavior
- obscures ownership

### 10.3 Routing must happen before execution

Correct:

```text
decide substrate → run_job(...)
```

Incorrect:

```text
run_job(...)
  → branch internally between unrelated substrates
```

Substrate selection must happen before the execution call.

### 10.4 Small-work fallback constraint

If execution uses fewer than the published thread count, then the active
participant count must be updated consistently.

In particular, if only one worker will run, then barrier participation must
also reflect one worker.

Otherwise:
- ggml barrier logic will deadlock

### 10.5 HPX must not own decode-sized work accidentally

HPX must not be used for decode-sized work by mistake.

Decode-like work and very small prefill must stay off the HPX substrate unless
a new design proves that choice beneficial.

---

## 11. Repo structure rules

### Root entry points

- `README_HPX.md`
  - current architecture and current results

- `docs/HPX_EXECUTOR_CONTRACT.md`
  - strict design and dependency rules

- `docs/HPX_PROVENANCE.md`
  - full chronological history
  - not a rulebook

### Main code areas

- `ggml/src/ggml-hpx/`
  - HPX-specific execution code

- `ggml/src/ggml-cpu/`
  - CPU executor seam and substrate ownership

- `hpx-bench/`
  - standalone benchmark executables

---

## 12. File responsibilities

### `ggml-hpx-region.h`
Defines coarse execution regions.

### `ggml-hpx-region-dag.h`
Defines fine CPU regions, dependency edges, resources, and `run_range`.

### `ggml-hpx-region-exec.h/.cpp`
Implements fine-region validation and direct fine-region execution helpers.

### `ggml-hpx-adapter.cpp`
Only allowed scheduler/backend translation unit.

### `ggml-hpx-plan.h/.cpp`
Structural plan creation and plan validity logic.

### `ggml-hpx-cache.h/.cpp`
Structural plan cache only.

### `ggml-hpx-runtime.h/.cpp`
HPX runtime ownership and substrate-level runtime support.

### `ggml-hpx-exec.h/.cpp`
Coarse-region orchestration path only.
Must not read scheduler internals directly.

---

## 13. Dependency rules

### Allowed
- adapter → scheduler/backend translation
- planner → immutable topology
- executor → adapter + plan + cache + runtime
- fine-region executor → fine-region structs + HPX executor headers

### Forbidden
- scheduler details outside adapter
- backend instances inside plans
- runtime ownership inside planner
- cache keys using ephemeral addresses
- fine-region code calling back into full graph execution

---

## 14. Fine-region execution rules

When using `run_range(...)`, code must:

- operate on explicit ranges
- be reentrant for distinct range tuples
- use only explicit resources passed in
- avoid graph re-entry

It must not:
- call `ggml_graph_compute_thread_run(...)`
- call `ggml_graph_compute(...)`
- call `ggml_backend_graph_compute(...)`
- depend on implicit worker identity semantics
- assume persistent worker participation or a ggml barrier contract

---

## 15. What Claude or future code should not change

Do not:
- merge pthread and HPX into one active substrate in one pool
- remove work-size routing without replacement evidence
- leak scheduler dependency outside the adapter
- reintroduce decode-sized HPX ownership casually
- treat provenance as a design contract
- replace fine-region execution with graph re-entry

---

## 16. What may evolve

These are allowed to evolve:
- threshold tuning for work-size routing
- region partitioning policy
- new direct `run_range` kernels
- fine-region DAG scheduling policy
- HPX executor configuration
- resource ownership details for reductions and scratch

These must evolve without violating earlier rules.

---

## 17. Short summary

- use pthread for small work
- use HPX for large work
- keep scheduler coupling isolated to the adapter
- keep plans structural
- keep BLAS opaque
- move HPX toward explicit fine-grained CPU work execution instead of
  thread-centric execution
