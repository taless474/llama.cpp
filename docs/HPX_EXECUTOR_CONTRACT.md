# HPX Executor Contract for ggml/llama.cpp
**Draft v0.2**

## Scope

This document defines the execution boundary for an HPX-based executor in a `llama.cpp` fork.

Goals:
- keep ggml graphs and tensors
- keep existing CPU kernels
- keep BLAS as a delegated backend
- support two execution modes: **decode** and **prefill**
- isolate scheduler coupling to one adapter translation unit
- make planning, caching, and execution testable independently

Non-goals:
- rewriting ggml kernels
- replacing ggml IR
- planning inside BLAS
- creating one HPX task per ggml op

---

## 1. Decode plan

A **decode plan** is a cached, reusable execution template for the low-latency single-token path.

It contains only structural execution metadata:
- mode = `decode`
- plan key
- coarse region list
- region order and dependencies
- chunking policy for CPU regions
- workspace requirement
- executor scratch layout
- optional instrumentation labels

A decode plan is optimized for:
- minimum dispatch overhead
- stable region structure
- minimal dynamic repartitioning

A decode plan must **not** depend on:
- ephemeral graph addresses
- live scheduler-owned objects
- prior allocation-owned tensor pointers

---

## 2. Prefill plan

A **prefill plan** is a cached, reusable execution template for prompt or batch throughput work.

It may depend on scheduler-derived split topology, but only through an immutable HPX-owned snapshot.

It contains:
- mode = `prefill`
- plan key
- region topology snapshot
- dependency edges
- per-region execution policy
- workspace requirement
- optional overlap policy

A prefill plan may be reused only while:
- backend placement assumptions remain compatible
- split topology remains compatible
- workspace/layout assumptions remain compatible
- planner policy version remains compatible

---

## 3. Execution unit

The execution unit is **not** a ggml op.

The execution unit is a **coarse region**, such as:
- a scheduler split
- a contiguous CPU-only region
- a delegated BLAS-supported region
- another planner-defined region with a single synchronization boundary

The planner creates regions.  
The executor consumes regions.  
The scheduler does not see HPX internal worker-level chunking.

---

## 4. BLAS handoff

A region is handed off to BLAS instead of being planned internally when:
- the region is executable by the existing BLAS backend
- no HPX-only fusion or kernel rewrite is required
- delegated execution is expected to be at least as good as decomposing it

BLAS regions are treated as opaque coarse units:
- HPX may schedule around them
- HPX may attach dependencies before and after them
- HPX does not plan inside them

---

## 5. Abort / cancel

Abort is **cooperative**, not preemptive.

At executor level, abort means:
- no new regions start after abort is observed
- a running CPU region stops at the next safe checkpoint
- in-flight delegated work such as BLAS is not forcibly preempted
- the executor returns aborted status after required cleanup

Abort checks must exist:
- before starting each region
- at safe checkpoints inside long CPU regions
- before launching dependent follow-on regions

---

## 6. Reusable state across runs

Safe to reuse:
- immutable plan metadata
- mode choice
- region topology
- dependency graph
- chunking policy
- workspace size requirement
- executor-owned scratch buffers sized for max requirement
- long-lived HPX runtime objects and teams
- instrumentation counters and labels

Not safe to reuse:
- raw `ggml_cgraph *` pointers
- raw tensor pointers
- backend buffer addresses from a previous allocation pass
- pointers into scheduler-owned temporary storage
- per-run input/output bindings
- transient split objects tied to a specific scheduler pass

---

## 7. Plan cache key

A plan cache key must include at least:
- mode: `decode` or `prefill`
- graph-shape signature
- backend assignment signature
- workspace/layout signature
- planner policy version

A plan cache key must **not** include ephemeral addresses.

---

## 8. Scheduler dependency rule

Scheduler coupling is isolated mechanically.

### Allowed scheduler dependency
`ggml-backend.h` may be included only by:
- `ggml-hpx-adapter.cpp`

### Forbidden scheduler dependency
`ggml-backend.h` must not be included by:
- `ggml-hpx-plan.cpp`
- `ggml-hpx-exec.cpp`
- cache, abort, topology, or region headers

This rule makes violations visible and testable.

---

## 9. Topology acquisition rule

The adapter may use two strategies.

### Decode topology
Default strategy: **independent graph walk**

Use this when the decode path is CPU-only and simple enough that HPX-owned topology is authoritative.

Reason:
- lowest overhead
- no scheduler-state dependency on the fast path

### Prefill topology
Default strategy: **scheduler-driven split translation**

Use this when split structure or mixed backend placement matters.

Flow:
1. adapter asks scheduler to derive split structure
2. adapter translates scheduler result into `ggml_hpx_region_topology`
3. plan builder consumes only that immutable snapshot

Fallback rule:
- if decode ever stops being simple or becomes mixed-backend, it may also use scheduler-driven topology through the adapter

---

## 10. Repo layout

```text
llama.cpp/
├── docs/
│   ├── HPX_EXECUTOR_CONTRACT.md
│   ├── HPX_REPO_LAYOUT.md
│   ├── HPX_MODE_POLICY.md
│   ├── HPX_TEST_PLAN.md
│   ├── HPX_BENCHMARK_PLAN.md
│   └── HPX_IMPL_NOTES.md
│
├── ggml/
│   ├── include/
│   │   └── ggml-hpx.h
│   │
│   └── src/
│       ├── CMakeLists.txt
│       │
│       ├── ggml-hpx/
│       │   ├── CMakeLists.txt
│       │   ├── ggml-hpx-region.h
│       │   ├── ggml-hpx-abort.h
│       │   ├── ggml-hpx-topo.h
│       │   ├── ggml-hpx-adapter.h
│       │   ├── ggml-hpx-adapter.cpp
│       │   ├── ggml-hpx-plan.h
│       │   ├── ggml-hpx-plan.cpp
│       │   ├── ggml-hpx-cache.h
│       │   ├── ggml-hpx-cache.cpp
│       │   ├── ggml-hpx-runtime.h
│       │   ├── ggml-hpx-runtime.cpp
│       │   ├── ggml-hpx-exec.h
│       │   ├── ggml-hpx-exec.cpp
│       │   ├── ggml-hpx-instrument.h
│       │   └── ggml-hpx-instrument.cpp
│       │
│       ├── ggml-cpu/
│       └── ggml-blas/
│
├── tests/
│   ├── hpx/
│   │   ├── test_hpx_contract.cpp
│   │   ├── test_hpx_decode_plan.cpp
│   │   ├── test_hpx_prefill_plan.cpp
│   │   ├── test_hpx_cache.cpp
│   │   ├── test_hpx_abort.cpp
│   │   ├── test_hpx_adapter.cpp
│   │   ├── test_hpx_exec.cpp
│   │   └── test_hpx_stress.cpp
│   │
│   └── ggml/
│
├── hpx-bench/
│   ├── CMakeLists.txt
│   ├── bench_dispatch_overhead.cpp
│   ├── bench_decode_latency.cpp
│   ├── bench_prefill_throughput.cpp
│   ├── bench_split_scaling.cpp
│   ├── bench_blas_interop.cpp
│   └── results/
│
└── src/
    └── llama-context.cpp
```

## 11. File responsibilities

### docs/

#### `HPX_EXECUTOR_CONTRACT.md`
Source of truth for:
- decode vs prefill semantics
- abort semantics
- plan reuse rules
- scheduler dependency rules

#### `HPX_REPO_LAYOUT.md`
Explains:
- why files are placed where they are
- allowed dependency directions
- what must remain isolated

#### `HPX_MODE_POLICY.md`
Defines:
- how mode is selected
- when decode uses graph-walk topology
- when prefill uses scheduler-derived topology
- fallback rules

#### `HPX_TEST_PLAN.md`
Lists:
- contract tests
- stress tests
- regression tests
- performance guardrails

#### `HPX_BENCHMARK_PLAN.md`
Defines:
- benchmark matrix
- input shapes
- decode vs prefill measurements
- BLAS and non-BLAS runs
- result reporting format

#### `HPX_IMPL_NOTES.md`
Holds:
- caveats
- unresolved questions
- follow-up refactors
- performance observations

---

### `ggml/include/ggml-hpx.h`
Thin public C-facing HPX backend header.

Responsibilities:
- init / destroy entry points
- capability query
- backend creation hooks
- no internal planner or runtime types

---

### `ggml/src/ggml-hpx/`

#### `ggml-hpx-region.h`
Defines the coarse execution unit.

Contains:
- region type enum
- region identity
- node span or equivalent structural bounds
- sync boundary marker
- region-level metadata only

#### `ggml-hpx-abort.h`
Defines the cooperative abort token.

Contains:
- atomic abort state
- inline check helpers
- no runtime ownership

#### `ggml-hpx-topo.h`
Defines immutable HPX-owned topology snapshot structs.

Contains:
- `ggml_hpx_region_topology`
- region list
- dependency edges
- topology metadata

Must never store:
- live scheduler-owned pointers
- allocation-pass-owned transient state

#### `ggml-hpx-adapter.h`
Adapter interface.

Responsibilities:
- accept graph and optional scheduler-facing inputs
- produce immutable topology snapshots
- expose separate decode/prefill topology entry points if needed

#### `ggml-hpx-adapter.cpp`
Only translation unit allowed to include `ggml-backend.h`.

Responsibilities:
- scheduler-driven split acquisition for prefill
- translation from scheduler result to HPX topology snapshot
- optional independent graph walk for decode
- no plan caching
- no execution

#### `ggml-hpx-plan.h`
Planner-facing types.

Contains:
- `plan_key`
- `decode_plan`
- `prefill_plan`
- plan validity contract

#### `ggml-hpx-plan.cpp`
Planner implementation.

Responsibilities:
- `build_decode_plan()`
- `build_prefill_plan()`
- `plan_valid()`

Rules:
- consumes topology snapshots
- does not read scheduler state directly
- must not include `ggml-backend.h`

#### `ggml-hpx-cache.h`
Plan cache interface.

Contains:
- lookup
- insert
- invalidate
- cache stats

#### `ggml-hpx-cache.cpp`
Cache implementation.

Responsibilities:
- keyed plan storage
- lifetime and invalidation rules
- no execution logic

#### `ggml-hpx-runtime.h`
HPX runtime ownership interface.

Contains:
- runtime creation / destroy
- team or pool ownership types
- scratch ownership interface

#### `ggml-hpx-runtime.cpp`
Runtime implementation.

Responsibilities:
- HPX runtime lifecycle
- worker team or pool ownership
- scratch buffer ownership
- pool sizing/config hooks

#### `ggml-hpx-exec.h`
Executor interface.

Contains:
- create
- run
- destroy
- status return contract

#### `ggml-hpx-exec.cpp`
Executor implementation.

Responsibilities:
- adapter → plan → cache orchestration
- mode-specific dispatch
- region execution
- abort checks
- delegated BLAS launch coordination

Must not:
- read scheduler state directly
- include `ggml-backend.h`

#### `ggml-hpx-instrument.h`
Instrumentation interface.

Contains:
- counters
- timing labels
- trace hooks
- benchmark-facing metrics

#### `ggml-hpx-instrument.cpp`
Instrumentation implementation.

Responsibilities:
- event counters
- timing collection
- trace helpers
- no planning policy

---

## 12. Dependency edges that matter

### Allowed
- `ggml-hpx-adapter.cpp`
  - includes `ggml-backend.h`
  - includes `ggml-hpx-topo.h`
  - includes `ggml-hpx-region.h`

- `ggml-hpx-plan.cpp`
  - includes `ggml-hpx-topo.h`
  - includes `ggml-hpx-region.h`
  - may include cache-free planner helpers
  - must not include scheduler headers

- `ggml-hpx-exec.cpp`
  - includes `ggml-hpx-adapter.h`
  - includes `ggml-hpx-plan.h`
  - includes `ggml-hpx-cache.h`
  - includes `ggml-hpx-abort.h`
  - includes `ggml-hpx-runtime.h`
  - includes `ggml-hpx-instrument.h`

### Forbidden
- `ggml-backend.h` outside `ggml-hpx-adapter.cpp`
- scheduler-owned objects escaping into plan/cache/executor structures
- cache keys containing ephemeral addresses

---

## 13. Testing implications

The tests should align to the contract.

### Contract tests
- decode plan does not depend on ephemeral pointers
- prefill plan reuses only when topology is compatible
- abort stops new regions from starting
- BLAS regions remain opaque coarse units

### Adapter tests
- scheduler-driven topology translation is stable
- decode graph walk produces valid topology
- adapter outputs immutable HPX-owned snapshots

### Cache tests
- key excludes ephemeral addresses
- invalidation occurs on policy/version/layout changes
- reuse works across repeated compatible runs

### Executor tests
- decode path stays low-overhead
- prefill path respects dependencies
- abort is cooperative
- BLAS handoff preserves ordering

---

## 14. Short summary

- decode uses a cached low-overhead plan
- prefill uses a cached topology-aware plan
- regions are coarse execution units
- BLAS is delegated, not internally planned
- abort is cooperative
- reusable state is structural only
- scheduler coupling is isolated to one adapter `.cpp`