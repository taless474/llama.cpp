# HPX Implementation Provenance
**For the llama.cpp HPX executor redesign**

## What this document is for

This file explains **why this HPX layer exists**, **what problem it is trying to solve**, and **why the code is organized the way it is**.

It is written for a reader who may know C/C++ and parallel programming in general, but may **not** already know:
- `llama.cpp`
- `ggml`
- HPX
- why “decode” and “prefill” are treated differently

---

## 1. Background: what llama.cpp is doing at a high level

Modern large language model inference is not one giant matrix multiply. It is a sequence of many tensor operations arranged as a **graph**.

In `llama.cpp`, that graph is built using **ggml**. ggml provides:
- tensor data structures
- operation nodes
- CPU kernels
- backend interfaces
- scheduler logic
- optional acceleration backends such as BLAS

So when we talk about “running llama.cpp,” we are really talking about:
1. building a graph of tensor operations
2. deciding which backend runs which parts
3. executing the graph efficiently

The key point for this project is:

**We are not trying to replace ggml.**  
We are trying to replace or redesign the **execution/orchestration layer** for CPU-side work while keeping:
- ggml graphs
- ggml tensors
- existing CPU kernels
- BLAS support

That design direction is already reflected in the headers:
- planning is separated from runtime execution
- BLAS is treated as delegated work, not something reimplemented inside HPX
- scheduler coupling is isolated to the adapter layer

---

## 2. Why HPX is being considered

HPX is a C++ runtime for asynchronous and parallel execution. It provides facilities for:
- task scheduling
- futures
- work distribution
- synchronization
- runtime-managed execution resources

A naive idea would be:

> “Replace the old threadpool with HPX and everything gets faster.” which we did in another branch :D

That is **not** the assumption here.

The experience from the earlier `hpx-threadpool-experiment` branch suggested a more careful conclusion:

- HPX can express the concurrency we need
- but a direct one-for-one replacement of an existing low-overhead threadpool is **not automatically faster**
- overhead matters a lot for inference, especially on the single-token path

That is why this redesign does **not** start from “make HPX look like pthreads.”  
It starts from:

> “What is the right execution contract if we know we have two different workloads: decode and prefill?”

---

## 3. Decode and prefill are different problems

This project treats model inference as having **two major execution modes**.

### Decode
Decode is the low-latency path used during token generation, often one token at a time.

Its priorities are:
- low overhead
- predictable dispatch
- minimal extra scheduling layers
- fast reuse of prior planning work

So decode wants a **small, stable, cached plan**.  
That is why the decode plan in `ggml-hpx-plan.h` is intentionally lean and stores:
- a plan key
- a vector of coarse regions
- chunking policy
- workspace size

### Prefill
Prefill is the prompt/batch path. It is usually larger and more throughput-oriented.

Its priorities are different:
- exploit more parallelism
- respect backend splits
- allow richer topology
- potentially overlap independent work when policy allows

So prefill carries a fuller topology snapshot and more region-level policy.  
That is why the prefill plan stores:
- a plan key
- an immutable topology snapshot
- per-region policy
- overlap policy
- workspace size

This decode/prefill split is one of the central outcomes of the redesign.

---

## 4. The execution unit is not a ggml op

A very tempting design would be:

> “Make one HPX task per ggml node.”

Instead, the execution unit is a **coarse region**.  
A region is one planner-defined unit with a single synchronization boundary. It can represent:
- a contiguous CPU-only subgraph
- a delegated BLAS region
- a scheduler-derived split boundary

This matters because fine-grained tasking often adds too much overhead, especially for decode. A region is large enough to make HPX scheduling worthwhile, while still giving the planner structure to work with.

The region definition was intentionally kept simple:
- `type`
- `node_begin`
- `node_end`
- `dep_mask`

That is enough to express coarse execution structure without coupling region objects to runtime-specific policy.

---

## 5. Scheduler coupling is isolated to the adapter

> **Only one translation unit should know about scheduler internals.**

That translation unit is `ggml-hpx-adapter.cpp`.

The adapter’s job is to convert external graph/scheduler state into **HPX-owned planning inputs**:
- immutable topology
- fully populated plan key

This is why:
- `ggml-hpx-plan.cpp` must not read scheduler state directly
- `ggml-hpx-exec.cpp` must not read scheduler state directly
- `ggml-backend.h` is allowed only in the adapter implementation path, not throughout the HPX layer

This decision has two big benefits:

### Benefit 1: cleaner planning
The plan builder consumes an immutable snapshot, not a live scheduler object.

That makes planning:
- easier to test
- easier to cache
- less sensitive to scheduler lifetime/reset rules

### Benefit 2: clearer ownership
The scheduler remains a ggml concern.  
The HPX planner/executor consumes a translated result.

This is especially important because prefill is intentionally **scheduler-driven**, while decode is intentionally **lighter weight** and may use an independent graph walk instead. That split is visible directly in the adapter API:
- `ggml_hpx_adapt_decode(...)`
- `ggml_hpx_adapt_prefill(...)`

---

## 6. BLAS is delegated instead of planned internally

This project keeps BLAS support.

That means the HPX layer does **not** try to reimplement BLAS-backed kernels or treat BLAS as something it can “look inside.” Instead, BLAS-capable work is modeled as a coarse delegated region. BLAS is already a specialized execution backend. If the HPX layer tried to internally decompose BLAS work, it would:
- duplicate backend logic
- blur the backend boundary
- complicate correctness
- risk oversubscription or confused ownership of parallelism

So the chosen approach is:

- HPX may schedule around BLAS regions
- HPX may add dependencies before and after them
- HPX does not plan inside them

That is why region type includes `blas_delegated` as a first-class category. 

---

## 7. Why plans are cached

Planning has a cost.  
If the graph shape and execution assumptions are compatible across runs, rebuilding the plan every time is wasted work.

So plans are cached using a structural key. The key includes:
- mode
- graph shape hash
- backend assignment hash
- workspace/layout signature
- planner policy version

The important idea is that the key must **not** contain ephemeral addresses. Pointer values from a prior graph allocation or scheduler pass are not stable enough to define reuse. Plan reuse should depend on **structure**, not memory accidents.

The current plan design also keeps mismatch categories separate:
- stale graph shape
- stale backend assignment
- stale workspace
- stale policy

That choice was made to improve diagnostics and future tuning. It lets us distinguish:
- “the policy changed”
from
- “the graph structure changed”

Those are very different reasons for cache invalidation.

---

## 8. Abort is cooperative, not preemptive

The abort model is intentionally simple and conservative.

`ggml-hpx-abort.h` defines a cooperative abort token with:
- `request()`
- `check()`
- `reset()`

The contract is:

- no new region starts after abort is observed
- a running CPU region stops at the next safe checkpoint
- in-flight delegated work such as BLAS is not forcibly preempted
- the executor returns an aborted status once cleanup is done
Why not preemptive cancellation?

Because preempting arbitrary low-level computation safely is much harder, especially when:
- CPU kernels are not written for interruption
- BLAS calls may already be in progress
- correctness matters more than aggressive interruption

So the executor’s job is to **observe** abort and stop launching more work, not to forcibly tear apart already-running work.

That model also appears cleanly in `ggml-hpx-exec.h`, where abort is exposed as an executor-level operation and the run status distinguishes `ok` from `aborted`.

---

## 9. Runtime and executor are separate

Another key design decision is the split between `runtime` and `exec`

### Runtime
The runtime owns long-lived HPX-related state:
- runtime lifecycle
- persistent worker teams
- scratch buffer ownership

It also exposes the primitive parallel dispatch operations:
- dispatch on decode team
- dispatch on prefill team

### Executor
The executor owns orchestration:
- reset abort token
- call adapter
- consult cache
- build/insert plan on miss
- iterate regions
- check abort
- fire instrumentation hooks

This split is important because it avoids a common design failure:

> letting the top-level executor silently become the place where every low-level runtime detail accumulates

Instead:
- `runtime` knows **how** to run work
- `exec` knows **what** to run and **when**

That keeps the architecture easier to explain and easier to evolve.

---
### Implementation
## 1. Instrumentation is passive first

Instrumentation often grows until it starts distorting the design.

To avoid that, the first instrumentation layer is intentionally passive.

`ggml-hpx-instrument.h` provides:
- counters
- timing accumulators
- scoped timers
- run and region identifiers
- optional benchmark-facing hooks 

What it does **not** do yet:
- HPX performance counter integration
- APEX integration
- framework-specific tracing

The first job of instrumentation here is to help answer simple questions:
- how many regions ran?
- how often did cache hits happen?
- how much time did a run or region take?

Those questions matter immediately, and they do not require deep integration with runtime-specific tooling.

Later, if we want richer tracing, we can add it without making the core contract depend on it.

```
ggml graph
   │
   ▼
adapter
   │   (decode: lighter graph walk)
   │   (prefill: scheduler-driven topology)
   ▼
immutable HPX planning input
   │
   ├──► structural cache lookup by plan_key
   │         │
   │         ├── hit  ─► reuse plan
   │         └── miss ─► build plan
   │
   ▼
plan
   ├── decode  : coarse regions + chunk policy
   └── prefill : topology + per-region policy + overlap policy
   │
   ▼
executor
   ├── reset abort
   ├── iterate ready regions
   ├── instrumentation hooks
   └── call runtime dispatch
   │
   ▼
runtime
   ├── decode team dispatch
   ├── prefill team dispatch
   └── workspace / scratch ownership
   │
   ▼
region execution
   ├── CPU region
   ├── BLAS delegated region
   └── next ready region
```


## 2. Adapter, plan, cache, runtime scratch, and test baseline

Since the last note (“Instrumentation is passive first”), the HPX layer moved from header-only design into a first working implementation slice.

The biggest completed piece is the **adapter / plan / cache path**. The adapter now has separate decode and prefill entry points, as intended by the contract. The decode path performs an independent graph walk and produces an HPX-owned topology plus a structural plan key. The prefill path is scheduler-informed: it materializes scheduler split state, then reconstructs regions from scheduler-assigned backend identity and verifies the resulting region count against the scheduler’s split count. This means the prefill path is no longer a generic graph walk, but it is still a **verified reconstruction** rather than a direct export of scheduler split boundaries. That distinction matters for future refinement. :contentReference[oaicite:0]{index=0}

The **plan layer** is now implemented as a real structural transformation rather than just a design sketch. Decode plan building copies regions and creates default chunk specs; prefill plan building copies topology and creates default per-region policies. Both decode and prefill plan checks now return precise mismatch reasons (`stale_graph_shape`, `stale_backend_assign`, `stale_workspace`, `stale_policy`) rather than a single stale/not-stale result. One consistency fix made during this step was to treat **empty topology as valid everywhere**. That matches the decode path, matches what the adapter can produce for empty graphs, and keeps the malformed-topology death tests focused on actual structural errors such as out-of-bounds spans or empty spans inside a region. 

The **cache layer** is now real and intentionally simple. It is a value-style cache, not a heap-owned opaque service. It stores at most one decode plan and one prefill plan, and the lookup/insert API returns `const*` so the execution layer can use cached plans without ambiguity about ownership. Inserting a second plan of the same mode replaces the first cleanly. This is enough for a first execution layer and keeps cache semantics easy to test and reason about.

The **runtime** is still only partially implemented, but it has moved beyond a pure stub. Scratch buffer support is now real: the runtime owns a scratch pointer and size, allocates on demand, grows when needed, and keeps the pointer stable when no grow is required. Dispatch is still synchronous and serial on the caller thread for now, so HPX worker teams are not yet wired, but the runtime contract for scratch ownership and “run each chunk exactly once” is already pinned down by tests. This means the next runtime step is narrower: replace the serial dispatch loops with HPX-backed dispatch without changing scratch semantics.

The **executor** remains a stub at this point. It can be created and destroyed, and it owns an abort token, but it does not yet orchestrate the real flow of reset-abort → adapt → validate → cache lookup/build → scratch ensure → region iteration → runtime dispatch. That flow is the next major implementation target.

A major outcome of today’s work is that the project now has a meaningful **test baseline**. The following suites are passing:
- abort token tests: 7/7
- decode plan tests: 8/8
- prefill plan tests: 5/5
- cache tests: 11/11
- adapter tests: 6/6 on the implemented decode path, plus 3 skipped prefill tests that still need broader backend coverage or helper support :contentReference[oaicite:1]{index=1}

The adapter tests are especially important because they now cover:
- stable key generation for the same decode graph
- propagation of `policy_version`
- bounded decode topology with no `sched_split` regions
- ownership-by-value of returned topology
- empty-graph decode behavior
- single-node decode behavior
- CPU-backed prefill split distinctness, once two distinct CPU backend handles are created in the test helper :contentReference[oaicite:2]{index=2}

One design clarification emerged during testing: for CPU-backed prefill split tests, using a generic “initialize backend by type” path can accidentally return the same backend object twice, which would hide split boundaries. The safer test strategy is to construct **distinct CPU backend instances** explicitly so adjacent CPU splits can be observed as separate `sched_split` regions. This reinforces a broader lesson from the adapter work: backend identity, not just backend category (“CPU” vs “non-CPU”), matters for preserving topology boundaries.

In short, the project is no longer just a contract and file layout. It now has:
1. a real adapter,
2. real plan validation/building,
3. a real cache,
4. a partially real runtime (scratch yes, HPX dispatch not yet),
5. a broad green test floor for the implemented pieces. 

## Update: tested implementation baseline established

Since the previous note (“Instrumentation is passive first”), the HPX layer has moved from interface design into a tested implementation baseline.

### What is implemented now

The adapter, plan, cache, runtime, and executor layers all have concrete `.cpp` files. The current state is intentionally incremental:

- **adapter** is real
  - decode path performs an independent graph walk
  - prefill path is scheduler-informed
  - both return an HPX-owned topology snapshot plus a fully populated plan key
- **plan** is real
  - decode and prefill plan builders are implemented
  - topology validation is implemented
  - plan check returns specific mismatch reasons
- **cache** is real
  - value-style cache
  - one decode plan slot and one prefill plan slot
  - replacement semantics are tested
- **runtime** is partially real
  - scratch buffer support is implemented
  - dispatch is still serial/synchronous for now
- **executor** is partially real
  - create/destroy implemented
  - abort is observed and consumed at run entry
  - empty/stub decode and prefill runs return correct status

### Important design choices that were exercised in code

#### 1. Empty topology is valid
The design was made consistent so that empty topology is accepted everywhere it needs to be:
- decode already allowed it
- the adapter can produce it
- prefill validation/building was aligned to that choice

Malformed topology is still rejected where appropriate, for example:
- out-of-bounds node spans
- empty spans inside a region
- decode plans containing `sched_split` regions

#### 2. Prefill is scheduler-informed, but still reconstructed
The current prefill adapter is no longer a generic graph walk. It calls into the scheduler, materializes split state, and reconstructs regions from scheduler-assigned backend identity. It then verifies the resulting region count against the scheduler’s split count.

This means the prefill path is stronger than a naive backend-classification pass, but it is still a **verified reconstruction**, not a direct export of scheduler split boundaries. That distinction remains important for future refinement.

#### 3. Backend identity matters
Testing made it clear that backend identity, not just backend category (“CPU” vs “non-CPU”), matters for preserving region boundaries.

For CPU-backed prefill tests, the fixture must use **distinct CPU backend instances**. Reusing one shared backend would hide boundaries and make the test meaningless. This was an important clarification for the adapter contract.

#### 4. Abort semantics are now observable
The executor now uses an **observe-and-consume** abort rule at run entry:
- if abort is already set, the run returns `aborted`
- the abort flag is cleared at the same time
- a subsequent run can succeed

This keeps the current stub executor behavior honest and matches the intended test contract.

### Test baseline

At this point the implementation baseline is:

- `test_hpx_abort`: **7 passed**
- `test_hpx_cache`: **11 passed**
- `test_hpx_decode_plan`: **8 passed**
- `test_hpx_prefill_plan`: **5 passed**
- `test_hpx_adapter`: **8 passed, 1 skipped**
- `test_hpx_runtime`: **5 passed**
- `test_hpx_exec`: **6 passed**

Total: **50 passed, 1 skipped, 0 failed**. The one skipped test is intentional: it requires a second non-CPU backend to verify delegated-backend distinctness in prefill. :contentReference[oaicite:0]{index=0}

### What this milestone means

The project is no longer just a design and file layout. It now has:

1. a real adapter,
2. real plan validation/building,
3. a real cache,
4. a partially real runtime (scratch yes, HPX dispatch not yet),
5. a partially real executor with observable abort behavior,
6. a broad green test floor for the implemented pieces. 

### What remains next

The next implementation step is to replace serial runtime dispatch with real HPX-backed dispatch while preserving the now-established runtime contract:
- scratch ownership stays the same
- each chunk still runs exactly once
- dispatch remains synchronous from the caller’s perspective

After that, the executor can be upgraded from stub orchestration to a full path that actually uses:
- adapter
- plan/cache
- scratch ensure
- region iteration
- runtime dispatch