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
- `prev_idx`

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


## 2026-04-07 — Runtime dispatch switched from serial stub to real HPX-backed prefill dispatch

### Summary

`ggml-hpx-runtime.cpp` now performs real HPX-backed parallel dispatch for prefill work while keeping the public runtime interface unchanged.

Current behavior:

- `ggml_hpx_runtime_dispatch_decode(...)` remains serial by design.
- `ggml_hpx_runtime_dispatch_prefill(...)` runs serially for `n_chunks < 2`.
- `ggml_hpx_runtime_dispatch_prefill(...)` uses `hpx::async` on the default HPX pool for `n_chunks >= 2`.
- The runtime still appears synchronous to callers: dispatch returns only after all chunk callbacks complete.

### Why this change

The earlier runtime implementation was only a structural stub:

- decode dispatch looped serially
- prefill dispatch looped serially
- tests covered correctness of callback invocation, but runtime did not yet exercise HPX underneath

A new runtime test was added to force the next step:

- `DispatchPrefillReturnsOnlyAfterAllChunksFinish`

This test blocks each chunk callback and waits until all chunks have started before releasing them. It fails against the serial stub and passes only once prefill dispatch launches chunks concurrently.

### Public API constraints kept intact

This change deliberately did **not** alter the public interface of `ggml-hpx-runtime.h`.

Unchanged API surface:

- `ggml_hpx_runtime_create(params)`
- `ggml_hpx_runtime_destroy(rt)`
- `ggml_hpx_runtime_scratch_ptr(rt)`
- `ggml_hpx_runtime_scratch_ensure(rt, bytes)`
- `ggml_hpx_runtime_dispatch_decode(rt, n_chunks, fn, user_data)`
- `ggml_hpx_runtime_dispatch_prefill(rt, n_chunks, fn, user_data)`

Important non-changes:

- no plan-type unification at runtime level
- no new runtime status return type
- no abort token added to runtime dispatch
- no public lane-scratch API
- no change to decode semantics

### HPX lifecycle decision

A refcounted start/stop-per-runtime design was considered first, but abandoned.

Reason:

- HPX cannot be initialized more than once per process
- HPX cannot be cleanly restarted for each `create()` / `destroy()` pair
- `hpx::finalize()` must run on an HPX thread

Final lifecycle design:

- HPX is started once per process via `std::call_once`
- startup happens on first `ggml_hpx_runtime_create(...)`
- shutdown is registered with `std::atexit(...)`
- the exit handler posts `hpx::finalize()` onto an HPX thread and then calls `hpx::stop()`
- `ggml_hpx_runtime_destroy(...)` frees runtime-owned scratch and deletes the runtime object, but does **not** stop HPX

This keeps the runtime wrapper compatible with unit tests, which create and destroy multiple runtime objects in one process.

### Implementation notes

Runtime struct currently stores:

- `n_prefill_threads`
- `scratch_ptr`
- `scratch_bytes`

Current dispatch policy:

- decode: always serial
- prefill:
  - no-op for `fn == nullptr` or `n_chunks == 0`
  - serial for `n_chunks < 2`
  - otherwise launch one `hpx::async` task per chunk, `hpx::wait_all(...)`, then surface exceptions with `future::get()`

The first implementation uses the default/global HPX pool.

`n_prefill_threads` is stored but not yet used to cap task submission or bind work to a dedicated pool. That refinement is deferred.

### Tests added

Three runtime tests were added:

- `DispatchPrefillReturnsOnlyAfterAllChunksFinish`
- `DispatchDecodeSingleChunkRunsExactlyOnce`
- `DispatchPrefillSingleChunkRunsExactlyOnce`

These were added on top of the existing runtime tests.

Interpretation:

- the blocking prefill test proves that prefill dispatch is no longer a serial loop
- the single-chunk tests protect the `n_chunks == 1` edge case, especially since prefill now has a serial fallback for tiny dispatches

### Validation result

After the runtime change:

- `test_hpx_runtime`: 8 passed, 0 failed
- full suite: 56 passed, 1 skipped, 0 failed

Suite breakdown at this milestone:

- `test_hpx_abort`: 7 passed
- `test_hpx_cache`: 11 passed
- `test_hpx_decode_plan`: 8 passed
- `test_hpx_prefill_plan`: 5 passed
- `test_hpx_adapter`: 8 passed, 1 skipped
- `test_hpx_runtime`: 8 passed
- `test_hpx_exec`: 6 passed

### What this milestone means

This is the first point where the runtime layer is doing real HPX work rather than only preserving structure.

The system now has:

- adapter tests green
- decode/prefill plan tests green
- cache tests green
- exec tests green
- runtime prefill dispatch actually using HPX underneath

### Known limitations left intentionally for later

- decode is still serial
- prefill uses the default HPX pool rather than a pool isolated to `n_prefill_threads`
- runtime dispatch still operates at callback/chunk level rather than on a richer execution object
- runtime does not yet expose abort-aware dispatch directly
- delegated-backend distinctness validation still depends on access to a second non-CPU backend

### Best next step

Wire `exec.cpp` through the real orchestration path:

1. observe-and-consume abort at run entry
2. adapt graph
3. determine decode vs prefill mode
4. build or fetch cached plan
5. ensure runtime scratch
6. dispatch through runtime
7. preserve existing exec result semantics

## 2026-04-08 — Exec wired through adapter → cache → runtime

### Summary

`ggml-hpx-exec.cpp` is now the real orchestration entry point for the HPX path. It owns a runtime, a structural plan cache, an abort token, and the policy version, and it drives both decode and prefill through the same high-level sequence: adapt, cache lookup/build, ensure scratch, then dispatch through the runtime. :contentReference[oaicite:0]{index=0}

This is a meaningful milestone because the project is no longer just a collection of separately-tested pieces. The exec layer is now connected end-to-end to the adapter, plan/cache, and runtime layers. The chunk functions are still stubs, so this does **not** execute real ggml compute yet, but the orchestration path is now real. :contentReference[oaicite:1]{index=1}

### Current exec shape

`ggml_hpx_exec` currently contains:

- `ggml_hpx_runtime* runtime`
- `ggml_hpx_plan_cache cache`
- `ggml_hpx_abort_token abort`
- `uint32_t policy_version`

This keeps plan caching and runtime ownership inside exec while leaving live backend handles out of the cached plan model. :contentReference[oaicite:2]{index=2}

### Abort rule implemented in exec

Abort semantics are now explicitly split across three stages:

- **Entry:** observe-and-consume (`check`, `reset`, return `aborted`)
- **Chunk fn:** check only, return early, never reset
- **Post-dispatch:** observe-and-consume again

The helper `consume_abort(...)` is the one place that resets the token. Chunk functions only call `check()`. 

This gives exec cooperative abort behavior without forcing the runtime API itself to become abort-aware.

### Decode path now wired

`ggml_hpx_exec_run_decode(...)` currently does:

1. entry abort consume
2. `ggml_hpx_adapt_decode(graph, policy_version)`
3. early-out if adapted topology has no regions
4. cache lookup by structural key
5. decode plan build + insert on cache miss
6. `ggml_hpx_runtime_scratch_ensure(...)` sized to `plan->regions.size()`
7. runtime dispatch using `ggml_hpx_runtime_dispatch_decode(...)`
8. post-dispatch abort consume

The decode dispatch context currently carries a decode plan pointer, abort token pointer, and scratch pointer fetched from the runtime. 

### Prefill path now wired

`ggml_hpx_exec_run_prefill(...)` mirrors the decode flow, with two notable differences:

- it requires `sched`; `sched == nullptr` is an honest early-out
- chunk count comes from `plan->topo.regions.size()`

The prefill path currently does:

1. entry abort consume
2. early-out on `sched == nullptr`
3. `ggml_hpx_adapt_prefill(graph, sched, policy_version)`
4. early-out if adapted topology has no regions
5. cache lookup by structural key
6. prefill plan build + insert on cache miss
7. `ggml_hpx_runtime_scratch_ensure(...)`
8. runtime dispatch using `ggml_hpx_runtime_dispatch_prefill(...)`
9. post-dispatch abort consume

This keeps scheduler-derived execution decisions in the prefill run path instead of baking them into the cached plan. 

### Chunk functions are still structural stubs

The current decode and prefill chunk functions intentionally do **not** run real ggml operations yet.

They currently:

- cast `user_data` to a run-context struct
- check abort and return early if requested
- validate `chunk_idx` against the plan
- validate region bounds
- touch one byte of scratch at offset `chunk_idx` to prove execution reached that chunk

This means the orchestration path is real, but compute semantics are still stubbed. The scratch touch is safe across chunks because each chunk writes only to its own byte offset. 

### Current status of the design

At this point the architecture is:

- **abort:** real and tested
- **adapter:** real and tested
- **plan/cache:** real and tested
- **runtime:** real HPX-backed prefill dispatch; decode remains serial by design
- **exec:** fully wired through adapter → cache → runtime, but chunk execution is still stubbed

This is the first point where the overall pipeline exists end-to-end even though the chunk body is still placeholder logic.

---

## Next agreed design step — live backends belong in run context, not in plans

### Decision

Plans remain **purely structural**. No live `ggml_backend_t` handles should be stored in a plan or in the plan cache.

Live backend handles are execution-time resources and should be assembled in the **per-run context** passed to chunk functions.

This preserves the invariant that cached plans are structural templates and avoids stale backend handles surviving a cache hit.

### Decode API direction

Decode can produce `blas_delegated` regions, so `run_decode(...)` should not be locked to a single backend handle.

The agreed direction is to add a small per-run decode backend bundle:

```cpp
struct ggml_hpx_decode_backends
{
    ggml_backend_t cpu  = nullptr;  // required
    ggml_backend_t blas = nullptr;  // optional
};
```

The bundle is passed per-run so backend ownership stays with the caller. Plans remain structural templates with no live handles inside them.

Prefill is unchanged: it already receives `sched` and derives backends per split at run time.

### Fallback rule

If a decode plan contains a `blas_delegated` region and `backends.blas` is `nullptr`, the region is routed to `backends.cpu`. This is semantically correct (CPU can execute any operation BLAS would handle, just slower) and requires no new status value.

---

## 2026-04-08 — Decode backend bundle implemented; HPX best-practices audit

### Summary

Two things happened in this session:

1. The `ggml_hpx_decode_backends` bundle was designed, implemented, and tested across four files.
2. A broader HPX best-practices audit was done, resulting in one concrete correction to the runtime.

---

### Decode backend bundle

#### What changed

**`ggml/src/ggml-hpx/ggml-hpx-fwd.h`**

Added a forward declaration for `ggml_backend` and `ggml_backend_t` so that `exec.h` can use the type without pulling in `ggml-backend.h`. The annotation mirrors `ggml-backend.h` line 27.

**`ggml/src/ggml-hpx/ggml-hpx-exec.h`**

Added the `ggml_hpx_decode_backends` bundle struct and updated `ggml_hpx_exec_run_decode` to accept it as a third parameter. `run_prefill` is unchanged — it already receives `sched` and derives backends from it per split.

**`ggml/src/ggml-hpx/ggml-hpx-exec.cpp`**

Added a hard precondition near the top of `run_decode`, after the abort check:

```cpp
GGML_ASSERT(backends.cpu != nullptr);
```

This fires before any plan lookup or dispatch. A null CPU backend makes the bundle meaningless and is treated as a programmer error rather than a graceful fallback.

Added a `backend_for_region` helper that selects `blas` for `blas_delegated` regions when `blas != nullptr`, and falls back to `cpu` otherwise.

Updated `decode_run_ctx` to carry `ggml_hpx_decode_backends backends` instead of a bare cpu handle.

Updated `decode_chunk_fn` to call `backend_for_region` and store the result in a local. This is **API and context shape change only** — no subgraph construction or `ggml_backend_graph_compute` call was added. Real compute semantics land when the subgraph_view constructor is implemented.

**`tests/hpx/test_hpx_exec.cpp`**

Added a RAII wrapper that owns backend lifetime automatically:

```cpp
struct cpu_backends_guard
{
    ggml_hpx_decode_backends b{};
    explicit cpu_backends_guard() { b.cpu = ggml_backend_cpu_init(); }
    ~cpu_backends_guard()        { ggml_backend_free(b.cpu); }
    // deleted copy/assign
};
```

All `run_decode` call sites declare a `cpu_backends_guard bg` and pass `bg.b`. No manual `ggml_backend_free` at call sites.

#### Design decisions

- **Plans stay structural.** No live `ggml_backend_t` handles are stored in the plan or plan cache. Cached plans are structural templates; handles are assembled per-run.
- **CPU is a required precondition, not a comment.** `GGML_ASSERT` fires before any dispatch if `backends.cpu == nullptr`.
- **BLAS fallback is silent and correct.** Missing `blas` routes to `cpu`. No new status value needed.
- **`backend_for_region` is plumbing today.** It shapes the run context correctly for the eventual compute step but does not yet invoke `ggml_backend_graph_compute`.

---

### HPX best-practices audit

#### Correct decisions

- **Lifecycle (`hpx::start` + `atexit`)**: canonical embedded-HPX pattern. `hpx::start` lets the main thread return; `hpx::init`/`hpx::main` would block it, which is wrong for a library. The `hpx::post(finalize)` trick is required because `hpx::finalize()` must be called from within an HPX thread.
- **`std::call_once` for the no-restart constraint**: correct.
- **`hpx::wait_all`**: correct HPX barrier primitive — see correction below.

#### Corrected: `wait_all` + `get()` loop was redundant

The prior implementation followed `hpx::wait_all(futures)` with a `f.get()` loop under the assumption that `wait_all` was a barrier-only primitive and `get()` was needed to surface exceptions.

This was wrong.

In current HPX, `hpx::wait_all` waits for all futures to become ready **and** rethrows any stored exceptions. A following `get()` loop is redundant for `future<void>` — there are no results to extract and exceptions are already surfaced. The loop was removed from `ggml-hpx-runtime.cpp` and the file-top comment was corrected.

A per-future `get()` loop is still acceptable if per-future error handling or explicit result consumption is needed, but it must not be added "for exception propagation" because that reason is incorrect.

#### Acceptable but non-idiomatic

- **`hpx::async` per chunk + manual future vector**: works, but HPX's idiomatic parallel loop is `hpx::experimental::for_loop` or `hpx::for_each` with `hpx::execution::par`. Those avoid the heap allocation for the future vector and compose better with executors. For the current small-N-chunks use case the difference is negligible, but this is a known style gap.

#### Known gaps, deferred intentionally

- **`n_prefill_threads` is stored but ignored.** The default HPX pool uses however many threads HPX was started with (env-controlled). Wiring `n_prefill_threads` to a per-dispatch executor is deferred until the dispatch model matures.
- **No `hpx::init_params` at startup.** Thread count and scheduler config rely on HPX auto-detection or environment variables. Acceptable for now.

---

### Test result at this milestone

No new tests were added in this session; the existing suite remained green:

- `test_hpx_abort`: 7 passed
- `test_hpx_cache`: 11 passed
- `test_hpx_decode_plan`: 8 passed
- `test_hpx_prefill_plan`: 5 passed
- `test_hpx_adapter`: 8 passed, 1 skipped
- `test_hpx_runtime`: 8 passed
- `test_hpx_exec`: 6 passed

**Total: 56 passed, 1 skipped, 0 failed**

---

### What remains next

The exec layer is now wired end-to-end with the correct context shape. The next implementation step is to make chunk functions do real compute:

1. Implement a `subgraph_view` constructor that presents `graph->nodes[begin..end)` as a complete `ggml_cgraph` without copying.
2. Call `ggml_backend_graph_compute(backend, &subgraph_view)` inside `decode_chunk_fn`.
3. Handle the `ggml_backend_graph_compute` return value and surface errors through the abort/status path.

After that, the prefill chunk function needs the same treatment, driven from the scheduler-split model: backend per split derived from `sched` at run time, not from a static bundle.

## 2026-04-08: llama.cpp end-to-end HPX exec smoke integration

### Summary

This change moves the project from isolated `ggml-hpx-exec` correctness tests to a tiny end-to-end llama.cpp integration path.

The HPX execution layer is now wired into `llama_context::graph_compute` behind an opt-in environment variable (`LLAMA_USE_HPX`) when `GGML_HPX` is enabled at build time. The goal of this milestone is correctness, not performance: prove that HPX-backed orchestration can replace the scheduler’s compute call on a CPU-only inference path and produce the same outputs as the reference path.

The initial llama integration was validated with a model-backed smoke test that compares HPX and reference logits on the same prompt. The integration also exposed three correctness bugs in region/dependency handling, all of which were fixed in this milestone.

### Integration point

The single replacement point is `src/llama-context.cpp`, inside:

- `llama_context::graph_compute(ggml_cgraph * gf, bool batched)`

The existing thread/threadpool setup in `graph_compute` remains unchanged. The only behavioral change is that, when `LLAMA_USE_HPX` is set and `hpx_exec` exists, the function routes compute through `ggml-hpx-exec` instead of calling the scheduler’s normal compute path.

Mapping is:

- `batched == false` → `ggml_hpx_exec_run_decode(...)`
- `batched == true`  → `ggml_hpx_exec_run_prefill(...)`

This keeps the hook at the narrowest possible point: all existing graph construction and llama-side batching logic remain intact.

### Public/API surface

No permanent public llama API knob was added for this milestone.

Instead:

- `llama_context` owns an optional private `ggml_hpx_exec * hpx_exec`
- the constructor reads `LLAMA_USE_HPX`
- when enabled, it creates the HPX exec object
- the destructor destroys it

This keeps the integration off by default and avoids exposing unstable HPX-exec plumbing through the public API before the path is better understood.

### Current compute behavior

#### Decode

Decode remains structurally simple:

- adaptation/build/cache still run through `ggml-hpx-exec`
- live backends are assembled per run
- the backend bundle currently uses:
  - `cpu = backend_cpu`
  - `blas = nullptr`

This means any `blas_delegated` decode region falls back to CPU execution for now. That is semantically correct and sufficient for the CPU-only smoke path.

#### Prefill

Prefill currently executes **sequentially** for correctness.

Earlier versions launched all prefill regions in parallel with `hpx::async`, but this was incorrect for sequential graphs because later regions consumed activations that earlier regions had not finished producing. The dispatch was changed to a serial loop, matching decode behavior for now.

This is intentional. Prefill parallelism now depends on explicit dependency-aware scheduling and should not be re-enabled until that logic exists.

### Real compute in chunk functions

This milestone completed the intended transition from stub chunk functions to real region execution:

- chunk functions now build region views with `ggml_graph_view(...)`
- chunk functions call `ggml_backend_graph_compute(...)`
- backend failures request abort and surface as exec failure

There is no per-node private compute-forward loop and no heap-copy subgraph construction.

### Scheduler / allocation contract

For the llama smoke path, the HPX branch in `graph_compute` now does:

1. `ggml_backend_sched_reset(sched.get())`
2. `ggml_backend_sched_alloc_graph(sched.get(), gf)`
3. route to HPX exec

This is correctness-first behavior. It is acceptable for a smoke/integration milestone, but it does give up graph-reuse efficiency because the scheduler is reset and allocation is re-established on each HPX compute call.

The project still needs a cleaner long-term allocation contract (for example, an explicit public “ensure allocated” helper or equivalent state tracking), but that is out of scope for this change.

### Bugs found and fixed during end-to-end integration

#### Bug 1: artificial 64-region cap

**Root cause**

`ggml_hpx_region` used:

- `dep_mask : uint64_t`

This imposed an artificial region limit because the old code assumed a bitmask dependency encoding. Real LLM graphs produce far more than 64 backend transitions, especially on macOS where Accelerate/BLAS and host CPU buffers alternate frequently.

**Fix**

Replaced:

- `dep_mask : uint64_t`

with:

- `prev_idx : uint32_t`

where:

- `UINT32_MAX` = no predecessor
- otherwise `prev_idx` is the immediate predecessor region

This removed:

- `GGML_HPX_MAX_REGIONS`
- `sequential_dep()`
- the related static assertion
- all bitmask-based assumptions

This is a more honest representation of the current execution model: regions are presently consumed as a linear chain, not a general DAG.

#### Bug 2: double `split_graph` in prefill adaptation

**Root cause**

`adapt_prefill(...)` originally called `split_graph(...)` internally.

After llama integration, the caller was already doing `ggml_backend_sched_alloc_graph(...)`, which itself populates scheduler split state. Calling `split_graph(...)` again inside the adapter changed scheduler state after allocation and invalidated assumptions about split counts.

**Fix**

Removed the internal `split_graph(...)` call from `adapt_prefill(...)`.

Prefill adaptation is now a **pure reader** of scheduler state. The caller is responsible for ensuring split/allocation state has already been populated before calling the adapter or `run_prefill(...)`.

Tests that called `adapt_prefill(...)` directly were updated to populate split state explicitly first.

#### Bug 3: invalid `regions.size() == expected_splits` assumption

**Root cause**

The old prefill region builder assumed that the number of contiguous backend regions would match the scheduler split count.

That was only valid before allocation. After `alloc_graph(...)`, the scheduler may insert extra copy nodes for backend transfers. Those nodes can create additional backend-type boundaries without changing the original split count in the way the adapter expected.

**Fix**

Removed the invalid equality assertion and the unused `expected_splits` parameter.

The correct invariants now are structural only:

- regions are non-empty
- first region starts at node 0
- last region ends at `ggml_graph_n_nodes(gf)`
- regions are contiguous and ordered
- each region is backend-uniform under current construction rules

#### Bug 4: parallel prefill data races / NaN outputs

**Root cause**

The first real-compute prefill runtime launched all regions concurrently.

For sequential graphs, region `N` depends on outputs from region `N-1`. Parallel launch allowed later regions to read tensors before their producers completed, causing bad values and NaNs.

**Fix**

Changed prefill dispatch to a sequential loop.

The new `prev_idx` field is the correct place to encode ordering information for future dependency-aware prefill parallelism, but no such scheduler exists yet. Correctness takes priority.

### Tests added / updated

#### ggml-hpx exec tests

Added real-compute correctness tests for both execution paths:

- decode CPU-only correctness on a nontrivial graph (`mul_mat` + `neg`)
- prefill CPU-only correctness on a scheduler-backed graph (`mul_mat` + `neg`)

These compare HPX execution results against the plain backend/scheduler baseline and verify that the outputs match exactly.

#### llama smoke test

Added:

- `tests/hpx/test_hpx_llama_smoke.cpp`

This test:

1. reads `LLAMACPP_TEST_MODELFILE`
2. skips cleanly if no model is provided
3. loads one model
4. creates two contexts:
   - reference path
   - HPX path (`LLAMA_USE_HPX=1`)
5. runs the same prompt through both
6. compares logits element-by-element

This is the first model-backed test proving that HPX exec can replace the normal compute call inside llama.cpp on a CPU-only path.

### Files changed in this milestone

- `ggml/src/ggml-hpx/ggml-hpx-region.h`
  - replace `dep_mask` with `prev_idx`
- `ggml/src/ggml-hpx/ggml-hpx-adapter.cpp`
  - prefill adapter no longer calls `split_graph(...)`
  - region construction updated for `prev_idx`
  - invalid split-count equality assumption removed
- `ggml/src/ggml-hpx/ggml-hpx-adapter.h`
  - contract updated: caller must prepare scheduler split state
- `ggml/src/ggml-hpx/ggml-hpx-exec.cpp`
  - real compute in chunk functions via `ggml_graph_view(...)` + `ggml_backend_graph_compute(...)`
  - prefill dispatch made sequential for correctness
- `ggml/src/ggml-hpx/ggml-hpx-exec.h`
  - exec entry points take `ggml_cgraph *`
- `src/llama-context.h`
  - private `ggml_hpx_exec * hpx_exec`
- `src/llama-context.cpp`
  - `LLAMA_USE_HPX` handling
  - HPX branch inside `graph_compute(...)`
- `src/CMakeLists.txt`
  - link llama against `ggml-hpx` when `GGML_HPX` is enabled
- `tests/hpx/test_hpx_exec.cpp`
  - new real-compute exec tests
- `tests/hpx/test_hpx_llama_smoke.cpp`
  - new model-backed smoke test
- `tests/hpx/CMakeLists.txt`
  - add smoke test target

### Current status after this milestone

What is now true:

- `ggml-hpx-exec` is wired into llama.cpp at a real end-to-end integration point
- decode chunk functions run real backend compute
- prefill chunk functions run real backend compute
- the llama CPU-only smoke path is in place
- the prior fake 64-region ceiling is gone
- prefill no longer mutates scheduler split state during adaptation
- prefill is correct, but currently sequential

What is intentionally **not** solved yet:

- no dependency-aware parallel prefill scheduling
- no graph-reuse optimization for the HPX branch in `graph_compute`
- no live BLAS backend in the llama decode bundle yet (`blas = nullptr`)
- no llama-side abort plumbing into `ggml_hpx_exec_abort`
- no tuning yet for `n_prefill_threads` / executor choice

### Next steps

1. Run `examples/simple` on the same model/prompt with and without `LLAMA_USE_HPX=1` and compare generated output.
2. Add light timing for:
   - prefill-heavy prompt
   - single-token decode
   - short decode loop
3. Design dependency-aware prefill scheduling using `prev_idx`.
4. Improve allocation reuse in the HPX llama path so `graph_compute` does not need reset+alloc on every call.
5. Revisit decode BLAS backend plumbing once correctness/perf baselines are established.