# HPX Implementation Provenance
**For the llama.cpp HPX executor redesign**

## What this document is for

This file explains **why this HPX layer exists**, **what problem it is trying to solve**, and **why the code is organized the way it is**.

### How to read this document

This document is chronological.

It records what happened in the order it actually happened, including intermediate benchmark results and design interpretations that were later corrected. Later sections sometimes revise earlier conclusions rather than replacing them.

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

### Why HPX?

HPX is a C++ runtime for asynchronous and parallel execution. It provides facilities for:
- task scheduling
- futures
- work distribution
- synchronization
- runtime-managed execution resources

A naive idea would be:

> “Replace the old threadpool with HPX and everything gets faster.”

That is **not** the assumption here.

The experience from the earlier `hpx-threadpool-experiment` branch suggested a more careful conclusion:

- HPX can express the concurrency we need
- but a direct one-for-one replacement of an existing low-overhead threadpool is **not automatically faster**
- overhead matters a lot for inference, especially on the single-token path

That is why this redesign does **not** start from “make HPX look like pthreads.”  
It starts from:

> “What is the right execution contract if we know we have two different workloads: decode and prefill?”

### Decode and prefill are different problems

This project treats model inference as having **two major execution modes**.

#### Decode
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

#### Prefill
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

### The execution unit is not a ggml op

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

### _Decision_: Scheduler coupling is isolated to the adapter

> **Only one translation unit should know about scheduler internals.**

That translation unit is `ggml-hpx-adapter.cpp`.

The adapter’s job is to convert external graph/scheduler state into **HPX-owned planning inputs**:
- immutable topology
- fully populated plan key

This is why:
- `ggml-hpx-plan.cpp` must not read scheduler state directly
- `ggml-hpx-exec.cpp` must not read scheduler state directly
- `ggml-backend.h` is allowed only in the adapter implementation path, not throughout the HPX layer

This decision has two big benefits.

#### Benefit 1: cleaner planning
The plan builder consumes an immutable snapshot, not a live scheduler object.

That makes planning:
- easier to test
- easier to cache
- less sensitive to scheduler lifetime/reset rules

#### Benefit 2: clearer ownership
The scheduler remains a ggml concern.  
The HPX planner/executor consumes a translated result.

This is especially important because prefill is intentionally **scheduler-driven**, while decode is intentionally **lighter weight** and may use an independent graph walk instead. That split is visible directly in the adapter API:
- `ggml_hpx_adapt_decode(...)`
- `ggml_hpx_adapt_prefill(...)`

### BLAS is delegated instead of planned internally

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

### Plans are cached

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

### Abort is cooperative, not preemptive

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

### _Decision_: Runtime and executor are separate

Another key design decision is the split between `runtime` and `exec`.

#### Runtime
The runtime owns long-lived HPX-related state:
- runtime lifecycle
- persistent worker teams
- scratch buffer ownership

It also exposes the primitive parallel dispatch operations:
- dispatch on decode team
- dispatch on prefill team

#### Executor
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

```text
HPX SIDE                            PLAIN LLAMA/GGML SIDE

adapter                             scheduler/split-prep world
ggml_hpx_adapt_*                    ggml_backend_sched_alloc_graph(...)
                                    split_graph(...)

plan                                implicit scheduler/backend setup
explicit HPX plan objects           graph + split/allocation state

cache                               prepared scheduler/allocation state
explicit plan cache                 what alloc_graph() has already set up

executor / manager                  llama_context::graph_compute(...)
ggml_hpx_exec_*                     src/llama-context.cpp

runtime / workers                   backend compute path
ggml-hpx-runtime                    ggml_backend_graph_compute(...)
```

---

## 2. Early implementation: adapter, plan, cache, runtime, and executor

### Instrumentation is passive first

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

### HPX orchestration pipeline

```text
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

### Adapter, plan, cache, runtime scratch, and test baseline

The HPX layer first moved from header-only design into a working implementation slice. The biggest completed piece at this stage was the **adapter / plan / cache path**.

The adapter has separate decode and prefill entry points, as intended by the contract.

- The **decode** path performs an independent graph walk and produces an HPX-owned topology plus a structural plan key.
- The **prefill** path is scheduler-informed: it materializes scheduler split state, then reconstructs regions from scheduler-assigned backend identity and verifies the resulting region count against the scheduler’s split count.

This means the prefill path is no longer a generic graph walk, but it is still a **verified reconstruction** rather than a direct export of scheduler split boundaries.

The **plan layer** became a real structural transformation rather than just a design sketch. Decode plan building copies regions and creates default chunk specs. Prefill plan building copies topology and creates default per-region policies. Both decode and prefill plan checks now return precise mismatch reasons:
- `stale_graph_shape`
- `stale_backend_assign`
- `stale_workspace`
- `stale_policy`

One consistency fix made during this step was to treat **empty topology as valid everywhere**. That matches the decode path, matches what the adapter can produce for empty graphs, and keeps malformed-topology death tests focused on actual structural errors such as out-of-bounds spans or empty spans inside a region.

The **cache layer** became real and intentionally simple. It is a value-style cache, not a heap-owned opaque service. It stores at most one decode plan and one prefill plan, and the lookup/insert API returns `const*` so the execution layer can use cached plans without ambiguity about ownership. Inserting a second plan of the same mode replaces the first cleanly.

The **runtime** was still only partially implemented, but it had moved beyond a pure stub. Scratch buffer support became real: the runtime owns a scratch pointer and size, allocates on demand, grows when needed, and keeps the pointer stable when no grow is required. Dispatch was still synchronous and serial on the caller thread for now, so HPX worker teams were not yet wired.

The **executor** remained a stub at that point. It could be created and destroyed, and it owned an abort token, but it did not yet orchestrate the full adapt → cache → scratch → dispatch flow.

### What was implemented at this stage

- **adapter** was real
  - decode path performs an independent graph walk
  - prefill path is scheduler-informed
  - both return an HPX-owned topology snapshot plus a fully populated plan key
- **plan** was real
  - decode and prefill plan builders implemented
  - topology validation implemented
  - plan check returns specific mismatch reasons
- **cache** was real
  - value-style cache
  - one decode plan slot and one prefill plan slot
  - replacement semantics tested
- **runtime** was partially real
  - scratch buffer support implemented
  - dispatch still serial/synchronous
- **executor** was partially real
  - create/destroy implemented
  - abort observed and consumed at run entry
  - empty/stub decode and prefill runs return correct status

### _Decision_: Important design choices exercised in code

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

This means the prefill path is stronger than a naive backend-classification pass, but it is still a **verified reconstruction**, not a direct export of scheduler split boundaries.

#### 3. Backend identity matters
Testing made it clear that backend identity, not just backend category (“CPU” vs “non-CPU”), matters for preserving region boundaries.

For CPU-backed prefill tests, the fixture must use **distinct CPU backend instances**. Reusing one shared backend would hide boundaries and make the test meaningless. This was an important clarification for the adapter contract.

#### 4. Abort semantics are observable
The executor now uses an **observe-and-consume** abort rule at run entry:
- if abort is already set, the run returns `aborted`
- the abort flag is cleared at the same time
- a subsequent run can succeed

This keeps the current stub executor behavior honest and matches the intended test contract.

### Test baseline at this stage

- `test_hpx_abort`: **7 passed**
- `test_hpx_cache`: **11 passed**
- `test_hpx_decode_plan`: **8 passed**
- `test_hpx_prefill_plan`: **5 passed**
- `test_hpx_adapter`: **8 passed, 1 skipped**
- `test_hpx_runtime`: **5 passed**
- `test_hpx_exec`: **6 passed**

Total: **50 passed, 1 skipped, 0 failed**

The one skipped test is intentional: it requires a second non-CPU backend to verify delegated-backend distinctness in prefill.

### Runtime dispatch switched from serial stub to real HPX-backed prefill dispatch

The next runtime step was to replace the serial prefill dispatch stub with real HPX-backed dispatch while keeping the public runtime interface unchanged.

Current behavior after this change:
- `ggml_hpx_runtime_dispatch_decode(...)` remains serial by design
- `ggml_hpx_runtime_dispatch_prefill(...)` runs serially for `n_chunks < 2`
- `ggml_hpx_runtime_dispatch_prefill(...)` uses `hpx::async` on the default HPX pool for `n_chunks >= 2`
- the runtime still appears synchronous to callers: dispatch returns only after all chunk callbacks complete


A new runtime test forced the next step:
- `DispatchPrefillReturnsOnlyAfterAllChunksFinish`

This test blocks each chunk callback and waits until all chunks have started before releasing them. It fails against the serial stub and passes only once prefill dispatch launches chunks concurrently.

### Public API constraints kept intact

This change deliberately did **not** alter the public interface of `ggml-hpx-runtime.h`.

- no plan-type unification at runtime level
- no new runtime status return type
- no abort token added to runtime dispatch
- no public lane-scratch API
- no change to decode semantics

### _Decision_: HPX runtime lifecycle starts once per process

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

### Implementation notes at this stage

Runtime struct currently stores:
- `n_prefill_threads`
- `scratch_ptr`
- `scratch_bytes`

Current dispatch policy:
- decode: always serial
- prefill:
  - no-op for `fn == nullptr` or `n_chunks == 0`
  - serial for `n_chunks < 2`
  - otherwise launch one `hpx::async` task per chunk, `hpx::wait_all(...)`, then surface exceptions

The first implementation uses the default/global HPX pool.

### Tests added

Three runtime tests were added:
- `DispatchPrefillReturnsOnlyAfterAllChunksFinish`
- `DispatchDecodeSingleChunkRunsExactlyOnce`
- `DispatchPrefillSingleChunkRunsExactlyOnce`

These were added on top of the existing runtime tests.

### Validation result after the runtime change

- `test_hpx_runtime`: **8 passed**
- full suite: **56 passed, 1 skipped, 0 failed**

Suite breakdown at this milestone:
- `test_hpx_abort`: 7 passed
- `test_hpx_cache`: 11 passed
- `test_hpx_decode_plan`: 8 passed
- `test_hpx_prefill_plan`: 5 passed
- `test_hpx_adapter`: 8 passed, 1 skipped
- `test_hpx_runtime`: 8 passed
- `test_hpx_exec`: 6 passed

### What this milestone meant

This was the first point where the runtime layer was doing real HPX work rather than only preserving structure.

The system now had:
- adapter tests green
- decode/prefill plan tests green
- cache tests green
- exec tests green
- runtime prefill dispatch actually using HPX underneath

### Known limitations intentionally left for later

- decode is still serial
- prefill uses the default HPX pool rather than a pool isolated to `n_prefill_threads`
- runtime dispatch still operates at callback/chunk level rather than on a richer execution object
- runtime does not yet expose abort-aware dispatch directly
- delegated-backend distinctness validation still depends on access to a second non-CPU backend

### Exec wired through adapter → cache → runtime

The next step was to make `ggml-hpx-exec.cpp` the real orchestration entry point.

`ggml-hpx-exec.cpp` now owned:
- a runtime
- a structural plan cache
- an abort token
- the policy version

Both decode and prefill were driven through the same high-level sequence:
- adapt
- cache lookup/build
- ensure scratch
- dispatch through the runtime

This was a meaningful milestone because the project was no longer just a collection of separately-tested pieces. The exec layer was connected end-to-end to the adapter, plan/cache, and runtime layers. The chunk functions were still stubs, so this did **not** yet execute real ggml compute, but the orchestration path was real.

### Current exec shape at this stage

`ggml_hpx_exec` currently contained:
- `ggml_hpx_runtime* runtime`
- `ggml_hpx_plan_cache cache`
- `ggml_hpx_abort_token abort`
- `uint32_t policy_version`

This kept plan caching and runtime ownership inside exec while leaving live backend handles out of the cached plan model.

### Abort rule implemented in exec

Abort semantics were explicitly split across three stages:
- **Entry:** observe-and-consume (`check`, `reset`, return `aborted`)
- **Chunk fn:** check only, return early, never reset
- **Post-dispatch:** observe-and-consume again

The helper `consume_abort(...)` is the one place that resets the token. Chunk functions only call `check()`.

### Decode path wired

`ggml_hpx_exec_run_decode(...)` did:
1. entry abort consume
2. `ggml_hpx_adapt_decode(graph, policy_version)`
3. early-out if adapted topology has no regions
4. cache lookup by structural key
5. decode plan build + insert on cache miss
6. `ggml_hpx_runtime_scratch_ensure(...)` sized to `plan->regions.size()`
7. runtime dispatch using `ggml_hpx_runtime_dispatch_decode(...)`
8. post-dispatch abort consume

### Prefill path wired

`ggml_hpx_exec_run_prefill(...)` mirrored the decode flow, with two notable differences:
- it required `sched`; `sched == nullptr` was an honest early-out
- chunk count came from `plan->topo.regions.size()`

The prefill path did:
1. entry abort consume
2. early-out on `sched == nullptr`
3. `ggml_hpx_adapt_prefill(graph, sched, policy_version)`
4. early-out if adapted topology has no regions
5. cache lookup by structural key
6. prefill plan build + insert on cache miss
7. `ggml_hpx_runtime_scratch_ensure(...)`
8. runtime dispatch using `ggml_hpx_runtime_dispatch_prefill(...)`
9. post-dispatch abort consume

This kept scheduler-derived execution decisions in the prefill run path instead of baking them into the cached plan.

### Chunk functions were still structural stubs

The current decode and prefill chunk functions intentionally did **not** run real ggml operations yet.

They:
- cast `user_data` to a run-context struct
- check abort and return early if requested
- validate `chunk_idx` against the plan
- validate region bounds
- touch one byte of scratch at offset `chunk_idx` to prove execution reached that chunk

This meant the orchestration path was real, but compute semantics were still stubbed.

### _Decision_: live backends belong in run context, not in plans

Plans remained **purely structural**. No live `ggml_backend_t` handles should be stored in a plan or in the plan cache.

Live backend handles are execution-time resources and should be assembled in the **per-run context** passed to chunk functions.

This preserved the invariant that cached plans are structural templates and avoided stale backend handles surviving a cache hit.

### Decode API direction

Decode can produce `blas_delegated` regions, so `run_decode(...)` should not be locked to a single backend handle.

The agreed direction was to add a small per-run decode backend bundle:

```cpp
struct ggml_hpx_decode_backends
{
    ggml_backend_t cpu  = nullptr;  // required
    ggml_backend_t blas = nullptr;  // optional
};
```

The bundle is passed per-run so backend ownership stays with the caller. Plans remain structural templates with no live handles inside them.

### Fallback rule

If a decode plan contains a `blas_delegated` region and `backends.blas` is `nullptr`, the region is routed to `backends.cpu`. This is semantically correct and requires no new status value.

### Decode backend bundle implemented

Implementation highlights:
- forward declarations added so exec headers could mention `ggml_backend_t` without pulling in heavy backend headers
- `ggml_hpx_exec_run_decode` updated to accept the backend bundle
- a hard precondition was added:
  ```cpp
  GGML_ASSERT(backends.cpu != nullptr);
  ```
- `backend_for_region` selects BLAS when available and otherwise falls back to CPU
- decode run context now carries the full backend bundle

### HPX best-practices audit

A broader HPX best-practices audit was done, resulting in one concrete correction to the runtime.

- **Lifecycle (`hpx::start` + `atexit`)**: canonical embedded-HPX pattern
- **`std::call_once` for the no-restart constraint**: correct
- **`hpx::wait_all`**: the earlier `wait_all` + `get()` loop was redundant

The prior implementation followed `hpx::wait_all(futures)` with a `f.get()` loop under the assumption that `wait_all` was a barrier-only primitive and `get()` was needed to surface exceptions.

This was wrong.

In current HPX, `hpx::wait_all` waits for all futures to become ready **and** rethrows any stored exceptions. A following `get()` loop is redundant for `future<void>`. The loop was removed from `ggml-hpx-runtime.cpp` and the file-top comment was corrected.

#### Acceptable but non-idiomatic

- **`hpx::async` per chunk + manual future vector**: works, but HPX's more idiomatic parallel loop tools are `hpx::experimental::for_loop` or `hpx::for_each` with `hpx::execution::par`. Those avoid the heap allocation for a future vector and compose better with executors. For the current small-`N` chunk counts, the practical difference is negligible, so this was left as a style gap rather than treated as a correctness issue.

#### Known gaps, deferred intentionally

- **`n_prefill_threads` is stored but ignored.** The default HPX pool still uses however many threads HPX was started with. Wiring `n_prefill_threads` to a dedicated pool or executor was deferred until the dispatch model stabilized.
- **No `hpx::init_params` at startup.** Thread count and scheduler configuration still rely on HPX auto-detection or environment variables. This was acceptable for the current phase.

### Test result at this stage

No new tests were added in that session; the existing suite remained green:

- `test_hpx_abort`: 7 passed
- `test_hpx_cache`: 11 passed
- `test_hpx_decode_plan`: 8 passed
- `test_hpx_prefill_plan`: 5 passed
- `test_hpx_adapter`: 8 passed, 1 skipped
- `test_hpx_runtime`: 8 passed
- `test_hpx_exec`: 6 passed

**Total: 56 passed, 1 skipped, 0 failed**

At this point the architecture had:
- real and tested abort
- real and tested adapter
- real and tested plan/cache
- real HPX-backed prefill runtime, with decode still serial
- exec fully wired through adapter → cache → runtime, but chunk execution still stubbed

---

## 3. End-to-end llama.cpp integration and the limit of “HPX above ggml”

At this point the HPX layer was no longer just an internal library. The next question was whether it could replace the normal compute path inside a real llama.cpp run.

This phase also turned out to be the point where the first hard limit of the “outer HPX above ggml” design became visible.

### Llama.cpp end-to-end HPX exec smoke integration

The HPX execution layer was wired into `llama_context::graph_compute(...)` behind an opt-in environment variable (`LLAMA_USE_HPX`) when `GGML_HPX` is enabled at build time.

The goal of this milestone was correctness, not performance:
- prove that HPX-backed orchestration can replace the scheduler’s compute call on a CPU-only inference path
- produce the same outputs as the reference path

### Integration point

The single replacement point is `src/llama-context.cpp`, inside:
- `llama_context::graph_compute(ggml_cgraph * gf, bool batched)`

The existing thread/threadpool setup in `graph_compute` remained unchanged. The only behavioral change was that, when `LLAMA_USE_HPX` is set and `hpx_exec` exists, the function routed compute through `ggml-hpx-exec` instead of calling the scheduler’s normal compute path.

Mapping:
- `batched == false` → `ggml_hpx_exec_run_decode(...)`
- `batched == true`  → `ggml_hpx_exec_run_prefill(...)`

### Public/API surface

No permanent public llama API knob was added for this milestone.

Instead:
- `llama_context` owns an optional private `ggml_hpx_exec * hpx_exec`
- the constructor reads `LLAMA_USE_HPX`
- when enabled, it creates the HPX exec object
- the destructor destroys it

### Current compute behavior at first integration

#### Decode
Decode remained structurally simple:
- adaptation/build/cache still run through `ggml-hpx-exec`
- live backends are assembled per run
- the backend bundle currently uses:
  - `cpu = backend_cpu`
  - `blas = nullptr`

This means any `blas_delegated` decode region falls back to CPU execution for now.

#### Prefill
Prefill initially launched regions in parallel, but that was found to be incorrect for sequential graphs. It was changed to a **sequential loop for correctness**.

This is intentional. Prefill parallelism now depends on explicit dependency-aware scheduling and should not be re-enabled until that logic exists.

### Real compute in chunk functions

This milestone completed the intended transition from stub chunk functions to real region execution:
- chunk functions now build region views with `ggml_graph_view(...)`
- chunk functions call `ggml_backend_graph_compute(...)`
- backend failures request abort and surface as exec failure

There is no per-node private compute-forward loop and no heap-copy subgraph construction.

### Scheduler / allocation contract

For the llama smoke path, the HPX branch originally did:
1. `ggml_backend_sched_reset(sched.get())`
2. `ggml_backend_sched_alloc_graph(sched.get(), gf)`
3. route to HPX exec

This turned out to be wrong, because llama had already done:
1. reset
2. alloc_graph
3. `set_inputs(...)`
4. `graph_compute(...)`

Repeating reset/allocation inside the HPX branch destroyed and recreated backend allocations **after** input tensors had already been populated.

### Bugs found and fixed during end-to-end integration

#### Bug 1: artificial 64-region cap
**Root cause:** `ggml_hpx_region` used a `uint64_t dep_mask`, which imposed an artificial region limit.

**Fix:** replaced `dep_mask` with `prev_idx : uint32_t`, using:
- `UINT32_MAX` = no predecessor
- otherwise `prev_idx` = immediate predecessor region

This also removed the old scaffolding tied to the bitmask model:
- `GGML_HPX_MAX_REGIONS`
- `sequential_dep()`
- the related static assertion

This removed the fake region ceiling and also matched the actual current execution model more honestly.

#### Bug 2: double `split_graph` in prefill adaptation
**Root cause:** `adapt_prefill(...)` originally called `split_graph(...)` internally, even after the caller had already done `ggml_backend_sched_alloc_graph(...)`.

**Fix:** removed the internal `split_graph(...)` call.  
Prefill adaptation is now a **pure reader** of scheduler state.

#### Bug 3: invalid `regions.size() == expected_splits` assumption
**Root cause:** the old prefill region builder assumed the number of contiguous backend regions would match the scheduler split count.

That was only valid before allocation. After `alloc_graph(...)`, the scheduler may insert extra copy nodes for backend transfers.

**Fix:** removed the invalid equality assertion and the unused `expected_splits` parameter. The remaining invariants are structural only.

#### Bug 4: parallel prefill data races / NaN outputs
**Root cause:** the first real-compute prefill runtime launched all regions concurrently, even though later regions depended on earlier ones.

**Fix:** changed prefill dispatch to a **sequential loop**.

#### Bug 5: HPX `graph_compute(...)` branch clobbered already-written inputs
**Root cause:** the HPX branch in `llama_context::graph_compute(...)` repeated:
- `ggml_backend_sched_reset(...)`
- `ggml_backend_sched_alloc_graph(...)`

after `set_inputs(...)` had already written prompt data.

**Observed symptom:** on a long-prompt, prefill-heavy TinyLlama run:
- reference path generated a whitespace token
- HPX path generated `<unk>`

**Fix:** removed the redundant scheduler reset/allocation from the HPX branch in `graph_compute(...)`.

The contract is now explicit:
- caller prepares scheduler state and writes inputs
- `graph_compute(...)` performs compute only

After this fix, the long-prompt divergence disappeared.

### Tests added and updated

#### ggml-hpx exec tests
Added real-compute correctness tests for both execution paths:
- decode CPU-only correctness on a nontrivial graph (`mul_mat` + `neg`)
- prefill CPU-only correctness on a scheduler-backed graph (`mul_mat` + `neg`)

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

### End-to-end TinyLlama results

Model used:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M

Driver:
- `llama-simple`

Runs compared:
- baseline
- `LLAMA_USE_HPX=1`

#### Correctness scenarios
All three scenarios matched after the input-clobber fix:
- short prompt, `-n 32` → match
- long prompt (~202 tokens), `-n 1` → match
- short prompt, `-n 128` → match

The earlier long-prompt `<unk>` divergence was fixed.

#### Performance signals (Release, CPU-only)
Observed trends after the correctness fix:
- **small prompt eval / prefill (2–5 tokens):** HPX still substantially slower (~3–4×), consistent with dispatch/setup overhead dominating very small batches
- **large prefill (~202 tokens):** HPX only about ~1.1× slower, indicating overhead amortizes as batch size grows
- **decode-heavy runs:** HPX decode near parity with baseline

This established the first clean baseline:
- correctness was clean
- decode was roughly at parity
- prefill overhead shrank as batch size grew
- tiny prefill batches remained the obvious weak spot

### Small-batch prefill bypass

A prefill threshold bypass was added:
- `GGML_HPX_PREFILL_MIN_TOKENS = 16`

Behavior:
- small prefill batches below the threshold use the normal scheduler path
- larger prefill batches continue to use HPX prefill

Observed effect:
- short prompt cases improved meaningfully
- long-prompt prefill remained only modestly slower than baseline
- decode remained near parity

### Scattered-node compute viability confirmed

A follow-on experiment tested whether `ggml_backend_graph_compute(...)` requires a contiguous `[node_begin, node_end)` slice from an original graph, or whether it can execute an arbitrary node list.

#### Result
It can execute an arbitrary node list.

A synthetic test graph was built with:
- `x` as input
- `A = neg(x)`
- `B = neg(A)` (dependent, interleaved)
- `C = relu(x)`
- `D = abs(x)`

A custom `ggml_cgraph` was then constructed whose `nodes[]` contained only the scattered subset `{A, C, D}` in topological order. The backend compute call executed `A`, `C`, and `D` correctly while skipping `B`.

#### Important allocator finding
The first version of the scattered-node test failed for a misleading reason:
- `ggml_gallocr` aggressively reuses buffers through in-place aliasing
- in the diagnostic graph, `x->data == D->data` and `A->data == B->data`
- zeroing “outputs” therefore also clobbered inputs/intermediates

For the scattered-node viability test, the allocation strategy was switched to `ggml_backend_alloc_ctx_tensors(...)`, which gives tensors distinct storage for the purposes of the experiment. With aliasing removed, the scattered-node experiment passed.

#### Design implication
- the limiting factor is **not** `ggml_backend_graph_compute(...)`
- the project can support a **scattered-node dispatch primitive** for CPU prefill work
- the real opportunity remains the intra-region parallel groups already identified in prefill CPU regions

### Intra-region parallel projection prototype

A narrow prototype then used the scattered-node capability to parallelize one obvious independent pattern inside a CPU prefill region:
- the FFN `gate` / `up` projection pair

The prototype decomposed a CPU region into:
- `before`
- `chain0`
- `chain1`
- `after`

and ran `chain0` and `chain1` in parallel via HPX while keeping the surrounding work serial.

#### Correctness
Correctness passed:
- baseline
- HPX serial
- HPX parallel prototype

all produced bit-exact outputs across the tested scenarios.

#### Performance
Performance did **not** improve.

A shared-budget check gave:
- serial HPX (4 threads, 1 pool): **410 ms**
- parallel HPX (2 chains × 2 threads): **631 ms**

So the parallel prototype was **1.54× slower** than the serial HPX version.

### Why the prototype lost

The slowdown is structural at this layer.

Each parallel chain used its own `ggml_backend_cpu_init()` backend, and each such backend creates its own independent ggml CPU threadpool. That means:
- the prototype did **not** redistribute the original thread budget across two chains
- instead, it created new threadpools on top of the existing backend/threadpool arrangement
- the original scheduler-owned threads sat idle during the parallel phase
- the new per-chain backends paid their own threadpool/setup/cache costs

This is the key negative result of the current branch:

**Outer HPX orchestration above ggml backends does not compose with ggml’s internal CPU threading model for intra-region parallelism.**

That negative result is what motivated the next architectural pivot: move inside ggml’s CPU executor ownership instead of layering more orchestration above it.

### Phase conclusion

At the end of this phase:
- llama.cpp integration through `ggml-hpx-exec` was correct end-to-end
- decode was near parity on CPU-only TinyLlama runs
- prefill overhead shrank as prompt size grew
- a small-batch prefill bypass helped the obvious tiny-prompt cases
- scattered-node execution was semantically viable
- but outer-parallel intra-region CPU execution was slower, because ggml backend thread ownership sits below the current orchestration layer

---

## 4. Shared executor attachment semantics and internal executor seam

At this point the outer-HPX approach had reached its limit, which led to the next design change: make ggml’s CPU executor ownership explicit and injectable.

### Goal

After separating per-dispatch job state from executor state, the next goal was to make executor sharing safe and to create a narrow internal seam so HPX could replace only the CPU executor substrate without changing kernels or higher-level ggml execution logic.

### Shared executor attachment semantics

The CPU backend originally had only `ggml_backend_cpu_set_threadpool(...)`. That API had a silent side effect: replacing the backend’s threadpool paused the previous threadpool. This was fine for a private backend↔threadpool relationship, but incorrect once the same executor could be shared across multiple backends.

To fix this, the backend context gained a mode bit distinguishing:
- **managed association** via `ggml_backend_cpu_set_threadpool(...)`
- **borrowed/shared association** via `ggml_backend_cpu_attach_threadpool(...)`

Semantics:
- `set_threadpool(...)` preserves legacy behavior and may pause the previously managed executor on replacement
- `attach_threadpool(...)` never pauses on behalf of the current backend and is safe for sharing one executor across multiple backends

### Validation

Added coverage for:
- two backends attached to the same executor, both computing correctly
- alternating dispatches across two backends sharing one executor
- replacing one backend’s executor does **not** pause the shared executor used by the other backend
- legacy managed `set_threadpool(...)` semantics remain unchanged

Both black-box compute tests and white-box pause-state checks passed.

### Executor/job ownership split

With sharing semantics validated, the CPU threadpool implementation was split into:
- long-lived **executor/substrate** state
- per-dispatch **job** state

This created a clean publication model:
- executor owns worker lifetime, wake/sleep policy, and long-lived control state
- job owns graph, plan, barrier state, chunk state, abort, and status

This split is what made a real HPX-backed executor substrate possible.

### Internal executor seam

After the ownership split, an internal executor-ops seam was added so the CPU layer could route only substrate-specific behavior through an internal interface:
- `init`
- `kickoff`
- `worker_wait`
- `destroy`

The first implementation was the existing pthread path, wired through the new seam with no intended behavior change.

### Header cleanup for C/C++

To support an HPX executor implementation from C++, a thin internal header was added with zero atomics/platform-type dependencies:
- `ggml-cpu-executor.h`

This exposes only the executor ops interface and cross-translation-unit function declarations needed by the HPX implementation, while keeping threadpool internals in CPU-only headers.

### Result of this phase

At this point:
- per-dispatch job ownership was separated cleanly from executor ownership
- shared executor attachment was safe and tested
- the CPU layer had a narrow internal executor seam
- pthread remained the first executor implementation
- the codebase was ready for an HPX executor substrate swap without rewriting kernels

---

## 5. CPU-side performance investigation

With the CPU executor seam in place, the next phase was empirical validation: swap in an HPX executor substrate and see what actually happens.

Some interpretations in this section were later revised. They are kept here in chronological order.

### HPX executor substrate integration

An HPX executor implementation was added under the internal CPU executor seam.

New pieces:
- `ggml-hpx-tpool.h`
- `ggml-hpx-tpool.cpp`

HPX runtime registration installs HPX executor ops during `hpx_acquire()` and restores pthread ops at shutdown.

The HPX executor implementation keeps the existing ggml job model intact:
- same `current_job` publication contract
- same barrier/chunk/job logic
- one active dispatch at a time
- no kernel rewrites

### Initial end-to-end validation

TinyLlama end-to-end output matched baseline exactly.

In the default Metal-enabled run:
- correctness passed
- timing was near parity
- CPU buffer remained very small, so this was not a meaningful CPU executor benchmark

This validated correctness but did not yet say much about HPX CPU-side speedup.

---

### Initial CPU-only benchmark campaign

**Goal:** establish a clean baseline for the HPX threadpool substrate.

All runs in this campaign were CPU-only (`GGML_METAL=OFF`). Two matrices were built:
- no-BLAS
- BLAS

Model:
- TinyLlama-1.1B Q4_K_M


#### Correctness
Pass. Generated text was bit-for-bit identical. The only diff was the `llama_context: HPX exec enabled` log line and timing noise.

#### Throughput summary (these are superseeded)

**No-BLAS** (build-base-cpu vs build-hpx-cpu):

| threads | case         | base pp | hpx pp | base tg | hpx tg | tg delta |
|---------|--------------|---------|--------|---------|--------|----------|
| 1       | prefill_long | 83.0    | 82.3   | 52.4    | 48.5   | -7%      |
| 1       | decode_heavy | 85.7    | 84.7   | 44.7    | 44.2   | -1%      |
| 2       | prefill_long | 128.4   | 126.8  | 66.8    | 54.3   | -19%     |
| 2       | decode_heavy | 130.1   | 127.3  | 56.4    | 57.6   | +2%      |
| 4       | prefill_long | 171.4   | 161.5  | 82.4    | 60.1   | -27%     |
| 4       | prefill_mid  | 196.2   | 196.7  | 90.0    | 87.3   | -3%      |
| 4       | decode_heavy | 160.3   | 141.9  | 52.7    | 35.0   | -34%     |
| 4       | decode_light | 161.4   | 160.3  | 59.3    | 74.1   | +25%*    |
| 8       | all cases    | (noisy) | (noisy)| (noisy) | (noisy)| unreliable |

**BLAS** (build-base-cpu-blas vs build-hpx-cpu-blas):

| threads | case         | base pp | hpx pp | base tg | hpx tg | tg delta |
|---------|--------------|---------|--------|---------|--------|----------|
| 1       | prefill_long | 88.1    | 87.6   | 48.7    | 45.5   | -7%      |
| 1       | decode_heavy | 89.2    | 86.7   | 45.2    | 45.1   | 0%       |
| 2       | prefill_long | 137.6   | 135.4  | 70.4    | 68.2   | -3%      |
| 2       | decode_heavy | 130.6   | 133.1  | 58.1    | 56.0   | -4%      |
| 4       | prefill_long | 185.0   | 176.2  | 85.6    | 82.6   | -4%      |
| 4       | prefill_mid  | 202.0   | 203.0  | 84.4    | 85.1   | +1%      |
| 4       | decode_heavy | 170.9   | 154.5  | 70.6    | 59.0   | -16%     |
| 4       | decode_light | 178.4   | 166.9  | 79.6    | 71.4   | -10%     |
| 8       | all cases    | (noisy) | (noisy)| (noisy) | (noisy)| unreliable |

\* No-BLAS decode_light t=4 HPX tg outlier (74.1 vs 59.3) is within the error bar and likely noise.

#### Initial interpretation at this stage

- t=1 and t=2 looked close
- t=4 prefill looked roughly near parity
- t=4 decode looked meaningfully slower on HPX
- BLAS helped both variants similarly

At this stage the working interpretation was:
- HPX threadpool substrate adds negligible overhead at t=1 and t=2
- prefill amortizes overhead
- decode suffers because short per-token graphs do not amortize per-dispatch cost

This interpretation was later corrected.

---

### Correction: the first CPU-only interpretation was wrong

The first CPU-only HPX campaign was later found to be **invalid for parallel HPX claims**.

Cause:
- the nested-HPX guard in `hpx_kickoff` used:
  - `hpx::get_worker_thread_num() != size_t(-1)`
- after `hpx::start`, HPX registers the main OS thread as worker 0
- so the guard fired even on the main thread
- as a result, every kickoff fell back to single-thread execution

This means the earlier CPU-only HPX numbers did **not** measure real parallel HPX execution and should not be used for HPX speedup conclusions.

### Guard fix

The guard was corrected to detect only HPX lightweight threads/coroutines, not OS threads registered with the runtime.

After the fix:
- HPX kickoff began posting real parallel work
- instrumentation showed nonzero `parallel-kickoffs` and `posts`
- the normal inference path was confirmed to use real HPX task posting

### Focused benchmark after the guard fix

With the guard fixed, real HPX parallel execution was active.

Focused decode/prefill benchmarks then showed:
- `pp32`: HPX still significantly slower than base
- `pp512`: HPX much closer to base
- `tg128`: HPX still behind base even with real parallel task posting

Instrumentation showed:
- task posting cost was only a few microseconds per post
- total posting overhead across the decode benchmark was tiny relative to the observed slowdown
- task posting itself explained only a very small fraction of the regression

#### Takeaway
The guard fix changed the question. The remaining decode slowdown could no longer be blamed on “HPX never parallelized.”

---

### Isolation: exec adapter vs threadpool substrate

To isolate the remaining decode regression, a temporary `GGML_HPX_TPOOL_ONLY=1` path was added so the HPX executor substrate could be measured both:
1. with the HPX exec/adapter layer enabled
2. with the exec/adapter layer bypassed, using only the HPX-backed threadpool substrate

#### Results

##### tg128 (decode)
- base: **99 t/s**
- HPX + exec layer: **67 t/s** (**−32%**)
- HPX + tpool only: **65 t/s** (**−35%**)

Posting cost remained small:
- HPX + exec layer: about **3.5 μs/post**
- HPX + tpool only: about **11.5 μs/post**

##### pp512 (prefill)
- base: **276 t/s**
- HPX + exec layer: **260 t/s** (**−6%**)
- HPX + tpool only: **260 t/s** (**−6%**)

#### Interpretation at this stage

This isolation showed that the HPX exec/adapter layer is **not** the dominant bottleneck for decode.

Evidence:
- removing the exec layer did **not** improve decode
- in fact, `tpool-only` was slightly worse than `exec + tpool`
- prefill results were identical between the two HPX modes

The working interpretation after this result was:
- the dominant remaining decode regression is **not** graph adaptation overhead above the executor
- the next plausible culprit is **per-dispatch HPX worker wake/scheduling latency**
- the natural next design step is **persistent HPX worker tasks** that stay alive across dispatches and wait on a signal/semaphore

#### Takeaway
This result redirected the investigation away from adapter caching and toward persistent-worker executor design.

---

### Persistent HPX workers

To reduce decode overhead from post-per-kickoff HPX tasks, the HPX executor was changed to keep persistent worker tasks alive across dispatches.

Each persistent worker waits on a signal/semaphore and runs the next dispatch when woken, instead of creating fresh HPX tasks for every kickoff.

#### Focused results

| test       | base         | post-per-kickoff (Option A) | persistent workers (Option B) |
|------------|--------------|-----------------------------|--------------------------------|
| tg128 t=4  | 14.62 t/s    | ~10 t/s (**−32%**)          | 12.70 t/s (**−13%**)           |
| pp512 t=4  | 42.04 t/s    | ~39.5 t/s (**−6%**)         | 42.89 t/s (**about +2%**)      |

#### Interpretation

Persistent workers materially improved decode performance, recovering roughly 19 percentage points relative to the earlier post-per-kickoff HPX design.

This supported the conclusion that the dominant decode penalty in Option A was not graph adaptation overhead, but per-dispatch worker cold-start behavior and associated synchronization latency.

Prefill remained healthy under the persistent-worker design, indicating that the change improved decode without harming larger work-unit behavior.

#### Takeaway
Persistent HPX workers were the right executor design change for decode-sized dispatches.

---

### Wake-latency follow-up

Because a meaningful but smaller decode gap still remained after switching to persistent workers, the next step was to measure wake latency directly.

Measured values:
- **avg-wake-latency-us:** **1.73 μs**
- **avg-entry-latency-us:** **0.03 μs**
- **avg-total-wake-to-run-us:** **1.76 μs per worker per dispatch**

For `tg128`:
- 3 secondary workers
- 129 dispatches total (128 decode steps + 1 prefill)

Estimated overhead:
- `1.76 μs × 3 × 128 ≈ 676 μs`

That is a tiny fraction of the total run time, so simple semaphore wake latency is **not** large enough to explain the remaining gap by itself.

### Revised interpretation after wake-latency measurement

This measurement weakened the earlier guess that the remaining decode gap was mostly raw wake/sleep latency.

What it now supports instead is:
- persistent workers removed the large task-creation / cold-start penalty
- the remaining gap is much smaller
- the remaining gap is **not** explained by simple wake latency alone
- the residual difference is more likely a combination of smaller effects such as synchronization structure, mutex/barrier behavior, or scheduler/cache effects

---

### Status

- end-to-end llama.cpp integration through the HPX path is correct on the tested TinyLlama scenarios
- the outer “HPX above ggml backends” approach hit a structural limit for intra-region CPU parallelism
- ggml-cpu now has a clean executor/job split and an internal executor seam
- shared executor attachment semantics are implemented and tested
- HPX has a real executor substrate implementation under that seam
- post-per-kickoff HPX workers were not good enough for decode
- persistent HPX workers materially improved decode and preserved healthy prefill behavior

Current performance conclusion:
- the large decode regression from the first HPX executor substrate design has been reduced substantially
- persistent workers recovered most of that loss
- prefill is effectively at parity in the focused persistent-worker result
- the remaining decode gap is small and is **not** explained by simple wake latency alone

What remains open:
- identify the source of the remaining small decode gap
- decide whether that remaining gap is worth another optimization phase
- continue only if the next phase has a clear, measurable target

## 6. Persistent Workers
### ggml-cpu executor pivot
This phase moved the project from outer HPX orchestration around ggml into ggml-cpu executor ownership itself.

Two key changes established that pivot:

- **Executor/job split**  
  Long-lived worker substrate state was separated from per-dispatch mutable job state. This made shared-vs-managed executor attachment semantics explicit and testable, and created the seam needed for multiple executor substrates behind the same ggml-cpu interface.

- **Persistent HPX workers**
  The HPX CPU executor no longer posts fresh HPX work on every dispatch. Instead, it creates long-lived secondary workers once during executor init, keeps them alive across dispatches, wakes them with semaphores at kickoff, and joins them only during destroy. This replaced the earlier per-dispatch task-posting design.

### HPX substrate model
The current HPX executor substrate is a **persistent-worker** design:

- secondary workers are launched once at init
- each worker waits on a semaphore between dispatches
- kickoff signals only the workers needed for the current graph
- workers run `ggml_graph_compute_thread_run(...)` and then return to waiting
- destroy sets stop, signals sleepers, waits for all persistent tasks, and tears down the substrate

This means the remaining cost is **not** due to recreating HPX tasks on every `ggml_graph_compute(...)` call.

### Correctness status
End-to-end llama.cpp integration remains correct on the HPX path. TinyLlama output matched baseline in the previously validated smoke scenarios, and the known correctness issues from earlier phases were already fixed before this baseline sweep.

### Compact sweep used for baseline
A compact matrix sweep was run with:

- **threads:** 1, 2, 4
- **workloads:** `pp32`, `pp512`, `tg128`
- **variants:** base pthread executor vs HPX persistent-worker executor
- **repetitions:** 5 warm + 10 measured

### Results summary
The sweep shows that the remaining HPX gap is **real** and concentrated in **short multithreaded calls**, not in single-thread execution.

#### Short prefill (`pp32`)
- `t=1`: modest loss (~7%)
- `t=2`: clear loss (~23%)
- `t=4`: clear loss (~29%)

#### Large prefill (`pp512`)
- `t=1`: parity / noise
- `t=2`: moderate loss (~8%)
- `t=4`: moderate loss (~5%)

#### Decode-like workload (`tg128`)
- `t=1`: parity / noise
- `t=2`: major loss (~41%)
- `t=4`: major loss (~31%)

### Interpretation
This baseline localizes the remaining problem to **per-dispatch multithreaded executor coordination overhead** in the HPX substrate.

Important takeaways:

- the HPX path is **not** intrinsically slower in single-thread execution
- the large earlier decode penalty was reduced by moving from per-dispatch task posting to persistent workers
- however, the current persistent-worker substrate still does **not** match pthread behavior on short multithreaded calls
- large prefill amortizes most of the remaining overhead, but short decode-like or short prefill calls do not

### Conclusion of this phase
At the end of this phase, the project has established a clean HPX persistent-worker baseline inside ggml-cpu:

- correctness is established
- executor/job split and shared attachment semantics are in place
- HPX persistent workers are implemented and reusable
- the remaining blocker is now narrowly identified as **short-call multithreaded coordination overhead**, not correctness and not per-dispatch task creation


---

## 7. HPX-native bulk-region redesign

### Motivation

The persistent-worker design was correct but pthread-shaped:
the vtable expressed a thread-lifecycle interface (init/kickoff/worker_wait/destroy), which forced persistent sleeping tasks, semaphore-based wakeup,
a nested-HPX coroutine guard, and a serial-decode heuristic — all scaffolding around an atomic spin barrier that required exactly N persistent threads.

The redesign grounds the implementation in two things simultaneously: the actual ggml seam and HPX's own primitives, to produce a design where HPX owns the parallel execution primitive rather than managing thread lifecycle.

### Design

**Vtable** reduced to three ops: `init`, `run_job`, `destroy`.

`run_job(tp, n_threads)` replaces `kickoff` + `worker_wait`. It is called
after `current_job` and `n_active_threads` are set, and returns only when
all workers have completed. The common layer in `ggml_graph_compute` checks
`ops->run_job` first; if non-NULL it calls it and skips the pthread kickoff
path entirely.

**Owned execution context**: `HpxTpoolState` holds a single
`hpx::execution::experimental::scheduler_executor<thread_pool_scheduler>`
constructed once at `hpx_init` with
`hpx::parallel::execution::with_processing_units_count(N)`. The scheduler
is a P2300 sender-based type; wrapping it in `scheduler_executor` exposes
the traditional executor interface used by `for_loop`.

**Dispatch**: `hpx_run_job` calls

```cpp
hpx::experimental::for_loop(
    hpx::execution::par.on(s->exec),
    0, n_threads,
    [=](int j) {
        ggml_graph_compute_thread_run(ggml_threadpool_worker(tp, j));
    });
```

`for_loop` with `par` dispatches all N tasks simultaneously (bulk
semantics) and blocks the caller until all complete. This is why the
ggml spin barrier remains correct: all N workers are live at the same
time, so every barrier cycle sees all `n_active_threads` participants.

### What was removed

- `HpxWorkerSlot`, persistent worker tasks, `hpx::counting_semaphore_var`
- `worker_loop` function
- `hpx_kickoff`, `hpx_worker_wait`
- Nested-HPX coroutine guard (`hpx::threads::get_self_ptr()` check)
- Serial-decode heuristic (`GGML_HPX_SERIAL_DECODE`)
- `GGML_HPX_STATS` instrumentation
- `ggml_threadpool_set_active_threads`, `ggml_threadpool_last_node_ne1`
  helper functions (were only used by `hpx_kickoff`)

`ggml-hpx-tpool.cpp` went from 357 lines to ~100.

### Correctness result

All 10 `test_hpx_exec` cases pass with the new substrate on the first
build attempt. The three files changed are:

- `ggml/src/ggml-cpu/ggml-cpu-executor.h` — vtable
- `ggml/src/ggml-cpu/ggml-cpu.c` — dispatch path + helper removals
- `ggml/src/ggml-hpx/ggml-hpx-tpool.cpp` — complete rewrite

### Future path

Performance benchmarks against the pthread baseline are the next step
(CPU-only, `-t 4`, short decode and 128-token prefill).

For richer HPX composition (dataflow dependencies, async prefill chains),
the `scheduler_executor` already supports the sender/receiver path:
`bulk(schedule(exec.sched()), N, fn)` + `sync_wait` can replace
`for_loop.on(exec)` without touching the seam.

### Initial bulk-region result

After the HPX-native bulk-region redesign landed, the new substrate was structurally correct and passed all `test_hpx_exec` cases. The common executor seam had been reduced to:

- `init`
- `run_job`
- `destroy`

and the HPX backend owned a reusable `scheduler_executor<thread_pool_scheduler>` created once at `hpx_init`. Dispatch was implemented as one blocking bulk call over logical worker ids using:

```cpp
hpx::experimental::for_loop(
    hpx::execution::par.on(s->exec),
    0, n_threads,
    [=](int j) {
        ggml_graph_compute_thread_run(ggml_threadpool_worker(tp, j));
    });
```

This preserved the intended HPX-native shape:
- no persistent worker slots
- no semaphore wakeup path
- no coroutine guard
- no serial-decode heuristic
- no thread-lifecycle-shaped executor contract

### Benchmark result: the redesign fixed the shape, but exposed a new single-thread issue

The first compact matrix sweep against the stock pthread baseline showed a mixed result:

- large prefill (`pp512`) remained near parity
- multithreaded short work improved relative to the earlier persistent-worker substrate in some cases
- but `tg128` at `t=1` regressed badly

This regression turned out not to be a correctness issue and not a multithreaded coordination problem. The cause was simpler:

- persistent-worker had effectively treated `n_threads <= 1` as a no-op on the HPX side, so worker 0 ran inline with no HPX dispatch overhead
- the new bulk-region `run_job` path always routed even `n_threads == 1` through HPX parallel-algorithm dispatch
- for decode-like workloads (`tg128`), that meant paying HPX dispatch overhead once per token

So the first bulk-region result showed that the redesign was directionally right for multithreaded work, but that the new seam had accidentally lost the old zero-overhead single-thread path.

### Fix: direct fast path for `n_threads <= 1`

The fix was intentionally small and local:

- keep the bulk-region architecture unchanged for `n_threads > 1`
- add an early return in `hpx_run_job(...)` for `n_threads <= 1`
- run worker 0 directly in that case

Conceptually:

```cpp
if (n_threads <= 1) {
    ggml_graph_compute_thread_run(ggml_threadpool_worker(tp, 0));
    return;
}
```
This restored the expected single-thread behavior without reintroducing:
- persistent workers
- semaphores
- coroutine guards
- serial heuristics
- extra executor hooks

### Three-way comparison: pthread vs persistent-worker vs bulk-region

After the `n_threads <= 1` fast path was added, the meaningful comparison became:

1. stock pthread baseline
2. HPX persistent-worker substrate
3. HPX bulk-region substrate

The resulting pattern was clear.

#### Single-thread (`t=1`)
Bulk-region returned to near parity with pthread:

- `pp32`: ~97% of pthread
- `pp512`: ~101% of pthread
- `tg128`: ~98% of pthread

This fixed the accidental single-thread decode regression completely enough for practical purposes.

#### Large prefill (`pp512`)
Bulk-region stayed healthy across all thread counts:

- `t=1`: ~101% of pthread
- `t=2`: ~99% of pthread
- `t=4`: ~97% of pthread

This is the strongest positive signal in the whole matrix. When compute dominates, the HPX bulk-region substrate is close to free.

#### Short multithreaded work (`pp32`, `tg128`)
The remaining gap is concentrated here:

- `pp32 t=2`: ~84% of pthread
- `pp32 t=4`: ~78% of pthread
- `tg128 t=2`: ~62% of pthread
- `tg128 t=4`: ~79% of pthread

So the redesign narrowed the problem substantially:
- single-thread is fixed
- large workloads are healthy
- the remaining weakness is short multithreaded dispatch

#### Comparison to persistent-worker
Bulk-region was consistently as good as or better than persistent-worker in the measured matrix. The biggest visible improvement was restoring single-thread decode to parity; large prefill also remained healthy, while short multithreaded decode still remained the main open problem.

### Experiment attempted and reverted: sender/receiver bulk path

A follow-up experiment temporarily replaced `for_loop(par.on(exec), ...)` with the P2300 sender path:

    bulk(schedule(exec.sched()), N, fn) + sync_wait(...)

The intent was to test whether the sender/receiver bulk form had lower overhead for short dispatches.

Benchmark result:
- it was slightly worse on the critical short multithreaded case
- for example, `tg128 t=2` dropped slightly relative to the `for_loop.on(exec)` version

That experiment was therefore reverted. The current branch remains on:
- `for_loop(par.on(exec), ...)` for `n_threads > 1`
- direct worker-0 execution for `n_threads <= 1`

### Interpretation

At this point the project has a cleaner and more stable result than either earlier phase:

- the HPX substrate is now HPX-native rather than pthread-shaped
- the single-thread decode regression introduced by the first bulk-region version was fixed
- large prefill workloads are near pthread parity
- bulk-region is consistently better than or equal to the persistent-worker substrate
- the remaining performance problem is specifically short multithreaded dispatch overhead, especially decode-like work

This means the architectural redesign itself is no longer the question. The open question is narrower:

> can the remaining short multithreaded dispatch cost be reduced enough to make decode-like work competitive with pthread?

### Near-no-op dispatch microbenchmark

The next step is a standalone near-no-op dispatch microbenchmark in `hpx-bench/` to measure raw `run_job(tp, n_threads)` overhead at `n_threads = 2, 4`, comparing:

- stock pthread dispatch
- current HPX bulk-region dispatch

The purpose of that benchmark is to isolate substrate overhead from ggml compute and determine whether the remaining `tg128` / `pp32` gap is primarily:

- raw dispatch overhead, or
- deeper interaction with ggml barrier/worker behavior

### Tail localization

To understand the remaining gap after the bulk-region redesign and the `n_threads <= 1` fast path, a standalone microbenchmark was added under `hpx-bench/` to measure raw `run_job(tp, n_threads)` overhead without ggml compute dominating the result.

The benchmark compared:

- stock pthread-style dispatch
- current HPX bulk-region dispatch using `for_loop(par.on(exec), ...)`

The first version used batched measurements and showed a heavy right tail on the HPX side:
- median overhead was only moderately worse than pthread
- but tail latency was much worse

This suggested that the remaining decode loss was not primarily caused by the median dispatch cost, but by rare expensive dispatches.

### Tail breakdown: where the delay actually occurs

The microbenchmark was then extended with a breakdown mode. For each dispatch, it recorded:

- `t_submit`: immediately before dispatch
- `t_enter[j]`: immediately when worker `j` entered the body
- `t_done`: immediately after dispatch returned

From those timestamps, it derived:

- `submit→first-start`
- `submit→last-start`
- `spread = last-start - first-start`
- total dispatch time

This revealed the key fact:

- the tail was almost entirely in `submit→first-start`
- the `spread` between first and last worker start was tiny

So the remaining problem was **not** gang formation, straggler arrival, or barrier spread.  
The problem was that the first HPX worker sometimes started late, while the rest of the workers followed almost immediately once pickup began.

### Restricted-thread executor probe

A comparison was made against `restricted_thread_pool_executor` to test whether explicit fixed OS-thread placement would eliminate the tail.

It did not reliably do so.

The executor changed the shape of the delay distribution somewhat, but did not remove the long startup stalls. In some cases it traded the original “first worker pickup delay” pattern for a wider first-to-last spread.

This ruled out a simple conclusion that runtime-managed placement alone was the root cause.

### Breakdown repetition count increased to 10,000

The first breakdown runs used too few repetitions to characterize the rare tail events reliably.

At 2,000 dispatches:
- p99 was under-sampled
- counts of extreme tail events varied too much between runs

The benchmark was therefore rerun with **10,000 breakdown repetitions**.

With 10K reps, the shape became much clearer and more stable:
- roughly 90–95% of dispatches landed in the fast bucket
- a small fraction landed in an intermediate 20 µs–1 ms region
- about ~1% landed in a >=1 ms tail bucket

This pattern appeared for both the scheduler-based HPX executor path and the restricted-thread executor probe. The conclusion from this phase was:

- the tail was real
- it was rare
- it was large enough to matter for decode-like workloads
- it looked more like startup/pickup delay than steady-state gang overhead

### Scheduler-policy sweep on the default HPX pool

Because the delay was concentrated in worker pickup rather than barrier spread, the next experiment was a scheduler-policy sweep with no structural code change.

The benchmark binary was updated so HPX runtime flags were forwarded correctly through `hpx::start(...)`, and a scheduler label was printed in both the header and breakdown rows. A sweep script was added to run multiple launches per scheduler and collect all output into a single timestamped result file.

The schedulers tested on the default HPX pool were:

- `local`
- `static`
- `local-priority`
- `static-priority`
- `shared-priority`

All five schedulers ran successfully at `--hpx:threads=4`. Earlier probe runs showed some scheduler instability at `--hpx:threads=2`, but the sweep itself completed.

### Scheduler sweep result

Scheduler choice had a clear effect on dispatch throughput.

At `t=2` for the near-no-op HPX dispatch benchmark:
- `static` was the clear winner
- `local-priority` was the slowest
- `shared-priority` and `local` landed in the middle
- `static-priority` was close to `static`, but not better

The most important outcome was that **`static` materially reduced total dispatch overhead compared with the previous default-like scheduler choice**.

This changed the interpretation in an important way:

- the earlier HPX loss was not an unavoidable fixed tax
- scheduler policy mattered significantly
- some of the remaining loss could be reduced without changing the seam or switching executor types

### End-to-end throughput with static scheduler

After the sweep identified `static` as the best candidate, the real throughput matrix was rerun against the stock pthread baseline for the important cases at `t=2` and `t=4`.

Results showed:

#### Prefill improved substantially
- `pp32` moved much closer to pthread
- `pp512` became very close to parity

Representative behavior:
- `pp512` was within about 1–4% of pthread
- `pp32` was within a few percent and even slightly positive in one noisy `t=4` run

This was a strong sign that the scheduler change addressed a meaningful part of the earlier HPX overhead.

#### Decode remained the persistent loser
The remaining weak point was `tg128`, the decode-heavy case:
- still clearly behind pthread at both `t=2` and `t=4`

The interpretation here is straightforward:
- decode-like execution consists of many short dispatches
- even with the better `static` scheduler, HPX still pays a higher per-dispatch cost than pthread
- that cost accumulates significantly across repeated short decode steps

### Interpretation after scheduler sweep

At this point the picture is much cleaner than earlier in the project:

- the HPX substrate is structurally HPX-native rather than pthread-shaped
- the accidental single-thread regression was fixed by the `n_threads <= 1` fast path
- scheduler policy was shown to matter, and `static` is clearly a better fit than `local-priority` for this workload
- large prefill is now near parity with pthread
- short prefill is much improved
- the main remaining problem is decode-heavy short dispatches

This means the project is no longer blocked on correctness, basic architecture, or general prefill performance.

The open question has narrowed to:

> can decode-like short dispatches be made competitive, or should they use a different execution policy than the current HPX bulk-region path?

### Next direction

The current evidence suggests that the remaining decode loss is tied to **dispatch granularity** more than to the basic executor seam.

Two plausible next directions are:

1. reduce the number of HPX dispatches on decode-like work, or
2. add a decode-specific cutoff/fallback so very small decode-sized calls avoid HPX dispatch entirely

The bulk-region design, plus the `static` scheduler, appears sufficient for large prefill and good enough for most non-decode-heavy work. Decode remains the one workload where the current HPX dispatch granularity is still not competitive with pthread.

### Work-size probe: finding a real decode/prefill discriminator

After scheduler-policy tuning, the remaining open question was whether decode-like calls could be separated from prefill-like calls using a simple runtime signal available inside the HPX substrate.

The first probe used graph node count (`cgraph->n_nodes`) as a candidate discriminator. That failed completely.

Observed result:
- `pp32`: 100% in `>=256` nodes
- `pp512`: 100% in `>=256` nodes
- `tg128` decode: 100% in `>=256` nodes
- single-token prefill (`p=1`): 100% in `>=256` nodes

This showed that TinyLlama’s graph topology is effectively invariant across decode and prefill. The number of nodes does not change meaningfully with batch size; only tensor shapes and work per node change.

That negative result was important because it ruled out graph topology as the gating signal.

### Work-size histogram: clean separation

The next probe used `cplan->work_size` instead.

Unlike node count, `work_size` separated the regimes cleanly:

- `tg128` decode: `4K–64K`
- single-token prefill (`p=1`): `4K–64K`
- `pp32`: `64K–1M`
- `pp512`: `>=16M`

This was the first clean decode/prefill discriminator found inside the current ggml/HPX seam.

Interpretation:
- the important distinction is not graph structure
- it is dispatch size / work per dispatch
- decode and large prefill use essentially the same graph topology, but radically different tensor sizes and scratch requirements

This matched the performance results much better than node count did:
- decode-like work remains the main HPX loser
- short prefill is much closer to pthread
- large prefill is near parity

### Consequence of the work-size result

The work-size probe changed the understanding of the remaining problem.

Before this probe, the main open question was whether decode-sized work could be identified at all from within the HPX substrate.

After this probe, the picture became much clearer:
- decode-sized dispatches can be identified
- the classifier is `work_size`, not `n_nodes`
- the likely policy boundary lies somewhere below the `pp32` regime and above the decode regime

A threshold around `64K` became the first plausible candidate:
- `<64K` catches decode-like dispatches and single-token prefill
- `>=64K` keeps short and large prefill on the HPX path

### Serial-fallback experiment and barrier hang

A first attempt was made to test a simple fallback policy for `work_size < 64K` by bypassing HPX and running only worker 0 directly.

That hung immediately.

Root cause:
- the caller had already set `current_job->n_active_threads = n_threads`
- the fallback executed only worker 0
- ggml’s spin barrier still expected all `n_active_threads` participants
- worker 0 blocked forever waiting for workers that were never launched

This clarified an important constraint of the current design:

A local “small-work bypass” is not safe unless the active participant count is also changed to match the fallback execution mode.

In other words:
- the fallback is not just “run worker 0 directly”
- it is “run worker 0 directly **and** override the barrier participant count to 1”

That is a useful design constraint for any future policy split.

### What the work-size result means strategically

The project is now past the phase of guessing at decode/prefill signals.

The current evidence supports the following conclusions:

- graph topology is not the right classifier
- work size is the right classifier
- the remaining HPX decode problem is tied to dispatch granularity
- any policy split must respect ggml’s barrier contract

This also sharpened the strategic options:

1. use `work_size` as a classifier for policy selection
2. keep HPX for prefill-like work
3. treat decode-like work as a separate policy problem

What remains unresolved is **which** alternate policy should be used for the decode-like regime:
- serial fallback for the smallest cases
- pthread fallback for small multithreaded calls
- or a deeper HPX-specific pool/placement redesign

### Status after all probes

At this point the project has established all of the following:

- the HPX substrate has been redesigned into an HPX-native bulk-region execution path
- the accidental single-thread regression was fixed by the `n_threads <= 1` fast path
- scheduler policy matters, and `static` is a much better fit than the earlier default-like scheduler for this workload
- large prefill is near parity with pthread
- short prefill is much closer to pthread than earlier HPX versions
- decode-heavy short dispatches remain the main loser
- node count does not distinguish decode from prefill
- `work_size` does distinguish decode from prefill cleanly
- barrier-safe fallback requires changing the active participant count as well as the execution path

### Updated interpretation

The project is no longer blocked on correctness or on finding the right substrate seam.

The open problem has narrowed to this:

> Given that `work_size` cleanly separates decode-like from prefill-like dispatches, what execution policy should the small-work regime use?

That is the current frontier of the project.

The evidence now points toward a policy split based on dispatch size, not graph topology. Whether that split should be implemented as a local serial fallback, a pthread fallback, or a deeper HPX pool/placement redesign remains the next design decision.

### Work-size probe: finding a real decode/prefill discriminator

After scheduler-policy tuning, the remaining open question was whether decode-like calls could be separated from prefill-like calls using a simple runtime signal available inside the current ggml/HPX seam.

The first probe used graph node count (`cgraph->n_nodes`) as a candidate discriminator. That failed completely.

Observed result:
- `pp32`: 100% in `>=256` nodes
- `pp512`: 100% in `>=256` nodes
- `tg128` decode: 100% in `>=256` nodes
- single-token prefill (`p=1`): 100% in `>=256` nodes

This showed that TinyLlama’s graph topology is effectively invariant across decode and prefill. The graph structure stays the same; only tensor shapes and work per node change.

That negative result ruled out graph topology as the gating signal.

### Work-size histogram: clean separation

The next probe used `cplan->work_size` instead.

Unlike node count, `work_size` separated the regimes cleanly:

- `tg128` decode: `4K–64K`
- single-token prefill (`p=1`): `4K–64K`
- `pp32`: `64K–1M`
- `pp512`: `>=16M`

This was the first clean decode/prefill discriminator found inside the current substrate seam.

Interpretation:
- the important distinction is not graph structure
- it is dispatch size / work per dispatch
- decode and prefill use essentially the same graph topology, but radically different tensor sizes and scratch requirements

This matched the performance story much better than node count did:
- decode-like work remains the main HPX loser
- short prefill is much closer to pthread
- large prefill is near parity

### Serial fallback probe inside the HPX path

A first attempt was made to use `work_size` as a policy signal inside `hpx_run_job(...)`.

The idea was:
- small work (`work_size < 64K`) would bypass HPX bulk dispatch
- the fallback would run worker 0 directly
- larger work would continue to use the HPX `for_loop(par.on(exec), ...)` path

The first version hung immediately.

Root cause:
- the caller had already set `current_job->n_active_threads = n_threads`
- the fallback executed only worker 0
- ggml’s spin barrier still expected all `n_active_threads` participants
- worker 0 blocked forever waiting for workers that were never launched

This clarified an important constraint of the current design:

A local serial fallback is not safe unless the active barrier participant count is also overridden to 1.

After fixing that, the serial fallback path did run correctly.

### Critical result: dispatch bypass did not restore decode performance

Even after the serial branch was confirmed to fire for decode-sized work, decode throughput remained far below the stock pthread baseline.

This was the key result of the probe.

Interpretation:
- the remaining decode loss is **not** explained only by the HPX `for_loop` dispatch overhead
- bypassing `for_loop` inside the HPX path is not enough
- the HPX runtime’s resident worker threads still exist in the background and compete for CPU time with the serial decode work

In other words:

> decode-sized calls do badly not just because of HPX dispatch, but because they are still running inside an HPX-owned runtime regime with active worker threads on the machine

This sharply changed the understanding of the remaining decode problem.

### Consequence: the real split is HPX vs pthread, not HPX vs serial branch

Before this probe, it was plausible that a simple branch inside `hpx_run_job(...)` could solve decode:
- small work → direct worker-0 path
- large work → HPX bulk dispatch

After this probe, that no longer looked sufficient.

The result implied that the meaningful boundary is not:

- HPX `for_loop` vs local serial fallback inside one HPX threadpool

but rather:

- true **pthread-owned execution** for decode-sized work
- true **HPX-owned execution** for prefill-sized work

### Proposed hybrid-threadpool design and why it was questioned

A follow-on design was then considered to remove the global executor switch and make threadpool creation explicit.

The good part of that plan was:
- stop relying on a global `g_executor_ops`
- add `ggml_threadpool_new_with_ops(...)`
- make HPX threadpool creation explicit via `ggml_hpx_tpool_create(...)`
- allow different contexts to coexist with different executor substrates

That infrastructure direction is sound.

However, the first routing idea proposed a **single hybrid threadpool object**:
- spawn pthread workers in `hpx_init`
- also create an HPX executor in the same object
- route by `work_size`
  - `<64K` → pthread kickoff-and-wait
  - `>=64K` → HPX `for_loop`

This was questioned for an important reason:

The serial-fallback experiment had already shown that decode loses not just because of HPX dispatch overhead, but because the HPX runtime’s resident worker threads continue to interfere with small-work execution.

So a single hybrid threadpool object containing:
- live pthread workers
- and live HPX workers

would likely preserve the same basic interference problem during the decode-sized path.

That means a one-object hybrid does not obviously solve the actual issue the probe uncovered.

### Updated interpretation

At this point the project has established all of the following:

- the HPX substrate has been redesigned into an HPX-native bulk-region execution path
- the accidental single-thread regression was fixed by the `n_threads <= 1` fast path
- scheduler policy matters, and `static` is a much better fit than the earlier default-like scheduler
- large prefill is near parity with pthread
- short prefill is much improved
- decode-heavy short dispatches remain the main loser
- graph node count does not distinguish decode from prefill
- `work_size` does distinguish decode from prefill cleanly
- a local serial fallback inside the HPX runtime does not restore decode performance
- the remaining decode loss is therefore not just `for_loop` dispatch overhead
- the likely fix is a **real substrate split**, not a local branch inside one HPX-owned path

### Current strategic direction

The project is now past the phase of guessing at workload signals.

The current evidence points toward this architecture:

- remove HPX’s dependence on the global executor switch
- make threadpool creation explicit
- keep a real pthread substrate available
- keep a real HPX substrate available
- choose between them per dispatch using `work_size`

The open design question is no longer whether decode and prefill can be separated.

That part is now answered.

The open question is:

> should the project implement a true work_size-based pthread/HPX split using two independent substrates, or go deeper into HPX pool/placement redesign first?

At this stage, the evidence favors the true substrate split.

### Explicit per-threadpool executor creation

At this stage, the global executor switch (`g_executor_ops`) had become the wrong abstraction.

Originally, HPX activation worked by replacing the global ggml CPU executor ops table, which meant:
- new threadpools created after HPX activation would use HPX ops
- the default path and the HPX path were still coupled through global process state

To remove that coupling, the threadpool creation path was refactored so executor ops could be passed explicitly.

Key change:
- `ggml_threadpool_new_with_ops(...)` was added as an explicit creation entry point
- the old global `ggml_cpu_set_executor_ops(...)` path was left in place only for the default/legacy route
- HPX threadpool creation no longer depends on mutating global executor state

This made executor selection per-threadpool instead of effectively process-wide.

### Explicit HPX threadpool creation

With the explicit-ops refactor in place, the HPX path gained an explicit creation function:
- `ggml_hpx_tpool_create(...)`

This function:
- starts HPX if needed
- creates a threadpool with HPX executor ops directly
- avoids touching the global default executor path

At the llama.cpp integration point, the HPX path now creates its own threadpool explicitly and stores it on the context. This means:
- the context can hold an HPX-backed threadpool independently of the default pthread one
- destruction is explicit and context-local
- HPX activation no longer globally redirects unrelated threadpool creation

This was an architectural cleanup, not just a performance change.

### First real substrate split

Once `work_size` had been validated as the correct decode/prefill discriminator, the next question was how to route work based on it.

A first attempt tried to do this *inside* the HPX threadpool by using a local serial fallback for `work_size < threshold`. That experiment showed that bypassing `for_loop(...)` alone was not enough; decode still lost because the HPX runtime’s resident worker threads were still alive and competing for CPU time.

That result changed the meaning of the split:

- the real boundary is not “HPX dispatch vs serial branch inside one HPX-owned threadpool”
- the real boundary is “true pthread-owned execution vs true HPX-owned execution”

The project then moved to a genuine substrate split.

### Routing by `work_size`

The llama.cpp dispatch path was updated so it computes a fresh `ggml_cplan` and uses `cplan->work_size` to decide which threadpool should own the dispatch.

The policy became:

- small work (`work_size < threshold`) → pthread threadpool
- larger work (`work_size >= threshold`) → HPX threadpool

This is the first point where the project stopped trying to make one substrate fit all workloads and instead used an explicit measured signal to route work to the more appropriate execution regime.

### Routing validation

The routing was then checked directly.

Observed behavior:
- decode-like work (`tg16` / `tg128`) never touched the HPX threadpool
- large prefill (`pp512`) consistently routed to the HPX threadpool

This confirmed that the split was not just configured in code; it was actually selecting the intended substrate at runtime.

### End-to-end result of the split

After the substrate split landed, the key benchmark matrix was rerun.

Measured result:

| case  | t | base | hpx | delta |
|------|---:|-----:|----:|------:|
| tg128 | 2 | 87.2 | 83.1 | -5% |
| tg128 | 4 | 97.0 | 96.2 | -1% |
| pp32  | 2 | 170.7 | 162.4 | -5% |
| pp32  | 4 | 236.7 | 233.3 | -1% |
| pp512 | 2 | 155.5 | 156.7 | +1% |
| pp512 | 4 | 210.9 | 214.3 | +2% |

Interpretation:
- `tg128` returned to near pthread baseline
- `pp32` stayed near parity
- `pp512` stayed at parity or a slight win

This was the first result where the architecture and the measurements fully lined up:

- decode-sized work no longer paid the HPX penalty
- large prefill still benefited from the HPX path being available
- no meaningful regression remained in the measured matrix

In practical terms, the split achieved the intended goal:
- protect decode performance
- preserve HPX competitiveness for prefill-sized work

### Threshold sweep and final default

A threshold sweep was run for:
- `32K`
- `64K`
- `128K`

Two important observations came out of the sweep:

1. Absolute throughput across sections was affected by thermal throttling, so the trustworthy comparison was the **within-section HPX/base ratio**, not raw tokens/s across the entire run.
2. Routing was identical across all three thresholds:
   - decode-sized work always stayed off HPX
   - `pp32` and `pp512` always stayed on HPX

Within-section ratios:

| threshold | case×t | HPX/base |
|----------|--------|----------|
| 32K  | tg128×t2 | 80.8 / 85.6 = 94% |
| 32K  | tg128×t4 | 78.4 / 83.9 = 93% |
| 32K  | pp32×t2  | 161.5 / 159.7 = 101% |
| 32K  | pp32×t4  | 208.4 / 202.7 = 103% |
| 32K  | pp512×t2 | 150.3 / 151.2 = 99% |
| 32K  | pp512×t4 | 184.8 / 187.2 = 99% |
| 64K  | tg128×t2 | 63.1 / 63.9 = 99% |
| 64K  | tg128×t4 | 71.6 / 73.0 = 98% |
| 64K  | pp32×t2  | 135.0 / 138.5 = 97% |
| 64K  | pp32×t4  | 204.8 / 199.1 = 103% |
| 64K  | pp512×t2 | 131.9 / 128.7 = 102% |
| 128K | tg128×t2 | 59.9 / 60.9 = 98% |
| 128K | tg128×t4 | 70.6 / 73.3 = 96% |
| 128K | pp32×t2  | 139.6 / 138.2 = 101% |
| 128K | pp32×t4  | 199.9 / 186.9 = 107% |
| 128K | pp512×t2 | 129.1 / 130.4 = 99% |

Because routing did not change across the tested thresholds, the smallest working threshold became the right default.

Final recommendation:
- set `GGML_HPX_WORK_THRESHOLD` default to **32K** (`32768` bytes)

Why `32K`:
- it is the smallest threshold that still routes correctly
- it keeps as much work as possible on the HPX path without leaking decode-sized work into HPX
- it leaves headroom for models whose decode-sized work is larger than TinyLlama’s while still staying conservative


# 8. HPX-native fine-region DAG redesign

## Motivation

The bulk-region executor redesign improved the HPX substrate, but still relied on ggml’s worker-loop model:

- launch worker j
- enter ggml_graph_compute_thread_run(...)
- synchronize via barriers

This meant HPX controlled *threads*, but not the **actual work units**.

The next question was:

> Can HPX execute CPU kernel work directly, without re-entering ggml’s worker loop?

---

## Coarse vs fine regions

Existing:
- `ggml_hpx_region` → coarse graph slice

New:
- `ggml_hpx_cpu_region` → fine work range inside one op
- `ggml_hpx_cpu_region_group` → DAG of fine regions

Hierarchy:

graph → coarse region → fine CPU region → run_range

---

## New contract (region DAG)

Defined in `ggml-hpx-region-dag.h`:

- `ggml_hpx_cpu_region_kind`
- `ggml_hpx_dep_edge { src, dst }`
- `ggml_hpx_region_resources`
- `ggml_hpx_run_range_fn`
- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`

Key idea:

> run_range is a **direct kernel entry**, no graph execution allowed.

---

## First executor layer

Implemented in:

- `ggml-hpx-region-exec.h`
- `ggml-hpx-region-exec.cpp`

Provides:

### Validation
`ggml_hpx_validate_region_group(...)`  
Checks structure and optional acyclicity.

### Direct kernel
`ggml_hpx_mul_mat_f32_run_range(...)`  
Calls `ggml_vec_dot_f32` directly (no ggml graph).

### Execution
- `run_single_region(...)`
- `run_region_group(...)`

Limitation:
- same-level regions executed serially (executor safety)

---

## Microbench: direct run_range

First standalone benchmark:

- no ggml graph execution
- direct kernel timing

Result:
- contract works
- large work ≈ parity
- small work still sensitive

---

## Region-chain benchmark

Adds:
- dependencies
- shared resources

Result:
- correctness matches
- overhead accumulates with scheduler executor

Conclusion:
- contract is correct
- executor choice is critical

---

## Executor comparison

### Bench 1 — matmul dispatch

| shape   | nth | scheduler_exec (prev) | fork_join_exec (now) |
|---------|-----|------------------------|----------------------|
| decode  | 1   | ~990 µs                | 993 µs (≈ same)      |
| decode  | 2   | ~880 µs                | 894 µs (≈ same)      |
| decode  | 4   | ~660 µs                | 642 µs (slightly faster) |
| prefill | 1   | ~31 ms                 | 31.2 ms (≈ same)     |
| prefill | 2   | ~28 ms                 | 27.8 ms (≈ same)     |
| prefill | 4   | ~21 ms                 | 20.5 ms (slightly faster) |

### Bench 2 — 3-region chain

| shape   | nth | pool      | fork_join_exec | speedup |
|---------|-----|-----------|----------------|---------|
| decode  | 4   | 1145 µs   | 643 µs         | 1.78×   |
| prefill | 4   | 23.4 ms   | 21.9 ms        | 1.07×   |

---

## Interpretation

- scheduler_executor + for_loop was wrong for this workload
- fork_join_executor matches the structure
- first real speedup appears on decode-scale chains

---

## Design consequence

HPX should:

- not just orchestrate regions
- not just replace threadpool
- but execute **explicit fine CPU work units**

---

## Limitations

- same-level regions still serialized
- run_single_region not optimized
- only matmul kernel implemented
- no full llama.cpp integration yet

---

## Conclusion

This phase moves HPX from:

- outer orchestration
- → executor substrate
- → direct CPU execution model

This is the first phase where HPX executes real work units, not just threads.


### Phase 1 — bridge to same-level fine-region concurrency

After the fine-region DAG contract, validator, and direct `run_range(...)` kernel entry were in place, the system still executed same-level regions **serially**.

A new test (`test_hpx_region_group_parallel`) exposed this:
- independent regions at the same level did not overlap
- the DAG existed structurally, but not behaviorally

### Key issue
Early attempts using `fork_join_executor` failed because:
- shared executor → unsafe concurrent use
- per-region pinned executors → PU contention → serialization

### Phase 1 design

Introduce a single HPX boundary:

```cpp
hpx::future<void> launch_region_async(...)
```

Changes:
- same-level regions launched via `hpx::async`
- intra-region fan-out via per-lane async
- level loop retained as temporary scaffolding

### Result
- real overlap achieved
- all existing tests still pass
- no graph re-entry introduced

---

### Phase 2 — HPX-native inter-region scheduling

Phase 1 still relied on a **manual level loop**.

Phase 2 removes that.

### Core idea
A region becomes runnable when its **predecessor futures are ready**.

### Implementation

- build predecessor lists
- topological sort (Kahn)
- one `shared_future` per region
- use `hpx::dataflow` for dependent regions

```cpp
region_fut[i] =
    dataflow(pred_futs..., launch_region_async(...))
```

### Result
- no explicit level loop
- true dependency-driven execution
- same correctness guarantees preserved

### Outcome

Fine-region execution is now:

- inter-region: HPX futures / dataflow
- intra-region: direct run_range execution

This is the first fully HPX-native execution model for the region DAG.
