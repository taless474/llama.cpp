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


## 8. HPX-native fine-region DAG redesign

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



### Context entering Phase 3

By the end of Phase 2, the project had already moved to the new fine-region DAG model:

- explicit `ggml_hpx_cpu_region` / `ggml_hpx_cpu_region_group`
- explicit dependencies via `ggml_hpx_dep_edge`
- direct `run_range` execution, not graph re-entry
- dependency-driven scheduling in `ggml_hpx_run_region_group(...)`
  using:
  - Kahn topo sort
  - one `shared_future` per region
  - `hpx::dataflow(...)` for dependent launch

So scheduling was already HPX-native and direct.

What was still *not* proven was the **resource model**:
- `lane_scratch`
- `reduction_buffer`
- `shared_scratch`

The key question became:

> Can the executor carry the right execution resources for real fine-region work, or is the region DAG still only a scheduling shell?

---

### Phase 3 — resource-model proof

#### Goal

Prove that `ggml_hpx_region_resources` is real and usable at the API boundary, without changing the scheduler and without integrating into the full llama / ggml path.

This phase was intentionally scoped as:

- no executor redesign
- no scheduler changes
- no integration yet
- no performance work
- prove resource plumbing by tests first

#### Why this phase happened

At this point, the DAG runner already existed and the fine-region contract was already defined, but the resource channels had only been designed, not justified by passing tests.

The immediate need was not “more scheduling,” but **evidence** that the new execution contract could safely carry shared and per-lane state.

### Work done

Two new tests were added in the HPX region-DAG test suite:

#### 1. `test_hpx_region_group_reduction`
A two-region DAG:

- `R0` (`REDUCTION`) computes a scalar reduction and writes it into
  `resources->reduction_buffer`
- `R1` depends on `R0` and reads that reduction result to produce output

This proved:
- `reduction_buffer` is visible across dependent regions
- dependency ordering is honored
- reduction remains single-threaded under the current runner contract

#### 2. `test_hpx_region_group_lane_scratch`
A lane-parallel region with `n_lanes = 4`.

Each callback invocation uses:
- `resources->lane_scratch[ith]`

This proved:
- per-lane scratch is threaded correctly to callbacks
- lane identity is stable at the API boundary
- each lane receives its own mutable slot for the dispatch

### Important design choice in Phase 3

These tests were written as **contract tests**, not as scheduler redesign tests.

That meant:
- no new runner logic
- no graph-path re-entry
- no broad new kernels
- only proving that the existing runner already satisfied the resource contract

### Outcome

All region-DAG tests passed, including the two new Phase 3 tests.

That established:

- the current dependency-driven runner already supports the resource model needed by the fine-region contract
- no code changes to the scheduler or executor were required
- `reduction_buffer` and `lane_scratch` are valid execution resources, not just unused fields in a struct

### Phase 3 conclusion

Phase 3 completed the **proof-of-contract** milestone:

> The fine-region DAG executor can carry real shared and per-lane execution resources correctly.

What it did **not** prove yet was that the DAG could host a real ggml-style op.

That led to the next question:

> Is the fine-region DAG only a validated executor contract, or can it actually run meaningful compute?

---

## Transition from Phase 3 to Phase 4

An intermediate “Phase 3-B” idea was considered:
- add more direct callbacks / more resource-backed toy paths

That idea was rejected as the main next milestone because it would mostly amount to writing more synthetic kernel-style code without proving that the DAG could express a *real* ggml-style operation.

The project direction was reframed:

> The next meaningful milestone is not “more resource tests.”
> It is “first real ggml-style CPU op on the fine-region DAG.”

That became Phase 4.

---

### Phase 4 — first real ggml-style CPU op on the fine-region DAG

#### Goal

Show that the fine-region DAG can execute a **real op-shaped computation**, not just synthetic callbacks.

Scope stayed intentionally narrow:

- F32 only
- one row
- CPU only
- correctness only
- no scheduler changes
- no full graph / llama integration yet

### Why RMS_NORM was chosen

`RMS_NORM` was chosen as the first real op because it naturally matches the fine-region model:

- lane-parallel partial work
- a reduction/finalize stage
- lane-parallel apply stage

It is also a real transformer-style computation, so succeeding here would mean the DAG is not merely a test harness anymore.

### Structural design of Phase 4

One RMS_NORM row was lowered into a 3-region DAG:

#### `R0` — partial sums of squares
Kind: `ELEMENTWISE`

Each lane computes:
- sum of `x[i]^2` over its chunk

and writes that partial into:
- `resources->lane_scratch[ith]`

#### `R1` — finalize
Kind: `REDUCTION`

Single-threaded.

Reads:
- `resources->lane_scratch[0 .. n_lanes)`

Computes:
- `sumsq`
- `scale = 1 / sqrt(sumsq / n + eps)`

Writes:
- `sumsq`
- `scale`

into:
- `resources->reduction_buffer`

#### `R2` — apply
Kind: `ELEMENTWISE`

Each lane reads `scale` from:
- `resources->reduction_buffer`

and writes:
- `dst[i] = x[i] * scale`

Dependencies:
- `R0 -> R1 -> R2`

This design deliberately exercised both:
- `lane_scratch`
- `reduction_buffer`

in one real computation.

### Test-first design

Before implementing the callbacks, a new test was written:

#### `test_hpx_region_group_rms_norm_f32`

Input:
- `x = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -8.f}`
- `eps = 1e-5f`
- `n_lanes = 4`

The test:
- built the exact `R0 -> R1 -> R2` DAG
- used a scalar reference RMS_NORM computation for the oracle
- checked the exact per-lane partial sums for the chosen split shape
- checked `reduce_buf.sumsq`
- checked `reduce_buf.scale`
- checked final `dst`

This test became the specification for the implementation.

### Production changes made in Phase 4

Two production files were changed:

#### 1. `ggml-hpx-region-exec.h`
Added 4 C-compatible structs:

- `ggml_hpx_rms_norm_f32_reduce_buffer`
- `ggml_hpx_rms_norm_partial_f32_ctx`
- `ggml_hpx_rms_norm_finalize_f32_ctx`
- `ggml_hpx_rms_norm_apply_f32_ctx`

And 3 new `run_range` declarations:

- `ggml_hpx_rms_norm_partial_f32_run_range`
- `ggml_hpx_rms_norm_finalize_f32_run_range`
- `ggml_hpx_rms_norm_apply_f32_run_range`

#### 2. `ggml-hpx-region-exec.cpp`
Added the 3 callback implementations:

- **partial**
  - accumulates `x[begin:end)^2` into `lane_scratch[ith]`

- **finalize**
  - asserts the current reduction contract
  - sums lane partials
  - computes `scale`
  - writes `sumsq` and `scale` into `reduction_buffer`

- **apply**
  - reads `scale` only from `reduction_buffer`
  - writes `dst[i] = x[i] * scale`

Notably:
- no scheduler changes were made
- no graph re-entry was introduced
- data was intentionally carried through resources, not smuggled through ctx structs

### Validation outcome

The new RMS_NORM DAG test passed.

All region-DAG tests passed as well.

This established that the fine-region DAG can now execute:

- real per-lane work
- a real reduction/finalize stage
- a real dependent apply stage

with correct numerical results.

### Phase 4 conclusion

Phase 4 completed the first **real compute** milestone:

> The fine-region DAG is not only a validated executor contract.
> It can host a real ggml-style CPU op decomposition and execute it correctly.

That was the bridge from:
- infrastructure proof

to:
- real compute substrate

---

## Net result of Phases 3 and 4

By the end of Phase 4, the project had established:

1. **Phase 3**
   - the fine-region executor carries real execution resources correctly

2. **Phase 4**
   - those resources are expressive enough to run a real op-shaped computation

Together, these phases transformed the fine-region DAG from:
- “an explicit HPX scheduling experiment”

into:
- “a viable execution substrate for real ggml-style CPU work”

without:
- re-entering the graph path
- redesigning the scheduler
- integrating into full llama execution yet

## HPX fine-region DAG provenance — Phase 5

### Context entering Phase 5

By the end of Phase 4, the project had already established that the fine-region DAG was no longer only a scheduling abstraction:

- dependency-driven region-group execution was working
- `lane_scratch` and `reduction_buffer` had been proven by tests
- a real ggml-style CPU op, F32 RMS_NORM, had been lowered into a 3-region DAG:
  - `R0` partial sumsq per lane
  - `R1` single-thread finalize
  - `R2` lane-parallel apply
- correctness was validated end to end

So the central question changed from:

> Can this design work?

to:

> Can this design work at a useful performance granularity?

Phase 5 was therefore not an architecture-design phase in the abstract sense. It was a measurement and diagnosis phase.

---

## Phase 5 — performance diagnosis and granularity study

### Goal

Determine whether the new HPX-native fine-region execution model is viable for real work at the tested granularity, and identify where the cost is coming from.

This phase was intentionally focused on:

- measurement
- overhead decomposition
- low-risk runner experiments
- deciding whether the current scheduling granularity is performance-viable

It was **not** about:

- adding more ops
- broad integration into llama / ggml graph execution
- rewriting the scheduler
- optimizing arithmetic kernels

---

## Phase 5A — baseline measurement

### Purpose

Establish a baseline comparison between:

- a direct scalar/reference RMS_NORM path
- the new 3-region HPX fine-region DAG path

### Work done

A new benchmark was created under:

- `hpx-bench/bench_hpx_rms_norm_f32.cpp`

with results saved under repo-local benchmark results directories rather than system temp space.

The benchmark measured:

- `ref`: scalar RMS_NORM baseline
- `dag`: real 3-region RMS_NORM DAG using production callbacks

Sweep dimensions included:
- multiple row sizes
- multiple lane counts

### Outcome

The benchmark showed that the new DAG path was much slower than the scalar reference at the tested sizes.

The first important signal was:

- correctness held
- but the path was **overhead-dominated**

This answered the first practical Phase 5 question:

> No, the current fine-region RMS_NORM execution is not yet a performance win.

---

## Phase 5B — overhead decomposition

### Purpose

Break the cost of the new path into understandable layers rather than treating “HPX overhead” as one undifferentiated number.

### Work done

The benchmark was extended with additional variants:

#### `direct`
A plain direct RMS_NORM path calling the production callbacks as ordinary function calls with no HPX and no region-group runner.

This answered:
> Are the production callbacks themselves expensive?

#### `dag_empty`
A synthetic near-no-op 3-region DAG with the same resource touch pattern but almost no math.

This answered:
> How much does the generic DAG machinery cost even when useful work is minimal?

### Main findings

The comparison ladder showed:

- `direct ≈ ref`
- `dag_empty >> direct`
- `dag` only modestly above `dag_empty`

This established:

1. the production callbacks and real RMS_NORM arithmetic were **not** the main problem
2. the dominant cost was the **generic per-call 3-region DAG machinery**

This was the key transition in understanding.

The project was no longer asking:
> Is HPX slow?

It was now asking:
> Is this scheduling granularity too fine for generic HPX DAG composition?

---

## Phase 5C — targeted overhead experiments

Once Phase 5B showed that the generic DAG path dominated cost, the project moved into small, targeted experiments to see which parts of that overhead were actually material.

### 5C.1 — remove the outer HPX crossing

A `nowrap` variant was introduced so the benchmark no longer paid an outer:

- `hpx::async([&]{ ... }).get()`

for each repetition.

This showed:

- removing the outer crossing saved a small but real amount of time
- the remaining floor stayed much larger than the pure math cost

Conclusion:
- the outer HPX boundary was **not** the main bottleneck

### 5C.2 — remove per-call heap churn inside the group runner

`ggml_hpx_run_region_group(...)` was given a small-buffer / stack-allocated fast path for tiny groups so that per-call heap allocations for runner bookkeeping could be avoided in the benchmarked RMS_NORM case.

The semantics remained unchanged:
- same Kahn topo logic
- same dependency-driven future composition
- same wait/join behavior

Re-running the benchmark showed that the min-time floor barely moved.

Conclusion:
- per-call heap allocation in the group runner was **not** the bottleneck

### 5C.3 — fused one-dispatch ceiling measurement

A fused benchmark path was added:

- `fused_async`

This performed the full RMS_NORM body inside one:

- `hpx::async(...).get()`

with no generic region DAG around the internal stages.

This created the decisive cost ladder:

- `direct`
- `fused_async`
- `dag_empty_nowrap`
- `dag_nowrap`

### Main findings from 5C.3

The fused result showed:

- `direct` = math only
- `fused_async` = math + one HPX post
- `dag_empty_nowrap` = 3-region DAG machinery with near-no-op callbacks
- `dag_nowrap` = real 3-region DAG

The comparison made the cost structure clear:

1. one HPX post was relatively cheap
2. the extra cost of the 3-region DAG itself was roughly a fixed several-microsecond tax
3. the real production callbacks added very little on top of the generic DAG machinery

This gave the cleanest Phase 5 conclusion:

> The dominant cost is not arithmetic, callback structure, outer HPX entry, or per-call heap churn.
> The dominant cost is the per-call generic 3-region HPX DAG dispatch/composition model at this granularity.

---

## What Phase 5 proved

Phase 5 proved something more precise than “fine-grained HPX is slow.”

It showed:

- the fine-region model is **correct**
- the production callbacks are **cheap**
- one HPX dispatch around coarser work is **plausible**
- the generic per-call tiny 3-stage DAG is **too expensive** at the tested sizes

This means the central issue is not merely “tuning HPX harder.”
It is the placement of the HPX runtime boundary.

The fine-region DAG remains valuable as:
- a semantic model
- a planning model
- a dependency/resource description

But Phase 5 showed that it is not necessarily the right **runtime scheduling granularity** for tiny per-op chains like this RMS_NORM example.

---

## Phase 5 conclusion

By the end of Phase 5, the project had a much sharper result:

> The current HPX-native fine-region representation is valid and expressive, but the generic per-call scheduling of tiny dependent region chains is too fine-grained to be performance-competitive at the tested sizes.

This was the important design insight.

Phase 5 therefore did not merely produce “bad benchmark numbers.”
It established where the HPX boundary likely needs to move:

- upward
- to coarser execution units
- so that HPX manages larger chunks of useful inference work rather than tiny per-op chains

That set up the next architectural question:

> What is the right coarse-grained HPX execution boundary for real inference work?

## 9. HPX-native frozen-packet redesign

### Motivation

The fine-region DAG design was already the right semantic representation:
it made dependencies explicit, preserved legality, and gave us a clean
source of truth for region-level execution.

But using the fine DAG directly as the runtime execution surface still left
too much generic per-call machinery in the hot path:
- per-call DAG composition
- dependency rebuild / topo handling
- per-region future-style execution structure
- per-call runtime orchestration around very small repeated units

The redesign started from a different premise:

> keep the fine-region DAG as the semantic and planning representation,
> but do **not** execute that generic structure directly every time.

Instead, compile it once into a reusable linear execution object.

### Design

The redesign introduced a second execution surface above the fine-region DAG:

fine-region DAG
→ sublayer-level lowering
→ frozen packet
→ per-invocation binding
→ packet execution

The fine DAG remains the authoritative input. The runtime executes a compiled
projection of it.

A **frozen packet** is:
- compiled once from a fine-region group
- immutable after compilation
- reusable across invocations
- executed as a straight-line step program instead of generic DAG machinery

Three-way split:

- `ggml_hpx_frozen_packet`
  - immutable compiled object
  - shareable across calls and threads

- `ggml_hpx_packet_frame`
  - mutable per-invocation state
  - receives the current call’s bindings
  - one frame per in-flight invocation

- `ggml_hpx_packet_runtime`
  - execution substrate
  - owns the pinned decode-side HPX execution context for packet execution

This enforced the central invariant:

> bind mutates the frame, never the packet.

### First prototype target: RMS_NORM_F32

The first prototype packet used the existing `RMS_NORM_F32` decomposition.

That target was chosen because it was:
- real
- small
- already understood
- simple enough to validate the packet model before moving to larger units

Its fine-region group was compiled into a 3-step packet.

### Packet execution model

The packet execution path deliberately removed generic DAG machinery from the
runtime hot path.

Execution is step-based:

- `SERIAL`
  - runs directly on the caller thread

- `LANE_FANOUT`
  - runs over lane indices using a pinned HPX execution context
  - uses `hpx::experimental::for_loop(...)`
  - synchronously joins before the next step

The runtime path therefore avoids:
- per-step `async`
- `dataflow`
- `shared_future`
- `wait_all`
- per-step heap allocation

At `n_lanes == 1`, fan-out steps are promoted to serial at compile time, so
the packet executes with zero HPX crossings.

### API direction

The packet surface was designed around a small explicit API.

Public structure:
- compile packet
- query frame size / alignment
- query resource requirements
- initialize caller-owned frame
- bind typed invocation values
- run frozen packet

For the prototype, binding stayed **typed per sublayer** rather than generic.

For `RMS_NORM_F32`, that meant a typed binding struct carrying:
- `x`
- `dst`
- `n`
- `eps`

This kept the first implementation honest and avoided introducing a generic
slot-walking ABI before the packet model itself was proven.

### Important correction during implementation

The first interpretation was that packet lane fan-out should reuse
`ggml_hpx_runtime_dispatch_decode/_prefill`.

Inspection showed those functions were currently serial stubs and therefore
not the real parallel execution primitive.

The actual reusable substrate was one layer deeper:
a pinned HPX execution object used with `hpx::experimental::for_loop(...)`.

The packet runtime was therefore corrected to use that pinned execution
context directly for `LANE_FANOUT` steps.

This changed the implementation detail, but not the design intent:
- keep persistent pinned HPX execution
- avoid generic DAG runtime machinery
- avoid per-step async/future composition

### Header and implementation work

The redesign was first turned into a real packet header and implementation.

The header established:
- immutable packet / mutable frame / runtime split
- structural plan key
- team identity in the key
- frame size and alignment queries
- resource requirement introspection
- typed RMS_NORM binding
- compile / bind / run entry points
- explicit invariants at the top of the file

The implementation then added:
- `ggml-hpx-packet.cpp`
- packet compile for `RMS_NORM_F32`
- frame init
- typed bind
- packet run loop
- CMake wiring into the HPX library

The benchmark binary re-linked cleanly against the updated library.

### RMS_NORM benchmark extension

A new `frozen_packet` row was added to the RMS_NORM benchmark ladder.

The timed body measured:
- `bind`
- `run_frozen_packet`

Compile and frame allocation happened once per `(n, lanes)` outside the timed
loop so the benchmark reflected the steady-state execution model rather than
one-time setup cost.

The benchmark used the same resource layout shape as the comparable
fine-DAG-based path so the comparison stayed fair.

### RMS_NORM result

The `RMS_NORM_F32` packet prototype validated the redesign.

Headline result:
- `frozen_packet` beat `dag_nowrap` in every measured `(n, lanes)` cell
- at `lanes = 1`, `frozen_packet` approached `direct`
- the remaining `n = 512` gap persisted under `NDEBUG`, showing it is real
  step-loop dispatch overhead rather than debug scaffolding

Representative min-ns results:

| n    | direct | frozen_packet(1) | frozen_packet(2) | frozen_packet(4) | dag_nowrap(1) | dag_nowrap(2) | dag_nowrap(4) | fused_async |
|------|--------|------------------|------------------|------------------|---------------|---------------|---------------|-------------|
| 512  | 250    | 459              | 4917             | 6125             | 7416          | 12375         | 12750         | 1000        |
| 2048 | 1125   | 1166             | 5958             | 6667             | 8958          | 13042         | 12416         | 3208        |
| 4096 | 2333   | 2250             | 6792             | 5625             | 10000         | 13875         | 13375         | 7500        |
| 8192 | 4916   | 4667             | 8709             | 9125             | 13709         | 14500         | 14875         | 10375       |

Ratios against `dag_nowrap` (min-ns):

| n    | lanes=1 | lanes=2 | lanes=4 |
|------|---------|---------|---------|
| 512  | 0.062   | 0.397   | 0.480   |
| 2048 | 0.130   | 0.457   | 0.537   |
| 4096 | 0.225   | 0.489   | 0.420   |
| 8192 | 0.340   | 0.601   | 0.613   |

`lanes = 1` sanity check against `direct`:

| n    | direct | frozen_packet(1) | ratio |
|------|--------|------------------|-------|
| 512  | 250    | 459              | 1.84  |
| 2048 | 1125   | 1166             | 1.04  |
| 4096 | 2333   | 2250             | 0.96  |
| 8192 | 4916   | 4667             | 0.95  |

### Interpretation of the first prototype

These results established three important points.

#### 1. The redesign removed the right overhead

`frozen_packet` beating `dag_nowrap` everywhere showed that the main cost
really was the generic per-call DAG machinery, not the mathematical work
itself.

#### 2. The packet model is correct for steady-state reuse

For meaningful sizes, `lanes = 1` collapsed close to direct execution.
That is exactly what the packet model was meant to recover:
- compile once
- bind cheaply
- run without rebuilding execution structure

#### 3. The remaining floor is small and concrete

The surviving `n = 512` gap after `NDEBUG` showed that the remaining fixed
cost is real packet step-loop dispatch overhead:
- frame indirection
- step dispatch
- small constant runtime overhead

That is fundamentally different from the old multi-microsecond generic DAG
floor and is small enough that it did not change the overall conclusion.

### What was still missing after RMS_NORM

At that point Section 9 had proven the packet mechanism, but only for one
prototype target. The codebase still did **not** have a reusable:

- `ggml op/tensor -> fine-region group`

lowering layer.

The shipped fine-region callbacks were only:
- `MUL_MAT F32`
- `RMS_NORM` partial / finalize / apply

That meant a real repeated subgraph, such as the llama-style MLP gate/up
pattern,

- `gate = MUL_MAT(W_gate, x)`
- `up = MUL_MAT(W_up, x)`
- `gate_act = SiLU(gate)`
- `out = MUL(gate_act, up)`

could not yet be built cleanly, because:
- `SiLU` and elementwise `MUL` were missing as fine-region callbacks
- lowering from real ggml nodes did not exist
- the existing packet success was still tied to the RMS_NORM prototype path

### Primitive fine-region expansion

To close that gap, two new primitive callbacks were added:

- `SiLU_F32`
- `MUL_F32`

These were implemented first and tested independently before lowering work
began.

Important design points:
- explicit ctx structs
- fixed work-range execution
- aliasing behavior documented and tested

### Real lowering layer

A real lowering layer was then added:

- `ggml_hpx_lower_op(...)`

Key design decision:
- one ggml op lowers to one **region group**
- single-region ops emit `n_regions = 1`
- multi-stage ops like `RMS_NORM` emit multiple regions

Lowering does **not** return borrowed ctx/region pointers into temporary
storage. Instead it lowers into a self-contained arena object that owns:
- region array
- dep array
- per-region ctx storage

That gave lowering a clean lifetime boundary and let later packet compile
consume a stable group representation safely.

Supported ops after this step:
- `MUL_MAT`
- `SiLU`
- `MUL`
- `RMS_NORM`

The first implementation was intentionally strict:
- F32 only
- explicit contiguity/layout checks
- shape consistency checks
- null-pointer rejection
- `RMS_NORM` limited to the single-row form already supported by the callback path

### Lowering correctness status

Lowering was validated with execution-based tests, not just structural checks.

Coverage included:
- region count / dep count
- kind / work ranges
- ctx placement
- actual execution vs scalar reference
- rejection cases
- aliasing cases

Test status at the end of this step:

| suite | tests | result |
|---|---:|---|
| `test_hpx_region_primitives_silu_mul_f32` | 6 | all passed |
| `test_hpx_lower_op` | 15 | all passed |
| pre-existing region DAG suites | 4 suites | no regressions |

### Real MLP gate/up composition

Once lowering existed, the next step was to prove that a **real repeated ggml
subgraph** could be composed without hand-building regions.

A new composer was added for the MLP gate/up unit:

- `gate = MUL_MAT(W_gate, x)`
- `up = MUL_MAT(W_up, x)`
- `gate_act = SiLU(gate)`
- `out = MUL(gate_act, up)`

The composed group uses a fixed 4-region layout:

| idx | op | kind |
|---:|---|---|
| 0 | `gate = MUL_MAT(W_gate, x)` | `MATMUL` |
| 1 | `up = MUL_MAT(W_up, x)` | `MATMUL` |
| 2 | `gate_act = SiLU(gate)` | `ELEMENTWISE` |
| 3 | `out = MUL(gate_act, up)` | `ELEMENTWISE` |

Cross-op deps:

| src | dst | meaning |
|---:|---:|---|
| 0 | 2 | `gate -> silu` |
| 1 | 3 | `up -> mul` |
| 2 | 3 | `silu -> mul` |

The composer also validates the expected topology:
- `gate_act` must be `silu(gate)`
- `out` must be `mul(gate_act, up)`
- commuted final `mul(up, gate_act)` is rejected intentionally
- wrong op kinds are rejected
- null inputs are rejected

This mattered because the composer is not just a convenience helper. It is the
first real proof that the new lowering layer can support a repeated ggml
subgraph without falling back to hand-built region groups.

### Frozen-packet support for MLP gate/up

With composition proven, packet support was extended to this new sublayer.

Added to `ggml-hpx-packet.h/.cpp`:
- `GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32 = 2`
- `ggml_hpx_mlp_gate_up_binding`
- `ggml_hpx_bind_mlp_gate_up_packet(frame *, const binding *)`

The packet-key shape convention for this sublayer was defined as:
- `shape[0] = out_cols`
- `shape[1] = cols`
- `shape[2] = rows`
- `shape[3] = 0`

Important packet design points:
- compile validates the composed 4-region group
- compile bakes structural dimensions into the packet template
- bind patches only the 7 data pointers
- dimensions are **not** rebound per invocation
- run executes a fixed 4-step topo-valid program

The MLP packet frame layout was fixed and explicit:
- 4 ctx slots
- total frame size 160 B
- bind patches only pointer fields at fixed offsets

### Packet correctness and hardening

The packet path was then validated in three layers:

1. **Packet correctness**
   - build ggml subgraph
   - compose group
   - compile packet
   - allocate/init frame
   - bind one invocation
   - run packet
   - compare to scalar reference

2. **Wrong key-shape rejection**
   - `shape[0] == 0`
   - mismatched `rows`

3. **Malformed-group rejection**
   - empty group
   - valid MLP group under the wrong sublayer key

This established that the packet path was not just happy-path correct; it also
rejected invalid compile contracts.

### Benchmark design for the MLP unit

With correctness established, the benchmark compared three execution surfaces
for the same MLP gate/up unit:

- `dag_group`
- `frozen_packet`
- `direct_manual`

Where:
- `dag_group` = composed fine-region DAG executed through the generic scheduler/runtime path
- `frozen_packet` = compile once, bind, run fixed packet program
- `direct_manual` = same callback kernels, same buffers, but hand-written 4-op execution order with no generic DAG machinery

This was the right benchmark structure because it separated:
- generic scheduler overhead
- packet overhead
- raw callback/math cost

The initial correctness check for the sweep used absolute tolerance, but larger
shapes produced legitimate larger absolute accumulation error. That was changed
to a relative tolerance of `1e-5`, after which all 12 shapes passed.

### MLP packet vs DAG results

Median times from the benchmark:

| shape (`out_cols × cols × rows`) | `dag_group` | `frozen_packet` | ratio (`dag/pkt`) |
|---|---:|---:|---:|
| `3×4×2` | 9083 ns | 125 ns | 73× |
| `64×64×1` | 9208 ns | 500 ns | 18× |
| `64×64×16` | 18666 ns | 7583 ns | 2.5× |
| `64×256×1` | 11916 ns | 1708 ns | 7× |

### MLP packet vs direct/manual results

The direct/manual comparison clarified what the packet cost really is.

| shape class | `direct_manual` vs `frozen_packet` | interpretation |
|---|---|---|
| tiny (`3×4×2`) | packet is ~40 ns slower | almost entirely bind cost (patching 7 pointers) |
| `64×64` and above | `dm/pkt ≈ 1.0` | packet overhead is effectively negligible above raw callback execution |

One representative datapoint:

| shape | `direct_manual` | `dag_group` | interpretation |
|---|---:|---:|---|
| `64×64×1` | 500 ns | 9583 ns | the extra ~9 µs is generic DAG scheduler/runtime overhead |

### Tail behavior

The p95 behavior reinforced the same conclusion:
- `dag_group` p95 regularly spikes to about **1 ms**
- `frozen_packet` p95 stays very close to the median
- `direct_manual` p95 also stays tight to the median

That means the long tail belongs to the generic scheduler path, not the packet
design and not the math kernels.

### Interpretation of the MLP results

These results establish three important points.

#### 1. The generic DAG path has a real fixed floor

The `dag_group` path sits at roughly **7–10 µs** regardless of shape. That
cost is not the math; it is:
- `shared_future` allocation
- dataflow chain construction
- topological dispatch
- generic runtime machinery around tiny repeated units

#### 2. The frozen packet removes almost all of that floor

Its cost scales with the actual mathematical work instead of paying a large
fixed orchestration tax.

#### 3. For meaningful shapes, packet is already at the callback-level lower bound

Above toy shapes, `direct_manual ≈ frozen_packet`.

So packetization is not just faster than the generic DAG path. It is also
already very close to the best-case hand-written execution surface for this
sublayer.

### What Section 9 established overall

Section 9 established all of the following:

- the project can compile a fine-region group into a reusable execution unit
- packet reuse works with immutable compiled state plus mutable per-call state
- typed bind per sublayer is sufficient for the first prototype and the first
  real composed ggml subgraph
- the packet runtime can execute the compiled schedule without generic DAG
  machinery in the hot path
- a real `ggml op -> region group` lowering layer now exists
- a real MLP gate/up subgraph can be composed from ggml nodes and frozen into
  a packet
- the result is materially faster than executing the equivalent work through
  the generic fine-region runtime path
- for meaningful MLP shapes, frozen-packet execution is effectively on par
  with direct/manual callback execution

### Conclusion of this phase

At the end of Section 9, the project has moved beyond packet feasibility.

The frozen-packet model is now validated in two ways:

1. as a reusable execution abstraction (`RMS_NORM_F32`)
2. as the right execution form for a real repeated ggml subgraph (MLP gate/up)

The main result of Section 9 is therefore:

> the frozen-packet design removes the generic fine-DAG scheduler floor while
> getting essentially all the way down to direct/manual callback execution
> cost for meaningful repeated ggml subgraphs.

That makes frozen packets the right next-level execution form for repeated
ggml work at this layer of the HPX design.

## 10. Selective mixed execution and first real llama.cpp integration

After the fine-region DAG executor was validated internally, the next step was to stop treating it as an isolated mechanism and ask a narrower integration question:

> can one supported op go through the fine-region path while everything else still runs through the normal CPU path?

This was the first real bridge from:
- fine-region unit tests

to:
- actual llama.cpp execution

### What already existed before this step

Before adding a new graph-level selective executor, one important check was made:

- **op-level lowering already existed**
- `ggml_hpx_lower_op(...)` was already tested for:
  - `SILU_F32`
  - `MUL_F32`
  - `MUL_MAT_F32`
  - `RMS_NORM_F32`
- rejection cases were already covered, including:
  - unsupported ops
  - non-F32 types
  - invalid null-data cases
  - multi-row `RMS_NORM`

So a new “does `MUL_MAT` lower?” unit test would have duplicated existing coverage rather than adding a new capability. The real missing piece was not op-level lowering; it was **graph-level mixed execution**.

### New graph-level selective executor

A new selective execution path was added:

- `ggml-hpx-exec-selective.h`
- `ggml-hpx-exec-selective.cpp`

Its job is intentionally narrow:

- walk `gf->nodes[]` in graph order
- try `ggml_hpx_lower_op(...)` on each node
- if lowering succeeds, execute that node through the fine-region DAG path
- if lowering fails, execute that node through the real CPU backend fallback path

This created the first graph-level mixed mode:

```text
supported lowered op
  → fine-region DAG path

unsupported op
  → normal CPU backend fallback
```

The first mixed execution target was deliberately small:

- lowered `mul_mat`
- fallback `neg`

### Why the fallback path mattered

The first design question was how unsupported nodes should be executed.

A toy hand-written scalar fallback (for example, a custom `neg` loop) would have been easy, but it would not have tested the actual intended integration. The selected fallback instead matched the existing coarse HPX decode execution style:

- build a 1-node `ggml_graph_view`
- call `ggml_backend_graph_compute(...)`

That is the same general mechanism the existing decode path already uses for graph slices: a `ggml_graph_view(...)` over the live graph and a real backend compute call. The decode executor already requires a live caller-owned CPU backend handle in `ggml_hpx_decode_backends`, with `cpu` required and used as the normal decode backend.

This made the fallback path a real execution path, not a test-only imitation.

### First mixed execution test

A new test was added:

- `tests/hpx/test_hpx_region_mixed_mul_mat.cpp`

Its graph shape was intentionally tiny:

```text
w, x
  → mul_mat   [supported: lowered]
  → neg       [unsupported: fallback]
```

The test used identity-like weights so the expected output was exact and simple:

- `mul_mat` produced known values
- `neg` flipped them
- final output could be checked element-by-element

This proved the first important mixed-mode property:

> one node can run through the fine-region path while a later node in the same graph falls back to the normal CPU backend path, and the final result still matches exactly

### Proving the lowered path really ran

After the first mixed test passed, two test-only counters were added under a dedicated testing guard:

- lowered count
- executed count

The mixed test then asserted:

- lowered == 1
- executed == 1

This was important because a passing output check alone would not prove that `mul_mat` actually went through the fine-region path. With the counters, the test established both:

- final output correctness
- routing correctness

### First llama.cpp integration

Once graph-level selective execution worked in isolation, the next step was to connect it to the real llama path.

The following changes were made:

- `src/llama-context.h`
  - added a `hpx_selective_mul_mat` flag under `#ifdef GGML_HPX`
- `src/llama-context.cpp`
  - read `LLAMA_HPX_SELECTIVE_MUL_MAT` at context construction
  - inside `graph_compute(...)`, under the existing HPX path and an additional `#ifdef GGML_HPX_REGION_DAG` guard, route to the selective graph executor when the flag is enabled
- `tests/hpx/test_hpx_llama_selective_mul_mat_smoke.cpp`
  - added a new smoke test reusing the same structure as the existing HPX smoke test:
    - reference context: `LLAMA_USE_HPX` unset
    - selective context: `LLAMA_USE_HPX=1` and `LLAMA_HPX_SELECTIVE_MUL_MAT=1`
    - run one decode batch
    - compare every logit exactly

This was the first real llama-integrated test of the fine-region work.

### First real integration bug: reduction-backed lowering

The first llama smoke integration crashed.

The initial suspicion was quantized `MUL_MAT`, but debug output showed the actual cause:

- `RMS_NORM` was still being lowered on the single-token decode path
- its lowered group had three regions and included a `REDUCTION` region
- the selective executor was constructing a minimal `ggml_hpx_region_resources` with null `lane_scratch` and null `reduction_buffer`
- the reduction path dereferenced `lane_scratch[ith]`

This matched the existing op-level tests:

- `RMS_NORM_F32` lowering is valid for single-row inputs
- but the execution tests only work when `lane_scratch` and `reduction_buffer` are populated for the lowered group

So the smoke crash was not caused by the selective path being fundamentally invalid. It was caused by the selective path lowering a resource-heavy group without also providing the resources that group requires.

### Narrow fix: reduction guard

The fix was intentionally narrow.

For the selective llama integration, lowered groups are now rejected from the selective path if they contain a `REDUCTION` region.

That means:

- resource-light lowered groups may still run through the fine-region path
- reduction-backed groups fall back to the real CPU backend path

This kept the selective integration small and safe without prematurely broadening the scope into full resource-backed lowering inside real llama execution.

The debug print used to confirm this root cause was then kept only as an optional env-gated diagnostic.

### Making the selective path measurement-ready

After the smoke test passed, the next question became not correctness but observability and fairness.

Two improvements were added:

#### 1. Lightweight selective stats
A per-call selective stats struct was added with:
- lowered node count
- fallback node count
- lowered execution time
- fallback execution time

`llama_context` stores the last selective stats and prints:

```text
[hpx-selective] lowered=N fallback=M lowered_ms=X fallback_ms=Y
```

when `LLAMA_HPX_SELECTIVE_STATS=1` is set.

This made the selective path observable during real llama runs.

#### 2. Real live-backend fallback
The selective path was updated to use the live CPU backend from the llama context rather than treating fallback as a standalone arena-style execution.

This followed the existing decode execution model more closely:
- decode already expects a live caller-owned CPU backend in the backend bundle
- decode slices are executed by building a `ggml_graph_view(...)` and calling `ggml_backend_graph_compute(...)` on that live backend

Using the live CPU backend made the selective path fairer to measure and avoided introducing obviously artificial per-call backend creation into the real llama path.

### What this milestone proved

At the end of this step, the project had established all of the following:

- op-level lowering already existed and was already tested
- graph-level mixed execution now exists
- a lowered node and a fallback node can coexist in one graph correctly
- routing correctness is testable through explicit counters
- the selective path is integrated into the real llama HPX branch
- a real model-backed smoke test passes with exact logit equality
- reduction-backed lowered groups are safely rejected to fallback for now
- the selective path is now stats-enabled and measurement-ready

### What this milestone did **not** prove

This step did **not** yet prove:

- end-to-end performance benefit from selective lowering
- that a quantized TinyLlama run actually lowers any `MUL_MAT`
- that resource-heavy lowered groups are ready for real llama execution

In particular, TinyLlama Q4_K_M is expected to produce `lowered=0` for quantized `MUL_MAT`, so this model validates:
- real llama-path routing
- real fallback correctness
- selective stats plumbing

but not necessarily the performance benefit of real selective `mul_mat` lowering.

### Interpretation

This is the point where the fine-region DAG work stopped being:
- a purely internal execution experiment

and became:
- a real llama-integrated execution experiment with selective lowering, fallback, and stats

That is a meaningful transition in the project.

The next step is no longer “does the selective path work at all?”

That part is now established.

The next step is:
- run real decode/prefill comparison cases
- inspect `lowered` / `fallback` stats
- determine whether the chosen model/path actually exercises meaningful selective lowering or only validates real fallback behavior

## 11. v1 packet integration in llama and the activation-proof finding

Section 10 closed with a working selective integration but a remaining gap:
the selective executor still dispatched lowered nodes through the generic
fine-region DAG runtime, not through the frozen-packet surface Section 9
had validated as the right execution form for repeated sublayers. Every
small elementwise op therefore paid the generic DAG per-call floor.

v1 was scoped to close that gap at the narrowest honest surface — one
sublayer (MLP gate/up), one compiled packet per shape, decode-only
dispatch, a single-lane packet runtime — and to do so without disturbing
any of the existing selective routing when the packet path is not engaged.
The goal of this milestone was explicitly a clean integration boundary,
not a performance claim.

### CPU-only guard — Section 10 refinement

Before the packet work started, one narrow refinement to Section 10's
selective wiring was required. Earlier 3-case runs on TinyLlama Q4_K_M
at default `-ngl` had been dispatching entire Metal-offloaded graphs
through the CPU backend, producing degenerate output (all `<unk>` tokens)
and meaningless timing — the selective path replaced
`ggml_backend_sched_graph_compute_async` and then fed a graph whose
nodes were tagged for Metal through a single CPU backend.

The fix was a three-condition structural guard at the selective call
site in `llama_context::graph_compute`:

- `ggml_backend_sched_get_n_splits(sched) == 1`
- `gf->n_nodes > 0`
- `ggml_backend_sched_get_tensor_backend(sched, gf->nodes[0]) == backend_cpu`

If any condition fails, the selective path is disabled for that call and
the graph flows through the normal scheduler. Mixed-backend graphs
(Metal-offloaded TinyLlama) therefore no longer contaminate selective
measurement; selective only engages on genuinely CPU-only graphs. This
is a caller-side decision; `ggml_hpx_exec_graph_selective_mul_mat`
itself does not inspect the scheduler.

### v1 API shape

The v1 design added five concrete pieces and deliberately did not modify
the shape of the existing selective entry point beyond two new optional
parameters.

1. **Stats buckets**
   `ggml_hpx_selective_stats` gained three fields: `packet_matches`,
   `packet_nodes`, `packet_dispatch_ns`. These are disjoint from
   `lowered_*` and `fallback_*`: when packet dispatch is enabled, every
   node in `gf` lands in exactly one of the three buckets.

2. **Opaque sublayer-specific cache**
   `ggml_hpx_mlp_gate_up_packet_cache`. Per-entry key is
   `(out_cols, cols, rows)`; cache-wide create-time parameters are
   `(n_lanes, seq_regime, policy_version)`. The cache is a **plan
   store** and holds no reference to a packet runtime. That split is
   load-bearing: if the cache had owned a runtime reference, plan and
   executor would have conflated, which Section 9 was careful to avoid.

3. **Extended selective entry point**
   `ggml_hpx_exec_graph_selective_mul_mat` gained two optional
   parameters `packet_rt` and `mlp_cache`. Both null → behaviour
   identical to Section 10's selective executor. Both non-null → the
   matcher pre-scan runs and recognized MLP gate/up subgraphs dispatch
   as frozen packets. Exactly one of the two non-null → debug-assert,
   release treats as "packet off". The contract is documented in the
   header directly.

4. **llama_context ownership**
   `llama_context` gained one flag (`hpx_mlp_gate_up_packet`, from
   `LLAMA_HPX_SELECTIVE_MLP_PACKET=1`) and two lazily-owned pointers.
   The runtime and cache are constructed on the first decode-side
   selective-eligible `graph_compute` call; if either construct fails,
   any partial state is destroyed and the call falls back cleanly to
   packet-off, rather than half-enabling the path.

5. **Decode-only gating at the caller**
   The MLP gate/up packet is compiled for `team = DECODE`, and prefill
   shapes (`rows > 1`) would require a separate packet per batch size.
   Gating is therefore done in `llama_context::graph_compute` by
   passing non-null packet arguments only when `!batched`. The
   selective executor itself does not inspect a "decode" flag.

### Matcher and dispatch site

The pre-scan and main loop are deliberately simple, with one load-
bearing invariant.

- One pass over `gf->nodes`. For each `GGML_OP_MUL`, walk back through
  its sources: `src[0]` must be a `GGML_OP_UNARY` with subop SiLU whose
  own `src[0]` is a `MUL_MAT` (the gate), and `src[1]` must be a
  `MUL_MAT` (the up). A shared-x check (`gate.src[1] == up.src[1]`)
  rules out coincidental matches. An overlap guard rejects any
  candidate whose four nodes are already claimed by a prior match.
  `consumed[]` and `match_at_final[]` are updated **only after**
  `lookup_or_compile_mlp` succeeds; a failed compose/compile leaves the
  nodes on the normal lowered/fallback path.

- Main loop state per index `i`:
  - `!consumed[i]` — standard lowered/fallback routing
  - `consumed[i] && match_at_final[i] < 0` — non-trigger packet member (skip)
  - `consumed[i] && match_at_final[i] >= 0` — final MUL of a match (dispatch packet)

The "final MUL owns dispatch" invariant means the packet's dependencies
(gate MUL_MAT, up MUL_MAT, SiLU output) have all been materialized in
ggml order by the time we dispatch; dispatching at the gate index would
read an unmaterialized up output.

The dispatch site binds from **live** ggml tensor `->data` pointers at
every call. Only `packet` and `frame` come from the cache; the seven
binding pointers (`w_gate`, `w_up`, `x`, `gate`, `up`, `gate_act`,
`out`) are read from the actual matched tensors on each dispatch.

Two test-only atomics provide the cache-reuse observable:
`g_hpx_mlp_packet_compile_count` is bumped on cache miss after compile
succeeds; `g_hpx_mlp_packet_dispatch_count` is bumped on every dispatch
(hit or miss).

### Isolation test

Before any llama wiring, a dedicated test
(`test_hpx_selective_mlp_gate_up_packet.cpp`) builds the 4-op graph
explicitly via ggml ops and drives it through the new selective entry
point with `n_lanes = 1`. The test asserts:

- matcher recognizes the 4-node pattern (`packet_matches == 1`,
  `packet_nodes == 4`)
- first call: `compile_count == 1`, `dispatch_count == 1`
- second call: `compile_count` still `1`, `dispatch_count == 2`
- `lowered_nodes == 0` and `fallback_nodes == 0` on both calls
- `packet_dispatch_ns > 0` (timing present; no further timing claim)
- output matches the scalar reference within relative `1e-5`, including
  after the output buffer is zeroed between the two calls (which proves
  bind really does read live tensor data each time)

Runtime: 7–19 ms including HPX startup. The test runs in the same
target structure as `test_hpx_region_mixed_mul_mat` — it re-compiles
`ggml-hpx-exec-selective.cpp` with `GGML_HPX_EXEC_SELECTIVE_TESTING` so
the atomics are visible.

### Q4_K_M smoke — transparent when the matcher finds nothing

Once v1 was wired through llama, the first end-to-end smoke ran on the
existing TinyLlama Q4_K_M model with `LLAMA_HPX_SELECTIVE_MLP_PACKET=1`
and `-ngl 0`.

Result: every `[hpx-selective]` line reported `packet=0(0 nodes)`.
`lowered` and `fallback` counters were identical to the pre-packet-flag
run on the same model (`lowered=2 fallback=687` at prefill,
`lowered=45 fallback=644` every decode). Output was coherent. The lazy-
init log `HPX MLP gate/up packet dispatch ready (n_lanes=1)` appeared
exactly once, on the first decode-side graph_compute. Context teardown
clean.

This matches what the header's "first-deployment scope" block said to
expect: on Q4_K_M the MLP MUL_MATs are quantized,
`ggml_hpx_lower_op` rejects them, `ggml_hpx_compose_mlp_gate_up_group`
fails, the cache never records an entry, and the selective path behaves
exactly as it did before v1. The integration is transparent when the
matcher finds nothing.

### Milestone A — activation-proof attempt on F32 weights

With v1 closed as a clean integration surface, a narrow follow-up
milestone (A) asked the simplest possible next question:

> does `packet_matches > 0` in real llama when the MLP projections are F32?

A was scoped explicitly as an activation proof — a mechanism milestone,
not a performance milestone. It was kept separate from any widening of
lowering or composition (the intended follow-up, milestone B).

The F32 model was produced by dequantizing the existing Q4_K_M GGUF:

```
build-hpx-dag/bin/llama-quantize --allow-requantize \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf \
    F32
```

5.3 s, 4196 MiB output. Every `ffn_gate.weight` and `ffn_up.weight` was
converted from `q4_K` to `f32` (logged per-tensor during quantize). This
is dequantization, not recovered F32 precision: values remain Q4_K-
rounded, only the tensor dtype changes. That is exactly what the
matcher needs — F32 **type**, not F32 **accuracy** — and was flagged
explicitly in the result-directory README so the run is not misread as
a quality baseline.

### Unexpected result — packet_matches stayed at 0

The smoke ran coherently, but every stats line still reported
`packet=0(0 nodes)`. Two things did change compared to the Q4_K_M run:

- `lowered` rose from 45 → 200 per decode (and 2 → 157 for prefill).
  The MLP and attention MUL_MATs now pass the F32 gate in
  `ggml_hpx_lower_op`, and the fine-region path does real work on a
  realistic graph.
- Per-decode `lowered_ms` rose to ~4 s, reflecting the cost of running
  22 layers of F32 MUL_MAT through the generic fine-region runtime
  without packet acceleration.

But zero packet dispatches.

### Root cause — GLU[SWIGLU] fusion

`src/llama-graph.cpp:1074` emits the MLP tail as a single fused op:

```
cur = ggml_swiglu_split(ctx0, cur, tmp);
```

which produces one `GGML_OP_GLU` node with subop `GGML_GLU_OP_SWIGLU`,
not the three-op decomposition `SiLU(gate) → MUL(silu, up)` that v1's
matcher scans for. The fused GLU node is a ggml-level optimization
independent of the HPX work; it simply predates v1.

So the real per-layer MLP tail in TinyLlama's decode graph is:

```
gate = MUL_MAT(W_gate, x)
up   = MUL_MAT(W_up,   x)
cur  = GLU[SWIGLU](gate, up)    <-- one node, not three
out  = MUL_MAT(W_down, cur)
```

None of that contains the 3-op SiLU+MUL tail v1 expects. `packet_matches
== 0` by construction, regardless of weight dtype.

### What Milestone A established

- v1's end-to-end path runs on a real F32 model without regression:
  env flag honoured, lazy creation fires exactly once, matcher scans
  every decode graph, selective stats include the packet bucket,
  teardown clean.
- Flipping MLP weight dtype from Q4_K to F32 really does unlock
  `ggml_hpx_lower_op` acceptance for the MUL_MAT nodes — `lowered`
  goes from 45 → 200 per decode step on this model. The fine-region
  path is capable of operating on a realistic graph once the dtype
  constraint is lifted.
- v1's matcher is **shape-incomplete** for real llama, independent of
  any quantization work. The gap is a ggml-level fusion, not an
  HPX-layer bug.

### What A does NOT prove

- Packet dispatch actually firing in llama — did not happen on this
  model because the graph does not contain the v1-assumed pattern.
- Any speedup claim at any layer.

### Consequence for Milestone B

A was designed to run to a conclusion and close. It did. It also
sharpened B's scope by turning the original "broaden lowering and
composition for quantized workloads" into a more precise pair of
sub-targets:

1. **GLU[SWIGLU]-aware composition.** Either a second matcher shape
   (two `MUL_MAT` + one `GLU`) or a GLU-aware composer variant, plus a
   new sublayer id
   (`GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_GLU_F32 = 3`) so the plan
   key stays structurally distinct from the existing 4-op form.
2. **Q4_K-aware MUL_MAT lowering.** The original B scope. Requires
   `dtype`/quant-scheme in the packet plan key so cached packets do
   not collide across quant schemes.

Landing (1) alone lets an F32-dequant model fire real packet dispatch
in llama. Landing (2) alone removes the dequantization dependency for
all quantized models. Landing both closes the full
Section 7 → 8 → 9 → 10 → 11 arc on realistic quantized inference.

### Conclusion

v1 is complete: the mechanism is proven in isolation, integrated into
llama behind the `LLAMA_HPX_SELECTIVE_MLP_PACKET` flag, transparent
when the matcher finds nothing, lazy-lifecycled, honest observables.

Milestone A is complete as a finding, not a speedup: activation on a
real llama graph does not happen with v1's matcher shape because llama
emits a single `GLU[SWIGLU]` node where v1 expects three ops. A's
value is in the finding, not in a positive activation count.

Milestone B is now scoped with one concrete first target
(GLU-aware composition) alongside its original quantized-MUL_MAT
target. B is its own milestone, with its own provenance section, and
deliberately not mixed into v1 or A.

### Opening phase of Milestone B.1 — graph shape before quantization

Milestone A closed with one specific finding: packet activation in real
llama was blocked first by **graph shape**, not by packet plumbing.
Real llama emits a fused `GGML_OP_GLU` node with subop
`GGML_GLU_OP_SWIGLU`; v1's packet path expected the unfused
`MUL_MAT, MUL_MAT, SiLU, MUL` tail and therefore could never match a
production llama MLP, even on the F32-dequant model.

That observation fixed the order of work for milestone B:

- **B.1 first:** make the packet/composer path match the real llama MLP
  graph shape (`MUL_MAT, MUL_MAT, GLU[SWIGLU]`)
- **B.2 later:** broaden lowering/composition to realistic quantized
  workloads

This ordering matters. If quantized-MUL_MAT work had been started first,
a failure to activate packet dispatch in llama would still have been
ambiguous: it could have been a packet bug, a quantized-lowering bug, or
the already-known graph-shape mismatch. By landing B.1 first, the graph-
shape problem is isolated and solved on its own terms.

### Item 1 — fused SWIGLU primitive

The first B.1 step was to add the missing primitive at the fine-region
layer: a fused `SWIGLU_F32` elementwise callback.

The existing `SiLU_F32` and `MUL_F32` kernels in
`ggml-hpx-region-exec.cpp` are both straight-line loops over an
elementwise context struct. `SWIGLU_F32` was therefore a trivial
extension of the same surface:

- new context struct:
  - `const float * gate`
  - `const float * up`
  - `float * dst`
  - `int64_t n`
- new run-range callback:
  - `dst[i] = silu(gate[i]) * up[i]`

No scratch, no reductions, no new region kind. The primitive is still
`GGML_HPX_CPU_REGION_KIND_ELEMENTWISE`, like `SiLU_F32` and `MUL_F32`.

A dedicated primitive test file established three things:

1. scalar equivalence against `(g / (1 + exp(-g))) * u`
2. alias safety for `dst == gate`
3. correct `[begin, end)` partial-range behavior

All three tests passed. This mattered not just for correctness, but for
resource shape: the fused GLU region has the same "no scratch" profile
as the earlier SiLU/MUL pair, so packet resources for the GLU-fused
sublayer remain zero-sized outside of `n_lanes`.

### Item 2 — lowering `GGML_OP_GLU` with `GGML_GLU_OP_SWIGLU`

With the primitive in place, the next step was to make the lowered
region surface capable of expressing the real llama graph node.

`ggml_hpx_lower_op` in `ggml-hpx-lower.cpp` gained a new
`case GGML_OP_GLU:` arm, gated strictly on

- `ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU`

and then on the same basic shape/type checks already used for existing
F32 elementwise lowering:

- F32 dtype
- contiguous storage
- matching element counts

The emitted lowering is a single-region `ELEMENTWISE` group backed by
the new `SWIGLU_F32` callback. In other words, the real llama fused GLU
node now lowers directly, without being decomposed back into `SiLU`
plus `MUL`.

That detail is load-bearing. Decomposing inside lowering or composition
would have reintroduced an intermediate `gate_act` buffer that the ggml
graph itself no longer materializes. Keeping the GLU node fused at the
HPX layer preserves the same "one node, one region" shape the real
graph already has.

The existing `test_hpx_lower_op` suite still passed unchanged after this
addition, so the new GLU case did not regress SiLU, MUL, MUL_MAT, or
RMS_NORM lowering.

### Item 3 — GLU-aware composer

Once `GGML_OP_GLU[SWIGLU]` could lower as a single region, the composer
layer could be made to reflect the real llama MLP tail directly.

The new composer adds a second MLP sublayer form alongside the earlier
4-op gate/up form:

- `gate = MUL_MAT(W_gate, x)`
- `up   = MUL_MAT(W_up,   x)`
- `glu  = GLU[SWIGLU](gate, up)`

This is a 3-op, 3-region, 2-dependency group:

- region 0: gate MATMUL
- region 1: up MATMUL
- region 2: fused GLU elementwise
- deps:
  - `0 -> 2`
  - `1 -> 2`

The new composed group mirrors the existing
`ggml_hpx_mlp_gate_up_group` pattern closely:

- fixed arrays for `lowers[]`, `combined_regions[]`, `combined_deps[]`
- outer `ggml_hpx_cpu_region_group`
- same ownership rule: once composed, do not move/copy the group
- same single-region / zero-dep validation on each constituent `lower_op`
  result

The new entry point takes three ggml nodes:

- `node_gate`
- `node_up`
- `node_glu`

and performs strict topology checks before lowering:

- `node_gate->op == GGML_OP_MUL_MAT`
- `node_up->op == GGML_OP_MUL_MAT`
- `node_glu->op == GGML_OP_GLU`
- `ggml_get_glu_op(node_glu) == GGML_GLU_OP_SWIGLU`
- `node_glu->src[0] == node_gate`
- `node_glu->src[1] == node_up`

The input order is asserted strictly rather than treated as commutative.
Even though the math of the fused SWIGLU tail could be seen as symmetric
at a high level, the ggml graph has a real producer/consumer convention,
and the composer preserves it directly.

With Item 3 landed, the fine-region and composer layers were no longer
assuming the old `SiLU + MUL` tail. The real llama graph shape was now
expressible all the way through composition.

### Item 4 — `MLP_GLU_F32` packet sublayer

The fourth step packetized that new composed sublayer.

A new packet sublayer id was added:

- `GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32 = 3`

This is append-only after `MLP_GATE_UP_F32 = 2`. It is a new structural
packet family, not a revision of the earlier 4-op form, so it begins its
own fresh per-sublayer version space at `policy_version = 1`.

The new binding drops the old `gate_act` pointer and becomes a six-
pointer structure:

- `w_gate`
- `w_up`
- `x`
- `gate`
- `up`
- `out`

The `shape[]` convention remains identical to `MLP_GATE_UP_F32`:

- `{out_cols, cols, rows, 0}`

so callers share the same structural dimension layout, but the packet
key remains distinct because `sublayer` differs.

The packet arena is smaller than the earlier 4-region packet:

- frame header
- gate MATMUL ctx
- up MATMUL ctx
- fused SWIGLU ctx

Total: 136 B, 8-byte aligned.

`ggml_hpx_compile_packet` gained a third packet branch for
`MLP_GLU_F32`:

1. gate MATMUL
2. up MATMUL
3. fused SWIGLU

The validator mirrors `validate_mlp_gate_up_group`, but for the fused
3-region shape:

- region kinds:
  - MATMUL
  - MATMUL
  - ELEMENTWISE
- run-range callbacks:
  - `mul_mat`
  - `mul_mat`
  - `swiglu`
- work ranges:
  - `[0, out_cols)`
  - `[0, out_cols)`
  - `[0, out_cols * rows)`
- deps:
  - `0 -> 2`
  - `1 -> 2`

As with the earlier packet, resources remain zero-sized outside
`n_lanes`: no reduction buffer, no lane scratch, no shared scratch.

A narrow packet test then proved the GLU packet surface in isolation.
The test mirrors the earlier `MLP_GATE_UP_F32` packet test, but builds
the fused ggml subgraph via `ggml_swiglu_split(gate, up)` and binds
through the new six-pointer structure. It asserts:

1. compile + bind + run matches the scalar reference
2. wrong key shape is rejected
3. malformed-group and wrong-sublayer cases are rejected

All three tests passed.

The wrong-sublayer case was intentionally made adversarial by using the
sibling `MLP_GATE_UP_F32` sublayer as the mismatched key rather than
something obviously unrelated like RMS_NORM. That matters because the
two MLP packet families share the same `shape[]` convention; the test
therefore exercises the per-sublayer validator dispatch path directly.

### Interim state at this pause point

At the end of Item 4, B.1 had completed the packetization groundwork
needed to match the real llama GLU-fused MLP shape:

- fused `SWIGLU_F32` primitive
- `GGML_OP_GLU[SWIGLU]` lowering
- GLU-aware composer
- `MLP_GLU_F32` packet sublayer with compile/bind/run support
- narrow tests for primitive, lowering, and packetization all green

What remained after this pause point was not packet construction, but
integration:

1. selective matcher/cache/env integration for the new GLU-fused
   sublayer
2. `llama_context` wiring for the second packet cache
3. real F32-dequant llama smoke with `packet_matches > 0`

In other words, by the end of Item 4, B.1 had already solved the
**representation problem** — the real llama MLP shape could now be
expressed and packetized at the HPX layer — but had not yet reconnected
that new packet family back into the selective llama path.

### Item 5 — selective matcher/cache/env integration for the GLU path

Item 5 closed the integration gap on the executor side.

A second prescan pass `prescan_mlp_glu_matches` was added, structurally
symmetric to the gate/up scan: walk `gf->nodes`; on each
`GGML_OP_GLU` with `GGML_GLU_OP_SWIGLU`, require both `src[0]` and
`src[1]` to be `GGML_OP_MUL_MAT` appearing earlier in the node order,
with a shared-x check between the two matrix multiplies. Overlap guard
and `consumed[]` discipline mirror the 4-node scan exactly. Only three
node indices are claimed per match (gate, up, glu-trigger) instead of
four.

A second cache type `ggml_hpx_mlp_glu_packet_cache` was added alongside
the gate/up cache. Both caches are owned by the same `llama_context`
lazy-creation block, and both key on `(out_cols, cols, rows)`, but they
remain separate types because the packet plan key carries a different
`sublayer` discriminator and a different per-shape policy_version
space.

The dispatch site in `ggml_hpx_exec_graph_selective_mul_mat` gained
three optional parameters (`packet_rt`, `mlp_cache`, `mlp_glu_cache`).
Null triple → legacy selective behavior. All three non-null → both
matchers pre-scan the graph and claim disjoint node sets; the main
loop treats a node as a GLU trigger, a gate/up trigger, a non-trigger
claimed member, or a normal lowered/fallback node, in that priority.
The execution order constraint is that `GLU` trigger dispatch must
retire gate and up results before firing — this matches the fine-region
dependency edges `0 -> 2, 1 -> 2` already enforced by the packet's
region DAG.

A narrow selective GLU isolation test was added to cover the integrated
path end-to-end (pattern recognition → compile → bind → dispatch)
without llama involvement, plus a regression test to confirm the
gate/up packet still fires on synthetic 4-node graphs where GLU is
absent.

### Item 6 — `llama_context` wiring and first F32-dequant smoke

Item 6 wired the new GLU cache through `llama_context` and ran the
first real-llama smoke.

The context-side env gate (`LLAMA_HPX_SELECTIVE_MLP_PACKET=1`) was
extended so that on first decode-eligible graph_compute the context
lazily creates one packet runtime plus **both** packet caches together.
If any of the three allocations fail, all partial state is destroyed
and the call falls back to packet-off, preserving the all-or-nothing
rule from v1. Prefill graphs continue to pass a null packet-argument
triple to the selective entry point, because both caches are compiled
for `team = DECODE` and `rows = 1`.

The 2026-04-18 smoke on TinyLlama F32 (dequantized from Q4_K_M), with
`-ngl 0 -n 16 "Hello"`, reported:

- `packet=22(66 nodes)` per decode graph — 22 transformer layers, each
  contributing `MUL_MAT(gate) + MUL_MAT(up) + GLU[SWIGLU]`
- `lowered=156` (down from the PACKET=0 baseline's 222)
- `fallback=467` unchanged
- the gate/up matcher did **not** fire — on real llama graphs, the
  SWIGLU fusion subsumes the older 4-node pattern entirely

This confirmed activation, but the initial speedup reading from that
smoke (2.0× decode, 3.9× prefill) was later shown to be largely a
measurement artifact.

### Warmup-cliff investigation and honest B.1 speedup

The 2026-04-18 smoke had noted an "HPX packet warmup cliff": per-token
`packet_ms` started high on the first decode tokens (225 ms on token
0, declining toward ~60 ms steady state) and the PACKET=0 comparator
appeared to suffer a thermal cliff on long F32 decode. Before scoping
B.2, the cliff was investigated directly.

### Hypotheses from code trace

Four candidate sources of one-time-per-process cost were identified by
reading the dispatch path end-to-end:

1. **Lazy packet-runtime init** on the first eligible decode token
   (`llama_context::graph_compute`), which constructs the scheduler
   and decode executor with processing-units pinning.
2. **First-shape packet compile** in `lookup_or_compile_mlp_glu`,
   including `ggml_hpx_compile_packet` + aligned frame allocation.
   TinyLlama has one MLP shape, so at most one compile per process.
3. **HPX runtime first-task cost**: cold TLB, NUMA first-touch,
   worker wake-up on the first `hpx::async` / first LANE_FANOUT
   `for_loop` on the decode executor.
4. **CPU-level warmup**: page faults on F32 weight matrices, cold L2/L3,
   DVFS ramp-up.

### Measurement protocol established during the investigation

The first retry of the smoke produced packet_ms values of 1100–3200 ms
and lowered_ms values varying 6× on identical work. A `ps` check
identified a stray `build-hpx-cpu/bin/llama-bench` consuming ~98% CPU;
after it was killed, a follow-up also surfaced `mds_stores` (Spotlight
indexing rebuilt binaries) at 56% CPU. Waiting out both produced a
quiet-machine run.

A rebuild at clean commit `26578ce22` (working tree stashed) was also
made to separate code drift from machine drift. That clean-commit run
produced stable lowered_ms in the 94–202 ms range but did not fire the
GLU matcher at all, confirming that the GLU packet path lives only in
the working-tree Items 5–6 changes. Code drift was ruled out as the
source of the earlier wild numbers.

### Quiet-machine Exp 1 v2 — no cliff at all

Rerunning the Apr 18 invocation exactly, on a verified-quiet machine
with working tree restored, produced per-token `packet_ms`:

- first three tokens: 39.4 / 40.3 / 38.4 ms
- last three tokens: 43.5 / 35.4 / 40.1 ms
- n=15, mean = 41.4 ms, stdev = 3.2 ms (8% of mean)

Token 0 is already at steady state. There is no monotonic decay.
The "warmup cliff" described in the Apr 18 smoke was a consequence of
CPU contention and thermal state, not a property of the HPX packet
path.

### Compile cost measured directly

An env-gated stderr log was added at both compile sites (`matcher=glu`
at `lookup_or_compile_mlp_glu` and `matcher=gate_up` at
`lookup_or_compile_mlp`), activated by
`LLAMA_HPX_PACKET_COMPILE_LOG=1`, with a cached one-shot `getenv`
read. Exactly one line fires per process on real TinyLlama:

```
[hpx-packet-compile] matcher=glu out_cols=5632 cols=2048 rows=1 compile_ns=1083
```

**1083 ns ≈ 1 µs.** Four orders of magnitude below the 42 ms first-token
dispatch cost. Compile is not a hidden warmup source. The `matcher=gate_up`
line never fires on real llama, matching the Item 5/6 finding that GLU
fusion subsumes the old 4-node pattern.

### Honest PACKET=0 vs PACKET=1 comparison on a quiet machine

Same binary, same invocation, back-to-back on the verified-quiet machine:

| bucket            | PACKET=0       | PACKET=1       | delta     |
|-------------------|---------------:|---------------:|----------:|
| `lowered_ms`      | 213.4 ± 57.2   | 144.2 ± 47.8   | −69.2 ms  |
| `fallback_ms`     |  33.5 ± 14.9   |  34.0 ± 16.3   |  +0.5 ms  |
| `packet_ms`       |   0            |  41.4 ± 3.2    | +41.4 ms  |
| **total**         | 246.9 ± 58.9   | 219.6 ± 55.6   | −27.3 ms  |

Eval throughput: 4.05 → 4.55 tok/s. **Honest B.1 decode speedup
on CPU-only F32 TinyLlama: 1.12× (≈11%).**

The arithmetic closes: moving 66 nodes out of `lowered` costs the
packet bucket 41.4 ms and saves 69.2 ms in lowered work, net 27.8 ms
saved per token, matching the observed 27.3 ms total delta within
rounding.

### Why the former "2×" was inflated

PACKET=0 has 66 more nodes flowing through the generic lowered path,
which is more sensitive to thread contention and scheduler jitter than
the frozen packet. On a contested machine, PACKET=0 suffers
disproportionately, enlarging the apparent PACKET=1 advantage. On a
quiet machine the gap shrinks from 2× to 1.12×.

### Outcome of this investigation

- The GLU packet path is real, correct, and fires as intended on real
  llama decode (22 matches per graph, 66 nodes reclaimed).
- The honest decode win on CPU-only F32 TinyLlama is ~12%, not 2×.
- No warmup cliff exists on a quiet machine; compile cost is
  negligible (~1 µs).
- A fair-comparison protocol was captured in `CLAUDE.local.md` for
  future HPX decode benchmarks: quiet-machine check, single rebuild,
  back-to-back runs with alternating arm order, per-bucket mean ±
  stdev reporting, arithmetic cross-check, no cross-machine
  comparisons.
- Instrumentation left behind: `LLAMA_HPX_PACKET_COMPILE_LOG` env gate
  at both compile sites, zero overhead when unset. Reference run
  directories: `hpx-bench/results/2026-04-20-packet-warmup-trace-v2/`,
  `2026-04-20-packet-baseline-quiet/`, `2026-04-20-compile-log/`.

With this, B.1 closes as an integration + honest speedup milestone,
and the remaining challenge — making the packet path useful on
realistic quantized models — remains scoped to B.2.

### Goal of B.2
B.2 is the point where the packet work stops being an F32 proof-path result and starts targeting real quantized llama models. The intended first milestone was not “all quantization,” but a narrow, real path: **TinyLlama, decode-only, one quant scheme, one sublayer family**, with the GLU path as the first production target.

The roadmap for B.2 had two candidate directions:

- **true quant-aware lowering**
- **quant→F32 bridge**

plus an early extension of the packet key so quantized and non-quantized cache entries cannot alias.

### What blocked the live GLU path on quantized models
The live GLU selective path already matched the 3-node pattern:

- `gate_mm`
- `up_mm`
- `glu`

However, the original compile path for a matched GLU pattern always routed through the existing F32 compose/lower pipeline. That pipeline called `ggml_hpx_lower_op` on the `MUL_MAT` nodes, and lowering rejected them when the weight tensor type was not `GGML_TYPE_F32`. On real quantized models such as `Q4_K_M`, this meant:

- prescan found the pattern correctly
- `lookup_or_compile_mlp_glu()` tried to compose the packet
- lowering rejected `gate_mm` / `up_mm`
- packet compilation failed
- no cache entry was created
- all three nodes fell back to the CPU path

So the immediate blocker for B.2 was **not** GLU pattern recognition, but the fact that the packet compile path still required **F32-weight MUL_MAT lowering**.

### Decision record: why B.2 started with QBRIDGE
Two candidate directions were considered.

#### Option A: true quant-aware lowering
Teach the lowering / compose / packet stack to accept quantized `MUL_MAT` directly, preserving the existing 3-step packet shape and moving toward a more native long-term design.

#### Option B: quant→F32 bridge
Let quantized `MUL_MAT` remain outside the packet for the first cut. Run those matmuls through the CPU backend first, then reuse the packet machinery only for the final fused SWIGLU elementwise stage.

The smallest cut that could make the real quantized TinyLlama GLU path fire was **Option B**, implemented as **QBRIDGE**.

### Chosen B.2 cut
The selected design was:

- keep the existing **3-step `MLP_GLU_F32`** packet path unchanged
- add a sibling packet sublayer:
  - **`GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE`**
- for quantized-weight GLU matches:
  - compile a **1-step SWIGLU-only packet**
  - execute `gate_mm` and `up_mm` through the CPU backend at dispatch time
  - then bind and run the SWIGLU packet over the already-produced F32 gate/up outputs

This preserved the B.1 F32 path, avoided quant kernel work in the first cut, and allowed the selective executor to start matching real quantized GLU patterns.

### Packet-key and cache-key extension
Packet-key extension was required so cached entries do not collide across quant schemes.

#### Internal selective cache key
`mlp_glu_cache_key` was extended with:

- `w_gate_type`
- `w_up_type`

so structurally identical GLU patterns with different weight dtypes now populate separate cache entries.

#### Frozen packet key
`ggml_hpx_packet_plan_key.extra` was assigned a documented per-sublayer convention for GLU packets:

- bits `[0:7]`   = `w_gate_ggml_type`
- bits `[8:15]`  = `w_up_ggml_type`

For example:

- F32/F32 weights → `extra = 0`
- Q4_K/Q4_K weights → `extra = 0x0C0C`

This keeps quantized and non-quantized GLU packet identities distinct even when shape and sublayer family are otherwise similar.

---

## File-level implementation provenance

### `ggml-hpx-packet.h`
This file was extended to define the new bridge sublayer and its binding contract.

Changes:
- added `GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE = 4`
- documented the `extra` field bit layout for GLU packet families
- added `ggml_hpx_mlp_glu_qbridge_binding`
- added `ggml_hpx_bind_mlp_glu_qbridge_packet(...)`

The new binding surface intentionally carries only:

- `gate`
- `up`
- `out`

because the quantized `MUL_MAT` stage is outside the packet in the bridge design.

### `ggml-hpx-packet.cpp`
This file gained the actual frozen-packet support for QBRIDGE.

Changes:
- added a new **arena layout** for a 1-step SWIGLU-only packet
- added `validate_mlp_glu_qbridge_group(...)`
- added a compile branch for `GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE`
- added `ggml_hpx_bind_mlp_glu_qbridge_packet(...)`

The frame contract is deliberately small:

- frame header
- one `ggml_hpx_swiglu_f32_ctx`
- total frame size: 40 bytes

The compile path bakes `n = out_cols * rows` into the SWIGLU context template and leaves only the pointers to be patched at dispatch time.

### `ggml-hpx-exec-selective.h`
This header was extended so the selective stats can represent the mixed bridge path honestly.

Changes:
- added `bridge_fallback_nodes`
- added `bridge_fallback_ns`

These fields count and time the CPU-backend work that still happens inside a QBRIDGE match:

- `bridge_fallback_nodes` = number of gate/up nodes executed through the CPU backend
- `bridge_fallback_ns` = wall time spent in those CPU backend calls

### `ggml-hpx-exec-selective.cpp`
This was the main behavioral change for B.2.

#### Cache separation
`mlp_glu_cache_key` and its hash/equality were extended with weight dtypes, so F32 and quantized GLU entries do not alias.

#### Relaxed prescan
`prescan_mlp_glu_matches(...)` no longer requires F32 weights. Instead, it accepts:

- `gate_mm` and `up_mm` with any weight dtype
- as long as the **MUL_MAT outputs are F32**
- and the existing shape / shared-input pattern still holds

That is the critical change that lets quantized GLU patterns enter the packet-selection path at all.

#### Branching compile path
`lookup_or_compile_mlp_glu(...)` now branches on weight dtype:

- **F32/F32** → existing `MLP_GLU_F32` compose + compile path
- **non-F32** → QBRIDGE path

For the QBRIDGE branch:
- it bypasses `ggml_hpx_compose_mlp_glu_group(...)`
- lowers only the GLU node itself
- compiles a 1-region SWIGLU packet
- writes the weight dtypes into `pkey.extra`

This is the key architectural point of the B.2 cut: the bridge path does **not** try to lower quantized `MUL_MAT`.

#### Dispatch behavior
At dispatch time, the GLU selective path now distinguishes between:

- full `MLP_GLU_F32` packet
- `MLP_GLU_QBRIDGE`

For `QBRIDGE`:
1. build a 1-node graph view for `gate_mm`
2. execute it through the CPU backend
3. build a 1-node graph view for `up_mm`
4. execute it through the CPU backend
5. bind the gate/up/output buffers to the QBRIDGE packet
6. run the 1-step SWIGLU packet

#### Stats accounting cleanup
The first QBRIDGE cut undercounted the real matched-path cost because the two fallback `MUL_MAT` calls happened outside packet timing and were not represented in stats. That was corrected by extending `ggml_hpx_selective_stats` and updating the dispatch branch so the bridge path is accounted for explicitly.

Current accounting for a QBRIDGE match is:

- `packet_nodes == 1` for the GLU trigger
- `bridge_fallback_nodes == 2` for `gate_mm` + `up_mm`
- `bridge_fallback_ns` accumulates the CPU backend time for those two calls
- `packet_dispatch_ns` covers only the SWIGLU packet execution

This makes the bridge path stats honest without pretending that all three matched nodes were executed inside the packet.

---

## Test provenance

### New isolation test
A new test file was added:

- `tests/hpx/test_hpx_selective_mlp_glu_qbridge.cpp`

It currently adds two targeted tests.

#### 1. `MatchesCompilesOnceDispatchesTwice`
This test builds a minimal 3-op GLU subgraph with:

- F16 gate weights
- F16 up weights
- F32 activations
- F32 gate/up/glu outputs

It verifies that:
- prescan accepts the pattern for non-F32 weights
- the first selective call compiles exactly one GLU packet
- the second selective call reuses the cached packet
- the output matches a scalar reference implementation
- the packet path fires twice across two invocations
- the updated stats are reported correctly:
  - `packet_matches == 1`
  - `packet_nodes == 1`
  - `bridge_fallback_nodes == 2`
  - `bridge_fallback_ns > 0`
  - `lowered_nodes == 0`
  - `fallback_nodes == 0`

This proves that the **QBRIDGE mechanism** is alive end-to-end and that the bridge portion of the work is now counted explicitly.

#### 2. `F32AndF16CacheEntriesAreDistinct`
This test uses the same shape twice:

- once with F32 weights
- once with F16 weights

and verifies that these populate **distinct cache entries** rather than aliasing through shape alone.

### Current test status
At this point the HPX test status is:

- **26 / 26 HPX tests passing**

A full background test run also reported two failing non-HPX tests:

- `test-tokenizers-ggml-vocabs`
- `test-jinja-py`

Those are llama.cpp tokenizer / Python tests and were already failing before this B.2 work. They are unrelated to the HPX selective / packet path.

---

## What B.2 has achieved so far
Up to this point, B.2 has established the following:

1. **The blocker is understood and isolated**  
   Quantized GLU was failing because the compile path still depended on F32-only `MUL_MAT` lowering.

2. **A minimal real-path bridge exists**  
   Quantized-weight GLU patterns can now enter the selective path and trigger a packet.

3. **Quantized and non-quantized packet identities are separated**  
   Both the internal selective cache and the frozen packet key now encode weight dtype.

4. **The F32 path stays intact**  
   The existing `MLP_GLU_F32` packet path remains unchanged.

5. **Bridge stats are now represented honestly**  
   The two CPU-backend `MUL_MAT` calls are no longer hidden behind packet-only accounting.

6. **There is a working isolation harness for the bridge path**  
   The new test covers compile, cache hit/miss behavior, output correctness, and the updated bridge stats.

7. **The HPX test surface remains green**  
   All 26 HPX tests pass after the QBRIDGE landing and stats cleanup.

---

## What landed in the validator + real-model smoke session 

Three changes closed the remaining open items from the previous session.

### 1. Validator tightening (`ggml-hpx-packet.cpp`)

`validate_mlp_glu_qbridge_group` now explicitly enforces the zero-dependency
contract:

```cpp
if (group->n_deps != 0)
    return "MLP_GLU_QBRIDGE group must have zero dependency edges";
```

This was a structurally guaranteed invariant before; it is now a hard validator
rejection.  All 26 HPX tests remain green.

### 2. Selective path gated to decode-only (`src/llama-context.cpp`)

The selective executor was previously engaged for **both** prefill and decode
graphs.  For prefill (batched=true), this caused the scalar SWIGLU kernel to
replace ggml's SIMD kernel in every layer, producing ~2–3% KV-cache drift that
propagated into the decode logits.

Fix: the outer guard changed from

```cpp
if (hpx_selective_mul_mat)
```
to
```cpp
if (hpx_selective_mul_mat && !batched)
```

Prefill now falls through to the normal ggml scheduler unconditionally.  The
selective path engages only for single-token decode calls (`batched = n_tokens > 1`
is false), which is the only regime the decode-team packets are designed for.

### 3. Real-model smoke restructured and passing (`tests/hpx/test_hpx_llama_selective_mul_mat_smoke.cpp`)

The smoke test previously fed all prompt tokens in a single `llama_batch_get_one`
call.  With `n_tokens > 1`, `batched = true` and `packet_eligible = false`, so
QBRIDGE never fired.

The test now runs two phases per context:

1. **Prefill** — all prompt tokens in one batch (`batched = true`, normal
   scheduler, no selective path).
2. **Decode** — one fixed token (`kDecodeToken = 1`, `batched = false`,
   `packet_eligible = true`).

Logits are captured only from the decode step and compared with
`EXPECT_NEAR(..., kLogitTol)` where `kLogitTol = 5e-6f`.  The tolerance covers
the ~4 ULP float32 noise between our scalar SWIGLU kernel and ggml's SIMD
kernel; the prefill-path bug we gated out was ~0.2 off (four orders of
magnitude larger), so the tolerance still catches any real error.

**Confirmed on TinyLlama `Q4_K_M`, CPU-only, M4:**

```
graph_compute: HPX MLP packet dispatch ready (gate/up + GLU, n_lanes=1)
[hpx-packet-compile] matcher=glu_qbridge out_cols=5632 cols=2048 rows=1 w_gate=12 w_up=12
[  PASSED  ] 1 test.
```

- `w_gate=12` = `GGML_TYPE_Q4_K`
- `out_cols=5632` = TinyLlama intermediate dimension
- `rows=1` = single-token decode
- logits within 5e-6 of reference

---

## Status summary
The current B.2 state is best described as:

- **design choice made:** quant→F32 bridge first
- **implementation landed:** QBRIDGE sibling sublayer
- **cache safety landed:** dtype-aware keying
- **selective path landed:** quantized GLU can now match and dispatch
- **stats cleanup landed:** bridge fallback work is counted explicitly
- **isolation coverage landed:** targeted bridge tests passing
- **validator tightened:** zero-dependency contract is now an explicit rejection
- **selective path decode-gated:** prefill falls through to normal scheduler
- **real-model smoke complete:** TinyLlama Q4_K_M, CPU-only, `matcher=glu_qbridge` confirmed
- **HPX test surface green:** 26 / 26 HPX tests passing

### Open items
The one remaining item from the original list is direct key inspection:

- `validate_mlp_glu_qbridge_group` and the isolation test prove cache separation
  indirectly through compile behavior.  A packet-level test that directly reads
  `ggml_hpx_packet_key(entry->packet)->sublayer` and the `extra` field would
  complete the coverage story.

That is lower priority now that the real-model smoke is confirmed.  The next
meaningful step is **honest comparison work**: measure QBRIDGE decode throughput
against the baseline (packet=0) under the fair-comparison protocol defined in
`CLAUDE.local.md`.

### Immediate next step when resuming
1. (optional) add direct packet-key inspection test
2. run A/B decode benchmark: QBRIDGE vs baseline, following the fair-comparison
   protocol (quiet machine, single binary, alternating order, per-bucket stats)

---

## B3: Can HPX run Q4_K MUL_MAT?

### Question

The QBRIDGE path routes Q4_K MUL_MAT nodes through the ggml CPU backend
(the "bridge fallback") before handing off only the SWIGLU step to HPX.  The
question for B3 is whether HPX can run the Q4_K MUL_MAT itself — removing the
CPU-backend dependency for the MLP projection step.

### What ggml actually does for Q4_K × F32

There is no `ggml_vec_dot_q4_K_f32` kernel.  ggml's design is:

1. **Quantize activations**: the F32 input row is quantized to Q8_K once via
   `quantize_row_q8_K`.  This is cheap relative to the dot products.
2. **Dot products**: `ggml_vec_dot_q4_K_q8_K(n, s, bs, vx, bx, vy, by, nrc)`
   computes the inner product of one Q4_K weight row with one Q8_K activation
   row.  This function dispatches to architecture-specific SIMD (ARM NEON,
   x86 AVX2, etc.) or a generic fallback.

The block layout:
- `block_q4_K`: 144 bytes, covers 256 elements (`QK_K = 256`).
  Fields: `d` (ggml_half scale), `dmin` (ggml_half min), `scales[12]`
  (6 bits each for 8 sub-scales and 8 sub-mins), `qs[128]` (4 bits/element).
- `block_q8_K`: 292 bytes, covers 256 elements.
  Fields: `d` (float scale), `qs[256]` (int8), `bsums[16]` (int16 subblock sums).

Row sizes (TinyLlama decode, cols = 2048):
- Q4_K row: `(2048/256) × 144 = 1152 bytes` (8 blocks)
- Q8_K row: `(2048/256) × 292 = 2336 bytes`

### Symbol visibility

Both `quantize_row_q8_K` and `ggml_vec_dot_q4_K_q8_K` are declared in
`ggml/src/ggml-cpu/quants.h` (internal, no `GGML_API`).  They are not
reachable from `ggml/include/`.  The correct approach — already used in this
codebase for `ggml_vec_dot_f32` — is `extern "C"` forward declarations in the
implementation file.  Their symbols are present in the linked `ggml` library.

### Two-region kernel design

A single `run_range` that quantizes inside the parallel fan-out would cause
every lane to redundantly quantize the same activation vector (wasteful) or
race on shared scratch (incorrect).  The correct decomposition is two regions
with a dep edge:

```
Region 0  REDUCTION (serial)   ggml_hpx_quantize_q8_k_f32_run_range
          begin=0, end=cols
          ctx: { x: F32 input, x_q8: scratch pointer, cols }
          → calls quantize_row_q8_K once

Region 1  MATMUL (parallel)    ggml_hpx_mul_mat_q4_k_q8_k_run_range
          begin=0, end=out_cols
          ctx: { w_q4k, x_q8 (shared), y, cols, out_cols, w_row_stride, q8k_row_bytes }
          → calls ggml_vec_dot_q4_K_q8_K per output column

Dep: 0 → 1
```

HPX owns the dependency edge.  The dep ensures region 1 does not fan out
until region 0 has written the Q8_K scratch.  No allocation inside any
`run_range` callback.

The REDUCTION kind was chosen for region 0 because `launch_region_async`
already enforces `ith=0, nth=1` for REDUCTION — exactly the serial-exactly-once
semantics the quantize step needs — without requiring a new region kind.

### Scratch storage

The Q8_K activation row (2336 bytes for cols=2048) is too large for the
128-byte `ctx_buf` slots.  A `scratch[4096]` arena was added to
`ggml_hpx_lowering`.  Both ctx structs hold a pointer into this arena; it is
zero-initialized by `ggml_hpx_lowering_init` via `memset`.  `lower_op` rejects
Q4_K nodes whose `ggml_row_size(Q8_K, cols)` exceeds 4096 (covers all
reasonable decode widths including TinyLlama's cols=2048).

### lower_op branch

The MUL_MAT case in `lower_op` was refactored from a single F32-only path into
two branches gated on weight type:

- `w->type == GGML_TYPE_F32` — existing 1-region path, unchanged.
- `w->type == GGML_TYPE_Q4_K` — new 2-region path, `rows == 1` only.

Contiguity check for Q4_K: `w->nb[0] == ggml_type_size(Q4_K)` and
`w->nb[1] == ggml_row_size(Q4_K, cols)`.  The top-level
`if (node->type != GGML_TYPE_F32) return false` in `lower_op` was not
changed — it remains correct because `MUL_MAT(Q4_K, F32)` always produces a
F32 output tensor.

### Selective executor interaction

The REDUCTION region in the Q4_K group causes `has_reduction = true` in
`ggml_hpx_exec_graph_selective_mul_mat`.  The selective executor currently
falls back to the ggml CPU backend for any group containing a REDUCTION region
(to avoid null `lane_scratch` / `reduction_buffer`).  For Q4_K nodes the
fallback is correct — ggml handles Q4_K natively — so no incorrect output
occurs.  Routing Q4_K through the HPX selective path without falling back
requires either a SERIAL region kind or relaxing the `has_reduction` check to
distinguish scratch-using reductions from scratch-free serial steps.  That is
left for a follow-on item.

### Test

`LowerOpMulMatQ4KF32.StructureAndExecution` in `test_hpx_lower_op.cpp`:

1. Builds a `ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 2)` weight tensor
   and a `[1 × 256]` F32 activation tensor using `no_alloc=true`.
2. Quantizes synthetic F32 weights to Q4_K via `quantize_row_q4_K` (also
   forward-declared via `extern "C"`).
3. Calls `lower_op` and asserts the 2-region structure, dep edge, and ctx
   field values.
4. Executes via `run_group(&lo, 1, nullptr, nullptr)`.
5. Computes a reference by calling `quantize_row_q8_K` + `ggml_vec_dot_q4_K_q8_K`
   directly.  Both paths are deterministic with the same inputs, so
   `EXPECT_FLOAT_EQ` (exact equality) is used.

`LowerOpRejects.Q4KMultiRowReturnsFalse`: verifies that `rows=2` is rejected.

### Result

```
[  PASSED  ] LowerOpMulMatQ4KF32.StructureAndExecution (0 ms)
[  PASSED  ] LowerOpRejects.Q4KMultiRowReturnsFalse (0 ms)
[  PASSED  ] 17 tests.  (full lower_op suite — no regressions)
```

**Answer: yes, HPX can run Q4_K MUL_MAT.**  The two kernels are proven correct
by the unit test.  The selective executor does not yet route Q4_K nodes through
HPX (falls back to ggml CPU); that integration is the remaining open item.

### Open items

- Add a SERIAL region kind (or relax the `has_reduction` guard) so the
  selective executor routes Q4_K MUL_MAT through HPX instead of falling back.
- Extend `rows > 1` once the rows=1 path is confirmed in the selective executor.
- Benchmark the HPX Q4_K path against the QBRIDGE baseline once the selective
  path integration is done.

## B4: Routing Q4_K through the selective executor (real-model integration)

### Question

After B3 proved that HPX can execute Q4_K `MUL_MAT` in isolation, the next
question was:

> Can the selective executor route real-model Q4_K nodes through HPX and
> preserve correctness at decode time?

---

### Initial symptom

Selective decode on TinyLlama `Q4_K_M` produced:

<s> Paris is a<unk><unk><unk>

CPU baseline produced:

<s> Paris is a great place to

Key observation:
- Divergence begins at decode step 1
- Pattern suggests state corruption, not numeric noise

---

### Isolation

Selective breakdown:

- MUL_MAT (Q4_K): 122
- MUL: 45
- GLU: 22
- RMS_NORM: 44 (fallback)
- Other: ~456 (fallback)

Guard experiment:

LLAMA_HPX_SELECTIVE_NO_Q4K=1

Result:
- With Q4_K → incorrect
- Without Q4_K → correct

Conclusion: bug is inside Q4_K lowering path

---

### Strided output hypothesis

Fix implemented:
- Relax contiguity guard for rows=1
- Add y_nb0 stride
- Write via byte arithmetic

Result:
- Tests pass
- No change in real model

Reason:
TinyLlama uses contiguous temporaries + ggml_cpy

---

### Kernel correctness

Tested at production shapes:
- 2048×256
- 2048×2048
- 2048×5632

All match CPU within tolerance

Conclusion: kernel is correct

---

### Root cause hypothesis

Q4_K lowering:

R0: quantize → writes Q8_K scratch
R1: parallel dot → reads scratch

Scratch:
ggml_hpx_lowering::scratch[4096]

In real graph:
- ~122 nodes per decode step
- multiple groups in flight

Hazard:
scratch reused/overwritten before read completes

---

### Ruled out

- Quantization noise
- QBRIDGE path
- GLU/MUL kernels
- KV cache striding

---

### Current state

- Q4_K lowering integrated ✔
- Kernel correct ✔
- Strided output fixed ✔
- Real decode correctness ✖

---

### Next step

Make scratch safe:

Options:
1. per-node scratch
2. per-lane scratch
3. synchronous execution
4. redesign ownership

---

### Big picture

This is a systems bug (memory/lifetime), not math.
