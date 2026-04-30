# HPX executor contract for ggml/llama.cpp

## Purpose

This is the durable rulebook for HPX-related CPU execution work in this branch.

It is not a provenance log, benchmark report, or milestone tracker. Old branch notes are reference evidence only. They should help explain what was tried, but they should not be treated as current design truth.

Use:

- `docs/hpx/prefill_branch_summary.md` for the old branch conclusion and evidence.
- `docs/hpx/selective_graph_map_reference.txt` for the old TinyLlama graph-map capture.
- this file for execution rules and review criteria going forward.

---

## Current direction

The project direction is **not**:

```text
replace pthread with HPX everywhere
```

The direction is:

```text
use HPX only where the execution unit is large, structured, and worth the scheduling cost
```

The fresh branch should move toward a **run-level executor**:

```text
analyze graph
build structural run plan
cache plan
bind live tensors per invocation
execute larger runs
```

HPX should schedule runs or structured region groups, not tiny decode nodes.

---

## 1. Evidence gate

Before optimizing or packetizing a path, prove:

- the path actually engages
- the graph exposes useful work at that level
- the baseline is measured under comparable conditions
- the change reduces dispatch or dependency overhead
- the result beats or plausibly approaches the scheduler baseline

Correctness alone is not enough. Reduced overhead inside a slow path is not enough.

---

## 2. Decode and prefill

Decode and prefill are operational regimes, not reliable graph-shape labels.

Guidance:

```text
whole-graph decode-sized substrate work -> avoid HPX unless proven beneficial
matched packetized decode work           -> possible only if dispatch granularity improves
large prefill-like substrate work        -> possible only if the topology exposes useful work
```

Do not assume prefill is a good HPX target just because it has more tokens. The old branch showed that a prefill region graph can be fully linear, which leaves HPX no coarse inter-region parallelism to exploit.

Forbidden hot-path shape:

```text
native llama thread
  -> one tiny node
  -> hpx::async(...).get()
  -> next tiny node
  -> hpx::async(...).get()
```

Preferred future shape:

```text
native llama thread
  -> one HPX-owned plan execution
  -> internal run scheduling
  -> one terminal wait
```

---

## 3. Execution units

### Run-level plan

A run is a larger execution unit created by graph analysis. Possible run kinds:

- `fallback_run`: contiguous graph slice executed by the normal backend
- `packet_run`: matched repeated sublayer with compile-once / bind-per-call behavior
- `lowered_run`: one or more HPX-native region groups
- `backend_run`: opaque backend-owned work

The planner should assign every ggml node to exactly one run.

### Fine CPU regions

Fine-region code uses explicit ranges, dependencies, and resource bundles:

```cpp
run_range(ctx, ith, nth, begin, end, resources)
```

Fine regions may be used inside lowered runs or packets. They should not become the public scheduling granularity for the whole graph unless measurement proves it is worthwhile.

### Packets

Packets are for repeated model sublayers with known op pattern, tensor types, shape/layout key, and scratch/resource ownership.

Packets should reduce dispatch granularity. They must not become thin wrappers around one tiny node.

### Fallback/backend runs

Unsupported or backend-owned work should remain opaque. Adjacent fallback nodes should be coalesced when correctness allows.

---

## 4. Structural plans and keys

Reusable plans and packet keys must be structural.

Allowed key material:

- operation pattern / sublayer kind
- shape
- dtype / trait class
- row stride or layout stride when baked into compiled context
- lane count or execution policy when it affects code shape
- policy version

Forbidden key material:

- raw tensor addresses
- live backend pointers
- scheduler-owned objects
- allocation-pass-owned transient state
- temporary lowering arenas
- per-invocation data pointers

If a compiled context bakes a stride or layout value, that value must be represented in the key or validated on every bind.

---

## 5. Compile vs bind

Compilation and binding are separate.

Compile-time responsibilities:

- validate structural shape
- build region/step layout
- allocate frame/template metadata
- store stable constants such as shape, stride, and work ranges

Bind-time responsibilities:

- patch live tensor pointers
- patch frame-local scratch pointers
- patch output pointers
- ensure no context points to temporary compose/lowering state
- ensure no context retains pointers from a previous invocation

A reusable plan or packet must not retain stale pointers into stack-local lowering state, temporary composer structs, or a previous invocation’s frame.

---

## 6. Fine-region and packet internals

Fine-region kernels and packet execution steps must not re-enter full graph execution.

Forbidden inside `run_range(...)` kernels and packet steps:

- `ggml_graph_compute_thread_run(...)`
- `ggml_graph_compute(...)`
- `ggml_backend_graph_compute(...)`
- reliance on ggml barrier participation
- reliance on implicit ggml worker identity

Allowed:

- direct CPU kernel calls
- explicit range partitioning
- explicit resources
- direct dependency edges
- local scratch/reduction buffers

The fallback/backend run layer is the exception. It may call the backend because backend work is its purpose.

---

## 7. Backend safety

Do not run HPX-lowered CPU code on nodes intended for another backend.

Mixed-backend graphs must be rejected or handled by opaque backend/fallback runs. Unified memory is not proof of CPU ownership.

A selective or run-level path must fail closed to the normal scheduler when backend ownership is unclear.

---

## 8. BLAS and delegated work

BLAS and delegated backend work remain opaque.

HPX may schedule before or after delegated work and represent dependencies around it.

HPX must not:

- decompose BLAS internals
- assume BLAS worker behavior
- plan inside a delegated backend kernel

---

## 9. Abort and cancellation

Abort is cooperative, not preemptive.

Abort means:

- no new region starts after abort is observed
- dependent follow-on work is not launched after abort
- running CPU work stops only at safe checkpoints
- in-flight BLAS/delegated work is not forcibly interrupted

Check abort before starting a run or region, before launching dependent work, and inside long CPU loops at safe checkpoints.

---

## 10. HPX usage

Use HPX for structured work.

Preferred:

- run-level execution
- packet-level dispatch
- region groups with meaningful work
- dependency-aware execution where dependencies matter
- stable executor/runtime ownership
- one terminal wait after a batched/plan-level dispatch

Avoid:

- HPX per cheap node
- repeated native-to-HPX crossings inside a node loop
- `hpx::async(...).get()` as a per-node bridge
- nested HPX fan-out before the outer unit is proven useful
- executor/runtime creation on the hot path

`.get()` and `wait_all` are not automatically wrong. They are wrong when they turn the hot path into repeated hard synchronization around tiny work.

---

## 11. Experiments and flags

Experimental paths must be:

- env-gated or compile-gated
- off-safe
- measurable
- reversible
- explicit in stats

A new packet/lowering/run path must document:

- exact match conditions
- fallback behavior when conditions fail
- stats/accounting
- correctness validation against baseline
- no regression when disabled

---

## 12. Do not do this

Do not:

- reintroduce HPX per tiny decode node
- write another packet before the run-level analyzer exists
- start from prefill-first assumptions without topology evidence
- revive known-slow parallel-projection work without new evidence
- treat old provenance notes as current design contracts
- use ephemeral pointers in reusable keys
- leave packet or run contexts bound to temporary compose/lowering scratch
- call backend graph compute from fine-region kernels or packet internals
- claim live speedup without direct measurement
- weaken CPU-only/backend ownership guards

---

## 13. May evolve

These may evolve:

- work-size thresholds
- packet matchers
- packet key layout
- run kinds
- run coalescing policy
- direct `run_range` kernel coverage
- fine-region scheduling policy
- HPX executor choice
- resource ownership details

They must evolve without violating the durable rules above.

---

## Short summary

- Use HPX for large or structured work.
- Do not use HPX per tiny decode node.
- Build structural plans before execution.
- Keep plans and packet keys structural.
- Separate compile from bind.
- Keep backend and BLAS work opaque.
- Keep fine-region and packet internals out of full graph re-entry.
- Fail closed when backend ownership is unclear.
- Measure before claiming speedup.
