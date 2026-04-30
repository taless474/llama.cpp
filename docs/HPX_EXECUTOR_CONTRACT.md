# HPX Executor Contract for ggml/llama.cpp

## Purpose

This is a compact rulebook for HPX-related CPU execution code in this fork.
It is not a provenance log, benchmark report, or milestone tracker.

Use:
- `README_HPX.md` for the project map and flags.
- `docs/HPX_PROVENANCE.md` for history and measurements.
- this file for durable execution rules and review criteria.

---

## Scope

Do not apply rules from one HPX path blindly to another.

The active paths are:

1. **ggml-cpu substrate path**  
   HPX-backed threadpool/substrate integration under the ggml CPU executor seam.

2. **Fine-region path**  
   Explicit CPU work regions using `run_range(...)`, dependency edges, and resource bundles.

3. **Selective executor path**  
   Real llama graphs are routed to lowered HPX regions, packetized sublayers, or CPU backend fallback.

4. **Packet path**  
   Repeated sublayers are matched, compiled once, bound per invocation, and dispatched as larger execution units.

The project direction is not “replace pthread with HPX everywhere.” It is:

```text
Use HPX where the execution unit is large and structured enough to justify it.
Avoid HPX per tiny decode node.
```

---

## 1. Substrate selection

For the **ggml-cpu substrate path**, substrate selection is based on:

```cpp
cplan->work_size
```

Default policy:

```text
small work -> pthread substrate
large work -> HPX substrate
```

This protects small decode-like work from accidental HPX overhead.

Do not generalize this to “HPX is never allowed in decode.” Packetized decode work may use HPX when it reduces dispatch granularity and is validated by measurement.

---

## 2. Decode and prefill

Decode and prefill are operational regimes, not reliable graph-shape labels.

Guidance:

```text
whole-graph decode-sized substrate work -> avoid HPX unless proven beneficial
large prefill-like substrate work        -> HPX may be appropriate
matched packetized decode work           -> HPX may be appropriate
```

Forbidden hot-path shape:

```text
native llama thread
  -> one tiny node
  -> hpx::async(...).get()
  -> next tiny node
  -> hpx::async(...).get()
```

Use HPX around meaningful regions, fallback runs, or packetized sublayers.

---

## 3. Execution units

### Coarse substrate dispatch

Used by the ggml-cpu substrate path. Examples: whole CPU graph job, scheduler split, large backend work unit. This path may use pthread or HPX depending on `work_size`.

### Fine CPU regions

Used by the HPX-native direct execution layer:

```cpp
run_range(ctx, ith, nth, begin, end, resources)
```

Fine-region code must operate on explicit ranges and explicit resources.

### Selective fallback runs

The selective executor may fall back to the live CPU backend for unsupported nodes or node runs. Fallback re-entry is allowed only in the selective fallback path. Adjacent fallback nodes should be coalesced when correctness allows.

### Packets

Packets are for repeated model sublayers with known op pattern, tensor types, shape/layout key, and scratch/resource ownership. They should reduce dispatch granularity and must not become thin wrappers around one tiny node.

---

## 4. Structural plans and packet keys

Reusable plans and packet keys must be structural.

Allowed key material:
- operation pattern / sublayer kind
- shape
- dtype / trait class
- row stride or layout stride when baked into compiled ctx
- lane count or execution policy when it affects code shape
- policy version

Forbidden key material:
- raw tensor addresses
- live backend pointers
- scheduler-owned objects
- allocation-pass-owned transient state
- temporary lowering arenas
- per-invocation data pointers

If a compiled packet ctx bakes a stride or layout value, that value must be represented in the key or validated on every bind.

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
- ensure no ctx points to a temporary compose/lowering arena

A packet must not retain stale pointers into `ggml_hpx_lowering::scratch`, temporary composer structs, stack-local lowering state, or a previous invocation’s frame.

---

## 6. Fine-region and packet internals

Fine-region kernels and packet internals must not re-enter full graph execution.

Forbidden inside `run_range(...)` kernels and packet execution steps:
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

The selective fallback path is the exception. It may call the CPU backend, but fallback runs should be coalesced when possible.

---

## 7. Scheduler isolation

Scheduler/backend coupling must remain mechanically isolated.

For the coarse planning path, scheduler-facing translation belongs in:

```text
ggml-hpx-adapter.cpp
```

Do not leak scheduler internals into plan/cache code, region definitions, fine-region execution, packet code, abort code, or general runtime ownership code.

Selective execution may inspect the real llama graph and tensor metadata, but unstable scheduler dependencies should still be isolated behind a small boundary.

---

## 8. Backend safety

Selective lowering and packet dispatch are valid only when backend ownership is known to be safe.

Do not run HPX-lowered CPU code on nodes intended for another backend. Mixed-backend graphs must be rejected or fall back safely.

Unified memory is not proof of CPU ownership.

---

## 9. BLAS and delegated work

BLAS and delegated backend work remain opaque.

HPX may schedule before or after delegated regions and represent dependencies around them.

HPX must not decompose BLAS internals, assume BLAS worker behavior, or plan inside a delegated backend kernel.

---

## 10. Abort

Abort is cooperative, not preemptive.

Abort means:
- no new region starts after abort is observed
- dependent follow-on work is not launched after abort
- running CPU work stops only at safe checkpoints
- in-flight BLAS/delegated work is not forcibly interrupted

Check abort before starting a region, before launching dependent work, and inside long CPU loops at safe checkpoints.

---

## 11. HPX usage

Use HPX for structured work.

Preferred:
- packet-level dispatch
- region groups with meaningful work
- coarse prefill-like substrate work
- dependency-aware execution where dependencies matter
- stable executor/runtime ownership

Avoid:
- HPX per cheap node
- repeated native-to-HPX crossings inside a node loop
- `hpx::async(...).get()` as a per-node bridge
- nested HPX fan-out before the outer unit is proven useful
- executor/runtime creation on the hot path

`.get()` and `wait_all` are not automatically wrong. They are wrong when they turn the hot path into repeated hard synchronization around tiny work. A final terminal wait after a batched dispatch may be acceptable.

Executor choice may evolve. Do not hard-code a preferred HPX executor in this contract unless current code and benchmark evidence agree.

---

## 12. File responsibilities

### Docs

- `README_HPX.md` — project map, flags, and how to run.
- `docs/HPX_EXECUTOR_CONTRACT.md` — durable rules and boundaries.
- `docs/HPX_PROVENANCE.md` — history, experiments, dead ends, and measurements.

### Code

- `ggml-hpx-runtime.*` — HPX runtime ownership and executor support.
- `ggml-hpx-tpool.*` — HPX-backed ggml threadpool/substrate integration.
- `ggml-hpx-adapter.*` — scheduler/backend topology translation for the coarse path.
- `ggml-hpx-plan.*` — structural planning for the coarse path.
- `ggml-hpx-cache.*` — structural cache for the coarse path.
- `ggml-hpx-exec.*` — coarse-region orchestration path.
- `ggml-hpx-region-dag.*` — fine-region, dependency, and resource data structures.
- `ggml-hpx-region-exec.*` — fine-region validation and direct `run_range(...)` kernels.
- `ggml-hpx-lower.*` — lowering from supported ggml ops to fine-region groups.
- `ggml-hpx-compose.*` — composition of matched sublayers into reusable region groups.
- `ggml-hpx-packet.*` — frozen packet compile/bind/dispatch surface.
- `ggml-hpx-exec-selective.*` — selective execution, fallback routing, packet matching, and packet cache integration.

---

## 13. Experiments and flags

Experimental paths must be:
- env-gated or compile-gated
- off-safe
- measurable
- reversible
- explicit in stats

A new packet/lowering path must document:
- exact match conditions
- fallback behavior when conditions fail
- stats/accounting
- correctness validation against baseline
- no regression when disabled

---

## 14. Do not do this

Do not:
- merge pthread and HPX workers into one active mixed substrate
- remove `work_size` routing for the substrate path without replacement evidence
- leak scheduler internals outside the adapter boundary
- reintroduce HPX per tiny decode node
- treat provenance notes as current design contracts
- use ephemeral pointers in reusable keys
- leave packet ctx pointers bound to temporary composer/lowering scratch
- call backend graph compute from fine-region kernels or packet internals
- claim live speedup without direct measurement

---

## 15. May evolve

These may evolve:
- work-size thresholds
- packet matchers
- packet key layout
- direct `run_range` kernel coverage
- fine-region scheduling policy
- HPX executor choice
- resource ownership details
- fallback coalescing policy

They must evolve without violating the durable rules above.

---

## Short summary

- Use pthread for small whole-graph substrate work.
- Use HPX for large or structured work.
- Do not use HPX per tiny decode node.
- Keep scheduler coupling isolated.
- Keep plans and packet keys structural.
- Keep BLAS/delegated work opaque.
- Keep fine-region and packet internals out of full graph re-entry.
- Use packets for repeated sublayers large enough to justify HPX dispatch.
