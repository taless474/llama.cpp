# README_HPX.md — HPX CPU Execution for llama.cpp

## Overview

This project redesigns the CPU execution side of `llama.cpp` using HPX.

It started with HPX-based orchestration above ggml, then moved into
`ggml-cpu` executor ownership, and now focuses on an HPX-native execution path
for explicit CPU work regions under `ggml/src/ggml-hpx/`.

The goal is not to replace ggml kernels or BLAS. The goal is to improve how CPU
work is:

- represented
- scheduled
- dispatched
- executed

The most important direction is no longer “HPX as orchestration.”
It is **HPX as the execution model for explicit CPU work regions and matched
repeated sublayers**.

---

## Core idea

The key question is not “pthread vs HPX.”

It is:

> What is the right execution contract for inference?

The project has evolved through four layers:

1. HPX above ggml as orchestration
2. HPX as a CPU executor substrate inside `ggml-cpu`
3. HPX as a direct executor of explicit fine-grained CPU work regions
4. Selective llama-integrated lowering and packet dispatch experiments

Today, layers 3 and 4 matter most.

---

## Current architecture

### Coarse CPU substrate policy

```text
small work (decode-like)
  → pthread substrate

large work (prefill-like)
  → HPX substrate
```

Routing is based on `cplan->work_size`.

### Fine-region execution layer

```text
coarse graph region
  → fine CPU region DAG
  → run_range(...)
  → HPX futures / dataflow
```

This path executes explicit CPU work units directly instead of routing through
`ggml_graph_compute_thread_run(...)`.

### Selective llama bridge

```text
real llama graph
  → selective matcher
  → lowered nodes / packetized sublayers / CPU fallback
```

This is the current bridge from HPX fine-region machinery into real llama.cpp
execution.

---

## What is implemented

### 1. HPX executor substrate inside `ggml-cpu`

- executor/job ownership split
- explicit executor seam (`init`, `run_job`, `destroy`)
- HPX-backed threadpool implementation
- work-size-based pthread / HPX substrate split

### 2. Fine-region DAG contract

Implemented under `GGML_HPX_REGION_DAG`:

- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`
- `ggml_hpx_dep_edge`
- `ggml_hpx_region_resources`
- `ggml_hpx_run_range_fn`

This defines explicit CPU work DAGs with:

- work ranges
- dependencies
- resource bundles

### 3. Fine-region execution

Implemented in `ggml-hpx-region-exec.*`:

- region-group validator
- direct F32 `mul_mat`
- direct F32 elementwise kernels:
  - `SiLU`
  - `MUL`
  - `SWIGLU`
- `ggml_hpx_run_single_region(...)`
- `ggml_hpx_run_region_group(...)`

Properties:

- direct `run_range(...)` execution
- no graph re-entry
- dependency-driven scheduling with HPX futures / `dataflow`
- real same-level overlap

### 4. Selective graph-level mixed execution

A selective execution path exists for real llama graphs:

- supported lowered ops use the fine-region path
- unsupported ops fall back to the live CPU backend
- reduction-containing groups are rejected from selective lowering
- selective stats and debug output are wired into the llama path
- mixed-backend graphs are excluded by a CPU-only guard

The selective path is valid only on CPU-only scheduled graphs.

### 5. Frozen-packet execution

A packetized path is integrated behind env gating:

- packet runtime is lazily created
- packet caches are owned by `llama_context`
- packet dispatch is decode-only in first deployment
- packet stats are reported through the selective stats stream

Packetization is meant for repeated steady-state sublayers where generic
fine-region DAG dispatch has a fixed overhead floor.

---

## Current validated behavior

### Fine-region tests

The core fine-region tests pass:

- `test_hpx_region_group_validate`
- `test_hpx_region_group_run`
- `test_hpx_region_group_parallel`
- `test_hpx_region_single_mul_mat`
- `test_hpx_region_mixed_mul_mat`

These prove:

- malformed groups are rejected
- small DAGs execute correctly
- same-level overlap is real
- direct region `mul_mat` works
- mixed lowered/fallback execution works

### Selective path correctness

The selective llama path is validated for:

- correct live-backend fallback
- safe rejection of reduction groups
- CPU-only guard on mixed Metal+CPU graphs
- stable stats/debug reporting

### Packet mechanism

The first packet path was proven in isolation with tests covering:

- exact-match recognition
- compile-once cache behavior
- bind-per-dispatch from live tensor storage
- numerically correct execution
- cache reuse on repeated dispatch

---

## Milestone status

### v1 — complete

v1 established the first llama-integrated packet path and proved it in
isolation. On TinyLlama `Q4_K_M`, packet dispatch stayed dormant because the
model did not present a compile-able F32 packet pattern. This was an honest
integration result, not a speedup result.

### Milestone A — complete

A asked a narrow question:

> Can packet dispatch fire inside real llama execution?

To test that, TinyLlama `Q4_K_M` was dequantized to an F32-typed GGUF.
This activated F32 `MUL_MAT` lowering, but packet dispatch still did not fire.

Root cause:

- the original packet matcher expected
  `MUL_MAT, MUL_MAT, SiLU, MUL`
- real llama graphs emit
  `MUL_MAT, MUL_MAT, GLU[SWIGLU]`

So A closed on a graph-shape finding, not a speedup.

### Milestone B.1 — complete

B.1 aligned the packet path with the real llama graph shape and proved
it on real decode.

Items delivered:

1. `SWIGLU_F32` primitive and tests.
2. lowering support for `GGML_OP_GLU` with `GGML_GLU_OP_SWIGLU`.
3. GLU-aware composer for:
   - `MUL_MAT(W_gate, x)`
   - `MUL_MAT(W_up, x)`
   - `GLU[SWIGLU](gate, up)`
4. `MLP_GLU_F32` packet sublayer (typed binding, compile, bind, narrow
   packet test).
5. Selective matcher/cache/env integration for the GLU packet path.
6. `llama_context` wiring for the GLU packet cache alongside the
   existing gate/up cache, lazy-created together on first eligible
   decode `graph_compute`.

Real-llama activation, TinyLlama F32 CPU-only decode:

- `packet=22(66 nodes)` per decode graph — every MLP block fires
- the gate/up matcher does **not** fire on real llama graphs, because
  SWIGLU fusion subsumes the older 4-node pattern; it is exercised
  only by isolation tests

Honest end-to-end speedup on a quiet machine, measured under the fair-
comparison protocol in `CLAUDE.local.md`:

- PACKET=0: 4.05 tok/s (247.1 ms/token)
- PACKET=1: 4.55 tok/s (219.8 ms/token)
- **Decode speedup: 1.12× (~12%)**

The earlier 2× reading from the 2026-04-18 smoke was a measurement
artifact of CPU contention, not a property of the packet path. The
investigation that closed B.1 is documented in
`docs/HPX_PROVENANCE.md` section 12 and in
`hpx-bench/results/2026-04-20-*/`.

Instrumentation left in place: `LLAMA_HPX_PACKET_COMPILE_LOG=1`
(env-gated stderr log at both compile sites; zero overhead when
unset; compile cost measured at ~1 µs per process).

### Milestone B.2 — future

B.2 is separate from B.1.

Its goal is to make the packet path useful on realistic quantized models by
adding quantized `MUL_MAT` lowering or an equivalent compose path.

---

## Performance takeaway

The current message is not “HPX is faster everywhere.”

It is:

- coarse routing protects small decode-like work
- HPX stays competitive on large prefill-like work
- fine-region execution gives a better execution model for explicit CPU work
- packetized execution is the right low-overhead surface for repeated matched
  sublayers
- the remaining challenge is matching realistic quantized workloads; the
  real llama graph shape is now matched as of B.1

First honest end-to-end packet result on CPU-only F32 TinyLlama decode:
**1.12× speedup**, obtained by moving the fused GLU MLP sublayer (66 nodes
per decode graph, 22 matches) from the generic lowered path into a frozen
packet. That number was measured under the fair-comparison protocol in
`CLAUDE.local.md`.

Packet compile cost is ~1 µs per process. No warmup cliff is present on a
quiet machine.

---

## Where to look

### Core implementation

- `ggml/src/ggml-hpx/`
  - lowering, region execution, packet code, selective execution, runtime code
- `ggml/src/ggml-cpu/`
  - CPU executor seam and substrate work

### Tests

- `tests/hpx/`
  - fine-region tests
  - primitive tests
  - packet tests
  - selective tests
  - llama smoke tests

### Benchmarks and logs

- `hpx-bench/`
  - microbenchmarks
  - selective/packet smoke result directories

### Docs

- `README_HPX.md`
  - current architecture and status
- `docs/HPX_EXECUTOR_CONTRACT.md`
  - design rules and execution boundaries
- `docs/HPX_PROVENANCE.md`
  - chronological project history and results

---

## Current limitations

- direct lowering coverage is still selective
- packet dispatch is activated on F32 CPU-only TinyLlama decode; broader
  model families and dtypes remain to be validated
- realistic quantized packet activation is B.2 scope
- measured F32 decode speedup is 1.12× on TinyLlama; larger gains are the
  subject of B.2 and later milestones
- benchmarking is sensitive to CPU contention and thermal state; follow
  the fair-comparison protocol in `CLAUDE.local.md`

---

## One-line summary

This project moves llama.cpp CPU execution from a thread-centric model toward a
structure-aware model where HPX executes explicit CPU work regions directly,
selectively lowers real llama graphs, and incrementally adds packetized
execution for repeated sublayers that match the model’s actual graph shape.
