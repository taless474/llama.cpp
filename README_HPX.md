# README_HPX.md — HPX CPU Execution for llama.cpp

## Purpose

This project explores an HPX-backed CPU execution path for `llama.cpp` / `ggml`.

The goal is **not** to replace ggml kernels, BLAS, or model logic. The goal is
to change how CPU work is represented and dispatched:

- from whole-graph threadpool execution,
- to explicit CPU work regions,
- to selective lowered execution,
- and, when useful, to packetized repeated sublayers.

The main design question is:

> What is the right execution unit for inference?

For this project, the answer has moved away from “one HPX task per ggml node.”
The useful direction is larger, structure-aware units: contiguous fallback runs,
fine-region DAGs, and frozen packets for repeated graph patterns.

---

## Architecture map

```text
llama.cpp graph_compute
        |
        v
ggml backend scheduler
        |
        +--> normal ggml CPU backend
        |
        +--> HPX paths, when enabled
                |
                +--> coarse CPU substrate
                |       ggml-cpu threadpool seam backed by HPX
                |
                +--> fine-region DAG
                |       explicit CPU regions + dependencies + run_range kernels
                |
                +--> selective executor
                |       lowered nodes + coalesced CPU fallback runs + packets
                |
                +--> frozen packets
                        compile once, bind per call, dispatch repeated sublayers
```

---

## Execution layers

### 1. Coarse CPU substrate

This was the first practical integration point inside `ggml-cpu`.

It added an executor seam around CPU graph execution:

```text
executor init
job dispatch
executor destroy
```

The HPX-backed substrate can run ggml's normal CPU worker function through HPX
workers. This path is useful as a compatibility layer and as a coarse-grained
substrate experiment.

Important lesson:

```text
HPX around the whole CPU job can be reasonable.
HPX around every tiny node is usually too expensive.
```

### 2. Fine-region DAG

The fine-region layer represents explicit CPU work directly:

```text
region group
  region[0]
  region[1]
  dependency edges
  shared resources
  run_range callbacks
```

Core concepts:

- `ggml_hpx_cpu_region`
- `ggml_hpx_cpu_region_group`
- `ggml_hpx_dep_edge`
- `ggml_hpx_region_resources`
- `ggml_hpx_run_range_fn`

This layer avoids re-entering `ggml_backend_graph_compute(...)` for work that
has already been lowered into explicit CPU kernels.

Use this layer when the operation has a safe direct kernel and the work unit is
large enough to justify HPX scheduling.

### 3. Selective executor

The selective executor is the bridge into real llama graphs.

For each scheduled CPU-only graph, it chooses among:

```text
packetized sublayer
lowered fine-region op
coalesced CPU fallback run
```

The selective path must preserve graph order and backend correctness. It should
only run on CPU-only scheduled graphs. Mixed backend graphs, such as Metal + CPU,
must stay on the normal backend path unless explicitly proven safe.

Important design rule:

```text
Selective execution should reduce dispatch count, not multiply it.
```

A selective path that slices the graph into many one-node HPX submissions is the
wrong shape, even if each individual lowered kernel is correct.

### 4. Frozen packets

A packet is a compiled execution template for a repeated graph pattern.

A packet has:

```text
matcher / prescan
plan key
compiled packet
frame / scratch
bind step
single dispatch
stats
```

Packets are useful when a subgraph appears repeatedly with stable structure,
tensor types, and shapes. They are meant to reduce generic DAG overhead and
avoid repeated per-node native-to-HPX crossings.

A packet should not mean “some adjacent nodes.” It should mean:

```text
known pattern
known constraints
safe ownership of intermediates
compile once
bind many times
same output as baseline
```

---

## Build-time gates

The exact CMake option names may evolve, but the implementation is guarded by
these project-level concepts:

| Gate | Purpose |
|---|---|
| `GGML_HPX` | Enables HPX-related code paths. |
| `GGML_HPX_REGION_DAG` | Enables fine-region DAG, lowering, selective execution, and packet code. |

When adding new HPX code, keep non-HPX builds clean. Headers should avoid
leaking HPX dependencies into unrelated llama.cpp code.

---

## Runtime flags and diagnostics

Common runtime environment variables used by the HPX path:

| Variable | Purpose |
|---|---|
| `LLAMA_USE_HPX=1` | Opt into the HPX execution path where wired. |
| `LLAMA_HPX_SELECTIVE_STATS=1` | Print selective execution counters. |
| `LLAMA_HPX_SELECTIVE_DEBUG=1` | Print selective routing/debug information. |
| `LLAMA_HPX_SELECTIVE_HIST=1` | Print fallback/lowering histograms when available. |
| `LLAMA_HPX_SELECTIVE_MLP_PACKET=1` | Enable existing MLP packet matching/dispatch where supported. |
| `LLAMA_HPX_PACKET_COMPILE_LOG=1` | Log packet compile/cache behavior. |

Rules for flags:

- flags should default to off unless the path is proven safe;
- diagnostics must be cold when disabled;
- packet-specific flags should be named by packet or packet family;
- stats must distinguish lowered nodes, fallback nodes, packet matches, and
  packet-consumed nodes.

---

## Correctness rules

HPX paths must preserve baseline llama.cpp behavior.

Required checks:

- CPU-only guard before selective execution;
- no mixed-backend contamination;
- fallback to live CPU backend for unsupported ops;
- no packet match unless all tensor-type, shape, stride, and ownership
  constraints hold;
- no skipped node unless its output is produced by the replacement path before
  any consumer can read it;
- output parity against selective-off / baseline before performance claims.

For packets, the matcher is part of correctness. A packet that matches too
broadly is worse than no packet.

---

## Performance rules

Do not claim HPX speedup from isolated kernel success alone.

The measured unit must match the claim:

```text
kernel benchmark       -> kernel claim only
packet microbenchmark  -> packet claim only
llama decode run       -> live decode claim
prefill run            -> prefill claim
```

Main performance risks:

- one HPX submission per ggml node;
- repeated `hpx::async(...).get()` in a graph loop;
- many one-node fallback `graph_compute` calls;
- rebuilding packet structure instead of compile-once / bind-many;
- nested HPX tasks inside packets before the outer packet dispatch is proven.

Preferred progression:

```text
coalesce fallback runs
batch lowered work where safe
packetize repeated sublayers
only then add internal packet parallelism
```

---

## Where the code lives

```text
ggml/src/ggml-hpx/
  runtime, adapter, planning/cache, fine-region execution,
  lowering, selective execution, packet code

ggml/src/ggml-cpu/
  CPU executor seam and substrate integration

tests/hpx/
  region tests, lowering tests, packet tests, selective tests

hpx-bench/
  microbenchmarks, llama smoke runs, result directories
```

---
## Command caution on llama-simple

For `llama-simple`, the prompt is positional.

Correct:

```bash
./build-hpx-dag/bin/llama-simple \
  -m model.gguf \
  -n 32 -ngl 0 \
  "Hello, my name is"
```

Do not use `-p` with `llama-simple`.
---

## Graph-map diagnostic

When selective behavior is unclear, inspect one decode graph as an execution-unit map:

```text
idx | op | name | shape | src0 type/trait | category L/F/P | run id | reason
```

Where:

```text
L = lowered by HPX
F = fallback to ggml
P = packet
```

Use this to identify fallback islands, lowered islands, packet opportunities, and unexpected dispatch fragmentation.

---
## Packet guidance

Do not call something a packet just because several nodes are adjacent. A packet should be a repeated, validated graph pattern with safe ownership of its intermediate tensors and a clear reduction in execution entries.

Current graph-map evidence shows a strong first packet candidate:

```text
MUL_MAT ffn_gate-N
MUL_MAT ffn_up-N
GLU/SWIGLU ffn_swiglu-N
```

This appears once per TinyLlama layer. For a first packet design, prefer the gate/up/GLU pattern and do not include `ffn_out` until its mixed Q4_K/Q6_K behavior is handled deliberately.

A useful first-cut packet should aim to own the pattern, not simply run gate/up through fallback and packetize only GLU.

---
## Result logging convention

Do not write benchmark or experiment results to `/tmp`.

Use:

```text
hpx-bench/results/<date>-<slug>/
local/results/
```

Each result directory should include:

```text
stdout or CSV
stderr / bench log
README.md
commit hash
binary path
exact command
findings
open questions
```
Each run gets its own directory with:

- raw stdout or CSV
- stderr / bench log
- `README.md`

The `README.md` should include:

- what was measured
- exact command or protocol
- key numbers
- findings and open questions
- git commit hash
- binary path

Redirect output at invocation time. Do not write to a temporary location and copy results afterward.

---

## One-line summary

This project moves llama.cpp CPU execution toward a structure-aware HPX model:
coarse substrate where useful, explicit fine-region DAGs for lowered CPU work,
selective graph execution for real llama graphs, and frozen packets for repeated
sublayers where one compiled dispatch can replace many small execution units.
