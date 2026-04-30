# HPX run-level executor prototype — branch research

**Date:** 2026-04-30
**Branch:** `hpx-prefill-orchestrator`
**Reference commit:** `7f05dcc2b` (CPU_REPACK Q4_Kx8 lowering + tests + scratch fix + fallback-run coalescing + fallback_runs counter)
**Mode:** read-only synthesis. No code change proposed for this branch.

The user-requested filename is preserved verbatim (`refill-summary.md`).

---

## 1. Executive summary

The current selective HPX path on this branch is correct end-to-end on
real llama (TinyLlama-1.1B Q4_K_M) and goes through three execution
shapes per node (lowered fine-region group / coalesced CPU fallback run /
frozen packet). It is also **structurally slow**: ~10–13 tok/s vs.
~99–103 tok/s for the plain scheduler path on the same model, hardware,
and weights. Per-token instrumentation shows the cost is dispatch
fragmentation, not kernel quality:

- 689 ggml nodes per decode graph (TinyLlama-1.1B Q4_K_M, 22 layers).
- After fallback-run coalescing: **135 lowered nodes + 102 fallback
  runs + 22 packet matches = 259 execution units per decode**, each
  starting on the native llama-context thread and bridging into HPX
  via `hpx::async([&]{...}).get()` or
  `ggml_backend_graph_compute(cpu_be, view)`.
- The plain scheduler does it as **1 unit per decode**.

Packetization (Q4_Kx8 MLP gate/up/SWIGLU) helped — A/B inside the
selective regime: **22(22 nodes) → 22(66 nodes), 6.67 → 9.78 tok/s,
+47% within selective, bit-identical output through n=128.** It does
not close the 10× gap to the scheduler; another handcrafted packet
will not change that conclusion.

The prefill executor, on the other hand, found **no inter-region
parallelism to exploit** on either TinyLlama (45 regions, fully
linear) or Llama 3.1 8B (65 regions, fully linear). HPX serial is
neutral-to-slow; HPX parallel-proj is 38–47 % slower than the
scheduler at every prompt size measured.

The **run-level executor** direction is the right answer to both
findings: separate planning from execution, classify a graph shape
into a small list of larger run units (fallback_run / packet_run /
lowered_run / backend_run), cache the plan, bind per invocation, and
let HPX schedule the runs (not the nodes). This collapses the
native↔HPX crossing count from O(nodes) to O(runs) and gives a
single place to express dependencies between opaque backend tasks
(`graph_compute`, `FLASH_ATTN_EXT`, BLAS, etc.).

The prototype should be built clean from upstream `llama.cpp`. The
current branch is useful as **evidence and reference** — not as a
base to refactor. Sections 12–14 list what to reuse and what to
deliberately leave behind.

---

## 2. What the current branch proved

### 2.1 HPX integration, lifecycle, substrate selection

- `ggml/src/ggml-hpx/ggml-hpx-runtime.cpp:43` — process-wide HPX
  startup via `std::call_once(g_hpx_start_flag)` with
  `--hpx:queuing=static` and an `atexit` finalize/stop. This is the
  one-shot lifecycle CLAUDE.md requires; it works and should be
  carried over.
- `src/llama-context.cpp:2274` — dual-substrate ggml threadpool: HPX
  for graphs with `cplan->work_size >= 32 KiB`
  (`GGML_HPX_WORK_THRESHOLD`), pthread otherwise. Confirmed safe;
  "small-work routes to pthread" is durable.
- `src/llama-context.cpp:2303` — selective executor is gated on
  `LLAMA_USE_HPX=1` ∧ `LLAMA_HPX_SELECTIVE_MUL_MAT=1` ∧ `!batched`
  ∧ `n_splits == 1` ∧ first-node-on-CPU ∧ `should_engage(gf)`. Five
  guards, all of them necessary.

### 2.2 Fine-region DAG model is correct

- `ggml/src/ggml-hpx/ggml-hpx-region-dag.h:46-178` defines the
  contract: `ggml_hpx_cpu_region` (begin/end/grain/ctx/run_range/
  uses_resources) + `ggml_hpx_dep_edge` (src→dst) +
  `ggml_hpx_region_resources` (shared_scratch / lane_scratch[] /
  reduction_buffer / n_lanes).
- `ggml-hpx-region-exec.cpp:545` `ggml_hpx_run_region_group` runs a
  validated DAG: Kahn topo + `hpx::dataflow` over predecessor
  `shared_future`s + `wait_all` barrier. The validator
  (`ggml-hpx-region-exec.cpp:172` `ggml_hpx_validate_region_group`)
  catches null `run_range`, empty ranges, out-of-range edges, and
  cycles.
- The model is reusable; the *generic per-call DAG dispatch* is
  what the packet design replaced (see §6).

### 2.3 Selective executor enters real llama graphs

`ggml-hpx-exec-selective.cpp:1400` `ggml_hpx_exec_graph_selective_mul_mat`
walks `gf->nodes[0..n_nodes)` once and routes every node:

- packet trigger → bind + `hpx::async(run_frozen_packet).get()`
- `ggml_hpx_lower_op` succeeds (no resource-reduction) →
  `hpx::async(ggml_hpx_run_region_group).get()`
- otherwise → coalesced fallback run via
  `scan_fallback_run_end` + `ggml_graph_view(gf, i, j)` +
  `ggml_backend_graph_compute(cpu_be, &view)`

Output is **bit-identical** to scheduler output across all configs
(C / A / B) and stable through n=128
(`local/results/q4k8-edit5-validate/E-stability-n128.log`,
md5 `239988b2f1ef09a88af084e37e473632`).

### 2.4 Lowering coverage (`ggml-hpx-lower.cpp`)

Single op → fine-region group conversion. Supported:

- `GGML_OP_UNARY/SILU` → 1 ELEMENTWISE region
- `GGML_OP_MUL` → 1 ELEMENTWISE region
- `GGML_OP_MUL_MAT`
  - F32×F32 → 1 MATMUL region
  - Q4_K standard (no `extra`, rows == 1) → 2-region (REDUCTION
    quantize-to-Q8_K + MATMUL gemv)
  - Q4_K CPU_REPACK with trait `q4_K_8x8_q8_K`, rows == 1 → same
    2-region with the 8x8 gemv kernel
  - Other quant traits, mixed backends, mixed dtypes → reject
- `GGML_OP_RMS_NORM` (ne[1]==1) → 3-region (partial / finalize / apply)
- `GGML_OP_GLU/SWIGLU` → 1 ELEMENTWISE region

Unsupported nodes return false → fall through to the CPU backend
fallback. The trait predicate
(`ggml-hpx-lower.cpp:38` `ggml_hpx_is_q4k_8x8_repacked`) is the
single source of truth used by both the lowering site and the
prescan classifier — they must agree, by design.

### 2.5 Fallback-run coalescing

`ggml-hpx-exec-selective.cpp:1595` `scan_fallback_run_end` walks
forward from `start` and grows a contiguous fallback view across
nodes that either `lower_op` rejects or have a resource-dependent
REDUCTION region. One `ggml_graph_view(gf, i, j)` +
`ggml_backend_graph_compute` per run replaces `j-i` per-node calls.

Measured impact: **488 fallback nodes → 102 fallback runs** per
decode (4.78× consolidation). Output unchanged, stable through
n=128. This was the single largest selective-side improvement.

### 2.6 Packetization (compile-once / bind-many)

The packet surface (`ggml-hpx-packet.h` + `ggml-hpx-packet.cpp`) is
a compiled-and-frozen execution template:

- `ggml_hpx_compile_packet(fine_group, key, &out_err)` — produces
  an immutable `ggml_hpx_frozen_packet` from a validated
  `ggml_hpx_cpu_region_group`.
- `ggml_hpx_packet_frame` is per-invocation mutable state
  (caller-owned over-aligned storage).
- Typed bind functions per sublayer; bind never mutates the packet.
- `ggml_hpx_run_frozen_packet` walks a precomputed `step[]` array of
  SERIAL / LANE_FANOUT steps. No `hpx::async`, no `hpx::dataflow`,
  no `shared_future`, no per-step heap. Lane fan-out uses
  `hpx::experimental::for_loop` on a pinned per-team Exec built once
  at runtime create.
- Sublayer caches (`ggml_hpx_mlp_gate_up_packet_cache`,
  `ggml_hpx_mlp_glu_packet_cache`,
  `ggml_hpx_mlp_gate_up_glu_q4k8_packet_cache`) amortize
  compose+compile across decode tokens via shape-keyed lookup.
- Defined sublayers: `MLP_GATE_UP_F32`, `MLP_GLU_F32`,
  `MLP_GLU_QBRIDGE`, `MLP_GATE_UP_GLU_Q4K8`, `QKV_PROJ_Q4K8`
  (designed only). Identity is structural
  (`ggml_hpx_packet_plan_key`, 64 bytes, no padding).

The Q4_Kx8 packet is **live** on TinyLlama Q4_K_M decode: 22
matches per graph (one per layer), `lookup_or_compile_mlp_gate_up_glu_q4k8`
fires on every cache miss, then hits thereafter
(memory: `feedback_glu_is_live_packet_path.md`).

---

## 3. What failed or stayed slow

### 3.1 Decode selective vs. scheduler

From `local/results/q4k8-edit5-validate/README.md`, TinyLlama-1.1B
Q4_K_M, M4 CPU-only, prompt `"Hello, my name is"`, `-n 16`:

| config | per-graph counters | tok/s | vs C |
|---|---|---:|---:|
| C — scheduler | (no selective) | **102.78** | — |
| A — selective, packet env on, Q4K8 off | 135 lowered / 488(102 runs) fallback / 22(22 nodes) packet | 6.67 | 0.065× |
| B — selective, Q4K8 on | 135 lowered / 488(102 runs) fallback / 22(66 nodes) packet | 9.78 | 0.095× |

A→B improvement (+47% within selective) is real and bit-identical,
but selective is **~10× slower than scheduler** in absolute terms.
Stability: D (n=64) 13.07 tok/s; E (n=128) 12.95 tok/s — the gap is
not a warmup artifact.

### 3.2 Per-token dispatch tally (from `local/results/2026-04-28-hpx-selective-hot-path-audit.md`, pre-coalescing)

- 689 ggml nodes / decode
- 189 lowered + 500 fallback (uncoalesced) / token
- ~4,500 HPX scheduler entries / token (lowered node ≈ 10 entries
  inside `run_region_group`'s small-fast-path; fallback view ≈ 5
  entries via the HPX threadpool's `hpx_run_job` + bulk for_loop)
- **689 native → HPX crossings / token** (one per node)
- For comparison, scheduler-on: ~5 scheduler entries, 1 crossing /
  token. **~900× more HPX scheduler entries under selective-on.**

Post-coalescing the fallback side drops from 500 entries to ~102
entries per token (489 → 102 entries per token), but the lowered
side and the per-node `.get()` bridge are unchanged. Total per-token
crossings drops from 689 to ~259 — better, still 50× the scheduler.

### 3.3 The forbidden hot-path shape, in code

`ggml-hpx-exec-selective.cpp:1837-1900`, the lowered branch:

```cpp
if (ggml_hpx_lower_op(gf->nodes[i], &lo))
{
    ...
    hpx::async([&lo, &res]() {
        ggml_hpx_run_region_group(&lo.group, &res);
    }).get();          // native thread blocks until this node finishes
    ...
    ++stats.lowered_nodes;
}
```

This is exactly the shape `docs/HPX_EXECUTOR_CONTRACT.md §2`
forbids:

```
native llama thread
  -> one tiny node
  -> hpx::async(...).get()
  -> next tiny node
  -> hpx::async(...).get()
```

The packet branches at lines 1676, 1720, 1810 have the same
`hpx::async([...](){run_frozen_packet(...)}).get()` pattern. So even
when packets fire, every match is a fresh native↔HPX bridge.

### 3.4 Lowering ownership prevents async batching

`ggml_hpx_lowering lo;` is iter-local on the stack
(`ggml-hpx-exec-selective.cpp:1834`). The `.get()` at line 1895 is
load-bearing for ctx lifetime: launching the async without waiting
would let `lo` (and the ctx pointers it owns) go out of scope before
HPX runs the body. This intentional serialization rules out cheaply
pipelining across nodes.

### 3.5 Prefill (linear-chain) has no parallelism to find

From `local/results/HPX_PREFILL_REPORT.md` (TinyLlama-1.1B Q4_K_M
+ Llama-3.1-8B Q4_K_M, M4, CPU-only):

| model | n_nodes | n_regions | sched_splits | dependency chain |
|---|---:|---:|---:|---|
| TinyLlama-1.1B | 689 | 45 | 1 | fully linear (prev=i−1 ∀ i) |
| Llama-3.1-8B | 999 | 65 | 1 | fully linear (prev=i−1 ∀ i) |

Results — 3 runs each, mean ms (TinyLlama, 566 tok prompt):

- A no-HPX: 1646 ms (344 tok/s)
- B HPX serial: 1676 ms (338 tok/s) — neutral
- C HPX parallel-proj: 2312 ms (245 tok/s) — **+40 % slower**

Llama-3.1-8B (472 tok prompt, 32 layers GQA-8) is the same shape:
serial ~6 % slower, parallel-proj ~47 % slower. The parallel-proj
overhead is HPX dispatch + future synchronization per-layer; even
with GQA-8 the Q projection dominates the layer's critical path so
fan-out does not buy time.

The structural reason is in the topology dump itself: every region
depends on the previous one. A single CPU-backend forward pass over
a transformer is a serial chain at the region level, regardless of
model size. The current prefill executor cannot create work that
isn't there.

### 3.6 Coarse HPX path (older, in `ggml-hpx-exec.cpp`)

Coarse HPX-around-graph dispatch: works and is integrated, but the
production scheduler is already efficient for decode. Wrapping it
without changing the execution unit does not produce a speedup.
This was the first lesson and remains true.

---

## 4. Llama graph facts we should design around

From the graph-map diagnostic
(`hpx-bench/results/2026-04-29-selective-graph-map/README.md`,
TinyLlama-1.1B Q4_K_M, one decode graph, single token):

```
[hpx-map-summary] nodes=689
                  lowered_nodes=201   fallback_nodes=488   packet_nodes=0
                  lowered_runs=101    fallback_runs=102    packet_runs=0
```

(L/F here is the *would-lower* classification with packets
disabled; with the GLU/Q4K8 packets engaged this becomes 135 L /
488 F / 66 P with 22 packet runs.)

### 4.1 Per-decode pattern (one graph)

- 22 layers × ~31 nodes/layer + tail = 689 nodes.
- Run-length distribution (top buckets):

| count | cat | len | island shape |
|---:|---|---:|---|
| 57 | L | 1 | isolated MUL_MAT (Q-/K-/output projection) |
| 46 | F | 2 | `ADD,RMS_NORM` (residual + norm) |
| 22 | L | 2 | `MUL,MUL_MAT` (norm-mul + Q proj) |
| 21 | F | 14 | KV-cache plumbing: `RESHAPE,ROPE,VIEW,SET_ROWS,...` |
| 13 | F | 1 | single fallback (often V MUL_MAT, stray RESHAPE) |
| 12 | L | 5 | **GLU sublayer**: `MUL,MUL_MAT,MUL_MAT,GLU,MUL_MAT` |
| 11 | F | 4 | `RESHAPE,ROPE,MUL_MAT(V),RESHAPE` (V projection) |
| 10 | L | 4 | **GLU sublayer (no down)**: `MUL,MUL_MAT,MUL_MAT,GLU` |

### 4.2 Per-layer asymmetries

- **Q proj** (`q4_K_8x8_q8_K`) → lowerable (L island).
- **K proj** (`q4_K_8x8_q8_K`) → lowerable (L island).
- **V proj** (`q6_K`, repacked, **non-8x8 trait**) → falls back, sits
  inside an F-len=4 island. This is the recurring single-MUL_MAT
  fallback. Adding `q6_K_8x8` lowering would convert ~22 F-len=4
  islands to L-len=1 + F-len=2.
- **MLP gate/up** → both `q4_K_8x8_q8_K` → packetable end-to-end
  (`MLP_GATE_UP_GLU_Q4K8` does it); 22 sites/decode, 1 per layer.
- **MLP down** → mixed Q4/Q6 site (rejected at pre-scan).
- **RMS_NORM** → 3-region; works.
- **FLASH_ATTN_EXT, KV-cache plumbing** → opaque backend work
  (RESHAPE, ROPE, VIEW, SET_ROWS, PERMUTE). 21 occurrences ≈ once
  per layer; the recurring 14-node F-burst dominates the fallback
  count.

### 4.3 Backend ownership and the `tensor->extra` rule

`CLAUDE.md` invariant repeated by the V-proj finding: `tensor->type`
does NOT fully describe the physical layout. Every Q4_K/Q6_K weight
on M4 has `extra != nullptr` (CPU_REPACK) and uses
`q*_K_Nx8_q8_K` gemv kernels. The lowering predicate must always
check `extra` + `ggml_repack_extra_traits_name` before lowering a
quantized tensor.
(`feedback_cpp_style.md`, `project_q4km_baseline_path.md`.)

### 4.4 Prefill region structure

Both TinyLlama and Llama-3.1-8B prefill graphs:

- alternating CPU (~27 nodes) / BLAS (4 nodes) regions
- prev = i−1 for every region → fully linear
- sched_splits = 1 on `-ngl 0`

Implication: any "schedule independent regions" technique cannot
help the single-CPU-backend prefill path. The parallelism at prefill
is **inside** a region (lane-level row partitioning), not across
regions.

---

## 5. Current selective executor anatomy

### 5.1 Entry and gating (`src/llama-context.cpp`)

`graph_compute()` body, lines 2258-2466. Order of operations:

1. Choose ggml threadpool based on `cplan->work_size` (line 2274).
2. `LLAMA_HPX_SELECTIVE_HIST=1` warned-once histogram (line 2310).
3. CPU-only graph guard: `n_splits == 1` ∧ first node on CPU
   (line 2312-2330).
4. Policy guard `ggml_hpx_selective_should_engage(gf)`: would any
   MUL_MAT actually lower? If not, fall through to scheduler
   (line 2332-2354).
5. Lazy create `ggml_hpx_packet_runtime` + the three sublayer caches
   on first eligible call (line 2371-2431). Atomic group: if any
   creation fails, all three are torn down and the path runs
   packet-off.
6. Bundle into `ggml_hpx_selective_packet_env` and call
   `ggml_hpx_exec_graph_selective_mul_mat` (line 2446-2448).
7. Print stats via `LLAMA_HPX_SELECTIVE_STATS=1` and return
   (line 2449-2463).

### 5.2 Inside `ggml_hpx_exec_graph_selective_mul_mat`
(`ggml-hpx-exec-selective.cpp:1400-1924`)

Two-phase shape, both inside the same call:

**Phase A — pre-scan (lines 1444-1474):**
Three sequential passes over `gf->nodes[]` sharing one `consumed[]`
bitmap. Order matters:

1. `prescan_mlp_gate_up_glu_q4k8_matches` — Q4_Kx8 fused MLP, 3
   nodes per match (gate_mm + up_mm + glu).
2. `prescan_mlp_matches` — F32 gate/up, 4 nodes per match.
3. `prescan_mlp_glu_matches` — F32/QBRIDGE GLU, 3 nodes per match.

A match is recorded only after `lookup_or_compile_*` returns
non-null. The Q4_Kx8 prescan runs first because GLU triggers
overlap and the larger packet must claim its trigger before the
smaller GLU matcher can.

**Phase B — dispatch loop (lines 1628-1917):**
For each node `i`:

- `consumed[i]` true → check three trigger tables; dispatch the
  matching packet via `hpx::async([&]{run_frozen_packet(...)}).get()`,
  or skip silently if non-trigger member.
- Else `ggml_hpx_lower_op(node, &lo)` succeeds and no resource
  reduction → `hpx::async([&]{run_region_group(&lo.group, &res)}).get()`.
- Else (lower rejected, or resource-reduction reduction) →
  `j = scan_fallback_run_end(i+1)`, `view = ggml_graph_view(gf, i, j)`,
  `graph_compute(cpu_be, &view)`, `i = j-1`.

### 5.3 The structural problem in one sentence

Every iteration of this loop crosses native ↔ HPX, even after
fallback coalescing and packet matching. There is no point at which
the executor batches "all lowered nodes in this graph" into one
async submission, because the per-iteration `ggml_hpx_lowering` ctx
lives on the iteration's stack frame and the per-iteration `.get()`
is what keeps it alive.

### 5.4 Stats wired and what they tell us
(`ggml-hpx-exec-selective.h:68-86` `ggml_hpx_selective_stats`)

`lowered_nodes`, `fallback_nodes`, `fallback_runs`, `packet_matches`,
`packet_nodes`, `bridge_fallback_nodes`, plus `_ns` timers. These
are the right counters for diagnosing dispatch fragmentation: in B
above, `fallback_runs=102` directly exposes the run-level grain.
Carry these forward into the new design.

---

## 6. Packetization result and limitation

### 6.1 What the Q4_Kx8 packet replaces

Before: one `lower_op` per MUL_MAT (gate, up) → two redundant
quantize-to-Q8_K REDUCTIONs over the same activation. Then GLU as a
separate single-region lowered node. Three lowered groups, each
with its own native↔HPX bridge.

After (composer + frozen packet, 4 SERIAL steps in one packet):

```
step 0 SERIAL  quantize x → frame-owned Q8_K scratch
step 1 SERIAL  gate gemv (reads quantized x, writes gate)
step 2 SERIAL  up   gemv (reads quantized x, writes up)
step 3 SERIAL  swiglu  (reads gate + up, writes out)
```

One bridge instead of three. One quantize instead of two. Output
unchanged (`local/results/q4k8-edit5-validate/*.tokens.final`,
md5 stable across A/B/C).

### 6.2 Numbers (Edit-5 validation)

| metric | A: q4k8 off | B: q4k8 on |
|---|---:|---:|
| packet matches / graph | 22 | 22 |
| packet nodes / graph | 22 | **66** (+44) |
| avg lowered_ms | 94.26 | 61.46 |
| avg fallback_ms | 36.70 | 26.43 |
| avg packet_ms | 3.90 | 13.73 |
| eval (15 decodes, ms) | 2247.99 | 1533.34 |
| tok/s | 6.67 | **9.78** |

Bit-identical output. Stable through n=128 (E run, 127/127
packet=22(66 nodes) lines).

### 6.3 Why it is not enough

Even with 44 ex-fallback nodes absorbed into packets, every decode
still has:

- 135 lowered nodes → 135 `.get()` bridges
- 102 fallback runs → 102 substrate entries
- 22 packet matches → 22 `.get()` bridges

= 259 native↔HPX crossings per token. Adding QKV_PROJ_Q4K8 (designed
in `local/results/q4k8-edit5-validate/QKV-packet-design.md`,
6 SERIAL steps, all-Q4_Kx8 only, 22 sites) would shrink the L count
further, but the executor shape stays the same: one bridge per
unit. **The packet surface is the right tool; it is being driven by
a per-node loop.**

### 6.4 QKV packet status

Designed (header `QKV_PROJ_Q4K8` sublayer with full structural-key
spec including RMS_NORM eps and weight stride encoding); composer
not implemented. Held to pivot to run-level planning instead.
Memory: `project_hpx_option_b.md` (items 1–3 done, 4–6 open).

---

## 7. What "run-level executor" should mean

A **run** is a contiguous slice of the ggml graph that we treat as
**one execution unit** with one HPX submission. The current branch
already has all four kinds in code, but it picks one per node; the
new design picks them per **plan**.

```cpp
enum class hpx_run_kind {
    fallback_run,    // contiguous ggml_graph_view, opaque graph_compute
    packet_run,      // compiled packet, bind + dispatch
    lowered_run,     // one or more region groups in dataflow
    backend_run,     // delegated/opaque backend op (FLASH_ATTN_EXT, BLAS)
};
```

For each kind:

### 7.1 `fallback_run`

- **inputs**: a half-open ggml node range `[i, j)`
- **outputs**: every node in the range writes through to its
  ggml-side tensor data
- **ownership**: the graph view is borrowed; the CPU backend owns
  scratch; data pointers are live ggml allocator state
- **opaque?**: yes — internally a single `ggml_backend_graph_compute(
  cpu_be, &view)` call
- **bind**: nothing — view is recomputed per invocation from
  `gf->nodes` because the live tensor data pointers may have moved
- **execute**: one CPU-backend submission
- **dependencies**: declared on the previous run's "all outputs
  ready" (in the linear-chain decode case this is just the prior
  run's completion)

### 7.2 `packet_run`

- **inputs**: typed binding struct (e.g. `ggml_hpx_mlp_gate_up_glu_q4k8_binding`)
  with N data pointers
- **outputs**: 1–N output tensor data writes
- **ownership**: packet immutable, frame mutable (caller-owned),
  resources caller-owned (per-team Exec inside the runtime)
- **opaque?**: from the run-level view, yes — the packet's
  step list is internal
- **bind**: one bind call patches ctx-arena pointers (cheap)
- **execute**: `ggml_hpx_run_frozen_packet`
- **dependencies**: producers of every binding pointer must be done
  by the time bind is called; consumers of every output pointer
  must wait for completion

### 7.3 `lowered_run`

- **inputs**: a list of nodes whose `lower_op` succeeds
- **outputs**: tensor data writes per node
- **ownership**: ctx storage must live long enough for the async
  body to finish; the current branch puts it on the loop stack —
  the new design must lift it into the plan or the run
- **opaque?**: no — the run's region group is visible (we wrote it)
- **bind**: patch live tensor pointers into each region's ctx
- **execute**: `ggml_hpx_run_region_group` (existing) once per run,
  not once per node
- **dependencies**: explicit dep edges across nodes belong to the
  run, not the outer planner; cross-run edges are the planner's job

### 7.4 `backend_run`

- **inputs**: a single ggml node or small group whose operator is
  opaque to us by design (e.g. `FLASH_ATTN_EXT`, BLAS-bound MUL_MAT,
  Metal/CUDA delegate)
- **outputs**: tensor write(s)
- **ownership**: the delegated backend owns everything inside;
  we own only the dispatch
- **opaque?**: yes — by contract (`HPX_EXECUTOR_CONTRACT.md §9`)
- **bind**: nothing
- **execute**: dispatch through whichever backend the scheduler
  assigned (or a single-node `graph_compute` if it stays on the
  active CPU backend)
- **dependencies**: declared on its inputs' producer runs

### 7.5 Plan, cache, bind separation

```
analyze graph (once per shape)
    -> classify nodes into runs
    -> validate (every node landed in exactly one run)
    -> compile lowered_runs / packet_runs (compose region groups,
       compile packets via existing ggml_hpx_compile_packet)
    -> emit hpx_run_plan { vector<run>, dep edges, output map }
cache the plan keyed on graph shape (n_nodes, op sequence,
    dtype/trait sequence, ne/strides where they affect lowering)

execute plan (per call):
    -> bind: rewalk gf->nodes; per run, patch live data pointers
       into the run's frame/ctx (no compose, no compile)
    -> submit: HPX-schedule the runs respecting deps
    -> wait once at the end
```

**Compile-once / bind-many** is already true for packets in this
branch; the run-level plan generalizes that idea to lowered runs and
fallback runs.

---

## 8. What "async backend orchestration" should mean

In this branch's terms:

- HPX's job is to **schedule runs** and the **edges between runs**.
- HPX must **not** decompose the body of a `backend_run` or a
  `fallback_run`. Those are opaque.
- A `lowered_run` and a `packet_run` are HPX-native because we wrote
  their internals; the planner can still treat them as units at the
  run-level.

Examples mapped onto llama:

| llama-side work | run kind | opaque to HPX? |
|---|---|:---:|
| `ggml_backend_graph_compute(cpu_be, view)` over RESHAPE/ROPE/VIEW/SET_ROWS island | fallback_run | yes |
| `FLASH_ATTN_EXT` node | backend_run | yes |
| BLAS-bound MUL_MAT | backend_run | yes |
| Metal/CUDA-assigned node | backend_run / scheduler-side, not eligible | yes (rejected at engagement) |
| MLP_GATE_UP_GLU_Q4K8 sublayer | packet_run | mostly (we own the steps) |
| F32 RMS_NORM 3-region group | lowered_run | no (we own it) |

Difference from the current selective loop:

- today: one `hpx::async(...).get()` between the native thread and
  *every* run, immediately
- run-level: one HPX submission for the entire run plan, with
  inter-run deps expressed as `hpx::shared_future` or
  `hpx::dataflow`, and one terminal wait at the end of the call

For the linear-chain decode case (which is most of the work) this
collapses to a serial run sequence with at most one `.get()` per
graph_compute call.

---

## 9. Proposed fresh architecture

### 9.1 Layered structure

```
llama-context::graph_compute
   └─ hpx_run_executor
        ├─ analyze(gf) ────────────► hpx_run_plan (cached)
        ├─ bind(plan, gf)  ─────────► live data pointers patched
        └─ run(plan)
             └─ for each run in topological order:
                  fallback_run  → ggml_backend_graph_compute(view)
                  backend_run   → ggml_backend_graph_compute(node)
                  packet_run    → run_frozen_packet (existing)
                  lowered_run   → run_region_group  (existing)
             └─ single terminal hpx::wait_all
```

The shape is closer to the `ggml-hpx-exec.cpp` prefill path than to
the selective per-node loop, but with the run kinds from selective
and the packet surface from `ggml-hpx-packet.h`.

### 9.2 Plan key (cache)

Reuse the discipline from `ggml_hpx_packet_plan_key` but at graph
scope:

- per-node op type
- per-node src dtype + repack-trait name (to detect Q4_Kx8 vs Q6_K)
- per-node ne[] / nb[] where they affect lowering
- backend assignment (CPU-only required to engage)
- policy_version (bump on classifier or compose change)

Output of analyze:

```cpp
struct hpx_run_plan {
    std::vector<hpx_run>    runs;
    std::vector<dep_edge>   inter_run_deps;     // src_run -> dst_run
    std::vector<int>        node_to_run;        // gf node idx -> run idx
    plan_key                key;                // for cache
};
```

### 9.3 Run-level dataflow vs serial

For decode:
- runs are ordered by graph node index
- the dependency chain is essentially linear (each run depends on
  the previous)
- HPX value: **lane fan-out inside lowered/packet runs** + **single
  terminal wait** instead of per-run sync

For prefill:
- region chain is also linear per HPX_PREFILL_REPORT.md
- HPX value at run level is small; the parallelism is intra-run
  (lane fan-out across columns)
- conclusion: the run-level executor for prefill is mostly a
  packaging exercise; the win lives inside each run, not between

### 9.4 Ownership / lifetimes

Three lifetime tiers:

- **persistent**: HPX runtime (one per process), packet runtime
  (one or one per team), packet caches, plan cache.
- **per-shape**: compiled `hpx_run_plan` (lives until the cache
  evicts or the graph shape changes). Lowered runs' ctx storage is
  in the plan, not the call stack — this is the change from today.
- **per-call**: bindings (typed structs of live pointers), shared
  scratch buffers that depend on activation values, one frame per
  in-flight run for any concurrent dispatch.

The hard rule from `HPX_EXECUTOR_CONTRACT.md §4`: plan/cache keys
are **structural only**. No raw tensor addresses, no scheduler
pointers, no per-invocation data in the key.

### 9.5 Backend opacity, declared at plan time

The planner inspects each node and chooses `backend_run` whenever:
- node's assigned backend is not the active CPU backend (rejected
  earlier — should not engage selective at all), OR
- node's op is on the opaque list (FLASH_ATTN_EXT, BLAS-bound
  MUL_MAT under the BLAS predicate already in ggml-cpu), OR
- node sits inside a contiguous fallback island the planner is
  about to coalesce into a `fallback_run`.

`backend_run` is only conceptually distinct from `fallback_run`;
both call into a backend. The distinction matters when we want
HPX to schedule **around** a known-opaque task (e.g. start a
follow-up lowered_run that doesn't read its outputs) — that is
phase 4+ work, not minimum prototype.

---

## 10. Minimal prototype plan

The outline you proposed is the right shape. Concrete suggestions
for each phase, given what's already in this branch:

### Phase 0 — Read-only design (this document)
Done. Output: this report. No code change.

### Phase 1 — Build the run list, no HPX async
- Add `hpx_run_plan analyze(gf)` to a new translation unit in a
  fresh checkout.
- Reuse the trait predicate `ggml_hpx_is_q4k_8x8_repacked` and the
  packet matchers (`prescan_mlp_*` from `ggml-hpx-exec-selective.cpp`)
  as **classifier components**, not as dispatchers.
- Output: list of `hpx_run` with kinds + node ranges + planned
  bindings.
- Validation: walk the plan, assert every node lands in exactly one
  run, dump the plan as the existing `[hpx-map]` lines.
- No execution, no HPX, no `.get()`.

### Phase 2 — Execute the run list serially on the native thread
- Run kinds dispatched directly without HPX:
  - fallback_run/backend_run → `ggml_backend_graph_compute(view)`
  - packet_run → bind + `ggml_hpx_run_frozen_packet`
  - lowered_run → `ggml_hpx_run_region_group`
- Goal: bit-identical output vs. current selective branch
  (compare against `q4k8-edit5-validate/B-q4k8-on.tokens.final`).
- Only at this point do we have a self-consistent run-level
  reference implementation.

### Phase 3 — Replace the selective loop
- Wire `hpx_run_executor` into `llama_context::graph_compute` in
  the same place the current selective call sits
  (`src/llama-context.cpp:2446`).
- Cache the plan keyed by graph shape; bind per call.
- Goal: same bit-identical output, decode tok/s **at least equal to
  current selective B (≈9.8–13 tok/s)**. If it regresses, the
  ownership model is wrong — fix before phase 4.

### Phase 4 — Add HPX async orchestration around runs
- One `hpx::async`/`hpx::dataflow` per run; one terminal
  `hpx::wait_all` per `graph_compute` call.
- Inter-run deps from the plan; runs that share no data may overlap
  (rare on decode, exists on prefill where two adjacent CPU regions
  could overlap with their successor BLAS region's BLAS-internal
  parallelism — measure first).
- Goal: native↔HPX crossings drop from ~259/token (today) to
  O(runs)/token, ideally O(1)/call. Measured tok/s should move
  toward the scheduler baseline.

### Phase 5 — Reintroduce packets/direct lowering selectively
- Extend the matcher set deliberately, **only** when the run-level
  plan shows a sublayer it would consolidate. The QKV_PROJ_Q4K8
  design from `q4k8-edit5-validate/QKV-packet-design.md` is the
  obvious next packet candidate: 22 sites × 6 SERIAL steps each
  would absorb a chunk of the V-projection F-len=4 islands
  (combined with q6_K_8x8 lowering).
- Avoid hand-writing a new packet before the run plan exists. New
  packets without run-level planning recreate the problem.

### Stop conditions

- Phase 1/2 must produce bit-identical output before moving on.
- Phase 3 must not regress against current selective B numbers.
- Phase 4 must show measurable native↔HPX crossing reduction *and*
  measurable tok/s improvement before adding more packets in
  phase 5.

---

## 11. Prefill benchmark implications

From `local/results/HPX_PREFILL_REPORT.md`:

### 11.1 What current paths apply to prefill

- The selective executor is **unreachable** for prefill: gated on
  `!batched` (`src/llama-context.cpp:2303`). The graph-map summary
  in HPX_PREFILL_REPORT.md confirms zero `[hpx-selective]` lines in
  any of 24 prefill stderr captures.
- The packet path is `team=DECODE` only and `!batched`-gated for
  the same reason.
- The active prefill code is `ggml-hpx-exec.cpp::ggml_hpx_exec_run_prefill`
  (regions = alternating CPU/BLAS, dispatched serially or via
  `LLAMA_HPX_PARALLEL_PROJ=1` for Q/K/V chains within CPU regions).

### 11.2 Decode-only assumptions to avoid

- "rows == 1" is built into the packet matchers and the Q4_K
  lowering branch. A run-level executor for prefill must have
  rows-aware variants (or refuse to engage), not silently lower
  to a rows==1 kernel.
- "GLU sublayer fires once per layer" is decode-graph specific.
  Prefill graphs may have multiple ubatches per call; per-ubatch
  graph shape may differ.
- "fully linear region chain" is true for the **single-CPU-backend
  prefill forward pass**, not for cross-batch / cross-sequence
  parallelism — that's a different design question.

### 11.3 Metrics to collect for the new design

For both decode and prefill, instrument:

- runs per call (by kind)
- native↔HPX crossings per call (native-thread `.get()` count
  attributed to a tagged source)
- packet matches per call
- per-run wall time (`*_ns` counters already in
  `ggml_hpx_selective_stats`)
- end-to-end tok/s vs. scheduler baseline at the same shape

Capture into `hpx-bench/results/<date>-<slug>/` per the result
logging convention; record the commit hash, command, and binary
path in `README.md` (rule from `feedback_results_location.md`).

### 11.4 How prefill should affect the fresh design

- The prefill region chain has no inter-region parallelism at a
  single-backend forward pass. The run-level executor should
  **acknowledge this** rather than try to schedule it.
- The win at prefill comes from intra-run parallelism (lane fan-out
  inside MUL_MAT regions) and from removing per-region dispatch
  overhead (HPX serial neutral or slow in the current shape).
- For longer term, prefill parallelism is structural: request
  batching, multi-sequence scheduling, or genuine independent work
  across regions — out of scope for this prototype but a sane
  long-term north star.

---

## 12. Risks and open questions

### 12.1 Risks

- **Plan-cache invalidation.** Graph shape can change across calls
  (n_tokens varies, ubatch boundaries shift). The plan key must
  capture every structural variable that affects classification, or
  we will run a stale plan against a different graph and corrupt
  output. Mitigation: validate cache hit by re-walking
  op/dtype/trait sequence, not just hash.
- **Lowered-run ctx ownership.** Today `ggml_hpx_lowering` is on
  the call stack. Lifting it into the plan means the plan owns
  arenas; a wrong ownership model could re-introduce the bug
  `HPX_EXECUTOR_CONTRACT.md §5` warns about (packet ctx pointing
  into temporary compose/lowering arenas).
- **Backend safety on partial offload.** The current path bails on
  `n_splits != 1`. The run plan must keep that guard; promoting a
  Metal-assigned node into a CPU `lowered_run` would break
  `HPX_EXECUTOR_CONTRACT.md §8`.
- **Packet runtime team count.** Today decode uses n_lanes=1 (SERIAL
  packets only). If lane fan-out is enabled inside packets and at
  the run-level simultaneously, oversubscription is plausible.
  Measure before mixing.
- **Doing the rewrite from clean upstream.** Tempting to refactor
  in place; the risk is preserving the current `.get()`-per-node
  shape by accident. A clean checkout with the run-level executor
  imported as a self-contained library is the safer path.

### 12.2 Open questions

- Is decode actually run-level-bound, or is there a kernel-quality
  ceiling we have not measured? The hot-path audit estimated ~4.5
  ms/token of pure scheduling overhead, comparable to ~9.75
  ms/token total scheduler-on. Removing all of it puts selective
  near scheduler — but does not exceed it. What HPX-native gain
  beyond parity is the eventual target?
- The 102 fallback runs / decode are dominated by KV-cache
  plumbing. Is any of that lowerable (SET_ROWS, PERMUTE), or is
  the right move to fuse the 14-node burst into a single
  delegated kernel?
- For the prefill linear-chain reality: do we keep an HPX prefill
  path at all in the prototype, or accept that prefill stays on
  the scheduler until cross-request parallelism exists? The data
  strongly suggests the latter.
- Does an `attention_full_block` packet (RMS_NORM + QKV proj +
  RoPE + KV-write + FlashAttn + output proj) make sense in the
  fresh design, given how much of attention is opaque
  (FLASH_ATTN_EXT, RoPE in-place on KV cache, SET_ROWS)? It might
  be the wrong abstraction; a backend_run sequence with explicit
  deps may be more honest.

---

## 13. Files to reuse or avoid copying

### 13.1 Reuse (port mostly intact)

These are **HPX-native and load-bearing**, and the durable lessons
of this branch live here:

- `ggml/src/ggml-hpx/ggml-hpx-region-dag.h` — region/dep/resources
  data model. Keep.
- `ggml/src/ggml-hpx/ggml-hpx-region-exec.{h,cpp}` —
  `validate_region_group`, `run_region_group`, all the
  `*_run_range` kernels (F32 mul_mat, RMS_NORM 3-region, SiLU,
  MUL, SWIGLU, quantize_q8_K, q4_K_q8_K gemv, **q4_K_8x8_q8_K
  gemv**). These are the well-tested CPU work.
- `ggml/src/ggml-hpx/ggml-hpx-lower.{h,cpp}` — lowering predicate
  and per-op compile. The trait predicate
  `ggml_hpx_is_q4k_8x8_repacked` and the
  `ggml_repack_extra_traits_name` extern are the durable bits.
- `ggml/src/ggml-hpx/ggml-hpx-packet.{h,cpp}` — frozen-packet
  surface, plan key, frame, runtime. The `MLP_GATE_UP_GLU_Q4K8`
  sublayer is the proven first one.
- `ggml/src/ggml-hpx/ggml-hpx-compose.{h,cpp}` — composers for the
  three MLP shapes. The `Q4K8` composer's "shared one quantize
  across two gemvs" pattern is the proof of value for hand-built
  composers.
- `ggml/src/ggml-hpx/ggml-hpx-runtime.{h,cpp}` — HPX one-shot
  startup with `--hpx:queuing=static`. Keep.
- `ggml/src/ggml-hpx/ggml-hpx-tpool.{h,cpp}` — HPX-backed ggml
  threadpool (the `cplan->work_size >= 32 KiB` substrate
  selection). Keep, but the run-level executor reduces the
  reliance on it.
- `docs/HPX_EXECUTOR_CONTRACT.md` — durable rules. Carry forward
  unchanged; the new design must obey it.
- The selective stats counter shape (`ggml-hpx-exec-selective.h:68`).

### 13.2 Reference-only (do not import the structure)

- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp` — the per-node
  dispatch loop (lines 1628-1917) is the **anti-pattern**.
  Reference for: prescan logic, fallback coalescer
  (`scan_fallback_run_end`, lines 1595-1626), trigger-table
  mechanism. Do not import the loop shape.
- `ggml/src/ggml-hpx/ggml-hpx-exec.{h,cpp}` — coarse path +
  current prefill executor. The `ggml_hpx_exec_run_prefill`
  region builder is a useful structural reference; the
  `LLAMA_HPX_PARALLEL_PROJ=1` parallel-proj logic is a known
  loss (HPX_PREFILL_REPORT.md), do not carry forward.
- `ggml/src/ggml-hpx/ggml-hpx-adapter.cpp`, `ggml-hpx-cache.cpp`,
  `ggml-hpx-plan.cpp` — coarse-path scheduler/plan/cache machinery,
  not on the selective hot path
  (`local/results/2026-04-28-hpx-selective-hot-path-audit.md §2`).
  Do not port.

### 13.3 Drop entirely

- The dual-flag soup that grew during exploration:
  `LLAMA_HPX_SELECTIVE_MUL_MAT`, `LLAMA_HPX_SELECTIVE_MLP_PACKET`,
  `LLAMA_HPX_PACKET_MLP_GATE_UP_GLU_Q4K8`. The new design needs
  one engagement gate (`LLAMA_USE_HPX=1` + a single run-executor
  flag) and the plan picks what runs.
- `LLAMA_HPX_PARALLEL_PROJ=1` — proven slow, reasoned dead.
- The historical provenance docs — `docs/HPX_PROVENANCE.md`,
  `docs/HPX_LOWER_OP.md`, `docs/HPX_BUILD.md` — keep but do not
  treat as design contracts (CLAUDE.md rule).

---

## 14. Recommended next action

Start a fresh branch on top of upstream `llama.cpp`. Implement the
prototype phases in order; keep this branch untouched as the
evidence/reference checkout.

Concretely, **next session**:

1. Create the new branch.
2. Copy the **§13.1 reuse set** verbatim into the new tree under
   `ggml/src/ggml-hpx/` (region-dag, region-exec, lower, packet,
   compose, runtime, tpool, contract doc).
3. Carry the **CLAUDE.md hard invariants** unchanged: process-wide
   one-shot HPX startup, CPU-only single-split engagement,
   structural keys only, fail-closed to scheduler.
4. Begin **Phase 1**: `hpx_run_plan analyze(gf)` + plan dump.
   No execution, no HPX, no `.get()`. Validation is byte-equal
   `[hpx-map]` output against the current branch on the same
   TinyLlama Q4_K_M graph.
5. Stop and review before Phase 2.

Do **not** write a new packet first. Do **not** start with a
prefill executor. The prototype's Phase 1 is a planner that exists
only as data; everything else builds on that.

---

## Appendix — concrete numbers to design against

| metric (TinyLlama 1.1B Q4_K_M, M4, single-token decode) | value |
|---|---:|
| ggml nodes per decode graph | 689 |
| layers | 22 |
| lowered nodes per decode (post-coalescing, packets on) | 135 |
| fallback nodes per decode | 488 |
| **fallback runs per decode** | **102** |
| packet matches per decode | 22 |
| packet nodes per decode (Q4K8 on) | 66 |
| native↔HPX crossings per decode (today, post-coalescing) | ~259 |
| native↔HPX crossings per decode (scheduler) | 1 |
| selective tok/s (Q4K8 on, n=128) | 12.95 |
| scheduler tok/s (same model) | 102.78 |

| metric (Llama-3.1-8B Q4_K_M, M4, prefill, 472 tok prompt) | value |
|---|---:|
| ggml nodes per prefill graph | 999 |
| HPX exec regions | 65 |
| sched_splits | 1 |
| dependency chain | fully linear |
| no-HPX prefill mean | 9322 ms (50.6 tok/s) |
| HPX serial mean | 9872 ms (47.8 tok/s, +5.9%) |
| HPX parallel-proj mean | 13742 ms (34.3 tok/s, +47%) |

These are the targets the run-level executor's first measurable
phase must move.
