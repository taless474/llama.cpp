# llama.cpp fork: HPX CPU execution redesign

# HPX selective + prefill orchestrator branch report

This branch explored several HPX-backed execution designs for `llama.cpp` / `ggml` on the CPU path. The branch reached a useful conclusion: HPX can be integrated correctly, packets can execute correctly, and selective execution can preserve output, but the current selective architecture is structurally slower than the plain llama.cpp scheduler.

The next direction is **not** another handwritten packet and not a prefill-first rewrite. The next direction is a fresh **run-level executor prototype** built from clean upstream `llama.cpp`, using this branch only as evidence and reference.

---

## Final conclusion

The current HPX selective path is correct but structurally slow.

On TinyLlama-1.1B Q4_K_M decode, the graph has 689 `ggml` nodes. With fallback coalescing and Q4K8 packetization enabled, the selective path still executes roughly:

```text
135 lowered nodes
488 fallback nodes, coalesced into 102 fallback runs
22 packet runs, claiming 66 packet nodes
≈259 execution units per decode token
```

The plain llama.cpp scheduler executes the same decode as one scheduler graph execution unit.

The important finding is:

```text
The bottleneck is dispatch fragmentation and native↔HPX crossing overhead, not just kernel quality.
```

Q4K8 packetization improved the selective regime substantially, but did not close the scheduler gap:

```text
Q4K8 packet off: 6.67 tok/s
Q4K8 packet on:  9.78 tok/s
Plain scheduler: ~99–103 tok/s
```

That means packetization worked, but the architecture driving the packets was still wrong.

---

## What this branch proved

### 1. HPX can be integrated into llama.cpp/ggml

The branch integrated HPX into the llama.cpp graph execution path and validated end-to-end execution on real llama graphs.

Important constraints were established:

- HPX runtime startup must be process-wide, not per graph/token/dispatch.
- CPU-only and single-split guards are required for the selective path.
- The selective path must fail closed to the normal llama.cpp scheduler.
- Backend assignment must be respected.
- Offloaded or backend-owned work must not accidentally be computed on CPU.

These are durable constraints for any future HPX work.

---

### 2. Fine-region DAGs are semantically useful

The branch introduced explicit CPU work regions:

```text
ggml_hpx_cpu_region
ggml_hpx_cpu_region_group
ggml_hpx_dep_edge
ggml_hpx_run_range_fn
```

This was useful because it gave HPX explicit ranges and dependencies. It validated that selected operations could be lowered into HPX-visible work units.

Supported lowering included operations such as:

- `SILU_F32`
- `MUL_F32`
- `MUL_MAT_F32`
- `MUL_MAT` for selected Q4_K CPU_REPACK cases
- `RMS_NORM_F32`
- `GLU/SWIGLU`

The lesson is not that this generic DAG should drive the whole runtime. The lesson is that the model is correct and reusable as an implementation detail.

The problem was that generic fine-region DAG dispatch has too much overhead for decode-sized work when used per node.

---

### 3. Selective execution can enter the real llama graph correctly

The selective executor entered the real llama.cpp decode graph and routed nodes through three execution shapes:

```text
lowered fine-region group
coalesced fallback graph view
frozen packet
```

It preserved correctness and produced bit-identical output in the validation runs.

The main problem was not correctness. The problem was execution shape:

```text
for each node:
    classify
    maybe lower
    maybe fallback
    maybe packet
    dispatch now
```

This interleaves planning and execution and creates far too many tiny dispatch units.

---

### 4. Fallback-run coalescing was the right improvement

Before coalescing, fallback execution was one `ggml_graph_view` + `ggml_backend_graph_compute` per fallback node.

```text
before: fallback node i -> graph_compute(view i:i+1)
after:  fallback run [i,j) -> graph_compute(view i:j)
```

This changed the fallback side from:

```text
488 fallback nodes
```

into:

```text
102 fallback runs
```

This was a real structural improvement. It showed that the executor should reason in larger units than individual nodes.

But even after coalescing, the graph still had too many execution units overall.

---

### 5. Packetization worked, but was not enough

The branch implemented the `MLP_GATE_UP_GLU_Q4K8` packet.

The packet replaces this pattern:

```text
gate MUL_MAT
up   MUL_MAT
SWIGLU
```

with one frozen packet that:

```text
quantizes x once
runs gate q4_K_8x8_q8_K gemv
runs up   q4_K_8x8_q8_K gemv
runs SWIGLU
```

This was the right packet shape because gate and up share the same activation input. The packet avoided duplicate quantization and reduced the number of selective execution units.

Measured result inside the selective regime:

```text
Q4K8 packet off: packet=22(22 nodes), 6.67 tok/s
Q4K8 packet on:  packet=22(66 nodes), 9.78 tok/s
```

This is a real improvement of roughly 47% within the selective path. Output was bit-identical and stable through `n=128`.

However, the plain scheduler was still about 10× faster:

```text
Plain scheduler: ~99–103 tok/s
Selective + Q4K8 packet: ~10–13 tok/s
```

So the packet proved the packet infrastructure, but it also produced the stop signal:

```text
Do not keep writing deeper packets before proving the execution path can compete with the scheduler.
```

---

## What failed or stayed slow

### 1. Per-node native↔HPX crossing

The current selective loop still uses the forbidden hot-path shape:

```cpp
hpx::async([&] {
    // tiny lowered node or packet
}).get();
```

This happens repeatedly from the native llama-context thread.

The result is many small native-to-HPX crossings per decode token.

Even after fallback coalescing and Q4K8 packetization, the path still has roughly 259 execution units per decode token. The plain scheduler path has one.

This is the central structural failure.

---

### 2. Lowered-node ownership forces immediate `.get()`

The current lowered path uses per-iteration stack-local lowering objects. The immediate `.get()` is load-bearing because it keeps those objects alive until the HPX work finishes.

That means the design cannot simply remove `.get()` and become async. The ownership model must change first.

Future design must separate:

```text
analyze / compile structure
bind live tensors
execute runs
```

and ensure that any ctx/frame/scratch storage lives long enough for async execution.

---

### 3. More HPX parallelism did not help prefill

Prefill looked promising because larger token batches sound like a better HPX target. The benchmark showed otherwise.

The observed prefill region graphs were fully linear:

```text
TinyLlama:      45 regions, fully linear
Llama-3.1-8B:   65 regions, fully linear
```

There was no inter-region parallelism for HPX to exploit.

Measured result on Llama-3.1-8B Q4_K_M prefill:

```text
No HPX:            9322 ms, 50.6 tok/s
HPX serial:        9872 ms, 47.8 tok/s
HPX parallel-proj: 13742 ms, 34.3 tok/s
```

The parallel-projection path was slower because it added HPX task overhead and synchronization without shortening the critical path.

The prefill lesson is:

```text
Bigger batches alone do not create useful HPX parallelism.
HPX only helps if the graph exposes independent coarse work or if it improves intra-run execution enough to pay for overhead.
```

---

### 4. Prefill could not validate packets

The current selective/fine-region/packet path is decode-only because it is gated by `!batched`.

That means prefill benchmarks could not validate QKV packet expectations or selective packet behavior.

Lesson:

```text
Before benchmarking an optimization, first prove the optimized path actually engages.
```

---

## Concrete graph facts from this branch

### Packet-disabled graph-map reference

From the `[hpx-map-*]` capture for TinyLlama Q4_K_M decode:

```text
nodes=689
lowered_nodes=201
fallback_nodes=488
packet_nodes=0
lowered_runs=101
fallback_runs=102
packet_runs=0
```

This is the packet-disabled graph-map reference. It is useful for validating the future analyzer before packet ownership is applied.

Observed recurring graph structure:

- 22 transformer layers.
- Repeated attention blocks with Q/K/V projections, RoPE, KV-cache writes, `FLASH_ATTN_EXT`, and output projection.
- Repeated MLP blocks with gate/up/down projections and SWIGLU.
- KV-cache / attention plumbing appears as long fallback islands around `RESHAPE`, `ROPE`, `VIEW`, `SET_ROWS`, `PERMUTE`, and `FLASH_ATTN_EXT`.
- Q and K projections are often lowerable when backed by `q4_K_8x8_q8_K`.
- V and down projections often fall back when backed by unsupported or non-matching quantized repack traits.

### Packet-enabled selective reference

With Q4K8 packetization enabled, the reference classification is:

```text
nodes=689
lowered_nodes=135
fallback_nodes=488
packet_runs=22
packet_nodes=66
fallback_runs=102
```

Invariant:

```text
135 + 488 + 66 = 689
```

This is the future analyzer sanity check. It does not validate speed. It validates that the analyzer understands the same graph structure and ownership rules as this branch.

---

## Why QKV packetization was stopped

A QKV packet design was drafted but not continued.

Reason:

```text
The Q4K8 MLP packet already proved packetization.
Another handcrafted packet would likely improve the selective regime again,
but it would not answer the bigger architecture question.
```

The bigger question is not whether another packet can claim more nodes. The bigger question is whether the executor can avoid the per-node/per-run dispatch fragmentation that keeps selective far behind the scheduler.

Therefore, the right next step is not QKV packetization.

---

## Durable lessons

### 1. Correctness is not enough

The branch proved correctness, bit-identical output, stable generation, and working packets.

That was still not enough because the scheduler was much faster.

Future HPX work must be evidence-gated:

```text
Does this path engage?
Is there independent work?
Does it beat the baseline?
Does overhead scale down with larger models or larger batches?
```

---

### 2. HPX must operate at the right granularity

Per-node HPX dispatch is too fine for decode.

The useful unit is not a single `ggml` node. The useful unit should be a larger run:

```text
fallback_run
packet_run
lowered_run
backend_run
```

The future executor should schedule runs, not nodes.

---

### 3. Packets are one run kind, not the architecture

Packets are useful for stable repeated patterns. They should be part of the executor, but not the whole design.

The mistake would be to keep hand-writing packets while still driving them through a fragmented per-node selective loop.

---

### 4. Backend work should remain opaque

HPX should not try to rewrite every backend kernel.

Backend-owned work should be treated as opaque tasks:

```text
ggml_backend_graph_compute(...)
fallback graph views
FLASH_ATTN_EXT
BLAS / Accelerate / Metal / ggml-cpu internal kernels
packet runs
```

HPX should know dependencies between runs, not the internals of every backend kernel.

---

### 5. Do not weaken backend guards

The current CPU-only and single-split guards are necessary.

Any future selective/run-level executor must fail closed to the scheduler when assumptions do not hold.

Do not compute Metal/CUDA/offloaded work on CPU by accident.

---

## Next direction

The next prototype should be a **run-level executor analyzer** first:

```text
analyze graph
classify nodes into runs
validate every node belongs to exactly one run
dump a run plan
no execution
no HPX async
no packet compile
no packet execution
no runtime wiring
```

The future design should separate phases:

```text
analyze graph shape
build run-level plan
cache plan
bind live tensors per invocation
execute runs
```

The first useful artifact is data only:

```text
hpx_run_plan analyze(gf)
```

It should reproduce this branch's reference numbers before any execution is added.

---

## Proposed future run kinds

A future run-level executor can start with:

```cpp
enum class hpx_run_kind {
    fallback_run,  // contiguous ggml graph view, opaque backend compute
    packet_run,    // compiled packet, bind + dispatch later
    lowered_run,   // one or more HPX region groups later
    backend_run,   // delegated/opaque backend work
};
```

For the analyzer-only phase, these are just classifications. No run should execute.

### `fallback_run`

A contiguous range of `ggml` nodes that should be executed by normal backend compute.

Example:

```text
ggml_graph_view(gf, i, j)
ggml_backend_graph_compute(cpu_be, &view)
```

In the analyzer phase, this is only a planned run, not executed.

### `packet_run`

A recognized repeated pattern such as MLP gate/up/SWIGLU Q4K8.

The analyzer should mark ownership of packet nodes first so those nodes are not also counted as lowered or fallback.

### `lowered_run`

A node or group of nodes that would be lowerable through the HPX fine-region path.

In the future implementation, lowered ctx ownership must be moved out of per-iteration stack storage before async execution is attempted.

### `backend_run`

Reserved for opaque backend-owned work. In Phase 1, it may remain unused if the analyzer mirrors today's selective behavior exactly and treats opaque backend work as fallback.

---

## Future analyzer validation references

The future analyzer should reproduce these classifications on the same TinyLlama Q4_K_M decode graph before any execution is added.

### Packet-disabled reference

```text
nodes=689
lowered_nodes=201
fallback_nodes=488
packet_nodes=0
lowered_runs=101
fallback_runs=102
packet_runs=0
```

### Packet-enabled Q4K8 reference

```text
nodes=689
lowered_nodes=135
fallback_nodes=488
packet_nodes=66
packet_runs=22
fallback_runs=102
```

### Validation invariant

Every `ggml` node must be assigned exactly once:

```text
lowered_nodes + fallback_nodes + packet_nodes == nodes
```

For packet-enabled Q4K8:

```text
135 + 488 + 66 == 689
```

If the analyzer does not reproduce these numbers, stop before execution and inspect:

- packet matcher order
- `consumed[]` behavior
- Q4K8 trait detection
- fallback coalescing boundaries
- graph/model/prompt/build flag mismatch

---

## What not to do next

Do not:

- write a new QKV packet first
- revive HPX parallel-projection prefill
- optimize inside a path before proving the path engages
- benchmark packet/lowering expectations through prefill while `!batched` skips the path
- add more HPX parallelism unless it shortens the critical path
- keep per-node `hpx::async(...).get()` in the hot path
- weaken CPU-only / single-split backend guards
- refactor this branch as the new architecture base

---

## Artifacts to preserve

Keep these as evidence/reference for the fresh design:

```text
local/refill-summary.md
local/results/q4k8-edit5-validate/
local/results/HPX_PREFILL_REPORT.md
hpx-bench/results/2026-04-29-selective-graph-map/
```

Important reference files include:

```text
hpx-bench/results/2026-04-29-selective-graph-map/stderr.txt
hpx-bench/results/2026-04-29-selective-graph-map/README.md
```

The `stderr.txt` file is the existing `[hpx-map-*]` packet-disabled graph-map capture.

---

These files contain useful implementation pieces, but they should be ported deliberately into a fresh branch rather than refactored in place here:

```text
ggml/src/ggml-hpx/ggml-hpx-region-dag.h
ggml/src/ggml-hpx/ggml-hpx-region-exec.{h,cpp}
ggml/src/ggml-hpx/ggml-hpx-lower.{h,cpp}
ggml/src/ggml-hpx/ggml-hpx-packet.{h,cpp}
ggml/src/ggml-hpx/ggml-hpx-compose.{h,cpp}
ggml/src/ggml-hpx/ggml-hpx-runtime.{h,cpp}
ggml/src/ggml-hpx/ggml-hpx-tpool.{h,cpp}
docs/HPX_EXECUTOR_CONTRACT.md
ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp
```

---

## One-line closure

This branch proved that HPX selective execution and packetization can be correct, but the current per-node/per-run dispatch architecture cannot compete with the llama.cpp scheduler. The next attempt should start from a run-level analyzer and cached run plan, not another packet.
