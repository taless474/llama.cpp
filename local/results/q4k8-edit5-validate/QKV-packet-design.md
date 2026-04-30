# Design note — attention QKV projection packet

Scope: a single fused packet that covers RMS_NORM + norm-scale MUL + Q + K + V
MUL_MAT for one TinyLlama-style attention sub-block. Everything downstream
(RESHAPE, ROPE, SET_ROWS, KV-cache views, FLASH_ATTN_EXT, output projection,
residual) stays on the existing per-node path.

This is a design note. **No code changes.**

Inputs to this note: the histogram run
`F-hist-attention-shapes.log`, the existing Q4_Kx8 FFN packet
(`mlp_gate_up_glu_q4k8`), and the per-tensor repack log captured during model
load.

---

## 1. Exact node pattern and trigger node

Pattern (graph order, per layer):

```
N0  RMS_NORM(x_in)              → norm_out   (F32, [d_model])
N1  MUL(norm_out, w_norm)       → normed_x   (F32, [d_model])
N2  MUL_MAT(W_Q, normed_x)      → q_proj     (F32, [d_model])
N3  MUL_MAT(W_K, normed_x)      → k_proj     (F32, [d_kv])
N4  MUL_MAT(W_V, normed_x)      → v_proj     (F32, [d_kv])
```

5 nodes per match. The packet writes through `q_proj`, `k_proj`, `v_proj` to
existing tensor storage; `norm_out` and `normed_x` are packet-internal.

**Trigger node**: by analogy with the existing FFN matchers (which trigger on
`GGML_OP_GLU` — the *last* node in their chain), the right choice is the
**V-projection MUL_MAT** (N4 in graph order; the last MUL_MAT among the
three).

Why V (last) is preferred over Q (first) or any middle node:

- The dispatcher convention is "skip indices that pre-scan marked as
  consumed; run the packet when the trigger index is reached." With trigger =
  V, every consumed index (N0 RMS_NORM, N1 MUL, N2 Q, N3 K) is *strictly less
  than* the trigger index N4. Linear scan reaches consumed→consumed→consumed→
  consumed→trigger, in that order, with no need for forward-look or rollback.
- Matches the existing FFN convention (GLU as last-node trigger), so the same
  pre-scan and dispatch templates carry over with minimal divergence.
- Symmetric topology: V is the unambiguously last MUL_MAT consuming
  `normed_x`, identical role to GLU's "unambiguously last consumer" position
  in the FFN chain.

If Q were chosen as trigger instead (N2, middle of the consumed set), the
dispatcher would need to handle skips on *both sides* of the trigger index
(N0/N1 before, N3/N4 after). Correctness still holds *if* the
no-consume-before-viable invariant is honored (pre-scan commits
consumed/skipped marks only after the packet entry is compiled and validated,
so by the time the executor reaches N0 there is no possibility of mid-stream
rollback) — but the bookkeeping is strictly more work for no benefit. There
is no need to introduce a rollback path: the invariant rules it out.

Pre-scan algorithm (with V as trigger):

1. Iterate gf->nodes. For each MUL_MAT node `v_t` (candidate trigger):
   - check `v_t->src[1]` is the output of a MUL `mul_t` whose `src[0]` is the
     output of a RMS_NORM `norm_t`;
   - walk backward through gf->nodes from `v_t` to find candidate Q and K
     MUL_MATs whose `src[1]` is the same `normed_x` tensor as `v_t->src[1]`;
   - confirm exactly three consumers of `normed_x` and exactly one consumer
     of `norm_out` (§10 exclusive-consumer checks).
2. Compile and bind a packet entry for the (cols, out_cols_q, out_cols_kv,
   row_strides, traits) shape. **If compile/bind fails, reject the match — do
   not mark any nodes consumed** (see *Dispatch invariant* below).
3. On success, record `(norm_idx, mul_idx, q_idx, k_idx, v_idx, entry*)`,
   mark N0..N3 as consumed, mark N4 (`v_idx`) as trigger.

## 2. Count per decode graph

22, one per transformer layer. Confirmed by histogram: 44 RMS_NORM (22 attn +
22 FFN) and 44 MUL_MAT of shape 2048×2048 (22 Q + 22 attn_output).

## 3. Tensor types and shapes for Q, K, V

| projection | shape (cols × out_cols) | weight type (across 22 layers) |
|---|---|---|
| Q (`attn_q`) | 2048 × 2048 | Q4_K-repacked, `q4_K_8x8_q8_K` trait, all 22 layers |
| K (`attn_k`) | 2048 × 256 | Q4_K-repacked, `q4_K_8x8_q8_K` trait, all 22 layers |
| V (`attn_v`) | 2048 × 256 | mixed: Q4_K-repacked in 12 layers, Q6_K-repacked in 10 layers |

256 = 4 KV heads × 64 head dim (GQA).

`rows = 1` (single-token decode is the target — `node->ne[1] == 1` like the
existing Q4K8 FFN packet).

`d_model = 2048`, `d_kv = 256`. Both are stable structural constants of the
model.

## 4. Which layers/projections are Q4_Kx8 vs Q6_Kx8

From the model load log:

- **`attn_q`**: all 22 layers Q4_Kx8.
- **`attn_k`**: all 22 layers Q4_Kx8.
- **`attn_v`**: Q6_Kx8 in blocks 0, 1, 4, 7, 8, 9, 12, 15, 18, 20 (10 layers);
  Q4_Kx8 in blocks 2, 3, 5, 6, 10, 11, 13, 14, 16, 17, 19, 21 (12 layers).

So the Q4/Q6 split is *only on V*. Q and K are uniform Q4_Kx8.

## 5. Whether Q/K/V all share the same normalized activation

Yes. The pattern in llama-graph is:

```
normed = rms_norm_with_scale(x_in, w_norm)
q = matmul(W_Q, normed)
k = matmul(W_K, normed)
v = matmul(W_V, normed)
```

A single `normed_x` tensor feeds all three projections. Confirmed by the
histogram (1 RMS_NORM and 1 MUL per attn block) and by reading
`src/llama-graph.cpp` (the QKV step uses one `cur` after norm).

## 6. Whether the normalized activation can be computed once and reused by Q/K/V

Yes — and this is the same fusion shape as the existing FFN packet, where
`Q8_K(x)` is quantized once and shared by gate/up. For QKV:

- Compute `normed_x` once inside the packet.
- Quantize `normed_x` to `Q8_K` once.
- Pass that `Q8_K` buffer to three `gemv_q4_K_8x8_q8_K` invocations (one each
  for Q, K, V weights).

This avoids 2 redundant quantizations per layer (each MUL_MAT today
re-quantizes its source independently).

## 7. Whether Q6_Kx8 has an existing prequantized-Q8_K helper like Q4_Kx8

Partially. ggml-cpu has the kernel:

- `ggml_gemv_q6_K_8x8_q8_K` — `repack.cpp:4007` (and generic at 1118)
- `q6_K_8x8_q8_K` trait instance — `repack.cpp:4551`
- It takes a `Q8_K`-quantized input identical in format to the Q4 path.

But ggml-hpx has **no Q6 wrapper**:

- `ggml-hpx-region-exec.cpp` only invokes `ggml_gemv_q4_K_8x8_q8_K` (line 519);
  there is no Q6 region kernel.
- `ggml-hpx-lower.cpp:44` recognizes only `q4_K_8x8_q8_K` as a lowerable trait.
- The Q4K8 packet kernel call site in the packet runtime hardcodes the Q4
  kernel; no Q6 dispatch.

So the underlying ggml-cpu kernel is ready, but a Q6 path through HPX is a
separate piece of work (region wrapper + lower-path trait check + packet
kernel dispatch + matching test like
`test_hpx_selective_mul_mat_q4_k_repacked.cpp`).

## 8. First cut: all-Q4_Kx8 sites only, or mixed Q4/Q6 required

**First cut should be all-Q4_Kx8 only.** This means the packet matcher
requires *all three* of `attn_q`, `attn_k`, `attn_v` to be Q4_Kx8-traited.
On TinyLlama Q4_K_M this gives **12 of 22 matches**; the other 10 layers (V is
Q6_Kx8) skip and continue through the existing per-node path.

Reasons:
- The new packet category and its plumbing (cache, prescan, dispatcher,
  write-through path, alias bookkeeping) is the bulk of the work and is
  identical for Q4 and Q6.
- The Q6 HPX wrapper is a separable, testable unit. Better to land it as a
  follow-up than couple it to the first QKV packet.
- 12-of-22 coverage still removes 12 × 4 = 48 dispatch boundaries per decode
  and validates the pattern end-to-end on real layers.

A future patch can add Q6_Kx8 support to:
1. ggml-hpx region exec (`ggml_gemv_q6_K_8x8_q8_K` wrapper),
2. lower-path trait recognition,
3. the QKV packet kernel dispatch (per-projection-type fan-out),
4. a corresponding repacked-Q6_K test.

After that, the QKV matcher can accept Q4 or Q6 per projection independently
and reach 22-of-22 coverage.

## 9. Which tensors must write through to ggml storage

Must write through (consumed outside the packet):
- `q_proj` — read by RESHAPE → ROPE downstream.
- `k_proj` — read by RESHAPE → ROPE → SET_ROWS into K-cache.
- `v_proj` — read by RESHAPE → SET_ROWS into V-cache.

Packet-internal (single-consumer chain inside the packet, see §10):
- `norm_out` — only consumer is the in-packet MUL (N1).
- `normed_x` — only consumers are the three in-packet MUL_MATs (N2..N4).

Internal tensors do not need to round-trip through ggml storage; they can
live in packet-frame scratch. This is the same pattern Q4K8 uses for the
quantized `Q8_K(x)` buffer.

## 10. Exclusive-consumer checks

The pre-scan must verify, on each candidate match, that:

- **`norm_out`** has exactly one consumer in the graph: the MUL at N1.
- **`normed_x`** has exactly three consumers in the graph: the Q, K, V
  MUL_MATs at N2..N4. No fourth consumer (no separate use of normed
  activation outside the QKV trio).
- **`q_proj`, `k_proj`, `v_proj`** each have at least one consumer (RESHAPE)
  and are not aliased into anything the packet writes. They are written
  through; downstream consumption is unaffected because the values are
  identical to the per-node path.

If any exclusive-consumer check fails, the match is rejected and the layer
falls through to the existing per-node path.

This is the standard "exclusive consumer / write-through" story from the
Packet Rule in `docs/HPX_LOWER_OP.md`. Q4K8 already does an analogous check
on `gate_proj` and `up_proj`.

## Dispatch invariant — no consume before viable

**No node is marked consumed/skipped unless the packet entry for that match
is known to be valid.**

Concretely, the pre-scan pipeline for one candidate match is:

1. Pattern-match the structural shape (RMS_NORM → MUL → 3 MUL_MAT sharing
   `normed_x`) and exclusive-consumer checks (§10).
2. Build the `qkv_proj_cache_key` and look it up in the cache.
3. If miss: compile and bind a new packet entry. If any of compile, kernel
   resolution (Q4_Kx8 trait support), or frame allocation fails, reject the
   whole match.
4. **Only after the entry is in hand**, write the consumed marks for
   N0..N3 and the trigger mark for N4 into the dispatcher's per-graph
   bookkeeping.

Consequence: at dispatch time, every consumed/skipped index has a
guaranteed-valid packet entry waiting at its associated trigger. The
dispatcher never has to "un-skip" a node, recompute upstream values, or
fall back to per-node execution mid-pattern. Failure is handled at pre-scan
time by simply not marking the nodes at all — the per-node path then
executes them in the normal way, identical to a graph that never had a
candidate match.

This invariant is what makes V-as-trigger and Q-as-trigger structurally
equivalent for correctness (both rely on it), and what eliminates the need
for a rollback mechanism inside the dispatcher.

## 11. Cache key fields

By analogy to `mlp_gate_up_glu_q4k8_cache_key` (`out_cols`, `cols`, `rows`,
`w_row_stride`):

```cpp
struct qkv_proj_cache_key {
    int64_t  cols;             // d_model       (input dim, shared)
    int64_t  out_cols_q;       // d_model       (Q output dim)
    int64_t  out_cols_kv;      // d_kv          (K and V output dim — equal)
    int64_t  rows;             // 1 for decode
    int64_t  w_q_row_stride;
    int64_t  w_k_row_stride;
    int64_t  w_v_row_stride;
    // Encoded weight traits, even though first cut is all-Q4:
    // makes the cache forward-compatible with Q6 V without re-keying.
    uint32_t w_q_trait;        // 0 = q4_K_8x8_q8_K, 1 = q6_K_8x8_q8_K, ...
    uint32_t w_k_trait;
    uint32_t w_v_trait;
};
```

Rationale:
- `cols` and `rows` cover the I/O shape.
- The three `out_cols` cover output shapes of Q, K, V (Q differs from K/V).
- Three `w_*_row_stride` cover repack stride per weight.
- Three `w_*_trait` keep Q6 extensibility cheap.

In practice on TinyLlama Q4_K_M with first-cut Q4-only, all 12 matching
layers will hash to a single cache entry (every Q has the same shape, every
K and V too) — same one-entry-per-shape behavior as Q4K8.

## Frame-owned scratch layout

The packet frame (one per cache entry) owns three scratch buffers, allocated
once at compile/bind time and reused across every dispatch of that entry.
None of these are ggml tensors — they live in the frame and never round-trip
through ggml storage:

| name | dtype | shape | purpose | bytes (TinyLlama, d_model=2048) |
|---|---|---|---|---|
| `norm_out` | F32 | `[d_model]` | output of RMS_NORM, single-consumer (in-packet MUL) | 8 KiB |
| `normed_x` | F32 | `[d_model]` | output of MUL(norm_out, w_norm); shared input to Q/K/V gemv | 8 KiB |
| `q8_x`     | Q8_K-quantized | `[d_model]` (`d_model / 256` super-blocks) | quantize-once-share-many input to the three `gemv_q4_K_8x8_q8_K` calls | ~1.2 KiB |

Total per frame: ~17 KiB, trivially fitting any sane allocator alignment.

Layout rules:

- All three buffers are aligned to `GGML_MEM_ALIGN` (matches the frame's
  base alignment used by Q4K8; required by the gemv kernels' load/store
  paths).
- `q8_x` is computed once per dispatch from `normed_x` via the existing
  `quantize_row_q8_K` path (same call Q4K8 already uses for its FFN
  `q8_x`).
- `norm_out` and `normed_x` could in principle be aliased (RMS_NORM result
  is overwritten by MUL element-wise), but keeping them separate keeps the
  packet body as three sequential sub-stages with clear data dependencies.
  The 8 KiB cost is negligible.
- None of the three buffers escape the packet. Nothing outside the packet
  reads them, and the §Dispatch-invariant guarantees that a partially-run
  packet is never observable.

This mirrors Q4K8's approach (which already owns its own `q8_x`); the only
new scratch slots are `norm_out` and `normed_x`.

## 12. Expected packet_nodes and boundary reduction

First cut (all-Q4_Kx8 only, 12 matches per decode):

- Packet nodes: 12 × 5 = **60 packet nodes per decode** (N0..N4).
- Dispatch boundaries removed: 12 × 4 = **48 fewer boundaries per decode**
  (5 nodes → 1 packet = 4 removed each).

After Q6 follow-up (full 22 matches):

- Packet nodes: 22 × 5 = 110 per decode.
- Dispatch boundaries removed: 22 × 4 = 88 per decode.

For comparison, Q4K8 currently absorbs 22 × 2 = 44 packet nodes and removes
22 × 2 = 44 boundaries. So the QKV first cut is roughly the same scale of
boundary reduction as Q4K8 (slightly larger), with comparable expected impact
on `tok/s` (the FFN block is bigger work-wise per node, but QKV has more
matmul mass per layer at the larger Q shape).

## 13. Correctness test plan

Mirror the Q4K8 test layout:

1. New unit test `tests/hpx/test_hpx_selective_qkv_packet_q4k8.cpp`:
   - Build a synthetic mini-graph: input → RMS_NORM → MUL → 3 MUL_MAT
     against repacked Q4_Kx8 weights.
   - Run reference (per-node ggml-cpu) and HPX (with QKV packet engaged).
   - Assert bit-equal Q, K, V outputs.
   - Assert the packet matcher fires (look at `[hpx-selective]` stats:
     packet_matches >= 1).
   - Mirror the trait check from `test_hpx_selective_mul_mat_q4_k_repacked`.
2. Extend `tests/hpx/CMakeLists.txt` accordingly.
3. Negative tests — exclusive-consumer guard:
   - Add a fourth consumer of `normed_x` (e.g., a side branch that uses it).
     Assert the matcher rejects and falls through.
   - Replace V weight with non-repacked Q4_K. Assert reject.
   - Replace V weight with Q6_Kx8 (first cut should reject; will be flipped
     to accept after Q6 follow-up).

## 14. A/B validation plan

End-to-end on TinyLlama Q4_K_M, mirroring the Q4K8 protocol exactly:

- **A**: selective + MLP packet on, **QKV packet OFF**, Q4K8 packet ON.
- **B**: selective + MLP packet on, **QKV packet ON**, Q4K8 packet ON.

Capture:
- bit-identical token sequences at n=16, 64, 128 (md5 compare).
- per-decode `[hpx-selective]` stats:
  - lowered nodes (expect: drop by ~12 × 3 = ~36; the three QKV MUL_MATs
    leave the lowered set).
  - fallback nodes / runs (expect: drop a bit; RMS_NORM and MUL leave
    fallback for those 12 layers).
  - packet matches (expect: 22 + 12 = 34, or 22 if QKV matches reuse the
    existing packet_match counter — clarify in implementation).
  - packet nodes (expect: 66 + 60 = 126 first-cut; 66 + 110 = 176 after Q6).
- eval ms / 15 decodes and tok/s.

The same structure as `local/results/q4k8-edit5-validate/`. Save under
`local/results/<date>-qkv-packet-validate/`.

Gate to declare success:
- Tokens bit-identical vs A across n=16, 64, 128.
- Stat-line `packet_nodes` increases by the expected delta on every decode.
- `tok/s` improves vs A; absolute number does not need to beat C (the
  scheduler), but the *direction* must be positive within the selective
  regime.

---

## Compliance audit vs the Packet Rule (`docs/HPX_LOWER_OP.md` lines 197-211)

| # | rule | verdict | evidence |
|---|---|---|---|
| 1 | pattern repeats in real graph | ✓ | 22× per decode on TinyLlama Q4_K_M (one per attn block); first-cut narrowing matches 12 layers, still repeats |
| 2 | removes ≥ 2 dispatch units | ✓ | 5 nodes → 1 packet = **4 boundaries removed per match**, well above the floor |
| 3 | stable dtype/layout/shape | ✓ (conditional) | Q is 2048×2048 Q4_Kx8 in all 22 layers; K is 2048×256 Q4_Kx8 in all 22; V is mixed Q4/Q6 across layers, so the first-cut matcher rejects mixed-trait sites — the *matched* pattern is stable; unstable cases are skipped at pre-scan |
| 4 | exclusive-consumer / write-through | ✓ | `norm_out` (1 consumer), `normed_x` (3 consumers, all in-packet) → packet-internal; `q_proj`/`k_proj`/`v_proj` write through; pre-scan rejects on any fourth consumer of `normed_x` (§9, §10) |
| 5 | cache key captures structural assumptions | ✓ | key carries `cols`, `out_cols_q`, `out_cols_kv`, `rows`, three `w_row_stride`, three `w_trait` — covers shape, repack stride, and per-projection type, with forward-compat for Q6 V (§11) |
| 6 | A/B validation plan | ✓ | all 5 sub-items present in §14: off vs on, bit-identical (md5 at n=16/64/128), dispatch count, packet_nodes, tok/s |
| 7 | not scheduler logic | ✓ | fused gemv pattern, identical category to existing FFN packets; no node reorder, backend assignment, or runtime mgmt |

**All 7 satisfied.** The only conditional is rule 3: stability holds *because
the matcher narrows to all-Q4_Kx8 sites at pre-scan*. If a future
implementation widens the matcher to mixed Q4/Q6 V before the Q6 HPX wrapper
exists, rule 3 would fail. The path to full 22/22 coverage is to land the
Q6 region/lower/packet plumbing first, then widen — at which point the
matcher would have a per-projection-trait dispatch and remain rule-3-stable.

---

## Implementation staging

The first cut can land in the same staged file order as Q4K8. Each file is a
separate, reviewable change; later files do not become callable until earlier
ones are present, so the build stays green between stages.

| stage | file(s) | change |
|---|---|---|
| 1 | `ggml-hpx-packet.h`, `.cpp` | New packet enum value (e.g. `GGML_HPX_PACKET_QKV_Q4K8`) and binding entry that names the per-projection weights, traits, and frame layout |
| 2 | `ggml-hpx-compose.h`, `.cpp` | New compose helper that emits the frozen sequence: `rms_norm → mul → quantize_q8_K → 3× gemv_q4_K_8x8_q8_K`, taking the frame-owned `norm_out`/`normed_x`/`q8_x` slots |
| 3 | `ggml-hpx-packet.cpp` | Compile/bind path for the new packet kind: allocate frame scratch (§Frame-owned scratch layout), wire kernels, validate trait support; fail fast on any unsupported trait so §Dispatch-invariant holds |
| 4 | `ggml-hpx-exec-selective.h` | Add the new cache type forward decl and a pointer field on `ggml_hpx_selective_packet_env` (mirrors `mlp_gate_up_glu_q4k8_cache`) |
| 5A | `ggml-hpx-exec-selective.cpp` | `qkv_proj_cache_key` + hash, `qkv_proj_packet_entry`, `ggml_hpx_qkv_proj_packet_cache` create/destroy |
| 5B | `ggml-hpx-exec-selective.cpp` | `lookup_or_compile_qkv_proj` — cache miss → compile/bind → store; honors §Dispatch-invariant by returning nullptr on any failure path |
| 5C | `ggml-hpx-exec-selective.cpp` | `prescan_qkv_proj_matches` — V-as-trigger pre-scan; populates `consumed_at_indices` and `trigger_at_index` only after `lookup_or_compile_qkv_proj` succeeds |
| 5D | `ggml-hpx-exec-selective.cpp` | Dispatch-loop branch: when an index is the trigger of a recorded match, call the packet entry; consumed indices are skipped |
| 6 | `src/llama-context.{h,cpp}` | Cache lifecycle (lazy create + destroy alongside `hpx_mlp_gate_up_glu_q4k8_cache`), `LLAMA_HPX_PACKET_QKV_Q4K8` env gate (chained behind `LLAMA_HPX_SELECTIVE_MLP_PACKET=1` like Q4K8), and the `penv.qkv_proj_cache = packet_active ? hpx_qkv_proj_cache : nullptr;` wiring step that mirrors the edit-5 pattern |

After stage 6, the same A/B protocol from §14 can run end-to-end. Until
stage 6 lands, each earlier stage is testable in isolation through the
existing test scaffolding (analogous to `test_hpx_selective_mul_mat_q4_k_repacked`).

---

## Verdict

**GOOD FIRST IMPLEMENTATION.**

The pattern is identical across 22 layers, the trigger and exclusive-consumer
story are clean, the cache key is small, the packet kernel reuses an existing
prequantized Q8_K helper (`gemv_q4_K_8x8_q8_K`) with minimal new math, and a
12-of-22 first cut already removes 48 dispatch boundaries per decode and
exercises every piece of the new plumbing end-to-end. The Q6_Kx8 path to
reach 22-of-22 is a clearly scoped, separate follow-up that should not gate
the first cut.
