# Design plan — CPU_REPACK-aware HPX lowering for Q4_K MUL_MAT

Status: design only, no live-path code yet.

This plan extends the selective HPX lowering path so that
`q4_K_8x8_q8_K`-traited (CPU_REPACK) Q4_K weights — which today fall back
to the CPU backend — are lowered into a two-region HPX group built on the
`ggml_gemv_q4_K_8x8_q8_K` kernel (the same kernel
`bench-cpu-repack-q4k-gemv` and `bench-hpx-route-q4k-gemv` use).

Bench evidence supporting this plan:
- `hpx-bench/results/2026-04-27-hpx-vs-ggml-4t-2048x5632/` (gate/up at 4w: 1.30x)
- `hpx-bench/results/2026-04-27-hpx-vs-ggml-3t-2048x2048/` (Q/K/attn_out at 3w: 1.59x)
- `hpx-bench/results/2026-04-27-hpx-bridge-ablation/` (in_hpx mode → 1.51x / 1.82x; bridge ≈ 3–7 us/iter)

Live-path savings target (Q4_K MUL_MAT alone, TinyLlama Q4_K_M decode, M4):
~2.29 ms / step (in_hpx-equivalent) vs ~5.88 ms / step today.

---

## 1. Exact integration point

**File:** `ggml/src/ggml-hpx/ggml-hpx-lower.cpp`

The current Q4_K MUL_MAT acceptance branch starts at line 243
(`if (w->type == GGML_TYPE_Q4_K && x->type == GGML_TYPE_F32)`).  Today
that branch immediately rejects any tensor whose `extra != nullptr`
(line 249) because it then unconditionally builds a region whose kernel is
`ggml_vec_dot_q4_K_q8_K` (region-exec.cpp:436), which cannot read the
8x8-repacked layout.

**Change:**

Split the Q4_K branch into two sub-branches, gated by a trait probe:

```
if (w->type == GGML_TYPE_Q4_K && x->type == GGML_TYPE_F32)
{
    ...common gates (env kill-switch, rows==1, cols%256, contiguity, scratch fits)...

    if (w->extra == nullptr) {
        // existing path: build R0 (quantize) + R1 (vec_dot_q4_K_q8_K)
    } else {
        const char * trait = ggml_repack_extra_traits_name(node);
        if (!trait || std::strcmp(trait, "q4_K_8x8_q8_K") != 0) return false;
        ...trait-specific gates (out_cols % NB_COLS, w nb stride for repacked layout)...
        // new path: build R0 (quantize, same as above) + R1' (gemv_q4_K_8x8_q8_K)
    }
}
```

Crucially:
- The line-249 hard reject becomes a fork.  Q4_K with no `extra` keeps the
  existing behaviour; Q4_K with the specific 8x8 trait gets the new path.
- The R0 quantize region is *identical* to the existing one — same
  ctx struct (`ggml_hpx_quantize_q8_k_f32_ctx`), same kernel function, same
  scratch sink.  Only R1 changes.
- Other repacked Q4_K traits (`q4_K_8x4_q8_K`, RISC-V `q4_K_16x1_q8_K`)
  fall through and remain on the CPU backend fallback.  No silent
  acceptance of unrelated traits.

**Companion site:** `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp:727-730`

`q4k_decode_safe` (the prescan-style classifier used in
`ggml_hpx_can_lower_graph`) currently requires `w->extra == nullptr`.
That check needs to widen to admit `extra != nullptr && trait ==
"q4_K_8x8_q8_K"`, in lockstep with the lower-op change above, so a graph
full of repacked Q4_K weights does not get classified as “nothing
lowerable, take the all-CPU path”.

---

## 2. Trait gating

**Public probe:** `ggml/src/ggml-cpu/repack.h:143`
`const char * ggml_repack_extra_traits_name(const struct ggml_tensor * op)`

Reads `op->src[0]->extra` and pointer-compares it against the program-
lifetime trait instances declared in `repack.cpp:4537-4577`.  Returns the
canonical name string (e.g. `"q4_K_8x8_q8_K"`) or `nullptr` if `extra` is
unset, or `"<unknown>"` if `extra` is non-null but doesn’t match any
known trait.

**Use in lowering:**

```
const char * trait = ggml_repack_extra_traits_name(node);
if (!trait) return false;                                  // not repacked
if (std::strcmp(trait, "q4_K_8x8_q8_K") != 0) return false; // wrong trait
```

Note: the helper expects the **op tensor**, not the weight tensor — it
reads `op->src[0]->extra`, so we pass `node` (the MUL_MAT op).  The
prototype bench (`tests/bench-cpu-repack-q4k-gemv.cpp:177`) already
follows that convention; the lower-op call site mirrors it.

This is a string compare against a named string literal — exactly two
ways for the gate to misfire (typo'd trait name; helper returns
`"<unknown>"`), both of which the unit test in §7 will catch.

**Symbol availability:** the helper is `extern "C"`, declared in
`repack.h`, and lives in the `ggml-cpu` static lib that `ggml-hpx`
already links against.  No CMake change needed.

---

## 3. Scratch layout

**Where it lives:**  `ggml_hpx_lowering::scratch` in
`ggml-hpx-lower.h` (the per-op lowering arena, `GGML_HPX_LOWERING_SCRATCH_BYTES`).
Already used by the existing Q4_K path (see lower.cpp:267-278 — `out->scratch`
is the destination of the F32→Q8_K quantize and the source for the dot kernel).

**Size:** `ggml_row_size(GGML_TYPE_Q8_K, cols)` for one Q8_K row.
For `cols=2048` this is **`(2048/256) × sizeof(block_q8_K)` = 8 × ≈292 = ~2.3 KB**.
Currently bounded by `GGML_HPX_LOWERING_SCRATCH_BYTES` (lower.cpp:268
already enforces `q8k_row <= GGML_HPX_LOWERING_SCRATCH_BYTES`).  No
change needed: the new path uses the **same row size** (cols are
unchanged; only out_cols differs between shapes; the activation row is
cols-sized).

**Owner / lifetime:** the `ggml_hpx_lowering` value `lo` is a stack
local in `ggml_hpx_exec_graph_selective_mul_mat`'s loop iteration
(exec-selective.cpp:1143).  `hpx::async([&lo, &res]{ ... }).get()` keeps
`lo` alive across the dispatch (line 1196-1201).  The new path
inherits this lifetime model verbatim — no new allocations, no
heap, no per-graph runtime scratch.

**No interaction with `ggml_hpx_runtime_scratch_*`.** That facility
exists for the packet/region-pool path and is orthogonal here.

---

## 4. Region structure

**Yes — still two regions, same shape as the existing F32 Q4_K branch.**

```
R0 — REDUCTION (serial, single-lane execution):
     ggml_hpx_quantize_q8_k_f32_run_range
     ctx: ggml_hpx_quantize_q8_k_f32_ctx { x = F32 input row,
                                            x_q8 = lo.scratch,
                                            cols }
     uses_resources: 0   (writes only into ctx->x_q8 which is lo.scratch)

R1 — MATMUL (parallel, n_lanes = effective worker count):
     **ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range**   ← NEW kernel wrapper
     ctx: ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx        ← NEW struct
       {
         w_q4k        : w->data,           (block_q4_Kx8 layout)
         x_q8         : lo.scratch,        (Q8_K activation, populated by R0)
         y            : node->data,        (F32 output)
         cols, out_cols,
         w_row_stride : w->nb[1],
         q8k_row_bytes: ggml_row_size(GGML_TYPE_Q8_K, cols),
         y_nb0        : node->nb[0],
       }
     uses_resources: 0

deps[0]: { src = R0, dst = R1 }
n_regions = 2, n_deps = 1
```

R1 body (the new run_range function in `ggml-hpx-region-exec.cpp`):

```cpp
void ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range(
    void * ctx_void, int /*ith*/, int /*nth*/,
    int64_t begin, int64_t end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * c = static_cast<ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx *>(ctx_void);

    // Align [begin, end) to NB_COLS=8 (mirrors repack.cpp:4370-4372).
    constexpr int64_t NB = 8;
    int64_t s = (begin % NB) ? begin + NB - (begin % NB) : begin;
    int64_t e = (end   % NB) ? end   + NB - (end   % NB) : end;
    e = std::min(e, c->out_cols);
    if (s >= e) return;

    ggml_gemv_q4_K_8x8_q8_K(
        static_cast<int>(c->cols),
        reinterpret_cast<float *>(reinterpret_cast<char *>(c->y) + s * c->y_nb0),
        c->y_nb0 / sizeof(float),                    // bs in elements? — verify against kernel signature
        reinterpret_cast<const char *>(c->w_q4k) + s * c->w_row_stride,
        c->x_q8,
        /*nrows=*/1,
        static_cast<int>(e - s));
}
```

Open detail to confirm before code: `ggml_gemv_q4_K_8x8_q8_K`'s `bs`
argument is the byte stride per output column from the kernel header —
the bench passes `static_cast<size_t>(out_cols)` (an element count) and
ggml's repack.cpp:4204 area passes the same — needs a re-read to make
sure we get this right.  Marking as a lowering-time invariant, not a
design-time blocker.

The MATMUL region's range-to-chunk mapping is handled by the existing
`launch_region_async` machinery in
`ggml-hpx-region-exec.cpp:475`/482 — same as the F32 Q4_K path and the
F32×F32 path.  We are not introducing new infrastructure.

The R1 work range is `[0, out_cols)`; HPX subdivides it across
`n_lanes` lanes inside `launch_region_async`.  This uses the
**fork-join executor's natural chunking**, not ggml's
`ggml_threadpool_chunk_set` formula.  This matches what the bench
showed wins.

---

## 5. HPX entry model

**One `hpx::async(...).get()` per lowered node — same as today.**

`ggml_hpx_exec_graph_selective_mul_mat` already wraps every lowered
group in a per-node async (exec-selective.cpp:1196).  The bridge
ablation says this hop costs **~3–7 us per node**.  At 134 lowered
Q4_K nodes per decode step, that is ~0.67 ms of bridge overhead — real
but a small fraction of the kernel time we save (~2.29 ms).

**Don't change the entry model in this PR.** Lifting the
per-node bridge requires a region-batching change to the executor (run
many regions inside one HPX async), which is independent of CPU_REPACK
support and orthogonal to this work.  The bridge ablation shows the
gain from removing it would be ~0.5 ms / step on top of the in-bridge
result — worth a follow-up but not a blocker.

The existing stat counters (`lowered_ns`, `fallback_ns`) wrap each
`hpx::async([&lo,&res]{...}).get()` so the live A/B will show the new
nodes moving from `fallback_ns` into `lowered_ns` automatically.  No
new instrumentation needed.

---

## 6. Worker count

**First cut: use `hpx::get_num_worker_threads()` — i.e., do not
shape-tune.**

Current behaviour (exec-selective.cpp:945-947):
```
const int lanes = (n_lanes > 0) ? n_lanes
                                : static_cast<int>(hpx::get_num_worker_threads());
```

Default startup uses `--hpx:threads=N` set from an env var or default
hardware concurrency.  In the live A/B, set `--hpx:threads=4` so all
nodes use 4 lanes.  This is the best compromise per the prior thread
sweep:
- 2048×5632 (gate/up, 44 ops):   peaks at 4t (we'll be at peak)
- 2048×2048 (Q/K/attn_out, 66 ops): peaks at 3t (we'll be 1 over peak — but on
  the in_hpx-mode HPX path the regression at +1 worker may be less
  severe than ggml's; the bridge ablation’s in_hpx 22.7us at 3w is already
  faster than ggml-3t at 41us, so even a moderate +1-worker regression
  on the smaller shape would still leave HPX ahead).

**Per-shape worker selection is a known follow-up**, not first-cut work.
It requires either (a) a per-node `n_lanes` override in the
`ggml_hpx_lowering` plan, set by the lower-op based on `out_cols`, or
(b) a new bench cell at 2048×2048 / 4 workers HPX to confirm the
single-thread-count assumption is good enough.  Option (b) costs one
bench run — defer the design lever for it until we have data.

---

## 7. Correctness tests

**File:** `tests/hpx/test_hpx_selective_mul_mat_q4_k_repacked.cpp` (new)

Pattern after `tests/hpx/test_hpx_selective_mul_mat_q4_k.cpp`
(GGML_HPX_REGION_DAG-gated, gtest, runs through
`ggml_hpx_exec_graph_selective_mul_mat`).  Differences:

1. **Allocate W via the CPU_REPACK buffer type** so `init_tensor`
   populates `extra` and `set_tensor` repacks in place — exactly what
   `bench-cpu-repack-q4k-gemv.cpp:110-151` does:

   ```
   ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
   ggml_backend_buffer_t       W_buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(W));
   ggml_tallocr_alloc(&W_alloc, W);
   ggml_backend_tensor_set(W, q4k_blob.data(), 0, w_bytes);
   ```

2. **Assert the trait** before running the executor:
   `ASSERT_STREQ(ggml_repack_extra_traits_name(y), "q4_K_8x8_q8_K");`

3. **Compare against ggml CPU backend** (the only legitimate reference for
   the repacked layout — the in-tree non-repacked vec_dot is
   numerically slightly different, so compare to whatever the CPU backend
   produces using its own kernel choice).  Tolerance same as the existing
   test (`EXPECT_NEAR ... 1e-3f` or as needed).

4. **Cells:** at minimum,
   - `MLPProjectionShape_2048x5632` (gate/up)
   - `SquareShape_2048x2048` (Q/K/attn_out)
   - one off-tile-boundary `out_cols` (e.g. 2056 = NB_COLS aligned but not
     a power of 2) to exercise the chunk-alignment math
   - `lowered_nodes == 1, fallback_nodes == 0` on each cell

This adds CMake target wiring in `tests/hpx/CMakeLists.txt`, mirroring
the existing `test_hpx_selective_mul_mat_q4_k` block (lines 213-228).

**Existing tests must continue to pass.** The non-repacked Q4_K path is
preserved verbatim, so `test_hpx_selective_mul_mat_q4_k.cpp`'s 5 cells
should still report `lowered_nodes == 1`.  CI signal: green on
`ctest -L main` and the new repacked test target.

---

## 8. Real-model test

**Existing instrument:** `ggml_hpx_selective_log_node_histogram`
(`ggml-hpx-exec-selective.cpp:749`) prints node counts including
`mul_mat_q4k_repacked` as a distinct bucket (already exists; see how it
sub-classifies MUL_MAT today).  This is the diff signal we read.

**Test recipe (CLAUDE.local.md-compliant, quiet machine, single rebuild):**

```
# baseline (today): no HPX selective
./build-baseline-no-hpx/bin/llama-bench -m TinyLlama-Q4_K_M.gguf -p 0 -n 16

# selective off:
LLAMA_HPX_SELECTIVE_MUL_MAT=0 \
  ./build-hpx-bench/bin/llama-bench -m TinyLlama-Q4_K_M.gguf -p 0 -n 16

# selective on, before this change (baseline-execution-map result):
LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_DEBUG=1 \
  ./build-hpx-bench/bin/llama-simple TinyLlama-Q4_K_M.gguf "..." -n 16

# selective on, AFTER this change:
# Same invocation; expected histogram diff — every Q4_K MUL_MAT row that
# previously printed under "fallback_nodes" should now print under
# "lowered_nodes". No new buckets.
```

**Pass criteria:**

- Histogram: 134 Q4_K-repacked nodes per decode step
  (= 22 layers × (gate=1+up=1+Q=1+K=1+V=1+attn_out=1) for TinyLlama)
  move from `fallback_nodes` to `lowered_nodes`.  Q6_K nodes
  (ffn_out × 22, lm_head × 1) stay in fallback (not in scope).
- Decode output text is identical between `selective=0` and
  `selective=1` runs (TinyLlama is deterministic at temperature 0; same
  prompt → same logits within FP rounding → same token sequence).  This
  is the live-path correctness check.
- No new fallback buckets, no error logs.

If the model outputs differ, the diff is a numerical bug in R1's
kernel-call wrapper (most likely the `bs`/stride/alignment math) — fix
in lower.cpp/region-exec.cpp before declaring done.

---

## 9. Performance test

**Three conditions on the same TinyLlama Q4_K_M model, alternating-order
A/B per `CLAUDE.local.md`:**

| condition          | binary                                | env                                               |
|---|---|---|
| C1 strict baseline | `build-baseline-no-hpx/bin/llama-bench` | (no HPX in build)                                |
| C2 HPX disabled    | `build-hpx-bench/bin/llama-bench`       | `LLAMA_HPX_SELECTIVE_MUL_MAT=0`                  |
| C3 HPX enabled     | `build-hpx-bench/bin/llama-bench`       | `LLAMA_HPX_SELECTIVE_MUL_MAT=1`                  |

C1 vs C2 measures the cost of the build itself (HPX runtime present but
inactive).  C2 vs C3 measures the live integration win.  C1 vs C3 is the
end-to-end story.

**Run protocol:**

- `-p 0 -n 16` (decode-only, 16 tokens) — enough samples for a
  stable eval-tok/s; matches the `2026-04-20-packet-baseline-quiet/`
  worked example.
- 3 batches, alternating order: B1=(C1,C2,C3), B2=(C3,C2,C1),
  B3=(C2,C1,C3).
- Quiet-machine pre/post snapshot per batch.
- Same `-n` across all conditions.
- Single rebuild per batch.

**Report buckets:**
- `eval_tok_s`, mean ± stdev across 3 batches per condition.
- From selective stats: `lowered_ms`, `fallback_ms`, `packet_ms`.
- Math sanity:
  `(C2.lowered_ms - C3.lowered_ms) + (C2.fallback_ms - C3.fallback_ms)`
  should equal the observed `C2.total - C3.total` total within rounding.
  If not, one of the buckets is mis-accounted.

**Win threshold to declare success:**
- C3 - C2 must show eval_tok/s improving with consistent direction
  across all 3 batches.
- The `lowered_ms` increase from C2→C3 must approximately equal the
  `fallback_ms` decrease — within rounding — proving the migration is
  net-fair (no work lost, no work double-counted).

**Predicted gain:** Bench upper bound is ~2.29 ms / step at the bridge-
included rate (~1.78 ms with bridge, ~2.29 ms without).  Per-step
budget on TinyLlama Q4_K_M decode at ~75 tok/s baseline ≈ 13.3 ms /
step, so 1.78 ms is **~13% improvement** in per-step time; eval tok/s
should improve commensurately, give or take overhead the bench did not
capture (e.g. attention KV writes, softmax — none of which we touch).

If observed tok/s improvement is significantly less than predicted, the
likely cause is bridge cost being higher in the integrated path than
in the bench (e.g. due to more small async hops per step from packet
dispatch interleaving with lowered nodes).  Diagnose via the
`lowered_ns` per-node histogram, not by re-running the perf bench.

---

## File-touch summary (preview only — not yet edited)

| file | change |
|---|---|
| `ggml/src/ggml-hpx/ggml-hpx-lower.cpp`            | fork Q4_K branch into `extra==null` (existing) and `trait==q4_K_8x8_q8_K` (new) |
| `ggml/src/ggml-hpx/ggml-hpx-region-exec.cpp`      | add `ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range` |
| `ggml/src/ggml-hpx/ggml-hpx-region-exec.h`        | add `ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx` + decl |
| `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp:727` | widen `q4k_decode_safe` to admit `extra != null && trait == q4_K_8x8_q8_K` |
| `ggml/src/ggml-hpx/ggml-hpx-lower.cpp:37`          | add static_assert for new ctx struct sizing |
| `ggml/src/ggml-cpu/repack.h`                       | (no change — public symbol already exists) |
| `tests/hpx/test_hpx_selective_mul_mat_q4_k_repacked.cpp` | NEW — gtest, mirrors existing repacked-Q4_K test |
| `tests/hpx/CMakeLists.txt`                         | wire the new test target |

No CMake gating change beyond the existing `if(GGML_HPX_REGION_DAG)`
block.  No public-API additions; the new ctx + run_range are internal
to ggml-hpx, the trait probe is already public.

---

## Out of scope (explicitly)

- Q6_K (ffn_out, lm_head) repacked lowering.  Same template, but a
  different kernel (`ggml_gemv_q6_K_8x8_q8_K`) and different trait
  (`q6_K_8x8_q8_K`).  Holdover for a follow-up; the design here makes
  Q6_K straightforward to add by analogy.
- Q5_K, Q2_K, Q4_0 repacked traits.  Not present in TinyLlama Q4_K_M.
- Per-node `n_lanes` selection (3 vs 4 by shape).  Defer pending data.
- Region-batching to amortize the per-node `hpx::async` bridge.
  Independent of CPU_REPACK support; addressed separately if/when the
  ~0.5 ms / step lever becomes a priority.
- Q8_Kx4 activation quantization (matrix variant).  Only matters for
  rows ≥ 4; decode is rows == 1.

---

## Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| `bs`/stride mismatch between bench and live (kernel argument convention) | medium | numerical mismatch on real model | unit test §7 with known-bit-equal CPU backend reference catches this before the model run |
| Per-shape thread sub-optimality (3 vs 4 worker question) | medium | up to ~10% of the gain | acceptable for first cut; revisit after live A/B if observed gain is well below predicted |
| Trait-name string typo | low | silent fallback, no correctness regression | unit test §7 explicitly compares trait string |
| `q4_K_8x4_q8_K` or RISC-V variants accidentally accepted | low | wrong kernel called → numerical mismatch | exact-match `strcmp`, not `strncmp`; risk lives only inside the trait-gate code we control |
| New ctx struct exceeds `GGML_HPX_LOWERING_CTX_BYTES_PER_REGION` | low | static_assert at compile-time — visible immediately | static_assert added at lower.cpp:37 |

---

## Decision points the user (you) should confirm before code

1. **First-cut worker count = 4 across all shapes.** OK to defer
   per-shape tuning to a post-merge follow-up?
2. **Bridge model unchanged.** OK to keep one async per node and tackle
   region batching separately?
3. **Reference for correctness tests = ggml CPU backend** (which uses
   the gemv kernel internally for repacked tensors).  OK?
4. **Live perf-test recipe** — `llama-bench -p 0 -n 16` plus 3
   alternating batches.  Should I extend `-n` for tighter
   bound-on-tok/s, or is 16 enough to start?
