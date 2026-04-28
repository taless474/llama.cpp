# Step 6 — real TinyLlama Q4_K_M correctness/stats with the new
# CPU_REPACK Q4_K_8x8_q8_K lowered path

## What was measured

Single A/B comparison on the same `build-hpx-bench` binary
(`-DGGML_HPX=ON -DGGML_HPX_REGION_DAG=ON`, commit `dcf25c1d8`):

- **run_off**: `LLAMA_USE_HPX=1` only — selective path disabled, full
  graph goes through the normal scheduler.
- **run_on**:  `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1
  LLAMA_HPX_SELECTIVE_HIST=1 LLAMA_HPX_SELECTIVE_STATS=1
  LLAMA_HPX_SELECTIVE_DEBUG=1` — selective path engaged.

Same model, prompt, `-n 16`, `-ngl 0`, same binary, ~30s apart.

Binary: `/Users/unick/Desktop/hpx/llama-hpx/build-hpx-bench/bin/llama-simple`
Model:  `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

## Goal of this step

Confirm two things, per the locked plan ("treat '134 nodes move to
lowered' as the expected result, not an assumption — if fewer lower,
log the skipped shapes/reasons"):

1. The post-patch predicate routes every CPU_REPACK Q4_K decode
   MUL_MAT through the lowered path, not the fallback.
2. Output text is bit-identical to the selective-off path.

This run is **not** a perf measurement — that is Step 7, on a quiet
machine with proper alternating-order A/B. See "Preliminary perf
observation" below for a flag, not a number.

## Result 1 — output bit-identical (correctness)

```
$ diff run_off.stdout run_on.stdout && echo OK
OK
```

The decoded 16-token completion is character-for-character identical
between selective-off and selective-on. No FP drift, no kernel
swap-out under us, no stride bug.

## Result 2 — node histogram match

`[hpx-selective-hist]` block from `run_on.stderr` (one block per
process, fired on the first decode graph):

```
[hpx-selective-hist] n_nodes=689
[hpx-selective-hist]   MUL_MAT/Q4_K-repacked    = 134  shapes(cols x out_cols x rows = count):
                                                          2048x2048x1=44 2048x256x1=34
                                                          2048x5632x1=44 5632x2048x1=12
[hpx-selective-hist]   MUL_MAT/Q4_K-nonrepacked = 0
[hpx-selective-hist]   MUL_MAT/F32xF32          = 0
[hpx-selective-hist]   MUL_MAT/other-quant      = 21  (Q6_K output proj + friends; not lowerable)
[hpx-selective-hist]   MUL_MAT/other            = 0
```

134 Q4_K-repacked MUL_MATs per decode graph, in four shapes. All four
satisfy `out_cols % 8 == 0`:

| cols | out_cols | nodes/step | NB_COLS=8-aligned? |
|-----:|---------:|-----------:|:-------------------|
| 2048 |     2048 |         44 | yes (256 tiles) |
| 2048 |      256 |         34 | yes (32 tiles)  |
| 2048 |     5632 |         44 | yes (704 tiles) |
| 5632 |     2048 |         12 | yes (256 tiles) |

So all 134 are accepted by `ggml_hpx_is_q4k_8x8_repacked` and routed
through `ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range` — none skipped.

## Result 3 — stats counters stable across all 15 decode steps

```
$ grep "hpx-selective\] lowered=" run_on.stderr | awk '{print $2,$3,$4,$5}' | sort -u
lowered=189 fallback=500 packet=0(0 nodes)
```

Every decode step reports the same `lowered=189 fallback=500
packet=0`. No shape-dependent fallback, no late-graph drift. 134 of
the 189 lowered nodes are the Q4_K MUL_MATs we just routed; the
other 55 are non-MUL_MAT ops (RMS_NORM/MUL/GLU/etc) that the lowered
path already handled. The 500 fallback nodes break down as:

- 21 Q6_K MUL_MATs (output projection — different trait, no lowered
  kernel; expected fallback).
- ADD (44), CPY (1), RESHAPE (88), VIEW (110), PERMUTE (66), GET_ROWS
  (3), SET_ROWS (44), ROPE (44), FLASH_ATTN_EXT (22), and others.
  All of these are either zero-cost view ops or ops that are not
  in the selective lowering case list. Untouched by this patch.

**Pre-patch baseline** (counterfactual): with the old prescan
(`w->extra == nullptr` only), all 134 Q4_K-repacked nodes would have
classified as fallback, and `selective_should_engage` likely would
have returned `false` (no lowerable MUL_MAT), so the whole graph
would have skipped the selective path entirely with the
`[hpx-selective] disabled: no lowerable MUL_MAT in graph` warning.

The post-patch run shows neither the `disabled: ...` warning nor any
unexpected fallback count — the prescan, lower-op acceptance, and
run_range dispatch are in lockstep.

## Preliminary perf observation (NOT a measurement)

For completeness, the `llama_perf_context_print` summary from the
two runs:

| condition           | eval time | tokens/s |
|---------------------|----------:|---------:|
| selective-off       |   146.24 ms / 15 runs |  102.57 t/s |
| selective-on (this) |  2024.27 ms / 15 runs |    7.41 t/s |

This is a ~14× slowdown. **Do not trust this number as a perf
result** — this run had HIST + STATS + DEBUG all on, with a
per-step `[hpx-selective] lowered=...` line going to stderr inside
the timed region. But the magnitude is large enough that print
overhead alone cannot explain it.

The most likely real factor: the 21 Q6_K fallback nodes per step
are dispatched as one-node `ggml_backend_graph_compute` calls
through the configured CPU threadpool — which is HPX, since
selective-on means we have HPX initialized — so each Q6_K node
pays coarse-substrate per-call cost on fine-grained work. 21 nodes
× 16 tokens × per-call overhead is consistent with a multi-second
penalty. The comment in `src/llama-context.cpp:2324-2333`
anticipates this exact problem ("paying coarse-substrate per-call
cost on fine-grained work"); the existing
`selective_should_engage` guard avoids it for graphs with **zero**
lowerable MUL_MAT, but not for graphs with mostly-but-not-all
lowerable MUL_MAT. We just turned a "all-fallback" graph into a
"mostly-lowered + 21 Q6_K fallbacks" graph, and the per-call cost
on those 21 nodes is now dominating.

This is exactly the kind of finding the user said to surface
before Step 7 ("if fewer lower, log the skipped shapes/reasons
before interpreting performance"). 134/134 lower as expected; the
slowdown is not from the lowered path itself but from the now-paid
per-node fallback dispatch through the HPX threadpool on the 21
Q6_K nodes.

## Files in this directory

- `run_off.stdout` — selective-off decoded text
- `run_off.stderr` — selective-off log (timings, no selective lines)
- `run_on.stdout`  — selective-on decoded text (bit-identical to off)
- `run_on.stderr`  — selective-on log (histogram + 15 stats lines)
- `run_on_fbhist.stdout` — selective-on text with FALLBACK_HIST only
- `run_on_fbhist.stderr` — selective-on log with per-shape fallback histogram
- `run_on_fbhist_8k.stdout` — same, after the scratch 4 KB → 8 KB fix
- `run_on_fbhist_8k.stderr` — same, after the fix; q4_K rows are gone

## Per-shape fallback histogram (cheap-instrumentation result)

Added in this session as namespace-local helpers in
`ggml-hpx-exec-selective.cpp`, gated on
`LLAMA_HPX_SELECTIVE_FALLBACK_HIST=1`. Buckets every fallback dispatch
by (op, w_type, cols, out_cols, rows), accumulates count and total ns,
prints sorted-by-total at exit. Records at both fallback sites (the
resource-rejected branch and the plain `lower_op == false` branch).

Run config: `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1
LLAMA_HPX_SELECTIVE_FALLBACK_HIST=1` (no STATS, no DEBUG, no HIST —
quiet stderr inside the timed region). Same prompt, `-n 16`, `-ngl 0`.
Eval reports `131.26 ms/token, 7.62 t/s` — same as the noisy
selective-on run, so print overhead is not the explanation.

Output, sorted by total time across all 15 decode steps:

| op             | w_type | shape (cxoxr)    | count | total_ms | mean_us |
|----------------|--------|------------------|------:|---------:|--------:|
| ROPE           | -      | -                |   660 |   143.16 |  216.91 |
| ADD            | -      | -                |   660 |    87.23 |  132.16 |
| MUL_MAT        | q6_K   | 5632 × 2048 × 1  |   150 |    65.79 |  438.56 |
| **MUL_MAT**    | **q4_K** | **5632 × 2048 × 1** | **180** | **61.62** | **342.32** |
| RMS_NORM       | -      | -                |   675 |    39.37 |   58.33 |
| FLASH_ATTN_EXT | -      | -                |   330 |    24.28 |   73.56 |
| MUL_MAT        | q6_K   | 2048 × 32000 × 1 |    15 |    18.69 | 1246.11 |
| RESHAPE        | -      | -                |  1320 |    11.42 |    8.65 |
| MUL_MAT        | q6_K   | 2048 × 256 × 1   |   150 |    10.87 |   72.47 |
| VIEW           | -      | -                |  1650 |    10.04 |    6.08 |
| PERMUTE        | -      | -                |   990 |     5.65 |    5.71 |
| SET_ROWS       | -      | -                |   660 |     4.25 |    6.43 |
| CPY            | -      | -                |    15 |     1.12 |   74.79 |
| GET_ROWS       | -      | -                |    45 |     0.43 |    9.57 |

Total fallback time across 15 decode steps: ~484 ms → ~32 ms/token.
Total lowered time per token (from earlier stats range): ~96 ms.
Sum ≈ 128 ms/token, matching the observed 131 ms/token decode.

## Two findings that change the picture

### Finding 1 — the original Q6_K hypothesis was wrong

Q6_K total = 65.79 + 18.69 + 10.87 = ~95.4 ms across 15 steps, only
about 20% of fallback cost. ROPE alone (143 ms) is bigger than all
Q6_K fallback combined; ROPE + ADD = 230 ms is 47% of fallback.

Even if we routed Q6_K through a hypothetical lowered kernel, we
would save at most ~6 ms/token. The selective-on regression is
~120 ms/token. So the original Step-7-blocking story ("the 21 Q6_K
nodes per step are dominating") is not the right diagnosis.

### Finding 2 — 12 Q4_K-repacked nodes/step are silently falling back

The fallback histogram includes a row of `MUL_MAT q4_K
5632×2048×1 count=180`. 180/15 = 12 per step. That matches exactly
the `5632×2048×1=12` shape in the prior `[hpx-selective-hist]`
output for Q4_K-repacked. So those 12 Q4_K-repacked nodes per step
**are not actually lowering** — they are accepted by the prescan
(`ggml_hpx_is_q4k_8x8_repacked` returns true and `out_cols=2048`
satisfies `% 8 == 0`), but rejected later inside `ggml_hpx_lower_op`.

The reject site is `ggml-hpx-lower.cpp:287-288`:

```cpp
const size_t q8k_row = ggml_row_size(GGML_TYPE_Q8_K, cols);
if (q8k_row > GGML_HPX_LOWERING_SCRATCH_BYTES) return false;
```

`GGML_HPX_LOWERING_SCRATCH_BYTES = 4096`. Q8_K row bytes by cols:

| cols | q8k_row_bytes | fits 4096? |
|-----:|--------------:|:-----------|
|  256 |           292 | yes |
| 2048 |          2336 | yes |
| **5632** | **6424** | **no — falls back** |

The MLP down-projection has cols=5632 (`5632→2048`). The Q8_K input
quantization scratch needed is 6.4 KB; the per-op stack arena is
4 KB. The lower-path can't fit it, so it bails to the CPU fallback.
This is why the synthetic test passed (its largest cols was 2048
in the QKV/MLP-up shapes) but the real graph has 12 nodes/step that
never lower.

Two ways to fix:

a) Bump `GGML_HPX_LOWERING_SCRATCH_BYTES` (8 KB or more). Costs
   ~4 KB extra per per-op stack frame on every selective node.
b) Heap-allocate scratch in the rejected branch (avoid stack growth
   for the rare large-cols case).

Option (a) is the minimal change. The arena is per-op stack-local,
so a one-time 4 KB bump on a path that's only entered for selective
graphs is acceptable. It needs a corresponding synthetic test cell
with `cols=5632` (the down-projection shape) to lock the fix.

### Coverage fix applied

- `GGML_HPX_LOWERING_SCRATCH_BYTES` 4096 → **8192**
  (`ggml/src/ggml-hpx/ggml-hpx-lower.h:41`). New comment cites this
  result dir and notes the new ceiling
  (`ggml_row_size(Q8_K, 7168) = 8176 bytes`, ≤ 8192).
- New synthetic test cell
  `SelectiveMulMatQ4KRepacked.MLPDownProjectionShape_5632x2048`
  (`tests/hpx/test_hpx_selective_mul_mat_q4_k_repacked.cpp`).
  All 5 cells `[ PASSED ]` after the bump.

### Confirmation on the real graph

`run_on_fbhist_8k.stderr` after the bump contains zero `q4_K` rows
in `[hpx-selective-fallback-hist]`:

```
q6_K | 5632x2048x1 | 150 | 67.67 ms | 451.15 us
q6_K | 2048x32000x1 |  15 | 18.27 ms | 1218.19 us
q6_K | 2048x256x1   | 150 | 11.99 ms |   79.92 us
```

Plus ROPE/ADD/RMS_NORM/FLASH_ATTN_EXT/RESHAPE/etc. as before. The
prescan / lower_op mismatch on cols=5632 is gone; the 12
Q4_K-repacked nodes/step at shape 5632×2048×1 now lower instead of
silently falling back. Decoded text bit-identical to selective-off.

This is a coverage/correctness fix, not a perf intervention. The
~120 ms/token regression remains; per Finding 2's math, the fix
saves at most ~4 ms/token. Step 7 framing question (when/whether
selective should engage on small decode graphs) is still open.

## What this changes about Step 7

Even fixing both findings does not close the perf gap:

- Finding 1 (Q6_K theory wrong): no easy savings.
- Finding 2 (5632 cols): saves ~62 ms across 15 steps = ~4 ms/token.
- ROPE+ADD fallback dispatch: ~15 ms/token of overhead from
  per-node `ggml_backend_graph_compute(view)` calls on cheap nodes.

Selective-off runs the entire graph in 9.75 ms/token because
ggml's backend compiles all 689 nodes into one whole-graph
dispatch, amortizing the per-call setup cost. Selective-on does
689 individual dispatches per token: 189 lowered (each via
`hpx::async([&]{run_region_group}).get()` ≈ 5–10 µs bridge cost)
+ 500 fallback (each via a one-node `graph_compute(view)` ≈
6–217 µs).

Per-node dispatch overhead alone, even with zero kernel cost,
accounts for ~30+ ms of the 120 ms regression. The lowered Q4_K
kernel itself is genuinely faster (1.30–1.59× per the earlier
synthetic bench), but it's running on 122/134 of the eligible
nodes and the saved time is being eaten by per-node setup on the
500 fallback nodes that now go one-at-a-time instead of batched.

This is a structural cost of the current selective design on
small-graph decode workloads, not a kernel-quality issue. The
synthetic A/B that motivated this work (1.30–1.59×) measured only
the gemv kernel, not the per-node dispatch overhead times the rest
of the graph.

## Open question for Step 7

Step 7 as originally framed (live perf A/B comparing selective-off
vs selective-on) is now likely to confirm a regression rather than
a win. Three paths from here:

1. **Fix Finding 2 (4 KB → 8 KB scratch + add `cols=5632`
   synthetic cell), run Step 7 anyway.** Confirms the regression
   magnitude; locks the synthetic test against the cols-5632 silent
   fallback. Cheap to do.

2. **Re-scope: don't use selective dispatch for tiny-graph decode.**
   The current `selective_should_engage` guard is "any lowerable
   MUL_MAT exists"; that's too permissive for decode. A stricter
   gate ("lowered MUL_MAT count × per-op work × workers >>
   selective overhead") would fall back to the whole-graph
   scheduler for TinyLlama-class decode and only engage on
   prefill-class graphs. The synthetic gemv win still applies on
   prefill or on larger models where dispatch overhead amortizes.

3. **Restructure: batch the lowered region groups.** Instead of
   one `hpx::async(...).get()` per lowered MUL_MAT, queue them
   into a single HPX `for_loop` over a list of (ctx, run_range).
   Eliminates the per-node bridge cost. Larger change; defer.

Recommendation: option 1 first (fix the silent fallback and lock
it with a synthetic cell — that's a correctness/coverage bug
regardless of perf), then option 2 framing for Step 7 itself
("when does selective help?") rather than option-3 structural
work.
