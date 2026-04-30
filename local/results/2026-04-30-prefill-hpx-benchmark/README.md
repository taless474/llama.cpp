# HPX Prefill Benchmark — 2026-04-30

**Question:** Does HPX have a better chance on larger prefill work units than on single-token decode?

**Short answer:** No. Under the current TinyLlama Q4_K_M CPU-only setup, HPX prefill does not
show a useful speedup. The prefill topology is fully linear at the region level, so the current
HPX prefill executor has no coarse-grained parallelism to exploit. HPX serial is roughly
equivalent to the scheduler path (within run-to-run noise), while HPX parallel-proj is
consistently ~40% slower. This argues against continuing the current prefill integration as-is.

---

## Environment

| | |
|---|---|
| Commit | `7f05dcc2b70d8231ff8b10305fb6c2765c15bb2d` |
| Binary | `build-hpx-dag/bin/llama-simple` |
| Model | `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| Model | TinyLlama 1.1B, Q4_K_M, 636 MiB, GQA-4 (4 KV heads, 32 Q heads) |
| Hardware | Apple M4, 10 logical CPUs |
| GPU layers | 0 (CPU-only, `-ngl 0`) |
| Generation | `-n 1` (1 token generated, so timing is dominated by prefill) |
| Date | 2026-04-30 |

---

## Configurations

| Config | Env vars | Code path |
|---|---|---|
| **A — no-HPX** | _(none)_ | `ggml_backend_sched_graph_compute_async` |
| **B — HPX serial** | `LLAMA_USE_HPX=1` | `ggml_hpx_exec_run_prefill` (serial region dispatch) |
| **C — HPX parallel-proj** | `LLAMA_USE_HPX=1 LLAMA_HPX_PARALLEL_PROJ=1` | Same as B + parallel Q/K/V chain detection within large CPU regions |

**Note on selective:** `LLAMA_HPX_SELECTIVE_MUL_MAT` was not set in any of these runs. Even if
it were set, the guard `if (hpx_selective_mul_mat && !batched)` in `llama-context.cpp:2303`
is structurally unreachable for prefill — `batched = (n_tokens > 1)` is always true here.

---

## What this benchmark does and does not test

**Tested:** The HPX prefill region executor (`ggml_hpx_exec_run_prefill`). This is the coarse
orchestration layer that splits the compute graph into regions (CPU and BLAS) and dispatches them
serially, or — with `LLAMA_HPX_PARALLEL_PROJ=1` — attempts to parallelize Q/K/V chains within
large CPU regions using HPX futures.

**Not tested:** The selective lowering / fine-region DAG / packet path. This path is distinct
from the prefill region executor and was not exercised in any of these runs. Specifically:

- `LLAMA_HPX_SELECTIVE_MUL_MAT=1` was not set, so the selective path was never configured.
- Even if it had been set, `ggml_hpx_exec_graph_selective_mul_mat` is gated by
  `!batched` (line 2303 in `llama-context.cpp`), making it structurally unreachable for
  any prefill call (`n_tokens > 1`). This is not a bug — the selective path is explicitly
  decode-only: its packets are `team=DECODE` and assume a single-row shape.
- No `[hpx-selective]` log lines appeared in any of the 18 run stderr files. No `lowered_nodes`,
  `fallback_nodes`, `fallback_runs`, `packet_matches`, or `packet_dispatch_ns` were reported.
  This is expected: the code path that prints those values (`llama-context.cpp:2449–2461`) sits
  inside the `!batched` block and was never entered.
- The MLP/QKV packet path (`hpx_mlp_gate_up_packet`) has the same guard:
  `packet_eligible = hpx_mlp_gate_up_packet && !batched` (line 2365). Packets require
  `rows == 1`; prefill shapes are `rows > 1`. The packet path was not entered.

**Implication:** The results here say nothing about whether the selective lowering, fine-region
DAG, or packetization work is correct or useful for decode. Those remain separate questions
evaluated in decode-mode runs. This benchmark only shows that the current HPX prefill region
orchestration path provides no useful speedup for TinyLlama Q4_K_M at 512–1024 token prefill.

---

## Prompts

| Label | Reps | Actual tokens |
|---|---:|---:|
| p512 | 47 | 566 |
| p1024 | 94 | 1130 |

Text: `"The quick brown fox jumps over the lazy dog. "` × N.

---

## Raw repeated-run data

### p512 (566 tokens)

| config | r1 ms | r2 ms | r3 ms | mean ms | stdev ms | mean tok/s |
|---|---:|---:|---:|---:|---:|---:|
| A no-HPX | 1702.7 | 1602.7 | 1632.6 | **1646.0** | 41.9 | 344 |
| B HPX serial | 1661.8 | 1679.5 | 1686.8 | **1676.1** | 10.5 | 338 |
| C HPX parallel | 2318.2 | 2321.0 | 2295.4 | **2311.5** | 11.5 | 245 |

### p1024 (1130 tokens)

| config | r1 ms | r2 ms | r3 ms | mean ms | stdev ms | mean tok/s |
|---|---:|---:|---:|---:|---:|---:|
| A no-HPX | 3378.4 | 3371.0 | 3438.8 | **3396.1** | 30.4 | 333 |
| B HPX serial | 3388.2 | 3466.9 | 3458.1 | **3437.8** | 35.2 | 329 |
| C HPX parallel | 4831.8 | 4686.6 | 4828.7 | **4782.3** | 67.7 | 236 |

---

## Summary table

| config | prompt size | prompt tokens | mean eval ms | stdev ms | mean tok/s | vs A |
|---|---:|---:|---:|---:|---:|---:|
| A no-HPX | ~512 | 566 | 1646 | 42 | 344 | — |
| B HPX serial | ~512 | 566 | 1676 | 11 | 338 | +1.8% |
| C HPX parallel | ~512 | 566 | 2312 | 12 | 245 | **+40%** slower |
| A no-HPX | ~1024 | 1130 | 3396 | 30 | 333 | — |
| B HPX serial | ~1024 | 1130 | 3438 | 35 | 329 | +1.2% |
| C HPX parallel | ~1024 | 1130 | 4782 | 68 | 236 | **+41%** slower |

---

## Is A vs B within noise?

**p512:** B mean − A mean = +30 ms. A stdev = 42 ms, B stdev = 11 ms. The difference is less
than one stdev of A and the distributions overlap. **Within noise.**

**p1024:** B mean − A mean = +42 ms. A stdev = 30 ms, B stdev = 35 ms. The difference is ~1.2
stdev. This is borderline — consistent with a small fixed overhead (~40 ms) from HPX region
analysis and region dispatch setup, or within noise. Either way the effect is **not useful**:
even if real, +42 ms on a 3400 ms prefill is a 1.2% penalty, not a benefit.

**Conclusion:** HPX serial prefill is at best neutral and at worst adds ~40 ms of fixed overhead.
It does not provide any measurable speedup at either prompt size.

## Is C consistently slower?

Yes, strongly. C is +40% at p512 and +41% at p1024. The stdev for C is small (12 ms at p512,
68 ms at p1024) relative to the ~665–1386 ms excess. All three C runs at both sizes are slower
than every A and B run at the corresponding size. **The parallel-proj path is consistently and
substantially slower.**

---

## Why HPX serial ≈ baseline

`LLAMA_HPX_DUMP_TOPO=1` on the p256 prompt reveals:

```
[hpx-topo] prefill plan:  n_nodes=689  n_regions=45  sched_splits=1
[hpx-topo] dependency chain: fully linear — no independent regions
```

The 45 regions (alternating CPU ~27-node blocks and BLAS 4-node blocks) form a **strictly serial
chain**: every region has `prev = i−1`. There is no coarse-grained parallelism at the region
level. `ggml_hpx_exec_run_prefill` in serial mode calls `ggml_backend_graph_compute` on each
region in sequence, which is functionally identical to what the scheduler does, just with an
extra topology-analysis pass on the first call.

## Why HPX parallel-proj is consistently slower

`LLAMA_HPX_PARALLEL_PROJ=1` runs `detect_parallel_chains` inside each large CPU region (≥20
nodes), attempting to run Q/K/V projections concurrently as HPX tasks.

TinyLlama uses GQA-4 (4 KV heads, 32 Q heads):
- Q projection: 2048 → 2048 (32 heads × 64 dim) — large
- K projection: 2048 → 256 (4 heads × 64 dim) — 8× smaller than Q
- V projection: 2048 → 256 (4 heads × 64 dim) — 8× smaller than Q

Even if Q, K, V run concurrently, K and V finish ~8× faster than Q. The critical path remains Q
alone. HPX task creation, future synchronization, and thread-pool interaction add ~665–1386 ms
per forward pass without changing the critical path at all. The overhead grows roughly linearly
with prompt length, consistent with it applying per-layer.

No size-scaling benefit emerges: the ratio C/A ≈ 1.40× is flat across p512 and p1024.

---

## Warnings / Errors

None. All 18 runs completed normally. No HPX abort, no fallback warnings.
`graphs reused = 0` in all runs (expected: different prompt sizes trigger different graph sizes).

---

## Commands (reproducible)

```bash
export MODEL=models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
RUN=local/results/2026-04-30-prefill-hpx-benchmark

# Run A — no HPX baseline (3× each size)
for SIZE in p512 p1024; do
  PROMPT=$(cat "$RUN/prompt_${SIZE}.txt")
  for R in 1 2 3; do
    build-hpx-dag/bin/llama-simple -m "$MODEL" -n 1 -ngl 0 "$PROMPT" \
      > "$RUN/A-nohpx-${SIZE}-r${R}.stdout.txt" \
      2> "$RUN/A-nohpx-${SIZE}-r${R}.stderr.txt"
  done
done

# Run B — HPX serial prefill (3× each size)
for SIZE in p512 p1024; do
  PROMPT=$(cat "$RUN/prompt_${SIZE}.txt")
  for R in 1 2 3; do
    LLAMA_USE_HPX=1 build-hpx-dag/bin/llama-simple -m "$MODEL" -n 1 -ngl 0 "$PROMPT" \
      > "$RUN/B-hpx-serial-${SIZE}-r${R}.stdout.txt" \
      2> "$RUN/B-hpx-serial-${SIZE}-r${R}.stderr.txt"
  done
done

# Run C — HPX parallel-proj (3× each size)
for SIZE in p512 p1024; do
  PROMPT=$(cat "$RUN/prompt_${SIZE}.txt")
  for R in 1 2 3; do
    LLAMA_USE_HPX=1 LLAMA_HPX_PARALLEL_PROJ=1 build-hpx-dag/bin/llama-simple \
      -m "$MODEL" -n 1 -ngl 0 "$PROMPT" \
      > "$RUN/C-hpx-parallel-${SIZE}-r${R}.stdout.txt" \
      2> "$RUN/C-hpx-parallel-${SIZE}-r${R}.stderr.txt"
  done
done

# Topology inspection
LLAMA_USE_HPX=1 LLAMA_HPX_DUMP_TOPO=1 build-hpx-dag/bin/llama-simple \
  -m "$MODEL" -n 1 -ngl 0 "$(cat $RUN/prompt_p256.txt)" 2>&1 | grep hpx-topo
```

---

## Recommendation

**Stop QKV packetizing. Stop current prefill executor work. Write up decode/selective/packetization findings.**

Under the current TinyLlama Q4_K_M CPU-only setup, HPX prefill does not show a useful speedup.
The prefill topology is fully linear at the region level, so the current HPX prefill executor has
no coarse-grained parallelism to exploit. HPX serial is roughly equivalent to the scheduler path,
while HPX parallel-proj is consistently ~40% slower. This argues against continuing the current
prefill integration as-is.

This result does not mean prefill is impossible for HPX or that HPX can never help prefill. It
means the current design — coarse linear region dispatch on a single transformer forward pass —
does not provide a viable integration point. The decode/selective/packetization findings stand
independently and are not affected by this result.

If HPX is revisited for prefill in a new design, the granularity would need to change
fundamentally — for example, run-level orchestration across multiple concurrent requests, true
multi-sequence batching, or request-level scheduling — not region-level dispatch within a single
linear forward pass graph.
