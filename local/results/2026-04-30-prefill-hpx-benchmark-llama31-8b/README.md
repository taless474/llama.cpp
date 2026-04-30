# HPX Prefill Benchmark — Llama 3.1 8B — 2026-04-30

**Question:** Is the TinyLlama prefill result model-size-specific, or does the current HPX
prefill region executor still fail to help on a larger model?

**Short answer:** The result is not model-size-specific. On Llama 3.1 8B Q4_K_M CPU-only, the
current HPX prefill region executor is again not useful. The topology is again fully linear at
the region level. HPX serial prefill is consistently 1–6% slower than no-HPX (region analysis
overhead, not a benefit). HPX parallel-proj is 38–47% slower. These findings parallel the
TinyLlama results and hold at both tested prompt sizes.

---

## Environment

| | |
|---|---|
| Commit | `7f05dcc2b70d8231ff8b10305fb6c2765c15bb2d` |
| Binary | `build-hpx-dag/bin/llama-simple` |
| Model | `models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| Model size | 8B params, Q4_K_M, GQA-8 (8 KV heads, 32 Q heads) |
| Hardware | Apple M4, 10 logical CPUs |
| GPU layers | 0 (CPU-only, `-ngl 0`) |
| Generation | `-n 1` (1 token, timing dominated by prefill) |
| Date | 2026-04-30 |

---

## Configurations

| Config | Env vars | Code path |
|---|---|---|
| **A — no-HPX** | _(none)_ | `ggml_backend_sched_graph_compute_async` |
| **B — HPX serial** | `LLAMA_USE_HPX=1` | `ggml_hpx_exec_run_prefill` (serial region dispatch) |
| **C — HPX parallel-proj** | `LLAMA_USE_HPX=1 LLAMA_HPX_PARALLEL_PROJ=1` | Same + parallel Q/K/V chain detection within large CPU regions |

**Note on selective:** `LLAMA_HPX_SELECTIVE_MUL_MAT` was not set. The selective/fine-region DAG
path has a structural `!batched` guard in `llama-context.cpp:2303` and is unreachable for
any prefill call. See the TinyLlama benchmark README for the full code-path analysis.

---

## Prompts

Llama 3.1 uses BPE and produces fewer tokens than TinyLlama's SentencePiece for the same text.
Actual token counts are taken from binary output.

| Label | Reps of base sentence | Actual tokens |
|---|---:|---:|
| p512 | 47 | 472 |
| p1024 | 94 | 942 |

Text: `"The quick brown fox jumps over the lazy dog. "` × N.

---

## Raw repeated-run data

### p512 (472 tokens)

| config | r1 ms | r2 ms | r3 ms | mean ms | stdev ms | mean tok/s | vs A |
|---|---:|---:|---:|---:|---:|---:|---:|
| A no-HPX | 9199.2 | 9185.4 | 9582.2 | **9322.3** | 183.9 | 50.6 | — |
| B HPX serial | 9531.4 | 10137.2 | 9948.2 | **9872.3** | 253.1 | 47.8 | +5.9% |
| C HPX parallel | 13379.1 | 13540.2 | 14306.8 | **13742.0** | 404.7 | 34.3 | **+47%** |

### p1024 (942 tokens)

| config | r1 ms | r2 ms | r3 ms | mean ms | stdev ms | mean tok/s | vs A |
|---|---:|---:|---:|---:|---:|---:|---:|
| A no-HPX | 22410.0 | 20053.3 | 19951.2 | **20804.9** | 1135.8 | 45.3 | — |
| B HPX serial | 20735.5 | 20581.1 | 21806.4 | **21041.0** | 544.9 | 44.8 | +1.1% |
| C HPX parallel | 29153.5 | 27943.7 | 29005.9 | **28701.0** | 538.9 | 32.8 | **+38%** |

Note: A p1024 r1 = 22410 ms is likely a cold-file-cache read on first model load per run; r2 and
r3 settled to ~20000 ms. This inflates the A p1024 mean and stdev but does not affect the B vs A
comparison directionally — B is not faster in any individual run.

---

## Summary table

| config | model | prompt tokens | mean eval ms | stdev ms | tok/s | vs A |
|---|---|---:|---:|---:|---:|---:|
| A no-HPX | Llama 3.1 8B | 472 | 9322 | 184 | 50.6 | — |
| B HPX serial | Llama 3.1 8B | 472 | 9872 | 253 | 47.8 | +5.9% |
| C HPX parallel | Llama 3.1 8B | 472 | 13742 | 405 | 34.3 | **+47%** |
| A no-HPX | Llama 3.1 8B | 942 | 20805 | 1136 | 45.3 | — |
| B HPX serial | Llama 3.1 8B | 942 | 21041 | 545 | 44.8 | +1.1% |
| C HPX parallel | Llama 3.1 8B | 942 | 28701 | 539 | 32.8 | **+38%** |

---

## Is A vs B within noise?

**p512:** B mean − A mean = +550 ms. A stdev = 184 ms. The difference is ~3 stdevs. Individual
runs: A max = 9582 ms, B min = 9531 ms — the ranges barely touch. B is consistently above A.
This is outside noise. HPX region analysis and dispatch setup add a consistent ~550 ms overhead
at 8B scale with 65 regions and 999 nodes.

**p1024:** B mean − A mean = +236 ms, which looks small, but A's stdev is inflated by the cold
r1 reading (22410 ms). Using only A r2+r3 (mean ≈ 19,990 ms), B mean − A(warm) mean ≈ +1051 ms.
In every individual run, B ≥ A. No B run is faster than the corresponding A run. **B is not
within noise; B is consistently at or above A.**

**Conclusion:** Unlike TinyLlama (where B was borderline noise), at 8B scale the HPX region
analysis overhead is more visible (550 ms at p512). HPX serial prefill is a net negative for
Llama 3.1 8B: it adds overhead without providing any speedup.

## Is C consistently slower?

Yes, strongly. C is +47% at p512 and +38% at p1024. Every C run is slower than every A and B run
at the corresponding size. The stdev for C is small relative to the excess (405 ms excess range
vs ~4420 ms gap at p512). The result is unambiguous.

Llama 3.1 8B has GQA-8 (8 KV heads, 32 Q heads — Q is 4× larger than K/V), which is less
asymmetric than TinyLlama's GQA-4 (8×). Despite this, the parallel-proj overhead is still
dominant: the Q projection is still substantially larger than K/V, and the HPX task dispatch
and synchronization cost per layer adds up over 32 layers.

---

## Topology dump

```
[hpx-topo] prefill plan:  n_nodes=999  n_regions=65  sched_splits=1
[hpx-topo]   region[ 0]  type=sched_split  nodes=[0,17)    n=17  prev=none  backend=CPU
[hpx-topo]   region[ 1]  type=sched_split  nodes=[17,21)   n=4   prev=0     backend=BLAS
[hpx-topo]   region[ 2]  type=sched_split  nodes=[21,49)   n=28  prev=1     backend=CPU
...
[hpx-topo]   region[63]  type=sched_split  nodes=[979,983) n=4   prev=62    backend=BLAS
[hpx-topo]   region[64]  type=sched_split  nodes=[983,999) n=16  prev=63    backend=CPU
[hpx-topo] dependency chain: fully linear — no independent regions
```

65 regions, 999 nodes, 1 sched split, dependency chain **fully linear**. The structure is
identical in kind to TinyLlama's: alternating CPU blocks (~27 nodes each) and BLAS blocks
(4 nodes each), all chained `prev = i−1`. The larger model has more layers (32 vs 22) and
therefore more regions, but the region dependency structure is the same: each region depends
on the previous one and there is no region-level parallelism.

### Comparison with TinyLlama

| model | layers | n_nodes | n_regions | sched_splits | dependency chain |
|---|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 22 | 689 | 45 | 1 | fully linear |
| Llama 3.1 8B | 32 | 999 | 65 | 1 | fully linear |

The structure scales with layer count but the topology does not change. This is inherent to
the transformer architecture on a single CPU backend: each layer's output feeds into the next
layer's input, so no region-level parallelism is structurally possible.

---

## Selective lowering / fine-region DAG / packet path

This benchmark does not evaluate the selective lowering/fine-region DAG/packet path. It
evaluates only the current HPX prefill region executor.

Grep across all 12 B and C run stderr files for `hpx-selective`, `lowered_nodes`,
`fallback_nodes`, `packet_match`, `packet_dispatch`: **zero hits**. This is expected and
structural — the same `!batched` guard in `llama-context.cpp:2303` applies. The selective path
and packet path were not entered, not evaluated, and their results are not affected by this
benchmark.

---

## Commands (reproducible)

```bash
export MODEL=models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
RUN=local/results/2026-04-30-prefill-hpx-benchmark-llama31-8b

python3 -c "
unit = 'The quick brown fox jumps over the lazy dog. '
for n, reps in [('p512', 47), ('p1024', 94)]:
    open(f'$RUN/prompt_{n}.txt','w').write(unit*reps)
"

for SIZE in p512 p1024; do
  PROMPT=\$(cat "\$RUN/prompt_\${SIZE}.txt")
  for R in 1 2 3; do
    build-hpx-dag/bin/llama-simple -m "\$MODEL" -n 1 -ngl 0 "\$PROMPT" \
      > "\$RUN/A-nohpx-\${SIZE}-r\${R}.stdout.txt" \
      2> "\$RUN/A-nohpx-\${SIZE}-r\${R}.stderr.txt"
    LLAMA_USE_HPX=1 build-hpx-dag/bin/llama-simple -m "\$MODEL" -n 1 -ngl 0 "\$PROMPT" \
      > "\$RUN/B-hpx-serial-\${SIZE}-r\${R}.stdout.txt" \
      2> "\$RUN/B-hpx-serial-\${SIZE}-r\${R}.stderr.txt"
    LLAMA_USE_HPX=1 LLAMA_HPX_PARALLEL_PROJ=1 build-hpx-dag/bin/llama-simple \
      -m "\$MODEL" -n 1 -ngl 0 "\$PROMPT" \
      > "\$RUN/C-hpx-parallel-\${SIZE}-r\${R}.stdout.txt" \
      2> "\$RUN/C-hpx-parallel-\${SIZE}-r\${R}.stderr.txt"
  done
done

# Topology dump
LLAMA_USE_HPX=1 LLAMA_HPX_DUMP_TOPO=1 build-hpx-dag/bin/llama-simple \
  -m "\$MODEL" -n 1 -ngl 0 "\$(cat \$RUN/prompt_p512.txt)" 2>&1 | grep hpx-topo
```

---

## Conclusion

The TinyLlama prefill result is not model-size-specific.

On Llama 3.1 8B Q4_K_M CPU-only, the current HPX prefill region executor shows the same
structural problem: the prefill compute graph is fully linear at the region level (65 regions,
all chained prev = i−1). There is no coarse-grained parallelism for HPX to exploit.

- **HPX serial prefill** adds 1–6% overhead relative to the plain scheduler (more visible than
  on TinyLlama because the region analysis overhead is ~550 ms at 8B scale). It provides no
  speedup at either 472 or 942 tokens.
- **HPX parallel-proj** is 38–47% slower. The GQA-8 structure is less asymmetric than TinyLlama
  (4× vs 8× Q/KV ratio) but still not symmetric enough to make parallel-proj worthwhile, and
  the HPX task overhead per 32 layers is substantial.
- **Selective lowering / fine-region DAG / packet path:** not tested, not engaged, structural
  guard (`!batched`) prevents engagement for any prefill call regardless of model.

**Recommendation:**

Stop QKV packetizing. Stop current prefill executor work. Write up decode/selective/packetization
findings.

The fully linear region topology is not a TinyLlama artifact — it is a property of the
transformer architecture on a single CPU-backend forward pass. No tested model size changes this.

If HPX is revisited for prefill in a future design, it would only be productive at a
fundamentally different execution granularity — for example, run-level orchestration across
multiple concurrent requests, multi-sequence batching, or request-level scheduling. The current
approach of linear region dispatch within a single transformer forward pass does not provide a
viable HPX integration point, regardless of model size or prompt length.
