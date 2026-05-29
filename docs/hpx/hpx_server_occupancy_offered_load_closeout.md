# hpx-server occupancy / offered-load closeout

## Question

Exp17 exposed a moderate-concurrency c=4 throughput gap between `llama-server`
and `hpx-server`. The project needed to determine whether the gap came from HPX
control-plane overhead, unfair thread settings, server-path overhead, broken
admission/backlog behavior, or a workload artifact.

## Scope

This branch does not replace:

- `llama_decode`
- ggml graph construction
- the backend scheduler
- kernels
- the tokenizer
- sampler math
- model loading

The investigation is about the HPX serving-control layer around llama.cpp, not
about model execution.

## What was ruled out

- Unequal ggml compute threads were suspected, then ruled out by rerunning with
  both backends pinned to 4 compute threads. The c=4 gap remained after equal
  thread pinning.
- HPX engine idle/wakeup was not the cause.
- HPX per-iteration orchestration was not the cause.
- hpx-server handler overhead was not the cause.
- JSON parse, tokenization, detokenization, response JSON, HTTP response,
  future-result propagation, client resubmit gap, and TCP connection setup were
  all too small to explain the gap.
- Diagnostics perturbation was negligible in the final request-cycle
  decomposition.

Key numbers:

- Equal-thread c=4 baseline: `llama-server` 72.7 tok/s, `hpx-server` 60.1 tok/s.
- Request-cycle decomposition showed server-side overhead around the engine was
  microsecond-scale, approximately 40 us/request.
- c=4 diagnostics showed the issue was request residency / effective batch
  occupancy, not engine compute overhead.

## Occupancy / offered-load POC

Final POC setup:

- `llama-hpx-server`
- TinyLlama F32
- `n_seq_max = 4`
- `max_concurrent = 16`
- `--n-threads 4`
- `--hpx-os-threads 2`
- `--ctx-size 4096`
- prompt: `"Hello, my name is"`
- `decode_budget = 8`
- greedy
- diagnostics enabled
- stopped through `POST /shutdown` so engine diagnostics flushed
- no C++ changes; only `local/runs/occupancy-offered-load-2026-05-28/drive.py`
  was added

| Metric                           |       c16 offered load |            c4 contrast |
| -------------------------------- | ---------------------: | ---------------------: |
| total requests                   |                    320 |                     80 |
| failed requests                  |                      0 |                      0 |
| 503 count                        |                      0 |                      0 |
| canonical b8 hash                | all 0x0619d4d1900c2365 | all 0x0619d4d1900c2365 |
| engine_summary present           |                    yes |                    yes |
| admitted_count                   |                    321 |                     81 |
| promises_fulfilled               |                    321 |                     81 |
| update_iterations                |                    642 |                    181 |
| mean active_seq_count, all iters |              3.989 / 4 |              3.541 / 4 |
| mean active_seq_count, queue > 0 |              4.000 / 4 |          n/a, no queue |
| fraction iters with queue > 0    |                  0.984 |                  0.000 |
| mean waiting_queue_depth         |                 11.185 |                  0.000 |
| mean batch_n_tokens              |                  6.483 |                  5.757 |

Note on `admitted_count` / `promises_fulfilled`: 321 = 320 measured requests + 1
warmup, and 81 = 80 measured requests + 1 warmup.

## Interpretation

- Under offered load, the existing HPX backlog/admission machinery keeps all 4
  slots full.
- With c16 and `max_concurrent=16`, the queue was non-empty in 98.4% of
  iterations and `active_seq_count` reached 4.000/4 in steady state.
- The c4 contrast had no queue at all, showing the closed-loop no-work-ready
  condition.
- Therefore the mechanism needed for Design A, HPX-owned pending/backlog
  admission, is already present and validated for the tested offered-load path.
- The inherited c=4 occupancy loss was a closed-loop no-work-ready artifact, not
  an HPX admission defect.
- The throughput ceiling in this branch is governed by single-context decode and
  batch density, not HPX orchestration overhead.

This conclusion is scoped to the tested workload: TinyLlama F32, `n_seq_max=4`,
`max_concurrent=16`, greedy, `decode_budget=8`, short prompt. It is a
serving-control attribution result, not a production-readiness claim and not a
claim that hpx-server is faster than `llama-server`. Different model sizes,
prompt lengths, decode budgets, or slot counts are not characterized here.

## Design implications

- Design A, HPX-owned backlog/pending queue: effectively present and validated
  for the tested offered-load path.
- Design B, persistent slots / prefix-KV reuse: the only plausible future
  performance lever, but it is a separate project because it requires slot
  residency, prefix matching, eviction/context-shift policy, and careful KV
  safety.
- Design C, multi-engine/context pool: stopped for this branch because it
  fragments batching, increases KV memory, and does not address the validated
  bottleneck.

## Future work

Future performance work should not continue as incremental cleanup of this
branch. The offered-load result shows the backlog/admission machinery already
fills the available slots, so the throughput ceiling here is single-context
decode and batch density, not orchestration. Squeezing this branch further would
not move that ceiling.

The plausible future lever is Design B: persistent slots / prefix-KV reuse. The
idea is to keep useful slot/KV state alive across related requests so repeated
work is not redone:

- keep useful slot/KV state alive across related requests
- reuse a shared system prompt or conversation prefix when safe
- avoid repeating full prefill for repeated-prefix chat workloads
- choose slots based on session identity or longest reusable prefix
- define eviction/LRU, context-shift, and KV-safety policies

Design B is a separate project, not a continuation of this closeout. It changes
the slot-residency and prompt-cache policy surface and carries its own KV-safety
obligations, so it should not be mixed into this branch. It would be worth
exploring for workloads where prefill cost is large enough to matter — repeated
long system prompts, multi-turn chat, shared prompt prefixes, and TTFT-sensitive
serving. It is not relevant for tiny prompts, independent one-shot requests, or
short b8 decode smoke workloads with no shared prefix, where there is no reusable
state to exploit.

Design C, a multi-engine/context pool, remains stopped for this branch: it
fragments batching and increases KV memory pressure, and the offered-load result
does not justify it.

The stop/go framing is that future work should start from a new question —
"Can HPX implement a safe persistent-slot / prefix-reuse policy around llama.cpp
that improves TTFT or throughput for repeated-prefix chat workloads?" — rather
than from "How do we keep tuning this branch to beat `llama-server`?" Design B is
a future lever to investigate, not a promised win.

## Performance path after this closeout

This closeout settles the backlog/admission question, and the performance path
for any follow-on work follows directly from it:

- The offered-load result closes the backlog/admission question: under queued
  offered load, HPX fills all active slots (mean `active_seq_count` 4.000/4 while
  the queue is non-empty). The serving-control / admission machinery is not the
  bottleneck.
- The remaining performance path is therefore **not** incremental cleanup of the
  current admission/backlog design, the HTTP handler, or engine orchestration.
  Those are already small or saturated; squeezing them does not move the
  single-context decode / batch-density ceiling.
- Later Design B measurements showed that **prefix-KV reuse is the meaningful
  next lever**:
  - exact-session continuation reuse can reduce TTFT for measured multi-turn
    chat continuation;
  - same-session longest-common-prefix reuse can recover shared-prefix benefit
    for same-session divergent-suffix workloads;
  - these are prefix-cache / slot-residency features, **not** fixes to the
    backlog/admission machinery.
- The correct future project is therefore a **prefix-cache / persistent-slot
  project**, treated separately from this occupancy closeout — not "keep tuning
  this branch."
- Cross-session LCP should remain a **separate policy decision**, because
  cross-session cache hits can create a timing side channel.
- Partial-LCP reuse should **not** claim byte-identity against fresh full-prefill
  under all batch shapes: deterministic cross-shape floating-point sensitivity
  was observed (a greedy near-tie flip into a still-coherent continuation), with
  correct KV positions and trim and no KV contamination.

Representative numbers belong to the prefix-reuse line of work, not to this
closeout: B1 exact-session reuse reduced TTFT p50 from 567.6 ms to 78.3 ms in the
measured W-chat continuation cell, and B+1 same-session LCP recovered large
shared-prefix gains. See the dedicated docs for the full results and caveats:

- [hpx_exact_session_prefix_reuse_poc.md](hpx_exact_session_prefix_reuse_poc.md)
  (B1 exact-session reuse)
- [hpx_same_session_lcp_reuse_poc.md](hpx_same_session_lcp_reuse_poc.md)
  (B+1 same-session LCP reuse and the cross-shape FP caveat)

## Final decision

Stop the current branch as a performance-competition project.

This is not a failure. The branch demonstrates:

- HPX can own the serving-control actor boundary correctly.
- HPX control overhead is small.
- futures/promises, cancellation, streaming, diagnostics, prefill-budget policy,
  and backlog admission are validated.
- The remaining path to a throughput or TTFT advantage over `llama-server` is not
  incremental cleanup of this branch; it is a separate prefix-KV / persistent-slot
  project.

## Artifacts

```text
local/runs/occupancy-offered-load-2026-05-28/
  drive.py
  c16/...
  c4/...
```

No C++ changes were made by the final POC.
