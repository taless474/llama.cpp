# Claude Guidance for HPX Serving Layer for llama.cpp

## Intent

This branch studies an HPX-owned serving-control layer over llama.cpp execution.

The old FIFO `llama-serving-bench` context-pool path is closed. It remains evidence and reference only. Do not continue tuning that design by default.

The generic hpx-server backlog/admission performance-competition question is also closed. The offered-load closeout showed that HPX backlog/admission fills all active slots when queued work exists. Do not keep tuning backlog, admission, HTTP handler, futures, or orchestration for generic throughput unless a new measurement identifies a specific bottleneck.

The current performance path, is prefix-cache / persistent-slot reuse:

* exact-session continuation reuse,
* same-session longest-common-prefix reuse,
* future cross-session prefix reuse only if explicitly approved as a separate policy decision.

Prefix-cache work is a separate line of work from the occupancy/admission closeout.

## Active surfaces

Current active implementation surfaces:

```text
tools/hpx-continuous-batch-gate/
```

Reusable HPX engine, continuous-batching validation, session residency, exact-session prefix reuse, same-session LCP reuse, and engine-level smokes.

```text
tools/hpx-server/
```

HTTP/SSE adapter over the engine, including the optional `session_id` request surface.

Pure llama.cpp reference gate:

```text
tools/multiseq-batch-gate/
```

Historical matched raw-measurement harness:

```text
hpx-bench/experiments/12_hpx_vs_llama_server_pair/
```

Use it as reference only unless the current task explicitly asks for it.

## Read first

Read these before working on the current phase:

```text
local/ahandoff.md
docs/hpx/hpx_server_occupancy_offered_load_closeout.md
docs/hpx/hpx_exact_session_prefix_reuse_poc.md
docs/hpx/hpx_same_session_lcp_reuse_poc.md
```

These define the current state:

```text
hpx_server_occupancy_offered_load_closeout.md:
  backlog/admission question is closed;
  HPX fills active slots under offered load;
  performance path is not more admission/HTTP/orchestration tuning.

hpx_exact_session_prefix_reuse_poc.md:
  B1 exact-session persistent-slot reuse;
  works for measured multi-turn continuation;
  not a general shared-prefix cache.

hpx_same_session_lcp_reuse_poc.md:
  B+1 same-session LCP reuse;
  recovers same-session shared-prefix benefit;
  partial-LCP reuse is not guaranteed byte-identical to fresh full-prefill
  because of cross-shape floating-point sensitivity.
```

Read these only if the task specifically touches older serving-layer design or historical validation:

```text
docs/hpx/hpx_serving_layer_m0_m8_milestone_summary.md
docs/hpx/continuous_batching_foundation.md
docs/hpx/continuous_batching_lifecycle_design.md
docs/hpx/continuous_batching_streaming_design.md
docs/hpx/serving_bench_fifo_closeout.md
docs/hpx/provenance.md
```

Read these only if needed for simulator or upstream-background questions:

```text
docs/hpx/continuous_batching_upstream_notes.md
docs/hpx/continuous_batching_simulator_design.md
hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md
```


## Ownership boundary

HPX owns orchestration:

```text
request metadata
slot/request lifecycle state
one engine task
futures/promises
traces
cancellation
streaming
backpressure
server adapter control path
session/prefix-reuse policy
```

llama.cpp owns model execution:

```text
llama_model
llama_context
llama_batch
llama_decode
llama_memory_seq_*
logits access
tokenizer
sampler math
ggml graph construction
backend scheduling
kernels
```

Hard rule:

```text
Only the engine task may touch llama_context, llama_batch, llama_decode,
llama_memory_seq_*, or llama_get_logits_ith.
```

Never run parallel `llama_decode` calls on the same `llama_context`.

## General correctness invariants

Correctness is token/lifecycle correctness, not speed.

Every submitted request/sequence must either complete successfully or fail with an explicit reason.

Decode budget is an upper bound. A request may stop earlier on EOG/EOS and still complete successfully. Exact decode-budget equality is only a gate when the slice explicitly defines it, such as the canonical p0_b8 anchor.

Generated-token hashes are correctness fingerprints, not performance metrics.

Hash comparisons must use the same model, prompt, decode policy, batch shape, and scheduling policy.

Do not use cross-shape long-budget hash equality as a correctness invariant.

Budget-64 and budget-256 hashes may differ between N=3 and N=99 because batch shape can change floating-point behavior.

Safe invariants:

```text
within-shape repeat determinism
within-run same-budget-class hash equality
expected token counts
expected done_iter / completion step
expected KV position behavior
every llama_decode returns 0
no residual KV for completed sequences unless intentionally resident
resident KV cleared at teardown
no cross-talk when clearing one seq_id
no cross-session reuse unless explicitly enabled by the current slice
```

Budget-8 canonical hash for the current TinyLlama / `"Hello, my name is"` / greedy shape:

```text
0x0619d4d1900c2365
```

Budget-16 canonical hash for the same shape:

```text
0x833045f1e2ebf49f
```

Do not make performance claims from correctness gates.

## HPX continuous-batching implementation invariants

There must be exactly one owner of a `llama_context` at a time.

Only the engine task may touch:

```text
llama_context
llama_batch
llama_decode
llama_memory_seq_*
llama_get_logits_ith
```

The engine task builds the shared `llama_batch`, calls `llama_decode`, reads logits, updates per-seq state, and performs KV cleanup or resident-KV bookkeeping.

Other HPX tasks may own request metadata, futures/promises, result collection, traces, or future cancellation/priority plumbing, but they must not touch llama.cpp context/batch/KV objects.

Per-request futures/promises, when used, are fulfilled only after:

```text
the sequence reaches its requested budget or stops on EOG/EOS
generated-token state is finalized
KV state is either cleared or intentionally marked resident
the result snapshot is independent of llama_context internals
```

HPX scheduling must not change correctness anchors accidentally.

If batch composition, seq ordering, prompt/decode mixing, session reuse, LCP reuse, or scheduling policy changes, regenerate same-shape reference results before comparing hashes.

HPX runtime startup/shutdown is process-level. Engine objects do not start or stop HPX.

Fail closed on HPX/runtime/model/decode errors. Do not silently fall back to a non-HPX path unless the slice explicitly defines that as the experiment.

## Prefix-cache / persistent-slot invariants

Prefix reuse is a serving/slot-residency policy around llama.cpp. It must not replace `llama_decode`, tokenizer behavior, sampler math, ggml graph construction, or backend execution.

Session-aware requests use optional `session_id`.

If `session_id` is absent, behavior must remain the pre-prefix-cache path.

Resident KV may be kept only after clean successful completion.

Cancelled or errored requests must not become resident.

Active slots must never be evicted.

Resident inactive slots may be evicted only by the engine task, using `llama_memory_seq_rm`.

Same-session exact reuse and same-session LCP reuse are allowed in the current POC.

Cross-session LCP is not allowed unless explicitly approved as a separate policy decision.

`llama_memory_seq_cp` is out of scope unless explicitly approved.

Context shift is out of scope unless explicitly approved.

Off-context prompt cache is out of scope unless explicitly approved.

All KV operations remain engine-task-only.

## LCP correctness and determinism policy

Token-level prefix equality is required before reusing KV.

For partial-LCP reuse, the old divergent KV tail must be removed with:

```text
llama_memory_seq_rm(mem, seq_id, matched, -1)
```

Suffix prefill must resume at the matched prefix length with contiguous positions.

No stale tail, position gap, or cross-session contamination is acceptable.

Byte-identity remains appropriate for fixed-shape anchors and controlled smokes.

Do not require byte-identity between partial-LCP reuse and fresh full-prefill across different batch shapes.

Cross-shape floating-point sensitivity can flip greedy near-ties. This was observed in B+1 and must be reported, not mistaken for a KV contamination bug unless structural invariants fail.

For partial-LCP comparisons, report:

```text
mismatch rate
first differing token
matched prefix length
trimmed token count
KV positions
whether old tail was removed
whether suffix positions are contiguous
```

## Performance interpretation

Do not claim HPX broadly beats `llama-server`.

Do not claim production readiness.

Do not make performance claims from smokes.

Performance claims require benchmark evidence and must name:

```text
model
workload
thread settings
decode budget
ctx size
session/reuse mode
whether reuse was exact-session or LCP
whether cross-shape byte mismatches occurred
```

The intended performance target for prefix-cache work is TTFT and prefill reduction on repeated-prefix workloads.

Backlog/admission and prefix cache are different mechanisms:

```text
backlog/admission:
  keeps batches full when queued work exists

prefix cache / persistent slots:
  avoids recomputing repeated prompt prefixes
```

The occupancy/offered-load closeout closed the backlog/admission question for the tested path. The evidence-backed performance path is prefix-cache / persistent-slot reuse.

## What not to do next by default

Do not keep tuning backlog/admission/HTTP handler/orchestration for generic throughput unless a new measurement shows a specific bottleneck.

Do not implement cross-session LCP by default.

Do not introduce `llama_memory_seq_cp`, context shift, off-context prompt cache, or a broad slot-cache manager without a new plan and approval.

Do not weaken correctness gates silently.

If partial-LCP output differs from fresh full-prefill, investigate and document whether structural KV invariants still hold.

Do not treat coherent but different generated text as automatically correct. First inspect tokens, positions, prefix match, tail trim, and mismatch reproducibility.

## HPX runtime invariants

HPX runtime startup is process-wide and one-shot.

Program entry owns HPX startup/shutdown.

Individual engine objects do not start or stop HPX.

Fail closed if HPX is unavailable.

Runtime traces are env-gated unless they are errors.

## Working style

Define the contract first.

Implement in small slices.

Run only the relevant slice checks.

Stop after each slice.

Do not proceed to the next slice without approval.

Do not commit unless explicitly asked.

Do not make performance claims from correctness gates.

For implementation work, prefer this order:

```text
plan
small source slice
focused smoke
regression smokes
benchmark only after correctness passes
documentation only after result interpretation is stable
```

## Paths

Prefer discovering paths from the current checkout and build configuration.

Common local layout may include:

```text
repo:
  /Users/unick/Desktop/hpx/llama-hpx

models:
  /Users/unick/Desktop/hpx/models

HPX install:
  /Users/unick/Desktop/hpx/hpx-install

Common build directories:
  /Users/unick/Desktop/hpx/builds/llama-base
  /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on
```

Do not assume these paths are universal. Confirm with `pwd`, `git status`, and the current build directory before running commands.

When changing engine code, rebuild the relevant server target before benchmarking:

```text
llama-hpx-server
```

Do not rely on smoke-target rebuilds alone when the benchmark launches the server binary.

## Saving outputs

Do not write outputs to `/tmp`.

For current correctness-gate captures, use:

```text
local/runs/<slice-or-experiment>/<short-label>/
```

For simulator-generated outputs, use the simulator's own gitignored result directory:

```text
hpx-bench/sim/continuous_batching/results/<run-id>/
```

For committed/shareable reports, use explicit docs/results files only when requested.

For local benchmark artifacts, keep the driver and summaries under the relevant `local/runs/...` directory.
