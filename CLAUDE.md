# Claude Guidance for HPX Serving Layer for llama.cpp

## Intent

This branch studies the HPX Serving Layer for llama.cpp: an HPX-owned serving
control plane layered over llama.cpp execution.

The old FIFO `llama-serving-bench` context-pool path is closed. It remains
evidence and reference only. Do not continue tuning that design by default.

Current active surfaces include:

- `tools/hpx-continuous-batch-gate/` — reusable HPX engine and gate validation
- `tools/hpx-server/` — HTTP/SSE adapter over the engine
- `hpx-bench/experiments/12_hpx_vs_llama_server_pair/` — matched raw-measurement harness

## Active tool

Current implementation target:

tools/hpx-continuous-batch-gate/


Pure llama.cpp reference gate:

tools/multiseq-batch-gate/


## Read first

Read these before working on the current slice:

local/ahandoff.md
docs/hpx/hpx_serving_layer_m0_m8_milestone_summary.md
docs/hpx/continuous_batching_foundation.md
docs/hpx/continuous_batching_lifecycle_design.md
docs/hpx/continuous_batching_streaming_design.md
docs/hpx/serving_bench_fifo_closeout.md
docs/hpx/provenance.md


## Background only

Read these only if needed:

docs/hpx/continuous_batching_upstream_notes.md
docs/hpx/continuous_batching_simulator_design.md
hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md



## Ownership boundary

HPX owns orchestration:

request metadata
slot/request lifecycle state
one engine task
futures/promises, traces, cancellation, streaming, sampling, backpressure, and server adapter control path


llama.cpp owns model execution:

llama_model
llama_context
llama_batch
llama_decode
llama_memory_seq_*
logits access
tokenizer

Hard rule:

Only the engine task may touch llama_context, llama_batch, llama_decode, or llama_memory_seq_*.
No parallel llama_decode.

## General correctness invariants

Correctness is token/lifecycle correctness, not speed.

Every submitted request/sequence must either complete successfully or fail with an explicit reason.

Decode budget is an upper bound. A request may stop earlier on EOG/EOS and still
complete successfully. Exact decode-budget equality is only a gate when the slice
explicitly defines it, such as the canonical p0_b8 anchor.

Generated-token hashes are correctness fingerprints, not performance metrics.

Hash comparisons must use the same model, prompt, decode policy, batch shape, and scheduling policy.

Do not use cross-shape long-budget hash equality as a correctness invariant.
Budget-64 and budget-256 hashes may differ between N=3 and N=99 because batch shape can change floating-point behavior.

Safe invariants:
  within-shape repeat determinism
  within-run same-budget-class hash equality
  expected token counts
  expected done_iter / completion step
  expected KV position behavior
  every llama_decode returns 0
  no residual KV for completed sequences
  no cross-talk when clearing one seq_id

Budget-8 canonical hash for the current TinyLlama / "Hello, my name is" / greedy shape:
  0x0619d4d1900c2365

Budget-16 canonical hash for the same shape:
  0x833045f1e2ebf49f

Do not make performance claims from correctness gates.

## HPX continuous-batching implementation invariants

HPX owns orchestration only.

llama.cpp owns model execution.

There must be exactly one owner of a llama_context at a time.

Only the engine task may touch:
  llama_context
  llama_batch
  llama_decode
  llama_memory_seq_*
  llama_get_logits_ith

Never run parallel llama_decode calls on the same llama_context.

The engine task builds the shared llama_batch, calls llama_decode, reads logits, updates per-seq state, and performs KV cleanup.

Other HPX tasks may own request metadata, futures/promises, result collection, traces, or future cancellation/priority plumbing, but they must not touch llama.cpp context/batch/KV objects.

Per-request futures/promises, when used, are fulfilled only after:
  the sequence reaches its requested budget or stops on EOG/EOS
  generated-token state is finalized
  per-seq KV clear has succeeded
  the result snapshot is independent of llama_context internals

HPX scheduling must not change correctness anchors accidentally.
If batch composition, seq ordering, prompt/decode mixing, or scheduling policy changes, regenerate same-shape reference results before comparing hashes.

HPX runtime startup/shutdown is process-level.
Engine objects do not start or stop HPX.

Fail closed on HPX/runtime/model/decode errors.
Do not silently fall back to a non-HPX path unless the slice explicitly defines that as the experiment.

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


## Paths

Prefer discovering paths from the current checkout and build configuration.

Common local layout may include:

repo:
  /Users/unick/Desktop/hpx/llama-hpx

models:
  /Users/unick/Desktop/hpx/models

HPX install:
  /Users/unick/Desktop/hpx/hpx-install

Common build directories:
  /Users/unick/Desktop/hpx/builds/llama-base
  /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on

Do not assume these paths are universal. Confirm with `pwd`, `git status`,
and the current build directory before running commands.


## Saving outputs

Do not write outputs to `/tmp`.

For current correctness-gate captures, use:

```text
local/
```
For simulator-generated outputs, use the simulator's own gitignored result directory:

```
hpx-bench/sim/continuous_batching/results/<run-id>/
```

For committed/shareable reports, use explicit docs/results files only when requested.