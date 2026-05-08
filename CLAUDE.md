# Claude Guidance for HPX / llama.cpp Continuous-Batching Work

## Intent

This branch studies HPX-owned orchestration around llama.cpp continuous-batching execution.

The old FIFO `llama-serving-bench` context-pool path is closed. It remains evidence and reference only. Do not continue tuning that design by default.

## Active tool

Current implementation target:

tools/hpx-continuous-batch-gate/


Pure llama.cpp reference gate:

tools/multiseq-batch-gate/


## Read first

Read these before working on the current slice:

local/ahandoff.md
docs/hpx/hpx_continuous_batching_prototype_design.md
tools/hpx-continuous-batch-gate/README.md
tools/multiseq-batch-gate/results.md


## Background only

Read these only if needed:

docs/hpx/multiseq_llama_batch_gate.md
docs/hpx/continuous_batching_upstream_notes.md
docs/hpx/continuous_batching_simulator_design.md
docs/hpx/continuous_batching_phase3_target.md
hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md
docs/hpx/serving_fifo_pool_closeout.md



## Ownership boundary

HPX owns orchestration:

request metadata
slot/request lifecycle state
one engine task
later: futures/promises, traces, cancellation, priority hooks


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

Every sequence must reach its requested decode budget unless the current slice explicitly tests cancellation.

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
  the sequence reaches its requested budget
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

Current account paths:

repo:        /Users/unick/Desktop/hpx/llama-hpx
models:      /Users/unick/Desktop/hpx/models
HPX install: /Users/unick/Desktop/hpx/hpx-install


Build directories:

HPX-OFF reference build:
  /Users/unick/Desktop/HPX/builds/llama-base

HPX-ON build:
  /Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on


Current HPX binary:

/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate


Pure llama.cpp reference binary:

/Users/unick/Desktop/HPX/builds/llama-base/bin/llama-multiseq-batch-gate


## Saving outputs

Do not write outputs to `/tmp`.

For current correctness-gate captures, use:

```text
local/
```

For simulator-generated outputs, use the simulator's own gitignored result directory:

```text
hpx-bench/sim/continuous_batching/results/<run-id>/
```

For committed/shareable reports, use explicit docs/results files only when requested.
