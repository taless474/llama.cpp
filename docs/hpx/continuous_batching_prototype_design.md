# HPX continuous-batching prototype design

This is a **design note**, not an implementation. No HPX code is
added. `tools/server/`, `tools/serving-bench/`, and
`tools/multiseq-batch-gate/` are not modified by this note.

## Framing

**Old, closed-out path.** HPX as a FIFO context-pool replacement
around opaque `llama_decode` did not outperform the std backend on
the tested CPU-only TinyLlama serving workloads (see
`docs/hpx/serving_fifo_pool_closeout.md`, experiments 10/11). That
direction is closed.

**New path.** HPX owns request and slot orchestration around a
production-style continuous-batching loop. `llama.cpp` keeps owning
model execution: `llama_model`, `llama_context`, `llama_batch`, the
KV primitives, logits, sampling, and `llama_decode`. The boundary is
`llama_decode` itself — HPX schedules around it, never inside it.

The simulator (`hpx-bench/sim/continuous_batching/`) and the
multi-seq `llama_batch` gate (`tools/multiseq-batch-gate/`) made
this design tractable:

- The simulator showed that symmetric all-short workloads do not
  separate `static_batching` from `continuous_batching`, while mixed
  decode lengths do (`hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md`).
- The gate proved that real `llama.cpp` can host 99 concurrent
  `seq_id`s in one shared `llama_batch` with deterministic per-class
  hashes and clean per-seq KV lifecycle
  (`tools/multiseq-batch-gate/results.md`).

The HPX prototype targets that proven primitive, not a fresh design.

## Goal

Build the smallest HPX-owned orchestration layer around the proven
Step 5 multi-seq `llama_batch` primitive.

The prototype must preserve the same core execution shape:

- one `llama_model`
- one `llama_context`
- many `seq_id`s
- one shared `llama_batch`
- mixed decode budgets `{8, 64, 256}`
- same prompt for every seq (initially)
- greedy argmax over `llama_get_logits_ith`
- per-seq completion at `n_decoded == budget`
- per-seq `llama_memory_seq_rm(..., -1, -1)` on completion

The first prototype is a **correctness-first** prototype, not a
performance claim. It exists to prove that HPX-owned request
lifecycle and completion plumbing can wrap the Step 5 primitive
without disturbing the gate's invariants.

## Non-goals

Explicitly out of scope for this prototype:

- HTTP server (no `cpp-httplib`, no routes, no JSON parsing).
- Streaming partial responses / SSE.
- Cancellation tokens or request abort paths.
- Priority scheduling or per-request priority classes.
- Prompt cache (host-RAM `server_prompt_cache`, prefix reuse).
- LoRA adapters, alora.
- Speculative decoding (no draft model).
- Multimodal inputs.
- `n_cmpl > 1` parent/child KV fan-out.
- Distributed serving / multi-host.
- Modifying `tools/server/` (upstream).
- Modifying `tools/serving-bench/` (closed-out FIFO benchmark).
- Modifying `tools/multiseq-batch-gate/` (closeout reference; must
  remain a clean pure-`llama.cpp` primitive).
- Benchmarking against upstream `llama-server`.
- Claiming any HPX-vs-std speedup.

Each item above is a separate axis. None are required to prove the
HPX orchestration primitive on the Phase 3 target shape.

## What HPX should own

HPX is the **orchestration layer**:

- Request objects (id, seq_id, prompt tokens, decode_budget,
  generated tokens, state, completion promise/future).
- The admission queue (initially trivial: all 99 requests admitted
  at `t = 0`; later: arrival timing, deferred queue).
- Slot lifecycle state (each `seq_id`'s claim — initially 1:1 with
  request, since there are 99 active seqs and `n_seq_max == 99`).
- Per-request completion futures/promises (`hpx::lcos::local::promise`
  or equivalent; future fulfilled when seq reaches budget and KV is
  cleared).
- Engine-loop task ownership — exactly one HPX task that runs the
  continuous-batching loop. Not parallelized.
- Result collection (the main thread waits on all futures, then
  validates per-class hashes/`done_iter`/`pos_max_at_clear`).
- Trace/event emission (admit, prefill, decode-row, complete, clear,
  future-fulfilled).
- Future extension points (cancellation tokens, priority queues,
  streaming channels). These are wiring stubs only in v1.

**Important rule.** HPX owns orchestration, not model execution.
HPX must not parallelize `llama_decode`. HPX must not call
`llama_decode` from more than one task at a time. HPX runtime
startup/shutdown is process-wide and one-shot, owned by the program
entry layer (the `main()` of the new tool), per the CLAUDE.md
invariant.

## What llama.cpp should own

`llama.cpp` is the **kernel layer**:

- `llama_model` (model weights, vocabulary).
- `llama_context` (one shared instance).
- `llama_batch` (one shared instance, allocated via
  `llama_batch_init`, populated via `common_batch_add`,
  reset each iteration via `common_batch_clear`).
- `llama_decode` (synchronous compute call).
- `llama_memory_seq_rm`, `llama_memory_seq_pos_min`,
  `llama_memory_seq_pos_max`, and any other KV primitives.
- Logits access via `llama_get_logits_ith`.
- Tokenizer (`common_tokenize`, `llama_vocab_*`).
- Greedy argmax — we keep the tiny local `argmax(logits, n_vocab)`
  helper from the gate to avoid `common_sampler_*` complexity. This
  is implementation detail; the model itself owns the underlying
  logits.

**Important rule.** Only the engine task may touch
`llama_context`, `llama_batch`, `llama_decode`, and
`llama_memory_seq_*`. HPX worker threads doing other work
(future fulfillment, trace emit, request bookkeeping) must not
reach into these objects directly.

## First prototype architecture

Minimal architecture:

- HPX runtime starts once, in the new tool's `main()`. Process-wide
  and one-shot. Errors abort fail-closed if HPX is unavailable.
- A vector of 99 `request` objects is created up-front from the
  fixed Phase 3 workload. Each request has:
  - `request_id` (== `seq_id` for v1, since there is no
    request-vs-slot indirection yet)
  - `seq_id`
  - `prompt_tokens` (shared via `std::shared_ptr` or by reference,
    since every request uses the same prompt)
  - `decode_budget`
  - `generated_tokens` (filled by the engine)
  - `state` (waiting → prefilled → decoding → done → cleared)
  - `completion_promise` / `completion_future` pair
  - `hash_state`, `n_decoded`, `pos_next`, `i_batch`, `last_token`
    (the gate's per-seq state, lifted into the request)
- A single HPX task — the **engine task** — runs the
  continuous-batching loop. The loop body is structurally identical
  to the gate's `run_multiseq()`:
  1. clear shared batch
  2. add prefill rows for all not-yet-prefilled requests, with
     `logits=true` only on the last prompt row per seq, recording
     `i_batch` per request
  3. for already-decoding requests, add one decode row per still-active
     request, `logits=true`, recording `i_batch`
  4. one `llama_decode` call
  5. for each active request, read its row's logits via
     `llama_get_logits_ith(ctx, i_batch)`, take argmax, fold the
     hash, push the token, bump `n_decoded`, advance `pos_next`
  6. for each request that just hit `n_decoded == budget`,
     mark done, clear its KV via `llama_memory_seq_rm`, snapshot
     siblings' `pos_min/pos_max` before the clear and verify
     no cross-talk after, fulfill the request's promise
  7. exit the loop when all promises are fulfilled
- The main thread waits on every request's completion future
  (`hpx::wait_all` on the future vector), then validates per-class
  invariants and emits the final `HPX_CB_PROTO: PASS` / `FAIL`
  line.

In v1 prefill is one-shot: all 99 prefill rows go in the first
shared batch (594 rows for `P=6, n_seqs=99` — fits in `n_batch=1024`
exactly as in Step 5). Mixing prefill rows with decode rows in the
same iteration is **out of scope for v1** — keeping prefill and
decode in temporally separate iterations is closer to the gate's
shape and removes one source of per-iteration scheduling complexity
until the request-lifecycle plumbing is proven.

**Offline workload, not live arrivals.** v1 admits all 99 requests
at `t = 0`. No arrival schedule, no deferred queue, no admission
backpressure. This keeps v1 as close to Step 5 as possible. Live
arrivals are a v2 axis; they're listed under "future extension
points" but not in the v1 path.

## Relationship to the Step 5 gate

The HPX prototype is, structurally, **Step 5 with a different
ownership shape**: request lifecycle and completion are represented
by HPX futures/promises, but the per-iteration loop body is
identical. The first goal is to reproduce Step 5 exactly under that
new ownership.

Reference Step 5 invariants the prototype must reproduce:

- `n_seqs == 99`
- 33 seqs per budget class for budgets `{8, 64, 256}`
- budget 8 hash = `0x0619d4d1900c2365` (canonical, cross-shape stable)
- budget 64 hash = exactly one unique hash per run
- budget 256 hash = exactly one unique hash per run
- `done_iter` sets:
  - 8 → `{7}`
  - 64 → `{63}`
  - 256 → `{255}`
- `pos_max_at_clear` sets:
  - 8 → `{12}`
  - 64 → `{68}`
  - 256 → `{260}`
- residual KV cleared for all seqs (`pos_min == pos_max == -1`)
- no cross-talk when clearing finished seqs
- `--repeat 2` deterministic within the same batch shape

**Hash-shape caveat carried forward.** Budget-64 and budget-256
hashes from this prototype must be compared to a **same-shape HPX-off
reference run** (i.e. the existing Step 5 captures with N=99, mix
`{8, 64, 256}`, n_batch=1024, ctx_size=32768). They must **not** be
compared to Step 4 (N=3) hashes for those budgets. Cross-shape FP
non-associativity makes those references invalid for long budgets,
as documented in `tools/multiseq-batch-gate/results.md`. The
budget-8 hash is the only currently-validated cross-shape anchor on
this model + prompt.

If the prototype changes the per-iteration batch composition (e.g.
mixing prefill and decode in the same iteration, or reordering
seqs), expect long-budget hashes to drift. This is not a bug; it is
the correctness boundary the prototype must respect when comparing
against references.

## Correctness gates

The prototype must verify all of the following before exiting `0`:

1. Every request future completes (no deadlock; main `wait_all`
   returns).
2. Every seq reaches `n_decoded == decode_budget`.
3. Counts per budget class are exact: 33 / 33 / 33 for
   `{8, 64, 256}`.
4. Budget-8 hash equals `0x0619d4d1900c2365`.
5. Budget-64 has exactly one unique hash within the run.
6. Budget-256 has exactly one unique hash within the run.
7. Same-shape repeat (`--repeat 2`) produces byte-identical per-class
   hashes, counts, `done_iter` sets, and `pos_max_at_clear` sets.
8. `done_iter` sets match `{7}`, `{63}`, `{255}` per class.
9. `pos_max_at_clear` sets match `{12}`, `{68}`, `{260}` per class.
10. Per-seq KV clear is effective: after each
    `llama_memory_seq_rm(..., -1, -1)` the cleared seq reads
    `pos_min == pos_max == -1`.
11. Clearing one seq does not disturb still-active siblings'
    `(pos_min, pos_max)` (the gate's `finalize_and_clear_seq`
    cross-talk check).
12. Residual KV at end is empty for all 99 seqs.
13. Every `llama_decode` call returns 0.
14. Final stdout line is exactly:
    - `HPX_CB_PROTO: PASS` on success, or
    - `HPX_CB_PROTO: FAIL: <reason>` with a specific reason string
      naming the gate that failed.

These are deliberately the same gates as Step 5 plus two
HPX-specific ones (futures complete; final emit line). Anything
else (timing, throughput, batch-utilization curves) is descriptive
metric, not a gate.

## Metrics to collect

Correctness-first. Print one block at end of run, after the gate
verdict but before exit:

- total wall time (steady_clock, milliseconds)
- number of update iterations (= number of `llama_decode` calls)
- rows-per-batch p50, p95, max
- active-seqs-per-iteration p50, p95, max
- completed seqs by budget (sanity, should be 33 / 33 / 33)
- time-to-completion by budget class (mean and p95 of
  `finish_time - start_time` per request, reported per class)
- optional HPX overhead counters if cheap to gather:
  - futures created
  - promises fulfilled
  - engine loop task count (should be 1)

**State clearly in the tool's output and in any results doc:**
performance numbers from the prototype are **not** comparable to
upstream `llama-server`, simulator output, or any prior FIFO-pool
experiment until the correctness gates above are stable across
multiple runs and the batch shape is held fixed.

## Proposed implementation location

### Recommendation: Option B — new tool

Create a new tool: `tools/hpx-continuous-batch-gate/`.

- `CMakeLists.txt` — links `llama-common`, `llama`, and the local
  HPX install (via `find_package(HPX)`).
- `hpx-continuous-batch-gate.cpp` — the prototype binary.
- `README.md` — slice status, build/run instructions, and the same
  out-of-scope list.
- `results.md` — added at closeout, mirroring the gate's results doc.

### Why Option B over Option A

| Concern | Option A (extend gate) | Option B (new tool) |
|---|---|---|
| Keep Step 5 gate clean as the closeout reference | **bad**: the gate identity already shipped (`tools/multiseq-batch-gate/results.md`) — adding HPX would muddy it | **good**: gate stays a pure `llama.cpp` primitive |
| Build complexity | gate's CMake gains conditional HPX linking and include paths | one new CMake target, opt-in, no impact on existing builds |
| Confusion between primitive and orchestration | the same binary would be both pure-llama and HPX-orchestrated, depending on a flag | clear physical separation |
| Code duplication | none | small (argmax, FNV fold, structural-checks helpers, ~50 LOC) |
| Independent build / debug cycles | tied | independent |
| Future cancellation/priority axes | grow inside the gate | grow inside the prototype, not the gate |

The duplication cost of Option B is small (the helpers are short,
header-style, and stable) and is the price paid to keep the gate's
closeout evidence intact. Option B wins.

If the duplication proves annoying after Slice 3, factor a tiny
shared header (e.g. `tools/multiseq-shared/hash_and_argmax.h`) at
that point, not before. v1 just inlines the helpers.

### What stays where

- `tools/multiseq-batch-gate/` — frozen as the closeout reference.
  Do not modify it for HPX work.
- `tools/hpx-continuous-batch-gate/` — new home for HPX-owned
  orchestration. Reaches into `libllama` and `libcommon` only;
  does not depend on the gate.

## Build gating

Add a CMake option in `tools/CMakeLists.txt`:

```cmake
option(LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE
       "Build the HPX continuous-batching prototype" OFF)

# ...

if (LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE)
    add_subdirectory(hpx-continuous-batch-gate)
endif()
```

- Default: **OFF**. Existing builds (including the multiseq gate
  build dir at `/Users/unick/Desktop/HPX/builds/llama-base`) are
  unaffected.
- Inside `tools/hpx-continuous-batch-gate/CMakeLists.txt`,
  `find_package(HPX REQUIRED)` (or equivalent for the local HPX
  install at the path the prior HPX-on serving-bench used). The
  target links `HPX::hpx` plus `llama-common`, `llama`,
  `${CMAKE_THREAD_LIBS_INIT}`.
- The binary is `llama-hpx-continuous-batch-gate`. Standalone target
  name (no shared TARGET with other tools).

The existing gate's build option (`LLAMA_BUILD_MULTISEQ_GATE`) is
not changed and is not required by the new option.

## Implementation slices

Each slice is a stop-and-check gate. The slice's binary must emit
its own `HPX_CB_STEPN: PASS` / `HPX_CB_STEPN: FAIL: <reason>` line
on the final stdout row before the work moves to the next slice.

### Slice 1 — HPX skeleton only

Goal: prove the HPX runtime starts inside this tool and does not
break `libllama`.

- Start the HPX runtime in `main()`. Use `hpx::init` /
  `hpx::start` / `hpx::finalize` as appropriate for the local HPX
  install pattern.
- Load model, create `llama_context`, print
  `actual n_ctx / n_seq_max / n_batch / prompt_tokens` (same
  Step-1-style structural prints).
- Build the 99 fixed request objects. No `llama_decode`, no
  `llama_batch`; request objects are metadata-only.
- Stop the HPX runtime cleanly. Free model + context.
- Emit `HPX_CB_STEP1: PASS` on the final stdout line.

Acceptance: same structural prints as the gate's Step 1, plus a
clean HPX runtime startup/shutdown in the same process as `libllama`.

### Slice 2 — engine object owns the loop, no futures yet

Goal: lift the gate's `run_multiseq()` body into an HPX-owned
engine task without changing observable behavior.

- Add an `engine` class that owns `llama_context *`, `llama_batch`,
  the request vector, and the iteration loop.
- The engine's `run()` method is the body; it is invoked as a
  single HPX task (`hpx::async` or equivalent), and `main()` waits
  on its completion future.
- No per-request futures yet. The engine still exposes results
  via the request vector (filled in place).
- Reproduce **all 14** Step 5 correctness gates before emit.
- Emit `HPX_CB_STEP2: PASS` on the final stdout line.

Acceptance: byte-identical per-class hashes, `done_iter`, and
`pos_max_at_clear` to a same-shape Step 5 capture.

### Slice 3 — per-request HPX futures/promises

Goal: thread completion through HPX async primitives instead of
direct vector access.

- Each request gets an `hpx::lcos::local::promise<request_result>`.
- The engine fulfills the promise inside the loop, **after** the
  per-seq KV clear and cross-talk check succeed for that request.
- `main()` collects futures via `hpx::wait_all(...)` then iterates
  results to validate gates.
- Reproduce all 14 gates plus gate #1 (every future completes).
- Emit `HPX_CB_STEP3: PASS` on the final stdout line.

Acceptance: byte-identical per-class hashes vs. a same-shape Slice 2
run; `wait_all` returns; no future left unfulfilled.

### Slice 4 — trace and events

Goal: instrument the lifecycle so future debugging and the eventual
metrics block have something to report.

- Emit one trace line per event (env-gated, default off) for:
  - request admitted
  - seq prefilled (with first sampled token id)
  - seq decode row added (with iter and pos)
  - seq complete (with `done_iter`, `pos_max_at_clear`, hash)
  - seq KV cleared (with cross-talk check result)
  - future fulfilled
- Add the metrics block (rows-per-batch p50/p95/max,
  active-seqs-per-iter p50/p95/max, total wall time, decode-call
  count, time-to-completion by class) to the **end of every run**,
  after gates, before the emit line.
- Emit `HPX_CB_STEP4: PASS` on the final stdout line.

Acceptance: same gates as Slice 3 plus a non-empty, deterministic
metrics block that survives `--repeat 2`.

### Slice 5 — same-shape comparison (only after correctness)

Goal: with everything green, capture a same-shape HPX-off reference
side-by-side and confirm hash equality where the shape is preserved.

- Run the existing Step 5 gate once with the same args.
- Run the HPX prototype once with the same shape.
- Compare per-class hashes. The budget-8 hash must equal across
  both runs (canonical anchor). The budget-64 and budget-256
  hashes must equal across both runs **only because the batch shape
  is held fixed** — explicitly, both runs use the same prompt, same
  budget mix, same `n_seqs`, same `n_batch`, same per-iteration
  composition.
- Emit `HPX_CB_PROTO: PASS` on the final stdout line of the HPX
  prototype run.

Acceptance: the prototype reproduces Step 5 byte-for-byte under
HPX ownership when batch shape is held fixed.

### Out of scope for the prototype

- Cancellation / abort propagation.
- Priority queues.
- Live arrival schedules.
- Mixing prefill + decode rows in the same iteration.
- Streaming partial responses.
- Multiple `llama_context` instances / multi-host.

These are the v2+ axes. None are in any of slices 1–5.

## Risks / open questions

1. **HPX runtime + `libllama` threading interaction.** `libllama`
   uses its own pthreads for ggml compute. HPX has its own thread
   pool. Risk: HPX hijacking or saturating cores that `libllama`
   expects. Mitigation: in v1 hold `n_threads` low (the gate uses
   `n_threads=2`), let `libllama` keep its own backend threads,
   and pin the engine task to its own HPX worker. Document the
   exact HPX thread-count config used.
2. **Single-owner discipline for `llama_context`.** All
   `llama_decode` / `llama_batch` / `llama_memory_seq_*` calls must
   originate from the engine task. Risk: a future-continuation
   running on a different HPX worker accidentally touches the
   context (e.g. during promise fulfillment). Mitigation: separate
   the "fulfill promise" continuation into a pure data step that
   does not reach into `llama_context`; the engine task owns all
   model-touching state and only hands snapshot data to the
   continuation.
3. **No accidental parallel `llama_decode`.** A single HPX engine
   task cannot run two `llama_decode` calls in parallel by
   construction. Risk emerges only if a future slice fans the engine
   into multiple tasks. v1 must use exactly one engine task. Add a
   debug-build assertion that fires if `llama_decode` is entered
   re-entrantly (e.g. a per-context atomic flag).
4. **Batch-shape preservation for correctness anchors.** Long-budget
   hashes are shape-dependent (gate's results.md table). Risk: a
   well-meaning future change (e.g. allowing prefill rows to mix
   with decode rows in the same iteration to fill `n_batch` more
   tightly) silently changes the reference hashes. Mitigation:
   correctness comparisons must always be against a same-shape
   reference produced by the **current code path**, not against
   archived hashes from a prior shape. Treat archived hashes as
   stale unless re-validated.
5. **Hash divergence across batch shapes.** Documented in
   `tools/multiseq-batch-gate/results.md`. The prototype must not
   claim cross-shape hash equality for budgets > 8.
6. **CMake / linking complexity.** Local HPX install path varies by
   account (CLAUDE.md mentions `/Users/Ashk/...`, this account has
   `/Users/unick/...`). `find_package(HPX REQUIRED)` with
   `HPX_DIR` configurable via `-DHPX_DIR=...` keeps this
   account-portable. Document the exact configure command in the
   tool's README.
7. **Sharing code with the multiseq gate vs. duplicating.** v1
   duplicates a few small helpers (FNV fold, argmax,
   `seq_state` struct, prefix-equality check). If duplication
   becomes painful at Slice 3+, factor a header-only
   `tools/multiseq-shared/` for the helpers. Don't pre-factor.
8. **HPX failure modes.** If HPX startup fails (HPX install
   missing, runtime config invalid, etc.), the tool must fail
   closed with a clear `HPX_CB_STEP1: FAIL: <reason>` rather than
   degrade silently to a non-HPX path. There is no fallback path.
9. **Repeat-2 nondeterminism through HPX worker scheduling.** HPX
   worker scheduling is internally nondeterministic, but the engine
   task is single-threaded and does not depend on which HPX worker
   runs it. So per-class hashes must be byte-identical across
   `--repeat 2`. If they aren't, that's a bug in how the engine
   uses HPX, not in HPX itself.
10. **Local HPX install version.** The prior FIFO-pool branch built
    against a specific HPX install (CLAUDE.md `HPX install:
    /Users/.../hpx-install`). Any v1 change to that install version
    is out of scope; the prototype reuses whatever is already
    installed. Pin the version in `results.md` once Slice 5 closes.

## Final recommendation

- **Recommended location**: Option B —
  `tools/hpx-continuous-batch-gate/`. Keeps `tools/multiseq-batch-gate/`
  intact as the closeout reference and isolates HPX linkage to one
  new opt-in target.
- **Recommended first slice**: Slice 1 (HPX skeleton only). Three
  files (`CMakeLists.txt`, `hpx-continuous-batch-gate.cpp`,
  `README.md`). Starts the HPX runtime, loads model + context,
  prints structural fields, builds 99 metadata-only request
  objects, exits with `HPX_CB_STEP1: PASS`. No `llama_decode`,
  no `llama_batch`; request objects are metadata-only.
- **Build gating**: `LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=OFF`,
  added to `tools/CMakeLists.txt` next to the existing
  `LLAMA_BUILD_MULTISEQ_GATE` and `LLAMA_BUILD_SERVING_BENCH`
  options.
- **Exact next implementation prompt** (suggested wording for the
  follow-up turn after this design is reviewed):

  > Proceed with Slice 1 only.
  >
  > Create `tools/hpx-continuous-batch-gate/` with:
  > - `CMakeLists.txt` (target `llama-hpx-continuous-batch-gate`,
  >   linking `llama-common`, `llama`, and the local HPX install via
  >   `find_package(HPX REQUIRED)`).
  > - `hpx-continuous-batch-gate.cpp` (starts HPX, loads model + ctx,
  >   prints `actual n_ctx / n_seq_max / n_batch / prompt_tokens`,
  >   builds 99 fixed request objects (no `llama_decode`, no
  >   `llama_batch`; request objects are metadata-only), stops HPX,
  >   frees model/context, emits `HPX_CB_STEP1: PASS` or
  >   `HPX_CB_STEP1: FAIL: <reason>` on the final stdout line).
  > - `README.md` (slice status, build/run, out-of-scope list).
  >
  > Wire `LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=OFF` into
  > `tools/CMakeLists.txt` next to the existing options.
  >
  > Build only the new target.
  > Run one smoke against TinyLlama with the Phase 3 args
  > (`--n-seqs 99 --decode-budget-mix 8,64,256 --ctx-size 32768
  > --n-batch 1024 --n-threads 2`).
  > Capture to `local/hpx_cb_step1.stdout` and `local/hpx_cb_step1.stderr`.
  > Stop and report files / build / run / final line / actual
  > capacities / HPX runtime startup notes / any uncertainty.
  >
  > Do not implement Slice 2 until I approve.

## Status

This document is **design only**. The HPX continuous-batching
prototype implementation has not started. No HPX code has been
added to the repo. `tools/server/`, `tools/serving-bench/`, and
`tools/multiseq-batch-gate/` remain unmodified.

If any of the source-of-truth signals below change in a way that
invalidates the architecture above, this note must be updated
before Slice 1 begins:

- `docs/hpx/continuous_batching_upstream_notes.md`
- `docs/hpx/continuous_batching_simulator_design.md`
- `docs/hpx/continuous_batching_phase3_target.md`
- `docs/hpx/multiseq_llama_batch_gate.md`
- `tools/multiseq-batch-gate/results.md`
