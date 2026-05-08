# HPX continuous-batching: cooperative cancellation — design note

Status: design only. No code yet. No commit yet.

This note defines how cancellation should work in the existing HPX
continuous-batching prototype
(`tools/hpx-continuous-batch-gate/`). It does not propose
implementing it yet; it scopes the work into small slices and
specifies the gates each slice must hit.

This is a correctness-and-lifecycle design. There are no
performance claims and none are planned for this track.

---

## 0. Architecture we are building on

Slices 1–5 are landed and PASS. The current shape (do not change
in this design):

- one `llama_model`, one `llama_context`, one shared `llama_batch`
- one HPX engine task per repeat iteration, scheduled via
  `hpx::async([&eng]{eng.run();})`
- one `hpx::promise<request_result>` per seq; futures handed out
  before the engine task starts
- the engine fulfills each promise only after per-seq KV clear
  and a cross-talk-against-still-active-siblings check
- main calls `hpx::wait_all` on the per-request futures, then
  `engine_fut.get()`, then validates correctness exclusively from
  `request_result` snapshots
- only the engine task touches `llama_context` / `llama_batch` /
  `llama_decode` / `llama_memory_seq_*` / `llama_get_logits_ith`
- residual-KV-empty check is performed inside the engine task

Cancellation must respect every one of those invariants.

---

## 1. Why cancellation is the next HPX-native feature

The HPX prototype today demonstrates orchestration mechanics
(engine-as-task, per-request promises, snapshot boundary), but it
does not yet exercise a serving invariant that a pure
`run-to-completion` driver could not also satisfy. Every request
runs to its budget; KV is reclaimed only at completion;
fulfillment is monotonic.

A named HPX orchestration pool / resource-partitioner setup
(Slice 3b) is structurally interesting but does not, with one
engine task, change runtime behavior or test a new serving
invariant. It would expand failure surface (partitioner config,
pool registration, executor lookup) against gates that are about
futures and request completion.

Cancellation, by contrast, introduces a serving invariant we do
not have yet:

> A request may leave the system before completing its budget,
> with its KV cells reclaimed, the engine task continuing to make
> forward progress on the surviving requests, and the cancelled
> request's future fulfilled with a non-completed status, exactly
> once.

This invariant is the precondition for many later capabilities:

- HTTP client disconnect → cancel the still-running request;
- deadline-driven scheduling → cancel requests that exceed budget
  even if they could keep going;
- priority preemption → cancel-and-requeue (later, much later);
- admission control under context-cell pressure → cancel oldest
  or lowest-priority to free KV.

Cancellation also forces the prototype to make explicit something
that is currently implicit: a request's lifecycle has multiple
terminal states, not just `completed`, and the snapshot-only
boundary between engine and main must accommodate those states
without leaking llama-side state across the line.

---

## 2. Cancellation semantics

Cancellation is **cooperative** and **iteration-boundary**. The
engine never interrupts an in-flight `llama_decode` call. The
batch shape that has already been built for a given iteration is
not retroactively edited; cancellation takes effect on the
following iteration boundary.

Per-request lifecycle states:

```
admitted ──► running ──► completed ──► future fulfilled (status=completed)
                  │
                  └──► cancel_requested ──► cancel_observed
                                                │
                                                ▼
                                           kv_cleared
                                                │
                                                ▼
                                  future fulfilled (status=cancelled)
```

State transitions in detail:

- **`cancel_requested`** — control plane sets a flag on the
  request. May happen from main thread (external trigger) or be
  pre-set in the request (deterministic trigger by token count).
  Has no immediate effect on llama state.
- **`cancel_observed`** — the engine task notices the flag at an
  iteration boundary and decides to stop scheduling further
  decode rows for this seq. Recorded as
  `cancel_observed_iter`.
- **`kv_cleared`** — the engine task calls `llama_memory_seq_rm`
  for the cancelled seq, verifies pos_min/pos_max are -1, and
  runs the same cross-talk check against all still-active seqs
  used today.
- **future fulfilled (status=cancelled)** — the engine builds a
  `request_result` snapshot, `set_value`s the per-request
  promise exactly once, with `status = cancelled` and
  `n_decoded_at_cancel < decode_budget`.

A cancelled seq must NOT:

- appear in any decode batch built after `cancel_observed_iter`;
- have its KV reclaimed before the cross-talk check has snapshot
  pos_min/pos_max for the still-active siblings;
- have its promise fulfilled before its KV is cleared;
- be fulfilled twice;
- silently disappear (the future MUST become ready, with
  `status = cancelled`, exactly once).

A cancelled seq MAY:

- have already produced 0..(decode_budget − 1) tokens;
- have a partially-populated `generated_tokens` vector — the
  snapshot carries whatever tokens were produced before
  `cancel_observed_iter`;
- have a hash that is well-defined over its produced tokens but
  is NOT comparable to any whole-budget canonical anchor.

---

## 3. Request / result model changes

These are the proposed field additions; nothing here is
implemented yet.

### Request side (engine-internal `seq_state` + the construction-time descriptor used by main)

```cpp
struct seq_state {
    // existing Slice-1..5 fields:
    int32_t                  seq_id;
    int32_t                  decode_budget;
    int32_t                  n_decoded;
    int32_t                  pos_next;
    int32_t                  i_batch;
    llama_token              last_token;
    bool                     done;
    int32_t                  done_iter;
    llama_pos                pos_max_at_clear;
    bool                     kv_cleared;
    bool                     promise_fulfilled;
    uint64_t                 hash_state;
    std::vector<llama_token> generated_tokens;

    // NEW for cancellation:
    std::atomic<bool>        cancel_requested{false};
    int32_t                  cancel_after_decoded_tokens = -1;  // -1 = never (deterministic test trigger)
    bool                     cancel_observed             = false;
    int32_t                  cancel_observed_iter        = -1;
    int32_t                  n_decoded_at_cancel         = -1;
};
```

Notes:

- `cancel_requested` is `std::atomic<bool>` because main may set
  it while the engine task reads it. The engine reads with
  `load(memory_order_acquire)` only at iteration boundaries.
  Atomic on a single bool with one writer and one reader is
  sufficient — no further fences needed.
- `cancel_after_decoded_tokens` is a deterministic test trigger,
  read only by the engine. No atomic needed. `-1` disables.
- `cancel_observed`, `cancel_observed_iter`, and
  `n_decoded_at_cancel` are write-once, engine-only.

### Result side (`request_result`, the snapshot crossing the boundary)

```cpp
enum class request_status : uint8_t {
    completed = 0,
    cancelled = 1,
    failed    = 2,   // reserved; not produced by Cancel Slices 1-4
};

struct request_result {
    int32_t                  request_id;
    int32_t                  seq_id;
    int32_t                  decode_budget;
    int32_t                  n_decoded;
    uint64_t                 hash;
    int32_t                  done_iter;          // last iter touched (completion or cancel observation)
    llama_pos                pos_max_at_clear;
    bool                     kv_cleared;
    int64_t                  ttc_us;
    std::vector<llama_token> generated_tokens;

    // NEW for cancellation:
    request_status           status               = request_status::completed;
    int32_t                  n_decoded_at_cancel  = -1;   // -1 unless status == cancelled
    int32_t                  cancel_observed_iter = -1;   // -1 unless status == cancelled
};
```

Invariants on the snapshot:

- `status == completed` ⇒ `n_decoded == decode_budget`,
  `n_decoded_at_cancel == -1`, `cancel_observed_iter == -1`.
- `status == cancelled` ⇒
  `0 ≤ n_decoded_at_cancel < decode_budget`,
  `n_decoded == n_decoded_at_cancel`,
  `cancel_observed_iter ≥ 0`, `kv_cleared == true`.
- `status == failed` is reserved for future slices and not
  produced by the cancel slices in this design.
- `kv_cleared` is `true` for both `completed` and `cancelled`
  (engine clears KV before fulfilling the promise either way).

---

## 4. Engine-loop changes

The cancellation observation point is **a single check per
still-active seq at the top of each decode iteration**, plus an
equivalent check **after prefill** (which is the same boundary
as "before decode iter 1's row-addition"). The four mention
points in the brief reduce to this:

| brief mention                          | actual check site                                  |
|----------------------------------------|----------------------------------------------------|
| after prefill                          | once, before entering the decode loop              |
| before adding decode rows              | top of each decode iter, before building `active_idx` |
| after each llama_decode iteration      | satisfied transitively — equivalent to "before iter K+1"; no separate check |
| before fulfilling futures              | trivially — cancellation observation IS the fulfillment path; KV clear precedes `set_value`, exactly as today |

Pseudocode for the engine loop with cancellation:

```text
run_body():
    [prefill: build batch, llama_decode, argmax per seq]

    // --- cancellation check after prefill ---
    for seq in seqs:
        if seq.done: continue
        if cancel_should_observe(seq):
            cancel_and_fulfill(seq, prefill_iter)  // clear KV, set_value(status=cancelled)

    iter = 0
    while any_active(seqs):
        iter += 1

        // --- cancellation check before adding decode rows ---
        for seq in seqs:
            if seq.done: continue
            if cancel_should_observe(seq):
                cancel_and_fulfill(seq, iter)
                // a cancelled seq does NOT become an active row this iter

        // --- build batch for surviving active seqs only ---
        common_batch_clear(batch)
        active_idx = []
        for seq in seqs:
            if seq.done: continue
            common_batch_add(batch, seq.last_token, ...)
            active_idx.push(s)

        if batch.n_tokens == 0: break

        llama_decode(ctx, batch); llama_synchronize(ctx)

        // --- existing completion path; cancellation NOT observed here ---
        for s in active_idx:
            seq = seqs[s]
            // argmax, fold hash, n_decoded++
            if eog or n_decoded >= decode_budget:
                finalize_and_fulfill(seq, iter, status=completed)

    // residual KV check stays inside the engine task (Slice 3 invariant)
```

Where `cancel_should_observe(seq)` is true iff:

- `!seq.cancel_observed`, AND
- (`seq.cancel_requested.load(acquire)` is true), OR
  (`seq.cancel_after_decoded_tokens >= 0` AND
  `seq.n_decoded >= seq.cancel_after_decoded_tokens`).

And `cancel_and_fulfill(seq, iter)`:

1. Set `seq.cancel_observed = true`,
   `seq.cancel_observed_iter = iter`,
   `seq.n_decoded_at_cancel = seq.n_decoded`.
2. Run the existing `clear_and_check(seq, iter, mem)` — same KV
   removal + cross-talk check used by completion today. If it
   fails, propagate the engine error exactly as today.
3. Build a `request_result` with
   `status = request_status::cancelled`, populated cancel fields,
   and call `promises_[seq.seq_id].set_value(...)` exactly once.
4. Mark `seq.promise_fulfilled = true`.
5. Increment a new metric `cancelled_count`.

Why not check inside the post-decode argmax loop? Because that
would put cancellation observation in two places, opening up the
question "what if both completion and cancellation could fire on
the same iter?". Keeping the check at iteration boundaries makes
the answer trivial: completion fires from the post-decode argmax
loop; cancellation fires only at the *next* iteration's
top-of-loop check; they can never collide on the same seq in the
same iter.

`hpx::wait_all` and `engine_fut.get()` in main do not change.
The drain-on-error path that fulfills any unfulfilled promise
with `set_exception` at engine exit (Slice 3) also does not
change.

---

## 5. First cancellation smoke shape

Deterministic, self-contained, no external trigger needed.

```text
n_seqs:              99
budget mix:          {8, 64, 256}, round-robin → 33 + 33 + 33
prompt:              "Hello, my name is"
ctx_size:            32768  (actual n_ctx 50688)
n_batch:             1024
n_threads:           2

cancel set:
  cancel 3 of the budget-256 seqs after 16 decoded tokens each
  cancel 3 of the budget-64  seqs after 16 decoded tokens each
  do not cancel any budget-8 seqs (16 > 8 anyway, would never trigger)
```

Round-robin layout means seq_id % 3 maps to budget {8, 64, 256}.
Concretely:

- budget 8   seqs:  ids 0, 3, 6, 9, ..., 96   (33 ids, every i with i%3==0)
- budget 64  seqs:  ids 1, 4, 7, 10, ..., 97  (33 ids, every i with i%3==1)
- budget 256 seqs:  ids 2, 5, 8, 11, ..., 98  (33 ids, every i with i%3==2)

Cancel set (proposed; pick small deterministic ids):

- cancel_after_decoded_tokens = 16 for seq_ids: **1, 4, 7** (budget 64)
- cancel_after_decoded_tokens = 16 for seq_ids: **2, 5, 8** (budget 256)

Expected outcome:

```text
budget   8 → 33 completed, 0 cancelled
budget  64 → 30 completed, 3 cancelled (seq 1, 4, 7)
budget 256 → 30 completed, 3 cancelled (seq 2, 5, 8)
```

Cancellation observed at the iteration boundary AFTER the seq has
decoded its 16th token. Concretely:

- 16th token is produced by decode iter 15 (since the 1st token
  comes from prefill, decode iters 1..15 produce tokens 2..16).
  At top of iter 16, `n_decoded == 16 >= 16` and the cancel fires.
- Therefore `cancel_observed_iter == 16` for all 6 cancelled seqs.
- `n_decoded_at_cancel == 16` for all 6 cancelled seqs.

(These exact numbers are an expected anchor for the gate, derived
from the same `1 prefill + (D-1) decode` accounting used in
Slices 1–5; they are subject to confirmation when Cancel Slice 2
runs.)

---

## 6. Correctness gates

The smoke shape above must satisfy all of:

- engine HPX task completes
- every request future completes (`futures_completed == 99`)
- `engine_task_count == 1`
- exactly 6 results have `status == cancelled`
- exactly 93 results have `status == completed`
- every result with `status == cancelled` has
  `0 ≤ n_decoded_at_cancel < decode_budget`
- every result with `status == cancelled` has
  `cancel_observed_iter == 16`
- every result with `status == cancelled` has `kv_cleared == true`
- every result with `status == completed` has
  `n_decoded == decode_budget` (existing invariant)
- non-cancelled budget-8 seqs still hash to
  `0x0619d4d1900c2365`
- non-cancelled budget-64 seqs all share one hash within the run
  (cross-shape note: this hash MAY differ from the all-99-running
  Slice-5 hash because the per-iter batch shape changes once the
  cancelled seqs leave; that is expected and is documented as a
  Slice-2-cancel-shape anchor, not an HPX vs reference equality)
- non-cancelled budget-256 seqs all share one hash within the run
  (same caveat)
- cancelled seqs do NOT appear in any decode batch built after
  their `cancel_observed_iter` — verifiable by ensuring
  `pos_next` does not advance past `n_decoded_at_cancel + n_prompt`
- `wasted_decode_rows_after_cancel == 0` (the metric defined in
  §7 must be 0 in this slice; if it is nonzero, the cancel
  observation logic regressed)
- clearing a cancelled seq's KV does not disturb still-active
  siblings (existing cross-talk check)
- residual KV at end is empty for all 99 seq_ids (engine-side)
- every `llama_decode` call returns 0
- every promise is fulfilled exactly once
- KV clear happens before promise fulfillment for both
  `completed` and `cancelled` results

`--repeat 2` determinism: the cancelled set, the
`cancel_observed_iter`, the `n_decoded_at_cancel`, and the
non-cancelled hashes/done_iter/pos_max_at_clear sets must all be
identical across repeats. ttc / wall_ms remain non-determinism.

Cross-shape caveat (must be documented in the cancel slice's
results.md): the budget-64 and budget-256 hashes WILL differ
between the cancellation run and the full Slice-5 run, because
the per-iter batch shape changes once the 6 cancelled seqs leave.
We do NOT compare those hashes across the two runs. The cancel
shape's hashes form their own canonical anchors.

---

## 7. New metrics

Descriptive only. No comparison. No speedup language.

Per repeat iteration:

- `cancelled_count` — number of results with
  `status == cancelled`
- `completed_count` — number of results with
  `status == completed`
- `cancel_observed_iter` — set of distinct values of
  `cancel_observed_iter` over cancelled results (printed as a
  set, e.g. `{16}`)
- `decoded_tokens_before_cancel` — distribution
  (mean / p95 / max) of `n_decoded_at_cancel` over cancelled
  results
- `wasted_decode_rows_after_cancel` — total count of decode rows
  added for cancelled seqs at iter > `cancel_observed_iter`.
  **Must be 0** when cancellation observation logic is correct;
  printing it as a metric makes regressions visible without
  changing what is a hard gate (gate listed in §6).
- `active_seqs_per_iter_after_cancellation` — same distribution
  shape as the existing `active_seqs_per_iter` Slice-4 metric,
  but printed alongside the surviving-active counts so a reader
  can see the step-down at iter 16.
- `decode_calls_saved` — total decoded budget that was NOT
  executed because of cancellation, computed as
  `Σ_cancelled (decode_budget - n_decoded_at_cancel)`. For the
  smoke shape this is
  `3*(64-16) + 3*(256-16) = 144 + 720 = 864`. Descriptive only;
  this is not a speedup metric and must not be presented as one.

Existing Slice-4 metrics (`wall_ms`, `decode_calls`,
`update_iterations`, `rows_per_batch` p50/p95/max,
`active_seqs_per_iter` p50/p95/max, per-budget completed,
ttc_ms by class, futures_created / promises_fulfilled /
futures_completed / engine_task_count) all stay. Some of them
will naturally shift in the cancel run (e.g.
`update_iterations` is still 255 because at least one budget-256
seq survives; `decode_calls` is still 256 by the same reasoning;
`active_seqs_per_iter` shows a step-down at iter 16). Note these
shifts in the results.md but do not call them performance.

---

## 8. Trace events

Add to the existing Slice-4 `LLAMA_HPX_CB_TRACE=1` event family.
Same `[hpx-cb-gate] event=<name> key=value ...` format.

New events:

- `cancel_requested seq=<id> source=<external|deterministic>` —
  fires once when the request enters the cancel path. For the
  deterministic trigger this is logged at engine start, for each
  seq with `cancel_after_decoded_tokens >= 0`. For the external
  trigger it would fire when the atomic flag is set; in this
  design's smoke shape, no external trigger is used.
- `cancel_observed seq=<id> iter=<I> n_decoded_at_cancel=<N>` —
  fires when the engine observes the cancel flag at an
  iteration boundary.
- `cancel_kv_cleared seq=<id> pos_max_at_clear=<P>
  cross_talk_ok=1` — fires after the per-seq KV clear succeeds
  (mirrors the existing `kv_cleared` event for the completion
  path, but with a different name so cancel and completion are
  trivially separable in trace logs).
- `cancel_future_fulfilled seq=<id> ttc_us=<T>` — fires after
  `set_value(status=cancelled)`.

Existing Slice-4 events (`engine_start`, `request_admitted`,
`seq_prefilled`, `decode_row`, `seq_complete`, `kv_cleared`,
`promise_fulfilled`, `engine_stop`) continue to fire on the
completion path exactly as today. They are NOT reused for cancel.

Trace gating remains: with `LLAMA_HPX_CB_TRACE` unset, the entire
trace family — old and new — must produce 0 stderr lines.

---

## 9. Out of scope (every cancel slice)

Explicitly excluded from this design and from the slices that
implement it:

- HTTP / network disconnect handling
- external client cancellation through any I/O channel
- async cancellation from a thread that is not main or the
  engine task
- priority scheduling
- preemption / requeue semantics
- streaming (partial-result emission to the client)
- live admission control / context-cell pressure responses
- performance claims (speedup, latency, throughput)
- interruption inside `llama_decode` (this is forbidden — see §2)
- modifying `tools/server/`, `tools/serving-bench/`,
  `tools/multiseq-batch-gate/`
- introducing a named orchestration pool / resource partitioner
  (Slice 3b is still the right home for that work)

External cancellation (atomic flag set from main) is *modeled* in
the data model (`cancel_requested` is atomic) so a future slice
can wire it up without redoing the request schema. It is NOT
exercised in any Cancel slice 1–4.

---

## 10. Implementation slices

Stop-and-check after each. Do not start the next without explicit
approval. No commits without explicit approval.

### Cancel Slice 1 — data model only, no behavior change

- Add `cancel_requested` (atomic),
  `cancel_after_decoded_tokens`, `cancel_observed`,
  `cancel_observed_iter`, `n_decoded_at_cancel` to `seq_state`.
- Add `request_status` enum and the cancel fields to
  `request_result` (default `status = completed`,
  `n_decoded_at_cancel = -1`,
  `cancel_observed_iter = -1`).
- Engine writes `status = completed` on every completion path.
- Main validates that every result has `status == completed` and
  the cancel fields stay at their defaults (this proves the data
  model is plumbed without changing behavior).
- All Slice-1..5 gates still pass on the standard 99-seq /
  {8,64,256} shape.
- Final stdout line: `HPX_CB_CANCEL_STEP1: PASS` (separate from
  the Slice-4 `HPX_CB_STEP4` so the cancel-track verdict is
  unambiguous in logs).
- No new CLI flags. No deterministic trigger wired up. No traces
  added yet.

### Cancel Slice 2 — deterministic trigger, engine observes, no future status

- Add CLI: `--cancel <seq_id>:<after_tokens>` (repeatable). Sets
  `cancel_after_decoded_tokens` on the named seq at construction
  time.
- Engine implements `cancel_should_observe` and the
  iteration-boundary check exactly as in §4.
- On observation, engine clears KV via `clear_and_check`,
  records `cancel_observed_iter` /
  `n_decoded_at_cancel`, and continues the loop. The seq does
  NOT appear in any subsequent batch.
- Promises for cancelled seqs are still fulfilled with the
  current `status = completed` default. (Status flips in
  Slice 3.)
- Validate: cancelled seqs have `n_decoded < decode_budget` in
  the snapshot; `wasted_decode_rows_after_cancel == 0`.
- All non-cancelled-shape correctness gates still pass.
- Final stdout line: `HPX_CB_CANCEL_STEP2: PASS`.
- No traces yet. No metrics block changes yet.

### Cancel Slice 3 — future status=cancelled and cancel-family traces

- Set `request_result::status = request_status::cancelled` on
  the cancellation path.
- Set `n_decoded_at_cancel`, `cancel_observed_iter` on the
  snapshot.
- Wire the four trace events from §8 behind the existing
  `LLAMA_HPX_CB_TRACE=1` gate.
- Main validates from snapshots: cancelled count, completed
  count, status enum coverage, no-promise-fulfilled-twice on
  either status, hash anchor for non-cancelled budget-8.
- All §6 correctness gates pass.
- Final stdout line: `HPX_CB_CANCEL_STEP3: PASS`.
- Trace gating remains intact (compact runs still emit zero
  cancel events).

### Cancel Slice 4 — cancelled vs non-cancelled lifecycle metrics

- Extend the Slice-4 metrics block with the new metrics from §7.
- `wasted_decode_rows_after_cancel` printed and gated to 0.
- `decode_calls_saved` printed as a descriptive number only.
- Update `tools/hpx-continuous-batch-gate/results.md` with a
  Cancel-track section: smoke shape, expected outcome, observed
  vs expected anchors, cross-shape caveat for the per-class
  hashes vs the Slice-5 reference run.
- Final stdout line: `HPX_CB_CANCEL_STEP4: PASS`.

After Cancel Slice 4 the prototype demonstrates a real serving
invariant the pre-cancel prototype did not. Anything beyond that
(external cancellation wiring, deadline integration,
admission-driven cancel) is a separate design doc.

---

## 11. Exact next implementation prompt

Below is the exact prompt to send back when Cancel Slice 1 is
ready to implement. **Do not implement it from this design note
alone.**

```text
Proceed with Cancel Slice 1 only.

Goal:
Add the cancellation data model to the HPX continuous-batching
gate without changing any runtime behavior.

Read first:
docs/hpx/hpx_continuous_batching_cancellation_design.md
tools/hpx-continuous-batch-gate/README.md
tools/hpx-continuous-batch-gate/results.md
local/ahandoff.md
CLAUDE.md

Do not modify:
- tools/server/
- tools/serving-bench/
- tools/multiseq-batch-gate/

Do not add (Cancel Slice 1):
- any cancellation behavior
- a deterministic trigger that actually fires
- traces
- metrics changes
- CLI flags
- HTTP / external cancellation
- orchestration pool / resource partitioner

Cancel Slice 1 scope:
- Add to engine-internal seq_state:
  - std::atomic<bool>  cancel_requested{false}
  - int32_t            cancel_after_decoded_tokens = -1
  - bool               cancel_observed             = false
  - int32_t            cancel_observed_iter        = -1
  - int32_t            n_decoded_at_cancel         = -1
- Add request_status enum with values:
  - completed
  - cancelled
  - failed
- Add to request_result:
  - request_status status               = completed
  - int32_t        n_decoded_at_cancel  = -1
  - int32_t        cancel_observed_iter = -1
- Engine writes status = request_status::completed on every
  completion path (existing prefill-completion and
  decode-loop-completion sites).
- Engine never observes cancellation in this slice — the new
  fields are present but unused.
- Main validates from request_result snapshots that every
  result has:
  - status == request_status::completed
  - n_decoded_at_cancel == -1
  - cancel_observed_iter == -1
  This proves the data model is plumbed without behavior change.

Correctness gates:
All Slice-3 / Slice-4 gates still pass on the 99-seq /
{8,64,256} shape:
- engine_task_count == 1
- futures_created == 99
- promises_fulfilled == 99
- futures_completed == 99
- 33 seqs per budget class
- budget 8 hash == 0x0619d4d1900c2365
- budget 64 has one unique hash within the run
- budget 256 has one unique hash within the run
- done_iter sets {7}/{63}/{255}
- pos_max_at_clear sets {12}/{68}/{260}
- per-seq KV clear succeeds before promise fulfillment
- residual KV empty at end (engine-side)
- every llama_decode returns 0
- --repeat 2 deterministic across token / hash / done_iter /
  pos_max_at_clear

Cancel-Slice-1-specific gates:
- every result.status == request_status::completed
- every result.n_decoded_at_cancel == -1
- every result.cancel_observed_iter == -1
- no new trace events fire (compact and trace runs both
  unchanged from Slice 4)

Final stdout line:
HPX_CB_CANCEL_STEP1: PASS
or
HPX_CB_CANCEL_STEP1: FAIL: <reason>

Update:
- tools/hpx-continuous-batch-gate/hpx-continuous-batch-gate.cpp
- tools/hpx-continuous-batch-gate/README.md (status block only)

Build only:

cmake --build /Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on \
  --target llama-hpx-continuous-batch-gate

Run without trace:

/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate \
  --model /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_cancel_step1.stdout \
  2> local/hpx_cb_cancel_step1.stderr

If that passes, also run repeat:

/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate \
  --model /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_cancel_step1_repeat2.stdout \
  2> local/hpx_cb_cancel_step1_repeat2.stderr

Stop and report:
- files modified
- build command
- run commands
- final stdout lines
- whether all Slice-3 / Slice-4 gates still pass
- whether every result has status == completed and the cancel
  fields at their defaults
- whether repeat 2 passed
- any HPX runtime warnings
- any llama_decode return-code issues

Do not proceed to Cancel Slice 2 until I approve.
```
