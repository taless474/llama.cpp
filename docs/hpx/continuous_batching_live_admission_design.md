# HPX continuous-batching: live admission — design note

Status: design only. No code yet. No commit yet.

This note defines how live admission should work in the existing
HPX continuous-batching prototype
(`tools/hpx-continuous-batch-gate/`). It does not propose
implementing it yet; it scopes the work into small slices and
specifies the gates each slice must hit.

This is a correctness-and-lifecycle design. There are no
performance claims and none are planned for this track.

---

## 0. Architecture we are building on

Slices 1–5 plus Cancel Slices 1–4 are landed and PASS. Live
admission must preserve every invariant they established. The
load-bearing ones:

- one `llama_model`, one `llama_context`, one shared
  `llama_batch`, one HPX engine task per repeat iteration; only
  the engine task touches `llama_context` / `llama_batch` /
  `llama_decode` / `llama_memory_seq_*` /
  `llama_get_logits_ith`.
- one `hpx::promise<request_result>` per seq; futures handed
  out before the engine task starts; each promise is fulfilled
  only after per-seq KV clear and a cross-talk check.
- main does `hpx::wait_all` on the per-request futures, then
  `engine_fut.get()`, then validates correctness exclusively
  from `request_result` snapshots.
- cooperative cancellation at iteration boundaries: cancelled
  seqs are removed from later decode batches, KV is cleared by
  the engine, futures fulfilled with `status = cancelled` only
  after KV clear; residual KV empty for all seq_ids at end of
  run.

---

## 1. Why live admission is next

Cancellation proved the leave path:

```
active seq → cancel observed → KV cleared → future fulfilled
```

Live admission proves the symmetric enter path — a waiting
request enters the active set:

```
waiting request → admitted into freed capacity →
prefilled at iteration boundary → joins later decode batches →
future fulfilled
```

This also forces the prototype to make explicit what
cancellation modeled implicitly: a `seq_id` is a *slot* that
can be reused after KV clear, not a permanent identity bound
to a single request. (Post-v1 capabilities — arrival
schedules, priority, streaming, an orchestration pool —
depend on this slot model and are out of scope here; see §10.)

---

## 2. What live admission means in this prototype

Vocabulary defined for this design:

- **request** — the user-level lifecycle object. Carries a
  `request_id`, a `decode_budget`, a prompt, a completion future.
  `request_id` is **not** the same as `seq_id` once admission
  exists.
- **seq_id / slot** — a `llama.cpp` index in `[0, n_seq_max)`
  that owns a strip of KV cells. A slot is *occupied* while a
  request is mapped to it; *free* before its first use and
  again after its previous owner's KV clear succeeds.
- **active request** — a request currently mapped to a `seq_id`,
  i.e. running through prefill or decode in the engine loop.
- **waiting request** — a request that has been queued but is
  not mapped to a `seq_id` yet. The engine has not seen it for
  decode purposes.
- **admission boundary** — the same iteration boundary where
  cancellation is observed. Concretely: the top of each decode
  iteration, before building the active row set, after the
  cancellation check has run for that iter. Admission, like
  cancellation, never interrupts an in-flight `llama_decode`.
- **slot reuse** — claiming a free `seq_id` for a waiting
  request. Requires the previous owner's KV to be empty
  (`pos_min == pos_max == -1`) and verified before any prefill
  row is added for the new owner.
- **prefill after admission** — at the admission boundary, the
  newly-admitted request's prompt rows are added to the same
  shared `llama_batch` that decode rows for surviving active
  seqs are added to. This is the first iteration where the
  prototype mixes prefill and decode rows in the same batch.
- **completion future** — the per-request
  `hpx::promise<request_result>` and its corresponding future.
  In v1 the future is created at admission time, not at
  request-queueing time, and is handed out to main before the
  engine binds the request to a `seq_id`.
- **free_due_to_cancel** — an ordered list (deterministic
  ascending `seq_id`) of slots that became reusable
  specifically because a cancellation KV clear succeeded. A
  `seq_id` is appended to this list only after its
  cancel-KV-clear step returns successfully. This is **not**
  the global free-slot list: naturally completed
  (e.g. budget-8) `seq_id`s are excluded from
  `free_due_to_cancel` and are not eligible for reuse in the
  first smoke. For the first smoke this set is exactly
  `{1, 2, 4, 5, 7, 8}` and is consumed in ascending `seq_id`
  order at the admission boundary.
- **admission_source** — a `request_result` field that records
  why the slot used by an admitted request became eligible.
  Allowed values in v1:
    - `none` — non-admitted / original active request (default
      for every result that was not produced by live
      admission).
    - `cancel_freed` — admitted request that reused a slot
      taken from `free_due_to_cancel`. This is the only
      admission source used in the first smoke (Slice 3).
    - `completion_freed` — admitted request that reused a
      slot taken from `free_due_to_completion`. Added in
      Live Admission Slice 5 behind the default-OFF
      `--reuse-completed` CLI flag; see §9 Slice 5 for the
      full data-model, demand-gate, source-priority, smoke
      shape, mapping, and gate definitions.
- **reused_seq_id** — for an admitted request, the `seq_id`
  the engine bound it to after the KV-empty verification
  (`pos_min == pos_max == -1`) succeeded. `-1` for
  non-admitted results.
- **previous_request_id** — for an admitted request, the
  `request_id` that previously owned the reused `seq_id` (the
  cancelled or completed prior owner whose KV clear made the
  slot eligible). `-1` for non-admitted results and for slots
  that have never been used before.

Flow on the smoke shape:

1. Main builds the request set: `N_active` requests admitted
   at `t = 0` plus `N_waiting` requests that sit in a queue.
2. Main creates per-request futures only for active requests
   in v1 Slice 2; waiting-request futures are created at
   admission time (Slice 3) and the future handles are returned
   to main via a shared collection that main reads after the
   engine task completes (or `hpx::wait_all`s once the future
   list is final).
3. Engine prefills the `N_active` requests at iter 0, decodes
   normally.
4. At cancellation boundaries, cancelled seqs leave; their
   `seq_id` slots become free after the KV clear succeeds.
5. At the next admission boundary, the engine pops up to
   `K = freed_slot_count` waiting requests in deterministic
   order, binds each to a freed `seq_id`, creates their
   per-request promise/future, adds prefill rows to the current
   iteration's `llama_batch`, and continues.
6. Admitted requests prefill exactly once. Their first token
   comes from the prefill argmax in the admission iteration.
   Subsequent decode iters produce the rest of their budget.
7. All requests (active-completed, active-cancelled,
   admitted-completed) eventually have their futures fulfilled
   with the correct status. Residual KV is empty at end.

---

## 3. What must stay invariant

These are non-negotiable. Every admission slice must preserve
every one of these.

- **Single engine ownership.** Only the engine task touches
  `llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`. Admission
  bookkeeping (queue-pop, slot-pick, future creation, snapshot
  fulfillment) happens inside the engine task. The waiting
  queue may be built before the engine task starts, but the
  engine consumes from it exclusively.
- **No parallel `llama_decode`.** Single engine task per repeat
  iteration. Admission does not introduce any new HPX task.
- **Snapshot-only futures.** No future consumer touches
  `llama_context` / `llama_batch` / `llama_decode` /
  `llama_memory_seq_*` / `llama_get_logits_ith`. The
  `request_result` snapshot is the only thing that crosses the
  engine→main boundary, for both active-completed,
  active-cancelled, and admitted-completed paths.
- **Future-before-decode.** Every request's future is handed
  out (`promises_[i].get_future()`) before any decode row is
  added on its behalf. For active requests this still happens
  before the engine task starts (Slice-3 invariant). For
  admitted requests this happens at the admission boundary,
  before the engine adds the request's prefill rows.
- **KV-clear-before-reuse.** Before a `seq_id` is bound to a
  newly-admitted request, the engine asserts
  `llama_memory_seq_pos_min(mem, seq_id) == -1` and
  `llama_memory_seq_pos_max(mem, seq_id) == -1`. This is the
  same shape used by the residual-KV-empty end-of-run check
  today. If the assertion fails, the engine sets a fatal error,
  drains the unfulfilled promises with `set_exception`, and
  returns; main observes a `FAIL` line.
- **One live owner per slot.** At any moment a `seq_id` is
  mapped to at most one live request. The engine never adds two
  rows from two different requests with the same `seq_id` to
  the same batch.
- **No mid-decode admission.** Admission, like cancellation, is
  observed only at iteration boundaries. The engine never
  admits a waiting request after rows have been added for the
  current iter and before `llama_decode` returns.
- **Admission is monotonic.** Within a single repeat iteration,
  a waiting request transitions waiting → admitted exactly
  once. It is never returned to the queue. (Re-queueing after
  failure is out of scope; an admitted request that hits an
  error follows the same `set_exception` drain path as today.)
- **Residual KV empty at end.** All 99 `seq_id`s satisfy
  `pos_min == pos_max == -1` at the end of the run, regardless
  of whether they ever owned an admitted request, the original
  active request only, or both in sequence.
- **Every future completes.** `hpx::wait_all` over the final
  future list returns. No future is left unfulfilled. Every
  promise is fulfilled exactly once.
- **`--repeat 2` deterministic** within the same admission
  schedule (same waiting-queue contents, same arrival order,
  same plan-vs-budget mapping).

---

## 4. First smoke shape — cancel-freed-slot live admission smoke

Deterministic, self-contained, no external arrival schedule
needed. Reuses the existing cancellation plan to free the
admission slots.

This first smoke is explicitly named the **cancel-freed-slot
live admission smoke**: admission intentionally reuses *only*
slots freed by the cancellation plan, not every free seq_id.
Naturally completed budget-8 free slots are ignored for this
smoke.

```text
n_seq_max:           99
n_active:            93     (request_ids 0..92, mapped 1:1 to seq_ids 0..92)
n_waiting:            6     (request_ids 93..98)
total requests:      99

prompt:              "Hello, my name is"      (same for every request)
ctx_size:            32768  (actual n_ctx 50688)
n_batch:             1024
n_threads:           2
hpx_os_threads:      1

active budget mix:   round-robin over {8, 64, 256} for request_ids 0..92
                       i % 3 == 0 → budget 8   (31 active requests)
                       i % 3 == 1 → budget 64  (31 active requests)
                       i % 3 == 2 → budget 256 (31 active requests)

cancellation plan (carries forward):
                     cancel request_ids {1, 4, 7, 2, 5, 8} after 16 decoded tokens
                     (3 budget-64 + 3 budget-256, same as the cancel-track smoke)

waiting requests:
                     request_ids 93..98, each decode_budget = 64
                     queued in FIFO order at construction time
                     no per-waiting-request arrival timing (they all become
                     eligible at t=0; admission is gated by slot availability,
                     not by an arrival clock)

admission rule:
                     at the top of each decode iter, after cancellation
                     observation has run, the engine binds up to K waiting
                     requests to up to K cancel-freed seq_ids (K =
                     min(waiting_queue size, free_due_to_cancel size)).
                     The eligible reuse set for this smoke is the
                     deterministic cancel-freed set: {1, 2, 4, 5, 7, 8}.
                     Naturally completed budget-8 free slots are NOT
                     eligible and must be ignored.

                     The engine maintains a free_due_to_cancel queue.
                     A seq_id is appended to that queue only after its
                     cancellation KV clear succeeds. At the admission
                     boundary the engine pops from free_due_to_cancel in
                     deterministic order (ascending seq_id) and binds the
                     FIFO head of the waiting queue to it. Each admitted
                     request's prefill rows are added to the SAME
                     llama_batch as decode rows for surviving active
                     seqs in this iter (mixed batch).
```

Round-robin layout means seq_id % 3 maps to budget {8, 64, 256}:

- budget 8   active seqs:  request_ids 0, 3, 6, 9, ..., 90
- budget 64  active seqs:  request_ids 1, 4, 7, 10, ..., 91
- budget 256 active seqs:  request_ids 2, 5, 8, 11, ..., 92

Cancellation fires at iter 16 for the 6 cancel-plan request_ids.
Their KV is cleared by the engine in iter 16, freeing seq_ids
`{1, 2, 4, 5, 7, 8}`.

At the top of iter 17 (the next iteration boundary after the
slots were freed), the engine admits the 6 waiting requests:

| popped from queue | freed seq_id picked |
|-------------------|---------------------|
| request_id 93     | seq_id 1            |
| request_id 94     | seq_id 2            |
| request_id 95     | seq_id 4            |
| request_id 96     | seq_id 5            |
| request_id 97     | seq_id 7            |
| request_id 98     | seq_id 8            |

Rationale for the cancel-freed-only rule: with `n_active = 93`,
budget-8 active seqs finish around `done_iter = 7`, so by iter
17 their slots `{0, 3, 6, 9, ...}` are already free. A generic
"lowest-numbered free seq_id among all free seqs" rule would
silently bind waiting requests to those completed-budget-8
slots instead of the cancel-freed slots, turning this smoke
into a reuse-after-completion smoke instead of the intended
reuse-after-cancellation smoke. This first smoke therefore
restricts reuse to `free_due_to_cancel` only; reuse-after-
natural-completion is out of scope here and belongs to a
later smoke.

Iter 17's `llama_batch` therefore contains:

- 36 prefill rows (6 admitted requests × 6 prompt tokens each),
  with `logits = true` only on the last prompt row per
  admitted seq;
- 56 decode rows (28 surviving budget-64 + 28 surviving
  budget-256 active seqs), with `logits = true` per row.

Total: 92 rows in iter 17, well within `n_batch = 1024`.

Iter 17 prefill argmax produces the first decoded token for
each admitted request. Decode iters 18..80 produce tokens 2..64
for each admitted budget-64 seq. `done_iter` for every admitted
seq is therefore 80.

Surviving active budget-64 seqs still finish at `done_iter = 63`
(the iter 17 prefill rows do not displace their decode row in
the mixed batch). Surviving active budget-256 seqs still finish
at `done_iter = 255`.

Caveat carried forward: surviving-active budget-64 and
budget-256 hashes in the admission run will not equal the
cancellation-only run hashes, because iter 17 is now a mixed
prefill+decode batch instead of a decode-only batch. FP order is
batch-shape-dependent. Within the admission run the same
within-class uniqueness gate applies.

---

## 5. Expected statuses

Three result groups in the snapshot population. Counts assume
the smoke shape from §4.

```text
original completed (status = completed):
  budget   8: 31 results (request_ids 0, 3, 6, ..., 90)
  budget  64: 28 results (request_ids 10, 13, ..., 91)
  budget 256: 28 results (request_ids 11, 14, ..., 92)
  subtotal:   87

original cancelled (status = cancelled):
  budget  64: 3 results  (request_ids 1, 4, 7)
  budget 256: 3 results  (request_ids 2, 5, 8)
  subtotal:   6

admitted completed (status = completed):
  budget  64: 6 results  (request_ids 93..98)
  subtotal:   6

totals:
  status = completed:   93   (87 original + 6 admitted)
  status = cancelled:   6
  futures total:        99
```

Per-result invariants (these become validation gates):

- **original completed**: `status == completed`,
  `n_decoded == decode_budget`, `done_iter == decode_budget − 1`,
  `pos_max_at_clear == n_prompt + decode_budget − 2`,
  `kv_cleared == true`, `cancel_observed_iter == -1`,
  `n_decoded_at_cancel == -1`, `admitted_at_iter == -1`,
  `reused_seq_id == -1`, `admission_source == none`,
  `previous_request_id == -1`.
- **original cancelled**: `status == cancelled`,
  `n_decoded == 16`, `n_decoded_at_cancel == 16`,
  `cancel_observed_iter == 16`, `kv_cleared == true`,
  `pos_max_at_clear == n_prompt + 16 − 2 == 20`,
  `admitted_at_iter == -1`, `reused_seq_id == -1`,
  `admission_source == none`, `previous_request_id == -1`.
- **admitted completed**: `status == completed`,
  `n_decoded == decode_budget` (= 64 in the smoke),
  `done_iter == admitted_at_iter + decode_budget − 1` (= 80 in
  the smoke), `pos_max_at_clear == n_prompt + decode_budget − 2`
  (= 68 for budget-64 admitted), `kv_cleared == true`,
  `cancel_observed_iter == -1`, `n_decoded_at_cancel == -1`,
  `admitted_at_iter == 17` (in the smoke), `reused_seq_id ∈
  {1, 2, 4, 5, 7, 8}` and is unique among admitted results,
  `admission_source == cancel_freed`,
  `request_id ∈ {93, 94, 95, 96, 97, 98}`,
  `previous_request_id` equals the original cancelled
  `request_id` that owned the reused slot (in this smoke each
  cancelled `request_id` matches its own `seq_id`, so for
  `reused_seq_id == s`, `previous_request_id == s`).

`admitted_at_iter`, `reused_seq_id`, `admission_source`, and
`previous_request_id` are new `request_result` fields. Default
values for non-admitted results: `admitted_at_iter == -1`,
`reused_seq_id == -1`, `admission_source == none`,
`previous_request_id == -1`.

---

## 6. Correctness gates

The smoke shape above must satisfy all of:

- engine HPX task completes
- `engine_task_count == 1`
- every request future completes (`futures_completed == 99`)
- exactly 6 results have `status == cancelled`
- exactly 93 results have `status == completed` (87 original + 6
  admitted)
- exactly 6 results have `admitted_at_iter >= 0`; for those,
  `admitted_at_iter == 17` in the smoke and `reused_seq_id`
  values form the set `{1, 2, 4, 5, 7, 8}`
- the multiset of `reused_seq_id` over admitted results has no
  duplicates (no slot is bound twice in the same run)
- cancel-freed-slot reuse only: admitted requests must reuse
  only seq_ids `{1, 2, 4, 5, 7, 8}`; no naturally completed
  budget-8 seq_id (i.e. any of `{0, 3, 6, 9, ..., 90}`) may be
  reused in this smoke
- `reused_seq_id` set equality: the set of `reused_seq_id`
  values across the 6 admitted results must equal
  `{1, 2, 4, 5, 7, 8}` exactly (every cancel-freed slot is
  reused exactly once and no other slot is reused)
- `admission_source == cancel_freed` for every admitted result
  (the only valid v1 admission source for this smoke)
- `admission_source == none` for every original completed and
  every original cancelled result (no admission ever occurred
  for the original active set)
- `previous_request_id` mapping for admitted results matches
  the cancel plan exactly:
    request 93 → previous_request_id 1, reused_seq_id 1
    request 94 → previous_request_id 2, reused_seq_id 2
    request 95 → previous_request_id 4, reused_seq_id 4
    request 96 → previous_request_id 5, reused_seq_id 5
    request 97 → previous_request_id 7, reused_seq_id 7
    request 98 → previous_request_id 8, reused_seq_id 8
- every admitted result has
  `kv_cleared == true`,
  `n_decoded == decode_budget`,
  `done_iter == admitted_at_iter + decode_budget − 1`
- admitted requests are not decoded before admission: every
  admitted seq's first appearance in any decode-row stream
  occurs at `iter == admitted_at_iter` (not earlier)
- admitted requests prefill exactly once: the engine emits
  exactly one `admitted_prefilled` event per admitted request
  (under `LLAMA_HPX_CB_TRACE=1`)
- admitted requests use seq_ids whose previous KV was cleared:
  before binding, the engine asserts
  `llama_memory_seq_pos_min(mem, reused_seq_id) == -1` and
  `llama_memory_seq_pos_max(mem, reused_seq_id) == -1`; if the
  assertion fails, the engine fails closed (no admission, no
  silent reuse)
- cancelled seqs are not decoded after cancel observation
  (existing `wasted_decode_rows_after_cancel == 0` invariant)
- non-cancelled budget-8 active seqs still hash to
  `0x0619d4d1900c2365` (budget-8 is not part of the cancel plan
  and not a slot reuse target, so its batch shape is unchanged
  from the cancellation-only run; the canonical anchor still
  applies)
- surviving-active budget-64 seqs all share one hash within the
  run (same-shape uniqueness gate); admitted budget-64 seqs all
  share one hash within the run (separate from surviving's
  hash, because they prefilled at a different iter); the two
  hashes are not required to match each other
- surviving-active budget-256 seqs all share one hash within
  the run
- clearing a cancelled or completed seq's KV does not disturb
  still-active siblings (existing cross-talk check; covers
  admitted siblings the same way)
- residual KV at end is empty for all 99 seq_ids — original
  active, original cancelled, and admitted (engine-side check)
- every `llama_decode` call returns 0
- `decode_failures == 0`
- every promise is fulfilled exactly once (no double-fulfillment
  on either status, including via slot reuse)
- KV clear happens before promise fulfillment for every result
  (already an invariant for completed and cancelled; admitted
  fulfilment uses the same `clear_and_check` → `fulfill_promise`
  ordering)
- `--repeat 2` deterministic within the same admission schedule:
  cancelled set, `cancel_observed_iter`, `n_decoded_at_cancel`,
  admitted set, `admitted_at_iter`, `reused_seq_id`, all
  hashes/`done_iter`/`pos_max_at_clear` sets are byte-identical
  across repeats. `wall_ms` and `ttc_us`-derived metrics remain
  outside the determinism contract.

---

## 7. Trace events

Add to the existing Slice-4 / Cancel-Slice-3
`LLAMA_HPX_CB_TRACE=1` event family. Same
`[hpx-cb-gate] event=<name> key=value ...` format. All gating
behavior is preserved (compact runs emit zero `event=` lines).

Existing events (unchanged):

```text
engine_start n_seqs=<int> prompt_tokens=<int> max_budget=<int>
request_admitted seq=<id> budget=<int>     (original active requests only)
seq_prefilled seq=<id> first_token=<id>    (original active requests only)
decode_row iter=<int> seq=<id> pos=<int>   (any seq, any iter)
seq_complete seq=<id> budget=<int> done_iter=<int> hash=<hex>
kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1
promise_fulfilled seq=<id> ttc_us=<int>
engine_stop decode_calls=<int> update_iterations=<int> wall_ms=<float>
cancel_requested seq=<id> budget=<int> cancel_after=<int>
cancel_observed seq=<id> iter=<int> n_decoded=<int>
cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1
cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>
```

New events for live admission:

```text
request_queued       request=<id> budget=<int> queue_pos=<int>
request_admitted_live request=<id> reused_seq_id=<id> iter=<int>
seq_reused           seq_id=<id> previous_owner=<request_id> new_owner=<request_id> iter=<int>
admitted_prefilled   request=<id> seq_id=<id> first_token=<id>
admitted_decode_row  request=<id> seq_id=<id> iter=<int> pos=<int>
admitted_complete    request=<id> seq_id=<id> budget=<int> done_iter=<int> hash=<hex>
```

Notes:

- `request_queued` fires once per waiting request at
  construction time, so the queued population is visible in
  trace before the engine starts.
- `request_admitted_live` is distinct from the existing
  `request_admitted` event so a grep can separate static
  admissions (start of run, original active set) from live
  admissions (slot reuse). Existing `request_admitted` keeps its
  semantics.
- `seq_reused` fires inside the engine at the admission boundary,
  immediately after the KV-empty assertion succeeds and before
  any prefill row is added for the new owner. `previous_owner`
  is the most recent prior request bound to that `seq_id`
  (whether completed or cancelled).
- `admitted_*` mirrors the completion-path family
  (`seq_prefilled`, `decode_row`, `seq_complete`) so a reader
  can audit the admitted lifecycle without ambiguity. The
  generic `decode_row` keeps firing for all decode rows (active
  and admitted) to preserve the existing 1:1 row count
  invariant; `admitted_decode_row` is an *additional* fine-
  grained event for admitted rows only.

Default-plan expected counts under `LLAMA_HPX_CB_TRACE=1`:

```text
engine_start              =     1
engine_stop               =     1
request_admitted          =    93   (original active only)
request_queued            =     6
request_admitted_live     =     6
seq_reused                =     6
seq_prefilled             =    93
admitted_prefilled        =     6
decode_row                = <derived>   (active + admitted decode rows)
admitted_decode_row       =   378       (6 admitted × 63 decode iters each)
seq_complete              =    87
admitted_complete         =     6
kv_cleared                =    93   (87 completed active + 6 admitted)
promise_fulfilled         =    93
cancel_requested          =     6
cancel_observed           =     6
cancel_kv_cleared         =     6
cancel_future_fulfilled   =     6
```

(`decode_row` total derives from the iter-by-iter active count;
the design does not pre-commit to its exact value, but the
expected count is computable once Slice 3 lands and is gated as
a Cancel-Slice-3-style anchor at that point.)

---

## 8. Metrics

Descriptive only. No comparison. No speedup language. Added to
the existing per-iter metrics block.

Per repeat iteration:

- `queued_count` — number of waiting requests at construction
  time. For the smoke: 6.
- `admitted_count` — number of waiting requests that became
  admitted during this iteration's run. For the smoke: 6.
- `admission_iter_set` — set of distinct iter values at which
  admission fired. For the smoke: `{17}`.
- `reused_seq_id_count` — number of distinct `seq_id`s that
  served a second request in this run. Equal to `admitted_count`
  in v1 (each admitted request consumes one previously-occupied
  slot). For the smoke: 6.
- `reused_seq_id_set` — explicit set of reused seq_ids printed
  for grep-friendly audit. For the smoke: `{1, 2, 4, 5, 7, 8}`.
- `admitted_ttc_ms[budget=B]` — ttc distribution (mean / p95) of
  admitted-completed results, segmented by budget class. Note
  the descriptive caveat: ttc here is wall-clock from the start
  of the engine run to admission-promise fulfillment, not from
  the admission boundary; that is the same convention used by
  the existing `ttc_ms_completed[budget=*]` field for original
  active requests, kept consistent for grep-friendly comparison.
- `waiting_queue_depth_after_admission_per_iter` — distribution
  of queue depth (p50 / p95 / max) sampled **after** each decode
  iteration's admission loop completes, so iters where admission
  fired record the post-consumption depth. For the smoke this
  yields `p50=0 p95=6 max=6 samples=255`: the 16 pre-admission
  iters (1..16) record `6`, iter 17 records `0` because admission
  consumed all 6 waiters that iter, and iters 18..255 record `0`.
- `active_seqs_per_iter` (existing) will show a different curve
  with admission. The metric stays unchanged in name and shape;
  its value reflects active + admitted.

Existing metrics (Slice 4 + Cancel Slice 4) all stay. Some
naturally shift in the admission run:

- `update_iterations` — stays at 255 because at least one
  surviving budget-256 active seq still runs to completion;
- `decode_calls` — stays at 256 (1 prefill + 255 decode iters);
- `rows_per_batch` — p50 / p95 / max curve will differ because
  iter 17 is a mixed prefill+decode batch with 92 rows; this is
  descriptive, not a determinism gate.

Document these shifts in the admission slice's results.md but do
not call them performance.

---

## 9. Implementation slices

Stop-and-check after each. Do not start the next without
explicit approval. No commits without explicit approval.

### Live Admission Slice 1 — data model only, no behavior change

- Add to engine-internal `seq_state`:
    - `int32_t request_id              = -1`  (== seq_id for
      pre-admission Slices, distinguishable thereafter)
    - `int32_t admitted_at_iter        = -1`
    - `int32_t previous_request_id     = -1`  (last owner of
      the slot; `-1` if the slot has never been used)
    - `admission_source                = none`  (enum or
      string field; allowed values per §2 are `none` and
      `cancel_freed`; Slice 1 only ever sets `none`)
- Add to `request_result`:
    - `int32_t request_id              = <copied from seq_state>`
    - `int32_t admitted_at_iter        = -1`
    - `int32_t reused_seq_id           = -1`
    - `int32_t previous_request_id     = -1`
    - `admission_source                = none`
- `reused_seq_id` lives only on `request_result` (it is the
  field consumers read), not on `seq_state`. The engine has
  enough information from `seq_state.request_id` plus the
  current `seq_id` index to populate `reused_seq_id` at the
  fulfillment boundary in Slice 3.
- Add an engine-level `free_due_to_cancel` placeholder data
  structure (e.g. a `std::deque<int32_t>` member of the engine
  object). In Slice 1 it must remain **empty and unused**. The
  cancellation path is **not** modified in Slice 1: it does
  not push into `free_due_to_cancel`, and nothing pops from
  it. The placeholder exists only to prepare the data model
  for Slice 3 wiring.
- Engine writes `admitted_at_iter = -1`, `reused_seq_id = -1`,
  `previous_request_id = -1`, and `admission_source = none` on
  every completion path AND every cancellation path (the
  cancel path stays unchanged in semantics; only the new
  fields are populated with their defaults).
- Main validates from `request_result` snapshots that every
  result has:
    - `admitted_at_iter == -1`
    - `reused_seq_id == -1`
    - `previous_request_id == -1`
    - `admission_source == none`
    - `request_id == seq_id` (no admission, no slot reuse, so
      `request_id` and `seq_id` stay 1:1 in this slice)
- Engine-side validation also checks:
    - `free_due_to_cancel` is empty at end of every iteration
      and at end of run (Slice 1 must not append to it)
- Cancel Slice 4 metrics block unchanged.
- All Cancel Slice 4 / Slice-3 / Slice-4 / Slice-5 gates still
  pass on the 99-seq / `{8,64,256}` shape.
- No CLI flags. No deterministic admission. No traces. No
  waiting queue plumbing yet.
- Final stdout line: `HPX_CB_ADMIT_STEP1: PASS` /
  `FAIL: <reason>`.

### Live Admission Slice 2 — waiting queue, no admission yet

- Add CLI:
    - `--n-active <int>` (default 99). When set < n_seqs,
      defines the number of original active requests that get
      admitted at `t = 0`.
    - `--n-waiting <int>` (default 0). When > 0, defines the
      number of waiting requests queued at construction time.
    - `--waiting-budget <int>` (default 64). Decode budget for
      every waiting request.
- Main builds the waiting queue at construction time. Each
  waiting request gets a `request_id ∈ [n_active, n_active +
  n_waiting)`. Their decode budgets are uniform per
  `--waiting-budget`. Waiting requests are NOT bound to any
  `seq_id`.
- **No futures are created for waiting requests in this slice.**
  Main creates only `n_active` per-request promises. The
  futures vector handed to `hpx::wait_all` has length
  `n_active`. This proves that `wait_all` cannot deadlock on
  unfulfilled waiting futures because there are none.
- Engine receives the waiting queue (read-only handle) but does
  NOT consume from it. Slice 2 only validates that the queue is
  visible to the engine and that all `n_active` futures still
  resolve cleanly (mirror of Cancel Slice 4 outcome on the
  reduced active set).
- Smoke shape for Slice 2:
    - `--n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64`
    - same cancel plan (`1,4,7,2,5,8 cancel_after=16`)
    - 93 active requests prefill, 6 cancelled at iter 16, 87
      complete normally; 6 waiting requests stay queued and
      have no future
- Main validates from snapshots:
    - exactly `n_active` results received
    - 87 with `status == completed`, 6 with `status == cancelled`
    - all 93 have `admitted_at_iter == -1`, `reused_seq_id == -1`
    - the waiting queue at engine end has `n_waiting` entries
      still present (engine did not consume any)
- Cancel Slice 4 metrics block unchanged. Add a single new line:
  `queued_count = <n>` (descriptive only).
- Final stdout line: `HPX_CB_ADMIT_STEP2: PASS` /
  `FAIL: <reason>`.

### Live Admission Slice 3 — admit and complete

- Engine implements the admission boundary check at the top of
  each decode iter, immediately after the cancellation
  observation pass and before building `active_idx`:

  ```text
  for each freed seq_id (KV verified empty):
      if waiting_queue is empty: break
      pop FIFO head w from waiting_queue
      bind w.request_id to this seq_id
      record seqs[seq_id].request_id = w.request_id
      record seqs[seq_id].admitted_at_iter = iter
      record seqs[seq_id].previous_request_id = <prior owner>
      create promises_[seq_id] = hpx::promise<request_result>
      hand its future out via the shared admission-future
      collection (main consumes it)
      add prefill rows for w.prompt to the current batch
  ```

- Main, after `engine_fut.get()` returns, retrieves any
  admitted-request futures from the shared collection and
  appends them to the wait list. (Equivalent: main owns one
  consolidated futures vector that grows as admission events
  fire; in the simplest implementation main uses a second
  `hpx::wait_all` over admitted-only futures after the engine
  task ends. Both shapes preserve the no-deadlock invariant
  because the engine fulfills every admitted promise before
  returning.)
- Engine fulfills each admitted promise with
  `status = completed` after the same `clear_and_check` →
  `fulfill_promise` ordering used today.
- All §6 correctness gates pass.
- No new traces yet (Slice 4 adds them). No new metrics yet
  beyond `queued_count` from Slice 2 (Slice 4 adds the rest).
- Final stdout line: `HPX_CB_ADMIT_STEP3: PASS` /
  `FAIL: <reason>`.

### Live Admission Slice 4 — traces, metrics, results.md closeout

- Wire the admission trace events from §7 behind the existing
  `LLAMA_HPX_CB_TRACE=1` gate.
- Extend the metrics block with the admission metrics from §8
  (`queued_count` already present; add `admitted_count`,
  `admission_iter_set`, `reused_seq_id_count`,
  `reused_seq_id_set`, `admitted_ttc_ms[budget=*]`,
  `waiting_queue_depth_after_admission_per_iter`).
- Update `tools/hpx-continuous-batch-gate/results.md` with a
  *Live admission results* section: smoke shape, expected
  outcome, observed vs expected anchors, trace counts, and the
  cross-shape caveat for surviving-active hashes vs the
  cancellation-only run.
- Update `tools/hpx-continuous-batch-gate/README.md` with the
  Live Admission Slice 4 status entry and the new CLI flags.
- All §6 correctness gates pass.
- Trace gating remains: with `LLAMA_HPX_CB_TRACE` unset,
  compact runs emit zero `event=` lines.
- Final stdout line: `HPX_CB_ADMIT_STEP4: PASS` /
  `FAIL: <reason>`.

### Live Admission Slice 5 — completion-freed-slot admission

Slice 5 adds the **second** admission source on top of the
Slice 3 cancel-freed-slot path: naturally completed slots can
also be reused by waiting requests. The data model from Slice 1
is extended (one new enum value), the engine grows a second
deque with a deliberate **demand gate**, and the admission loop
gains a **source priority** ordering. No new trace event names
and no new event counts — only payload enrichment.

CLI:

- New default-OFF flag `--reuse-completed`. With the flag OFF
  the engine never reads or writes `free_due_to_completion`,
  no `completion_freed` admissions occur, and
  `completion_freed_pool_size_at_run_end` is `0`. Slice 3 / 4
  semantics (Slice 3 cancel-freed mapping byte-for-byte: same
  `reused_seq_id_set = {1,2,4,5,7,8}`, same `admission_iter_set
  = {17}`, same admitted-budget-64 hash `0x3b15a0474dfe11be`)
  are preserved. The label advances to
  `HPX_CB_ADMIT_STEP5` and the per-iter audit line is renamed
  `admit_step5:` (with new fields appended); regression
  evidence between Slice 4 and Slice 5 default-OFF runs is
  therefore compared **semantically**, not byte-for-byte.
- `--cancel-plan none` (existing sentinel from Cancel Slice 2)
  is used to make the no-cancellation smoke explicit; the
  smoke does not rely on an implicit empty plan.

Data model extension (`admission_source` enum):

```text
admission_source ::= none | cancel_freed | completion_freed
```

`completion_freed` joins the existing values. It is carried
through `seq_state`, `request_result`, and the trace payloads
(see below). `previous_request_id == reused_seq_id` is the
Slice-1 invariant on the original-active set (since prior
owners have `request_id == seq_id`), and it continues to hold
for both `cancel_freed` and `completion_freed` admissions.

New engine deque and demand gate:

- `std::deque<int32_t> free_due_to_completion_` lives on the
  engine alongside `free_due_to_cancel_`.
- Push site: inside `finalize_and_fulfill`, **after**
  `clear_and_check` succeeds (so the slot is truly KV-empty at
  queue-entry time), under two conjunctive conditions:
  1. `--reuse-completed` is ON.
  2. `waiting_queue_consumable_` is non-empty.
  This is the **demand gate**: a naturally completed slot is
  pooled only while there is at least one outstanding waiting
  request that has not yet been admitted. Once the waiting
  queue is drained, subsequent natural completions stop pooling
  for the remainder of the run. This keeps the residual pool
  size a tight invariant — the test focuses on the *first
  freed wave*, not on a persistent free-slot inventory across
  the whole run.
- Pop site: a second loop inside the existing admission
  boundary at the top of each decode iter, after the cancel
  queue is drained. No snapshot is needed for the completion
  queue (pushes happen at end of iter, pops happen at top of
  the next iter — entries are settled).
- Reset site: `run_body()` clears `free_due_to_completion_` at
  the start of every repeat, so residuals never leak across
  repeats. The residual size at engine end is snapshotted into
  `engine_result.completion_freed_pool_size_at_run_end` for
  reporting and gating.

Admission source priority:

```text
1. free_due_to_cancel       (preserves Slice 3 mapping)
2. free_due_to_completion   (Slice 5)
```

Both queues are drained in this order on every admission
boundary. Slice 5 itself does **not** exercise a combined
cancel + completion run; a mixed-source smoke is a future
slice and is out of scope here. The priority ordering exists
so the future mixed-source slice can land without changing
the cancel-freed mapping; Slice 5 only proves it on the
single-source paths.

Trace event payloads (no new event names):

The five existing admission trace events from Slice 4 gain an
explicit `admission_source=<value>` key:

- `seq_reused seq_id=<id> previous_owner=<request_id>
  new_owner=<request_id> iter=<int>
  admission_source=<cancel_freed|completion_freed>`
- `request_admitted_live request=<id> reused_seq_id=<id>
  iter=<int> admission_source=<value>`
- `admitted_prefilled request=<id> seq_id=<id>
  first_token=<id> admission_source=<value>`
- `admitted_decode_row request=<id> seq_id=<id> iter=<int>
  pos=<int> admission_source=<value>`
- `admitted_complete request=<id> seq_id=<id> budget=<int>
  done_iter=<int> hash=<hex> admission_source=<value>`

Event *counts* are unchanged in shape; only payloads become
richer. Slice 3 cancel-freed-slot trace captures now emit
`admission_source=cancel_freed` on these five events, so the
Slice 4 trace-on capture must be compared semantically.
Trace-off captures stay byte-quiet (zero
`[hpx-cb-gate] event=` lines).

Slice 5 smoke shape (deterministic):

```text
n_seq_max       = 99
n_active        = 90
n_waiting       = 9
waiting_budget  = 8
active budgets  = round-robin {8, 64, 256}
cancel_plan     = none (empty)
```

This shape is deliberately distinct from the Slice 3 / 4
shape (which uses `n_active=93`, `n_waiting=6`,
`waiting_budget=64`, non-empty `cancel_plan`), so both smokes
coexist and there is no mapping collision.

Admission mapping (FIFO over the first 9 budget-8 round-robin
slots, in ascending `seq_id` order):

```text
request 90 -> seq  0
request 91 -> seq  3
request 92 -> seq  6
request 93 -> seq  9
request 94 -> seq 12
request 95 -> seq 15
request 96 -> seq 18
request 97 -> seq 21
request 98 -> seq 24
```

Every admitted request: `budget=8`, `admitted_at_iter=8`,
`done_iter=15`, `pos_max_at_clear = n_prompt + budget - 2 =
6 + 8 - 2 = 12`, `admission_src=completion_freed`. The 21
remaining budget-8 slots (`{27, 30, …, 87}`) remain pooled in
`free_due_to_completion` and KV-empty for the rest of the run.
Later natural completions (budget-64 at iter 63, budget-256 at
iter 255) do **not** push to the pool because the waiting
queue is empty by then (demand gate).

Hash-gating policy:

- `admission_src=none, budget=8` partition runs entirely
  before admission iter 8, so its canonical anchor
  `0x0619d4d1900c2365` is **gated strictly** (same as in
  Slice 4).
- `admission_src=none, budget=64` and
  `admission_src=none, budget=256` surviving partitions cross
  iter 8's mixed prefill+decode batch shape. They are gated
  only by within-run uniqueness and `--repeat 2` determinism.
  If they happen to match canonical anchors `0x88a4dc75a31d4325`
  and `0x8a1a3bd01360aada`, that is recorded as observed,
  **not** promoted to a strict gate — to avoid baking in
  accidental backend-specific hash stability.
- `admission_src=completion_freed, budget=8` partition:
  within-run uniqueness gated; the observed hash is recorded
  descriptively in `results.md`, not canonical-gated.

Correctness gates (in addition to all §6 gates):

- `admission_iter_set == {8}`
- `reused_seq_id_set == {0,3,6,9,12,15,18,21,24}`
- `admission_src == completion_freed` for every admitted
  result
- `admitted_count == 9`
- `admitted_prefill_events == 9`
- `completion_freed_pool_size_at_run_end == 21`
- `free_due_to_completion` size at end of iter 7 is `30`
  (implicit via residual `21 = 30 − 9` gate)
- `free_due_to_completion` size after admission at top of
  iter 8 is `21` (implicit via residual gate)
- `free_due_to_completion` remains `21` after the budget-64
  and budget-256 completion waves (demand-gate guarantee;
  residual `21` at run end)
- `free_due_to_cancel` remains empty for the whole Slice 5
  smoke (no cancel admissions)
- with `--reuse-completed` OFF: zero `completion_freed`
  admissions AND `completion_freed_pool_size_at_run_end == 0`
- cross-source confusion fail-closed: cancel-freed
  `reused_seq_id` must be in `cancel_plan`; completion-freed
  `reused_seq_id` must **not** be
- residual KV empty for all `n_seq_max` slots (existing
  Slice 1 sweep covers the 21 pooled residual slots too)
- `--repeat 2` determinism includes
  `(admission_src, admitted_at_iter, reused_seq_id,
  previous_request_id)`

Out of scope for Slice 5: a single run that exercises
cancel + completion admission together. That belongs to a
later mixed-source slice. Slice 5 keeps the failure-mode
space small by exercising exactly one source at a time.

- Update `tools/hpx-continuous-batch-gate/results.md` with a
  *Live Admission Slice 5 results — completion-freed slot
  reuse* section: smoke shape, capture commands, final lines,
  audit line, mapping, partition hashes, trace counts, and
  repeat determinism.
- Update `tools/hpx-continuous-batch-gate/README.md` with the
  Live Admission Slice 5 status entry and the new
  `--reuse-completed` CLI flag.
- All §6 correctness gates pass.
- Trace gating remains: with `LLAMA_HPX_CB_TRACE` unset, all
  Slice 5 captures emit zero `event=` lines.
- Final stdout line: `HPX_CB_ADMIT_STEP5: PASS` /
  `FAIL: <reason>`.

After Live Admission Slice 5 the prototype demonstrates both
the cancel-freed and the completion-freed admission paths.
Anything beyond that (mixed cancel + completion admission in
one run, arrival schedules with non-zero start times, priority
on the waiting queue, multi-cycle slot reuse, async-submitted
waiting requests, streaming partial responses) is a separate
design doc.

### Live Admission Slice 6 — async external arrivals

Adds **deterministic async external arrivals** on top of the
existing Slice 3 cancel-freed admission flow. A single
scripted HPX submitter task pushes external arrivals into an
engine-owned inbox under a **release + ack barrier**; the
engine drains the inbox at the top of the next decode iter
and admits arrivals via the existing cancel-freed FIFO path.

Path under test:

```text
external HPX submitter task
  -> hpx::future<void> release_future           (set by engine at end of iter K)
  -> engine::submit(arrival_msg)                (acquires hpx::spinlock, push)
  -> hpx::promise<void> ack_promise.set_value() (submitter)
  -> engine resumes after ack_future.get()
  -> drain_external_inbox(iter)                 (engine task only, top of iter K+1)
  -> waiting_queue_consumable_                  (arrival_source = external)
  -> existing cancel_freed admission boundary   (cancel_after + 1)
  -> external future fulfilled with completed status
```

For the smoke shape with `--external-release-iter 8` and
`--cancel-after 16`:

```text
release_iter  = 8
drain_iter    = 9
admit_iter    = 17
```

**HPX-native constraints (hard rules):**

- No `std::thread`. No `std::condition_variable`. No
  `std::this_thread::sleep_for`. No wall-clock timing. No new
  `std::mutex`. The only new lock is an **`hpx::spinlock`**
  (`inbox_mtx_`) guarding the engine inbox. (Note: the public
  alias exported by the installed HPX is `hpx::spinlock`, not
  `hpx::lcos::local::spinlock` — the implementation uses
  `hpx::spinlock`.)
- Coordination is `hpx::promise<void>` / `hpx::future<void>`
  only. The engine sets a release promise at end of iter K and
  suspends on the matching ack future; the submitter awaits the
  release future, pushes its K-block via `engine::submit()`,
  and sets the ack promise. There is no "same-iteration
  release/drain" assumption — drain happens at the top of
  iter K+1 with no other synchronization.
- The submitter helper body **must not call any `llama_*` API**.
  Its allowed surface is: wait on `release_future`, construct
  `arrival_msg`, move per-arrival promises into it, call
  `engine::submit()`, and set `ack_promise`. A grep gate at
  review time scopes this to the submitter helper body, not
  the whole TU.
- `engine::submit()` **must not mutate `engine_result`** —
  Correction 1 of the Slice 6 plan. It only acquires the
  inbox spinlock and pushes the message; counters are bumped
  by the engine when it drains. There is no
  `external_arrival_count` field on `engine_result`.
- Only the engine task touches llama.cpp state
  (`llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`).
  `engine::submit()` and the scripted submitter never call
  any `llama_*` API.

**New types (engine-internal data model):**

```text
arrival_source         ::= preloaded | external
arrival_msg            { request_id, decode_budget,
                         hpx::promise<request_result> promise,
                         arrival_source src }
external_release_handle{ hpx::future<void> release_future,
                         hpx::promise<void> ack_promise }
scripted_arrival       { request_id, decode_budget,
                         release_iter }
```

`arrival_src` is added to `seq_state`, `request_result`,
and `waiting_request` and is copied through
`fulfill_promise()` onto every snapshot. Default for every
initially-bound active seq is `preloaded`. Admission
overwrites `seq.arrival_src` from the bound
`waiting_request::src`.

**Engine API additions:**

```text
engine::submit(arrival_msg msg)
engine::register_external_release_iter(int32_t K) -> external_release_handle
```

Both are called from the submitter task only, and both are
called BEFORE / ALONGSIDE `engine::run()` (registration must
happen before the engine task is scheduled so the engine
sees a ready ack future to wait on at end of iter K).

**Engine-internal additions:**

```text
hpx::spinlock                                inbox_mtx_;
std::deque<arrival_msg>                      inbox_;
std::unordered_map<int32_t, hpx::promise<request_result>> external_promises_;
std::vector<hpx::promise<void>>              iter_release_promises_;
std::vector<hpx::future<void>>               submitter_ack_futures_;
std::set<int32_t>                            iter_release_set_;
int32_t                                      max_decode_iters_;
```

`drain_external_inbox(iter)` is engine-only; it swaps
`inbox_` out under the spinlock (O(1) critical section),
then iterates the local deque without the lock: stashes
each `arrival_msg::promise` in `external_promises_` keyed by
`request_id`, pushes a `waiting_request{src=external}` onto
`waiting_queue_consumable_`, bumps
`result_.arrival_drained_count`, and latches
`first_external_drain_iter` on first drain.

`admit_one` (already in Slice 3) is extended so that on
binding an external waiter it moves the matching promise out
of `external_promises_` into `promises_[reuse_seq]` (the
submitter already holds the future, so no admitted-futures
push happens for external arrivals).

**Engine-result counters (engine-side only):**

```text
arrival_drained_count
external_admitted_count
first_external_drain_iter      (-1 when no external arrivals)
iter_release_fired_set         (engine-side observation order)
submitter_ack_set              (engine-side observation order)
```

`external_arrival_count` is intentionally absent (Correction
1; submission-side counters are not engine state).

**Trace events (gated on `LLAMA_HPX_CB_TRACE=1`):**

```text
request_submitted_external request=<id> budget=<int> arrival_source=external
arrival_drained            request=<id> iter=<int> budget=<int>
iter_release_fired         iter=<int>
submitter_ack_observed     iter=<int>
```

Existing events extended with `arrival_source=<preloaded|external>`:

```text
request_queued
request_admitted_live
```

Trace-on smoke event counts per repeat:

```text
request_submitted_external = 6
arrival_drained            = 6
iter_release_fired         = 1
submitter_ack_observed     = 1
request_admitted_live with arrival_source=external = 6
```

**Smoke shape:**

```text
n_seq_max               = 99
n_active                = 93
n_waiting               = 0
n_external_arrivals     = 6
external_arrival_budget = 64
external_release_iter   = 8
cancel_plan             = 1,4,7,2,5,8
cancel_after            = 16
--reuse-completed       = OFF
```

External request IDs: `93, 94, 95, 96, 97, 98`
(`n_active + n_waiting + i` for `i = 0..5`).

**Expected mapping** (FIFO over sorted cancel-plan):

```text
request 93 -> seq 1
request 94 -> seq 2
request 95 -> seq 4
request 96 -> seq 5
request 97 -> seq 7
request 98 -> seq 8
```

Every external admitted result: `arrival_src = external`,
`admission_src = cancel_freed`, `admitted_at_iter = 17`,
`decode_budget = 64`, `done_iter = 17 + 64 - 1 = 80`,
`pos_max_at_clear = n_prompt_tokens + 64 - 2`, admitted
budget-64 hash `= 0x3b15a0474dfe11be` (same anchor as the
Slice 3 cancel-freed budget-64 admission, proving the
async-external surface routes byte-identically through the
existing admission and decode paths).

**Required Slice 6 source-side gates** (in addition to all
Slice 1–5 gates):

- `er.arrival_drained_count == args.n_external_arrivals`
- `er.external_admitted_count == args.n_external_arrivals`
- `er.first_external_drain_iter == args.external_release_iter + 1`
- `er.iter_release_fired_set == {args.external_release_iter}`
- `er.submitter_ack_set == {args.external_release_iter}`
- request→seq mapping per sorted cancel-plan FIFO
- `arrival_src == external` on every admitted result with
  `request_id ∈ [n_active, n_active + n_external_arrivals)`;
  `admission_src == cancel_freed` on each; `admitted_at_iter
  == cancel_after + 1` on each.
- `arrival_src == preloaded` on every NON-external result
  (original actives, preloaded waiters).
- admitted budget-64 hash anchor for the slice 6 smoke
  (`0x3b15a0474dfe11be`).
- every external future is fulfilled with
  `status = completed`; `promises_fulfilled == n_active +
  admitted_count` (preloaded admissions + external
  admissions).
- inbox is empty at engine end, `external_promises_` is empty
  at engine end. Both are asserted BEFORE the residual-KV
  sweep; either being non-empty fails closed with an explicit
  reason.
- residual KV empty for all `n_seq_max` slots (existing
  Slice 1 sweep).
- with `--n-external-arrivals == 0` (the inert path) every
  Slice 6 counter and set is at its default; no submitter
  task is spawned and no release/ack handle is registered.

**Required Slice 6 HPX-native review gates:**

- the scripted submitter helper body does not call any
  `llama_*` API
- no `std::thread`
- no `std::condition_variable`
- no `std::this_thread::sleep_for`
- no new `std::mutex`
- the only new lock is `hpx::spinlock inbox_mtx_`

**Quietness / determinism:**

- trace-off runs emit zero `[hpx-cb-gate] event=` lines on
  stderr
- `--repeat 2` is deterministic, including the Slice 6
  counters and the per-external-result snapshot tuple

**CLI flags:**

```text
--n-external-arrivals <int>      default: 0
--external-arrival-budget <int>  default: 64
--external-release-iter <int>    default: 0
```

With `--n-external-arrivals == 0` the binary's run is
byte-equivalent to Slice 5 with `--reuse-completed` OFF:
no submitter task, no release handle, no inbox traffic, no
admission via the external path.

Closeout evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Live Admission Slice 6 results — async external arrivals*
section.

Final stdout line: `HPX_CB_ADMIT_STEP6: PASS` /
`FAIL: <reason>`.

After Live Admission Slice 6 the prototype additionally
demonstrates a deterministic, HPX-native async-arrival
surface. Out-of-scope items unchanged: multi-K release
schedules, wall-clock arrivals, completion-freed external
admission, streaming.

### Live Admission Slice 7 — mixed-source admission priority

Goal: prove the engine's **source-priority** rule end-to-end.
When `free_due_to_cancel_` and `free_due_to_completion_` are
both non-empty when an admission step runs, admission must
drain `cancel_freed` first, then `completion_freed`. Earlier
slices proved each admission source in isolation (Slice 3
cancel-only; Slice 5 completion-only; Slice 6 cancel-only
with external arrivals) but never put both pools on the same
admission boundary in one deterministic run.

**Smoke shape (forces the mixed boundary):**

```text
n_seq_max               = 99
n_active                = 84
n_waiting               = 9        (preloaded; req_ids 84..92, budget 8)
waiting_budget          = 8
n_external_arrivals     = 6        (async;     req_ids 93..98, budget 64)
external_arrival_budget = 64
external_release_iter   = 16
--reuse-completed       = ON
cancel_plan             = 1,4,7,2,5,8
cancel_after            = 16
active budgets          = round-robin {8, 64, 256}
```

`n_active + n_waiting + n_external_arrivals = 99 = n_seqs`.
cancel_plan ⊂ budget-64 ∪ budget-256, all in `[0, 84)`.

**Two-phase admission timeline:**

```text
iter 0       prefill of all 84 actives
iter 1..7    budget-8 actives decode toward done_iter=7
iter 7       28 budget-8 actives complete naturally; demand gate
             pushes ALL 28 onto free_due_to_completion_ (waiting
             queue still has 9 entries — gate stays open)
iter 8       admission boundary #1 (Phase 1)
             cancel_eligible_snapshot = 0  (cancel hasn't fired)
             completion-freed pass: 9 admissions, FIFO ascending
                 req 84 -> seq  0
                 req 85 -> seq  3
                 req 86 -> seq  6
                 req 87 -> seq  9
                 req 88 -> seq 12
                 req 89 -> seq 15
                 req 90 -> seq 18
                 req 91 -> seq 21
                 req 92 -> seq 24
             free_due_to_completion_ residual = 19  (28 − 9)
             waiting_queue_consumable_ empty
iter 9..15   admitted budget-8 admissions decode
iter 15      admitted budget-8 admissions complete; demand gate
             stops (waiting queue empty) — pool stays at 19
iter 16      cancel observation fires for {1,2,4,5,7,8}
             (seq_id ascending) — pushes onto free_due_to_cancel_
             admission_eligible_snapshot taken BEFORE cancel pass = 0
             (so no iter-16 admissions of cancel-freed)
             end of iter 16: release+ack barrier — engine sets
             release[16]; submitter pushes 6 external arrivals; ack;
             engine resumes
iter 17      admission boundary #2 (Phase 2) — MIXED-SOURCE
             drain_external_inbox(17) pushes 6 onto waiting_queue
             Both pools non-empty:
                 free_due_to_cancel_     = [1,2,4,5,7,8]   (6)
                 free_due_to_completion_ = [27,30,…,81]    (19)
                 waiting_queue           = [93,94,95,96,97,98]
             Source-priority rule:
                 cancel pass     -> admits 6
                     req 93 -> seq 1
                     req 94 -> seq 2
                     req 95 -> seq 4
                     req 96 -> seq 5
                     req 97 -> seq 7
                     req 98 -> seq 8
                 completion pass -> no-op  (waiting queue empty)
             free_due_to_completion_ residual STAYS AT 19
iter 18..80  admitted budget-64 (externals) decode
iter 18..255 surviving budget-64 and budget-256 actives complete;
             demand gate stops their pool pushes (waiting queue empty)
```

**Mappings (per phase, per source):**

```text
Phase 1  iter  8  completion_freed  preloaded
    req 84 -> seq  0
    req 85 -> seq  3
    req 86 -> seq  6
    req 87 -> seq  9
    req 88 -> seq 12
    req 89 -> seq 15
    req 90 -> seq 18
    req 91 -> seq 21
    req 92 -> seq 24

Phase 2  iter 17  cancel_freed  external
    req 93 -> seq 1
    req 94 -> seq 2
    req 95 -> seq 4
    req 96 -> seq 5
    req 97 -> seq 7
    req 98 -> seq 8
```

**Per-iter source-priority gates (slice7_strict):**

```text
G7-S1   admission_iter_set == {min_active_budget, cancel_after+1}
        == {8, 17}

G7-S2   for every admitted result with admitted_at_iter == 8:
            admission_src == completion_freed
            arrival_src   == preloaded

G7-S3   for every admitted result with admitted_at_iter == 17:
            admission_src == cancel_freed
            arrival_src   == external

G7-S4   Phase 1 mapping (FIFO over the first n_waiting min-budget
        actives in seq_id ascending order):
            request n_active + i -> min_budget_slots[i]

G7-S5   Phase 2 mapping (FIFO over sorted cancel_plan):
            request n_active + n_waiting + i -> sorted(cancel_plan)[i]

G7-S6   No result with admitted_at_iter == 17 has
        reused_seq_id in the completion-freed residual set
            min_budget_slots[n_waiting:]
        — proves cancel-freed priority was respected.

G7-S7   completion_freed_pool_size_at_run_end
            == first_wave_size − n_waiting
            == 28 − 9 == 19
```

**Per-`(admitted_at_iter, admission_src)` reused_seq_id
ordering gate** (replaces the pre-Slice-7 global "strictly
ascending" gate):

```text
G7-O1   reused_seq_id_set.size() == admitted_count
        (carried over from Slice 3)

G7-O2   No duplicate seq_id appears across the entire
        reused_seq_id_set (was implied by global ascending
        in Slice 3..6).

G7-O3   For each pair (K, src) with K in admission_iter_set
        and src in {cancel_freed, completion_freed}, the
        sub-sequence of reused_seq_ids drawn from results
        with admitted_at_iter == K AND admission_src == src,
        ordered by request_id ascending (waiting-queue FIFO),
        is strictly ascending.
```

G7-O3 holds for every prior slice (Slice 3 single iter,
single source — global ascending implies per-(K, src)
ascending; Slice 5 same; Slice 6 single iter, single source)
and now also for the Slice 7 mixed shape, where the engine
emits two strictly-ascending sub-sequences `{0,3,…,24}` at
iter 8 and `{1,2,4,5,7,8}` at iter 17.

**Residual completion-freed pool proof:**

The residual gate G7-S7 only holds when **no iter-17 admission
consumed from the completion pool**, because the engine's pool
push events are demand-gated and only fire while
`waiting_queue_consumable_` is non-empty. Concretely:

- 28 budget-8 actives complete at iter 7; all 28 push to the
  pool (waiting queue size 9 throughout iter 7's finalize phase).
- 9 are consumed at iter 8.
- The 9 admitted-budget-8 reqs complete at iter 15; waiting
  queue is empty by then — demand gate stops pushes.
- 6 cancel-freed slots are admitted at iter 17 to the 6
  external arrivals; the completion pool is not touched.
- Surviving budget-64 actives complete at iter 63 and
  surviving budget-256 actives at iter 255; waiting queue is
  still empty — demand gate stops pushes.

So `completion_freed_pool_size_at_run_end == 28 − 9 == 19`
exactly, and the 19 residual slots stay KV-empty under the
existing `n_seq_max`-wide residual sweep.

**No new CLI flags / trace event names / HPX primitives.**

- All Slice 7 behavior is exercised through the existing
  `--n-waiting / --waiting-budget / --n-external-arrivals /
  --external-arrival-budget / --external-release-iter /
  --reuse-completed / --cancel-plan / --cancel-after` surface.
- The Slice 4–6 trace surface already partitions by
  `admission_source=` and `arrival_source=` on
  `request_admitted_live`, `admitted_complete`,
  `admitted_decode_row`, `admitted_prefilled`, `seq_reused`,
  and `request_queued`. Slice 7 does not add or rename any
  trace event.
- The Slice 6 `hpx::spinlock` inbox and release+ack barrier
  are reused unchanged. No new `std::thread`, no
  `std::condition_variable`, no wall-clock sleeps, no new
  `std::mutex`. The HPX-native and llama-cpp-ownership
  invariants from Slice 6 carry over verbatim.

**No performance claim.** Slice 7 is correctness-only — it
proves an ordering invariant of the admission loop, not a
throughput property. Observed admitted hashes are recorded
descriptively and gated only on within-partition uniqueness
and `--repeat 2` determinism (the same policy used in Slice 5
for the completion-freed admitted budget-8 partition).

**New audit line** (`admit_step7:`) on stdout, in addition to
the existing `admit_step5:` and `admit_step6:` lines:

```text
iter[r] admit_step7: phase1@iter=8 completion_freed=9
                    phase2@iter=17 cancel_freed=6
                    pool_residual=19
                    admission_iter_set={8,17}
                    first_external_drain_iter=17
```

Closeout evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Live Admission Slice 7 results — mixed-source admission
priority* section.

Final stdout line: `HPX_CB_ADMIT_STEP7: PASS` /
`FAIL: <reason>`.

After Live Admission Slice 7 the prototype additionally
demonstrates the source-priority rule under a deterministic
two-phase smoke that puts both pools on the same admission
boundary. Out-of-scope items unchanged: multi-K release
schedules, wall-clock arrivals, completion-freed external
admission, streaming, per-request priority queues, multi-
cycle slot reuse, multiple engine tasks.

---

## 10. Out of scope (every admission slice)

Explicitly excluded from this design and from the slices that
implement it:

- HTTP / network arrival path
- async submission of waiting requests from a thread other than
  main (the construction-time waiting queue is the only
  admission source in v1)
- arrival schedules with per-request start times (all waiting
  requests are eligible at `t = 0` and admission is gated only
  by slot availability)
- priority on the waiting queue (FIFO only)
- preemption / requeue (an admitted request runs to completion
  or hits the existing cancellation path; it is never returned
  to the waiting queue)
- multi-cycle slot reuse (a `seq_id` is reused at most once per
  run; chaining waiting → admitted → completed → second
  waiting → second admitted is not exercised in v1)
- per-admitted-request prompts (every request shares the same
  prompt in v1, mirroring the earlier slices)
- streaming partial-result emission to a client
- multiple `llama_context` instances
- modifying `tools/server/`, `tools/serving-bench/`,
  `tools/multiseq-batch-gate/`
- introducing a named orchestration pool / resource partitioner
  (Slice 3b is still the right home for that work; admission
  itself does not require a second HPX task)
- performance claims (speedup, latency, throughput)
- interruption inside `llama_decode` (forbidden; admission is
  cooperative at iteration boundaries, exactly like
  cancellation)
- async external admission (v1 uses a construction-time queue
  only; a thread-safe queue, e.g. `hpx::lcos::local::channel`,
  can be added later without changing the engine loop or
  snapshot boundary)

---

## 11. Risks / open questions

1. **Mixed prefill+decode batches.** Iter 17's mixed batch
   (36 prefill rows + 56 decode rows) is the first time this
   prototype builds a heterogeneous batch. Risk: a subtle bug
   in `i_batch` accounting between the two row kinds. Mitigation:
   Slice 3 must validate `seq.i_batch` independently for every
   admitted seq (it points to the *last* prefill row for that
   seq) and for every surviving active seq (it points to the
   single decode row for that seq), and the post-`llama_decode`
   argmax loop must visit each row exactly once.

2. **`promises_` indexing.** `promises_` is currently keyed by
   `seq_id`; after admission two requests share a `seq_id` over
   time. Risk: double-fulfill of the prior owner or fulfill
   against a stale promise. Mitigation: Slice 3 replaces the
   slot's promise with a fresh `hpx::promise<request_result>`
   at admission time (prior owner already fulfilled and its
   future already moved to main's wait list). Storage shape
   (vector re-keyed at admission vs. map keyed by `request_id`)
   is deferred to Slice 3.

3. **Future delivery to main without leaks.** Admitted-request
   futures are created inside the engine task. Main needs to
   wait on them too. Risk: a race between the engine fulfilling
   an admitted promise and main moving its future onto the wait
   list. Mitigation: simplest pattern is for the engine to push
   futures into a shared `std::vector<hpx::future<request_result>>`
   protected by a `std::mutex`, and for main to consume the
   collection only AFTER `engine_fut.get()` returns. The engine
   has fulfilled every admitted promise by then, so every
   future in the collection is already ready; main's
   `hpx::wait_all` over them returns immediately. No race
   window. Document this ordering invariant in Slice 3.

4. **Slot-reuse KV-empty assertion failure.** If the prior
   owner's KV clear didn't take effect, reusing the slot would
   silently mix two requests' KV. Mitigation: the §3 assertion
   is fail-closed — admission aborts, the engine
   `set_exception`s unfulfilled promises, main observes
   `FAIL: <reason>`. v1 keeps assertion-only to make the
   failure loud.

5. **`request_admitted` event semantics.** Existing event
   keeps its original-active-only meaning; live admissions emit
   a distinct `request_admitted_live` event (see §7).

6. **`update_iterations` and `decode_calls` are descriptive,
   not gates.** They depend on whether a budget-256 seq
   survives; correctness rides on per-result invariants and
   residual-KV-empty only.

7. **Hash anchors after admission.** Iter 17 is a mixed batch,
   so surviving-active budget-64/256 hashes differ from the
   cancellation-only run. §4 / §6 commit only to within-run
   uniqueness for those budgets; the canonical anchor remains
   for budget-8 only.

8. **Determinism across `--repeat 2`.** Admission determinism
   relies on (a) the waiting queue being built once at
   construction and consumed FIFO, and (b) `free_due_to_cancel`
   being appended in deterministic ascending-`seq_id` order
   (cancellation already fires deterministically at iter 16).
   Slice 3 validates with a per-result determinism check on
   `(request_id, seq_id, admitted_at_iter, reused_seq_id,
   previous_request_id, admission_source, hash,
   generated_tokens, done_iter, pos_max_at_clear)`.

9. **CLI surface.** `--n-active`, `--n-waiting`,
   `--waiting-budget` are sufficient for v1; later flags should
   namespace under `--admit-*` / `--wait-*` if they grow.

---

## 12. Exact next implementation prompt

Below is the exact prompt to send back when Live Admission
Slice 1 is ready to implement. **Do not implement it from this
design note alone.**

```text
Proceed with Live Admission Slice 1 only.

Goal:
Add the live-admission data model to the HPX continuous-batching
gate without changing any runtime behavior.

Read first:
docs/hpx/continuous_batching_live_admission_design.md
docs/hpx/continuous_batching_cancellation_design.md
docs/hpx/continuous_batching_prototype_design.md
tools/hpx-continuous-batch-gate/README.md
tools/hpx-continuous-batch-gate/results.md
local/ahandoff.md
CLAUDE.md

Do not modify:
- tools/server/
- tools/serving-bench/
- tools/multiseq-batch-gate/

Do not add (Live Admission Slice 1):
- a waiting queue
- any admission behavior
- new CLI flags
- traces
- metrics changes
- HTTP / external admission
- orchestration pool / resource partitioner
- any push into free_due_to_cancel from the cancel path
  (the placeholder must stay empty in Slice 1)

Live Admission Slice 1 scope (full spec: §9 of this design):
- Add to seq_state: request_id (init = seq_id at construction),
  admitted_at_iter (-1), previous_request_id (-1),
  admission_source (none).
- Add to request_result: request_id (copied from seq_state),
  admitted_at_iter (-1), reused_seq_id (-1),
  previous_request_id (-1), admission_source (none).
- Add an engine-level free_due_to_cancel placeholder (e.g.
  std::deque<int32_t>). In Slice 1 it MUST remain empty and
  unused — the cancel path does not push, nothing pops.
- Engine writes the four new request_result fields with their
  defaults on every completion AND every cancellation path.
- Main validates every result has admitted_at_iter == -1,
  reused_seq_id == -1, previous_request_id == -1,
  admission_source == none, and request_id == seq_id.
- Engine validates free_due_to_cancel is empty at end of every
  iteration and at end of run.

Correctness gates:
All Cancel Slice 4 gates still pass on the 99-seq /
{8,64,256} shape:
- engine_task_count == 1
- futures_created == 99
- promises_fulfilled == 99
- futures_completed == 99
- 33 seqs per budget class
- budget 8 hash == 0x0619d4d1900c2365
- budget 64 has one unique hash within the run (completed only)
- budget 256 has one unique hash within the run (completed only)
- done_iter sets {7}/{63}/{255} for completed seqs
- pos_max_at_clear sets {12}/{68}/{260} for completed seqs
- per-seq KV clear succeeds before promise fulfillment
- residual KV empty at end (engine-side)
- every llama_decode returns 0
- --repeat 2 deterministic across token / hash / done_iter /
  pos_max_at_clear
- Cancel Slice 4: completed_count = 93, cancelled_count = 6,
  cancel_observed_iter_set = {16}, n_decoded_at_cancel_set = {16},
  wasted_decode_rows_after_cancel = 0

Live-Admission-Slice-1-specific gates:
- every result.admitted_at_iter == -1
- every result.reused_seq_id == -1
- every result.previous_request_id == -1
- every result.admission_source == none
- every result.request_id == result.seq_id
- engine free_due_to_cancel placeholder stays empty for the
  entire run (no push from the cancel path, no pop from
  anywhere)
- no new trace events fire (compact and trace runs both
  unchanged from Cancel Slice 4)
- metrics block unchanged

Final stdout line:
HPX_CB_ADMIT_STEP1: PASS
or
HPX_CB_ADMIT_STEP1: FAIL: <reason>

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
  > local/hpx_cb_admit_step1.stdout \
  2> local/hpx_cb_admit_step1.stderr

If that passes, also run repeat:

/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate \
  --model /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_admit_step1_repeat2.stdout \
  2> local/hpx_cb_admit_step1_repeat2.stderr

Stop and report:
- files modified
- build command
- run commands
- final stdout lines
- whether all Cancel Slice 4 gates still pass
- whether every result has admitted_at_iter == -1,
  reused_seq_id == -1, previous_request_id == -1,
  admission_source == none, and request_id == seq_id
- whether the engine's free_due_to_cancel placeholder stayed
  empty / unused for the entire run
- whether repeat 2 passed
- any HPX runtime warnings
- any llama_decode return-code issues

Do not proceed to Live Admission Slice 2 until I approve.
```
