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

Slices 1–5 plus Cancel Slices 1–4 are landed and PASS. The
current shape (do not change in this design):

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
- cooperative cancellation observed at iteration boundaries:
  cancelled seqs are removed from later decode batches, KV is
  cleared by the engine, futures are fulfilled with
  `status = cancelled` only after KV clear, non-cancelled seqs
  continue to completion, residual KV is empty for all seq_ids
  at end of run

Live admission must respect every one of those invariants.

---

## 1. Why live admission is next

Cancellation proved that an active request can leave the active
set:

```
active seq → cancel observed → KV cleared → future fulfilled
```

Live admission proves the symmetric other half of dynamic
continuous batching — that a waiting request can enter the
active set:

```
waiting request → admitted into freed capacity →
prefilled at iteration boundary → joins later decode batches →
future fulfilled
```

Until live admission lands, the prototype runs a fixed set of
requests admitted only at `t = 0`. Admission is the precondition
for every subsequent serving capability:

- arrival-driven workloads (mixed start times);
- priority scheduling (priority is a no-op axis until requests
  can saturate batch capacity over time);
- an orchestration pool / resource partitioner becoming
  meaningful (it earns its keep once HPX has ≥ 2 concurrent task
  kinds, e.g. engine + admission watcher);
- streaming partial-result emission as admitted requests stay in
  the system longer than the original cohort.

Admission also forces the prototype to make explicit something
the cancellation design only modeled implicitly — that a `seq_id`
is a *slot* that can be reused after KV clear, not a permanent
identity bound to a single request.

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

## 4. First smoke shape

Deterministic, self-contained, no external arrival schedule
needed. Reuses the existing cancellation plan to free the
admission slots.

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
                     requests to up to K freed seq_ids (K = min(waiting_queue
                     size, free_seq_id count)). The pairing is deterministic:
                     pop the FIFO head of the waiting queue, assign it to the
                     lowest-numbered free seq_id, repeat until either side
                     is empty. Each admitted request's prefill rows are
                     added to the SAME llama_batch as decode rows for
                     surviving active seqs in this iter (mixed batch).
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
  `reused_seq_id == -1`.
- **original cancelled**: `status == cancelled`,
  `n_decoded == 16`, `n_decoded_at_cancel == 16`,
  `cancel_observed_iter == 16`, `kv_cleared == true`,
  `pos_max_at_clear == n_prompt + 16 − 2 == 20`,
  `admitted_at_iter == -1`, `reused_seq_id == -1`.
- **admitted completed**: `status == completed`,
  `n_decoded == decode_budget` (= 64 in the smoke),
  `done_iter == admitted_at_iter + decode_budget − 1` (= 80 in
  the smoke), `pos_max_at_clear == n_prompt + decode_budget − 2`
  (= 68 for budget-64 admitted), `kv_cleared == true`,
  `cancel_observed_iter == -1`, `n_decoded_at_cancel == -1`,
  `admitted_at_iter == 17` (in the smoke), `reused_seq_id ∈
  {1, 2, 4, 5, 7, 8}` and is unique among admitted results.

`admitted_at_iter` and `reused_seq_id` are new
`request_result` fields; their default values for non-admitted
results are both `-1`.

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
- `waiting_queue_depth_per_iter` — distribution of queue depth
  observed at the top of each decode iteration (p50 / p95 / max).
  For the smoke this is `6` for iters 0..16 and `0` for iters
  17..255.
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
- Add to `request_result`:
    - `int32_t request_id              = <copied from seq_state>`
    - `int32_t admitted_at_iter        = -1`
    - `int32_t reused_seq_id           = -1`
- Engine writes `admitted_at_iter = -1` and `reused_seq_id = -1`
  on every completion path (and continues to write
  `cancel_observed_iter == -1` etc. on completion as today).
- Main validates from `request_result` snapshots that every
  result has:
    - `admitted_at_iter == -1`
    - `reused_seq_id == -1`
    - `request_id == seq_id` (no admission, no slot reuse, so
      request_id and seq_id stay 1:1 in this slice)
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
  `waiting_queue_depth_per_iter`).
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

After Live Admission Slice 4 the prototype demonstrates the
other half of dynamic continuous batching. Anything beyond that
(arrival schedules with non-zero start times, priority on the
waiting queue, multi-cycle slot reuse, async-submitted waiting
requests, streaming partial responses) is a separate design doc.

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

Async external admission is *not* modeled in v1's data path.
Adding it later requires a thread-safe waiting queue (e.g.
`hpx::lcos::local::channel`) but does not change the engine
loop or the snapshot boundary.

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

2. **`promises_` indexing.** The current Slice-3 implementation
   indexes `promises_` by `seq_id`. After admission, two
   distinct requests share a `seq_id` over time. Risk: the
   admission slice accidentally fulfills the prior owner's
   future twice, or fulfills the new owner's future against a
   stale promise. Mitigation: Slice 3 must replace
   `promises_[seq_id]` with a fresh `hpx::promise<request_result>`
   at admission time (the prior owner's promise was already
   fulfilled and its future already moved to main's wait list).
   The data model in Slice 1 needs to anticipate this — the
   `promises_` storage may need to migrate from
   `std::vector<hpx::promise<request_result>>` keyed by
   `seq_id` to a `std::unordered_map<int32_t, hpx::promise<...>>`
   keyed by `request_id`, or a per-slot vector with
   construction-on-binding semantics. Pick the one that
   minimizes lifetime risk in Slice 3; defer the choice in
   Slice 1.

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

4. **Slot-reuse KV-empty assertion failure mode.** If for any
   reason the prior owner's KV clear didn't fully take effect
   (a llama.cpp bug, an engine logic error, or a future
   refactor that splits cancel-clear from cleanup), reusing the
   slot would silently mix two requests' KV state. Mitigation:
   the assertion in §3 is fail-closed; admission aborts and the
   engine drains the unfulfilled promises with `set_exception`,
   surfacing a clean `FAIL: <reason>` to main. A future Slice
   should consider running an explicit `llama_memory_seq_rm`
   immediately before binding (idempotent on an already-empty
   slot, defensive against a missed cleanup) but v1 keeps the
   assertion-only form to keep the failure mode loud.

5. **`request_admitted` event semantics.** The existing event
   fires for original active requests at engine start. Live
   admission introduces a second class of admissions. Risk: a
   trace reader expecting one event class. Mitigation: keep the
   existing event firing only for original active requests, and
   add `request_admitted_live` for admission-driven entries.
   Document the split in the admission slice's README and
   results.md.

6. **`update_iterations` and `decode_calls` cross-shape
   stability.** With surviving budget-256 active seqs running
   to iter 255, `update_iterations` stays at 255 even with
   admission. Risk: a future plan with no surviving budget-256
   seq would reduce the iteration count and break the anchor.
   Mitigation: do not use `update_iterations == 255` as a
   correctness gate; treat it as a descriptive metric.
   Correctness depends only on per-result invariants and the
   residual-KV-empty check.

7. **Hash anchors after admission.** Surviving-active
   budget-64 / budget-256 hashes will differ from the
   cancellation-only run because iter 17 is a mixed batch. This
   is expected and parallels the Cancel-Slice-2 caveat.
   Mitigation: the design (§4 caveat, §6 gate wording) commits
   only to the same-shape uniqueness gate (one unique hash per
   class within the run) for those budgets, and to the
   canonical anchor only for budget-8.

8. **Determinism across `--repeat 2`.** Determinism of the
   admission outcome (which `request_id` lands on which
   `seq_id`, at which iter, with which budget) depends on the
   FIFO+lowest-seq-id rule being deterministic. As long as the
   waiting queue is built once at construction time and not
   re-ordered, and the engine consumes it in declared order at
   each admission boundary, both repeats produce identical
   admission schedules. Slice 3 must validate this with a
   per-result determinism check on `(request_id, seq_id,
   admitted_at_iter, reused_seq_id, hash, generated_tokens,
   done_iter, pos_max_at_clear)`.

9. **CLI surface stability.** New flags `--n-active`,
   `--n-waiting`, `--waiting-budget` add a small surface. Risk:
   later interaction with priority/streaming flags. Mitigation:
   namespace future flags under `--admit-*` and `--wait-*`
   prefixes if they grow; for v1 the three flags above are
   sufficient.

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
docs/hpx/hpx_continuous_batching_cancellation_design.md
docs/hpx/hpx_continuous_batching_prototype_design.md
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

Live Admission Slice 1 scope:
- Add to engine-internal seq_state:
  - int32_t request_id          = -1   (== seq_id while no
                                         admission has happened;
                                         distinguishable later)
  - int32_t admitted_at_iter    = -1
  - int32_t previous_request_id = -1
- Add to request_result:
  - int32_t request_id          = <copied from seq_state>
  - int32_t admitted_at_iter    = -1
  - int32_t reused_seq_id       = -1
- Engine initializes seq_state.request_id = seq_id at
  construction (no admission yet, so request_id and seq_id are
  1:1).
- Engine writes admitted_at_iter = -1 and reused_seq_id = -1 on
  every completion path AND every cancellation path (the cancel
  path stays unchanged in semantics; only the new fields are
  populated with their default).
- Engine never admits in this slice — the new fields are
  present but unused except for the request_id-equals-seq_id
  default.
- Main validates from request_result snapshots that every
  result has:
  - admitted_at_iter == -1
  - reused_seq_id == -1
  - request_id == seq_id
  This proves the data model is plumbed without behavior change.

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
- every result.request_id == result.seq_id
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
  reused_seq_id == -1, and request_id == seq_id
- whether repeat 2 passed
- any HPX runtime warnings
- any llama_decode return-code issues

Do not proceed to Live Admission Slice 2 until I approve.
```
