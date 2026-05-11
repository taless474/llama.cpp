# hpx-continuous-batch-gate

The HPX continuous-batching prototype. Wraps the proven Step 5
multi-seq `llama_batch` primitive (see
`tools/multiseq-batch-gate/results.md`) with HPX-owned request and
slot orchestration.

This tool is a **correctness-first prototype**, not a benchmark. No
HPX-vs-std performance claim is made.

## Status

- **Slice 1 (done):** HPX runtime skeleton. Starts the HPX runtime,
  loads the model, creates one `llama_context`, prints the
  structural capacity fields, builds N metadata-only request
  objects, stops the HPX runtime, and exits with `HPX_CB_STEP1: PASS`.
- **Slice 2 (done):** an `engine` class owns `llama_context*`, the
  shared `llama_batch`, the per-seq state vector, and decode-loop
  state. `engine::run()` is the sole code path that touches
  `llama_context` / `llama_batch` / `llama_decode` /
  `llama_memory_seq_*` / `llama_get_logits_ith`, and is launched
  as exactly one HPX task per repeat iteration via `hpx::async`;
  `main()` waits on the returned future.
- **Slice 3 (done):** per-request HPX promise/future on top of the
  Slice-2 engine. One HPX promise/future per seq; futures are
  handed out before the engine task is scheduled; the engine
  fulfills each promise only after the seq reaches its budget,
  records `pos_max_at_clear`, has its KV cleared, and passes the
  cross-talk-against-still-active-siblings check. Main uses
  `hpx::wait_all` on the per-request futures, calls
  `engine_fut.get()` to surface engine-task exceptions, and
  validates correctness only from `request_result` snapshots.
  Main does not touch `llama_context`, `llama_batch`,
  `llama_decode`, `llama_get_logits_ith`, or `llama_memory_seq_*`
  — the residual-KV-empty check is performed inside the engine
  task.
- **Slice 3b (optional later — not scheduled):** named HPX
  orchestration pool / resource-partitioner setup. See
  *Why no orchestration pool yet?* below.
- **Slice 4 (done):** lifecycle trace events
  (`LLAMA_HPX_CB_TRACE=1`) and a descriptive metrics block. Trace
  emits `engine_start`, `request_admitted`, `seq_prefilled`,
  `decode_row`, `seq_complete`, `kv_cleared`, `promise_fulfilled`,
  `engine_stop` to stderr (off by default; off-path is one atomic
  load + early return per call site). Each repeat iteration prints
  a metrics block to stdout: `wall_ms`, `decode_calls`,
  `update_iterations`, `rows_per_batch` (p50/p95/max),
  `active_seqs_per_iter` (p50/p95/max), per-budget completed
  counts, per-budget `ttc_ms` (mean / p95), `futures_created`,
  `promises_fulfilled`, `futures_completed`, `engine_task_count`.
  Metrics are descriptive only — no comparisons or speedup
  language. Final emit is `HPX_CB_STEP4: PASS` / `FAIL: <reason>`.
- **Slice 5 (done):** same-shape correctness equivalence between
  the pure llama.cpp reference (`tools/multiseq-batch-gate/`,
  `GATE_STEP5: PASS`) and the HPX prototype (`HPX_CB_STEP4: PASS`)
  on the 99-seq / `{8,64,256}` shape. Verdict
  `HPX_CB_PROTO: PASS` is recorded in
  `tools/hpx-continuous-batch-gate/results.md`. Correctness
  equivalence only; not a performance comparison.
- **Cancel Slice 1 (done):** cancellation **data model only**.
  Added the `request_status` enum
  (`completed` / `cancelled` / `failed_reserved`) and new cancel
  fields on `seq_state` (`std::atomic<bool> cancel_requested`,
  `cancel_after_decoded_tokens`, `cancel_observed`,
  `cancel_observed_iter`, `n_decoded_at_cancel`) and on
  `request_result` (`status`, `cancel_observed_iter`,
  `n_decoded_at_cancel`). Added CLI: `--cancel-plan <csv>`
  (default `1,4,7,2,5,8`) and `--cancel-after <int>`
  (default `16`). The plan was parsed, printed, and each plan
  seq's actual budget was asserted against the round-robin mix
  prediction. The plan was **not** propagated into `seq_state`
  and the engine did **not** observe cancellation in that slice.
  Final emit was `HPX_CB_CANCEL_STEP1: PASS` / `FAIL: <reason>`.
- **Cancel Slice 2 (done):** cooperative cancellation enabled
  inside the engine loop. The plan from `--cancel-plan` is
  propagated into `seq_state` at engine construction
  (`cancel_after_decoded_tokens` per seq). The engine task
  observes cancellation only at iteration boundaries: once
  post-prefill, and once at the top of each decode iteration
  **before** building the active row set. No `llama_decode`
  call is interrupted. On observation the engine records
  `cancel_observed`, `cancel_observed_iter`,
  `n_decoded_at_cancel`; runs the existing KV-clear +
  cross-talk check; and fulfills the request promise once with
  `status = cancelled`. Cancelled seqs do not appear in any
  later decode batch (they become `seq.done = true` via the
  shared clear path), so `wasted_decode_rows_after_cancel` is
  structurally `0`. The `--cancel-plan` parser is generalized
  to accept `seq_id 0` while still rejecting negative seq ids.
  Final emit was `HPX_CB_CANCEL_STEP2: PASS` / `FAIL: <reason>`.
- **Cancel Slice 3 (done):** result/status path polish on top
  of Cancel Slice 2 — no behavioral change. Cancel-family
  trace events were aligned to the spec format
  (`cancel_requested seq=<id> budget=<int> cancel_after=<int>`,
  `cancel_observed seq=<id> iter=<int> n_decoded=<int>`,
  `cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1`,
  `cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>`).
  Lifecycle events `seq_complete`, `kv_cleared`, and
  `promise_fulfilled` fire only on the completion path; the
  cancellation path emits its own `cancel_*` events. Default-plan
  trace event counts: `seq_complete=93`, `kv_cleared=93`,
  `promise_fulfilled=93`, `cancel_requested=6`,
  `cancel_observed=6`, `cancel_kv_cleared=6`,
  `cancel_future_fulfilled=6`. Main also prints an explicit
  per-iter `status_summary: completed=<> cancelled=<> total=<>`
  line and gates `n_decoded == n_decoded_at_cancel` for
  cancelled results. Final emit was `HPX_CB_CANCEL_STEP3: PASS`
  / `FAIL: <reason>`.
- **Cancel Slice 4 (done):** closeout of the
  cooperative-cancellation line — no new behavior. The per-iter
  metrics block now reports completed/cancelled splits
  correctly: the `completed[budget=B]` line counts only
  `status == completed` (the legacy form printed every seq of
  that budget, which was misleading once cancellation became
  real), and a paired `cancelled[budget=B]` line is emitted for
  budgets with any cancelled seqs. The single
  `ttc_ms[budget=B]` field is split into
  `ttc_ms_completed[budget=B]` and (when present)
  `ttc_ms_cancelled[budget=B]`, both descriptive only and not
  part of the determinism contract. Top-level metrics added
  after `engine_task_count`: `completed_count`,
  `cancelled_count`, `decode_failures`,
  `wasted_decode_rows_after_cancel`, `residual_kv_empty`. All
  Cancel Slice 2 / 3 correctness gates still apply unchanged.
  The cooperative-cancellation evidence is summarized in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Cooperative cancellation results* section. Final emit was
  `HPX_CB_CANCEL_STEP4: PASS` / `FAIL: <reason>`.
- **Live Admission Slice 1 (done):** live-admission
  **data model only**, no behavior change. Adds the
  `admission_source` enum (`none`, `cancel_freed`; only `none`
  is set in this slice) and four new fields on `seq_state`
  (`request_id`, `admitted_at_iter`, `previous_request_id`,
  `admission_src`) and `request_result` (`admitted_at_iter`,
  `reused_seq_id`, `previous_request_id`, `admission_src`).
  `request_id` is initialized to `seq_id` at engine
  construction and stays equal in this slice (no admission
  rebinds it). `reused_seq_id` lives only on `request_result`
  and stays at default `-1` on every result. An engine-level
  `free_due_to_cancel` placeholder (a `std::deque<int32_t>`)
  is added but **must remain empty and unused**: the cancel
  path does not push to it and no admission path pops from
  it; `engine_result.free_due_to_cancel_violations` counts
  any non-empty observation at iter end / run end and is
  gated to `0`. `fulfill_promise` populates the four new
  `request_result` fields with their defaults from
  `seq_state` on every completion AND every cancellation
  path; `reused_seq_id` is intentionally not copied. Main
  prints a per-iter `admit_step1: …` audit line and gates
  every result for `admitted_at_iter == -1`,
  `reused_seq_id == -1`, `previous_request_id == -1`,
  `admission_source == none`, and `request_id == seq_id`.
  The `--repeat 2` determinism tuple is extended with
  `request_id`, `admitted_at_iter`, `reused_seq_id`,
  `previous_request_id`, `admission_src`. No CLI flags, no
  waiting queue, no traces, no metrics changes, no admission
  behavior. All Cancel Slice 4 correctness gates still pass
  on the 99-seq / `{8,64,256}` shape against a Metal-enabled
  build (the recorded canonical-hash baseline). The
  data-model evidence is summarized in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live admission Slice 1 results* section. Final emit was
  `HPX_CB_ADMIT_STEP1: PASS` / `FAIL: <reason>`.
- **Live Admission Slice 2 (done):** waiting queue,
  **no admission behavior**. The engine gains a read-only
  handle to a construction-time `std::vector<waiting_request>`
  built by main; the engine samples its size at run start
  (`queued_count`) and at run end
  (`waiting_queue_size_at_engine_end`) and fails closed if
  they differ. The bound active population is now distinct
  from the model's slot capacity: `seqs_` and `promises_` are
  sized `n_active` while the residual-KV-empty sweep covers
  all `n_seq_max` slots (so the not-yet-bound slots
  `[n_active, n_seq_max)` are also asserted empty). Three new
  CLI flags, all with backwards-compatible defaults:
  `--n-active <int>` (default = `n_seqs`),
  `--n-waiting <int>` (default `0`),
  `--waiting-budget <int>` (default `64`). Validation enforces
  `1 <= n_active <= n_seqs` and `n_active + n_waiting <= n_seqs`,
  and the cancel-plan range tightens from `[0, n_seqs)` to
  `[0, n_active)` so cancellation only targets bound seqs. The
  per-iter audit line is renamed to `admit_step2: …` and now
  also reports `queued_count` and
  `waiting_queue_size_at_engine_end`; the metrics block adds
  a single descriptive `queued_count = <n>` line. Hard scope:
  no waiting futures, no waiting prefill, no waiting decode
  rows, no admission, no traces, no new trace events; the
  `free_due_to_cancel` placeholder remains empty/unused (no
  cancel-path push, no admission-path pop). All Slice-1 field
  gates still pass on every result and Cancel Slice 4
  carry-over invariants hold for the active population. Final
  emit was `HPX_CB_ADMIT_STEP2: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live Admission Slice 2 results* section.
- **Live Admission Slice 3 (done):** actual
  **cancel-freed-slot live admission**. This is the first
  slice with a real admission behavior: waiting requests are
  now consumed by the engine, bound to slots freed by
  cooperative cancellation, prefilled in a mixed
  prefill+decode batch, and run to completion alongside
  surviving active seqs. Wiring:
  - `cancel_and_fulfill` appends each cleared slot's
    `seq_id` to `free_due_to_cancel` **after** its KV clear
    succeeds (no push from any other source; no naturally
    completed budget-8 free slot is added).
  - At the top of every decode iter, the engine snapshots
    `free_due_to_cancel.size()` **before** the iter's
    cancellation observation pass and consumes only that
    many entries. This produces the design's one-iter delay
    — cancellations observed at iter `K` admit at iter
    `K + 1` — while keeping a single deque (`{1, 2, 4, 5,
    7, 8}` ascending order is preserved by the cancel
    observation loop walking seqs in `seq_id` order).
  - Admission pops a cancel-freed `seq_id` and the FIFO
    head of the waiting queue, asserts
    `llama_memory_seq_pos_{min,max}(reused_seq) == -1`,
    resets the slot's per-seq dynamic state, sets
    `request_id`, `previous_request_id`, `admitted_at_iter`,
    `admission_src = cancel_freed`, and
    `decode_budget = waiting_budget`. The slot's promise is
    **replaced** with a fresh `hpx::promise<request_result>`;
    the new future is pushed to the engine-internal
    `admitted_futures_` collection (mutex-guarded; main
    drains it strictly after `engine_fut.get()`).
  - The row-build loop emits prompt/prefill rows
    (`logits=true` only on the last) for any seq with
    `admitted_at_iter == iter && n_decoded == 0`, and a
    single decode row for every other still-active seq.
    Iter-17 in the smoke is therefore a mixed batch of 36
    prefill rows (6 admitted × 6 prompt tokens) + 56 decode
    rows (28 surviving budget-64 + 28 surviving budget-256).
  - `fulfill_promise` populates `rr.reused_seq_id` to the
    bound `seq_id` for admitted results, and `-1` for
    non-admitted.

  Per-result validation in main is now partitioned by
  `(admission_src, status, decode_budget)`: surviving-active
  budget-64 and admitted budget-64 each hold a single
  in-partition hash, the two are not required to match
  (admitted budget-64 hash on the smoke shape is
  `0x3b15a0474dfe11be`; surviving-active budget-64 stays
  `0x88a4dc75a31d4325`; budget-8 canonical anchor
  `0x0619d4d1900c2365` still applies to the untouched
  `admission_src == none` partition). Results are sorted
  and deduplicated by `request_id` because cancelled and
  admitted results can share the same `seq_id`. Future /
  promise / result counts use `n_active + admitted_count`
  rather than `n_active + n_waiting`, so partial-admission
  shapes still pass cleanly.

  The Slice 1 / Slice 2 "free_due_to_cancel must remain
  empty" guard is retired: in default mode (`--n-waiting 0`)
  the cancel plan still fires, leaves `cancel_plan.size()`
  harmless residue, and the residue is reset at the start of
  every repeat. The Slice 2 "queue must not be consumed"
  guard is replaced with a `consumed == admitted_count`
  cross-check.

  No new trace event names yet — the existing
  `decode_row` / `seq_complete` / `kv_cleared` /
  `promise_fulfilled` lifecycle events fire for admitted
  seqs through the same paths. Admission-specific traces
  (`request_queued`, `request_admitted_live`, `seq_reused`,
  `admitted_prefilled`, `admitted_decode_row`,
  `admitted_complete`) and admission metrics
  (`admission_iter_set`, `reused_seq_id_count`,
  `admitted_ttc_ms[budget=*]`,
  `waiting_queue_depth_per_iter`) are deferred to Slice 4.
  Slice 3 only adds three descriptive lines:
  `admitted_count`, `waiting_queue_size_at_engine_end`, and
  `reused_seq_id_set`. Final emit was
  `HPX_CB_ADMIT_STEP3: PASS` / `FAIL: <reason>`. Closeout
  evidence is in `tools/hpx-continuous-batch-gate/results.md`
  under the *Live Admission Slice 3 results* section.
- **Live Admission Slice 4 (done):** admission-
  specific **traces and descriptive metrics only — no
  admission behavior change from Slice 3**. The smoke shape's
  per-result anchors (87 orig completed, 6 cancelled, 6
  admitted, total 99, residual KV empty, partitioned hashes
  including the admitted-budget-64 anchor
  `0x3b15a0474dfe11be`) reproduce exactly. Six new trace
  events ride the existing `LLAMA_HPX_CB_TRACE=1` gate, so
  trace-off captures stay byte-identical to Slice 3 (zero
  `[hpx-cb-gate] event=` lines on stderr in default, compact,
  and repeat-2 trace-off runs).

  New trace events (format unchanged: `[hpx-cb-gate]
  event=<name> key=value …`):
  - `request_queued request=<id> budget=<int> queue_pos=<int>`
    — emitted from main once per waiting request at queue
    construction time, before the engine task runs.
  - `seq_reused seq_id=<id> previous_owner=<request_id>
    new_owner=<request_id> iter=<int>` — emitted in the
    admission loop immediately after the KV-empty assertion
    succeeds and before any state mutation, so
    `previous_owner` carries the prior `request_id`.
  - `request_admitted_live request=<id> reused_seq_id=<id>
    iter=<int>` — emitted right after `seq_reused`, distinct
    from the existing `request_admitted` (which keeps its
    original-active-only semantics).
  - `admitted_prefilled request=<id> seq_id=<id>
    first_token=<id>` — emitted in the post-decode argmax
    loop on the first-token of an admitted seq, gated on a
    predicate (`admission_src != none && admitted_at_iter ==
    iter && n_decoded == 0`) captured **before** any
    `n_decoded` mutation, so it fires exactly once per
    admitted request.
  - `admitted_decode_row request=<id> seq_id=<id> iter=<int>
    pos=<int>` — emitted in the row-build decode-row branch
    when `admission_src != none`, **in addition to** the
    generic `decode_row` (the 1:1 `decode_row`-per-row
    invariant is preserved).
  - `admitted_complete request=<id> seq_id=<id> budget=<int>
    done_iter=<int> hash=<hex>` — emitted in
    `finalize_and_fulfill` next to `seq_complete` when the
    completed seq was admitted.

  All existing generic events (`engine_start`,
  `request_admitted`, `seq_prefilled`, `decode_row`,
  `seq_complete`, `kv_cleared`, `promise_fulfilled`,
  `engine_stop`, the `cancel_*` family) keep firing
  unchanged.

  New descriptive metrics on the per-iter metrics block:
  - `admitted_count` (already in Slice 3, kept here)
  - `reused_seq_id_set` (already in Slice 3, kept here)
  - `reused_seq_id_count` — equal to `reused_seq_id_set.size()`
    in v1; printed for grep-friendly closeout.
  - `admitted_prefill_events` — structural counter
    incremented next to the `admitted_prefilled` trace site
    (unconditionally, regardless of trace gating).
  - `admission_iter_set` — distinct decode-iter values at
    which admission fired (e.g. `{17}` on the smoke shape).
  - `waiting_queue_depth_after_admission_per_iter
    p50=<int> p95=<int> max=<int>` — sampled after each
    iter's admission loop, so iter-17 in the smoke records
    `0` (admission consumed all 6 waiters this iter); 16
    earlier iters record `6`. p50/p95/max via the existing
    `percentile()` helper. Note: post-admission sampling, so
    "depth before this iter would have admitted" is one
    sample later than the `admission_iter_set` value.
  - `admitted_ttc_ms[budget=B]` — completion-time
    distribution (mean / p95) for admitted-completed results
    per budget class. Same wall-clock-from-engine-start
    convention as `ttc_ms_completed` (descriptive only, not
    part of the determinism contract).

  New structural gate (independent of trace flag):
  `er.admitted_prefill_events == er.admitted_count` per
  repeat. Catches a misplaced `admitted_prefilled` predicate
  even when traces are disabled.

  Final emit was `HPX_CB_ADMIT_STEP4: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live Admission Slice 4 results* section.
- **Live Admission Slice 5 (this version):** completion-
  freed-slot live admission. Adds the second admission source
  on top of Slice 4: naturally completed slots can now be
  reused by waiting requests, in addition to the cancel-freed
  reuse path proven in Slice 3. Behavior is gated by a new
  default-OFF CLI flag `--reuse-completed`. With
  `--reuse-completed` OFF the Slice 3 cancel-freed admission
  flow is preserved semantically — the engine never touches
  the completion pool, residual `completion_freed_pool_size_at_run_end`
  is `0`, no `completion_freed` admissions occur, and the
  Slice 3 / Slice 4 anchors (87 orig completed, 6 cancelled,
  6 admitted, admitted-budget-64 hash `0x3b15a0474dfe11be`,
  same `reused_seq_id_set = {1,2,4,5,7,8}`, same
  `admission_iter_set = {17}`) reproduce. With
  `--reuse-completed` ON and `--cancel-plan none`, naturally
  completed budget-8 actives populate a second engine-level
  deque, `free_due_to_completion`, which is **demand-gated**:
  a completed slot is pushed only while the waiting queue
  still has unadmitted entries, so once the waiting queue is
  drained later natural completions (budget-64, budget-256)
  do **not** grow the pool. The admission loop consumes
  cancel-freed entries first, then completion-freed entries —
  this priority preserves the Slice 3 mapping byte-for-byte
  when both sources are present, even though Slice 5 itself
  deliberately exercises only one source at a time (combined
  cancel+completion admission in one run is out of scope and
  is deferred to a later mixed-source slice).

  New CLI flag (default OFF):
  - `--reuse-completed` — enables the completion-freed
    admission path. When OFF, the engine never reads or writes
    `free_due_to_completion`. When ON, the demand gate fires
    in `finalize_and_fulfill` immediately after `clear_and_check`
    succeeds.

  New admission source enum value:
  - `admission_source::completion_freed` — joins
    `admission_source::cancel_freed` and `admission_source::none`.
    Carried through `seq_state`, `request_result`, the engine
    metrics, and the trace payloads.

  Trace payload extension (no new event names): the five
  admission events from Slice 4 now carry an explicit
  `admission_source=<value>` key so cancel-freed and
  completion-freed admissions can be distinguished without
  adding new events. Affected events:
  - `seq_reused`
  - `request_admitted_live`
  - `admitted_prefilled`
  - `admitted_decode_row`
  - `admitted_complete`

  Event *counts* keep the same shape (one event per admitted
  request for the per-request events; one per admitted decode
  row for `admitted_decode_row`); only payloads become richer.
  Slice 3 cancel-freed-slot trace captures now emit
  `admission_source=cancel_freed` on the five events above; the
  Slice-4 trace-on capture must therefore be compared
  semantically, not byte-for-byte.

  New descriptive metric on the per-iter metrics block:
  - `completion_freed_pool_size_at_run_end` — residual size of
    `free_due_to_completion` at engine end. Under the demand
    gate, this equals `first_wave_count − admitted_count`. For
    the Slice 5 smoke (round-robin `{8,64,256}` actives,
    `n_active=90`, `n_waiting=9`) this is `30 − 9 = 21`.

  The `admit_step3:` per-iter audit line is renamed
  `admit_step5:` and now also reports the
  `cancel_freed=<n>` / `completion_freed=<n>` split and the
  pool residual. The label is the only stdout name that
  changes between Slice 4 and Slice 5 in the default-OFF
  regression — content under the audit line continues to
  match Slice 4 semantically.

  The `admitted_ttc_ms[budget=B]` line is split by source so
  cancel-freed and completion-freed admissions are no longer
  blurred together: `admitted_ttc_ms[src=cancel_freed,budget=B]`
  and `admitted_ttc_ms[src=completion_freed,budget=B]`.
  Descriptive only.

  Slice 5 smoke shape (deterministic):

  ```text
  n_seq_max       = 99
  n_active        = 90
  n_waiting       = 9
  waiting_budget  = 8
  active budgets  = round-robin {8, 64, 256}
  cancel_plan     = none (empty)
  ```

  Slice 5 admission mapping (FIFO over the first 9 entries of
  the budget-8 round-robin slots, in ascending `seq_id` order):

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

  Every admitted request: `budget=8 admitted_at_iter=8
  done_iter=15 pos_max_at_clear=12
  admission_source=completion_freed`. The 21 remaining
  budget-8 slots (`{27, 30, …, 87}`) stay pooled in
  `free_due_to_completion` and KV-empty for the rest of the
  run; later natural completions of budget-64 (iter 63) and
  budget-256 (iter 255) **do not** push to the pool because
  the waiting queue is empty by then (demand gate).

  Source priority (when both queues are populated, which
  Slice 5 itself does **not** exercise):

  ```text
  1. free_due_to_cancel      (preserves Slice 3 mapping)
  2. free_due_to_completion  (new in Slice 5)
  ```

  Hash gating policy:
  - Natural budget-8 partition (`admission_src=none, budget=8`)
    runs entirely before admission iter 8, so its canonical
    anchor `0x0619d4d1900c2365` is gated strictly.
  - Surviving natural budget-64 and budget-256 partitions
    cross iter 8's mixed prefill+decode batch shape; gated
    only by within-run uniqueness + `--repeat 2` determinism.
    They happen to match canonical anchors in this run, but
    that is recorded as observed, not as a strict gate.
  - Admitted `completion_freed` budget-8 partition: observed
    hash recorded descriptively (also `0x0619d4d1900c2365` in
    this run); not canonical-gated.

  New structural gates (in addition to all Slice 4 gates):
  - `admission_iter_set == {8}`
  - `reused_seq_id_set == {0,3,6,9,12,15,18,21,24}`
  - `admission_src == completion_freed` for every admitted
    result
  - `admitted_count == 9` and
    `admitted_prefill_events == 9`
  - `completion_freed_pool_size_at_run_end == 21`
  - the 21 residual `seq_id`s remain KV-empty at run end
    (covered by the existing all-`n_seq_max`-slots residual
    sweep)
  - with `--reuse-completed` OFF: zero `completion_freed`
    admissions AND `completion_freed_pool_size_at_run_end == 0`
  - `free_due_to_cancel` remains empty for the whole Slice 5
    smoke run (no cancel admissions); cross-source
    confusion is fail-closed (cancel-freed `reused_seq_id`
    must be in `cancel_plan`; completion-freed `reused_seq_id`
    must **not** be).

  Out of scope for Slice 5: a single run that exercises
  cancel + completion admission together. That is a future
  mixed-source slice. Slice 5 deliberately keeps the failure
  modes separable by exercising exactly one source per smoke.

  Final emit is `HPX_CB_ADMIT_STEP5: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live Admission Slice 5 results — completion-freed slot
  reuse* section.
- **Live Admission Slice 6 (this version):** deterministic
  **async external arrivals**. Adds a single scripted HPX
  submitter task that pushes external arrivals into an
  engine-owned inbox under a release+ack barrier; the engine
  drains the inbox at the top of the next iter and admits via
  the existing Slice 3 cancel-freed path. The path proven by
  the slice is:

  ```text
  external HPX submitter task
    -> release_future (engine sets it at end of iter K)
    -> engine::submit(arrival_msg)  [acquires hpx::spinlock]
    -> ack_promise   (submitter sets it after pushing all K-arrivals)
    -> engine inbox
    -> drain at top of iter K+1     (engine task only)
    -> waiting_queue_consumable_    (arrival_source=external)
    -> existing cancel_freed admission at iter K+1+...
    -> external future fulfilled
  ```

  For the smoke shape with `--external-release-iter 8` and
  `--cancel-after 16`: `release_iter = 8`, `drain_iter = 9`,
  `admit_iter = 17`.

  HPX-native constraints satisfied:
  - **No `std::thread`, no `std::condition_variable`, no
    `std::this_thread::sleep_for`, no new `std::mutex`**. The
    only new lock is an `hpx::spinlock` (`inbox_mtx_`)
    guarding the engine inbox.
  - Timing is **deterministic, not wall-clock** — the engine
    releases the submitter at end of iter K via
    `hpx::promise<void>::set_value()` and suspends on a
    matching `hpx::future<void>`; the submitter pushes its
    K-block and sets the ack promise, after which the engine
    resumes. No `sleep_for`-style coordination.
  - **Only the engine task touches llama.cpp state**
    (`llama_context`, `llama_batch`, `llama_decode`,
    `llama_memory_seq_*`, `llama_get_logits_ith`). The
    scripted submitter helper body must not call any `llama_*`
    API; it only awaits the release future, calls
    `engine::submit()`, and sets the ack promise.
  - `engine::submit()` **does not mutate `engine_result`** —
    it only acquires the inbox spinlock and pushes the
    message; counters are bumped by the engine when it drains
    (Correction 1 from the Slice 6 plan; there is no
    `external_arrival_count` on `engine_result`).

  New CLI flags (default OFF, fully inert path when 0):

  ```
  --n-external-arrivals <int>     default: 0
  --external-arrival-budget <int> default: 64
  --external-release-iter <int>   default: 0
  ```

  When `--n-external-arrivals == 0` no submitter task is
  spawned, no release/ack handle is registered, the inbox is
  never written, and every Slice 6 counter stays at its
  default (`arrival_drained_count == 0`,
  `external_admitted_count == 0`,
  `first_external_drain_iter == -1`,
  `iter_release_fired_set == {}`, `submitter_ack_set == {}`).
  Default runs are byte-equivalent to Slice 5 semantics.

  New trace events (gated on `LLAMA_HPX_CB_TRACE=1`):
  - `request_submitted_external request=<id> budget=<int>
    arrival_source=external` — emitted by `engine::submit()`
    from the submitter task at push time.
  - `arrival_drained request=<id> iter=<int> budget=<int>` —
    emitted by the engine at the top of iter K+1 as it moves
    arrivals out of the inbox.
  - `iter_release_fired iter=<int>` — emitted by the engine at
    end of iter K when it sets the release promise.
  - `submitter_ack_observed iter=<int>` — emitted by the
    engine after the matching ack future returns from
    `.get()`.

  Existing events extended with an explicit `arrival_source=`
  key so the preloaded and external paths partition cleanly:
  - `request_queued` (both the main-side preloaded emission
    and the engine-side external emission carry
    `arrival_source=preloaded|external`)
  - `request_admitted_live` (admission carries
    `arrival_source=preloaded|external` alongside the existing
    `admission_source=...`)

  New `engine_result` counters (engine-side only):

  ```
  arrival_drained_count
  external_admitted_count
  first_external_drain_iter
  iter_release_fired_set
  submitter_ack_set
  ```

  Engine-side residual checks before the existing residual-KV
  sweep: the inbox MUST be empty at run end and
  `external_promises_` MUST be empty at run end; either being
  non-empty fails closed with an explicit reason.

  Smoke gates (`--n-external-arrivals 6
  --external-arrival-budget 64 --external-release-iter 8
  --n-active 93 --n-waiting 0 --cancel-plan 1,4,7,2,5,8
  --cancel-after 16`):
  - `arrival_drained_count == 6`
  - `external_admitted_count == 6`
  - `first_external_drain_iter == 9`
  - `iter_release_fired_set == {8}`
  - `submitter_ack_set == {8}`
  - request→seq mapping (sorted cancel-plan FIFO):
    `93→1`, `94→2`, `95→4`, `96→5`, `97→7`, `98→8`
  - every external admitted result has
    `arrival_src == external`, `admission_src == cancel_freed`,
    `admitted_at_iter == 17`
  - admitted budget-64 hash == `0x3b15a0474dfe11be`
  - `residual_kv_empty == true`, inbox + `external_promises_`
    asserted empty before the KV sweep

  Final emit is `HPX_CB_ADMIT_STEP6: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live Admission Slice 6 results — async external arrivals*
  section.
- **Live Admission Slice 7 (this version):** mixed-source
  admission **priority**. Proves the engine's source-priority
  rule end-to-end with both pools non-empty at the same
  admission boundary: when `free_due_to_cancel_` and
  `free_due_to_completion_` both have entries when an admission
  step runs, **cancel_freed drains first, then
  completion_freed**. Earlier slices only ever exercised one
  source at a time (Slice 3 cancel-only, Slice 5 completion-
  only, Slice 6 cancel-freed only with external arrivals).

  Smoke shape that forces the mixed boundary:

  ```text
  n_seq_max               = 99
  n_active                = 84
  n_waiting               = 9
  waiting_budget          = 8
  n_external_arrivals     = 6
  external_arrival_budget = 64
  external_release_iter   = 16
  --reuse-completed       = ON
  cancel_plan             = 1,4,7,2,5,8
  cancel_after            = 16
  active budgets          = round-robin {8,64,256}
  ```

  Two admission phases:

  - **Phase 1 — iter 8, completion_freed + preloaded.** All 28
    budget-8 actives complete naturally at iter 7; demand gate
    pushes all 28 onto `free_due_to_completion_` (waiting queue
    still has 9 entries). At top of iter 8, admission's
    completion-freed pass binds the first 9 in FIFO order to
    the preloaded waiters:

    ```text
    req 84 -> seq  0
    req 85 -> seq  3
    req 86 -> seq  6
    req 87 -> seq  9
    req 88 -> seq 12
    req 89 -> seq 15
    req 90 -> seq 18
    req 91 -> seq 21
    req 92 -> seq 24
    ```

    `admission_source=completion_freed`,
    `arrival_source=preloaded`, `admitted_at_iter=8`,
    `decode_budget=8`, `done_iter=15`.
    `free_due_to_completion_` residual after iter 8 = 19.

  - **Phase 2 — iter 17, cancel_freed + external.** At iter 16
    the cancel plan fires and pushes 6 cancel-freed slots
    `{1,2,4,5,7,8}` (ascending seq_id order). The release+ack
    barrier at end of iter 16 publishes the 6 external arrivals
    to the inbox; iter 17's drain moves them into
    `waiting_queue_consumable_` as
    `arrival_source=external`. **Both pools are now non-empty:**
    `free_due_to_cancel_` has 6, `free_due_to_completion_` has
    19 (untouched since iter 8). The admission loop drains
    cancel-freed first; the FIFO matches the sorted cancel-plan
    to the FIFO of external arrivals:

    ```text
    req 93 -> seq 1
    req 94 -> seq 2
    req 95 -> seq 4
    req 96 -> seq 5
    req 97 -> seq 7
    req 98 -> seq 8
    ```

    `admission_source=cancel_freed`,
    `arrival_source=external`, `admitted_at_iter=17`,
    `decode_budget=64`, `done_iter=80`. After cancel-freed
    drains, the waiting queue is empty, so the completion-
    freed pass at iter 17 is a no-op.
    `free_due_to_completion_` residual stays at 19.

  Priority is observable in both directions of the rule: if it
  were reversed (completion-freed first), iter 17 would admit
  the 6 externals onto `{27,30,33,36,39,42}` instead of
  `{1,2,4,5,7,8}`, and `free_due_to_cancel_` would leak 6
  entries. The slice7 strict gate fails closed if either:

  - any iter-17 admission has `admission_source !=
    cancel_freed`,
  - any iter-17 admission's `reused_seq_id` is in the residual
    completion-freed pool
    `{27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81}`,
  - or `completion_freed_pool_size_at_run_end != 19`.

  Source-side changes (minimal):

  - The pre-Slice-7 global "strictly-ascending
    `reused_seq_id_set`" gate (added in Slice 3) was correct
    only for single-iter, single-source admissions and now
    breaks for the mixed shape (iter-8's `{0,3,…,24}` followed
    by iter-17's `{1,2,…,8}`). Slice 7 replaces it with two
    weaker but still strict checks: **global no-duplicate**
    plus **per-`(admitted_at_iter, admission_src)` strictly
    ascending** (in FIFO admission order, i.e. request_id-
    ascending). The cancel pool and completion pool each pop
    in seq_id-ascending order, so this property holds for
    every existing slice and now also for Slice 7.
  - New `slice7_strict` gate that fires on the mixed-source
    shape and asserts the two-phase mapping, the source-
    priority invariant, and the residual = 19 invariant.
  - New `admit_step7:` audit line on stdout:

    ```text
    iter[r] admit_step7: phase1@iter=8 completion_freed=9
                        phase2@iter=17 cancel_freed=6
                        pool_residual=19
                        admission_iter_set={8,17}
                        first_external_drain_iter=17
    ```

  Out of scope (kept for a future slice):

  - **No new CLI flags.** The existing
    `--n-waiting / --waiting-budget / --n-external-arrivals /
    --external-arrival-budget / --external-release-iter /
    --reuse-completed / --cancel-plan / --cancel-after`
    surface is enough.
  - **No new trace event names.** The Slice 4–6 trace surface
    already partitions by `admission_source=` and
    `arrival_source=` on `request_admitted_live`,
    `admitted_complete`, `admitted_decode_row`,
    `admitted_prefilled`, `seq_reused`, and `request_queued`.
  - **No new HPX primitives.** The Slice 6 release/ack barrier
    and `hpx::spinlock`-protected inbox are unchanged. No new
    `std::thread`, no `std::condition_variable`, no wall-clock
    sleeps, no new `std::mutex`.
  - **No performance claim.** Slice 7 is a deterministic
    correctness slice — it proves an ordering invariant of the
    admission loop, not a throughput property.

  Hash policy: observed admitted hashes are recorded
  descriptively in `results.md` and gated only on within-
  partition uniqueness and `--repeat 2` determinism. The Slice
  6 budget-64 anchor `0x3b15a0474dfe11be` is *not* strict-
  gated against the Slice 7 batch shape (it happens to match,
  but that is recorded as observed-equal-to-Slice-6, not as a
  required equality).

  Final emit is `HPX_CB_ADMIT_STEP7: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Live Admission Slice 7 results — mixed-source admission
  priority* section.

See `docs/hpx/continuous_batching_prototype_design.md` for the
full slice plan, correctness gates, and out-of-scope list, and
`docs/hpx/continuous_batching_live_admission_design.md` for the
live-admission-specific design.

## Why no orchestration pool yet?

A named HPX orchestration pool / resource-partitioner setup is
intentionally deferred. With only one engine task, it would not
change runtime behavior in a meaningful way. It becomes useful once
the prototype has multiple HPX-side tasks, such as admission,
cancellation, priority scheduling, streaming/result dispatch, or
metrics continuations. Until then, adding pool/executor wiring
would increase failure surface without testing a new serving
invariant.

Slice 3b may be added later with gates such as:

- engine task submitted through a named orchestration executor
- engine records pool/thread identity
- no llama.cpp context access escapes the engine task
- all Slice 3 correctness gates still pass

## Build

The target is opt-in (default OFF) so default builds are unaffected.
HPX install path is provided through `HPX_DIR`.

```sh
cmake -S . -B <build_dir> \
    -DLLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=ON \
    -DHPX_DIR=/Users/unick/Desktop/hpx/hpx-install/lib/cmake/HPX
cmake --build <build_dir> --target llama-hpx-continuous-batch-gate
```

(Adjust `HPX_DIR` to your local HPX install. The path used by the
existing HPX-on serving-bench build dir is shown above.)

## Usage

```text
llama-hpx-continuous-batch-gate --model <path> [options]

  --model <path>            (required) path to .gguf model file
  --prompt <string>         default: "Hello, my name is"
  --ctx-size <int>          default: 32768
  --n-seq-max <int>         default: 99
  --n-batch <int>           default: 1024
  --n-threads <int>         default: 2  (libllama compute threads)
  --n-seqs <int>            default: 99
  --decode-budget-mix <csv> default: "8,64,256"
  --hpx-os-threads <int>    default: 1  (HPX orchestration threads)
  --repeat <int>            default: 1
  --cancel-plan <csv>       default: "1,4,7,2,5,8" (use "none" for empty)
                            non-negative seq_ids cancelled at the
                            iteration boundary once they reach
                            --cancel-after decoded tokens.
  --cancel-after <int>      default: 16
                            (decoded-token threshold; 0 fires post-prefill)
  --n-active <int>          default: n_seqs (when 0)
                            size of the bound active set; must satisfy
                            1 <= n_active <= n_seqs and
                            n_active + n_waiting <= n_seqs.
  --n-waiting <int>         default: 0
                            number of waiting requests queued at
                            construction time. Slice 3 consumes the
                            FIFO head at each admission boundary,
                            binding it to a cancel-freed slot.
  --waiting-budget <int>    default: 64
                            uniform decode budget for every waiting
                            request (Slice 3 admitted requests run
                            this many decode tokens after admission).
  --reuse-completed         default: OFF
                            Slice 5: enable completion-freed-slot
                            admission.
  --n-external-arrivals <int>     default: 0
                            Slice 6: number of async external
                            arrivals the scripted submitter HPX task
                            pushes under the release+ack barrier. 0
                            disables the path entirely (no submitter
                            spawned, no release handles registered,
                            inbox never written).
  --external-arrival-budget <int> default: 64
                            Slice 6: uniform decode budget for every
                            external arrival.
  --external-release-iter <int>   default: 0
                            Slice 6: decode iter K at which the
                            engine fires the release promise.
                            Submitter pushes its K-block then sets
                            the ack; engine drains at top of iter K+1.
```

Set `LLAMA_HPX_CB_TRACE=1` to enable HPX runtime startup/shutdown
trace lines plus the Slice-4 lifecycle events (`engine_start`,
`request_admitted`, `seq_prefilled`, `decode_row`, `seq_complete`,
`kv_cleared`, `promise_fulfilled`, `engine_stop`), the
Cancel-Slice-2 events (`cancel_requested`, `cancel_observed`,
`cancel_kv_cleared`, `cancel_future_fulfilled`), the Live
Admission Slice 4 events (`request_queued`,
`request_admitted_live`, `seq_reused`, `admitted_prefilled`,
`admitted_decode_row`, `admitted_complete`), and the Live
Admission Slice 6 events (`request_submitted_external`,
`arrival_drained`, `iter_release_fired`,
`submitter_ack_observed`) on stderr.

### Trace event format

Each lifecycle event is a single stderr line:

```
[hpx-cb-gate] event=<name> key=value [key=value ...]
```

Examples:

```
[hpx-cb-gate] event=engine_start n_seqs=99 prompt_tokens=6 max_budget=256
[hpx-cb-gate] event=request_admitted seq=0 budget=8
[hpx-cb-gate] event=seq_prefilled seq=0 first_token=29871
[hpx-cb-gate] event=decode_row iter=1 seq=0 pos=6
[hpx-cb-gate] event=seq_complete seq=0 budget=8 done_iter=7 hash=0x0619d4d1900c2365
[hpx-cb-gate] event=kv_cleared seq=0 pos_max_at_clear=12 cross_talk_ok=1
[hpx-cb-gate] event=promise_fulfilled seq=0 ttc_us=...
[hpx-cb-gate] event=engine_stop decode_calls=256 update_iterations=255 wall_ms=...
[hpx-cb-gate] event=request_queued request=93 budget=64 queue_pos=0
[hpx-cb-gate] event=seq_reused seq_id=1 previous_owner=1 new_owner=93 iter=17
[hpx-cb-gate] event=request_admitted_live request=93 reused_seq_id=1 iter=17
[hpx-cb-gate] event=admitted_prefilled request=93 seq_id=1 first_token=2259
[hpx-cb-gate] event=admitted_decode_row request=93 seq_id=1 iter=18 pos=6
[hpx-cb-gate] event=admitted_complete request=93 seq_id=1 budget=64 done_iter=80 hash=0x3b15a0474dfe11be
[hpx-cb-gate] event=request_submitted_external request=93 budget=64 arrival_source=external
[hpx-cb-gate] event=iter_release_fired iter=8
[hpx-cb-gate] event=submitter_ack_observed iter=8
[hpx-cb-gate] event=arrival_drained request=93 iter=9 budget=64
[hpx-cb-gate] event=request_queued request=93 budget=64 arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=93 reused_seq_id=1 iter=17 admission_source=cancel_freed arrival_source=external
```

`decode_row` and `admitted_decode_row` are high-cardinality (one
event per row, per decode iter). All trace output is gated on
`LLAMA_HPX_CB_TRACE=1` and is silent by default; the off-path is
a single atomic load per call site.

### Metrics block

A descriptive metrics block is printed to stdout near the end of
every repeat iteration, before the final PASS/FAIL line. Fields:

```
iter[N] metrics:
  wall_ms                 = <float>
  decode_calls            = <int>
  update_iterations       = <int>
  rows_per_batch          p50=<int> p95=<int> max=<int>
  active_seqs_per_iter    p50=<int> p95=<int> max=<int>
  completed[budget=B]     = <int>            (one line per budget class)
  ttc_ms[budget=B]        mean=<float> p95=<float>
  futures_created         = <int>
  promises_fulfilled      = <int>
  futures_completed       = <int>
  engine_task_count       = <int>
```

Metrics are descriptive only. No comparisons or speedup language.
Timing-derived fields (`wall_ms`, `ttc_ms`) are not part of the
`--repeat` determinism contract.

### Final stdout line

Current final stdout line:

- `HPX_CB_ADMIT_STEP7: PASS`

  or

- `HPX_CB_ADMIT_STEP7: FAIL: <reason>`

Historical slice labels (kept for reference only — the binary now
emits the Live Admission Slice 7 label):

- `HPX_CB_STEP1` — Slice 1
- `HPX_CB_STEP2` — Slice 2
- `HPX_CB_STEP3` — Slice 3
- `HPX_CB_STEP4` — Slice 4
- `HPX_CB_CANCEL_STEP1` — Cancel Slice 1
- `HPX_CB_CANCEL_STEP2` — Cancel Slice 2
- `HPX_CB_CANCEL_STEP3` — Cancel Slice 3
- `HPX_CB_CANCEL_STEP4` — Cancel Slice 4
- `HPX_CB_ADMIT_STEP1` — Live Admission Slice 1
- `HPX_CB_ADMIT_STEP2` — Live Admission Slice 2
- `HPX_CB_ADMIT_STEP3` — Live Admission Slice 3
- `HPX_CB_ADMIT_STEP4` — Live Admission Slice 4
- `HPX_CB_ADMIT_STEP5` — Live Admission Slice 5
- `HPX_CB_ADMIT_STEP6` — Live Admission Slice 6
- `HPX_CB_ADMIT_STEP7` — Live Admission Slice 7 (current)

## Structural prerequisites

- `llama_n_seq_max(ctx) >= --n-seq-max`
- `llama_n_ctx(ctx) >= prompt_tokens + max_decode_budget`
- `llama_n_batch(ctx) >= --n-seqs * prompt_tokens` for one-shot
  prefill
- HPX runtime starts and stops cleanly within the same process as
  `libllama`

Note: for the 99-seq Phase 3 shape, actual `n_ctx` may be larger
than requested because llama.cpp rounds/partitions context capacity
internally. Treat actual capacity as authoritative.

## Slice 4 correctness gates

All Slice-3 correctness gates still apply, listed below. Slice 4
adds trace and metrics output but does not change correctness
behavior or batch shape.

### Slice 3 correctness gates (still apply)

- engine task completes
- `futures_created = 99`
- `promises_fulfilled = 99`
- `futures_completed = 99`
- no promise fulfilled twice
- all seqs reach requested budgets
- exactly 33 seqs per budget class
- budget 8 hash == `0x0619d4d1900c2365`
- budget 64 has one unique hash within the run
- budget 256 has one unique hash within the run
- `done_iter` sets:
    - 8 → `{7}`
    - 64 → `{63}`
    - 256 → `{255}`
- `pos_max_at_clear` sets:
    - 8 → `{12}`
    - 64 → `{68}`
    - 256 → `{260}`
- per-seq KV clear succeeds before promise fulfillment
- clearing one seq does not disturb still-active siblings
- residual KV empty check is performed inside the engine task
- all `llama_decode` calls return 0
- `--repeat 2` is deterministic

## HPX ownership invariant

Futures carry `request_result` snapshots only. No future consumer
may touch `llama_context`, `llama_batch`, `llama_decode`,
`llama_get_logits_ith`, or `llama_memory_seq_*`.

## Out of scope (every slice)

Same as the design note: HTTP, streaming, cancellation, priority,
prompt cache, LoRA, speculative decoding, multimodal, `n_cmpl > 1`,
distributed serving, modifying `tools/server/` /
`tools/serving-bench/` / `tools/multiseq-batch-gate/`, performance
comparisons, multi-backend toggling, context shift, SWA.
