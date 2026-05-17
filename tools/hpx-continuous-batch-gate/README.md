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
- **Streaming Slice 1 (done):** first HPX-native per-
  request token streaming boundary, layered on top of the closed
  admission gate sequence at `HPX_CB_ADMIT_STEP7: PASS`. Adds an
  HPX local-channel token stream per bound active seq: when
  `--stream-all` is ON, the engine publishes every generated
  token id onto a per-seq channel before the existing
  `request_result` promise is fulfilled. Main consumes via a
  matching `receive_channel`. The admission gate sequence is
  preserved unchanged — this is a new gate sequence
  (`HPX_CB_STREAM_STEP1`) layered on top, not an extension of
  the admission line.

  HPX-native abstraction (replaces the earlier future-chain
  design that was explored in source review):

  ```text
  token_stream_channel  = hpx::lcos::local::channel<token_stream_event>
  token_stream_sender   = hpx::lcos::local::send_channel<token_stream_event>
  token_stream_receiver = hpx::lcos::local::receive_channel<token_stream_event>
  ```

  Event payload:

  ```text
  enum class stream_event_kind   { token, closed };
  enum class stream_close_reason { completed, cancelled, error };

  struct token_stream_event {
      stream_event_kind    kind;          // token or closed
      int32_t              token_id;      // valid iff kind == token
      stream_close_reason  close_reason;  // valid iff kind == closed
  };
  ```

  Ownership rules:
  - Engine task is the **sole producer**. Only the engine
    publishes onto the channel and only the engine calls
    `channel.close()`.
  - Main / caller is the **sole consumer**, holding the
    matching `receive_channel`. The caller never touches
    `llama_context`, `llama_batch`, `llama_decode`,
    `llama_get_logits_ith`, or `llama_memory_seq_*`.
  - Stream events carry **only** an `int32_t` token id and a
    close reason. No `llama_context` / KV / logits state ever
    crosses the channel.
  - The terminal event (`kind=closed`) is sent **before**
    `channel.close()`. Reversing the order would make `set()`
    throw `invalid_status`.
  - Underlying critical section is an `hpx::spinlock` (already
    accepted as HPX-native in Slice 6). No `std::thread`, no
    `std::condition_variable`, no `std::this_thread::sleep_for`,
    no new `std::mutex`.

  New CLI flag (default OFF):

  ```text
  --stream-all   default: OFF
                 Slice 8: enable HPX-native per-request token
                 streaming. Each bound active seq gets an
                 engine-owned hpx::lcos::local::channel; main
                 holds the matching receive_channel and drains
                 the chain after engine_fut.get().
  ```

  With `--stream-all` OFF the engine allocates no channel,
  emits no `token_stream_*` events, and keeps every Slice 8
  stream counter at zero. The Slice 7 smoke shape under
  `--stream-all` OFF reproduces Slice 7 semantics; only the
  final stdout label line changes (label-line excluded
  comparison, see Slice 7 evidence).

  New trace events (gated on the existing `LLAMA_HPX_CB_TRACE=1`
  flag — trace-off is a single atomic load per call site, as
  with every earlier slice):

  ```text
  token_stream_opened request=<id> seq_id=<id>
  token_stream_token  request=<id> seq_id=<id> pos=<int> token=<int>
  token_stream_closed request=<id> seq_id=<id> n_tokens=<int> reason=completed|cancelled|error
  ```

  `token_stream_token` is high-cardinality (one event per
  emitted token); the 1:1 `token_stream_token`-per-streamed-
  token invariant holds.

  New `engine_result` counters (maintained by the engine task
  regardless of trace state):

  ```text
  streams_opened
  streams_closed_completed
  streams_closed_cancelled
  streams_closed_error
  stream_tokens_emitted_total
  ```

  Slice 8 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 3
  --decode-budget-mix 8,64,256
  --cancel-plan none
  --n-waiting 0
  --n-external-arrivals 0
  ```

  Smoke gates (per repeat, per streaming request):

  - `stream_token_count == rr.n_decoded`
  - `streamed_hash == rr.hash` (FNV-1a over int32_t token ids,
    same fold as the per-seq result hash)
  - stream close count == 1
  - stream close reason == `completed`
  - residual KV empty (Slice 1+ carry-over)
  - `--repeat 2` deterministic on per-seq streamed token
    vectors AND close reasons

  Engine-side counter gates (Slice 8 smoke, `--stream-all` ON):

  ```text
  streams_opened            == 3
  streams_closed_completed  == 3
  streams_closed_cancelled  == 0
  streams_closed_error      == 0
  stream_tokens_emitted_total == 328   (= 8 + 64 + 256)
  ```

  Observed per-request hashes on the smoke shape (Metal build):

  ```text
  budget 8   hash=0x0619d4d1900c2365  (canonical anchor; gated)
  budget 64  hash=0x3b15a0474dfe11be  (within-run uniqueness only)
  budget 256 hash=0x8790fbe5a60c9ae6  (within-run uniqueness only)
  ```

  OFF-mode regression gate: with `--stream-all` OFF every
  Slice 8 stream counter must be exactly zero and the
  receiver vector must be empty; fail-closed otherwise.

  Hard scope of Streaming Slice 1:
  - No HTTP/API or server integration (boundary note unchanged).
  - No cancellation+streaming smoke yet; `cancel_and_fulfill`
    defensively closes with `cancelled`, but no smoke
    exercises the path. A dedicated streaming-cancellation
    slice will gate it.
  - No backpressure / bounded-channel policy. The unlimited
    `channel<T>` is used so the engine never suspends on the
    consumer. Bounded channels are deferred.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No streaming through admitted reuser slots in the smoke
    (smoke has no admission). The wiring is admission-
    agnostic, but the gates target the original-active path.
  - **No performance claim.** Streaming Slice 1 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP1: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 1 results — HPX local-channel token stream*
  section.
- **Streaming Slice 2 (done):** cancellation-aware
  token streams. Slice 1 proved per-request streaming on the
  completion path only; Slice 2 closes the cancellation path
  end-to-end on the same HPX local-channel substrate, with no
  new HPX primitive, no new CLI flag, and no engine-side
  behavior change beyond what Slice 1 already wired
  defensively. The engine-side `cancel_and_fulfill` flow
  observes cancellation at an iteration boundary, runs the
  KV-clear + cross-talk check, and closes the per-seq stream
  channel with `close_reason = cancelled` **before**
  fulfilling the per-request `request_result` promise with
  `status = cancelled`. The streamed token vector for the
  cancelled seq carries exactly `rr.n_decoded_at_cancel`
  tokens, and the stream closes with `reason=cancelled`.
  Slice 2 gates this semantically through stream length,
  close reason, `rr.status`, and `streamed_hash == rr.hash`,
  not through a per-token trace-order assertion.

  The gate is what changes in Slice 2. The Slice 1 close-
  reason / token-count gates become status-aware:

  ```text
  per streamed request:
    expected_close_reason = (rr.status == cancelled)
                              ? cancelled
                              : completed
    streamed_tokens[seq].size() == rr.n_decoded
    if rr.status == cancelled:
      streamed_tokens[seq].size() == rr.n_decoded_at_cancel
    streamed_hash == rr.hash       # completed and cancelled

  engine-side stream counters:
    streams_closed_completed == count(rr.status == completed)
    streams_closed_cancelled == count(rr.status == cancelled)
    streams_closed_error     == 0
    streams_opened           == streams_closed_completed
                              + streams_closed_cancelled
                              + streams_closed_error
    stream_tokens_emitted_total
                             == sum(rr.n_decoded over streamed)
  ```

  The `--stream-all` OFF regression on the Slice 7 admission
  shape still gates every stream counter at exactly zero.
  Slice 1 lifecycle invariants — single producer (engine
  task), single consumer (main), terminal `kind=closed` event
  sent **before** `channel.close()`, only `int32_t` token id
  and a close reason cross the channel — all carry over
  unchanged. Slice 1's Slice 7 carry-over also carries over:
  every admission/cancellation/lifecycle gate from Slices 1–7
  remains a strict invariant under `HPX_CB_STREAM_STEP2`.

  Slice 2 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 3
  --decode-budget-mix 8,64,256
  --cancel-plan 1
  --cancel-after 16
  --n-waiting 0
  --n-external-arrivals 0
  --repeat 2
  ```

  Expected per-request behavior on the smoke:

  ```text
  seq 0  budget=8    status=completed  close=completed  streamed=8
  seq 1  budget=64   status=cancelled  close=cancelled  streamed=16
                     n_decoded=16  n_decoded_at_cancel=16
                     cancel_observed_iter=16
  seq 2  budget=256  status=completed  close=completed  streamed=256
  ```

  Engine-side counter gates (Slice 2 smoke, `--stream-all` ON):

  ```text
  streams_opened              == 3
  streams_closed_completed    == 2
  streams_closed_cancelled    == 1
  streams_closed_error        == 0
  stream_tokens_emitted_total == 280   (= 8 + 16 + 256)
  ```

  Trace event counts on the trace-on smoke (`LLAMA_HPX_CB_TRACE=1`,
  per repeat — the smoke runs `--repeat 2`, so totals are
  double):

  ```text
  token_stream_opened           == 3   (per repeat)
  token_stream_token            == 280 (per repeat)
  token_stream_closed           == 3   (per repeat)
    reason=completed            == 2   (per repeat)
    reason=cancelled            == 1   (per repeat)
  cancel_requested              == 1   (per repeat)
  cancel_observed               == 1   (per repeat)
  cancel_kv_cleared             == 1   (per repeat)
  cancel_future_fulfilled       == 1   (per repeat)
  ```

  Observed hashes on the smoke (Metal build):

  ```text
  budget 8   completed         hash=0x0619d4d1900c2365   (canonical anchor; gated)
  budget 256 completed         hash=0x8790fbe5a60c9ae6   (within-run uniqueness only)
  budget 64  cancelled-prefix  streamed_hash == rr.hash   (gated; concrete value
                                                            not surfaced in stdout —
                                                            partition row prints only
                                                            unique_completed_hashes)
  ```

  The budget-64 cancelled-prefix hash is **not** compared to
  the Slice 1 budget-64 completed hash (`0x3b15a0474dfe11be`):
  Slice 1 hashed 64 tokens, Slice 2 hashes only the 16-token
  cancelled prefix.

  Hard scope of Streaming Slice 2:
  - No HTTP/API or server integration (boundary note carry-
    over).
  - No backpressure / bounded-channel policy. `channel<T>` is
    still unbounded so the engine never suspends on the
    consumer. Bounded channels remain deferred.
  - No engine-failure stream smoke. The defensive `error`
    close path still exists; no smoke exercises it yet.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No new HPX primitive: same `hpx::lcos::local::channel<token_stream_event>`,
    same `hpx::spinlock`, same `hpx::promise<request_result>`.
  - No source change to `cancel_and_fulfill` /
    `close_stream` / `token_stream_event` /
    `engine_result` counter set — Slice 2 changes only the
    gate logic and the STEP label.
  - **No performance claim.** Streaming Slice 2 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP2: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 2 results — cancellation-aware streaming*
  section.
- **Streaming Slice 3 (done):** admitted-request
  streaming over a completion-freed slot. Slices 1 and 2
  proved per-request streaming on the completion and
  cancellation paths for **original-active** seqs. Admitted
  requests — those bound to a slot freed at runtime by a
  sibling's completion (Slice 5 `--reuse-completed`
  surface) — were excluded from the Slice 1/2 gates: the
  gate's per-seq streamed loop filtered on
  `rr.admission_src == none && rr.request_id == rr.seq_id`,
  and engine-side, an admitted slot inherited
  `stream_closed = true` from the prior occupant so
  `publish_token` short-circuited silently. Slice 3 closes
  that gap on the completion-freed admission path:
  `admit_one` rebinds the per-slot stream channel
  (fresh `token_stream_channel{}`, reset
  `stream_closed = false`, reset cumulative
  `stream_tokens_emitted = 0` so the close-event trace
  reads the admitted-request count, not prev+admitted) and
  pushes an explicit `{ request_id, receiver }` bundle
  onto a new per-admission handoff vector
  (`admitted_stream_handoffs_`). Main drains the bundle
  after `engine_fut.get()` and keys streamed tokens by
  `request_id`, so the existing seq-id-indexed Slice 1
  loop and the new per-request-id Slice 3 loop coexist
  without aliasing.

  HPX-native design note (Correction acknowledged):
  the new bundle vector is guarded by the **same existing
  `admitted_futures_mtx_` critical section** that already
  serializes admitted-future handoff. No new `std::mutex`
  is introduced. No `hpx::spinlock`, `std::thread`,
  `std::condition_variable`, `std::this_thread::sleep_for`,
  or wall-clock sleep is introduced. The lock guards
  engine→main result-handoff metadata only; it does NOT
  guard `llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, or `llama_get_logits_ith` access.

  Safe claim (verbatim):

  ```text
  A waiting request admitted into a completion-freed slot
  opens a fresh HPX local-channel stream, emits exactly
  rr.n_decoded token events, has streamed_hash == rr.hash,
  and closes with reason=completed.
  ```

  The gate-side change generalizes the streaming gate to
  range over the **union** of original-active streamed
  requests (keyed by `seq_id`) and completion-freed
  admitted streamed requests (keyed by `request_id`). The
  Slice 2 status-aware close-reason / token-count / hash
  gates apply uniformly to both. Slice 3 adds:

  ```text
  per admitted-streamed request (admission_src=completion_freed):
    admitted_streamed_seen[rr.request_id] == true
    admitted_streamed_tokens[rr.request_id].size()
                                == rr.n_decoded
    streamed_hash == rr.hash
    streamed_close == completed   (Slice 3 scope; cancelled
                                   admitted streaming is
                                   deferred)
    admitted_streamed_tokens[rr.request_id]
                                != streamed_tokens[rr.seq_id]
                                (no inheritance of prev
                                 occupant's token vector)
    rr.previous_request_id >= 0
    rr.seq_id ∈ [0, n_active)

  engine-side stream counters (expected-from-results,
  admission-aware):
    streams_opened              == count(streamed requests
                                          in results)
    streams_closed_completed    == count(rr.status==completed
                                          over streamed)
    streams_closed_cancelled    == count(rr.status==cancelled
                                          over streamed)
    streams_closed_error        == 0
    streams_opened              == streams_closed_completed
                                  + streams_closed_cancelled
                                  + streams_closed_error
    stream_tokens_emitted_total == sum(rr.n_decoded over
                                       streamed)
  ```

  Coverage gate (new in Slice 3): when `--stream-all` AND
  `--reuse-completed` AND `--n-waiting > 0`, at least one
  streamed completion-freed admitted request must be
  witnessed; misconfigured smoke shapes fail-closed
  instead of silently proving nothing.

  Slice 3 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 2
  --n-active 1
  --n-waiting 1
  --decode-budget-mix 8
  --waiting-budget 16
  --reuse-completed
  --cancel-plan none
  --n-external-arrivals 0
  --repeat 2
  ```

  Expected per-request behavior:

  ```text
  request_id=0 (original active, seq_id=0, admission_src=none)
    budget=8   status=completed  close=completed  streamed=8

  request_id=1 (admitted waiter, seq_id=0,
                admission_src=completion_freed,
                arrival_source=preloaded,
                reused_seq_id=0,
                previous_request_id=0,
                admitted_at_iter=8)
    budget=16  status=completed  close=completed  streamed=16
  ```

  Engine-side counter gates (Slice 3 smoke, `--stream-all`
  ON, per repeat):

  ```text
  streams_opened              == 2   (1 ctor + 1 admit)
  streams_closed_completed    == 2
  streams_closed_cancelled    == 0
  streams_closed_error        == 0
  stream_tokens_emitted_total == 24  (= 8 + 16)
  ```

  Observed hashes on the smoke (Metal build):

  ```text
  budget 8   original  completed  hash=0x0619d4d1900c2365   (canonical anchor; gated)
  budget 16  admitted  completed  hash=0x833045f1e2ebf49f   (observed and
                                                              repeat-deterministic;
                                                              NOT a new
                                                              cross-shape canonical
                                                              claim)
  ```

  OFF-mode regression gate: with `--stream-all` OFF on the
  canonical Slice 7 admission shape every stream counter
  is exactly zero, the receiver vector is empty, and the
  new admitted-stream handoff vector is empty;
  fail-closed otherwise.

  Hard scope of Streaming Slice 3:
  - **Completion-freed admission only.** Cancel-freed
    admitted streaming is not gated; external-arrival
    admitted streaming is not gated. The `admit_one`
    rebind is explicitly predicated on
    `src == admission_source::completion_freed`, so the
    other admission paths still produce admitted requests
    whose slots inherit `stream_closed=true` and skip
    streaming (carry-over of Slice 1/2 phantom behavior).
  - No HTTP/API or server integration (boundary note
    carry-over).
  - No backpressure / bounded-channel policy.
    `channel<T>` is still unbounded so the engine never
    suspends on the consumer.
  - No engine-failure stream smoke. The defensive
    `error` close path remains untested.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No new HPX primitive. Same
    `hpx::lcos::local::channel<token_stream_event>`,
    same `hpx::spinlock` for the existing inbox, same
    `hpx::promise<request_result>` /
    `hpx::future<request_result>`.
  - **No performance claim.** Streaming Slice 3 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP3: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 3 results — admitted-request streaming
  over completion-freed slot* section.

- **Streaming Slice 4 (done):** admitted-request
  streaming over a **cancel-freed** slot. Slice 3 closed
  the completion-freed admission path; Slice 4 extends the
  same `admit_one` stream-rebind to the cancel-freed
  admission path so that a waiting request bound to a slot
  freed by a sibling's cooperative cancellation is itself
  streamed end-to-end on a fresh HPX local channel. The
  engine-side delta is one predicate change: the rebind
  block in `admit_one` now fires when
  `src == admission_source::completion_freed` **or**
  `src == admission_source::cancel_freed`. No new HPX
  primitive, no new `std::mutex`, no new CLI flag, no new
  trace event name, no new channel type, no CMake change.
  The streaming substrate, ownership rules, payload, and
  trace event set are unchanged from Slices 1–3.

  HPX-native design note (carry-over from Slice 3):
  the rebound handoff is still pushed inside the same
  existing `admitted_futures_mtx_` critical section that
  serializes admitted-future handoff. No new mutex / no
  new spinlock is introduced for the cancel-freed path.
  The lock guards engine→main result-handoff metadata
  only; it does **not** guard `llama_context`,
  `llama_batch`, `llama_decode`, `llama_memory_seq_*`, or
  `llama_get_logits_ith` access. Only the engine HPX task
  touches llama.cpp execution state.

  Safe claim (verbatim):

  ```text
  A waiting request admitted into a cancel-freed slot
  opens a fresh HPX local-channel stream, emits exactly
  rr.n_decoded token events, has streamed_hash == rr.hash,
  and closes with reason=completed without inheriting the
  cancelled occupant's close reason or stream state.
  ```

  The gate-side change is the symmetric extension of the
  Slice 3 generalization: the streamed-request loop now
  ranges over the **union** of original-active streamed
  requests (`admission_src=none`), completion-freed
  admitted streamed requests (`admission_src=completion_freed`),
  and cancel-freed admitted streamed requests
  (`admission_src=cancel_freed`). The Slice 2 status-aware
  close-reason / token-count / hash gates apply uniformly
  to all three. Slice 4 adds:

  ```text
  per cancel-freed admitted-streamed request:
    rr.admission_src                       == cancel_freed
    admitted_streamed_seen[rr.request_id]  == true
    admitted_streamed_close[rr.request_id] == completed
                                              (cancelled
                                               occupant's
                                               close_reason
                                               is NOT
                                               inherited)
    admitted_streamed_tokens[rr.request_id].size()
                                            == rr.n_decoded
    streamed_hash                           == rr.hash
    admitted_streamed_tokens[rr.request_id]
                                  != streamed_tokens[rr.seq_id]
                                  (no inheritance of the
                                   cancelled occupant's
                                   token vector)
    rr.previous_request_id                  >= 0
    rr.seq_id ∈ [0, n_active)               true
  ```

  Coverage gate (new in Slice 4): when `--stream-all` AND
  the configured `cancel_plan` is non-empty AND
  `--n-waiting > 0`, at least one streamed cancel-freed
  admitted request must be witnessed; misconfigured smoke
  shapes fail-closed instead of silently proving nothing.
  The Slice 3 completion-freed coverage gate is untouched.

  Slice 4 smoke-shape adjustment (versus the natural
  one-active sketch):

  The originally suggested `--n-active 1` shape **cannot**
  fire cancel-freed admission: when the single active seq
  is cancelled at iter K, the engine exits via
  `if (!any_active())` before iter K+1 (the design's
  one-iter cancel→admit delay). The smoke therefore uses
  `--n-active 2` with one cancelled and one surviving
  active seq, plus one preloaded waiter, so the decode
  loop stays alive across the cancel→admit boundary.
  Additionally, because greedy decoding from the same
  prompt produces the same first-N tokens regardless of
  admission path, the cancelled prefix length is set to
  8 tokens and the admitted budget to 16 so the
  independence-gate token-vector compare is naturally
  inequality-by-length.

  Slice 4 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 3
  --n-active 2
  --n-waiting 1
  --decode-budget-mix 64,256
  --waiting-budget 16
  --cancel-plan 0
  --cancel-after 8
  --n-external-arrivals 0
  --repeat 2
  ```

  Expected per-request behavior:

  ```text
  request_id=0 (original active, seq_id=0,
                admission_src=none)
    budget=64  status=cancelled  close=cancelled
    n_decoded=8  n_decoded_at_cancel=8
    cancel_observed_iter=8  streamed=8

  request_id=1 (original active, seq_id=1,
                admission_src=none)
    budget=256  status=completed  close=completed
    streamed=256

  request_id=2 (admitted waiter, seq_id=0,
                admission_src=cancel_freed,
                arrival_source=preloaded,
                reused_seq_id=0,
                previous_request_id=0,
                admitted_at_iter=9)
    budget=16  status=completed  close=completed
    streamed=16
  ```

  Engine-side counter gates (Slice 4 smoke, `--stream-all`
  ON, per repeat):

  ```text
  streams_opened              == 3   (2 ctor + 1 admit)
  streams_closed_completed    == 2   (orig 1 + admitted 2)
  streams_closed_cancelled    == 1   (orig 0)
  streams_closed_error        == 0
  stream_tokens_emitted_total == 280 (= 8 + 256 + 16)
  ```

  Observed hashes on the smoke (Metal build):

  ```text
  budget 16   admitted (cancel_freed)   completed  hash=0x833045f1e2ebf49f
  budget 256  original                  completed  hash=0x8790fbe5a60c9ae6
  budget 64   original                  cancelled  (cancelled-prefix
                                                     hash not surfaced
                                                     separately; gated
                                                     through
                                                     streamed_hash
                                                     == rr.hash)
  ```

  The admitted budget-16 hash equals the Slice 3
  completion-freed admitted budget-16 hash for the same
  prompt/policy/budget — this is expected under greedy
  decoding, **not** a new cross-shape canonical anchor.

  OFF-mode regression gate: with `--stream-all` OFF on the
  canonical Slice 7 admission shape every stream counter
  is exactly zero, the receiver vector is empty, and the
  admitted-stream handoff vector is empty; fail-closed
  otherwise.

  Hard scope of Streaming Slice 4:
  - **Cancel-freed admission path covered.** The
    `admit_one` stream rebind predicate now includes
    `cancel_freed` alongside `completion_freed`.
  - **Completion-freed admission path remains covered**
    from Slice 3 (carry-over invariant; Slice 3 smoke
    and gates unchanged).
  - **External-arrival admitted streaming still
    deferred.** The `admit_one` predicate still excludes
    `arrival_source::external` admissions; those slots
    inherit `stream_closed=true` and silently skip
    streaming until a later slice extends the rebind to
    the external arrival path.
  - No HTTP / API / server / network streaming. The
    stream is in-process only between the engine HPX task
    and `main()`.
  - No backpressure / bounded-channel policy.
    `channel<T>` is still unbounded so the engine never
    suspends on the consumer.
  - No engine-failure stream smoke. The defensive
    `error` close path remains untested.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No new HPX primitive. Same
    `hpx::lcos::local::channel<token_stream_event>`,
    same `hpx::spinlock` for the existing inbox, same
    `hpx::promise<request_result>` /
    `hpx::future<request_result>`.
  - **No performance claim.** Streaming Slice 4 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP4: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 4 results — admitted-request streaming
  over cancel-freed slot* section.

- **Streaming Slice 5 (done):** external-arrival
  admitted streaming over a **completion-freed** slot.
  Slices 3 and 4 closed admitted-request streaming for
  **preloaded** waiters on the completion-freed and
  cancel-freed admission paths. Slice 5 extends streaming
  to **external arrivals**: a request that entered through
  `engine::submit()` from a scripted HPX submitter task,
  admitted into a slot freed by a sibling's natural
  completion, opens a fresh HPX local-channel stream and
  is streamed end-to-end. The engine-side delta is one
  new rebind block at the end of the external branch of
  `admit_one`, gated on
  `stream_all_ && src == admission_source::completion_freed`
  (Slice 5 scope; the predicate was broadened in Streaming
  Slice 6 to also cover `cancel_freed`). No new HPX
  primitive, no new `std::mutex`, no new CLI flag, no new
  trace event name, no new channel type, no CMake change.

  HPX-native design note (carry-over from Slice 3 / 4):
  the rebound handoff is pushed inside the same existing
  `admitted_futures_mtx_` critical section that already
  serializes admitted-future handoff and the Slice 3 / 4
  stream handoff. No new mutex / no new spinlock is
  introduced. The submitter-held
  `hpx::future<request_result>` remains the result route
  (main does NOT push to `admitted_futures_` for external
  arrivals); the stream receiver is pushed to
  `admitted_stream_handoffs_` keyed by `request_id`, so
  main drains it via the same Slice 3 / 4 path. The lock
  guards engine→main result-handoff metadata only; it
  does **not** guard `llama_context`, `llama_batch`,
  `llama_decode`, `llama_memory_seq_*`, or
  `llama_get_logits_ith` access. The scripted submitter
  helper body still calls no `llama_*` API. Only the
  engine HPX task touches llama.cpp execution state.

  Safe claim (verbatim):

  ```text
  An externally arriving request admitted into a
  completion-freed slot opens a fresh HPX local-channel
  stream, emits exactly rr.n_decoded token events, has
  streamed_hash == rr.hash, and closes with
  reason=completed.
  ```

  The gate-side change adds:

  ```text
  per external-arrival admitted-streamed request
    (admission_src=completion_freed, arrival_src=external):
      admitted_streamed_seen[rr.request_id]  == true
      admitted_streamed_close[rr.request_id] == completed
      admitted_streamed_tokens[rr.request_id].size()
                                              == rr.n_decoded
      streamed_hash                           == rr.hash
      rr.previous_request_id                  >= 0
      rr.reused_seq_id                        == prior occupant's seq_id
      rr.seq_id ∈ [0, n_active)               true
      admitted_streamed_tokens[rr.request_id]
                                    != streamed_tokens[rr.seq_id]
                                    (length-based inequality;
                                     not a semantic
                                     token-divergence claim)
  ```

  Coverage gate (new in Slice 5): when `--stream-all` AND
  `args.n_external_arrivals > 0` AND `--reuse-completed`
  AND `args.cancel_plan.empty()`, at least one streamed
  external-arrival admitted request must be witnessed;
  misconfigured smoke shapes fail-closed instead of
  silently proving nothing. The Slice 3 and Slice 4
  coverage gates are untouched.

  Pre-existing Live Admission Slice 6 gate adjustment:
  the original gate at the Slice 6 results-validation
  block required every external arrival to come via
  `admission_src=cancel_freed` (its own comment admitted
  "no completion-freed external path exercised yet"). Slice
  5 relaxed it to accept either `cancel_freed` (when
  `cancel_plan` non-empty) or `completion_freed` (when
  `--reuse-completed` is on). The Slice 7 mixed-source
  `slice7_strict` block is unaffected: its preconditions
  (`reuse_completed && !cancel_plan.empty() && n_waiting > 0
   && n_external_arrivals > 0`) exclude the Slice 5 smoke
  shape, so no Slice 7 invariants moved.

  Slice 5 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 3
  --n-active 2
  --n-waiting 0
  --n-external-arrivals 1
  --decode-budget-mix 8,256
  --external-arrival-budget 16
  --external-release-iter 3
  --reuse-completed
  --cancel-plan none
  --repeat 2
  ```

  Expected per-request behavior:

  ```text
  request_id=0 (original active, seq_id=0,
                admission_src=none, arrival_src=preloaded)
    budget=8     status=completed   close=completed
    streamed=8   hash=0x0619d4d1900c2365

  request_id=1 (original active, seq_id=1,
                admission_src=none, arrival_src=preloaded)
    budget=256   status=completed   close=completed
    streamed=256 hash=0x8790fbe5a60c9ae6

  request_id=2 (external arrival, seq_id=0,
                admission_src=completion_freed,
                arrival_src=external,
                reused_seq_id=0,
                previous_request_id=0,
                admitted_at_iter=8)
    budget=16    status=completed   close=completed
    streamed=16  hash=0x833045f1e2ebf49f
                 done_iter=23  pos_max_at_clear=20
  ```

  Engine-side counter gates (Slice 5 smoke, `--stream-all`
  ON, per repeat):

  ```text
  streams_opened              == 3   (2 ctor + 1 external admit)
  streams_closed_completed    == 3
  streams_closed_cancelled    == 0
  streams_closed_error        == 0
  stream_tokens_emitted_total == 280  (= 8 + 256 + 16)
  ```

  External-arrival metrics (per repeat):

  ```text
  admitted_count              == 1
  external_admitted_count     == 1
  arrival_drained_count       == 1
  first_external_drain_iter   == 4    (= external_release_iter + 1)
  iter_release_fired_set      == {3}
  submitter_ack_set           == {3}
  ```

  Observed hashes on the smoke (Metal build):

  ```text
  budget  8   original  completed             hash=0x0619d4d1900c2365   (canonical anchor)
  budget 256  original  completed             hash=0x8790fbe5a60c9ae6
  budget 16   external admitted completion_freed
                                              hash=0x833045f1e2ebf49f
  ```

  The external admitted budget-16 hash equals the Slice 3
  / Slice 4 preloaded admitted budget-16 hashes for the
  same prompt / policy / budget — expected under greedy
  decoding, **not** a new cross-shape canonical anchor.

  OFF-mode regression gate: with `--stream-all` OFF on the
  canonical Slice 7 admission shape every stream counter
  is exactly zero, the receiver vector is empty, and the
  admitted-stream handoff vector is empty; fail-closed
  otherwise.

  Hard scope of Streaming Slice 5:
  - **External-arrival admitted streaming over the
    completion-freed path covered.** The external branch
    of `admit_one` now rebinds the stream channel when
    `--stream-all` is on and `src == completion_freed`.
  - **External-arrival admitted streaming over the
    cancel-freed path was deferred at Slice 5; covered
    in Streaming Slice 6 below.**
  - **Slice 3 / Slice 4 preloaded admitted streaming
    invariants remain strict carry-overs.**
  - No HTTP / API / server / network streaming. The
    stream is in-process only between the engine HPX
    task and `main()`. The scripted submitter is an HPX
    task, not a network adapter.
  - No backpressure / bounded-channel policy.
    `channel<T>` is still unbounded so the engine never
    suspends on the consumer.
  - No engine-failure stream smoke. The defensive
    `error` close path remains untested.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No new HPX primitive. Same
    `hpx::lcos::local::channel<token_stream_event>`,
    same `hpx::spinlock` for the existing inbox, same
    `hpx::promise<request_result>` /
    `hpx::future<request_result>`.
  - **No performance claim.** Streaming Slice 5 is a
    correctness/lifecycle gate.

  Final emit was `HPX_CB_STREAM_STEP5: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 5 results — external-arrival streaming
  over completion-freed slot* section.

- **Streaming Slice 6 (done):** external-arrival
  admitted streaming over a **cancel-freed** slot. Slice 5
  closed external-arrival admitted streaming on the
  completion-freed path only; Slice 6 broadens the
  external-branch rebind predicate in `admit_one` by one
  disjunct so that an externally arriving request admitted
  into a slot freed by a sibling's cooperative cancellation
  is itself streamed end-to-end. The engine-side delta is a
  single-line predicate change at the end of the external
  branch of `admit_one`:
  `stream_all_ && (src == admission_source::completion_freed
   || src == admission_source::cancel_freed)`.
  This is structurally symmetric to the Slice 4 → Slice 6
  pattern Slice 4 already proved on the preloaded branch.
  No new HPX primitive, no new `std::mutex`, no new CLI
  flag, no new trace event name, no new channel type, no
  CMake change.

  HPX-native design note (carry-over from Slices 3 / 4 / 5):
  the rebound handoff for the external + cancel_freed
  admission is pushed inside the same existing
  `admitted_futures_mtx_` critical section that already
  serializes the Slice 3 / 4 / 5 stream handoff. No new
  mutex / no new spinlock is introduced. The submitter-held
  `hpx::future<request_result>` remains the result route
  (main does NOT push to `admitted_futures_` for external
  arrivals); the stream receiver is pushed to
  `admitted_stream_handoffs_` keyed by `request_id`, so
  main drains it via the same Slice 3 / 4 / 5 path. The
  lock guards engine→main result-handoff metadata only; it
  does **not** guard `llama_context`, `llama_batch`,
  `llama_decode`, `llama_memory_seq_*`, or
  `llama_get_logits_ith` access. The scripted submitter
  helper body still calls no `llama_*` API. Only the
  engine HPX task touches llama.cpp execution state.

  Safe claim (verbatim):

  ```text
  An externally arriving request admitted into a
  cancel-freed slot opens a fresh HPX local-channel
  stream, emits exactly rr.n_decoded token events, has
  streamed_hash == rr.hash, and closes with
  reason=completed without inheriting the cancelled
  previous occupant's stream state or close reason.
  ```

  The gate-side change adds:

  ```text
  per external-arrival admitted-streamed request
    (admission_src=cancel_freed, arrival_src=external):
      admitted_streamed_seen[rr.request_id]  == true
      admitted_streamed_close[rr.request_id] == completed
      admitted_streamed_tokens[rr.request_id].size()
                                              == rr.n_decoded
      streamed_hash                           == rr.hash
      rr.previous_request_id                  >= 0
      rr.reused_seq_id                        == prior occupant's seq_id
                                                  (the cancelled slot)
      rr.seq_id ∈ [0, n_active)               true
      admitted_streamed_tokens[rr.request_id]
                                    != streamed_tokens[rr.seq_id]
                                    (length-based inequality;
                                     not a semantic
                                     token-divergence claim)
  ```

  Coverage gate (new in Slice 6): when `--stream-all` AND
  `args.n_external_arrivals > 0` AND
  `!args.cancel_plan.empty()`, at least one streamed
  external-arrival admitted request with
  `admission_src == cancel_freed` must be witnessed;
  misconfigured smoke shapes fail-closed instead of
  silently proving nothing. The gate lives inside the
  enclosing `if (args.stream_all)` block so stream-off
  regression runs are unaffected. The Slice 3 / 4 / 5
  coverage gates are untouched.

  Slice 6 smoke shape (deterministic):

  ```text
  --stream-all
  --n-seqs 3
  --n-active 2
  --n-waiting 0
  --n-external-arrivals 1
  --decode-budget-mix 64,256
  --external-arrival-budget 16
  --external-release-iter 3
  --cancel-plan 0
  --cancel-after 8
  --repeat 2
  ```

  Expected per-request behavior:

  ```text
  request_id=0 (original active, seq_id=0,
                admission_src=none, arrival_src=preloaded)
    budget=64    status=cancelled  close=cancelled
    cancel_observed_iter=8         n_decoded_at_cancel=8
    streamed=8

  request_id=1 (original active, seq_id=1,
                admission_src=none, arrival_src=preloaded)
    budget=256   status=completed  close=completed
    streamed=256 hash=0x8790fbe5a60c9ae6

  request_id=2 (external arrival, seq_id=0,
                admission_src=cancel_freed,
                arrival_src=external,
                reused_seq_id=0,
                previous_request_id=0,
                admitted_at_iter=9       (== cancel_after+1))
    budget=16    status=completed  close=completed
    streamed=16  hash=0x833045f1e2ebf49f
                 done_iter=24  pos_max_at_clear=20
  ```

  Engine-side counter gates (Slice 6 smoke, `--stream-all`
  ON, per repeat):

  ```text
  streams_opened              == 3
                                (1 ctor cancelled
                               + 1 ctor surviving
                               + 1 external admit-rebind)
  streams_closed_completed    == 2     (req 1 + req 2)
  streams_closed_cancelled    == 1     (req 0)
  streams_closed_error        == 0
  stream_tokens_emitted_total == 280   (= 8 + 256 + 16)
  ```

  External / cancel metrics (per repeat):

  ```text
  admitted_count                == 1
  external_admitted_count       == 1
  arrival_drained_count         == 1
  first_external_drain_iter     == 4    (== external_release_iter + 1)
  iter_release_fired_set        == {3}
  submitter_ack_set             == {3}
  cancel_observed               == 1
  cancel_kv_cleared             == 1
  cancel_future_fulfilled       == 1
  request_admitted_live         == 1    (with
                                          admission_source=cancel_freed,
                                          arrival_source=external)
  seq_reused                    == 1
  admitted_prefilled            == 1
  ```

  Observed hashes on the smoke (Metal build):

  ```text
  budget  64  original cancelled              (no completion hash)
  budget 256  original completed              hash=0x8790fbe5a60c9ae6
  budget  16  external admitted cancel_freed
                                              hash=0x833045f1e2ebf49f
  ```

  The external admitted budget-16 hash equals the Slice 3
  / Slice 4 / Slice 5 admitted budget-16 hashes for the
  same prompt / policy / budget — expected under greedy
  decoding, **not** a new cross-shape canonical anchor.

  OFF-mode regression gate: with `--stream-all` OFF on the
  canonical Slice 7 admission shape every stream counter
  is exactly zero, the receiver vector is empty, and the
  admitted-stream handoff vector is empty; fail-closed
  otherwise.

  Hard scope of Streaming Slice 6:
  - **External-arrival admitted streaming over the
    cancel-freed path covered.** The external branch of
    `admit_one` now rebinds the stream channel when
    `--stream-all` is on and `src == cancel_freed`, in
    addition to the Slice 5 `completion_freed` case.
  - **External-arrival + completion-freed path remains
    covered (Slice 5 carry-over).** The optional Slice 5
    regression run on the Slice 5 smoke shape still passes
    after the Slice 6 predicate broadening.
  - **Preloaded completion-freed and cancel-freed admitted
    streaming remain covered (Slices 3 / 4 carry-overs).**
  - No HTTP / API / server / network streaming. The
    stream is in-process only between the engine HPX
    task and `main()`. The scripted submitter is an HPX
    task, not a network adapter.
  - No backpressure / bounded-channel policy.
    `channel<T>` is still unbounded so the engine never
    suspends on the consumer.
  - No multi-cycle slot reuse with streaming. A `seq_id`
    is still reused at most once per run.
  - No engine-failure stream smoke. The defensive
    `error` close path remains untested.
  - No tokenizer / prompt generalization (still
    `"Hello, my name is"` greedy on TinyLlama).
  - No per-request sampling configuration.
  - No new HPX primitive. Same
    `hpx::lcos::local::channel<token_stream_event>`,
    same `hpx::spinlock` for the existing inbox, same
    `hpx::promise<request_result>` /
    `hpx::future<request_result>`.
  - **No performance claim.** Streaming Slice 6 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP6: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 6 results — external-arrival streaming
  over cancel-freed slot* section.

- **Streaming Slice 7 (this version):** completion-freed
  multi-cycle slot reuse with streaming. Earlier streaming
  slices closed the width of the first-admission surface:
  original active completion, original active cancellation,
  preloaded admitted requests over `completion_freed` and
  `cancel_freed`, and external admitted requests over
  `completion_freed` and `cancel_freed`. Slice 7 closes the
  next depth case: one `seq_id` slot can host more than one
  admitted streamed request in the same engine run. The
  Slice 7 source change is validation-only — `admit_one`
  already rebound a fresh channel per admitted request, reset
  per-slot stream state, asserted KV-empty before rebind,
  cleared generated tokens, and reset hash state. Slice 7
  relaxes one-reuse-per-slot validation assumptions and
  replaces them with chain-aware validation. No new HPX
  primitive, no new `std::mutex`, no new CLI flag, no new
  trace event name, no new channel type, no CMake change.

  HPX-native design note: the stream substrate remains
  `hpx::lcos::local::channel<token_stream_event>`, stream
  payload is still token id plus close reason only, the
  engine HPX task remains the sole producer, `main` remains
  the consumer, and only the engine HPX task touches
  llama.cpp execution state.

  Slice 7 smoke shape (the design proposal used `--n-seqs 2`,
  but the CLI requires `n_active + n_waiting <= n_seqs`, so
  the smoke uses `--n-seqs 3`; only slot 0 is live at any
  moment):

  ```text
  --n-seqs 3
  --n-active 1
  --n-waiting 2
  --waiting-budget 8
  --decode-budget-mix 8
  --reuse-completed
  --cancel-plan none
  --stream-all
  --repeat 2
  ```

  Slot 0 hosts a three-occupant completion-freed chain
  (`request 0 -> request 1 -> request 2`) in one engine run.
  Expected per-request behavior:

  ```text
  request_id=0 (original active, seq_id=0,
                admission_src=none, arrival_src=preloaded)
    budget=8     status=completed close=completed
    done_iter=7  streamed=8  hash=0x0619d4d1900c2365

  request_id=1 (admitted, seq_id=0,
                admission_src=completion_freed,
                arrival_src=preloaded,
                reused_seq_id=0,
                previous_request_id=0,
                admitted_at_iter=8)
    budget=8     status=completed close=completed
    done_iter=15 streamed=8  hash=0x0619d4d1900c2365

  request_id=2 (admitted, seq_id=0,
                admission_src=completion_freed,
                arrival_src=preloaded,
                reused_seq_id=0,
                previous_request_id=1,
                admitted_at_iter=16)
    budget=8     status=completed close=completed
    done_iter=23 streamed=8  hash=0x0619d4d1900c2365
  ```

  All three occupants use the same prompt, model, greedy
  policy, and budget. Under those conditions equal token
  vectors and equal hashes are expected. Slice 7 therefore
  uses **Path alpha: structural independence**, not
  token-vector inequality, as the independence proof: fresh
  channel per admission, per-slot stream counter reset
  before rebind, KV-empty assertion before each bind, the
  `previous_request_id` chain walk, per-cycle
  `streamed_hash == rr.hash`, and independent stream
  open/close counters.

  Engine-side counter gates (Slice 7 smoke, `--stream-all`
  ON, per repeat):

  ```text
  streams_opened              == 3
  streams_closed_completed    == 3
  streams_closed_cancelled    == 0
  streams_closed_error        == 0
  stream_tokens_emitted_total == 24
  admitted_count              == 2
  external_admitted_count     == 0
  reused_seq_id_set           == {0,0}  (multiset; the
                                          duplicate is the
                                          multi-cycle evidence)
  ```

  OFF-mode regression: with `--stream-all` OFF on the
  canonical Slice 7 admission shape, every stream counter is
  exactly zero and the admitted-stream handoff vector is
  empty; fail-closed otherwise. The representative Slice 6
  regression also passes under the Slice 7 binary, confirming
  that the Slice 7 validation relaxations are conditional on
  the multi-cycle shape and do not weaken the previous
  external + cancel_freed single-cycle path.

  Hard scope of Streaming Slice 7:
  - **Completion-freed multi-cycle reuse covered.** A single
    `seq_id` slot can host a three-occupant chain in one
    engine run.
  - **Cancel-freed multi-cycle reuse is deferred.**
  - **External-arrival multi-cycle reuse is deferred.**
  - No engine-failure `reason=error` stream semantics.
  - No HTTP / gRPC / Unix-socket / WebSocket streaming.
  - No backpressure / bounded-channel policy.
  - No tokenizer / prompt generalization.
  - No per-request sampling configuration.
  - **No performance claim.** Streaming Slice 7 is a
    correctness/lifecycle gate.

  Final emit is `HPX_CB_STREAM_STEP7: PASS` / `FAIL: <reason>`.
  Closeout evidence is in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Streaming Slice 7 results — multi-cycle slot reuse with
  streaming* section.

See `docs/hpx/continuous_batching_prototype_design.md` for the
full slice plan, correctness gates, and out-of-scope list,
`docs/hpx/continuous_batching_live_admission_design.md` for the
live-admission-specific design,
`docs/hpx/continuous_batching_streaming_slice1_design.md` for
the Streaming Slice 1 design,
`docs/hpx/continuous_batching_streaming_slice2_design.md` for
the Streaming Slice 2 design,
`docs/hpx/continuous_batching_streaming_slice3_design.md` for
the Streaming Slice 3 design,
`docs/hpx/continuous_batching_streaming_slice4_design.md` for
the Streaming Slice 4 design,
`docs/hpx/continuous_batching_streaming_slice5_design.md` for
the Streaming Slice 5 design,
`docs/hpx/continuous_batching_streaming_slice6_design.md` for
the Streaming Slice 6 design, and
`docs/hpx/continuous_batching_streaming_slice7_design.md` for
the Streaming Slice 7 design.

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
  --stream-all              default: OFF
                            Streaming Slice 1: enable HPX-native
                            per-request token streaming via
                            hpx::lcos::local::channel<token_stream_event>.
                            Each bound active seq gets an
                            engine-owned channel; main holds the
                            matching receive_channel and drains
                            the chain after engine_fut.get(). OFF
                            preserves Slice 7 semantics.
```

Set `LLAMA_HPX_CB_TRACE=1` to enable HPX runtime startup/shutdown
trace lines plus the Slice-4 lifecycle events (`engine_start`,
`request_admitted`, `seq_prefilled`, `decode_row`, `seq_complete`,
`kv_cleared`, `promise_fulfilled`, `engine_stop`), the
Cancel-Slice-2 events (`cancel_requested`, `cancel_observed`,
`cancel_kv_cleared`, `cancel_future_fulfilled`), the Live
Admission Slice 4 events (`request_queued`,
`request_admitted_live`, `seq_reused`, `admitted_prefilled`,
`admitted_decode_row`, `admitted_complete`), the Live
Admission Slice 6 events (`request_submitted_external`,
`arrival_drained`, `iter_release_fired`,
`submitter_ack_observed`), and the Streaming Slice 1 events
(`token_stream_opened`, `token_stream_token`,
`token_stream_closed`) on stderr.

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
[hpx-cb-gate] event=token_stream_opened request=0 seq_id=0
[hpx-cb-gate] event=token_stream_token request=0 seq_id=0 pos=0 token=2259
[hpx-cb-gate] event=token_stream_closed request=0 seq_id=0 n_tokens=8 reason=completed
```

`decode_row`, `admitted_decode_row`, and `token_stream_token` are
high-cardinality (one event per row / per emitted token, per
decode iter). All trace output is gated on `LLAMA_HPX_CB_TRACE=1`
and is silent by default; the off-path is a single atomic load per
call site.

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

- `HPX_CB_STREAM_STEP7: PASS`

  or

- `HPX_CB_STREAM_STEP7: FAIL: <reason>`

`HPX_CB_STREAM_STEP7` is the seventh label in the streaming
gate sequence opened by `HPX_CB_STREAM_STEP1`, layered on top
of — and not replacing — the closed admission gate sequence at
`HPX_CB_ADMIT_STEP7: PASS`. Every Slice 1 … Slice 7
admission/cancellation/lifecycle invariant and every
Streaming Slice 1 + Streaming Slice 2 + Streaming Slice 3
+ Streaming Slice 4 + Streaming Slice 5 + Streaming Slice 6
streaming invariant remains a carry-over invariant under
`HPX_CB_STREAM_STEP7`.

Historical slice labels (kept for reference only — the binary now
emits the Streaming Slice 7 label):

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
- `HPX_CB_ADMIT_STEP7` — Live Admission Slice 7 (closing
  label of the admission gate sequence)
- `HPX_CB_STREAM_STEP1` — Streaming Slice 1 (opens the
  streaming gate sequence)
- `HPX_CB_STREAM_STEP2` — Streaming Slice 2
  (cancellation-aware streaming)
- `HPX_CB_STREAM_STEP3` — Streaming Slice 3
  (admitted-request streaming over completion-freed slot)
- `HPX_CB_STREAM_STEP4` — Streaming Slice 4
  (admitted-request streaming over cancel-freed slot)
- `HPX_CB_STREAM_STEP5` — Streaming Slice 5
  (external-arrival admitted streaming over completion-freed
  slot)
- `HPX_CB_STREAM_STEP6` — Streaming Slice 6
  (external-arrival admitted streaming over cancel-freed
  slot)
- `HPX_CB_STREAM_STEP7` — Streaming Slice 7 (current;
  completion-freed multi-cycle slot reuse with streaming)

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
