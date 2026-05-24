# Results — Experiment 13 control-plane responsiveness

All numbers are microseconds (µs), measured records only (warmups
excluded). These are control-plane **latency** measurements, not
throughput or correctness claims.

This doc has two phases. **Phase 1** (serial W3 stream drain) is preserved
below as accepted reference. **Phase 2** (concurrent foreign-thread W3
stream drain) re-measures W3 only and is appended at the end; it does not
overwrite Phase 1.

---

# Phase 1 — serial W3 drain (accepted, reference)

## Run identity

```text
run_id        : 20260523-143446-phase1
git SHA        : 7c0242201d12ff5d96344af2d4865fe1703e0156   branch: hpx-run-level-analyzer   dirty: true
binary         : builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-responsiveness-bench
model          : tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  (668,788,096 bytes)
model sha256   : 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
modes          : default_os1, default_os2, engine_pool_os2
workloads      : w2, w3
trials         : 30   warmup-trials: 1   n-streams: 4   decode-budget: 8
overall_ok     : PASS (6/6 cells)
```

(`dirty: true` reflects the in-progress branch worktree, including the
new experiment-13 harness; the bench binary itself was built from the
accepted engine source.)

## Cell status

| cell | rc | PASS line | measured |
|------|----|-----------|----------|
| default_os1_w2 | 0 | PASS | 30 |
| default_os1_w3 | 0 | PASS | 120 |
| default_os2_w2 | 0 | PASS | 30 |
| default_os2_w3 | 0 | PASS | 120 |
| engine_pool_os2_w2 | 0 | PASS | 30 |
| engine_pool_os2_w3 | 0 | PASS | 120 |

## W2 — queued-cancel (µs)

| mode | metric | N | p50 | p95 | p99 | max | mean |
|------|--------|---|-----|-----|-----|-----|------|
| default_os1 | cancel_to_observed_us | 30 | 61.5 | 89.8 | 105.5 | 111 | 65.5 |
| default_os1 | cancel_to_future_ready_us | 30 | 64.0 | 91.8 | 107.5 | 113 | 68.0 |
| default_os1 | cancel_to_stream_close_us | 30 | 64.0 | 92.2 | 107.5 | 113 | 68.3 |
| default_os2 | cancel_to_observed_us | 30 | 65.0 | 111.3 | 122.5 | 126 | 71.2 |
| default_os2 | cancel_to_future_ready_us | 30 | 70.0 | 117.8 | 129.2 | 133 | 78.0 |
| default_os2 | cancel_to_stream_close_us | 30 | 71.0 | 118.3 | 130.2 | 134 | 78.5 |
| engine_pool_os2 | cancel_to_observed_us | 30 | 61.0 | 69.6 | 82.8 | 88 | 62.5 |
| engine_pool_os2 | cancel_to_future_ready_us | 30 | 66.0 | 75.6 | 88.8 | 94 | 68.2 |
| engine_pool_os2 | cancel_to_stream_close_us | 30 | 67.0 | 76.0 | 89.5 | 95 | 68.6 |

Queued-cancel resolves in **tens of microseconds** across all modes. The
cancel-observe → future-ready → stream-close chain is monotonic and tight
(≈2–4 µs between the three stages). No mode exceeds ~135 µs at max.

## W3 — multi-request streaming, K=4 (µs)

| mode | metric | N | p50 | p95 | p99 | max | mean |
|------|--------|---|-----|-----|-----|-----|------|
| default_os1 | submit_to_admitted_us | 120 | 36043 | 40515 | 41286 | 41286 | 24467 |
| default_os1 | submit_to_first_token_us | 120 | 99324 | 105414 | 106670 | 106835 | 78703 |
| default_os1 | submit_to_complete_us | 120 | 299259 | 312462 | 314755 | 315180 | 296524 |
| default_os1 | submit_to_stream_close_us | 120 | 299015 | 312257 | 314470 | 314916 | 296203 |
| default_os1 | inter_token_gap_us | 840 | 1 | 32314 | 64609 | 65169 | 9414 |
| default_os2 | submit_to_admitted_us | 120 | 6 | 35856 | 38057 | 38058 | 10474 |
| default_os2 | submit_to_first_token_us | 120 | 37802 | 100171 | 100507 | 100597 | 54198 |
| default_os2 | submit_to_complete_us | 120 | 275819 | 287368 | 295639 | 295681 | 271627 |
| default_os2 | submit_to_stream_close_us | 120 | 275692 | 287275 | 295640 | 295682 | 271508 |
| default_os2 | inter_token_gap_us | 840 | 1 | 29527 | 64467 | 65030 | 8598 |
| engine_pool_os2 | submit_to_admitted_us | 120 | 33466 | 38050 | 38095 | 38096 | 21993 |
| engine_pool_os2 | submit_to_first_token_us | 120 | 93787 | 99742 | 100008 | 100047 | 73041 |
| engine_pool_os2 | submit_to_complete_us | 120 | 284231 | 296558 | 296939 | 297076 | 283232 |
| engine_pool_os2 | submit_to_stream_close_us | 120 | 284132 | 296449 | 296864 | 297004 | 283109 |
| engine_pool_os2 | inter_token_gap_us | 840 | 1 | 29788 | 61637 | 62022 | 9027 |

W3 latencies are dominated by **decode time**: completion sits at
~270–300 ms for 4 streams × 8 tokens regardless of mode. Admission and
first-token are **bimodal** — some of the K requests are admitted on the
first engine iteration (~µs) and the rest one decode iteration later
(~33–38 ms) — so the median reflects which side of that split a mode
lands on, not a smooth latency. `inter_token_gap_us` p50 = 1 µs is a
drain artifact (see caveats).

## Engine-pool vs default_os2

Δ = engine_pool_os2 − default_os2 (negative = engine_pool faster).

W2 (cancel path) — engine_pool tightens the tail:

| metric | Δp50 | Δp95 | Δp99 |
|--------|------|------|------|
| cancel_to_observed_us | −4 | **−41.7** | **−39.7** |
| cancel_to_future_ready_us | −4 | **−42.2** | **−40.5** |
| cancel_to_stream_close_us | −4 | **−42.3** | **−40.7** |

W3 (streaming under load) — engine_pool does **not** help, slightly worse:

| metric | Δp50 | Δp95 | Δp99 |
|--------|------|------|------|
| submit_to_admitted_us | +33460 | +2194 | +38 |
| submit_to_first_token_us | +55985 | −428 | −498 |
| submit_to_complete_us | +8412 | +9191 | +1300 |
| inter_token_gap_us | 0 | +261 | −2830 |

**Reading:** the named engine pool helps exactly where the control plane
is the only work — the W2 cancel path, where it cuts p95/p99 by ~40 µs
and produces a markedly tighter, more predictable tail than default_os2.
On W3, decode dominates and engine-task placement has little leverage:
median admission/first-token are worse for engine_pool (more requests
fall on the later-iteration side of the bimodal admission split),
completion is ~8–9 ms slower at the median/p95, and the tails converge.
Net: **engine-pool placement is a control-plane-tail win (W2), not a
streaming-throughput win (W3)** — consistent with HPX owning
orchestration while llama.cpp owns execution.

## Caveats

- TinyLlama Q4_K_M on Apple M4; single machine; timings include OS
  scheduling jitter. Treat tails as indicative, not contractual.
- W2 requests never decode (cancelled while queued); W2 measures the
  cancel path only.
- W3 `inter_token_gap_us` p50 = 1 µs is a **measurement artifact**: the
  bench submits all K streams, then drains each serially. By the time a
  given stream is drained, its tokens are already buffered in the
  channel, so consecutive `get()`s return near-instantly. The gap
  reflects consumer drain speed, not engine production cadence; the
  p95/p99 (~29–65 ms) appear where the consumer catches up to production.
  True per-token production cadence needs a concurrent multi-consumer
  drain (a later-phase design decision).
- W3 admission/first-token are **bimodal** (iter-0 vs next-iteration
  admission), so medians are split-position-dependent, not smooth.
- Microsecond stamps are `steady_clock`; absolute values are not wall
  time and are only meaningful as differences.
- This run does not assert token hashes; correctness fingerprints come
  from the gate tools, not this responsiveness harness.

---

# Phase 2 — concurrent foreign-thread W3 drain

Phase 2 re-measures **W3 only**, draining the K streams concurrently with
one foreign `std::thread` per consumer (`drain_mode="concurrent"`). It
fixes the Phase 1 `inter_token_gap_us` ~1 µs serial-drain artifact. W2 was
not re-run (unchanged code; the Phase 1 W2 result stands). No engine,
cancellation, or stream-close-ordering change.

## Run identity

```text
run_id        : 20260523-145138-phase2
git SHA        : 7c0242201 (worktree dirty: Phase 2 bench + harness edits)
binary         : builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-responsiveness-bench
model          : tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  sha256 9fecc3b3…2776a0
workload       : w3   drain_mode: concurrent
modes          : default_os1, default_os2, engine_pool_os2
trials         : 30   warmup-trials: 1   n-streams: 4   decode-budget: 8
overall_ok     : PASS (3/3 cells, 120 measured each)
```

Validation gates (all passed): build of engine + bench + gate; W2-unchanged
smoke 3/3 PASS across modes; W3 tiny validation 3/3 PASS; per-record
invariants over 36 records — 0 problems (every request `completed`;
`len(token_receive_us)==n_decoded`; per-request monotonic; terminal close
observed; `drain_mode=="concurrent"`); artifact-gone check passed.

## Headline: inter-token receive cadence is now real

W3 `inter_token_gap_us` p50, Phase 1 (serial) vs Phase 2 (concurrent):

| mode | Phase 1 p50 | Phase 2 p50 | Phase 2 min |
|------|------|------|------|
| default_os1 | 1 µs (artifact) | **30023 µs** | 15467 |
| default_os2 | 1 µs (artifact) | **29292 µs** | 7799 |
| engine_pool_os2 | 1 µs (artifact) | **29479 µs** | 7950 |

The ~1 µs buffer-drain artifact is gone. The honest cadence is ~29–30 ms
p50 and is **essentially mode-independent** — consistent with the engine
producing one token per stream per decode iteration, so the gap is set by
decode-iteration time, which engine-task placement does not change.

## W3 Phase 2 — concurrent drain (µs)

| mode | metric | N | p50 | p95 | p99 | max | mean |
|------|--------|---|-----|-----|-----|-----|------|
| default_os1 | submit_to_admitted_us | 120 | 39369 | 40493 | 40537 | 40539 | 27412 |
| default_os1 | submit_to_first_token_us | 120 | 102876 | 104977 | 105416 | 105537 | 83490 |
| default_os1 | submit_to_complete_us | 120 | 305925 | 310582 | 311477 | 312084 | 299391 |
| default_os1 | submit_to_stream_close_us | 120 | 305654 | 310118 | 311123 | 311719 | 299159 |
| default_os1 | inter_token_gap_us | 840 | 30023 | 32939 | 63840 | 65144 | 30807 |
| default_os2 | submit_to_admitted_us | 120 | 5 | 35741 | 39576 | 39577 | 9069 |
| default_os2 | submit_to_first_token_us | 120 | 37879 | 100008 | 101129 | 101176 | 52008 |
| default_os2 | submit_to_complete_us | 120 | 275402 | 284936 | 297065 | 297273 | 266772 |
| default_os2 | submit_to_stream_close_us | 120 | 275224 | 284832 | 297018 | 297133 | 266648 |
| default_os2 | inter_token_gap_us | 840 | 29292 | 60149 | 64563 | 64747 | 30662 |
| engine_pool_os2 | submit_to_admitted_us | 120 | 8 | 38308 | 38484 | 38485 | 16572 |
| engine_pool_os2 | submit_to_first_token_us | 120 | 38227 | 100970 | 102091 | 102532 | 64185 |
| engine_pool_os2 | submit_to_complete_us | 120 | 281389 | 300371 | 310642 | 310831 | 280265 |
| engine_pool_os2 | submit_to_stream_close_us | 120 | 281353 | 300290 | 310454 | 310734 | 280139 |
| engine_pool_os2 | inter_token_gap_us | 840 | 29479 | 60336 | 65115 | 66005 | 30849 |

## Engine-pool vs default_os2 (Phase 2 W3)

Δ = engine_pool_os2 − default_os2 (positive = engine_pool slower).

| metric | Δp50 | Δp95 | Δp99 |
|--------|------|------|------|
| submit_to_admitted_us | +3 | +2567 | −1092 |
| submit_to_first_token_us | +348 | +962 | +962 |
| submit_to_complete_us | +5988 | +15434 | +13577 |
| inter_token_gap_us | +188 | +187 | +552 |

**Reading:** with the honest concurrent measurement, the Phase 1 W3
conclusion **holds and is now trustworthy**: engine-pool placement does
not improve W3 streaming responsiveness. Admission, first-token, and
inter-token cadence are near-identical to `default_os2`; completion is
~6 ms slower at the median and ~13.5 ms slower at p99 for engine_pool.
Decode dominates W3; the named pool's benefit is confined to the pure
control-plane path (W2 cancel tail, Phase 1). No decode-speed or
throughput claim is made.

## Phase 2 caveats

- `inter_token_gap_us` is **adapter/consumer-observed** receive cadence:
  engine production + HPX channel wakeup + OS scheduling + foreign-thread
  consumer wake latency. It is not pure engine `publish_token()` cadence,
  by design — this benchmark measures the client/adapter-visible side.
- Foreign-thread wake and thread-spawn (~tens of µs) are folded into
  receive times; the first gap per stream can be small if a token was
  buffered before its consumer started. The p50 over all gaps is the
  robust figure (~29–30 ms), not the per-stream first gap.
- `default_os1` runs the engine + 4 consumer threads + main; consumers
  mostly block, but oversubscription jitter is real — its admission p50
  (~39 ms) reflects the contended single-HPX-worker case.
- W3 admission/first-token remain **bimodal** (iter-0 vs next-iteration);
  medians are split-position dependent.
- Same machine/model/steady_clock caveats as Phase 1; no token-hash
  assertions.
