# Phase 5 extension — frozen_packet vs dag_nowrap / fused_async

Date: 2026-04-16
Commit: 116452bdb86182446bc9824506656761a0d47d7f (branch hpx-prefill-orchestrator)
Binary: build-hpx/bin/bench_hpx_rms_norm_f32
Host: darwin 25.3.0 (Apple BLAS, Release)

## What was measured

A new `frozen_packet` row was added to `bench_hpx_rms_norm_f32.cpp`. It exercises
the compiled 3-step RMS_NORM_F32 packet end-to-end:

- `ggml_hpx_compile_packet` once per (n, lanes) outside the timed loop
- caller-owned aligned frame allocated via `::operator new(size, align_val_t)`
- `bind + run_frozen_packet` inside the timed loop, using the same lane_ptrs /
  reduction buffer layout as the `dag_nowrap` row (byte-for-byte comparable)

`n_lanes == 1` promotes LANE_FANOUT → SERIAL at compile time, so frozen_packet
with lanes=1 performs zero HPX crossings per rep. lanes>1 performs exactly two
`hpx::experimental::for_loop` crossings per rep (R0 partial + R2 apply), on a
pinned `scheduler_executor` built once at runtime_create.

## Headline numbers (min_ns)

| n    | direct | frozen_packet(1) | frozen_packet(2) | frozen_packet(4) | dag_nowrap(1) | dag_nowrap(2) | dag_nowrap(4) | fused_async |
|------|--------|------------------|------------------|------------------|---------------|---------------|---------------|-------------|
| 512  | 250    | 459              | 4917             | 6125             | 7416          | 12375         | 12750         | 1000        |
| 2048 | 1125   | 1166             | 5958             | 6667             | 8958          | 13042         | 12416         | 3208        |
| 4096 | 2333   | 2250             | 6792             | 5625             | 10000         | 13875         | 13375         | 7500        |
| 8192 | 4916   | 4667             | 8709             | 9125             | 13709         | 14500         | 14875         | 10375       |

## Ratios (min_ns)

`frozen_packet / dag_nowrap` — the primary target:

| n    | lanes=1 | lanes=2 | lanes=4 |
|------|---------|---------|---------|
| 512  | 0.062   | 0.397   | 0.480   |
| 2048 | 0.130   | 0.457   | 0.537   |
| 4096 | 0.225   | 0.489   | 0.420   |
| 8192 | 0.340   | 0.601   | 0.613   |

`frozen_packet / direct` at lanes=1 — the sanity-check ratio:

| n    | direct | frozen_packet(1) | ratio |
|------|--------|------------------|-------|
| 512  | 250    | 459              | 1.84  |
| 2048 | 1125   | 1166             | 1.04  |
| 4096 | 2333   | 2250             | 0.96  |
| 8192 | 4916   | 4667             | 0.95  |

## Findings

1. **Primary goal met.** frozen_packet is faster than dag_nowrap in every
   (n, lanes) cell of the sweep, by 1.6× – 16×. The largest wins are at small
   n and low lanes, where the fine-region generic machinery's fixed cost is
   the dominant term.

2. **Lanes=1 sanity check: mostly passes, one outlier.** For n ≥ 2048,
   frozen_packet(lanes=1) matches `direct` within 5 %. At n=512 there is a
   ~200 ns fixed gap (459 vs 250). This is consistent with the step-loop
   dispatch cost (3 function-pointer calls through the `PacketStep` program)
   plus a debug assert in `run_frozen_packet`. It vanishes into the math at
   n ≥ 2048. Not a hidden HPX crossing — lanes=1 promotes to SERIAL-only at
   compile time, so the cost floor is pure C++ dispatch.

3. **fused_async is no longer the ceiling.** At lanes=1 frozen_packet is
   actually *faster* than fused_async (e.g. n=2048: 1166 vs 3208 ns), because
   lanes=1 avoids HPX entirely while fused_async still pays one `hpx::async`
   per rep. For lanes>1, frozen_packet pays two `for_loop` crossings per rep
   and lands between `direct` and `dag_nowrap`, much closer to the fused
   ceiling than the generic DAG path was.

4. **Step-loop overhead is small but real, and it is not scaffolding.**
   Even at lanes=1 with no HPX, there is ~200 ns of fixed overhead above
   `direct` at n=512. Verified against the Release build used here
   (`-O3 -DNDEBUG`): every `assert()` in `ggml-hpx-packet.cpp` comes from
   `<cassert>` and is already a no-op under NDEBUG. The gap is genuine
   dispatch cost — the step loop, frame-pointer indirection, and three
   function-pointer calls through `PacketStep.run_range`. Options if we
   ever want to tighten it:
     - inline the step loop for known-length sublayer packets
     - accept it — ~200 ns is a rounding error compared to the several-µs
       floor the old dag_nowrap path imposed, and it disappears into the
       math at n ≥ 2048

## Success criteria (from docs/bench-phase5-frozen-packet.md)

- ✅ frozen_packet faster than dag_nowrap across the entire sweep
- ✅ frozen_packet closer to fused_async than dag_nowrap — at lanes=1 it is
  strictly faster than fused_async; at lanes>1 the gap is much smaller than
  dag_nowrap's
- ⚠ frozen_packet(n, 1) ≈ direct(n) — holds for n ≥ 2048 (within 5 %); at
  n=512 there is a ~200 ns fixed gap. Confirmed in a Release (`-O3 -DNDEBUG`)
  build, so it is not debug scaffolding — it is real step-loop dispatch cost

## Open questions

- Is the n=512 gap worth tightening? With NDEBUG confirmed, the remaining
  cost is genuine dispatch, not scaffolding. Worth revisiting only if
  packet-sized work below ~2 K shows up in real decode traffic.
- p95 variance is very large for every HPX path (both dag_nowrap and
  frozen_packet spike to the 500 µs — 2 ms range in p95), while min/median
  are stable. Likely an OS noise effect, not a frozen_packet regression;
  dag_nowrap shows the same shape. Not chasing in this phase.

## Next step

RMS_NORM_F32 has served its purpose as the smallest repeated sublayer that
proves the model. The next interesting question is whether the packet model
holds up at a slightly larger granularity — e.g. a QKV-projection or an
MLP gate block — where a single packet spans more than three regions and
the dispatch-floor share of the overall cost is naturally smaller. The
architectural question at that scale is whether a multi-sublayer packet
changes the compile/bind/run surface in meaningful ways, not whether the
per-packet overhead is measurable in isolation.
