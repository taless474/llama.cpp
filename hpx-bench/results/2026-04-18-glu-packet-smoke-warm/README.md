# B.1 GLU packet smoke — warm-cache run (INVALID for comparison — see findings)

**Date:** 2026-04-18
**Commit:** hpx-prefill-orchestrator branch (post Item 5/6)
**Binary:** `build-hpx-dag/bin/llama-simple`
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf`
**Protocol:** warmup pass (-n 1, same flags) immediately before each measured pass (-n 16)

---

## Why this run is not a valid comparison

Two separate confounders make the results uninterpretable:

### 1. PACKET=0 thermal cliff

| token | p0 total ms |
|---|---|
| 1 | 873 |
| 2 | 368 |
| 3 | 484 |
| 4 | 635 |
| 5 | 811 |
| 6 | 1702 |
| 9+ | 4200–5400 |

PACKET=0 (`lowered=222` F32 matrix-ops per decode graph) progressively degrades from ~400 ms
to ~5000 ms over 15 tokens. This is thermal throttling: 222 HPX-lowered F32 MUL_MAT ops
sustain the M4 at full utilisation across all tokens until the chip throttles. The warmup
pass ran before this (adding more prior heat), making the cliff arrive earlier.

### 2. PACKET=1 HPX warmup cliff

| token | packet_ms |
|---|---|
| 1 | 3416 |
| 2 | 3107 |
| 3 | 2292 |
| 4–6 | 2158–2537 |
| 7 | 234 |
| 8–15 | 68–222 |

packet_ms starts at 3000+ ms for the first 7 tokens, then drops to 70–120 ms. Unknown
cause — likely a mix of HPX thread pool not yet at cruise speed AND thermal throttling
inherited from the PACKET=0 measured run (50 s of heavy compute) immediately before.

### 3. Sequential heat buildup

The runs execute in order: PACKET=0 warmup → PACKET=0 measured (50 s) → PACKET=1 warmup →
PACKET=1 measured. PACKET=1 measured starts with the chip already hot from PACKET=0's 50 s
of sustained parallel F32 work. The two runs see fundamentally different thermal states.

---

## What the data does confirm

All four correctness checks from the previous run hold:
- `packet=0(0 nodes)` on the prefill graph in both runs — `!batched` gate enforced ✓
- `packet=22(66 nodes)` on every decode graph in PACKET=1 — GLU prescan fires correctly ✓
- `lowered` drops 222 → 156 in PACKET=1 — 66 nodes transferred to packet ✓
- `fallback=467` unchanged — packet path does not disturb the fallback bucket ✓

---

## What a valid comparison requires

The F32 model (4.1 GB, 222 lowered ops per decode) is too heavy for back-to-back runs on
the M4 without thermal saturation. Options for the next attempt:

1. **Use the Q4_K_M model.** The MLP packet doesn't fire (quantised MUL_MAT rejected),
   but it establishes a stable decode baseline. Alternatively, find a smaller F32 model.

2. **Run each variant in isolation with a cooldown between them.** At minimum 30–60 s
   of idle time between PACKET=0 and PACKET=1 measured passes.

3. **Run longer and discard the first N tokens.** If the thermal cliff is at token ~6 for
   PACKET=0, a run of 64 tokens would still have 58 stable tokens before the cliff — but
   the cliff itself is the problem, not the sample size.

4. **Investigate the PACKET=1 HPX warmup cliff independently.** The drop from 3000 ms to
   70 ms in packet_ms over 7 tokens is a separate issue worth understanding — it is likely
   the HPX pool warmup dominating and unrelated to packet correctness.
