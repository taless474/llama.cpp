# multiseq-batch-gate

Smallest real `llama.cpp` technical gate before HPX integration.

This tool is a **correctness gate**, not a benchmark. Subsequent steps
(see `docs/hpx/multiseq_llama_batch_gate.md` Q7) build it up to a
multi-`seq_id` shared `llama_batch` decode.

## Status

- **Step 1 (done, now a prerequisite):** load model, create one
  `llama_context`, tokenize prompt, run three structural checks
  (`n_seq_max`, `n_ctx`, `n_batch`).
- **Step 2 (done, `--n-seqs 1`):** single-seq baseline. seq_id=0,
  one shared `llama_batch` (allocated via `llama_batch_init`,
  populated via `common_batch_add`), prefill with `logits=true` only
  on the last prompt row, then per-step greedy-argmax decode, with a
  deterministic FNV-1a 64-bit token hash that matches
  `tools/serving-bench/harness.h`. Verifies the canonical hash for
  `--decode-budget 8` (`0x0619d4d1900c2365`) and, when run with
  `--decode-budget 16`, the canonical 16-token TinyLlama fingerprint
  (`0x833045f1e2ebf49f`). The two hashes are shape-specific and must
  not be claimed for each other.
- **Step 3 (done, `--n-seqs >= 2` with uniform budget):** N seqs
  with the same prompt and same `--decode-budget`, sharing one
  `llama_context` and one shared `llama_batch`. Prefill rows for all
  seqs go in the same batch; each step's decode rows likewise share
  one batch. Verifies pairwise equality of generated token sequences
  and hashes, plus per-seq `llama_memory_seq_rm(..., -1, -1)` and a
  `pos_min == -1` sentinel check.
- **Step 4 (this version, `--decode-budgets <csv>`):** N seqs with
  the same prompt but **mixed** decode budgets (e.g. `8,64,256`).
  Each seq's KV is cleared as soon as it reaches its budget; the
  next decode iteration must omit cleared seqs. The gate verifies:
  - per-seq `n_decoded == budget`
  - pairwise prefix equality up to `min(budget_i, budget_j)`
  - finish-order monotonicity in `done_iter` w.r.t. budget
  - `pos_max_at_clear == P + budget - 2` per seq
  - cross-talk isolation: clearing seq `s` must not disturb any
    still-active seq's `pos_min`/`pos_max`
  - residual KV after the run: every seq reads `(-1, -1)`
- **Step 5 (this version, `--decode-budget-mix <csv>`):** scale-up
  to the Phase 3 primary target shape. Combine with `--n-seqs N` to
  round-robin the mix across N seqs, e.g.
  `--n-seqs 99 --decode-budget-mix 8,64,256` yields 33+33+33 seqs.
  Output is collapsed to a per-budget-class summary: budget, count,
  unique_hash_count, hash, expected/observed `done_iter` set,
  expected/observed `pos_max_at_clear` set. Per-seq lines are
  suppressed for `n_seqs > 8`. All Step 4 invariants (per-seq
  budgets, prefix equality, finish-order monotonicity, KV
  isolation, residual KV) still apply at scale, plus the hard gates
  "1 unique hash per budget class", "1 unique done_iter per class",
  "1 unique pos_max_at_clear per class".

## Build

The target is opt-in (default OFF) so default builds are unaffected.

```sh
cmake -S . -B <build_dir> -DLLAMA_BUILD_MULTISEQ_GATE=ON
cmake --build <build_dir> --target llama-multiseq-batch-gate
```

## Usage

```text
llama-multiseq-batch-gate --model <path> [options]

  --model <path>          (required) path to .gguf model file
  --prompt <string>       default: "Hello, my name is"
  --ctx-size <int>        default: 1024
  --n-seq-max <int>       default: 3
  --n-batch <int>         default: 512
  --n-threads <int>       default: 2
  --decode-budget <int>   default: 8
  --repeat <int>          default: 1
  --n-seqs <int>          default: 1
  --decode-budgets <csv>  per-seq budgets, e.g. "8,64,256". If given,
                          n_seqs = number of values and
                          --decode-budget is ignored.
  --decode-budget-mix <csv> mix to round-robin over --n-seqs, e.g.
                          "8,64,256" with --n-seqs 99 yields 33+33+33.
                          Triggers the GATE_STEP5 emit.
```

The final stdout line is either:

```
GATE_STEP<N>: PASS
```

or

```
GATE_STEP<N>: FAIL: <reason>
```

where `N` is the highest step the binary executed:

- 1 seq                          → `GATE_STEP2`
- N seqs, all budgets equal      → `GATE_STEP3`
- N seqs, mixed budgets          → `GATE_STEP4`
- `--decode-budget-mix` used     → `GATE_STEP5`

so an external script can grep for it.

## Structural prerequisites (from Step 1)

After init the binary verifies three things and fails closed if any
is violated. These remain prerequisites for every later step:

1. `llama_n_seq_max(ctx) >= --n-seq-max`
2. `llama_n_ctx(ctx) >= prompt_tokens + 256`
3. `llama_n_batch(ctx) >= --n-seq-max * prompt_tokens`

## KV position convention

For prompt length `P` and decode budget `D`, the per-seq KV state
immediately before clear is:

- `pos_min == 0`
- `pos_max == P + D - 2` (**not** `P + D - 1`)

Reason: generated token #1 comes from the prefill's final logits row
and does not occupy a new KV cell. Only `D - 1` decoded tokens are
fed back as decode rows, so KV holds `P + (D - 1)` cells at positions
`0 .. P + D - 2`.

## Step 2 hash semantics

- The token hash is FNV-1a 64-bit with the same constants and fold
  arithmetic as `tools/serving-bench/harness.h`
  (`k_token_hash_init = 0xcbf29ce484222325`,
  prime = `0x100000001b3`,
  `state ^= u32(int32_t(token_id)); state *= prime`).
  `n_decoded == 0` ⇒ hash = `0`.
- Tokenization uses `add_special=true, parse_special=true` to match
  serving-bench.
- End-of-generation tokens, if sampled, terminate the loop **without**
  being folded into the hash.
- Canonical references (TinyLlama 1.1B Q4_K_M, prompt
  `"Hello, my name is"`, greedy argmax):
  - `--decode-budget 8`  ⇒ `0x0619d4d1900c2365`
  - `--decode-budget 16` ⇒ `0x833045f1e2ebf49f`
  Each hash is shape-specific. A budget-8 run cannot produce the
  16-token canonical hash, and vice versa.

## Out of scope (for every step in this tool)

HPX, HTTP, streaming, cancellation, priority, prompt cache, LoRA,
speculative decoding, multimodal, `n_cmpl > 1`, distributed serving,
sampler chain, chat templates, performance comparisons, multi-backend
toggling, context shift, SWA.
