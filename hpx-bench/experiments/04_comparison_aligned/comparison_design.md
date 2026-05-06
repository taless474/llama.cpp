# Side-by-side comparison design — `llama-server` vs `llama-serving-bench std`

Snapshot date: 2026-05-04.
Branch: `hpx-run-level-analyzer`. HEAD: `63bae8a49`.
Status: **design only**. No experiment was run. No source was changed. No HPX was started. No concurrency was tested.

## Question

Can the completed `llama-server` timing smoke (`local/baselines/server_timing/`) and the completed `llama-serving-bench std` timing smoke (`local/baselines/serving_bench_std_timing/`) be compared to each other, and what must be aligned before any fair comparison?

The honest answer is: **not yet, not on wall-clock numbers.** The two runs share enough structure that hash- and token-level correctness signals already line up, but they do not share enough timing semantics for `wall_ms` and `total_ms` to be compared as if they measured the same thing. This note enumerates what is aligned, what is not, what the current descriptive numbers say, what would have to change before a fair comparison is meaningful, and what we are deliberately not doing yet.

## Sources used

Read-only references for this note:

```text
local/baselines/server_timing/                  # llama-server timing smoke (PASS)
local/baselines/serving_bench_std_timing/       # llama-serving-bench std timing smoke (PASS)
local/baselines/server_repeat/                  # llama-server cross-process determinism (PASS)
local/baselines/README.md                       # parent index
```

No additional commands, builds, or runs were performed for this note.

## 1. What is already aligned between the two timing smokes

These properties hold by construction across both `local/baselines/server_timing/` and `local/baselines/serving_bench_std_timing/`:

```text
model              tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  (same absolute path)
prompt             "Hello, my name is"                   (byte-equal string)
n_predict / max-tokens
                   16                                    (both runs)
generated tokens
                   16 in every successful request        (16/16 server, 16/16 bench)
prompt token count 6                                     (server: tokens_evaluated=6;
                                                          bench: "prompt fits: 6 prompt tokens")
ngl                0                                     (CPU-only on both;
                                                          server flag, bench hardcoded in backend_std.cpp)
slot / context     1                                     (server: -np 1 -nocb;
                                                          bench: --n-contexts 1 --n-concurrent 1)
continuous batching
                   disabled / not present                (server: -nocb; bench: not implemented)
prompt path        raw                                   (server: --no-jinja and chat_format=Content-only;
                                                          bench: prompt fed verbatim through llama_tokenize)
sampling           greedy / argmax                       (server: temperature=0.0, top_k=1;
                                                          bench: argmax(logits) in backend_std.cpp,
                                                          seed-base ignored by std backend)
warmup + measured shape
                   1 warmup + 5 measured                 (both runs explicitly)
session            single                                (server: one HTTP server session;
                                                          bench: one harness process)
deterministic output
                   yes                                   (server: byte-equal content across measured;
                                                          bench: identical token-id hash across all 6)
correctness gates  PASS                                  (both summary.txt: OVERALL: PASS)
```

The strongest cross-binary signal we already have is **prompt-token parity**: both binaries independently tokenize `"Hello, my name is"` against the same model and report 6 prompt tokens. That is byte-equal evidence at the tokenizer layer.

The strongest within-binary signal is **determinism**:

- `llama-server` produced byte-equal generated text across all 5 measured requests, and across the two fresh server sessions in `local/baselines/server_repeat/`.
- `llama-serving-bench std` produced an identical FNV-1a 64-bit token-id hash (`0x833045f1e2ebf49f`) across all 6 requests in this run, and the same hash also appears in `local/bench_repeat/A_std_r1.stdout` from the earlier local matrix.

## 2. What is not aligned — and why this blocks a wall-clock comparison

These are real differences. Each one would have to be either eliminated or explicitly normalized before the two timing numbers can be compared as if they meant the same thing.

### 2.1 Generated text vs token hash

`llama-server` returns generated text in the response body and we gate on byte-equality of that text against a canonical string. `llama-serving-bench` does not emit generated text — it emits a 64-bit FNV-1a fold over generated token IDs (`generated_token_hash` in `harness.h:23-30`). These are different signals over different surfaces:

- text equality includes the detokenizer path,
- hash equality only proves the same token-ID sequence was produced, with no statement about how those IDs would render.

Both are valid correctness signals. They are not directly substitutable for cross-binary content equality.

### 2.2 HTTP-client wall time vs harness-internal `total_ms`

This is the core wall-clock mismatch:

- `llama-server`'s `wall_ms` is measured in `_run_requests.py` from `time.monotonic()` immediately before `urllib.request.urlopen` to immediately after `resp.read()`. It includes JSON serialization, HTTP framing, kernel socket round-trip, and HTTP response parsing on the loopback interface, on top of the actual generation work.
- `llama-serving-bench`'s `total_ms` is measured inside `backend_std.cpp:run_one` from `t_submit` (set by `engine_std::submit` immediately on enqueue) to the moment the last decode completes. It excludes nothing inside the same process but includes nothing outside it.

So `server_timing.wall_ms` and `serving_bench_std_timing.total_ms` measure overlapping but non-identical time spans. A direct numerical comparison silently attributes HTTP/loopback overhead to the std backend's slowness (or vice versa).

### 2.3 Different binaries and code paths

Even on the same model, prompt, and decoding policy, the two binaries take different paths to the same logits:

- `llama-server` runs through the upstream server `slot` machinery, including its own thread-pool sizing, KV management policy, request/response queueing, and per-request state-reset logic.
- `llama-serving-bench std` runs through `backend_std.cpp`'s minimal in-process queue: a single worker thread per context, `llama_memory_clear(ctx, false)` per request, batch construction via `llama_batch_get_one`, and pure `argmax` sampling.

These are not the same code paths. Performance differences are expected and are not necessarily the experiments' fault.

### 2.4 Possibly different thread behavior

`llama-server` chooses its thread count via its own logic (the value is whatever the server's defaults resolve to — neither `server_timing` nor this design note has captured it explicitly). `backend_std::init` chooses `n_threads = hardware_concurrency / n_workers` when `--n-threads 0` is passed; with `n_workers = 1`, that resolves to `hardware_concurrency` (`backend_std.cpp:86-94`). These two heuristics are unlikely to land on the same number on this machine, and decode latency is sensitive to the chosen thread count. Until both runs pin or print the actual number used, this difference is silent.

### 2.5 Possibly different timing interval definitions

Even within a single binary, "submit" and "completion" mean different things:

- For `llama-server`, "submit" is implicit at HTTP request acceptance; the server-reported `timings.predicted_ms` and `timings.prompt_ms` cover the prompt-eval and generation phases inside the slot, not the full HTTP-server-side wall.
- For `llama-serving-bench`, `ttft_ms` is submit-to-first-logits and `total_ms` is submit-to-last-decode — both purely steady-clock inside the engine, and `submit` is the moment `engine_std::submit` is called (queueing happens after).

Mapping `server_timing.timings.predicted_ms` (mean ≈ 146.570 ms) to `serving_bench_std_timing.total_ms - ttft_ms` (mean ≈ 247.619 − 39.053 = 208.566 ms) is closer to apples-to-apples than `wall_ms` vs `total_ms`, but it is still not exact: the server's `predicted_ms` is the inner decode time as measured by the server, while the harness's `total_ms - ttft_ms` includes any synchronization/post-processing after first-token. A meaningful comparison needs an explicit definition.

### 2.6 Possibly different cache / memory clear semantics

- `llama-server` is run with `cache_prompt=false`. The server's per-request behavior under that flag is "do not reuse a previously-cached prompt prefix"; this is consumed by the slot but not echoed back in `generation_settings` (we already documented this in `local/baselines/server_repeat/`).
- `llama-serving-bench std` calls `llama_memory_clear(llama_get_memory(ctx), false)` at the start of every `run_one` (`backend_std.cpp:213`).

Functionally both should produce a fresh KV cache state per request for a single-slot, single-context configuration, but the implementation paths are not identical. We have not verified bit-equivalence at the memory layer; we have only verified that the produced outputs are deterministic in each binary, separately.

### 2.7 Different observability surfaces

- `llama-server` exposes Prometheus metrics: `prompt_tokens_total`, `tokens_predicted_total`, `n_decode_total`, plus average gauges. We gate these in `server_timing` with exact deltas (36 / 96 / 96).
- `llama-serving-bench` exposes per-request fields (`ttft_ms`, `total_ms`, `n_tokens_generated`, `generated_token_hash`) and an aggregate block (`n_ok`, `n_cancelled`, `n_error`, `wall`, `agg_tok/s`, p50/p95/p99 percentiles, `per_req_tps_cv`).

There is no common observability schema. Mapping between the two requires an explicit translation table; nothing automatic is correct.

## 3. What the current numbers say — descriptively, not comparatively

These numbers are the **measured-only stats** from each smoke. They are recorded here for completeness. They are not a comparison.

```text
llama-server timing smoke           (local/baselines/server_timing/summary.txt)
  measured client wall_ms       n=5  mean ≈ 173.353  ms     stdev_pop ≈ 1.126   ms
  measured client tokens/sec    n=5  mean ≈  92.301  tok/s  stdev_pop ≈ 0.600   tps
  measured server predicted_ms  n=5  mean ≈ 146.570  ms     stdev_pop ≈ 0.129   ms
  measured server prompt_ms     n=5  mean ≈  26.150  ms     stdev_pop ≈ 1.097   ms

llama-serving-bench std timing smoke (local/baselines/serving_bench_std_timing/summary.txt)
  measured total_ms             n=5  mean ≈ 247.619  ms     stdev_pop ≈ 35.005  ms
  measured tokens_per_second    n=5  mean ≈  65.928  tok/s  stdev_pop ≈  9.316  tps
  measured ttft_ms              n=5  mean ≈  39.053  ms     stdev_pop ≈  7.218  ms
```

**Do not treat any of this as a performance conclusion.** In particular:

- The 173 ms vs 248 ms gap is not "std backend is slower than `llama-server`". It is "two different binaries measured two different intervals on two different code paths with possibly different thread counts, on a small sample, on the same machine, and the numbers happen to look like that."
- The 92 tok/s vs 66 tok/s gap is the same story, derived from those same intervals.
- The relatively high `stdev_pop` on `total_ms` and `tokens_per_second` for the bench (35 ms, 9.3 tps) compared to `llama-server`'s very tight `predicted_per_second` (≈ 0.10 tps) is also not a verdict — it could be wholly explained by the bench's `total_ms` measuring a longer span that includes more kernel-thread scheduling jitter, by a different thread count, or by sampling noise on n=5.

Both runs `OVERALL: PASS` against their own gates. That is what these smokes are for. They are correctness baselines.

## 4. What a fair comparison would require next

To turn the two existing artifact sets into a defensible side-by-side, the following alignments would have to be made explicit. None of them are difficult; none of them have been done yet. Each is its own small slice with its own approval.

### 4.1 Pin and print thread counts on both sides

- For `llama-server`, pass `-t N` explicitly and capture the value to the artifacts.
- For `llama-serving-bench std`, pass `--n-threads N` explicitly with the same `N`.

Until the thread count matches, the comparison is silently confounded. The simplest choice is whatever the std backend's `--n-threads 0` heuristic landed on in this run (we should grep `bench.stderr` for `engine_std ready: ... n_threads_per_ctx=...` to read the actual value), and pass that same value to `llama-server`.

### 4.2 Decide which interval(s) to compare

Three plausible options, in order of increasing strictness:

1. **Client wall-only**: wrap the harness invocation in an outer steady-clock measurement on the client side, and compare to `server_timing`'s client `wall_ms`. This makes both numbers "what an external caller observes." Cheapest. Adds one number to the bench artifacts.
2. **Server-internal generation only**: compare `server_timing.timings.predicted_ms` to `serving_bench_std_timing.total_ms - ttft_ms`, with the explicit caveat that "generation" is defined slightly differently in each. Requires no new code. Requires writing the mapping into the comparison artifact.
3. **Both**: report client wall and inner generation side by side, with an explicit translation block. Most honest. Most artifact volume.

Option 3 is the least likely to produce a misleading number. Option 2 is the cheapest defensible.

### 4.3 Make the content signal cross-binary

Today, `llama-server` reports text and `llama-serving-bench` reports a token-ID hash. Two ways to bridge:

- Add `return_tokens=true` to the `llama-server` request body (it already supports this). The response will include `tokens: [...]`. Compute the same FNV-1a fold on those token IDs in the post-processing helper and assert it equals `0x833045f1e2ebf49f`. No source change.
- Or, on the bench side, write a small helper that detokenizes from a saved token-ID list. Requires a harness change; out of scope here.

The first option is the cheaper one and yields a real cross-binary equality check.

### 4.4 Optional: outer wall-clock around the harness

If interval choice (1) is taken, add a thin `time.monotonic()` wrap around the harness subprocess invocation in `_run_bench.py` and record `process_wall_ms` per invocation. This is one number per process, not per request — but with `--n-requests 6`, it is comparable to `5 × server_timing.wall_ms + warmup` and gives a defensible upper-bound comparison.

### 4.5 Preserve every existing structural alignment

Anything in §1 that is currently aligned must stay aligned in the comparison run:

```text
same model, same prompt, same n_predict=16
CPU-only, single slot/context
no continuous batching
raw prompt path, deterministic greedy output
1 warmup + 5 measured
single session per binary
correctness gates retained verbatim
```

### 4.6 Preserve the existing correctness gates verbatim

The comparison run does not relax any existing gate from either smoke. It adds gates:

- thread count printed and matches across both binaries
- (if §4.3 is taken) `llama-server` token-ID FNV-1a hash equals `serving-bench` token-ID FNV-1a hash
- intervals declared explicitly in the comparison artifact's README

If any of these added gates fail, the comparison stops and reports, the same way the existing smokes do.

## 5. Decision

For now:

- **No HPX comparison until the std-vs-server comparison basis above is defined and an aligned run has been executed.** Comparing HPX against an undefined baseline is strictly worse than comparing it against a defined one.
- **No concurrency comparison yet.** Both completed smokes are single-slot / single-concurrent. Concurrency adds queue-depth, batch-pressure, and KV-management dynamics that the current artifact set cannot pin down.
- **No source changes for this comparison design.** Everything in §4 can be done with helper-script changes only, except possibly §4.3, which only requires flipping a request-body flag on the `llama-server` side.

This note is the design checkpoint. It does not authorize or schedule any of §4. The next slice, when approved, would be to pick one option from §4.2 and one option from §4.3, sketch the helper-script delta, and stop again before any run.

## What this note explicitly is not

- not a benchmark
- not a performance verdict
- not an HPX comparison or HPX dry-run
- not a concurrency study
- not an authorization to run anything
- not a re-run or modification of any existing artifact under `local/baselines/`

## Next action

Wait for explicit approval before:

- choosing a comparison-interval option from §4.2,
- choosing a content-signal option from §4.3,
- writing any helper or harness script that would produce a comparison run.

Until that approval, this note stands as the comparison-readiness reference. The two source experiments (`server_timing/` and `serving_bench_std_timing/`) remain valid as their own correctness baselines, on their own terms.
