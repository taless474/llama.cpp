# `llama-serving-bench std` timing smoke — `local/baselines/serving_bench_std_timing/`

This is HPX-OFF baseline discovery, not an HPX comparison.

## Question

Can the upstream `llama-serving-bench` harness, built std-only (HPX backend compiled as a stub via `LLAMA_SERVING_BENCH_HPX=OFF`), run the same deterministic 1-warmup + 5-measured shape we just validated for `llama-server`, on the same model and prompt, with the same correctness-first gates, and produce a small honest timing record?

The goal is to validate the std harness baseline on its own terms — token-id hash stability, prompt-token parity with `llama-server`, no errors, clean exit — before any HPX comparison is attempted. It is the smallest run that exercises this binary end-to-end.

## Shape

```text
binary           /Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-serving-bench
backend          std (LLAMA_SERVING_BENCH_HPX=OFF; HPX backend compiled as stub)
model            /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt           "Hello, my name is"
n-contexts       1
n-concurrent     1
n-requests       6   (req[0] = warmup, req[1..5] = measured)
max-tokens       16
ctx-size         2048
batch-size       512
n-threads        0   (auto: hardware_concurrency / n_workers)
seed-base        1234   (std uses argmax(logits), so seed has no observable effect)
sampler path     pure greedy via argmax in backend_std.cpp
sessions         1 process; repeats inside the harness
PID lookup       not needed (single foreground subprocess)
```

Ordering, as executed:

```text
1. _run_bench.py invokes the binary once with the argv above.
2. Captures bench.stdout, bench.stderr, bench.exit_code.txt.
3. Parses per-request stdout lines into per_repeat.csv (header + 6 rows).
4. Writes request_summary.txt, binary_used.txt, git_head.txt, build_info.txt.
5. _summarize_timing.py reads per_repeat.csv + bench.stdout + bench.stderr +
   bench.exit_code.txt, runs the gate list, computes measured-only stats, and
   writes summary.txt.
```

Per-request stdout line shape (from `tools/serving-bench/main.cpp:189-194`):

```text
[serving-bench] req[N] status=ok n_tokens_generated=16 \
  generated_token_hash=0x... ttft_ms=... total_ms=...
```

Aggregate line (from `tools/serving-bench/harness.cpp:117-118`):

```text
[serving-bench] n_ok=N n_cancelled=N n_error=N
```

Prompt-fits line on stderr (from `tools/serving-bench/main.cpp:129-131`):

```text
[serving-bench] prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size
```

## Artifacts

```text
_run_bench.py                          # invokes harness once, writes csv + provenance
_summarize_timing.py                   # reads csv + raw output, runs gates, writes summary.txt
bench.stdout                           # full harness stdout (per-req lines + aggregate)
bench.stderr                           # full harness stderr (config + prompt-fits + diagnostics)
bench.exit_code.txt                    # subprocess exit code
per_repeat.csv                         # header + 6 data rows
request_summary.txt                    # one line per parsed request
summary.txt                            # full stats + gates + OVERALL
binary_used.txt                        # absolute path, size, sha256
build_info.txt                         # build dir + serving-bench flags + binary path
git_head.txt                           # rev-parse, log -1 --oneline, status --short
configure_serving_bench.stdout/stderr  # cmake reconfigure logs (LLAMA_BUILD_SERVING_BENCH=ON)
build_serving_bench.stdout/stderr      # cmake --build logs for the llama-serving-bench target
```

`per_repeat.csv` columns:

```text
req_index, kind, status, n_tokens_generated,
generated_token_hash, ttft_ms, total_ms, tokens_per_second
```

`tokens_per_second = n_tokens_generated / (total_ms / 1000)`.

## Acceptance gates

All 11 gates passed. Source of truth: `summary.txt`.

```text
[PASS] harness exit code == 0
[PASS] all 6 req lines parsed
[PASS] stdout aggregate line contains: 'n_ok=6 n_cancelled=0 n_error=0'
[PASS] all 6 status == ok
[PASS] all 6 n_tokens_generated == 16
[PASS] req[1..5] generated_token_hash all identical
[PASS] req[0] generated_token_hash equals measured hash
[PASS] stderr contains: 'prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size'
[PASS] stderr has no '[serving-bench] error'
[PASS] stderr has no 'failed'
[PASS] per_repeat.csv has 6 data rows + header
```

## Result

```text
OVERALL: PASS
6/6 rows parsed
aggregate: n_ok=6 n_cancelled=0 n_error=0
all 6 status=ok
all 6 n_tokens_generated=16
prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size
warmup hash = measured hash = 0x833045f1e2ebf49f
```

Per-request rows:

```text
req[0] kind=warmup   status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=179.492 total_ms=356.518 tps=44.879
req[1] kind=measured status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=37.780  total_ms=246.215 tps=64.984
req[2] kind=measured status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=53.019  total_ms=293.325 tps=54.547
req[3] kind=measured status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=35.216  total_ms=206.752 tps=77.387
req[4] kind=measured status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=32.401  total_ms=280.250 tps=57.092
req[5] kind=measured status=ok n_tok=16 hash=0x833045f1e2ebf49f ttft_ms=36.850  total_ms=211.551 tps=75.632
```

Measured statistics (req[1..5], n=5):

```text
ttft_ms             n=5 min=32.401   max=53.019   mean=39.053   median=36.850   stdev_pop=7.218    ms
total_ms            n=5 min=206.752  max=293.325  mean=247.619  median=246.215  stdev_pop=35.005   ms
tokens_per_second   n=5 min=54.547   max=77.387   mean=65.928   median=64.984   stdev_pop=9.316    tps
```

Token-hash stability:

```text
warmup hash:   0x833045f1e2ebf49f
measured hash: 0x833045f1e2ebf49f   (n_distinct=1)
warmup hash equals measured hash:   PASS
```

The hash also matches the value already recorded in the prior local matrix (`local/bench_repeat/A_std_r1.stdout`: `0x833045f1e2ebf49f`), giving cross-run hash continuity for this prompt + greedy decoding configuration.

## Interpretation

This establishes a small HPX-OFF `llama-serving-bench` `std` timing smoke for the single-context, single-concurrent, raw prompt path.

It establishes hash stability across all 6 requests (1 warmup + 5 measured) and prompt-token parity with the prior `llama-server` baseline (`6 prompt tokens` reported by both binaries on the same prompt) for this harness configuration.

It does not establish HPX comparison, concurrency behavior, or benchmark-grade performance.

It also confirms the corrected protocol — invoke the harness once with `--n-requests 6`, treat `req[0]` as warmup in the analysis layer, gate `req[0..5]` for status/token-count parity and `req[1..5]` for hash stability and measured stats — produces a clean, parsable artifact set that mirrors `local/baselines/server_timing/` in shape while honoring the std harness's signal differences (hash instead of byte-equal text, harness-internal `total_ms`/`ttft_ms` instead of HTTP-client wall time).

## Caveats

- Small smoke (n=5 measured). Reported `stdev_pop` values are descriptive of this run, not a statistical claim about general behavior.
- `total_ms` and `ttft_ms` are harness-internal timings: submit-to-first-token and submit-to-completion as measured by `steady_clock` inside `backend_std::run_one`. There is no separate "client wall_ms" because the harness is the client.
- The `tokens_per_second` value reported here is a derived quantity, `n_tokens_generated / (total_ms / 1000)`, not the harness's own `agg_tok/s` (which is computed over the full wall and is reported separately on stdout).
- Single host, single session, single model, single prompt, CPU-only, single context, single in-flight request, raw greedy-argmax decoding. No concurrency. No batching pressure. No HPX.
- The `seed-base 1234` value is recorded for completeness; the std backend uses pure argmax over logits and does not consult the seed, so its outputs are seed-independent. The hash gate is therefore a structural-determinism gate, not a sampler-determinism gate.
- The observed `total_ms` mean is higher than the earlier `llama-server` client `wall_ms` mean, but **this is not an apples-to-apples comparison and should not be interpreted as std-backend slowness**. The two binaries measure different things: `llama-server` measures wall time as seen by an HTTP client over a localhost socket, while `llama-serving-bench` measures harness submit-to-completion inside the same process. Stack overhead, thread-count heuristics, transport, and timing boundaries all differ. A meaningful side-by-side comparison would require its own protocol and is explicitly out of scope here.
- This is not a benchmark. No claim about std-backend or HPX-backend performance is made by this experiment.

