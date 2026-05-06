# `llama-server` timing smoke — `local/baselines/server_timing/`

This is HPX-OFF baseline discovery, not an HPX comparison.

## Question

Can a single, unmodified upstream `llama-server` session, configured for the same deterministic raw `/completion` shape used in the prior single-request and repeatability smokes, give us a stable client-side and server-side timing record for 5 measured requests, with token-count and content stability and Prometheus metrics deltas matching expectation?

The goal is a small, honest timing record — not a benchmark. It is the smallest run that exercises the timing protocol end-to-end so the protocol itself can be trusted before being reused for comparison work.

## Shape

```text
binary               /HPX/builds/llama-base/bin/llama-server
model                /HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
host:port            127.0.0.1:18083
sessions             1 (one server start, one shutdown)
warmup               1 request (excluded from measured statistics, included in HTTP/JSON-parse checks and metrics deltas)
measured             5 requests
flags                -ngl 0  -np 1  -nocb  --no-jinja  --metrics  --seed 1234
path                 POST /completion
prompt               "Hello, my name is"
n_predict            16
temperature          0.0
top_k                1
seed                 1234
cache_prompt         false
stream               false
PID lookup           lsof -i :18083 -sTCP:LISTEN -t
shutdown             SIGINT, then wait until lsof reports the port released
```

Ordering, as executed:

```text
1. Start llama-server.
2. Health check (GET /health → 200, {"status":"ok"}).
3. Capture metrics_before.txt (GET /metrics).
4. Run _run_requests.py
   - sends 1 warmup + 5 measured POST /completion requests
   - writes per-request headers, JSON, and curl_meta artifacts
   - writes per_repeat.csv (header + 6 rows)
   - writes request_summary.txt
5. Capture metrics_after.txt (GET /metrics).
6. Run _summarize_timing.py
   - reads per_repeat.csv
   - reads each request_*.json
   - reads metrics_before.txt and metrics_after.txt
   - writes summary.txt with per-request rows, measured stats, and metrics deltas
   - emits PASS/FAIL checks
7. SIGINT to PID from server.pid.
8. Verify port 18083 is no longer listening (port_released.txt).
```

## Artifacts

```text
_run_requests.py                         # warmup + 5 measured POST /completion
_summarize_timing.py                     # CSV + JSON + metrics → summary.txt + checks
health.json                              # /health response captured before any request
server.pid                               # listener PID captured from lsof at startup
server_start.stdout                      # llama-server stdout (empty)
server_start.stderr                      # llama-server stderr, includes shutdown trace
metrics_before.txt                       # /metrics before any request (all counters 0)
metrics_after.txt                        # /metrics after all 6 requests
request_warmup.headers                   # raw response headers per request
request_warmup.json                      # raw response body per request
request_warmup.curl_meta.txt             # status, wall_ms, tp, te per request
request_measured_{1..5}.headers
request_measured_{1..5}.json
request_measured_{1..5}.curl_meta.txt
per_repeat.csv                           # 1 header row + 6 data rows
request_summary.txt                      # one-line summary per request
summary.txt                              # full timing + checks summary
server_shutdown.txt                      # ps -p <pid> after SIGINT (process gone)
port_released.txt                        # lsof -i :18083 -sTCP:LISTEN -t (empty)
```

`per_repeat.csv` columns:

```text
label, kind, http_status, wall_ms, client_tokens_per_sec,
tokens_predicted, tokens_evaluated,
prompt_ms, predicted_ms, prompt_per_second, predicted_per_second,
content_len_chars
```

## Acceptance gates

All gates from the protocol passed. Source of truth: `summary.txt`.

```text
[PASS] all 6 requests HTTP 200
[PASS] all 6 JSON responses parse
[PASS] 5/5 measured content byte-equal to canonical content
[PASS] 5/5 measured tokens_evaluated == 6
[PASS] 5/5 measured tokens_predicted == 16
[PASS] llamacpp:prompt_tokens_total delta == 36
[PASS] llamacpp:tokens_predicted_total delta == 96
[PASS] llamacpp:n_decode_total delta == 96   (counter exposed by this build)
[PASS] per_repeat.csv has 6 data rows + header
[PASS] summary.txt exists
[PASS] all artifacts under local/baselines/server_timing/
[PASS] server shutdown clean (SIGINT → port released, ps -p <pid> empty)
```

Canonical content (from prior single-request and repeatability smokes):

```text
" John Smith. I am a software engineer at XYZ Company. I have"
```

## Result

Token-count stability (all 6 requests):

```text
tokens_evaluated  = 6   for every request
tokens_predicted  = 16  for every request
content           = canonical, byte-equal, for every measured request
```

Client wall_ms (5 measured):

```text
n=5  min=171.881  max=174.611  mean=173.353  median=173.381  stdev_pop=1.126   ms
warmup wall_ms = 192.553 ms (excluded from measured stats)
```

Client tokens/sec (predicted / wall, 5 measured):

```text
n=5  min=91.632  max=93.088  mean=92.301  median=92.282  stdev_pop=0.600   tok/s
```

Server-reported timings (5 measured):

```text
prompt_ms              n=5  min=24.825   max=27.617   mean=26.150   median=26.037   stdev_pop=1.097    ms
predicted_ms           n=5  min=146.403  max=146.728  mean=146.570  median=146.589  stdev_pop=0.129    ms
prompt_per_second      n=5  min=217.257  max=241.692  mean=229.845  median=230.441  stdev_pop=9.605    tok/s
predicted_per_second   n=5  min=109.045  max=109.287  mean=109.163  median=109.149  stdev_pop=0.096    tok/s
```

Prometheus `/metrics` deltas (after − before, all matched expected):

```text
llamacpp:prompt_tokens_total      before=0  after=36  delta=36  expected=36
llamacpp:tokens_predicted_total   before=0  after=96  delta=96  expected=96
llamacpp:n_decode_total           before=0  after=96  delta=96  expected=96
```

Shutdown:

```text
SIGINT to PID from server.pid
ps -p <pid>  → exit 1, no row (process gone)
lsof -i :18083 -sTCP:LISTEN -t  → exit 1, empty (port released)
server_start.stderr tail: "update_slots: all slots are idle" → "operator(): cleaning up before exit..."
```

## Interpretation

This run establishes that, for this exact deterministic raw `/completion` shape on this account, a fresh `llama-server` session:

- accepts 6 sequential requests cleanly,
- produces byte-equal content for every measured request (matches the canonical content already established by the single-request and repeatability smokes),
- produces token counts that are exactly stable across all 6 requests,
- produces Prometheus counter deltas that match `6 prompt tokens × 6 requests = 36` and `16 predicted × 6 = 96` and `16 decode steps × 6 = 96` exactly,
- shuts down cleanly under SIGINT and releases its listening port.

It also establishes that the corrected ordering — capture `metrics_before.txt` before any request, run the request helper, capture `metrics_after.txt` after, then summarize — works as a protocol. Metrics-delta verification happens in `_summarize_timing.py` after both snapshots exist, so the request helper never needs to compute deltas itself.

It is the first run on this branch that produces a small, honest client-side and server-side timing record for `llama-server` under the deterministic single-slot, no-continuous-batching, no-jinja path.

## Caveats

- This is a small smoke (n=5 measured). The reported `stdev_pop` values are descriptive of this run, not a statistical claim about general behavior.
- The number `predicted_per_second ≈ 109.16 tok/s` is a property of this specific binary, this specific TinyLlama Q4_K_M model, and this specific machine under these flags. It is not a general `llama-server` claim.
- Single host, single session, single model, single prompt, CPU-only, single slot, continuous batching disabled, raw `/completion`, no jinja, greedy sampling. No concurrency. No batching pressure.
- This is not an HPX comparison. No HPX-on build was started or queried. No serving-bench numbers are referenced or compared here. No claim that HPX is faster, slower, or equivalent is made by this experiment.
- This is not an apples-to-apples reference for `llama-serving-bench`. `llama-parallel` and `llama-serving-bench` use different scaffolding and different sequence shapes.
- Warmup wall_ms (192.553 ms) is ~19 ms higher than the measured mean (173.353 ms). The protocol kept warmup as a row in `per_repeat.csv` and counted it for HTTP/JSON-parse and metrics-delta gates, but excluded it from measured statistics and content-stability gates.
- `n_decode_total` is exposed by this build, so the `delta = 96` gate is real here. On builds where it is not exposed, the summarizer treats it as skipped rather than failing.
- No system clock anomalies were observed; client wall-clock timing used `time.monotonic()`.

