# Aligned comparison — `llama-server` vs `llama-serving-bench std`

This directory operationalizes §4.1–§4.4 of
`04_comparison_aligned/comparison_design.md` by carrying the smallest
possible helper-script delta over the two existing per-binary PASS smokes:

```text
local/baselines/server_timing/
local/baselines/serving_bench_std_timing/
```

It does not modify those existing artifact directories. It does not modify any
source under `tools/serving-bench/` or `tools/server/`.

## Question

Can the completed upstream `llama-server` timing smoke and the completed
`llama-serving-bench --backend std` timing smoke be compared under a tighter,
explicitly aligned protocol?

This experiment asks a narrower question:

```text
After pinning both binaries to the same thread count and bridging the content
signal through token IDs, do the two binaries produce the same generated token
sequence, and do their closest available inner-generation timing intervals line
up for this single-request CPU-only greedy shape?
```

This is **not** an HPX comparison. It is a comparison-basis validation step for
the std/upstream baselines.

## Interval declaration

interval pair under comparison: llama-server timings.predicted_ms vs llama-serving-bench std (total_ms - ttft_ms)

This single line is the §4.2 commitment for this experiment. The cross-binary
summarizer (`_summarize_comparison.py`) gates on the verbatim presence of this
string in this file.

## Decisions resolved

```text
§4.1 thread parity         pin both binaries to n_threads = 4
                           server  : -t 4
                           bench   : --n-threads 4
                           rationale: matches llama-server's natural default on
                           this machine. The bench's previous default of 10 was
                           the silent confounder §4.1 warned about.

§4.2 interval choice       Option 4.2.2 — server timings.predicted_ms vs
                           bench (total_ms - ttft_ms). No new in-process timing
                           code on either binary.

§4.3 content-signal bridge Option A — set "return_tokens": true on the server
                           request body, recover body.tokens, fold with the same
                           FNV-1a 64-bit routine as tools/serving-bench/harness.h,
                           and assert equality against the bench's
                           generated_token_hash.

§4.4 outer wall            informational only. process_wall_ms.txt is captured
                           around the bench subprocess and an outer-wall line is
                           included in the cross-binary summary. NOT gated.
```

## Shape

### Common shape

```text
model                 /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt                "Hello, my name is"
generated tokens      16
prompt tokens         6
CPU-only              yes
single slot/context   yes
concurrency           1
continuous batching   disabled / not present
sampling              greedy / argmax
warmup policy         1 warmup request, excluded from measured timing stats
measured requests     5
```

### `llama-server` side

```text
binary                /Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-server
directory             local/baselines/comparison_aligned/server/
endpoint              /completion
raw prompt path       --no-jinja
thread count          -t 4
request addition      return_tokens=true
interval used         timings.predicted_ms
```

### `llama-serving-bench std` side

```text
binary                /Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-serving-bench
directory             local/baselines/comparison_aligned/bench/
backend               --backend std
thread count          --n-threads 4
interval used         total_ms - ttft_ms
```

## Layout and artifacts

```text
local/baselines/comparison_aligned/
  README.md
  _summarize_comparison.py
  comparison_summary.txt

  server/
    _run_requests.py
    _summarize_timing.py
    health.json
    server.pid
    server_start.stdout
    server_start.stderr
    metrics_before.txt
    metrics_after.txt
    request_warmup.{headers,json,curl_meta.txt}
    request_measured_{1..5}.{headers,json,curl_meta.txt}
    per_repeat.csv
    request_summary.txt
    summary.txt
    server_shutdown.txt
    port_released.txt

  bench/
    _run_bench.py
    _summarize_timing.py
    bench.stdout
    bench.stderr
    bench.exit_code.txt
    per_repeat.csv
    request_summary.txt
    summary.txt
    process_wall_ms.txt
    binary_used.txt
    build_info.txt
    git_head.txt
```

Important schema additions:

```text
server/per_repeat.csv:
  token_id_count
  token_id_hash

bench/per_repeat.csv:
  total_minus_ttft_ms
```

## Acceptance gates

### Per-binary gates

Server side:

```text
server/summary.txt ends with OVERALL: PASS
HTTP 200 for all requests
JSON parsed for all requests
content stable
tokens_evaluated == 6
tokens_predicted == 16
metrics deltas match expected token counts
clean shutdown
port released
```

Bench side:

```text
bench/summary.txt ends with OVERALL: PASS
harness exit code == 0
all 6 request lines parsed
n_ok=6 n_cancelled=0 n_error=0
all status=ok
all n_tokens_generated=16
hash stability across all requests
prompt-fits line present
no error / failed diagnostics
```

### Cross-binary gates

```text
comparison_summary.txt ends with OVERALL: PASS
server thread count == 4
bench thread count == 4
server token_id_hash matches bench generated_token_hash for all measured requests
warmup hashes also match
canonical hash is 0x833045f1e2ebf49f
server token_id_count == 16 for measured requests
README declares the interval pair explicitly
new CSV schema columns are present
old artifact directories were not modified
```

## Result

All gates passed:

```text
server summary:      OVERALL: PASS
bench summary:       OVERALL: PASS
comparison summary:  OVERALL: PASS
```

Thread parity:

```text
server stderr: system_info: n_threads = 4
bench stderr:  n_threads_per_ctx = 4
```

Canonical hash:

```text
0x833045f1e2ebf49f
```

Cross-binary token hash result:

```text
server measured (n=5, n_distinct=1): 0x833045f1e2ebf49f
bench measured  (n=5, n_distinct=1): 0x833045f1e2ebf49f
server warmup:                       0x833045f1e2ebf49f
bench warmup:                        0x833045f1e2ebf49f
```

Every measured request and both warmups, on both binaries, produced the same
canonical token hash.

## Interval table

Measured requests only:

```text
interval pair:
  llama-server timings.predicted_ms
  vs
  llama-serving-bench std (total_ms - ttft_ms)

server timings.predicted_ms:
  min=145.454 ms
  max=146.543 ms
  mean=146.076 ms
  median=146.007 ms
  stdev_pop=0.411 ms

bench total_ms - ttft_ms:
  min=145.329 ms
  max=146.247 ms
  mean=145.736 ms
  median=145.816 ms
  stdev_pop=0.322 ms
```

Mean difference for the declared interval pair:

```text
146.076 ms - 145.736 ms = 0.340 ms
```

This gap is inside the per-side population standard deviations for this small run.

## Informational outer-wall numbers

```text
server: 861.871 ms summed across 5 measured client wall_ms
bench:  1320.214 ms process wall
```

These outer-wall numbers are **not** part of the gate set and are not treated as
directly comparable. The server number excludes server startup/shutdown. The bench
number includes model load, the warmup request, and harness teardown inside the
same process.

## Interpretation

This run validates the comparison basis for this narrow shape.

After aligning thread count and bridging the content signal through token IDs:

```text
llama-server and llama-serving-bench std produce the same generated token IDs.
```

The closest available inner-generation timing intervals also line up tightly:

```text
server predicted_ms mean:        146.076 ms
bench total_ms - ttft_ms mean:   145.736 ms
```

This strongly suggests that the earlier apparent gap between `llama-server`
client wall time and `serving-bench std` total time was a measurement-interval /
thread-count artifact, not evidence of a backend-speed difference.

What this establishes:

```text
cross-binary token-ID equality for this prompt
thread-count parity
a declared inner-generation interval pair
per-binary correctness gates still pass
a small descriptive timing record for the aligned interval
```

What this does not establish:

```text
a general performance result
a benchmark-grade conclusion
HPX behavior
concurrency behavior
behavior on other models, prompts, token counts, quantizations, or machines
exact equivalence between predicted_ms and total_ms - ttft_ms
fairness of outer-wall numbers
```

## Caveats

This is still a small smoke:

```text
n=5 measured requests
single machine
single model
single prompt
n_predict=16
CPU-only
single context / single slot
single concurrent request
greedy decoding
no HPX
no concurrency
```

The interval pair is explicitly declared but not perfect. These are the closest
available inner-generation-style intervals without source changes, not guaranteed
identical measurement spans.

The bench `seed-base` is recorded but not behaviorally important here because the
std backend uses argmax. Determinism is structural, not sampler-driven.

