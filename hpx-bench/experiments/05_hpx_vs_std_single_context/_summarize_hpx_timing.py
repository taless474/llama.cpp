"""
Per-binary llama-serving-bench summarizer for the std-vs-HPX comparison.

Default target is the hpx/ subdirectory (--backend hpx artifacts produced by
_run_hpx.py). The same script can be invoked with a single CLI argument to
target hpx_off_regression/ instead (--backend std artifacts produced by
_run_hpx_off_regression.py); the std-timing gates are identical and the
lifecycle gates are inverted (zero trace lines expected at --backend std even
with LLAMA_SERVING_BENCH_HPX_TRACE=1 set).

Usage:
  python3 _summarize_hpx_timing.py                    # default: hpx/
  python3 _summarize_hpx_timing.py hpx                # explicit
  python3 _summarize_hpx_timing.py hpx_off_regression # regression variant

Reads (under <variant>/):
  per_repeat.csv         (header + 6 data rows: req[0]=warmup, req[1..5]=measured)
  bench.stdout           (per-request lines + aggregate line)
  bench.stderr           (config dump + prompt-fits line + diagnostics)
  bench.exit_code.txt    (subprocess exit code)
  hpx_trace.txt          (filtered HPX lifecycle / pool lines)

Writes:
  <variant>/summary.txt  (per-row table, measured stats, gates, OVERALL PASS/FAIL, caveats)

Std timing gates (applied in both variants):
  - harness exit code == 0
  - all 6 req lines parsed
  - stdout aggregate line: n_ok=6 n_cancelled=0 n_error=0
  - all 6 status == ok
  - all 6 n_tokens_generated == 16
  - req[1..5] generated_token_hash all identical
  - req[0] generated_token_hash equals measured hash
  - stderr contains prompt-fits line
  - stderr has no '[serving-bench] error'
  - stderr has no 'failed'
  - per_repeat.csv has 6 data rows + header

HPX lifecycle gates (variant=hpx, positive form):
  - exactly one '[serving-bench] hpx_runtime_start_once: starting (os_threads=1)'
  - exactly one '[serving-bench] engine_hpx ready: n_contexts=1 pool_size=1'
  - exactly one '[serving-bench] hpx_runtime_stop: stopping'
  - for each i in 0..5: exactly one 'req[i] acquire ctx=0' and one 'req[i] release ctx=0'
  - for each i in 0..5: acquire line precedes release line

HPX lifecycle gate (variant=hpx_off_regression, inverted form):
  - hpx_trace.txt contains zero matching trace lines

OVERALL: PASS iff every gate above passes.
"""
import csv
import re
import sys
from pathlib import Path
from statistics import mean, median, pstdev

BASE = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_hpx_vs_std")

VALID_VARIANTS = ("hpx", "hpx_off_regression")
variant = sys.argv[1] if len(sys.argv) > 1 else "hpx"
if variant not in VALID_VARIANTS:
    print(f"error: variant must be one of {VALID_VARIANTS}; got {variant!r}",
          file=sys.stderr)
    sys.exit(2)

ROOT = BASE / variant

EXPECTED_N_TOKENS         = 16
EXPECTED_PROMPT_FITS_LINE = "prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size"
EXPECTED_AGG_LINE         = "n_ok=6 n_cancelled=0 n_error=0"

START_LINE   = "[serving-bench] hpx_runtime_start_once: starting (os_threads=1)"
READY_LINE   = "[serving-bench] engine_hpx ready: n_contexts=1 pool_size=1"
STOP_LINE    = "[serving-bench] hpx_runtime_stop: stopping"
ACQUIRE_RE   = re.compile(r"^\[serving-bench\] req\[(\d+)\] acquire ctx=0$")
RELEASE_RE   = re.compile(r"^\[serving-bench\] req\[(\d+)\] release ctx=0$")
ANY_TRACE_RE = re.compile(
    r"^\[serving-bench\] (?:"
    r"hpx_runtime_start_once|"
    r"engine_hpx ready|"
    r"hpx_runtime_stop|"
    r"req\[\d+\] acquire|"
    r"req\[\d+\] release"
    r")"
)


def stats(xs, unit=""):
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} min={min(xs):.3f}{unit} max={max(xs):.3f}{unit} "
        f"mean={mean(xs):.3f}{unit} median={median(xs):.3f}{unit} "
        f"stdev_pop={(pstdev(xs) if len(xs) > 1 else 0.0):.3f}{unit}"
    )


# ---- read inputs ----
exit_code_text = (ROOT / "bench.exit_code.txt").read_text().strip()
try:
    exit_code = int(exit_code_text)
except ValueError:
    exit_code = -1

stdout = (ROOT / "bench.stdout").read_text()
stderr = (ROOT / "bench.stderr").read_text()

trace_path = ROOT / "hpx_trace.txt"
trace_text = trace_path.read_text() if trace_path.exists() else ""
trace_lines_all = [ln for ln in trace_text.splitlines() if ln.strip() != ""]

rows = []
with (ROOT / "per_repeat.csv").open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

warmup_rows   = [r for r in rows if int(r["req_index"]) == 0]
measured_rows = [r for r in rows if 1 <= int(r["req_index"]) <= 5]

# ---- std timing gates ----
checks = []

checks.append(("harness exit code == 0", exit_code == 0))
checks.append(("all 6 req lines parsed", len(rows) == 6))
checks.append((f"stdout aggregate line contains: {EXPECTED_AGG_LINE!r}",
               EXPECTED_AGG_LINE in stdout))

status_ok = (len(rows) == 6) and all(r["status"] == "ok" for r in rows)
checks.append(("all 6 status == ok", status_ok))

n_tokens_ok = (len(rows) == 6) and all(
    int(r["n_tokens_generated"]) == EXPECTED_N_TOKENS for r in rows
)
checks.append((f"all 6 n_tokens_generated == {EXPECTED_N_TOKENS}", n_tokens_ok))

measured_hashes = {r["generated_token_hash"] for r in measured_rows}
hash_stable     = (len(measured_rows) == 5) and (len(measured_hashes) == 1)
checks.append(("req[1..5] generated_token_hash all identical", hash_stable))

if measured_hashes and warmup_rows:
    warmup_hash   = warmup_rows[0]["generated_token_hash"]
    measured_hash = next(iter(measured_hashes))
    warmup_eq     = warmup_hash == measured_hash
else:
    warmup_eq = False
checks.append(("req[0] generated_token_hash equals measured hash", warmup_eq))

checks.append((f"stderr contains: {EXPECTED_PROMPT_FITS_LINE!r}",
               EXPECTED_PROMPT_FITS_LINE in stderr))
checks.append(("stderr has no '[serving-bench] error'",
               "[serving-bench] error" not in stderr))
checks.append(("stderr has no 'failed'", "failed" not in stderr))
checks.append(("per_repeat.csv has 6 data rows + header", len(rows) == 6))

# ---- lifecycle gates (variant-specific) ----
if variant == "hpx":
    n_start = sum(1 for ln in trace_lines_all if ln == START_LINE)
    n_ready = sum(1 for ln in trace_lines_all if ln == READY_LINE)
    n_stop  = sum(1 for ln in trace_lines_all if ln == STOP_LINE)

    checks.append((f"exactly one start line: {START_LINE!r}", n_start == 1))
    checks.append((f"exactly one ready line: {READY_LINE!r}", n_ready == 1))
    checks.append((f"exactly one stop line: {STOP_LINE!r}",  n_stop  == 1))

    acquire_idx = {}  # req_index -> [position-in-trace, ...]
    release_idx = {}
    for pos, ln in enumerate(trace_lines_all):
        m_a = ACQUIRE_RE.match(ln)
        m_r = RELEASE_RE.match(ln)
        if m_a:
            acquire_idx.setdefault(int(m_a.group(1)), []).append(pos)
        elif m_r:
            release_idx.setdefault(int(m_r.group(1)), []).append(pos)

    for i in range(6):
        a = acquire_idx.get(i, [])
        r = release_idx.get(i, [])
        checks.append((f"req[{i}] exactly one acquire ctx=0", len(a) == 1))
        checks.append((f"req[{i}] exactly one release ctx=0", len(r) == 1))
        precedes = (len(a) == 1 and len(r) == 1 and a[0] < r[0])
        checks.append((f"req[{i}] acquire precedes release", precedes))

elif variant == "hpx_off_regression":
    matching_traces = [ln for ln in trace_lines_all if ANY_TRACE_RE.match(ln)]
    checks.append(("hpx_trace.txt contains zero HPX lifecycle/pool lines",
                   len(matching_traces) == 0))

# ---- measured stats ----
ttft_ms  = [float(r["ttft_ms"])           for r in measured_rows]
total_ms = [float(r["total_ms"])          for r in measured_rows]
tps      = [float(r["tokens_per_second"]) for r in measured_rows]

# ---- summary text ----
out = []
out.append(f"=== llama-serving-bench {variant} timing summary "
           f"(comparison_hpx_vs_std/{variant}) ===")
out.append("")
out.append(f"binary exit code: {exit_code}")
out.append("requests: 1 warmup (req[0]) + 5 measured (req[1..5])")
out.append(f"variant:  {variant}")
out.append("")
out.append("--- per-request rows ---")
for r in rows:
    out.append(
        f"req[{r['req_index']}] kind={r['kind']:<8} status={r['status']} "
        f"n_tok={r['n_tokens_generated']} hash={r['generated_token_hash']} "
        f"ttft_ms={float(r['ttft_ms']):.3f} "
        f"total_ms={float(r['total_ms']):.3f} "
        f"tps={float(r['tokens_per_second']):.3f}"
    )
out.append("")
out.append("--- measured ttft_ms ---")
out.append(stats(ttft_ms, "ms"))
out.append("--- measured total_ms ---")
out.append(stats(total_ms, "ms"))
out.append("--- measured tokens_per_second ---")
out.append(stats(tps, "tps"))
out.append("")
out.append("--- token hash ---")
if warmup_rows:
    out.append(f"warmup hash:   {warmup_rows[0]['generated_token_hash']}")
if measured_hashes:
    out.append(
        f"measured hash: {next(iter(measured_hashes))}  "
        f"(n_distinct={len(measured_hashes)})"
    )
out.append("")
out.append("--- HPX trace lines (filtered) ---")
out.append(f"trace_lines_total: {len(trace_lines_all)}")
if variant == "hpx":
    out.append(f"start lines: {sum(1 for ln in trace_lines_all if ln == START_LINE)}")
    out.append(f"ready lines: {sum(1 for ln in trace_lines_all if ln == READY_LINE)}")
    out.append(f"stop lines:  {sum(1 for ln in trace_lines_all if ln == STOP_LINE)}")
    out.append(
        f"acquire lines: {sum(1 for ln in trace_lines_all if ACQUIRE_RE.match(ln))}"
    )
    out.append(
        f"release lines: {sum(1 for ln in trace_lines_all if RELEASE_RE.match(ln))}"
    )
elif variant == "hpx_off_regression":
    out.append("(expected zero trace lines at --backend std)")
out.append("")
out.append("--- checks ---")
overall = True
for label, ok in checks:
    flag = "PASS" if ok else "FAIL"
    if not ok:
        overall = False
    out.append(f"[{flag}] {label}")
out.append("")
out.append(f"OVERALL: {'PASS' if overall else 'FAIL'}")
out.append("")
out.append("--- caveats ---")
out.append("This is one bench-side leg of the std-vs-HPX comparison.")
out.append("Cross-binary gates live in comparison_hpx_vs_std/comparison_summary.txt.")
out.append("No concurrency. Single context, single concurrent. Not benchmark-grade.")
out.append("Timing stats above are descriptive only; gates do not depend on them.")
out.append("")

text = "\n".join(out)
(ROOT / "summary.txt").write_text(text)
print(text)
