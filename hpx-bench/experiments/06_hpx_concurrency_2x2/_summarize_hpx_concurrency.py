"""
Per-backend summarizer for the (2,2) HPX concurrency slice.

Variants (single CLI argument):
  python3 _summarize_hpx_concurrency.py                  # default: hpx/
  python3 _summarize_hpx_concurrency.py hpx              # explicit
  python3 _summarize_hpx_concurrency.py std_control      # std-control variant

Reads (under <variant>/):
  per_repeat.csv         (header + 8 data rows; req_index 0..7, kind=request)
  bench.stdout           (per-request lines + aggregate line)
  bench.stderr           (config dump + prompt-fits line + diagnostics)
  bench.exit_code.txt    (subprocess exit code)
  hpx_trace.txt          (filtered HPX lifecycle / pool lines)

Writes:
  <variant>/summary.txt  (per-row table, descriptive stats, gates, OVERALL PASS/FAIL, caveats)

Common gates (applied in both variants):
  - harness exit code == 0
  - all 8 req lines parsed
  - stdout aggregate line: n_ok=8 n_cancelled=0 n_error=0
  - all 8 status == ok
  - all 8 n_tokens_generated == 16
  - all 8 generated_token_hash equal canonical 0x833045f1e2ebf49f
  - stderr contains the prompt-fits line
  - stderr has no '[serving-bench] error'
  - stderr has no 'failed'
  - per_repeat.csv has 8 data rows + header

HPX lifecycle gates (variant=hpx, positive form):
  - exactly one '[serving-bench] hpx_runtime_start_once: starting (os_threads=2)'
  - exactly one '[serving-bench] engine_hpx ready: n_contexts=2 pool_size=2'
  - exactly one '[serving-bench] hpx_runtime_stop: stopping'
  - for each i in 0..7: exactly one acquire and exactly one release
  - for each i in 0..7: acquire ctx id and release ctx id are both in {0, 1}
  - for each i in 0..7: release ctx id equals acquire ctx id (same-ctx pairing)
  - for each i in 0..7: acquire line precedes release line in trace order
  - the set of acquire ctx ids across all 8 requests equals {0, 1}
  - total filtered HPX trace lines == 19   (1 + 1 + 1 + 8 + 8)

Lifecycle gate (variant=std_control, inverted form):
  - hpx_trace.txt contains zero matching HPX lifecycle / pool lines

Explicitly NOT gated:
  - the order in which req indices are acquired
  - the order in which ctx ids appear (no alternation requirement)
  - the count of requests served by ctx=0 vs ctx=1 (any split that uses both is fine)
  - whether any two requests overlap in time (no trace timestamps)

OVERALL: PASS iff every gate above passes.
"""
import csv
import re
import sys
from pathlib import Path
from statistics import mean, median, pstdev

BASE = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_hpx_concurrency")

VALID_VARIANTS = ("hpx", "std_control")
variant = sys.argv[1] if len(sys.argv) > 1 else "hpx"
if variant not in VALID_VARIANTS:
    print(f"error: variant must be one of {VALID_VARIANTS}; got {variant!r}",
          file=sys.stderr)
    sys.exit(2)

ROOT = BASE / variant

CANONICAL_HASH            = "0x833045f1e2ebf49f"
EXPECTED_N_TOKENS         = 16
EXPECTED_N_REQUESTS       = 8
EXPECTED_PROMPT_FITS_LINE = "prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size"
EXPECTED_AGG_LINE         = "n_ok=8 n_cancelled=0 n_error=0"
EXPECTED_TRACE_LINE_COUNT = 19  # 1 start + 1 ready + 1 stop + 8 acquire + 8 release

START_LINE = "[serving-bench] hpx_runtime_start_once: starting (os_threads=2)"
READY_LINE = "[serving-bench] engine_hpx ready: n_contexts=2 pool_size=2"
STOP_LINE  = "[serving-bench] hpx_runtime_stop: stopping"
ACQUIRE_RE = re.compile(r"^\[serving-bench\] req\[(\d+)\] acquire ctx=(\d+)$")
RELEASE_RE = re.compile(r"^\[serving-bench\] req\[(\d+)\] release ctx=(\d+)$")
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
rows.sort(key=lambda r: int(r["req_index"]))

# ---- common gates ----
checks = []

checks.append(("harness exit code == 0", exit_code == 0))
checks.append((f"all {EXPECTED_N_REQUESTS} req lines parsed",
               len(rows) == EXPECTED_N_REQUESTS))
checks.append((f"stdout aggregate line contains: {EXPECTED_AGG_LINE!r}",
               EXPECTED_AGG_LINE in stdout))

status_ok = (len(rows) == EXPECTED_N_REQUESTS) and all(
    r["status"] == "ok" for r in rows
)
checks.append((f"all {EXPECTED_N_REQUESTS} status == ok", status_ok))

n_tokens_ok = (len(rows) == EXPECTED_N_REQUESTS) and all(
    int(r["n_tokens_generated"]) == EXPECTED_N_TOKENS for r in rows
)
checks.append((f"all {EXPECTED_N_REQUESTS} n_tokens_generated == {EXPECTED_N_TOKENS}",
               n_tokens_ok))

all_canonical_ok = (len(rows) == EXPECTED_N_REQUESTS) and all(
    r["generated_token_hash"] == CANONICAL_HASH for r in rows
)
checks.append((f"all {EXPECTED_N_REQUESTS} generated_token_hash equal canonical "
               f"{CANONICAL_HASH}", all_canonical_ok))

checks.append((f"stderr contains: {EXPECTED_PROMPT_FITS_LINE!r}",
               EXPECTED_PROMPT_FITS_LINE in stderr))
checks.append(("stderr has no '[serving-bench] error'",
               "[serving-bench] error" not in stderr))
checks.append(("stderr has no 'failed'", "failed" not in stderr))
checks.append((f"per_repeat.csv has {EXPECTED_N_REQUESTS} data rows + header",
               len(rows) == EXPECTED_N_REQUESTS))

# ---- lifecycle gates (variant-specific) ----
acquire_ctx_ids_seen = set()

if variant == "hpx":
    n_start = sum(1 for ln in trace_lines_all if ln == START_LINE)
    n_ready = sum(1 for ln in trace_lines_all if ln == READY_LINE)
    n_stop  = sum(1 for ln in trace_lines_all if ln == STOP_LINE)

    checks.append((f"exactly one start line: {START_LINE!r}", n_start == 1))
    checks.append((f"exactly one ready line: {READY_LINE!r}", n_ready == 1))
    checks.append((f"exactly one stop line: {STOP_LINE!r}",  n_stop  == 1))

    # req_index -> [(position-in-trace, ctx_id), ...]
    acquire_idx = {}
    release_idx = {}
    for pos, ln in enumerate(trace_lines_all):
        m_a = ACQUIRE_RE.match(ln)
        m_r = RELEASE_RE.match(ln)
        if m_a:
            acquire_idx.setdefault(int(m_a.group(1)), []).append(
                (pos, int(m_a.group(2)))
            )
        elif m_r:
            release_idx.setdefault(int(m_r.group(1)), []).append(
                (pos, int(m_r.group(2)))
            )

    for i in range(EXPECTED_N_REQUESTS):
        a = acquire_idx.get(i, [])
        r = release_idx.get(i, [])
        checks.append((f"req[{i}] exactly one acquire", len(a) == 1))
        checks.append((f"req[{i}] exactly one release", len(r) == 1))
        a_ctx_ok = len(a) == 1 and a[0][1] in (0, 1)
        r_ctx_ok = len(r) == 1 and r[0][1] in (0, 1)
        checks.append((f"req[{i}] acquire ctx id in {{0, 1}}", a_ctx_ok))
        checks.append((f"req[{i}] release ctx id in {{0, 1}}", r_ctx_ok))
        same_ctx = (len(a) == 1 and len(r) == 1 and a[0][1] == r[0][1])
        checks.append((f"req[{i}] release ctx equals acquire ctx", same_ctx))
        precedes = (len(a) == 1 and len(r) == 1 and a[0][0] < r[0][0])
        checks.append((f"req[{i}] acquire precedes release", precedes))
        if len(a) == 1:
            acquire_ctx_ids_seen.add(a[0][1])

    checks.append(("set of acquire ctx ids equals {0, 1}",
                   acquire_ctx_ids_seen == {0, 1}))
    checks.append((f"total filtered HPX trace lines == {EXPECTED_TRACE_LINE_COUNT}",
                   len(trace_lines_all) == EXPECTED_TRACE_LINE_COUNT))

elif variant == "std_control":
    matching_traces = [ln for ln in trace_lines_all if ANY_TRACE_RE.match(ln)]
    checks.append(("hpx_trace.txt contains zero HPX lifecycle/pool lines",
                   len(matching_traces) == 0))

# ---- descriptive timing over all 8 rows ----
ttft_ms             = [float(r["ttft_ms"])             for r in rows]
total_ms            = [float(r["total_ms"])            for r in rows]
tps                 = [float(r["tokens_per_second"])   for r in rows]
total_minus_ttft_ms = [float(r["total_minus_ttft_ms"]) for r in rows]

# ---- summary text ----
out = []
out.append(f"=== llama-serving-bench {variant} concurrency summary "
           f"(comparison_hpx_concurrency/{variant}) ===")
out.append("")
out.append(f"binary exit code: {exit_code}")
out.append(f"requests: {EXPECTED_N_REQUESTS} requests, all correctness-gated (no warmup phase)")
out.append(f"variant:  {variant}")
out.append("shape:    n_contexts=2, n_concurrent=2, n_requests=8, max_tokens=16, n_threads=4")
out.append("")
out.append("--- per-request rows ---")
for r in rows:
    out.append(
        f"req[{r['req_index']}] kind={r['kind']:<8} status={r['status']} "
        f"n_tok={r['n_tokens_generated']} hash={r['generated_token_hash']} "
        f"ttft_ms={float(r['ttft_ms']):.3f} "
        f"total_ms={float(r['total_ms']):.3f} "
        f"total_minus_ttft_ms={float(r['total_minus_ttft_ms']):.3f} "
        f"tps={float(r['tokens_per_second']):.3f}"
    )
out.append("")
out.append("--- ttft_ms ---")
out.append(stats(ttft_ms, "ms"))
out.append("--- total_ms ---")
out.append(stats(total_ms, "ms"))
out.append("--- total_ms - ttft_ms ---")
out.append(stats(total_minus_ttft_ms, "ms"))
out.append("--- tokens_per_second ---")
out.append(stats(tps, "tps"))
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
    if acquire_ctx_ids_seen:
        out.append(f"acquire ctx ids seen: {sorted(acquire_ctx_ids_seen)}")
elif variant == "std_control":
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
out.append("This is one bench-side leg of the (2,2) HPX concurrency slice.")
out.append("Cross-backend gates live in comparison_hpx_concurrency/comparison_summary.txt.")
out.append("All 8 requests are correctness-gated. No warmup separation.")
out.append("Timing stats above are descriptive only; gates do not depend on them.")
out.append("Oversubscription (2 contexts x 4 threads = 8 kernel threads on a 4-core box)")
out.append("is expected and is NOT a regression.")
out.append("Trace lines are event-ordered but not timestamped; we do NOT claim")
out.append("true overlap of in-flight requests in time.")
out.append("")

text = "\n".join(out)
(ROOT / "summary.txt").write_text(text)
print(text)
