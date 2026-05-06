"""
Cross-backend summarizer for the (2, 4, 12) HPX waiter-pressure slice.

Reads:
  hpx/summary.txt
  hpx/per_repeat.csv
  std_control/summary.txt
  std_control/per_repeat.csv
  std_control/hpx_trace.txt

Writes:
  comparison_summary.txt

Cross-backend gates:
  - hpx/summary.txt has 'OVERALL: PASS'
  - std_control/summary.txt has 'OVERALL: PASS'
  - std_control/hpx_trace.txt is empty (zero non-empty lines)
  - HPX hash equals std_control hash for each req[0..11] (per-request equality)
  - all 24 hashes equal canonical 0x833045f1e2ebf49f
  - all 24 n_tokens_generated == 16
  - HPX lifecycle / capacity-correctness gates passed
    (per-line PASS in hpx/summary.txt)

Descriptive timing (NOT gated):
  - HPX (all 12 rows):  ttft_ms / total_ms / total_minus_ttft_ms / tokens_per_second
                        (min, max, mean, median)
  - HPX (req[4..11] trim, presentation only): total_minus_ttft_ms (mean, median)
  - std_control: same all-rows view and same trim view
  - delta of means (HPX - std_control) for total_minus_ttft_ms,
    both all-rows and trim views

OVERALL: PASS iff every cross-backend gate above passes.

Reference integrity is a discipline note, not a gate: this script does not
read or write any prior baseline directory (e.g. comparison_hpx_vs_std/,
comparison_aligned/, comparison_hpx_concurrency/). Helpers and summarizers
should leave those alone.

Scope reminder: this slice can test capacity correctness / no-starvation
under n_concurrent > n_contexts. It cannot prove FIFO waiter ordering
because the current trace does not emit queued/wake events.
"""
import csv
from pathlib import Path
from statistics import mean, median

BASE = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_hpx_waiters")
HPX_DIR = BASE / "hpx"
STD_DIR = BASE / "std_control"

CANONICAL_HASH      = "0x833045f1e2ebf49f"
EXPECTED_N_TOKENS   = 16
EXPECTED_N_REQUESTS = 12
TRIM_INDICES        = set(range(4, EXPECTED_N_REQUESTS))   # req[4..11]


def read_text_safe(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def read_csv_rows(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r["req_index"]))
    return rows


def total_minus_ttft(rows, indices=None):
    if indices is None:
        return [float(r["total_minus_ttft_ms"]) for r in rows]
    return [
        float(r["total_minus_ttft_ms"])
        for r in rows
        if int(r["req_index"]) in indices
    ]


def all_n_tokens_match(rows, expected):
    return bool(rows) and all(
        int(r["n_tokens_generated"]) == expected for r in rows
    )


def all_canonical(rows, canonical):
    return bool(rows) and all(
        r["generated_token_hash"] == canonical for r in rows
    )


def lifecycle_gates_passed(summary_text: str) -> bool:
    """All [PASS] / [FAIL] lines whose label mentions a lifecycle term must be PASS."""
    if not summary_text:
        return False
    lifecycle_terms = (
        "start line", "ready line", "stop line",
        "acquire", "release",
        "ctx ids", "total filtered",
    )
    saw_any = False
    for line in summary_text.splitlines():
        if not (line.startswith("[PASS]") or line.startswith("[FAIL]")):
            continue
        if not any(term in line for term in lifecycle_terms):
            continue
        saw_any = True
        if line.startswith("[FAIL]"):
            return False
    return saw_any


def stats_line(xs, unit="ms"):
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} min={min(xs):.3f}{unit} max={max(xs):.3f}{unit} "
        f"mean={mean(xs):.3f}{unit} median={median(xs):.3f}{unit}"
    )


def mean_or_none(xs):
    return mean(xs) if xs else None


# ---- read inputs ----
hpx_summary = read_text_safe(HPX_DIR / "summary.txt")
std_summary = read_text_safe(STD_DIR / "summary.txt")
std_trace   = read_text_safe(STD_DIR / "hpx_trace.txt")

hpx_rows = read_csv_rows(HPX_DIR / "per_repeat.csv")
std_rows = read_csv_rows(STD_DIR / "per_repeat.csv")

# ---- gates ----
checks = []

checks.append(("hpx/summary.txt has 'OVERALL: PASS'",
               "OVERALL: PASS" in hpx_summary))
checks.append(("std_control/summary.txt has 'OVERALL: PASS'",
               "OVERALL: PASS" in std_summary))

std_trace_lines = [ln for ln in std_trace.splitlines() if ln.strip() != ""]
checks.append(("std_control/hpx_trace.txt is empty (zero non-empty lines)",
               len(std_trace_lines) == 0))

per_req_match = (
    len(hpx_rows) == EXPECTED_N_REQUESTS and
    len(std_rows) == EXPECTED_N_REQUESTS and
    all(
        hpx_rows[i]["generated_token_hash"] == std_rows[i]["generated_token_hash"]
        for i in range(EXPECTED_N_REQUESTS)
    )
)
checks.append((f"HPX hash equals std_control hash for each "
               f"req[0..{EXPECTED_N_REQUESTS - 1}]", per_req_match))

all_can = (
    all_canonical(hpx_rows, CANONICAL_HASH) and
    all_canonical(std_rows, CANONICAL_HASH)
)
checks.append((f"all {2 * EXPECTED_N_REQUESTS} hashes equal canonical "
               f"{CANONICAL_HASH}", all_can))

n_tok_ok = (
    all_n_tokens_match(hpx_rows, EXPECTED_N_TOKENS) and
    all_n_tokens_match(std_rows, EXPECTED_N_TOKENS)
)
checks.append((f"all {2 * EXPECTED_N_REQUESTS} n_tokens_generated == "
               f"{EXPECTED_N_TOKENS}", n_tok_ok))

checks.append(("HPX lifecycle / capacity-correctness gates passed "
               "(per-line PASS in hpx/summary.txt)",
               lifecycle_gates_passed(hpx_summary)))

# ---- descriptive timing (not gated) ----
hpx_all_ttft   = [float(r["ttft_ms"])           for r in hpx_rows]
hpx_all_total  = [float(r["total_ms"])          for r in hpx_rows]
hpx_all_minus  = total_minus_ttft(hpx_rows)
hpx_all_tps    = [float(r["tokens_per_second"]) for r in hpx_rows]
hpx_trim_minus = total_minus_ttft(hpx_rows, TRIM_INDICES)

std_all_ttft   = [float(r["ttft_ms"])           for r in std_rows]
std_all_total  = [float(r["total_ms"])          for r in std_rows]
std_all_minus  = total_minus_ttft(std_rows)
std_all_tps    = [float(r["tokens_per_second"]) for r in std_rows]
std_trim_minus = total_minus_ttft(std_rows, TRIM_INDICES)

delta_all = None
if hpx_all_minus and std_all_minus:
    delta_all = mean_or_none(hpx_all_minus) - mean_or_none(std_all_minus)

delta_trim = None
if hpx_trim_minus and std_trim_minus:
    delta_trim = mean_or_none(hpx_trim_minus) - mean_or_none(std_trim_minus)

# ---- summary text ----
out = []
out.append("=== HPX vs std_control waiter-pressure comparison summary "
           "(comparison_hpx_waiters) ===")
out.append("")
out.append("inputs:")
out.append(f"  hpx:         {HPX_DIR}")
out.append(f"  std_control: {STD_DIR}")
out.append("")
out.append("shape: n_contexts=2, n_concurrent=4, n_requests=12, max_tokens=16, n_threads=4")
out.append("")
out.append("--- token hashes ---")
out.append(f"hpx        : {[r['generated_token_hash'] for r in hpx_rows]}")
out.append(f"std_control: {[r['generated_token_hash'] for r in std_rows]}")
out.append(f"canonical:   {CANONICAL_HASH}")
out.append("")
out.append("--- std_control trace lines ---")
out.append(f"std_control/hpx_trace.txt non-empty lines: {len(std_trace_lines)}")
out.append("")
out.append("--- descriptive timing, all 12 rows (NOT gated) ---")
out.append(f"hpx        ttft_ms             : {stats_line(hpx_all_ttft)}")
out.append(f"hpx        total_ms            : {stats_line(hpx_all_total)}")
out.append(f"hpx        total_ms - ttft_ms  : {stats_line(hpx_all_minus)}")
out.append(f"hpx        tokens_per_second   : {stats_line(hpx_all_tps, unit='tps')}")
out.append(f"std_control ttft_ms            : {stats_line(std_all_ttft)}")
out.append(f"std_control total_ms           : {stats_line(std_all_total)}")
out.append(f"std_control total_ms - ttft_ms : {stats_line(std_all_minus)}")
out.append(f"std_control tokens_per_second  : {stats_line(std_all_tps, unit='tps')}")
out.append("")
out.append("--- descriptive timing, req[4..11] trim, presentation only (NOT gated) ---")
out.append(f"hpx        total_ms - ttft_ms  : {stats_line(hpx_trim_minus)}")
out.append(f"std_control total_ms - ttft_ms : {stats_line(std_trim_minus)}")
out.append("")
out.append("--- delta of means (HPX - std_control) ---")
out.append(
    f"all-rows  total_ms - ttft_ms : "
    f"{f'{delta_all:+.3f}ms' if delta_all is not None else 'n/a'}"
)
out.append(
    f"trim view total_ms - ttft_ms : "
    f"{f'{delta_trim:+.3f}ms' if delta_trim is not None else 'n/a'}"
)
out.append("")
out.append("note: this is an n=12 smoke at n_contexts=2, n_concurrent=4.")
out.append("note: oversubscription (2 ctx x 4 threads on a 4-core box, 4 in-flight) is")
out.append("      more severe than the (2, 2) slice and is expected.")
out.append("note: not a performance claim. timing is informational only.")
out.append("note: trace lines are NOT timestamped; we do NOT claim true in-flight overlap.")
out.append("note: this slice does NOT establish FIFO waiter ordering — no queued/wake")
out.append("      trace events are emitted today, so the FIFO claim is not observable.")
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
out.append("Cross-backend leg of the (2, 4, 12) HPX waiter-pressure slice.")
out.append("Per-backend gates live in <variant>/summary.txt.")
out.append("No HPX speed claim from this n=12 smoke.")
out.append("No claim of true in-flight overlap; trace is event-ordered, not timestamped.")
out.append("No claim of FIFO waiter ordering; current trace lacks queued/wake events.")
out.append("No claim of per-request wait time; no queued event to anchor it.")
out.append("This slice does NOT exercise n_concurrent >> n_contexts (e.g. 8x2, 16x2).")
out.append("This slice does NOT exercise cancellation under backpressure.")
out.append("Reference integrity (prior baselines unmodified) is a discipline note, not")
out.append("a gate; this script does not read or write any prior baseline directory.")
out.append("")

text = "\n".join(out)
(BASE / "comparison_summary.txt").write_text(text)
print(text)
