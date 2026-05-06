"""
Cross-binary summarizer for the std-vs-HPX comparison.

Reads:
  hpx/summary.txt
  hpx/per_repeat.csv
  hpx_off_regression/summary.txt
  hpx_off_regression/per_repeat.csv
  hpx_off_regression/hpx_trace.txt
  ../comparison_aligned/bench/summary.txt           (existing std reference)
  ../comparison_aligned/bench/per_repeat.csv         (existing std reference)

Writes:
  comparison_summary.txt

Cross-binary gates:
  - hpx/summary.txt has 'OVERALL: PASS'
  - hpx_off_regression/summary.txt has 'OVERALL: PASS'
  - hpx_off_regression/hpx_trace.txt is empty (no HPX trace lines at --backend std)
  - HPX warmup hash equals std reference warmup hash
  - HPX measured hash equals std reference measured hash
  - HPX-off regression warmup hash equals std reference warmup hash
  - HPX-off regression measured hash equals std reference measured hash
  - all warmup + measured hashes equal canonical 0x833045f1e2ebf49f
  - all n_tokens_generated == 16 across hpx/, hpx_off_regression/, std reference
  - HPX lifecycle gates passed (each per-req acquire/release line and each
    runtime start/ready/stop line shows [PASS] in hpx/summary.txt)
  - existing std reference still passes its own per-binary gates
    (../comparison_aligned/bench/summary.txt still has 'OVERALL: PASS' and
    its measured/warmup hashes still equal canonical) -- this stands in for
    the "std reference was not modified" requirement; we cannot prove the
    file bytes are unchanged without a stored snapshot, but a still-green
    reference confirms it has not been functionally broken by the comparison.

Descriptive timing (NOT gated):
  - HPX measured (total_ms - ttft_ms) min / max / mean / median
  - std reference measured (total_ms - ttft_ms) min / max / mean / median
  - delta of means (HPX - std)

OVERALL: PASS iff every cross-binary gate above passes.
"""
import csv
from pathlib import Path
from statistics import mean, median

BASE = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_hpx_vs_std")
HPX_DIR  = BASE / "hpx"
REG_DIR  = BASE / "hpx_off_regression"
STD_DIR  = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_aligned/bench")

CANONICAL_HASH    = "0x833045f1e2ebf49f"
EXPECTED_N_TOKENS = 16


def read_text_safe(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def read_csv_rows(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def measured_minus_ttft(rows):
    return [
        float(r["total_ms"]) - float(r["ttft_ms"])
        for r in rows
        if 1 <= int(r["req_index"]) <= 5
    ]


def warmup_hash(rows):
    for r in rows:
        if int(r["req_index"]) == 0:
            return r["generated_token_hash"]
    return None


def measured_hashes(rows):
    return {
        r["generated_token_hash"]
        for r in rows
        if 1 <= int(r["req_index"]) <= 5
    }


def all_n_tokens_match(rows, expected):
    return bool(rows) and all(
        int(r["n_tokens_generated"]) == expected for r in rows
    )


def lifecycle_gates_passed(summary_text: str) -> bool:
    """All [PASS] / [FAIL] lines whose label mentions a lifecycle term must be PASS."""
    if not summary_text:
        return False
    lifecycle_terms = (
        "start line", "ready line", "stop line",
        "acquire", "release",
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


# ---- read inputs ----
hpx_summary   = read_text_safe(HPX_DIR / "summary.txt")
reg_summary   = read_text_safe(REG_DIR / "summary.txt")
std_summary   = read_text_safe(STD_DIR / "summary.txt")
reg_trace     = read_text_safe(REG_DIR / "hpx_trace.txt")

hpx_rows = read_csv_rows(HPX_DIR / "per_repeat.csv")
reg_rows = read_csv_rows(REG_DIR / "per_repeat.csv")
std_rows = read_csv_rows(STD_DIR / "per_repeat.csv")

# ---- gates ----
checks = []

checks.append(("hpx/summary.txt has 'OVERALL: PASS'",
               "OVERALL: PASS" in hpx_summary))
checks.append(("hpx_off_regression/summary.txt has 'OVERALL: PASS'",
               "OVERALL: PASS" in reg_summary))

reg_trace_lines = [ln for ln in reg_trace.splitlines() if ln.strip() != ""]
checks.append(("hpx_off_regression/hpx_trace.txt is empty (zero trace lines)",
               len(reg_trace_lines) == 0))

hpx_w  = warmup_hash(hpx_rows)
reg_w  = warmup_hash(reg_rows)
std_w  = warmup_hash(std_rows)
hpx_m  = measured_hashes(hpx_rows)
reg_m  = measured_hashes(reg_rows)
std_m  = measured_hashes(std_rows)

hpx_m_one = next(iter(hpx_m)) if len(hpx_m) == 1 else None
reg_m_one = next(iter(reg_m)) if len(reg_m) == 1 else None
std_m_one = next(iter(std_m)) if len(std_m) == 1 else None

checks.append(("HPX warmup hash equals std reference warmup hash",
               hpx_w is not None and hpx_w == std_w))
checks.append(("HPX measured hash equals std reference measured hash",
               hpx_m_one is not None and hpx_m_one == std_m_one))
checks.append(("HPX-off regression warmup hash equals std reference warmup hash",
               reg_w is not None and reg_w == std_w))
checks.append(("HPX-off regression measured hash equals std reference measured hash",
               reg_m_one is not None and reg_m_one == std_m_one))

all_hashes = []
for w, m_one in ((hpx_w, hpx_m_one), (reg_w, reg_m_one), (std_w, std_m_one)):
    if w is not None:
        all_hashes.append(w)
    if m_one is not None:
        all_hashes.append(m_one)
checks.append((f"all warmup + measured hashes equal canonical {CANONICAL_HASH}",
               bool(all_hashes) and all(h == CANONICAL_HASH for h in all_hashes)))

n_tok_ok = (
    all_n_tokens_match(hpx_rows, EXPECTED_N_TOKENS) and
    all_n_tokens_match(reg_rows, EXPECTED_N_TOKENS) and
    all_n_tokens_match(std_rows, EXPECTED_N_TOKENS)
)
checks.append((f"all n_tokens_generated == {EXPECTED_N_TOKENS} "
               "across hpx/, hpx_off_regression/, std reference", n_tok_ok))

checks.append(("HPX lifecycle gates passed (per-line PASS in hpx/summary.txt)",
               lifecycle_gates_passed(hpx_summary)))

std_unmodified = (
    "OVERALL: PASS" in std_summary
    and std_w == CANONICAL_HASH
    and std_m_one == CANONICAL_HASH
)
checks.append(("std reference still passes its own gates "
               "(integrity stand-in for 'not modified')", std_unmodified))

# ---- descriptive timing (not gated) ----
hpx_minus = measured_minus_ttft(hpx_rows)
std_minus = measured_minus_ttft(std_rows)
reg_minus = measured_minus_ttft(reg_rows)
hpx_mean  = mean(hpx_minus) if hpx_minus else None
std_mean  = mean(std_minus) if std_minus else None
delta     = (hpx_mean - std_mean) if (hpx_mean is not None and std_mean is not None) else None

# ---- summary text ----
out = []
out.append("=== std-vs-HPX comparison summary (comparison_hpx_vs_std) ===")
out.append("")
out.append("inputs:")
out.append(f"  hpx:                     {HPX_DIR}")
out.append(f"  hpx_off_regression:      {REG_DIR}")
out.append(f"  std reference (existing): {STD_DIR}")
out.append("")
out.append("--- token hashes ---")
out.append(f"hpx        warmup={hpx_w}   measured={hpx_m_one}   "
           f"distinct_measured={len(hpx_m)}")
out.append(f"regression warmup={reg_w}   measured={reg_m_one}   "
           f"distinct_measured={len(reg_m)}")
out.append(f"std ref    warmup={std_w}   measured={std_m_one}   "
           f"distinct_measured={len(std_m)}")
out.append(f"canonical:  {CANONICAL_HASH}")
out.append("")
out.append("--- regression trace lines ---")
out.append(f"hpx_off_regression/hpx_trace.txt non-empty lines: {len(reg_trace_lines)}")
out.append("")
out.append("--- descriptive timing (not gated) ---")
out.append(f"hpx        measured (total_ms - ttft_ms): {stats_line(hpx_minus)}")
out.append(f"std ref    measured (total_ms - ttft_ms): {stats_line(std_minus)}")
out.append(f"regression measured (total_ms - ttft_ms): {stats_line(reg_minus)}")
if delta is not None:
    out.append(f"delta of means (hpx - std_ref): {delta:+.3f}ms")
else:
    out.append("delta of means (hpx - std_ref): n/a (missing rows)")
out.append("note: this is an n=5 smoke at single-context, single-concurrent.")
out.append("note: not a performance claim. timing is informational only.")
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
out.append("This is the cross-binary leg of the std-vs-HPX comparison.")
out.append("Per-binary gates live in <variant>/summary.txt.")
out.append("No HPX speed claim from this n=5 smoke.")
out.append("No concurrency. Single context, single concurrent.")
out.append("Not benchmark-grade. Correctness + lifecycle only.")
out.append("'std reference was not modified' is verified indirectly by checking")
out.append("that the existing reference still produces the canonical hash and")
out.append("still passes its own per-binary gates; byte-level immutability is")
out.append("not asserted.")
out.append("")

text = "\n".join(out)
(BASE / "comparison_summary.txt").write_text(text)
print(text)
