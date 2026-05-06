"""
Cross-binary summarizer for the aligned llama-server vs llama-serving-bench std
comparison run.

This file does not run any binary. It only reads the artifacts produced by:
  - comparison_aligned/server/_run_requests.py
  - comparison_aligned/server/_summarize_timing.py
  - comparison_aligned/bench/_run_bench.py
  - comparison_aligned/bench/_summarize_timing.py

Inputs:
  comparison_aligned/server/per_repeat.csv          (must contain token_id_count, token_id_hash)
  comparison_aligned/server/server_start.stderr     (must contain "system_info: n_threads = N")
  comparison_aligned/server/summary.txt             (must contain "OVERALL: PASS")
  comparison_aligned/bench/per_repeat.csv           (must contain total_minus_ttft_ms)
  comparison_aligned/bench/bench.stderr             (must contain "n_threads_per_ctx=N")
  comparison_aligned/bench/summary.txt              (must contain "OVERALL: PASS")
  comparison_aligned/bench/process_wall_ms.txt      (informational; tolerated if missing)

Output:
  comparison_aligned/comparison_summary.txt

Cross-binary gates (additive; no per-binary gate is relaxed):
  [G1] thread parity:
       server "system_info: n_threads = N" matches bench "n_threads_per_ctx=N"
       and equals the expected value (4 by default; override via EXPECTED_N_THREADS)
  [G2] cross-binary content parity:
       every measured server token_id_hash equals the bench-measured hash
       AND equals the canonical hash (default: 0x833045f1e2ebf49f)
       AND token_id_count == 16 for every measured server row
  [G3] warmup hash parity:
       server warmup token_id_hash == bench warmup generated_token_hash
  [G4] csv schema additions:
       server csv has token_id_count and token_id_hash columns
       bench csv  has total_minus_ttft_ms column
  [G5] interval declaration:
       comparison_aligned/README.md contains the verbatim interval-pair line
       (so the artifact set declares which numbers are being compared)
  [G6] per-binary OVERALL PASS:
       comparison_aligned/server/summary.txt contains "OVERALL: PASS"
       comparison_aligned/bench/summary.txt  contains "OVERALL: PASS"

Descriptive table (not a gate):
  server timings.predicted_ms        n=5  min/max/mean/median/stdev_pop
  bench  total_ms - ttft_ms          n=5  min/max/mean/median/stdev_pop

Optional informational rows (not a gate):
  server  sum of 5 measured client wall_ms
  bench   process_wall_ms (from process_wall_ms.txt)
"""
import csv
import re
from pathlib import Path
from statistics import mean, median, pstdev

ROOT        = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_aligned")
SERVER_DIR  = ROOT / "server"
BENCH_DIR   = ROOT / "bench"
README_PATH = ROOT / "README.md"

EXPECTED_N_THREADS = 4
CANONICAL_BENCH_HASH = "0x833045f1e2ebf49f"
EXPECTED_N_GENERATED_TOKENS = 16
INTERVAL_DECLARATION = (
    "interval pair under comparison: llama-server timings.predicted_ms "
    "vs llama-serving-bench std (total_ms - ttft_ms)"
)

SERVER_THREADS_RE = re.compile(r"system_info:\s*n_threads\s*=\s*(\d+)")
BENCH_THREADS_RE  = re.compile(r"n_threads_per_ctx\s*=\s*(\d+)")


def stat_block(xs, unit=""):
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} min={min(xs):.3f}{unit} max={max(xs):.3f}{unit} "
        f"mean={mean(xs):.3f}{unit} median={median(xs):.3f}{unit} "
        f"stdev_pop={(pstdev(xs) if len(xs) > 1 else 0.0):.3f}{unit}"
    )


def read_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def first_match(regex: re.Pattern, text: str):
    m = regex.search(text)
    return m.group(1) if m else None


# ---- read everything ----
server_rows  = read_csv(SERVER_DIR / "per_repeat.csv")
bench_rows   = read_csv(BENCH_DIR  / "per_repeat.csv")

server_stderr  = (SERVER_DIR / "server_start.stderr").read_text()
bench_stderr   = (BENCH_DIR  / "bench.stderr").read_text()

server_summary = (SERVER_DIR / "summary.txt").read_text()
bench_summary  = (BENCH_DIR  / "summary.txt").read_text()

readme_text    = README_PATH.read_text() if README_PATH.exists() else ""

# Optional informational source.
process_wall_path = BENCH_DIR / "process_wall_ms.txt"
process_wall_first_line = None
if process_wall_path.exists():
    raw = process_wall_path.read_text().splitlines()
    if raw:
        process_wall_first_line = raw[0].strip()

# ---- partition rows ----
server_warmup   = [r for r in server_rows if r["kind"] == "warmup"]
server_measured = [r for r in server_rows if r["kind"] == "measured"]
bench_warmup    = [r for r in bench_rows  if int(r["req_index"]) == 0]
bench_measured  = [r for r in bench_rows  if 1 <= int(r["req_index"]) <= 5]

# ---- gate computations ----
checks = []

# G6: per-binary OVERALL PASS first, so we know per-binary gates didn't regress.
checks.append((
    "G6a server per-binary OVERALL: PASS",
    "OVERALL: PASS" in server_summary,
))
checks.append((
    "G6b bench per-binary OVERALL: PASS",
    "OVERALL: PASS" in bench_summary,
))

# G1: thread parity
server_n_threads = first_match(SERVER_THREADS_RE, server_stderr)
bench_n_threads  = first_match(BENCH_THREADS_RE,  bench_stderr)

checks.append((
    f"G1a server stderr declares n_threads = {EXPECTED_N_THREADS}",
    server_n_threads is not None and int(server_n_threads) == EXPECTED_N_THREADS,
))
checks.append((
    f"G1b bench  stderr declares n_threads_per_ctx = {EXPECTED_N_THREADS}",
    bench_n_threads is not None and int(bench_n_threads) == EXPECTED_N_THREADS,
))
checks.append((
    "G1c thread parity (server == bench)",
    server_n_threads is not None
    and bench_n_threads is not None
    and server_n_threads == bench_n_threads,
))

# G4: csv schema additions
server_cols = set(server_rows[0].keys()) if server_rows else set()
bench_cols  = set(bench_rows[0].keys())  if bench_rows  else set()

checks.append((
    "G4a server csv has token_id_count and token_id_hash columns",
    "token_id_count" in server_cols and "token_id_hash" in server_cols,
))
checks.append((
    "G4b bench csv has total_minus_ttft_ms column",
    "total_minus_ttft_ms" in bench_cols,
))

# G2: cross-binary content parity (over measured rows).
def norm_hash(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    if not s.startswith("0x"):
        return s
    return "0x" + s[2:].lstrip("0").rjust(16, "0")  # canonical 16-hex form

bench_measured_hashes = {norm_hash(r["generated_token_hash"]) for r in bench_measured}
server_measured_hashes = {norm_hash(r.get("token_id_hash", "")) for r in server_measured}

bench_measured_singleton = (len(bench_measured_hashes) == 1)
server_measured_singleton = (len(server_measured_hashes) == 1)

if bench_measured_singleton:
    bench_measured_hash_value = next(iter(bench_measured_hashes))
else:
    bench_measured_hash_value = None

if server_measured_singleton:
    server_measured_hash_value = next(iter(server_measured_hashes))
else:
    server_measured_hash_value = None

checks.append((
    "G2a server measured token_id_hash all identical (n_distinct == 1)",
    server_measured_singleton,
))
checks.append((
    "G2b bench measured generated_token_hash all identical (n_distinct == 1)",
    bench_measured_singleton,
))
checks.append((
    "G2c server measured hash == bench measured hash",
    server_measured_singleton
    and bench_measured_singleton
    and server_measured_hash_value == bench_measured_hash_value,
))
checks.append((
    f"G2d measured hash equals canonical {CANONICAL_BENCH_HASH}",
    bench_measured_hash_value == norm_hash(CANONICAL_BENCH_HASH),
))

server_token_counts_ok = all(
    int(r.get("token_id_count", "0") or "0") == EXPECTED_N_GENERATED_TOKENS
    for r in server_measured
) and len(server_measured) == 5
checks.append((
    f"G2e server measured token_id_count == {EXPECTED_N_GENERATED_TOKENS} on all 5 rows",
    server_token_counts_ok,
))

# G3: warmup hash parity
server_warmup_hash = norm_hash(server_warmup[0].get("token_id_hash", "")) if server_warmup else ""
bench_warmup_hash  = norm_hash(bench_warmup[0]["generated_token_hash"]) if bench_warmup else ""
checks.append((
    "G3 warmup hash parity (server warmup == bench warmup)",
    bool(server_warmup_hash) and server_warmup_hash == bench_warmup_hash,
))

# G5: interval declaration in README.md
checks.append((
    "G5 README.md declares the interval pair verbatim",
    INTERVAL_DECLARATION in readme_text,
))

# ---- descriptive interval-aligned table ----
server_predicted_ms = [
    float(r["predicted_ms"]) for r in server_measured if r.get("predicted_ms")
]
bench_total_minus_ttft = [
    float(r["total_minus_ttft_ms"]) for r in bench_measured if r.get("total_minus_ttft_ms")
]

# ---- informational outer-wall rows ----
server_measured_wall_sum = (
    sum(float(r["wall_ms"]) for r in server_measured) if server_measured else None
)

# ---- assemble summary text ----
out = []
out.append("=== Aligned comparison summary: llama-server vs llama-serving-bench std ===")
out.append("")
out.append(f"server n_threads (parsed from server_start.stderr): {server_n_threads}")
out.append(f"bench  n_threads_per_ctx (parsed from bench.stderr): {bench_n_threads}")
out.append(f"expected n_threads on both sides:                   {EXPECTED_N_THREADS}")
out.append("")
out.append("--- token-hash parity ---")
out.append(f"server measured hashes (n_distinct={len(server_measured_hashes)}): "
           + (server_measured_hash_value or "(none)"))
out.append(f"bench  measured hashes (n_distinct={len(bench_measured_hashes)}): "
           + (bench_measured_hash_value or "(none)"))
out.append(f"server warmup hash: {server_warmup_hash or '(missing)'}")
out.append(f"bench  warmup hash: {bench_warmup_hash or '(missing)'}")
out.append(f"canonical bench hash: {norm_hash(CANONICAL_BENCH_HASH)}")
out.append("")
out.append("--- interval-aligned table (descriptive, not a gate) ---")
out.append(INTERVAL_DECLARATION)
out.append("server timings.predicted_ms (measured, n=5):  "
           + stat_block(server_predicted_ms, "ms"))
out.append("bench  total_ms - ttft_ms   (measured, n=5):  "
           + stat_block(bench_total_minus_ttft, "ms"))
out.append("")
out.append("--- informational outer-wall (not a gate, scope §4.4) ---")
if server_measured_wall_sum is not None:
    out.append(f"server  sum of 5 measured client wall_ms: {server_measured_wall_sum:.3f} ms")
else:
    out.append("server  sum of 5 measured client wall_ms: (no measured rows)")
if process_wall_first_line is not None:
    out.append(f"bench   process_wall_ms (line 1 of process_wall_ms.txt): {process_wall_first_line}")
else:
    out.append("bench   process_wall_ms.txt: (missing)")
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
out.append("--- scope ---")
out.append("This summary is the cross-binary leg of the aligned comparison run.")
out.append("Per-binary gates live in comparison_aligned/server/summary.txt and")
out.append("comparison_aligned/bench/summary.txt; G6 above asserts both PASS.")
out.append("No HPX. No concurrency. n=5 measured. Descriptive only.")
out.append("")

text = "\n".join(out)
(ROOT / "comparison_summary.txt").write_text(text)
print(text)
