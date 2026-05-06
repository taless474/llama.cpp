"""
HPX-OFF llama-serving-bench std timing smoke summarizer.

Reads:
  - per_repeat.csv         (header + 6 data rows: req[0]=warmup, req[1..5]=measured)
  - bench.stdout           (per-request lines + aggregate line)
  - bench.stderr           (config dump + prompt-fits line + diagnostics)
  - bench.exit_code.txt    (subprocess exit code)

Writes:
  - summary.txt            (per-row table, measured stats, gates, OVERALL PASS/FAIL, caveats)

Stats computed over measured rows only (req[1..5]):
  - ttft_ms:           min, max, mean, median, stdev_pop
  - total_ms:          min, max, mean, median, stdev_pop
  - tokens_per_second: min, max, mean, median, stdev_pop

Gates (mirroring the design plan):
  - harness exit code == 0
  - all 6 req lines parsed
  - aggregate line: n_ok=6 n_cancelled=0 n_error=0
  - all 6 status == ok
  - all 6 n_tokens_generated == 16
  - req[1..5] generated_token_hash all identical
  - req[0] generated_token_hash equals measured hash
  - stderr contains "prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size"
  - stderr has no "[serving-bench] error"
  - stderr has no "failed"
  - per_repeat.csv has 6 data rows + header
"""
import csv
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/serving_bench_std_timing")

EXPECTED_N_TOKENS         = 16
EXPECTED_PROMPT_FITS_LINE = "prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size"
EXPECTED_AGG_LINE         = "n_ok=6 n_cancelled=0 n_error=0"


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

rows = []
with (ROOT / "per_repeat.csv").open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

warmup_rows   = [r for r in rows if int(r["req_index"]) == 0]
measured_rows = [r for r in rows if 1 <= int(r["req_index"]) <= 5]

# ---- gates ----
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

# ---- measured stats ----
ttft_ms  = [float(r["ttft_ms"])           for r in measured_rows]
total_ms = [float(r["total_ms"])          for r in measured_rows]
tps      = [float(r["tokens_per_second"]) for r in measured_rows]

# ---- summary text ----
out = []
out.append("=== HPX-OFF llama-serving-bench std timing smoke summary ===")
out.append("")
out.append(f"binary exit code: {exit_code}")
out.append("requests: 1 warmup (req[0]) + 5 measured (req[1..5])")
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
out.append("This is a small std-backend timing smoke, not a benchmark.")
out.append("No HPX comparison.")
out.append("No concurrency behavior.")
out.append("No cross-binary wall-clock comparison to llama-server yet.")
out.append("")

text = "\n".join(out)
(ROOT / "summary.txt").write_text(text)
print(text)
