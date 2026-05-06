"""
Per-binary llama-server summarizer for the aligned comparison run.

Identical in behavior to local/baselines/server_timing/_summarize_timing.py.
Only the ROOT path differs. The CSV in this directory carries two extra
columns (token_id_count, token_id_hash); csv.DictReader handles them
transparently and they are not consulted by these per-binary gates. The
cross-binary _summarize_comparison.py is the one that reads those columns.

Inputs (under local/baselines/comparison_aligned/server/):
  - per_repeat.csv
  - request_warmup.json, request_measured_{1..5}.json
  - metrics_before.txt, metrics_after.txt

Output:
  - summary.txt
  - prints PASS/FAIL summary
"""
import csv
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_aligned/server")

CANONICAL_CONTENT = " John Smith. I am a software engineer at XYZ Company. I have"
EXPECTED_PROMPT_TOKENS_PER_REQUEST = 6
EXPECTED_PREDICTED_PER_REQUEST = 16
N_REQUESTS = 6  # 1 warmup + 5 measured

EXPECTED_DELTAS = {
    "llamacpp:prompt_tokens_total": 6 * N_REQUESTS,        # 36
    "llamacpp:tokens_predicted_total": 16 * N_REQUESTS,    # 96
    "llamacpp:n_decode_total": 16 * N_REQUESTS,            # 96 (if exposed)
}

METRIC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_:]*)\s+([-+]?[0-9]*\.?[0-9]+)\s*$")


def parse_metrics(path: Path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line.strip())
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def stats(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "max": max(xs),
        "mean": mean(xs),
        "median": median(xs),
        "stdev_pop": pstdev(xs) if len(xs) > 1 else 0.0,
    }


def fmt_stats(s, unit=""):
    if s.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={s['n']} min={s['min']:.3f}{unit} max={s['max']:.3f}{unit} "
        f"mean={s['mean']:.3f}{unit} median={s['median']:.3f}{unit} "
        f"stdev_pop={s['stdev_pop']:.3f}{unit}"
    )


# ---- read CSV ----
csv_path = ROOT / "per_repeat.csv"
rows = []
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

assert len(rows) == N_REQUESTS, f"expected {N_REQUESTS} csv data rows, got {len(rows)}"

warmup_rows = [r for r in rows if r["kind"] == "warmup"]
measured_rows = [r for r in rows if r["kind"] == "measured"]
assert len(warmup_rows) == 1
assert len(measured_rows) == 5

# ---- HTTP 200 + JSON parse for all 6 ----
checks = []

for r in rows:
    checks.append((f"{r['label']} HTTP 200", r["http_status"] == "200"))

# ---- per-request JSON checks ----
def load_json(label):
    return json.loads((ROOT / f"request_{label}.json").read_text())

bodies = {r["label"]: load_json(r["label"]) for r in rows}

for r in rows:
    label = r["label"]
    body = bodies[label]
    checks.append((f"{label} JSON parsed", isinstance(body, dict)))

# ---- measured-only correctness checks ----
for r in measured_rows:
    label = r["label"]
    body = bodies[label]
    checks.append(
        (f"{label} content byte-equal canonical", body.get("content") == CANONICAL_CONTENT)
    )
    checks.append(
        (f"{label} tokens_evaluated == {EXPECTED_PROMPT_TOKENS_PER_REQUEST}",
         body.get("tokens_evaluated") == EXPECTED_PROMPT_TOKENS_PER_REQUEST)
    )
    checks.append(
        (f"{label} tokens_predicted == {EXPECTED_PREDICTED_PER_REQUEST}",
         body.get("tokens_predicted") == EXPECTED_PREDICTED_PER_REQUEST)
    )

# ---- metrics deltas ----
m_before = parse_metrics(ROOT / "metrics_before.txt")
m_after = parse_metrics(ROOT / "metrics_after.txt")

delta_lines = []
for key, expected in EXPECTED_DELTAS.items():
    if key not in m_before or key not in m_after:
        if key == "llamacpp:n_decode_total":
            delta_lines.append(f"{key}: not exposed by this build (skipped)")
            continue
        checks.append((f"metrics has {key}", False))
        delta_lines.append(f"{key}: MISSING in before/after")
        continue
    actual = m_after[key] - m_before[key]
    ok = abs(actual - expected) < 1e-6
    checks.append((f"{key} delta == {expected}", ok))
    delta_lines.append(
        f"{key}: before={m_before[key]:.0f} after={m_after[key]:.0f} delta={actual:.0f} expected={expected}"
    )

# ---- timing stats ----
measured_wall_ms = [float(r["wall_ms"]) for r in measured_rows]
measured_client_tps = [float(r["client_tokens_per_sec"]) for r in measured_rows]
measured_prompt_ms = [float(r["prompt_ms"]) for r in measured_rows if r["prompt_ms"]]
measured_predicted_ms = [float(r["predicted_ms"]) for r in measured_rows if r["predicted_ms"]]
measured_server_pps = [float(r["prompt_per_second"]) for r in measured_rows if r["prompt_per_second"]]
measured_server_gps = [float(r["predicted_per_second"]) for r in measured_rows if r["predicted_per_second"]]

warmup_wall_ms = float(warmup_rows[0]["wall_ms"])

# ---- summary text ----
out = []
out.append("=== Aligned llama-server timing summary (comparison_aligned/server) ===")
out.append("")
out.append("requests: 1 warmup + 5 measured")
out.append(f"warmup wall_ms: {warmup_wall_ms:.3f}")
out.append("")
out.append("--- measured client wall_ms (ms) ---")
out.append(fmt_stats(stats(measured_wall_ms), "ms"))
out.append("")
out.append("--- measured client tokens/sec (predicted/wall) ---")
out.append(fmt_stats(stats(measured_client_tps), "tps"))
out.append("")
out.append("--- measured server prompt_ms (ms) ---")
out.append(fmt_stats(stats(measured_prompt_ms), "ms"))
out.append("--- measured server predicted_ms (ms) ---")
out.append(fmt_stats(stats(measured_predicted_ms), "ms"))
out.append("--- measured server prompt_per_second (tokens/s) ---")
out.append(fmt_stats(stats(measured_server_pps), "tps"))
out.append("--- measured server predicted_per_second (tokens/s) ---")
out.append(fmt_stats(stats(measured_server_gps), "tps"))
out.append("")
out.append("--- per-request rows ---")
for r in rows:
    out.append(
        f"{r['label']:<12} kind={r['kind']:<8} status={r['http_status']} "
        f"wall_ms={float(r['wall_ms']):.3f} "
        f"tp={r['tokens_predicted']} te={r['tokens_evaluated']} "
        f"prompt_ms={r['prompt_ms']} predicted_ms={r['predicted_ms']}"
    )
out.append("")
out.append("--- metrics deltas ---")
out.extend(delta_lines)
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

text = "\n".join(out) + "\n"
(ROOT / "summary.txt").write_text(text)
print(text)
