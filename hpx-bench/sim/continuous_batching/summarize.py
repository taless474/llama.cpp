"""Re-summarize an existing simulator run directory.

Reads `<run_dir>/config.json` and each `<run_dir>/<policy>/{requests,iterations}.csv`,
regenerates `summary.md` with the side-by-side comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r") as f:
        return list(csv.DictReader(f))


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _row_for_policy(policy_dir: Path) -> Dict[str, Any]:
    reqs = _read_csv(policy_dir / "requests.csv")
    its = _read_csv(policy_dir / "iterations.csv")
    if not reqs:
        return {"policy": policy_dir.name, "n_completed": 0}
    latencies = [float(r["total_latency_ms"]) for r in reqs]
    ttfts = [float(r["time_to_first_token_ms"]) for r in reqs]
    finishes = [float(r["finish_time_ms"]) for r in reqs]
    arrivals = [float(r["arrival_time_ms"]) for r in reqs]
    decode_calls = sum(
        int(it["decode_calls_in_iteration"]) for it in its
    )
    total_rows = sum(int(it["batch_size"]) for it in its)
    makespan = max(finishes) - min(arrivals)
    return {
        "policy": policy_dir.name,
        "n_completed": len(reqs),
        "makespan_ms": makespan,
        "decode_calls": decode_calls,
        "tokens_per_decode_call": (
            (total_rows / decode_calls) if decode_calls else 0.0
        ),
        "total_latency_ms_p50": _percentile(latencies, 50),
        "total_latency_ms_p95": _percentile(latencies, 95),
        "total_latency_ms_p99": _percentile(latencies, 99),
        "ttft_ms_p50": _percentile(ttfts, 50),
        "ttft_ms_p95": _percentile(ttfts, 95),
        "ttft_ms_p99": _percentile(ttfts, 99),
    }


def summarize_run(run_dir: Path) -> int:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        print(f"no config.json in {run_dir}", file=sys.stderr)
        return 2
    meta = json.loads(cfg_path.read_text())

    rows: List[Dict[str, Any]] = []
    for policy_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if not (policy_dir / "requests.csv").exists():
            continue
        rows.append(_row_for_policy(policy_dir))

    headers = [
        "policy",
        "n_completed",
        "makespan_ms",
        "decode_calls",
        "tokens_per_decode_call",
        "total_latency_ms_p50",
        "total_latency_ms_p95",
        "total_latency_ms_p99",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
    ]
    lines: List[str] = []
    lines.append(
        f"# Simulator run `{meta.get('run_id', '?')}` (re-summarized)"
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- command: `{meta.get('command', '?')}`")
    lines.append(f"- seed: `{meta.get('seed', '?')}`")
    lines.append(f"- run_id: `{meta.get('run_id', '?')}`")
    lines.append(f"- config_hash: `{meta.get('config_hash', '?')}`")
    lines.append(f"- git_revision: `{meta.get('git_revision', '?')}`")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells: List[str] = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines))
    print(f"updated {run_dir / 'summary.md'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-summarize a simulator run dir."
    )
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    return summarize_run(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
