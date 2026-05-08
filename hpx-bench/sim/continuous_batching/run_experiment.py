"""CLI entry point for the continuous-batching simulator.

Examples:

    python run_experiment.py \\
        --workload exp11_like_all_short \\
        --policies fifo_context_pool,static_batching,continuous_batching \\
        --n-slots 4 --n-batch 128 --seed 42

    python run_experiment.py --workload mixed_realistic --seed 7

The default `run_id` is a deterministic 12-char config hash, so the same
config + same seed lands in the same `results/<run_id>/` directory. Use
`--timestamped` to prefix with a wallclock for ad-hoc sessions.

See `docs/hpx/continuous_batching_simulator_design.md`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure sibling modules import.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sim import CostParams, SchedulerState, run_simulation  # noqa: E402
from policies import POLICIES  # noqa: E402
from workloads import WORKLOADS  # noqa: E402
from metrics import (  # noqa: E402
    aggregate,
    write_iterations_csv,
    write_requests_csv,
    write_summary_csv,
)

DEFAULT_RESULTS_ROOT = _HERE / "results"


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous-batching simulator")
    p.add_argument(
        "--workload",
        default="exp11_like_all_short",
        choices=list(WORKLOADS.keys()),
    )
    p.add_argument(
        "--policies",
        default="fifo_context_pool,static_batching,continuous_batching",
        help="Comma-separated list of policy names.",
    )
    p.add_argument("--n-slots", type=int, default=4)
    p.add_argument("--n-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--workload-cap",
        type=int,
        default=100,
        help=(
            "total_cap for cap-aware workloads (mixed_realistic and the "
            "Phase 2B mixed/bursty workloads). Workloads that don't use "
            "a cap (e.g. exp11_like_all_short) ignore this flag."
        ),
    )
    p.add_argument("--cost-base", type=float, default=1.0)
    p.add_argument("--cost-prompt", type=float, default=0.05)
    p.add_argument("--cost-decode", type=float, default=0.5)
    p.add_argument("--cost-slot", type=float, default=0.0)
    p.add_argument("--cost-stream", type=float, default=0.0)
    p.add_argument(
        "--run-id",
        default=None,
        help="Override run id (default: 12-char config hash).",
    )
    p.add_argument(
        "--timestamped",
        action="store_true",
        help="Prefix the default run id with a UTC timestamp.",
    )
    p.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Base directory for run outputs.",
    )
    return p.parse_args(argv)


def build_config_payload(
    args: argparse.Namespace, policies: List[str], cp: CostParams
) -> Dict[str, Any]:
    return {
        "workload": args.workload,
        "workload_cap": args.workload_cap,
        "policies": sorted(policies),
        "n_slots": args.n_slots,
        "n_batch": args.n_batch,
        "seed": args.seed,
        "cost_params": asdict(cp),
    }


def config_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_revision() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            cwd=_HERE,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_summary_md(
    run_dir: Path, meta: Dict[str, Any], rows: List[Dict[str, Any]]
) -> None:
    lines: List[str] = []
    lines.append(f"# Simulator run `{meta['run_id']}`")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- command: `{meta['command']}`")
    lines.append(f"- seed: `{meta['seed']}`")
    lines.append(f"- run_id: `{meta['run_id']}`")
    lines.append(f"- config_hash: `{meta['config_hash']}`")
    lines.append(f"- git_revision: `{meta['git_revision']}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    headers = [
        "policy",
        "n_slots",
        "n_batch",
        "n_completed",
        "makespan_ms",
        "tokens_per_decode_call",
        "decode_calls",
        "total_latency_ms_p50",
        "total_latency_ms_p95",
        "total_latency_ms_p99",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
    ]
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


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for name in policies:
        if name not in POLICIES:
            print(f"unknown policy: {name}", file=sys.stderr)
            return 2

    cp = CostParams(
        base_decode_step_cost_ms=args.cost_base,
        per_prompt_token_cost_ms=args.cost_prompt,
        per_decode_token_cost_ms=args.cost_decode,
        per_active_slot_overhead_ms=args.cost_slot,
        streaming_emit_cost_ms=args.cost_stream,
    )

    payload = build_config_payload(args, policies, cp)
    chash = config_hash(payload)
    run_id = args.run_id or (
        f"{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{chash}"
        if args.timestamped
        else chash
    )

    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(payload)
    meta.update(
        {
            "run_id": run_id,
            "config_hash": chash,
            "git_revision": git_revision(),
            "command": " ".join(shlex.quote(a) for a in sys.argv),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )

    # All workload callables accept **_kw, so passing seed/total_cap to every
    # one of them is safe; workloads that don't use a parameter just ignore it.
    # `exp11_like_all_short` ignores both and keeps its native 200-request
    # baseline regardless of --workload-cap.
    requests = WORKLOADS[args.workload](
        seed=args.seed, total_cap=args.workload_cap
    )

    summary_rows: List[Dict[str, Any]] = []
    for policy_name in policies:
        policy_cls = POLICIES[policy_name]
        policy = policy_cls()
        cont_batching = policy_name != "fifo_context_pool"
        state = SchedulerState.initialize(
            requests=list(requests),
            n_slots=args.n_slots,
            n_batch=args.n_batch,
            cont_batching=cont_batching,
        )
        run_simulation(state, policy, cp)

        sub = run_dir / policy_name
        sub.mkdir(parents=True, exist_ok=True)
        write_requests_csv(state, sub / "requests.csv")
        write_iterations_csv(state, sub / "iterations.csv")
        agg = aggregate(state)
        agg["policy"] = policy_name
        agg["workload"] = args.workload
        agg["n_slots"] = args.n_slots
        agg["n_batch"] = args.n_batch
        agg["seed"] = args.seed
        summary_rows.append(agg)

    write_summary_csv(summary_rows, run_dir / "summary.csv")
    write_summary_md(run_dir, meta, summary_rows)

    print(f"run_id: {run_id}")
    print(f"output: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
