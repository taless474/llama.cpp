"""
Phase 2 helper — drive the HPX-vs-std performance matrix by invoking
_run_one_trial.py in schedule order. Does NOT summarize. Does NOT gate.
Summarizers run separately afterwards.

Reads:
  local/baselines/perf_hpx_vs_std_matrix/_schedule.json

Calls (per selected entry):
  python3 _run_one_trial.py --global-index <N>

Skips entries whose <output_dir>/bench.exit_code.txt already exists,
unless --force is passed. This makes a re-run idempotent and resumable
after interruption.

Selection flags (compose via intersection):
  --start-index N    inclusive lower bound on global_index (default 0)
  --end-index   N    inclusive upper bound on global_index (default last)
  --limit       N    cap the number of entries processed after the
                     start/end window is applied
  --force            do not skip even if bench.exit_code.txt exists

Driver logs:
  local/baselines/perf_hpx_vs_std_matrix/driver_logs/run-<timestamp>/
    driver_summary.csv   one row per processed entry: gidx, cell, layer,
                         backend, trial, action (RUN|SKIP), helper_rc, wall_ms
    driver.log           full subprocess stdout/stderr per RUN entry

Stop policy:
  Stops on the first nonzero return code from _run_one_trial.py and exits
  with that code. SKIP entries do not affect the stop policy.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT     = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_hpx_vs_std_matrix")
SCHEDULE = ROOT / "_schedule.json"
RUN_ONE  = ROOT / "_run_one_trial.py"
LOGS     = ROOT / "driver_logs"


def parse_args():
    p = argparse.ArgumentParser(
        description="run HPX-vs-std performance matrix schedule entries"
    )
    p.add_argument("--start-index", type=int, default=None,
                   help="inclusive lower bound on global_index")
    p.add_argument("--end-index", type=int, default=None,
                   help="inclusive upper bound on global_index")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on number of entries to process")
    p.add_argument("--force", action="store_true",
                   help="re-run entries whose bench.exit_code.txt already exists")
    return p.parse_args()


def load_schedule():
    if not SCHEDULE.exists():
        sys.exit(
            f"schedule not found: {SCHEDULE}\n"
            "run _make_schedule.py first."
        )
    return json.loads(SCHEDULE.read_text())


def select_entries(schedule, args):
    entries = sorted(schedule["schedule"], key=lambda e: e["global_index"])
    n = len(entries)
    if n == 0:
        sys.exit("schedule is empty")

    lo = 0 if args.start_index is None else args.start_index
    hi = (n - 1) if args.end_index is None else args.end_index

    if lo < 0 or lo >= n:
        sys.exit(f"--start-index out of range: {lo} (valid 0..{n - 1})")
    if hi < lo or hi >= n:
        sys.exit(f"--end-index out of range: {hi} (valid {lo}..{n - 1})")

    selected = [e for e in entries if lo <= e["global_index"] <= hi]
    if args.limit is not None:
        if args.limit < 0:
            sys.exit(f"--limit must be >= 0; got {args.limit}")
        selected = selected[:args.limit]
    return selected, lo, hi


def main():
    args = parse_args()
    schedule = load_schedule()
    selected, lo, hi = select_entries(schedule, args)

    LOGS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOGS / f"run-{ts}"
    log_dir.mkdir(parents=True, exist_ok=False)

    summary_path = log_dir / "driver_summary.csv"
    full_path    = log_dir / "driver.log"

    summary = summary_path.open("w")
    full    = full_path.open("w")

    summary.write(
        "global_index,cell_id,layer,backend,trial_index,action,"
        "helper_rc,wall_ms\n"
    )
    full.write(f"# driver run started: {ts}\n")
    full.write(f"# schedule entries total: {len(schedule['schedule'])}\n")
    full.write(f"# selected window: start={lo} end={hi}  count={len(selected)}\n")
    full.write(f"# limit: {args.limit}  force: {args.force}\n")
    full.write(f"# helper: {RUN_ONE}\n")
    full.write("# --- per-entry log follows ---\n")
    full.flush()

    n_run = 0
    n_skip = 0
    overall_rc = 0
    stopped_at = None

    for entry in selected:
        gidx = entry["global_index"]
        out_dir = ROOT / entry["output_dir"]
        ec_path = out_dir / "bench.exit_code.txt"

        if ec_path.exists() and not args.force:
            summary.write(
                f"{gidx},{entry['cell_id']},{entry['layer']},"
                f"{entry['backend']},{entry['trial_index']},SKIP,,0\n"
            )
            summary.flush()
            n_skip += 1
            continue

        cmd = ["python3", str(RUN_ONE), "--global-index", str(gidx)]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.monotonic()
        wall_ms = (t1 - t0) * 1000.0

        full.write(f"\n=== gidx={gidx} cell={entry['cell_id']} "
                   f"layer={entry['layer']} backend={entry['backend']} "
                   f"trial={entry['trial_index']} ===\n")
        full.write(f"command: {' '.join(cmd)}\n")
        full.write(f"returncode: {proc.returncode}\n")
        full.write(f"wall_ms: {wall_ms:.3f}\n")
        full.write("--- helper stdout ---\n")
        full.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            full.write("\n")
        full.write("--- helper stderr ---\n")
        full.write(proc.stderr)
        if proc.stderr and not proc.stderr.endswith("\n"):
            full.write("\n")
        full.flush()

        summary.write(
            f"{gidx},{entry['cell_id']},{entry['layer']},"
            f"{entry['backend']},{entry['trial_index']},RUN,"
            f"{proc.returncode},{wall_ms:.3f}\n"
        )
        summary.flush()
        n_run += 1

        if proc.returncode != 0:
            overall_rc = proc.returncode
            stopped_at = gidx
            break

    end_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    full.write(f"\n# driver run ended: {end_ts}\n")
    full.write(f"# entries RUN: {n_run}\n")
    full.write(f"# entries SKIP: {n_skip}\n")
    full.write(f"# stopped_at_gidx: {stopped_at}\n")
    full.write(f"# overall_rc: {overall_rc}\n")
    full.close()
    summary.close()

    print(f"driver log dir: {log_dir}")
    print(f"selected: {len(selected)}  RUN: {n_run}  SKIP: {n_skip}")
    if stopped_at is not None:
        print(f"stopped at gidx={stopped_at}; helper rc={overall_rc}")
    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
