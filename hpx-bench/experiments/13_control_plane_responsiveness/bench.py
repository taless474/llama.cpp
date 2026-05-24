#!/usr/bin/env python3
"""Experiment 13 — control-plane responsiveness driver (W2/W3).

This driver invokes the standalone C++ benchmark binary

    llama-hpx-continuous-batch-responsiveness-bench

once per (mode, workload) cell, strictly sequentially. Each invocation
is its own process that loads the model, runs the workload on the HPX
engine task, emits one JSONL record per request to a per-cell file, and
prints a terminal "RESPONSIVENESS_BENCH: PASS|FAIL" line to stdout.

Scope:
  modes     : default_os1, default_os2, engine_pool_os2
  workloads : w2 (queued-cancel), w3 (multi-request streaming)

Stream-drain model (recorded per record as drain_mode):
  Phase 1 drained W3 streams serially on one consumer (drain_mode would
  read "legacy" for that old raw). Phase 2 drains the K W3 streams
  concurrently, one foreign consumer thread per stream
  (drain_mode="concurrent"); W2 is always "single". The driver itself is
  unchanged across phases — concurrency lives inside the binary's W3
  workload. summary.csv carries drain_mode so phases stay separable.

Hard constraint: NEVER run two model-loading processes at once. Every
cell is a blocking subprocess.run; there is no concurrency in THIS
driver. This is deliberate — concurrent model loads would perturb the
microsecond-scale control-plane timings this experiment measures.

This driver does not touch engine semantics, cancellation semantics, or
stream-close ordering. It only launches the binary and aggregates the
JSONL it emits.

Output layout (all under results/<run_id>/):
  manifest.json                      run metadata + per-cell status
  commands.txt                       exact command lines, one per cell
  raw/<mode>_<workload>.jsonl        per-request JSONL (incl. warmups)
  logs/<mode>_<workload>.stdout      binary stdout (PASS/FAIL line)
  logs/<mode>_<workload>.stderr      binary stderr (llama/ggml chatter)
  summary.csv                        p50/p95/p99/max/mean/min/N per metric

Aggregation excludes warmup records (is_warmup==1), matching the
binary's own invariant pass.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Local defaults. These match the current checkout's common layout but are
# all overridable on the command line — do not assume they are universal.
# ---------------------------------------------------------------------------
DEFAULT_BINARY = (
    "/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/"
    "llama-hpx-continuous-batch-responsiveness-bench"
)
DEFAULT_MODEL = (
    "/Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

ALL_MODES = ["default_os1", "default_os2", "engine_pool_os2"]
ALL_WORKLOADS = ["w2", "w3"]

# Metrics aggregated per workload. Scalar metrics are read straight from
# the JSONL fields the binary emits; list metrics (inter_token_gap_us) are
# flattened across all measured records before percentiles are computed.
# submit_to_stream_close_us is derived here from raw submitter stamps.
W2_SCALAR_METRICS = [
    "cancel_to_observed_us",
    "cancel_to_future_ready_us",
    "cancel_to_stream_close_us",
]
W3_SCALAR_METRICS = [
    "submit_to_admitted_us",
    "submit_to_first_token_us",
    "submit_to_complete_us",
    "submit_to_stream_close_us",  # derived below
]
W3_LIST_METRICS = [
    "inter_token_gap_us",
]

PASS_TOKEN = "RESPONSIVENESS_BENCH: PASS"
FAIL_TOKEN = "RESPONSIVENESS_BENCH: FAIL"


# ---------------------------------------------------------------------------
# Small pure-stdlib statistics (no numpy dependency).
# ---------------------------------------------------------------------------
def percentile(sorted_vals, q):
    """Linear-interpolation percentile (numpy 'linear'), q in [0,100]."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def summarize(values):
    """Return dict of N/mean/min/p50/p95/p99/max for a value list."""
    if not values:
        return {
            "N": 0, "mean": None, "min": None,
            "p50": None, "p95": None, "p99": None, "max": None,
        }
    s = sorted(values)
    return {
        "N": len(s),
        "mean": sum(s) / len(s),
        "min": float(s[0]),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "max": float(s[-1]),
    }


# ---------------------------------------------------------------------------
# Metadata helpers.
# ---------------------------------------------------------------------------
def git_info(repo_dir):
    out = {"sha": None, "branch": None, "dirty": None}
    def run(args):
        try:
            r = subprocess.run(
                ["git"] + args, cwd=repo_dir,
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            return None
        return None
    out["sha"] = run(["rev-parse", "HEAD"])
    out["branch"] = run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["status", "--porcelain"])
    if status is not None:
        out["dirty"] = bool(status.strip())
    return out


def model_fingerprint(model_path, max_hash_bytes):
    """Always record size+mtime; sha256 only when cheap (<= max_hash_bytes)."""
    fp = {"path": model_path, "size_bytes": None, "mtime": None,
          "sha256": None, "sha256_skipped_reason": None}
    try:
        st = os.stat(model_path)
        fp["size_bytes"] = st.st_size
        fp["mtime"] = datetime.datetime.utcfromtimestamp(
            st.st_mtime).isoformat() + "Z"
    except OSError as e:
        fp["sha256_skipped_reason"] = "stat failed: %s" % e
        return fp
    if fp["size_bytes"] is not None and fp["size_bytes"] > max_hash_bytes:
        fp["sha256_skipped_reason"] = (
            "size %d > max_hash_bytes %d" % (fp["size_bytes"], max_hash_bytes))
        return fp
    try:
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    except OSError as e:
        fp["sha256_skipped_reason"] = "read failed: %s" % e
    return fp


# ---------------------------------------------------------------------------
# Per-cell run.
# ---------------------------------------------------------------------------
def build_cmd(args, mode, workload, out_jsonl):
    cmd = [
        args.binary,
        "--model", args.model,
        "--mode", mode,
        "--workload", workload,
        "--trials", str(args.trials),
        "--warmup-trials", str(args.warmup_trials),
        "--decode-budget", str(args.decode_budget),
        "--prompt", args.prompt,
        "--output", out_jsonl,
        "--trial-id", "%s_%s" % (mode, workload),
    ]
    if workload == "w3":
        cmd += ["--n-streams", str(args.n_streams)]
    return cmd


def parse_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A malformed line is a binary-side fault; surface by
                # leaving it out — the cell's status will already be FAIL
                # if the PASS line was not printed.
                pass
    return records


def measured_records(records):
    return [r for r in records if not r.get("is_warmup")]


def derive_w3_fields(records):
    """Inject submit_to_stream_close_us from raw submitter stamps."""
    for r in records:
        sc = r.get("stream_close_us")
        su = r.get("submit_us")
        if isinstance(sc, int) and isinstance(su, int) and sc >= 0 and su >= 0:
            r["submit_to_stream_close_us"] = sc - su
        else:
            r["submit_to_stream_close_us"] = None


def collect_scalar(records, field):
    vals = []
    for r in records:
        v = r.get(field)
        if isinstance(v, (int, float)) and v >= 0:
            vals.append(v)
    return vals


def collect_list(records, field):
    vals = []
    for r in records:
        v = r.get(field)
        if isinstance(v, list):
            vals.extend(x for x in v if isinstance(x, (int, float)))
    return vals


def metrics_for(workload):
    if workload == "w2":
        return [(m, "scalar") for m in W2_SCALAR_METRICS]
    return ([(m, "scalar") for m in W3_SCALAR_METRICS]
            + [(m, "list") for m in W3_LIST_METRICS])


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # repo root: .../hpx-bench/experiments/13_.../ -> up 3
    repo_dir = os.path.abspath(os.path.join(here, "..", "..", ".."))

    p = argparse.ArgumentParser(description="Experiment 13 driver (W2/W3).")
    p.add_argument("--binary", default=DEFAULT_BINARY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--modes", default=",".join(ALL_MODES),
                   help="comma-separated subset of: " + ",".join(ALL_MODES))
    p.add_argument("--workloads", default=",".join(ALL_WORKLOADS),
                   help="comma-separated subset of: " + ",".join(ALL_WORKLOADS))
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--warmup-trials", type=int, default=1)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--decode-budget", type=int, default=8)
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--run-id", default=None,
                   help="override the auto timestamp run id")
    p.add_argument("--label", default="phase1",
                   help="suffix appended to the auto run id")
    p.add_argument("--notes", default="",
                   help="free-text run notes recorded in manifest.json")
    p.add_argument("--max-hash-mib", type=int, default=1536,
                   help="hash the model only if <= this many MiB")
    p.add_argument("--timeout", type=int, default=900,
                   help="per-cell subprocess timeout (seconds)")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    for m in modes:
        if m not in ALL_MODES:
            sys.exit("unknown mode: %s" % m)
    for w in workloads:
        if w not in ALL_WORKLOADS:
            sys.exit("unknown workload: %s" % w)

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s" % args.binary)
    if not os.path.exists(args.model):
        sys.exit("model not found: %s" % args.model)

    if args.run_id:
        run_id = args.run_id
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = "%s-%s" % (ts, args.label)

    run_dir = os.path.join(here, "results", run_id)
    raw_dir = os.path.join(run_dir, "raw")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("Experiment 13 — control-plane responsiveness")
    print("  run_id     : %s" % run_id)
    print("  run_dir    : %s" % run_dir)
    print("  modes      : %s" % ", ".join(modes))
    print("  workloads  : %s" % ", ".join(workloads))
    print("  trials     : %d  (warmup %d)" % (args.trials, args.warmup_trials))
    print("  n_streams  : %d   decode_budget : %d"
          % (args.n_streams, args.decode_budget))
    print()

    cells = []          # list of cell-status dicts for manifest
    cmd_lines = []      # exact command strings for commands.txt
    summary_rows = []   # rows for summary.csv

    overall_ok = True

    # Strictly sequential: outer mode, inner workload. One blocking
    # subprocess per cell; no concurrent model loads.
    for mode in modes:
        for workload in workloads:
            cell = "%s_%s" % (mode, workload)
            out_jsonl = os.path.join(raw_dir, cell + ".jsonl")
            out_stdout = os.path.join(log_dir, cell + ".stdout")
            out_stderr = os.path.join(log_dir, cell + ".stderr")
            cmd = build_cmd(args, mode, workload, out_jsonl)
            cmd_str = " ".join(cmd)
            cmd_lines.append(cmd_str)

            print("[run] %-20s ..." % cell, end="", flush=True)

            status = {
                "cell": cell, "mode": mode, "workload": workload,
                "cmd": cmd, "returncode": None, "pass_line": None,
                "passed": False, "raw_jsonl": os.path.relpath(out_jsonl, run_dir),
                "n_records": 0, "n_measured": 0, "drain_mode": None,
                "error": None,
            }

            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.timeout)
                status["returncode"] = r.returncode
                with open(out_stdout, "w") as f:
                    f.write(r.stdout)
                with open(out_stderr, "w") as f:
                    f.write(r.stderr)
                pass_line = ""
                for line in r.stdout.splitlines():
                    if line.startswith(PASS_TOKEN) or line.startswith(FAIL_TOKEN):
                        pass_line = line.strip()
                status["pass_line"] = pass_line
                status["passed"] = (r.returncode == 0
                                   and pass_line.startswith(PASS_TOKEN))
            except subprocess.TimeoutExpired:
                status["error"] = "timeout after %ds" % args.timeout
            except Exception as e:  # noqa: BLE001
                status["error"] = str(e)

            records = parse_jsonl(out_jsonl)
            status["n_records"] = len(records)
            meas = measured_records(records)
            status["n_measured"] = len(meas)

            # drain_mode is emitted per record by the binary ("single"
            # for W2, "concurrent" for W3 Phase 2). Old Phase 1 raw files
            # predate the field -> "legacy". Captured so Phase 1 (serial)
            # and Phase 2 (concurrent) data stay distinguishable.
            sample = meas[0] if meas else (records[0] if records else {})
            drain_mode = sample.get("drain_mode", "legacy")
            status["drain_mode"] = drain_mode

            if workload == "w3":
                derive_w3_fields(meas)

            for metric, kind in metrics_for(workload):
                if kind == "scalar":
                    vals = collect_scalar(meas, metric)
                else:
                    vals = collect_list(meas, metric)
                s = summarize(vals)
                summary_rows.append({
                    "run_id": run_id, "mode": mode, "workload": workload,
                    "drain_mode": drain_mode,
                    "metric": metric, "unit": "us",
                    "N": s["N"], "mean": s["mean"], "min": s["min"],
                    "p50": s["p50"], "p95": s["p95"], "p99": s["p99"],
                    "max": s["max"],
                })

            if status["passed"]:
                print(" PASS (%d measured)" % status["n_measured"])
            else:
                overall_ok = False
                print(" FAIL (rc=%s err=%s line=%r)"
                      % (status["returncode"], status["error"],
                         status["pass_line"]))

            cells.append(status)

    # ---- write commands.txt -------------------------------------------------
    with open(os.path.join(run_dir, "commands.txt"), "w") as f:
        for line in cmd_lines:
            f.write(line + "\n")

    # ---- write summary.csv --------------------------------------------------
    csv_fields = ["run_id", "mode", "workload", "drain_mode", "metric", "unit",
                  "N", "mean", "min", "p50", "p95", "p99", "max"]
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in summary_rows:
            out = dict(row)
            for k in ("mean", "min", "p50", "p95", "p99", "max"):
                if out[k] is not None:
                    out[k] = round(out[k], 3)
            w.writerow(out)

    # ---- write manifest.json ------------------------------------------------
    manifest = {
        "experiment": "13_control_plane_responsiveness",
        "phase": "phase1_w2_w3",
        "run_id": run_id,
        "timestamp_local": datetime.datetime.now().isoformat(),
        "binary": args.binary,
        "git": git_info(repo_dir),
        "model": model_fingerprint(args.model, args.max_hash_mib * (1 << 20)),
        "config": {
            "modes": modes, "workloads": workloads,
            "trials": args.trials, "warmup_trials": args.warmup_trials,
            "n_streams": args.n_streams, "decode_budget": args.decode_budget,
            "prompt": args.prompt,
        },
        "notes": args.notes,
        "overall_ok": overall_ok,
        "cells": cells,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print()
    print("wrote: %s" % os.path.join(run_dir, "summary.csv"))
    print("wrote: %s" % os.path.join(run_dir, "manifest.json"))
    print("overall_ok: %s" % overall_ok)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
