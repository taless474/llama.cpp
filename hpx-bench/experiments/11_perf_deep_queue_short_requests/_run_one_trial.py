"""
Run exactly one entry of the deep-queue short-request schedule. No driving;
no gate enforcement. Summarizers will gate later.

Selection (must pass exactly one of):
    --global-index N
    --layer <layer> --backend <std|hpx> --trial-index N

Reads:
    runs/_schedule.json

Writes (under runs/<entry.output_dir>/):
    bench.stdout
    bench.stderr
    bench.exit_code.txt
    process_wall_ms.txt
    per_repeat.csv               (one row per harness output line, in
                                  completion order; the "completion_index"
                                  column is the i printed in "req[i]" by
                                  the harness, not the original
                                  request_index — see facts.md "Harness
                                  output convention")
    request_summary.txt
    hpx_trace.txt
    binary_used.txt
    git_head.txt
    build_info.txt
    schedule_entry.json

Trace env:
    LLAMA_SERVING_BENCH_HPX_TRACE=1 only when entry.trace_enabled is true;
    otherwise the variable is removed from the child env.

Binary / model paths are resolved in this order:
    1. environment override
            LLAMA_SERVING_BENCH_BIN
            LLAMA_MODEL
    2. defaults derived from the repo's grandparent directory
            <repo>/../builds/llama-hpx-hpx-on/bin/llama-serving-bench
            <repo>/../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

This helper does NOT enforce correctness or lifecycle gates. Per-trial
artifacts are captured raw; a separate summarizer applies the gates and
records PASS/FAIL.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
RUNS    = EXP_DIR / "runs"
REPO    = EXP_DIR.parents[2]

DEFAULT_BIN   = REPO.parent / "builds" / "llama-hpx-hpx-on" / "bin" / "llama-serving-bench"
DEFAULT_MODEL = REPO.parent / "models"  / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

BINARY = Path(os.environ.get("LLAMA_SERVING_BENCH_BIN", str(DEFAULT_BIN)))
MODEL  = Path(os.environ.get("LLAMA_MODEL",            str(DEFAULT_MODEL)))

PROMPT = "Hello, my name is"

SCHEDULE_PATH = RUNS / "_schedule.json"

REQ_RE = re.compile(
    r"^\[serving-bench\] req\[(\d+)\] "
    r"status=(\S+) "
    r"n_tokens_generated=(\d+) "
    r"generated_token_hash=(0x[0-9a-fA-F]+) "
    r"ttft_ms=([0-9.]+) "
    r"total_ms=([0-9.]+)\s*$"
)

TRACE_RE = re.compile(
    r"^\[serving-bench\] (?:"
    r"hpx_runtime_start_once|"
    r"engine_hpx ready|"
    r"hpx_runtime_stop|"
    r"req\[\d+\] acquire|"
    r"req\[\d+\] release"
    r")"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="run one deep-queue short-request schedule entry"
    )
    p.add_argument("--global-index", type=int, default=None)
    p.add_argument("--layer", type=str, default=None,
                   choices=["correctness_trace_on", "timing_trace_off"])
    p.add_argument("--backend", type=str, default=None,
                   choices=["std", "hpx"])
    p.add_argument("--trial-index", type=int, default=None)
    return p.parse_args()


def load_schedule():
    if not SCHEDULE_PATH.exists():
        sys.exit(
            f"schedule not found: {SCHEDULE_PATH}\n"
            "run _make_schedule.py first."
        )
    with SCHEDULE_PATH.open() as f:
        return json.load(f)


def select_entry(schedule, args):
    entries = schedule["schedule"]

    if args.global_index is not None:
        explicit_any = any(
            v is not None for v in (args.layer, args.backend, args.trial_index)
        )
        if explicit_any:
            sys.exit(
                "pass either --global-index OR the explicit "
                "--layer/--backend/--trial-index triple, not both."
            )
        for e in entries:
            if e["global_index"] == args.global_index:
                return e
        sys.exit(f"no entry with global_index={args.global_index}")

    needed = ("layer", "backend", "trial_index")
    vals   = (args.layer, args.backend, args.trial_index)
    if any(v is None for v in vals):
        sys.exit(
            "must pass either --global-index, or all of "
            "--layer, --backend, --trial-index."
        )
    for e in entries:
        if (e["layer"], e["backend"], e["trial_index"]) == vals:
            return e
    sys.exit(f"no schedule entry matches {dict(zip(needed, vals))}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_run(args):
    p = subprocess.run(
        ["git", *args], cwd=str(REPO), capture_output=True, text=True
    )
    return p.stdout


def build_argv(entry):
    plan_csv = ",".join(str(x) for x in entry["max_tokens_plan"])
    return [
        str(BINARY),
        "--backend",          entry["backend"],
        "--model",            str(MODEL),
        "--prompt",           PROMPT,
        "--n-contexts",       str(entry["n_contexts"]),
        "--n-concurrent",     str(entry["n_concurrent"]),
        "--n-requests",       str(entry["n_requests"]),
        "--max-tokens",       str(entry["max_tokens"]),
        "--max-tokens-plan",  plan_csv,
        "--ctx-size",         str(entry["ctx_size"]),
        "--batch-size",       str(entry["batch_size"]),
        "--n-threads",        str(entry["n_threads"]),
        "--seed-base",        str(entry["seed_base"]),
    ]


def main():
    args     = parse_args()
    schedule = load_schedule()
    entry    = select_entry(schedule, args)

    out_dir = RUNS / entry["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if entry["trace_enabled"]:
        env["LLAMA_SERVING_BENCH_HPX_TRACE"] = "1"
    else:
        env.pop("LLAMA_SERVING_BENCH_HPX_TRACE", None)

    argv = build_argv(entry)

    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, env=env)
    t1 = time.monotonic()
    process_wall_ms = (t1 - t0) * 1000.0

    stdout    = proc.stdout.decode("utf-8", errors="replace")
    stderr    = proc.stderr.decode("utf-8", errors="replace")
    exit_code = proc.returncode

    (out_dir / "bench.stdout").write_text(stdout)
    (out_dir / "bench.stderr").write_text(stderr)
    (out_dir / "bench.exit_code.txt").write_text(f"{exit_code}\n")
    (out_dir / "process_wall_ms.txt").write_text(
        f"{process_wall_ms:.3f}\n"
        "note: outer time.monotonic() around subprocess.run; informational only.\n"
        f"layer: {entry['layer']}\n"
        f"backend: {entry['backend']}\n"
        f"trial_index: {entry['trial_index']}\n"
        f"measured_for_timing: {entry['measured_for_timing']}\n"
        "process wall includes runtime startup/shutdown, model load, harness teardown.\n"
    )

    trace_lines = [line for line in stderr.splitlines() if TRACE_RE.match(line)]
    (out_dir / "hpx_trace.txt").write_text(
        ("\n".join(trace_lines) + "\n") if trace_lines else ""
    )

    rows = []
    summary_lines = []
    for line in stdout.splitlines():
        m = REQ_RE.match(line)
        if not m:
            continue
        completion_index = int(m.group(1))
        status           = m.group(2)
        n_tok            = int(m.group(3))
        h                = m.group(4)
        ttft_ms          = float(m.group(5))
        total_ms         = float(m.group(6))
        tps              = (n_tok / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        total_minus_ttft_ms = total_ms - ttft_ms
        rows.append((
            completion_index, status, n_tok, h, ttft_ms, total_ms, tps,
            total_minus_ttft_ms,
        ))
        summary_lines.append(
            f"req[{completion_index}] status={status} n_tok={n_tok} "
            f"hash={h} ttft_ms={ttft_ms:.3f} total_ms={total_ms:.3f} "
            f"total_minus_ttft_ms={total_minus_ttft_ms:.3f} tps={tps:.3f}"
        )

    rows.sort(key=lambda r: r[0])

    csv_path = out_dir / "per_repeat.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "completion_index", "status", "n_tokens_generated",
            "generated_token_hash", "ttft_ms", "total_ms",
            "tokens_per_second", "total_minus_ttft_ms",
        ])
        for r in rows:
            w.writerow([
                r[0], r[1], r[2], r[3],
                f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}",
            ])

    (out_dir / "request_summary.txt").write_text(
        "\n".join(summary_lines) + ("\n" if summary_lines else "")
    )

    if BINARY.exists():
        bin_size = BINARY.stat().st_size
        bin_sha  = sha256_of(BINARY)
        (out_dir / "binary_used.txt").write_text(
            f"path: {BINARY}\n"
            f"size_bytes: {bin_size}\n"
            f"sha256: {bin_sha}\n"
        )
    else:
        (out_dir / "binary_used.txt").write_text(
            f"path: {BINARY}\n"
            "missing: true\n"
        )

    git_text = (
        f"rev: {git_run(['rev-parse', '--short', 'HEAD']).strip()}\n"
        f"log -1 --oneline:\n{git_run(['log', '-1', '--oneline'])}"
        f"status --short:\n{git_run(['status', '--short'])}"
    )
    (out_dir / "git_head.txt").write_text(git_text)

    plan_csv = ",".join(str(x) for x in entry["max_tokens_plan"])
    (out_dir / "build_info.txt").write_text(
        f"binary: {BINARY}\n"
        f"model: {MODEL}\n"
        f"prompt: {PROMPT!r}\n"
        f"global_index: {entry['global_index']}\n"
        f"layer: {entry['layer']}\n"
        f"backend: {entry['backend']}\n"
        f"trial_index: {entry['trial_index']}\n"
        f"n_contexts: {entry['n_contexts']}\n"
        f"n_concurrent: {entry['n_concurrent']}\n"
        f"n_requests: {entry['n_requests']}\n"
        f"max_tokens: {entry['max_tokens']}\n"
        f"max_tokens_plan: {plan_csv}\n"
        f"ctx_size: {entry['ctx_size']}\n"
        f"batch_size: {entry['batch_size']}\n"
        f"n_threads: {entry['n_threads']}\n"
        f"seed_base: {entry['seed_base']}\n"
        f"trace_enabled: {entry['trace_enabled']}\n"
        f"measured_for_timing: {entry['measured_for_timing']}\n"
        f"expected_os_threads: {entry['expected_os_threads']}\n"
        f"expected_pool_size: {entry['expected_pool_size']}\n"
        f"expected_trace_lines: {entry['expected_trace_lines']}\n"
    )

    (out_dir / "schedule_entry.json").write_text(
        json.dumps(entry, indent=2) + "\n"
    )

    print(f"global_index={entry['global_index']}")
    print(
        f"layer={entry['layer']}  backend={entry['backend']}  "
        f"trial_index={entry['trial_index']}"
    )
    print(f"output_dir={out_dir}")
    print(
        f"trace_enabled={entry['trace_enabled']}  "
        f"measured_for_timing={entry['measured_for_timing']}"
    )
    print(f"exit_code={exit_code}")
    print(f"process_wall_ms={process_wall_ms:.3f}")
    print(f"rows_parsed={len(rows)}  (expected {entry['n_requests']})")
    print(
        f"hpx_trace_lines={len(trace_lines)}  "
        f"(expected_when_trace_on={entry['expected_trace_lines']})"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
