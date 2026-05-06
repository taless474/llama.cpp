"""
Phase 1 helper — run exactly one entry of the C_2x4 n_threads sweep
schedule. No driving. No gate enforcement. Summarizers will gate later.

Selection (must pass exactly one of):
  --global-index N
  --n-threads <1|2|4> --layer <layer> --backend <std|hpx> --trial-index N

Reads:
  local/baselines/perf_thread_sweep/_schedule.json

Writes (under <ROOT>/<entry.output_dir>/):
  bench.stdout
  bench.stderr
  bench.exit_code.txt
  process_wall_ms.txt
  per_repeat.csv               (header + n_requests data rows)
  request_summary.txt
  hpx_trace.txt                (filtered HPX lifecycle/pool lines; empty when trace disabled)
  binary_used.txt
  git_head.txt
  build_info.txt
  schedule_entry.json

Trace env:
  LLAMA_SERVING_BENCH_HPX_TRACE=1 only when entry.trace_enabled is true;
  otherwise the variable is removed from the child env.

This helper does NOT enforce correctness or lifecycle gates. Per-trial
artifacts are captured raw; a separate summarizer applies the README's
§5 gates and records PASS/FAIL.
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

ROOT   = Path(
    "/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_thread_sweep"
)
BINARY = Path("/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench")
MODEL  = "/Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
REPO   = Path("/Users/Ashk/Desktop/HPX/llama-hpx")
PROMPT = "Hello, my name is"

SCHEDULE_PATH = ROOT / "_schedule.json"

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
        description="run one C_2x4 n_threads-sweep schedule entry"
    )
    p.add_argument("--global-index", type=int, default=None)
    p.add_argument("--n-threads", type=int, default=None, choices=[1, 2, 4])
    p.add_argument("--layer", type=str, default=None,
                   choices=["correctness_trace_on", "timing_trace_off"])
    p.add_argument("--backend", type=str, default=None, choices=["std", "hpx"])
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
            v is not None
            for v in (args.n_threads, args.layer, args.backend, args.trial_index)
        )
        if explicit_any:
            sys.exit(
                "pass either --global-index OR the explicit "
                "--n-threads/--layer/--backend/--trial-index quad, not both."
            )
        for e in entries:
            if e["global_index"] == args.global_index:
                return e
        sys.exit(f"no entry with global_index={args.global_index}")

    needed = ("n_threads", "layer", "backend", "trial_index")
    vals = (args.n_threads, args.layer, args.backend, args.trial_index)
    if any(v is None for v in vals):
        sys.exit(
            "must pass either --global-index, or all of "
            "--n-threads, --layer, --backend, --trial-index."
        )
    for e in entries:
        if (e["n_threads"], e["layer"], e["backend"], e["trial_index"]) == vals:
            return e
    sys.exit(
        "no schedule entry matches "
        f"{dict(zip(needed, vals))}"
    )


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
    return [
        str(BINARY),
        "--backend",      entry["backend"],
        "--model",        MODEL,
        "--prompt",       PROMPT,
        "--n-contexts",   str(entry["n_contexts"]),
        "--n-concurrent", str(entry["n_concurrent"]),
        "--n-requests",   str(entry["n_requests"]),
        "--max-tokens",   str(entry["max_tokens"]),
        "--ctx-size",     str(entry["ctx_size"]),
        "--batch-size",   str(entry["batch_size"]),
        "--n-threads",    str(entry["n_threads"]),
        "--seed-base",    str(entry["seed_base"]),
    ]


def main():
    args = parse_args()
    schedule = load_schedule()
    entry = select_entry(schedule, args)

    out_dir = ROOT / entry["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- env ----
    env = os.environ.copy()
    if entry["trace_enabled"]:
        env["LLAMA_SERVING_BENCH_HPX_TRACE"] = "1"
    else:
        env.pop("LLAMA_SERVING_BENCH_HPX_TRACE", None)

    argv = build_argv(entry)

    # ---- run ----
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, env=env)
    t1 = time.monotonic()
    process_wall_ms = (t1 - t0) * 1000.0

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    exit_code = proc.returncode

    (out_dir / "bench.stdout").write_text(stdout)
    (out_dir / "bench.stderr").write_text(stderr)
    (out_dir / "bench.exit_code.txt").write_text(f"{exit_code}\n")
    (out_dir / "process_wall_ms.txt").write_text(
        f"{process_wall_ms:.3f}\n"
        "note: outer time.monotonic() around subprocess.run; informational only.\n"
        f"cell_id: {entry['cell_id']}\n"
        f"n_threads: {entry['n_threads']}\n"
        f"layer: {entry['layer']}\n"
        f"backend: {entry['backend']}\n"
        f"trial_index: {entry['trial_index']}\n"
        f"measured_for_timing: {entry['measured_for_timing']}\n"
        "process wall includes runtime startup/shutdown, model load, harness teardown.\n"
    )

    # ---- hpx_trace.txt (filtered lifecycle + pool lines) ----
    trace_lines = [line for line in stderr.splitlines() if TRACE_RE.match(line)]
    (out_dir / "hpx_trace.txt").write_text(
        ("\n".join(trace_lines) + "\n") if trace_lines else ""
    )

    # ---- per-request parsing ----
    rows = []
    summary_lines = []
    for line in stdout.splitlines():
        m = REQ_RE.match(line)
        if not m:
            continue
        idx      = int(m.group(1))
        status   = m.group(2)
        n_tok    = int(m.group(3))
        h        = m.group(4)
        ttft_ms  = float(m.group(5))
        total_ms = float(m.group(6))
        tps      = n_tok / (total_ms / 1000.0) if total_ms > 0 else 0.0
        total_minus_ttft_ms = total_ms - ttft_ms
        kind     = "request"
        rows.append(
            (idx, kind, status, n_tok, h, ttft_ms, total_ms, tps, total_minus_ttft_ms)
        )
        summary_lines.append(
            f"req[{idx}] kind={kind:<8} status={status} n_tok={n_tok} "
            f"hash={h} ttft_ms={ttft_ms:.3f} total_ms={total_ms:.3f} "
            f"total_minus_ttft_ms={total_minus_ttft_ms:.3f} tps={tps:.3f}"
        )

    rows.sort(key=lambda r: r[0])

    csv_path = out_dir / "per_repeat.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "req_index", "kind", "status", "n_tokens_generated",
            "generated_token_hash", "ttft_ms", "total_ms", "tokens_per_second",
            "total_minus_ttft_ms",
        ])
        for r in rows:
            w.writerow([
                r[0], r[1], r[2], r[3], r[4],
                f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}", f"{r[8]:.6f}",
            ])

    (out_dir / "request_summary.txt").write_text(
        "\n".join(summary_lines) + ("\n" if summary_lines else "")
    )

    # ---- binary_used.txt ----
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

    # ---- git_head.txt ----
    git_text = (
        f"rev: {git_run(['rev-parse', '--short', 'HEAD']).strip()}\n"
        f"log -1 --oneline:\n{git_run(['log', '-1', '--oneline'])}"
        f"status --short:\n{git_run(['status', '--short'])}"
    )
    (out_dir / "git_head.txt").write_text(git_text)

    # ---- build_info.txt ----
    (out_dir / "build_info.txt").write_text(
        "build_dir: /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on\n"
        "LLAMA_BUILD_SERVING_BENCH: ON\n"
        "LLAMA_SERVING_BENCH_HPX: ON\n"
        f"binary: {BINARY}\n"
        "HPX install: /Users/Ashk/Desktop/HPX/hpx-install\n"
        f"global_index: {entry['global_index']}\n"
        f"cell_id: {entry['cell_id']}\n"
        f"n_threads: {entry['n_threads']}\n"
        f"layer: {entry['layer']}\n"
        f"backend: {entry['backend']}\n"
        f"trial_index: {entry['trial_index']}\n"
        f"n_contexts: {entry['n_contexts']}\n"
        f"n_concurrent: {entry['n_concurrent']}\n"
        f"n_requests: {entry['n_requests']}\n"
        f"max_tokens: {entry['max_tokens']}\n"
        f"ctx_size: {entry['ctx_size']}\n"
        f"batch_size: {entry['batch_size']}\n"
        f"seed_base: {entry['seed_base']}\n"
        f"trace_enabled: {entry['trace_enabled']}\n"
        f"measured_for_timing: {entry['measured_for_timing']}\n"
        f"expected_os_threads: {entry['expected_os_threads']}\n"
        f"expected_pool_size: {entry['expected_pool_size']}\n"
        f"expected_trace_lines: {entry['expected_trace_lines']}\n"
    )

    # ---- schedule_entry.json (verbatim copy of the selected entry) ----
    (out_dir / "schedule_entry.json").write_text(
        json.dumps(entry, indent=2) + "\n"
    )

    # ---- console echo ----
    print(f"global_index={entry['global_index']}")
    print(
        f"cell_id={entry['cell_id']}  n_threads={entry['n_threads']}  "
        f"layer={entry['layer']}  backend={entry['backend']}  "
        f"trial_index={entry['trial_index']}"
    )
    print(f"output_dir={out_dir}")
    print(f"trace_enabled={entry['trace_enabled']}  "
          f"measured_for_timing={entry['measured_for_timing']}")
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
