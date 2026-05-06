"""
HPX-on llama-serving-bench timing helper for the (2, 4, 12) waiter-pressure
slice.

Mirrors local/baselines/comparison_hpx_concurrency/_run_hpx_concurrency.py
with these changes:
  - ROOT points at the waiter-pressure slice's hpx/ subdirectory.
  - ARGV uses --n-contexts 2 --n-concurrent 4 --n-requests 12.
  - per_repeat.csv has 12 data rows; every row carries kind=request
    (no warmup phase — all 12 requests are correctness-gated).
  - At n_concurrent > n_contexts, two requests at saturation are queued on
    the hpx_context_pool::waiters_ deque while two are in-flight. The trace
    does NOT distinguish queued vs. fast-path requests (no queued/wake
    events emitted today); the runner just captures the events that exist.

Captures (under hpx/):
  bench.stdout
  bench.stderr
  bench.exit_code.txt
  process_wall_ms.txt          (informational, not a gate)
  per_repeat.csv               (header + 12 data rows, with total_minus_ttft_ms column)
  request_summary.txt
  binary_used.txt
  build_info.txt
  git_head.txt
  hpx_trace.txt                (filtered HPX lifecycle + pool lines)

Stdout per-request line shape (from main.cpp):
  [serving-bench] req[N] status=ok n_tokens_generated=16 \\
    generated_token_hash=0x... ttft_ms=... total_ms=...

HPX trace lines (only emitted when LLAMA_SERVING_BENCH_HPX_TRACE=1 and
backend=hpx):
  [serving-bench] hpx_runtime_start_once: starting (os_threads=2)
  [serving-bench] engine_hpx ready: n_contexts=2 pool_size=2
  [serving-bench] hpx_runtime_stop: stopping
  [serving-bench] req[i] acquire ctx=N
  [serving-bench] req[i] release ctx=N
"""
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_hpx_waiters/hpx")
BINARY = Path("/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench")
MODEL  = "/Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
REPO   = Path("/Users/Ashk/Desktop/HPX/llama-hpx")
PROMPT = "Hello, my name is"

ARGV = [
    str(BINARY),
    "--backend",      "hpx",
    "--model",        MODEL,
    "--prompt",       PROMPT,
    "--n-contexts",   "2",
    "--n-concurrent", "4",
    "--n-requests",   "12",
    "--max-tokens",   "16",
    "--ctx-size",     "2048",
    "--batch-size",   "512",
    "--n-threads",    "4",
    "--seed-base",    "1234",
]

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

ROOT.mkdir(parents=True, exist_ok=True)


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


# ---- run the harness with HPX trace enabled ----
env = os.environ.copy()
env["LLAMA_SERVING_BENCH_HPX_TRACE"] = "1"

t0 = time.monotonic()
proc = subprocess.run(ARGV, capture_output=True, env=env)
t1 = time.monotonic()
process_wall_ms = (t1 - t0) * 1000.0

stdout = proc.stdout.decode("utf-8", errors="replace")
stderr = proc.stderr.decode("utf-8", errors="replace")
exit_code = proc.returncode

(ROOT / "bench.stdout").write_text(stdout)
(ROOT / "bench.stderr").write_text(stderr)
(ROOT / "bench.exit_code.txt").write_text(f"{exit_code}\n")
(ROOT / "process_wall_ms.txt").write_text(
    f"{process_wall_ms:.3f}\n"
    "note: outer time.monotonic() around subprocess.run; informational only.\n"
    "scope: HPX-on (2, 4, 12) waiter-pressure slice. Not a gate.\n"
    "process wall here includes HPX runtime startup/shutdown, model load, and\n"
    "harness teardown. With n_concurrent > n_contexts, total wall also folds\n"
    "in waiter-queue residency time, which is not separately observable.\n"
)

# ---- extract HPX lifecycle / pool trace lines ----
trace_lines = [line for line in stderr.splitlines() if TRACE_RE.match(line)]
(ROOT / "hpx_trace.txt").write_text(
    ("\n".join(trace_lines) + "\n") if trace_lines else ""
)

# ---- parse per-request stdout lines ----
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
    rows.append((idx, kind, status, n_tok, h, ttft_ms, total_ms, tps, total_minus_ttft_ms))
    summary_lines.append(
        f"req[{idx}] kind={kind:<8} status={status} n_tok={n_tok} "
        f"hash={h} ttft_ms={ttft_ms:.3f} total_ms={total_ms:.3f} "
        f"total_minus_ttft_ms={total_minus_ttft_ms:.3f} tps={tps:.3f}"
    )

rows.sort(key=lambda r: r[0])

# ---- per_repeat.csv ----
csv_path = ROOT / "per_repeat.csv"
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

# ---- request_summary.txt ----
(ROOT / "request_summary.txt").write_text(
    "\n".join(summary_lines) + ("\n" if summary_lines else "")
)

# ---- binary_used.txt ----
bin_size = BINARY.stat().st_size
bin_sha  = sha256_of(BINARY)
(ROOT / "binary_used.txt").write_text(
    f"path: {BINARY}\n"
    f"size_bytes: {bin_size}\n"
    f"sha256: {bin_sha}\n"
)

# ---- git_head.txt ----
git_text = (
    f"rev: {git_run(['rev-parse', '--short', 'HEAD']).strip()}\n"
    f"log -1 --oneline:\n{git_run(['log', '-1', '--oneline'])}"
    f"status --short:\n{git_run(['status', '--short'])}"
)
(ROOT / "git_head.txt").write_text(git_text)

# ---- build_info.txt ----
(ROOT / "build_info.txt").write_text(
    "build_dir: /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on\n"
    "LLAMA_BUILD_SERVING_BENCH: ON\n"
    "LLAMA_SERVING_BENCH_HPX: ON\n"
    f"binary: {BINARY}\n"
    "HPX install: /Users/Ashk/Desktop/HPX/hpx-install\n"
    "thread choice: --n-threads 4 (per-context kernel threads)\n"
    "concurrency shape: --n-contexts 2 --n-concurrent 4 --n-requests 12\n"
    "trace env: LLAMA_SERVING_BENCH_HPX_TRACE=1\n"
    "expected HPX runtime carriers: os_threads = max(1, min(4, 2)) = 2\n"
    "expected pool_size: 2\n"
    "expected total filtered HPX trace lines: 27 = 1 start + 1 ready + 1 stop + 12 acquire + 12 release\n"
    "scope: capacity correctness / no-starvation under n_concurrent > n_contexts.\n"
    "FIFO waiter ordering is NOT claimed (no queued/wake trace events emitted today).\n"
)

# ---- console echo ----
print(f"exit_code={exit_code}")
print(f"process_wall_ms={process_wall_ms:.3f}")
print(f"rows_parsed={len(rows)}  (expected 12)")
print(f"hpx_trace_lines={len(trace_lines)}  (expected 27)")
for line in summary_lines:
    print(line)
print(f"per_repeat.csv: {csv_path}")
sys.exit(0)
