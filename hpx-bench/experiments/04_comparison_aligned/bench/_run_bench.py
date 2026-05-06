"""
Aligned llama-serving-bench std timing helper for the std-vs-server comparison.

Copy of local/baselines/serving_bench_std_timing/_run_bench.py with three
changes:
  - --n-threads "0" is replaced with "4" so the bench runs on the same thread
    count llama-server picks by default on this machine. This is the §4.1
    parity decision documented in local/baselines/comparison_design/README.md.
  - per_repeat.csv adds a total_minus_ttft_ms column (= total_ms - ttft_ms),
    which is the bench-side leg of the §4.2 interval pair the cross-binary
    summarizer compares against llama-server's timings.predicted_ms.
  - the subprocess invocation is wrapped in time.monotonic() and the outer
    wall is written to process_wall_ms.txt as informational evidence for
    §4.4. This number is captured but not gated on.

Captures:
  - bench.stdout
  - bench.stderr
  - bench.exit_code.txt
  - process_wall_ms.txt          NEW (informational, not a gate)

Parses per-request stdout lines into:
  - per_repeat.csv               (header + 6 data rows, with total_minus_ttft_ms)
  - request_summary.txt

Writes provenance:
  - binary_used.txt
  - git_head.txt
  - build_info.txt

Stdout per-request line shape (from main.cpp:189-194):
  [serving-bench] req[N] status=ok n_tokens_generated=16 \\
    generated_token_hash=0x... ttft_ms=... total_ms=...
"""
import csv
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_aligned/bench")
BINARY = Path("/Users/Ashk/Desktop/HPX/builds/llama-base/bin/llama-serving-bench")
MODEL  = "/Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
REPO   = Path("/Users/Ashk/Desktop/HPX/llama-hpx")
PROMPT = "Hello, my name is"

ARGV = [
    str(BINARY),
    "--backend",      "std",
    "--model",        MODEL,
    "--prompt",       PROMPT,
    "--n-contexts",   "1",
    "--n-concurrent", "1",
    "--n-requests",   "6",
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


# ---- run the harness ----
t0 = time.monotonic()
proc = subprocess.run(ARGV, capture_output=True)
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
    "scope: §4.4 of local/baselines/comparison_design/README.md.\n"
    "not a gate. not a comparable number to llama-server wall_ms without\n"
    "explicit interval translation.\n"
)

# ---- parse per-request lines ----
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
    kind     = "warmup" if idx == 0 else "measured"
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
    "build_dir: /Users/Ashk/Desktop/HPX/builds/llama-base\n"
    "LLAMA_BUILD_SERVING_BENCH: ON\n"
    "LLAMA_SERVING_BENCH_HPX: OFF\n"
    f"binary: {BINARY}\n"
    "note: std-only upstream baseline build "
    "(HPX backend present in source but compiled as stub)\n"
    "thread choice: --n-threads 4 (matches llama-server's natural default on this machine)\n"
)

# ---- console echo ----
print(f"exit_code={exit_code}")
print(f"process_wall_ms={process_wall_ms:.3f}")
print(f"rows_parsed={len(rows)}")
for line in summary_lines:
    print(line)
print(f"per_repeat.csv: {csv_path}")
sys.exit(0)
