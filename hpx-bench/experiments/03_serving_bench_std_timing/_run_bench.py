"""
HPX-OFF llama-serving-bench std timing smoke runner.

Invokes the harness exactly once with the agreed argv (1 warmup + 5 measured,
single context, single concurrent, std backend, deterministic argmax).

Captures:
  - bench.stdout
  - bench.stderr
  - bench.exit_code.txt

Parses per-request stdout lines into:
  - per_repeat.csv     (header + 6 data rows)
  - request_summary.txt

Writes provenance:
  - binary_used.txt    (absolute path, size, sha256)
  - git_head.txt       (rev-parse, log -1 --oneline, status --short)
  - build_info.txt     (build dir, serving-bench flags, binary path, note)

Stdout per-request line shape (from main.cpp:189-194):
  [serving-bench] req[N] status=ok n_tokens_generated=16 \\
    generated_token_hash=0x... ttft_ms=... total_ms=...

Acceptance gates are NOT enforced here. _summarize_timing.py owns gates.
"""
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT   = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/serving_bench_std_timing")
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
    "--n-threads",    "0",
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
proc = subprocess.run(ARGV, capture_output=True)
stdout = proc.stdout.decode("utf-8", errors="replace")
stderr = proc.stderr.decode("utf-8", errors="replace")
exit_code = proc.returncode

(ROOT / "bench.stdout").write_text(stdout)
(ROOT / "bench.stderr").write_text(stderr)
(ROOT / "bench.exit_code.txt").write_text(f"{exit_code}\n")

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
    kind     = "warmup" if idx == 0 else "measured"
    rows.append((idx, kind, status, n_tok, h, ttft_ms, total_ms, tps))
    summary_lines.append(
        f"req[{idx}] kind={kind:<8} status={status} n_tok={n_tok} "
        f"hash={h} ttft_ms={ttft_ms:.3f} total_ms={total_ms:.3f} tps={tps:.3f}"
    )

rows.sort(key=lambda r: r[0])

# ---- per_repeat.csv ----
csv_path = ROOT / "per_repeat.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "req_index", "kind", "status", "n_tokens_generated",
        "generated_token_hash", "ttft_ms", "total_ms", "tokens_per_second",
    ])
    for r in rows:
        w.writerow([
            r[0], r[1], r[2], r[3], r[4],
            f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}",
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
)

# ---- console echo ----
print(f"exit_code={exit_code}")
print(f"rows_parsed={len(rows)}")
for line in summary_lines:
    print(line)
print(f"per_repeat.csv: {csv_path}")
sys.exit(0)
