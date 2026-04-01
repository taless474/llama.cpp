#!/usr/bin/env bash
# run_all_benchmarks.sh — run Level A, B, C benchmarks and save all results as CSV
#
# Usage: ./run_all_benchmarks.sh [n_threads] [model_path]
#   n_threads   default 4
#   model_path  default: llama3.1-8b Q4_K_M (auto-detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

N_THREADS="${1:-4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Auto-detect model
MODEL_BIG="$REPO_DIR/models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_SMALL="/Users/unick/Desktop/hpx/triton-hpx-llm/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL="${2:-$MODEL_BIG}"

echo "============================================================"
echo "  HPX vs pthread benchmark suite"
echo "  threads=$N_THREADS  timestamp=$TIMESTAMP"
echo "  results → $RESULTS_DIR"
echo "============================================================"
echo ""

# ── Level A: dispatch overhead ────────────────────────────────────────────────
echo "── Level A: dispatch overhead ──"
for backend in pthread-noblas hpx-noblas; do
  bin="$SCRIPT_DIR/build-$backend/bench_dispatch_overhead"
  [ -x "$bin" ] || { echo "  SKIP $backend (not built)"; continue; }
  echo "  running $backend..."
  "$bin" "$N_THREADS" 2>&1
done
echo ""

# ── Level B: mul_mat at realistic sizes ───────────────────────────────────────
echo "── Level B: mul_mat ──"
for backend in pthread-noblas hpx-noblas; do
  bin="$SCRIPT_DIR/build-$backend/bench_mul_mat"
  [ -x "$bin" ] || { echo "  SKIP $backend (not built)"; continue; }
  echo "  running $backend..."
  "$bin" "$N_THREADS" 2>&1
done
echo ""

# ── Level C: llama-bench end-to-end ──────────────────────────────────────────
echo "── Level C: llama-bench ──"
[ -f "$MODEL" ] || { echo "  ERROR: model not found: $MODEL"; exit 1; }

for build in pthread-noblas hpx-noblas pthread-blas hpx-blas; do
  bin="$REPO_DIR/build-$build/bin/llama-bench"
  [ -x "$bin" ] || { echo "  SKIP $build (not built)"; continue; }

  csv="$RESULTS_DIR/llama_bench_${build}_${TIMESTAMP}.csv"
  echo "  running $build → $csv"
  # Run once: save CSV and tee markdown to stdout
  "$bin" -m "$MODEL" -t "$N_THREADS" -p 512 -n 128 -r 3 \
         --output csv 2>/dev/null > "$csv"
  "$bin" -m "$MODEL" -t "$N_THREADS" -p 512 -n 128 -r 1 \
         2>/dev/null | grep -E "pp|tg"
  echo ""
done

echo "============================================================"
echo "  Done. CSV files:"
ls -1 "$RESULTS_DIR"/*.csv 2>/dev/null | sed 's/^/    /'
echo "============================================================"
