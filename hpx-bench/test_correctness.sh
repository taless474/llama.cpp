#!/usr/bin/env bash
# hpx-bench/test_correctness.sh
#
# Correctness test: verifies HPX threading produces byte-identical output to
# pthreads for the same model, seed, and prompt.
#
# Uses the *-noblas builds so BLAS is not bypassing the thread pool —
# every GEMM goes through ggml_mul_mat and exercises the barrier.
#
# Usage:
#   ./hpx-bench/test_correctness.sh <model.gguf> [n_tokens]
#   ./hpx-bench/test_correctness.sh               # auto-detects model in ./models/
#
# Requirements:
#   cmake --build build-pthread-noblas --target llama-cli
#   cmake --build build-hpx-noblas     --target llama-cli

set -euo pipefail
cd "$(dirname "$0")/.."      # always run from repo root

# ── Config ────────────────────────────────────────────────────────────────────

N_TOK=${2:-50}
SEED=42
STRESS_RUNS=20
PROMPT="Hello"

PTHREAD_BIN=./build-pthread-noblas/bin/llama-cli
HPX_BIN=./build-hpx-noblas/bin/llama-cli
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

NCORES=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# ── Helpers ───────────────────────────────────────────────────────────────────

pass() { printf "  PASS  %s\n" "$1"; }
fail() {
    printf "  FAIL  %s\n" "$1" >&2
    exit 1
}

check_binary() {
    local bin=$1
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found."
        echo "Build it with:"
        local bd
        bd=$(dirname "$(dirname "$bin")")
        echo "  cmake --build $bd --target llama-cli"
        exit 1
    fi
}

# Run inference, write output to a file, suppress all log/timing chatter.
# Timing lines like "[ Prompt: X t/s | Generation: X t/s ]" are stripped
# even when --log-disable doesn't fully suppress them.
# Args: binary  n_threads  output_file
run_inference() {
    local bin=$1 threads=$2 outfile=$3
    "$bin" \
        -m    "$MODEL"   \
        -p    "$PROMPT"  \
        -n    "$N_TOK"   \
        --seed "$SEED"   \
        -t    "$threads" \
        -ngl  0          \
        --single-turn    \
        --log-disable    \
        2>/dev/null \
        | grep -v '^\[.*t/s.*\]' \
        > "$outfile"
}

# ── Model detection ───────────────────────────────────────────────────────────

if [[ $# -ge 1 && -f "$1" ]]; then
    MODEL="$1"
else
    MODEL=$(find ./models -name "*.gguf" -not -name "*.part*" | sort | head -1 || true)
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: no .gguf model found."
        echo "Pass the model path as the first argument, or place a .gguf in ./models/"
        exit 1
    fi
fi

# ── Pre-flight ────────────────────────────────────────────────────────────────

check_binary "$PTHREAD_BIN"
check_binary "$HPX_BIN"

echo ""
echo "=== HPX threading correctness test ==="
printf "Model   : %s\n"    "$MODEL"
printf "Prompt  : \"%s\"\n" "$PROMPT"
printf "Tokens  : %d  Seed : %d  Cores : %d\n" "$N_TOK" "$SEED" "$NCORES"
echo ""

# ── Test 1: pthread t=T  vs  HPX t=T ─────────────────────────────────────────
# Same thread count → same work partitioning → identical floating-point results.
# BLAS is OFF so every matmul goes through ggml_mul_mat and uses the barrier.

echo "── Test 1: HPX output == pthread (same thread count) ──"
for T in 1 2 4 "$NCORES"; do
    REF="$TMPDIR_LOCAL/ref_t${T}.txt"
    GOT="$TMPDIR_LOCAL/hpx_t${T}.txt"
    run_inference "$PTHREAD_BIN" "$T" "$REF"
    run_inference "$HPX_BIN"    "$T" "$GOT"
    if diff -q "$REF" "$GOT" > /dev/null; then
        pass "t=$T  HPX == pthread"
    else
        echo "  FAIL  t=$T  output differs:"
        diff "$REF" "$GOT" | head -20
        fail "t=$T mismatch"
    fi
done

# ── Test 2: barrier determinism ───────────────────────────────────────────────
# A broken hpx::barrier would cause different outputs across runs with the same
# seed.  Test the four counts most likely to expose a race:
#   t=1  → barrier bypass (early return, no arrive_and_wait)
#   t=2  → minimal two-thread sync
#   t=4  → typical workload
#   t=10 → above typical core count, stresses HPX task scheduling

echo ""
echo "── Test 2: Barrier determinism (same seed → same output every run) ──"
for T in 1 2 4 10; do
    A="$TMPDIR_LOCAL/det_a_t${T}.txt"
    B="$TMPDIR_LOCAL/det_b_t${T}.txt"
    run_inference "$HPX_BIN" "$T" "$A"
    run_inference "$HPX_BIN" "$T" "$B"
    if diff -q "$A" "$B" > /dev/null; then
        pass "t=$T  deterministic across two runs"
    else
        echo "  FAIL  t=$T  non-deterministic output (barrier race?):"
        diff "$A" "$B" | head -20
        fail "t=$T non-deterministic"
    fi
done

# ── Test 3: stress ────────────────────────────────────────────────────────────
# Race conditions that only surface intermittently need many iterations.
# Run N inferences sequentially on HPX t=4 and compare each to the reference.

echo ""
echo "── Test 3: Stress ($STRESS_RUNS sequential runs, t=4) ──"
REF4="$TMPDIR_LOCAL/ref_t4.txt"
run_inference "$PTHREAD_BIN" 4 "$REF4"

fail_count=0
for i in $(seq 1 "$STRESS_RUNS"); do
    GOT="$TMPDIR_LOCAL/stress_${i}.txt"
    run_inference "$HPX_BIN" 4 "$GOT"
    if ! diff -q "$REF4" "$GOT" > /dev/null; then
        echo "  FAIL  stress run $i differs:"
        diff "$REF4" "$GOT" | head -10
        fail_count=$((fail_count + 1))
    fi
done

if [[ $fail_count -eq 0 ]]; then
    pass "$STRESS_RUNS sequential HPX runs all match pthread reference"
else
    fail "$fail_count / $STRESS_RUNS stress runs differed from reference"
fi

# ── Test 4: thread count edge cases ──────────────────────────────────────────

echo ""
echo "── Test 4: Edge cases ──"

# n_threads=1: barrier is bypassed (early return in ggml_barrier)
REF1="$TMPDIR_LOCAL/ref_t1.txt"
GOT1="$TMPDIR_LOCAL/hpx_edge_t1.txt"
run_inference "$PTHREAD_BIN" 1  "$REF1"
run_inference "$HPX_BIN"    1  "$GOT1"
diff -q "$REF1" "$GOT1" > /dev/null \
    && pass "t=1  (barrier bypass path)  HPX == pthread" \
    || { diff "$REF1" "$GOT1" | head -10; fail "t=1 mismatch"; }

# n_threads=n_cores: maximum OS-level contention
REFM="$TMPDIR_LOCAL/ref_tmax.txt"
GOTM="$TMPDIR_LOCAL/hpx_edge_tmax.txt"
run_inference "$PTHREAD_BIN" "$NCORES" "$REFM"
run_inference "$HPX_BIN"    "$NCORES" "$GOTM"
diff -q "$REFM" "$GOTM" > /dev/null \
    && pass "t=$NCORES  (max contention)  HPX == pthread" \
    || { diff "$REFM" "$GOTM" | head -10; fail "t=$NCORES mismatch"; }

echo ""
echo "All correctness tests passed."
echo ""
