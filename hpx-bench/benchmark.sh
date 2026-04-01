#!/usr/bin/env bash
# hpx-bench/benchmark.sh
# Baseline thread-sweep benchmark: runs llama-bench across thread counts 1–10,
# records system state, and prints a summary table.
# Run from the repo root: bash hpx-bench/benchmark.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH="models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLAMA_BENCH="build-pthread-blas/bin/llama-bench"
RESULTS_DIR="results"
THREADS=(1 2 3 4 5 6 7 8 9 10)
N_PROMPT=512
N_GEN=128
N_GPU_LAYERS=0       # CPU-only
REPS=3               # reps per config; llama-bench averages internally → one output row
# ──────────────────────────────────────────────────────────────────────────────

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
header(){ echo -e "\n${BOLD}=== $* ===${NC}"; }

# ── Preconditions ─────────────────────────────────────────────────────────────
header "Precondition checks"

# 1. Power source
POWER_SOURCE=$(pmset -g ps | head -1)
if echo "$POWER_SOURCE" | grep -q "Battery Power"; then
    warn "Running on battery. Plug in AC power for consistent results."
else
    info "Power source: AC (good)"
fi

# 2. CPU load — sample for 1 s via top, look at idle %
CPU_IDLE=$(top -l 1 -n 0 | awk '/^CPU usage/ { gsub(/%/,""); print $7 }')
CPU_USED=$(echo "100 - ${CPU_IDLE:-0}" | bc 2>/dev/null || echo "?")
if [[ "$CPU_USED" != "?" ]] && (( $(echo "$CPU_USED > 30" | bc -l) )); then
    warn "CPU load is ${CPU_USED}% — other heavy processes may skew results."
else
    info "CPU idle: ${CPU_IDLE:-?}% (load ~${CPU_USED}%)"
fi

# 3. Thermal state via powermetrics (one sample, requires sudo)
get_thermal_state() {
    if ! command -v powermetrics &>/dev/null; then
        echo "unavailable"
        return
    fi
    # powermetrics needs root; skip gracefully if not available
    if sudo -n powermetrics --samplers smc -n 1 -i 100 2>/dev/null \
            | grep -i "thermal level" | awk '{print $NF}'; then
        :
    else
        echo "unavailable (run with sudo for thermal data)"
    fi
}

info "Sampling thermal state (may be skipped without sudo)..."
THERMAL_BEFORE=$(get_thermal_state)
info "Thermal state (before): ${THERMAL_BEFORE}"
if [[ "$THERMAL_BEFORE" =~ ^[0-9]+$ ]] && (( THERMAL_BEFORE > 3 )); then
    warn "Thermal level ${THERMAL_BEFORE} is elevated. Consider waiting for the machine to cool."
fi

# ── Validate paths ─────────────────────────────────────────────────────────────
[[ -f "$LLAMA_BENCH" ]] || die "llama-bench not found at: $LLAMA_BENCH"
[[ -f "$MODEL_PATH"  ]] || die "Model not found at: $MODEL_PATH"
mkdir -p "$RESULTS_DIR"

# ── System state snapshot ──────────────────────────────────────────────────────
header "System state"

MACOS_VER=$(sw_vers -productVersion)
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model)
P_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
E_CORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "?")
CPU_FREQ_NOM=$(sysctl -n hw.cpufrequency_max 2>/dev/null \
    | awk '{printf "%.0f MHz", $1/1e6}' 2>/dev/null || echo "unavailable")
GIT_COMMIT=$(git -C "$(dirname "$LLAMA_BENCH")/../.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
BENCH_ABS=$(realpath "$LLAMA_BENCH")

info "macOS:      $MACOS_VER"
info "Chip:       $CHIP"
info "P-cores:    $P_CORES"
info "E-cores:    $E_CORES"
info "CPU freq:   $CPU_FREQ_NOM"
info "Binary:     $BENCH_ABS"
info "Git commit: $GIT_COMMIT"

# ── Run benchmarks ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_OUT="${RESULTS_DIR}/pthread_blas_${TIMESTAMP}.csv"
# Accumulate per-thread averages for the summary table
declare -a SUMMARY_PP SUMMARY_TG

header "Running benchmarks → $CSV_OUT"
info "Configs: threads=[${THREADS[*]}], -p $N_PROMPT -n $N_GEN, -ngl $N_GPU_LAYERS"
info "Reps per config: $REPS (averaged internally by llama-bench)"
echo ""

rm -f /tmp/llama_bench_*.csv

FIRST_RUN=1
for T in "${THREADS[@]}"; do
    echo -n "  threads=$T ... "

    TMP_CSV=$(mktemp /tmp/llama_bench_XXXXXX)
    TMP_CSV="${TMP_CSV}.csv"

    "$LLAMA_BENCH" \
        --model "$MODEL_PATH" \
        --threads "$T" \
        -p "$N_PROMPT" \
        -n "$N_GEN" \
        -ngl "$N_GPU_LAYERS" \
        -r "$REPS" \
        -o csv \
        > "$TMP_CSV" 2>/dev/null

    # First invocation: keep the CSV header
    if [[ $FIRST_RUN -eq 1 ]]; then
        cp "$TMP_CSV" "$CSV_OUT"
        FIRST_RUN=0
    else
        # Append data rows only (skip header line)
        tail -n +2 "$TMP_CSV" >> "$CSV_OUT"
    fi

    # llama-bench -r N outputs one row per test type with avg_ts already averaged.
    # n_prompt > 0 → pp row; n_gen > 0 → tg row.
    DATA_ROWS=$(tail -n +2 "$TMP_CSV")

    read_avg_ts() {
        # $1 = newline-separated CSV rows; read avg_ts (col 38) from first row
        echo "$1" | awk -F',' 'NR==1 { gsub(/"/, "", $38); if ($38+0>0) printf "%.2f", $38; else print "?" }'
    }

    PP_ROWS=$(echo "$DATA_ROWS" | awk -F',' '{ gsub(/"/, "", $32); gsub(/"/, "", $33); if ($32+0 > 0 && $33+0 == 0) print }')
    TG_ROWS=$(echo "$DATA_ROWS" | awk -F',' '{ gsub(/"/, "", $32); gsub(/"/, "", $33); if ($32+0 == 0 && $33+0 > 0) print }')

    PP_AVG=$(read_avg_ts "$PP_ROWS")
    TG_AVG=$(read_avg_ts "$TG_ROWS")
    SUMMARY_PP+=("$PP_AVG")
    SUMMARY_TG+=("$TG_AVG")

    echo "pp=${PP_AVG} t/s  tg=${TG_AVG} t/s"
    rm -f "$TMP_CSV"
done

# ── Post-run system state ──────────────────────────────────────────────────────
header "Post-run system state"

THERMAL_AFTER=$(get_thermal_state)
info "Thermal state (after): ${THERMAL_AFTER}"

if [[ "$THERMAL_BEFORE" =~ ^[0-9]+$ && "$THERMAL_AFTER" =~ ^[0-9]+$ ]]; then
    DELTA=$(( THERMAL_AFTER - THERMAL_BEFORE ))
    if (( DELTA >= 2 )); then
        warn "Thermal level rose by ${DELTA} during the run (${THERMAL_BEFORE} → ${THERMAL_AFTER}). Results may be affected by throttling."
    else
        info "Thermal delta: ${DELTA} (stable)"
    fi
fi

# ── Summary table ──────────────────────────────────────────────────────────────
header "Summary (avg_ts per thread count)"

printf "\n  %-10s %20s %20s\n" "threads" "prompt (t/s, pp)" "generate (t/s, tg)"
printf "  %-10s %20s %20s\n" "-------" "----------------" "------------------"
for i in "${!THREADS[@]}"; do
    printf "  %-10s %20s %20s\n" "${THREADS[$i]}" "${SUMMARY_PP[$i]}" "${SUMMARY_TG[$i]}"
done
echo ""

info "Full results saved to: $CSV_OUT"
