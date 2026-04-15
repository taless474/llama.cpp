#!/usr/bin/env bash
# run_tpool_matrix.sh — focused tpool regression matrix
#
# threads: 1, 2, 4
# workloads: pp32, pp512, tg128
# protocol: 5 warm reps (discarded) + 10 measured reps per cell
# variants: base, hpx (LLAMA_USE_HPX=1)
#
# Usage:
#   MODEL=/path/to/model.gguf \
#   BASE_BUILD=build-base-cpu \
#   HPX_BUILD=build-hpx-cpu \
#   bash hpx-bench/run_tpool_matrix.sh

set -euo pipefail

MODEL="${MODEL:?set MODEL=/absolute/path/to/model.gguf}"
BASE_BUILD="${BASE_BUILD:-build-base-cpu}"
HPX_BUILD="${HPX_BUILD:-build-hpx-cpu}"
RESULT_ROOT="${RESULT_ROOT:-hpx-bench/results}"

STAMP="$(date +%F-%H%M%S)"
OUTDIR="${RESULT_ROOT}/${STAMP}-tpool-matrix"
mkdir -p "${OUTDIR}/meta" "${OUTDIR}/raw"

BASE_BENCH="${BASE_BUILD}/bin/llama-bench"
HPX_BENCH="${HPX_BUILD}/bin/llama-bench"

WARM_REPS=5
MEAS_REPS=10

THREADS=(1 2 4)

# Each entry: "label  -p N  -n M"
declare -A CASE_P=( [pp32]=32  [pp512]=512 [tg128]=1   )
declare -A CASE_N=( [pp32]=0   [pp512]=0   [tg128]=128 )
CASES=(pp32 pp512 tg128)

# ---------- metadata ----------
{
  echo "date: $(date)"
  echo "model: ${MODEL}"
  echo "base_build: ${BASE_BUILD}"
  echo "hpx_build: ${HPX_BUILD}"
  echo "warm_reps: ${WARM_REPS}"
  echo "meas_reps: ${MEAS_REPS}"
  echo "threads: ${THREADS[*]}"
  echo "cases: ${CASES[*]}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "uname: $(uname -a)"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "logical_cpu: $(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  echo "physical_cpu: $(sysctl -n hw.physicalcpu 2>/dev/null || true)"
} > "${OUTDIR}/meta/run_info.txt"

# ---------- bench loop ----------
# TSV summary: variant  threads  case  t/s_mean  t/s_stddev  raw_file
printf "variant\tthreads\tcase\tt/s\traw_file\n" > "${OUTDIR}/summary.tsv"

run_cell() {
  local variant="$1"   # base | hpx
  local exe="$2"
  local threads="$3"
  local case_name="$4"
  local p="${CASE_P[$case_name]}"
  local n="${CASE_N[$case_name]}"
  local logfile="${OUTDIR}/raw/${variant}.t${threads}.${case_name}.log"

  echo "  [${variant}] t=${threads} ${case_name} (p=${p} n=${n})"

  # warm pass — discard output
  echo "    warm (${WARM_REPS} reps)..."
  if [[ "${variant}" == "hpx" ]]; then
    LLAMA_USE_HPX=1 "${exe}" -m "${MODEL}" -ngl 0 \
      -t "${threads}" -p "${p}" -n "${n}" \
      --no-warmup -r "${WARM_REPS}" -o csv \
      > /dev/null 2>&1
  else
    "${exe}" -m "${MODEL}" -ngl 0 \
      -t "${threads}" -p "${p}" -n "${n}" \
      --no-warmup -r "${WARM_REPS}" -o csv \
      > /dev/null 2>&1
  fi

  # measured pass
  echo "    measure (${MEAS_REPS} reps)..."
  if [[ "${variant}" == "hpx" ]]; then
    LLAMA_USE_HPX=1 "${exe}" -m "${MODEL}" -ngl 0 \
      -t "${threads}" -p "${p}" -n "${n}" \
      --no-warmup -r "${MEAS_REPS}" \
      > "${logfile}" 2>&1
  else
    "${exe}" -m "${MODEL}" -ngl 0 \
      -t "${threads}" -p "${p}" -n "${n}" \
      --no-warmup -r "${MEAS_REPS}" \
      > "${logfile}" 2>&1
  fi

  # extract t/s field (last data line, column 7 of md table: "  123.45 ± 6.78 ")
  local ts
  ts=$(grep -E '^\| llama' "${logfile}" | grep "${case_name}" | \
       awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $8); print $8}' | tail -1)
  [[ -z "${ts}" ]] && ts="(parse error)"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" "${threads}" "${case_name}" "${ts}" "${logfile}" \
    >> "${OUTDIR}/summary.tsv"
}

for t in "${THREADS[@]}"; do
  echo "=== threads=${t} ==="
  for c in "${CASES[@]}"; do
    run_cell "base" "${BASE_BENCH}" "${t}" "${c}"
    run_cell "hpx"  "${HPX_BENCH}"  "${t}" "${c}"
  done
done

# ---------- pretty summary ----------
echo ""
echo "=== RESULTS: ${OUTDIR} ==="
echo ""
printf "%-8s  %7s  %-8s  %s\n" "variant" "threads" "case" "t/s (mean ± stddev)"
printf "%-8s  %7s  %-8s  %s\n" "-------" "-------" "----" "-------------------"
tail -n +2 "${OUTDIR}/summary.tsv" | while IFS=$'\t' read -r variant threads case_name ts raw; do
  printf "%-8s  %7s  %-8s  %s\n" "${variant}" "${threads}" "${case_name}" "${ts}"
done

echo ""
echo "Full logs: ${OUTDIR}/raw/"
echo "Summary:   ${OUTDIR}/summary.tsv"
