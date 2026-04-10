#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?set MODEL=/absolute/path/to/model.gguf}"

BASE_BUILD="${BASE_BUILD:-build-base-cpu}"
HPX_BUILD="${HPX_BUILD:-build-hpx-cpu}"

RESULT_ROOT="${RESULT_ROOT:-hpx-bench/results}"
STAMP="$(date +%F-%H%M%S)"
OUTDIR="${RESULT_ROOT}/${STAMP}-cpu-campaign"

BASE_SIMPLE="${BASE_BUILD}/bin/llama-simple"
HPX_SIMPLE="${HPX_BUILD}/bin/llama-simple"
BASE_BENCH="${BASE_BUILD}/bin/llama-bench"
HPX_BENCH="${HPX_BUILD}/bin/llama-bench"

mkdir -p "${OUTDIR}/meta" "${OUTDIR}/correctness" "${OUTDIR}/bench"

# ---------- metadata ----------
{
  echo "date: $(date)"
  echo "pwd: $(pwd)"
  echo "model: ${MODEL}"
  echo "base_build: ${BASE_BUILD}"
  echo "hpx_build: ${HPX_BUILD}"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "uname: $(uname -a)"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "logical_cpu: $(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  echo "physical_cpu: $(sysctl -n hw.physicalcpu 2>/dev/null || true)"
  echo "mem_bytes: $(sysctl -n hw.memsize 2>/dev/null || true)"
  echo "sw_vers:"
  sw_vers 2>/dev/null || true
  echo
  echo "base_cmake_cache:"
  grep -E 'GGML_|CMAKE_BUILD_TYPE|HPX' "${BASE_BUILD}/CMakeCache.txt" 2>/dev/null || true
  echo
  echo "hpx_cmake_cache:"
  grep -E 'GGML_|CMAKE_BUILD_TYPE|HPX' "${HPX_BUILD}/CMakeCache.txt" 2>/dev/null || true
} > "${OUTDIR}/meta/run_info.txt"

# ---------- correctness ----------
PROMPT="Tell me a short story about a young girl named Lily."

echo "Running correctness check..."
"${BASE_SIMPLE}" -m "${MODEL}" -t 4 -ngl 0 -n 32 "${PROMPT}" \
  > "${OUTDIR}/correctness/base.txt" 2>&1

LLAMA_USE_HPX=1 "${HPX_SIMPLE}" -m "${MODEL}" -t 4 -ngl 0 -n 32 "${PROMPT}" \
  > "${OUTDIR}/correctness/hpx.txt" 2>&1

# Save a normalized diff that ignores most timing noise.
grep -vE '^(main:|build:|system_info:|llama_perf_|ggml_|AVX|NEON|Metal|CUDA|OpenCL|Vulkan|graphs reused|sampler seed)' \
  "${OUTDIR}/correctness/base.txt" | sed '/^[[:space:]]*$/d' \
  > "${OUTDIR}/correctness/base.norm.txt" || true

grep -vE '^(main:|build:|system_info:|llama_perf_|ggml_|AVX|NEON|Metal|CUDA|OpenCL|Vulkan|graphs reused|sampler seed)' \
  "${OUTDIR}/correctness/hpx.txt" | sed '/^[[:space:]]*$/d' \
  > "${OUTDIR}/correctness/hpx.norm.txt" || true

diff -u "${OUTDIR}/correctness/base.norm.txt" "${OUTDIR}/correctness/hpx.norm.txt" \
  > "${OUTDIR}/correctness/diff.txt" || true

# ---------- benchmark matrix ----------
THREADS=(1 2 4 8)

# name prompt_tokens gen_tokens
CASES=(
  "prefill_tiny 16 1"
  "prefill_mid 128 1"
  "prefill_long 512 1"
  "decode_light 32 32"
  "decode_heavy 32 128"
)

run_one() {
  local label="$1"
  local exe="$2"
  local threads="$3"
  local prompt_tokens="$4"
  local gen_tokens="$5"
  local logfile="$6"

  {
    echo "label=${label}"
    echo "exe=${exe}"
    echo "threads=${threads}"
    echo "prompt_tokens=${prompt_tokens}"
    echo "gen_tokens=${gen_tokens}"
    echo "command=${exe} -m ${MODEL} -ngl 0 -t ${threads} -p ${prompt_tokens} -n ${gen_tokens}"
    echo
    "${exe}" -m "${MODEL}" -ngl 0 -t "${threads}" -p "${prompt_tokens}" -n "${gen_tokens}"
  } > "${logfile}" 2>&1
}

echo -e "variant\tthreads\tcase\tprompt_tokens\tgen_tokens\tlogfile" \
  > "${OUTDIR}/bench/index.tsv"

for t in "${THREADS[@]}"; do
  for entry in "${CASES[@]}"; do
    set -- ${entry}
    case_name="$1"
    p="$2"
    n="$3"

    base_log="${OUTDIR}/bench/base.t${t}.${case_name}.log"
    hpx_log="${OUTDIR}/bench/hpx.t${t}.${case_name}.log"

    echo "  bench: base t=${t} ${case_name} p=${p} n=${n}"
    run_one "base" "${BASE_BENCH}" "${t}" "${p}" "${n}" "${base_log}"
    echo "  bench: hpx  t=${t} ${case_name} p=${p} n=${n}"
    run_one "hpx"  "${HPX_BENCH}"  "${t}" "${p}" "${n}" "${hpx_log}"

    echo -e "base\t${t}\t${case_name}\t${p}\t${n}\t${base_log}" >> "${OUTDIR}/bench/index.tsv"
    echo -e "hpx\t${t}\t${case_name}\t${p}\t${n}\t${hpx_log}"  >> "${OUTDIR}/bench/index.tsv"
  done
done

cat > "${OUTDIR}/README.txt" <<EOF
This directory contains:
- meta/run_info.txt            : machine/build metadata
- correctness/base.txt         : raw base correctness run
- correctness/hpx.txt          : raw HPX correctness run
- correctness/diff.txt         : normalized text diff
- bench/*.log                  : raw llama-bench outputs
- bench/index.tsv              : run index

Suggested manual review:
1. correctness/diff.txt should be empty or explainable
2. bench/*.log should show CPU-only behavior
3. compare throughput by case and thread count
EOF

echo "Saved results to: ${OUTDIR}"
