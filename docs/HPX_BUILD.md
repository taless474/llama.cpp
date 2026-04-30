# HPX build / test / benchmark cheatsheet

Use this when you need to rebuild, run HPX tests, run `llama-simple`, or run
HPX benchmarks without searching old notes.

---

## 0. Important CMake flags

| Flag | Scope | Purpose |
|---|---|---|
| `GGML_HPX` | top-level, declared in `ggml/CMakeLists.txt` | Builds the `ggml-hpx` backend library. Pulls in `hpx-bench/`; pulls in `tests/hpx/` when `LLAMA_BUILD_TESTS=ON`. |
| `GGML_HPX_REGION_DAG` | inner, declared in `ggml/src/ggml-hpx/CMakeLists.txt` | Adds the experimental fine-region DAG + frozen-packet sources, tests, and benchmarks. Requires `GGML_HPX=ON`. |

Build trees:

| Directory | `GGML_HPX` | `GGML_HPX_REGION_DAG` | Use for |
|---|---:|---:|---|
| `build-base` | OFF | OFF | Clean non-HPX baseline |
| `build-hpx` | ON | OFF | Stable HPX backend only |
| `build-hpx-dag` | ON | ON | Fine-region DAG, selective executor, packets, HPX benchmarks |

Default for current HPX work:

```bash
build-hpx-dag
```

---

## 1. Local paths

Edit these if your local machine changes.

```bash
export HPX_DIR=/Users/unick/hpx-install/lib/cmake/HPX
export GTest_DIR=/opt/homebrew/lib/cmake/GTest
export JOBS=$(sysctl -n hw.logicalcpu)
export MODEL=models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## 2. Configure and build `build-hpx-dag`

Use this for current HPX work: region DAG, selective executor, packet work,
and HPX benchmarks.

```bash
cmake -S . -B build-hpx-dag \
  -DGGML_HPX=ON \
  -DGGML_HPX_REGION_DAG=ON \
  -DHPX_DIR="$HPX_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGTest_DIR="$GTest_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build-hpx-dag -j"$JOBS"
ln -sf build-hpx-dag/compile_commands.json compile_commands.json
```

---

## 3. Run HPX tests

All HPX tests:

```bash
ctest --test-dir build-hpx-dag -R "^test_hpx_" --output-on-failure
```

List HPX test binaries:

```bash
ls build-hpx-dag/bin/test_hpx_*
```

Common focused tests:

```bash
./build-hpx-dag/bin/test_hpx_region_group_validate
./build-hpx-dag/bin/test_hpx_region_group_run
./build-hpx-dag/bin/test_hpx_region_group_parallel
./build-hpx-dag/bin/test_hpx_region_group_reduction
./build-hpx-dag/bin/test_hpx_region_group_lane_scratch
./build-hpx-dag/bin/test_hpx_region_group_rms_norm_f32
./build-hpx-dag/bin/test_hpx_selective_mul_mat_q4_k_repacked
```

Smoke test with a real model:

```bash
LLAMACPP_TEST_MODELFILE="$MODEL" ./build-hpx-dag/bin/test_hpx_llama_smoke
```

---

## 4. Run `llama-simple`

Important: `llama-simple` takes the prompt as a positional argument. Do **not**
use `-p`.

CPU-only is required for selective HPX work:

```text
-ngl 0
```

Baseline-ish run from the HPX build, with HPX disabled:

```bash
LLAMA_USE_HPX=0 ./build-hpx-dag/bin/llama-simple \
  -m "$MODEL" \
  -n 32 -ngl 0 \
  "Hello, my name is"
```

HPX-enabled run:

```bash
LLAMA_USE_HPX=1 ./build-hpx-dag/bin/llama-simple \
  -m "$MODEL" \
  -n 32 -ngl 0 \
  "Hello, my name is"
```

Longer decode stability check:

```bash
for N in 16 32 64 128; do
  echo "=== n=$N ==="
  LLAMA_USE_HPX=1 ./build-hpx-dag/bin/llama-simple \
    -m "$MODEL" \
    -n "$N" -ngl 0 \
    "Hello, my name is"
done
```

---

## 5. Useful runtime env flags

Basic HPX path:

```bash
LLAMA_USE_HPX=1
```

Selective stats/debug/histogram, if available in the current branch:

```bash
LLAMA_HPX_SELECTIVE_STATS=1
LLAMA_HPX_SELECTIVE_DEBUG=1
LLAMA_HPX_SELECTIVE_HIST=1
```

Packet-related flags, if available in the current branch:

```bash
LLAMA_HPX_PACKET_COMPILE_LOG=1
LLAMA_HPX_SELECTIVE_MLP_PACKET=1
```

Only enable packet-specific flags that exist in the branch you are running.

---

## 6. HPX benchmarks

List benchmark binaries:

```bash
ls build-hpx-dag/bin/bench_*
```

Common HPX benchmarks:

```bash
./build-hpx-dag/bin/bench_run_job_overhead
./build-hpx-dag/bin/bench_hpx_run_range_mul_mat
./build-hpx-dag/bin/bench_hpx_rms_norm_f32
```

If the branch has extra HPX benchmarks, run them directly from
`build-hpx-dag/bin/bench_*`.

Run one benchmark and save output:

```bash
RUN=hpx-bench/results/$(date +%Y-%m-%d)-bench-rms-norm
mkdir -p "$RUN"

./build-hpx-dag/bin/bench_hpx_rms_norm_f32 \
  > "$RUN/bench_rms_norm.csv" \
  2> "$RUN/bench.log"

{
  echo "# bench_hpx_rms_norm_f32"
  echo
  echo "- commit: $(git rev-parse HEAD)"
  echo "- binary: ./build-hpx-dag/bin/bench_hpx_rms_norm_f32"
  echo "- command: ./build-hpx-dag/bin/bench_hpx_rms_norm_f32"
  echo
  echo "## Findings"
  echo
  echo "- TODO"
  echo
  echo "## Open questions"
  echo
  echo "- TODO"
} > "$RUN/README.md"
```

Run all available benchmarks one by one:

```bash
RUN=hpx-bench/results/$(date +%Y-%m-%d)-all-hpx-benches
mkdir -p "$RUN"

for B in build-hpx-dag/bin/bench_*; do
  NAME=$(basename "$B")
  echo "=== $NAME ==="
  "$B" > "$RUN/${NAME}.stdout.txt" 2> "$RUN/${NAME}.stderr.txt"
done

{
  echo "# All HPX benchmarks"
  echo
  echo "- commit: $(git rev-parse HEAD)"
  echo "- binary dir: build-hpx-dag/bin"
  echo "- command: for B in build-hpx-dag/bin/bench_*; do \$B; done"
  echo
  echo "## Findings"
  echo
  echo "- TODO"
  echo
  echo "## Open questions"
  echo
  echo "- TODO"
} > "$RUN/README.md"
```

---

## 7. Save a `llama-simple` run correctly

Never write results to `/tmp`.

Use:

```text
hpx-bench/results/<date>-<slug>/
local/results/
```

Example:

```bash
RUN=hpx-bench/results/$(date +%Y-%m-%d)-tinyllama-hpx-smoke
mkdir -p "$RUN"

LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_STATS=1 \
  ./build-hpx-dag/bin/llama-simple \
  -m "$MODEL" \
  -n 32 -ngl 0 \
  "Hello, my name is" \
  > "$RUN/stdout.txt" \
  2> "$RUN/stderr.txt"

{
  echo "# TinyLlama HPX smoke"
  echo
  echo "- commit: $(git rev-parse HEAD)"
  echo "- binary: ./build-hpx-dag/bin/llama-simple"
  echo "- model: $MODEL"
  echo "- command: LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_STATS=1 ./build-hpx-dag/bin/llama-simple -m \$MODEL -n 32 -ngl 0 \"Hello, my name is\""
  echo
  echo "## Findings"
  echo
  echo "- TODO"
  echo
  echo "## Open questions"
  echo
  echo "- TODO"
} > "$RUN/README.md"
```

---

## 8. Configure and build `build-base`

Use this to check the non-HPX baseline.

```bash
cmake -S . -B build-base \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build-base -j"$JOBS"
ctest --test-dir build-base --output-on-failure
```

Run baseline `llama-simple`:

```bash
./build-base/bin/llama-simple \
  -m "$MODEL" \
  -n 32 -ngl 0 \
  "Hello, my name is"
```

---

## 9. Configure and build `build-hpx`

Use this for older/stable HPX backend work without the experimental DAG/packet
sources.

```bash
cmake -S . -B build-hpx \
  -DGGML_HPX=ON \
  -DHPX_DIR="$HPX_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGTest_DIR="$GTest_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build-hpx -j"$JOBS"
ctest --test-dir build-hpx -R "^test_hpx_" --output-on-failure
```

Stable HPX benchmark:

```bash
./build-hpx/bin/bench_run_job_overhead
```

---

## 10. Switch `compile_commands.json`

```bash
# Current DAG / selective / packet work
ln -sf build-hpx-dag/compile_commands.json compile_commands.json

# Stable HPX backend only
ln -sf build-hpx/compile_commands.json compile_commands.json

# Upstream/non-HPX baseline
ln -sf build-base/compile_commands.json compile_commands.json
```

---

## 11. Common mistakes

- Do not use `-p` with `llama-simple`; the prompt is positional.
- Use `-ngl 0` for CPU-only selective HPX tests.
- Use `build-hpx-dag` for region DAG, selective executor, packets, and HPX benchmarks.
- Set `HPX_DIR` explicitly.
- Set `GTest_DIR` explicitly if CMake cannot find GTest.
- Do not save outputs to `/tmp`.
- Save stdout, stderr, command, binary path, model path, and commit hash.
