# HPX build and test reference

This project uses HPX as an alternative CPU executor for ggml. Two
dedicated CMake options drive it:

| Flag                  | Scope                                                 | Purpose                                                                                          |
|-----------------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `GGML_HPX`            | top-level (declared in `ggml/CMakeLists.txt`)          | Builds the `ggml-hpx` backend library. Pulls in `hpx-bench/`; pulls in `tests/hpx/` when `LLAMA_BUILD_TESTS=ON`. |
| `GGML_HPX_REGION_DAG` | inner (declared in `ggml/src/ggml-hpx/CMakeLists.txt`) | Adds the experimental fine-region DAG + frozen-packet sources, tests, and benchmarks. Requires `GGML_HPX=ON`. |

Three build trees are used. The naming mirrors the flag state so it is
obvious which tree is which at a glance:

| Directory        | `GGML_HPX` | `GGML_HPX_REGION_DAG` | Purpose                                                           |
|------------------|------------|-----------------------|-------------------------------------------------------------------|
| `build-base`     | OFF        | OFF                   | Upstream build; verifies non-HPX paths still compile and test.    |
| `build-hpx`      | ON         | OFF                   | Stable HPX backend only. Production-ish surface, no experimental. |
| `build-hpx-dag`  | ON         | ON                    | Stable HPX + experimental fine-region DAG + frozen-packet.        |

---

## Prerequisites

- HPX installed at `/Users/unick/hpx-install` (HPX 1.10.0)
- GTest installed via `brew install googletest`
- CMake ≥ 3.21

Every HPX configure below sets `HPX_DIR` explicitly so CMake finds the
local install without relying on system search paths.

---

## build-base — upstream, no HPX

Used to check that HPX-gated changes have not broken anything in the
non-HPX code paths.

### Configure

```bash
cmake -S . -B build-base \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Build

```bash
cmake --build build-base -j$(sysctl -n hw.logicalcpu)
```

### Run upstream tests

```bash
ctest --test-dir build-base --output-on-failure
```

---

## build-hpx — stable HPX backend only

`GGML_HPX=ON`, region-DAG experimental path OFF. This tree builds the
production HPX backend library and the non-experimental HPX tests and
benchmarks.

### Configure

```bash
cmake -S . -B build-hpx \
  -DGGML_HPX=ON \
  -DHPX_DIR=/Users/unick/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGTest_DIR=/opt/homebrew/lib/cmake/GTest \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Build the whole tree

```bash
cmake --build build-hpx -j$(sysctl -n hw.logicalcpu)
```

### What gets built

- Library: `ggml-hpx`
- Tests (`bin/test_hpx_*`):
  `test_hpx_abort`, `test_hpx_cache`, `test_hpx_decode_plan`,
  `test_hpx_prefill_plan`, `test_hpx_adapter`, `test_hpx_runtime`,
  `test_hpx_exec`, `test_hpx_llama_smoke`
- Benchmarks (`bin/bench_*`):
  `bench_run_job_overhead`

### Run the HPX tests

Individual binaries (recommended for readable per-suite output):

```bash
./build-hpx/bin/test_hpx_abort
./build-hpx/bin/test_hpx_cache
./build-hpx/bin/test_hpx_decode_plan
./build-hpx/bin/test_hpx_prefill_plan
./build-hpx/bin/test_hpx_adapter
./build-hpx/bin/test_hpx_runtime
./build-hpx/bin/test_hpx_exec
```

All HPX tests through CTest:

```bash
ctest --test-dir build-hpx -R "^test_hpx_" --output-on-failure
```

Note: `test_hpx_llama_smoke` is skipped unless `LLAMACPP_TEST_MODELFILE`
is set to a valid model path.

### Run the benchmark

```bash
./build-hpx/bin/bench_run_job_overhead
```

---

## build-hpx-dag — stable HPX + experimental fine-region DAG

`GGML_HPX=ON` **and** `GGML_HPX_REGION_DAG=ON`. This adds the
experimental fine-region executor (`ggml-hpx-region-exec.cpp`), the
frozen-packet sources (`ggml-hpx-packet.cpp`), the seven region-DAG
unit tests, and the two region-DAG benchmarks.

### Configure

```bash
cmake -S . -B build-hpx-dag \
  -DGGML_HPX=ON \
  -DGGML_HPX_REGION_DAG=ON \
  -DHPX_DIR=/Users/unick/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGTest_DIR=/opt/homebrew/lib/cmake/GTest \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Build

```bash
cmake --build build-hpx-dag -j$(sysctl -n hw.logicalcpu)
```

### What gets built (in addition to the `build-hpx` set)

- Tests (`bin/test_hpx_region_*`):
  `test_hpx_region_group_validate`, `test_hpx_region_group_run`,
  `test_hpx_region_single_mul_mat`, `test_hpx_region_group_parallel`,
  `test_hpx_region_group_reduction`, `test_hpx_region_group_lane_scratch`,
  `test_hpx_region_group_rms_norm_f32`
- Benchmarks (`bin/bench_*`):
  `bench_hpx_run_range_mul_mat`, `bench_hpx_rms_norm_f32`

### Run the region-DAG tests

```bash
./build-hpx-dag/bin/test_hpx_region_group_validate
./build-hpx-dag/bin/test_hpx_region_group_run
./build-hpx-dag/bin/test_hpx_region_single_mul_mat
./build-hpx-dag/bin/test_hpx_region_group_parallel
./build-hpx-dag/bin/test_hpx_region_group_reduction
./build-hpx-dag/bin/test_hpx_region_group_lane_scratch
./build-hpx-dag/bin/test_hpx_region_group_rms_norm_f32
```

All HPX tests (both always-on and region-DAG-gated) through CTest:

```bash
ctest --test-dir build-hpx-dag -R "^test_hpx_" --output-on-failure
```

### Run the region-DAG benchmarks

```bash
./build-hpx-dag/bin/bench_hpx_run_range_mul_mat
./build-hpx-dag/bin/bench_hpx_rms_norm_f32
```

See `CLAUDE.md` for where benchmark outputs live (`hpx-bench/results/<date>-<slug>/`).

---

## compile_commands.json (IDE / clangd)

The symlink at the project root points to whichever build tree is
active for IDE navigation. Switch it when changing trees:

```bash
# DAG work (default during active region-DAG / packet development):
ln -sf build-hpx-dag/compile_commands.json compile_commands.json

# Stable HPX backend only:
ln -sf build-hpx/compile_commands.json compile_commands.json

# Upstream-only:
ln -sf build-base/compile_commands.json compile_commands.json
```

---

## Which tree do I want?

- Fixing something in the **stable HPX backend** (adapter / cache /
  plan / exec / runtime) and don't want experimental sources compiled
  in → `build-hpx`.
- Working on the **fine-region DAG or frozen packets** (tests under
  `tests/hpx/test_hpx_region_*`, benches under
  `hpx-bench/bench_hpx_rms_norm_f32.cpp`, etc.) → `build-hpx-dag`.
- Confirming a change does not regress anything **outside the HPX
  subtree** → `build-base`.
