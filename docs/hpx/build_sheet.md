# llama-hpx Build Cheatsheet

```bash
REPO=/Users/unick/Desktop/hpx/llama-hpx
MODEL=/Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## 1. Plain llama.cpp base

Use this when testing **llama.cpp itself**, not the serving benchmark.

### Configure

```bash
cmake -S $REPO \
      -B /Users/unick/Desktop/hpx/builds/llama-base \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVING_BENCH=OFF \
      -DLLAMA_SERVING_BENCH_HPX=OFF
```

### Build `llama-cli`

```bash
cmake --build /Users/unick/Desktop/hpx/builds/llama-base \
  --target llama-cli -j
```

### Run a simple prompt

```bash
/Users/unick/Desktop/hpx/builds/llama-base/bin/llama-cli \
  -m $MODEL \
  -p "Write one short sentence about the moon." \
  -n 32 \
  -ngl 0
```

### Quieter run into a log

```bash
mkdir -p local

/Users/unick/Desktop/hpx/builds/llama-base/bin/llama-cli \
  -m $MODEL \
  -p "Write one short sentence about the moon." \
  -n 32 \
  -ngl 0 \
  > local/llama_base_run.log 2>&1

tail -80 local/llama_base_run.log
```

---

## 2. std serving-bench build

Use this when testing the `llama-serving-bench` std backend. No HPX.

### Configure

```bash
cmake -S $REPO \
      -B /Users/unick/Desktop/hpx/builds/llama-hpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVING_BENCH=ON \
      -DLLAMA_SERVING_BENCH_HPX=OFF
```

### Build

```bash
cmake --build /Users/unick/Desktop/hpx/builds/llama-hpx \
  --target llama-serving-bench -j
```

### Help / sanity

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench --help
```

### std smoke

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench \
  -m $MODEL \
  -p "Hello, my name is" \
  --n-contexts 1 \
  --n-concurrent 1 \
  --n-requests 1 \
  --max-tokens 16 \
  --backend std
```

Expected key lines:

```text
n_ok=1
n_error=0
n_tokens_generated=16
generated_token_hash=0x833045f1e2ebf49f
```

### std empty-generation check

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench \
  -m $MODEL \
  -p "Hello, my name is" \
  --n-contexts 1 \
  --n-concurrent 1 \
  --n-requests 1 \
  --max-tokens 0 \
  --backend std 2>&1 | grep "serving-bench"
```

Expected hash:

```text
generated_token_hash=0x0000000000000000
```

### HPX-OFF stub check

```bash
/Users/unick/Desktop/hpx/builds/llama-hpx/bin/llama-serving-bench \
  -m $MODEL \
  -p "Hello, my name is" \
  --backend hpx
echo "exit=$?"
```

Expected:

```text
[serving-bench] HPX backend not built (LLAMA_SERVING_BENCH_HPX=OFF)
exit=1
```

---

## 3. HPX-ON serving-bench build

Use this when testing the real HPX-enabled build. Keep it in a separate build directory.

### Configure

```bash
cmake -S $REPO \
      -B /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVING_BENCH=ON \
      -DLLAMA_SERVING_BENCH_HPX=ON \
      -DHPX_DIR=/Users/unick/Desktop/hpx/hpx-install/lib/cmake/HPX
```

### Build

```bash
cmake --build /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on \
  --target llama-serving-bench -j
```

### Check HPX linkage

```bash
otool -L /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-serving-bench | grep hpx
```

Expected: HPX dylibs should come from the configured HPX install path via rpath.

### std regression inside HPX-ON build

This checks that enabling HPX support did not perturb the std backend.

```bash
LLAMA_SERVING_BENCH_HPX_TRACE=1 \
/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-serving-bench \
  -m $MODEL \
  -p "Hello, my name is" \
  --n-contexts 1 \
  --n-concurrent 1 \
  --n-requests 1 \
  --max-tokens 16 \
  --backend std
```

Expected:

```text
n_ok=1
n_error=0
generated_token_hash=0x833045f1e2ebf49f
```

Also expected: no HPX lifecycle trace should appear for `--backend std`.

### Later, when real HPX backend exists

```bash
LLAMA_SERVING_BENCH_HPX_TRACE=1 \
/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-serving-bench \
  -m $MODEL \
  -p "Hello, my name is" \
  --n-contexts 1 \
  --n-concurrent 1 \
  --n-requests 1 \
  --max-tokens 16 \
  --backend hpx
```

Expected after HPX backend implementation:

```text
n_ok=1
n_error=0
n_tokens_generated=16
generated_token_hash=0x833045f1e2ebf49f
```

---

## Quick output filter

For `llama-serving-bench`, the useful lines usually start with `[serving-bench]`:

```bash
...your command... 2>&1 | grep "serving-bench"
```

For plain `llama-cli`, save output and inspect the end:

```bash
...your llama-cli command... > local/llama_cli.log 2>&1
tail -80 local/llama_cli.log
```
