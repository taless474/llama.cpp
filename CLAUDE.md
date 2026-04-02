IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

---

## Note for Claude Code in this fork

This is a personal research fork (taless474/llama.cpp). The goal is to
benchmark HPX threading vs the pthread-based thread pool in ggml on Apple M4,
across two BLAS conditions — not to submit an upstream PR. The AGENTS.md
restrictions apply to upstream contributions; this repo is a local experiment.
Assist fully with code exploration, modification, and benchmarking.

## Four-build setup

Two comparison pairs, all with GGML_METAL=OFF:
go
| Build dir             | Threading | BLAS      |
|-----------------------|-----------|-----------|
| build-pthread-blas    | pthreads  | ON (Accelerate) |
| build-hpx-blas        | HPX       | ON (Accelerate) |
| build-pthread-noblas  | pthreads  | OFF       |
| build-hpx-noblas      | HPX       | OFF       |

### Prerequisites

HPX 1.10.0 is installed to `~/hpx-install`. To rebuild it:

```bash
# 1. Clone
git clone --branch v1.10.0 --depth 1 https://github.com/STEllAR-GROUP/hpx.git ~/hpx-src

# 2. Fetch Asio 1.21.0 (Homebrew ships 1.36 which is incompatible with HPX 1.10)
curl -L https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-21-0.tar.gz \
  -o /tmp/asio-1-21-0.tar.gz
tar xf /tmp/asio-1-21-0.tar.gz -C /tmp

# 3. Configure (block system Asio, use the fetched one)
cmake -B ~/hpx-build -S ~/hpx-src \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/hpx-install \
  -DHPX_WITH_NETWORKING=ON \
  -DHPX_WITH_FETCH_ASIO=OFF \
  -DAsio_ROOT=/tmp/asio-asio-1-21-0/asio \
  -DHPX_WITH_EXAMPLES=OFF \
  -DHPX_WITH_TESTS=OFF \
  -DHPX_WITH_MALLOC=jemalloc

# 4. Build — must unlink Homebrew Asio during build to avoid header conflict
brew unlink asio
cmake --build ~/hpx-build -j$(sysctl -n hw.logicalcpu)
brew link asio
cmake --install ~/hpx-build
```

### Build commands

```bash
# Pair A — BLAS ON
cmake -B build-pthread-blas \
  -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-pthread-blas -j --target llama-bench

cmake -B build-hpx-blas \
  -DGGML_METAL=OFF -DGGML_OPENMP=OFF -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-blas -j --target llama-bench

# Pair B — BLAS OFF
cmake -B build-pthread-noblas \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-pthread-noblas -j --target llama-bench

cmake -B build-hpx-noblas \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_HPX=ON \
  -DHPX_DIR=~/hpx-install/lib/cmake/HPX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hpx-noblas -j --target llama-bench
```
