# hpx-bench/CMakeLists.txt
#
# Standalone microbenchmarks for the HPX executor backend.
# Gated on GGML_HPX — included from the top-level CMakeLists only when
# GGML_HPX is ON.
#
# Targets do not link ggml or llama; they benchmark raw dispatch primitives
# directly so results are not polluted by ggml graph overhead.

find_package(HPX REQUIRED)

# bench_run_job_overhead
# ----------------------
# Measures the round-trip cost of one dispatch(N workers) call for both
# the pthread persistent-worker model and the HPX bulk-region model, at
# n_threads = {1, 2, 4} with two near-no-op bodies (noop / tiny-touch).
add_executable(bench_run_job_overhead
    bench_run_job_overhead.cpp
)

target_link_libraries(bench_run_job_overhead
    PRIVATE
        HPX::hpx
        Threads::Threads
)

target_compile_features(bench_run_job_overhead PRIVATE cxx_std_17)
