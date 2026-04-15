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

# bench_hpx_run_range_mul_mat
# ---------------------------
# Compares spawn / persistent-condvar-pool / fork_join_executor dispatch on
# ggml_vec_dot_f32, plus a 3-region chain exercising ggml_hpx_region_resources.
# Gated on GGML_HPX_REGION_DAG (declared in ggml/src/ggml-hpx/CMakeLists.txt,
# processed before this file).  The region-dag header is found via the PUBLIC
# include directory exported by ggml-hpx; no ad hoc path needed here.
if(GGML_HPX_REGION_DAG)
    add_executable(bench_hpx_run_range_mul_mat
        bench_hpx_run_range_mul_mat.cpp
    )
    target_compile_definitions(bench_hpx_run_range_mul_mat
        PRIVATE
            GGML_HPX_REGION_DAG
    )
    target_link_libraries(bench_hpx_run_range_mul_mat
        PRIVATE
            ggml-hpx
            ggml-cpu
            ggml
            HPX::hpx
    )
    target_compile_features(bench_hpx_run_range_mul_mat PRIVATE cxx_std_17)
    message(STATUS "hpx-bench: bench_hpx_run_range_mul_mat target added")
endif()
