# ggml-hpx backend CMakeLists.txt
#
# Gated by GGML_HPX in the parent CMakeLists.
# Included only when GGML_HPX is ON.
#
# Note:
#   ggml-backend.h is pulled in by ggml-hpx-exec-selective.h (its public
#   interface uses ggml_backend_t) and therefore transitively by every TU
#   that includes that header.  The three TUs that use ggml-backend.h
#   directly are:
#     ggml-hpx-adapter.cpp          — scheduler coupling and split state
#     ggml-hpx-exec.cpp             — ggml_backend_graph_compute and
#                                     ggml_backend_sched_get_tensor_backend at run time
#     ggml-hpx-exec-selective.cpp   — per-node CPU fallback via ggml_backend_graph_compute

find_package(HPX REQUIRED)

message(STATUS "HPX found: ${HPX_VERSION}")

ggml_add_backend_library(ggml-hpx
    ggml-hpx-adapter.cpp
    ggml-hpx-cache.cpp
    ggml-hpx-exec.cpp
    ggml-hpx-plan.cpp
    ggml-hpx-runtime.cpp
    ggml-hpx-tpool.cpp
    ggml-hpx-instrument.cpp
)

target_include_directories(ggml-hpx
    # PUBLIC: ggml/src/ggml-hpx/ is exposed to consumers (e.g. hpx-bench)
    # so they can include ggml-hpx-region-dag.h without an ad hoc path.
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ggml-hpx
    PRIVATE
        HPX::hpx
        # ggml-cpu: needed by the parallel-chain prototype in ggml-hpx-exec.cpp
        # to create per-chain CPU backends (ggml_backend_cpu_init /
        # ggml_backend_cpu_set_n_threads).  Remove when the prototype is
        # replaced by a registration-based backend factory.
        ggml-cpu
)

target_compile_features(ggml-hpx PRIVATE cxx_std_17)

# Experimental: fine-grained HPX region DAG (ggml-hpx-region-dag.h).
# OFF by default. Enable to develop and benchmark the DAG executor path
# without affecting the stable dispatch pipeline.
# Benchmarks that exercise this path live in hpx-bench/ and are gated on
# the same option there.
option(GGML_HPX_REGION_DAG
    "Enable experimental HPX fine-grained region DAG structures" OFF)
if(GGML_HPX_REGION_DAG)
    target_compile_definitions(ggml-hpx PRIVATE GGML_HPX_REGION_DAG)
    # Add the region executor implementation to the library.
    # The source is gated here rather than in the main source list so it is
    # only compiled (and GGML_HPX_REGION_DAG can be assumed present) when
    # the flag is explicitly enabled.
    target_sources(ggml-hpx PRIVATE ggml-hpx-region-exec.cpp ggml-hpx-packet.cpp ggml-hpx-lower.cpp ggml-hpx-compose.cpp ggml-hpx-exec-selective.cpp)
    message(STATUS "ggml-hpx: GGML_HPX_REGION_DAG enabled")
endif()
