# M7a: minimal HPX-driven HTTP serving binary (`llama-hpx-server`) plus
# two in-process smokes. Depends on the existing `llama-hpx-engine`
# static library defined in tools/hpx-continuous-batch-gate/. The
# build is therefore gated by LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE
# at the tools/CMakeLists.txt level.
#
# cpp-httplib is an explicit, documented non-HPX network adapter
# boundary: it spawns internal std::thread workers for HTTP I/O. Our
# own sources (this directory) introduce no std::thread / std::mutex /
# std::condition_variable / std::this_thread::sleep_for; HPX still
# owns the request lifecycle and engine control plane.

# Reuse the HPX runtime helpers + trace headers/sources defined in
# the gate directory. We only need the headers in this TU because the
# linked llama-hpx-engine static library already provides the symbols.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hpx-continuous-batch-gate)

find_package(HPX REQUIRED)

# llama-hpx-server — main binary. Long-running process; lifecycle is
# srv.listen() blocking until somebody calls srv.stop(). M7a has no
# signal handler, so the standalone binary is meant for manual curl
# experimentation and is terminated externally. The two smokes below
# spin a server up in-process and stop it through normal control
# flow.
add_executable(llama-hpx-server hpx-server.cpp)
target_compile_features(llama-hpx-server PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# Round-trip smoke. Boots model + engine + server in-process, drives
# a request via an in-process httplib::Client, asserts the canonical
# greedy round-trip, then srv.stop() + clean engine drain. Built by
# default; intentionally not installed.
add_executable(llama-hpx-server-smoke hpx_server_smoke.cpp)
target_compile_features(llama-hpx-server-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# Oversize-prompt reject smoke. Configures --max-prompt-tokens=8 and
# asserts that prompts exceeding the cap return HTTP 413 while the
# server remains alive (a subsequent in-bounds POST succeeds). Built
# by default; intentionally not installed.
add_executable(llama-hpx-server-oversize-smoke hpx_server_oversize_smoke.cpp)
target_compile_features(llama-hpx-server-oversize-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-oversize-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7b: positive-path sampling smoke. Drives four sequential POSTs
# against one in-process server — omitted sampling (greedy canonical),
# stochastic seed=42 (M6b anchor), stochastic seed=43 (seed
# sensitivity), and full non-default knobs. Built by default;
# intentionally not installed.
add_executable(llama-hpx-server-sampling-smoke hpx_server_sampling_smoke.cpp)
target_compile_features(llama-hpx-server-sampling-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-sampling-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7b: negative-path sampling smoke. Six invalid `sampling` POSTs must
# return HTTP 422 with error.code == "invalid_sampling"; a final
# greedy/default POST must still hit the canonical greedy hash,
# proving the rejection path leaves the engine state clean. Built by
# default; intentionally not installed.
add_executable(llama-hpx-server-invalid-sampling-smoke hpx_server_invalid_sampling_smoke.cpp)
target_compile_features(llama-hpx-server-invalid-sampling-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-invalid-sampling-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7c: SSE streaming round-trip smoke. Drives a single POST /completion
# with "stream": true, parses the chunked text/event-stream body, and
# asserts at least one token event plus a terminal done event matching
# the budget-8 greedy canonical hash. Built by default; intentionally
# not installed.
add_executable(llama-hpx-server-stream-smoke hpx_server_stream_smoke.cpp)
target_compile_features(llama-hpx-server-stream-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-stream-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7d: SSE client-disconnect smoke. Drives a streaming POST whose
# client aborts mid-response via a ContentReceiver returning false,
# then sends a non-streaming sanity POST. Asserts the server stayed
# alive, the canonical greedy hash is unchanged, and the engine
# counters reflect that the disconnect path issued a fire-and-forget
# cancellation (cancel_request_calls >= 1, cancel_unknown_request_id
# == 0, decode_failures == 0, residual_kv_ok == true). Built by
# default; intentionally not installed.
add_executable(llama-hpx-server-stream-disconnect-smoke hpx_server_stream_disconnect_smoke.cpp)
target_compile_features(llama-hpx-server-stream-disconnect-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-stream-disconnect-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7e: HTTP-layer backpressure smoke. Boots an in-process server with
# --n-seq-max=1 --max-concurrent=1 and drives two concurrent POSTs
# coordinated through HPX-native primitives (no std::thread / sleep);
# asserts exactly one HTTP 200 with the canonical greedy hash and
# exactly one HTTP 503 carrying `error.code == "server_busy"` and
# `Retry-After: 1`. A follow-up serial canonical POST must return
# 200 with the same hash — proves capacity is released after
# completion. Built by default; intentionally not installed.
add_executable(llama-hpx-server-backpressure-smoke hpx_server_backpressure_smoke.cpp)
target_compile_features(llama-hpx-server-backpressure-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-backpressure-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# N4: in-process placement smoke that runs the round-trip and SSE
# client-disconnect paths with the engine HPX task scheduled on the
# named single-PU "engine" thread pool (start_once with
# enable_engine_pool=true; spawn via hpx_runtime::async_on_engine;
# engine_options::lib.cooperative_yield_on_pump=false). Acts as the
# permanent regression guard for the cpp-httplib + engine-pool
# integration path. Built by default; intentionally not installed.
add_executable(llama-hpx-server-engine-pool-smoke hpx_server_engine_pool_smoke.cpp)
target_compile_features(llama-hpx-server-engine-pool-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-engine-pool-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})

# M7f: composition of M7d (SSE client-disconnect → cancel) and M7e
# (HTTP-door capacity lease). Boots in-process server with
# --n-seq-max=1 --max-concurrent=1; drives a streaming holder POST
# that the client aborts mid-response via a ContentReceiver returning
# false after ~64 bytes; observes cap.in_flight transitions to gate
# (a) the challenger POST which must receive HTTP 503 server_busy
# while the holder is mid-stream, and (b) the serial canonical
# follow-up which must receive HTTP 200 with the budget-8 greedy
# canonical hash after the holder's disconnect/finalize path has
# unwound and released the lease. Asserts both the M7d engine
# cancellation counters and the M7e capacity counters. Built by
# default; intentionally not installed.
add_executable(llama-hpx-server-backpressure-stream-disconnect-smoke hpx_server_backpressure_stream_disconnect_smoke.cpp)
target_compile_features(llama-hpx-server-backpressure-stream-disconnect-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-server-backpressure-stream-disconnect-smoke
    PRIVATE llama-hpx-engine
            llama-common
            llama
            cpp-httplib
            HPX::hpx
            ${CMAKE_THREAD_LIBS_INIT})
