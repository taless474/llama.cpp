set(TARGET llama-hpx-continuous-batch-gate)

# HPX is required for both the engine library and the gate executable.
# The local install path is provided via -DHPX_DIR=... when the parent
# build is configured (see the tool's README).
find_package(HPX REQUIRED)

# Reusable engine library: HPX-native orchestration only.
# Only the engine task touches llama_context mutable state; this library
# owns engine.cpp, the process-level HPX runtime, and trace helpers.
add_library(llama-hpx-engine STATIC
    engine.cpp
    hpx_runtime.cpp
    trace.cpp)
target_compile_features(llama-hpx-engine PUBLIC cxx_std_17)
target_include_directories(llama-hpx-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-hpx-engine
    PUBLIC llama-common llama HPX::hpx ${CMAKE_THREAD_LIBS_INIT})

# Gate executable: CLI parsing, scripted external submitter, validation
# driver, and the gate's main(). Gate-only sources stay out of the
# library.
add_executable(${TARGET}
    hpx-continuous-batch-gate.cpp
    cli.cpp
    gate_validation.cpp
    submitter.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET}
    PRIVATE llama-hpx-engine
            llama-common llama HPX::hpx ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# M2e: minimal in-process smoke client. Proves the llama-hpx-engine
# static library is usable from a non-gate consumer that only sees the
# library's public headers. Built by default; intentionally not
# installed (proof-of-reusability artifact, not a deployable tool).
add_executable(llama-hpx-engine-smoke engine_smoke.cpp)
target_compile_features(llama-hpx-engine-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-smoke PRIVATE llama-hpx-engine)

# M2f: initial-idle-slot smoke client. Proves submit_request can be
# admitted into a slot popped from the engine's startup idle pool
# (engine_options::initial_idle_slots) without a dummy initial active.
# Built by default; intentionally not installed.
add_executable(llama-hpx-engine-idle-smoke engine_idle_smoke.cpp)
target_compile_features(llama-hpx-engine-idle-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-idle-smoke PRIVATE llama-hpx-engine)

# M2g: per-request streaming smoke client. Proves
# submit_request(want_stream=true) returns a real token_stream_receiver
# through submit_handle.stream, without any gate-style
# admitted_stream_handoffs_ drain. Built by default; intentionally not
# installed.
add_executable(llama-hpx-engine-stream-smoke engine_stream_smoke.cpp)
target_compile_features(llama-hpx-engine-stream-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-stream-smoke PRIVATE llama-hpx-engine)
