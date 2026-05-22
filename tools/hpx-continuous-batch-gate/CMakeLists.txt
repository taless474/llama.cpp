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

# M3a: long-running keep-alive smoke client. Proves engine.run() can
# be launched before any submission, idle-wait on inbox_cv_, process
# requests submitted at runtime via submit_request(), and exit cleanly
# when request_shutdown() is observed on a drained engine. Built by
# default; intentionally not installed.
add_executable(llama-hpx-engine-keepalive-smoke engine_keepalive_smoke.cpp)
target_compile_features(llama-hpx-engine-keepalive-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-keepalive-smoke PRIVATE llama-hpx-engine)

# M3b: queued-before-admission cancellation smoke client. Proves
# engine::cancel_request(rid) resolves a request drained into
# waiting_queue_consumable_ but never admitted, fulfilling its
# promise with status=cancelled and closing the caller-supplied
# stream with reason=cancelled. Built by default; intentionally not
# installed.
add_executable(llama-hpx-engine-queued-cancel-smoke engine_queued_cancel_smoke.cpp)
target_compile_features(llama-hpx-engine-queued-cancel-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-queued-cancel-smoke PRIVATE llama-hpx-engine)

# M3c: active-request cancellation smoke client. Proves
# engine::cancel_request(rid) cooperatively cancels a request that
# has been admitted to a seq and is actively decoding — the engine
# task sets seq.cancel_requested at the next apply_queued_cancellations
# call, and the existing cancel_should_observe → cancel_and_fulfill
# iter-boundary pipeline closes the stream with reason=cancelled,
# clears KV, and fulfills the promise with status=cancelled. Built
# by default; intentionally not installed.
add_executable(llama-hpx-engine-active-cancel-smoke engine_active_cancel_smoke.cpp)
target_compile_features(llama-hpx-engine-active-cancel-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-active-cancel-smoke PRIVATE llama-hpx-engine)

# M5a: epoch-aware stale-token identity smoke. Proves that a token
# captured for an earlier completed incarnation of a request_id does
# NOT cancel a later same-rid submission — instead the engine bumps
# engine_result::cancel_stale_epoch and leaves the live request
# untouched. Validates the (rid, epoch) matching path through
# engine::cancel_request(const cancel_token &) end-to-end without
# adding any new public output to the gate driver.
add_executable(llama-hpx-engine-stale-token-smoke engine_stale_token_smoke.cpp)
target_compile_features(llama-hpx-engine-stale-token-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-stale-token-smoke PRIVATE llama-hpx-engine)

# M5b: handle-cancel convenience-forwarder smoke. Proves
# submit_handle::cancel(engine &) is behavior-equivalent to
# engine::cancel_request(handle.token) across three sub-cases —
# queued cancel, active cancel, and stale-token rid-reuse — all
# driven via h.cancel(eng) on captured handles. Built by default;
# intentionally not installed.
add_executable(llama-hpx-engine-handle-cancel-smoke engine_handle_cancel_smoke.cpp)
target_compile_features(llama-hpx-engine-handle-cancel-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-handle-cancel-smoke PRIVATE llama-hpx-engine)

# M6a/M6b: per-request sampling_config carry + activation smoke.
# M6a established the data-model carry; M6b activates the stochastic
# branch (per-seq llama_sampler chain in admit_one when
# sampling.mode == stochastic). The smoke now asserts: (A) greedy/
# default reproduces the canonical TinyLlama / "Hello, my name is" /
# budget-8 hash 0x0619d4d1900c2365 (greedy argmax path bit-identical),
# and (B) a stochastic seed=42 config produces a hash distinct from
# the greedy canonical (proves the chain was built and the sampling
# site dispatched onto it). Built by default; intentionally not
# installed.
add_executable(llama-hpx-engine-sampling-config-carry-smoke engine_sampling_config_carry_smoke.cpp)
target_compile_features(llama-hpx-engine-sampling-config-carry-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-sampling-config-carry-smoke PRIVATE llama-hpx-engine)

# Diagnostic: minimal greedy reproducer for the keep-alive 3rd-or-
# later sequential submit_request hang exposed during M6b validation.
# Three sequential greedy submit_request calls on a single
# keep_alive engine with initial_idle_slots=2. Bounded by
# hpx::future::wait_for(30s) so the smoke does not hang
# indefinitely if the bug is present. Pre-fix expected output:
# "BUG_REPRODUCED" (engine task hung; smoke force-exits via _Exit).
# Post-fix expected output: "ALL_COMPLETED" (slot-recycle works).
# Built by default; intentionally not installed.
add_executable(llama-hpx-engine-keepalive-multi-submit-smoke engine_keepalive_multi_submit_smoke.cpp)
target_compile_features(llama-hpx-engine-keepalive-multi-submit-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-keepalive-multi-submit-smoke PRIVATE llama-hpx-engine)

# M6b: seeded stochastic sampling smoke. Proves the four M6b
# correctness properties on a single keep-alive engine:
#   (1) same-seed reproducibility — two seed=42 requests collapse to
#       byte-identical token sequences and hashes;
#   (2) seed sensitivity — seed=42 vs seed=43 diverge;
#   (3) stochastic differs from the greedy canonical fingerprint;
#   (4) greedy-after-stochastic still reproduces the canonical hash
#       (proves admit_one's sampler_chain.reset() runs before the
#       conditional build and finalize_and_fulfill's reset runs
#       before slot reuse — no stochastic-state bleed across
#       admissions). n_seq_max=2 forces at least one slot reuse.
# Built by default; intentionally not installed.
add_executable(llama-hpx-engine-sampling-stochastic-smoke engine_sampling_stochastic_smoke.cpp)
target_compile_features(llama-hpx-engine-sampling-stochastic-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-sampling-stochastic-smoke PRIVATE llama-hpx-engine)

# M6c: per-seq sampler isolation smoke. Single keep_alive engine
# with n_seq_max=2 and initial_idle_slots=2. Submits two stochastic
# requests (seed=42 / seed=43) before awaiting either, so both are
# bound concurrently to the available slots. Then a greedy submit
# exercises slot recycle (greedy hash must equal canonical), and a
# final default-knob stochastic seed=42 submit confirms a fresh
# chain is rebuilt on slot reuse. Built by default; intentionally
# not installed.
add_executable(llama-hpx-engine-sampling-independence-smoke engine_sampling_independence_smoke.cpp)
target_compile_features(llama-hpx-engine-sampling-independence-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-sampling-independence-smoke PRIVATE llama-hpx-engine)
