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

# N2.7: engine-pool variant of the queued-cancel smoke. Same scenario
# as engine-queued-cancel-smoke but with the engine task running on
# the named single-PU HPX engine pool (start_once with
# enable_engine_pool=true; spawn via hpx_runtime::async_on_engine).
# Sets engine_options::lib.cooperative_yield_on_pump=false. Acts as
# the permanent regression guard for the N2.6b / N2.6c scheduler-
# livelock hazard: a regression that re-enables the yield under
# engine-pool placement will hang here. Built by default;
# intentionally not installed.
add_executable(llama-hpx-engine-queued-cancel-engine-pool-smoke
    engine_queued_cancel_engine_pool_smoke.cpp)
target_compile_features(llama-hpx-engine-queued-cancel-engine-pool-smoke
    PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-queued-cancel-engine-pool-smoke
    PRIVATE llama-hpx-engine)

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

# N5b: shutdown-vs-queued-but-unadmittable liveness reproducer. Single
# engine (n_seq_max=1, initial_idle_slots=0, budgets={}, keep_alive=true)
# with one non-streaming request drained to waiting_queue_consumable_ but
# never admittable. Tests whether request_shutdown() joins the engine.
# Pre-fix: engine re-parks (outer-tail shutdown-break is gated on an
# empty waiting queue), so the smoke FAILs cleanly via bounded join
# timeout + _Exit(nonzero). Post-fix: engine joins and the queued
# request resolves to a terminal status. Built by default; intentionally
# not installed.
add_executable(llama-hpx-engine-shutdown-queued-unadmittable-smoke engine_shutdown_queued_unadmittable_smoke.cpp)
target_compile_features(llama-hpx-engine-shutdown-queued-unadmittable-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-shutdown-queued-unadmittable-smoke PRIVATE llama-hpx-engine)

# N5a: cancel_freed -> natural-completion -> later-admission slot-recovery
# reproducer. Single slot (n_seq_max=1, initial_idle_slots=1, budgets={},
# reuse_completed=false, keep_alive=true): streaming request A is
# cancelled while active (slot -> cancel_freed), non-streaming request B
# is admitted from cancel_freed and completes naturally, then
# non-streaming request C probes admission. Pre-fix C is never admitted
# (the cancel_freed slot is leaked by finalize_and_fulfill) and is
# resolved as failed_reserved by the N5b shutdown drain, so the smoke
# FAILs cleanly (engine joins; no _Exit on the expected path). Post-fix C
# admits from the recovered free_idle_ slot and completes with the
# canonical b8 hash. Built by default; intentionally not installed.
add_executable(llama-hpx-engine-cancel-freed-completion-admit-smoke engine_cancel_freed_completion_admit_smoke.cpp)
target_compile_features(llama-hpx-engine-cancel-freed-completion-admit-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-cancel-freed-completion-admit-smoke PRIVATE llama-hpx-engine)

# Slice C: gated single-request chunked-prefill correctness smoke. Drives
# one greedy request ("Hello, my name is", budget=8) through the
# live-admission build path (Path A) on a single-slot keep-alive engine
# with engine_options::lib.prefill_budget_rows set to several values:
#   B=0 (unbounded) and B>=prompt_len (collapse) must reproduce the
#   canonical b8 hash 0x0619d4d1900c2365 (one-chunk == whole-prompt);
#   B=2 and B=4 must split prefill across multiple iters (verified via the
#   in-memory Phase-1 prefill_rows_per_iter / tokens_emitted_per_iter
#   diagnostics, which the smoke self-enables), complete with n_decoded==8
#   and no token before prefill completes, leave no residual KV, and be
#   deterministic across same-B repeats. A chunked hash differing from b8
#   is allowed (batch-shape FP) and recorded, not gated. Path B (preloaded)
#   is not exercised. Built by default; intentionally not installed.
add_executable(llama-hpx-engine-chunked-prefill-smoke engine_chunked_prefill_smoke.cpp)
target_compile_features(llama-hpx-engine-chunked-prefill-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-chunked-prefill-smoke PRIVATE llama-hpx-engine)

# Slice D: cancel-during-partial-prefill correctness smoke. A single-slot
# keep-alive engine (prefill_budget_rows=2, prompt "Hello, my name is")
# admits one streaming request and is parked at the end of the first work
# iter (K0=1) via the gate release/ack barrier, with the seq mid-prefill
# (partial KV, no token). The submitter cancels the active seq and releases
# the barrier; the cancel is observed in iter 2. Asserts: status=cancelled,
# n_decoded=0, n_decoded_at_cancel=0, no token_stream_event before a
# terminal close=cancelled; partial-prefill proof (0 < prefill rows < n,
# 0 tokens) snapshotted BEFORE the follow-up so it cannot be contaminated;
# then a follow-up non-streaming request reuses the freed slot and
# completes with canonical b8 0x0619d4d1900c2365; cancel_active_observed=1,
# cancelled_count=1, queued_cancelled=0, decode_failures=0,
# residual_kv_ok=true. Built by default; intentionally not installed.
add_executable(llama-hpx-engine-cancel-mid-prefill-smoke engine_cancel_mid_prefill_smoke.cpp)
target_compile_features(llama-hpx-engine-cancel-mid-prefill-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-cancel-mid-prefill-smoke PRIVATE llama-hpx-engine)

# Slice E: concurrent chunked-prefill correctness smoke (n_seq_max=2). One
# anchor request A (admitted before run, prefilled to a decoding state via a
# gate barrier at K=ceil(n/B)) decodes while a probe request B does chunked
# prefill in the SAME llama_decode. Proves per-seq causal masking: with the
# batch shape fixed, varying B's prompt content (same token length) must NOT
# change A's hash, and varying A's content must NOT change B's hash. Also
# gates within-shape repeat determinism, an actual overlap iter
# (prefill_rows>0 & decode_rows>0 & active_seqs==2), B prefill chunks <= B
# attributed via barrier snapshot (aggregate prefill_rows is not per-seq),
# decode_failures==0, residual_kv_ok, no cancelled/error. Solo-vs-concurrent
# hashes are NOT gated (cross-shape FP). Built by default; not installed.
add_executable(llama-hpx-engine-concurrent-chunked-prefill-smoke engine_concurrent_chunked_prefill_smoke.cpp)
target_compile_features(llama-hpx-engine-concurrent-chunked-prefill-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-concurrent-chunked-prefill-smoke PRIVATE llama-hpx-engine)

# Slice F (Option A): engine-only W1c-style PrefillBudgetPolicy comparison
# harness. NOT an apples-to-apples rerun of the original server-driven W1c
# (no HTTP, no SSE, no llama-server). Drives the same d2-staggered SHAPE —
# one anchor sequence decoding while a long-prefill probe is admitted —
# across engine_options::lib.prefill_budget_rows values B in {0,32,64,128}.
# Probe classes (L8/L64/L256/L1024) mirror the W1c driver's prompts. Per-B
# fresh llama_context per scenario, no cross-scenario KV carryover. Uses
# the gate release/ack barrier at iter K=1 (anchor "Hello, my name is"
# fits in one prefill chunk for every tested B) to deterministically
# stagger the probe so it lands while the anchor is decoding, then
# validates: anchor+probe completed with budgets exhausted, decode_failures
# ==0, cancelled_count==0, residual_kv_ok, at least one interference iter
# (prefill_rows>0 AND decode_rows>0 AND active_seq_count==2) is present
# post-barrier, and for B>0 the max per-iter prefill rows is <= B (policy
# bound). Emits Phase-1 diag JSONL via LLAMA_HPX_DIAG_METRICS_PATH plus
# its own scenario_begin / scenario_end marker lines for analyzer
# splitting. Built by default; intentionally not installed.
add_executable(llama-hpx-engine-w1c-policy-smoke engine_w1c_policy_smoke.cpp)
target_compile_features(llama-hpx-engine-w1c-policy-smoke PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-engine-w1c-policy-smoke PRIVATE llama-hpx-engine)

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

# N3.1 / Experiment 13: control-plane responsiveness benchmark.
# Dedicated binary linked against llama-hpx-engine. Phase 1 supports
# the W2 (queued-cancel) and W3 (multi-request streaming) workloads.
# Always sets engine_options::lib.enable_responsiveness_timing=true
# so the four request_result responsiveness timestamp fields are
# populated. The gate, all existing smokes, and hpx-server leave that
# option off, so they are unaffected by this binary. Built by
# default; intentionally not installed (benchmark artifact, not a
# deployable tool).
add_executable(llama-hpx-continuous-batch-responsiveness-bench
    hpx_continuous_batch_responsiveness_bench.cpp)
target_compile_features(llama-hpx-continuous-batch-responsiveness-bench
    PRIVATE cxx_std_17)
target_link_libraries(llama-hpx-continuous-batch-responsiveness-bench
    PRIVATE llama-hpx-engine)

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
