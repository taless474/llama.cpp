#pragma once

// ggml-hpx-runtime.h
//
// HPX runtime lifecycle, persistent worker teams, and scratch buffer pool.
//
// The runtime is the only object in the HPX layer that owns long-lived HPX
// state. It is created once per executor instance and destroyed when the
// executor shuts down. Nothing here is per-run; per-run dispatch is handled
// by ggml-hpx-exec.cpp, which calls into the runtime.
//
// Design:
//   ggml_hpx_runtime is an opaque struct defined in ggml-hpx-runtime.cpp.
//   This header does not include any HPX headers; those are confined to the
//   .cpp. The dispatch interface uses ggml_hpx_chunk_fn so that exec.cpp
//   can call it without pulling in HPX types through this header.
//
// Worker teams:
//   Two persistent teams: decode and prefill. The decode team is sized for
//   low-latency work (typically fewer threads, pinned). The prefill team is
//   sized for throughput. Both teams are created at runtime init and persist
//   until destroy. The exec layer does not create or tear down threads.
//
// Scratch buffer:
//   One pre-allocated scratch region owned by the runtime. Sized to the
//   maximum workspace_bytes seen across all plans. The exec layer fetches
//   the current pointer before each run via ggml_hpx_runtime_scratch_ptr
//   and passes it to workers through the dispatch user_data. The pointer
//   must not be used after ggml_hpx_runtime_scratch_ensure is called again
//   (reallocation may occur).

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Opaque runtime handle
// ---------------------------------------------------------------------------

// Full definition is in ggml-hpx-runtime.cpp. Callers hold a pointer only.
struct ggml_hpx_runtime;

// ---------------------------------------------------------------------------
// Creation parameters
// ---------------------------------------------------------------------------

struct ggml_hpx_runtime_params
{
    uint32_t n_decode_threads;      // worker count for the decode team
    uint32_t n_prefill_threads;     // worker count for the prefill team
    size_t initial_scratch_bytes;   // scratch pre-allocation at init; may grow later
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create and start the HPX runtime and worker teams. Returns a non-null
// pointer on success. The HPX runtime is started as a side effect; it must
// not already be running.
//
// Defined in ggml-hpx-runtime.cpp.
ggml_hpx_runtime* ggml_hpx_runtime_create(ggml_hpx_runtime_params const& params);

// Stop the HPX runtime, join worker teams, and free all owned resources
// including the scratch buffer. The pointer is invalid after this call.
//
// Defined in ggml-hpx-runtime.cpp.
void ggml_hpx_runtime_destroy(ggml_hpx_runtime* rt);

// ---------------------------------------------------------------------------
// Scratch buffer
// ---------------------------------------------------------------------------

// Return a pointer to the start of the scratch buffer. The pointer is valid
// until the next call to ggml_hpx_runtime_scratch_ensure on this runtime.
// Returns nullptr if no scratch has been allocated yet.
//
// Defined in ggml-hpx-runtime.cpp.
void* ggml_hpx_runtime_scratch_ptr(ggml_hpx_runtime* rt);

// Ensure the scratch buffer is at least bytes large. A no-op if the current
// allocation already satisfies the request. May reallocate; any pointer
// previously returned by ggml_hpx_runtime_scratch_ptr is invalid after a
// reallocation occurs.
//
// Defined in ggml-hpx-runtime.cpp.
void ggml_hpx_runtime_scratch_ensure(ggml_hpx_runtime* rt, size_t bytes);

// ---------------------------------------------------------------------------
// Parallel dispatch
// ---------------------------------------------------------------------------

// Dispatch callback type. Called from HPX worker threads.
//
// chunk_idx  - index of this chunk in [0, n_chunks)
// user_data  - passed through from the dispatch call unchanged; the exec
//              layer uses this to carry the graph pointer, scratch pointer,
//              abort token, and any other per-run state workers need
using ggml_hpx_chunk_fn = void (*)(uint32_t chunk_idx, void* user_data);

// Fan out n_chunks work items across the decode worker team and block until
// all complete. fn is called exactly once per chunk_idx in [0, n_chunks).
// Chunks may execute in any order and in parallel up to the team size.
//
// This call is synchronous from the caller's perspective: it returns only
// after all chunks have finished. Abort checking inside fn is the
// responsibility of the exec layer; the runtime does not inspect abort state.
//
// Defined in ggml-hpx-runtime.cpp.
void ggml_hpx_runtime_dispatch_decode(ggml_hpx_runtime* rt,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data);

// Fan out n_chunks work items across the prefill worker team and block until
// all complete. Semantics are identical to ggml_hpx_runtime_dispatch_decode
// except that the prefill team is used.
//
// Defined in ggml-hpx-runtime.cpp.
void ggml_hpx_runtime_dispatch_prefill(ggml_hpx_runtime* rt,
    uint32_t n_chunks, ggml_hpx_chunk_fn fn, void* user_data);
