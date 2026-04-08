#pragma once

// ggml-hpx-fwd.h
//
// Forward declarations for external ggml types used across the HPX layer.
//
// These mirror stable declarations in ggml.h and ggml-backend.h. The full
// headers are not included here; ggml-backend.h in particular is confined to
// ggml-hpx-adapter.cpp. If either upstream declaration changes, this header
// must be updated to match.
//
//   ggml_cgraph          - ggml.h line 386
//   ggml_backend         - pointee type of ggml_backend_t
//   ggml_backend_t       - ggml-backend.h line 27
//   ggml_backend_sched   - pointee type of ggml_backend_sched_t
//   ggml_backend_sched_t - ggml-backend.h line 294

struct ggml_cgraph;

struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;
