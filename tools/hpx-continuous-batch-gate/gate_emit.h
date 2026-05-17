// Final stdout label helpers for the HPX continuous-batch gate.
//
// `STEP` is the binary's current slice label; `emit_pass` / `emit_fail`
// print the terminal stdout line. Both `main` and the validation
// harness need to call these — so they live in a shared header.
// Definitions are `inline` to keep the header self-contained and ODR-
// safe across translation units.

#pragma once

#include <cstdio>

namespace gate_emit {

inline constexpr const char * STEP = "HPX_CB_STREAM_STEP7";

inline void emit_pass() {
    fprintf(stdout, "%s: PASS\n", STEP);
}

inline void emit_fail(const char * reason) {
    fprintf(stdout, "%s: FAIL: %s\n", STEP, reason);
}

}  // namespace gate_emit
