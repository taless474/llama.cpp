// Env-gated lifecycle trace for the HPX continuous-batch gate.
//
// `init()` reads `LLAMA_HPX_CB_TRACE`. When set to exactly "1", every
// `event(fmt, ...)` call emits a line of the form
//   [hpx-cb-gate] event=<fmt-output>
// on stderr. When trace is disabled (default), `event()` is one atomic
// load + early return per call site — no formatting, no allocation.
//
// `fmt` should start with the event name and continue with
// space-separated key=value pairs (grep-friendly, stable). The body is
// variadic (va_list); it is therefore defined in trace.cpp rather than
// inlined into the header.

#pragma once

namespace trace {

void init() noexcept;

bool enabled() noexcept;

void event(const char * fmt, ...) noexcept
    __attribute__((format(printf, 1, 2)));

}  // namespace trace
