#include "trace.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace trace {

namespace {

std::atomic<bool> g_enabled{false};
std::once_flag    g_init_flag;

}  // namespace

void init() noexcept {
    std::call_once(g_init_flag, []() {
        const char * v = std::getenv("LLAMA_HPX_CB_TRACE");
        const bool on = v != nullptr && v[0] == '1' && v[1] == '\0';
        g_enabled.store(on, std::memory_order_release);
    });
}

bool enabled() noexcept {
    return g_enabled.load(std::memory_order_acquire);
}

void event(const char * fmt, ...) noexcept {
    if (!enabled()) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[hpx-cb-gate] event=");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

}  // namespace trace
