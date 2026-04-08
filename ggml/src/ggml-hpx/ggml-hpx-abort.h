#pragma once

// ggml-hpx-abort.h
//
// Cooperative abort token for the HPX executor.
//
// Abort is NOT preemptive. The contract:
//   - no new region starts after check() returns true
//   - a running CPU region stops at the next safe checkpoint
//   - in-flight delegated work (e.g. a BLAS call) is not forcibly preempted
//   - the executor returns an aborted status after all started cleanup is done
//
// Abort checks must occur:
//   - before starting each region
//   - at safe checkpoints inside long CPU regions
//   - before launching dependent follow-on regions
//
// The token is owned by the executor and reset between runs. Callers must
// not hold a raw pointer to a token across a run boundary.

#include <atomic>

struct ggml_hpx_abort_token
{
    void request() noexcept
    {
        flag.store(true, std::memory_order_release);
    }

    // Returns true if abort has been requested. Safe to call from any thread.
    bool check() const noexcept
    {
        return flag.load(std::memory_order_acquire);
    }

    // Reset before a new run. Must be called before any worker sees this token.
    void reset() noexcept
    {
        flag.store(false, std::memory_order_relaxed);
    }

private:
    std::atomic<bool> flag{false};
};
