#pragma once

// ggml-hpx-instrument.h
//
// Passive instrumentation for the HPX executor.
//
// No HPX performance counters. No APEX. No runtime tracing framework.
// All types here are self-contained and composable into any struct that
// wants to carry metrics.
//
// Layers:
//   ggml_hpx_counter       - relaxed atomic accumulator, suitable for hot paths
//   ggml_hpx_timing        - named nanosecond accumulator
//   ggml_hpx_scoped_timer  - RAII helper; adds elapsed ns to a ggml_hpx_timing
//   ggml_hpx_run_id        - opaque monotonic identifier for one executor run
//   ggml_hpx_region_id     - identifies a region within a run
//   ggml_hpx_metrics_hooks - optional benchmark-facing callback table

#include <atomic>
#include <chrono>
#include <cstdint>

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

// Monotonic run counter. The executor increments this before each run.
// Zero is reserved for "no run" / uninitialized.
using ggml_hpx_run_id = uint64_t;

// Index of a region within the plan executed during a particular run.
using ggml_hpx_region_id = uint32_t;

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

// A simple relaxed-atomic accumulator for per-run or lifetime event counts
// (regions started, abort checks fired, cache hits, etc.).
struct ggml_hpx_counter
{
    void increment() noexcept
    {
        value_.fetch_add(1, std::memory_order_relaxed);
    }

    void add(uint64_t n) noexcept
    {
        value_.fetch_add(n, std::memory_order_relaxed);
    }

    uint64_t load() const noexcept
    {
        return value_.load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        value_.store(0, std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> value_{0};
};

// ---------------------------------------------------------------------------
// Timing accumulator
// ---------------------------------------------------------------------------

// Named nanosecond accumulator. label is a string literal; not owned here.
struct ggml_hpx_timing
{
    char const* label = nullptr;    // string literal, not owned

    explicit ggml_hpx_timing(char const* lbl = nullptr) noexcept : label(lbl) {}

    void add_ns(uint64_t ns) noexcept
    {
        total_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    uint64_t total_ns() const noexcept
    {
        return total_ns_.load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        total_ns_.store(0, std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> total_ns_{0};
};

// ---------------------------------------------------------------------------
// Scoped timer
// ---------------------------------------------------------------------------

// RAII helper. Records start time on construction; adds elapsed nanoseconds
// to the target ggml_hpx_timing on destruction.
//
// Usage:
//   {
//       ggml_hpx_scoped_timer t(my_timing);
//       // ... work ...
//   } // elapsed added to my_timing here
struct ggml_hpx_scoped_timer
{
    explicit ggml_hpx_scoped_timer(ggml_hpx_timing& target) noexcept
      : target_(target)
      , start_(std::chrono::steady_clock::now())
    {
    }

    ~ggml_hpx_scoped_timer() noexcept
    {
        auto end = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start_)
                      .count();
        target_.add_ns(static_cast<uint64_t>(ns));
    }

    // Non-copyable, non-movable: the timer must outlive the scope it wraps.
    ggml_hpx_scoped_timer(ggml_hpx_scoped_timer const&) = delete;
    ggml_hpx_scoped_timer& operator=(ggml_hpx_scoped_timer const&) = delete;

private:
    ggml_hpx_timing& target_;
    std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Metrics hooks (benchmark-facing)
// ---------------------------------------------------------------------------

// Optional callback table. All pointers default to null (no-op).
// The executor checks each pointer before calling; callers install hooks
// before the first run and leave them in place.
//
// user_data is passed through to every callback unchanged.
//
// on_run_begin    - called just before the first region of a run dispatches
// on_run_end      - called after the last region completes (or abort returns)
// on_region_begin - called before a region starts
// on_region_end   - called after a region completes; elapsed_ns is wall time
//                   for that region only
// on_abort        - called when cooperative abort is observed during a run
struct ggml_hpx_metrics_hooks
{
    void (*on_run_begin)(ggml_hpx_run_id run, void* user_data) = nullptr;
    void (*on_run_end)(ggml_hpx_run_id run, uint64_t elapsed_ns,
        void* user_data) = nullptr;
    void (*on_region_begin)(ggml_hpx_run_id run, ggml_hpx_region_id region,
        void* user_data) = nullptr;
    void (*on_region_end)(ggml_hpx_run_id run, ggml_hpx_region_id region,
        uint64_t elapsed_ns, void* user_data) = nullptr;
    void (*on_abort)(ggml_hpx_run_id run, void* user_data) = nullptr;
    void* user_data = nullptr;
};
