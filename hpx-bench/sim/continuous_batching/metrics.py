"""Metrics aggregation and CSV writers.

See `docs/hpx/continuous_batching_simulator_design.md` §8.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from sim import Completion, IterationRecord, SchedulerState


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def aggregate(state: SchedulerState) -> Dict[str, Any]:
    completions: List[Completion] = state.completed_requests
    if not completions:
        return {
            "n_completed": 0,
            "makespan_ms": 0.0,
            "tokens_per_second": 0.0,
            "decode_calls": 0,
            "tokens_per_decode_call": 0.0,
            "average_batch_rows": 0.0,
            "n_iterations": 0,
        }
    arrivals = [c.arrival_time_ms for c in completions]
    finishes = [c.finish_time_ms for c in completions]
    makespan = max(finishes) - min(arrivals)
    total_tokens = sum(
        c.prompt_tokens + c.decode_tokens for c in completions
    )
    total_rows = sum(it.batch_size for it in state.iteration_records)
    decode_calls = state.decode_call_count

    latencies = [c.total_latency_ms for c in completions]
    ttfts = [c.time_to_first_token_ms for c in completions]
    queue_waits = [c.queue_wait_ms for c in completions]
    batch_sizes = [it.batch_size for it in state.iteration_records]
    actives = [it.active_slots for it in state.iteration_records]

    out: Dict[str, Any] = {
        "n_completed": len(completions),
        "makespan_ms": makespan,
        "tokens_per_second": (
            (total_tokens * 1000.0 / makespan) if makespan > 0 else 0.0
        ),
        "decode_calls": decode_calls,
        "tokens_per_decode_call": (
            (total_rows / decode_calls) if decode_calls > 0 else 0.0
        ),
        "average_batch_rows": (
            (total_rows / len(state.iteration_records))
            if state.iteration_records
            else 0.0
        ),
        "n_iterations": len(state.iteration_records),
        "total_latency_ms_p50": percentile(latencies, 50),
        "total_latency_ms_p90": percentile(latencies, 90),
        "total_latency_ms_p95": percentile(latencies, 95),
        "total_latency_ms_p99": percentile(latencies, 99),
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p90": percentile(ttfts, 90),
        "ttft_ms_p95": percentile(ttfts, 95),
        "ttft_ms_p99": percentile(ttfts, 99),
        "queue_wait_ms_p50": percentile(queue_waits, 50),
        "queue_wait_ms_p95": percentile(queue_waits, 95),
        "batch_size_p50": percentile([float(b) for b in batch_sizes], 50),
        "batch_size_p90": percentile([float(b) for b in batch_sizes], 90),
        "batch_size_p95": percentile([float(b) for b in batch_sizes], 95),
        "batch_size_p99": percentile([float(b) for b in batch_sizes], 99),
        "active_slots_p50": percentile([float(a) for a in actives], 50),
        "active_slots_p90": percentile([float(a) for a in actives], 90),
        "active_slots_p95": percentile([float(a) for a in actives], 95),
        "active_slots_p99": percentile([float(a) for a in actives], 99),
    }

    # Sorted (stable) class-split metrics.
    classes = sorted(
        {c.class_label for c in completions if c.class_label is not None}
    )
    for cls in classes:
        cls_lat = [
            c.total_latency_ms
            for c in completions
            if c.class_label == cls
        ]
        cls_ttft = [
            c.time_to_first_token_ms
            for c in completions
            if c.class_label == cls
        ]
        out[f"class_{cls}_count"] = len(cls_lat)
        out[f"class_{cls}_total_latency_ms_p50"] = percentile(cls_lat, 50)
        out[f"class_{cls}_total_latency_ms_p95"] = percentile(cls_lat, 95)
        out[f"class_{cls}_ttft_ms_p50"] = percentile(cls_ttft, 50)
        out[f"class_{cls}_ttft_ms_p95"] = percentile(cls_ttft, 95)
    return out


_REQUEST_FIELDS = [
    "request_id",
    "arrival_time_ms",
    "assigned_at_ms",
    "first_token_time_ms",
    "finish_time_ms",
    "prompt_tokens",
    "decode_tokens",
    "queue_wait_ms",
    "time_to_first_token_ms",
    "total_latency_ms",
    "prefill_ms",
    "decode_ms",
    "class_label",
]

_ITERATION_FIELDS = [
    "iteration_index",
    "t_iter_start_ms",
    "t_iter_end_ms",
    "prefill_rows",
    "decode_rows",
    "batch_size",
    "active_slots",
    "decode_calls_in_iteration",
    "iter_cost_ms",
    "admitted_this_iteration",
    "completed_this_iteration",
]


def write_requests_csv(state: SchedulerState, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_REQUEST_FIELDS)
        w.writeheader()
        for c in sorted(
            state.completed_requests, key=lambda c: c.request_id
        ):
            w.writerow({k: getattr(c, k) for k in _REQUEST_FIELDS})


def write_iterations_csv(state: SchedulerState, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ITERATION_FIELDS)
        w.writeheader()
        for r in state.iteration_records:
            w.writerow({k: getattr(r, k) for k in _ITERATION_FIELDS})


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    front = [
        "policy",
        "workload",
        "n_slots",
        "n_batch",
        "seed",
        "n_completed",
        "makespan_ms",
        "tokens_per_second",
        "decode_calls",
        "tokens_per_decode_call",
    ]
    ordered = [k for k in front if k in keys] + [
        k for k in keys if k not in front
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ordered})
