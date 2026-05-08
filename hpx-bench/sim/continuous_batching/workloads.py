"""Deterministic workloads.

See `docs/hpx/continuous_batching_simulator_design.md` §7.

Phase 2 (existing):
- exp11_like_all_short: 200 short requests at t=0; mirrors
  `hpx-bench/experiments/11_perf_deep_queue_short_requests`.
- mixed_realistic: bursty Poisson arrivals across short/medium/long
  classes, deterministic from a seed.

Phase 2B (added to isolate which workload axis makes continuous_batching
separate from static_batching):
- staggered_identical_short: identical short reqs, deterministic stagger.
- mixed_decode_only: identical prompt, three decode-length classes @ t=0.
- mixed_prompt_only: identical decode, three prompt-length classes @ t=0.
- mixed_prompt_and_decode: three (prompt, decode) classes @ t=0.
- bursty_mixed_deterministic: same three classes in deterministic bursts.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List

from sim import Request


def exp11_like_all_short(**_kw: Any) -> List[Request]:
    """200 short requests, all arriving at t=0, P=6 D=8."""
    return [
        Request(
            request_id=i,
            arrival_time_ms=0.0,
            prompt_tokens=6,
            decode_tokens=8,
            class_label="all_short",
        )
        for i in range(200)
    ]


_MIXED_CLASSES: Dict[str, tuple] = {
    "short": (16, 8),
    "medium": (128, 64),
    "long": (1024, 256),
}

_MIXED_WINDOWS = [
    # (t_start_ms, t_end_ms, lambda_per_s, class_mix)
    (0.0, 1000.0, 50.0, {"short": 0.8, "medium": 0.2, "long": 0.0}),
    (1000.0, 3000.0, 20.0, {"short": 0.4, "medium": 0.5, "long": 0.1}),
    (3000.0, 5000.0, 5.0, {"short": 0.2, "medium": 0.3, "long": 0.5}),
]


def mixed_realistic(
    seed: int = 42, total_cap: int = 100, **_kw: Any
) -> List[Request]:
    """Bursty Poisson arrivals across three classes.

    Each window has its own arrival rate λ (req/s) and a class mix.
    Inter-arrival times are exponential. The list is bounded at
    `total_cap` for reproducibility.
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    rid = 0
    for (t_start, t_end, lam_per_s, mix) in _MIXED_WINDOWS:
        if len(requests) >= total_cap:
            break
        last_t = (
            requests[-1].arrival_time_ms if requests else 0.0
        )
        t = max(t_start, last_t)
        lam_per_ms = lam_per_s / 1000.0
        keys = sorted(mix.keys())
        weights = [mix[k] for k in keys]
        while t < t_end and len(requests) < total_cap:
            dt = rng.expovariate(lam_per_ms)
            t = t + dt
            if t >= t_end:
                break
            cls = rng.choices(keys, weights=weights, k=1)[0]
            P, D = _MIXED_CLASSES[cls]
            requests.append(
                Request(
                    request_id=rid,
                    arrival_time_ms=t,
                    prompt_tokens=P,
                    decode_tokens=D,
                    class_label=cls,
                )
            )
            rid += 1
    return requests


# --- Phase 2B workloads (deterministic, no rng dependence) -----------------
#
# All Phase 2B mixed workloads use round-robin assignment of (rid mod 3) to
# the three classes, so total_cap divisible by 3 (e.g. 99) gives equal class
# counts. None of them depend on `seed`, but they accept it for CLI symmetry.

_THREE_DECODE_CLASSES = [
    ("short", 8),
    ("medium", 64),
    ("long", 256),
]
_THREE_PROMPT_CLASSES = [
    ("short", 16),
    ("medium", 128),
    ("long", 1024),
]
_THREE_PD_CLASSES = [
    # (label, prompt_tokens, decode_tokens)
    ("short", 16, 8),
    ("medium", 128, 64),
    ("long", 1024, 256),
]


def staggered_identical_short(
    total_cap: int = 200, stride_ms: float = 5.0, **_kw: Any
) -> List[Request]:
    """B. Same shape as exp11_like_all_short (P=6, D=8) but staggered.

    Arrivals are deterministic: request i arrives at i * stride_ms.
    No length variance, only arrival staggering — isolates the
    "staggered arrivals" axis.
    """
    return [
        Request(
            request_id=i,
            arrival_time_ms=i * stride_ms,
            prompt_tokens=6,
            decode_tokens=8,
            class_label="all_short",
        )
        for i in range(total_cap)
    ]


def mixed_decode_only(
    total_cap: int = 99, **_kw: Any
) -> List[Request]:
    """C. All arrive at t=0, identical prompt P=6, decode mix {8, 64, 256}.

    Round-robin class assignment: rid % 3.
    """
    out: List[Request] = []
    for i in range(total_cap):
        cls_label, decode = _THREE_DECODE_CLASSES[i % 3]
        out.append(
            Request(
                request_id=i,
                arrival_time_ms=0.0,
                prompt_tokens=6,
                decode_tokens=decode,
                class_label=cls_label,
            )
        )
    return out


def mixed_prompt_only(
    total_cap: int = 99, **_kw: Any
) -> List[Request]:
    """D. All arrive at t=0, identical decode D=8, prompt mix {16, 128, 1024}."""
    out: List[Request] = []
    for i in range(total_cap):
        cls_label, prompt = _THREE_PROMPT_CLASSES[i % 3]
        out.append(
            Request(
                request_id=i,
                arrival_time_ms=0.0,
                prompt_tokens=prompt,
                decode_tokens=8,
                class_label=cls_label,
            )
        )
    return out


def mixed_prompt_and_decode(
    total_cap: int = 99, **_kw: Any
) -> List[Request]:
    """E. All arrive at t=0, three (prompt, decode) classes round-robin.

    short=(16, 8), medium=(128, 64), long=(1024, 256).
    """
    out: List[Request] = []
    for i in range(total_cap):
        cls_label, prompt, decode = _THREE_PD_CLASSES[i % 3]
        out.append(
            Request(
                request_id=i,
                arrival_time_ms=0.0,
                prompt_tokens=prompt,
                decode_tokens=decode,
                class_label=cls_label,
            )
        )
    return out


def bursty_mixed_deterministic(
    total_cap: int = 99,
    n_bursts: int = 3,
    burst_spacing_ms: float = 500.0,
    **_kw: Any,
) -> List[Request]:
    """F. Same three (prompt, decode) classes as E, but in deterministic bursts.

    `total_cap` requests are split into `n_bursts` equal-sized groups
    arriving at t = k * burst_spacing_ms for k in [0, n_bursts).  Within a
    burst, all requests share an arrival time; class is round-robin by
    request_id (so every burst has a balanced class mix).
    """
    if n_bursts <= 0:
        raise ValueError("n_bursts must be >= 1")
    out: List[Request] = []
    per_burst = total_cap // n_bursts
    leftover = total_cap - per_burst * n_bursts
    rid = 0
    for k in range(n_bursts):
        size = per_burst + (1 if k < leftover else 0)
        t_arr = k * burst_spacing_ms
        for _ in range(size):
            cls_label, prompt, decode = _THREE_PD_CLASSES[rid % 3]
            out.append(
                Request(
                    request_id=rid,
                    arrival_time_ms=t_arr,
                    prompt_tokens=prompt,
                    decode_tokens=decode,
                    class_label=cls_label,
                )
            )
            rid += 1
    return out


WORKLOADS: Dict[str, Callable[..., List[Request]]] = {
    "exp11_like_all_short": exp11_like_all_short,
    "mixed_realistic": mixed_realistic,
    "staggered_identical_short": staggered_identical_short,
    "mixed_decode_only": mixed_decode_only,
    "mixed_prompt_only": mixed_prompt_only,
    "mixed_prompt_and_decode": mixed_prompt_and_decode,
    "bursty_mixed_deterministic": bursty_mixed_deterministic,
}
