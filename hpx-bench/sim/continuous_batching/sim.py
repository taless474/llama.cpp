"""Discrete-event continuous-batching simulator core.

Pure logic, no I/O. Models the upstream `llama-server` `update_slots()` loop
at a level sufficient to compare scheduling policies. See
`docs/hpx/continuous_batching_simulator_design.md` and
`docs/hpx/continuous_batching_upstream_notes.md`.

This module knows nothing about HPX, llama.cpp, or real inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SlotPhase(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"


class RowKind(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class CostMode(Enum):
    SHARED = "shared"
    PER_SLOT = "per_slot"


class SanityGateViolation(AssertionError):
    """Raised when a §9 invariant is violated. Aborts the run."""


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time_ms: float
    prompt_tokens: int
    decode_tokens: int
    priority: int = 0
    cancel_at_ms: Optional[float] = None
    class_label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0:
            raise ValueError(
                f"request {self.request_id}: prompt_tokens must be > 0"
            )
        if self.decode_tokens <= 0:
            raise ValueError(
                f"request {self.request_id}: decode_tokens must be > 0"
            )


@dataclass
class Slot:
    slot_id: int
    seq_id: int  # equals slot_id by construction
    request_id: Optional[int] = None
    phase: SlotPhase = SlotPhase.WAITING
    prompt_remaining: int = 0
    decode_remaining: int = 0
    generated_tokens: int = 0
    first_token_time_ms: Optional[float] = None
    finish_time_ms: Optional[float] = None
    assigned_at_ms: Optional[float] = None
    arrival_time_ms: Optional[float] = None
    prompt_tokens: int = 0
    decode_tokens: int = 0
    class_label: Optional[str] = None


@dataclass(frozen=True)
class BatchRow:
    slot_id: int
    kind: RowKind


@dataclass
class Iteration:
    rows: List[BatchRow]
    cost_mode: CostMode
    decode_calls: int


@dataclass
class IterationRecord:
    iteration_index: int
    t_iter_start_ms: float
    t_iter_end_ms: float
    prefill_rows: int
    decode_rows: int
    batch_size: int
    active_slots: int
    decode_calls_in_iteration: int
    iter_cost_ms: float
    admitted_this_iteration: int
    completed_this_iteration: int


@dataclass
class Completion:
    request_id: int
    arrival_time_ms: float
    assigned_at_ms: float
    first_token_time_ms: float
    finish_time_ms: float
    prompt_tokens: int
    decode_tokens: int
    queue_wait_ms: float
    time_to_first_token_ms: float
    total_latency_ms: float
    prefill_ms: float
    decode_ms: float
    class_label: Optional[str] = None


@dataclass
class CostParams:
    base_decode_step_cost_ms: float = 1.0
    per_prompt_token_cost_ms: float = 0.05
    per_decode_token_cost_ms: float = 0.5
    per_active_slot_overhead_ms: float = 0.0
    streaming_emit_cost_ms: float = 0.0


@dataclass
class SchedulerState:
    current_time_ms: float
    waiting_queue: List[Request]
    active_slots: List[Slot]
    completed_requests: List[Completion]
    n_slots: int
    n_batch: int
    cont_batching: bool
    n_ubatch: Optional[int] = None
    iteration_count: int = 0
    decode_call_count: int = 0
    pending_arrivals: List[Request] = field(default_factory=list)
    iteration_records: List[IterationRecord] = field(default_factory=list)
    submitted_count: int = 0
    last_admitted: int = 0
    last_completed: int = 0

    @classmethod
    def initialize(
        cls,
        requests: List[Request],
        n_slots: int,
        n_batch: int,
        cont_batching: bool,
        n_ubatch: Optional[int] = None,
    ) -> "SchedulerState":
        sorted_arrivals = sorted(
            requests, key=lambda r: (r.arrival_time_ms, r.request_id)
        )
        slots = [Slot(slot_id=i, seq_id=i) for i in range(n_slots)]
        return cls(
            current_time_ms=0.0,
            waiting_queue=[],
            active_slots=slots,
            completed_requests=[],
            n_slots=n_slots,
            n_batch=n_batch,
            cont_batching=cont_batching,
            n_ubatch=n_ubatch,
            pending_arrivals=list(sorted_arrivals),
            submitted_count=len(sorted_arrivals),
        )


def assign_request_to_slot(
    state: SchedulerState, slot: Slot, req: Request
) -> None:
    """Move `req` into `slot`. Assert the slot is actually free first.

    This is gate #2 (no slot owns more than one request at a time) and
    gate #8 (admission only into free slots), enforced at the moment of
    admission.
    """
    if slot.request_id is not None or slot.phase != SlotPhase.WAITING:
        raise SanityGateViolation(
            f"slot {slot.slot_id} not free: phase={slot.phase}, "
            f"request_id={slot.request_id}"
        )
    slot.request_id = req.request_id
    slot.phase = SlotPhase.PREFILL
    slot.prompt_remaining = req.prompt_tokens
    slot.decode_remaining = req.decode_tokens
    slot.generated_tokens = 0
    slot.first_token_time_ms = None
    slot.finish_time_ms = None
    slot.assigned_at_ms = state.current_time_ms
    slot.arrival_time_ms = req.arrival_time_ms
    slot.prompt_tokens = req.prompt_tokens
    slot.decode_tokens = req.decode_tokens
    slot.class_label = req.class_label


def reset_slot(slot: Slot) -> None:
    slot.request_id = None
    slot.phase = SlotPhase.WAITING
    slot.prompt_remaining = 0
    slot.decode_remaining = 0
    slot.generated_tokens = 0
    slot.first_token_time_ms = None
    slot.finish_time_ms = None
    slot.assigned_at_ms = None
    slot.arrival_time_ms = None
    slot.prompt_tokens = 0
    slot.decode_tokens = 0
    slot.class_label = None


def advance_arrivals(state: SchedulerState) -> None:
    while (
        state.pending_arrivals
        and state.pending_arrivals[0].arrival_time_ms <= state.current_time_ms
    ):
        state.waiting_queue.append(state.pending_arrivals.pop(0))


def next_arrival_time(state: SchedulerState) -> Optional[float]:
    if state.pending_arrivals:
        return state.pending_arrivals[0].arrival_time_ms
    return None


def all_idle(state: SchedulerState) -> bool:
    return all(s.phase == SlotPhase.WAITING for s in state.active_slots)


def all_done(state: SchedulerState) -> bool:
    if state.pending_arrivals or state.waiting_queue:
        return False
    return all_idle(state)


def compute_iter_cost(
    rows: List[BatchRow],
    cost_mode: CostMode,
    n_active: int,
    cp: CostParams,
) -> float:
    """Linear cost model. SHARED policies pay base once; PER_SLOT pays once
    per contributing slot."""
    if not rows:
        return 0.0
    if cost_mode == CostMode.SHARED:
        n_p = sum(1 for r in rows if r.kind == RowKind.PREFILL)
        n_d = sum(1 for r in rows if r.kind == RowKind.DECODE)
        return (
            cp.base_decode_step_cost_ms
            + cp.per_prompt_token_cost_ms * n_p
            + cp.per_decode_token_cost_ms * n_d
            + cp.per_active_slot_overhead_ms * n_active
            + cp.streaming_emit_cost_ms * n_d
        )
    by_slot: Dict[int, List[BatchRow]] = {}
    for r in rows:
        by_slot.setdefault(r.slot_id, []).append(r)
    total = 0.0
    for slot_rows in by_slot.values():
        n_p = sum(1 for r in slot_rows if r.kind == RowKind.PREFILL)
        n_d = sum(1 for r in slot_rows if r.kind == RowKind.DECODE)
        total += (
            cp.base_decode_step_cost_ms
            + cp.per_prompt_token_cost_ms * n_p
            + cp.per_decode_token_cost_ms * n_d
            + cp.streaming_emit_cost_ms * n_d
        )
    total += cp.per_active_slot_overhead_ms * n_active
    return total


def check_iteration_constraints(
    state: SchedulerState, iteration: Iteration
) -> None:
    """Gate #7: batch_size <= n_batch (per call, not per iteration)."""
    if iteration.cost_mode == CostMode.SHARED:
        if len(iteration.rows) > state.n_batch:
            raise SanityGateViolation(
                f"shared batch_size {len(iteration.rows)} > n_batch "
                f"{state.n_batch}"
            )
    else:
        by_slot: Dict[int, int] = {}
        for r in iteration.rows:
            by_slot[r.slot_id] = by_slot.get(r.slot_id, 0) + 1
        for sid, count in by_slot.items():
            if count > state.n_batch:
                raise SanityGateViolation(
                    f"per_slot batch for slot {sid} = {count} > "
                    f"n_batch {state.n_batch}"
                )


def check_state_invariants(state: SchedulerState) -> None:
    """Gates #2, #5, #6 plus the slot-uniqueness check."""
    n_active = sum(
        1 for s in state.active_slots if s.phase != SlotPhase.WAITING
    )
    if n_active > state.n_slots:
        raise SanityGateViolation(
            f"active_slots {n_active} > n_slots {state.n_slots}"
        )
    seen: set = set()
    for slot in state.active_slots:
        if slot.prompt_remaining < 0:
            raise SanityGateViolation(
                f"slot {slot.slot_id} prompt_remaining < 0"
            )
        if slot.decode_remaining < 0:
            raise SanityGateViolation(
                f"slot {slot.slot_id} decode_remaining < 0"
            )
        if slot.generated_tokens > slot.decode_tokens:
            raise SanityGateViolation(
                f"slot {slot.slot_id} generated {slot.generated_tokens} "
                f"> decode_tokens {slot.decode_tokens}"
            )
        if slot.request_id is not None:
            if slot.request_id in seen:
                raise SanityGateViolation(
                    f"request {slot.request_id} owned by multiple slots"
                )
            seen.add(slot.request_id)


def apply_iteration(
    state: SchedulerState,
    iteration: Iteration,
    t_start_ms: float,
    t_end_ms: float,
) -> int:
    """Apply row effects, transition phases, emit completions, release slots.

    Returns the number of requests completed in this iteration.
    """
    by_slot: Dict[int, List[BatchRow]] = {}
    for r in iteration.rows:
        by_slot.setdefault(r.slot_id, []).append(r)

    completed = 0
    for slot_id, rows in by_slot.items():
        slot = state.active_slots[slot_id]
        for row in rows:
            if row.kind == RowKind.PREFILL:
                if slot.phase != SlotPhase.PREFILL:
                    raise SanityGateViolation(
                        f"slot {slot_id} got prefill row in phase "
                        f"{slot.phase}"
                    )
                slot.prompt_remaining -= 1
                if slot.prompt_remaining == 0:
                    slot.phase = SlotPhase.DECODE
            else:  # DECODE
                if slot.phase != SlotPhase.DECODE:
                    raise SanityGateViolation(
                        f"slot {slot_id} got decode row in phase "
                        f"{slot.phase}"
                    )
                slot.decode_remaining -= 1
                slot.generated_tokens += 1
                if slot.first_token_time_ms is None:
                    slot.first_token_time_ms = t_end_ms
                if slot.decode_remaining == 0:
                    slot.phase = SlotPhase.DONE
                    slot.finish_time_ms = t_end_ms

    for slot in state.active_slots:
        if slot.phase != SlotPhase.DONE:
            continue
        comp = Completion(
            request_id=slot.request_id,  # type: ignore[arg-type]
            arrival_time_ms=slot.arrival_time_ms,  # type: ignore[arg-type]
            assigned_at_ms=slot.assigned_at_ms,  # type: ignore[arg-type]
            first_token_time_ms=slot.first_token_time_ms,  # type: ignore[arg-type]
            finish_time_ms=slot.finish_time_ms,  # type: ignore[arg-type]
            prompt_tokens=slot.prompt_tokens,
            decode_tokens=slot.decode_tokens,
            queue_wait_ms=slot.assigned_at_ms - slot.arrival_time_ms,  # type: ignore[operator]
            time_to_first_token_ms=(
                slot.first_token_time_ms - slot.arrival_time_ms  # type: ignore[operator]
            ),
            total_latency_ms=slot.finish_time_ms - slot.arrival_time_ms,  # type: ignore[operator]
            prefill_ms=slot.first_token_time_ms - slot.assigned_at_ms,  # type: ignore[operator]
            decode_ms=slot.finish_time_ms - slot.first_token_time_ms,  # type: ignore[operator]
            class_label=slot.class_label,
        )
        if comp.first_token_time_ms < comp.assigned_at_ms:
            raise SanityGateViolation(
                f"req {comp.request_id} first_token < assigned_at"
            )
        if comp.finish_time_ms < comp.first_token_time_ms:
            raise SanityGateViolation(
                f"req {comp.request_id} finish < first_token"
            )
        state.completed_requests.append(comp)
        reset_slot(slot)
        completed += 1

    return completed


def run_simulation(
    state: SchedulerState,
    policy,
    cp: CostParams,
    max_iterations: int = 1_000_000,
) -> SchedulerState:
    """The main engine loop. Mirrors `update_slots()` shape:

    1. advance arrivals to current_time_ms
    2. policy.admit -> may move waiting requests into free slots
    3. policy.build_iteration -> shared batch (or per-slot for fifo)
    4. compute cost; advance current_time_ms
    5. apply_iteration -> phase transitions, completions, slot release
    6. record metrics; check invariants

    Aborts via SanityGateViolation if any §9 invariant fails.
    """
    while not all_done(state):
        if state.iteration_count >= max_iterations:
            raise SanityGateViolation(
                f"exceeded max_iterations {max_iterations}; possible "
                "infinite loop"
            )

        advance_arrivals(state)
        if not state.waiting_queue and all_idle(state):
            nxt = next_arrival_time(state)
            if nxt is None:
                break
            if nxt > state.current_time_ms:
                state.current_time_ms = nxt
                advance_arrivals(state)

        n_admitted = policy.admit(state)
        state.last_admitted = n_admitted

        iteration = policy.build_iteration(state)
        check_iteration_constraints(state, iteration)

        n_active = sum(
            1 for s in state.active_slots if s.phase != SlotPhase.WAITING
        )
        cost = compute_iter_cost(
            iteration.rows, iteration.cost_mode, n_active, cp
        )
        t_start = state.current_time_ms
        t_end = t_start + cost

        if not iteration.rows:
            nxt = next_arrival_time(state)
            if nxt is None:
                if not all_done(state):
                    raise SanityGateViolation(
                        "simulator stuck: no rows, no arrivals"
                    )
                break
            if nxt <= state.current_time_ms:
                raise SanityGateViolation(
                    "non-progressing iteration: empty rows, "
                    "no time advance possible"
                )
            state.current_time_ms = nxt
            continue

        completed = apply_iteration(state, iteration, t_start, t_end)
        state.last_completed = completed

        if t_end < state.current_time_ms:
            raise SanityGateViolation(
                f"time non-monotonic: t_end {t_end} < "
                f"current {state.current_time_ms}"
            )
        state.current_time_ms = t_end

        rec = IterationRecord(
            iteration_index=state.iteration_count,
            t_iter_start_ms=t_start,
            t_iter_end_ms=t_end,
            prefill_rows=sum(
                1 for r in iteration.rows if r.kind == RowKind.PREFILL
            ),
            decode_rows=sum(
                1 for r in iteration.rows if r.kind == RowKind.DECODE
            ),
            batch_size=len(iteration.rows),
            active_slots=n_active,
            decode_calls_in_iteration=iteration.decode_calls,
            iter_cost_ms=cost,
            admitted_this_iteration=n_admitted,
            completed_this_iteration=completed,
        )
        state.iteration_records.append(rec)
        state.iteration_count += 1
        state.decode_call_count += iteration.decode_calls

        check_state_invariants(state)

    if len(state.completed_requests) != state.submitted_count:
        raise SanityGateViolation(
            f"completed {len(state.completed_requests)} != "
            f"submitted {state.submitted_count}"
        )
    return state
