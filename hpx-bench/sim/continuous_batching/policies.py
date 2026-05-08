"""Three scheduling policies for the simulator.

See `docs/hpx/continuous_batching_simulator_design.md` §6.

- FifoContextPool: each active slot pays its own decode-call base cost
  (PER_SLOT cost mode), no inter-slot batching. Approximates the current
  llama-serving-bench shape.
- StaticBatching: shared batch within a group, no admission until the
  whole group drains.
- ContinuousBatching: shared batch, admission as soon as a slot frees.
"""
from __future__ import annotations

from typing import List, Optional

from sim import (
    BatchRow,
    CostMode,
    Iteration,
    RowKind,
    SchedulerState,
    Slot,
    SlotPhase,
    assign_request_to_slot,
)


def _first_free_slot(state: SchedulerState) -> Optional[Slot]:
    for s in state.active_slots:
        if s.phase == SlotPhase.WAITING:
            return s
    return None


def _build_shared_iteration(state: SchedulerState) -> Iteration:
    """Two-pass: PASS A decode rows, PASS B prefill rows.

    PASS B is gated by `state.cont_batching`: if false, prefill rows are
    only added when no decode rows are present (fallback semantics from
    upstream `update_slots`). Total rows are capped at `n_batch`.
    """
    rows: List[BatchRow] = []

    for slot in state.active_slots:
        if len(rows) >= state.n_batch:
            break
        if slot.phase == SlotPhase.DECODE and slot.decode_remaining > 0:
            rows.append(BatchRow(slot.slot_id, RowKind.DECODE))

    if state.cont_batching or not rows:
        for slot in state.active_slots:
            if len(rows) >= state.n_batch:
                break
            if slot.phase == SlotPhase.PREFILL and slot.prompt_remaining > 0:
                take = min(slot.prompt_remaining, state.n_batch - len(rows))
                for _ in range(take):
                    rows.append(BatchRow(slot.slot_id, RowKind.PREFILL))

    decode_calls = 1 if rows else 0
    return Iteration(
        rows=rows, cost_mode=CostMode.SHARED, decode_calls=decode_calls
    )


class FifoContextPool:
    name = "fifo_context_pool"

    def admit(self, state: SchedulerState) -> int:
        n = 0
        while state.waiting_queue:
            slot = _first_free_slot(state)
            if slot is None:
                break
            req = state.waiting_queue.pop(0)
            assign_request_to_slot(state, slot, req)
            n += 1
        return n

    def build_iteration(self, state: SchedulerState) -> Iteration:
        rows: List[BatchRow] = []
        decode_calls = 0
        for slot in state.active_slots:
            if (
                slot.phase == SlotPhase.PREFILL
                and slot.prompt_remaining > 0
            ):
                take = min(slot.prompt_remaining, state.n_batch)
                for _ in range(take):
                    rows.append(BatchRow(slot.slot_id, RowKind.PREFILL))
                if take > 0:
                    decode_calls += 1
            elif (
                slot.phase == SlotPhase.DECODE
                and slot.decode_remaining > 0
            ):
                rows.append(BatchRow(slot.slot_id, RowKind.DECODE))
                decode_calls += 1
        return Iteration(
            rows=rows,
            cost_mode=CostMode.PER_SLOT,
            decode_calls=decode_calls,
        )


class StaticBatching:
    name = "static_batching"

    def admit(self, state: SchedulerState) -> int:
        # Group barrier: only admit when ALL slots are WAITING.
        if not all(
            s.phase == SlotPhase.WAITING for s in state.active_slots
        ):
            return 0
        n = 0
        while state.waiting_queue and n < state.n_slots:
            slot = _first_free_slot(state)
            if slot is None:
                break
            req = state.waiting_queue.pop(0)
            assign_request_to_slot(state, slot, req)
            n += 1
        return n

    def build_iteration(self, state: SchedulerState) -> Iteration:
        return _build_shared_iteration(state)


class ContinuousBatching:
    name = "continuous_batching"

    def admit(self, state: SchedulerState) -> int:
        n = 0
        while state.waiting_queue:
            slot = _first_free_slot(state)
            if slot is None:
                break
            req = state.waiting_queue.pop(0)
            assign_request_to_slot(state, slot, req)
            n += 1
        return n

    def build_iteration(self, state: SchedulerState) -> Iteration:
        return _build_shared_iteration(state)


POLICIES = {
    "fifo_context_pool": FifoContextPool,
    "static_batching": StaticBatching,
    "continuous_batching": ContinuousBatching,
}
