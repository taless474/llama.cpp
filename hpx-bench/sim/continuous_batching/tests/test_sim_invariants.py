"""Invariant tests for the simulator core.

Covers:
1. one request, one slot
2. two requests, one slot (serialization)
3. two slots, two requests (parallel)
4. n_batch smaller than desired rows
8. invariant failure if a slot double-owns a request
9. deterministic output for same config/seed
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sim import (  # noqa: E402
    CostParams,
    Request,
    SanityGateViolation,
    SchedulerState,
    SlotPhase,
    assign_request_to_slot,
    run_simulation,
)
from policies import (  # noqa: E402
    ContinuousBatching,
    FifoContextPool,
    StaticBatching,
)


def _cp() -> CostParams:
    return CostParams(
        base_decode_step_cost_ms=1.0,
        per_prompt_token_cost_ms=0.05,
        per_decode_token_cost_ms=0.5,
    )


def _run(policy_cls, requests, n_slots, n_batch):
    policy = policy_cls()
    cont_batching = policy_cls is not FifoContextPool
    state = SchedulerState.initialize(
        requests=list(requests),
        n_slots=n_slots,
        n_batch=n_batch,
        cont_batching=cont_batching,
    )
    run_simulation(state, policy, _cp())
    return state


class TestOneRequestOneSlot(unittest.TestCase):
    def test_all_policies(self):
        reqs = [Request(0, 0.0, 4, 4)]
        for cls in (FifoContextPool, StaticBatching, ContinuousBatching):
            with self.subTest(policy=cls.name):
                state = _run(cls, reqs, n_slots=1, n_batch=8)
                self.assertEqual(len(state.completed_requests), 1)
                comp = state.completed_requests[0]
                self.assertEqual(comp.request_id, 0)
                self.assertEqual(comp.prompt_tokens, 4)
                self.assertEqual(comp.decode_tokens, 4)
                self.assertGreater(comp.total_latency_ms, 0.0)
                self.assertGreaterEqual(
                    comp.first_token_time_ms, comp.assigned_at_ms
                )
                self.assertGreaterEqual(
                    comp.finish_time_ms, comp.first_token_time_ms
                )


class TestTwoRequestsOneSlot(unittest.TestCase):
    """One slot serializes two requests; req 1 must wait."""

    def test_all_policies(self):
        reqs = [Request(0, 0.0, 2, 2), Request(1, 0.0, 2, 2)]
        for cls in (FifoContextPool, StaticBatching, ContinuousBatching):
            with self.subTest(policy=cls.name):
                state = _run(cls, reqs, n_slots=1, n_batch=8)
                self.assertEqual(len(state.completed_requests), 2)
                by_id = {
                    c.request_id: c for c in state.completed_requests
                }
                self.assertEqual(by_id[0].queue_wait_ms, 0.0)
                self.assertGreater(by_id[1].queue_wait_ms, 0.0)
                # req 1 must be assigned at or after req 0's finish.
                self.assertGreaterEqual(
                    by_id[1].assigned_at_ms, by_id[0].finish_time_ms
                )


class TestTwoSlotsTwoRequests(unittest.TestCase):
    """Two slots, two requests: both run in parallel; neither waits."""

    def test_all_policies(self):
        reqs = [Request(0, 0.0, 2, 2), Request(1, 0.0, 2, 2)]
        for cls in (FifoContextPool, StaticBatching, ContinuousBatching):
            with self.subTest(policy=cls.name):
                state = _run(cls, reqs, n_slots=2, n_batch=8)
                self.assertEqual(len(state.completed_requests), 2)
                for c in state.completed_requests:
                    self.assertEqual(c.queue_wait_ms, 0.0)


class TestNBatchSmallerThanPrompt(unittest.TestCase):
    """Single request with prompt larger than n_batch must be processed
    correctly across multiple iterations."""

    def test_continuous(self):
        reqs = [Request(0, 0.0, 10, 2)]
        state = _run(ContinuousBatching, reqs, n_slots=1, n_batch=4)
        self.assertEqual(len(state.completed_requests), 1)
        # Prefill 10 rows / n_batch 4 → ceil(10/4)=3 prefill iterations.
        prefill_iters = sum(
            1 for it in state.iteration_records if it.prefill_rows > 0
        )
        self.assertGreaterEqual(prefill_iters, 3)
        self.assertEqual(
            state.completed_requests[0].prompt_tokens, 10
        )

    def test_static(self):
        reqs = [Request(0, 0.0, 10, 2)]
        state = _run(StaticBatching, reqs, n_slots=1, n_batch=4)
        self.assertEqual(len(state.completed_requests), 1)


class TestDoubleOwnership(unittest.TestCase):
    """assign_request_to_slot must reject a slot that's already busy."""

    def test_reject_busy_slot(self):
        reqs = [Request(0, 0.0, 2, 2)]
        state = SchedulerState.initialize(
            requests=reqs, n_slots=1, n_batch=8, cont_batching=True
        )
        slot = state.active_slots[0]
        slot.request_id = 99
        slot.phase = SlotPhase.DECODE
        bad = Request(1, 0.0, 2, 2)
        with self.assertRaises(SanityGateViolation):
            assign_request_to_slot(state, slot, bad)


class TestDeterministicOutput(unittest.TestCase):
    """Same config / seed → identical completion sequences."""

    def _completion_signature(self, state):
        return tuple(
            (
                c.request_id,
                round(c.assigned_at_ms, 6),
                round(c.first_token_time_ms, 6),
                round(c.finish_time_ms, 6),
                c.prompt_tokens,
                c.decode_tokens,
            )
            for c in sorted(
                state.completed_requests, key=lambda c: c.request_id
            )
        )

    def test_continuous_deterministic(self):
        from workloads import exp11_like_all_short

        reqs = exp11_like_all_short()
        state_a = _run(
            ContinuousBatching, reqs, n_slots=4, n_batch=128
        )
        state_b = _run(
            ContinuousBatching, reqs, n_slots=4, n_batch=128
        )
        self.assertEqual(
            self._completion_signature(state_a),
            self._completion_signature(state_b),
        )

    def test_mixed_deterministic(self):
        from workloads import mixed_realistic

        reqs_a = mixed_realistic(seed=42, total_cap=50)
        reqs_b = mixed_realistic(seed=42, total_cap=50)
        # Workload itself must be deterministic.
        self.assertEqual(
            tuple((r.request_id, r.arrival_time_ms, r.prompt_tokens, r.decode_tokens, r.class_label) for r in reqs_a),
            tuple((r.request_id, r.arrival_time_ms, r.prompt_tokens, r.decode_tokens, r.class_label) for r in reqs_b),
        )
        state_a = _run(
            ContinuousBatching, reqs_a, n_slots=2, n_batch=128
        )
        state_b = _run(
            ContinuousBatching, reqs_b, n_slots=2, n_batch=128
        )
        self.assertEqual(
            self._completion_signature(state_a),
            self._completion_signature(state_b),
        )


if __name__ == "__main__":
    unittest.main()
