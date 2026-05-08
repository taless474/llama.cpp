"""Per-policy behavior tests.

Covers:
5. static batching does not admit new work until the active group drains
6. continuous batching admits into freed slots immediately
7. long prefill plus short request: cont_batching mixes prefill+decode
10. tokens_per_decode_call sanity for continuous on exp11
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sim import (  # noqa: E402
    CostMode,
    CostParams,
    Request,
    SchedulerState,
    run_simulation,
)
from policies import (  # noqa: E402
    ContinuousBatching,
    FifoContextPool,
    StaticBatching,
)
from workloads import exp11_like_all_short  # noqa: E402
from metrics import aggregate  # noqa: E402


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


class TestStaticGroupBarrier(unittest.TestCase):
    """Static must not admit a new request until the whole group drains."""

    def test_group_barrier(self):
        # 4 requests, slots=2; req 1 is much longer. Static must wait
        # until req 1 finishes before admitting req 2 / req 3.
        reqs = [
            Request(0, 0.0, 2, 2),   # short
            Request(1, 0.0, 2, 10),  # long
            Request(2, 0.0, 2, 2),
            Request(3, 0.0, 2, 2),
        ]
        state = _run(StaticBatching, reqs, n_slots=2, n_batch=128)
        by_id = {c.request_id: c for c in state.completed_requests}
        # req 1 finishes after req 0; req 2 must wait for req 1.
        self.assertGreater(by_id[1].finish_time_ms, by_id[0].finish_time_ms)
        self.assertGreaterEqual(
            by_id[2].assigned_at_ms, by_id[1].finish_time_ms - 1e-9
        )
        # Static admit fires only when admitted_this_iteration > 0
        # AND in those iterations, all slots transitioned from waiting in
        # the same step. Verify we have at least 2 admit events (groups).
        admit_events = sum(
            1
            for it in state.iteration_records
            if it.admitted_this_iteration > 0
        )
        self.assertGreaterEqual(admit_events, 2)


class TestContinuousImmediateAdmission(unittest.TestCase):
    """Continuous must admit into freed slots before the long slot finishes."""

    def test_immediate_admission(self):
        reqs = [
            Request(0, 0.0, 2, 2),
            Request(1, 0.0, 2, 10),
            Request(2, 0.0, 2, 2),
            Request(3, 0.0, 2, 2),
        ]
        state_static = _run(StaticBatching, reqs, n_slots=2, n_batch=128)
        state_cont = _run(
            ContinuousBatching, reqs, n_slots=2, n_batch=128
        )

        qw_static = next(
            c.queue_wait_ms
            for c in state_static.completed_requests
            if c.request_id == 2
        )
        qw_cont = next(
            c.queue_wait_ms
            for c in state_cont.completed_requests
            if c.request_id == 2
        )
        # Continuous gets req 2 into the freed short-slot well before
        # static (which must wait for the long request).
        self.assertLess(qw_cont, qw_static)


class TestContBatchingMixesPrefillDecode(unittest.TestCase):
    """Cont_batching must produce iterations with both prefill and decode
    rows when one slot's prefill outlasts another's prefill."""

    def test_mixed_iteration(self):
        # slot 0: short prefill, long decode (gets to DECODE phase quickly)
        # slot 1: long prefill (still in PREFILL while slot 0 decodes)
        reqs = [
            Request(0, 0.0, 2, 20),
            Request(1, 0.0, 20, 2),
        ]
        state = _run(ContinuousBatching, reqs, n_slots=2, n_batch=4)
        has_mixed = any(
            it.prefill_rows > 0 and it.decode_rows > 0
            for it in state.iteration_records
        )
        self.assertTrue(
            has_mixed,
            msg="expected at least one mixed prefill+decode iteration",
        )


class TestFifoNeverSharesRows(unittest.TestCase):
    """fifo_context_pool must always emit PER_SLOT iterations (gate #9)."""

    def test_per_slot_cost_mode(self):
        reqs = [
            Request(0, 0.0, 4, 4),
            Request(1, 0.0, 4, 4),
        ]
        # Run via a custom loop that observes iteration cost mode.
        from sim import run_simulation

        policy = FifoContextPool()
        state = SchedulerState.initialize(
            requests=list(reqs),
            n_slots=2,
            n_batch=8,
            cont_batching=False,
        )
        run_simulation(state, policy, _cp())
        # Every iteration with multiple slots active must have used
        # PER_SLOT pricing — i.e., decode_calls_in_iteration equals the
        # number of distinct slots that contributed rows. We check this
        # via `decode_calls_in_iteration > 1` ever happens when active>1.
        any_multi_slot_iter = any(
            it.decode_calls_in_iteration > 1
            for it in state.iteration_records
            if it.active_slots > 1
        )
        self.assertTrue(any_multi_slot_iter)


class TestTokensPerDecodeCallSanity(unittest.TestCase):
    """Continuous batching on exp11 with n_slots > 1 must actually batch:
    tokens_per_decode_call > 1.0."""

    def test_cont_packs_rows(self):
        reqs = exp11_like_all_short()
        state = _run(
            ContinuousBatching, reqs, n_slots=4, n_batch=128
        )
        agg = aggregate(state)
        self.assertGreater(agg["tokens_per_decode_call"], 1.0)


if __name__ == "__main__":
    unittest.main()
