"""
Unit tests for ParallelDispatcher — lightweight parallel dispatch utility.

Tests cover:
- Sequential mode (enabled=False)
- Parallel mode (enabled=True)
- Result ordering matches submission order
- Individual failure isolation
- Duration tracking
- Max workers concurrency cap
- Edge cases (empty list, single item)
"""

import threading
import time
from collections import OrderedDict

import pytest

from chal.utilities.parallel import ParallelDispatcher, WorkItem, WorkResult


# ==============================================
# Sequential Mode Tests
# ==============================================

class TestSequentialMode:
    """Tests for ParallelDispatcher with enabled=False."""

    def test_sequential_runs_in_order(self):
        """enabled=False runs items sequentially in submission order."""
        dispatcher = ParallelDispatcher(max_workers=4, enabled=False)

        call_order = []

        def make_fn(label):
            def fn():
                call_order.append(label)
                return label
            return fn

        items = [
            WorkItem(key="A", callable=make_fn("A")),
            WorkItem(key="B", callable=make_fn("B")),
            WorkItem(key="C", callable=make_fn("C")),
        ]

        results = dispatcher.run(items)

        assert call_order == ["A", "B", "C"]
        assert list(results.keys()) == ["A", "B", "C"]
        assert results["A"].result == "A"
        assert results["B"].result == "B"
        assert results["C"].result == "C"

    def test_sequential_captures_errors(self):
        """In sequential mode, exceptions are captured per-item."""
        dispatcher = ParallelDispatcher(enabled=False)

        def fail():
            raise ValueError("boom")

        items = [
            WorkItem(key="ok", callable=lambda: "fine"),
            WorkItem(key="fail", callable=fail),
            WorkItem(key="ok2", callable=lambda: "also fine"),
        ]

        results = dispatcher.run(items)

        assert results["ok"].result == "fine"
        assert results["ok"].error is None
        assert results["fail"].error is not None
        assert isinstance(results["fail"].error, ValueError)
        assert results["ok2"].result == "also fine"

    def test_sequential_preserves_context(self):
        """Context dict passes through unchanged in sequential mode."""
        dispatcher = ParallelDispatcher(enabled=False)

        items = [
            WorkItem(key="A", callable=lambda: 1, context={"idx": 0, "name": "first"}),
        ]

        results = dispatcher.run(items)
        assert results["A"].context == {"idx": 0, "name": "first"}


# ==============================================
# Parallel Mode Tests
# ==============================================

class TestParallelMode:
    """Tests for ParallelDispatcher with enabled=True."""

    def test_parallel_returns_all_results(self):
        """All work items produce results in parallel mode."""
        dispatcher = ParallelDispatcher(max_workers=4, enabled=True)

        items = [
            WorkItem(key=f"item-{i}", callable=lambda i=i: i * 10)
            for i in range(5)
        ]

        results = dispatcher.run(items)

        assert len(results) == 5
        for i in range(5):
            assert results[f"item-{i}"].result == i * 10
            assert results[f"item-{i}"].error is None

    def test_parallel_result_order_matches_submission(self):
        """Results are keyed in submission order, regardless of completion order."""
        dispatcher = ParallelDispatcher(max_workers=4, enabled=True)

        # Items with varying sleep to finish in different order
        def slow():
            time.sleep(0.1)
            return "slow"

        def fast():
            return "fast"

        items = [
            WorkItem(key="slow", callable=slow),
            WorkItem(key="fast", callable=fast),
        ]

        results = dispatcher.run(items)

        # Keys should be in submission order
        assert list(results.keys()) == ["slow", "fast"]
        assert results["slow"].result == "slow"
        assert results["fast"].result == "fast"

    def test_parallel_failure_isolation(self):
        """One item raising an exception doesn't cancel others."""
        dispatcher = ParallelDispatcher(max_workers=4, enabled=True)

        def succeed():
            time.sleep(0.05)
            return "ok"

        def fail():
            raise RuntimeError("task failed")

        items = [
            WorkItem(key="good1", callable=succeed),
            WorkItem(key="bad", callable=fail),
            WorkItem(key="good2", callable=succeed),
        ]

        results = dispatcher.run(items)

        assert results["good1"].result == "ok"
        assert results["good1"].error is None
        assert results["bad"].error is not None
        assert isinstance(results["bad"].error, RuntimeError)
        assert results["good2"].result == "ok"
        assert results["good2"].error is None

    def test_parallel_preserves_context(self):
        """Context dict passes through unchanged in parallel mode."""
        dispatcher = ParallelDispatcher(max_workers=2, enabled=True)

        items = [
            WorkItem(key="A", callable=lambda: 1, context={"role": "challenger"}),
            WorkItem(key="B", callable=lambda: 2, context={"role": "target"}),
        ]

        results = dispatcher.run(items)
        assert results["A"].context == {"role": "challenger"}
        assert results["B"].context == {"role": "target"}

    def test_parallel_actually_concurrent(self):
        """Multiple items run concurrently (total time < sum of individual times)."""
        dispatcher = ParallelDispatcher(max_workers=4, enabled=True)

        def slow_task():
            time.sleep(0.1)
            return "done"

        items = [WorkItem(key=f"t-{i}", callable=slow_task) for i in range(4)]

        start = time.monotonic()
        results = dispatcher.run(items)
        elapsed = time.monotonic() - start

        assert all(r.result == "done" for r in results.values())
        # 4 tasks * 0.1s each = 0.4s sequential; should be ~0.1s parallel
        assert elapsed < 0.35, f"Expected parallel execution, took {elapsed:.2f}s"


# ==============================================
# Duration Tracking Tests
# ==============================================

class TestDurationTracking:
    """Tests for WorkResult.duration_seconds."""

    def test_duration_populated_sequential(self):
        """duration_seconds is populated in sequential mode."""
        dispatcher = ParallelDispatcher(enabled=False)

        def slow():
            time.sleep(0.05)
            return "done"

        items = [WorkItem(key="A", callable=slow)]
        results = dispatcher.run(items)

        assert results["A"].duration_seconds >= 0.04

    def test_duration_populated_parallel(self):
        """duration_seconds is populated in parallel mode."""
        dispatcher = ParallelDispatcher(max_workers=2, enabled=True)

        def slow():
            time.sleep(0.05)
            return "done"

        items = [
            WorkItem(key="A", callable=slow),
            WorkItem(key="B", callable=slow),
        ]
        results = dispatcher.run(items)

        assert results["A"].duration_seconds >= 0.04
        assert results["B"].duration_seconds >= 0.04

    def test_duration_on_error(self):
        """duration_seconds is populated even when the callable raises."""
        dispatcher = ParallelDispatcher(enabled=False)

        def fail():
            time.sleep(0.02)
            raise ValueError("fail")

        items = [WorkItem(key="A", callable=fail)]
        results = dispatcher.run(items)

        assert results["A"].duration_seconds >= 0.01
        assert results["A"].error is not None


# ==============================================
# Max Workers Concurrency Tests
# ==============================================

class TestMaxWorkers:
    """Tests that max_workers caps concurrency."""

    def test_max_workers_respected(self):
        """No more than max_workers items run concurrently."""
        max_workers = 2
        dispatcher = ParallelDispatcher(max_workers=max_workers, enabled=True)

        concurrent_count = threading.Semaphore(0)
        max_concurrent = [0]
        lock = threading.Lock()

        def tracked_task():
            concurrent_count.release()
            # Small sleep to allow overlap detection
            time.sleep(0.05)
            # Count active concurrent tasks via a counter
            return "done"

        # Use a barrier-style approach with active count
        active = [0]

        def counted_task():
            with lock:
                active[0] += 1
                if active[0] > max_concurrent[0]:
                    max_concurrent[0] = active[0]
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return "done"

        items = [WorkItem(key=f"t-{i}", callable=counted_task) for i in range(6)]

        results = dispatcher.run(items)

        assert max_concurrent[0] <= max_workers, (
            f"Max concurrent was {max_concurrent[0]}, expected <= {max_workers}"
        )
        assert all(r.result == "done" for r in results.values())


# ==============================================
# Edge Cases
# ==============================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_work_list(self):
        """Empty list returns empty OrderedDict."""
        dispatcher = ParallelDispatcher(enabled=True)
        results = dispatcher.run([])

        assert isinstance(results, OrderedDict)
        assert len(results) == 0

    def test_single_item_sequential(self):
        """Single-item list works in sequential mode."""
        dispatcher = ParallelDispatcher(enabled=False)
        items = [WorkItem(key="only", callable=lambda: 42)]

        results = dispatcher.run(items)
        assert results["only"].result == 42

    def test_single_item_parallel(self):
        """Single-item list works in parallel mode (falls back to sequential)."""
        dispatcher = ParallelDispatcher(enabled=True)
        items = [WorkItem(key="only", callable=lambda: 42)]

        results = dispatcher.run(items)
        assert results["only"].result == 42

    def test_work_result_defaults(self):
        """WorkResult dataclass defaults are correct."""
        wr = WorkResult(key="test")
        assert wr.result is None
        assert wr.error is None
        assert wr.duration_seconds == 0.0
        assert wr.context == {}

    def test_work_item_default_context(self):
        """WorkItem default context is empty dict."""
        wi = WorkItem(key="test", callable=lambda: None)
        assert wi.context == {}
