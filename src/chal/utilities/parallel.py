"""
parallel.py

Lightweight parallel dispatch utility for the gather-then-apply pattern.

Wraps ``concurrent.futures.ThreadPoolExecutor`` to fire independent API calls
concurrently and return results in deterministic (submission) order.  When
disabled, runs items sequentially in a plain for-loop — producing identical
behavior to the pre-parallelization codebase.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class WorkItem:
    """A single unit of parallel work.

    Attributes:
        key: Unique identifier for this item (e.g. agent name, pair key).
        callable: Zero-argument callable that performs the work
                  (e.g. ``lambda: agent.generate(messages)``).
        context: Arbitrary metadata passed through to the result unchanged.
    """
    key: str
    callable: Callable[[], Any]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkResult:
    """Result of executing a single :class:`WorkItem`.

    Attributes:
        key: Matches the originating ``WorkItem.key``.
        result: Return value from the callable, or ``None`` on error.
        error: The exception if the callable raised, else ``None``.
        duration_seconds: Wall-clock time for this item.
        context: Pass-through of the originating ``WorkItem.context``.
    """
    key: str
    result: Any = None
    error: Optional[Exception] = None
    duration_seconds: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


class ParallelDispatcher:
    """Dispatches work items and returns results in deterministic order.

    When ``enabled=True``, items run concurrently in a
    :class:`~concurrent.futures.ThreadPoolExecutor`.  When ``enabled=False``,
    items run sequentially in a simple for-loop — identical to the legacy
    sequential behavior.

    Each item's exceptions are caught individually; one failure does not
    cancel or affect the others.

    Usage::

        dispatcher = ParallelDispatcher(max_workers=5, enabled=True)
        items = [
            WorkItem(key="Agent-A", callable=lambda: agent_a.generate(msg)),
            WorkItem(key="Agent-B", callable=lambda: agent_b.generate(msg)),
        ]
        results = dispatcher.run(items)
        # results["Agent-A"].result contains the response
    """

    def __init__(self, max_workers: int = 5, enabled: bool = True) -> None:
        self.max_workers = max_workers
        self.enabled = enabled

    def run(self, items: List[WorkItem]) -> OrderedDict[str, WorkResult]:
        """Execute all work items and return results in submission order.

        Args:
            items: Work items to execute.  Keys should be unique within
                   a single ``run()`` call.

        Returns:
            An :class:`~collections.OrderedDict` mapping each item's key
            to its :class:`WorkResult`, in the same order as *items*.
        """
        if not items:
            return OrderedDict()

        if self.enabled and len(items) > 1:
            actual = min(self.max_workers, len(items))
            print(f"[Parallel] Dispatching {len(items)} items across {actual} threads")
            results = self._run_parallel(items)
            total = sum(r.duration_seconds for r in results.values())
            wall = max(r.duration_seconds for r in results.values())
            print(f"[Parallel] Done — wall time ≈ {wall:.1f}s (saved ≈ {total - wall:.1f}s vs sequential)")
            return results
        return self._run_sequential(items)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_sequential(self, items: List[WorkItem]) -> OrderedDict[str, WorkResult]:
        """Run items one-by-one in a plain for-loop."""
        results: OrderedDict[str, WorkResult] = OrderedDict()
        for item in items:
            start = time.monotonic()
            try:
                value = item.callable()
                results[item.key] = WorkResult(
                    key=item.key,
                    result=value,
                    duration_seconds=time.monotonic() - start,
                    context=item.context,
                )
            except Exception as exc:
                results[item.key] = WorkResult(
                    key=item.key,
                    error=exc,
                    duration_seconds=time.monotonic() - start,
                    context=item.context,
                )
        return results

    def _run_parallel(self, items: List[WorkItem]) -> OrderedDict[str, WorkResult]:
        """Run items concurrently in a ThreadPoolExecutor."""
        results: OrderedDict[str, WorkResult] = OrderedDict()

        # Pre-allocate slots in submission order so the final OrderedDict
        # preserves that order regardless of completion order.
        futures_map: Dict[Future, WorkItem] = {}
        start_times: Dict[str, float] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for item in items:
                start_times[item.key] = time.monotonic()
                future = executor.submit(item.callable)
                futures_map[future] = item

            # Wait for all futures (executor.__exit__ blocks until done)

        # Collect results in original submission order
        # Build a lookup from futures
        future_results: Dict[str, WorkResult] = {}
        for future, item in futures_map.items():
            elapsed = time.monotonic() - start_times[item.key]
            exc = future.exception()
            if exc is not None:
                future_results[item.key] = WorkResult(
                    key=item.key,
                    error=exc,
                    duration_seconds=elapsed,
                    context=item.context,
                )
            else:
                future_results[item.key] = WorkResult(
                    key=item.key,
                    result=future.result(),
                    duration_seconds=elapsed,
                    context=item.context,
                )

        # Return in submission order
        for item in items:
            results[item.key] = future_results[item.key]

        return results
