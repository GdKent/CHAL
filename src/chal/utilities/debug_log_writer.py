"""
debug_log_writer.py

Thread-safe debug log that streams entries to disk in real time.

Replaces the in-memory ``list[str]`` used by DebateController so that
``log.txt`` is written incrementally — even if the debate crashes,
everything up to the failure is on disk.

When ``file_path`` is None the writer operates in memory-only mode,
which preserves backward compatibility for tests and programmatic use.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path


class DebugLogWriter:
    """Thread-safe debug log writer with real-time file streaming.

    Every call to :meth:`write` atomically appends the line to an
    in-memory list **and** writes + flushes it to the backing file (if
    one was supplied).  A ``threading.Lock`` serialises concurrent
    writes from the ``ParallelDispatcher``'s worker threads.

    Supports context-manager protocol for automatic cleanup.
    """

    def __init__(self, file_path: Path | None = None) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._file_handle = None
        if file_path is not None:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(file_path, "w", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def write(self, line: str) -> None:
        """Append *line* to the log (memory + file).

        Each line is written with a trailing newline to the file.
        """
        with self._lock:
            self._lines.append(line)
            if self._file_handle is not None:
                self._file_handle.write(line + "\n")
                self._file_handle.flush()

    def get_contents(self) -> str:
        """Return the full log joined by newlines."""
        with self._lock:
            return "\n".join(self._lines)

    def close(self) -> None:
        """Flush and close the backing file (idempotent)."""
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.flush()
                    self._file_handle.close()
                except OSError:
                    pass
                self._file_handle = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of lines written so far (used by final logging)."""
        with self._lock:
            return len(self._lines)

    def __enter__(self) -> DebugLogWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class DebugLogHandler(logging.Handler):
    """Bridges Python ``logging`` records into a :class:`DebugLogWriter`.

    Attach this handler to the ``chal`` logger so that low-level retry
    messages, agent API errors, and any other ``logger.*()`` calls are
    captured in the same real-time log file.
    """

    def __init__(self, writer: DebugLogWriter) -> None:
        super().__init__()
        self.writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.writer.write(self.format(record))
        except Exception:
            self.handleError(record)
