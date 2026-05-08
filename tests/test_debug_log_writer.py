"""Tests for the DebugLogWriter real-time logging infrastructure."""

from __future__ import annotations

import logging
import threading

import pytest

from chal.utilities.debug_log_writer import DebugLogHandler, DebugLogWriter


# ==============================================
# DebugLogWriter
# ==============================================


@pytest.mark.unit
class TestDebugLogWriterMemoryOnly:
    """Memory-only mode (file_path=None)."""

    def test_write_and_get_contents(self):
        w = DebugLogWriter()
        w.write("line 1")
        w.write("line 2")
        assert w.get_contents() == "line 1\nline 2"

    def test_len(self):
        w = DebugLogWriter()
        assert len(w) == 0
        w.write("a")
        w.write("b")
        assert len(w) == 2

    def test_close_is_idempotent(self):
        w = DebugLogWriter()
        w.write("x")
        w.close()
        w.close()  # no error
        assert w.get_contents() == "x"

    def test_context_manager(self):
        with DebugLogWriter() as w:
            w.write("hello")
        assert w.get_contents() == "hello"


@pytest.mark.unit
class TestDebugLogWriterFileMode:
    """File-backed mode — verifies real-time streaming."""

    def test_writes_to_file_immediately(self, tmp_path):
        path = tmp_path / "log.txt"
        w = DebugLogWriter(file_path=path)
        w.write("first")
        # File should already contain the line (flushed)
        assert path.read_text(encoding="utf-8") == "first\n"

        w.write("second")
        assert path.read_text(encoding="utf-8") == "first\nsecond\n"
        w.close()

    def test_file_survives_without_close(self, tmp_path):
        """File is readable even if close() was never called (crash scenario)."""
        path = tmp_path / "log.txt"
        w = DebugLogWriter(file_path=path)
        w.write("crash line")
        # Don't close — simulate crash
        content = path.read_text(encoding="utf-8")
        assert "crash line" in content
        w.close()  # cleanup

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "log.txt"
        w = DebugLogWriter(file_path=path)
        w.write("ok")
        w.close()
        assert path.exists()

    def test_get_contents_matches_file(self, tmp_path):
        path = tmp_path / "log.txt"
        w = DebugLogWriter(file_path=path)
        w.write("a")
        w.write("b")
        w.close()
        # Memory contents use \n joiner; file uses trailing \n per line
        assert w.get_contents() == "a\nb"
        assert path.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.unit
class TestDebugLogWriterThreadSafety:
    """Concurrent writes from multiple threads."""

    def test_concurrent_writes(self, tmp_path):
        path = tmp_path / "log.txt"
        w = DebugLogWriter(file_path=path)
        n_threads = 10
        n_writes = 50

        def worker(thread_id):
            for i in range(n_writes):
                w.write(f"thread-{thread_id}-line-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        w.close()

        assert len(w) == n_threads * n_writes
        file_lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(file_lines) == n_threads * n_writes


# ==============================================
# DebugLogHandler
# ==============================================


@pytest.mark.unit
class TestDebugLogHandler:
    """Bridge from Python logging to DebugLogWriter."""

    def test_handler_forwards_records(self):
        w = DebugLogWriter()
        handler = DebugLogHandler(w)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        test_logger = logging.getLogger("test_debug_log_handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.info("hello world")

        test_logger.removeHandler(handler)

        contents = w.get_contents()
        assert "INFO: hello world" in contents

    def test_handler_captures_debug_level(self):
        w = DebugLogWriter()
        handler = DebugLogHandler(w)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.DEBUG)

        test_logger = logging.getLogger("test_debug_handler_debug")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.debug("debug msg")

        test_logger.removeHandler(handler)

        assert "debug msg" in w.get_contents()
