"""
Unit tests for the CHAL debate history module (history.py).

Tests cover:
- log_debate: ID generation, file creation, config snapshots, field validation,
  appending entries, and winner determination
- list_debates: empty-file and populated scenarios
- load_debate_config: loading from snapshot and missing-file error
- format_history_table: Rich table rendering with entries and empty state
"""

import json
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console

from chal.cli.history import (
    log_debate,
    list_debates,
    load_debate_config,
    format_history_table,
)
from chal.config import DebateConfig, AgentConfig, AdjudicationConfig, OutputConfig


# ========================================
# Helpers
# ========================================

def _make_config(tmp_path: Path) -> DebateConfig:
    """Return a minimal DebateConfig pointing at *tmp_path* for outputs."""
    return DebateConfig(
        topic="Test topic",
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        outputs=OutputConfig(storage_dir=tmp_path),
    )


def _make_results() -> dict:
    """Return a minimal results dict with agent stats and convergence."""
    return {
        "agent_stats": {
            "Agent-A": {"performance_score": 0.55},
            "Agent-B": {"performance_score": 0.35},
        },
        "convergence_history": [{"convergence_score": 0.72}],
        "synthesis": "",
        "debug_log": "",
        "initial_positions": [],
        "final_positions": [],
    }


def _patch_paths(monkeypatch, tmp_path: Path):
    """Redirect HISTORY_DIR and HISTORY_FILE to tmp_path."""
    monkeypatch.setattr("chal.cli.history.HISTORY_DIR", tmp_path / "history")
    monkeypatch.setattr("chal.cli.history.HISTORY_FILE", tmp_path / "history.json")


# ==============================================
# 1. TestLogDebate
# ==============================================

class TestLogDebate:

    @pytest.mark.unit
    def test_returns_8_char_id(self, tmp_path, monkeypatch):
        """log_debate returns an 8-character hex debate ID."""
        _patch_paths(monkeypatch, tmp_path)
        config = _make_config(tmp_path)
        results = _make_results()

        debate_id = log_debate(config, results, duration_s=12.3)

        assert isinstance(debate_id, str)
        assert len(debate_id) == 8
        # Must be valid hexadecimal
        int(debate_id, 16)

    @pytest.mark.unit
    def test_creates_history_file(self, tmp_path, monkeypatch):
        """log_debate creates the HISTORY_FILE on disk."""
        _patch_paths(monkeypatch, tmp_path)
        history_file = tmp_path / "history.json"
        config = _make_config(tmp_path)
        results = _make_results()

        log_debate(config, results)

        assert history_file.exists()
        data = json.loads(history_file.read_text(encoding="utf-8"))
        assert "debates" in data
        assert len(data["debates"]) == 1

    @pytest.mark.unit
    def test_saves_config_snapshot(self, tmp_path, monkeypatch):
        """log_debate saves a YAML config snapshot as HISTORY_DIR/{id}.yaml."""
        _patch_paths(monkeypatch, tmp_path)
        history_dir = tmp_path / "history"
        config = _make_config(tmp_path)
        results = _make_results()

        debate_id = log_debate(config, results)

        snapshot = history_dir / f"{debate_id}.yaml"
        assert snapshot.exists()
        # Snapshot should be non-empty YAML
        assert snapshot.stat().st_size > 0

    @pytest.mark.unit
    def test_entry_has_required_fields(self, tmp_path, monkeypatch):
        """The history entry contains all required fields."""
        _patch_paths(monkeypatch, tmp_path)
        history_file = tmp_path / "history.json"
        config = _make_config(tmp_path)
        results = _make_results()

        debate_id = log_debate(config, results, duration_s=42.5)

        data = json.loads(history_file.read_text(encoding="utf-8"))
        entry = data["debates"][0]

        required_keys = {
            "id",
            "timestamp",
            "topic",
            "agents",
            "rounds",
            "duration_s",
            "convergence",
            "winner",
            "config_snapshot",
            "output_dir",
        }
        assert required_keys.issubset(entry.keys())

        # Verify specific values
        assert entry["id"] == debate_id
        assert entry["topic"] == "Test topic"
        assert entry["agents"] == ["Agent-A", "Agent-B"]
        assert entry["duration_s"] == 42.5
        assert entry["convergence"] == 0.72
        assert entry["winner"] == "Agent-A"

    @pytest.mark.unit
    def test_appends_to_existing_history(self, tmp_path, monkeypatch):
        """Calling log_debate twice produces two entries in the history file."""
        _patch_paths(monkeypatch, tmp_path)
        history_file = tmp_path / "history.json"
        config = _make_config(tmp_path)
        results = _make_results()

        id_1 = log_debate(config, results, duration_s=1.0)
        id_2 = log_debate(config, results, duration_s=2.0)

        assert id_1 != id_2

        data = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(data["debates"]) == 2
        assert data["debates"][0]["id"] == id_1
        assert data["debates"][1]["id"] == id_2

    @pytest.mark.unit
    def test_determines_winner_by_score(self, tmp_path, monkeypatch):
        """The winner field is the agent with the highest performance_score."""
        _patch_paths(monkeypatch, tmp_path)
        history_file = tmp_path / "history.json"
        config = _make_config(tmp_path)

        # Agent-B wins this time
        results = _make_results()
        results["agent_stats"] = {
            "Agent-A": {"performance_score": 0.20},
            "Agent-B": {"performance_score": 0.75},
        }

        log_debate(config, results)

        data = json.loads(history_file.read_text(encoding="utf-8"))
        assert data["debates"][0]["winner"] == "Agent-B"


# ==============================================
# 2. TestListDebates
# ==============================================

class TestListDebates:

    @pytest.mark.unit
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        """list_debates returns [] when no history file exists."""
        _patch_paths(monkeypatch, tmp_path)
        # No history.json has been created
        assert list_debates() == []

    @pytest.mark.unit
    def test_returns_logged_entries(self, tmp_path, monkeypatch):
        """list_debates returns previously logged entries."""
        _patch_paths(monkeypatch, tmp_path)
        config = _make_config(tmp_path)
        results = _make_results()

        id_1 = log_debate(config, results, duration_s=10.0)
        id_2 = log_debate(config, results, duration_s=20.0)

        debates = list_debates()
        assert len(debates) == 2
        assert debates[0]["id"] == id_1
        assert debates[1]["id"] == id_2
        assert debates[0]["topic"] == "Test topic"


# ==============================================
# 3. TestLoadDebateConfig
# ==============================================

class TestLoadDebateConfig:

    @pytest.mark.unit
    def test_loads_config_from_snapshot(self, tmp_path, monkeypatch):
        """load_debate_config reloads the DebateConfig from the saved YAML snapshot."""
        _patch_paths(monkeypatch, tmp_path)
        config = _make_config(tmp_path)
        results = _make_results()

        debate_id = log_debate(config, results)

        loaded = load_debate_config(debate_id)

        assert isinstance(loaded, DebateConfig)
        assert loaded.topic == "Test topic"
        assert len(loaded.agents) == 2
        assert loaded.agents[0].name == "Agent-A"
        assert loaded.agents[1].name == "Agent-B"
        assert loaded.agents[0].persona == "EMPIRICIST"

    @pytest.mark.unit
    def test_raises_file_not_found(self, tmp_path, monkeypatch):
        """load_debate_config raises FileNotFoundError for a non-existent ID."""
        _patch_paths(monkeypatch, tmp_path)
        # Ensure the history dir exists but has no snapshot for this id
        (tmp_path / "history").mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="not found"):
            load_debate_config("deadbeef")


# ==============================================
# 4. TestFormatHistoryTable
# ==============================================

class TestFormatHistoryTable:

    @pytest.mark.unit
    def test_renders_table_with_entries(self, tmp_path, monkeypatch):
        """format_history_table prints a Rich table when given debate entries."""
        _patch_paths(monkeypatch, tmp_path)
        config = _make_config(tmp_path)
        results = _make_results()

        log_debate(config, results, duration_s=90.0)

        debates = list_debates()
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)

        format_history_table(debates, console)

        output = buf.getvalue()
        assert "Debate History" in output
        assert "Test topic" in output
        assert "Agent-A" in output  # winner column

    @pytest.mark.unit
    def test_shows_message_when_empty(self):
        """format_history_table prints an informational message for an empty list."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)

        format_history_table([], console)

        output = buf.getvalue()
        assert "No debate history found" in output
