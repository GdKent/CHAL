"""
Unit tests for training data recording and export.

Tests cover:
- DebateRecorder initialization with config/agents/topic
- Each record_* method appends correct event type to timeline
- export_jsonl() produces valid JSONL
- export_belief_training_pairs() extracts formation and update pairs
- Belief pairs contain correct input/target structure
- Metadata fields populated correctly
- Edge cases (no updates, single round, many rounds)
- Raw response fields captured for all event types
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock
from chal.utilities.training_data import DebateRecorder
from tests.utils import create_sample_belief


# ========================================
# Helpers
# ========================================

def _make_mock_config(mode="rebuttal", max_rounds=2):
    """Create a mock config for recorder tests."""
    config = Mock()
    config.stage3_mode = mode
    config.max_rounds = max_rounds
    config.adjudication = Mock()
    config.adjudication.model = "gpt-4o"
    config.adjudication.provider = "openai"

    return config


def _make_mock_agent(name, model="gpt-4o", provider="openai", persona="EMPIRICIST"):
    """Create a mock agent for recorder tests."""
    agent = Mock()
    agent.name = name
    agent.model = model
    agent.provider = provider
    agent.persona_label = persona
    agent.temperature = 0.7
    return agent


def _make_recorder(mode="rebuttal", agents=None):
    """Create a DebateRecorder with mock config and agents."""
    config = _make_mock_config(mode=mode)
    if agents is None:
        agents = [
            _make_mock_agent("Agent-A"),
            _make_mock_agent("Agent-B", model="claude-sonnet-4-5-20250929", provider="anthropic", persona="RATIONALIST"),
        ]
    return DebateRecorder(config, agents, "Does free will exist?")


# ============================================================
# 1. Initialization Tests
# ============================================================

class TestDebateRecorderInit:
    """Tests for DebateRecorder initialization."""

    @pytest.mark.unit
    def test_has_debate_id(self):
        recorder = _make_recorder()
        assert recorder.debate_id is not None
        assert len(recorder.debate_id) > 0

    @pytest.mark.unit
    def test_topic_stored(self):
        recorder = _make_recorder()
        assert recorder.topic == "Does free will exist?"

    @pytest.mark.unit
    def test_timeline_starts_empty(self):
        recorder = _make_recorder()
        assert recorder.timeline == []

    @pytest.mark.unit
    def test_metadata_populated(self):
        recorder = _make_recorder()
        meta = recorder.metadata

        assert meta["topic"] == "Does free will exist?"
        assert meta["num_agents"] == 2
        assert len(meta["agents"]) == 2

    @pytest.mark.unit
    def test_agent_metadata_fields(self):
        recorder = _make_recorder()
        agent_a = recorder.metadata["agents"][0]

        assert agent_a["agent_id"] == "Agent-A"
        assert agent_a["model"] == "gpt-4o"
        assert agent_a["provider"] == "openai"
        assert agent_a["persona"] == "EMPIRICIST"

    @pytest.mark.unit
    def test_adjudicator_metadata(self):
        recorder = _make_recorder()
        adj = recorder.metadata["adjudicator"]

        assert adj["model"] == "gpt-4o"
        assert adj["provider"] == "openai"

    @pytest.mark.unit
    def test_config_snapshot(self):
        recorder = _make_recorder()
        snapshot = recorder.metadata["config_snapshot"]

        assert "stage3_mode" in snapshot
        assert "max_rounds" in snapshot


# ============================================================
# 2. Record Methods Tests
# ============================================================

class TestRecordMethods:
    """Tests for each record_* method."""

    @pytest.mark.unit
    def test_record_belief_formation(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation(
            agent_id="Agent-A",
            inputs={"topic": "Free will", "persona": "EMPIRICIST"},
            belief=belief,
            raw_response="raw text",
        )

        assert len(recorder.timeline) == 1
        event = recorder.timeline[0]
        assert event["type"] == "belief_formation"
        assert event["stage"] == 1
        assert event["agent_id"] == "Agent-A"
        assert event["outputs"]["belief"] == belief
        assert event["outputs"]["raw_response"] == "raw text"

    @pytest.mark.unit
    def test_record_cross_examination(self):
        recorder = _make_recorder()
        recorder.record_cross_examination(
            agent_id="Agent-A",
            target_id="Agent-B",
            inputs={"own_belief": {}, "opponent_belief": {}},
            challenges=["Q1", "Q2"],
            raw_response="raw",
        )

        assert len(recorder.timeline) == 1
        event = recorder.timeline[0]
        assert event["type"] == "cross_examination"
        assert event["stage"] == 2
        assert event["target_agent_id"] == "Agent-B"
        assert event["outputs"]["challenges"] == ["Q1", "Q2"]

    @pytest.mark.unit
    def test_record_rebuttal(self):
        recorder = _make_recorder()
        recorder.record_rebuttal(
            agent_id="Agent-B",
            challenger_id="Agent-A",
            inputs={"own_belief": {}},
            rebuttals=["R1"],
            raw_response="raw",
        )

        event = recorder.timeline[0]
        assert event["type"] == "rebuttal"
        assert event["stage"] == 3
        assert event["challenger_agent_id"] == "Agent-A"

    @pytest.mark.unit
    def test_record_adjudication(self):
        recorder = _make_recorder()
        recorder.record_adjudication(
            challenger_id="Agent-A",
            target_id="Agent-B",
            inputs={"challenge": "Q", "rebuttal": "R"},
            verdict="critique_valid",
            reasoning="Valid point",
            raw_response="raw",
        )

        event = recorder.timeline[0]
        assert event["type"] == "adjudication"
        assert event["stage"] == 4
        assert event["outputs"]["verdict"] == "critique_valid"

    @pytest.mark.unit
    def test_record_belief_update(self):
        recorder = _make_recorder()
        before = create_sample_belief(confidence=0.8)
        after = create_sample_belief(confidence=0.6)
        recorder.record_belief_update(
            agent_id="Agent-A",
            belief_before=before,
            belief_after=after,
            adjudication_results=[{"status": "critique_valid"}],
            patches=[{"op": "update_claim", "target_id": "C1"}],
            raw_response="raw",
        )

        event = recorder.timeline[0]
        assert event["type"] == "belief_update"
        assert event["stage"] == 5
        assert event["inputs"]["belief_before"] == before
        assert event["outputs"]["belief_after"] == after
        assert len(event["outputs"]["patches_applied"]) == 1

    @pytest.mark.unit
    def test_record_concluding_remarks(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_concluding_remarks(
            agent_id="Agent-A",
            final_belief=belief,
            remarks="My conclusion",
            raw_response="raw",
        )

        event = recorder.timeline[0]
        assert event["type"] == "concluding_remarks"
        assert event["stage"] == 6
        assert event["round"] is None
        assert event["outputs"]["remarks"] == "My conclusion"

    @pytest.mark.unit
    def test_set_round(self):
        recorder = _make_recorder()
        recorder.set_round(2)
        recorder.record_belief_formation(
            agent_id="Agent-A", inputs={}, belief={}, raw_response=""
        )

        assert recorder.timeline[0]["round"] == 2

    @pytest.mark.unit
    def test_multiple_events_accumulate(self):
        recorder = _make_recorder()
        for i in range(5):
            recorder.record_belief_formation(
                agent_id=f"Agent-{i}", inputs={}, belief={}, raw_response=""
            )
        assert len(recorder.timeline) == 5


# ============================================================
# 3. Export Tests
# ============================================================

class TestExportJsonl:
    """Tests for export_jsonl()."""

    @pytest.mark.unit
    def test_creates_file(self, tmp_path):
        recorder = _make_recorder()
        recorder.record_belief_formation("A", {}, {}, "raw")
        output = tmp_path / "test.jsonl"

        recorder.export_jsonl(output)

        assert output.exists()

    @pytest.mark.unit
    def test_produces_valid_jsonl(self, tmp_path):
        recorder = _make_recorder()
        recorder.record_belief_formation("A", {}, {}, "raw")
        output = tmp_path / "test.jsonl"

        recorder.export_jsonl(output)

        with open(output, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "debate_id" in record
        assert "metadata" in record
        assert "timeline" in record

    @pytest.mark.unit
    def test_appends_to_existing_file(self, tmp_path):
        output = tmp_path / "test.jsonl"

        recorder1 = _make_recorder()
        recorder1.record_belief_formation("A", {}, {}, "raw")
        recorder1.export_jsonl(output)

        recorder2 = _make_recorder()
        recorder2.record_belief_formation("B", {}, {}, "raw")
        recorder2.export_jsonl(output)

        with open(output, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

    @pytest.mark.unit
    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "nested" / "dir" / "test.jsonl"

        recorder = _make_recorder()
        recorder.export_jsonl(output)

        assert output.exists()

    @pytest.mark.unit
    def test_timeline_events_in_output(self, tmp_path):
        recorder = _make_recorder()
        recorder.record_belief_formation("Agent-A", {"topic": "FW"}, {"thesis": "T"}, "raw")
        recorder.record_adjudication("A", "B", {}, "rebuttal_valid", "Good", "raw")
        output = tmp_path / "test.jsonl"

        recorder.export_jsonl(output)

        with open(output, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert len(record["timeline"]) == 2
        assert record["timeline"][0]["type"] == "belief_formation"
        assert record["timeline"][1]["type"] == "adjudication"


# ============================================================
# 4. Belief Training Pairs Tests
# ============================================================

class TestExportBeliefTrainingPairs:
    """Tests for export_belief_training_pairs() and _extract_belief_pairs()."""

    @pytest.mark.unit
    def test_extracts_formation_pair(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation(
            agent_id="Agent-A",
            inputs={"topic": "Free will", "persona": "EMPIRICIST", "system_prompt": "sys"},
            belief=belief,
            raw_response="raw",
        )

        pairs = recorder._extract_belief_pairs()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["pair_type"] == "belief_formation"
        assert pair["input"]["topic"] == "Free will"
        assert pair["input"]["persona"] == "EMPIRICIST"
        assert pair["target"] == belief
        assert pair["metadata"]["agent_id"] == "Agent-A"
        assert pair["metadata"]["debate_id"] == recorder.debate_id

    @pytest.mark.unit
    def test_extracts_update_pair(self):
        recorder = _make_recorder()
        before = create_sample_belief(confidence=0.8)
        after = create_sample_belief(confidence=0.6)
        recorder.set_round(2)
        recorder.record_belief_update(
            agent_id="Agent-A",
            belief_before=before,
            belief_after=after,
            adjudication_results=[{"status": "critique_valid"}],
            patches=[{"op": "update_claim"}],
            raw_response="raw",
        )

        pairs = recorder._extract_belief_pairs()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["pair_type"] == "belief_update"
        assert pair["input"]["belief_before"] == before
        assert pair["target"] == after
        assert pair["metadata"]["round"] == 2
        assert pair["metadata"]["patches_applied"] == [{"op": "update_claim"}]

    @pytest.mark.unit
    def test_extracts_both_pair_types(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation("Agent-A", {"topic": "T"}, belief, "raw")
        recorder.record_belief_update("Agent-A", belief, belief, [], [], "raw")

        pairs = recorder._extract_belief_pairs()

        types = {p["pair_type"] for p in pairs}
        assert types == {"belief_formation", "belief_update"}

    @pytest.mark.unit
    def test_skips_events_with_none_belief(self):
        recorder = _make_recorder()
        recorder.record_belief_formation("Agent-A", {}, None, "raw")

        pairs = recorder._extract_belief_pairs()
        assert len(pairs) == 0

    @pytest.mark.unit
    def test_skips_update_with_none_before_or_after(self):
        recorder = _make_recorder()
        recorder.record_belief_update("Agent-A", None, {}, [], [], "raw")
        recorder.record_belief_update("Agent-A", {}, None, [], [], "raw")

        pairs = recorder._extract_belief_pairs()
        assert len(pairs) == 0

    @pytest.mark.unit
    def test_agent_model_in_metadata(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation("Agent-A", {"topic": "T"}, belief, "raw")

        pairs = recorder._extract_belief_pairs()
        assert pairs[0]["metadata"]["agent_model"] == "gpt-4o"

    @pytest.mark.unit
    def test_agent_model_for_second_agent(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation("Agent-B", {"topic": "T"}, belief, "raw")

        pairs = recorder._extract_belief_pairs()
        assert pairs[0]["metadata"]["agent_model"] == "claude-sonnet-4-5-20250929"

    @pytest.mark.unit
    def test_debate_context_in_update_pair(self):
        recorder = _make_recorder()
        before = create_sample_belief()
        after = create_sample_belief()
        recorder.set_round(3)
        recorder.record_belief_update("Agent-A", before, after, [], [], "raw")

        pairs = recorder._extract_belief_pairs()
        context = pairs[0]["input"]["debate_context"]

        assert context["topic"] == "Does free will exist?"
        assert context["round"] == 3
        assert context["mode"] == "rebuttal"

    @pytest.mark.unit
    def test_export_writes_pairs_to_file(self, tmp_path):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation("Agent-A", {"topic": "T"}, belief, "raw")
        output = tmp_path / "pairs.jsonl"

        recorder.export_belief_training_pairs(output)

        assert output.exists()
        with open(output, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1

        pair = json.loads(lines[0])
        assert pair["pair_type"] == "belief_formation"

    @pytest.mark.unit
    def test_export_multiple_pairs(self, tmp_path):
        recorder = _make_recorder()
        belief = create_sample_belief()
        for i in range(5):
            recorder.set_round(i + 1)
            recorder.record_belief_formation(f"Agent-A", {"topic": "T"}, belief, "raw")

        output = tmp_path / "pairs.jsonl"
        recorder.export_belief_training_pairs(output)

        with open(output, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 5


# ============================================================
# 5. get_debate_record() Tests
# ============================================================

class TestGetDebateRecord:
    """Tests for get_debate_record()."""

    @pytest.mark.unit
    def test_returns_dict(self):
        recorder = _make_recorder()
        record = recorder.get_debate_record()
        assert isinstance(record, dict)

    @pytest.mark.unit
    def test_has_required_keys(self):
        recorder = _make_recorder()
        record = recorder.get_debate_record()

        assert "debate_id" in record
        assert "metadata" in record
        assert "timeline" in record

    @pytest.mark.unit
    def test_is_json_serializable(self):
        recorder = _make_recorder()
        belief = create_sample_belief()
        recorder.record_belief_formation("Agent-A", {"topic": "T"}, belief, "raw")
        recorder.record_adjudication("A", "B", {}, "rebuttal_valid", "Good", "raw")

        record = recorder.get_debate_record()
        serialized = json.dumps(record)
        assert len(serialized) > 0

    @pytest.mark.unit
    def test_timeline_reflects_recorded_events(self):
        recorder = _make_recorder()
        recorder.record_belief_formation("A", {}, {}, "raw")
        recorder.record_cross_examination("A", "B", {}, [], "raw")
        recorder.record_adjudication("A", "B", {}, "critique_valid", "reason", "raw")

        record = recorder.get_debate_record()
        assert len(record["timeline"]) == 3
        types = [e["type"] for e in record["timeline"]]
        assert types == ["belief_formation", "cross_examination", "adjudication"]
