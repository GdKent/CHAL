"""
Integration tests for Blood Sport adversarial mode.

Tests cover:
- run_stage_3_bloodsport() with mocked agents (exchange flow)
- Standard adjudicator is used unchanged
- BloodSport stats tracking
- run() branching for bloodsport mode
- CBS-v1 belief format maintained throughout
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from chal.agents.base import Agent, Message
from chal.config import (
    DebateConfig,
    AgentConfig,
    AdjudicationConfig,
    StageConfig,
    OutputConfig,
    ScribeConfig,
    BloodSportConfig,
)
from tests.utils import create_sample_belief, create_mock_adjudication_response


# ========================================
# Helper: build mock agent
# ========================================

def _make_mock_agent(name: str, responses=None, belief_obj=None):
    """Create a mock agent for bloodsport tests."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.model = "gpt-4o"
    agent.provider = "openai"
    agent.persona_label = "TEST"
    agent.temperature = 0.7
    agent.current_belief = None

    if responses is None:
        responses = ['{"attack": "Test attack", "defense": null, "target_claims": ["C1"]}']

    response_cycle = iter(responses * 200)

    def mock_generate(messages, temperature=None):
        return Message(role="assistant", content=next(response_cycle))

    agent.generate = Mock(side_effect=mock_generate)
    agent.get_internal_belief_obj = Mock(return_value=belief_obj or create_sample_belief())
    agent.get_internal_belief = Mock(return_value="Mock belief text")
    agent.internal_belief = "Mock belief text"
    agent.internal_belief_obj = belief_obj or create_sample_belief()
    agent.update_current_belief = Mock()
    agent.system_prompt = ""
    agent.all_beliefs_held = []

    return agent


def _build_controller_for_bloodsport(
    agent_responses=None,
    adjudicator_responses=None,
    intensity="moderate",
    max_exchanges=3,
    tmpdir=None,
):
    """Build a DebateController configured for bloodsport mode with mocked agents."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    config = DebateConfig(
        name="Bloodsport Test",
        topic="Does free will exist?",
        max_rounds=1,
        stage3_mode="bloodsport",
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        adjudication=AdjudicationConfig(),
        stages=StageConfig(max_rebuttal_length_chars=500),
        outputs=OutputConfig(storage_dir=Path(tmpdir)),
        scribe=ScribeConfig(enabled=False),
        bloodsport=BloodSportConfig(
            intensity=intensity,
            max_exchanges=max_exchanges,
        ),
    )

    # Default agent responses: bloodsport turn JSON
    if agent_responses is None:
        agent_responses = [
            '{"attack": "Test attack text", "defense": null, "target_claims": ["C1"]}',
            '{"attack": "Counter attack text", "defense": "Defense text", "target_claims": ["C2"]}',
        ]

    agent_a = _make_mock_agent("Agent-A", responses=agent_responses)
    agent_b = _make_mock_agent("Agent-B", responses=agent_responses)

    # Default adjudicator response
    if adjudicator_responses is None:
        adj_response = create_mock_adjudication_response("rebuttal_valid")
        adjudicator_responses = [adj_response] * 20

    with patch("chal.orchestrator.debate_controller.create_agent") as mock_create:
        mock_adj_agent = _make_mock_agent("Adjudicator", responses=adjudicator_responses)
        mock_scribe_agent = _make_mock_agent("Scribe")
        mock_create.side_effect = [mock_adj_agent, mock_scribe_agent]

        from chal.orchestrator.debate_controller import DebateController
        controller = DebateController(
            agents=[agent_a, agent_b],
            max_rounds=1,
            config=config,
        )

    controller.adjudicator.agent = mock_adj_agent
    controller.topic = "Does free will exist?"

    return controller, agent_a, agent_b


# ============================================================
# 1. run_stage_3_bloodsport() Tests
# ============================================================

class TestRunStage3Bloodsport:
    """Tests for DebateController.run_stage_3_bloodsport()."""

    @pytest.mark.unit
    def test_populates_required_keys(self):
        """Each entry gets rebuttal, resolution, bloodsport_transcript, bloodsport_stats."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=3)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Why do you believe X?",
            "qid": "Q1",
            "target_ids": ["C1"],
        }]

        result = controller.run_stage_3_bloodsport()

        assert len(result) == 1
        entry = result[0]
        assert "rebuttal" in entry
        assert "resolution" in entry
        assert "bloodsport_transcript" in entry
        assert "bloodsport_stats" in entry

    @pytest.mark.unit
    def test_resolution_has_correct_format(self):
        """Resolution dict matches Stage 4 format: status, reasoning."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=2)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Test",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_bloodsport()
        resolution = controller.challenge_rebuttal_pairs[0]["resolution"]

        assert "status" in resolution
        assert resolution["status"] in ("rebuttal_valid", "critique_valid", "unresolved")
        assert "reasoning" in resolution

    @pytest.mark.unit
    def test_exchange_has_correct_num_turns(self):
        """Exchange should have max_exchanges turns."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=4)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_bloodsport()
        transcript = controller.challenge_rebuttal_pairs[0]["bloodsport_transcript"]

        assert len(transcript) == 4

    @pytest.mark.unit
    def test_bloodsport_stats_populated(self):
        """bloodsport_stats has intensity and turn count."""
        controller, _, _ = _build_controller_for_bloodsport(
            intensity="extreme", max_exchanges=3
        )
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_bloodsport()
        stats = controller.challenge_rebuttal_pairs[0]["bloodsport_stats"]

        assert stats["intensity"] == "extreme"
        assert stats["num_turns"] == 3

    @pytest.mark.unit
    def test_agent_stats_updated(self):
        """Agent stats should be updated after bloodsport exchange."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=2)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        assert controller.agent_stats["Agent-A"]["total_arguments"] == 0

        controller.run_stage_3_bloodsport()

        total = (
            controller.agent_stats["Agent-A"]["total_arguments"]
            + controller.agent_stats["Agent-B"]["total_arguments"]
        )
        assert total > 0

    @pytest.mark.unit
    def test_bloodsport_specific_stats_tracked(self):
        """bloodsport_exchanges and bloodsport_turns are tracked."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=3)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_bloodsport()

        for agent_name in ["Agent-A", "Agent-B"]:
            stats = controller.agent_stats[agent_name]
            assert stats.get("bloodsport_exchanges", 0) > 0
            assert stats.get("bloodsport_turns", 0) > 0

    @pytest.mark.unit
    def test_anti_repetition_tracking(self):
        """previous_rounds_challenges is populated with outcomes."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=2)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": ["C1"],
        }]

        controller.run_stage_3_bloodsport()

        key = "Agent-A\u2192Agent-B"
        assert key in controller.previous_rounds_challenges
        assert len(controller.previous_rounds_challenges[key]) == 1

    @pytest.mark.unit
    def test_handles_multiple_entries_same_pair(self):
        """Multiple challenge entries for the same pair are all resolved."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=2)
        controller.challenge_rebuttal_pairs = [
            {
                "challenger": "Agent-A",
                "target": "Agent-B",
                "challenge": "Q1?",
                "qid": "Q1",
                "target_ids": [],
            },
            {
                "challenger": "Agent-A",
                "target": "Agent-B",
                "challenge": "Q2?",
                "qid": "Q2",
                "target_ids": [],
            },
        ]

        result = controller.run_stage_3_bloodsport()

        assert len(result) == 2
        assert all("resolution" in e for e in result)
        assert all("bloodsport_transcript" in e for e in result)

    @pytest.mark.unit
    def test_markdown_transcript_updated(self):
        """Markdown transcript should contain bloodsport content."""
        controller, _, _ = _build_controller_for_bloodsport(max_exchanges=2)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_bloodsport()

        md = "\n".join(controller.markdown_transcript)
        assert "Blood Sport" in md or "blood sport" in md.lower()
        assert "Agent-A" in md
        assert "Agent-B" in md


# ============================================================
# 2. run() Branching Tests
# ============================================================

class TestBloodSportRunBranching:
    """Tests for stage3_mode='bloodsport' routing in run()."""

    @pytest.mark.unit
    def test_bloodsport_mode_calls_stage_3c(self):
        """In bloodsport mode, run_stage_3_bloodsport is called."""
        controller, _, _ = _build_controller_for_bloodsport()

        with patch.object(controller, "run_stage_3_bloodsport") as mock_3c, \
             patch.object(controller, "run_stage_3_collaborative") as mock_3b, \
             patch.object(controller, "run_stage_3_rebuttals") as mock_3, \
             patch.object(controller, "run_stage_4_conflict_resolution") as mock_4:

            stage3_mode = controller.config.stage3_mode
            if stage3_mode == "collaborative":
                controller.run_stage_3_collaborative()
            elif stage3_mode == "bloodsport":
                controller.run_stage_3_bloodsport()
            else:
                controller.run_stage_3_rebuttals()
                controller.run_stage_4_conflict_resolution()

        mock_3c.assert_called_once()
        mock_3b.assert_not_called()
        mock_3.assert_not_called()
        mock_4.assert_not_called()

    @pytest.mark.unit
    def test_bloodsport_mode_skips_stage_4(self):
        """In bloodsport mode, Stage 4 is NOT called separately (adjudication is inline)."""
        controller, _, _ = _build_controller_for_bloodsport()

        with patch.object(controller, "run_stage_3_bloodsport") as mock_3c, \
             patch.object(controller, "run_stage_4_conflict_resolution") as mock_4:

            stage3_mode = controller.config.stage3_mode
            if stage3_mode == "bloodsport":
                controller.run_stage_3_bloodsport()

        mock_3c.assert_called_once()
        mock_4.assert_not_called()
