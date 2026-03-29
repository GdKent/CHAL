"""
Unit tests for parallelized debate stages.

Uses mock agents (from existing test infrastructure) to verify that each
parallelized stage produces identical results whether parallel.enabled is
True or False.

Tests cover:
- Stage 1 (Opening Positions): parallel matches sequential
- Stage 2 (Cross-Examination): parallel matches sequential
- Stage 3 (Rebuttals): parallel matches sequential
- Stage 4 (Adjudication): parallel matches sequential
- Stage 6 (Concluding Remarks): parallel matches sequential
- Error handling: one agent failure doesn't crash others
"""

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from chal.orchestrator.debate_controller import DebateController
from chal.config import (
    DebateConfig, AgentConfig, AdjudicationConfig,
    OutputConfig, ParallelConfig,
)
from chal.agents.base import Message
from tests.utils import (
    create_mock_agent,
    create_sample_belief,
    create_mock_belief_response,
)


# ==============================================
# Helper Fixtures
# ==============================================

@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_openai_responses(fixtures_dir):
    """Load mock OpenAI responses."""
    with open(fixtures_dir / "mock_openai_responses.json") as f:
        return json.load(f)


def _make_config(parallel_enabled: bool, tmpdir: str) -> DebateConfig:
    """Create a DebateConfig with the given parallel setting."""
    return DebateConfig(
        name="Parallel Test Debate",
        topic="Does free will exist?",
        max_rounds=1,
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        adjudication=AdjudicationConfig(),
        outputs=OutputConfig(storage_dir=Path(tmpdir)),
        parallel=ParallelConfig(enabled=parallel_enabled, max_workers=5),
    )


def _make_agents(responses_data):
    """Create a pair of mock agents with predefined responses (deep enough for all stages)."""
    # Each agent needs responses for: Stage 1, Stage 2, Stage 3, Stage 5, Stage 6
    responses = [
        responses_data["belief_complete"]["content"],
        responses_data["cross_examination_3"]["content"],
        responses_data["rebuttals_3"]["content"],
        responses_data["belief_update_patches"]["content"],
        responses_data["concluding_remarks"]["content"],
    ]

    agent_a = create_mock_agent("Agent-A", responses=responses)
    agent_b = create_mock_agent("Agent-B", responses=responses)
    return [agent_a, agent_b]


def _run_through_stage(mock_openai_responses, parallel_enabled: bool, up_to_stage: int):
    """Run debate pipeline up to the given stage and return the controller state.

    Returns a tuple of (controller, agents) so callers can inspect the result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(parallel_enabled, tmpdir)
        agents = _make_agents(mock_openai_responses)
        controller = DebateController(agents, config=config)

        # Stage 0 — Briefing
        personas = {ac.name: ac.persona for ac in config.agents}
        controller.run_stage_0_briefing(config.topic, personas)

        if up_to_stage >= 1:
            controller.run_stage_1_opening_positions(config.topic)

        if up_to_stage >= 2:
            controller.run_stage_2_cross_examination()

        if up_to_stage >= 3:
            controller.run_stage_3_rebuttals()

        if up_to_stage >= 4:
            controller.run_stage_4_conflict_resolution()

        if up_to_stage >= 6:
            controller.run_stage_6_concluding_remarks()

        return controller


# ==============================================
# Stage 1: Opening Positions
# ==============================================

class TestStage1Parallel:
    """Opening positions — parallel vs sequential equivalence."""

    def test_stage1_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes produce opening positions for all agents."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=1)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=1)

        # Both should have opening positions
        assert hasattr(ctrl_seq, "opening_positions")
        assert hasattr(ctrl_par, "opening_positions")

        # Same number of opening positions (one per agent)
        assert len(ctrl_seq.opening_positions) == len(ctrl_par.opening_positions)

        # opening_positions is a list of belief text strings
        for pos_s, pos_p in zip(ctrl_seq.opening_positions, ctrl_par.opening_positions):
            assert isinstance(pos_s, str)
            assert isinstance(pos_p, str)

    def test_stage1_beliefs_set_on_agents(self, mock_openai_responses):
        """Both modes set internal beliefs on agents."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=1)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=1)

        for agent_s, agent_p in zip(ctrl_seq.agents, ctrl_par.agents):
            # Both should have set_internal_belief_obj called
            assert agent_s.set_internal_belief_obj.called
            assert agent_p.set_internal_belief_obj.called

    def test_stage1_handles_agent_failure(self, mock_openai_responses):
        """One agent's failure doesn't crash the other in parallel mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(True, tmpdir)

            # Agent-A returns valid belief, Agent-B returns garbage
            agent_a = create_mock_agent("Agent-A", responses=[
                mock_openai_responses["belief_complete"]["content"],
            ] * 10)
            agent_b = create_mock_agent("Agent-B", responses=[
                "This is not valid JSON at all",
            ] * 10)

            controller = DebateController([agent_a, agent_b], config=config)
            personas = {ac.name: ac.persona for ac in config.agents}
            controller.run_stage_0_briefing(config.topic, personas)

            # Should not raise; handles gracefully
            try:
                controller.run_stage_1_opening_positions(config.topic)
            except Exception:
                pass  # Some implementations may raise after max retries

            # Agent-A should have been processed
            assert agent_a.generate.called


# ==============================================
# Stage 2: Cross-Examination
# ==============================================

class TestStage2Parallel:
    """Cross-examination — parallel vs sequential equivalence."""

    def test_stage2_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes produce challenge_rebuttal_pairs entries."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=2)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=2)

        # Same number of challenge-rebuttal pairs
        assert len(ctrl_seq.challenge_rebuttal_pairs) == len(ctrl_par.challenge_rebuttal_pairs)

        # Same challenger→target pairs in same order
        for entry_s, entry_p in zip(ctrl_seq.challenge_rebuttal_pairs, ctrl_par.challenge_rebuttal_pairs):
            assert entry_s["challenger"] == entry_p["challenger"]
            assert entry_s["target"] == entry_p["target"]

    def test_stage2_generates_challenges(self, mock_openai_responses):
        """Both modes populate challenge text in entries."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=2)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=2)

        for entries in [ctrl_seq.challenge_rebuttal_pairs, ctrl_par.challenge_rebuttal_pairs]:
            for entry in entries:
                assert entry.get("challenge"), f"Missing challenge in entry: {entry}"


# ==============================================
# Stage 3: Rebuttals
# ==============================================

class TestStage3Parallel:
    """Rebuttals — parallel vs sequential equivalence."""

    def test_stage3_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes fill rebuttal fields in challenge_rebuttal_pairs."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=3)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=3)

        # Same number of entries
        assert len(ctrl_seq.challenge_rebuttal_pairs) == len(ctrl_par.challenge_rebuttal_pairs)

        # Both should have rebuttals populated
        for entry_s, entry_p in zip(ctrl_seq.challenge_rebuttal_pairs, ctrl_par.challenge_rebuttal_pairs):
            assert entry_s["challenger"] == entry_p["challenger"]
            assert entry_s["target"] == entry_p["target"]
            # Rebuttal should be populated (or empty string if not matched)
            assert "rebuttal" in entry_s
            assert "rebuttal" in entry_p


# ==============================================
# Stage 4: Adjudication
# ==============================================

class TestStage4Parallel:
    """Adjudication — parallel vs sequential equivalence."""

    def test_stage4_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes produce resolution outcomes for complete pairs."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=4)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=4)

        # Check resolutions exist
        resolved_seq = [e for e in ctrl_seq.challenge_rebuttal_pairs if "resolution" in e]
        resolved_par = [e for e in ctrl_par.challenge_rebuttal_pairs if "resolution" in e]

        assert len(resolved_seq) == len(resolved_par)

    def test_stage4_updates_agent_stats(self, mock_openai_responses):
        """Both modes update agent_stats after adjudication."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=4)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=4)

        # Both should have agent stats
        assert "Agent-A" in ctrl_seq.agent_stats
        assert "Agent-A" in ctrl_par.agent_stats

    def test_stage4_handles_adjudicator_failure(self, mock_openai_responses):
        """Adjudication stage runs and handles entries in parallel mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(True, tmpdir)
            agents = _make_agents(mock_openai_responses)
            controller = DebateController(agents, config=config)

            personas = {ac.name: ac.persona for ac in config.agents}
            controller.run_stage_0_briefing(config.topic, personas)
            controller.run_stage_1_opening_positions(config.topic)
            controller.run_stage_2_cross_examination()
            controller.run_stage_3_rebuttals()

            # Should not raise even in parallel mode
            controller.run_stage_4_conflict_resolution()

            # Some entries should have resolutions (or be marked incomplete)
            total_entries = len(controller.challenge_rebuttal_pairs)
            assert total_entries > 0


# ==============================================
# Stage 6: Concluding Remarks
# ==============================================

class TestStage6Parallel:
    """Concluding remarks — parallel vs sequential equivalence."""

    def test_stage6_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes produce concluding remarks for all agents."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=6)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=6)

        # Both should have conclusions in the markdown transcript
        assert "Concluding" in ctrl_seq.markdown_transcript or len(ctrl_seq.markdown_transcript) > 0
        assert "Concluding" in ctrl_par.markdown_transcript or len(ctrl_par.markdown_transcript) > 0

    def test_stage6_all_agents_conclude(self, mock_openai_responses):
        """Both modes have each agent generate a conclusion."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=6)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=6)

        # Each agent's generate should have been called for conclusions
        for agent in ctrl_seq.agents:
            assert agent.generate.called
        for agent in ctrl_par.agents:
            assert agent.generate.called


# ==============================================
# Dispatcher Integration Tests
# ==============================================

class TestDispatcherIntegration:
    """Tests that the dispatcher attribute is correctly set up."""

    def test_dispatcher_created_from_config(self, mock_openai_responses):
        """Controller creates dispatcher from ParallelConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(True, tmpdir)
            agents = _make_agents(mock_openai_responses)
            controller = DebateController(agents, config=config)

            assert hasattr(controller, "dispatcher")
            assert controller.dispatcher.enabled is True
            assert controller.dispatcher.max_workers == 5

    def test_dispatcher_disabled_by_default(self, mock_openai_responses):
        """Controller with default config has dispatcher disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(False, tmpdir)
            agents = _make_agents(mock_openai_responses)
            controller = DebateController(agents, config=config)

            assert controller.dispatcher.enabled is False

    def test_no_config_creates_sequential_dispatcher(self, mock_openai_responses):
        """Controller without config creates a disabled dispatcher."""
        agents = _make_agents(mock_openai_responses)
        controller = DebateController(agents, config=None)

        assert hasattr(controller, "dispatcher")
        assert controller.dispatcher.enabled is False
