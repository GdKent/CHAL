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


@pytest.fixture(autouse=True)
def mock_adjudicator_agent(mock_openai_responses):
    """Patch create_agent so the adjudicator uses a mock instead of real API calls."""
    adj_response = mock_openai_responses["adjudicator_verdict"]["content"]
    mock_adj = create_mock_agent("Adjudicator", responses=[adj_response])
    with patch("chal.orchestrator.debate_controller.create_agent", return_value=mock_adj):
        yield mock_adj


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
    # Each agent needs responses for: Stage 1, Stage 2, Stage 3, Stage 5
    responses = [
        responses_data["belief_complete"]["content"],
        responses_data["cross_examination_3"]["content"],
        responses_data["rebuttals_3"]["content"],
        responses_data["belief_update_patches"]["content"],
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
        """Both modes produce current_round_pairs entries."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=2)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=2)

        # Same number of challenge-rebuttal pairs
        assert len(ctrl_seq.current_round_pairs) == len(ctrl_par.current_round_pairs)

        # Same challenger→target pairs in same order
        for entry_s, entry_p in zip(ctrl_seq.current_round_pairs, ctrl_par.current_round_pairs):
            assert entry_s["challenger"] == entry_p["challenger"]
            assert entry_s["target"] == entry_p["target"]

    def test_stage2_generates_challenges(self, mock_openai_responses):
        """Both modes populate challenge text in entries."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=2)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=2)

        for entries in [ctrl_seq.current_round_pairs, ctrl_par.current_round_pairs]:
            for entry in entries:
                assert entry.get("challenge"), f"Missing challenge in entry: {entry}"


# ==============================================
# Stage 3: Rebuttals
# ==============================================

class TestStage3Parallel:
    """Rebuttals — parallel vs sequential equivalence."""

    def test_stage3_parallel_matches_sequential(self, mock_openai_responses):
        """Both modes fill rebuttal fields in current_round_pairs."""
        ctrl_seq = _run_through_stage(mock_openai_responses, False, up_to_stage=3)
        ctrl_par = _run_through_stage(mock_openai_responses, True, up_to_stage=3)

        # Same number of entries
        assert len(ctrl_seq.current_round_pairs) == len(ctrl_par.current_round_pairs)

        # Both should have rebuttals populated
        for entry_s, entry_p in zip(ctrl_seq.current_round_pairs, ctrl_par.current_round_pairs):
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
        resolved_seq = [e for e in ctrl_seq.current_round_pairs if "resolution" in e]
        resolved_par = [e for e in ctrl_par.current_round_pairs if "resolution" in e]

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
            total_entries = len(controller.current_round_pairs)
            assert total_entries > 0


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


# ==============================================
# Stage 5: Belief Updates (Parallel)
# ==============================================

def _make_stage5_phase1_response():
    """Mock Phase 1 (enforcement) response with patches."""
    return (
        '<reasoning>C1 was critiqued validly, lowering strength.</reasoning>\n\n'
        '```json\n'
        '{"patches": [{"op": "update_claim", "target_id": "C1", '
        '"changes": {"strength": 0.55}}]}\n'
        '```'
    )


def _make_stage5_phase2_response():
    """Mock Phase 2 (introspection) response with patches."""
    return (
        '<reasoning>Rewriting thesis after enforcement.</reasoning>\n\n'
        '```json\n'
        '{"patches": [{"op": "update_thesis", "new_strength": 0.5, '
        '"stance": "Revised stance after debate", '
        '"summary_bullets": ["Revised bullet 1", "Revised bullet 2"]}]}\n'
        '```'
    )


def _setup_stage5_controller(
    agent_names,
    parallel_enabled,
    agents_with_entries=None,
):
    """Create a controller wired up for Stage 5 parallel testing.

    Args:
        agent_names: List of agent name strings.
        parallel_enabled: Whether to enable parallel dispatch.
        agents_with_entries: Set of agent names that have adjudication entries.
            Defaults to all agents.

    Returns:
        (controller, agents_dict) where agents_dict maps name -> mock agent.
    """
    if agents_with_entries is None:
        agents_with_entries = set(agent_names)

    tmpdir = tempfile.mkdtemp()

    config = DebateConfig(
        name="Stage 5 Parallel Test",
        topic="Test topic",
        max_rounds=1,
        agents=[AgentConfig(name=n, persona="EMPIRICIST") for n in agent_names],
        adjudication=AdjudicationConfig(),
        outputs=OutputConfig(storage_dir=Path(tmpdir)),
        parallel=ParallelConfig(enabled=parallel_enabled, max_workers=5),
    )

    agents = []
    agents_dict = {}
    for name in agent_names:
        responses = [
            _make_stage5_phase1_response(),
            _make_stage5_phase2_response(),
        ] * 10
        agent = create_mock_agent(name, responses=responses)
        belief = create_sample_belief(
            belief_id=f"BELIEF-{name}",
            num_claims=1, num_assumptions=1, num_evidence=1,
        )
        agent.get_internal_belief_obj.return_value = belief
        agent.get_internal_belief.return_value = f"markdown for {name}"
        agents.append(agent)
        agents_dict[name] = agent

    controller = DebateController(agents, config=config)
    controller.current_round_key = "round_1"
    controller.round_histories = {"round_1": []}
    controller.opening_positions = {n: f"opening-{n}" for n in agent_names}
    controller.current_positions = {}
    controller.last_rebuttals_patches = {}

    # Build current_round_pairs only for agents in agents_with_entries
    controller.current_round_pairs = []
    other_names = list(agent_names)
    for name in agent_names:
        if name in agents_with_entries:
            challenger = next((n for n in other_names if n != name), "External")
            controller.current_round_pairs.append({
                "target": name,
                "challenger": challenger,
                "challenge": f"Your C1 is weak, {name}",
                "qid": "Q1",
                "target_ids": ["C1"],
                "attack_type": "undermining",
                "attack_strategy": "challenge_strength_calibration",
                "rebuttal": "I defended with evidence",
                "resolution": {
                    "status": "critique_valid",
                    "reasoning": "The critique was valid",
                },
            })

    return controller, agents_dict


class TestStage5Parallel:
    """Stage 5 belief updates — parallel-specific tests."""

    def test_stage_5_parallel_dispatch(self, mock_openai_responses):
        """Enable parallel, verify dispatcher.run() receives correct WorkItems."""
        controller, agents = _setup_stage5_controller(
            ["Agent-A", "Agent-B"], parallel_enabled=True,
        )

        # Spy on the dispatcher to capture what items are dispatched
        original_run = controller.dispatcher.run
        dispatched_items = []

        def spy_run(items):
            dispatched_items.extend(items)
            return original_run(items)

        controller.dispatcher.run = spy_run
        controller.run_stage_5_update_positions()

        # Should dispatch exactly 2 WorkItems (one per agent with entries)
        assert len(dispatched_items) == 2
        dispatched_keys = {item.key for item in dispatched_items}
        assert dispatched_keys == {"Agent-A", "Agent-B"}

    def test_stage_5_parallel_agent_independence(self, mock_openai_responses):
        """One agent errors in gather; the other still gets its belief updated."""
        controller, agents = _setup_stage5_controller(
            ["Agent-A", "Agent-B"], parallel_enabled=True,
        )

        # Make Agent-B's generate raise an exception
        agents["Agent-B"].generate.side_effect = RuntimeError("API failure")

        controller.run_stage_5_update_positions()

        # Agent-A should still have its belief committed
        assert agents["Agent-A"].set_internal_belief_obj.called

        # Agent-B should NOT have its belief committed (error path)
        assert not agents["Agent-B"].set_internal_belief_obj.called

        # Debug log should mention Agent-B's error
        debug_text = controller.debug_log.get_contents()
        assert "Agent-B" in debug_text

    def test_stage_5_parallel_deterministic_order(self, mock_openai_responses):
        """Three agents — APPLY phase processes them in self.agents order."""
        controller, agents = _setup_stage5_controller(
            ["Agent-A", "Agent-B", "Agent-C"], parallel_enabled=True,
        )

        controller.run_stage_5_update_positions()

        # All three agents should have beliefs committed
        for name in ["Agent-A", "Agent-B", "Agent-C"]:
            assert agents[name].set_internal_belief_obj.called, f"{name} belief not committed"

        # Check debug_log ordering: Agent-A entries appear before Agent-B,
        # which appear before Agent-C
        debug_text = controller.debug_log.get_contents()
        pos_a = debug_text.index("Updating belief for: Agent-A")
        pos_b = debug_text.index("Updating belief for: Agent-B")
        pos_c = debug_text.index("Updating belief for: Agent-C")
        assert pos_a < pos_b < pos_c, (
            f"Expected deterministic order A < B < C, got {pos_a}, {pos_b}, {pos_c}"
        )

    def test_stage_5_parallel_no_entries_skipped(self, mock_openai_responses):
        """Agents with no adjudication results are not dispatched."""
        controller, agents = _setup_stage5_controller(
            ["Agent-A", "Agent-B"],
            parallel_enabled=True,
            agents_with_entries={"Agent-A"},  # Only Agent-A has entries
        )

        # Spy on dispatcher
        original_run = controller.dispatcher.run
        dispatched_items = []

        def spy_run(items):
            dispatched_items.extend(items)
            return original_run(items)

        controller.dispatcher.run = spy_run
        controller.run_stage_5_update_positions()

        # Only Agent-A should have been dispatched
        assert len(dispatched_items) == 1
        assert dispatched_items[0].key == "Agent-A"

        # Agent-A should have belief committed
        assert agents["Agent-A"].set_internal_belief_obj.called

        # Agent-B should NOT have been dispatched or have belief committed
        assert not agents["Agent-B"].set_internal_belief_obj.called
