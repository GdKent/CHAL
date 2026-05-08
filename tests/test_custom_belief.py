"""
Tests for custom belief agent support.

Covers:
- Stage 1 skip logic for pre-loaded agents
- Mixed custom + generated agent scenarios
- Bookkeeping for custom agents (belief history, transcript)
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from chal.agents.base import Message
from chal.beliefs.io import belief_to_markdown
from chal.config import DebateConfig, AgentConfig
from chal.orchestrator.debate_controller import DebateController
from tests.utils import create_sample_belief, create_mock_agent, create_mock_belief_response


# ==============================================
# Fixtures
# ==============================================

@pytest.fixture(autouse=True)
def mock_adjudicator():
    """Patch create_agent so the adjudicator uses a mock instead of real API calls."""
    mock_adj = create_mock_agent("Adjudicator", responses=['{"test": "adjudicator"}'])
    with patch("chal.orchestrator.debate_controller.create_agent", return_value=mock_adj):
        yield mock_adj


@pytest.fixture
def custom_belief():
    """A valid pre-loaded belief for testing."""
    return create_sample_belief(belief_id="BELIEF-CUSTOM")


@pytest.fixture
def mock_custom_agent(custom_belief):
    """An agent with a pre-loaded belief (simulates custom belief agent)."""
    agent = create_mock_agent(name="Agent-Custom")
    # Simulate runner pre-loading: set belief obj and markdown
    md = belief_to_markdown(custom_belief)
    agent.get_internal_belief_obj.return_value = custom_belief
    agent.get_internal_belief.return_value = md
    agent.internal_belief = md
    agent.internal_belief_obj = custom_belief
    return agent


@pytest.fixture
def mock_generation_agent():
    """An agent without a pre-loaded belief (needs Stage 1 generation)."""
    belief = create_sample_belief(belief_id="BELIEF-GEN")
    response_text = create_mock_belief_response(belief)
    agent = create_mock_agent(name="Agent-Generated", responses=[response_text])
    # get_internal_belief_obj returns None — needs generation
    agent.get_internal_belief_obj.return_value = None
    return agent


def _make_config():
    """Helper to build a minimal DebateConfig for two-agent tests."""
    return DebateConfig(
        topic="Test topic",
        agents=[
            AgentConfig(name="Agent-Custom", persona="NONE"),
            AgentConfig(name="Agent-Generated", persona="EMPIRICIST"),
        ],
    )


def _build_controller(custom_agent, gen_agent, config=None):
    """Helper to instantiate a DebateController and run Stage 0."""
    if config is None:
        config = _make_config()
    controller = DebateController(
        agents=[custom_agent, gen_agent],
        config=config,
    )
    controller.run_stage_0_briefing("Test topic", {
        "Agent-Custom": "",
        "Agent-Generated": "You are an empiricist.",
    })
    return controller


# ==============================================
# Tests
# ==============================================

class TestStage1CustomBeliefSkip:
    """Test that Stage 1 correctly skips pre-loaded custom agents."""

    def test_custom_agent_not_dispatched_for_generation(
        self, mock_custom_agent, mock_generation_agent
    ):
        """Custom agent should NOT trigger an LLM generate() call in Stage 1."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        # Custom agent should NOT have had generate() called
        mock_custom_agent.generate.assert_not_called()
        # Generated agent SHOULD have had generate() called
        assert mock_generation_agent.generate.call_count > 0

    def test_custom_agent_appears_in_opening_positions(
        self, mock_custom_agent, mock_generation_agent
    ):
        """Custom agent's belief should appear in controller.opening_positions."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        # Both agents should have entries in opening_positions
        assert len(controller.opening_positions) == 2

    def test_custom_agent_belief_tracked_in_history(
        self, mock_custom_agent, mock_generation_agent, custom_belief
    ):
        """Custom agent's belief should be appended to all_beliefs_held."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        # all_beliefs_held is a plain list on MagicMock; the controller appends
        # the JSON-serialised belief object to it.
        assert len(mock_custom_agent.all_beliefs_held) > 0
        # The appended value should be the JSON string of the custom belief
        recorded = json.loads(mock_custom_agent.all_beliefs_held[0])
        assert recorded["belief_id"] == "BELIEF-CUSTOM"

    def test_custom_agent_in_markdown_transcript(
        self, mock_custom_agent, mock_generation_agent
    ):
        """Custom agent should appear in the markdown transcript."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        transcript = "\n".join(controller.markdown_transcript)
        assert "Agent-Custom" in transcript
        assert "Custom Belief" in transcript

    def test_generated_agent_also_tracked_in_history(
        self, mock_custom_agent, mock_generation_agent
    ):
        """Generated agent's belief should also be appended to all_beliefs_held."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        # The generation agent should also have had its beliefs recorded
        assert len(mock_generation_agent.all_beliefs_held) > 0

    def test_all_custom_agents_skip_generation(self):
        """When ALL agents are custom, no LLM calls should be made at all."""
        belief_a = create_sample_belief(belief_id="BELIEF-A")
        belief_b = create_sample_belief(belief_id="BELIEF-B")

        agent_a = create_mock_agent(name="Agent-A")
        agent_a.get_internal_belief_obj.return_value = belief_a
        agent_a.get_internal_belief.return_value = belief_to_markdown(belief_a)
        agent_a.internal_belief = belief_to_markdown(belief_a)

        agent_b = create_mock_agent(name="Agent-B")
        agent_b.get_internal_belief_obj.return_value = belief_b
        agent_b.get_internal_belief.return_value = belief_to_markdown(belief_b)
        agent_b.internal_belief = belief_to_markdown(belief_b)

        config = DebateConfig(
            topic="Test topic",
            agents=[
                AgentConfig(name="Agent-A", persona="NONE"),
                AgentConfig(name="Agent-B", persona="NONE"),
            ],
        )
        controller = DebateController(
            agents=[agent_a, agent_b],
            config=config,
        )
        controller.run_stage_0_briefing("Test topic", {
            "Agent-A": "",
            "Agent-B": "",
        })
        controller.run_stage_1_opening_positions("Test topic")

        agent_a.generate.assert_not_called()
        agent_b.generate.assert_not_called()
        assert len(controller.opening_positions) == 2

    def test_custom_agent_notify_called(
        self, mock_custom_agent, mock_generation_agent
    ):
        """_notify should be called for custom agent completion."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)

        # Attach a spy to _notify
        events = []
        original_notify = controller._notify

        def spy_notify(event, data=None):
            events.append((event, data))
            original_notify(event, data)

        controller._notify = spy_notify
        controller.run_stage_1_opening_positions("Test topic")

        # There should be an agent_complete event for the custom agent
        custom_events = [
            (e, d) for e, d in events
            if e == "agent_complete" and d and d.get("agent_name") == "Agent-Custom"
        ]
        assert len(custom_events) == 1
        assert custom_events[0][1]["stage"] == 1

    def test_round_history_contains_custom_agent_message(
        self, mock_custom_agent, mock_generation_agent
    ):
        """Custom agent should produce a synthetic Message in round_histories."""
        controller = _build_controller(mock_custom_agent, mock_generation_agent)
        controller.run_stage_1_opening_positions("Test topic")

        round_0 = controller.round_histories.get("round-0", [])
        # Should have at least 2 messages (one per agent)
        assert len(round_0) >= 2
        # The first message (custom agent is processed first) should be
        # an assistant Message containing the belief markdown
        custom_msg = round_0[0]
        assert custom_msg.role == "assistant"
        assert "Thesis" in custom_msg.content  # belief_to_markdown always starts with "# Thesis"
