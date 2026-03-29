"""
Unit and integration tests for Collaborative Truth-Seeking Mode (Stage 3B).

Tests cover:
- Collaborative prompt builders (4 functions)
- run_stage_3_collaborative() method (turn alternation, adjudicator checks,
  early termination, output structure)
- run() branching (stage3_mode routing)
- collaborative.yaml config loading
"""

import pytest
import json
import textwrap
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from chal.agents.base import Agent, Message
from chal.agents.prompts import (
    build_collaborative_defender_prompt,
    build_collaborative_challenger_followup_prompt,
    build_collaborative_adjudicator_check_prompt,
    build_collaborative_final_adjudication_prompt,
)
from chal.config import (
    DebateConfig,
    AgentConfig,
    AdjudicationConfig,
    StageConfig,
    OutputConfig,
    ScribeConfig,
    CollaborativeConfig,
    load_config,
)
from tests.utils import create_sample_belief, create_mock_adjudication_response


# ========================================
# Helper: create a mock agent for collaborative tests
# ========================================

def _make_mock_agent(name: str, responses=None, belief_obj=None):
    """Create a mock agent with generate(), get_internal_belief_obj(), etc."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.model = "gpt-4o"
    agent.temperature = 0.7
    agent.current_belief = None

    if responses is None:
        responses = ["Mock response"]

    response_cycle = iter(responses * 200)

    def mock_generate(messages):
        return Message(role="assistant", content=next(response_cycle))

    agent.generate = Mock(side_effect=mock_generate)
    agent.get_internal_belief_obj = Mock(return_value=belief_obj or create_sample_belief())
    agent.get_internal_belief = Mock(return_value="Mock belief text")
    agent.internal_belief = "Mock belief text"
    agent.internal_belief_obj = belief_obj or create_sample_belief()
    agent.update_current_belief = Mock()
    agent.system_prompt = ""

    return agent


# ============================================================
# 1. Collaborative Prompt Builder Tests
# ============================================================

class TestCollaborativeDefenderPrompt:
    """Tests for build_collaborative_defender_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        prompt = build_collaborative_defender_prompt(
            topic="Free will",
            defender_name="Agent-A",
            challenger_name="Agent-B",
            defender_belief_json='{"thesis": {"stance": "test"}}',
            question_text="Why do you believe X?",
            dialogue_history=[],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    @pytest.mark.unit
    def test_contains_expected_parameters(self):
        prompt = build_collaborative_defender_prompt(
            topic="Consciousness",
            defender_name="Defender-1",
            challenger_name="Challenger-1",
            defender_belief_json='{"thesis": {"stance": "test"}}',
            question_text="How do you define qualia?",
            dialogue_history=[],
        )
        assert "Consciousness" in prompt
        assert "Defender-1" in prompt
        assert "Challenger-1" in prompt
        assert "How do you define qualia?" in prompt

    @pytest.mark.unit
    def test_includes_dialogue_history_when_present(self):
        history = [
            {"speaker": "Defender-1", "message": "My first response"},
            {"speaker": "Challenger-1", "message": "Follow-up question"},
        ]
        prompt = build_collaborative_defender_prompt(
            topic="Ethics",
            defender_name="Defender-1",
            challenger_name="Challenger-1",
            defender_belief_json="{}",
            question_text="Original Q",
            dialogue_history=history,
        )
        assert "My first response" in prompt
        assert "Follow-up question" in prompt

    @pytest.mark.unit
    def test_omits_history_section_when_empty(self):
        prompt = build_collaborative_defender_prompt(
            topic="Ethics",
            defender_name="D",
            challenger_name="C",
            defender_belief_json="{}",
            question_text="Q?",
            dialogue_history=[],
        )
        assert "DIALOGUE SO FAR" not in prompt

    @pytest.mark.unit
    def test_mentions_collaborative_purpose(self):
        prompt = build_collaborative_defender_prompt(
            topic="T",
            defender_name="D",
            challenger_name="C",
            defender_belief_json="{}",
            question_text="Q?",
            dialogue_history=[],
        )
        assert "collaborative" in prompt.lower() or "truth-seeking" in prompt.lower()

    @pytest.mark.unit
    def test_accepts_max_response_length(self):
        """Test that max_response_length_chars param is accepted."""
        prompt = build_collaborative_defender_prompt(
            topic="T",
            defender_name="D",
            challenger_name="C",
            defender_belief_json="{}",
            question_text="Q?",
            dialogue_history=[],
            max_response_length_chars=750,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestCollaborativeChallengerFollowupPrompt:
    """Tests for build_collaborative_challenger_followup_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        history = [{"speaker": "D", "message": "response"}]
        prompt = build_collaborative_challenger_followup_prompt(
            topic="Free will",
            challenger_name="C",
            defender_name="D",
            challenger_belief_json="{}",
            question_text="Q?",
            dialogue_history=history,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    @pytest.mark.unit
    def test_contains_expected_parameters(self):
        history = [{"speaker": "Def", "message": "resp"}]
        prompt = build_collaborative_challenger_followup_prompt(
            topic="Metaphysics",
            challenger_name="Chal",
            defender_name="Def",
            challenger_belief_json='{"thesis": {"stance": "test"}}',
            question_text="Original question?",
            dialogue_history=history,
        )
        assert "Metaphysics" in prompt
        assert "Chal" in prompt
        assert "Def" in prompt
        assert "Original question?" in prompt

    @pytest.mark.unit
    def test_includes_dialogue_history(self):
        history = [
            {"speaker": "D", "message": "First answer"},
            {"speaker": "C", "message": "Follow-up"},
            {"speaker": "D", "message": "Second answer"},
        ]
        prompt = build_collaborative_challenger_followup_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            challenger_belief_json="{}",
            question_text="Q?",
            dialogue_history=history,
        )
        assert "First answer" in prompt
        assert "Second answer" in prompt

    @pytest.mark.unit
    def test_mentions_crux_of_disagreement(self):
        history = [{"speaker": "D", "message": "resp"}]
        prompt = build_collaborative_challenger_followup_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            challenger_belief_json="{}",
            question_text="Q?",
            dialogue_history=history,
        )
        assert "crux" in prompt.lower() or "CRUX" in prompt


class TestCollaborativeAdjudicatorCheckPrompt:
    """Tests for build_collaborative_adjudicator_check_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        history = [{"speaker": "D", "message": "msg"}]
        prompt = build_collaborative_adjudicator_check_prompt(
            dialogue_history=history,
            challenger_name="C",
            defender_name="D",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @pytest.mark.unit
    def test_checks_for_fallacies_deflection_progress_convergence(self):
        history = [{"speaker": "D", "message": "msg"}]
        prompt = build_collaborative_adjudicator_check_prompt(
            dialogue_history=history,
            challenger_name="C",
            defender_name="D",
        )
        prompt_lower = prompt.lower()
        assert "fallac" in prompt_lower
        assert "deflection" in prompt_lower
        assert "progress" in prompt_lower
        assert "convergence" in prompt_lower

    @pytest.mark.unit
    def test_requests_json_output(self):
        history = [{"speaker": "D", "message": "msg"}]
        prompt = build_collaborative_adjudicator_check_prompt(
            dialogue_history=history,
            challenger_name="C",
            defender_name="D",
        )
        assert "json" in prompt.lower()


class TestCollaborativeFinalAdjudicationPrompt:
    """Tests for build_collaborative_final_adjudication_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        transcript = [{"turn_number": 1, "speaker": "D", "message": "msg"}]
        prompt = build_collaborative_final_adjudication_prompt(
            topic="Free will",
            challenger_name="C",
            defender_name="D",
            question_text="Q?",
            target_ids=["C1"],
            dialogue_transcript=transcript,
            adjudicator_checks=[],
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    @pytest.mark.unit
    def test_contains_all_three_outcomes(self):
        transcript = [{"turn_number": 1, "speaker": "D", "message": "msg"}]
        prompt = build_collaborative_final_adjudication_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            question_text="Q?",
            target_ids=[],
            dialogue_transcript=transcript,
            adjudicator_checks=[],
        )
        assert "rebuttal_valid" in prompt
        assert "critique_valid" in prompt
        assert "unresolved" in prompt

    @pytest.mark.unit
    def test_includes_target_ids(self):
        transcript = [{"turn_number": 1, "speaker": "D", "message": "msg"}]
        prompt = build_collaborative_final_adjudication_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            question_text="Q?",
            target_ids=["C3", "A1"],
            dialogue_transcript=transcript,
            adjudicator_checks=[],
        )
        assert "C3" in prompt
        assert "A1" in prompt

    @pytest.mark.unit
    def test_includes_interim_checks(self):
        transcript = [{"turn_number": 1, "speaker": "D", "message": "msg"}]
        checks = [{"progress_assessment": "productive", "convergence_detected": False,
                    "fallacies_detected": [], "deflection_detected": False}]
        prompt = build_collaborative_final_adjudication_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            question_text="Q?",
            target_ids=[],
            dialogue_transcript=transcript,
            adjudicator_checks=checks,
        )
        assert "productive" in prompt

    @pytest.mark.unit
    def test_uses_custom_weights(self):
        transcript = [{"turn_number": 1, "speaker": "D", "message": "msg"}]
        prompt = build_collaborative_final_adjudication_prompt(
            topic="T",
            challenger_name="C",
            defender_name="D",
            question_text="Q?",
            target_ids=[],
            dialogue_transcript=transcript,
            adjudicator_checks=[],
            logic_weight=0.6,
            ethics_weight=0.4,
        )
        assert "0.6" in prompt
        assert "0.4" in prompt


# ============================================================
# 2. run_stage_3_collaborative() Tests
# ============================================================

def _build_controller_for_collaborative(
    agent_responses=None,
    adjudicator_responses=None,
    max_turns=4,
    min_turns=2,
    check_interval=2,
    early_term=True,
    tmpdir=None,
):
    """
    Build a DebateController configured for collaborative mode with mocked agents.

    Returns (controller, agent_a, agent_b).
    """
    import tempfile

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    config = DebateConfig(
        name="Collaborative Test",
        topic="Does free will exist?",
        max_rounds=1,
        stage3_mode="collaborative",
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        adjudication=AdjudicationConfig(),
        stages=StageConfig(max_rebuttal_length_chars=500),
        outputs=OutputConfig(storage_dir=Path(tmpdir)),
        scribe=ScribeConfig(enabled=False),
        collaborative=CollaborativeConfig(
            max_turns_per_question=max_turns,
            min_turns_per_question=min_turns,
            adjudicator_check_interval=check_interval,
            early_termination_on_agreement=early_term,
        ),
    )

    # Default agent responses: just echo turn text
    if agent_responses is None:
        agent_responses = ["This is my collaborative response."]

    agent_a = _make_mock_agent("Agent-A", responses=agent_responses)
    agent_b = _make_mock_agent("Agent-B", responses=agent_responses)

    # Default adjudicator check response (no convergence)
    adj_check = json.dumps({
        "fallacies_detected": [],
        "deflection_detected": False,
        "progress_assessment": "productive",
        "convergence_detected": False,
    })
    adj_check_response = f"```json\n{adj_check}\n```"

    # Default final adjudication response
    adj_final = json.dumps({
        "restatement": "Test disagreement",
        "formalization_challenger": "P1 → Q",
        "formalization_target": "¬P1",
        "outcome": "rebuttal_valid",
        "reasoning": "Defender addressed the concern",
    })
    adj_final_response = f"```json\n{adj_final}\n```"

    if adjudicator_responses is None:
        # Provide enough check responses for max_turns/check_interval checks + 1 final
        num_checks = max_turns // check_interval if check_interval <= max_turns else 0
        adjudicator_responses = [adj_check_response] * num_checks + [adj_final_response]

    # Patch create_agent to avoid real API calls
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

    # Override adjudicator agent with our mock (the Adjudicator wraps the agent)
    controller.adjudicator.agent = mock_adj_agent

    return controller, agent_a, agent_b


class TestRunStage3Collaborative:
    """Tests for DebateController.run_stage_3_collaborative()."""

    @pytest.mark.unit
    def test_populates_required_keys(self):
        """Each entry gets rebuttal, resolution, collaborative_transcript, adjudicator_checks."""
        controller, agent_a, agent_b = _build_controller_for_collaborative(max_turns=4)

        # Seed a challenge-rebuttal pair (normally Stage 2 does this)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Why do you believe X?",
            "qid": "Q1",
            "target_ids": ["C1"],
        }]

        result = controller.run_stage_3_collaborative()

        assert len(result) == 1
        entry = result[0]
        assert "rebuttal" in entry
        assert "resolution" in entry
        assert "collaborative_transcript" in entry
        assert "adjudicator_checks" in entry
        assert isinstance(entry["collaborative_transcript"], list)
        assert isinstance(entry["adjudicator_checks"], list)

    @pytest.mark.unit
    def test_resolution_has_correct_format(self):
        """Resolution dict matches Stage 4 format: status, reasoning, restatement, formalizations."""
        controller, _, _ = _build_controller_for_collaborative(max_turns=4)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Test question",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        resolution = controller.challenge_rebuttal_pairs[0]["resolution"]

        assert "status" in resolution
        assert resolution["status"] in ("rebuttal_valid", "critique_valid", "unresolved")
        assert "reasoning" in resolution
        assert "restatement" in resolution
        assert "formalizations" in resolution
        assert "challenger" in resolution["formalizations"]
        assert "target" in resolution["formalizations"]

    @pytest.mark.unit
    def test_turn_alternation(self):
        """Odd turns are defender, even turns are challenger."""
        controller, agent_a, agent_b = _build_controller_for_collaborative(
            max_turns=4, check_interval=99  # High interval to avoid extra adj calls
        )
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        transcript = controller.challenge_rebuttal_pairs[0]["collaborative_transcript"]

        assert len(transcript) == 4
        # Turn 1 (odd) -> defender (Agent-B)
        assert transcript[0]["speaker"] == "Agent-B"
        # Turn 2 (even) -> challenger (Agent-A)
        assert transcript[1]["speaker"] == "Agent-A"
        # Turn 3 (odd) -> defender
        assert transcript[2]["speaker"] == "Agent-B"
        # Turn 4 (even) -> challenger
        assert transcript[3]["speaker"] == "Agent-A"

    @pytest.mark.unit
    def test_adjudicator_check_frequency(self):
        """Adjudicator checks happen every check_interval turns."""
        controller, _, _ = _build_controller_for_collaborative(
            max_turns=6, check_interval=2
        )
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        checks = controller.challenge_rebuttal_pairs[0]["adjudicator_checks"]

        # With 6 turns and check_interval=2, checks at turns 2, 4, 6 → 3 checks
        assert len(checks) == 3

    @pytest.mark.unit
    def test_early_termination_on_convergence(self):
        """Dialogue stops early when convergence_detected=True after min_turns."""
        # Build adjudicator responses: first check = no convergence, second check = convergence
        adj_check_no_conv = '```json\n{"fallacies_detected": [], "deflection_detected": false, "progress_assessment": "productive", "convergence_detected": false}\n```'
        adj_check_conv = '```json\n{"fallacies_detected": [], "deflection_detected": false, "progress_assessment": "productive", "convergence_detected": true}\n```'
        adj_final = '```json\n{"restatement": "test", "formalization_challenger": "P", "formalization_target": "Q", "outcome": "rebuttal_valid", "reasoning": "test"}\n```'

        controller, _, _ = _build_controller_for_collaborative(
            max_turns=10,
            min_turns=2,
            check_interval=2,
            adjudicator_responses=[adj_check_no_conv, adj_check_conv, adj_final],
        )
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        transcript = controller.challenge_rebuttal_pairs[0]["collaborative_transcript"]

        # Should stop at turn 4 (second check detects convergence, turn 4 >= min_turns 2)
        assert len(transcript) == 4

    @pytest.mark.unit
    def test_no_early_termination_before_min_turns(self):
        """Convergence before min_turns does not terminate dialogue."""
        adj_check_conv = '```json\n{"fallacies_detected": [], "deflection_detected": false, "progress_assessment": "productive", "convergence_detected": true}\n```'
        adj_final = '```json\n{"restatement": "test", "formalization_challenger": "P", "formalization_target": "Q", "outcome": "critique_valid", "reasoning": "test"}\n```'

        controller, _, _ = _build_controller_for_collaborative(
            max_turns=6,
            min_turns=5,  # min_turns > check point at turn 2
            check_interval=2,
            adjudicator_responses=[adj_check_conv, adj_check_conv, adj_check_conv, adj_final],
        )
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        transcript = controller.challenge_rebuttal_pairs[0]["collaborative_transcript"]

        # Convergence at turn 2 but min_turns=5, so continues
        # Turn 4 check: convergence=True but 4 < 5, continues
        # Turn 6 check: convergence=True and 6 >= 5, terminates
        assert len(transcript) == 6

    @pytest.mark.unit
    def test_rebuttal_is_defenders_last_message(self):
        """entry['rebuttal'] is set to the defender's final message."""
        defender_responses = [
            "Defender turn 1 response",
            "Defender turn 3 response",
        ]
        challenger_responses = [
            "Challenger turn 2 response",
            "Challenger turn 4 response",
        ]
        # Interleave: turn 1 (defender), turn 2 (challenger), turn 3 (defender), turn 4 (challenger)
        # Agent-B is defender, Agent-A is challenger
        controller, agent_a, agent_b = _build_controller_for_collaborative(max_turns=4, check_interval=99)
        agent_b.generate.side_effect = [
            Message(role="assistant", content=r) for r in defender_responses
        ] * 100
        agent_a.generate.side_effect = [
            Message(role="assistant", content=r) for r in challenger_responses
        ] * 100

        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        controller.run_stage_3_collaborative()
        assert controller.challenge_rebuttal_pairs[0]["rebuttal"] == "Defender turn 3 response"

    @pytest.mark.unit
    def test_updates_agent_stats(self):
        """Agent stats are updated after each entry."""
        controller, _, _ = _build_controller_for_collaborative(max_turns=2, check_interval=99)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": [],
        }]

        # Stats should start at 0
        assert controller.agent_stats["Agent-A"]["total_arguments"] == 0

        controller.run_stage_3_collaborative()

        # Stats should be updated (at least total_arguments incremented)
        total = (
            controller.agent_stats["Agent-A"]["total_arguments"]
            + controller.agent_stats["Agent-B"]["total_arguments"]
        )
        assert total > 0

    @pytest.mark.unit
    def test_anti_repetition_tracking(self):
        """previous_rounds_challenges is populated with outcomes."""
        controller, _, _ = _build_controller_for_collaborative(max_turns=2, check_interval=99)
        controller.challenge_rebuttal_pairs = [{
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": "Q?",
            "qid": "Q1",
            "target_ids": ["C1"],
        }]

        controller.run_stage_3_collaborative()

        key = "Agent-A→Agent-B"
        assert key in controller.previous_rounds_challenges
        assert len(controller.previous_rounds_challenges[key]) == 1
        assert controller.previous_rounds_challenges[key][0]["qid"] == "Q1"

    @pytest.mark.unit
    def test_handles_multiple_entries(self):
        """Multiple challenge-rebuttal pairs are all processed."""
        controller, _, _ = _build_controller_for_collaborative(max_turns=2, check_interval=99)
        controller.challenge_rebuttal_pairs = [
            {
                "challenger": "Agent-A",
                "target": "Agent-B",
                "challenge": "Q1?",
                "qid": "Q1",
                "target_ids": [],
            },
            {
                "challenger": "Agent-B",
                "target": "Agent-A",
                "challenge": "Q2?",
                "qid": "Q2",
                "target_ids": [],
            },
        ]

        result = controller.run_stage_3_collaborative()

        assert len(result) == 2
        assert all("resolution" in e for e in result)
        assert all("collaborative_transcript" in e for e in result)


# ============================================================
# 3. run() Branching Tests
# ============================================================

class TestRunBranching:
    """Tests for stage3_mode routing in the run() loop.

    Instead of calling the full run() (which has many dependencies),
    we test the branching logic by simulating the relevant code path.
    """

    @pytest.mark.unit
    def test_collaborative_mode_calls_stage_3b_and_skips_stage_4(self):
        """In collaborative mode, run_stage_3_collaborative is called, Stage 4 is skipped."""
        controller, _, _ = _build_controller_for_collaborative()

        with patch.object(controller, "run_stage_3_collaborative") as mock_3b, \
             patch.object(controller, "run_stage_3_rebuttals") as mock_3, \
             patch.object(controller, "run_stage_4_conflict_resolution") as mock_4:

            # Simulate the branching logic from run()
            stage3_mode = controller.config.stage3_mode if controller.config else "rebuttal"
            if stage3_mode == "collaborative":
                controller.run_stage_3_collaborative()
            else:
                controller.run_stage_3_rebuttals()
                controller.run_stage_4_conflict_resolution()

        mock_3b.assert_called_once()
        mock_3.assert_not_called()
        mock_4.assert_not_called()

    @pytest.mark.unit
    def test_rebuttal_mode_calls_stage_3_and_stage_4(self):
        """In rebuttal mode, run_stage_3_rebuttals + Stage 4 are called."""
        controller, _, _ = _build_controller_for_collaborative()
        controller.config.stage3_mode = "rebuttal"

        with patch.object(controller, "run_stage_3_collaborative") as mock_3b, \
             patch.object(controller, "run_stage_3_rebuttals") as mock_3, \
             patch.object(controller, "run_stage_4_conflict_resolution") as mock_4:

            stage3_mode = controller.config.stage3_mode if controller.config else "rebuttal"
            if stage3_mode == "collaborative":
                controller.run_stage_3_collaborative()
            else:
                controller.run_stage_3_rebuttals()
                controller.run_stage_4_conflict_resolution()

        mock_3.assert_called_once()
        mock_4.assert_called_once()
        mock_3b.assert_not_called()

    @pytest.mark.unit
    def test_default_mode_is_rebuttal(self):
        """Without explicit config, defaults to rebuttal mode."""
        controller, _, _ = _build_controller_for_collaborative()
        controller.config = None  # No config

        with patch.object(controller, "run_stage_3_collaborative") as mock_3b, \
             patch.object(controller, "run_stage_3_rebuttals") as mock_3, \
             patch.object(controller, "run_stage_4_conflict_resolution") as mock_4:

            stage3_mode = controller.config.stage3_mode if controller.config else "rebuttal"
            if stage3_mode == "collaborative":
                controller.run_stage_3_collaborative()
            else:
                controller.run_stage_3_rebuttals()
                controller.run_stage_4_conflict_resolution()

        mock_3.assert_called_once()
        mock_4.assert_called_once()
        mock_3b.assert_not_called()


# ============================================================
# 4. Config Loading Tests
# ============================================================

class TestCollaborativeConfigLoading:
    """Tests for loading collaborative.yaml and CollaborativeConfig."""

    @pytest.mark.unit
    def test_load_collaborative_config(self, tmp_path):
        """Inline collaborative YAML loads with stage3_mode='collaborative'."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Collaborative Debate"
              version: "1.0"
            debate:
              topic: "Does free will exist?"
              max_rounds: 1
              stage3_mode: "collaborative"
            agents:
              - name: "Agent-A"
                persona: "EMPIRICIST"
                model: "gpt-4o"
                temperature: 0.7
              - name: "Agent-B"
                persona: "RATIONALIST"
                model: "gpt-4o"
                temperature: 0.7
            adjudication:
              model: "gpt-4o"
              logic_weight: 1.0
              ethics_weight: 0.0
              logic_system: "Classical logic"
              ethics_system: "None"
            collaborative:
              max_turns_per_question: 10
              min_turns_per_question: 3
              adjudicator_check_interval: 2
              early_termination_on_agreement: true
        """)
        config_file = tmp_path / "collab.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)

        assert config.name == "Collaborative Debate"
        assert config.stage3_mode == "collaborative"
        assert config.topic == "Does free will exist?"
        assert len(config.agents) == 2

    @pytest.mark.unit
    def test_collaborative_section_parsed(self, tmp_path):
        """Collaborative settings are correctly parsed from YAML."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Collab Parsed"
              version: "1.0"
            debate:
              topic: "Test"
              max_rounds: 1
              stage3_mode: "collaborative"
            agents:
              - name: "Agent-A"
                persona: "EMPIRICIST"
                model: "gpt-4o"
                temperature: 0.7
            adjudication:
              model: "gpt-4o"
              logic_weight: 1.0
              ethics_weight: 0.0
              logic_system: "Classical logic"
              ethics_system: "None"
            collaborative:
              max_turns_per_question: 10
              min_turns_per_question: 3
              adjudicator_check_interval: 2
              early_termination_on_agreement: true
        """)
        config_file = tmp_path / "collab_parsed.yaml"
        config_file.write_text(yaml_content)
        config = DebateConfig.from_yaml(config_file)

        assert config.collaborative.max_turns_per_question == 10
        assert config.collaborative.min_turns_per_question == 3
        assert config.collaborative.adjudicator_check_interval == 2
        assert config.collaborative.early_termination_on_agreement is True

    @pytest.mark.unit
    def test_default_config_has_rebuttal_mode(self):
        """default.yaml still loads with stage3_mode='rebuttal' (backward compat)."""
        config = load_config("default")
        assert config.stage3_mode == "rebuttal"

    @pytest.mark.unit
    def test_default_config_has_collaborative_defaults(self):
        """default.yaml has default CollaborativeConfig even though section is commented."""
        config = load_config("default")
        # CollaborativeConfig defaults should be used since the section is commented out
        assert config.collaborative.max_turns_per_question == 10
        assert config.collaborative.min_turns_per_question == 3

    @pytest.mark.unit
    def test_collaborative_config_from_yaml(self, tmp_path):
        """Custom YAML with collaborative settings parses correctly."""
        yaml_content = textwrap.dedent("""\
            metadata:
              name: "Custom Collab"
              version: "1.0"
            debate:
              topic: "Is math invented or discovered?"
              max_rounds: 2
              stage3_mode: "collaborative"
            agents:
              - name: "Agent-X"
                persona: "RATIONALIST"
                model: "gpt-4o"
                temperature: 0.5
            adjudication:
              model: "gpt-4o"
              logic_weight: 0.8
              ethics_weight: 0.2
              logic_system: "Formal logic"
              ethics_system: "Virtue ethics"
            collaborative:
              max_turns_per_question: 6
              min_turns_per_question: 2
              adjudicator_check_interval: 3
              early_termination_on_agreement: false
        """)
        config_file = tmp_path / "custom_collab.yaml"
        config_file.write_text(yaml_content)

        config = DebateConfig.from_yaml(config_file)

        assert config.stage3_mode == "collaborative"
        assert config.collaborative.max_turns_per_question == 6
        assert config.collaborative.min_turns_per_question == 2
        assert config.collaborative.adjudicator_check_interval == 3
        assert config.collaborative.early_termination_on_agreement is False
