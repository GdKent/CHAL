"""
Integration tests for moderated debate flow (stage2_mode == "moderated").

Tests verify:
- DebateController creates Moderator when stage2_mode == "moderated"
- Roadmap is generated before debate rounds begin
- Focus subtopics are passed to stage 2 cross-examination
- Open mode (stage2_mode == "open") produces identical behavior to default
- Training data recorder captures roadmap events
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from chal.agents.base import Agent, Message
from chal.config import (
    DebateConfig,
    AgentConfig,
    AdjudicationConfig,
    StageConfig,
    OutputConfig,
    ScribeConfig,
    ModeratorConfig,
)
from chal.orchestrator.moderator import Moderator, SubTopic, Roadmap
from tests.utils import create_sample_belief, create_mock_adjudication_response


# ========================================
# Helpers
# ========================================

def _make_mock_agent(name: str, responses=None, belief_obj=None):
    """Create a mock agent with all methods needed by the debate pipeline."""
    agent = Mock()
    agent.name = name
    agent.model = "gpt-4o"
    agent.provider = "openai"
    agent.persona_label = "TEST"
    agent.temperature = 0.7
    agent.current_belief = None

    belief = belief_obj or create_sample_belief()

    if responses is None:
        responses = ['{"test": "response"}']

    response_cycle = iter(responses * 200)

    def mock_generate(messages, temperature=None):
        return Message(role="assistant", content=next(response_cycle))

    agent.generate = Mock(side_effect=mock_generate)
    agent.get_internal_belief_obj = Mock(return_value=belief)
    agent.get_internal_belief = Mock(return_value=json.dumps(belief))
    agent.set_internal_belief = Mock()
    agent.set_internal_belief_obj = Mock()
    agent.internal_belief = json.dumps(belief)
    agent.internal_belief_obj = belief
    agent.update_current_belief = Mock()
    agent.system_prompt = ""
    agent.all_beliefs_held = [json.dumps(belief)]

    return agent


def _build_controller(stage2_mode="moderated", stage3_mode="rebuttal", tmpdir=None):
    """Build a DebateController with mocked create_agent calls."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    config = DebateConfig(
        name="Moderated Test",
        topic="Does free will exist?",
        max_rounds=2,
        stage2_mode=stage2_mode,
        stage3_mode=stage3_mode,
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        adjudication=AdjudicationConfig(),
        stages=StageConfig(),
        outputs=OutputConfig(storage_dir=Path(tmpdir)),
        scribe=ScribeConfig(enabled=False),
        moderator=ModeratorConfig(model="gpt-4o", provider="openai", temperature=0.3),
    )

    adj_response = create_mock_adjudication_response("rebuttal_valid")

    with patch("chal.orchestrator.debate_controller.create_agent") as mock_dc_create, \
         patch("chal.orchestrator.moderator.create_agent") as mock_mod_create:

        mock_adj = _make_mock_agent("Adjudicator", responses=[adj_response] * 50)
        mock_scribe = _make_mock_agent("Scribe")
        mock_dc_create.side_effect = [mock_adj, mock_scribe]
        mock_mod_create.return_value = _make_mock_agent("Moderator")

        from chal.orchestrator.debate_controller import DebateController
        agent_a = _make_mock_agent("Agent-A")
        agent_b = _make_mock_agent("Agent-B")

        controller = DebateController(
            agents=[agent_a, agent_b],
            max_rounds=2,
            config=config,
        )

    return controller, agent_a, agent_b


def _stub_all_stages(controller):
    """
    Stub out all pipeline stages so run() only exercises the dispatch logic
    (roadmap generation, round loop structure, focus subtopic passing).
    """
    controller.run_stage_0_briefing = Mock()
    controller.run_stage_1_opening_positions = Mock()
    # Don't stub stage 2 — we want to spy on it
    controller.run_stage_3_rebuttals = Mock()
    controller.run_stage_3_collaborative = Mock()
    controller.run_stage_3_bloodsport = Mock()
    controller.run_stage_4_conflict_resolution = Mock()
    controller.run_stage_5_update_positions = Mock()
    controller.run_stage_6_concluding_remarks = Mock()
    controller.run_stage_7_scribe = Mock()

    # Attributes set by stubbed stages that run() reads at the end
    controller.conclusions = {}
    controller.final_synthesis = ""


# ============================================================
# 1. Controller Initialization
# ============================================================

class TestModeratedControllerInit:

    @pytest.mark.unit
    def test_moderated_creates_moderator(self):
        """DebateController creates Moderator when stage2_mode == 'moderated'."""
        controller, _, _ = _build_controller(stage2_mode="moderated")

        assert controller.stage2_mode == "moderated"
        assert controller.moderator is not None
        assert isinstance(controller.moderator, Moderator)
        assert controller.roadmap is None

    @pytest.mark.unit
    def test_open_mode_no_moderator(self):
        """DebateController does NOT create Moderator when stage2_mode == 'open'."""
        controller, _, _ = _build_controller(stage2_mode="open")

        assert controller.stage2_mode == "open"
        assert controller.moderator is None
        assert controller.roadmap is None


# ============================================================
# 2. Roadmap Generation in run()
# ============================================================

class TestRoadmapGenerationInRun:

    @pytest.mark.unit
    def test_moderated_generates_roadmap(self):
        """In moderated mode, run() generates a roadmap before rounds begin."""
        controller, _, _ = _build_controller(stage2_mode="moderated")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        mock_roadmap = Roadmap(
            sub_topics=[
                SubTopic(title="Definitions", description="Define terms."),
                SubTopic(title="Evidence", description="Review evidence."),
            ],
            overall_rationale="Test rationale",
            sufficiency_note="Adequate",
            raw_response="raw text",
        )
        controller.moderator.generate_roadmap = Mock(return_value=mock_roadmap)

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Does free will exist?", personas=personas)

        controller.moderator.generate_roadmap.assert_called_once()
        assert controller.roadmap is not None
        assert len(controller.roadmap.sub_topics) == 2
        assert controller.roadmap.sub_topics[0].title == "Definitions"

    @pytest.mark.unit
    def test_open_mode_no_roadmap(self):
        """In open mode, run() does NOT generate a roadmap."""
        controller, _, _ = _build_controller(stage2_mode="open")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Does free will exist?", personas=personas)

        assert controller.moderator is None
        assert controller.roadmap is None

    @pytest.mark.unit
    def test_roadmap_logged_to_transcript(self):
        """Roadmap should be added to markdown transcript."""
        controller, _, _ = _build_controller(stage2_mode="moderated")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        mock_roadmap = Roadmap(
            sub_topics=[SubTopic(title="Topic A", description="Desc A")],
            overall_rationale="Test",
            sufficiency_note="OK",
            raw_response="raw text",
        )
        controller.moderator.generate_roadmap = Mock(return_value=mock_roadmap)

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        transcript = "\n".join(controller.markdown_transcript)
        assert "Debate Roadmap" in transcript
        assert "Topic A" in transcript


# ============================================================
# 3. Focus SubTopic Passing
# ============================================================

class TestFocusSubtopicPassing:

    @pytest.mark.unit
    def test_moderated_passes_focus_subtopic_to_stage2(self):
        """In moderated mode, stage 2 receives the focus subtopic for each round."""
        controller, _, _ = _build_controller(stage2_mode="moderated")
        _stub_all_stages(controller)

        # Set up roadmap
        roadmap = Roadmap(
            sub_topics=[
                SubTopic(title="Definitions", description="Define terms.",
                         guiding_questions=["Q1?"]),
                SubTopic(title="Evidence", description="Review evidence.",
                         guiding_questions=["Q2?"]),
            ],
            raw_response="raw",
        )

        def mock_generate_roadmap(*args, **kwargs):
            controller.moderator.roadmap = roadmap
            return roadmap

        controller.moderator.generate_roadmap = Mock(side_effect=mock_generate_roadmap)

        # Spy on stage 2 — mock it so it doesn't actually run
        controller.run_stage_2_cross_examination = Mock()

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        # Should have been called twice (2 rounds)
        assert controller.run_stage_2_cross_examination.call_count == 2

        # Round 1: focus_subtopic should be "Definitions"
        _, kwargs1 = controller.run_stage_2_cross_examination.call_args_list[0]
        assert kwargs1.get("focus_subtopic") is not None
        assert kwargs1["focus_subtopic"]["title"] == "Definitions"
        assert "Q1?" in kwargs1["focus_subtopic"]["guiding_questions"]

        # Round 2: focus_subtopic should be "Evidence"
        _, kwargs2 = controller.run_stage_2_cross_examination.call_args_list[1]
        assert kwargs2.get("focus_subtopic") is not None
        assert kwargs2["focus_subtopic"]["title"] == "Evidence"

    @pytest.mark.unit
    def test_open_mode_no_focus_subtopic(self):
        """In open mode, stage 2 receives focus_subtopic=None."""
        controller, _, _ = _build_controller(stage2_mode="open")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        assert controller.run_stage_2_cross_examination.call_count == 2

        for call_args in controller.run_stage_2_cross_examination.call_args_list:
            _, kwargs = call_args
            assert kwargs.get("focus_subtopic") is None

    @pytest.mark.unit
    def test_focus_subtopic_dict_structure(self):
        """Focus subtopic passed to stage 2 has correct dict structure."""
        controller, _, _ = _build_controller(stage2_mode="moderated")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        roadmap = Roadmap(
            sub_topics=[
                SubTopic(
                    title="My Topic",
                    description="My description.",
                    guiding_questions=["GQ1?", "GQ2?"],
                ),
                SubTopic(title="Other", description="Other."),
            ],
            raw_response="raw",
        )

        def mock_generate_roadmap(*args, **kwargs):
            controller.moderator.roadmap = roadmap
            return roadmap

        controller.moderator.generate_roadmap = Mock(side_effect=mock_generate_roadmap)

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        _, kwargs = controller.run_stage_2_cross_examination.call_args_list[0]
        focus = kwargs["focus_subtopic"]

        assert isinstance(focus, dict)
        assert "title" in focus
        assert "description" in focus
        assert "guiding_questions" in focus
        assert focus["title"] == "My Topic"
        assert focus["description"] == "My description."
        assert len(focus["guiding_questions"]) == 2


# ============================================================
# 4. Training Data Recording
# ============================================================

class TestTrainingDataWithModerator:

    @pytest.mark.unit
    def test_moderated_records_roadmap_in_training_data(self):
        """When training data is enabled, roadmap generation is recorded."""
        tmpdir = tempfile.mkdtemp()
        controller, _, _ = _build_controller(stage2_mode="moderated", tmpdir=tmpdir)
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()
        controller.config.outputs.save_training_data = True

        mock_roadmap = Roadmap(
            sub_topics=[SubTopic(title="Topic A", description="Desc A")],
            overall_rationale="Test",
            sufficiency_note="OK",
            raw_response="raw",
        )
        controller.moderator.generate_roadmap = Mock(return_value=mock_roadmap)

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Does free will exist?", personas=personas)

        assert controller.recorder is not None
        roadmap_events = [
            e for e in controller.recorder.timeline
            if e["type"] == "roadmap_generation"
        ]
        assert len(roadmap_events) == 1
        assert roadmap_events[0]["stage"] == 0
        assert "roadmap" in roadmap_events[0]["outputs"]

    @pytest.mark.unit
    def test_training_data_metadata_has_stage2_mode(self):
        """Training data metadata includes stage2_mode."""
        tmpdir = tempfile.mkdtemp()
        controller, _, _ = _build_controller(stage2_mode="moderated", tmpdir=tmpdir)
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()
        controller.config.outputs.save_training_data = True

        mock_roadmap = Roadmap(
            sub_topics=[SubTopic(title="A", description="D")],
            raw_response="raw",
        )
        controller.moderator.generate_roadmap = Mock(return_value=mock_roadmap)

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        assert controller.recorder.metadata["stage2_mode"] == "moderated"

    @pytest.mark.unit
    def test_open_mode_no_roadmap_event(self):
        """In open mode, no roadmap_generation event is recorded."""
        tmpdir = tempfile.mkdtemp()
        controller, _, _ = _build_controller(stage2_mode="open", tmpdir=tmpdir)
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()
        controller.config.outputs.save_training_data = True

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        assert controller.recorder is not None
        roadmap_events = [
            e for e in controller.recorder.timeline
            if e["type"] == "roadmap_generation"
        ]
        assert len(roadmap_events) == 0


# ============================================================
# 5. Verify open mode produces identical behavior
# ============================================================

class TestOpenModeUnchanged:

    @pytest.mark.unit
    def test_open_mode_calls_same_stages(self):
        """In open mode, all standard stages are called in order."""
        controller, _, _ = _build_controller(stage2_mode="open")
        _stub_all_stages(controller)
        controller.run_stage_2_cross_examination = Mock()

        personas = {"Agent-A": "Empiricist", "Agent-B": "Rationalist"}
        controller.run(topic="Topic", personas=personas)

        controller.run_stage_0_briefing.assert_called_once()
        controller.run_stage_1_opening_positions.assert_called_once()
        assert controller.run_stage_2_cross_examination.call_count == 2  # 2 rounds
        assert controller.run_stage_3_rebuttals.call_count == 2
        assert controller.run_stage_4_conflict_resolution.call_count == 2
        assert controller.run_stage_5_update_positions.call_count == 2
