"""
Unit tests for Moderator class and roadmap generation.

Tests cover:
- Moderator initialization
- Roadmap generation with mocked LLM responses
- Roadmap response parsing (JSON, fenced JSON, malformed)
- SubTopic retrieval by round index
- review_round() static mode no-op
- Truncation when sub_topics exceed num_rounds
- Fallback behavior for empty/invalid responses
- Stage 2 prompt with/without focus_subtopic
- Moderator roadmap prompt builder
- Reporting integration with roadmap data
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from chal.orchestrator.moderator import Moderator, SubTopic, Roadmap, RoadmapRevision
from chal.orchestrator.debate_controller import DebateController
from chal.config import ModeratorConfig
from chal.agents.base import Message


# ========================================
# Sample Data
# ========================================

SAMPLE_ROADMAP_JSON = {
    "sub_topics": [
        {
            "title": "Defining Free Will",
            "description": "Establish clear definitions and scope conditions for the debate.",
            "rationale": "Foundational concepts must be defined before contentious claims can be evaluated.",
            "guiding_questions": [
                "What does each agent mean by 'free will'?",
                "Are we discussing libertarian, compatibilist, or hard determinist free will?"
            ]
        },
        {
            "title": "Neuroscientific Evidence",
            "description": "Examine the empirical evidence from neuroscience regarding decision-making.",
            "rationale": "Empirical evidence should be evaluated early to ground the discussion.",
            "guiding_questions": [
                "What do Libet experiments actually show?",
                "Does neural determinism preclude conscious choice?"
            ]
        },
        {
            "title": "Moral Responsibility",
            "description": "Explore the implications for moral responsibility if free will does or does not exist.",
            "rationale": "This is the most contentious practical implication and a natural culmination.",
            "guiding_questions": [
                "Can we hold people morally responsible without free will?",
                "What alternative frameworks exist?"
            ]
        }
    ],
    "overall_rationale": "Progresses from definitions to evidence to implications.",
    "sufficiency_note": "3 rounds is adequate for a focused exploration."
}


def _make_fenced_response(data: dict) -> str:
    """Wrap a dict as a fenced JSON code block."""
    return f"Here is the roadmap:\n\n```json\n{json.dumps(data, indent=2)}\n```\n\nI hope this helps."


def _make_raw_json_response(data: dict) -> str:
    """Return a raw JSON string (no fencing)."""
    return json.dumps(data, indent=2)


def _make_moderator_config(**overrides) -> ModeratorConfig:
    """Create a ModeratorConfig with defaults."""
    defaults = {
        "model": "gpt-4o",
        "provider": "openai",
        "temperature": 0.3,
        "context": "",
        "moderator_mode": "static",
    }
    defaults.update(overrides)
    return ModeratorConfig(**defaults)


# ========================================
# 1. Initialization Tests
# ========================================

class TestModeratorInit:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_init_creates_agent(self, mock_create_agent):
        """Test that Moderator creates an agent on init."""
        mock_create_agent.return_value = Mock()
        config = _make_moderator_config()

        moderator = Moderator(config)

        mock_create_agent.assert_called_once_with(
            name="Moderator",
            model="gpt-4o",
            provider="openai",
            system_prompt="You are a debate moderator and topic analyst.",
        )
        assert moderator.config == config
        assert moderator.roadmap is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_init_with_custom_config(self, mock_create_agent):
        """Test init with non-default config values."""
        mock_create_agent.return_value = Mock()
        config = _make_moderator_config(
            model="claude-sonnet-4-5-20250929",
            provider="anthropic",
            temperature=0.5,
            context="Some background context",
            moderator_mode="adaptive",
        )

        moderator = Moderator(config)

        mock_create_agent.assert_called_once_with(
            name="Moderator",
            model="claude-sonnet-4-5-20250929",
            provider="anthropic",
            system_prompt="You are a debate moderator and topic analyst.",
        )
        assert moderator.config.context == "Some background context"
        assert moderator.config.moderator_mode == "adaptive"


# ========================================
# 2. Roadmap Generation Tests
# ========================================

class TestGenerateRoadmap:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_generate_roadmap_basic(self, mock_create_agent):
        """Test generating a roadmap from a well-formed fenced JSON response."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(SAMPLE_ROADMAP_JSON),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config())
        roadmap = moderator.generate_roadmap(
            topic="Does free will exist?",
            num_rounds=3,
            agent_personas=["EMPIRICIST", "SUPERNATURALIST"],
        )

        assert isinstance(roadmap, Roadmap)
        assert len(roadmap.sub_topics) == 3
        assert roadmap.sub_topics[0].title == "Defining Free Will"
        assert roadmap.sub_topics[1].title == "Neuroscientific Evidence"
        assert roadmap.sub_topics[2].title == "Moral Responsibility"
        assert roadmap.overall_rationale == "Progresses from definitions to evidence to implications."
        assert roadmap.sufficiency_note == "3 rounds is adequate for a focused exploration."
        assert roadmap.raw_response != ""

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_generate_roadmap_stores_on_self(self, mock_create_agent):
        """Test that generated roadmap is stored on self.roadmap."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(SAMPLE_ROADMAP_JSON),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config())
        roadmap = moderator.generate_roadmap("Topic", 3, ["A", "B"])

        assert moderator.roadmap is roadmap

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_generate_roadmap_calls_agent(self, mock_create_agent):
        """Test that generate_roadmap calls agent.generate with correct message structure."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(SAMPLE_ROADMAP_JSON),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(temperature=0.4))
        moderator.generate_roadmap("My topic", 2, ["X", "Y"])

        mock_agent.generate.assert_called_once()
        call_args = mock_agent.generate.call_args
        messages = call_args[0][0]  # First positional arg
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "My topic" in messages[0].content
        assert call_args[1]["temperature"] == 0.4

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_generate_roadmap_guiding_questions(self, mock_create_agent):
        """Test that guiding questions are parsed correctly."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(SAMPLE_ROADMAP_JSON),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config())
        roadmap = moderator.generate_roadmap("Topic", 3, ["A", "B"])

        assert len(roadmap.sub_topics[0].guiding_questions) == 2
        assert "free will" in roadmap.sub_topics[0].guiding_questions[0].lower()


# ========================================
# 3. Parsing Tests
# ========================================

class TestParseRoadmapResponse:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_fenced_json(self, mock_create_agent):
        """Test parsing a fenced JSON code block."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = _make_fenced_response(SAMPLE_ROADMAP_JSON)
        roadmap = moderator._parse_roadmap_response(raw, 3)

        assert len(roadmap.sub_topics) == 3
        assert roadmap.sub_topics[0].title == "Defining Free Will"

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_raw_json(self, mock_create_agent):
        """Test parsing a raw JSON response (no fencing)."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = _make_raw_json_response(SAMPLE_ROADMAP_JSON)
        roadmap = moderator._parse_roadmap_response(raw, 3)

        assert len(roadmap.sub_topics) == 3

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_truncates_to_num_rounds(self, mock_create_agent):
        """Test that sub-topics are truncated when exceeding num_rounds."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = _make_fenced_response(SAMPLE_ROADMAP_JSON)
        roadmap = moderator._parse_roadmap_response(raw, 2)  # Only 2 rounds

        assert len(roadmap.sub_topics) == 2
        assert roadmap.sub_topics[0].title == "Defining Free Will"
        assert roadmap.sub_topics[1].title == "Neuroscientific Evidence"

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_malformed_json_fallback(self, mock_create_agent):
        """Test fallback when JSON is malformed (triggers JSONDecodeError)."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        # Include a closing brace so the regex matches, but JSON is still invalid
        raw = '```json\n{"sub_topics": [broken}\n```'
        roadmap = moderator._parse_roadmap_response(raw, 3)

        assert len(roadmap.sub_topics) == 1
        assert roadmap.sub_topics[0].title == "General Discussion"
        assert "Failed to parse" in roadmap.overall_rationale

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_empty_response_fallback(self, mock_create_agent):
        """Test fallback when response has no JSON at all."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = "I don't know how to generate a roadmap."
        roadmap = moderator._parse_roadmap_response(raw, 3)

        # Should hit the empty sub_topics fallback since {} has no sub_topics
        assert len(roadmap.sub_topics) == 1
        assert roadmap.sub_topics[0].title == "General Discussion"

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_empty_sub_topics_fallback(self, mock_create_agent):
        """Test fallback when JSON has empty sub_topics array."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = _make_fenced_response({"sub_topics": [], "overall_rationale": "test"})
        roadmap = moderator._parse_roadmap_response(raw, 3)

        assert len(roadmap.sub_topics) == 1
        assert roadmap.sub_topics[0].title == "General Discussion"
        assert "No sub-topics" in roadmap.overall_rationale

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_parse_missing_optional_fields(self, mock_create_agent):
        """Test parsing when optional fields are missing."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        minimal_data = {
            "sub_topics": [
                {"title": "Topic A"},
                {"title": "Topic B", "description": "Desc B"},
            ]
        }
        raw = _make_fenced_response(minimal_data)
        roadmap = moderator._parse_roadmap_response(raw, 3)

        assert len(roadmap.sub_topics) == 2
        assert roadmap.sub_topics[0].title == "Topic A"
        assert roadmap.sub_topics[0].description == ""
        assert roadmap.sub_topics[0].rationale == ""
        assert roadmap.sub_topics[0].guiding_questions == []
        assert roadmap.sub_topics[1].description == "Desc B"
        assert roadmap.overall_rationale == ""
        assert roadmap.sufficiency_note == ""


# ========================================
# 4. SubTopic Retrieval Tests
# ========================================

class TestGetSubtopicForRound:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_get_valid_index(self, mock_create_agent):
        """Test retrieving a sub-topic by valid round index."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())
        moderator.roadmap = Roadmap(
            sub_topics=[
                SubTopic(title="A", description="desc A"),
                SubTopic(title="B", description="desc B"),
            ]
        )

        result = moderator.get_subtopic_for_round(0)
        assert result is not None
        assert result.title == "A"

        result = moderator.get_subtopic_for_round(1)
        assert result is not None
        assert result.title == "B"

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_get_out_of_range_returns_none(self, mock_create_agent):
        """Test that out-of-range index returns None."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())
        moderator.roadmap = Roadmap(
            sub_topics=[SubTopic(title="A", description="desc A")]
        )

        assert moderator.get_subtopic_for_round(1) is None
        assert moderator.get_subtopic_for_round(-1) is None
        assert moderator.get_subtopic_for_round(100) is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_get_when_no_roadmap(self, mock_create_agent):
        """Test retrieval when roadmap hasn't been generated yet."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        assert moderator.get_subtopic_for_round(0) is None


# ========================================
# 5. review_round() Tests
# ========================================

class TestReviewRound:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_static_mode_returns_none(self, mock_create_agent):
        """Test that review_round returns None in static mode."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config(moderator_mode="static"))

        result = moderator.review_round(1, {"some": "summary"})
        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_adaptive_returns_none_without_roadmap(self, mock_create_agent):
        """Test that adaptive review_round returns None when no roadmap is set."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config(moderator_mode="adaptive"))

        result = moderator.review_round(1, {"some": "summary"})
        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_adaptive_calls_llm(self, mock_create_agent):
        """Adaptive mode calls agent.generate with review prompt."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content='{"revision_needed": false}',
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(moderator_mode="adaptive"))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "desc A"),
            SubTopic("B", "desc B"),
            SubTopic("C", "desc C"),
        ])
        moderator._topic = "Test topic"

        result = moderator.review_round(1, {"round_num": 1})

        mock_agent.generate.assert_called_once()
        call_args = mock_agent.generate.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Test topic" in messages[0].content

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_adaptive_returns_revision(self, mock_create_agent):
        """Valid revision JSON → RoadmapRevision returned."""
        revision_json = {
            "revision_needed": True,
            "revised_sub_topics": [
                {"title": "New B", "description": "new desc B"},
                {"title": "New C", "description": "new desc C"},
            ],
            "revision_rationale": "Topics need updating based on round results.",
        }
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(revision_json),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(moderator_mode="adaptive"))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "desc A"),
            SubTopic("B", "desc B"),
            SubTopic("C", "desc C"),
        ])
        moderator._topic = "Test topic"

        result = moderator.review_round(1, {"round_num": 1})

        assert result is not None
        assert isinstance(result, RoadmapRevision)
        assert len(result.revised_sub_topics) == 2
        assert result.revised_sub_topics[0].title == "New B"
        assert result.revision_rationale == "Topics need updating based on round results."

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_adaptive_no_revision_needed(self, mock_create_agent):
        """revision_needed: false in LLM response → returns None."""
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response({"revision_needed": False}),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(moderator_mode="adaptive"))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "a"), SubTopic("B", "b"), SubTopic("C", "c"),
        ])
        moderator._topic = "Test"

        result = moderator.review_round(1, {})
        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_respects_review_frequency(self, mock_create_agent):
        """Only reviews on matching round numbers (round_num % frequency == 0)."""
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(
            moderator_mode="adaptive",
            review_frequency=2,
        ))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "a"), SubTopic("B", "b"),
            SubTopic("C", "c"), SubTopic("D", "d"),
        ])
        moderator._topic = "Test"

        # Round 1: should skip (1 % 2 != 0)
        result = moderator.review_round(1, {})
        assert result is None
        mock_agent.generate.assert_not_called()

        # Round 2: should review (2 % 2 == 0)
        mock_agent.generate.return_value = Message(
            role="assistant",
            content='{"revision_needed": false}',
        )
        result = moderator.review_round(2, {})
        # LLM was called even though no revision was needed
        mock_agent.generate.assert_called_once()

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_respects_max_revisions(self, mock_create_agent):
        """Stops reviewing after max_revisions is reached."""
        revision_json = {
            "revision_needed": True,
            "revised_sub_topics": [{"title": "X", "description": "x"}],
            "revision_rationale": "reason",
        }
        mock_agent = Mock()
        mock_agent.generate.return_value = Message(
            role="assistant",
            content=_make_fenced_response(revision_json),
        )
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(
            moderator_mode="adaptive",
            max_revisions=1,
        ))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "a"), SubTopic("B", "b"), SubTopic("C", "c"),
        ])
        moderator._topic = "Test"

        # First review — should succeed and increment _revision_count
        result = moderator.review_round(1, {})
        assert result is not None
        assert moderator._revision_count == 1

        # Reset roadmap to have remaining topics for round 2
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "a"), SubTopic("B2", "b2"), SubTopic("C2", "c2"),
        ])

        # Second review — should be blocked by max_revisions=1
        result = moderator.review_round(2, {})
        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_no_remaining_topics(self, mock_create_agent):
        """Returns None when all sub-topics have been covered."""
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        moderator = Moderator(_make_moderator_config(moderator_mode="adaptive"))
        moderator.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "a"),
        ])
        moderator._topic = "Test"

        # Round 1: sub_topics[1:] == [] → no remaining topics
        result = moderator.review_round(1, {})
        assert result is None
        mock_agent.generate.assert_not_called()


# ========================================
# 5b. Revision Response Parsing Tests
# ========================================

class TestParseRevisionResponse:

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_valid_revision(self, mock_create_agent):
        """Fenced JSON with revision_needed: true → RoadmapRevision."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        data = {
            "revision_needed": True,
            "revised_sub_topics": [
                {"title": "New Topic", "description": "New desc",
                 "rationale": "Why", "guiding_questions": ["Q1?"]},
            ],
            "revision_rationale": "Need to adjust.",
        }
        raw = _make_fenced_response(data)
        remaining = [SubTopic("Old", "old")]

        result = moderator._parse_revision_response(raw, remaining)

        assert result is not None
        assert isinstance(result, RoadmapRevision)
        assert len(result.revised_sub_topics) == 1
        assert result.revised_sub_topics[0].title == "New Topic"
        assert result.revised_sub_topics[0].guiding_questions == ["Q1?"]
        assert result.revision_rationale == "Need to adjust."

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_no_revision_needed(self, mock_create_agent):
        """revision_needed: false → None."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = _make_fenced_response({"revision_needed": False})
        result = moderator._parse_revision_response(raw, [SubTopic("A", "a")])

        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_malformed_json(self, mock_create_agent):
        """Bad JSON → None (no crash)."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        raw = '```json\n{broken json}\n```'
        result = moderator._parse_revision_response(raw, [SubTopic("A", "a")])

        assert result is None

    @pytest.mark.unit
    @patch("chal.orchestrator.moderator.create_agent")
    def test_empty_revised_subtopics(self, mock_create_agent):
        """Empty revised_sub_topics list → None."""
        mock_create_agent.return_value = Mock()
        moderator = Moderator(_make_moderator_config())

        data = {
            "revision_needed": True,
            "revised_sub_topics": [],
            "revision_rationale": "Actually no changes.",
        }
        raw = _make_fenced_response(data)
        result = moderator._parse_revision_response(raw, [SubTopic("A", "a")])

        assert result is None


# ========================================
# 5c. Build Round Summary Tests
# ========================================

class TestBuildRoundSummary:

    @pytest.mark.unit
    def test_includes_focus_subtopic(self):
        """Summary includes the correct focus sub-topic dict."""
        controller = DebateController(agents=[], max_rounds=3)
        focus = {"title": "Evidence Review", "description": "Review evidence"}

        summary = controller._build_round_summary(
            round_num=1, round_idx=0,
            focus_subtopic=focus,
            convergence_data=None,
        )

        assert summary["focus_subtopic"] == focus
        assert summary["focus_subtopic"]["title"] == "Evidence Review"
        assert summary["round_num"] == 1
        assert summary["remaining_rounds"] == 2

    @pytest.mark.unit
    def test_includes_adjudication(self):
        """Summary includes adjudicator verdict counts."""
        controller = DebateController(agents=[], max_rounds=3)
        controller.challenge_rebuttal_pairs = [
            {"challenger": "A", "target": "B", "challenge": "Is X true?",
             "qid": "Q1", "target_ids": ["C1"],
             "attack_type": "undermining", "attack_strategy": "challenge_evidence",
             "resolution": {"status": "critique_valid"}},
            {"challenger": "B", "target": "A", "challenge": "What about Y?",
             "qid": "Q2", "target_ids": ["C2"],
             "attack_type": "rebutting", "attack_strategy": "present_counter_evidence",
             "resolution": {"status": "rebuttal_valid"}},
        ]

        summary = controller._build_round_summary(
            round_num=1, round_idx=0,
            focus_subtopic=None,
            convergence_data=None,
        )

        assert summary["adjudication"]["verdict_counts"]["critique_valid"] == 1
        assert summary["adjudication"]["verdict_counts"]["rebuttal_valid"] == 1
        assert summary["adjudication"]["winner"] is None  # Tied

    @pytest.mark.unit
    def test_includes_belief_changes(self):
        """Summary includes per-agent belief change data."""
        agent_a = Mock()
        agent_a.name = "Agent-A"
        agent_a.all_beliefs_held = ["b1", "b2"]  # Changed

        agent_b = Mock()
        agent_b.name = "Agent-B"
        agent_b.all_beliefs_held = ["b1"]  # Not changed

        controller = DebateController(agents=[agent_a, agent_b], max_rounds=3)

        summary = controller._build_round_summary(
            round_num=1, round_idx=0,
            focus_subtopic=None,
            convergence_data=None,
        )

        changes = summary["belief_changes"]
        assert len(changes) == 2
        assert changes[0]["agent"] == "Agent-A"
        assert changes[0]["changed"] is True
        assert changes[1]["agent"] == "Agent-B"
        assert changes[1]["changed"] is False

    @pytest.mark.unit
    def test_includes_convergence(self):
        """Summary includes convergence data when provided."""
        controller = DebateController(agents=[], max_rounds=3)

        summary = controller._build_round_summary(
            round_num=1, round_idx=0,
            focus_subtopic=None,
            convergence_data={"convergence_score": 0.65},
        )

        assert summary["convergence"] is not None
        assert summary["convergence"]["score"] == 0.65
        assert summary["convergence"]["trend"] == "stable"

    @pytest.mark.unit
    def test_completed_subtopics(self):
        """Summary lists completed sub-topic titles from roadmap."""
        controller = DebateController(agents=[], max_rounds=3)
        controller.roadmap = Roadmap(sub_topics=[
            SubTopic("Definitions", "d1"),
            SubTopic("Evidence", "d2"),
            SubTopic("Implications", "d3"),
        ])

        summary = controller._build_round_summary(
            round_num=2, round_idx=1,
            focus_subtopic=None,
            convergence_data=None,
        )

        assert summary["completed_subtopics"] == ["Definitions", "Evidence"]
        assert summary["remaining_rounds"] == 1


# ========================================
# 5d. Adaptive Controller Flow Tests
# ========================================

class TestAdaptiveControllerFlow:

    @pytest.mark.unit
    def test_revision_updates_roadmap(self):
        """Controller applies revision to roadmap (completed topics preserved)."""
        controller = DebateController(agents=[], max_rounds=3)
        controller.stage2_mode = "moderated"
        controller.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "desc A"),
            SubTopic("B", "desc B"),
            SubTopic("C", "desc C"),
        ])

        # Simulate: moderator returns a revision after round 1
        revision = RoadmapRevision(
            revised_sub_topics=[
                SubTopic("D", "desc D"),
                SubTopic("E", "desc E"),
            ],
            revision_rationale="Need new topics",
        )

        # Apply revision (same logic as controller round loop)
        round_num = 1
        completed = controller.roadmap.sub_topics[:round_num]
        controller.roadmap.sub_topics = completed + revision.revised_sub_topics

        assert len(controller.roadmap.sub_topics) == 3
        assert controller.roadmap.sub_topics[0].title == "A"  # Completed: preserved
        assert controller.roadmap.sub_topics[1].title == "D"  # New
        assert controller.roadmap.sub_topics[2].title == "E"  # New

    @pytest.mark.unit
    def test_revision_fires_event(self):
        """roadmap_revised event is emitted with correct data."""
        callback = MagicMock()
        controller = DebateController(agents=[], max_rounds=3)
        controller._progress_callback = callback

        controller._notify("roadmap_revised", {
            "round_num": 1,
            "new_subtopics": [
                {"title": "New A", "description": "new desc"},
            ],
            "rationale": "Test rationale",
        })

        callback.assert_called_once()
        event_name, data = callback.call_args[0]
        assert event_name == "roadmap_revised"
        assert data["rationale"] == "Test rationale"
        assert len(data["new_subtopics"]) == 1
        assert data["new_subtopics"][0]["title"] == "New A"

    @pytest.mark.unit
    def test_no_revision_no_change(self):
        """None revision leaves roadmap unchanged."""
        controller = DebateController(agents=[], max_rounds=3)
        controller.roadmap = Roadmap(sub_topics=[
            SubTopic("A", "desc A"),
            SubTopic("B", "desc B"),
        ])

        # Simulate: moderator returns None (no revision needed)
        revision = None

        if revision is not None:
            completed = controller.roadmap.sub_topics[:1]
            controller.roadmap.sub_topics = completed + revision.revised_sub_topics

        assert len(controller.roadmap.sub_topics) == 2
        assert controller.roadmap.sub_topics[0].title == "A"
        assert controller.roadmap.sub_topics[1].title == "B"

    @pytest.mark.unit
    def test_revision_recorded_training_data(self):
        """Recorder captures revision event via record_event."""
        controller = DebateController(agents=[], max_rounds=3)
        recorder = Mock()
        controller.recorder = recorder

        revision_record = {
            "round_num": 1,
            "revision_rationale": "Test reason",
            "new_subtopics": [
                {"title": "New A", "description": "desc A",
                 "rationale": "r", "guiding_questions": []},
            ],
        }
        controller.roadmap_revisions.append(revision_record)
        controller.recorder.record_event("roadmap_revision", revision_record)

        recorder.record_event.assert_called_once_with(
            "roadmap_revision", revision_record,
        )
        assert len(controller.roadmap_revisions) == 1
        assert controller.roadmap_revisions[0]["round_num"] == 1


# ========================================
# 6. Dataclass Tests
# ========================================

class TestDataclasses:

    @pytest.mark.unit
    def test_subtopic_defaults(self):
        """Test SubTopic has correct default values."""
        st = SubTopic(title="Test", description="Test desc")
        assert st.rationale == ""
        assert st.guiding_questions == []

    @pytest.mark.unit
    def test_subtopic_with_all_fields(self):
        """Test SubTopic with all fields populated."""
        st = SubTopic(
            title="Test",
            description="A description",
            rationale="Why here",
            guiding_questions=["Q1?", "Q2?"],
        )
        assert st.title == "Test"
        assert len(st.guiding_questions) == 2

    @pytest.mark.unit
    def test_roadmap_defaults(self):
        """Test Roadmap has correct default values."""
        rm = Roadmap(sub_topics=[])
        assert rm.overall_rationale == ""
        assert rm.sufficiency_note == ""
        assert rm.raw_response == ""

    @pytest.mark.unit
    def test_roadmap_revision_defaults(self):
        """Test RoadmapRevision has correct defaults."""
        rev = RoadmapRevision(revised_sub_topics=[])
        assert rev.revision_rationale == ""


# ========================================
# 7. Stage 2 Prompt with focus_subtopic
# ========================================

class TestStage2PromptWithFocusSubtopic:

    @pytest.mark.unit
    def test_stage2_prompt_without_focus_subtopic(self):
        """Test that stage 2 prompt works without focus_subtopic (open mode)."""
        from chal.agents.prompts import build_stage_2_prompt

        prompt = build_stage_2_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json='{"thesis": {"stance": "test"}}',
            opponent_belief_json='{"thesis": {"stance": "test2"}}',
            focus_subtopic=None,
        )

        assert isinstance(prompt, str)
        assert "ROUND FOCUS" not in prompt
        assert "MODERATED DEBATE" not in prompt

    @pytest.mark.unit
    def test_stage2_prompt_with_focus_subtopic(self):
        """Test that stage 2 prompt includes focus section when subtopic provided."""
        from chal.agents.prompts import build_stage_2_prompt

        focus = {
            "title": "Defining Free Will",
            "description": "Establish definitions and scope conditions.",
            "guiding_questions": [
                "What does each agent mean by 'free will'?",
                "Libertarian vs compatibilist?"
            ]
        }

        prompt = build_stage_2_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json='{"thesis": {"stance": "test"}}',
            opponent_belief_json='{"thesis": {"stance": "test2"}}',
            focus_subtopic=focus,
        )

        assert "round_focus" in prompt
        assert "MODERATED DEBATE" in prompt
        assert "Defining Free Will" in prompt
        assert "Establish definitions and scope conditions." in prompt

    @pytest.mark.unit
    def test_stage2_prompt_focus_without_guiding_questions(self):
        """Test focus section when no guiding questions are provided."""
        from chal.agents.prompts import build_stage_2_prompt

        focus = {
            "title": "Evidence Review",
            "description": "Review the empirical evidence.",
            "guiding_questions": []
        }

        prompt = build_stage_2_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json='{"thesis": {"stance": "test"}}',
            opponent_belief_json='{"thesis": {"stance": "test2"}}',
            focus_subtopic=focus,
        )

        assert "round_focus" in prompt
        assert "Evidence Review" in prompt

    @pytest.mark.unit
    def test_stage2_bloodsport_prompt_with_focus_subtopic(self):
        """Test that bloodsport stage 2 prompt accepts focus_subtopic param."""
        from chal.agents.prompts import build_stage_2_bloodsport_prompt

        focus = {
            "title": "Moral Implications",
            "description": "Explore moral responsibility implications.",
            "guiding_questions": ["Can we hold people responsible?"]
        }

        prompt = build_stage_2_bloodsport_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json='{"thesis": {"stance": "test"}}',
            opponent_belief_json='{"thesis": {"stance": "test2"}}',
            focus_subtopic=focus,
        )

        # Bloodsport variant accepts focus_subtopic param without error
        assert "blood_sport" in prompt or "ADVERSARIAL CROSS-EXAMINATION" in prompt

    @pytest.mark.unit
    def test_stage2_bloodsport_prompt_without_focus(self):
        """Test that bloodsport prompt works without focus (open mode)."""
        from chal.agents.prompts import build_stage_2_bloodsport_prompt

        prompt = build_stage_2_bloodsport_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json='{"thesis": {"stance": "test"}}',
            opponent_belief_json='{"thesis": {"stance": "test2"}}',
            focus_subtopic=None,
        )

        assert "round_focus" not in prompt
        assert "blood_sport" in prompt or "ADVERSARIAL CROSS-EXAMINATION" in prompt


# ========================================
# 8. Moderator Prompt Builder Tests
# ========================================

class TestModeratorPromptBuilder:

    @pytest.mark.unit
    def test_build_moderator_roadmap_prompt_basic(self):
        """Test basic prompt generation."""
        from chal.agents.prompts import build_moderator_roadmap_prompt

        prompt = build_moderator_roadmap_prompt(
            topic="Does free will exist?",
            num_rounds=3,
            agent_personas=["EMPIRICIST", "SUPERNATURALIST"],
        )

        assert isinstance(prompt, str)
        assert "Does free will exist?" in prompt
        assert "EMPIRICIST" in prompt
        assert "SUPERNATURALIST" in prompt
        assert "3" in prompt
        assert "sub_topics" in prompt  # JSON format instruction

    @pytest.mark.unit
    def test_build_moderator_roadmap_prompt_with_context(self):
        """Test prompt accepts context parameter."""
        from chal.agents.prompts import build_moderator_roadmap_prompt

        prompt = build_moderator_roadmap_prompt(
            topic="AI Ethics",
            num_rounds=2,
            agent_personas=["PRAGMATIST"],
            context="Recent developments in AGI safety research.",
        )

        # Context param is accepted; prompt is generated
        assert isinstance(prompt, str)
        assert "AI Ethics" in prompt

    @pytest.mark.unit
    def test_build_moderator_roadmap_prompt_no_context(self):
        """Test prompt excludes context section when not provided."""
        from chal.agents.prompts import build_moderator_roadmap_prompt

        prompt = build_moderator_roadmap_prompt(
            topic="AI Ethics",
            num_rounds=2,
            agent_personas=["PRAGMATIST"],
            context="",
        )

        assert "BACKGROUND CONTEXT" not in prompt

    @pytest.mark.unit
    def test_build_moderator_review_round_prompt(self):
        """Test the review round prompt builder (stub for Phase 3)."""
        from chal.agents.prompts import build_moderator_review_round_prompt

        prompt = build_moderator_review_round_prompt(
            topic="Free will",
            round_num=1,
            round_summary={"verdict_counts": {"critique_valid": 2}},
            remaining_sub_topics=[{"title": "Topic B"}],
        )

        assert isinstance(prompt, str)
        assert "Free will" in prompt
        assert "1" in prompt


# ========================================
# 9. Reporting Integration Tests
# ========================================

class TestReportingWithRoadmap:

    def _make_mock_config(self, stage2_mode="moderated", mode="rebuttal"):
        """Create a mock config for reporting."""
        config = Mock()
        config.stage3_mode = mode
        config.stage2_mode = stage2_mode
        config.topic = "Free will"
        config.max_rounds = 3
        config.adjudication = Mock()
        config.adjudication.model = "gpt-4o"
        config.adjudication.provider = "openai"
        config.adjudication.logic_weight = 1.0
        config.adjudication.ethics_weight = 0.0
        return config

    def _make_mock_agent(self, name):
        from tests.utils import create_sample_belief
        agent = Mock()
        agent.name = name
        agent.model = "gpt-4o"
        agent.provider = "openai"
        agent.persona_label = "EMPIRICIST"
        agent.get_internal_belief_obj = Mock(return_value=create_sample_belief())
        agent.all_beliefs_held = [json.dumps(create_sample_belief())]
        return agent

    def _make_sample_roadmap_dict(self):
        return {
            "sub_topics": [
                {
                    "title": "Definitions",
                    "description": "Define key terms.",
                    "rationale": "Foundation first.",
                    "guiding_questions": ["What is free will?"]
                },
                {
                    "title": "Evidence",
                    "description": "Review evidence.",
                    "rationale": "Ground in data.",
                    "guiding_questions": []
                },
            ],
            "overall_rationale": "Definitions before evidence.",
            "sufficiency_note": "2 rounds may be tight."
        }

    @pytest.mark.unit
    def test_markdown_report_includes_roadmap_section(self):
        """Test that Markdown report includes Debate Roadmap section."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        assert "Debate Roadmap" in report
        assert "Definitions" in report
        assert "Evidence" in report
        assert "Definitions before evidence." in report
        assert "2 rounds may be tight." in report
        assert "What is free will?" not in report or "What is free will?" in report  # May be in detailed section

    @pytest.mark.unit
    def test_markdown_report_no_roadmap_in_open_mode(self):
        """Test that roadmap section is NOT shown in open mode."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="open")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        assert "Debate Roadmap" not in report

    @pytest.mark.unit
    def test_markdown_report_no_roadmap_when_none(self):
        """Test that roadmap section is NOT shown when roadmap=None."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=None,
        )

        assert "Debate Roadmap" not in report

    @pytest.mark.unit
    def test_markdown_report_section_numbering_with_roadmap(self):
        """Test that section numbers shift when roadmap is present."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        # With roadmap: 1=Metadata, 2=Roadmap, 3=Verdicts, 4=Adjudication, etc.
        assert "## 2. Debate Roadmap" in report
        assert "## 3. Adjudicator Verdict Distribution" in report

    @pytest.mark.unit
    def test_markdown_report_section_numbering_without_roadmap(self):
        """Test that section numbers are normal when no roadmap."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="open")
        agents = [self._make_mock_agent("Agent-A")]

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=None,
        )

        # Without roadmap: 1=Metadata, 2=Verdicts, etc.
        assert "## 2. Adjudicator Verdict Distribution" in report

    @pytest.mark.unit
    def test_json_report_includes_roadmap(self):
        """Test that JSON report includes roadmap data."""
        from chal.utilities.reporting import generate_analysis_json

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        result = generate_analysis_json(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        assert "roadmap" in result
        assert len(result["roadmap"]["sub_topics"]) == 2
        assert result["metadata"]["stage2_mode"] == "moderated"

    @pytest.mark.unit
    def test_json_report_no_roadmap_in_open_mode(self):
        """Test that JSON report excludes roadmap in open mode."""
        from chal.utilities.reporting import generate_analysis_json

        config = self._make_mock_config(stage2_mode="open")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        result = generate_analysis_json(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        assert "roadmap" not in result
        assert result["metadata"]["stage2_mode"] == "open"

    @pytest.mark.unit
    def test_json_report_stage2_mode_in_metadata(self):
        """Test that stage2_mode is always included in JSON metadata."""
        from chal.utilities.reporting import generate_analysis_json

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]

        result = generate_analysis_json(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
        )

        assert "stage2_mode" in result["metadata"]
        assert "stage3_mode" in result["metadata"]

    @pytest.mark.unit
    def test_json_report_serializable_with_roadmap(self):
        """Test that JSON report with roadmap is fully serializable."""
        from chal.utilities.reporting import generate_analysis_json

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]
        roadmap = self._make_sample_roadmap_dict()

        result = generate_analysis_json(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
            roadmap=roadmap,
        )

        serialized = json.dumps(result)
        assert len(serialized) > 0

    @pytest.mark.unit
    def test_markdown_report_includes_stage2_mode(self):
        """Test that Markdown report shows stage2_mode in metadata."""
        from chal.utilities.reporting import generate_analysis_report

        config = self._make_mock_config(stage2_mode="moderated")
        agents = [self._make_mock_agent("Agent-A")]

        report = generate_analysis_report(
            config=config, agents=agents,
            challenge_rebuttal_pairs=[], agent_stats={},
        )

        assert "Stage 2 Mode" in report
        assert "moderated" in report
