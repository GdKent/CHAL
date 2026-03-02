"""
Unit tests for the CHAL CLI roadmap review module (roadmap_review.py).

Tests cover:
- show_roadmap renders subtopics in a Rich panel (including overall_rationale)
- ask_roadmap_action returns correct action strings (mocked questionary)
- _ask_reorder reorders subtopics correctly
- _ask_add appends a new subtopic
- _ask_remove removes the selected subtopic
- _ask_edit modifies a subtopic's title and description
- _ask_adjust_rounds adjusts max_rounds
- run_roadmap_review approve flow exits immediately (returns tuple)
- run_roadmap_review regenerate calls moderator.generate_roadmap
- run_roadmap_review returns empty tuple when roadmap is None
- was_modified tracking
- KeyboardInterrupt handling
"""

import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from rich.console import Console

from chal.orchestrator.moderator import Moderator, SubTopic, Roadmap
from chal.cli.roadmap_review import (
    show_roadmap,
    ask_roadmap_action,
    _ask_reorder,
    _ask_add,
    _ask_remove,
    _ask_edit,
    _ask_adjust_rounds,
    run_roadmap_review,
)


# =========================================================================
# Helpers
# =========================================================================

def _console() -> Console:
    return Console(file=StringIO(), force_terminal=True, width=120)


def _make_subtopics(n: int = 3) -> list:
    """Create n sample SubTopic objects."""
    titles = ["Defining Free Will", "Neuroscientific Evidence", "Moral Responsibility",
              "Quantum Indeterminacy", "Compatibilism"]
    return [
        SubTopic(
            title=titles[i % len(titles)],
            description=f"Description for topic {i + 1}",
            rationale=f"Rationale {i + 1}",
            guiding_questions=[f"Question {i + 1}a?", f"Question {i + 1}b?"],
        )
        for i in range(n)
    ]


def _make_moderator(subtopics: list = None) -> MagicMock:
    """Create a mock Moderator with a roadmap."""
    moderator = MagicMock(spec=Moderator)
    if subtopics is None:
        subtopics = _make_subtopics(3)
    moderator.roadmap = Roadmap(
        sub_topics=subtopics,
        overall_rationale="Test rationale",
        sufficiency_note="3 rounds is adequate.",
    )
    return moderator


# =========================================================================
# 1. show_roadmap
# =========================================================================

class TestShowRoadmap:

    @pytest.mark.unit
    def test_renders_subtopic_titles(self):
        """show_roadmap renders each subtopic title."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(3)
        show_roadmap(console, subtopics)
        output = buf.getvalue()
        assert "Defining Free Will" in output
        assert "Neuroscientific Evidence" in output
        assert "Moral Responsibility" in output

    @pytest.mark.unit
    def test_renders_round_numbers(self):
        """show_roadmap numbers subtopics starting from 1."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(2)
        show_roadmap(console, subtopics)
        output = buf.getvalue()
        assert "1" in output
        assert "2" in output

    @pytest.mark.unit
    def test_renders_sufficiency_note(self):
        """show_roadmap includes sufficiency note as subtitle."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(1)
        show_roadmap(console, subtopics, sufficiency="Coverage is adequate")
        output = buf.getvalue()
        assert "Coverage is adequate" in output

    @pytest.mark.unit
    def test_renders_guiding_questions(self):
        """show_roadmap shows guiding questions if present."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(1)
        show_roadmap(console, subtopics)
        output = buf.getvalue()
        assert "Question 1a?" in output

    @pytest.mark.unit
    def test_truncates_long_descriptions(self):
        """show_roadmap truncates descriptions longer than 80 chars."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = [SubTopic(title="Long", description="X" * 100)]
        show_roadmap(console, subtopics)
        output = buf.getvalue()
        assert "..." in output

    @pytest.mark.unit
    def test_renders_overall_rationale(self):
        """show_roadmap displays the overall_rationale text."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(2)
        show_roadmap(console, subtopics, overall_rationale="Start with fundamentals first")
        output = buf.getvalue()
        assert "Rationale:" in output
        assert "Start with fundamentals first" in output

    @pytest.mark.unit
    def test_omits_rationale_when_empty(self):
        """show_roadmap omits rationale paragraph when overall_rationale is empty."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        subtopics = _make_subtopics(1)
        show_roadmap(console, subtopics, overall_rationale="")
        output = buf.getvalue()
        assert "Rationale:" not in output


# =========================================================================
# 2. ask_roadmap_action
# =========================================================================

class TestAskRoadmapAction:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_returns_approve(self, mock_q):
        """ask_roadmap_action returns 'approve' when selected."""
        mock_q.select.return_value.ask.return_value = "approve"
        assert ask_roadmap_action() == "approve"

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_returns_regenerate(self, mock_q):
        """ask_roadmap_action returns 'regenerate' when selected."""
        mock_q.select.return_value.ask.return_value = "regenerate"
        assert ask_roadmap_action() == "regenerate"

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_none_raises_keyboard_interrupt(self, mock_q):
        """ask_roadmap_action raises KeyboardInterrupt if user cancels."""
        mock_q.select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            ask_roadmap_action()


# =========================================================================
# 3. Reorder
# =========================================================================

class TestReorder:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_valid_reorder(self, mock_q):
        """_ask_reorder reorders subtopics with valid input."""
        mock_q.text.return_value.ask.return_value = "3,1,2"
        subtopics = _make_subtopics(3)
        original_titles = [st.title for st in subtopics]
        result = _ask_reorder(subtopics)
        result_titles = [st.title for st in result]
        assert result_titles == [original_titles[2], original_titles[0], original_titles[1]]

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_invalid_reorder_returns_unchanged(self, mock_q):
        """_ask_reorder returns unchanged list on invalid input."""
        mock_q.text.return_value.ask.return_value = "abc"
        subtopics = _make_subtopics(3)
        result = _ask_reorder(subtopics)
        assert [st.title for st in result] == [st.title for st in subtopics]

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_incomplete_reorder_returns_unchanged(self, mock_q):
        """_ask_reorder returns unchanged if not all indices provided."""
        mock_q.text.return_value.ask.return_value = "1,2"  # missing 3
        subtopics = _make_subtopics(3)
        result = _ask_reorder(subtopics)
        # Returns unchanged because len(reordered) != len(subtopics)
        assert [st.title for st in result] == [st.title for st in subtopics]

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_reorder_none_raises_keyboard_interrupt(self, mock_q):
        """_ask_reorder raises KeyboardInterrupt if user cancels."""
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            _ask_reorder(_make_subtopics(2))


# =========================================================================
# 4. Add
# =========================================================================

class TestAdd:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_add_appends_subtopic(self, mock_q):
        """_ask_add appends a new SubTopic to the list."""
        mock_q.text.return_value.ask.side_effect = ["New Topic", "New description"]
        subtopics = _make_subtopics(2)
        result = _ask_add(subtopics)
        assert len(result) == 3
        assert result[-1].title == "New Topic"
        assert result[-1].description == "New description"

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_add_none_title_raises_keyboard_interrupt(self, mock_q):
        """_ask_add raises KeyboardInterrupt if title is cancelled."""
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            _ask_add(_make_subtopics(1))


# =========================================================================
# 5. Remove
# =========================================================================

class TestRemove:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_remove_pops_selected(self, mock_q):
        """_ask_remove removes the selected subtopic."""
        mock_q.select.return_value.ask.return_value = 1  # index 1 (second item)
        subtopics = _make_subtopics(3)
        original_titles = [st.title for st in subtopics]
        result = _ask_remove(subtopics)
        assert len(result) == 2
        assert original_titles[1] not in [st.title for st in result]

    @pytest.mark.unit
    def test_remove_single_item_returns_unchanged(self):
        """_ask_remove returns unchanged list when only 1 subtopic."""
        subtopics = _make_subtopics(1)
        result = _ask_remove(subtopics)
        assert len(result) == 1

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_remove_none_raises_keyboard_interrupt(self, mock_q):
        """_ask_remove raises KeyboardInterrupt if user cancels."""
        mock_q.select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            _ask_remove(_make_subtopics(3))


# =========================================================================
# 6. Edit
# =========================================================================

class TestEdit:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_edit_modifies_subtopic(self, mock_q):
        """_ask_edit updates the title and description of the selected subtopic."""
        mock_q.select.return_value.ask.return_value = 0  # first item
        # First call = title, second call = description
        mock_q.text.return_value.ask.side_effect = ["Edited Title", "Edited Desc"]
        subtopics = _make_subtopics(2)
        result = _ask_edit(subtopics)
        assert result[0].title == "Edited Title"
        assert result[0].description == "Edited Desc"
        # Rationale and guiding questions preserved
        assert result[0].rationale == subtopics[0].rationale
        assert result[0].guiding_questions == subtopics[0].guiding_questions

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_edit_none_select_raises_keyboard_interrupt(self, mock_q):
        """_ask_edit raises KeyboardInterrupt if selection is cancelled."""
        mock_q.select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            _ask_edit(_make_subtopics(2))


# =========================================================================
# 7. Adjust Rounds
# =========================================================================

class TestAdjustRounds:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_returns_suggested_count(self, mock_q):
        """_ask_adjust_rounds returns the number the user enters."""
        mock_q.text.return_value.ask.return_value = "5"
        result = _ask_adjust_rounds(_make_subtopics(3), current_rounds=3)
        assert result == 5

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_returns_default_on_enter(self, mock_q):
        """_ask_adjust_rounds defaults to subtopic count when user presses Enter."""
        mock_q.text.return_value.ask.return_value = "3"
        result = _ask_adjust_rounds(_make_subtopics(3), current_rounds=4)
        assert result == 3

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_invalid_input_returns_current(self, mock_q):
        """_ask_adjust_rounds returns current_rounds on invalid input."""
        mock_q.text.return_value.ask.return_value = "abc"
        result = _ask_adjust_rounds(_make_subtopics(3), current_rounds=4)
        assert result == 4

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_zero_returns_current(self, mock_q):
        """_ask_adjust_rounds returns current_rounds if user enters 0."""
        mock_q.text.return_value.ask.return_value = "0"
        result = _ask_adjust_rounds(_make_subtopics(3), current_rounds=4)
        assert result == 4

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.questionary")
    def test_none_raises_keyboard_interrupt(self, mock_q):
        """_ask_adjust_rounds raises KeyboardInterrupt if user cancels."""
        mock_q.text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            _ask_adjust_rounds(_make_subtopics(2), current_rounds=3)


# =========================================================================
# 8. run_roadmap_review
# =========================================================================

class TestRunRoadmapReview:

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_approve_exits_immediately(self, mock_action):
        """run_roadmap_review returns subtopics on immediate approve."""
        mock_action.return_value = "approve"
        moderator = _make_moderator()
        subtopics, num_rounds, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test topic",
            num_rounds=3,
            agent_personas=["EMPIRICIST", "RATIONALIST"],
        )
        assert len(subtopics) == 3
        assert subtopics[0].title == "Defining Free Will"
        assert num_rounds == 3
        assert was_modified is False

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_approve_updates_moderator_roadmap(self, mock_action):
        """run_roadmap_review writes back to moderator.roadmap.sub_topics."""
        mock_action.return_value = "approve"
        moderator = _make_moderator()
        subtopics, _, _ = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert moderator.roadmap.sub_topics == subtopics

    @pytest.mark.unit
    def test_returns_empty_when_no_roadmap(self):
        """run_roadmap_review returns empty tuple if moderator.roadmap is None."""
        moderator = MagicMock(spec=Moderator)
        moderator.roadmap = None
        subtopics, num_rounds, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert subtopics == []
        assert num_rounds == 3
        assert was_modified is False

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_regenerate_calls_moderator(self, mock_action):
        """run_roadmap_review calls moderator.generate_roadmap on regenerate."""
        # First call: regenerate, second call: approve
        mock_action.side_effect = ["regenerate", "approve"]

        moderator = _make_moderator()
        new_subtopics = _make_subtopics(2)
        moderator.generate_roadmap.return_value = Roadmap(
            sub_topics=new_subtopics,
            sufficiency_note="Revised",
        )

        subtopics, _, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Free will",
            num_rounds=2,
            agent_personas=["EMPIRICIST"],
        )
        moderator.generate_roadmap.assert_called_once_with(
            topic="Free will",
            num_rounds=2,
            agent_personas=["EMPIRICIST"],
        )
        assert len(subtopics) == 2
        assert was_modified is True

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_add")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_add_then_approve(self, mock_action, mock_add):
        """run_roadmap_review supports add action before approve."""
        mock_action.side_effect = ["add", "approve"]
        original_subtopics = _make_subtopics(2)
        added = original_subtopics + [SubTopic(title="New", description="New desc")]
        mock_add.return_value = added

        moderator = _make_moderator(subtopics=_make_subtopics(2))
        subtopics, _, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert len(subtopics) == 3
        assert subtopics[-1].title == "New"
        assert was_modified is True

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_remove")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_remove_then_approve(self, mock_action, mock_remove):
        """run_roadmap_review supports remove action before approve."""
        mock_action.side_effect = ["remove", "approve"]
        remaining = _make_subtopics(3)[:2]
        mock_remove.return_value = remaining

        moderator = _make_moderator()
        subtopics, _, _ = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert len(subtopics) == 2

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_reorder")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_reorder_then_approve(self, mock_action, mock_reorder):
        """run_roadmap_review supports reorder action before approve."""
        mock_action.side_effect = ["reorder", "approve"]
        subtopics = _make_subtopics(3)
        reordered = [subtopics[2], subtopics[0], subtopics[1]]
        mock_reorder.return_value = reordered

        moderator = _make_moderator()
        result_subtopics, _, _ = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert result_subtopics[0].title == subtopics[2].title

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_edit")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_edit_then_approve(self, mock_action, mock_edit):
        """run_roadmap_review supports edit action before approve."""
        mock_action.side_effect = ["edit", "approve"]
        subtopics = _make_subtopics(3)
        edited = list(subtopics)
        edited[0] = SubTopic(title="Edited", description="New desc")
        mock_edit.return_value = edited

        moderator = _make_moderator()
        result_subtopics, _, _ = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert result_subtopics[0].title == "Edited"

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_adjust_rounds")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_adjust_rounds_then_approve(self, mock_action, mock_adjust):
        """run_roadmap_review supports adjust_rounds action before approve."""
        mock_action.side_effect = ["adjust_rounds", "approve"]
        mock_adjust.return_value = 5

        moderator = _make_moderator()
        subtopics, num_rounds, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert num_rounds == 5
        assert was_modified is True
        assert len(subtopics) == 3

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_was_modified_false_on_direct_approve(self, mock_action):
        """Direct approve results in was_modified=False."""
        mock_action.return_value = "approve"
        moderator = _make_moderator()
        _, _, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert was_modified is False

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review._ask_edit")
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_was_modified_true_after_edit(self, mock_action, mock_edit):
        """Editing before approve results in was_modified=True."""
        mock_action.side_effect = ["edit", "approve"]
        mock_edit.return_value = _make_subtopics(3)

        moderator = _make_moderator()
        _, _, was_modified = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert was_modified is True

    @pytest.mark.unit
    @patch("chal.cli.roadmap_review.ask_roadmap_action")
    def test_return_type_is_tuple(self, mock_action):
        """run_roadmap_review returns a 3-tuple."""
        mock_action.return_value = "approve"
        moderator = _make_moderator()
        result = run_roadmap_review(
            console=_console(),
            moderator=moderator,
            topic="Test",
            num_rounds=3,
            agent_personas=[],
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
