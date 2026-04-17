"""
Unit tests for the CHAL CLI display module (display.py).

Tests cover:
- DebateDisplay.handle_event dispatches to correct handler methods
- Unknown events are silently ignored
- debate_start renders panel with topic, agents, rounds
- stage_start renders stage header with icon and name
- agent_complete renders agent status line
- adjudication_result collects results for table display
- round_start resets adjudication collector
- round_complete shows adjudication table and performance scores
- debate_complete stops progress bar and shows completion panel
- Adjudication table content and styling
- Performance table content
- Verbose vs non-verbose agent_start behavior
"""

import pytest
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from chal.cli.display import DebateDisplay, _STAGE_NAMES, _STAGE_ICONS, _convergence_label


# =========================================================================
# Helpers
# =========================================================================

def _make_display(verbose: bool = False, num_rounds: int = 3, num_agents: int = 2, interactive: bool = True) -> tuple:
    """Create a DebateDisplay with a captured console and return both."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    display = DebateDisplay(console=console, num_rounds=num_rounds, num_agents=num_agents, verbose=verbose, interactive=interactive)
    return display, buf


def _get_output(buf: StringIO) -> str:
    """Get all output written to the console buffer."""
    return buf.getvalue()


# =========================================================================
# 1. Event Dispatch
# =========================================================================

class TestEventDispatch:

    @pytest.mark.unit
    def test_known_event_dispatches(self):
        """handle_event calls the correct _on_<event> method."""
        display, buf = _make_display()
        # Manually call a known event
        display.handle_event("stage_start", {"stage": 0, "name": "Briefing"})
        output = _get_output(buf)
        assert "Stage 0" in output
        assert "Briefing" in output

    @pytest.mark.unit
    def test_unknown_event_does_not_crash(self):
        """handle_event silently ignores unknown event names."""
        display, buf = _make_display()
        # Should not raise
        display.handle_event("totally_unknown_event", {"foo": "bar"})

    @pytest.mark.unit
    def test_empty_data_does_not_crash(self):
        """handle_event works with empty data dict."""
        display, buf = _make_display()
        display.handle_event("stage_start", {})
        # Should produce some output without crashing
        output = _get_output(buf)
        assert "Stage" in output


# =========================================================================
# 2. Debate Start
# =========================================================================

class TestDebateStart:

    @pytest.mark.unit
    def test_debate_start_shows_topic(self):
        """debate_start event renders the topic in a panel."""
        display, buf = _make_display()
        display.handle_event("debate_start", {
            "topic": "Does free will exist?",
            "num_agents": 2,
            "num_rounds": 3,
        })
        output = _get_output(buf)
        assert "Does free will exist?" in output
        assert "Debate Starting" in output

    @pytest.mark.unit
    def test_debate_start_shows_agent_count(self):
        """debate_start shows number of agents and rounds."""
        display, buf = _make_display()
        display.handle_event("debate_start", {
            "topic": "Test",
            "num_agents": 4,
            "num_rounds": 5,
        })
        output = _get_output(buf)
        assert "4 agents" in output
        assert "5 round(s)" in output

    @pytest.mark.unit
    def test_debate_start_creates_progress_bar(self):
        """debate_start initializes the progress bar."""
        display, buf = _make_display(num_rounds=5)
        assert display._progress is None
        display.handle_event("debate_start", {"topic": "Test", "num_agents": 2, "num_rounds": 5})
        assert display._progress is not None
        assert display._round_task_id is not None


# =========================================================================
# 3. Stage Events
# =========================================================================

class TestStageEvents:

    @pytest.mark.unit
    def test_stage_start_shows_name(self):
        """stage_start renders stage number and name."""
        display, buf = _make_display()
        display.handle_event("stage_start", {"stage": 2, "name": "Cross-Examination"})
        output = _get_output(buf)
        assert "Stage 2" in output
        assert "Cross-Examination" in output

    @pytest.mark.unit
    def test_stage_start_falls_back_to_lookup(self):
        """stage_start uses _STAGE_NAMES if name not in data."""
        display, buf = _make_display()
        display.handle_event("stage_start", {"stage": 1})
        output = _get_output(buf)
        assert "Opening Positions" in output

    @pytest.mark.unit
    def test_stage_start_shows_extra(self):
        """stage_start appends extra info if provided."""
        display, buf = _make_display()
        display.handle_event("stage_start", {"stage": 2, "name": "Cross-Exam", "extra": "Focus: evidence"})
        output = _get_output(buf)
        assert "Focus: evidence" in output

    @pytest.mark.unit
    def test_stage_complete_shows_completion(self):
        """stage_complete shows completion message."""
        display, buf = _make_display()
        display.handle_event("stage_complete", {"stage": 3, "name": "Rebuttals"})
        output = _get_output(buf)
        assert "Stage 3" in output
        assert "complete" in output


# =========================================================================
# 4. Agent Events
# =========================================================================

class TestAgentEvents:

    @pytest.mark.unit
    def test_agent_complete_shows_name(self):
        """agent_complete always shows agent name and action."""
        display, buf = _make_display(verbose=False)
        display.handle_event("agent_complete", {
            "agent_name": "Agent-Empiricist",
            "action": "Opening statement received",
        })
        output = _get_output(buf)
        assert "Agent-Empiricist" in output
        assert "Opening statement received" in output

    @pytest.mark.unit
    def test_agent_start_hidden_when_not_verbose(self):
        """agent_start produces no output when verbose=False."""
        display, buf = _make_display(verbose=False)
        display.handle_event("agent_start", {
            "agent_name": "Agent-X",
            "action": "Generating response",
        })
        output = _get_output(buf)
        assert "Agent-X" not in output

    @pytest.mark.unit
    def test_agent_start_shown_when_verbose(self):
        """agent_start shows agent info when verbose=True."""
        display, buf = _make_display(verbose=True)
        display.handle_event("agent_start", {
            "agent_name": "Agent-X",
            "action": "Generating response",
        })
        output = _get_output(buf)
        assert "Agent-X" in output


# =========================================================================
# 5. Adjudication Events
# =========================================================================

class TestAdjudicationEvents:

    @pytest.mark.unit
    def test_adjudication_result_collects(self):
        """adjudication_result appends to the internal collector."""
        display, buf = _make_display()
        display.handle_event("adjudication_result", {
            "challenger": "A", "target": "B", "outcome": "Sustained",
        })
        display.handle_event("adjudication_result", {
            "challenger": "B", "target": "A", "outcome": "Overruled",
        })
        assert len(display._round_adjudications) == 2

    @pytest.mark.unit
    def test_adjudication_verbose_shows_detail(self):
        """adjudication_result prints detail when verbose."""
        display, buf = _make_display(verbose=True)
        display.handle_event("adjudication_result", {
            "challenger": "Agent-A", "target": "Agent-B", "outcome": "Sustained",
        })
        output = _get_output(buf)
        assert "Agent-A" in output
        assert "Agent-B" in output
        assert "Sustained" in output

    @pytest.mark.unit
    def test_adjudication_nonverbose_no_inline_detail(self):
        """adjudication_result does not print inline when not verbose."""
        display, buf = _make_display(verbose=False)
        display.handle_event("adjudication_result", {
            "challenger": "Agent-A", "target": "Agent-B", "outcome": "Sustained",
        })
        output = _get_output(buf)
        # Non-verbose: no inline detail (only table at round_complete)
        assert "Agent-A ->" not in output


# =========================================================================
# 6. Round Events
# =========================================================================

class TestRoundEvents:

    @pytest.mark.unit
    def test_round_start_shows_round_number(self):
        """round_start shows round number and total."""
        display, buf = _make_display()
        display.handle_event("round_start", {"round": 2, "total_rounds": 5})
        output = _get_output(buf)
        assert "Round 2 of 5" in output

    @pytest.mark.unit
    def test_round_start_resets_adjudications(self):
        """round_start clears the adjudication collector."""
        display, buf = _make_display()
        display._round_adjudications = [{"a": "b"}]
        display.handle_event("round_start", {"round": 1, "total_rounds": 3})
        assert display._round_adjudications == []

    @pytest.mark.unit
    def test_round_complete_shows_adjudication_table(self):
        """round_complete renders adjudication table if results exist."""
        display, buf = _make_display()
        display._round_adjudications = [
            {"challenger": "A", "target": "B", "outcome": "Sustained"},
            {"challenger": "B", "target": "A", "outcome": "Overruled"},
        ]
        display.handle_event("round_complete", {"round": 1})
        output = _get_output(buf)
        assert "Adjudication Results" in output
        assert "Sustained" in output
        assert "Overruled" in output

    @pytest.mark.unit
    def test_round_complete_shows_performance_scores(self):
        """round_complete renders performance table when scores provided."""
        display, buf = _make_display()
        display.handle_event("round_complete", {
            "round": 1,
            "scores": {
                "Agent-A": {"performance_score": 7.2, "sustained": 1, "overruled": 0},
                "Agent-B": {"performance_score": 6.8, "sustained": 0, "overruled": 1},
            },
        })
        output = _get_output(buf)
        assert "Performance Scores" in output
        assert "Agent-A" in output
        assert "7.2" in output

    @pytest.mark.unit
    def test_round_complete_shows_convergence(self):
        """round_complete shows convergence score when provided."""
        display, buf = _make_display()
        display.handle_event("round_complete", {
            "round": 1,
            "convergence": {"convergence_score": 0.72},
        })
        output = _get_output(buf)
        assert "0.72" in output

    @pytest.mark.unit
    def test_round_complete_no_convergence_when_missing(self):
        """round_complete does not show convergence when not provided."""
        display, buf = _make_display()
        display.handle_event("round_complete", {"round": 1})
        output = _get_output(buf)
        assert "Convergence" not in output


# =========================================================================
# 7. Debate Complete
# =========================================================================

class TestDebateComplete:

    @pytest.mark.unit
    def test_debate_complete_shows_panel(self):
        """debate_complete shows the completion panel."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {})
        output = _get_output(buf)
        assert "Debate complete!" in output

    @pytest.mark.unit
    def test_debate_complete_stops_progress(self):
        """debate_complete stops and clears the progress bar."""
        display, buf = _make_display()
        # Start progress first
        display.handle_event("debate_start", {"topic": "T", "num_agents": 2, "num_rounds": 3})
        assert display._progress is not None
        display.handle_event("debate_complete", {})
        assert display._progress is None


# =========================================================================
# 8. Table Helpers
# =========================================================================

class TestTableHelpers:

    @pytest.mark.unit
    def test_adjudication_table_sustained_and_overruled(self):
        """_show_adjudication_table renders both outcomes."""
        display, buf = _make_display()
        results = [
            {"challenger": "Alice", "target": "Bob", "outcome": "Sustained"},
            {"challenger": "Bob", "target": "Alice", "outcome": "Overruled"},
        ]
        display._show_adjudication_table(results)
        output = _get_output(buf)
        assert "Alice" in output
        assert "Bob" in output
        assert "Sustained" in output
        assert "Overruled" in output

    @pytest.mark.unit
    def test_performance_table_renders_scores(self):
        """_show_performance_table renders agent scores."""
        display, buf = _make_display()
        stats = {
            "Agent-A": {"performance_score": 8.5, "sustained": 2, "overruled": 1},
            "Agent-B": {"performance_score": 6.0, "sustained": 0, "overruled": 2},
        }
        display._show_performance_table(stats)
        output = _get_output(buf)
        assert "Agent-A" in output
        assert "8.5" in output
        assert "Agent-B" in output
        assert "6.0" in output



# =========================================================================
# 9. Stage Names and Icons
# =========================================================================

class TestStageMetadata:

    @pytest.mark.unit
    def test_all_stages_have_names(self):
        """All 6 stages (0-5) have entries in _STAGE_NAMES."""
        for i in range(6):
            assert i in _STAGE_NAMES

    @pytest.mark.unit
    def test_all_stages_have_icons(self):
        """All 6 stages (0-5) have entries in _STAGE_ICONS."""
        for i in range(6):
            assert i in _STAGE_ICONS


# =========================================================================
# 10. Post-Debate Summary (Phase 3)
# =========================================================================

class TestPostDebateSummary:

    @pytest.mark.unit
    def test_debate_complete_shows_duration(self):
        """debate_complete shows duration when total_duration_s is provided."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {"total_duration_s": 272})
        output = _get_output(buf)
        assert "4m 32s" in output

    @pytest.mark.unit
    def test_debate_complete_shows_topic(self):
        """debate_complete shows topic when provided."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {"topic": "Does free will exist?"})
        output = _get_output(buf)
        assert "Does free will exist?" in output

    @pytest.mark.unit
    def test_debate_complete_shows_agent_stats(self):
        """debate_complete renders performance table from agent_stats."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {
            "agent_stats": {
                "Agent-A": {"performance_score": 7.5, "sustained": 2, "overruled": 1},
                "Agent-B": {"performance_score": 5.0, "sustained": 1, "overruled": 2},
            },
        })
        output = _get_output(buf)
        assert "Agent-A" in output
        assert "7.5" in output

    @pytest.mark.unit
    def test_debate_complete_shows_convergence(self):
        """debate_complete shows convergence from convergence_history."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {
            "convergence_history": [
                {"convergence_score": 0.45},
                {"convergence_score": 0.72},
            ],
        })
        output = _get_output(buf)
        assert "0.72" in output
        assert "moderate agreement" in output

    @pytest.mark.unit
    def test_debate_complete_minimal_data(self):
        """debate_complete works with empty data dict."""
        display, buf = _make_display()
        display.handle_event("debate_complete", {})
        output = _get_output(buf)
        assert "Debate complete!" in output

    @pytest.mark.unit
    def test_output_files_saved_event(self):
        """output_files_saved event shows file names."""
        display, buf = _make_display()
        display.handle_event("output_files_saved", {
            "files": ["transcript.txt", "final_beliefs.txt", "stats.json"],
        })
        output = _get_output(buf)
        assert "transcript.txt" in output
        assert "final_beliefs.txt" in output
        assert "stats.json" in output


# =========================================================================
# 11. Convergence Label Helper
# =========================================================================

class TestConvergenceLabel:

    @pytest.mark.unit
    def test_strong_agreement(self):
        assert _convergence_label(0.90) == "strong agreement"

    @pytest.mark.unit
    def test_moderate_agreement(self):
        assert _convergence_label(0.65) == "moderate agreement"

    @pytest.mark.unit
    def test_partial_agreement(self):
        assert _convergence_label(0.40) == "partial agreement"

    @pytest.mark.unit
    def test_divergent(self):
        assert _convergence_label(0.20) == "divergent"


# =========================================================================
# 12. Error Handler (Phase 3)
# =========================================================================

class TestErrorHandler:

    @pytest.mark.unit
    def test_headless_retries_once(self):
        """Headless mode retries on first error."""
        display, buf = _make_display(interactive=False)
        action = display.handle_error("Agent-A", RuntimeError("rate limit"), retry_count=0)
        assert action == "retry"

    @pytest.mark.unit
    def test_headless_aborts_after_retry(self):
        """Headless mode aborts after first retry fails."""
        display, buf = _make_display(interactive=False)
        action = display.handle_error("Agent-A", RuntimeError("rate limit"), retry_count=1)
        assert action == "abort"

    @pytest.mark.unit
    def test_error_panel_shows_agent_name(self):
        """Error handler shows agent name and error in output."""
        display, buf = _make_display(interactive=False)
        display.handle_error("Agent-X", ValueError("bad response"), retry_count=0)
        output = _get_output(buf)
        assert "Agent-X" in output
        assert "bad response" in output

    @pytest.mark.unit
    @patch("questionary.select")
    def test_interactive_returns_user_choice(self, mock_select):
        """Interactive mode returns user's selected action."""
        mock_select.return_value.ask.return_value = "skip"
        display, buf = _make_display(interactive=True)
        action = display.handle_error("Agent-A", RuntimeError("error"), retry_count=0)
        assert action == "skip"

    @pytest.mark.unit
    @patch("questionary.select")
    def test_interactive_abort_on_ctrl_c(self, mock_select):
        """Interactive mode returns 'abort' when user presses Ctrl+C."""
        mock_select.return_value.ask.return_value = None
        display, buf = _make_display(interactive=True)
        action = display.handle_error("Agent-A", RuntimeError("error"), retry_count=0)
        assert action == "abort"


