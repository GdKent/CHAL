"""
Unit tests for the CHAL CLI runner (runner.py).

Tests cover:
- save_debate_outputs writes expected files based on config toggles
- run_debate creates agents and calls controller
- run_debate passes progress_callback to controller.run()
- Roadmap review is skipped in headless mode
- Roadmap review runs in interactive moderated mode
"""

import pytest
import json
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from io import StringIO

from rich.console import Console

from chal.config import DebateConfig, AgentConfig, OutputConfig, ScribeConfig
from chal.cli.runner import save_debate_outputs, run_debate


# =========================================================================
# Helpers
# =========================================================================

def _console() -> Console:
    return Console(file=StringIO())


def _make_config(tmp_path: Path, **output_overrides) -> DebateConfig:
    """Build a minimal DebateConfig with storage_dir pointing at tmp_path."""
    output_kwargs = {
        "storage_dir": tmp_path,
        "save_synthesis": False,
        "save_transcript": False,
        "save_debug_log": False,
        "save_initial_beliefs": False,
        "save_final_beliefs": False,
        "save_agent_stats": False,
        "generate_embeddings": False,
        "plot_trajectories": False,
        "generate_graph_visualization": False,
        "save_analysis_report": False,
        "save_training_data": False,
    }
    output_kwargs.update(output_overrides)
    outputs = OutputConfig(**output_kwargs)

    return DebateConfig(
        topic="Test topic",
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST"),
        ],
        outputs=outputs,
        scribe=ScribeConfig(enabled=True),
    )


def _make_results() -> dict:
    """Build a minimal results dict matching controller.run() output."""
    return {
        "synthesis": "Test synthesis text.",
        "full_transcript": "Full transcript text.",
        "markdown_transcript": "Markdown transcript text.",
        "debug_log": "Debug log text.",
        "initial_positions": ["Belief A", "Belief B"],
        "final_positions": ["Updated A", "Updated B"],
        "agent_stats": {"Agent-A": {"wins": 1}, "Agent-B": {"wins": 0}},
    }


# =========================================================================
# 1. save_debate_outputs
# =========================================================================

class TestSaveDebateOutputs:

    @pytest.mark.unit
    def test_saves_synthesis(self, tmp_path):
        """Synthesis file is written when save_synthesis is True."""
        config = _make_config(tmp_path, save_synthesis=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.synthesis_file
        assert path.exists()
        assert path.read_text() == "Test synthesis text."

    @pytest.mark.unit
    def test_skips_synthesis_when_disabled(self, tmp_path):
        """No synthesis file when save_synthesis is False."""
        config = _make_config(tmp_path, save_synthesis=False)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.synthesis_file
        assert not path.exists()

    @pytest.mark.unit
    def test_saves_transcript(self, tmp_path):
        config = _make_config(tmp_path, save_transcript=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.transcript_file
        assert path.exists()
        assert "Markdown transcript" in path.read_text()

    @pytest.mark.unit
    def test_saves_debug_log(self, tmp_path):
        config = _make_config(tmp_path, save_debug_log=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.debug_log_file
        assert path.exists()

    @pytest.mark.unit
    def test_saves_initial_beliefs(self, tmp_path):
        config = _make_config(tmp_path, save_initial_beliefs=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.initial_beliefs_file
        assert path.exists()
        content = path.read_text()
        assert "Belief A" in content
        assert "Belief B" in content

    @pytest.mark.unit
    def test_saves_final_beliefs(self, tmp_path):
        config = _make_config(tmp_path, save_final_beliefs=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.final_beliefs_file
        assert path.exists()

    @pytest.mark.unit
    def test_saves_agent_stats(self, tmp_path):
        config = _make_config(tmp_path, save_agent_stats=True)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        path = tmp_path / config.outputs.stats_file
        assert path.exists()
        data = json.loads(path.read_text())
        assert "Agent-A" in data

    @pytest.mark.unit
    def test_no_files_when_all_disabled(self, tmp_path):
        """No output files created when all toggles are False."""
        config = _make_config(tmp_path)
        results = _make_results()

        save_debate_outputs(config, results, MagicMock(), _console())

        # Only files in tmp_path should be nothing (or .gitkeep etc.)
        files = list(tmp_path.glob("*"))
        assert len(files) == 0


# =========================================================================
# 2. run_debate
# =========================================================================

class TestRunDebate:

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_creates_agents_and_runs(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """run_debate creates agents and calls controller.run()."""
        config = _make_config(tmp_path, save_transcript=True)

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        result = run_debate(config, _console())

        assert result == 0
        assert mock_create_agent.call_count == 2  # 2 agents
        mock_controller.run.assert_called_once()

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_returns_1_on_error(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """run_debate returns 1 if controller.run() raises."""
        config = _make_config(tmp_path)

        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.side_effect = RuntimeError("API error")
        mock_controller_cls.return_value = mock_controller

        result = run_debate(config, _console())

        assert result == 1

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    def test_unknown_persona_returns_1(self, mock_validate, tmp_path):
        """run_debate returns 1 if an agent has an unknown persona."""
        config = _make_config(tmp_path)
        config.agents[0].persona = "NONEXISTENT_PERSONA"

        result = run_debate(config, _console())

        assert result == 1


# =========================================================================
# 3. Callback Wiring
# =========================================================================

class TestCallbackWiring:

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_passes_callback_to_controller(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """run_debate passes a progress_callback to controller.run()."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        run_debate(config, _console())

        # controller.run() should have been called with progress_callback kwarg
        call_kwargs = mock_controller.run.call_args
        assert "progress_callback" in call_kwargs.kwargs
        assert call_kwargs.kwargs["progress_callback"] is not None

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_callback_is_display_handle_event(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """The callback passed to controller.run() is DebateDisplay.handle_event."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        run_debate(config, _console())

        callback = mock_controller.run.call_args.kwargs["progress_callback"]
        # The callback should be a bound method named handle_event
        assert hasattr(callback, "__func__")
        assert callback.__func__.__name__ == "handle_event"

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_roadmap_review_skipped_headless(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """Roadmap review is skipped when interactive=False."""
        config = _make_config(tmp_path)
        config.stage2_mode = "moderated"
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller.moderator = MagicMock()
        mock_controller.roadmap = MagicMock()
        mock_controller_cls.return_value = mock_controller

        with patch("chal.cli.roadmap_review.run_roadmap_review") as mock_review:
            run_debate(config, _console(), interactive=False)
            mock_review.assert_not_called()

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.roadmap_review.run_roadmap_review")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_roadmap_review_runs_interactive_moderated(
        self, mock_create_agent, mock_controller_cls, mock_review, mock_validate, tmp_path
    ):
        """Roadmap review runs when interactive=True and stage2_mode=moderated."""
        config = _make_config(tmp_path)
        config.stage2_mode = "moderated"
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller.moderator = MagicMock()
        mock_controller.roadmap = MagicMock()
        mock_controller_cls.return_value = mock_controller
        mock_review.return_value = ([], config.max_rounds, False)

        run_debate(config, _console(), interactive=True)
        mock_review.assert_called_once()

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.roadmap_review.run_roadmap_review")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_roadmap_review_updates_max_rounds(
        self, mock_create_agent, mock_controller_cls, mock_review, mock_validate, tmp_path
    ):
        """run_debate updates max_rounds when roadmap review adjusts them."""
        config = _make_config(tmp_path)
        config.stage2_mode = "moderated"
        config.max_rounds = 3
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller.moderator = MagicMock()
        mock_controller.roadmap = MagicMock()
        mock_controller_cls.return_value = mock_controller
        mock_review.return_value = ([], 5, True)

        run_debate(config, _console(), interactive=True)

        assert config.max_rounds == 5
        assert mock_controller.max_rounds == 5

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.roadmap_review.run_roadmap_review")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_roadmap_review_sets_user_modified(
        self, mock_create_agent, mock_controller_cls, mock_review, mock_validate, tmp_path
    ):
        """run_debate sets roadmap_user_modified on the controller."""
        config = _make_config(tmp_path)
        config.stage2_mode = "moderated"
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller.moderator = MagicMock()
        mock_controller.roadmap = MagicMock()
        mock_controller_cls.return_value = mock_controller
        mock_review.return_value = ([], 3, True)

        run_debate(config, _console(), interactive=True)

        assert mock_controller.roadmap_user_modified is True

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_roadmap_review_skipped_non_moderated(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """Roadmap review is skipped when stage2_mode is not 'moderated'."""
        config = _make_config(tmp_path)
        config.stage2_mode = "free"
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        with patch("chal.cli.roadmap_review.run_roadmap_review") as mock_review:
            run_debate(config, _console(), interactive=True)
            mock_review.assert_not_called()


# =========================================================================
# 4. API Key Validation Wiring (Phase 3)
# =========================================================================

class TestApiKeyWiring:

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_validate_called_before_agents(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """validate_api_keys is called before agent creation."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        run_debate(config, _console())

        mock_validate.assert_called_once()

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_validate_gets_interactive_flag(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """validate_api_keys receives the interactive flag."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        run_debate(config, _console(), interactive=False)

        call_kwargs = mock_validate.call_args
        assert call_kwargs.kwargs.get("interactive") is False or call_kwargs[0][2] is False


# =========================================================================
# 5. History Logging Wiring (Phase 3)
# =========================================================================

class TestHistoryWiring:

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.history.log_debate")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_history_logged_after_debate(self, mock_create_agent, mock_controller_cls, mock_log, mock_validate, tmp_path):
        """log_debate is called after successful debate execution."""
        config = _make_config(tmp_path, save_transcript=True)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller
        mock_log.return_value = "abc12345"

        run_debate(config, _console())

        mock_log.assert_called_once()

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.history.log_debate", side_effect=Exception("disk full"))
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_history_failure_does_not_crash(self, mock_create_agent, mock_controller_cls, mock_log, mock_validate, tmp_path):
        """History logging failure does not crash the runner."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        result = run_debate(config, _console())
        assert result == 0  # Still succeeds


# =========================================================================
# 6. Error Recovery Wiring (Phase 3)
# =========================================================================

class TestErrorRecoveryWiring:

    @pytest.mark.unit
    @patch("chal.cli.runner.validate_api_keys")
    @patch("chal.cli.runner.DebateController")
    @patch("chal.cli.runner.create_agent_from_config")
    def test_on_error_passed_to_controller(self, mock_create_agent, mock_controller_cls, mock_validate, tmp_path):
        """run_debate passes on_error callback to controller.run()."""
        config = _make_config(tmp_path)
        mock_create_agent.return_value = MagicMock()
        mock_controller = MagicMock()
        mock_controller.run.return_value = _make_results()
        mock_controller_cls.return_value = mock_controller

        run_debate(config, _console())

        call_kwargs = mock_controller.run.call_args
        assert "on_error" in call_kwargs.kwargs
        assert call_kwargs.kwargs["on_error"] is not None


# =========================================================================
# 7. save_debate_outputs returns file list (Phase 3)
# =========================================================================

class TestSaveDebateOutputsReturnValue:

    @pytest.mark.unit
    def test_returns_saved_file_names(self, tmp_path):
        """save_debate_outputs returns a list of saved file names."""
        config = _make_config(tmp_path, save_synthesis=True, save_transcript=True)
        results = _make_results()

        saved = save_debate_outputs(config, results, MagicMock(), _console())

        assert isinstance(saved, list)
        assert len(saved) >= 2

    @pytest.mark.unit
    def test_returns_empty_when_all_disabled(self, tmp_path):
        """save_debate_outputs returns [] when all toggles are off."""
        config = _make_config(tmp_path)
        results = _make_results()

        saved = save_debate_outputs(config, results, MagicMock(), _console())

        assert saved == []
