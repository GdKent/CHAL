"""
Unit tests for the CHAL CLI runner (runner.py).

Tests cover:
- save_debate_outputs writes expected files based on config toggles
- run_debate creates agents and calls controller
- run_debate passes progress_callback to controller.run()
"""

import pytest
import json
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from io import StringIO

from rich.console import Console

from chal.config import DebateConfig, AgentConfig, OutputConfig
from chal.cli.runner import save_debate_outputs, run_debate, _write_best_agent_beliefs


# =========================================================================
# Helpers
# =========================================================================

def _console() -> Console:
    return Console(file=StringIO())


def _make_config(tmp_path: Path, **output_overrides) -> DebateConfig:
    """Build a minimal DebateConfig with storage_dir pointing at tmp_path."""
    output_kwargs = {
        "storage_dir": tmp_path,
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
    )


def _make_results() -> dict:
    """Build a minimal results dict matching controller.run() output."""
    return {
        "markdown_transcript": "Markdown transcript text.",
        "debug_log": "Debug log text.",
        "initial_positions": ["Belief A", "Belief B"],
        "final_positions": ["Updated A", "Updated B"],
        "agent_stats": {"Agent-A": {"performance_score": 0.5}, "Agent-B": {"performance_score": -0.1}},
    }


# =========================================================================
# 1. save_debate_outputs
# =========================================================================

class TestSaveDebateOutputs:

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
        controller = _make_controller_with_agents()

        save_debate_outputs(config, results, controller, _console())

        beliefs_dir = tmp_path / config.outputs.initial_beliefs_dir
        assert beliefs_dir.is_dir()
        assert (beliefs_dir / "Agent-A.json").exists()
        assert (beliefs_dir / "Agent-B.json").exists()
        # Verify at least one file contains valid CBS data
        data = json.loads((beliefs_dir / "Agent-A.json").read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "thesis" in data

    @pytest.mark.unit
    def test_saves_final_beliefs(self, tmp_path):
        config = _make_config(tmp_path, save_final_beliefs=True)
        results = _make_results()
        controller = _make_controller_with_agents()

        save_debate_outputs(config, results, controller, _console())

        beliefs_dir = tmp_path / config.outputs.final_beliefs_dir
        assert beliefs_dir.is_dir()
        assert (beliefs_dir / "Agent-A.json").exists()
        assert (beliefs_dir / "Agent-B.json").exists()
        # Verify at least one file contains valid CBS data
        data = json.loads((beliefs_dir / "Agent-A.json").read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "thesis" in data

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

    @pytest.mark.unit
    @patch("chal.embeddings.embedding_visualizer.generate_pca_trajectory_plot")
    @patch("chal.embeddings.embedding_visualizer.generate_belief_trajectory_plot")
    def test_plot_trajectories_calls_generate_function(self, mock_gen_plot, mock_gen_pca, tmp_path):
        """When plot_trajectories=True, generate_belief_trajectory_plot is called."""
        config = _make_config(tmp_path, plot_trajectories=True)
        results = _make_results()

        mock_gen_plot.return_value = tmp_path / config.outputs.trajectory_plot_file
        mock_gen_pca.return_value = tmp_path / config.outputs.pca_plot_file

        saved = save_debate_outputs(config, results, MagicMock(), _console())

        mock_gen_plot.assert_called_once_with(config)
        assert config.outputs.trajectory_plot_file in saved

    @pytest.mark.unit
    @patch("chal.embeddings.embedding_visualizer.generate_pca_trajectory_plot")
    @patch("chal.embeddings.embedding_visualizer.generate_belief_trajectory_plot")
    def test_pca_plot_trajectories_calls_generate_function(self, mock_gen_plot, mock_gen_pca, tmp_path):
        """When plot_trajectories=True, generate_pca_trajectory_plot is also called."""
        config = _make_config(tmp_path, plot_trajectories=True)
        results = _make_results()

        mock_gen_plot.return_value = tmp_path / config.outputs.trajectory_plot_file
        mock_gen_pca.return_value = tmp_path / config.outputs.pca_plot_file

        saved = save_debate_outputs(config, results, MagicMock(), _console())

        mock_gen_pca.assert_called_once_with(config)
        assert config.outputs.pca_plot_file in saved


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
        config = _make_config(tmp_path, save_transcript=True, save_debug_log=True)
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


# =========================================================================
# 8. Best-Agent Beliefs (Phase 3 of stats-expansion roadmap)
# =========================================================================

def _make_best_belief_results(
    a_score: float = 0.65,
    b_score: float = 0.15,
    a_initial: str = "Initial belief for Agent-A.",
    a_final: str = "Final belief for Agent-A.",
    b_initial: str = "Initial belief for Agent-B.",
    b_final: str = "Final belief for Agent-B.",
) -> dict:
    """Build a results dict with the keys _write_best_agent_beliefs reads."""
    return {
        "markdown_transcript": "markdown transcript",
        "debug_log": "log",
        "initial_positions": [a_initial, b_initial],
        "final_positions": [a_final, b_final],
        "agent_stats": {
            "Agent-A": {"performance_score": a_score},
            "Agent-B": {"performance_score": b_score},
        },
    }


def _make_controller_with_agents(best_score: float = 0.65) -> MagicMock:
    """Build a mock controller with two agents (A and B)."""
    initial_a = json.dumps({
        "schema_version": "CBS",
        "thesis": {"strength": 0.5, "stance": "A initial"},
        "claims": [{"id": "C1", "status": "active"}],
    })
    initial_b = json.dumps({
        "schema_version": "CBS",
        "thesis": {"strength": 0.4, "stance": "B initial"},
        "claims": [],
    })

    agent_a = MagicMock()
    agent_a.name = "Agent-A"
    agent_a.all_beliefs_held = [initial_a]
    agent_a.get_internal_belief_obj.return_value = {
        "schema_version": "CBS",
        "thesis": {"strength": 0.8, "stance": "A final"},
        "claims": [{"id": "C1", "status": "active"}, {"id": "C2", "status": "active"}],
    }

    agent_b = MagicMock()
    agent_b.name = "Agent-B"
    agent_b.all_beliefs_held = [initial_b]
    agent_b.get_internal_belief_obj.return_value = {
        "schema_version": "CBS",
        "thesis": {"strength": 0.3, "stance": "B final"},
        "claims": [],
    }

    controller = MagicMock()
    controller.agents = [agent_a, agent_b]
    return controller


class TestWriteBestAgentBeliefs:

    @pytest.mark.unit
    def test_best_initial_final_beliefs_json_is_written(self, tmp_path):
        """_write_best_agent_beliefs produces a CBS-shaped JSON with correct best_agent."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results(a_score=0.65, b_score=0.15)
        controller = _make_controller_with_agents()

        written = _write_best_agent_beliefs(config, results, controller)

        json_path = tmp_path / config.outputs.best_beliefs_json_file
        assert json_path.name in written
        assert json_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["best_agent"] == "Agent-A"
        assert payload["performance_score"] == pytest.approx(0.65)
        assert payload["topic"] == config.topic
        # Initial belief parsed from all_beliefs_held[0]; final from get_internal_belief_obj()
        assert payload["initial_belief"]["thesis"]["stance"] == "A initial"
        assert payload["final_belief"]["thesis"]["stance"] == "A final"
        assert "selection_rule" in payload

    @pytest.mark.unit
    def test_best_initial_final_beliefs_txt_is_written(self, tmp_path):
        """The markdown output reuses the results positions for the best agent."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results(
            a_initial="A-INITIAL-MARKDOWN-BLOCK",
            a_final="A-FINAL-MARKDOWN-BLOCK",
        )
        controller = _make_controller_with_agents()

        written = _write_best_agent_beliefs(config, results, controller)

        text_path = tmp_path / config.outputs.best_beliefs_text_file
        assert text_path.name in written
        assert text_path.exists()

        text = text_path.read_text(encoding="utf-8")
        assert "Best Agent Beliefs" in text
        assert "Agent-A" in text
        assert "A-INITIAL-MARKDOWN-BLOCK" in text
        assert "A-FINAL-MARKDOWN-BLOCK" in text
        # Must NOT leak the loser's markdown block into the best-agent file.
        assert "Initial belief for Agent-B" not in text

    @pytest.mark.unit
    def test_best_agent_picks_highest_score(self, tmp_path):
        """When Agent-B has the higher score, it becomes best_agent."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results(a_score=0.10, b_score=0.75)
        controller = _make_controller_with_agents()

        _write_best_agent_beliefs(config, results, controller)

        payload = json.loads(
            (tmp_path / config.outputs.best_beliefs_json_file).read_text(encoding="utf-8")
        )
        assert payload["best_agent"] == "Agent-B"
        assert payload["performance_score"] == pytest.approx(0.75)

    @pytest.mark.unit
    def test_tiebreaker_picks_first_in_config_order(self, tmp_path):
        """Equal scores resolve via first-in-config.agents order."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results(a_score=0.50, b_score=0.50)
        controller = _make_controller_with_agents()

        _write_best_agent_beliefs(config, results, controller)

        payload = json.loads(
            (tmp_path / config.outputs.best_beliefs_json_file).read_text(encoding="utf-8")
        )
        # Agent-A is earlier in config.agents → wins the tie.
        assert payload["best_agent"] == "Agent-A"

    @pytest.mark.unit
    def test_unparseable_initial_belief_falls_back_to_error_payload(self, tmp_path):
        """Malformed initial JSON produces {'error':..., 'raw':...} in initial_belief."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results()

        controller = _make_controller_with_agents()
        controller.agents[0].all_beliefs_held = ["<<<NOT JSON>>>"]

        _write_best_agent_beliefs(config, results, controller)

        payload = json.loads(
            (tmp_path / config.outputs.best_beliefs_json_file).read_text(encoding="utf-8")
        )
        initial = payload["initial_belief"]
        assert isinstance(initial, dict)
        assert "error" in initial
        assert initial.get("raw") == "<<<NOT JSON>>>"

    @pytest.mark.unit
    def test_missing_initial_belief_raises(self, tmp_path):
        """Agent with empty all_beliefs_held raises ValueError (caller should catch)."""
        config = _make_config(tmp_path)
        results = _make_best_belief_results()
        controller = _make_controller_with_agents()
        controller.agents[0].all_beliefs_held = []  # best agent has no initial snapshot

        with pytest.raises(ValueError):
            _write_best_agent_beliefs(config, results, controller)


# =========================================================================
# 9. Save-outputs always writes best-agent files (Phase 3 always-on behavior)
# =========================================================================

class TestSaveOutputsWritesBestAgentFiles:

    @pytest.mark.unit
    def test_best_beliefs_written_even_when_all_toggles_off(self, tmp_path):
        """Best-agent beliefs are ALWAYS written (no gating flag)."""
        config = _make_config(tmp_path)  # every save_* toggle off
        results = _make_best_belief_results()
        controller = _make_controller_with_agents()

        saved = save_debate_outputs(config, results, controller, _console())

        json_path = tmp_path / config.outputs.best_beliefs_json_file
        text_path = tmp_path / config.outputs.best_beliefs_text_file
        assert json_path.exists()
        assert text_path.exists()
        assert json_path.name in saved
        assert text_path.name in saved

    @pytest.mark.unit
    def test_best_beliefs_failure_does_not_abort_save(self, tmp_path):
        """If best-agent writer raises, save_debate_outputs still returns (other files intact)."""
        config = _make_config(tmp_path, save_transcript=True)
        results = _make_best_belief_results()
        # Controller with no matching agent will trigger select-best failure downstream.
        controller = MagicMock()
        controller.agents = []  # no agents → writer will raise

        saved = save_debate_outputs(config, results, controller, _console())

        # Transcript still gets written despite best-agent failure.
        transcript = tmp_path / config.outputs.transcript_file
        assert transcript.exists()
        assert transcript.name in saved
