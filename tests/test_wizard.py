"""
Unit tests for the CHAL CLI configuration wizard (wizard.py).

Tests cover:
- Individual step functions (mocked questionary prompts)
- Review panel rendering
- Wizard orchestration (build config, cancel, save, edit loop)
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call
from io import StringIO

from rich.console import Console

from chal.config import (
    AgentConfig,
    AdjudicationConfig,
    DebateConfig,
    OutputConfig,
    ParallelConfig,
    DEFAULT_STORAGE_DIR,
)
from chal.cli.wizard import (
    ask_topic,
    ask_num_agents,
    ask_agent_config,
    ask_stage3_mode,
    ask_num_rounds,
    ask_adjudicator_config,
    ask_output_toggles,
    ask_parallelization,
    ask_max_workers,
    ask_api_keys_for_config,
    show_review_panel,
    ask_review_action,
    ask_edit_section,
    ask_preset,
    ask_main_menu,
    _scan_presets,
    run_wizard,
    _validate_float_range,
    _validate_int_range,
    PERSONA_CHOICES,
    OUTPUT_TOGGLES,
    ABOUT_CHAL,
    WizardBack,
)


# =========================================================================
# Helpers
# =========================================================================

def _console() -> Console:
    return Console(file=StringIO())


# =========================================================================
# 1. Validators
# =========================================================================

class TestValidators:

    @pytest.mark.unit
    def test_float_range_valid(self):
        assert _validate_float_range("0.5") is True

    @pytest.mark.unit
    def test_float_range_boundary(self):
        assert _validate_float_range("0.0") is True
        assert _validate_float_range("1.0") is True

    @pytest.mark.unit
    def test_float_range_out_of_bounds(self):
        result = _validate_float_range("1.5")
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_float_range_not_a_number(self):
        result = _validate_float_range("abc")
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_int_range_valid(self):
        assert _validate_int_range("3", 2, 6) is True

    @pytest.mark.unit
    def test_int_range_out_of_bounds(self):
        result = _validate_int_range("7", 2, 6)
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_int_range_not_a_number(self):
        result = _validate_int_range("abc", 2, 6)
        assert isinstance(result, str)


# =========================================================================
# 2. Individual Step Functions
# =========================================================================

class TestAskTopic:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_returns_topic(self, mock_text):
        mock_text.return_value.ask.return_value = "Does free will exist?"
        result = ask_topic()
        assert result == "Does free will exist?"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_uses_default(self, mock_text):
        mock_text.return_value.ask.return_value = "My topic"
        ask_topic(default="Default topic")
        mock_text.assert_called_once()
        _, kwargs = mock_text.call_args
        assert kwargs["default"] == "Default topic"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_ctrl_c_raises(self, mock_text):
        mock_text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            ask_topic()


class TestAskNumAgents:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_returns_int(self, mock_text):
        mock_text.return_value.ask.return_value = "3"
        result = ask_num_agents()
        assert result == 3

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_uses_default(self, mock_text):
        mock_text.return_value.ask.return_value = "4"
        ask_num_agents(default=4)
        _, kwargs = mock_text.call_args
        assert kwargs["default"] == "4"


class TestAskAgentConfig:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.autocomplete")
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_agent_config(self, mock_select, mock_auto):
        # select calls: persona, provider
        mock_select.return_value.ask.side_effect = ["EMPIRICIST", "openai"]
        # autocomplete call: model
        mock_auto.return_value.ask.return_value = "gpt-4o"

        result = ask_agent_config(0)

        assert isinstance(result, AgentConfig)
        assert result.persona == "EMPIRICIST"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.temperature == 1.0
        assert "Empiricist" in result.name

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.autocomplete")
    @patch("chal.cli.wizard.questionary.select")
    def test_uses_default(self, mock_select, mock_auto):
        default = AgentConfig(
            name="Agent-Test", persona="SKEPTIC", model="o1-mini",
            provider="openai", temperature=1.0
        )
        mock_select.return_value.ask.side_effect = ["SKEPTIC", "openai"]
        mock_auto.return_value.ask.return_value = "o1-mini"

        result = ask_agent_config(0, default=default)

        assert result.name == "Agent-Test"
        assert result.persona == "SKEPTIC"


class TestAskStage3Mode:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_rebuttal_no_sub_options(self, mock_select):
        mock_select.return_value.ask.return_value = "rebuttal"
        mode, opts = ask_stage3_mode()
        assert mode == "rebuttal"
        assert opts == {}

class TestAskNumRounds:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_returns_int(self, mock_text):
        mock_text.return_value.ask.return_value = "5"
        assert ask_num_rounds() == 5


class TestAskAdjudicatorConfig:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.autocomplete")
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_adjudication_config(self, mock_select, mock_auto):
        mock_select.return_value.ask.side_effect = [
            "openai",                        # provider
            "CLASSICAL_INFORMAL_BAYESIAN",   # logic system
            "NONE",                          # ethics system
            "pure_logic",                    # balance preset
        ]
        mock_auto.return_value.ask.return_value = "o1-mini"

        result = ask_adjudicator_config()

        assert isinstance(result, AdjudicationConfig)
        assert result.model == "o1-mini"
        assert result.provider == "openai"
        assert result.logic_system == "CLASSICAL_INFORMAL_BAYESIAN"
        assert result.ethics_system == "NONE"
        assert result.logic_weight == 1.0
        assert result.ethics_weight == 0.0


class TestAskOutputToggles:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.checkbox")
    def test_returns_dict_of_bools(self, mock_checkbox):
        # Simulate selecting only save_transcript
        mock_checkbox.return_value.ask.return_value = ["save_transcript"]

        result = ask_output_toggles()

        assert isinstance(result, dict)
        assert result["save_transcript"] is True
        assert result["save_debug_log"] is False  # not selected

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.checkbox")
    def test_all_attrs_present(self, mock_checkbox):
        mock_checkbox.return_value.ask.return_value = []

        result = ask_output_toggles()

        expected_attrs = {attr for _, attr, _ in OUTPUT_TOGGLES}
        assert set(result.keys()) == expected_attrs


# =========================================================================
# 3. Review Panel
# =========================================================================

class TestShowReviewPanel:

    @pytest.mark.unit
    def test_renders_without_error(self):
        config = DebateConfig(
            topic="Test topic",
            agents=[
                AgentConfig(name="A1", persona="EMPIRICIST"),
                AgentConfig(name="A2", persona="RATIONALIST"),
            ],
        )
        console = _console()
        show_review_panel(config, console)

    @pytest.mark.unit
    def test_shows_parallelization_enabled(self):
        """Review panel shows 'Enabled (N threads)' when parallelization is on."""
        config = DebateConfig(
            topic="Test",
            agents=[AgentConfig(name="A1", persona="EMPIRICIST")],
            parallel=ParallelConfig(enabled=True, max_workers=5),
        )
        buf = StringIO()
        console = Console(file=buf)
        show_review_panel(config, console)
        output = buf.getvalue()
        assert "Parallelization" in output
        assert "Enabled" in output
        assert "5 threads" in output

    @pytest.mark.unit
    def test_shows_parallelization_disabled(self):
        """Review panel shows 'Disabled' when parallelization is off."""
        config = DebateConfig(
            topic="Test",
            agents=[AgentConfig(name="A1", persona="EMPIRICIST")],
            parallel=ParallelConfig(enabled=False),
        )
        buf = StringIO()
        console = Console(file=buf)
        show_review_panel(config, console)
        output = buf.getvalue()
        assert "Parallelization" in output
        assert "Disabled" in output


# =========================================================================
# 3b. Parallelization & API Keys
# =========================================================================

class TestAskParallelization:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.confirm")
    def test_returns_true_when_accepted(self, mock_confirm):
        """ask_parallelization returns True when user accepts."""
        mock_confirm.return_value.ask.return_value = True
        assert ask_parallelization() is True

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.confirm")
    def test_returns_false_when_declined(self, mock_confirm):
        """ask_parallelization returns False when user declines."""
        mock_confirm.return_value.ask.return_value = False
        assert ask_parallelization() is False

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.confirm")
    def test_default_is_true(self, mock_confirm):
        """Default value passed to questionary.confirm is True."""
        mock_confirm.return_value.ask.return_value = True
        ask_parallelization()
        _, kwargs = mock_confirm.call_args
        assert kwargs.get("default") is True


class TestAskMaxWorkers:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_returns_entered_value(self, mock_text):
        """ask_max_workers returns the integer the user types."""
        mock_text.return_value.ask.return_value = "8"
        assert ask_max_workers() == 8

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_default_is_5(self, mock_text):
        """Default value passed to questionary.text is '5'."""
        mock_text.return_value.ask.return_value = "5"
        ask_max_workers()
        _, kwargs = mock_text.call_args
        assert kwargs.get("default") == "5"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_invalid_input_returns_default(self, mock_text):
        """Non-integer input falls back to the default."""
        mock_text.return_value.ask.return_value = "abc"
        assert ask_max_workers(default=5) == 5

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_zero_returns_one(self, mock_text):
        """Value < 1 is clamped to 1."""
        mock_text.return_value.ask.return_value = "0"
        assert ask_max_workers() == 1


class TestAskApiKeysForConfig:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.confirm")
    @patch("chal.cli.wizard.questionary.text")
    @patch.dict(os.environ, {}, clear=False)
    def test_single_key_entry(self, mock_text, mock_confirm, monkeypatch):
        """Entering one key and declining more sets env var."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_text.return_value.ask.return_value = "sk-test"
        mock_confirm.return_value.ask.return_value = False

        state = {
            'agent_configs': [AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            'adjudication': AdjudicationConfig(provider="openai"),
        }
        ask_api_keys_for_config(state)
        assert os.environ.get("OPENAI_API_KEY") == "sk-test"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.confirm")
    @patch("chal.cli.wizard.questionary.text")
    @patch.dict(os.environ, {}, clear=False)
    def test_multi_key_entry(self, mock_text, mock_confirm, monkeypatch):
        """Entering two keys produces comma-separated env var."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_text.return_value.ask.side_effect = ["sk-a", "sk-b"]
        mock_confirm.return_value.ask.side_effect = [True, False]

        state = {
            'agent_configs': [AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            'adjudication': AdjudicationConfig(provider="openai"),
        }
        ask_api_keys_for_config(state)
        assert os.environ.get("OPENAI_API_KEY") == "sk-a,sk-b"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-present"}, clear=False)
    def test_skips_already_set(self, mock_text):
        """Providers with keys already set are skipped."""
        state = {
            'agent_configs': [AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            'adjudication': AdjudicationConfig(provider="openai"),
        }
        ask_api_keys_for_config(state)
        mock_text.assert_not_called()

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    def test_skips_ollama(self, mock_text):
        """Ollama provider is skipped (no API key needed)."""
        state = {
            'agent_configs': [AgentConfig(name="A", persona="EMPIRICIST", provider="ollama")],
            'adjudication': AdjudicationConfig(provider="ollama"),
        }
        ask_api_keys_for_config(state)
        mock_text.assert_not_called()

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.text")
    @patch.dict(os.environ, {}, clear=False)
    def test_skip_when_empty_input(self, mock_text, monkeypatch):
        """Pressing Enter (empty input) skips the provider."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_text.return_value.ask.return_value = ""

        state = {
            'agent_configs': [AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            'adjudication': AdjudicationConfig(provider="openai"),
        }
        ask_api_keys_for_config(state)
        assert os.environ.get("OPENAI_API_KEY") is None


# =========================================================================
# 4. Review Action
# =========================================================================

class TestReviewAction:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_action(self, mock_select):
        mock_select.return_value.ask.return_value = "launch"
        assert ask_review_action() == "launch"


class TestEditSection:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_section(self, mock_select):
        mock_select.return_value.ask.return_value = "topic"
        assert ask_edit_section() == "topic"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_parallelization_in_choices(self, mock_select):
        """Edit section includes parallelization option."""
        mock_select.return_value.ask.return_value = "parallelization"
        result = ask_edit_section()
        assert result == "parallelization"
        _, kwargs = mock_select.call_args
        choice_values = [c.value for c in kwargs["choices"]]
        assert "parallelization" in choice_values


# =========================================================================
# 5. Wizard Orchestration
# =========================================================================

class TestRunWizard:

    def _mock_all_steps(self):
        """Return a dict of patches for all questionary calls in wizard steps."""
        return {
            "text": patch("chal.cli.wizard.questionary.text"),
            "select": patch("chal.cli.wizard.questionary.select"),
            "autocomplete": patch("chal.cli.wizard.questionary.autocomplete"),
            "checkbox": patch("chal.cli.wizard.questionary.checkbox"),
            "confirm": patch("chal.cli.wizard.questionary.confirm"),
        }

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_wizard_cancel(self):
        """Wizard returns (None, 'cancel') when user cancels at review."""
        with patch("chal.cli.wizard.questionary.text") as m_text, \
             patch("chal.cli.wizard.questionary.select") as m_select, \
             patch("chal.cli.wizard.questionary.autocomplete") as m_auto, \
             patch("chal.cli.wizard.questionary.checkbox") as m_checkbox, \
             patch("chal.cli.wizard.questionary.confirm") as m_confirm:

            m_text.return_value.ask.side_effect = [
                "Test topic",  # topic
                "2",           # num_agents
                "3",           # num_rounds
                "5",           # max_workers
            ]
            m_select.return_value.ask.side_effect = [
                "debate",                           # main menu
                "__custom__",                       # ask_preset
                "EMPIRICIST", "openai",            # agent 1
                "RATIONALIST", "openai",           # agent 2
                "open",                             # stage 2
                "rebuttal",                         # stage 3
                "openai",                           # adjudicator provider
                "CLASSICAL_INFORMAL_BAYESIAN",     # adjudicator logic system
                "NONE",                             # adjudicator ethics system
                "pure_logic",                       # balance preset
                "cancel",                           # review action
            ]
            m_auto.return_value.ask.side_effect = [
                "gpt-4o",  # agent 1 model
                "gpt-4o",  # agent 2 model
                "o1-mini", # adjudicator model
            ]
            m_checkbox.return_value.ask.return_value = ["save_transcript", "save_debug_log"]
            m_confirm.return_value.ask.return_value = True  # parallelization

            config, action = run_wizard(_console())

            assert config is None
            assert action == "cancel"

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "ant-test"}, clear=False)
    def test_wizard_launch(self):
        """Wizard builds a valid DebateConfig when user launches."""
        with patch("chal.cli.wizard.questionary.text") as m_text, \
             patch("chal.cli.wizard.questionary.select") as m_select, \
             patch("chal.cli.wizard.questionary.autocomplete") as m_auto, \
             patch("chal.cli.wizard.questionary.checkbox") as m_checkbox, \
             patch("chal.cli.wizard.questionary.confirm") as m_confirm:

            m_text.return_value.ask.side_effect = [
                "Does free will exist?",  # topic
                "2",                       # num_agents
                "3",                       # num_rounds
                "5",                       # max_workers
            ]
            m_select.return_value.ask.side_effect = [
                "debate",                        # main menu
                "__custom__",                    # ask_preset
                "EMPIRICIST", "openai",
                "SKEPTIC", "anthropic",
                "open",
                "rebuttal",
                "openai",                        # adjudicator provider
                "CLASSICAL_INFORMAL_BAYESIAN",   # adjudicator logic system
                "NONE",                          # adjudicator ethics system
                "pure_logic",                    # balance preset
                "launch",                        # review action
            ]
            m_auto.return_value.ask.side_effect = [
                "gpt-4o",
                "claude-sonnet-4-5-20250929",
                "o1-mini",
            ]
            m_checkbox.return_value.ask.return_value = [
                "save_transcript", "save_debug_log",
            ]
            m_confirm.return_value.ask.return_value = True  # parallelization

            config, action = run_wizard(_console())

            assert action == "launch"
            assert isinstance(config, DebateConfig)
            assert config.topic == "Does free will exist?"
            assert len(config.agents) == 2
            assert config.agents[0].persona == "EMPIRICIST"
            assert config.agents[1].persona == "SKEPTIC"
            assert config.agents[1].provider == "anthropic"
            assert config.stage3_mode == "rebuttal"
            assert config.max_rounds == 3
            assert config.outputs.save_transcript is True
            assert config.outputs.save_training_data is False
            assert config.parallel.enabled is True

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_wizard_save(self, tmp_path):
        """Wizard saves config to YAML when user selects save."""
        yaml_path = tmp_path / "test_debate.yaml"

        with patch("chal.cli.wizard.questionary.text") as m_text, \
             patch("chal.cli.wizard.questionary.select") as m_select, \
             patch("chal.cli.wizard.questionary.autocomplete") as m_auto, \
             patch("chal.cli.wizard.questionary.checkbox") as m_checkbox, \
             patch("chal.cli.wizard.questionary.confirm") as m_confirm:

            m_text.return_value.ask.side_effect = [
                "Save test topic",  # topic
                "2",                # num_agents
                "1",                # num_rounds
                "5",                # max_workers
                str(yaml_path),     # save path
            ]
            m_select.return_value.ask.side_effect = [
                "debate",                        # main menu
                "__custom__",                    # ask_preset
                "EMPIRICIST", "openai",
                "RATIONALIST", "openai",
                "open",
                "rebuttal",
                "openai",                        # adjudicator provider
                "CLASSICAL_INFORMAL_BAYESIAN",   # adjudicator logic system
                "NONE",                          # adjudicator ethics system
                "pure_logic",                    # balance preset
                "save",    # review action: save (loops back to review)
                "launch",  # review action: launch
            ]
            m_auto.return_value.ask.side_effect = [
                "gpt-4o", "gpt-4o", "o1-mini",
            ]
            m_checkbox.return_value.ask.return_value = ["save_transcript"]
            m_confirm.return_value.ask.return_value = True  # parallelization

            config, action = run_wizard(_console())

            assert action == "launch"
            assert yaml_path.exists()

            # Verify saved YAML can be loaded
            from chal.config import DebateConfig as DC
            reloaded = DC.from_yaml(yaml_path)
            assert reloaded.topic == "Save test topic"


# =========================================================================
# 6. Constants
# =========================================================================

class TestConstants:

    @pytest.mark.unit
    def test_persona_choices_has_12(self):
        """PERSONA_CHOICES has all 12 personas."""
        assert len(PERSONA_CHOICES) == 12

    @pytest.mark.unit
    def test_persona_choice_values_are_uppercase(self):
        """All persona choice values are uppercase strings."""
        for choice in PERSONA_CHOICES:
            assert choice.value == choice.value.upper()

    @pytest.mark.unit
    def test_output_toggles_all_have_three_elements(self):
        """Each OUTPUT_TOGGLE is a (label, attr, default) triple."""
        for toggle in OUTPUT_TOGGLES:
            assert len(toggle) == 3
            label, attr, default = toggle
            assert isinstance(label, str)
            assert isinstance(attr, str)
            assert isinstance(default, bool)


# =========================================================================
# 7. Preset Selection
# =========================================================================

class TestScanPresets:

    @pytest.mark.unit
    def test_scans_configurations_directory(self):
        """_scan_presets returns entries from the configurations directory."""
        presets = _scan_presets()
        # We know there are 4 YAML files in src/chal/configurations/
        assert len(presets) >= 1
        # Each entry is (label, config_name, path)
        for label, name, path in presets:
            assert isinstance(label, str)
            assert isinstance(name, str)
            assert path.exists()

    @pytest.mark.unit
    def test_preset_names_include_default(self):
        """_scan_presets includes the 'default' configuration."""
        presets = _scan_presets()
        names = [name for _, name, _ in presets]
        assert "default" in names

    @pytest.mark.unit
    def test_scan_returns_empty_if_dir_missing(self, monkeypatch):
        """_scan_presets returns [] if CONFIG_DIR doesn't exist."""
        from pathlib import Path
        monkeypatch.setattr("chal.cli.wizard.CONFIG_DIR", Path("/nonexistent/dir"))
        assert _scan_presets() == []


class TestAskPreset:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_none_for_custom(self, mock_select):
        """ask_preset returns None when user selects 'Custom'."""
        mock_select.return_value.ask.return_value = "__custom__"
        result = ask_preset()
        assert result is None

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    @patch("chal.cli.wizard.DebateConfig.from_yaml")
    def test_returns_config_for_preset(self, mock_from_yaml, mock_select):
        """ask_preset returns a DebateConfig when a preset path is selected."""
        mock_config = MagicMock(spec=DebateConfig)
        mock_from_yaml.return_value = mock_config
        mock_select.return_value.ask.return_value = "/some/path.yaml"

        result = ask_preset()

        assert result == mock_config
        mock_from_yaml.assert_called_once()

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_ctrl_c_raises(self, mock_select):
        """ask_preset raises KeyboardInterrupt on Ctrl+C."""
        mock_select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            ask_preset()


# =========================================================================
# 8. Main Menu
# =========================================================================

class TestMainMenu:

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_debate(self, mock_select):
        """ask_main_menu returns 'debate' when selected."""
        mock_select.return_value.ask.return_value = "debate"
        assert ask_main_menu() == "debate"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_about(self, mock_select):
        """ask_main_menu returns 'about' when selected."""
        mock_select.return_value.ask.return_value = "about"
        assert ask_main_menu() == "about"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_gauntlet(self, mock_select):
        """ask_main_menu returns 'gauntlet' when selected."""
        mock_select.return_value.ask.return_value = "gauntlet"
        assert ask_main_menu() == "gauntlet"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_returns_exit(self, mock_select):
        """ask_main_menu returns 'exit' when selected."""
        mock_select.return_value.ask.return_value = "exit"
        assert ask_main_menu() == "exit"

    @pytest.mark.unit
    @patch("chal.cli.wizard.questionary.select")
    def test_ctrl_c_raises(self, mock_select):
        """ask_main_menu raises KeyboardInterrupt on Ctrl+C."""
        mock_select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            ask_main_menu()

    @pytest.mark.unit
    def test_about_chal_text_is_nonempty(self):
        """ABOUT_CHAL contains substantive content."""
        assert len(ABOUT_CHAL) > 100
        assert "CHAL" in ABOUT_CHAL
        assert "belief" in ABOUT_CHAL.lower()

    @pytest.mark.unit
    def test_wizard_exit_on_main_menu(self):
        """Selecting 'exit' at main menu returns (None, 'cancel')."""
        with patch("chal.cli.wizard.questionary.select") as m_select:
            m_select.return_value.ask.return_value = "exit"
            config, action = run_wizard(_console())
            assert config is None
            assert action == "cancel"

    @pytest.mark.unit
    def test_wizard_about_then_exit(self):
        """Selecting 'about' shows info, then 'exit' cancels."""
        with patch("chal.cli.wizard.questionary.select") as m_select:
            m_select.return_value.ask.side_effect = ["about", "exit"]
            config, action = run_wizard(_console())
            assert config is None
            assert action == "cancel"

    @pytest.mark.unit
    def test_wizard_gauntlet_then_exit(self):
        """Selecting 'gauntlet' shows coming soon, then 'exit' cancels."""
        with patch("chal.cli.wizard.questionary.select") as m_select:
            m_select.return_value.ask.side_effect = ["gauntlet", "exit"]
            config, action = run_wizard(_console())
            assert config is None
            assert action == "cancel"

    @pytest.mark.unit
    def test_ctrl_z_at_preset_returns_to_menu(self):
        """Ctrl+Z at preset step (step 0) returns to main menu."""
        with patch("chal.cli.wizard.questionary.select") as m_select, \
             patch("chal.cli.wizard.questionary.text"):
            # First call: main menu -> debate
            # Second call: preset -> raises WizardBack (simulating Ctrl+Z)
            # Third call: back at main menu -> exit
            call_count = [0]
            def select_side_effect(*args, **kwargs):
                mock = MagicMock()
                idx = call_count[0]
                call_count[0] += 1
                if idx == 0:
                    mock.ask.return_value = "debate"    # main menu
                elif idx == 1:
                    mock.ask.side_effect = WizardBack   # Ctrl+Z at preset
                elif idx == 2:
                    mock.ask.return_value = "exit"      # back at main menu
                return mock
            m_select.side_effect = select_side_effect

            config, action = run_wizard(_console())
            assert config is None
            assert action == "cancel"
