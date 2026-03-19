"""
Unit tests for the CHAL CLI API key validation module (api_keys.py).

Tests cover:
- Provider collection from debate config (_collect_providers)
- API key presence checking (check_api_keys)
- Interactive prompting for missing keys (prompt_missing_keys)
- Non-interactive warning for missing keys (warn_missing_keys)
- Orchestrator dispatch (validate_api_keys)
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from rich.console import Console

from chal.config import (
    AgentConfig,
    AdjudicationConfig,
    DebateConfig,
    ModeratorConfig,
)
from chal.cli.api_keys import (
    _collect_providers,
    check_api_keys,
    prompt_missing_keys,
    warn_missing_keys,
    validate_api_keys,
    PROVIDER_ENV_VARS,
)


# =========================================================================
# Helpers
# =========================================================================

def _console() -> Console:
    return Console(file=StringIO())


def _minimal_config(**overrides) -> DebateConfig:
    """Return a minimal DebateConfig suitable for most tests."""
    defaults = dict(
        topic="test",
        agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
        adjudication=AdjudicationConfig(provider="openai"),
    )
    defaults.update(overrides)
    return DebateConfig(**defaults)


# =========================================================================
# 1. _collect_providers
# =========================================================================

class TestCollectProviders:

    @pytest.mark.unit
    def test_collects_agent_providers(self):
        """Providers are collected from all agents."""
        config = _minimal_config(
            agents=[
                AgentConfig(name="A", persona="EMPIRICIST", provider="openai"),
                AgentConfig(name="B", persona="RATIONALIST", provider="anthropic"),
            ],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        providers = _collect_providers(config)
        assert "openai" in providers
        assert "anthropic" in providers

    @pytest.mark.unit
    def test_collects_adjudication_provider(self):
        """The adjudication provider is included."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        providers = _collect_providers(config)
        assert "anthropic" in providers

    @pytest.mark.unit
    def test_collects_moderator_provider_when_moderated(self):
        """The moderator provider is included when stage2_mode is 'moderated'."""
        config = _minimal_config(
            stage2_mode="moderated",
            moderator=ModeratorConfig(provider="google"),
        )
        providers = _collect_providers(config)
        assert "google" in providers

    @pytest.mark.unit
    def test_excludes_moderator_when_not_moderated(self):
        """The moderator provider is excluded when stage2_mode is not 'moderated'."""
        config = _minimal_config(
            stage2_mode="open",
            moderator=ModeratorConfig(provider="google"),
        )
        providers = _collect_providers(config)
        assert "google" not in providers


# =========================================================================
# 2. check_api_keys
# =========================================================================

class TestCheckApiKeys:

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=False)
    def test_returns_true_when_key_present(self):
        """check_api_keys returns True for a provider whose key is set."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        result = check_api_keys(config)
        assert result["openai"] is True

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_returns_false_when_key_missing(self):
        """check_api_keys returns False for a provider whose key is not set."""
        # Ensure the key is absent
        env_var = PROVIDER_ENV_VARS["anthropic"]
        os.environ.pop(env_var, None)

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="anthropic")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        result = check_api_keys(config)
        assert result["anthropic"] is False

    @pytest.mark.unit
    @patch.dict(os.environ, {"XAI_API_KEY": "xai-test123"}, clear=False)
    def test_xai_key_present(self):
        """check_api_keys returns True for xAI when XAI_API_KEY is set."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="xai")],
            adjudication=AdjudicationConfig(provider="xai"),
        )
        result = check_api_keys(config)
        assert result["xai"] is True

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_xai_key_missing(self):
        """check_api_keys returns False for xAI when XAI_API_KEY is not set."""
        os.environ.pop("XAI_API_KEY", None)
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="xai")],
            adjudication=AdjudicationConfig(provider="xai"),
        )
        result = check_api_keys(config)
        assert result["xai"] is False

    @pytest.mark.unit
    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test123"}, clear=False)
    def test_perplexity_key_present(self):
        """check_api_keys returns True for Perplexity when PERPLEXITY_API_KEY is set."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="perplexity")],
            adjudication=AdjudicationConfig(provider="perplexity"),
        )
        result = check_api_keys(config)
        assert result["perplexity"] is True

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_perplexity_key_missing(self):
        """check_api_keys returns False for Perplexity when PERPLEXITY_API_KEY is not set."""
        os.environ.pop("PERPLEXITY_API_KEY", None)
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="perplexity")],
            adjudication=AdjudicationConfig(provider="perplexity"),
        )
        result = check_api_keys(config)
        assert result["perplexity"] is False

    @pytest.mark.unit
    def test_ollama_no_key_required(self):
        """Ollama is not in PROVIDER_ENV_VARS, so it always returns True (no key needed)."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="ollama")],
            adjudication=AdjudicationConfig(provider="ollama"),
        )
        result = check_api_keys(config)
        assert result["ollama"] is True

    @pytest.mark.unit
    def test_unknown_provider_returns_true(self):
        """Unknown providers (not in PROVIDER_ENV_VARS) are treated as valid."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="custom_llm")],
            adjudication=AdjudicationConfig(provider="custom_llm"),
        )
        result = check_api_keys(config)
        assert result["custom_llm"] is True


# =========================================================================
# 3. prompt_missing_keys
# =========================================================================

class TestPromptMissingKeys:

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {}, clear=False)
    def test_prompts_for_missing_key(self, mock_questionary):
        """When a key is missing, the user is prompted and the env var is set."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        mock_questionary.text.return_value.ask.return_value = "sk-ant-secret"

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="anthropic")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        mock_questionary.text.assert_called_once()
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-secret"

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {}, clear=False)
    def test_skip_when_empty_input(self, mock_questionary):
        """When the user presses Enter (empty input), the key is not set."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        mock_questionary.text.return_value.ask.return_value = ""

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="anthropic")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        assert os.environ.get("ANTHROPIC_API_KEY") is None

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {}, clear=False)
    def test_ctrl_c_raises_keyboard_interrupt(self, mock_questionary):
        """When questionary returns None (Ctrl+C), KeyboardInterrupt is raised."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        mock_questionary.text.return_value.ask.return_value = None

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="anthropic")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        console = _console()
        with pytest.raises(KeyboardInterrupt):
            prompt_missing_keys(config, console)

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-present"}, clear=False)
    def test_skips_present_keys(self, mock_questionary):
        """Keys already present in the environment are not prompted for."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        mock_questionary.text.assert_not_called()

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {}, clear=False)
    def test_prompts_for_missing_xai_key(self, mock_questionary):
        """When XAI_API_KEY is missing, the user is prompted and the env var is set."""
        os.environ.pop("XAI_API_KEY", None)

        mock_questionary.text.return_value.ask.return_value = "xai-secret"

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="xai")],
            adjudication=AdjudicationConfig(provider="xai"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        mock_questionary.text.assert_called_once()
        assert os.environ.get("XAI_API_KEY") == "xai-secret"

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    @patch.dict(os.environ, {}, clear=False)
    def test_prompts_for_missing_perplexity_key(self, mock_questionary):
        """When PERPLEXITY_API_KEY is missing, the user is prompted and the env var is set."""
        os.environ.pop("PERPLEXITY_API_KEY", None)

        mock_questionary.text.return_value.ask.return_value = "pplx-secret"

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="perplexity")],
            adjudication=AdjudicationConfig(provider="perplexity"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        mock_questionary.text.assert_called_once()
        assert os.environ.get("PERPLEXITY_API_KEY") == "pplx-secret"


# =========================================================================
# 4. warn_missing_keys
# =========================================================================

class TestWarnMissingKeys:

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_warns_about_missing_key(self):
        """A warning is printed for a missing key."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="anthropic")],
            adjudication=AdjudicationConfig(provider="anthropic"),
        )
        buf = StringIO()
        console = Console(file=buf)
        warn_missing_keys(config, console)

        output = buf.getvalue()
        assert "ANTHROPIC_API_KEY" in output
        assert "not set" in output

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-present"}, clear=False)
    def test_no_warning_when_present(self):
        """No warning is printed when the key is present."""
        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        buf = StringIO()
        console = Console(file=buf)
        warn_missing_keys(config, console)

        output = buf.getvalue()
        assert "OPENAI_API_KEY" not in output

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_warns_missing_xai_key(self):
        """A warning is printed when XAI_API_KEY is missing."""
        os.environ.pop("XAI_API_KEY", None)

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="xai")],
            adjudication=AdjudicationConfig(provider="xai"),
        )
        buf = StringIO()
        console = Console(file=buf)
        warn_missing_keys(config, console)

        output = buf.getvalue()
        assert "XAI_API_KEY" in output
        assert "not set" in output

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=False)
    def test_warns_missing_perplexity_key(self):
        """A warning is printed when PERPLEXITY_API_KEY is missing."""
        os.environ.pop("PERPLEXITY_API_KEY", None)

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="perplexity")],
            adjudication=AdjudicationConfig(provider="perplexity"),
        )
        buf = StringIO()
        console = Console(file=buf)
        warn_missing_keys(config, console)

        output = buf.getvalue()
        assert "PERPLEXITY_API_KEY" in output
        assert "not set" in output


# =========================================================================
# 5. validate_api_keys
# =========================================================================

class TestValidateApiKeys:

    @pytest.mark.unit
    @patch("chal.cli.api_keys.prompt_missing_keys")
    def test_interactive_calls_prompt(self, mock_prompt):
        """In interactive mode, prompt_missing_keys is called."""
        config = _minimal_config()
        console = _console()
        validate_api_keys(config, console, interactive=True)

        mock_prompt.assert_called_once_with(config, console)

    @pytest.mark.unit
    @patch("chal.cli.api_keys.warn_missing_keys")
    def test_headless_calls_warn(self, mock_warn):
        """In headless mode, warn_missing_keys is called."""
        config = _minimal_config()
        console = _console()
        validate_api_keys(config, console, interactive=False)

        mock_warn.assert_called_once_with(config, console)

    @pytest.mark.unit
    @patch("chal.cli.api_keys.warn_missing_keys")
    @patch("chal.cli.api_keys.prompt_missing_keys")
    def test_interactive_does_not_call_warn(self, mock_prompt, mock_warn):
        """In interactive mode, warn_missing_keys is NOT called."""
        config = _minimal_config()
        console = _console()
        validate_api_keys(config, console, interactive=True)

        mock_prompt.assert_called_once()
        mock_warn.assert_not_called()

    @pytest.mark.unit
    @patch("chal.cli.api_keys.prompt_missing_keys")
    @patch("chal.cli.api_keys.warn_missing_keys")
    def test_headless_does_not_call_prompt(self, mock_warn, mock_prompt):
        """In headless mode, prompt_missing_keys is NOT called."""
        config = _minimal_config()
        console = _console()
        validate_api_keys(config, console, interactive=False)

        mock_warn.assert_called_once()
        mock_prompt.assert_not_called()
