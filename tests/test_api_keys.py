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
    def test_returns_false_when_key_missing(self, monkeypatch):
        """check_api_keys returns False for a provider whose key is not set."""
        monkeypatch.delenv(PROVIDER_ENV_VARS["anthropic"], raising=False)

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
    def test_xai_key_missing(self, monkeypatch):
        """check_api_keys returns False for xAI when XAI_API_KEY is not set."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
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
    def test_perplexity_key_missing(self, monkeypatch):
        """check_api_keys returns False for Perplexity when PERPLEXITY_API_KEY is not set."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
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
    def test_prompts_for_missing_key(self, mock_questionary, monkeypatch):
        """When a key is missing, the user is prompted and the env var is set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        mock_questionary.text.return_value.ask.return_value = "sk-ant-secret"
        mock_questionary.confirm.return_value.ask.return_value = False

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
    def test_skip_when_empty_input(self, mock_questionary, monkeypatch):
        """When the user presses Enter (empty input), the key is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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
    def test_ctrl_c_raises_keyboard_interrupt(self, mock_questionary, monkeypatch):
        """When questionary returns None (Ctrl+C), KeyboardInterrupt is raised."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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
    def test_prompts_for_missing_xai_key(self, mock_questionary, monkeypatch):
        """When XAI_API_KEY is missing, the user is prompted and the env var is set."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        mock_questionary.text.return_value.ask.return_value = "xai-secret"
        mock_questionary.confirm.return_value.ask.return_value = False

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
    def test_prompts_for_missing_perplexity_key(self, mock_questionary, monkeypatch):
        """When PERPLEXITY_API_KEY is missing, the user is prompted and the env var is set."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

        mock_questionary.text.return_value.ask.return_value = "pplx-secret"
        mock_questionary.confirm.return_value.ask.return_value = False

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="perplexity")],
            adjudication=AdjudicationConfig(provider="perplexity"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        mock_questionary.text.assert_called_once()
        assert os.environ.get("PERPLEXITY_API_KEY") == "pplx-secret"

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    def test_multi_key_entry(self, mock_questionary, monkeypatch):
        """User enters two keys, confirm yes then no → comma-separated env var."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_questionary.text.return_value.ask.side_effect = ["sk-key1", "sk-key2"]
        mock_questionary.confirm.return_value.ask.side_effect = [True, False]

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        assert os.environ.get("OPENAI_API_KEY") == "sk-key1,sk-key2"

    @pytest.mark.unit
    @patch("chal.cli.api_keys.questionary")
    def test_single_key_declined_extra(self, mock_questionary, monkeypatch):
        """User enters one key, declines adding more → single value in env var."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_questionary.text.return_value.ask.return_value = "sk-only"
        mock_questionary.confirm.return_value.ask.return_value = False

        config = _minimal_config(
            agents=[AgentConfig(name="A", persona="EMPIRICIST", provider="openai")],
            adjudication=AdjudicationConfig(provider="openai"),
        )
        console = _console()
        prompt_missing_keys(config, console)

        assert os.environ.get("OPENAI_API_KEY") == "sk-only"


# =========================================================================
# 4. warn_missing_keys
# =========================================================================

class TestWarnMissingKeys:

    @pytest.mark.unit
    def test_warns_about_missing_key(self, monkeypatch):
        """A warning is printed for a missing key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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
    def test_warns_missing_xai_key(self, monkeypatch):
        """A warning is printed when XAI_API_KEY is missing."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)

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
    def test_warns_missing_perplexity_key(self, monkeypatch):
        """A warning is printed when PERPLEXITY_API_KEY is missing."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

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
