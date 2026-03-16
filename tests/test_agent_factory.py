"""
Unit tests for the agent factory (factory.py).

All tests mock the individual agent constructors so that no SDK or network
calls are needed. The factory is tested for:
- Correct dispatch to each provider class
- Case-insensitive provider names
- Default provider behaviour
- system_prompt forwarding
- Unknown provider rejection
- create_agent_from_config convenience wrapper
"""

import pytest
from unittest.mock import patch, MagicMock
from chal.agents.factory import create_agent, create_agent_from_config
from chal.config import AgentConfig


# ==============================================
# 1. Provider Dispatch Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_create_openai_agent(MockOpenAI):
    """provider='openai' dispatches to OpenAIAgent."""
    agent = create_agent("MyAgent", "gpt-4o", "openai")
    MockOpenAI.assert_called_once_with(model="gpt-4o", name="MyAgent", system_prompt="")
    assert agent is MockOpenAI.return_value


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.AnthropicAgent')
def test_create_anthropic_agent(MockAnthropic):
    """provider='anthropic' dispatches to AnthropicAgent."""
    agent = create_agent("MyAgent", "claude-sonnet-4-6", "anthropic")
    MockAnthropic.assert_called_once_with(
        model="claude-sonnet-4-6", name="MyAgent", system_prompt=""
    )
    assert agent is MockAnthropic.return_value


@pytest.mark.unit
@patch('chal.agents.google_agent.GoogleAgent')
def test_create_google_agent(MockGoogle):
    """provider='google' dispatches to GoogleAgent."""
    agent = create_agent("MyAgent", "gemini-2.0-flash", "google")
    MockGoogle.assert_called_once_with(
        model="gemini-2.0-flash", name="MyAgent", system_prompt=""
    )
    assert agent is MockGoogle.return_value


# ==============================================
# 2. Unknown Provider Test
# ==============================================

@pytest.mark.unit
def test_unknown_provider_raises():
    """An unsupported provider string raises ValueError immediately."""
    with pytest.raises(ValueError, match="Unknown provider"):
        create_agent("MyAgent", "gpt-4o", "cohere")


@pytest.mark.unit
def test_unknown_provider_error_message_lists_valid():
    """The ValueError message names the valid options."""
    with pytest.raises(ValueError) as exc_info:
        create_agent("MyAgent", "gpt-4o", "mistral")
    msg = str(exc_info.value)
    assert "openai" in msg
    assert "anthropic" in msg
    assert "google" in msg


# ==============================================
# 3. Case-Insensitive Provider Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_provider_case_insensitive_openai(MockOpenAI):
    """'OpenAI', 'OPENAI', 'openai' all dispatch correctly."""
    create_agent("A", "gpt-4o", "OpenAI")
    create_agent("B", "gpt-4o", "OPENAI")
    create_agent("C", "gpt-4o", "openai")
    assert MockOpenAI.call_count == 3


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.AnthropicAgent')
def test_provider_case_insensitive_anthropic(MockAnthropic):
    """'ANTHROPIC' and 'Anthropic' both dispatch correctly."""
    create_agent("A", "claude-sonnet-4-6", "ANTHROPIC")
    create_agent("B", "claude-sonnet-4-6", "Anthropic")
    assert MockAnthropic.call_count == 2


@pytest.mark.unit
@patch('chal.agents.google_agent.GoogleAgent')
def test_provider_case_insensitive_google(MockGoogle):
    """'Google' and 'GOOGLE' both dispatch correctly."""
    create_agent("A", "gemini-2.0-flash", "Google")
    create_agent("B", "gemini-2.0-flash", "GOOGLE")
    assert MockGoogle.call_count == 2


# ==============================================
# 4. Default Provider Test
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_default_provider_is_openai(MockOpenAI):
    """Omitting the provider argument defaults to OpenAI."""
    create_agent("MyAgent", "gpt-4o")
    MockOpenAI.assert_called_once()


# ==============================================
# 5. system_prompt Forwarding Test
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_system_prompt_forwarded(MockOpenAI):
    """system_prompt argument is forwarded to the agent constructor."""
    create_agent("MyAgent", "gpt-4o", "openai", system_prompt="Custom prompt")
    MockOpenAI.assert_called_once_with(
        model="gpt-4o", name="MyAgent", system_prompt="Custom prompt"
    )


# ==============================================
# 6. create_agent_from_config Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_create_agent_from_config_openai(MockOpenAI):
    """AgentConfig with provider='openai' is correctly unpacked."""
    cfg = AgentConfig(
        name="Agent-Empiricist", persona="EMPIRICIST", model="gpt-4o", provider="openai"
    )
    agent = create_agent_from_config(cfg)
    MockOpenAI.assert_called_once_with(
        model="gpt-4o", name="Agent-Empiricist", system_prompt=""
    )
    assert agent is MockOpenAI.return_value


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.AnthropicAgent')
def test_create_agent_from_config_anthropic(MockAnthropic):
    """AgentConfig with provider='anthropic' is correctly unpacked."""
    cfg = AgentConfig(
        name="Agent-Claude",
        persona="RATIONALIST",
        model="claude-sonnet-4-6",
        provider="anthropic",
    )
    agent = create_agent_from_config(cfg)
    MockAnthropic.assert_called_once_with(
        model="claude-sonnet-4-6", name="Agent-Claude", system_prompt=""
    )
    assert agent is MockAnthropic.return_value


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAIAgent')
def test_create_agent_from_config_default_provider(MockOpenAI):
    """AgentConfig without explicit provider defaults to openai."""
    cfg = AgentConfig(name="Agent-Default", persona="SKEPTIC")
    # provider defaults to "openai" in AgentConfig dataclass
    create_agent_from_config(cfg)
    MockOpenAI.assert_called_once()


# ==============================================
# 7. Ollama Provider Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.ollama_agent.OllamaAgent')
def test_create_ollama_agent(MockOllama):
    """provider='ollama' dispatches to OllamaAgent."""
    agent = create_agent("MyAgent", "deepseek-r1:14b", "ollama")
    MockOllama.assert_called_once_with(
        model="deepseek-r1:14b", name="MyAgent", system_prompt=""
    )
    assert agent is MockOllama.return_value


@pytest.mark.unit
@patch('chal.agents.ollama_agent.OllamaAgent')
def test_provider_case_insensitive_ollama(MockOllama):
    """'Ollama' and 'OLLAMA' both dispatch correctly."""
    create_agent("A", "deepseek-r1:14b", "Ollama")
    create_agent("B", "deepseek-r1:8b", "OLLAMA")
    assert MockOllama.call_count == 2


@pytest.mark.unit
def test_unknown_provider_error_message_lists_ollama():
    """The ValueError message includes 'ollama' in the valid options list."""
    with pytest.raises(ValueError) as exc_info:
        create_agent("MyAgent", "gpt-4o", "unknown")
    msg = str(exc_info.value)
    assert "ollama" in msg


@pytest.mark.unit
@patch('chal.agents.ollama_agent.OllamaAgent')
def test_create_agent_from_config_ollama(MockOllama):
    """AgentConfig with provider='ollama' is correctly unpacked."""
    cfg = AgentConfig(
        name="Agent-Local",
        persona="EMPIRICIST",
        model="deepseek-r1:14b",
        provider="ollama",
    )
    agent = create_agent_from_config(cfg)
    MockOllama.assert_called_once_with(
        model="deepseek-r1:14b", name="Agent-Local", system_prompt=""
    )
    assert agent is MockOllama.return_value


# ==============================================
# 8. xAI Provider Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.xai_agent.XAIAgent')
def test_create_xai_agent(MockXAI):
    """provider='xai' dispatches to XAIAgent."""
    agent = create_agent("MyAgent", "grok-2", "xai")
    MockXAI.assert_called_once_with(model="grok-2", name="MyAgent", system_prompt="")
    assert agent is MockXAI.return_value


@pytest.mark.unit
@patch('chal.agents.xai_agent.XAIAgent')
def test_provider_case_insensitive_xai(MockXAI):
    """'XAI' and 'xAi' both dispatch correctly."""
    create_agent("A", "grok-2", "XAI")
    create_agent("B", "grok-beta", "xAi")
    assert MockXAI.call_count == 2


@pytest.mark.unit
@patch('chal.agents.xai_agent.XAIAgent')
def test_create_agent_from_config_xai(MockXAI):
    """AgentConfig with provider='xai' is correctly unpacked."""
    cfg = AgentConfig(
        name="Agent-Grok",
        persona="EMPIRICIST",
        model="grok-2",
        provider="xai",
    )
    agent = create_agent_from_config(cfg)
    MockXAI.assert_called_once_with(model="grok-2", name="Agent-Grok", system_prompt="")
    assert agent is MockXAI.return_value


# ==============================================
# 9. Perplexity Provider Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.perplexity_agent.PerplexityAgent')
def test_create_perplexity_agent(MockPerplexity):
    """provider='perplexity' dispatches to PerplexityAgent."""
    agent = create_agent("MyAgent", "sonar-pro", "perplexity")
    MockPerplexity.assert_called_once_with(model="sonar-pro", name="MyAgent", system_prompt="")
    assert agent is MockPerplexity.return_value


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.PerplexityAgent')
def test_provider_case_insensitive_perplexity(MockPerplexity):
    """'Perplexity' and 'PERPLEXITY' both dispatch correctly."""
    create_agent("A", "sonar-pro", "Perplexity")
    create_agent("B", "sonar-reasoning", "PERPLEXITY")
    assert MockPerplexity.call_count == 2


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.PerplexityAgent')
def test_create_agent_from_config_perplexity(MockPerplexity):
    """AgentConfig with provider='perplexity' is correctly unpacked."""
    cfg = AgentConfig(
        name="Agent-Sonar",
        persona="RATIONALIST",
        model="sonar-pro",
        provider="perplexity",
    )
    agent = create_agent_from_config(cfg)
    MockPerplexity.assert_called_once_with(model="sonar-pro", name="Agent-Sonar", system_prompt="")
    assert agent is MockPerplexity.return_value


@pytest.mark.unit
def test_unknown_provider_error_message_lists_xai_and_perplexity():
    """The ValueError message lists xai and perplexity as valid options."""
    with pytest.raises(ValueError) as exc_info:
        create_agent("MyAgent", "grok-2", "unknown")
    msg = str(exc_info.value)
    assert "xai" in msg
    assert "perplexity" in msg
