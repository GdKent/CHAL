"""
Unit tests for AnthropicAgent implementation.

All tests use mocked API calls — no actual Anthropic API calls are made.

Tests cover:
- Agent initialization
- Message generation (mocked)
- System-role filtering
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from chal.agents.base import Message


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_instantiation(mock_anthropic_class):
    """AnthropicAgent sets all belief-state attributes correctly."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Skeptic")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "claude-sonnet-4-6"
    assert agent.internal_belief == ""
    assert agent.internal_belief_obj is None
    assert agent.belief_graph is None
    assert agent.persona_label == "Skeptic"
    assert agent.all_beliefs_held == []
    assert agent.system_prompt == ""


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_instantiation_system_prompt(mock_anthropic_class):
    """System prompt is stored correctly at construction."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(
        model="claude-sonnet-4-6", name="Agent-Test", system_prompt="Custom prompt"
    )
    assert agent.system_prompt == "Custom prompt"


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_persona_label_no_prefix(mock_anthropic_class):
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Standalone")
    assert agent.persona_label == "Standalone"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
@patch('chal.agents.anthropic_agent.retry_anthropic_message')
def test_generate_success(mock_retry, mock_anthropic_class):
    """Mocked Anthropic response is parsed into a Message."""
    from chal.agents.anthropic_agent import AnthropicAgent

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked response")]
    mock_response.model = "claude-sonnet-4-6"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked response"


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
@patch('chal.agents.anthropic_agent.retry_anthropic_message')
def test_generate_filters_system_role(mock_retry, mock_anthropic_class):
    """'system' messages in history are stripped before the API call."""
    from chal.agents.anthropic_agent import AnthropicAgent

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]
    mock_response.model = "claude-sonnet-4-6"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    history = [
        Message(role="system", content="System instruction"),
        Message(role="user", content="User question"),
        Message(role="assistant", content="Prior answer"),
    ]
    agent.generate(history)

    _args, call_kwargs = mock_retry.call_args
    messages_sent = call_kwargs["messages"]
    assert not any(m["role"] == "system" for m in messages_sent)
    assert len(messages_sent) == 2  # user + assistant only


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
@patch('chal.agents.anthropic_agent.retry_anthropic_message')
def test_generate_error_returns_error_message(mock_retry, mock_anthropic_class):
    """Exceptions from retry are caught and returned as a labelled error Message."""
    from chal.agents.anthropic_agent import AnthropicAgent

    mock_retry.side_effect = RuntimeError("API failure")

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert "[Error from Agent-Test]" in result.content


# ==============================================
# 3. System Prompt / Role Card Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_receive_system_prompt(mock_anthropic_class):
    """receive_system_prompt replaces the system_prompt."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_receive_role_card(mock_anthropic_class):
    """receive_role_card appends to system_prompt."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(
        model="claude-sonnet-4-6", name="Agent-Test", system_prompt="Base prompt"
    )
    agent.receive_role_card("Role card content")
    assert "Base prompt" in agent.system_prompt
    assert "Role card content" in agent.system_prompt


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_set_get_internal_belief(mock_anthropic_class):
    """Belief text roundtrips correctly."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    agent.set_internal_belief("Free will is determined.")
    assert agent.get_internal_belief() == "Free will is determined."


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_set_internal_belief_obj_stores_dict(mock_anthropic_class):
    """set_internal_belief_obj stores the belief dict (graph build may silently fail)."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_set_internal_belief_obj_none_clears_graph(mock_anthropic_class):
    """Setting belief obj to None clears the belief graph."""
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests (Direct on retry function)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.time.sleep')
def test_retry_on_rate_limit(mock_sleep):
    """Rate limit error triggers retry; succeeds on second attempt."""
    import chal.agents.anthropic_agent as mod

    mock_client = MagicMock()
    mock_success = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_success,
    ]

    with patch.object(mod.anthropic, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIStatusError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIConnectionError', FakeRateLimitError):
        result = mod.retry_anthropic_message(
            client=mock_client,
            model="claude-sonnet-4-6",
            system_prompt="",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    assert result is mock_success
    assert mock_client.messages.create.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.time.sleep')
def test_retry_exhausted(mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    import chal.agents.anthropic_agent as mod

    mock_client = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = FakeRateLimitError("rate limited")

    with patch.object(mod.anthropic, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIStatusError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIConnectionError', FakeRateLimitError):
        with pytest.raises(RuntimeError, match="Exceeded max retries"):
            mod.retry_anthropic_message(
                client=mock_client,
                model="claude-sonnet-4-6",
                system_prompt="",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_retries=3,
            )

    assert mock_client.messages.create.call_count == 3


@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
@patch('chal.agents.anthropic_agent.retry_anthropic_message')
def test_generate_catches_exhausted_retry(mock_retry, mock_anthropic_class):
    """generate() wraps a RuntimeError from exhausted retries as an error Message."""
    from chal.agents.anthropic_agent import AnthropicAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for Anthropic API call.")

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Question")])

    assert "[Error from Agent-Test]" in result.content


# ==============================================
# 6. API Key Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.anthropic.Anthropic')
def test_api_key_from_env(mock_anthropic_class, monkeypatch):
    """Key is read from ANTHROPIC_API_KEY env var when not passed explicitly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-env")
    from chal.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(model="claude-sonnet-4-6", name="Agent-Test")
    assert agent.api_key == "test-key-from-env"
