"""
Unit tests for PerplexityAgent implementation.

All tests use mocked SDK calls — no actual Perplexity API access needed.

Tests cover:
- Agent initialization
- Message generation (mocked)
- System prompt prepending
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
import os
import httpx
import perplexity as perplexity_module
from unittest.mock import Mock, patch, MagicMock
from chal.agents.base import Message


def _make_rate_limit_error():
    """Create a perplexity.RateLimitError suitable for testing."""
    mock_response = httpx.Response(429, request=httpx.Request("POST", "https://api.perplexity.ai"))
    return perplexity_module.RateLimitError("Rate limited", response=mock_response, body=None)


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_instantiation():
    """All belief-state attributes initialized correctly."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Skeptic", api_key="test-key")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "sonar-pro"
    assert agent.api_key == "test-key"
    assert agent.internal_belief == ""
    assert agent.internal_belief_obj is None
    assert agent.belief_graph is None
    assert agent.persona_label == "Skeptic"
    assert agent.all_beliefs_held == []
    assert agent.system_prompt == ""


@pytest.mark.unit
def test_instantiation_system_prompt():
    """System prompt is stored correctly at construction."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k", system_prompt="Be concise.")
    assert agent.system_prompt == "Be concise."


@pytest.mark.unit
def test_persona_label_no_prefix():
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Standalone", api_key="k")
    assert agent.persona_label == "Standalone"


@pytest.mark.unit
def test_api_key_from_env(monkeypatch):
    """Key is read from PERPLEXITY_API_KEY env var when not passed explicitly."""
    from chal.agents.perplexity_agent import PerplexityAgent

    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-perplexity-key")
    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test")
    assert agent.api_key == "env-perplexity-key"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_generate_success(mock_retry):
    """Mocked Perplexity SDK response is parsed into a Message with role='assistant'."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked Perplexity response"
    mock_response.model = "sonar-pro"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked Perplexity response"


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_system_prompt_prepended(mock_retry):
    """Non-empty system_prompt appears as first message in the messages kwarg."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "sonar-pro"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k", system_prompt="Be concise.")
    agent.generate([Message(role="user", content="Question?")])

    messages = mock_retry.call_args.kwargs['messages']
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be concise."


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_generate_conversation_history(mock_retry):
    """Multi-turn history is preserved in the messages kwarg."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "sonar-pro"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    history = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message"),
    ]
    agent.generate(history)

    messages_sent = mock_retry.call_args.kwargs['messages']
    assert len(messages_sent) >= 3
    assert messages_sent[0]["role"] == "user"
    assert messages_sent[1]["role"] == "assistant"
    assert messages_sent[2]["role"] == "user"


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_generate_error_returns_error_message(mock_retry):
    """Exception from retry is caught and returned as a labelled error Message."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_retry.side_effect = RuntimeError("Perplexity failure")

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert "[Error from Agent-Test]" in result.content


# ==============================================
# 3. System Prompt / Role Card Tests
# ==============================================

@pytest.mark.unit
def test_receive_system_prompt():
    """receive_system_prompt replaces the system_prompt."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
def test_receive_role_card():
    """receive_role_card appends to system_prompt."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k", system_prompt="Base prompt")
    agent.receive_role_card("Role card content")
    assert "Base prompt" in agent.system_prompt
    assert "Role card content" in agent.system_prompt


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
def test_set_get_internal_belief():
    """Belief text roundtrips correctly."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    agent.set_internal_belief("Free will is determined.")
    assert agent.get_internal_belief() == "Free will is determined."


@pytest.mark.unit
def test_set_internal_belief_obj_stores_dict():
    """set_internal_belief_obj stores the belief dict."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
def test_set_internal_belief_obj_none_clears_graph():
    """Setting belief obj to None clears both the graph and the obj."""
    from chal.agents.perplexity_agent import PerplexityAgent

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.perplexity_agent.time.sleep')
def test_retry_on_rate_limit(mock_sleep):
    """RateLimitError triggers retry; succeeds on second attempt."""
    import chal.agents.perplexity_agent as mod

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OK"

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_rate_limit_error(),
        mock_response,
    ]

    result = mod.retry_perplexity_chat_completion(
        client=mock_client, model="sonar-pro", messages=[], temperature=0.7,
        max_retries=5, base_delay=1.0,
    )
    assert result.choices[0].message.content == "OK"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.time.sleep')
def test_retry_exhausted_raises(mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    import chal.agents.perplexity_agent as mod

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        mod.retry_perplexity_chat_completion(
            client=mock_client, model="sonar-pro", messages=[], temperature=0.7,
            max_retries=3, base_delay=1.0,
        )

    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_generate_catches_runtime_error(mock_retry):
    """generate() wraps any RuntimeError from retry as a labelled error Message."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for Perplexity API call.")

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Question")])

    assert "[Error from Agent-Test]" in result.content
    assert result.role == "assistant"


# ==============================================
# 6. Error Handling Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.perplexity_agent.retry_perplexity_chat_completion')
def test_generate_empty_response(mock_retry):
    """Empty content from model is returned without crash."""
    from chal.agents.perplexity_agent import PerplexityAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.model = "sonar-pro"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = PerplexityAgent(model="sonar-pro", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Question")])

    assert result.content == ""
