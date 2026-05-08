"""
Unit tests for XAIAgent implementation.

All tests use mocked SDK calls — no actual xAI API access needed.

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
import grpc
from unittest.mock import Mock, patch, MagicMock
from chal.agents.base import Message


class _MockRpcError(grpc.RpcError):
    """Concrete grpc.RpcError for testing retry logic."""
    def __init__(self, code, details="mock error"):
        self._code = code
        self._details = details
    def code(self):
        return self._code
    def details(self):
        return self._details


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_instantiation():
    """All belief-state attributes initialized correctly."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Skeptic", api_key="test-key")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "grok-2"
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
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k", system_prompt="Be concise.")
    assert agent.system_prompt == "Be concise."


@pytest.mark.unit
def test_persona_label_no_prefix():
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Standalone", api_key="k")
    assert agent.persona_label == "Standalone"


@pytest.mark.unit
def test_api_key_from_env(monkeypatch):
    """Key is read from XAI_API_KEY env var when not passed explicitly."""
    from chal.agents.xai_agent import XAIAgent

    monkeypatch.setenv("XAI_API_KEY", "env-xai-key")
    agent = XAIAgent(model="grok-2", name="Agent-Test")
    assert agent.api_key == "env-xai-key"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.xai_agent.retry_api_call')
def test_generate_success(mock_retry):
    """Mocked xAI SDK response is parsed into a Message with role='assistant'."""
    from chal.agents.xai_agent import XAIAgent

    mock_response = MagicMock()
    mock_response.content = "Mocked xAI response"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked xAI response"


@pytest.mark.unit
@patch('chal.agents.xai_agent.xai_user')
@patch('chal.agents.xai_agent.xai_system')
@patch('chal.agents.xai_agent.retry_api_call')
def test_system_prompt_prepended(mock_retry, mock_system, mock_user):
    """Non-empty system_prompt causes xai_system() to be called."""
    from chal.agents.xai_agent import XAIAgent

    sentinel_sys = object()
    sentinel_usr = object()
    mock_system.return_value = sentinel_sys
    mock_user.return_value = sentinel_usr

    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k", system_prompt="Be concise.")
    agent.generate([Message(role="user", content="Question?")])

    mock_system.assert_called_once_with("Be concise.")
    assert mock_retry.call_args.kwargs['provider'] == 'xai'


@pytest.mark.unit
@patch('chal.agents.xai_agent.xai_user')
@patch('chal.agents.xai_agent.xai_assistant')
@patch('chal.agents.xai_agent.retry_api_call')
def test_generate_conversation_history(mock_retry, mock_assistant, mock_user):
    """Multi-turn history is preserved in the call to retry_api_call."""
    from chal.agents.xai_agent import XAIAgent

    sentinel_u1 = object()
    sentinel_a1 = object()
    sentinel_u2 = object()
    mock_user.side_effect = [sentinel_u1, sentinel_u2]
    mock_assistant.return_value = sentinel_a1

    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    history = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message"),
    ]
    agent.generate(history)

    assert mock_retry.call_args.kwargs['provider'] == 'xai'


@pytest.mark.unit
@patch('chal.agents.xai_agent.retry_api_call')
def test_generate_error_returns_error_message(mock_retry):
    """Exception from retry is re-raised by generate()."""
    from chal.agents.xai_agent import XAIAgent

    mock_retry.side_effect = RuntimeError("xAI failure")

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")

    with pytest.raises(RuntimeError, match="xAI failure"):
        agent.generate([Message(role="user", content="Hello?")])


# ==============================================
# 3. System Prompt / Role Card Tests
# ==============================================

@pytest.mark.unit
def test_receive_system_prompt():
    """receive_system_prompt replaces the system_prompt."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
def test_receive_role_card():
    """receive_role_card appends to system_prompt."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k", system_prompt="Base prompt")
    agent.receive_role_card("Role card content")
    assert "Base prompt" in agent.system_prompt
    assert "Role card content" in agent.system_prompt


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
def test_set_get_internal_belief():
    """Belief text roundtrips correctly."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    agent.set_internal_belief("Free will is determined.")
    assert agent.get_internal_belief() == "Free will is determined."


@pytest.mark.unit
def test_set_internal_belief_obj_stores_dict():
    """set_internal_belief_obj stores the belief dict."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
def test_set_internal_belief_obj_none_clears_graph():
    """Setting belief obj to None clears both the graph and the obj."""
    from chal.agents.xai_agent import XAIAgent

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_on_rate_limit(mock_sleep):
    """Rate limit error triggers retry; succeeds on second attempt."""
    from chal.utilities.retry import retry_api_call
    from chal.agents.xai_agent import _XAIRateLimitError

    mock_success = MagicMock()
    mock_success.content = "OK"

    call_count = 0
    def _make_call(rotated_client):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _XAIRateLimitError("resource exhausted")
        return mock_success

    result = retry_api_call(
        call_fn=_make_call,
        provider="xai",
        rate_limit_errors=(_XAIRateLimitError,),
        retryable_errors=(),
        max_retries=5,
        base_delay=1.0,
    )
    assert result.content == "OK"
    assert call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_exhausted_raises(mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    from chal.utilities.retry import retry_api_call
    from chal.agents.xai_agent import _XAIRetryableError

    def _make_call(rotated_client):
        raise _XAIRetryableError("unavailable")

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        retry_api_call(
            call_fn=_make_call,
            provider="xai",
            rate_limit_errors=(),
            retryable_errors=(_XAIRetryableError,),
            max_retries=3,
            base_delay=1.0,
        )


@pytest.mark.unit
@patch('chal.agents.xai_agent.retry_api_call')
def test_generate_catches_runtime_error(mock_retry):
    """generate() re-raises RuntimeError from exhausted retries."""
    from chal.agents.xai_agent import XAIAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for xai API call.")

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        agent.generate([Message(role="user", content="Question")])


# ==============================================
# 6. Error Handling Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.xai_agent.retry_api_call')
def test_generate_empty_response(mock_retry):
    """Empty content from model is returned without crash."""
    from chal.agents.xai_agent import XAIAgent

    mock_response = MagicMock()
    mock_response.content = ""
    mock_response.usage = None
    mock_retry.return_value = mock_response

    agent = XAIAgent(model="grok-2", name="Agent-Test", api_key="k")
    result = agent.generate([Message(role="user", content="Question")])

    assert result.content == ""
