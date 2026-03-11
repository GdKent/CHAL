"""
Unit tests for OllamaAgent implementation.

All tests use mocked Ollama calls — no actual Ollama server or models needed.

Tests cover:
- Agent initialization
- Message generation (mocked)
- System prompt prepending
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from chal.agents.base import Message


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_instantiation():
    """All belief-state attributes initialized correctly; api_key is not stored."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Skeptic")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "deepseek-r1:14b"
    assert agent.internal_belief == ""
    assert agent.internal_belief_obj is None
    assert agent.belief_graph is None
    assert agent.persona_label == "Skeptic"
    assert agent.all_beliefs_held == []
    assert agent.system_prompt == ""
    assert not hasattr(agent, "api_key")


@pytest.mark.unit
def test_instantiation_system_prompt():
    """System prompt is stored correctly at construction."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(
        model="deepseek-r1:14b", name="Agent-Test", system_prompt="Custom prompt"
    )
    assert agent.system_prompt == "Custom prompt"


@pytest.mark.unit
def test_persona_label_no_prefix():
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Standalone")
    assert agent.persona_label == "Standalone"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.ollama_agent.retry_ollama_chat')
def test_generate_success(mock_retry):
    """Mocked Ollama response is parsed into a Message with role='assistant'."""
    from chal.agents.ollama_agent import OllamaAgent

    mock_response = MagicMock()
    mock_response.message.content = "Mocked response"
    mock_retry.return_value = mock_response

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked response"


@pytest.mark.unit
@patch('chal.agents.ollama_agent.retry_ollama_chat')
def test_system_prompt_prepended(mock_retry):
    """Non-empty system_prompt appears as first message passed to retry_ollama_chat."""
    from chal.agents.ollama_agent import OllamaAgent

    mock_response = MagicMock()
    mock_response.message.content = "Response"
    mock_retry.return_value = mock_response

    agent = OllamaAgent(
        model="deepseek-r1:14b", name="Agent-Test", system_prompt="Be concise."
    )
    agent.generate([Message(role="user", content="Question?")])

    messages_sent = mock_retry.call_args[0][1]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[0]["content"] == "Be concise."


@pytest.mark.unit
@patch('chal.agents.ollama_agent.retry_ollama_chat')
def test_generate_error_returns_error_message(mock_retry):
    """Exception from retry is caught and returned as a labelled error Message."""
    from chal.agents.ollama_agent import OllamaAgent

    mock_retry.side_effect = RuntimeError("Ollama failure")

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
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
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
def test_receive_role_card():
    """receive_role_card appends to system_prompt."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(
        model="deepseek-r1:14b", name="Agent-Test", system_prompt="Base prompt"
    )
    agent.receive_role_card("Role card content")
    assert "Base prompt" in agent.system_prompt
    assert "Role card content" in agent.system_prompt


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
def test_set_get_internal_belief():
    """Belief text roundtrips correctly."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    agent.set_internal_belief("Free will is determined.")
    assert agent.get_internal_belief() == "Free will is determined."


@pytest.mark.unit
def test_set_internal_belief_obj_stores_dict():
    """set_internal_belief_obj stores the belief dict (graph build may silently fail)."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
def test_set_internal_belief_obj_none_clears_graph():
    """Setting belief obj to None clears both the graph and the obj."""
    from chal.agents.ollama_agent import OllamaAgent

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests (Direct on retry_ollama_chat)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.ollama_agent.time.sleep')
@patch('chal.agents.ollama_agent.ollama')
def test_retry_on_server_error(mock_ollama_mod, mock_sleep):
    """5xx ResponseError triggers retry; succeeds on second attempt."""
    import chal.agents.ollama_agent as mod

    class FakeResponseError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    mock_success = MagicMock()
    mock_ollama_mod.ResponseError = FakeResponseError
    mock_ollama_mod.chat.side_effect = [
        FakeResponseError("server error", 500),
        mock_success,
    ]

    result = mod.retry_ollama_chat("deepseek-r1:14b", [], 0.7, max_retries=5)

    assert result is mock_success
    assert mock_ollama_mod.chat.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.agents.ollama_agent.time.sleep')
@patch('chal.agents.ollama_agent.ollama')
def test_retry_exhausted(mock_ollama_mod, mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    import chal.agents.ollama_agent as mod

    class FakeResponseError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    mock_ollama_mod.ResponseError = FakeResponseError
    mock_ollama_mod.chat.side_effect = FakeResponseError("server error", 500)

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        mod.retry_ollama_chat("deepseek-r1:14b", [], 0.7, max_retries=3)

    assert mock_ollama_mod.chat.call_count == 3


@pytest.mark.unit
@patch('chal.agents.ollama_agent.ollama')
def test_model_not_found_raises_immediately(mock_ollama_mod):
    """404 ResponseError raises RuntimeError immediately without retrying."""
    import chal.agents.ollama_agent as mod

    class FakeResponseError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    mock_ollama_mod.ResponseError = FakeResponseError
    mock_ollama_mod.chat.side_effect = FakeResponseError("not found", 404)

    with pytest.raises(RuntimeError, match="ollama pull"):
        mod.retry_ollama_chat("no-such-model:7b", [], 0.7, max_retries=5)

    # Only called once — no retries on 404
    assert mock_ollama_mod.chat.call_count == 1


@pytest.mark.unit
@patch('chal.agents.ollama_agent.ollama')
def test_connection_refused_raises_immediately(mock_ollama_mod):
    """ConnectionRefusedError raises RuntimeError immediately without retrying."""
    import chal.agents.ollama_agent as mod

    # FakeResponseError must NOT be a parent of ConnectionRefusedError so the
    # except clauses in retry_ollama_chat are tested independently.
    class FakeResponseError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    mock_ollama_mod.ResponseError = FakeResponseError
    mock_ollama_mod.chat.side_effect = ConnectionRefusedError("connection refused")

    with pytest.raises(RuntimeError, match="ollama serve"):
        mod.retry_ollama_chat("deepseek-r1:14b", [], 0.7, max_retries=5)

    assert mock_ollama_mod.chat.call_count == 1


@pytest.mark.unit
@patch('chal.agents.ollama_agent.retry_ollama_chat')
def test_generate_catches_runtime_error(mock_retry):
    """generate() wraps any RuntimeError from retry as a labelled error Message."""
    from chal.agents.ollama_agent import OllamaAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for Ollama model 'deepseek-r1:14b'.")

    agent = OllamaAgent(model="deepseek-r1:14b", name="Agent-Test")
    result = agent.generate([Message(role="user", content="Question")])

    assert "[Error from Agent-Test]" in result.content
    assert result.role == "assistant"
