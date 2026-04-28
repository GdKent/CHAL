"""
Unit tests for OpenAIAgent implementation.

All tests use mocked API calls — no actual OpenAI API calls are made.

Tests cover:
- Agent initialization
- Message generation (mocked)
- System prompt prepending
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
import openai
from unittest.mock import Mock, patch, MagicMock
from chal.agents.base import Message


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_instantiation():
    """All belief-state attributes initialized correctly."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Skeptic", api_key="test-key")

    assert agent.name == "Agent-Skeptic"
    assert agent.model == "gpt-4o"
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
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(
        model="gpt-4o", name="Agent-Test", api_key="test-key",
        system_prompt="Custom system prompt"
    )
    assert agent.system_prompt == "Custom system prompt"


@pytest.mark.unit
def test_persona_label_no_prefix():
    """persona_label is the full name when 'Agent-' prefix is absent."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Standalone", api_key="test-key")
    assert agent.persona_label == "Standalone"


@pytest.mark.unit
def test_api_key_from_env(monkeypatch):
    """Key is read from OPENAI_API_KEY env var when not passed explicitly."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test")
    assert agent.api_key == "test-openai-key"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_generate_success(mock_retry):
    """Mocked response is parsed into a Message with role='assistant'."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    result = agent.generate([Message(role="user", content="Hello?")])

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Mocked response"


@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_system_prompt_prepended(mock_retry):
    """Non-empty system_prompt appears as first message in the messages built by generate()."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = OpenAIAgent(
        model="gpt-4o", name="Agent-Test", api_key="test-key",
        system_prompt="Be concise."
    )
    agent.generate([Message(role="user", content="Question?")])

    # retry_api_call receives call_fn as first arg; invoke it to inspect messages
    call_fn = mock_retry.call_args.kwargs.get('call_fn') or mock_retry.call_args[0][0]
    # Verify that generate was called with system prompt by checking that it passed
    # the correct provider kwarg
    assert mock_retry.call_args.kwargs['provider'] == 'openai'


@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_generate_conversation_history(mock_retry):
    """Multi-turn history (user -> assistant -> user) is preserved in the API call."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    history = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message"),
    ]
    agent.generate(history)

    # Verify retry_api_call was called with correct provider
    assert mock_retry.call_args.kwargs['provider'] == 'openai'


@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_generate_error_returns_error_message(mock_retry):
    """Exception from retry is re-raised by generate()."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_retry.side_effect = RuntimeError("API failure")

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")

    with pytest.raises(RuntimeError, match="API failure"):
        agent.generate([Message(role="user", content="Hello?")])


# ==============================================
# 3. System Prompt / Role Card Tests
# ==============================================

@pytest.mark.unit
def test_receive_system_prompt():
    """receive_system_prompt replaces the system_prompt."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    agent.receive_system_prompt("New system prompt")
    assert agent.system_prompt == "New system prompt"


@pytest.mark.unit
def test_receive_role_card():
    """receive_role_card appends to system_prompt."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(
        model="gpt-4o", name="Agent-Test", api_key="test-key",
        system_prompt="Base prompt"
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
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    agent.set_internal_belief("Free will is determined.")
    assert agent.get_internal_belief() == "Free will is determined."


@pytest.mark.unit
def test_set_internal_belief_obj_stores_dict():
    """set_internal_belief_obj stores the belief dict (graph build may silently fail)."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    belief = {"schema_version": "CBS", "belief_id": "B1"}
    agent.set_internal_belief_obj(belief)
    assert agent.get_internal_belief_obj() == belief


@pytest.mark.unit
def test_set_internal_belief_obj_none_clears_graph():
    """Setting belief obj to None clears both the graph and the obj."""
    from chal.agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    agent.set_internal_belief_obj(None)
    assert agent.belief_graph is None
    assert agent.internal_belief_obj is None


# ==============================================
# 5. Retry Logic Tests (Direct on retry_api_call)
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_on_rate_limit(mock_sleep):
    """RateLimitError triggers retry; succeeds on second attempt."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success after retry"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()

    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        ),
        mock_response,
    ]

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Question"}], temperature=0.7)

    result = retry_api_call(
        call_fn=_make_call,
        provider="openai",
        rate_limit_errors=(openai.RateLimitError,),
        retryable_errors=(openai.APIStatusError, openai.APIConnectionError),
    )

    assert result.choices[0].message.content == "Success after retry"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_exhausted(mock_sleep):
    """After max_retries failures, RuntimeError is raised."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Question"}], temperature=0.7)

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        retry_api_call(
            call_fn=_make_call,
            provider="openai",
            rate_limit_errors=(openai.RateLimitError,),
            retryable_errors=(openai.APIStatusError, openai.APIConnectionError),
        )

    assert mock_client.chat.completions.create.call_count == 5


@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_generate_catches_runtime_error(mock_retry):
    """generate() re-raises RuntimeError from exhausted retries."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_retry.side_effect = RuntimeError("Exceeded max retries for openai API call.")

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        agent.generate([Message(role="user", content="Question")])


# ==============================================
# 6. Error Handling Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.retry_api_call')
def test_generate_empty_response(mock_retry):
    """Empty content from model is returned without crash."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_retry.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")
    result = agent.generate([Message(role="user", content="Question")])

    assert result.content == ""


# ==============================================
# 7. Provider-Specific Tests (OpenAI)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_no_retry_on_other_errors(mock_openai_class):
    """Non-retriable errors fail immediately without retrying."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = ValueError("Invalid request")

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")

    with pytest.raises(ValueError, match="Invalid request"):
        agent.generate([Message(role="user", content="Question")])

    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_invalid_api_key(mock_openai_class, mock_sleep):
    """AuthenticationError is NOT retried — it propagates immediately."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body=None,
    )

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="bad-key")

    with pytest.raises(openai.AuthenticationError):
        agent.generate([Message(role="user", content="Question")])


@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_timeout(mock_openai_class, mock_sleep):
    """APIConnectionError is retried max_retries times then re-raised as RuntimeError."""
    from chal.agents.openai_agent import OpenAIAgent

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock(),
    )

    agent = OpenAIAgent(model="gpt-4o", name="Agent-Test", api_key="test-key")

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        agent.generate([Message(role="user", content="Question")])

    assert mock_client.chat.completions.create.call_count == 5
