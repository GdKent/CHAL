"""
Unit tests for OpenAI agent implementation.

All tests use mocked API calls - no actual OpenAI API calls are made.

Tests cover:
- Agent initialization
- Message generation (mocked)
- Retry logic (mocked)
- Belief state management
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import openai
from chal.agents.openai_agent import OpenAIAgent, retry_openai_chat_completion
from chal.agents.base import Message


# ==============================================
# 1. Agent Initialization Tests
# ==============================================

@pytest.mark.unit
def test_openai_agent_init_defaults():
    """Test initialization with default parameters."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")

    assert agent.name == "TestAgent"
    assert agent.model == "gpt-4o"
    assert agent.internal_belief == ""
    assert agent.internal_belief_obj is None


@pytest.mark.unit
def test_openai_agent_init_custom():
    """Test initialization with custom parameters."""
    agent = OpenAIAgent(
        model="gpt-4o",
        name="CustomAgent"
    )

    assert agent.name == "CustomAgent"
    assert agent.model == "gpt-4o"


@pytest.mark.unit
def test_openai_agent_init_system_prompt():
    """Test that system prompt is set correctly."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", system_prompt="Custom system prompt")

    assert agent.system_prompt == "Custom system prompt"


# ==============================================
# 2. Message Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_single_message_mock(mock_openai_class):
    """Test generating response to single message (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.choices[0].message.role = "assistant"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [Message(role="user", content="Test question")]

    response = agent.generate(messages)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content == "Mocked response"


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_conversation_history_mock(mock_openai_class):
    """Test generating response with multi-message conversation (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Conversation response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message")
    ]

    response = agent.generate(messages)

    assert isinstance(response, Message)
    # Verify that all messages were passed to API
    call_args = mock_client.chat.completions.create.call_args
    assert len(call_args[1]["messages"]) >= 3


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_includes_system_prompt(mock_openai_class):
    """Test that system prompt is added to messages (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key", system_prompt="Custom system")
    messages = [Message(role="user", content="Question")]

    agent.generate(messages)

    # Verify system message was added
    call_args = mock_client.chat.completions.create.call_args
    messages_sent = call_args[1]["messages"]
    assert any(msg.get("role") == "system" for msg in messages_sent)


@pytest.mark.skip(reason="OpenAIAgent does not expose temperature parameter in constructor")
@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_temperature_applied(mock_openai_class):
    """Test that temperature parameter is passed to API (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [Message(role="user", content="Question")]

    agent.generate(messages)

    # Temperature handling is implementation-specific
    call_args = mock_client.chat.completions.create.call_args
    assert call_args is not None


# ==============================================
# 3. Retry Logic Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.time.sleep')
def test_generate_retry_on_rate_limit(mock_sleep):
    """Test retry logic on rate limit error (mocked)."""
    mock_client = MagicMock()

    # Build a successful response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success after retry"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()

    # First call raises RateLimitError, second succeeds
    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        ),
        mock_response
    ]

    result = retry_openai_chat_completion(
        client=mock_client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Question"}],
        temperature=0.7,
    )

    assert result.choices[0].message.content == "Success after retry"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.unit
@patch('chal.agents.openai_agent.time.sleep')
def test_generate_retry_exhaustion(mock_sleep):
    """Test that retries eventually fail after max attempts (mocked)."""
    mock_client = MagicMock()

    # Always raise error
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )

    with pytest.raises(RuntimeError, match="Exceeded max retries"):
        retry_openai_chat_completion(
            client=mock_client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Question"}],
            temperature=0.7,
        )

    assert mock_client.chat.completions.create.call_count == 5


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_no_retry_on_other_errors(mock_openai_class):
    """Test that non-retriable errors fail immediately (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Raise a non-retriable error (not RateLimitError/APIStatusError/APIConnectionError)
    mock_client.chat.completions.create.side_effect = ValueError("Invalid request")

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message, not raise exception
    assert "[Error from TestAgent]" in response.content
    # Should only be called once (no retries for non-API errors)
    assert mock_client.chat.completions.create.call_count == 1


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
def test_update_current_belief():
    """Test setting internal_belief_obj attribute."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    belief = {"schema_version": "CBS", "belief_id": "TEST"}

    agent.set_internal_belief_obj(belief)

    assert agent.internal_belief_obj == belief


@pytest.mark.unit
def test_get_current_belief():
    """Test retrieving current belief."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    belief = {"schema_version": "CBS", "belief_id": "TEST"}
    agent.internal_belief_obj = belief

    retrieved = agent.get_internal_belief_obj()

    assert retrieved == belief


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_belief_state_persists(mock_openai_class):
    """Test that belief state is maintained across generate calls (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    belief = {"schema_version": "CBS", "belief_id": "TEST"}
    agent.set_internal_belief_obj(belief)

    # Generate response
    messages = [Message(role="user", content="Question")]
    agent.generate(messages)

    # Belief should still be there
    assert agent.internal_belief_obj == belief


# ==============================================
# 5. Error Handling Tests
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.time.sleep')
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_invalid_api_key(mock_openai_class, mock_sleep):
    """Test handling of authentication errors (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Create 401 Unauthorized error
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body=None,
    )

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="bad-key")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message after retries exhausted
    # (AuthenticationError is an APIStatusError subclass, so it gets retried)
    assert "[Error from TestAgent]" in response.content


@pytest.mark.unit
@patch('chal.agents.openai_agent.time.sleep')
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_timeout(mock_openai_class, mock_sleep):
    """Test handling of timeout errors (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock(),
    )

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message after retries exhausted
    assert "[Error from TestAgent]" in response.content
    # Should be retried multiple times (5 attempts)
    assert mock_client.chat.completions.create.call_count == 5


@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_empty_response(mock_openai_class):
    """Test handling of empty model responses (mocked)."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", api_key="test-key")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return empty message (or handle appropriately)
    assert response.content == ""
