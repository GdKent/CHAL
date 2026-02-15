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
from chal.agents.openai_agent import OpenAIAgent
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
@patch('chal.agents.openai_agent.httpx.post')
def test_generate_single_message_mock(mock_post):
    """Test generating response to single message (mocked)."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Mocked response"
            }
        }]
    }
    mock_post.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Test question")]

    response = agent.generate(messages)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content == "Mocked response"


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
def test_generate_conversation_history_mock(mock_post):
    """Test generating response with multi-message conversation (mocked)."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Conversation response"
            }
        }]
    }
    mock_post.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [
        Message(role="user", content="First message"),
        Message(role="assistant", content="First response"),
        Message(role="user", content="Second message")
    ]

    response = agent.generate(messages)

    assert isinstance(response, Message)
    # Verify that all messages were passed to API
    call_args = mock_post.call_args
    assert len(call_args[1]["json"]["messages"]) >= 3


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
def test_generate_includes_system_prompt(mock_post):
    """Test that system prompt is added to messages (mocked)."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Response"
            }
        }]
    }
    mock_post.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent", system_prompt="Custom system")
    messages = [Message(role="user", content="Question")]

    agent.generate(messages)

    # Verify system message was added
    call_args = mock_post.call_args
    messages_sent = call_args[1]["json"]["messages"]
    assert any(msg.get("role") == "system" for msg in messages_sent)


@pytest.mark.skip(reason="OpenAIAgent does not expose temperature parameter in constructor")
@pytest.mark.unit
@patch('chal.agents.openai_agent.OpenAI')
def test_generate_temperature_applied(mock_openai_class):
    """Test that temperature parameter is passed to API (mocked)."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_client.chat.completions.create.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    agent.generate(messages)

    # Temperature handling is implementation-specific
    call_args = mock_client.chat.completions.create.call_args
    assert call_args is not None


# ==============================================
# 3. Retry Logic Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
@patch('chal.agents.openai_agent.time.sleep')  # Mock sleep to speed up test
def test_generate_retry_on_rate_limit(mock_sleep, mock_post):
    """Test retry logic on rate limit error (mocked)."""
    # First call raises HTTPStatusError (rate limit), second succeeds
    from httpx import HTTPStatusError, Request, Response

    # Create mock request and response for HTTPStatusError
    mock_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_error_response = Response(429, text="Rate limit exceeded")

    # Second call returns success
    mock_success_response = MagicMock()
    mock_success_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Success after retry"
            }
        }]
    }

    # First call raises error, second succeeds
    mock_post.side_effect = [
        HTTPStatusError("429 Rate Limit", request=mock_request, response=mock_error_response),
        mock_success_response
    ]

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    assert response.content == "Success after retry"
    assert mock_post.call_count == 2


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
@patch('chal.agents.openai_agent.time.sleep')
def test_generate_retry_exhaustion(mock_sleep, mock_post):
    """Test that retries eventually fail after max attempts (mocked)."""
    # NOTE: Implementation catches all exceptions and returns error Message
    # So we expect an error message, not an exception
    from httpx import HTTPStatusError, Request, Response

    # Create mock request and response for HTTPStatusError
    mock_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_error_response = Response(429, text="Rate limit exceeded")

    # Always raise error
    mock_post.side_effect = HTTPStatusError(
        "429 Rate Limit", request=mock_request, response=mock_error_response
    )

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message (not raise exception)
    assert "[Error from TestAgent]" in response.content
    assert "RuntimeError" in response.content or "Exceeded max retries" in response.content


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
def test_generate_no_retry_on_other_errors(mock_post):
    """Test that non-retriable errors fail immediately (mocked)."""
    # NOTE: Implementation catches all exceptions and returns error Message
    # Non-retriable errors (like ValueError) should only be called once

    # Raise a non-retriable error (not HTTPStatusError/TimeoutException/RequestError)
    mock_post.side_effect = ValueError("Invalid request")

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message, not raise exception
    assert "[Error from TestAgent]" in response.content
    # Should only be called once (no retries for non-HTTP errors)
    assert mock_post.call_count == 1


# ==============================================
# 4. Belief State Management Tests
# ==============================================

@pytest.mark.unit
def test_update_current_belief():
    """Test setting internal_belief_obj attribute."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    belief = {"schema_version": "CBS-v1", "belief_id": "TEST"}

    agent.set_internal_belief_obj(belief)

    assert agent.internal_belief_obj == belief


@pytest.mark.unit
def test_get_current_belief():
    """Test retrieving current belief."""
    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    belief = {"schema_version": "CBS-v1", "belief_id": "TEST"}
    agent.internal_belief_obj = belief

    retrieved = agent.get_internal_belief_obj()

    assert retrieved == belief


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
def test_belief_state_persists(mock_post):
    """Test that belief state is maintained across generate calls (mocked)."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Response"
            }
        }]
    }
    mock_post.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    belief = {"schema_version": "CBS-v1", "belief_id": "TEST"}
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
@patch('chal.agents.openai_agent.httpx.post')
@patch('chal.agents.openai_agent.time.sleep')  # Mock sleep to prevent hanging
def test_generate_invalid_api_key(mock_sleep, mock_post):
    """Test handling of authentication errors (mocked)."""
    # NOTE: Implementation catches all exceptions and returns error Message
    # HTTPStatusError triggers retry logic, so we need to mock sleep
    from httpx import HTTPStatusError, Request, Response

    # Create 401 Unauthorized response
    mock_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_error_response = Response(401, text="Invalid API key")

    mock_post.side_effect = HTTPStatusError(
        "401 Unauthorized", request=mock_request, response=mock_error_response
    )

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message after retries exhausted
    assert "[Error from TestAgent]" in response.content


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
@patch('chal.agents.openai_agent.time.sleep')
def test_generate_timeout(mock_sleep, mock_post):
    """Test handling of timeout errors (mocked)."""
    # NOTE: Implementation catches all exceptions and returns error Message
    # TimeoutException gets retried, so after max retries it returns error
    from httpx import TimeoutException

    mock_post.side_effect = TimeoutException("Request timed out")

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return error message after retries exhausted
    assert "[Error from TestAgent]" in response.content
    # Should be retried multiple times (5 attempts)
    assert mock_post.call_count == 5


@pytest.mark.unit
@patch('chal.agents.openai_agent.httpx.post')
def test_generate_empty_response(mock_post):
    """Test handling of empty model responses (mocked)."""
    # Setup mock with empty content
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": ""
            }
        }]
    }
    mock_post.return_value = mock_response

    agent = OpenAIAgent(model="gpt-4o", name="TestAgent")
    messages = [Message(role="user", content="Question")]

    response = agent.generate(messages)

    # Should return empty message (or handle appropriately)
    assert response.content == ""
