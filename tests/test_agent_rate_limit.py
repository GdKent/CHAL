"""
Unit tests for the on_rate_limit callback in agent retry functions.

Tests cover all three provider retry utilities:
- retry_anthropic_message  (anthropic_agent.py)
- retry_openai_chat_completion  (openai_agent.py)
- retry_google_generate  (google_agent.py)

All tests use mocked API calls and fake exception classes so that no real
SDK clients or network calls are needed.
"""

import pytest
from unittest.mock import MagicMock, patch


# ==============================================
# 1. Anthropic — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.time.sleep')
def test_retry_anthropic_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when RateLimitError occurs."""
    import chal.agents.anthropic_agent as mod

    mock_client = MagicMock()
    mock_response = MagicMock()

    # Use a fake exception that the isinstance() check will match
    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    callback = MagicMock()

    with patch.object(mod.anthropic, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIStatusError', type('Unused', (Exception,), {})), \
         patch.object(mod.anthropic, 'APIConnectionError', type('Unused', (Exception,), {})):
        result = mod.retry_anthropic_message(
            client=mock_client,
            model="claude-sonnet-4-6",
            system_prompt="test",
            messages=[],
            temperature=0.7,
            on_rate_limit=callback,
        )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 2. OpenAI — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.agents.openai_agent.time.sleep')
def test_retry_openai_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when RateLimitError occurs."""
    import chal.agents.openai_agent as mod

    mock_client = MagicMock()
    mock_response = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.chat.completions.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    callback = MagicMock()

    with patch.object(mod.openai, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.openai, 'APIStatusError', type('Unused', (Exception,), {})), \
         patch.object(mod.openai, 'APIConnectionError', type('Unused', (Exception,), {})):
        result = mod.retry_openai_chat_completion(
            client=mock_client,
            model="gpt-4o",
            messages=[],
            temperature=0.7,
            on_rate_limit=callback,
        )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 3. Google — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.agents.google_agent.time.sleep')
def test_retry_google_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when a 429 APIError occurs."""
    import chal.agents.google_agent as mod

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "test response"

    # Google checks getattr(e, 'code', None) == 429 or '429' in str(e)
    class FakeAPIError(Exception):
        pass

    rate_limit_err = FakeAPIError("rate limited")
    rate_limit_err.code = 429

    mock_client.models.generate_content.side_effect = [
        rate_limit_err,
        mock_response,
    ]

    callback = MagicMock()

    with patch.object(mod.genai_errors, 'APIError', FakeAPIError):
        result = mod.retry_google_generate(
            client=mock_client,
            model="gemini-2.0-flash",
            contents=[],
            system_prompt="test",
            temperature=0.7,
            on_rate_limit=callback,
        )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 4. on_rate_limit=None (default) still retries
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.time.sleep')
def test_retry_without_on_rate_limit_still_works(mock_sleep):
    """Passing on_rate_limit=None (the default) still retries without crashing."""
    import chal.agents.anthropic_agent as mod

    mock_client = MagicMock()
    mock_response = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    with patch.object(mod.anthropic, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIStatusError', type('Unused', (Exception,), {})), \
         patch.object(mod.anthropic, 'APIConnectionError', type('Unused', (Exception,), {})):
        # on_rate_limit is intentionally omitted (defaults to None)
        result = mod.retry_anthropic_message(
            client=mock_client,
            model="claude-sonnet-4-6",
            system_prompt="test",
            messages=[],
            temperature=0.7,
        )

    assert result is mock_response
    assert mock_client.messages.create.call_count == 2


# ==============================================
# 5. Callback NOT called for non-rate-limit errors
# ==============================================

@pytest.mark.unit
@patch('chal.agents.anthropic_agent.time.sleep')
def test_on_rate_limit_not_called_on_non_rate_limit_error(mock_sleep):
    """on_rate_limit callback is NOT invoked for non-rate-limit errors."""
    import chal.agents.anthropic_agent as mod

    mock_client = MagicMock()
    mock_response = MagicMock()

    # Two distinct fake exception types so isinstance() can tell them apart
    class FakeRateLimitError(Exception):
        pass

    class FakeAPIConnectionError(Exception):
        pass

    # First call raises a connection error (not rate limit), second succeeds
    mock_client.messages.create.side_effect = [
        FakeAPIConnectionError("connection failed"),
        mock_response,
    ]

    callback = MagicMock()

    with patch.object(mod.anthropic, 'RateLimitError', FakeRateLimitError), \
         patch.object(mod.anthropic, 'APIStatusError', FakeAPIConnectionError), \
         patch.object(mod.anthropic, 'APIConnectionError', FakeAPIConnectionError):
        result = mod.retry_anthropic_message(
            client=mock_client,
            model="claude-sonnet-4-6",
            system_prompt="test",
            messages=[],
            temperature=0.7,
            on_rate_limit=callback,
        )

    # The callback must NOT have been called for a non-rate-limit error
    assert callback.call_count == 0
    assert result is mock_response
