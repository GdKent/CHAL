"""
Unit tests for the on_rate_limit callback in the generic retry_api_call function.

Tests cover the callback behavior across different provider scenarios:
- on_rate_limit callback fires on rate limit errors
- on_rate_limit=None (default) still retries without crashing
- Callback NOT called for non-rate-limit errors

All tests use mocked API calls and fake exception classes so that no real
SDK clients or network calls are needed.
"""

import pytest
from unittest.mock import MagicMock, patch


# ==============================================
# 1. Anthropic scenario — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_anthropic_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when a rate limit error occurs (Anthropic scenario)."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_response = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    callback = MagicMock()

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.messages.create(model="claude-sonnet-4-6", messages=[], temperature=0.7)

    result = retry_api_call(
        call_fn=_make_call,
        provider="anthropic",
        rate_limit_errors=(FakeRateLimitError,),
        retryable_errors=(),
        on_rate_limit=callback,
    )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 2. OpenAI scenario — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_openai_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when a rate limit error occurs (OpenAI scenario)."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_response = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.chat.completions.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    callback = MagicMock()

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.chat.completions.create(model="gpt-4o", messages=[], temperature=0.7)

    result = retry_api_call(
        call_fn=_make_call,
        provider="openai",
        rate_limit_errors=(FakeRateLimitError,),
        retryable_errors=(),
        on_rate_limit=callback,
    )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 3. Google scenario — on_rate_limit callback fires
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_google_calls_on_rate_limit_callback(mock_sleep):
    """on_rate_limit callback is called when a rate limit error occurs (Google scenario)."""
    from chal.utilities.retry import retry_api_call

    mock_response = MagicMock()
    mock_response.text = "test response"

    class FakeRateLimitError(Exception):
        pass

    call_count = 0
    def _make_call(rotated_client):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FakeRateLimitError("rate limited")
        return mock_response

    callback = MagicMock()

    result = retry_api_call(
        call_fn=_make_call,
        provider="google",
        rate_limit_errors=(FakeRateLimitError,),
        retryable_errors=(),
        on_rate_limit=callback,
    )

    assert callback.call_count == 1
    assert result is mock_response


# ==============================================
# 4. on_rate_limit=None (default) still retries
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_retry_without_on_rate_limit_still_works(mock_sleep):
    """Passing on_rate_limit=None (the default) still retries without crashing."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_response = MagicMock()

    class FakeRateLimitError(Exception):
        pass

    mock_client.messages.create.side_effect = [
        FakeRateLimitError("rate limited"),
        mock_response,
    ]

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.messages.create(model="claude-sonnet-4-6", messages=[], temperature=0.7)

    # on_rate_limit is intentionally omitted (defaults to None)
    result = retry_api_call(
        call_fn=_make_call,
        provider="anthropic",
        rate_limit_errors=(FakeRateLimitError,),
        retryable_errors=(),
    )

    assert result is mock_response
    assert mock_client.messages.create.call_count == 2


# ==============================================
# 5. Callback NOT called for non-rate-limit errors
# ==============================================

@pytest.mark.unit
@patch('chal.utilities.retry.time.sleep')
def test_on_rate_limit_not_called_on_non_rate_limit_error(mock_sleep):
    """on_rate_limit callback is NOT invoked for non-rate-limit (retryable) errors."""
    from chal.utilities.retry import retry_api_call

    mock_client = MagicMock()
    mock_response = MagicMock()

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

    def _make_call(rotated_client):
        c = rotated_client if rotated_client is not None else mock_client
        return c.messages.create(model="claude-sonnet-4-6", messages=[], temperature=0.7)

    result = retry_api_call(
        call_fn=_make_call,
        provider="anthropic",
        rate_limit_errors=(FakeRateLimitError,),
        retryable_errors=(FakeAPIConnectionError,),
        on_rate_limit=callback,
    )

    # The callback must NOT have been called for a non-rate-limit error
    assert callback.call_count == 0
    assert result is mock_response
