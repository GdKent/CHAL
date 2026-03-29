"""
Integration tests for API key rotation under rate limits.

Tests cover:
- OpenAI retry wrapper rotates key on RateLimitError
- All keys exhausted → waits for cooldown → succeeds
- Mixed providers: rate-limiting one provider doesn't affect another
- Retry wrapper without key pool behaves identically to pre-pool behavior
"""

import time
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from chal.utilities.key_pool import KeyPool


# ==============================================
# OpenAI Key Rotation Tests
# ==============================================

class TestOpenAIKeyRotation:
    """Tests for key rotation in the OpenAI retry wrapper."""

    def test_retry_rotates_key_on_rate_limit(self):
        """On RateLimitError, the wrapper rotates to the next key and retries."""
        import openai
        from chal.agents.openai_agent import retry_openai_chat_completion

        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "success"

        # First call: RateLimitError, second call: success
        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = [
            rate_limit_err,
            mock_response,
        ]

        # Patch OpenAI constructor to return our mock
        with patch("chal.agents.openai_agent.OpenAI") as MockOpenAI:
            new_client = MagicMock()
            new_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = new_client

            result = retry_openai_chat_completion(
                client=mock_client,
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_retries=3,
                base_delay=0.01,
                key_pool=pool,
                current_key="sk-a",
            )

        assert result == mock_response
        # Original client was called once (rate limited)
        assert mock_client.chat.completions.create.call_count == 1
        # New client was called once (success)
        assert new_client.chat.completions.create.call_count == 1

    def test_retry_without_pool_uses_backoff(self):
        """Without a key pool, RateLimitError triggers exponential backoff."""
        import openai
        from chal.agents.openai_agent import retry_openai_chat_completion

        mock_client = MagicMock()
        mock_response = MagicMock()

        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = [
            rate_limit_err,
            mock_response,
        ]

        with patch("time.sleep") as mock_sleep:
            result = retry_openai_chat_completion(
                client=mock_client,
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_retries=3,
                base_delay=0.01,
                key_pool=None,
                current_key="",
            )

        assert result == mock_response
        # Should have slept (backoff)
        mock_sleep.assert_called_once()

    def test_retry_max_retries_exhausted(self):
        """After max retries, RuntimeError is raised."""
        import openai
        from chal.agents.openai_agent import retry_openai_chat_completion

        mock_client = MagicMock()
        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = rate_limit_err

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Exceeded max retries"):
                retry_openai_chat_completion(
                    client=mock_client,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.7,
                    max_retries=2,
                    base_delay=0.01,
                    key_pool=None,
                    current_key="",
                )


# ==============================================
# Anthropic Key Rotation Tests
# ==============================================

class TestAnthropicKeyRotation:
    """Tests for key rotation in the Anthropic retry wrapper."""

    def test_retry_rotates_key_on_rate_limit(self):
        """On RateLimitError, the wrapper rotates to the next key."""
        import anthropic
        from chal.agents.anthropic_agent import retry_anthropic_message

        pool = KeyPool()
        pool.register_keys("anthropic", ["ant-a", "ant-b"])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="success")]
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20
        )

        rate_limit_err = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.messages.create.side_effect = [
            rate_limit_err,
            mock_response,
        ]

        with patch("chal.agents.anthropic_agent.anthropic.Anthropic") as MockAnthropicCls:
            new_client = MagicMock()
            new_client.messages.create.return_value = mock_response
            MockAnthropicCls.return_value = new_client

            result = retry_anthropic_message(
                client=mock_client,
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "test"}],
                system_prompt="test",
                temperature=0.7,
                max_retries=3,
                base_delay=0.01,
                key_pool=pool,
                current_key="ant-a",
            )

        assert result == mock_response


# ==============================================
# Cross-Provider Isolation Tests
# ==============================================

class TestCrossProviderIsolation:
    """Tests that rate-limiting one provider doesn't affect another."""

    def test_mixed_providers_no_cross_contamination(self):
        """Rate-limiting an OpenAI key doesn't affect the Anthropic pool."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])
        pool.register_keys("anthropic", ["ant-a", "ant-b"])

        # Rate-limit all OpenAI keys
        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=60)
        pool.mark_rate_limited("openai", "sk-b", cooldown_seconds=60)

        # Anthropic should be completely unaffected
        key = pool.get_key("anthropic")
        assert key in ("ant-a", "ant-b")

    def test_key_pool_tracks_cooldowns_independently(self):
        """Each provider's cooldown state is fully independent."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])
        pool.register_keys("google", ["goog-a"])

        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=0.1)

        # Google is immediately available
        assert pool.get_key("google") == "goog-a"

        # Wait for OpenAI cooldown
        time.sleep(0.15)
        assert pool.get_key("openai") == "sk-a"


# ==============================================
# All Keys Exhausted Tests
# ==============================================

class TestAllKeysExhausted:
    """Tests for behavior when all keys are rate-limited."""

    def test_all_keys_exhausted_eventually_succeeds(self):
        """When all keys are cooling, get_key() blocks until soonest expires."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])

        # Rate limit both with short cooldowns
        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=0.2)
        pool.mark_rate_limited("openai", "sk-b", cooldown_seconds=0.1)

        start = time.monotonic()
        key = pool.get_key("openai")  # Should block ~0.1s
        elapsed = time.monotonic() - start

        assert key == "sk-b"
        assert elapsed >= 0.05  # Should have waited
        assert elapsed < 0.5   # But not too long
