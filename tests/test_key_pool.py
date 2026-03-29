"""
Unit tests for KeyPool — thread-safe API key pool with rate-limit-aware rotation.

Tests cover:
- Loading keys (single, comma-separated, whitespace)
- Round-robin key retrieval
- Cooldown / rate-limit marking
- Blocking when all keys are cooling down
- Thread safety under concurrent access
- Error handling for unknown / empty providers
- Introspection helpers (key_count, has_keys, registered_providers)
"""

import threading
import time
from unittest.mock import patch

import pytest

from chal.utilities.key_pool import KeyPool


# ==============================================
# Loading Tests
# ==============================================

class TestKeyPoolLoading:
    """Tests for key registration and environment loading."""

    def test_register_single_key(self):
        """Single key registered for a provider."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-abc"])
        assert pool.key_count("openai") == 1
        assert pool.get_key("openai") == "sk-abc"

    def test_register_multiple_keys(self):
        """Multiple keys registered for a provider."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b", "sk-c"])
        assert pool.key_count("openai") == 3

    def test_load_from_env_single_key(self):
        """load_from_env with a single key per provider."""
        env = {"OPENAI_API_KEY": "sk-only"}
        pool = KeyPool()
        with patch.dict("os.environ", env, clear=False):
            pool.load_from_env()
        assert pool.get_key("openai") == "sk-only"

    def test_load_from_env_comma_separated(self):
        """load_from_env parses comma-separated keys into a list."""
        env = {"OPENAI_API_KEY": "sk-a,sk-b,sk-c"}
        pool = KeyPool()
        with patch.dict("os.environ", env, clear=False):
            pool.load_from_env()
        assert pool.key_count("openai") == 3

    def test_load_from_env_whitespace_trimmed(self):
        """Keys with surrounding whitespace are trimmed."""
        env = {"OPENAI_API_KEY": " sk-a , sk-b , sk-c "}
        pool = KeyPool()
        with patch.dict("os.environ", env, clear=False):
            pool.load_from_env()
        assert pool.get_key("openai") == "sk-a"  # first key, trimmed

    def test_load_from_env_trailing_comma(self):
        """Trailing commas produce no empty tokens."""
        env = {"OPENAI_API_KEY": "sk-a,sk-b,"}
        pool = KeyPool()
        with patch.dict("os.environ", env, clear=False):
            pool.load_from_env()
        assert pool.key_count("openai") == 2

    def test_load_from_env_empty_var(self):
        """Empty env var registers no keys."""
        env = {"OPENAI_API_KEY": ""}
        pool = KeyPool()
        with patch.dict("os.environ", env, clear=False):
            pool.load_from_env()
        assert pool.key_count("openai") == 0

    def test_load_from_env_missing_var(self):
        """Missing env var registers no keys."""
        pool = KeyPool()
        with patch.dict("os.environ", {}, clear=True):
            pool.load_from_env()
        assert pool.key_count("openai") == 0

    def test_register_overwrites_previous(self):
        """Re-registering keys overwrites the previous set."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-old"])
        pool.register_keys("openai", ["sk-new-a", "sk-new-b"])
        assert pool.key_count("openai") == 2
        assert pool.get_key("openai") == "sk-new-a"

    def test_register_clears_cooldowns(self):
        """Re-registering keys clears any existing cooldowns."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])
        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=100)
        # Re-register — cooldown should be cleared
        pool.register_keys("openai", ["sk-a"])
        # Should return immediately (no cooldown)
        assert pool.get_key("openai") == "sk-a"


# ==============================================
# Round-Robin Retrieval Tests
# ==============================================

class TestKeyPoolRoundRobin:
    """Tests for round-robin key cycling."""

    def test_round_robin_cycles(self):
        """Sequential get_key() calls cycle through keys."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b", "sk-c"])

        keys = [pool.get_key("openai") for _ in range(6)]
        assert keys == ["sk-a", "sk-b", "sk-c", "sk-a", "sk-b", "sk-c"]

    def test_single_key_always_returns_same(self):
        """Single key always returned regardless of call count."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-only"])

        keys = [pool.get_key("openai") for _ in range(5)]
        assert all(k == "sk-only" for k in keys)


# ==============================================
# Cooldown / Rate-Limit Tests
# ==============================================

class TestKeyPoolCooldown:
    """Tests for rate-limit marking and cooldown behavior."""

    def test_skips_cooling_down_key(self):
        """A rate-limited key is skipped; the next key is returned."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])

        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=60)
        assert pool.get_key("openai") == "sk-b"

    def test_cooldown_expiry(self):
        """After cooldown elapses, key becomes available again."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])

        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=0.1)
        time.sleep(0.15)  # Wait for cooldown to expire
        # Both keys should be available; round-robin continues
        key = pool.get_key("openai")
        assert key in ("sk-a", "sk-b")

    def test_blocks_when_all_cooling(self):
        """When all keys are cooling, get_key() blocks until soonest expires."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])

        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=0.3)
        pool.mark_rate_limited("openai", "sk-b", cooldown_seconds=0.1)

        start = time.monotonic()
        key = pool.get_key("openai")
        elapsed = time.monotonic() - start

        assert key == "sk-b"  # sk-b has shorter cooldown
        assert elapsed >= 0.05  # Should have waited (with some tolerance)

    def test_mark_unknown_key_raises(self):
        """Marking a key not registered for the provider raises ValueError."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])

        with pytest.raises(ValueError, match="Key not found"):
            pool.mark_rate_limited("openai", "sk-unknown")

    def test_mark_unknown_provider_raises(self):
        """Marking a key on an unregistered provider raises ValueError."""
        pool = KeyPool()

        with pytest.raises(ValueError, match="Key not found"):
            pool.mark_rate_limited("openai", "sk-a")


# ==============================================
# Error Handling Tests
# ==============================================

class TestKeyPoolErrors:
    """Tests for error conditions."""

    def test_get_key_unknown_provider_raises(self):
        """get_key() for an unregistered provider raises ValueError."""
        pool = KeyPool()

        with pytest.raises(ValueError, match="No API keys registered"):
            pool.get_key("unknown_provider")

    def test_get_key_empty_provider_raises(self):
        """Provider with zero registered keys raises ValueError."""
        pool = KeyPool()
        # Don't register any keys for openai
        with pytest.raises(ValueError, match="No API keys registered"):
            pool.get_key("openai")


# ==============================================
# Introspection Tests
# ==============================================

class TestKeyPoolIntrospection:
    """Tests for key_count, has_keys, registered_providers."""

    def test_key_count_returns_correct_count(self):
        """key_count() returns the number of registered keys."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b", "sk-c"])
        pool.register_keys("anthropic", ["ant-a"])

        assert pool.key_count("openai") == 3
        assert pool.key_count("anthropic") == 1
        assert pool.key_count("google") == 0

    def test_has_keys(self):
        """has_keys() returns True when keys exist, False otherwise."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])

        assert pool.has_keys("openai") is True
        assert pool.has_keys("google") is False

    def test_registered_providers(self):
        """registered_providers() lists all providers with keys."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])
        pool.register_keys("google", ["goog-a"])

        providers = pool.registered_providers()
        assert set(providers) == {"openai", "google"}

    def test_registered_providers_empty(self):
        """registered_providers() returns empty list when none registered."""
        pool = KeyPool()
        assert pool.registered_providers() == []


# ==============================================
# Thread Safety Tests
# ==============================================

class TestKeyPoolThreadSafety:
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_get_key(self):
        """Concurrent get_key() calls from multiple threads don't corrupt state."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b", "sk-c"])

        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    key = pool.get_key("openai")
                    results.append(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 500
        assert all(k in ("sk-a", "sk-b", "sk-c") for k in results)

    def test_concurrent_get_and_mark(self):
        """Concurrent get_key() and mark_rate_limited() don't corrupt state."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b", "sk-c"])

        errors = []

        def getter():
            try:
                for _ in range(30):
                    pool.get_key("openai")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def marker():
            try:
                for _ in range(10):
                    key = pool.get_key("openai")
                    pool.mark_rate_limited("openai", key, cooldown_seconds=0.01)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=getter) for _ in range(5)]
        threads += [threading.Thread(target=marker) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ==============================================
# Multi-Provider Isolation Tests
# ==============================================

class TestKeyPoolProviderIsolation:
    """Tests that provider pools are independent."""

    def test_different_providers_isolated(self):
        """Rate-limiting one provider doesn't affect another."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a"])
        pool.register_keys("anthropic", ["ant-a"])

        pool.mark_rate_limited("openai", "sk-a", cooldown_seconds=60)

        # Anthropic should be unaffected
        assert pool.get_key("anthropic") == "ant-a"

    def test_round_robin_independent_per_provider(self):
        """Round-robin indices are independent per provider."""
        pool = KeyPool()
        pool.register_keys("openai", ["sk-a", "sk-b"])
        pool.register_keys("google", ["goog-a", "goog-b"])

        assert pool.get_key("openai") == "sk-a"
        assert pool.get_key("google") == "goog-a"
        assert pool.get_key("openai") == "sk-b"
        assert pool.get_key("google") == "goog-b"
