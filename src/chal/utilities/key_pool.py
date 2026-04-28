"""
key_pool.py

Thread-safe API key pool with rate-limit-aware rotation.

Supports multiple API keys per provider.  When a key hits a rate limit,
it enters a cooldown period and the pool rotates to the next available key.
If all keys for a provider are cooling down, get_key() blocks until the
soonest one recovers.

Design pattern: Resource Pool + Circuit Breaker.
"""

from __future__ import annotations

import os
import threading
import time

from chal.constants import PROVIDER_ENV_VARS


class KeyPool:
    """Thread-safe API key pool with rate-limit-aware rotation.

    Supports multiple API keys per provider.  When a key hits a rate
    limit, it enters a cooldown period and the pool rotates to the
    next available key.  If all keys for a provider are cooling down,
    get_key() blocks until the soonest one recovers.

    Usage::

        pool = KeyPool()
        pool.load_from_env()          # reads OPENAI_API_KEY="sk-a,sk-b"
        key = pool.get_key("openai")  # returns "sk-a"

        # On rate limit:
        pool.mark_rate_limited("openai", key, cooldown_seconds=60)
        key = pool.get_key("openai")  # returns "sk-b"
    """

    def __init__(self) -> None:
        self._keys: dict[str, list[str]] = {}          # provider -> [key1, key2, ...]
        self._cooldowns: dict[str, float] = {}          # "provider:idx" -> expiry timestamp
        self._next_index: dict[str, int] = {}            # provider -> round-robin index
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_env(self) -> None:
        """Load keys from environment variables.

        Supports comma-separated lists::

            OPENAI_API_KEY="sk-abc,sk-def,sk-ghi"

        Single keys continue to work unchanged::

            OPENAI_API_KEY="sk-abc"

        Keys are stripped of surrounding whitespace.  Empty tokens
        (from trailing commas, etc.) are silently dropped.
        """
        for provider, env_var in PROVIDER_ENV_VARS.items():
            raw = os.environ.get(env_var, "")
            if not raw:
                continue
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if keys:
                self.register_keys(provider, keys)

    def register_keys(self, provider: str, keys: list[str]) -> None:
        """Register one or more API keys for a provider.

        Overwrites any previously registered keys for this provider.
        """
        with self._lock:
            self._keys[provider] = list(keys)
            self._next_index[provider] = 0
            # Clear any existing cooldowns for this provider
            prefix = f"{provider}:"
            stale = [k for k in self._cooldowns if k.startswith(prefix)]
            for k in stale:
                del self._cooldowns[k]

    # ------------------------------------------------------------------
    # Key retrieval
    # ------------------------------------------------------------------

    def get_key(self, provider: str) -> str:
        """Get the next available (non-cooling-down) key for a provider.

        Uses round-robin rotation.  If all keys are cooling down, sleeps
        until the soonest key recovers, then returns it.

        Returns:
            An API key string.

        Raises:
            ValueError: If no keys are registered for the provider.
        """
        with self._lock:
            keys = self._keys.get(provider)
            if not keys:
                raise ValueError(
                    f"No API keys registered for provider '{provider}'. "
                    f"Set {PROVIDER_ENV_VARS.get(provider, provider.upper() + '_API_KEY')} "
                    f"in your environment or .env file."
                )
            n = len(keys)
            start = self._next_index.get(provider, 0)
            now = time.monotonic()

            # Try each key starting from round-robin position
            for offset in range(n):
                idx = (start + offset) % n
                cooldown_key = f"{provider}:{idx}"
                expiry = self._cooldowns.get(cooldown_key, 0.0)
                if now >= expiry:
                    # This key is available
                    self._next_index[provider] = (idx + 1) % n
                    # Clean up expired cooldown entry
                    self._cooldowns.pop(cooldown_key, None)
                    return keys[idx]

            # All keys are cooling down — find the soonest expiry
            soonest_expiry = min(
                self._cooldowns[f"{provider}:{i}"]
                for i in range(n)
                if f"{provider}:{i}" in self._cooldowns
            )
            wait_time = soonest_expiry - now

        # Sleep outside the lock so other threads aren't blocked
        if wait_time > 0:
            time.sleep(wait_time)

        # Retry after sleeping (recursive, but bounded — at least one key is now available)
        return self.get_key(provider)

    # ------------------------------------------------------------------
    # Rate limit marking
    # ------------------------------------------------------------------

    def mark_rate_limited(
        self, provider: str, key: str, cooldown_seconds: float = 60.0
    ) -> None:
        """Mark a specific key as rate-limited.

        The key will be skipped by get_key() until the cooldown expires.

        Args:
            provider: Provider name (e.g. "openai").
            key: The API key string that was rate-limited.
            cooldown_seconds: How long to cool down (default 60s).

        Raises:
            ValueError: If the key is not found for the provider.
        """
        with self._lock:
            keys = self._keys.get(provider, [])
            try:
                idx = keys.index(key)
            except ValueError:
                raise ValueError(
                    f"Key not found for provider '{provider}'. "
                    f"Cannot mark an unregistered key as rate-limited."
                ) from None
            cooldown_key = f"{provider}:{idx}"
            self._cooldowns[cooldown_key] = time.monotonic() + cooldown_seconds

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def key_count(self, provider: str) -> int:
        """Return the number of keys registered for a provider."""
        with self._lock:
            return len(self._keys.get(provider, []))

    def has_keys(self, provider: str) -> bool:
        """Return True if at least one key is registered for a provider."""
        return self.key_count(provider) > 0

    def registered_providers(self) -> list[str]:
        """Return list of providers that have at least one key registered."""
        with self._lock:
            return [p for p, keys in self._keys.items() if keys]
