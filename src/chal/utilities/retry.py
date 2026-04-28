"""
retry.py

Core retry infrastructure for structured LLM output validation.

Provides `generate_with_retry`, a wrapper around Agent.generate() that
validates the response using a caller-supplied validator function and
retries with targeted correction hints on failure.

Also provides `retry_api_call`, a generic retry wrapper for provider API
calls with optional key rotation and exponential backoff.

Each stage in the debate pipeline supplies its own validator
(see validators.py); this module provides the shared retry loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chal.agents.base import Agent, Message
from chal.log import logger


def retry_api_call(
    call_fn,
    provider: str,
    rate_limit_errors: tuple = (),
    retryable_errors: tuple = (),
    key_pool=None,
    current_key: str = "",
    rebuild_client_fn=None,
    max_retries: int = 5,
    base_delay: float = 60.0,
    on_rate_limit=None,
):
    """Generic retry wrapper for provider API calls with key rotation.

    Args:
        call_fn: Callable that takes a client (or None) and returns the API response.
                 Signature: call_fn(client) -> response, where client may be None
                 if no key rotation is needed.
        provider: Provider name for logging (e.g., "openai", "anthropic").
        rate_limit_errors: Exception types that indicate rate limiting.
        retryable_errors: Exception types that should trigger exponential backoff.
        key_pool: Optional KeyPool instance for key rotation.
        current_key: Current API key being used.
        rebuild_client_fn: Callable that takes a new API key and returns a new client.
                          Required if key_pool is provided.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        on_rate_limit: Optional callback fired when a rate limit is hit.

    Returns:
        The API response from call_fn.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    client = None  # Will be set by rebuild_client_fn if key rotation occurs

    for attempt in range(max_retries):
        try:
            return call_fn(client)
        except rate_limit_errors as e:
            if on_rate_limit:
                on_rate_limit()
            if key_pool is not None and rebuild_client_fn is not None:
                key_pool.mark_rate_limited(provider, current_key, cooldown_seconds=60)
                current_key = key_pool.get_key(provider)
                client = rebuild_client_fn(current_key)
                logger.warning(f"[KeyPool] Rotated {provider} API key after rate limit (attempt {attempt+1}/{max_retries}).")
                continue
            # No key pool — fall through to backoff
            wait = base_delay * (2 ** attempt)
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {provider} rate limit: {e}. Retrying in {wait:.1f}s.")
            time.sleep(wait)
        except retryable_errors as e:
            wait = base_delay * (2 ** attempt)
            logger.warning(f"[Retry {attempt+1}/{max_retries}] {provider} API error: {e}. Retrying in {wait:.1f}s.")
            time.sleep(wait)

    raise RuntimeError(f"Exceeded max retries for {provider} API call.")


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    parsed_data: Any = None


@dataclass
class RetryRecord:
    """Record of a single failed validation attempt."""

    attempt: int
    errors: list[str]
    raw_response_preview: str
    timestamp: float


def generate_with_retry(
    agent: Agent,
    messages: list[Message],
    validator_fn: Callable[[str], ValidationResult],
    max_retries: int = 3,
    stage_label: str = "Unknown",
    log_fn: Callable[[str, str], None] | None = None,
    temperature: float | None = None,
    remediation_hints: str = "",
) -> tuple[Message, list[RetryRecord]]:
    """Wrap an Agent.generate() call with validation and retry.

    On each attempt the response is passed through *validator_fn*.  If
    validation fails, the failed assistant message **and** a user
    correction message (listing the specific errors plus any static
    *remediation_hints*) are appended to the conversation before the
    next attempt.  This gives the model full context of what it did
    wrong.

    Args:
        agent: The LLM agent to call.
        messages: Initial message list (prompt).  Not mutated.
        validator_fn: ``(raw_response_text) -> ValidationResult``.
        max_retries: Maximum retry attempts (default 3).  Total calls =
            1 + max_retries in the worst case (attempt 0 is the initial
            call, attempts 1..N are retries).
        stage_label: Human-readable label for log messages.
        log_fn: Optional ``(message, level) -> None`` callback.
        temperature: Optional temperature forwarded to
            ``agent.generate()``.
        remediation_hints: Stage-specific static hints appended to every
            correction message.

    Returns:
        ``(response, retry_records)`` — the best response obtained (or
        the last one if all attempts failed) together with a list of
        :class:`RetryRecord` for every failed attempt.
    """
    retry_records: list[RetryRecord] = []
    working_messages = list(messages)  # shallow copy — don't mutate caller

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response: Message | None = None

    for attempt in range(max_retries + 1):
        response = agent.generate(working_messages, **kwargs)

        result = validator_fn(response.content)

        if result.is_valid:
            if attempt > 0 and log_fn:
                log_fn(
                    f"[{stage_label}] Retry {attempt} succeeded "
                    f"(resolved after {attempt} {'retry' if attempt == 1 else 'retries'})",
                    "INFO",
                )
            return response, retry_records

        # --- Validation failed ---
        record = RetryRecord(
            attempt=attempt,
            errors=list(result.errors),
            raw_response_preview=response.content[:500],
            timestamp=time.time(),
        )
        retry_records.append(record)

        if log_fn:
            error_summary = "; ".join(result.errors[:3])
            if len(result.errors) > 3:
                error_summary += " ..."
            log_fn(
                f"[{stage_label}] Attempt {attempt} failed validation "
                f"({len(result.errors)} error(s)): {error_summary}",
                "WARN",
            )

        # Append correction messages for the next attempt
        if attempt < max_retries:
            error_list = "\n".join(f"- {e}" for e in result.errors)
            correction = (
                "Your previous response had formatting issues that must be fixed:\n\n"
                f"{error_list}\n\n"
                "Please regenerate your response with these issues corrected."
            )
            if remediation_hints:
                correction += f"\n\n{remediation_hints}"

            working_messages.append(
                Message(role="assistant", content=response.content)
            )
            working_messages.append(Message(role="user", content=correction))

    # All retries exhausted
    if log_fn:
        log_fn(
            f"[{stage_label}] All {max_retries} retries exhausted. "
            "Using last response despite validation failures.",
            "ERROR",
        )

    assert response is not None  # guaranteed by loop executing at least once
    return response, retry_records
