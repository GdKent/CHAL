"""
Tests for the core retry infrastructure (generate_with_retry).

Covers:
- Success on first attempt (no retries)
- Retry on validation failure (eventually succeeds)
- All retries exhausted (returns last response)
- Correction messages contain error strings
- Remediation hints included in corrections
- Original messages list not mutated
- Temperature forwarded to agent.generate()
- RetryRecord.raw_response_preview truncated to 500 chars
"""

import pytest
from unittest.mock import Mock, call

from chal.agents.base import Message
from chal.utilities.retry import (
    ValidationResult,
    RetryRecord,
    generate_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(responses: list[str]) -> Mock:
    """Create a mock agent that returns successive responses."""
    agent = Mock()
    agent.name = "TestAgent"
    it = iter(responses)

    def _gen(messages, **kwargs):
        return Message(role="assistant", content=next(it))

    agent.generate = Mock(side_effect=_gen)
    return agent


def _always_valid(text: str) -> ValidationResult:
    return ValidationResult(is_valid=True, parsed_data={"ok": True})


def _always_invalid(text: str) -> ValidationResult:
    return ValidationResult(is_valid=False, errors=["bad format", "missing field"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateWithRetry:

    def test_passes_on_first_attempt(self):
        """Validator returns valid on first call — no retries."""
        agent = _make_agent(["good response"])
        messages = [Message(role="user", content="hello")]

        response, records = generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_always_valid,
            max_retries=3,
        )

        assert response.content == "good response"
        assert len(records) == 0
        assert agent.generate.call_count == 1

    def test_retries_on_validation_failure(self):
        """Validator fails first 2 attempts, passes on 3rd."""
        call_count = 0

        def _validator(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ValidationResult(is_valid=False, errors=[f"error on call {call_count}"])
            return ValidationResult(is_valid=True, parsed_data={"ok": True})

        agent = _make_agent(["bad1", "bad2", "good"])
        messages = [Message(role="user", content="hello")]

        response, records = generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_validator,
            max_retries=3,
        )

        assert response.content == "good"
        assert len(records) == 2
        assert agent.generate.call_count == 3
        # Verify correction messages were appended (working_messages grows)
        # The 3rd call should have 5 messages: original + 2*(assistant + user correction)
        third_call_messages = agent.generate.call_args_list[2][0][0]
        assert len(third_call_messages) == 5

    def test_exhausts_retries(self):
        """Validator always fails, max_retries=2 — total 3 calls."""
        agent = _make_agent(["bad1", "bad2", "bad3"])
        messages = [Message(role="user", content="hello")]
        log_entries = []

        response, records = generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_always_invalid,
            max_retries=2,
            stage_label="TestStage",
            log_fn=lambda msg, lvl: log_entries.append((msg, lvl)),
        )

        assert response.content == "bad3"  # last response returned
        assert len(records) == 3  # attempt 0, 1, 2 all failed
        assert agent.generate.call_count == 3

        # Log should include ERROR level for exhaustion
        error_logs = [e for e in log_entries if e[1] == "ERROR"]
        assert len(error_logs) == 1
        assert "exhausted" in error_logs[0][0].lower()

    def test_correction_message_includes_errors(self):
        """Verify the correction message appended to messages contains error strings."""
        call_count = 0

        def _validator(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ValidationResult(
                    is_valid=False,
                    errors=["missing 'patches' key", "invalid JSON format"],
                )
            return ValidationResult(is_valid=True)

        agent = _make_agent(["bad", "good"])
        messages = [Message(role="user", content="hello")]

        generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_validator,
            max_retries=2,
        )

        # Second call should have correction message
        second_call_messages = agent.generate.call_args_list[1][0][0]
        correction_msg = second_call_messages[-1]  # last message is user correction
        assert correction_msg.role == "user"
        assert "missing 'patches' key" in correction_msg.content
        assert "invalid JSON format" in correction_msg.content

    def test_includes_remediation_hints(self):
        """Verify remediation_hints string appears in the correction message."""
        call_count = 0

        def _validator(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ValidationResult(is_valid=False, errors=["some error"])
            return ValidationResult(is_valid=True)

        agent = _make_agent(["bad", "good"])
        messages = [Message(role="user", content="hello")]
        hints = "Output exactly ONE fenced ```json block."

        generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_validator,
            max_retries=2,
            remediation_hints=hints,
        )

        second_call_messages = agent.generate.call_args_list[1][0][0]
        correction_msg = second_call_messages[-1]
        assert hints in correction_msg.content

    def test_preserves_original_messages(self):
        """Verify the caller's original messages list is not mutated."""
        agent = _make_agent(["bad", "good"])
        messages = [Message(role="user", content="hello")]
        original_len = len(messages)

        call_count = 0

        def _validator(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ValidationResult(is_valid=False, errors=["error"])
            return ValidationResult(is_valid=True)

        generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_validator,
            max_retries=2,
        )

        assert len(messages) == original_len

    def test_passes_temperature(self):
        """Verify temperature is forwarded to agent.generate()."""
        agent = _make_agent(["response"])
        messages = [Message(role="user", content="hello")]

        generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_always_valid,
            temperature=0.42,
        )

        _, kwargs = agent.generate.call_args
        assert kwargs["temperature"] == 0.42

    def test_retry_record_contains_preview(self):
        """Verify RetryRecord.raw_response_preview is truncated to 500 chars."""
        long_content = "X" * 1000
        agent = _make_agent([long_content, "good"])
        messages = [Message(role="user", content="hello")]

        call_count = 0

        def _validator(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ValidationResult(is_valid=False, errors=["too long"])
            return ValidationResult(is_valid=True)

        _, records = generate_with_retry(
            agent=agent,
            messages=messages,
            validator_fn=_validator,
            max_retries=2,
        )

        assert len(records) == 1
        assert len(records[0].raw_response_preview) == 500
        assert records[0].attempt == 0
        assert records[0].errors == ["too long"]
        assert records[0].timestamp > 0
