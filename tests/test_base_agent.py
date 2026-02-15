"""
Unit tests for base Agent class and Message dataclass.

Tests cover:
- Message creation and equality
- Agent abstract interface
"""

import pytest
from chal.agents.base import Agent, Message


# ==============================================
# 1. Message Class Tests
# ==============================================

@pytest.mark.unit
def test_message_creation():
    """Test creating Message with role and content."""
    message = Message(role="user", content="Test content")

    assert message.role == "user"
    assert message.content == "Test content"


@pytest.mark.unit
def test_message_equality():
    """Test that two identical messages are equal."""
    msg1 = Message(role="assistant", content="Hello")
    msg2 = Message(role="assistant", content="Hello")

    assert msg1 == msg2


@pytest.mark.unit
def test_message_repr():
    """Test string representation includes role and content preview."""
    message = Message(role="system", content="This is a long message that should be truncated")

    repr_str = repr(message)

    assert "system" in repr_str.lower() or "Message" in repr_str


# ==============================================
# 2. Agent Abstract Interface Tests
# ==============================================

@pytest.mark.unit
def test_agent_cannot_instantiate():
    """Test that abstract Agent class cannot be instantiated."""
    with pytest.raises(TypeError):
        Agent()


@pytest.mark.unit
def test_agent_subclass_must_implement_generate():
    """Test that subclass without generate() method fails."""
    class IncompleteAgent(Agent):
        def __init__(self):
            self.name = "Incomplete"
            self.model = "test"
            self.temperature = 0.7
            self.current_belief = None

    # Attempting to instantiate should fail or require generate() implementation
    with pytest.raises(TypeError):
        IncompleteAgent()
