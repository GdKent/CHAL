"""
Pytest configuration and shared fixtures for CHAL test suite.

This module provides common fixtures, test data, and configuration
used across all test modules.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock


# ========================================
# Test Data Paths
# ========================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ========================================
# Pytest Configuration
# ========================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across modules"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )


# ========================================
# Mock Agent Fixtures
# ========================================

@pytest.fixture
def mock_adjudicator_agent():
    """Create a mock adjudicator agent for testing."""
    from chal.agents.base import Message

    agent = Mock()
    agent.name = "MockAdjudicator"

    def mock_adjudicate(messages):
        return Message(
            role="assistant",
            content='''```json
{
  "restatement": "Test disagreement",
  "formalization_challenger": "P1 → Q",
  "formalization_target": "¬Q",
  "outcome": "rebuttal_valid",
  "reasoning": "Test reasoning"
}
```'''
        )

    agent.generate = Mock(side_effect=mock_adjudicate)

    return agent
