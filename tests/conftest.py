"""
Pytest configuration and shared fixtures for CHAL test suite.

This module provides common fixtures, test data, and configuration
used across all test modules.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock


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
# Sample Belief Fixtures
# ========================================

@pytest.fixture
def sample_minimal_belief() -> Dict[str, Any]:
    """
    Minimal valid CBS belief structure with only required fields.
    """
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-TEST-001",
        "version": 1,
        "metadata": {
            "topic_query": "Does free will exist?",
            "agent_persona": "Empiricist",
            "created_at": "2026-02-15T12:00:00Z"
        },
        "thesis": {
            "stance": "Free will is an illusion created by deterministic processes.",
            "summary_bullets": [
                "Neuroscience shows decisions occur before conscious awareness",
                "Physical determinism governs all events"
            ],
            "strength": 0.75
        }
    }


@pytest.fixture
def sample_complete_belief() -> Dict[str, Any]:
    """
    Complete CBS belief structure with all optional fields populated.
    """
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-TEST-002",
        "version": 1,
        "metadata": {
            "topic_query": "Does free will exist?",
            "agent_persona": "Rationalist",
            "created_at": "2026-02-15T12:00:00Z",
            "last_updated": "2026-02-15T12:30:00Z",
            "scope_conditions": "Discussion limited to compatibilist framework",
            "definitions": [
                {"term": "free will", "definition": "The ability to choose between alternatives without external coercion"}
            ]
        },
        "thesis": {
            "stance": "Free will exists within a compatibilist framework.",
            "summary_bullets": [
                "Determinism and free will are compatible",
                "Humans have the capacity for rational deliberation"
            ],
            "strength": 0.80
        },
        "assumptions": [
            {"id": "A1", "type": "empirical", "statement": "Rational deliberation is a real phenomenon", "strength": 0.9, "status": "active", "strength_justification": "Well-established through cognitive science research"}
        ],
        "claims": [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Humans can make choices based on reasons",
                "depends_on": ["A1", "E1"],
                "strength": 0.85,
                "status": "active",
                "predictions": [
                    {
                        "statement": "People will report feeling in control when making deliberate choices",
                        "test": "Survey participants after deliberate vs. reflexive choices",
                        "decision_criterion": "If >70% report feeling in control during deliberate choices, prediction is confirmed"
                    }
                ],
                "inference_chain": [
                    {"step": "P1: Rational beings deliberate", "justification": "Empirical observation"}
                ],
                "strength_justification": "Strong empirical support"
            }
        ],
        "evidence": [
            {
                "id": "E1",
                "type": "empirical",
                "summary": "Studies show humans consider alternatives before deciding",
                "source": "Libet et al. (1983)",
                "relevance_to_claims": ["C1"],
                "strength": 0.8,
                "status": "active",
                "strength_justification": "Replicated across labs with converging methods"
            }
        ]
    }


@pytest.fixture
def sample_belief_with_cycle() -> Dict[str, Any]:
    """
    Invalid belief with circular dependency (C1 → C2 → C1).
    """
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-INVALID-001",
        "version": 1,
        "metadata": {
            "topic_query": "Test topic",
            "agent_persona": "Test",
            "created_at": "2026-02-15T12:00:00Z"
        },
        "thesis": {
            "stance": "Test stance",
            "summary_bullets": ["Test bullet"],
            "strength": 0.5
        },
        "claims": [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Claim 1 depends on Claim 2",
                "depends_on": ["C2"],
                "strength": 0.7,
                "strength_justification": "Test justification",
                "status": "active",
                "inference_chain": ["Step 1: C2 implies C1"],
                "predictions": [{"statement": "P1", "test": "T1", "decision_criterion": "DC1"}]
            },
            {
                "id": "C2",
                "type": "deductive",
                "statement": "Claim 2 depends on Claim 1",
                "depends_on": ["C1"],
                "strength": 0.7,
                "strength_justification": "Test justification",
                "status": "active",
                "inference_chain": ["Step 1: C1 implies C2"],
                "predictions": [{"statement": "P2", "test": "T2", "decision_criterion": "DC2"}]
            }
        ]
    }


@pytest.fixture
def sample_belief_with_orphan() -> Dict[str, Any]:
    """
    Invalid belief with orphaned claim (no supporting evidence or assumptions).
    """
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-INVALID-002",
        "version": 1,
        "metadata": {
            "topic_query": "Test topic",
            "agent_persona": "Test",
            "created_at": "2026-02-15T12:00:00Z"
        },
        "thesis": {
            "stance": "Test stance",
            "summary_bullets": ["Test bullet"],
            "strength": 0.5
        },
        "claims": [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Orphaned claim with no support",
                "depends_on": [],
                "strength": 0.7,
                "strength_justification": "Test justification",
                "status": "active",
                "inference_chain": ["Step 1: Orphaned reasoning"],
                "predictions": [{"statement": "P1", "test": "T1", "decision_criterion": "DC1"}]
            }
        ]
    }


# ========================================
# Patch Fixtures
# ========================================

@pytest.fixture
def sample_patches_update_thesis() -> List[Dict[str, Any]]:
    """Sample patches for updating thesis strength."""
    return [
        {"op": "update_thesis", "change": "weaken"}
    ]


@pytest.fixture
def sample_patches_update_claim() -> List[Dict[str, Any]]:
    """Sample patches for updating claim properties."""
    return [
        {
            "op": "update_claim",
            "target_id": "C1",
            "changes": {
                "strength": 0.6,
                "status": "revised"
            }
        }
    ]


# ========================================
# Mock Agent Fixtures
# ========================================

@pytest.fixture
def mock_openai_agent():
    """Create a mock OpenAI agent for testing."""
    from chal.agents.base import Message

    agent = Mock()
    agent.name = "MockAgent"
    agent.model = "gpt-4o"
    agent.temperature = 0.7
    agent.current_belief = None

    # Mock generate method
    def mock_generate(messages: List[Message]) -> Message:
        return Message(
            role="assistant",
            content='{"response": "Mock response"}'
        )

    agent.generate = Mock(side_effect=mock_generate)
    agent.update_current_belief = Mock()

    return agent


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


# ========================================
# Configuration Fixtures
# ========================================

@pytest.fixture
def sample_config():
    """Sample debate configuration for testing."""
    from chal.config import DebateConfig, AgentConfig, AdjudicationConfig

    config = DebateConfig(
        name="Test Debate",
        topic="Test topic",
        max_rounds=1,
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST"),
            AgentConfig(name="Agent-B", persona="RATIONALIST")
        ],
        adjudication=AdjudicationConfig(
            logic_weight=1.0,
            ethics_weight=0.0
        )
    )

    return config


# ========================================
# Helper Fixtures
# ========================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory for test outputs."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage
