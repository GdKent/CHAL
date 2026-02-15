"""
Unit tests for Adjudicator class.

All tests use mocked agents - no API calls are made.

Tests cover:
- Adjudicator initialization
- Resolution outcomes (rebuttal_valid, critique_valid, unresolved)
- Response parsing (JSON and fallback)
- Error handling
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from chal.orchestrator.adjudicator import Adjudicator
from chal.agents.base import Message
from tests.utils import create_mock_agent


# ==============================================
# Test Fixtures
# ==============================================

@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_adjudications(fixtures_dir):
    """Load test adjudication outcomes from fixtures."""
    with open(fixtures_dir / "test_adjudications.json") as f:
        return json.load(f)


@pytest.fixture
def mock_adjudicator_agent():
    """Create a mock agent for adjudication."""
    agent = create_mock_agent("Adjudicator")
    return agent


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_adjudicator_init(mock_adjudicator_agent):
    """Test initialization with agent and weights."""
    adjudicator = Adjudicator(
        adjudicator_agent=mock_adjudicator_agent,
        logic_weight=0.8,
        ethics_weight=0.2
    )

    assert adjudicator.agent == mock_adjudicator_agent
    assert adjudicator.logic_weight == 0.8
    assert adjudicator.ethics_weight == 0.2


@pytest.mark.unit
def test_adjudicator_default_weights(mock_adjudicator_agent):
    """Test default weights: logic_weight=1.0, ethics_weight=0.0."""
    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    assert adjudicator.logic_weight == 1.0
    assert adjudicator.ethics_weight == 0.0


# ==============================================
# 2. Resolution Outcomes Tests (Mocked)
# ==============================================

@pytest.mark.unit
def test_run_rebuttal_valid(mock_adjudicator_agent, test_adjudications):
    """Test adjudication returning 'rebuttal_valid' status."""
    # Setup mock to return rebuttal_valid outcome
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    challenge = "Your argument relies on compatibilism, but doesn't this just redefine free will?"
    rebuttal = "This mischaracterizes compatibilism. We're analyzing what freedom actually means."

    result = adjudicator.run(challenge, rebuttal, challenger="Agent-A", target="Agent-B")

    assert result["status"] == "rebuttal_valid"
    assert "restatement" in result
    assert "reasoning" in result


@pytest.mark.unit
def test_run_critique_valid(mock_adjudicator_agent, test_adjudications):
    """Test adjudication returning 'critique_valid' status."""
    # Setup mock to return critique_valid outcome
    adjudication = test_adjudications["critique_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    challenge = "Libet experiments show decisions occur before conscious awareness."
    rebuttal = "The conscious mind has veto power."

    result = adjudicator.run(challenge, rebuttal, challenger="Agent-A", target="Agent-B")

    assert result["status"] == "critique_valid"
    assert "reasoning" in result


@pytest.mark.unit
def test_run_unresolved(mock_adjudicator_agent, test_adjudications):
    """Test adjudication returning 'unresolved' status."""
    # Setup mock to return unresolved outcome
    adjudication = test_adjudications["unresolved"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    challenge = "Moral responsibility requires genuine alternatives."
    rebuttal = "Moral responsibility requires action from one's character."

    result = adjudicator.run(challenge, rebuttal, challenger="Agent-A", target="Agent-B")

    assert result["status"] == "unresolved"
    assert "reasoning" in result


# ==============================================
# 3. Response Parsing Tests
# ==============================================

@pytest.mark.unit
def test_run_parses_json_response(mock_adjudicator_agent, test_adjudications):
    """Test extraction of structured JSON from response."""
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"""Here is my adjudication:

```json
{json.dumps(adjudication)}
```

The rebuttal successfully addresses the critique."""
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert result["status"] == "rebuttal_valid"
    assert isinstance(result, dict)


@pytest.mark.unit
def test_run_parses_fallback_format(mock_adjudicator_agent):
    """Test fallback parsing when JSON is not in code block."""
    # Response without code block but with JSON structure
    mock_response = Message(
        role="assistant",
        content="""
Outcome: rebuttal_valid

Restatement: The disagreement concerns the nature of compatibilism.

Reasoning: The target's rebuttal successfully demonstrates that compatibilism
is analyzing what freedom means rather than arbitrarily redefining it.
"""
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    # Should extract outcome even from unstructured format
    assert "status" in result


@pytest.mark.unit
def test_run_includes_restatement(mock_adjudicator_agent, test_adjudications):
    """Test that restatement is extracted."""
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert "restatement" in result
    assert len(result["restatement"]) > 0


@pytest.mark.unit
def test_run_includes_formalizations(mock_adjudicator_agent, test_adjudications):
    """Test that challenger and target formalizations are extracted."""
    adjudication = test_adjudications["critique_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert "formalizations" in result
    assert "challenger" in result["formalizations"]
    assert "target" in result["formalizations"]


# ==============================================
# 4. Error Handling Tests
# ==============================================

@pytest.mark.unit
def test_run_malformed_json(mock_adjudicator_agent, test_adjudications):
    """Test handling of unparseable JSON."""
    # Response with broken JSON
    mock_response = Message(
        role="assistant",
        content="```json\n{\"outcome\": \"rebuttal_valid\", broken\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    # Should either handle gracefully or use fallback parsing
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert isinstance(result, dict)


@pytest.mark.unit
def test_run_missing_outcome(mock_adjudicator_agent):
    """Test that missing outcome returns 'unknown' or error."""
    # Response without outcome field
    mock_response = Message(
        role="assistant",
        content="""```json
{
  "restatement": "Test disagreement",
  "reasoning": "Test reasoning"
}
```"""
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    # Should have outcome field, possibly "unknown"
    assert "status" in result


@pytest.mark.unit
def test_run_empty_response(mock_adjudicator_agent):
    """Test handling of empty response from agent."""
    mock_response = Message(role="assistant", content="")
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    # Should handle gracefully
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert isinstance(result, dict)


@pytest.mark.unit
def test_run_with_agent_error(mock_adjudicator_agent):
    """Test handling when agent raises exception."""
    mock_adjudicator_agent.generate = Mock(side_effect=Exception("API Error"))

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    # Should either propagate or handle gracefully
    with pytest.raises(Exception):
        adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")


# ==============================================
# 5. Logic/Ethics Weight Tests
# ==============================================

@pytest.mark.unit
def test_adjudicator_logic_only(mock_adjudicator_agent):
    """Test adjudicator with logic_weight=1.0, ethics_weight=0.0."""
    adjudicator = Adjudicator(
        adjudicator_agent=mock_adjudicator_agent,
        logic_weight=1.0,
        ethics_weight=0.0
    )

    assert adjudicator.logic_weight == 1.0
    assert adjudicator.ethics_weight == 0.0


@pytest.mark.unit
def test_adjudicator_ethics_only(mock_adjudicator_agent):
    """Test adjudicator with logic_weight=0.0, ethics_weight=1.0."""
    adjudicator = Adjudicator(
        adjudicator_agent=mock_adjudicator_agent,
        logic_weight=0.0,
        ethics_weight=1.0
    )

    assert adjudicator.logic_weight == 0.0
    assert adjudicator.ethics_weight == 1.0


@pytest.mark.unit
def test_adjudicator_balanced_weights(mock_adjudicator_agent):
    """Test adjudicator with balanced weights."""
    adjudicator = Adjudicator(
        adjudicator_agent=mock_adjudicator_agent,
        logic_weight=0.5,
        ethics_weight=0.5
    )

    assert adjudicator.logic_weight == 0.5
    assert adjudicator.ethics_weight == 0.5


# ==============================================
# 6. Integration with Prompts Tests
# ==============================================

@pytest.mark.unit
def test_adjudicator_uses_custom_prompt(mock_adjudicator_agent, test_adjudications):
    """Test that adjudicator can use custom system prompt."""
    custom_prompt = "Use strict logical analysis only"
    mock_adjudicator_agent.system_prompt = custom_prompt

    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert result["status"] == "rebuttal_valid"


@pytest.mark.unit
def test_adjudicator_generates_prompt_correctly(mock_adjudicator_agent):
    """Test that adjudicator constructs proper prompt with challenge and rebuttal."""
    mock_response = Message(
        role="assistant",
        content='```json\n{"outcome": "unresolved", "reasoning": "Test"}\n```'
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    challenge = "Test challenge question?"
    rebuttal = "Test rebuttal response."

    adjudicator.run(challenge, rebuttal, challenger="Agent-A", target="Agent-B")

    # Verify generate was called
    assert mock_adjudicator_agent.generate.called
    call_args = mock_adjudicator_agent.generate.call_args
    messages = call_args[0][0]

    # Should include challenge and rebuttal in messages
    assert any(challenge in str(msg) for msg in messages)
    assert any(rebuttal in str(msg) for msg in messages)
