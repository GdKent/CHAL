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
from chal.orchestrator.adjudicator import Adjudicator, enforce_verdict
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
    assert "scores" in result
    assert isinstance(result["scores"], dict)


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
    assert "scores" in result


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
    assert "scores" in result


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
    """Test that missing outcome defaults to 'unresolved'."""
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

    assert result["status"] == "unresolved"


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


# ==============================================
# 7. Score Extraction Tests (v3)
# ==============================================

@pytest.mark.unit
def test_run_extracts_scores(mock_adjudicator_agent, test_adjudications):
    """Test that scores are extracted from v3 JSON format."""
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert "scores" in result
    scores = result["scores"]
    assert "challenger_logic" in scores
    assert "defender_logic" in scores
    assert "challenger_combined" in scores
    assert "defender_combined" in scores
    assert scores["challenger_logic"] == 0.4
    assert scores["defender_logic"] == 0.7


@pytest.mark.unit
def test_run_scores_empty_when_missing(mock_adjudicator_agent):
    """Test that scores is empty dict when not in response."""
    mock_response = Message(
        role="assistant",
        content='```json\n{"outcome": "unresolved", "reasoning": "Test"}\n```'
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert "scores" in result
    assert result["scores"] == {}


@pytest.mark.unit
def test_run_with_belief_excerpts(mock_adjudicator_agent, test_adjudications):
    """Test that adjudicator accepts belief excerpt parameters."""
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)

    challenger_excerpt = '{"claims": [{"id": "C1", "statement": "Test"}]}'
    target_excerpt = '{"claims": [{"id": "C2", "statement": "Counter"}]}'

    result = adjudicator.run(
        "Challenge", "Rebuttal",
        challenger="Agent-A", target="Agent-B",
        challenger_belief_excerpt_json=challenger_excerpt,
        target_belief_excerpt_json=target_excerpt
    )

    assert result["status"] == "rebuttal_valid"
    # Verify the excerpts were included in the prompt
    call_args = mock_adjudicator_agent.generate.call_args
    messages = call_args[0][0]
    prompt_content = str(messages[0].content)
    assert "challenger_belief_excerpt" in prompt_content
    assert "target_belief_excerpt" in prompt_content


# ==============================================
# 9. Reasoning Extraction Tests
# ==============================================

@pytest.mark.unit
def test_run_prefers_reasoning_tags_over_json_summary(mock_adjudicator_agent):
    """Full <reasoning> block is used instead of the short JSON summary."""
    full_reasoning = (
        "1. Restatement\n"
        "The challenger argues X. The defender argues Y.\n\n"
        "2. Formalization\n"
        "P1: ...\nP2: ...\nC: ...\n\n"
        "3. Adjudication\n"
        "The defender successfully refutes the challenge."
    )
    mock_response = Message(
        role="assistant",
        content=(
            f"<reasoning>\n{full_reasoning}\n</reasoning>\n\n"
            '```json\n'
            '{"outcome": "rebuttal_valid", "restatement": "Test", '
            '"formalization_challenger": "", "formalization_target": "", '
            '"scores": {}, "reasoning": "Short summary."}\n'
            '```'
        )
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="A", target="B")

    assert result["reasoning"] == full_reasoning
    assert "Short summary" not in result["reasoning"]


@pytest.mark.unit
def test_run_falls_back_to_json_reasoning_without_tags(mock_adjudicator_agent):
    """When no <reasoning> tags are present, fall back to JSON reasoning field."""
    mock_response = Message(
        role="assistant",
        content=(
            '```json\n'
            '{"outcome": "unresolved", "reasoning": "JSON-only reasoning."}\n'
            '```'
        )
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="A", target="B")

    assert result["reasoning"] == "JSON-only reasoning."


@pytest.mark.unit
def test_run_reasoning_tags_used_in_fallback_path(mock_adjudicator_agent):
    """<reasoning> tags are preferred even when JSON parsing fails entirely."""
    full_reasoning = "Detailed analysis of the exchange."
    mock_response = Message(
        role="assistant",
        content=(
            f"<reasoning>\n{full_reasoning}\n</reasoning>\n\n"
            "Outcome: critique_valid\n"
            "Reasoning: Short fallback text."
        )
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="A", target="B")

    assert result["reasoning"] == full_reasoning


# ==============================================
# 10. Verdict Enforcement Tests
# ==============================================

@pytest.mark.unit
def test_enforce_verdict_critique_valid():
    """Scores with clear challenger advantage produce critique_valid."""
    scores = {
        "challenger_logic": 0.8, "challenger_ethics": 0.5,
        "defender_logic": 0.4, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=0.5, ethics_weight=0.5)
    assert result["computed_verdict"] == "critique_valid"
    assert result["gap"] >= 0.15


@pytest.mark.unit
def test_enforce_verdict_rebuttal_valid():
    """Scores with clear defender advantage produce rebuttal_valid."""
    scores = {
        "challenger_logic": 0.3, "challenger_ethics": 0.5,
        "defender_logic": 0.7, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=0.5, ethics_weight=0.5)
    assert result["computed_verdict"] == "rebuttal_valid"
    assert result["gap"] <= -0.15


@pytest.mark.unit
def test_enforce_verdict_unresolved():
    """Small score gap produces unresolved."""
    scores = {
        "challenger_logic": 0.55, "challenger_ethics": 0.5,
        "defender_logic": 0.5, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=0.5, ethics_weight=0.5)
    assert result["computed_verdict"] == "unresolved"
    assert abs(result["gap"]) < 0.15


@pytest.mark.unit
def test_enforce_verdict_edge_at_positive_threshold():
    """Gap exactly at +threshold uses >= so produces critique_valid."""
    # logic_weight=1.0, ethics_weight=0.0
    # challenger_logic=0.65, defender_logic=0.5 => gap=0.15
    scores = {
        "challenger_logic": 0.65, "challenger_ethics": 0.5,
        "defender_logic": 0.5, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=1.0, ethics_weight=0.0, threshold=0.15)
    assert result["computed_verdict"] == "critique_valid"
    assert result["gap"] == pytest.approx(0.15)


@pytest.mark.unit
def test_enforce_verdict_edge_at_negative_threshold():
    """Gap exactly at -threshold uses <= so produces rebuttal_valid."""
    scores = {
        "challenger_logic": 0.5, "challenger_ethics": 0.5,
        "defender_logic": 0.65, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=1.0, ethics_weight=0.0, threshold=0.15)
    assert result["computed_verdict"] == "rebuttal_valid"
    assert result["gap"] == pytest.approx(-0.15)


@pytest.mark.unit
def test_enforce_verdict_edge_balanced_weights_fp_regression():
    """Balanced weights (0.5/0.5) at exact threshold must not lose to FP noise.

    Regression test: with cl=0.65, ce=0.5, dl=0.35, de=0.5 and weights 0.5/0.5,
    the mathematical gap is exactly 0.15 but IEEE 754 arithmetic produces
    0.14999999999999991 which is less than float(0.15). The gap must be rounded
    before comparison to avoid spurious 'unresolved' verdicts.
    """
    scores = {
        "challenger_logic": 0.65, "challenger_ethics": 0.5,
        "defender_logic": 0.35, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=0.5, ethics_weight=0.5, threshold=0.15)
    assert result["computed_verdict"] == "critique_valid"
    assert result["gap"] == pytest.approx(0.15)

    # Negative direction: same gap magnitude should produce rebuttal_valid
    scores_neg = {
        "challenger_logic": 0.35, "challenger_ethics": 0.5,
        "defender_logic": 0.65, "defender_ethics": 0.5,
    }
    result_neg = enforce_verdict(scores_neg, logic_weight=0.5, ethics_weight=0.5, threshold=0.15)
    assert result_neg["computed_verdict"] == "rebuttal_valid"
    assert result_neg["gap"] == pytest.approx(-0.15)


@pytest.mark.unit
def test_enforce_verdict_custom_threshold():
    """Custom threshold of 0.30 makes a 0.20 gap produce unresolved."""
    scores = {
        "challenger_logic": 0.7, "challenger_ethics": 0.5,
        "defender_logic": 0.5, "defender_ethics": 0.5,
    }
    result = enforce_verdict(scores, logic_weight=1.0, ethics_weight=0.0, threshold=0.30)
    assert result["computed_verdict"] == "unresolved"
    assert result["gap"] == pytest.approx(0.2)


@pytest.mark.unit
def test_run_overrides_llm_verdict_when_math_disagrees(mock_adjudicator_agent, test_adjudications):
    """When LLM verdict contradicts the scores, math wins."""
    # The rebuttal_valid fixture has defender_logic=0.7 > challenger_logic=0.4
    # but we'll change the outcome to "critique_valid" to force a disagreement
    adjudication = dict(test_adjudications["rebuttal_valid"])
    adjudication["outcome"] = "critique_valid"  # LLM says critique_valid
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    # Math says rebuttal_valid (defender has higher scores)
    assert result["status"] == "rebuttal_valid"
    assert result["override_occurred"] is True
    assert result["llm_verdict"] == "critique_valid"


@pytest.mark.unit
def test_run_no_override_when_math_agrees(mock_adjudicator_agent, test_adjudications):
    """When LLM verdict matches math, no override occurs."""
    adjudication = test_adjudications["rebuttal_valid"]
    mock_response = Message(
        role="assistant",
        content=f"```json\n{json.dumps(adjudication)}\n```"
    )
    mock_adjudicator_agent.generate = Mock(return_value=mock_response)

    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    result = adjudicator.run("Challenge", "Rebuttal", challenger="Agent-A", target="Agent-B")

    assert result["status"] == "rebuttal_valid"
    assert result["override_occurred"] is False


@pytest.mark.unit
def test_adjudicator_init_accepts_threshold(mock_adjudicator_agent):
    """Adjudicator accepts and stores threshold parameter."""
    adjudicator = Adjudicator(
        adjudicator_agent=mock_adjudicator_agent,
        threshold=0.20,
    )
    assert adjudicator.threshold == 0.20


@pytest.mark.unit
def test_adjudicator_default_threshold(mock_adjudicator_agent):
    """Adjudicator default threshold is 0.15."""
    adjudicator = Adjudicator(adjudicator_agent=mock_adjudicator_agent)
    assert adjudicator.threshold == 0.15
