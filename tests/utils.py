"""
Testing utilities and helper functions for CHAL test suite.

This module provides reusable utilities for creating test data,
mocking LLM responses, and performing custom assertions.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
from chal.agents.base import Agent, Message
import json


# ========================================
# Belief Creation Utilities
# ========================================

def create_sample_belief(
    belief_id: str = "BELIEF-TEST",
    confidence: float = 0.75,
    num_claims: int = 1,
    num_assumptions: int = 1,
    num_evidence: int = 1
) -> Dict[str, Any]:
    """
    Create a sample CBS belief structure for testing.

    Args:
        belief_id: Unique identifier for the belief
        confidence: Thesis confidence (0.0-1.0)
        num_claims: Number of claims to include
        num_assumptions: Number of assumptions to include
        num_evidence: Number of evidence items to include

    Returns:
        Valid CBS belief dictionary
    """
    belief = {
        "schema_version": "CBS",
        "belief_id": belief_id,
        "version": 1,
        "metadata": {
            "topic_query": "Test topic",
            "agent_persona": "Test Persona",
            "created_at": "2026-02-15T12:00:00Z"
        },
        "thesis": {
            "stance": f"Test stance for {belief_id}",
            "summary_bullets": ["Test bullet 1", "Test bullet 2"],
            "confidence": confidence
        }
    }

    # Add assumptions
    if num_assumptions > 0:
        belief["assumptions"] = [
            {
                "id": f"A{i+1}",
                "statement": f"Assumption {i+1}"
            }
            for i in range(num_assumptions)
        ]

    # Add evidence
    if num_evidence > 0:
        belief["evidence"] = [
            {
                "id": f"E{i+1}",
                "type": "empirical",
                "summary": f"Evidence {i+1}",
                "citation": f"Test et al. (2026)"
            }
            for i in range(num_evidence)
        ]

    # Add claims
    if num_claims > 0:
        belief["claims"] = [
            {
                "id": f"C{i+1}",
                "type": "deductive",
                "statement": f"Claim {i+1}",
                "depends_on": ["A1"] if num_assumptions > 0 else [],
                "backing_evidence_ids": ["E1"] if num_evidence > 0 else [],
                "confidence": confidence - 0.1,
                "confidence_justification": "Test justification",
                "status": "accepted",
                "known_weaknesses": [],
                "known_rebuttals": []
            }
            for i in range(num_claims)
        ]

    return belief


def create_invalid_belief(error_type: str) -> Dict[str, Any]:
    """
    Create an invalid belief for testing validation.

    Args:
        error_type: Type of validation error to create
            - "missing_schema_version": Missing schema_version field
            - "wrong_schema_version": Incorrect schema version
            - "missing_belief_id": Missing belief_id field
            - "invalid_version": Version is 0 or negative
            - "missing_metadata": Missing metadata field
            - "missing_thesis": Missing thesis field
            - "confidence_out_of_bounds": Thesis confidence > 1.0 or < 0.0
            - "empty_bullets": Empty summary_bullets array
            - "circular_dependency": Claims with circular dependency
            - "orphaned_claim": Claim with no support

    Returns:
        Invalid belief dictionary
    """
    base_belief = create_sample_belief()

    if error_type == "missing_schema_version":
        del base_belief["schema_version"]
    elif error_type == "wrong_schema_version":
        base_belief["schema_version"] = "CBS-v0"
    elif error_type == "missing_belief_id":
        del base_belief["belief_id"]
    elif error_type == "invalid_version":
        base_belief["version"] = 0
    elif error_type == "missing_metadata":
        del base_belief["metadata"]
    elif error_type == "missing_thesis":
        del base_belief["thesis"]
    elif error_type == "confidence_out_of_bounds":
        base_belief["thesis"]["confidence"] = 1.5
    elif error_type == "empty_bullets":
        base_belief["thesis"]["summary_bullets"] = []
    elif error_type == "circular_dependency":
        base_belief["claims"] = [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Claim 1",
                "depends_on": ["C2"],
                "backing_evidence_ids": [],
                "confidence": 0.7,
                "status": "accepted"
            },
            {
                "id": "C2",
                "type": "deductive",
                "statement": "Claim 2",
                "depends_on": ["C1"],
                "backing_evidence_ids": [],
                "confidence": 0.7,
                "status": "accepted"
            }
        ]
    elif error_type == "orphaned_claim":
        base_belief["claims"] = [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Orphaned claim",
                "depends_on": [],
                "backing_evidence_ids": [],
                "confidence": 0.7,
                "status": "accepted"
            }
        ]
        base_belief["assumptions"] = []
        base_belief["evidence"] = []

    return base_belief


# ========================================
# Mock Agent Creation
# ========================================

def create_mock_agent(
    name: str = "MockAgent",
    model: str = "gpt-4o",
    responses: Optional[List[str]] = None
) -> Mock:
    """
    Create a mock agent with configurable responses.

    Args:
        name: Agent name
        model: Model identifier
        responses: List of responses to return (cycles through them)

    Returns:
        Mock agent instance
    """
    agent = MagicMock()
    agent.name = name
    agent.model = model
    agent.temperature = 0.7
    agent.current_belief = None
    agent.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
    agent.system_prompt = ""
    agent.internal_belief = ""
    agent.all_beliefs_held = []
    agent.get_internal_belief_obj.return_value = None
    agent.get_internal_belief.return_value = ""

    if responses is None:
        responses = ['{"test": "response"}']

    response_cycle = iter(responses * 100)  # Repeat responses

    def mock_generate(messages: List[Message], **kwargs) -> Message:
        return Message(role="assistant", content=next(response_cycle))

    agent.generate = Mock(side_effect=mock_generate)
    agent.update_current_belief = Mock()

    return agent


def create_mock_belief_response(belief: Dict[str, Any]) -> str:
    """
    Create a mock LLM response containing a CBS belief.

    Args:
        belief: Belief dictionary to encode

    Returns:
        Formatted response string with JSON code block
    """
    return f"""```json
{json.dumps(belief, indent=2)}
```

Here is my belief structure in CBS format.
"""


def create_mock_challenge_response(num_challenges: int = 3) -> str:
    """
    Create a mock LLM response with cross-examination questions.

    Args:
        num_challenges: Number of questions to generate

    Returns:
        Formatted response string with numbered questions
    """
    challenges = [
        f"{i+1}. Test challenge {i+1}?" for i in range(num_challenges)
    ]
    return "\n".join(challenges)


def create_mock_rebuttal_response(num_rebuttals: int = 3) -> str:
    """
    Create a mock LLM response with structured rebuttals.

    Args:
        num_rebuttals: Number of rebuttals to generate

    Returns:
        Formatted response string with Critique/Response pairs
    """
    pairs = []
    for i in range(num_rebuttals):
        pairs.append(f"Critique {i+1}:\nTest critique {i+1}\n\nResponse {i+1}:\nTest rebuttal {i+1}")
    return "\n\n".join(pairs)


def create_mock_adjudication_response(outcome: str = "rebuttal_valid") -> str:
    """
    Create a mock adjudication response.

    Args:
        outcome: One of "rebuttal_valid", "critique_valid", "unresolved"

    Returns:
        Formatted JSON response
    """
    return f"""```json
{{
  "restatement": "Test disagreement restatement",
  "formalization_challenger": "P1 → Q",
  "formalization_target": "¬P1",
  "outcome": "{outcome}",
  "reasoning": "Test reasoning for {outcome}"
}}
```"""


# ========================================
# Custom Assertions
# ========================================

def assert_belief_valid(belief: Dict[str, Any]) -> None:
    """
    Assert that a belief structure is valid according to CBS.

    Args:
        belief: Belief dictionary to validate

    Raises:
        AssertionError: If belief is invalid
    """
    from chal.beliefs.schema import validate_belief

    errors = validate_belief(belief)
    assert len(errors) == 0, f"Belief validation failed: {errors}"


def assert_belief_has_errors(belief: Dict[str, Any], expected_errors: Optional[List[str]] = None) -> None:
    """
    Assert that a belief structure has validation errors.

    Args:
        belief: Belief dictionary to validate
        expected_errors: Optional list of expected error substrings

    Raises:
        AssertionError: If belief is valid or doesn't have expected errors
    """
    from chal.beliefs.schema import validate_belief

    errors = validate_belief(belief)
    assert len(errors) > 0, "Expected validation errors but belief was valid"

    if expected_errors:
        for expected in expected_errors:
            assert any(expected in error for error in errors), \
                f"Expected error containing '{expected}' but got: {errors}"


def assert_graph_acyclic(belief: Dict[str, Any]) -> None:
    """
    Assert that a belief graph has no cycles.

    Args:
        belief: Belief dictionary to check

    Raises:
        AssertionError: If graph has cycles
    """
    from chal.beliefs.belief_graph import BeliefGraph

    graph = BeliefGraph(belief)
    assert not graph._has_cycle(), "Expected acyclic graph but found cycles"


def assert_graph_has_cycle(belief: Dict[str, Any]) -> None:
    """
    Assert that a belief graph contains at least one cycle.

    Args:
        belief: Belief dictionary to check

    Raises:
        AssertionError: If graph is acyclic
    """
    from chal.beliefs.belief_graph import BeliefGraph

    graph = BeliefGraph(belief)
    assert graph._has_cycle(), "Expected cycles but graph is acyclic"


def assert_no_orphaned_claims(belief: Dict[str, Any]) -> None:
    """
    Assert that a belief has no orphaned claims.

    Args:
        belief: Belief dictionary to check

    Raises:
        AssertionError: If orphaned claims exist
    """
    from chal.beliefs.belief_graph import BeliefGraph

    graph = BeliefGraph(belief)
    orphans = graph._find_orphaned_claims()
    assert len(orphans) == 0, f"Expected no orphaned claims but found: {orphans}"


def assert_has_orphaned_claims(belief: Dict[str, Any], expected_count: Optional[int] = None) -> None:
    """
    Assert that a belief has orphaned claims.

    Args:
        belief: Belief dictionary to check
        expected_count: Optional expected number of orphaned claims

    Raises:
        AssertionError: If no orphaned claims or count doesn't match
    """
    from chal.beliefs.belief_graph import BeliefGraph

    graph = BeliefGraph(belief)
    orphans = graph._find_orphaned_claims()
    assert len(orphans) > 0, "Expected orphaned claims but found none"

    if expected_count is not None:
        assert len(orphans) == expected_count, \
            f"Expected {expected_count} orphaned claims but found {len(orphans)}"


# ========================================
# Comparison Utilities
# ========================================

def beliefs_structurally_equal(belief1: Dict[str, Any], belief2: Dict[str, Any], ignore_fields: Optional[List[str]] = None) -> bool:
    """
    Compare two beliefs for structural equality, optionally ignoring certain fields.

    Args:
        belief1: First belief
        belief2: Second belief
        ignore_fields: List of top-level fields to ignore (e.g., ["version", "metadata"])

    Returns:
        True if beliefs are structurally equal (ignoring specified fields)
    """
    import copy

    b1 = copy.deepcopy(belief1)
    b2 = copy.deepcopy(belief2)

    if ignore_fields:
        for field in ignore_fields:
            b1.pop(field, None)
            b2.pop(field, None)

    return b1 == b2
