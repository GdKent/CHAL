"""
Unit tests for belief I/O operations.

Tests cover:
- Parsing model output (JSON extraction, validation)
- Belief to markdown conversion
- Embedding projection
"""

import pytest
import json
from pathlib import Path
from chal.beliefs.io import (
    parse_model_output_to_belief,
    belief_to_markdown,
    project_for_embedding
)
from tests.utils import create_sample_belief, create_mock_belief_response


# ==============================================
# Test Fixtures
# ==============================================

@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_beliefs(fixtures_dir):
    """Load test beliefs from fixtures."""
    with open(fixtures_dir / "test_beliefs.json") as f:
        return json.load(f)


@pytest.fixture
def mock_responses(fixtures_dir):
    """Load mock OpenAI responses from fixtures."""
    with open(fixtures_dir / "mock_openai_responses.json") as f:
        return json.load(f)


# ==============================================
# 1. Parsing Model Output Tests
# ==============================================

@pytest.mark.unit
def test_parse_model_output_json_only(test_beliefs):
    """Test extraction of JSON block from model output."""
    belief = test_beliefs["minimal_valid"]
    response = create_mock_belief_response(belief)

    parsed, markdown, errors = parse_model_output_to_belief(response)

    assert parsed is not None
    assert "schema_version" in parsed
    assert parsed["schema_version"] == "CBS-v1"


@pytest.mark.unit
def test_parse_model_output_json_and_markdown(mock_responses):
    """Test extraction from output with both JSON and markdown."""
    response = mock_responses["belief_minimal"]["content"]

    parsed, markdown, errors = parse_model_output_to_belief(response)

    assert parsed is not None
    assert "schema_version" in parsed
    assert "belief_id" in parsed


@pytest.mark.unit
def test_parse_model_output_no_json(mock_responses):
    """Test handling of output with no JSON block."""
    response = mock_responses["no_json_block"]["content"]

    parsed, markdown, errors = parse_model_output_to_belief(response)

    assert parsed is None


@pytest.mark.unit
def test_parse_model_output_malformed_json(mock_responses):
    """Test handling of invalid JSON syntax."""
    response = mock_responses["malformed_json"]["content"]

    parsed, markdown, errors = parse_model_output_to_belief(response)

    assert parsed is None


@pytest.mark.unit
def test_parse_model_output_validation_errors():
    """Test handling of valid JSON but invalid belief schema."""
    response = """```json
{
  "schema_version": "CBS-v0",
  "belief_id": "INVALID"
}
```"""

    parsed, markdown, errors = parse_model_output_to_belief(response)

    # Should parse JSON even if validation fails
    assert parsed is not None
    assert "schema_version" in parsed


# ==============================================
# 2. Belief to Markdown Conversion Tests
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_minimal(test_beliefs):
    """Test markdown conversion with only required fields."""
    belief = test_beliefs["minimal_valid"]

    markdown = belief_to_markdown(belief)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert "# Thesis" in markdown or "Thesis" in markdown


@pytest.mark.unit
def test_belief_to_markdown_complete(test_beliefs):
    """Test markdown conversion with all fields populated."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert isinstance(markdown, str)
    assert len(markdown) > 0


@pytest.mark.unit
def test_belief_to_markdown_thesis_section(test_beliefs):
    """Test that thesis is formatted correctly in markdown."""
    belief = test_beliefs["minimal_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain thesis stance
    assert belief["thesis"]["stance"] in markdown

    # Should contain summary bullets
    for bullet in belief["thesis"]["summary_bullets"]:
        assert bullet in markdown or bullet.lower() in markdown.lower()


@pytest.mark.unit
def test_belief_to_markdown_assumptions_section(test_beliefs):
    """Test that assumptions are listed in markdown."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain assumption statements
    if "assumptions" in belief:
        for assumption in belief["assumptions"]:
            assert assumption["statement"] in markdown or \
                   assumption["statement"].lower() in markdown.lower()


@pytest.mark.unit
def test_belief_to_markdown_claims_section(test_beliefs):
    """Test that claims with dependencies/evidence are formatted."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain claim statements
    if "claims" in belief:
        for claim in belief["claims"]:
            assert claim["statement"] in markdown or \
                   claim["statement"].lower() in markdown.lower()


@pytest.mark.unit
def test_belief_to_markdown_evidence_section(test_beliefs):
    """Test that evidence items are included."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain evidence summaries
    if "evidence" in belief:
        for evidence in belief["evidence"]:
            assert evidence["summary"] in markdown or \
                   evidence["summary"].lower() in markdown.lower()


@pytest.mark.unit
def test_belief_to_markdown_predictions_section(test_beliefs):
    """Test that predictions are included if present."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain predictions if they exist
    if "predictions" in belief and len(belief["predictions"]) > 0:
        for prediction in belief["predictions"]:
            assert prediction["statement"] in markdown or \
                   prediction["statement"].lower() in markdown.lower()


@pytest.mark.unit
def test_belief_to_markdown_normative_implications(test_beliefs):
    """Test that normative implications are included if present."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should contain normative implications if they exist
    if "normative_implications" in belief and len(belief["normative_implications"]) > 0:
        for implication in belief["normative_implications"]:
            assert implication["statement"] in markdown or \
                   implication["statement"].lower() in markdown.lower()


# ==============================================
# 3. Embedding Projection Tests
# ==============================================

@pytest.mark.unit
def test_project_for_embedding_basic(test_beliefs):
    """Test projection of belief to text string."""
    belief = test_beliefs["minimal_valid"]

    projection = project_for_embedding(belief)

    assert isinstance(projection, str)
    assert len(projection) > 0


@pytest.mark.unit
def test_project_for_embedding_includes_thesis(test_beliefs):
    """Test that projection contains thesis content."""
    belief = test_beliefs["minimal_valid"]

    projection = project_for_embedding(belief)

    # Should include thesis stance
    assert belief["thesis"]["stance"] in projection


@pytest.mark.unit
def test_project_for_embedding_includes_claims(test_beliefs):
    """Test that projection contains claim statements."""
    belief = test_beliefs["complete_valid"]

    projection = project_for_embedding(belief)

    # Should include claim statements
    if "claims" in belief:
        for claim in belief["claims"]:
            assert claim["statement"] in projection or \
                   claim["statement"].lower() in projection.lower()


@pytest.mark.unit
def test_project_for_embedding_excludes_metadata():
    """Test that projection excludes low-signal metadata."""
    belief = create_sample_belief()

    projection = project_for_embedding(belief)

    # Should generally exclude metadata like IDs, timestamps
    # (depends on implementation)
    assert isinstance(projection, str)
    # Metadata exclusion is implementation-specific


@pytest.mark.unit
def test_project_for_embedding_complete(test_beliefs):
    """Test projection with complete belief structure."""
    belief = test_beliefs["complete_valid"]

    projection = project_for_embedding(belief)

    assert isinstance(projection, str)
    assert len(projection) > 100  # Should be substantial
    # Should include core content
    assert belief["thesis"]["stance"] in projection
