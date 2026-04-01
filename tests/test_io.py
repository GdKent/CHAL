"""
Unit tests for belief I/O operations.

Tests cover:
- Parsing model output (JSON extraction, ID normalization, validation)
- Belief to markdown conversion (strength, predictions inline, no P#/N#)
- Embedding projection (strength-sorted, no standalone predictions)
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
    assert parsed["schema_version"] == "CBS"


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


@pytest.mark.unit
def test_parse_model_output_normalizes_aceux_ids():
    """Test that A#, C#, E#, U#, X# IDs are normalized (hash removed)."""
    response = '''```json
{
  "schema_version": "CBS",
  "belief_id": "BELIEF-TEST",
  "version": 1,
  "metadata": {"topic_query": "Test", "agent_persona": "Test"},
  "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
  "claims": [
    {
      "id": "C#1",
      "type": "deductive",
      "statement": "Test claim",
      "depends_on": ["A#1", "E#1"],
      "strength": 0.7,
      "status": "active",
      "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
    }
  ],
  "assumptions": [{"id": "A#1", "type": "empirical", "statement": "Test", "strength": 0.8}],
  "evidence": [{"id": "E#1", "type": "empirical", "summary": "Test", "source": "Test (2026)", "relevance_to_claims": ["C#1"], "strength": 0.8}]
}
```'''

    parsed, _, errors = parse_model_output_to_belief(response)

    assert parsed is not None
    # IDs should have hash removed
    assert parsed["claims"][0]["id"] == "C1"
    assert parsed["claims"][0]["depends_on"] == ["A1", "E1"]
    assert parsed["assumptions"][0]["id"] == "A1"
    assert parsed["evidence"][0]["id"] == "E1"


@pytest.mark.unit
def test_parse_model_output_does_not_normalize_pn_ids():
    """Test that P# and N# IDs are NOT normalized (removed from schema)."""
    response = '''```json
{
  "schema_version": "CBS",
  "belief_id": "BELIEF-TEST",
  "version": 1,
  "metadata": {"topic_query": "Test", "agent_persona": "Test"},
  "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
  "extra_field_with_P#1": "should stay as P#1",
  "extra_field_with_N#1": "should stay as N#1"
}
```'''

    parsed, _, _ = parse_model_output_to_belief(response)

    assert parsed is not None
    # P# and N# should NOT be normalized since they're removed from the regex
    assert "extra_field_with_P#1" in parsed
    assert "extra_field_with_N#1" in parsed


# ==============================================
# 2. Belief to Markdown — Thesis
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_minimal(test_beliefs):
    """Test markdown conversion with only required fields."""
    belief = test_beliefs["minimal_valid"]

    markdown = belief_to_markdown(belief)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert "# Thesis" in markdown


@pytest.mark.unit
def test_belief_to_markdown_thesis_strength(test_beliefs):
    """Test that thesis renders 'Strength' not 'Confidence'."""
    belief = test_beliefs["minimal_valid"]

    markdown = belief_to_markdown(belief)

    assert "Strength: 0.75" in markdown
    assert "Confidence" not in markdown


@pytest.mark.unit
def test_belief_to_markdown_thesis_section(test_beliefs):
    """Test that thesis is formatted correctly in markdown."""
    belief = test_beliefs["minimal_valid"]

    markdown = belief_to_markdown(belief)

    assert belief["thesis"]["stance"] in markdown
    for bullet in belief["thesis"]["summary_bullets"]:
        assert bullet in markdown


# ==============================================
# 3. Belief to Markdown — Assumptions with Strength
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_assumptions_with_strength(test_beliefs):
    """Test that assumptions render with strength values."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert "# Assumptions" in markdown
    for assumption in belief["assumptions"]:
        assert assumption["statement"] in markdown
        assert f"Strength: {assumption['strength']}" in markdown


# ==============================================
# 4. Belief to Markdown — Claims with Strength & Predictions
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_claims_strength(test_beliefs):
    """Test that claims render 'Strength' not 'Confidence'."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert "# Claims" in markdown
    for claim in belief["claims"]:
        assert claim["statement"] in markdown
    # Should use "Strength" not "Confidence"
    assert "Confidence:" not in markdown


@pytest.mark.unit
def test_belief_to_markdown_claims_inline_predictions(test_beliefs):
    """Test that predictions render inline under their parent claims."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    # Should have "Predictions:" under claims
    assert "Predictions:" in markdown
    # Each claim's prediction should appear
    for claim in belief["claims"]:
        for pred in claim.get("predictions", []):
            assert pred["statement"] in markdown


@pytest.mark.unit
def test_belief_to_markdown_no_known_weaknesses():
    """Test that known_weaknesses is not rendered (removed from schema)."""
    belief = create_sample_belief()
    # Even if someone includes known_weaknesses, it shouldn't render
    belief["claims"][0]["known_weaknesses"] = ["Some weakness"]

    markdown = belief_to_markdown(belief)

    assert "Known weaknesses" not in markdown
    assert "Some weakness" not in markdown


@pytest.mark.unit
def test_belief_to_markdown_strength_justification(test_beliefs):
    """Test that strength_justification renders under claims."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    for claim in belief["claims"]:
        if claim.get("strength_justification"):
            assert claim["strength_justification"] in markdown


# ==============================================
# 5. Belief to Markdown — Evidence with Strength
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_evidence_with_strength(test_beliefs):
    """Test that evidence renders with strength values."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert "# Evidence" in markdown
    for evidence in belief["evidence"]:
        assert evidence["summary"] in markdown
        assert f"Strength: {evidence['strength']}" in markdown


# ==============================================
# 6. Belief to Markdown — Uncertainties with Targets/Status
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_uncertainties_targets_status():
    """Test that uncertainties render with targets and status."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {
            "id": "U1",
            "targets": ["C1", "A1"],
            "question": "Is the mechanism clear?",
            "status": "active",
            "importance": "High value"
        }
    ]

    markdown = belief_to_markdown(belief)

    assert "# Uncertainties" in markdown
    assert "U1" in markdown
    assert "(active)" in markdown
    assert "targets: C1, A1" in markdown
    assert "Is the mechanism clear?" in markdown
    assert "Importance: High value" in markdown


@pytest.mark.unit
def test_belief_to_markdown_uncertainties_resolved():
    """Test that resolved uncertainties render with resolution_note."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {
            "id": "U1",
            "targets": ["C1"],
            "question": "Was the gap resolved?",
            "status": "resolved",
            "resolution_note": "Resolved by new evidence E3",
            "importance": ""
        }
    ]

    markdown = belief_to_markdown(belief)

    assert "(resolved)" in markdown
    assert "Resolution: Resolved by new evidence E3" in markdown


@pytest.mark.unit
def test_belief_to_markdown_no_cruciality():
    """Test that uncertainties do NOT render cruciality (removed)."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {
            "id": "U1",
            "targets": ["C1"],
            "question": "Test",
            "status": "active",
            "importance": ""
        }
    ]

    markdown = belief_to_markdown(belief)

    assert "cruciality" not in markdown


# ==============================================
# 7. Belief to Markdown — Counterpositions (No Strength)
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_counterpositions_no_strength(test_beliefs):
    """Test that counterpositions do NOT render numeric strength."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert "# Counterpositions" in markdown
    # Find the counterpositions section and check
    x_section_start = markdown.index("# Counterpositions")
    x_section = markdown[x_section_start:]
    # Should have Sufficiency but NOT "Strength:"
    assert "Sufficiency:" in x_section
    assert "Strength:" not in x_section


@pytest.mark.unit
def test_belief_to_markdown_counterpositions_content(test_beliefs):
    """Test that counterpositions render statement, target, attack type."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    for x in belief["counterpositions"]:
        assert x["statement"] in markdown
        assert x["attack_type"] in markdown


# ==============================================
# 8. Belief to Markdown — Removed Sections
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_no_standalone_predictions_section():
    """Test that no standalone 'Predictions (falsifiable)' section is rendered."""
    belief = create_sample_belief()

    markdown = belief_to_markdown(belief)

    assert "# Predictions (falsifiable)" not in markdown
    assert "Likelihood:" not in markdown
    assert "Importance:" not in markdown


@pytest.mark.unit
def test_belief_to_markdown_no_normative_implications_section():
    """Test that no 'Normative Implications' section is rendered."""
    belief = create_sample_belief()

    markdown = belief_to_markdown(belief)

    assert "Normative Implications" not in markdown


# ==============================================
# 9. Belief to Markdown — Graph Structure
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_no_argument_structure(test_beliefs):
    """Test that Argument Structure section is not included in markdown."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert "# Argument Structure" not in markdown
    assert "Total nodes:" not in markdown


@pytest.mark.unit
def test_belief_to_markdown_complete(test_beliefs):
    """Test markdown conversion with all fields populated."""
    belief = test_beliefs["complete_valid"]

    markdown = belief_to_markdown(belief)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    # All major sections present
    assert "# Thesis" in markdown
    assert "# Assumptions" in markdown
    assert "# Claims" in markdown
    assert "# Evidence" in markdown
    assert "# Counterpositions" in markdown


# ==============================================
# 10. Embedding Projection Tests
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

    assert belief["thesis"]["stance"] in projection


@pytest.mark.unit
def test_project_for_embedding_uses_strength():
    """Test that projection uses 'Strength' not 'Confidence'."""
    belief = create_sample_belief()

    projection = project_for_embedding(belief)

    assert "Strength:" in projection
    assert "Confidence:" not in projection


@pytest.mark.unit
def test_project_for_embedding_sorts_claims_by_strength():
    """Test that claims are sorted by strength descending."""
    belief = create_sample_belief(num_claims=3)
    belief["claims"][0]["strength"] = 0.3
    belief["claims"][1]["strength"] = 0.9
    belief["claims"][2]["strength"] = 0.5

    projection = project_for_embedding(belief)

    # C2 (0.9) should appear before C3 (0.5) which should appear before C1 (0.3)
    c2_pos = projection.index("Claim C2")
    c3_pos = projection.index("Claim C3")
    c1_pos = projection.index("Claim C1")
    assert c2_pos < c3_pos < c1_pos


@pytest.mark.unit
def test_project_for_embedding_includes_claims(test_beliefs):
    """Test that projection contains claim statements."""
    belief = test_beliefs["complete_valid"]

    projection = project_for_embedding(belief)

    for claim in belief["claims"]:
        assert claim["statement"] in projection


@pytest.mark.unit
def test_project_for_embedding_no_standalone_predictions():
    """Test that projection does NOT include standalone prediction lines."""
    belief = create_sample_belief()

    projection = project_for_embedding(belief)

    # Should NOT have "Prediction P1:" or standalone prediction lines
    assert "Prediction P" not in projection
    assert "Likelihood" not in projection


@pytest.mark.unit
def test_project_for_embedding_uncertainty_with_targets():
    """Test that uncertainty embedding includes targets and status."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {
            "id": "U1",
            "targets": ["C1"],
            "question": "Is the mechanism clear?",
            "status": "active",
            "importance": ""
        }
    ]

    projection = project_for_embedding(belief)

    assert "Uncertainty U1" in projection
    assert "targets=C1" in projection
    assert "status=active" in projection


@pytest.mark.unit
def test_project_for_embedding_counterpositions_no_strength():
    """Test that counterposition embedding does NOT include numeric strength."""
    belief = create_sample_belief()
    belief["counterpositions"] = [
        {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Counter to C1",
            "my_response": "Response",
            "response_sufficiency": "partial"
        }
    ]

    projection = project_for_embedding(belief)

    assert "Counterposition X1" in projection
    assert "sufficiency=partial" in projection
    # Should NOT have "strength=" for counterpositions
    x_line = [l for l in projection.split("\n") if "Counterposition" in l][0]
    assert "strength=" not in x_line


@pytest.mark.unit
def test_project_for_embedding_excludes_metadata():
    """Test that projection excludes low-signal metadata."""
    belief = create_sample_belief()

    projection = project_for_embedding(belief)

    assert isinstance(projection, str)


@pytest.mark.unit
def test_project_for_embedding_complete(test_beliefs):
    """Test projection with complete belief structure."""
    belief = test_beliefs["complete_valid"]

    projection = project_for_embedding(belief)

    assert isinstance(projection, str)
    assert len(projection) > 100
    assert belief["thesis"]["stance"] in projection
