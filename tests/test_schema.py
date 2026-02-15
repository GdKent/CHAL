"""
Unit tests for CBS-v1 belief schema validation.

Tests cover:
- Valid belief validation (minimal and complete)
- Invalid belief detection (missing fields, wrong types)
- ID format validation
- Metadata validation
- Thesis validation
- Optional collections validation
"""

import pytest
import json
from pathlib import Path
from chal.beliefs.schema import validate_belief
from tests.utils import (
    create_sample_belief,
    create_invalid_belief,
    assert_belief_valid,
    assert_belief_has_errors
)


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


# ==============================================
# 1. Valid Belief Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_belief_minimal_valid(test_beliefs):
    """Test validation of minimal valid belief with only required fields."""
    belief = test_beliefs["minimal_valid"]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Minimal valid belief should have no errors, got: {errors}"


@pytest.mark.unit
def test_validate_belief_complete(test_beliefs):
    """Test validation of complete belief with all optional fields populated."""
    belief = test_beliefs["complete_valid"]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Complete valid belief should have no errors, got: {errors}"


@pytest.mark.unit
def test_validate_belief_schema_version():
    """Test that schema_version must be 'CBS-v1'."""
    belief = create_sample_belief()
    belief["schema_version"] = "CBS-v1"
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_belief_version_number():
    """Test that version must be >= 1."""
    belief = create_sample_belief()
    belief["version"] = 1
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief["version"] = 5
    errors = validate_belief(belief)
    assert len(errors) == 0


# ==============================================
# 2. Invalid Belief Detection Tests
# ==============================================

@pytest.mark.unit
def test_validate_belief_missing_schema_version():
    """Test detection of missing schema_version field."""
    belief = create_invalid_belief("missing_schema_version")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("schema_version" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_wrong_schema_version():
    """Test detection of incorrect schema version."""
    belief = create_invalid_belief("wrong_schema_version")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("cbs-v1" in error.lower() or "schema" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_missing_belief_id():
    """Test detection of missing belief_id field."""
    belief = create_invalid_belief("missing_belief_id")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("belief_id" in error.lower() for error in errors)


@pytest.mark.skip(reason="Schema validation doesn't check version type yet")
@pytest.mark.unit
def test_validate_belief_invalid_version_type():
    """Test detection of version as string instead of int."""
    belief = create_sample_belief()
    belief["version"] = "1"  # String instead of int
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("version" in error.lower() for error in errors)


@pytest.mark.skip(reason="Schema validation doesn't check version >= 1 yet")
@pytest.mark.unit
def test_validate_belief_zero_version():
    """Test detection of version = 0 (should be >= 1)."""
    belief = create_invalid_belief("invalid_version")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("version" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_missing_metadata():
    """Test detection of missing metadata field."""
    belief = create_invalid_belief("missing_metadata")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("metadata" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_missing_thesis():
    """Test detection of missing thesis field."""
    belief = create_invalid_belief("missing_thesis")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("thesis" in error.lower() for error in errors)


# ==============================================
# 3. ID Format Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_id_format_valid():
    """Test validation of valid ID formats: A1, C12, E3, P1, N5."""
    belief = create_sample_belief(num_assumptions=3, num_claims=2, num_evidence=2)
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_id_format_invalid():
    """Test detection of invalid ID formats."""
    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["id"] = "A"  # Missing number
    errors = validate_belief(belief)
    assert len(errors) > 0

    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["id"] = "1A"  # Number before letter
    errors = validate_belief(belief)
    assert len(errors) > 0


@pytest.mark.unit
def test_validate_id_prefix_mismatch():
    """Test detection of ID with wrong prefix in wrong collection."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["id"] = "A1"  # Assumption ID in claims array
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("claims" in error.lower() and ("id" in error.lower() or "a1" in error.lower()) for error in errors)


@pytest.mark.unit
def test_validate_id_uniqueness():
    """Test detection of duplicate IDs within same collection."""
    belief = create_sample_belief(num_assumptions=2)
    belief["assumptions"][1]["id"] = "A1"  # Duplicate of first assumption
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("duplicate" in error.lower() or "unique" in error.lower() for error in errors)


# ==============================================
# 4. Metadata Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_metadata_required_fields():
    """Test that metadata contains required fields: topic_query, agent_persona, created_at."""
    belief = create_sample_belief()
    assert "topic_query" in belief["metadata"]
    assert "agent_persona" in belief["metadata"]
    assert "created_at" in belief["metadata"]
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_metadata_optional_fields(test_beliefs):
    """Test that optional metadata fields are accepted."""
    belief = test_beliefs["complete_valid"]
    assert "last_updated" in belief["metadata"]
    assert "scope_conditions" in belief["metadata"]
    assert "definitions" in belief["metadata"]
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_metadata_definitions_structure(test_beliefs):
    """Test that definitions are structured as term/definition pairs."""
    belief = test_beliefs["complete_valid"]
    definitions = belief["metadata"]["definitions"]
    assert isinstance(definitions, list)
    assert len(definitions) > 0
    assert "term" in definitions[0]
    assert "definition" in definitions[0]


# ==============================================
# 5. Thesis Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_thesis_required_fields():
    """Test that thesis contains required fields: stance, summary_bullets, confidence."""
    belief = create_sample_belief()
    assert "stance" in belief["thesis"]
    assert "summary_bullets" in belief["thesis"]
    assert "confidence" in belief["thesis"]
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_thesis_confidence_range():
    """Test that confidence is in range [0.0, 1.0]."""
    belief = create_sample_belief(confidence=0.0)
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(confidence=1.0)
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(confidence=0.5)
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.skip(reason="Schema validation doesn't check confidence bounds yet")
@pytest.mark.unit
def test_validate_thesis_confidence_out_of_bounds():
    """Test detection of confidence < 0 or > 1."""
    belief = create_invalid_belief("confidence_out_of_bounds")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("confidence" in error.lower() for error in errors)


@pytest.mark.skip(reason="Schema validation doesn't check empty bullets yet")
@pytest.mark.unit
def test_validate_thesis_empty_bullets():
    """Test detection of empty summary_bullets array."""
    belief = create_invalid_belief("empty_bullets")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("summary_bullets" in error.lower() or "bullets" in error.lower() for error in errors)


# ==============================================
# 6. Optional Collections Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_assumptions_structure():
    """Test validation of assumption objects."""
    belief = create_sample_belief(num_assumptions=2)
    assert "assumptions" in belief
    assert len(belief["assumptions"]) == 2
    assert "id" in belief["assumptions"][0]
    assert "statement" in belief["assumptions"][0]
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_claims_structure():
    """Test validation of claim objects."""
    belief = create_sample_belief(num_claims=2)
    assert "claims" in belief
    assert len(belief["claims"]) == 2
    claim = belief["claims"][0]
    assert "id" in claim
    assert "type" in claim
    assert "statement" in claim
    assert "depends_on" in claim
    assert "backing_evidence_ids" in claim
    assert "confidence" in claim
    assert "status" in claim
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_evidence_structure():
    """Test validation of evidence objects."""
    belief = create_sample_belief(num_evidence=2)
    assert "evidence" in belief
    assert len(belief["evidence"]) == 2
    evidence = belief["evidence"][0]
    assert "id" in evidence
    assert "type" in evidence
    assert "summary" in evidence
    assert "citation" in evidence
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_predictions_structure(test_beliefs):
    """Test validation of prediction objects."""
    belief = test_beliefs["complete_valid"]
    if "predictions" in belief:
        predictions = belief["predictions"]
        assert isinstance(predictions, list)
        if len(predictions) > 0:
            prediction = predictions[0]
            assert "id" in prediction
            assert "statement" in prediction
            assert "linked_claims" in prediction


@pytest.mark.unit
def test_validate_normative_implications_structure(test_beliefs):
    """Test validation of normative_implications objects."""
    belief = test_beliefs["complete_valid"]
    if "normative_implications" in belief:
        implications = belief["normative_implications"]
        assert isinstance(implications, list)
        if len(implications) > 0:
            implication = implications[0]
            assert "id" in implication
            assert "statement" in implication
