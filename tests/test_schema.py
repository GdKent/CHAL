"""
Unit tests for CBS belief schema validation.

Tests cover:
- Valid belief validation (minimal and complete)
- Invalid belief detection (missing fields, wrong types)
- ID format validation
- Metadata validation
- Thesis validation
- Optional collections validation
- Assumption type and strength validation
- Counterposition validation (no numeric strength)
- Claims validation (strength, predictions)
- Evidence validation (strength)
- Uncertainty validation (targets, status)
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
    """Test that schema_version must be 'CBS'."""
    belief = create_sample_belief()
    belief["schema_version"] = "CBS"
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
    assert any("cbs-v0" in error.lower() or "schema" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_missing_belief_id():
    """Test detection of missing belief_id field."""
    belief = create_invalid_belief("missing_belief_id")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("belief_id" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_belief_invalid_version_type():
    """Test detection of version as string instead of int."""
    belief = create_sample_belief()
    belief["version"] = "1"  # String instead of int
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("version" in error.lower() for error in errors)


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
    """Test validation of valid ID formats: A1, C12, E3."""
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


@pytest.mark.unit
def test_validate_id_p_and_n_prefixes_rejected():
    """Test that P# and N# IDs are rejected by the ID regex."""
    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["id"] = "P1"
    errors = validate_belief(belief)
    assert len(errors) > 0, "P# IDs should be rejected"

    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["id"] = "N1"
    errors = validate_belief(belief)
    assert len(errors) > 0, "N# IDs should be rejected"


# ==============================================
# 4. Metadata Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_metadata_required_fields():
    """Test that metadata contains required fields: topic_query, agent_persona."""
    belief = create_sample_belief()
    assert "topic_query" in belief["metadata"]
    assert "agent_persona" in belief["metadata"]
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
    """Test that thesis contains required fields: stance, summary_bullets, strength."""
    belief = create_sample_belief()
    assert "stance" in belief["thesis"]
    assert "summary_bullets" in belief["thesis"]
    assert "strength" in belief["thesis"]
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_thesis_strength_range():
    """Test that strength is in range [0.0, 1.0]."""
    belief = create_sample_belief(confidence=0.0)
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(confidence=1.0)
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(confidence=0.5)
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_thesis_strength_out_of_bounds():
    """Test detection of strength < 0 or > 1."""
    belief = create_invalid_belief("strength_out_of_bounds")
    errors = validate_belief(belief)
    assert len(errors) > 0
    assert any("strength" in error.lower() for error in errors)


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
    """Test validation of assumption objects with typed categories and strength."""
    belief = create_sample_belief(num_assumptions=2)
    assert "assumptions" in belief
    assert len(belief["assumptions"]) == 2
    assert "id" in belief["assumptions"][0]
    assert "type" in belief["assumptions"][0]
    assert "statement" in belief["assumptions"][0]
    assert "strength" in belief["assumptions"][0]
    assert belief["assumptions"][0]["type"] in ("foundational", "empirical", "methodological")
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
    assert "depends_on" in claim
    assert "strength" in claim
    assert "status" in claim
    assert "predictions" in claim
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_evidence_structure():
    """Test validation of evidence objects with strength."""
    belief = create_sample_belief(num_evidence=2)
    assert "evidence" in belief
    assert len(belief["evidence"]) == 2
    evidence = belief["evidence"][0]
    assert "id" in evidence
    assert "type" in evidence
    assert "summary" in evidence
    assert "source" in evidence
    assert "strength" in evidence
    errors = validate_belief(belief)
    assert len(errors) == 0


# ==============================================
# 7. Assumption Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_assumption_type_valid():
    """Test that valid assumption types are accepted."""
    for atype in ("foundational", "empirical", "methodological"):
        belief = create_sample_belief(num_assumptions=1)
        belief["assumptions"][0]["type"] = atype
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Type '{atype}' should be valid, got: {errors}"


@pytest.mark.unit
def test_validate_assumption_type_invalid():
    """Test that invalid assumption type is rejected."""
    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["type"] = "speculative"
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid assumption type 'speculative' should produce errors"


@pytest.mark.unit
def test_validate_assumption_type_normative_rejected():
    """Test that 'normative' assumption type is rejected."""
    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["type"] = "normative"
    errors = validate_belief(belief)
    assert len(errors) > 0, "'normative' assumption type should be rejected"


@pytest.mark.unit
def test_validate_assumption_missing_type():
    """Test that missing assumption type is rejected."""
    belief = create_sample_belief(num_assumptions=1)
    del belief["assumptions"][0]["type"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Missing assumption type should produce errors"


@pytest.mark.unit
def test_validate_assumption_strength_range():
    """Test that assumption strength is validated in [0.0, 1.0]."""
    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["strength"] = 0.5
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["strength"] = 1.5
    errors = validate_belief(belief)
    assert len(errors) > 0, "Assumption strength 1.5 should produce errors"
    assert any("strength" in e.lower() for e in errors)

    belief = create_sample_belief(num_assumptions=1)
    belief["assumptions"][0]["strength"] = -0.1
    errors = validate_belief(belief)
    assert len(errors) > 0, "Assumption strength -0.1 should produce errors"


# ==============================================
# 8. Counterposition Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_counterpositions_structure(test_beliefs):
    """Test validation of counterposition objects with all required fields (no numeric strength)."""
    belief = test_beliefs["complete_valid"]
    assert "counterpositions" in belief
    assert len(belief["counterpositions"]) >= 2

    cp = belief["counterpositions"][0]
    assert "id" in cp
    assert "targets" in cp
    assert "attack_type" in cp
    assert "statement" in cp
    assert "my_response" in cp
    assert "response_sufficiency" in cp
    # X# should NOT have a numeric strength field
    assert "strength" not in cp
    errors = validate_belief(belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_counterposition_valid_without_strength():
    """Test that counterpositions are valid without a strength field."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Test counterposition",
        "my_response": "Test response",
        "response_sufficiency": "partial"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Counterposition without strength should pass, got: {errors}"


@pytest.mark.unit
def test_validate_counterposition_attack_type_valid():
    """Test that valid attack types are accepted."""
    for attack_type in ("undermining", "rebutting", "undercutting"):
        belief = create_sample_belief(num_claims=1)
        belief["counterpositions"] = [{
            "id": "X1",
            "targets": ["C1"],
            "attack_type": attack_type,
            "statement": "Test counterposition",
            "my_response": "Test response",
            "response_sufficiency": "partial"
        }]
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Attack type '{attack_type}' should be valid, got: {errors}"


@pytest.mark.unit
def test_validate_counterposition_attack_type_invalid():
    """Test that invalid attack type is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "destroying",
        "statement": "Test counterposition",
        "my_response": "Test response",
        "response_sufficiency": "partial"
    }]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid attack_type 'destroying' should produce errors"


@pytest.mark.unit
def test_validate_counterposition_response_sufficiency_valid():
    """Test that valid response_sufficiency values are accepted."""
    for sufficiency in ("sufficient", "partial", "unaddressed"):
        belief = create_sample_belief(num_claims=1)
        belief["counterpositions"] = [{
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Test counterposition",
            "my_response": "Test response",
            "response_sufficiency": sufficiency
        }]
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Sufficiency '{sufficiency}' should be valid, got: {errors}"


@pytest.mark.unit
def test_validate_counterposition_response_sufficiency_invalid():
    """Test that invalid response_sufficiency is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Test counterposition",
        "my_response": "Test response",
        "response_sufficiency": "maybe"
    }]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid response_sufficiency 'maybe' should produce errors"


@pytest.mark.unit
def test_validate_counterposition_id_prefix():
    """Test that counterposition IDs must start with X."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "C99",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Test counterposition",
        "my_response": "Test response",
        "response_sufficiency": "partial"
    }]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Counterposition with C-prefix ID should produce errors"
    assert any("prefix" in e.lower() or "wrong" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_counterposition_field_name():
    """Test that the field is called 'counterpositions' not 'counterposition_map'."""
    belief = create_sample_belief(num_claims=1)
    # Using the old field name should NOT be validated
    belief["counterposition_map"] = [{"id": "X1", "statement": "test"}]
    # The validator only checks 'counterpositions', not 'counterposition_map'
    # So counterposition_map is just an unknown extra field (allowed by additionalProperties: True)
    errors = validate_belief(belief)
    # No errors from the old field name (it's just ignored)
    # Verify that the correct field name works
    belief2 = create_sample_belief(num_claims=1)
    belief2["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Test",
        "my_response": "Response",
        "response_sufficiency": "sufficient"
    }]
    errors2 = validate_belief(belief2)
    assert len(errors2) == 0


# ==============================================
# 9. Claims Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_claim_valid_structure():
    """Test that a full claim with all required fields passes validation."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["claims"][0]["inference_chain"] = ["Step 1: premise", "Step 2: conclusion"]
    belief["claims"][0]["strength_justification"] = "Strong empirical support"
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Full claim should pass validation, got: {errors}"


@pytest.mark.unit
def test_validate_claim_valid_with_all_required():
    """Test that claim with all required fields passes validation."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    # All fields are now required — no optional fields remain on claims
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Claim with all required fields should pass, got: {errors}"


@pytest.mark.unit
def test_validate_claim_invalid_status():
    """Test that invalid claim status is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["status"] = "pending_review"
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid claim status 'pending_review' should produce errors"
    assert any("status" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_claim_strength_out_of_bounds():
    """Test that claim strength outside [0.0, 1.0] is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["strength"] = 1.5
    errors = validate_belief(belief)
    assert len(errors) > 0, "Claim strength 1.5 should produce errors"
    assert any("strength" in e.lower() for e in errors)

    belief2 = create_sample_belief(num_claims=1)
    belief2["claims"][0]["strength"] = -0.1
    errors2 = validate_belief(belief2)
    assert len(errors2) > 0, "Claim strength -0.1 should produce errors"
    assert any("strength" in e.lower() for e in errors2)


@pytest.mark.unit
def test_validate_claim_predictions_required():
    """Test that claims require a predictions array with at least one item."""
    belief = create_sample_belief(num_claims=1)
    del belief["claims"][0]["predictions"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Claim without predictions should produce errors"


@pytest.mark.unit
def test_validate_claim_predictions_empty_rejected():
    """Test that a claim with empty predictions array is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["predictions"] = []
    errors = validate_belief(belief)
    assert len(errors) > 0, "Claim with empty predictions should produce errors"


@pytest.mark.unit
def test_validate_claim_prediction_required_fields():
    """Test that each prediction requires statement, test, decision_criterion."""
    belief = create_sample_belief(num_claims=1)
    # Prediction missing 'test' field
    belief["claims"][0]["predictions"] = [
        {"statement": "Test prediction", "decision_criterion": "Some criterion"}
    ]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Prediction missing 'test' should produce errors"


@pytest.mark.unit
def test_validate_claim_prediction_with_optional_falsifiers():
    """Test that prediction with optional potential_falsifiers passes."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["predictions"] = [{
        "statement": "Test prediction",
        "test": "Test method",
        "decision_criterion": "Some criterion",
        "potential_falsifiers": ["Falsifier A", "Falsifier B"]
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Prediction with falsifiers should pass, got: {errors}"


# ==============================================
# 10. Evidence Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_evidence_valid_with_all_fields():
    """Test that evidence with all required fields passes."""
    belief = create_sample_belief(num_evidence=1, num_claims=1)
    # All required fields (including status, strength_justification) are set by create_sample_belief
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Evidence with all fields should pass, got: {errors}"


@pytest.mark.unit
def test_validate_evidence_valid_with_extra_fields():
    """Test that evidence with extra fields still passes (additionalProperties=True)."""
    belief = create_sample_belief(num_evidence=1, num_claims=1)
    belief["evidence"][0]["limitations"] = "Limited to Western populations"
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Evidence with extra fields should pass, got: {errors}"


@pytest.mark.unit
def test_validate_evidence_invalid_type():
    """Test that invalid evidence type is rejected."""
    belief = create_sample_belief(num_evidence=1, num_claims=1)
    belief["evidence"][0]["type"] = "anecdotal"
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid evidence type 'anecdotal' should produce errors"
    assert any("evidence type" in e.lower() or "anecdotal" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_evidence_strength_range():
    """Test that evidence strength is validated in [0.0, 1.0]."""
    belief = create_sample_belief(num_evidence=1, num_claims=1)
    belief["evidence"][0]["strength"] = 0.5
    errors = validate_belief(belief)
    assert len(errors) == 0

    belief = create_sample_belief(num_evidence=1, num_claims=1)
    belief["evidence"][0]["strength"] = 1.5
    errors = validate_belief(belief)
    assert len(errors) > 0, "Evidence strength 1.5 should produce errors"
    assert any("strength" in e.lower() for e in errors)

    belief = create_sample_belief(num_evidence=1, num_claims=1)
    belief["evidence"][0]["strength"] = -0.1
    errors = validate_belief(belief)
    assert len(errors) > 0, "Evidence strength -0.1 should produce errors"


# ==============================================
# 11. Uncertainty Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_uncertainty_valid_structure():
    """Test that a valid uncertainty with targets and status passes validation."""
    belief = create_sample_belief(num_claims=1)
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Can the explanatory gap be closed?",
        "status": "active",
        "importance": "Resolving this directly determines viability"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Full uncertainty should pass, got: {errors}"


@pytest.mark.unit
def test_validate_uncertainty_valid_without_optional():
    """Test that uncertainty without optional fields still passes."""
    belief = create_sample_belief(num_claims=1)
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Is this assumption valid?",
        "status": "active"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Uncertainty without optional fields should pass, got: {errors}"


@pytest.mark.unit
def test_validate_uncertainty_resolved_with_note():
    """Test that a resolved uncertainty with resolution_note passes."""
    belief = create_sample_belief(num_claims=1)
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Can the explanatory gap be closed?",
        "status": "resolved",
        "resolution_note": "Resolved by new claim C2 with evidence E2"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Resolved uncertainty should pass, got: {errors}"


@pytest.mark.unit
def test_validate_uncertainty_invalid_status():
    """Test that invalid uncertainty status is rejected."""
    belief = create_sample_belief(num_claims=1)
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Test question",
        "status": "pending"
    }]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Invalid status 'pending' should produce errors"
    assert any("status" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_uncertainty_valid_without_cruciality():
    """Test that uncertainties are valid without a cruciality field (removed)."""
    belief = create_sample_belief(num_claims=1)
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Test question",
        "status": "active"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Uncertainty without cruciality should pass, got: {errors}"


# ==============================================
# 12. Changelog Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_changelog_valid_structure():
    """Test that a valid changelog entry passes validation."""
    belief = create_sample_belief()
    belief["changelog"] = [{
        "version": 2,
        "changes": ["Updated thesis strength", "Revised claim C1"],
        "timestamp": "2026-02-16T10:00:00Z"
    }]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Valid changelog should pass, got: {errors}"


@pytest.mark.unit
def test_validate_changelog_multiple_entries():
    """Test that multiple changelog entries pass validation."""
    belief = create_sample_belief()
    belief["changelog"] = [
        {
            "version": 2,
            "changes": ["Initial revision"],
            "timestamp": "2026-02-16T10:00:00Z"
        },
        {
            "version": 3,
            "changes": ["Added evidence E3", "Updated claim C2"],
            "timestamp": "2026-02-17T14:30:00Z"
        }
    ]
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Multiple changelog entries should pass, got: {errors}"


# ==============================================
# 13. Assumption Status Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_assumption_status_valid():
    """Test that valid assumption statuses are accepted."""
    for status in ("active", "revised", "retracted"):
        belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
        belief["assumptions"][0]["status"] = status
        errors = validate_belief(belief)
        assert not any("assumption status" in e.lower() for e in errors), \
            f"Valid status '{status}' should not produce status errors, got: {errors}"


@pytest.mark.unit
def test_validate_assumption_status_invalid():
    """Test that invalid assumption status is rejected."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["assumptions"][0]["status"] = "pending_review"
    errors = validate_belief(belief)
    assert any("assumption status" in e.lower() for e in errors), \
        f"Invalid assumption status should produce errors, got: {errors}"


@pytest.mark.unit
def test_validate_evidence_status_valid():
    """Test that valid evidence statuses are accepted."""
    for status in ("active", "revised", "retracted"):
        belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
        belief["evidence"][0]["status"] = status
        errors = validate_belief(belief)
        assert not any("evidence status" in e.lower() for e in errors), \
            f"Valid status '{status}' should not produce status errors, got: {errors}"


@pytest.mark.unit
def test_validate_evidence_status_invalid():
    """Test that invalid evidence status is rejected."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["evidence"][0]["status"] = "archived"
    errors = validate_belief(belief)
    assert any("evidence status" in e.lower() for e in errors), \
        f"Invalid evidence status should produce errors, got: {errors}"


@pytest.mark.unit
def test_strength_justification_required_for_claims():
    """Test that missing strength_justification on claims triggers error."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    del belief["claims"][0]["strength_justification"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Missing strength_justification on claims should produce errors"


@pytest.mark.unit
def test_strength_justification_required_for_assumptions():
    """Test that missing strength_justification on assumptions triggers error."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    del belief["assumptions"][0]["strength_justification"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Missing strength_justification on assumptions should produce errors"


@pytest.mark.unit
def test_strength_justification_required_for_evidence():
    """Test that missing strength_justification on evidence triggers error."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    del belief["evidence"][0]["strength_justification"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Missing strength_justification on evidence should produce errors"


@pytest.mark.unit
def test_inference_chain_required_for_claims():
    """Test that missing inference_chain on claims triggers error."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    del belief["claims"][0]["inference_chain"]
    errors = validate_belief(belief)
    assert len(errors) > 0, "Missing inference_chain on claims should produce errors"


@pytest.mark.unit
def test_inference_chain_empty_rejected():
    """Test that empty inference_chain on claims triggers error."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["claims"][0]["inference_chain"] = []
    errors = validate_belief(belief)
    assert len(errors) > 0, "Empty inference_chain on claims should produce errors"
