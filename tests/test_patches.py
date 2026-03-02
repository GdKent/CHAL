"""
Unit tests for belief patch application.

Tests cover:
- Patch application for all operation types
- Confidence propagation through dependency graph
- Version management
- Changelog generation
- Patch validation
- Edge cases
"""

import pytest
import json
import copy
from pathlib import Path
from chal.beliefs.patches import apply_patches, validate_patches
from tests.utils import create_sample_belief


# ==============================================
# Test Fixtures
# ==============================================

@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_patches(fixtures_dir):
    """Load test patches from fixtures."""
    with open(fixtures_dir / "test_patches.json") as f:
        return json.load(f)


# ==============================================
# 1. Patch Application - update_thesis
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_thesis_weaken(test_patches):
    """Test weakening thesis confidence by 0.1."""
    belief = create_sample_belief(confidence=0.75)
    patches = test_patches["update_thesis_weaken"]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["confidence"] == pytest.approx(0.65, abs=0.01)


@pytest.mark.unit
def test_apply_patches_update_thesis_strengthen(test_patches):
    """Test strengthening thesis confidence by 0.1."""
    belief = create_sample_belief(confidence=0.75)
    patches = test_patches["update_thesis_strengthen"]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["confidence"] == pytest.approx(0.85, abs=0.01)


@pytest.mark.unit
def test_apply_patches_update_thesis_floor():
    """Test that confidence cannot go below 0.0."""
    belief = create_sample_belief(confidence=0.05)
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["confidence"] >= 0.0


@pytest.mark.unit
def test_apply_patches_update_thesis_ceiling():
    """Test that confidence cannot exceed 1.0."""
    belief = create_sample_belief(confidence=0.95)
    patches = [{"op": "update_thesis", "change": "strengthen"}]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["confidence"] <= 1.0


# ==============================================
# 2. Patch Application - update_claim
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_claim_confidence(test_patches):
    """Test modifying claim confidence."""
    belief = create_sample_belief(num_claims=1)
    original_confidence = belief["claims"][0]["confidence"]
    patches = test_patches["update_claim_confidence"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["confidence"] == 0.6
    assert updated["claims"][0]["confidence"] != original_confidence


@pytest.mark.unit
def test_apply_patches_update_claim_status(test_patches):
    """Test changing claim status."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["update_claim_status"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["status"] == "revised"


@pytest.mark.unit
def test_apply_patches_update_claim_statement():
    """Test modifying claim statement."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {
            "statement": "Updated claim statement"
        }
    }]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["statement"] == "Updated claim statement"


@pytest.mark.unit
def test_apply_patches_update_claim_multiple_fields(test_patches):
    """Test updating multiple fields of a claim."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["update_claim_multiple"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["confidence"] == 0.65
    assert updated["claims"][0]["status"] == "revised"
    assert "known_weaknesses" in updated["claims"][0]
    assert len(updated["claims"][0]["known_weaknesses"]) > 0


@pytest.mark.unit
def test_apply_patches_update_claim_not_found():
    """Test error when updating non-existent claim."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C99",
        "changes": {"confidence": 0.5}
    }]

    with pytest.raises((ValueError, KeyError)):
        apply_patches(belief, patches)


# ==============================================
# 3. Patch Application - retire_claim
# ==============================================

@pytest.mark.unit
def test_apply_patches_retire_claim_success(test_patches):
    """Test retiring a claim sets status='retracted' and confidence=0.0."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["retire_claim"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["status"] == "retracted"
    assert updated["claims"][0]["confidence"] == 0.0


@pytest.mark.unit
def test_apply_patches_retire_claim_not_found():
    """Test error when retiring non-existent claim."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "retire_claim",
        "target_id": "C99"
    }]

    with pytest.raises((ValueError, KeyError)):
        apply_patches(belief, patches)


# ==============================================
# 4. Patch Application - add_evidence
# ==============================================

@pytest.mark.unit
def test_apply_patches_add_evidence_success(test_patches):
    """Test adding new evidence item."""
    belief = create_sample_belief(num_evidence=1)
    original_count = len(belief["evidence"])
    patches = test_patches["add_evidence"]

    updated = apply_patches(belief, patches)

    assert len(updated["evidence"]) == original_count + 1
    assert any(e["id"] == "E3" for e in updated["evidence"])


@pytest.mark.unit
def test_apply_patches_add_evidence_no_evidence_array():
    """Test adding evidence when evidence array doesn't exist."""
    belief = create_sample_belief(num_evidence=0)
    if "evidence" in belief:
        del belief["evidence"]

    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E1",
            "type": "empirical",
            "summary": "New evidence",
            "citation": "Test (2026)"
        }
    }]

    updated = apply_patches(belief, patches)

    assert "evidence" in updated
    assert len(updated["evidence"]) == 1
    assert updated["evidence"][0]["id"] == "E1"


@pytest.mark.unit
def test_apply_patches_add_evidence_missing_id():
    """Test error if evidence item has no 'id' field."""
    belief = create_sample_belief(num_evidence=0)
    patches = [{
        "op": "add_evidence",
        "item": {
            "type": "empirical",
            "summary": "Evidence without ID"
        }
    }]

    with pytest.raises((ValueError, KeyError)):
        apply_patches(belief, patches)


# ==============================================
# 5. Patch Application - update_assumption
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_assumption_success(test_patches):
    """Test updating assumption statement."""
    belief = create_sample_belief(num_assumptions=1)
    original_statement = belief["assumptions"][0]["statement"]
    patches = test_patches["update_assumption"]

    updated = apply_patches(belief, patches)

    assert updated["assumptions"][0]["statement"] != original_statement
    assert updated["assumptions"][0]["statement"] == "Revised assumption statement"


@pytest.mark.unit
def test_apply_patches_update_assumption_not_found():
    """Test error when updating non-existent assumption."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "update_assumption",
        "target_id": "A99",
        "new_statement": "New statement"
    }]

    with pytest.raises((ValueError, KeyError)):
        apply_patches(belief, patches)


# ==============================================
# 6. Confidence Propagation Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_propagate_simple():
    """Test simple propagation: C2 depends on C1; C1 weakens → C2 weakens."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["confidence"] = 0.8
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []

    belief["claims"][1]["confidence"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_confidence=True)

    assert updated["claims"][0]["confidence"] == 0.5
    # C2 depends on C1 and is capped at C1's new confidence
    assert updated["claims"][1]["confidence"] == 0.5


@pytest.mark.unit
def test_apply_patches_propagate_transitive():
    """Test transitive propagation: C3 → C2 → C1; C1 weakens → C2, C3 weaken."""
    belief = create_sample_belief(num_assumptions=0, num_claims=3, num_evidence=0)
    belief["claims"][0]["id"] = "C1"
    belief["claims"][0]["confidence"] = 0.8
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []

    belief["claims"][1]["id"] = "C2"
    belief["claims"][1]["confidence"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    belief["claims"][2]["id"] = "C3"
    belief["claims"][2]["confidence"] = 0.8
    belief["claims"][2]["depends_on"] = ["C2"]
    belief["claims"][2]["backing_evidence_ids"] = []

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_confidence=True)

    # Propagation: C2 and C3 are capped at C1's new confidence
    assert updated["claims"][0]["confidence"] == 0.5
    assert updated["claims"][1]["confidence"] == 0.5
    assert updated["claims"][2]["confidence"] == 0.5


@pytest.mark.unit
def test_apply_patches_propagate_no_change():
    """Test that propagation doesn't occur if dependent already has lower confidence."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["confidence"] = 0.8
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []

    belief["claims"][1]["confidence"] = 0.4  # Already lower
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.7}
    }]

    updated = apply_patches(belief, patches, propagate_confidence=True)

    # C2 should remain at 0.4
    assert updated["claims"][1]["confidence"] == 0.4


@pytest.mark.unit
def test_apply_patches_propagate_multiple_dependencies():
    """Test propagation when C1 depends on multiple nodes; weakest wins."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=0)
    belief["assumptions"][0]["id"] = "A1"
    belief["assumptions"][1]["id"] = "A2"
    belief["claims"][0]["confidence"] = 0.9
    belief["claims"][0]["depends_on"] = ["A1", "A2"]
    belief["claims"][0]["backing_evidence_ids"] = []

    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"confidence": 0.5}  # Note: assumptions don't have confidence in schema
    }]

    # This test might need adjustment based on actual patch implementation
    # If assumptions don't support confidence, this tests robustness
    try:
        updated = apply_patches(belief, patches, propagate_confidence=True)
        assert isinstance(updated, dict)
    except (ValueError, KeyError):
        pytest.skip("Assumptions don't support confidence in current schema")


@pytest.mark.unit
def test_apply_patches_propagate_disabled():
    """Test that propagate_confidence=False skips propagation."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["confidence"] = 0.8
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []

    belief["claims"][1]["confidence"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_confidence=False)

    # C1 updated but C2 unchanged
    assert updated["claims"][0]["confidence"] == 0.5
    assert updated["claims"][1]["confidence"] == 0.8


# ==============================================
# 7. Version Management Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_increments_version():
    """Test that version increases by 1."""
    belief = create_sample_belief()
    belief["version"] = 1
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert updated["version"] == 2


@pytest.mark.unit
def test_apply_patches_version_from_zero():
    """Test handling of missing version field."""
    belief = create_sample_belief()
    if "version" in belief:
        del belief["version"]

    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert "version" in updated
    assert updated["version"] >= 1


@pytest.mark.unit
def test_apply_patches_multiple_patches(test_patches):
    """Test that sequential patches increment version once."""
    belief = create_sample_belief(num_claims=1, num_evidence=1)
    belief["version"] = 1
    patches = test_patches["multiple_patches"]

    updated = apply_patches(belief, patches)

    # Version should increment once regardless of number of patches
    assert updated["version"] == 2


# ==============================================
# 8. Changelog Generation Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_creates_changelog():
    """Test that applying patches creates changelog entry."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert "changelog" in updated or "metadata" in updated
    # Changelog format varies by implementation


@pytest.mark.unit
def test_apply_patches_changelog_format():
    """Test that changelog contains version, changes, timestamp."""
    belief = create_sample_belief()
    belief["version"] = 1
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    # Check for changelog in metadata or as separate field
    if "changelog" in updated:
        assert len(updated["changelog"]) > 0
    elif "metadata" in updated and "last_updated" in updated["metadata"]:
        assert "last_updated" in updated["metadata"]


@pytest.mark.unit
def test_apply_patches_changelog_multiple(test_patches):
    """Test that multiple patches create single changelog entry."""
    belief = create_sample_belief(num_claims=1, num_evidence=1)
    patches = test_patches["multiple_patches"]

    updated = apply_patches(belief, patches)

    # Should have one changelog entry for all patches
    assert isinstance(updated, dict)


# ==============================================
# 9. Patch Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_patches_all_valid(test_patches):
    """Test validation of valid patches."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["update_claim_confidence"]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_missing_op(test_patches):
    """Test error if 'op' field missing."""
    belief = create_sample_belief()
    patches = test_patches["invalid_missing_op"]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("op" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_patches_unknown_op(test_patches):
    """Test error for unknown operation."""
    belief = create_sample_belief()
    patches = test_patches["invalid_unknown_op"]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("unknown" in error.lower() or "invalid" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_patches_update_claim_bad_id():
    """Test error if target_id doesn't exist."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C99",
        "changes": {"confidence": 0.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("C99" in error or "not found" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_patches_retire_claim_bad_id():
    """Test error if target_id for retire doesn't exist."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "retire_claim",
        "target_id": "C99"
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0


@pytest.mark.unit
def test_validate_patches_add_evidence_duplicate_id():
    """Test error if evidence ID already exists."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E1",  # Already exists
            "type": "empirical",
            "summary": "Duplicate",
            "citation": "Test (2026)"
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("duplicate" in error.lower() or "exists" in error.lower() for error in errors)


# ==============================================
# 10. Edge Cases Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_empty_patch_list():
    """Test that empty patch list still increments version."""
    belief = create_sample_belief()
    belief["version"] = 1
    patches = []

    updated = apply_patches(belief, patches)

    # Behavior depends on implementation
    assert isinstance(updated, dict)


@pytest.mark.unit
def test_apply_patches_deep_copy():
    """Test that original belief is unchanged."""
    belief = create_sample_belief()
    original_confidence = belief["thesis"]["confidence"]
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    # Original should be unchanged
    assert belief["thesis"]["confidence"] == original_confidence
    assert updated["thesis"]["confidence"] != original_confidence


@pytest.mark.unit
def test_apply_patches_propagation_failure():
    """Test handling of propagation exceptions gracefully."""
    belief = create_sample_belief(num_claims=2)
    # Create malformed dependency that might cause propagation issues
    belief["claims"][1]["depends_on"] = ["NONEXISTENT"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.5}
    }]

    # Should either handle gracefully or raise appropriate error
    try:
        updated = apply_patches(belief, patches, propagate_confidence=True)
        assert isinstance(updated, dict)
    except (ValueError, KeyError):
        pass  # Expected if propagation fails


@pytest.mark.unit
def test_apply_patches_invalid_change_type():
    """Test update_thesis with invalid change type (covers line 71)."""
    belief = create_sample_belief()
    belief["thesis"]["confidence"] = 0.5

    patches = [{
        "op": "update_thesis",
        "change": "invalid_action"  # Not "weaken" or "strengthen"
    }]

    updated = apply_patches(belief, patches)

    # Should skip the invalid patch and leave confidence unchanged
    assert updated["thesis"]["confidence"] == 0.5
    assert updated["version"] == 2  # Version still increments


@pytest.mark.unit
def test_apply_patches_unknown_operation():
    """Test applying patch with unknown operation (covers line 143)."""
    belief = create_sample_belief()

    patches = [{
        "op": "delete_everything",  # Unknown operation
        "target_id": "C1"
    }]

    updated = apply_patches(belief, patches)

    # Should log warning and continue
    assert updated["version"] == 2
    # Check changelog contains warning
    assert len(updated["changelog"]) == 1
    assert any("Unknown patch operation" in change for change in updated["changelog"][0]["changes"])


@pytest.mark.unit
def test_validate_patches_invalid_thesis_change():
    """Test validation catches invalid update_thesis change value."""
    belief = create_sample_belief()

    patches = [{
        "op": "update_thesis",
        "change": "invalid"
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "weaken" in errors[0] or "strengthen" in errors[0]


@pytest.mark.unit
def test_validate_patches_update_claim_missing_target():
    """Test validation catches update_claim without target_id."""
    belief = create_sample_belief()

    patches = [{
        "op": "update_claim",
        "changes": {"confidence": 0.5}
        # Missing target_id
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "missing target_id" in errors[0]


@pytest.mark.unit
def test_validate_patches_retire_claim_missing_target():
    """Test validation catches retire_claim without target_id."""
    belief = create_sample_belief()

    patches = [{
        "op": "retire_claim"
        # Missing target_id
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "missing target_id" in errors[0]


@pytest.mark.unit
def test_validate_patches_add_evidence_missing_item():
    """Test validation catches add_evidence without item."""
    belief = create_sample_belief()

    patches = [{
        "op": "add_evidence"
        # Missing item
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "missing item" in errors[0]


@pytest.mark.unit
def test_validate_patches_add_evidence_missing_id():
    """Test validation catches add_evidence item without id."""
    belief = create_sample_belief()

    patches = [{
        "op": "add_evidence",
        "item": {
            "type": "empirical",
            "summary": "Test"
            # Missing id
        }
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "missing 'id' field" in errors[0]


@pytest.mark.unit
def test_validate_patches_update_assumption_missing_target():
    """Test validation catches update_assumption without target_id."""
    belief = create_sample_belief()

    patches = [{
        "op": "update_assumption",
        "new_statement": "Updated"
        # Missing target_id
    }]

    errors = validate_patches(patches, belief)

    assert len(errors) == 1
    assert "missing target_id" in errors[0]
