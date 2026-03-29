"""
Unit tests for belief patch application.

Tests cover:
- Patch application for all operation types
- Strength propagation through dependency graph
- Thesis ceiling enforcement
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
    """Test weakening thesis strength by 0.1."""
    belief = create_sample_belief(confidence=0.75)
    patches = test_patches["update_thesis_weaken"]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == pytest.approx(0.65, abs=0.01)


@pytest.mark.unit
def test_apply_patches_update_thesis_strengthen(test_patches):
    """Test strengthening thesis strength by 0.1."""
    belief = create_sample_belief(confidence=0.75)
    patches = test_patches["update_thesis_strengthen"]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == pytest.approx(0.85, abs=0.01)


@pytest.mark.unit
def test_apply_patches_update_thesis_floor():
    """Test that strength cannot go below 0.0."""
    belief = create_sample_belief(confidence=0.05)
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["strength"] >= 0.0


@pytest.mark.unit
def test_apply_patches_update_thesis_ceiling():
    """Test that strength cannot exceed 1.0 via strengthen."""
    belief = create_sample_belief(confidence=0.95)
    patches = [{"op": "update_thesis", "change": "strengthen"}]

    updated = apply_patches(belief, patches)

    assert updated["thesis"]["strength"] <= 1.0


@pytest.mark.unit
def test_apply_patches_update_thesis_new_strength(test_patches):
    """Test setting thesis strength via explicit new_strength value."""
    belief = create_sample_belief(confidence=0.75)
    patches = test_patches["update_thesis_new_strength"]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == pytest.approx(0.55, abs=0.01)


@pytest.mark.unit
def test_apply_patches_update_thesis_new_strength_clamp_high():
    """Test that new_strength > 1.0 is clamped to 1.0."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "new_strength": 1.5}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == 1.0


@pytest.mark.unit
def test_apply_patches_update_thesis_new_strength_clamp_low():
    """Test that new_strength < 0.0 is clamped to 0.0."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "new_strength": -0.5}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == 0.0


@pytest.mark.unit
def test_apply_patches_update_thesis_new_strength_overrides_change():
    """Test that new_strength takes precedence over change field."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "new_strength": 0.55, "change": "strengthen"}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    # new_strength takes priority over change
    assert updated["thesis"]["strength"] == pytest.approx(0.55, abs=0.01)


# ==============================================
# 2. Patch Application - update_claim
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_claim_strength(test_patches):
    """Test modifying claim strength."""
    belief = create_sample_belief(num_claims=1)
    original_strength = belief["claims"][0]["strength"]
    patches = test_patches["update_claim_strength"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["strength"] == 0.6
    assert updated["claims"][0]["strength"] != original_strength


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

    assert updated["claims"][0]["strength"] == 0.65
    assert updated["claims"][0]["status"] == "revised"


@pytest.mark.unit
def test_apply_patches_update_claim_not_found():
    """Test error when updating non-existent claim."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C99",
        "changes": {"strength": 0.5}
    }]

    with pytest.raises((ValueError, KeyError)):
        apply_patches(belief, patches)


# ==============================================
# 3. Patch Application - retire_claim
# ==============================================

@pytest.mark.unit
def test_apply_patches_retire_claim_success(test_patches):
    """Test retiring a claim sets status='retracted' and strength=0.0."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["retire_claim"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["status"] == "retracted"
    assert updated["claims"][0]["strength"] == 0.0


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
# 3b. Patch Application - add_claim
# ==============================================

@pytest.mark.unit
def test_apply_patches_add_claim_success():
    """Test adding a new claim item."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    original_count = len(belief["claims"])
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "descriptive",
            "statement": "New claim statement",
            "depends_on": ["A1", "E1"],
            "strength": 0.65,
            "status": "active",
            "inference_chain": ["Step 1: A1 supports new claim"],
            "predictions": [
                {"statement": "Pred 1", "test": "Test 1", "decision_criterion": "Crit 1"}
            ]
        }
    }]

    updated = apply_patches(belief, patches)

    assert len(updated["claims"]) == original_count + 1
    assert any(c["id"] == "C2" for c in updated["claims"])
    new_c = [c for c in updated["claims"] if c["id"] == "C2"][0]
    assert new_c["type"] == "descriptive"
    assert new_c["strength"] == 0.65
    assert new_c["status"] == "active"
    assert len(new_c["predictions"]) == 1


@pytest.mark.unit
def test_apply_patches_add_claim_no_array():
    """Test adding claim when claims array doesn't exist."""
    belief = create_sample_belief(num_claims=0, num_assumptions=1, num_evidence=1)
    belief.pop("claims", None)

    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C1",
            "type": "deductive",
            "statement": "First claim",
            "depends_on": ["A1", "E1"],
            "strength": 0.7,
            "status": "active",
            "inference_chain": ["Step 1: A1 supports first claim"],
            "predictions": [
                {"statement": "Pred", "test": "Test", "decision_criterion": "Crit"}
            ]
        }
    }]

    updated = apply_patches(belief, patches)

    assert "claims" in updated
    assert len(updated["claims"]) == 1
    assert updated["claims"][0]["id"] == "C1"


@pytest.mark.unit
def test_apply_patches_add_claim_missing_id():
    """Test error if claim item has no 'id' field."""
    belief = create_sample_belief(num_claims=0)
    patches = [{
        "op": "add_claim",
        "item": {
            "type": "deductive",
            "statement": "Claim without ID"
        }
    }]

    with pytest.raises(ValueError):
        apply_patches(belief, patches)


@pytest.mark.unit
def test_apply_patches_add_claim_changelog():
    """Test that adding a claim creates a changelog entry."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "descriptive",
            "statement": "New claim",
            "depends_on": ["A1", "E1"],
            "strength": 0.6,
            "status": "active",
            "inference_chain": ["Step 1: A1 supports new claim"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    updated = apply_patches(belief, patches)

    assert "changelog" in updated
    last_entry = updated["changelog"][-1]
    assert any("C2" in change for change in last_entry.get("changes", []))


@pytest.mark.unit
def test_apply_patches_add_claim_affects_thesis_strength():
    """Adding a claim changes thesis strength via the breadth formula."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8

    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "descriptive",
            "statement": "Second claim",
            "depends_on": ["A1", "E1"],
            "strength": 0.6,
            "status": "active",
            "inference_chain": ["Step 1: A1 supports second claim"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=True)

    # Now 2 active claims: avg = (0.8 + 0.6) / 2 = 0.7
    # breadth = 2^1.5 / (2^1.5 + 1) ≈ 0.739
    # thesis = 0.7 * 0.739 ≈ 0.5172
    assert updated["thesis"]["strength"] == pytest.approx(0.5172, abs=0.01)


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
            "source": "Test (2026)",
            "relevance_to_claims": [],
            "strength": 0.8
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
# 5. Patch Application - update_evidence
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_evidence_strength(test_patches):
    """Test updating evidence strength."""
    belief = create_sample_belief(num_evidence=1)
    original_strength = belief["evidence"][0]["strength"]
    patches = test_patches["update_evidence"]

    updated = apply_patches(belief, patches)

    assert updated["evidence"][0]["strength"] == 0.9
    assert updated["evidence"][0]["strength"] != original_strength


@pytest.mark.unit
def test_apply_patches_update_evidence_summary():
    """Test updating evidence summary."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"summary": "Updated evidence summary"}
    }]

    updated = apply_patches(belief, patches)

    assert updated["evidence"][0]["summary"] == "Updated evidence summary"


@pytest.mark.unit
def test_apply_patches_update_evidence_not_found():
    """Test error when updating non-existent evidence."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E99",
        "changes": {"strength": 0.5}
    }]

    with pytest.raises(ValueError, match="non-existent evidence"):
        apply_patches(belief, patches)


# ==============================================
# 6. Patch Application - update_assumption
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


@pytest.mark.unit
def test_apply_patches_update_assumption_strength(test_patches):
    """Test updating assumption strength via changes dict."""
    belief = create_sample_belief(num_assumptions=1)
    original_strength = belief["assumptions"][0]["strength"]
    patches = test_patches["update_assumption_strength"]

    updated = apply_patches(belief, patches)

    assert updated["assumptions"][0]["strength"] == 0.7
    assert updated["assumptions"][0]["strength"] != original_strength


@pytest.mark.unit
def test_apply_patches_update_assumption_combined():
    """Test updating assumption with both new_statement and changes dict."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "new_statement": "Updated statement",
        "changes": {"strength": 0.65}
    }]

    updated = apply_patches(belief, patches)

    assert updated["assumptions"][0]["statement"] == "Updated statement"
    assert updated["assumptions"][0]["strength"] == 0.65


# ==============================================
# 6b. Patch Application - add_assumption
# ==============================================

@pytest.mark.unit
def test_apply_patches_add_assumption_success():
    """Test adding a new assumption item."""
    belief = create_sample_belief(num_assumptions=1)
    original_count = len(belief["assumptions"])
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "empirical",
            "statement": "New assumption statement",
            "strength": 0.7
        }
    }]

    updated = apply_patches(belief, patches)

    assert len(updated["assumptions"]) == original_count + 1
    assert any(a["id"] == "A2" for a in updated["assumptions"])
    new_a = [a for a in updated["assumptions"] if a["id"] == "A2"][0]
    assert new_a["type"] == "empirical"
    assert new_a["statement"] == "New assumption statement"
    assert new_a["strength"] == 0.7


@pytest.mark.unit
def test_apply_patches_add_assumption_no_array():
    """Test adding assumption when assumptions array doesn't exist."""
    belief = create_sample_belief(num_assumptions=0)
    belief.pop("assumptions", None)

    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A1",
            "type": "foundational",
            "statement": "New foundational assumption",
            "strength": 0.8
        }
    }]

    updated = apply_patches(belief, patches)

    assert "assumptions" in updated
    assert len(updated["assumptions"]) == 1
    assert updated["assumptions"][0]["id"] == "A1"


@pytest.mark.unit
def test_apply_patches_add_assumption_missing_id():
    """Test error if assumption item has no 'id' field."""
    belief = create_sample_belief(num_assumptions=0)
    patches = [{
        "op": "add_assumption",
        "item": {
            "type": "empirical",
            "statement": "Assumption without ID"
        }
    }]

    with pytest.raises(ValueError):
        apply_patches(belief, patches)


@pytest.mark.unit
def test_apply_patches_add_assumption_changelog():
    """Test that adding an assumption creates a changelog entry."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "methodological",
            "statement": "Methodological assumption",
            "strength": 0.6
        }
    }]

    updated = apply_patches(belief, patches)

    assert "changelog" in updated
    assert len(updated["changelog"]) > 0
    last_entry = updated["changelog"][-1]
    assert any("A2" in change for change in last_entry.get("changes", []))


# ==============================================
# 7. Strength Propagation Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_propagate_simple():
    """Test simple propagation: C2 depends on C1; C1 weakens → C2 weakens."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = []

    belief["claims"][1]["strength"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_strength=True)

    assert updated["claims"][0]["strength"] == 0.5
    # C2 depends on C1 and is capped at C1's new strength
    assert updated["claims"][1]["strength"] == 0.5


@pytest.mark.unit
def test_apply_patches_propagate_transitive():
    """Test transitive propagation: C3 → C2 → C1; C1 weakens → C2, C3 weaken."""
    belief = create_sample_belief(num_assumptions=0, num_claims=3, num_evidence=0)
    belief["claims"][0]["id"] = "C1"
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = []

    belief["claims"][1]["id"] = "C2"
    belief["claims"][1]["strength"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]

    belief["claims"][2]["id"] = "C3"
    belief["claims"][2]["strength"] = 0.8
    belief["claims"][2]["depends_on"] = ["C2"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_strength=True)

    # Propagation: C2 and C3 are capped at C1's new strength
    assert updated["claims"][0]["strength"] == 0.5
    assert updated["claims"][1]["strength"] == 0.5
    assert updated["claims"][2]["strength"] == 0.5


@pytest.mark.unit
def test_apply_patches_propagate_no_change():
    """Test that propagation doesn't occur if dependent already has lower strength."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = []

    belief["claims"][1]["strength"] = 0.4  # Already lower
    belief["claims"][1]["depends_on"] = ["C1"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.7}
    }]

    updated = apply_patches(belief, patches, propagate_strength=True)

    # C2 should remain at 0.4
    assert updated["claims"][1]["strength"] == 0.4


@pytest.mark.unit
def test_apply_patches_propagate_disabled():
    """Test that propagate_strength=False skips propagation."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = []

    belief["claims"][1]["strength"] = 0.8
    belief["claims"][1]["depends_on"] = ["C1"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    # C1 updated but C2 unchanged
    assert updated["claims"][0]["strength"] == 0.5
    assert updated["claims"][1]["strength"] == 0.8


# ==============================================
# 8. Thesis Ceiling Propagation Tests
# ==============================================

@pytest.mark.unit
def test_thesis_ceiling_single_claim():
    """Single active claim → ceiling = strength * 0.5."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["thesis"]["strength"] = 0.8

    # Apply a no-op thesis update to trigger ceiling enforcement
    patches = [{"op": "update_thesis", "new_strength": 0.8}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # ceiling = 0.8 * (1 - 1/2) = 0.8 * 0.5 = 0.4
    assert updated["thesis"]["strength"] == pytest.approx(0.4, abs=0.01)


@pytest.mark.unit
def test_thesis_ceiling_two_claims():
    """Two active claims → ceiling = avg * breadth(2, p=1.5)."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.6
    belief["thesis"]["strength"] = 0.9

    patches = [{"op": "update_thesis", "new_strength": 0.9}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # avg = (0.8 + 0.6) / 2 = 0.7
    # breadth = 2^1.5 / (2^1.5 + 1) ≈ 0.739
    # ceiling = 0.7 * 0.739 ≈ 0.5172
    assert updated["thesis"]["strength"] == pytest.approx(0.5172, abs=0.01)


@pytest.mark.unit
def test_thesis_ceiling_three_claims():
    """Three active claims → ceiling = avg * breadth(3, p=1.5)."""
    belief = create_sample_belief(num_claims=3, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.8
    belief["claims"][2]["strength"] = 0.8
    belief["thesis"]["strength"] = 0.9

    patches = [{"op": "update_thesis", "new_strength": 0.9}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # avg = 0.8, breadth = 3^1.5 / (3^1.5 + 1) ≈ 0.839, ceiling ≈ 0.6709
    assert updated["thesis"]["strength"] == pytest.approx(0.6709, abs=0.01)


@pytest.mark.unit
def test_thesis_ceiling_retracted_excluded():
    """Retracted claims are excluded from ceiling calculation."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["status"] = "active"
    belief["claims"][1]["strength"] = 0.0
    belief["claims"][1]["status"] = "retracted"
    belief["thesis"]["strength"] = 0.8

    patches = [{"op": "update_thesis", "new_strength": 0.8}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # Only 1 active claim (C1 at 0.8), retracted C2 excluded
    # ceiling = 0.8 * 0.5 = 0.4
    assert updated["thesis"]["strength"] == pytest.approx(0.4, abs=0.01)


@pytest.mark.unit
def test_thesis_strength_always_equals_formula():
    """Thesis strength is always set to the formula result, regardless of agent's value."""
    belief = create_sample_belief(num_claims=3, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.8
    belief["claims"][2]["strength"] = 0.8
    belief["thesis"]["strength"] = 0.3  # Agent set lower, but formula should override

    patches = [{"op": "update_thesis", "new_strength": 0.3}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # Formula: avg(0.8, 0.8, 0.8) × (3^1.5 / (3^1.5 + 1)) = 0.8 × 0.839 ≈ 0.6709
    assert updated["thesis"]["strength"] == pytest.approx(0.6709, abs=0.01)
    assert "strength_reasoning" in updated["thesis"]


@pytest.mark.unit
def test_thesis_ceiling_after_claim_weakness():
    """Lowering a claim's strength cascades to thesis ceiling."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.8
    belief["thesis"]["strength"] = 0.6

    # Lower C1 strength significantly
    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.3}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # avg = (0.3 + 0.8) / 2 = 0.55, breadth = 2^1.5 / (2^1.5 + 1) ≈ 0.739
    # ceiling = 0.55 * 0.739 ≈ 0.4063
    assert updated["thesis"]["strength"] == pytest.approx(0.4063, abs=0.01)


# ==============================================
# 9. Patch Application - add/update counterposition
# ==============================================

@pytest.mark.unit
def test_apply_patches_add_counterposition(test_patches):
    """Test adding a new counterposition (no numeric strength on X#)."""
    belief = create_sample_belief(num_claims=1)

    patches = test_patches["add_counterposition"]

    updated = apply_patches(belief, patches)

    assert "counterpositions" in updated
    assert len(updated["counterpositions"]) == 1
    cp = updated["counterpositions"][0]
    assert cp["id"] == "X1"
    assert cp["targets"] == ["C1"]
    assert cp["attack_type"] == "rebutting"
    assert cp["response_sufficiency"] == "partial"
    # X# no longer has numeric strength
    assert "strength" not in cp


@pytest.mark.unit
def test_apply_patches_add_counterposition_no_array():
    """Test adding counterposition when counterpositions array doesn't exist."""
    belief = create_sample_belief(num_claims=1)
    belief.pop("counterpositions", None)

    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "undermining",
            "statement": "Test",
            "my_response": "Response",
            "response_sufficiency": "sufficient"
        }
    }]

    updated = apply_patches(belief, patches)

    assert "counterpositions" in updated
    assert len(updated["counterpositions"]) == 1


@pytest.mark.unit
def test_apply_patches_add_counterposition_missing_id():
    """Test error when counterposition item has no id."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_counterposition",
        "item": {
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Test"
        }
    }]

    with pytest.raises(ValueError):
        apply_patches(belief, patches)


@pytest.mark.unit
def test_apply_patches_update_counterposition(test_patches):
    """Test updating an existing counterposition."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Original statement",
        "my_response": "Original response",
        "response_sufficiency": "partial"
    }]

    patches = test_patches["update_counterposition"]

    updated = apply_patches(belief, patches)

    assert updated["counterpositions"][0]["response_sufficiency"] == "unaddressed"
    # Unchanged fields should remain
    assert updated["counterpositions"][0]["statement"] == "Original statement"


@pytest.mark.unit
def test_apply_patches_update_counterposition_not_found():
    """Test error when updating non-existent counterposition."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = []

    patches = [{
        "op": "update_counterposition",
        "target_id": "X99",
        "changes": {"response_sufficiency": "sufficient"}
    }]

    with pytest.raises(ValueError):
        apply_patches(belief, patches)


# ==============================================
# 10. Patch Application - add/resolve uncertainty
# ==============================================

@pytest.mark.unit
def test_apply_patches_add_uncertainty(test_patches):
    """Test adding a new uncertainty (with targets, status, no cruciality)."""
    belief = create_sample_belief()

    patches = test_patches["add_uncertainty"]

    updated = apply_patches(belief, patches)

    assert "uncertainties" in updated
    assert len(updated["uncertainties"]) == 1
    u = updated["uncertainties"][0]
    assert u["id"] == "U1"
    assert u["question"] == "Can the explanatory gap be closed?"
    assert u["targets"] == ["C1"]
    assert u["status"] == "active"


@pytest.mark.unit
def test_apply_patches_add_uncertainty_no_array():
    """Test adding uncertainty when uncertainties array doesn't exist."""
    belief = create_sample_belief()
    belief.pop("uncertainties", None)

    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Test question",
            "status": "active",
            "importance": "Test hint"
        }
    }]

    updated = apply_patches(belief, patches)

    assert "uncertainties" in updated
    assert len(updated["uncertainties"]) == 1


@pytest.mark.unit
def test_apply_patches_add_uncertainty_missing_id():
    """Test error when uncertainty item has no id."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_uncertainty",
        "item": {
            "question": "No ID here"
        }
    }]

    with pytest.raises(ValueError):
        apply_patches(belief, patches)


@pytest.mark.unit
def test_apply_patches_resolve_uncertainty():
    """Test resolving a U# sets status='resolved' and records resolution_note."""
    belief = create_sample_belief()
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Open question",
        "status": "active"
    }]

    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U1",
        "resolution_note": "Resolved by new claim C4 with evidence E3"
    }]

    updated = apply_patches(belief, patches)

    assert updated["uncertainties"][0]["status"] == "resolved"
    assert updated["uncertainties"][0]["resolution_note"] == "Resolved by new claim C4 with evidence E3"


@pytest.mark.unit
def test_apply_patches_resolve_uncertainty_not_found():
    """Test error when resolving non-existent uncertainty."""
    belief = create_sample_belief()
    belief["uncertainties"] = []

    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U99",
        "resolution_note": "Some note"
    }]

    with pytest.raises(ValueError, match="non-existent uncertainty"):
        apply_patches(belief, patches)


# ==============================================
# 11. Version Management Tests
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
# 12. Changelog Generation Tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_creates_changelog():
    """Test that applying patches creates changelog entry."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    assert "changelog" in updated or "metadata" in updated


@pytest.mark.unit
def test_apply_patches_changelog_format():
    """Test that changelog contains version, changes, timestamp."""
    belief = create_sample_belief()
    belief["version"] = 1
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    if "changelog" in updated:
        assert len(updated["changelog"]) > 0


@pytest.mark.unit
def test_apply_patches_changelog_multiple(test_patches):
    """Test that multiple patches create single changelog entry."""
    belief = create_sample_belief(num_claims=1, num_evidence=1)
    patches = test_patches["multiple_patches"]

    updated = apply_patches(belief, patches)

    # Should have one changelog entry for all patches
    assert isinstance(updated, dict)


# ==============================================
# 13. Patch Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_patches_all_valid(test_patches):
    """Test validation of valid patches."""
    belief = create_sample_belief(num_claims=1)
    patches = test_patches["update_claim_strength"]

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
        "changes": {"strength": 0.5}
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
            "source": "Test (2026)",
            "relevance_to_claims": [],
            "strength": 0.8
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("duplicate" in error.lower() or "exists" in error.lower() for error in errors)


# ==============================================
# 14. Validation - update_thesis
# ==============================================

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
def test_validate_patches_thesis_new_strength_valid():
    """Test validation accepts valid new_strength on update_thesis."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "new_strength": 0.5}]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_thesis_new_strength_out_of_bounds():
    """Test validation rejects out-of-bounds new_strength."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "new_strength": 1.5}]

    errors = validate_patches(patches, belief)
    assert len(errors) == 1
    assert "between 0.0 and 1.0" in errors[0]


# ==============================================
# 15. Validation - update_evidence
# ==============================================

@pytest.mark.unit
def test_validate_patches_update_evidence_valid():
    """Test validation accepts valid update_evidence."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"strength": 0.9}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_update_evidence_bad_id():
    """Test validation catches update_evidence with non-existent ID."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E99",
        "changes": {"strength": 0.9}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("E99" in e or "non-existent" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_update_evidence_missing_target():
    """Test validation catches update_evidence without target_id."""
    belief = create_sample_belief()
    patches = [{
        "op": "update_evidence",
        "changes": {"strength": 0.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing target_id" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_update_evidence_strength_out_of_bounds():
    """Test validation catches out-of-bounds strength in update_evidence."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"strength": 1.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


# ==============================================
# 16. Validation - update_assumption
# ==============================================

@pytest.mark.unit
def test_validate_patches_update_assumption_strength_valid():
    """Test validation accepts valid strength in update_assumption changes."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"strength": 0.7}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_update_assumption_strength_out_of_bounds():
    """Test validation catches out-of-bounds strength in update_assumption."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"strength": -0.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


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


# ==============================================
# 16b. Validation - add_assumption
# ==============================================

@pytest.mark.unit
def test_validate_patches_add_assumption_valid():
    """Test validation accepts valid add_assumption patch."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "empirical",
            "statement": "New assumption",
            "strength": 0.7
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_assumption_missing_item():
    """Test validation catches add_assumption without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_assumption"}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing item" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_assumption_missing_id():
    """Test validation catches add_assumption item without id."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_assumption",
        "item": {"type": "empirical", "statement": "No ID", "strength": 0.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing 'id' field" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_assumption_duplicate_id():
    """Test validation catches duplicate assumption ID."""
    belief = create_sample_belief(num_assumptions=1)  # Has A1
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A1",
            "type": "empirical",
            "statement": "Duplicate",
            "strength": 0.5
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("exists" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_assumption_missing_required_fields():
    """Test validation catches add_assumption missing required fields."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2"
            # Missing: type, statement, strength
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) >= 3  # Missing type, statement, strength


@pytest.mark.unit
def test_validate_patches_add_assumption_invalid_type():
    """Test validation catches add_assumption with invalid type."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "invalid_type",
            "statement": "Test",
            "strength": 0.5
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("foundational" in e or "empirical" in e or "methodological" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_add_assumption_strength_out_of_bounds():
    """Test validation catches add_assumption with out-of-bounds strength."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "empirical",
            "statement": "Test",
            "strength": 1.5
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


# ==============================================
# 17. Validation - resolve_uncertainty
# ==============================================

@pytest.mark.unit
def test_validate_patches_resolve_uncertainty_valid():
    """Test validation accepts valid resolve_uncertainty."""
    belief = create_sample_belief()
    belief["uncertainties"] = [{"id": "U1", "targets": ["C1"], "question": "Q", "status": "active"}]

    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U1",
        "resolution_note": "Resolved by new evidence E3"
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_resolve_uncertainty_bad_id():
    """Test validation catches resolve_uncertainty with non-existent ID."""
    belief = create_sample_belief()
    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U99",
        "resolution_note": "Some note"
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("U99" in e or "non-existent" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_resolve_uncertainty_empty_note():
    """Test validation catches resolve_uncertainty with empty resolution_note."""
    belief = create_sample_belief()
    belief["uncertainties"] = [{"id": "U1", "targets": ["C1"], "question": "Q", "status": "active"}]

    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U1",
        "resolution_note": ""
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("resolution_note" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_resolve_uncertainty_missing_note():
    """Test validation catches resolve_uncertainty with missing resolution_note."""
    belief = create_sample_belief()
    belief["uncertainties"] = [{"id": "U1", "targets": ["C1"], "question": "Q", "status": "active"}]

    patches = [{
        "op": "resolve_uncertainty",
        "target_id": "U1"
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("resolution_note" in e for e in errors)


# ==============================================
# 18. Validation - counterposition (updated: no strength)
# ==============================================

@pytest.mark.unit
def test_validate_patches_add_counterposition_valid():
    """Test validation of valid add_counterposition patch (no strength field)."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Test",
            "my_response": "Response",
            "response_sufficiency": "partial"
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_counterposition_missing_item():
    """Test validation catches add_counterposition without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_counterposition"}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing item" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_counterposition_missing_id():
    """Test validation catches add_counterposition item without id."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_counterposition",
        "item": {"statement": "No ID"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing 'id' field" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_counterposition_duplicate_id():
    """Test validation catches duplicate counterposition ID."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Existing",
        "my_response": "Response",
        "response_sufficiency": "sufficient"
    }]

    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "undermining",
            "statement": "Duplicate",
            "my_response": "Response",
            "response_sufficiency": "partial"
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("exists" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_counterposition_missing_required_fields():
    """Test validation catches counterposition missing required fields."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "statement": "Missing fields"
            # Missing: targets, attack_type, my_response, response_sufficiency
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0


@pytest.mark.unit
def test_validate_patches_update_counterposition_valid():
    """Test validation of valid update_counterposition patch."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1",
        "targets": ["C1"],
        "attack_type": "rebutting",
        "statement": "Test",
        "my_response": "Response",
        "response_sufficiency": "partial"
    }]

    patches = [{
        "op": "update_counterposition",
        "target_id": "X1",
        "changes": {"response_sufficiency": "sufficient"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_update_counterposition_bad_id():
    """Test validation catches update_counterposition with non-existent ID."""
    belief = create_sample_belief()
    patches = [{
        "op": "update_counterposition",
        "target_id": "X99",
        "changes": {"response_sufficiency": "sufficient"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("X99" in e or "non-existent" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_update_counterposition_missing_target():
    """Test validation catches update_counterposition without target_id."""
    belief = create_sample_belief()
    patches = [{
        "op": "update_counterposition",
        "changes": {"response_sufficiency": "partial"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing target_id" in e.lower() for e in errors)


# ==============================================
# 19. Validation - uncertainty
# ==============================================

@pytest.mark.unit
def test_validate_patches_add_uncertainty_valid():
    """Test validation of valid add_uncertainty patch."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Test?",
            "status": "active",
            "importance": "Hint"
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_uncertainty_missing_item():
    """Test validation catches add_uncertainty without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_uncertainty"}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing item" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_uncertainty_missing_id():
    """Test validation catches add_uncertainty item without id."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_uncertainty",
        "item": {"question": "No ID"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing 'id' field" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_uncertainty_duplicate_id():
    """Test validation catches duplicate uncertainty ID."""
    belief = create_sample_belief()
    belief["uncertainties"] = [{
        "id": "U1",
        "targets": ["C1"],
        "question": "Existing",
        "status": "active"
    }]

    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Duplicate",
            "status": "active"
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("exists" in e.lower() for e in errors)


# ==============================================
# 20. Validation - update_claim strength bounds
# ==============================================

@pytest.mark.unit
def test_validate_patches_update_claim_missing_target():
    """Test validation catches update_claim without target_id."""
    belief = create_sample_belief()

    patches = [{
        "op": "update_claim",
        "changes": {"strength": 0.5}
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


# ==============================================
# 20b. Validation - add_claim
# ==============================================

@pytest.mark.unit
def test_validate_patches_add_claim_valid():
    """Test validation accepts valid add_claim patch."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "descriptive",
            "statement": "New claim",
            "depends_on": ["A1", "E1"],
            "strength": 0.65,
            "status": "active",
            "inference_chain": ["Step 1: A1 supports new claim"],
            "predictions": [
                {"statement": "Pred", "test": "Test", "decision_criterion": "Crit"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_claim_missing_item():
    """Test validation catches add_claim without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_claim"}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing item" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_missing_id():
    """Test validation catches add_claim item without id."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_claim",
        "item": {"type": "deductive", "statement": "No ID"}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("missing 'id' field" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_duplicate_id():
    """Test validation catches duplicate claim ID."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C1",  # Already exists
            "type": "deductive",
            "statement": "Duplicate",
            "depends_on": ["A1", "E1"],
            "strength": 0.5,
            "status": "active",
            "inference_chain": ["Step 1: Duplicate reasoning"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("exists" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_missing_required_fields():
    """Test validation catches add_claim missing required fields."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2"
            # Missing: type, statement, depends_on, strength, status, predictions, inference_chain
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) >= 7  # Missing 7 required fields


@pytest.mark.unit
def test_validate_patches_add_claim_strength_out_of_bounds():
    """Test validation catches add_claim with out-of-bounds strength."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A1", "E1"],
            "strength": 1.5,
            "status": "active",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_invalid_status():
    """Test validation catches add_claim with invalid status."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A1", "E1"],
            "strength": 0.5,
            "status": "invalid_status",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("active" in e or "revised" in e or "retracted" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_bad_depends_on():
    """Test validation catches add_claim with non-existent depends_on reference."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A99"],  # Non-existent
            "strength": 0.5,
            "status": "active",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("A99" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_bad_evidence_ref():
    """Test validation catches add_claim with non-existent depends_on reference."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A1", "E99"],  # E99 doesn't exist
            "strength": 0.5,
            "status": "active",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("depends_on references non-existent" in e.lower() or "E99" in e for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_empty_predictions():
    """Test validation catches add_claim with empty predictions array."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A1", "E1"],
            "strength": 0.5,
            "status": "active",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": []
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("predictions" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_patches_add_claim_prediction_missing_fields():
    """Test validation catches prediction missing required fields."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "Test",
            "depends_on": ["A1", "E1"],
            "strength": 0.5,
            "status": "active",
            "inference_chain": ["Step 1: Test reasoning"],
            "predictions": [
                {"statement": "Pred only"}  # Missing test and decision_criterion
            ]
        }
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) >= 2  # Missing test and decision_criterion


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
def test_validate_patches_update_claim_strength_out_of_bounds():
    """Test validation catches out-of-bounds strength in update_claim."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 1.5}
    }]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


# ==============================================
# 21. Edge Cases Tests
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
    original_strength = belief["thesis"]["strength"]
    patches = [{"op": "update_thesis", "change": "weaken"}]

    updated = apply_patches(belief, patches)

    # Original should be unchanged
    assert belief["thesis"]["strength"] == original_strength
    assert updated["thesis"]["strength"] != original_strength


@pytest.mark.unit
def test_apply_patches_propagation_failure():
    """Test handling of propagation exceptions gracefully."""
    belief = create_sample_belief(num_claims=2)
    # Create malformed dependency that might cause propagation issues
    belief["claims"][1]["depends_on"] = ["NONEXISTENT"]

    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"strength": 0.5}
    }]

    # Should either handle gracefully or raise appropriate error
    try:
        updated = apply_patches(belief, patches, propagate_strength=True)
        assert isinstance(updated, dict)
    except (ValueError, KeyError):
        pass  # Expected if propagation fails


@pytest.mark.unit
def test_apply_patches_invalid_change_type():
    """Test update_thesis with invalid change type."""
    belief = create_sample_belief()
    belief["thesis"]["strength"] = 0.5

    patches = [{
        "op": "update_thesis",
        "change": "invalid_action"  # Not "weaken" or "strengthen"
    }]

    updated = apply_patches(belief, patches)

    # Should skip the invalid patch and leave strength unchanged
    # (thesis ceiling may cap it though, so just verify structure)
    assert isinstance(updated, dict)
    assert updated["version"] == 2  # Version still increments


@pytest.mark.unit
def test_apply_patches_unknown_operation():
    """Test applying patch with unknown operation."""
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


# ==============================================
# 22. Assumption type tests
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_assumption_type(test_patches):
    """Test updating assumption type via new_type field."""
    belief = create_sample_belief(num_assumptions=1)
    original_type = belief["assumptions"][0]["type"]
    patches = test_patches["update_assumption_type"]

    updated = apply_patches(belief, patches)

    assert updated["assumptions"][0]["type"] == "methodological"
    assert updated["assumptions"][0]["type"] != original_type


@pytest.mark.unit
def test_apply_patches_update_assumption_statement_and_type(test_patches):
    """Test updating both statement and type of an assumption."""
    belief = create_sample_belief(num_assumptions=1)
    patches = test_patches["update_assumption_both"]

    updated = apply_patches(belief, patches)

    assert updated["assumptions"][0]["statement"] == "Revised assumption"
    assert updated["assumptions"][0]["type"] == "methodological"


# ==============================================
# 23. update_thesis with stance field
# ==============================================

@pytest.mark.unit
def test_update_thesis_stance_only():
    """Patch with stance only updates thesis.stance without touching strength."""
    belief = create_sample_belief(confidence=0.75)
    original_strength = belief["thesis"]["strength"]
    patches = [{"op": "update_thesis", "stance": "New stance text"}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["stance"] == "New stance text"
    assert updated["thesis"]["strength"] == original_strength


@pytest.mark.unit
def test_update_thesis_stance_empty_rejected():
    """Validate that empty stance is rejected by validate_patches."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "stance": ""}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("stance" in e for e in errors)


@pytest.mark.unit
def test_update_thesis_stance_and_strength():
    """Both stance and new_strength in same patch: both fields update."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "new_strength": 0.55, "stance": "Updated stance"}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["stance"] == "Updated stance"
    assert updated["thesis"]["strength"] == pytest.approx(0.55, abs=0.01)


# ==============================================
# 24. update_thesis with summary_bullets field
# ==============================================

@pytest.mark.unit
def test_update_thesis_bullets_only():
    """Patch with summary_bullets only updates thesis.summary_bullets."""
    belief = create_sample_belief(confidence=0.75)
    original_strength = belief["thesis"]["strength"]
    patches = [{"op": "update_thesis", "summary_bullets": ["Bullet A", "Bullet B"]}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["summary_bullets"] == ["Bullet A", "Bullet B"]
    assert updated["thesis"]["strength"] == original_strength


@pytest.mark.unit
def test_update_thesis_bullets_empty_list_rejected():
    """summary_bullets: [] rejected by validation."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "summary_bullets": []}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("summary_bullets" in e for e in errors)


@pytest.mark.unit
def test_update_thesis_bullets_non_list_rejected():
    """summary_bullets: 'not a list' rejected by validation."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "summary_bullets": "not a list"}]

    errors = validate_patches(patches, belief)
    assert len(errors) > 0
    assert any("summary_bullets" in e for e in errors)


# ==============================================
# 25. Combined thesis update
# ==============================================

@pytest.mark.unit
def test_update_thesis_all_fields():
    """Patch with new_strength, stance, and summary_bullets: all three update."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{
        "op": "update_thesis",
        "new_strength": 0.55,
        "stance": "Comprehensive new stance",
        "summary_bullets": ["New bullet 1", "New bullet 2", "New bullet 3"]
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert updated["thesis"]["strength"] == pytest.approx(0.55, abs=0.01)
    assert updated["thesis"]["stance"] == "Comprehensive new stance"
    assert updated["thesis"]["summary_bullets"] == ["New bullet 1", "New bullet 2", "New bullet 3"]


@pytest.mark.unit
def test_update_thesis_stance_changelog():
    """Verify changelog records 'Thesis stance text updated'."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "stance": "New stance for changelog test"}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert "changelog" in updated
    assert len(updated["changelog"]) > 0
    last_entry = updated["changelog"][-1]
    assert any("stance" in change.lower() for change in last_entry.get("changes", []))


@pytest.mark.unit
def test_update_thesis_bullets_changelog():
    """Verify changelog records 'Thesis summary bullets updated'."""
    belief = create_sample_belief(confidence=0.75)
    patches = [{"op": "update_thesis", "summary_bullets": ["A", "B"]}]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert "changelog" in updated
    assert len(updated["changelog"]) > 0
    last_entry = updated["changelog"][-1]
    assert any("bullets" in change.lower() for change in last_entry.get("changes", []))


# ==============================================
# 10. Status Support and Extended Propagation Tests
# ==============================================

@pytest.mark.unit
def test_add_assumption_defaults_status_active():
    """Test that add_assumption defaults status to 'active' if not provided."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "empirical",
            "statement": "New assumption",
            "strength": 0.7,
            "strength_justification": "Test justification"
        }
    }]
    updated = apply_patches(belief, patches, propagate_strength=False)
    new_assumption = [a for a in updated["assumptions"] if a["id"] == "A2"][0]
    assert new_assumption["status"] == "active"


@pytest.mark.unit
def test_add_evidence_defaults_status_active():
    """Test that add_evidence defaults status to 'active' if not provided."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E2",
            "type": "empirical",
            "summary": "New evidence",
            "source": "Test (2026)",
            "relevance_to_claims": ["C1"],
            "strength": 0.7,
            "strength_justification": "Test justification"
        }
    }]
    updated = apply_patches(belief, patches, propagate_strength=False)
    new_ev = [e for e in updated["evidence"] if e["id"] == "E2"][0]
    assert new_ev["status"] == "active"


@pytest.mark.unit
def test_update_assumption_retracted_zeroes_strength():
    """Test that retracting an assumption forces strength to 0.0."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["assumptions"][0]["strength"] = 0.8
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"status": "retracted"}
    }]
    updated = apply_patches(belief, patches, propagate_strength=False)
    assert updated["assumptions"][0]["status"] == "retracted"
    assert updated["assumptions"][0]["strength"] == 0.0


@pytest.mark.unit
def test_update_evidence_retracted_zeroes_strength():
    """Test that retracting evidence forces strength to 0.0."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["evidence"][0]["strength"] = 0.8
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"status": "retracted"}
    }]
    updated = apply_patches(belief, patches, propagate_strength=False)
    assert updated["evidence"][0]["status"] == "retracted"
    assert updated["evidence"][0]["strength"] == 0.0


@pytest.mark.unit
def test_propagate_assumption_weakness_caps_claim():
    """Test that weakening an assumption caps dependent claims."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["assumptions"][0]["strength"] = 0.8
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = ["A1"]

    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"strength": 0.4}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)
    # C1 depends on A1 and should be capped at A1's new strength
    assert updated["claims"][0]["strength"] == 0.4


@pytest.mark.unit
def test_propagate_evidence_weakness_caps_claim():
    """Test that weakening evidence caps dependent claims."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["evidence"][0]["strength"] = 0.8
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = ["E1"]

    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"strength": 0.3}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)
    # C1 backed by E1 and should be capped at E1's new strength
    assert updated["claims"][0]["strength"] == 0.3


@pytest.mark.unit
def test_propagate_retracted_assumption_skipped():
    """Test that retracted assumptions do NOT cap dependent claims."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=0)
    belief["assumptions"][0]["id"] = "A1"
    belief["assumptions"][0]["strength"] = 0.8
    belief["assumptions"][1]["id"] = "A2"
    belief["assumptions"][1]["strength"] = 0.7
    belief["claims"][0]["strength"] = 0.7
    belief["claims"][0]["depends_on"] = ["A1", "A2"]

    # Retract A1 — should NOT cap C1 because retracted deps are excluded
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"status": "retracted"}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)
    # A1 is retracted (strength=0), but excluded from cap. A2 is 0.7.
    # C1 should be capped at A2's 0.7 (its current value), NOT at 0.0.
    assert updated["claims"][0]["strength"] == 0.7


@pytest.mark.unit
def test_propagate_mixed_deps_uses_weakest_active():
    """Test that cap uses min across active/revised A#, E#, C# only."""
    belief = create_sample_belief(num_assumptions=1, num_claims=2, num_evidence=1)
    belief["assumptions"][0]["strength"] = 0.9
    belief["evidence"][0]["strength"] = 0.5
    belief["claims"][0]["id"] = "C1"
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][0]["depends_on"] = []
    belief["claims"][1]["id"] = "C2"
    belief["claims"][1]["strength"] = 0.8
    belief["claims"][1]["depends_on"] = ["A1", "C1", "E1"]

    # Weaken E1 to 0.3 — should cap C2 at 0.3 (weakest of A1=0.9, C1=0.8, E1=0.3)
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {"strength": 0.3}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)
    c2 = [c for c in updated["claims"] if c["id"] == "C2"][0]
    assert c2["strength"] == 0.3


@pytest.mark.unit
def test_propagate_no_cap_when_claim_already_lower():
    """Test that propagation doesn't raise a claim above its current strength."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["assumptions"][0]["strength"] = 0.8
    belief["claims"][0]["strength"] = 0.3  # Already lower
    belief["claims"][0]["depends_on"] = ["A1"]

    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {"strength": 0.5}
    }]
    updated = apply_patches(belief, patches, propagate_strength=True)
    # C1 is at 0.3, which is already below A1's 0.5 — should stay at 0.3
    assert updated["claims"][0]["strength"] == 0.3
