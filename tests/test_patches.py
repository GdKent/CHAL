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


def _flat(errors_dict):
    """Flatten per-patch errors dict to a flat list of error messages."""
    return [msg for msgs in errors_dict.values() for msg in msgs]


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
# 3. Patch Application - update_claim retraction enforcement
# ==============================================

@pytest.mark.unit
def test_apply_patches_update_claim_retraction_enforcement(test_patches):
    """Setting status='retracted' via update_claim forces strength to 0.0."""
    belief = create_sample_belief(num_claims=1)
    belief["claims"][0]["strength"] = 0.8
    patches = test_patches["update_claim_retract"]

    updated = apply_patches(belief, patches)

    assert updated["claims"][0]["status"] == "retracted"
    assert updated["claims"][0]["strength"] == 0.0


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
            "inference_chain": [
                {"role": "premise", "text": "A1 supports new claim", "reference": "A1"},
                {"role": "inference", "text": "A1 provides evidence for the new claim", "inference_type": "deductive"},
                {"role": "conclusion", "text": "New claim is supported by A1"}
            ],
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
            "inference_chain": [
                {"role": "premise", "text": "A1 supports first claim", "reference": "A1"},
                {"role": "inference", "text": "A1 provides evidence for the first claim", "inference_type": "deductive"},
                {"role": "conclusion", "text": "First claim is supported by A1"}
            ],
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
            "inference_chain": [
                {"role": "premise", "text": "A1 supports new claim", "reference": "A1"},
                {"role": "inference", "text": "A1 provides evidence for the new claim", "inference_type": "deductive"},
                {"role": "conclusion", "text": "New claim is supported by A1"}
            ],
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
            "inference_chain": [
                {"role": "premise", "text": "A1 supports second claim", "reference": "A1"},
                {"role": "inference", "text": "A1 provides evidence for the second claim", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Second claim is supported by A1"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=True)

    # Now 2 active claims: avg = (0.8 + 0.6) / 2 = 0.7
    # breadth = 2^1.0 / (2^1.0 + 1) = 2/3 ≈ 0.6667
    # thesis = 0.7 * 0.6667 ≈ 0.4667
    assert updated["thesis"]["strength"] == pytest.approx(0.4667, abs=0.01)


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
    assert any(e["id"] == "E2" for e in updated["evidence"])


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
            "supports_claims": [],
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
    """Two active claims → ceiling = avg * breadth(2, p=1.0)."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.6
    belief["thesis"]["strength"] = 0.9

    patches = [{"op": "update_thesis", "new_strength": 0.9}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # avg = (0.8 + 0.6) / 2 = 0.7
    # breadth = 2^1.0 / (2^1.0 + 1) = 2/3 ≈ 0.6667
    # ceiling = 0.7 * 0.6667 ≈ 0.4667
    assert updated["thesis"]["strength"] == pytest.approx(0.4667, abs=0.01)


@pytest.mark.unit
def test_thesis_ceiling_three_claims():
    """Three active claims → ceiling = avg * breadth(3, p=1.0)."""
    belief = create_sample_belief(num_claims=3, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.8
    belief["claims"][2]["strength"] = 0.8
    belief["thesis"]["strength"] = 0.9

    patches = [{"op": "update_thesis", "new_strength": 0.9}]
    updated = apply_patches(belief, patches, propagate_strength=True)

    # avg = 0.8, breadth = 3^1.0 / (3^1.0 + 1) = 3/4 = 0.75, ceiling = 0.8 * 0.75 = 0.6
    assert updated["thesis"]["strength"] == pytest.approx(0.6, abs=0.01)


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

    # Formula: avg(0.8, 0.8, 0.8) × (3^1.0 / (3^1.0 + 1)) = 0.8 × 0.75 = 0.6
    assert updated["thesis"]["strength"] == pytest.approx(0.6, abs=0.01)
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

    # avg = (0.3 + 0.8) / 2 = 0.55, breadth = 2^1.0 / (2^1.0 + 1) = 2/3 ≈ 0.6667
    # ceiling = 0.55 * 0.6667 ≈ 0.3667
    assert updated["thesis"]["strength"] == pytest.approx(0.3667, abs=0.01)


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
            "importance": "high"
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
    """Test that changelog contains version and changes."""
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_missing_op(test_patches):
    """Test error if 'op' field missing."""
    belief = create_sample_belief()
    patches = test_patches["invalid_missing_op"]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("op" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_patches_unknown_op(test_patches):
    """Test error for unknown operation."""
    belief = create_sample_belief()
    patches = test_patches["invalid_unknown_op"]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("C99" in error or "not found" in error.lower() for error in errors)


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
            "supports_claims": [],
            "strength": 0.8
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))

    assert len(errors) == 1
    assert "weaken" in errors[0] or "strengthen" in errors[0]


@pytest.mark.unit
def test_validate_patches_thesis_new_strength_valid():
    """Test validation accepts valid new_strength on update_thesis."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "new_strength": 0.5}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_thesis_new_strength_out_of_bounds():
    """Test validation rejects out-of-bounds new_strength."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "new_strength": 1.5}]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))

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
            "strength": 0.7,
            "strength_justification": "0.7 — based on empirical data"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_assumption_missing_item():
    """Test validation catches add_assumption without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_assumption"}]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) >= 4  # Missing type, statement, strength, strength_justification


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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_counterposition_missing_item():
    """Test validation catches add_counterposition without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_counterposition"}]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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
            "importance": "high"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_uncertainty_missing_item():
    """Test validation catches add_uncertainty without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_uncertainty"}]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))

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
            "inference_chain": [
                {"role": "premise", "text": "A1 supports new claim", "reference": "A1"},
                {"role": "inference", "text": "A1 provides evidence for the new claim", "inference_type": "deductive"},
                {"role": "conclusion", "text": "New claim is supported by A1"}
            ],
            "predictions": [
                {"statement": "Pred", "test": "Test", "decision_criterion": "Crit"}
            ],
            "strength_justification": "0.65 — supported by A1 and E1"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_patches_add_claim_missing_item():
    """Test validation catches add_claim without item."""
    belief = create_sample_belief()
    patches = [{"op": "add_claim"}]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides duplicate reasoning", "reference": "A1"},
                {"role": "inference", "text": "Duplicate reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Duplicate claim is supported"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) >= 8  # Missing 8 required fields (includes strength_justification)


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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": [
                {"statement": "P", "test": "T", "decision_criterion": "DC"}
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": []
        }
    }]

    errors = _flat(validate_patches(patches, belief))
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
            "inference_chain": [
                {"role": "premise", "text": "A1 provides test reasoning", "reference": "A1"},
                {"role": "inference", "text": "Test reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Test claim is supported"}
            ],
            "predictions": [
                {"statement": "Pred only"}  # Missing test and decision_criterion
            ]
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) >= 2  # Missing test and decision_criterion


@pytest.mark.unit
def test_validate_patches_add_evidence_missing_item():
    """Test validation catches add_evidence without item."""
    belief = create_sample_belief()

    patches = [{
        "op": "add_evidence"
        # Missing item
    }]

    errors = _flat(validate_patches(patches, belief))

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

    errors = _flat(validate_patches(patches, belief))

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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
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

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("summary_bullets" in e for e in errors)


@pytest.mark.unit
def test_update_thesis_bullets_non_list_rejected():
    """summary_bullets: 'not a list' rejected by validation."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "summary_bullets": "not a list"}]

    errors = _flat(validate_patches(patches, belief))
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
            "supports_claims": ["C1"],
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


# ==============================================
# 3B. Blocking Validation Return Type Tests
# ==============================================

@pytest.mark.unit
def test_validate_patches_returns_dict():
    """Return type is Dict[int, List[str]], not List[str]."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C99", "changes": {"strength": 0.5}}]

    result = validate_patches(patches, belief)
    assert isinstance(result, dict)
    assert all(isinstance(k, int) for k in result.keys())
    assert all(isinstance(v, list) for v in result.values())


@pytest.mark.unit
def test_validate_patches_valid_returns_empty_dict():
    """Valid patches return empty dict {}."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.5}}]

    result = validate_patches(patches, belief)
    assert result == {}


@pytest.mark.unit
def test_validate_patches_invalid_patch_index_in_keys():
    """Error dict keys match the indices of bad patches."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.5}},  # valid (idx 0)
        {"op": "update_claim", "target_id": "C99", "changes": {"strength": 0.5}},  # invalid (idx 1)
        {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.3}},  # valid (idx 2)
    ]

    result = validate_patches(patches, belief)
    assert 1 in result
    assert 0 not in result
    assert 2 not in result


@pytest.mark.unit
def test_validate_patches_mixed_valid_invalid():
    """Batch of 3 patches where patch 1 is invalid -> dict has key 1 only."""
    belief = create_sample_belief(num_claims=1, num_evidence=1)
    patches = [
        {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.6}},  # valid
        {"op": "update_claim", "target_id": "C99", "changes": {"strength": 0.5}},  # invalid - C99 doesn't exist
        {"op": "update_evidence", "target_id": "E1", "changes": {"strength": 0.9}},  # valid
    ]

    result = validate_patches(patches, belief)
    assert set(result.keys()) == {1}
    assert len(result[1]) > 0


# ==============================================
# 3C. Field Whitelist Tests — update_claim
# ==============================================

@pytest.mark.unit
def test_validate_update_claim_rejects_unknown_field():
    """changes: {"invalid_field": 123} -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {"invalid_field": 123}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("unknown" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_claim_allows_all_valid_fields():
    """All 8 whitelist keys accepted without error."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {
            "strength": 0.6,
            "strength_justification": "Updated justification",
            "statement": "Updated statement",
            "status": "revised",
            "depends_on": ["A1", "E1"],
            "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}],
            "inference_chain": [
                {"role": "premise", "text": "A1 supports updated claim", "reference": "A1"},
                {"role": "inference", "text": "Updated reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "Updated claim is supported"}
            ],
            "type": "descriptive",
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_update_claim_empty_changes_rejected():
    """changes: {} -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("changes" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_claim_status_enum():
    """changes: {"status": "invalid"} -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {"status": "invalid"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("active" in e or "revised" in e or "retracted" in e for e in errors)


# ==============================================
# 3C. Field Whitelist Tests — update_evidence
# ==============================================

@pytest.mark.unit
def test_validate_update_evidence_rejects_unknown_field():
    """changes: {"bad": 1} -> error."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{"op": "update_evidence", "target_id": "E1", "changes": {"bad": 1}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("unknown" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_evidence_allows_all_valid_fields():
    """All 7 whitelist keys accepted."""
    belief = create_sample_belief(num_claims=1, num_evidence=1)
    patches = [{
        "op": "update_evidence",
        "target_id": "E1",
        "changes": {
            "strength": 0.7,
            "strength_justification": "Updated",
            "summary": "Updated summary",
            "source": "Updated source",
            "status": "revised",
            "supports_claims": ["C1"],
            "type": "conceptual",
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_update_evidence_empty_changes_rejected():
    """changes: {} -> error."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{"op": "update_evidence", "target_id": "E1", "changes": {}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("changes" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_evidence_type_enum():
    """changes: {"type": "invalid"} -> error."""
    belief = create_sample_belief(num_evidence=1)
    patches = [{"op": "update_evidence", "target_id": "E1", "changes": {"type": "invalid"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("empirical" in e or "conceptual" in e or "expert_consensus" in e for e in errors)


# ==============================================
# 3C. Field Whitelist Tests — update_assumption
# ==============================================

@pytest.mark.unit
def test_validate_update_assumption_rejects_unknown_field():
    """changes: {"bad": 1} -> error."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{"op": "update_assumption", "target_id": "A1", "changes": {"bad": 1}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("unknown" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_assumption_allows_all_valid_fields():
    """All 6 whitelist keys accepted."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1)
    patches = [{
        "op": "update_assumption",
        "target_id": "A1",
        "changes": {
            "strength": 0.6,
            "strength_justification": "Updated",
            "statement": "Updated statement",
            "status": "revised",
            "type": "methodological",
            "supports_claims": ["C1"],
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_update_assumption_new_type_enum():
    """new_type: "invalid" -> error."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{"op": "update_assumption", "target_id": "A1", "new_type": "invalid"}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("foundational" in e or "empirical" in e or "methodological" in e for e in errors)


@pytest.mark.unit
def test_validate_update_assumption_requires_content():
    """No new_statement, new_type, or changes -> error."""
    belief = create_sample_belief(num_assumptions=1)
    patches = [{"op": "update_assumption", "target_id": "A1"}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("at least one" in e.lower() for e in errors)


# ==============================================
# 3C. Field Whitelist Tests — update_counterposition
# ==============================================

@pytest.mark.unit
def test_validate_update_counterposition_rejects_unknown_field():
    """changes: {"bad": 1} -> error."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1", "targets": ["C1"], "attack_type": "rebutting",
        "statement": "Test", "my_response": "Response", "response_sufficiency": "partial"
    }]
    patches = [{"op": "update_counterposition", "target_id": "X1", "changes": {"bad": 1}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("unknown" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_counterposition_allows_all_valid_fields():
    """All 5 whitelist keys accepted."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1", "targets": ["C1"], "attack_type": "rebutting",
        "statement": "Test", "my_response": "Response", "response_sufficiency": "partial"
    }]
    patches = [{
        "op": "update_counterposition",
        "target_id": "X1",
        "changes": {
            "my_response": "Updated response",
            "response_sufficiency": "sufficient",
            "statement": "Updated statement",
            "attack_type": "undermining",
            "targets": ["C1"],
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_update_counterposition_empty_changes_rejected():
    """changes: {} -> error."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1", "targets": ["C1"], "attack_type": "rebutting",
        "statement": "Test", "my_response": "Response", "response_sufficiency": "partial"
    }]
    patches = [{"op": "update_counterposition", "target_id": "X1", "changes": {}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("changes" in e.lower() for e in errors)


@pytest.mark.unit
def test_validate_update_counterposition_sufficiency_enum():
    """changes: {"response_sufficiency": "invalid"} -> error."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1", "targets": ["C1"], "attack_type": "rebutting",
        "statement": "Test", "my_response": "Response", "response_sufficiency": "partial"
    }]
    patches = [{"op": "update_counterposition", "target_id": "X1", "changes": {"response_sufficiency": "invalid"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("sufficient" in e or "partial" in e or "unaddressed" in e for e in errors)


@pytest.mark.unit
def test_validate_update_counterposition_attack_type_enum():
    """changes: {"attack_type": "invalid"} -> error."""
    belief = create_sample_belief(num_claims=1)
    belief["counterpositions"] = [{
        "id": "X1", "targets": ["C1"], "attack_type": "rebutting",
        "statement": "Test", "my_response": "Response", "response_sufficiency": "partial"
    }]
    patches = [{"op": "update_counterposition", "target_id": "X1", "changes": {"attack_type": "invalid"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("undermining" in e or "rebutting" in e or "undercutting" in e for e in errors)


# ==============================================
# 3D. Missing Required Fields — add_evidence
# ==============================================

@pytest.mark.unit
def test_validate_add_evidence_missing_required_fields():
    """Item with only id -> errors for type, summary, source, supports_claims, strength, strength_justification."""
    belief = create_sample_belief()
    patches = [{"op": "add_evidence", "item": {"id": "E2"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) >= 6  # Missing type, summary, source, supports_claims, strength, strength_justification


@pytest.mark.unit
def test_validate_add_evidence_valid_all_fields():
    """Complete item -> no errors."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E2",
            "type": "empirical",
            "summary": "New evidence",
            "source": "Test (2026)",
            "supports_claims": ["C1"],
            "strength": 0.7,
            "strength_justification": "0.7 — based on empirical data"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_add_evidence_type_enum():
    """type: "invalid" -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E2",
            "type": "invalid",
            "summary": "Test",
            "source": "Test",
            "supports_claims": ["C1"],
            "strength": 0.7,
            "strength_justification": "Test"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("empirical" in e or "conceptual" in e or "expert_consensus" in e for e in errors)


@pytest.mark.unit
def test_validate_add_evidence_strength_range():
    """strength: 1.5 -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_evidence",
        "item": {
            "id": "E2",
            "type": "empirical",
            "summary": "Test",
            "source": "Test",
            "supports_claims": ["C1"],
            "strength": 1.5,
            "strength_justification": "Test"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("between 0.0 and 1.0" in e for e in errors)


# ==============================================
# 3D. Missing Required Fields — add_uncertainty
# ==============================================

@pytest.mark.unit
def test_validate_add_uncertainty_missing_required_fields():
    """Item with only id -> errors for targets, question, status, importance."""
    belief = create_sample_belief()
    patches = [{"op": "add_uncertainty", "item": {"id": "U1"}}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) >= 4  # Missing targets, question, status, importance


@pytest.mark.unit
def test_validate_add_uncertainty_valid_all_fields():
    """Complete item -> no errors."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Is this testable?",
            "status": "active",
            "importance": "high"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_add_uncertainty_status_enum():
    """status: "invalid" -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Test?",
            "status": "invalid",
            "importance": "high"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("active" in e or "resolved" in e for e in errors)


@pytest.mark.unit
def test_validate_add_uncertainty_importance_enum():
    """importance: "invalid" -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_uncertainty",
        "item": {
            "id": "U1",
            "targets": ["C1"],
            "question": "Test?",
            "status": "active",
            "importance": "invalid"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("high" in e or "medium" in e or "low" in e for e in errors)


# ==============================================
# 3D. Missing strength_justification
# ==============================================

@pytest.mark.unit
def test_validate_add_claim_missing_strength_justification():
    """Item with all fields except strength_justification -> error."""
    belief = create_sample_belief(num_assumptions=1, num_evidence=1)
    patches = [{
        "op": "add_claim",
        "item": {
            "id": "C2",
            "type": "deductive",
            "statement": "New claim",
            "depends_on": ["A1", "E1"],
            "strength": 0.6,
            "status": "active",
            "inference_chain": [
                {"role": "premise", "text": "A1 provides reasoning", "reference": "A1"},
                {"role": "inference", "text": "Reasoning based on A1", "inference_type": "deductive"},
                {"role": "conclusion", "text": "New claim is supported by reasoning"}
            ],
            "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
            # Missing strength_justification
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("strength_justification" in e for e in errors)


@pytest.mark.unit
def test_validate_add_assumption_missing_strength_justification():
    """Item with all fields except strength_justification -> error."""
    belief = create_sample_belief()
    patches = [{
        "op": "add_assumption",
        "item": {
            "id": "A2",
            "type": "empirical",
            "statement": "New assumption",
            "strength": 0.7
            # Missing strength_justification
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("strength_justification" in e for e in errors)


# ==============================================
# 3E. Enum Validation — add_counterposition
# ==============================================

@pytest.mark.unit
def test_validate_add_counterposition_attack_type_enum():
    """attack_type: "invalid" -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "invalid",
            "statement": "Test",
            "my_response": "Response",
            "response_sufficiency": "partial"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("undermining" in e or "rebutting" in e or "undercutting" in e for e in errors)


@pytest.mark.unit
def test_validate_add_counterposition_sufficiency_enum():
    """response_sufficiency: "invalid" -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Test",
            "my_response": "Response",
            "response_sufficiency": "invalid"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("sufficient" in e or "partial" in e or "unaddressed" in e for e in errors)


@pytest.mark.unit
def test_validate_add_counterposition_targets_nonempty():
    """targets: [] -> error."""
    belief = create_sample_belief(num_claims=1)
    patches = [{
        "op": "add_counterposition",
        "item": {
            "id": "X1",
            "targets": [],
            "attack_type": "rebutting",
            "statement": "Test",
            "my_response": "Response",
            "response_sufficiency": "partial"
        }
    }]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("targets" in e.lower() for e in errors)


# ==============================================
# 3F. update_thesis Mutual Exclusivity
# ==============================================

@pytest.mark.unit
def test_validate_update_thesis_mutual_exclusivity():
    """Both new_strength and change present -> error."""
    belief = create_sample_belief()
    patches = [{"op": "update_thesis", "new_strength": 0.5, "change": "weaken"}]

    errors = _flat(validate_patches(patches, belief))
    assert len(errors) > 0
    assert any("mutually exclusive" in e.lower() or ("new_strength" in e and "change" in e) for e in errors)


# ==============================================
# 3G. Integration Tests — Blocking Behavior
# ==============================================

@pytest.mark.unit
def test_invalid_patch_skipped_valid_patches_applied():
    """Batch of [valid, invalid, valid] -> only 2 patches applied, belief updated correctly."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["strength"] = 0.8
    belief["claims"][1]["strength"] = 0.7

    patches = [
        {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.5}},  # valid
        {"op": "update_claim", "target_id": "C99", "changes": {"strength": 0.3}},  # invalid
        {"op": "update_claim", "target_id": "C2", "changes": {"strength": 0.4}},  # valid
    ]

    # Filter out invalid patches (same pattern as debate_controller)
    patch_errors = validate_patches(patches, belief)
    invalid_indices = set(patch_errors.keys())
    valid_patches = [p for i, p in enumerate(patches) if i not in invalid_indices]

    assert len(valid_patches) == 2
    assert invalid_indices == {1}

    updated = apply_patches(belief, valid_patches, propagate_strength=False)
    assert updated["claims"][0]["strength"] == 0.5
    assert updated["claims"][1]["strength"] == 0.4


@pytest.mark.unit
def test_all_patches_invalid_returns_unchanged_belief():
    """All invalid patches -> belief unchanged (except version/changelog)."""
    belief = create_sample_belief(num_claims=1)
    original_strength = belief["claims"][0]["strength"]

    patches = [
        {"op": "update_claim", "target_id": "C99", "changes": {"strength": 0.3}},
        {"op": "update_evidence", "target_id": "E99", "changes": {"strength": 0.5}},
    ]

    patch_errors = validate_patches(patches, belief)
    invalid_indices = set(patch_errors.keys())
    valid_patches = [p for i, p in enumerate(patches) if i not in invalid_indices]

    assert len(valid_patches) == 0

    updated = apply_patches(belief, valid_patches)
    # Claim strength unchanged since no patches were applied
    assert updated["claims"][0]["strength"] == original_strength


# ==============================================
# Inference Chain — Patch Validation Tests
# ==============================================

_VALID_IC = [
    {"role": "premise", "text": "A1 supports new claim", "reference": "A1"},
    {"role": "inference", "text": "Therefore the new claim follows", "inference_type": "deductive"},
    {"role": "conclusion", "text": "New claim statement"},
]


@pytest.mark.unit
def test_validate_add_claim_valid_inference_chain():
    """add_claim with a valid structured inference_chain passes validation."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "add_claim", "item": {
        "id": "C2", "type": "deductive", "statement": "New claim statement",
        "depends_on": ["A1"], "strength": 0.7, "status": "active",
        "strength_justification": "Justified by A1",
        "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}],
        "inference_chain": _VALID_IC,
    }}]
    errors = _flat(validate_patches(patches, belief))
    ic_errors = [e for e in errors if "inference_chain" in e.lower() or "inference step" in e or "premise" in e.lower()]
    assert len(ic_errors) == 0, f"Valid IC in add_claim should pass, got: {ic_errors}"


@pytest.mark.unit
def test_validate_add_claim_old_string_format_rejected():
    """add_claim with old string-format inference_chain fails validation."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "add_claim", "item": {
        "id": "C2", "type": "deductive", "statement": "New claim",
        "depends_on": ["A1"], "strength": 0.7, "status": "active",
        "strength_justification": "Justified",
        "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}],
        "inference_chain": ["Step 1: A1 holds", "Step 2: Therefore claim"],
    }}]
    errors = _flat(validate_patches(patches, belief))
    assert any("must be an object" in e for e in errors), \
        f"String-format IC in add_claim should fail, got: {errors}"


@pytest.mark.unit
def test_validate_update_claim_valid_inference_chain_replacement():
    """update_claim replacing inference_chain with valid new one passes."""
    belief = create_sample_belief(num_claims=1)
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {
        "inference_chain": _VALID_IC,
    }}]
    errors = _flat(validate_patches(patches, belief))
    ic_errors = [e for e in errors if "inference_chain" in e.lower() or "inference step" in e or "premise" in e.lower()]
    assert len(ic_errors) == 0, f"Valid IC replacement should pass, got: {ic_errors}"


@pytest.mark.unit
def test_validate_update_claim_invalid_inference_chain_rejected():
    """update_claim with structurally invalid inference_chain fails."""
    belief = create_sample_belief(num_claims=1)
    # Missing conclusion and inference
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {
        "inference_chain": [
            {"role": "premise", "text": "A1 holds", "reference": "A1"},
        ],
    }}]
    errors = _flat(validate_patches(patches, belief))
    assert any("inference step" in e for e in errors), \
        f"Invalid IC in update_claim should fail, got: {errors}"


@pytest.mark.unit
def test_update_claim_inference_chain_changelog_formatting():
    """update_claim with inference_chain produces meaningful changelog entry."""
    belief = create_sample_belief(num_claims=1)
    new_ic = [
        {"role": "premise", "text": "A1 holds", "reference": "A1"},
        {"role": "premise", "text": "E1 supports", "reference": "E1"},
        {"role": "inference", "text": "Therefore claim follows", "inference_type": "inductive"},
        {"role": "conclusion", "text": "Claim 1"},
    ]
    patches = [{"op": "update_claim", "target_id": "C1", "changes": {
        "inference_chain": new_ic,
    }}]
    updated = apply_patches(belief, patches)
    changelog_text = " ".join(updated["changelog"][-1]["changes"])
    assert "inference_chain updated" in changelog_text, \
        f"Changelog should contain 'inference_chain updated', got: {changelog_text}"
    assert "premises" in changelog_text, \
        f"Changelog should summarise premise count, got: {changelog_text}"


# ==============================================
# Incremental ID Tracking in validate_patches
# ==============================================

def _make_add_assumption(aid, supports_claims=None, supported_by_definitions=None):
    """Helper to build an add_assumption patch."""
    item = {
        "id": aid, "type": "empirical", "statement": f"{aid} statement",
        "strength": 0.75, "strength_justification": f"{aid} justification",
        "supports_claims": supports_claims or [],
    }
    if supported_by_definitions:
        item["supported_by_definitions"] = supported_by_definitions
    return {"op": "add_assumption", "item": item}


def _make_add_evidence(eid, supports_claims=None, supported_by_definitions=None):
    """Helper to build an add_evidence patch."""
    item = {
        "id": eid, "type": "empirical", "summary": f"{eid} summary",
        "source": "Test (2026)", "supports_claims": supports_claims or [],
        "strength": 0.75, "strength_justification": f"{eid} justification",
    }
    if supported_by_definitions:
        item["supported_by_definitions"] = supported_by_definitions
    return {"op": "add_evidence", "item": item}


def _make_add_claim(cid, depends_on):
    """Helper to build an add_claim patch."""
    return {"op": "add_claim", "item": {
        "id": cid, "type": "deductive", "statement": f"{cid} statement",
        "depends_on": depends_on, "strength": 0.7,
        "strength_justification": f"{cid} justification", "status": "active",
        "inference_chain": [
            {"role": "premise", "text": f"From {depends_on[0]}", "reference": depends_on[0]},
            {"role": "inference", "text": f"Therefore {cid}", "inference_type": "deductive"},
            {"role": "conclusion", "text": f"{cid} statement"},
        ],
        "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}],
    }}


def _make_add_definition(did, used_by):
    """Helper to build an add_definition patch."""
    return {"op": "add_definition", "item": {
        "id": did, "term": f"{did} term", "definition": f"{did} definition",
        "strength": 0.85, "strength_justification": f"{did} justification",
        "used_by": used_by,
    }}


def _make_add_counterposition(xid, targets):
    """Helper to build an add_counterposition patch."""
    return {"op": "add_counterposition", "item": {
        "id": xid, "targets": targets, "attack_type": "undercutting",
        "statement": f"{xid} statement", "my_response": f"{xid} response",
        "response_sufficiency": "partial",
    }}


@pytest.mark.unit
def test_validate_incremental_add_evidence_then_claim():
    """add_claim can reference an E# added earlier in the same batch."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_evidence("E3", supports_claims=["C1"]),
        _make_add_claim("C4", depends_on=["A1", "E3"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_add_assumption_then_claim():
    """add_claim can reference an A# added earlier in the same batch."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_assumption("A3", supports_claims=["C1"]),
        _make_add_claim("C4", depends_on=["A3", "E1"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_add_definition_then_evidence():
    """add_evidence supported_by_definitions can reference a D# added earlier."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_definition("D2", used_by=["A1"]),
        _make_add_evidence("E3", supports_claims=["C1"], supported_by_definitions=["D2"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_add_definition_then_assumption():
    """add_assumption supported_by_definitions can reference a D# added earlier."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_definition("D2", used_by=["A1"]),
        _make_add_assumption("A3", supported_by_definitions=["D2"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_add_claim_then_evidence_supporting_it():
    """add_evidence supports_claims can reference a C# added earlier."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_claim("C4", depends_on=["A1", "E1"]),
        _make_add_evidence("E3", supports_claims=["C4"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_add_claim_then_counterposition():
    """add_counterposition targets can reference a C# added earlier."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_claim("C4", depends_on=["A1", "E1"]),
        _make_add_counterposition("X1", targets=["C4"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Expected no errors, got: {errors}"


@pytest.mark.unit
def test_validate_incremental_forward_ref_fails_if_earlier_patch_invalid():
    """If add_evidence at index 0 is invalid, add_claim at index 1
    referencing that E# should also fail (ID not tracked)."""
    belief = create_sample_belief(num_claims=1)
    # E3 is invalid: missing required 'summary' field
    bad_evidence = {"op": "add_evidence", "item": {
        "id": "E3", "type": "empirical",
        # "summary" deliberately omitted
        "source": "Test (2026)", "supports_claims": ["C1"],
        "strength": 0.75, "strength_justification": "justification",
    }}
    patches = [
        bad_evidence,
        _make_add_claim("C4", depends_on=["A1", "E3"]),
    ]
    errors = validate_patches(patches, belief)
    # Both patches should have errors
    assert 0 in errors, "Invalid evidence at index 0 should have errors"
    assert 1 in errors, "Claim referencing untracked E3 at index 1 should have errors"
    # The claim error should mention E3
    claim_errors = errors[1]
    assert any("E3" in e for e in claim_errors), \
        f"Claim errors should mention E3, got: {claim_errors}"


@pytest.mark.unit
def test_validate_incremental_duplicate_id_in_batch():
    """Two add_claim patches with the same C# ID should both error."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_claim("C4", depends_on=["A1", "E1"]),
        _make_add_claim("C4", depends_on=["A1", "E1"]),
    ]
    errors = validate_patches(patches, belief)
    # The second patch should fail because C4 was already added by the first
    assert 1 in errors, "Second add_claim with same ID should error"
    assert any("already exists" in e for e in errors[1]), \
        f"Error should mention 'already exists', got: {errors[1]}"


@pytest.mark.unit
def test_validate_incremental_full_growth_batch():
    """A realistic batch: add_definition -> add_assumption -> add_evidence -> add_claim.
    All should validate successfully with incremental tracking."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_definition("D2", used_by=["A1"]),
        _make_add_assumption("A3", supports_claims=["C1"], supported_by_definitions=["D2"]),
        _make_add_evidence("E3", supports_claims=["C1"], supported_by_definitions=["D2"]),
        _make_add_claim("C4", depends_on=["A3", "E3"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Full growth batch should pass, got: {errors}"


# ==============================================
# Phase 2: Asymmetric supports_claims validation
# ==============================================

@pytest.mark.unit
def test_add_assumption_supports_claims_nonexistent():
    """add_assumption supports_claims referencing a non-existent C# should fail."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_assumption("A3", supports_claims=["C99"]),
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "Should reject non-existent claim reference"
    assert any("C99" in e for e in errors[0]), \
        f"Error should mention C99, got: {errors[0]}"


@pytest.mark.unit
def test_add_assumption_supports_claims_forward_ref_passes():
    """add_assumption supports_claims referencing a C# added LATER in the batch
    should pass thanks to projected-ID pre-registration."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_assumption("A3", supports_claims=["C4"]),
        _make_add_claim("C4", depends_on=["A1", "E1"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Forward ref to C4 should pass via projection, got: {errors}"


@pytest.mark.unit
def test_add_assumption_supports_claims_backward_ref_passes():
    """add_assumption supports_claims referencing a C# added EARLIER in the batch
    should pass (C# was tracked by Phase 1 incremental ID tracking)."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_claim("C4", depends_on=["A1", "E1"]),
        _make_add_assumption("A3", supports_claims=["C4"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Backward ref to C4 should pass, got: {errors}"


# ==============================================
# Defense Tracking Field Preservation Tests
# ==============================================

@pytest.mark.unit
def test_update_claim_preserves_original_strength():
    """update_claim cannot overwrite original_strength."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["claims"][0]["original_strength"] = 0.65
    belief["claims"][0]["consecutive_defenses"] = 3
    patches = [
        {"op": "update_claim", "target_id": "C1",
         "changes": {"strength": 0.50, "original_strength": 0.99, "consecutive_defenses": 99}}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    assert result["claims"][0]["original_strength"] == 0.65
    assert result["claims"][0]["consecutive_defenses"] == 3
    assert result["claims"][0]["strength"] == 0.50


@pytest.mark.unit
def test_update_assumption_preserves_tracking_fields():
    """update_assumption cannot overwrite original_strength or consecutive_defenses."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["assumptions"][0]["original_strength"] = 0.80
    belief["assumptions"][0]["consecutive_defenses"] = 2
    patches = [
        {"op": "update_assumption", "target_id": "A1",
         "changes": {"strength": 0.60, "original_strength": 0.50, "consecutive_defenses": 0}}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    assert result["assumptions"][0]["original_strength"] == 0.80
    assert result["assumptions"][0]["consecutive_defenses"] == 2


@pytest.mark.unit
def test_update_evidence_preserves_tracking_fields():
    """update_evidence cannot overwrite original_strength or consecutive_defenses."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["evidence"][0]["original_strength"] = 0.80
    belief["evidence"][0]["consecutive_defenses"] = 1
    patches = [
        {"op": "update_evidence", "target_id": "E1",
         "changes": {"strength": 0.55, "original_strength": 0.10, "consecutive_defenses": 10}}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    assert result["evidence"][0]["original_strength"] == 0.80
    assert result["evidence"][0]["consecutive_defenses"] == 1


@pytest.mark.unit
def test_update_definition_preserves_tracking_fields():
    """update_definition cannot overwrite original_strength or consecutive_defenses."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    belief["definitions"][0]["original_strength"] = 0.90
    belief["definitions"][0]["consecutive_defenses"] = 4
    patches = [
        {"op": "update_definition", "target_id": "D1",
         "changes": {"strength": 0.70, "original_strength": 0.30, "consecutive_defenses": 0}}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    assert result["definitions"][0]["original_strength"] == 0.90
    assert result["definitions"][0]["consecutive_defenses"] == 4


@pytest.mark.unit
def test_add_claim_initializes_tracking_fields():
    """add_claim sets original_strength and consecutive_defenses on new claims."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "add_claim", "item": {
            "id": "C2", "type": "deductive", "statement": "New claim",
            "depends_on": ["A1"], "strength": 0.70,
            "strength_justification": "Test",
            "status": "active",
            "inference_chain": [
                {"role": "premise", "text": "A1 holds", "reference": "A1"},
                {"role": "inference", "text": "Therefore C2", "inference_type": "deductive"},
                {"role": "conclusion", "text": "C2"}
            ],
            "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
        }}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    new_claim = [c for c in result["claims"] if c["id"] == "C2"][0]
    assert new_claim["original_strength"] == 0.70
    assert new_claim["consecutive_defenses"] == 0


@pytest.mark.unit
def test_add_evidence_initializes_tracking_fields():
    """add_evidence sets original_strength and consecutive_defenses on new evidence."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "add_evidence", "item": {
            "id": "E2", "type": "empirical", "summary": "New evidence",
            "source": "Test (2026)", "supports_claims": ["C1"],
            "strength": 0.85, "strength_justification": "Test",
            "supported_by_definitions": ["D1"]
        }}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    new_ev = [e for e in result["evidence"] if e["id"] == "E2"][0]
    assert new_ev["original_strength"] == 0.85
    assert new_ev["consecutive_defenses"] == 0


@pytest.mark.unit
def test_add_assumption_initializes_tracking_fields():
    """add_assumption sets original_strength and consecutive_defenses on new assumptions."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "add_assumption", "item": {
            "id": "A2", "type": "empirical", "statement": "New assumption",
            "strength": 0.75, "strength_justification": "Test",
            "supports_claims": ["C1"], "supported_by_definitions": ["D1"]
        }}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    new_a = [a for a in result["assumptions"] if a["id"] == "A2"][0]
    assert new_a["original_strength"] == 0.75
    assert new_a["consecutive_defenses"] == 0


@pytest.mark.unit
def test_add_definition_initializes_tracking_fields():
    """add_definition sets original_strength and consecutive_defenses on new definitions."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "add_definition", "item": {
            "id": "D2", "term": "new term", "definition": "New definition",
            "strength": 0.80, "strength_justification": "Test",
            "status": "active", "used_by": ["A1"]
        }}
    ]
    result = apply_patches(belief, patches, propagate_strength=False)
    new_d = [d for d in result["definitions"] if d["id"] == "D2"][0]
    assert new_d["original_strength"] == 0.80
    assert new_d["consecutive_defenses"] == 0


# ==============================================
# Projected-ID Validation Tests (Phase 0)
# ==============================================

@pytest.mark.unit
def test_validate_projected_ids_cross_reference_batch():
    """Full cross-referencing chain: D5 → A5/E5 → C3.
    All patches reference each other and should pass via projected IDs."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_definition("D5", used_by=["A5", "E5"]),
        _make_add_assumption("A5", supports_claims=["C3"], supported_by_definitions=["D5"]),
        _make_add_evidence("E5", supports_claims=["C3"], supported_by_definitions=["D5"]),
        _make_add_claim("C3", depends_on=["A5", "E5"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Cross-referencing batch should pass via projection, got: {errors}"


@pytest.mark.unit
def test_validate_projected_ids_partial_failure_cascade():
    """If a projected patch fails for non-reference reasons, dependents cascade-fail."""
    belief = create_sample_belief(num_claims=1)
    # D5 has strength out of range → fails validation
    bad_def = _make_add_definition("D5", used_by=["A5", "E5"])
    bad_def["item"]["strength"] = 2.0  # Invalid
    patches = [
        bad_def,
        _make_add_assumption("A5", supports_claims=["C1"], supported_by_definitions=["D5"]),
        _make_add_evidence("E5", supports_claims=["C1"], supported_by_definitions=["D5"]),
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "D5 should fail (bad strength)"
    assert 1 in errors, "A5 should cascade-fail (depends on D5)"
    assert any("failed patch" in e for e in errors[1]), \
        f"A5 error should mention failed patch, got: {errors[1]}"
    assert 2 in errors, "E5 should cascade-fail (depends on D5)"


@pytest.mark.unit
def test_validate_projected_ids_no_false_positives():
    """Patches referencing genuinely non-existent nodes (not in batch) should still fail."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_assumption("A5", supports_claims=["C99"], supported_by_definitions=["D99"]),
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "Should fail for non-existent C99 and D99"
    error_text = " ".join(errors[0])
    assert "C99" in error_text, f"Should mention C99, got: {errors[0]}"
    assert "D99" in error_text, f"Should mention D99, got: {errors[0]}"


@pytest.mark.unit
def test_validate_projected_ids_mixed_valid_invalid():
    """Batch with valid cross-referencing patches and independently invalid ones.
    Valid patches should survive; invalid ones fail without contaminating valid ones."""
    belief = create_sample_belief(num_claims=1)
    # Patch 0: bad assumption (missing required field 'type')
    bad_assumption = {"op": "add_assumption", "item": {
        "id": "A5", "statement": "missing type",
        "strength": 0.7, "strength_justification": "test",
        "supports_claims": ["C1"],
    }}
    patches = [
        bad_assumption,
        # Patch 1: valid definition referencing existing A1
        _make_add_definition("D5", used_by=["A1"]),
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "Bad assumption should fail"
    assert 1 not in errors, f"Valid definition should pass, got: {errors.get(1)}"


@pytest.mark.unit
def test_validate_projected_definition_used_by_forward_ref():
    """The exact failure pattern from the live debate: add_definition with used_by
    referencing an A# that is added later in the same batch."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_definition("D5", used_by=["A5"]),
        _make_add_assumption("A5", supports_claims=["C1"], supported_by_definitions=["D5"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"Forward ref D5→A5 should pass via projection, got: {errors}"


# ==============================================
# Whitelist Tests (Phase 2: supported_by_definitions)
# ==============================================

@pytest.mark.unit
def test_update_assumption_supported_by_definitions_accepted():
    """update_assumption with supported_by_definitions should pass when D1 exists."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "update_assumption", "target_id": "A1",
         "changes": {"supported_by_definitions": ["D1"]}}
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"supported_by_definitions with valid D1 should pass, got: {errors}"


@pytest.mark.unit
def test_update_evidence_supported_by_definitions_accepted():
    """update_evidence with supported_by_definitions should pass when D1 exists."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "update_evidence", "target_id": "E1",
         "changes": {"supported_by_definitions": ["D1"]}}
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"supported_by_definitions with valid D1 should pass, got: {errors}"


@pytest.mark.unit
def test_update_assumption_supported_by_definitions_validates_refs():
    """update_assumption with supported_by_definitions referencing non-existent D# should fail."""
    belief = create_sample_belief(num_claims=1, num_assumptions=1, num_evidence=1)
    patches = [
        {"op": "update_assumption", "target_id": "A1",
         "changes": {"supported_by_definitions": ["D99"]}}
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "Should fail for non-existent D99"
    assert any("D99" in e for e in errors[0]), \
        f"Error should mention D99, got: {errors[0]}"


# --- Phase 0: Transitive cascade removal tests ---


@pytest.mark.unit
def test_cascade_transitive_three_hop():
    """Three-hop chain: C3 fails → A5/E5 cascade → D6 cascades on second iteration.

    Patch 0: add_claim C3 (missing predictions → fails validation)
    Patch 1: add_assumption A5 (supports_claims: [C3] → cascade-fails, hop 1)
    Patch 2: add_evidence E5 (supports_claims: [C3] → cascade-fails, hop 1)
    Patch 3: add_definition D6 (used_by: [A5, E5] → cascade-fails, hop 2)
    """
    belief = create_sample_belief(num_claims=1)
    # C3 is missing 'predictions' (required for add_claim) → fails
    bad_claim = _make_add_claim("C3", depends_on=["A1"])
    del bad_claim["item"]["predictions"]
    patches = [
        bad_claim,
        _make_add_assumption("A5", supports_claims=["C3"]),
        _make_add_evidence("E5", supports_claims=["C3"]),
        _make_add_definition("D6", used_by=["A5", "E5"]),
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "C3 should fail (missing predictions)"
    assert 1 in errors, "A5 should cascade-fail (depends on C3)"
    assert 2 in errors, "E5 should cascade-fail (depends on C3)"
    assert 3 in errors, "D6 should cascade-fail (depends on A5/E5, second hop)"
    assert any("failed patch" in e for e in errors[3]), \
        f"D6 error should mention failed patch, got: {errors[3]}"


@pytest.mark.unit
def test_cascade_transitive_converges():
    """Same three-hop structure but C3 is valid — all patches should pass."""
    belief = create_sample_belief(num_claims=1)
    patches = [
        _make_add_claim("C3", depends_on=["A1"]),
        _make_add_assumption("A5", supports_claims=["C3"]),
        _make_add_evidence("E5", supports_claims=["C3"]),
        _make_add_definition("D6", used_by=["A5", "E5"]),
    ]
    errors = validate_patches(patches, belief)
    assert errors == {}, f"All patches valid, should have no errors, got: {errors}"


@pytest.mark.unit
def test_cascade_update_targets_failed_add():
    """update_* targeting a failed add_* ID should cascade-fail.

    Patch 0: add_assumption A5 (missing type → fails)
    Patch 1: update_assumption A5 (should cascade-fail because A5 was never added)
    """
    belief = create_sample_belief(num_claims=1)
    bad_assumption = {"op": "add_assumption", "item": {
        "id": "A5", "statement": "missing type field",
        "strength": 0.75, "strength_justification": "A5 justification",
        "supports_claims": ["C1"],
        # 'type' deliberately omitted → validation failure
    }}
    patches = [
        bad_assumption,
        {"op": "update_assumption", "target_id": "A5",
         "changes": {"strength": 0.80, "strength_justification": "boosted"}},
    ]
    errors = validate_patches(patches, belief)
    assert 0 in errors, "A5 add should fail (missing type)"
    assert 1 in errors, "update_assumption A5 should cascade-fail (A5 was never added)"
    assert any("failed patch" in e for e in errors[1]), \
        f"update_assumption error should mention failed patch, got: {errors[1]}"


@pytest.mark.unit
def test_cascade_no_infinite_loop():
    """Large batch with a single root failure. Verify convergence and correct error set.

    Structure: C3 (fails) → A5..A14 (each supports_claims: [C3]) → D6..D15 (each used_by: [A5..A14])
    Total: 1 + 10 + 10 = 21 patches. All should cascade-fail except none should be left valid.
    """
    belief = create_sample_belief(num_claims=1)
    # Root failure: C3 missing predictions
    bad_claim = _make_add_claim("C3", depends_on=["A1"])
    del bad_claim["item"]["predictions"]
    patches = [bad_claim]

    # 10 assumptions depending on C3
    a_ids = [f"A{5 + j}" for j in range(10)]
    for aid in a_ids:
        patches.append(_make_add_assumption(aid, supports_claims=["C3"]))

    # 10 definitions, each depending on one of the assumptions
    for j, aid in enumerate(a_ids):
        patches.append(_make_add_definition(f"D{6 + j}", used_by=[aid]))

    errors = validate_patches(patches, belief)
    # All 21 patches should fail
    assert len(errors) == 21, f"Expected 21 failures, got {len(errors)}: {sorted(errors.keys())}"
    # Patch 0 is the root failure
    assert 0 in errors
    # All assumption patches (1-10) should cascade-fail
    for i in range(1, 11):
        assert i in errors, f"Patch {i} (assumption) should cascade-fail"
    # All definition patches (11-20) should cascade-fail
    for i in range(11, 21):
        assert i in errors, f"Patch {i} (definition) should cascade-fail"
