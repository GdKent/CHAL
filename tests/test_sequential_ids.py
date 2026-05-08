"""
Unit tests for sequential ID enforcement and edge-type validation.

Tests cover:
- Sequential ID validation for all node types (D#, A#, C#, E#, U#, X#)
- Edge-type prefix validation in validate_belief() (schema layer)
- Edge-type prefix validation in validate_patches() (patch layer)
- Fixture verification (create_sample_belief outputs valid IDs and edges)

Implements Phase 4 of ID_AND_EDGE_VALIDATION_ROADMAP.md.
"""

import pytest
import copy
from chal.beliefs.schema import validate_belief, ALLOWED_REF_PREFIXES
from chal.beliefs.patches import validate_patches
from tests.utils import create_sample_belief


def _flat(errors_dict):
    """Flatten per-patch errors dict to a flat list of error messages."""
    return [msg for msgs in errors_dict.values() for msg in msgs]


# ========================================================
# Helper: build a fully valid belief for mutation testing
# ========================================================

def _make_belief(**overrides):
    """Create a fully valid belief, then apply overrides.

    Includes D#, A#, E#, C#, X#, U# with correct sequential IDs and
    valid cross-references so mutations can be tested in isolation.
    """
    belief = {
        "schema_version": "CBS",
        "belief_id": "BELIEF-TEST-SEQ",
        "version": 1,
        "metadata": {"topic_query": "Test", "agent_persona": "Tester"},
        "definitions": [
            {
                "id": "D1", "term": "term1",
                "definition": "Definition of term1.",
                "strength": 0.9, "strength_justification": "Well-defined",
                "status": "active", "used_by": ["A1", "E1"]
            },
            {
                "id": "D2", "term": "term2",
                "definition": "Definition of term2.",
                "strength": 0.85, "strength_justification": "Well-defined",
                "status": "active", "used_by": ["A2"]
            },
        ],
        "assumptions": [
            {
                "id": "A1", "type": "empirical",
                "statement": "Assumption 1",
                "supports_claims": ["C1"], "strength": 0.8,
                "status": "active",
                "strength_justification": "Strong assumption",
                "supported_by_definitions": ["D1"]
            },
            {
                "id": "A2", "type": "foundational",
                "statement": "Assumption 2",
                "supports_claims": ["C1"], "strength": 0.75,
                "status": "active",
                "strength_justification": "Moderate assumption",
                "supported_by_definitions": ["D2"]
            },
        ],
        "claims": [
            {
                "id": "C1", "type": "deductive",
                "statement": "Claim 1",
                "depends_on": ["A1", "A2", "E1"],
                "strength": 0.75, "status": "active",
                "strength_justification": "0.75 — limited by A2 (0.75)",
                "inference_chain": [{"role": "premise", "text": "A1 holds", "reference": "A1"}, {"role": "inference", "text": "Therefore C1 follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "Claim 1"}],
                "predictions": [
                    {"statement": "P1", "test": "T1", "decision_criterion": "DC1"}
                ]
            },
        ],
        "evidence": [
            {
                "id": "E1", "type": "empirical",
                "summary": "Evidence 1", "source": "Test 2026",
                "supports_claims": ["C1"],
                "strength": 0.8, "status": "active",
                "strength_justification": "Solid evidence",
                "supported_by_definitions": ["D1"]
            },
        ],
        "counterpositions": [
            {
                "id": "X1",
                "targets": ["C1"], "attack_type": "rebutting",
                "attack_strategy": "present_counter_example",
                "statement": "Counter 1", "my_response": "Response 1",
                "response_sufficiency": "sufficient"
            },
            {
                "id": "X2",
                "targets": ["A1"], "attack_type": "undermining",
                "attack_strategy": "challenge_evidence",
                "statement": "Counter 2", "my_response": "Response 2",
                "response_sufficiency": "partial"
            },
        ],
        "uncertainties": [
            {
                "id": "U1",
                "targets": ["A2"], "question": "Is A2 well-founded?",
                "status": "active", "importance": "medium"
            },
        ],
        "thesis": {
            "stance": "Test stance",
            "summary_bullets": ["Bullet 1", "Bullet 2"],
            "strength": 0.7
        },
        "changelog": [{"version": 1, "changes": ["Initial"]}],
    }
    for k, v in overrides.items():
        belief[k] = v
    return belief


# ========================================================
# 1. Sequential ID Tests — validate_belief()
# ========================================================

class TestSequentialIDs:
    """Tests for _validate_sequential_ids via validate_belief()."""

    @pytest.mark.unit
    def test_valid_sequential_ids_all_types(self):
        """Fully valid belief with sequential IDs for every type passes."""
        belief = _make_belief()
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Expected no errors, got: {errors}"

    @pytest.mark.unit
    def test_single_node_per_type(self):
        """A belief with exactly one node per type is sequential (prefix1)."""
        belief = _make_belief()
        # Trim to single nodes
        belief["definitions"] = [belief["definitions"][0]]
        belief["definitions"][0]["used_by"] = ["A1", "E1"]
        belief["assumptions"] = [belief["assumptions"][0]]
        belief["assumptions"][0]["supported_by_definitions"] = ["D1"]
        belief["counterpositions"] = [belief["counterpositions"][0]]
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Expected no errors, got: {errors}"

    @pytest.mark.unit
    def test_empty_collection_passes(self):
        """Empty collections are valid (no sequential check needed)."""
        belief = _make_belief()
        belief["counterpositions"] = []
        belief["uncertainties"] = []
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Expected no errors, got: {errors}"

    @pytest.mark.unit
    def test_gap_in_d_ids(self):
        """D1, D3 (missing D2) should fail."""
        belief = _make_belief()
        belief["definitions"][1]["id"] = "D3"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "D# IDs must be sequential" in e]
        assert len(seq_errors) == 1
        assert "['D1', 'D2']" in seq_errors[0]
        assert "['D1', 'D3']" in seq_errors[0]

    @pytest.mark.unit
    def test_gap_in_a_ids(self):
        """A1, A3 (missing A2) should fail."""
        belief = _make_belief()
        belief["assumptions"][1]["id"] = "A3"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "A# IDs must be sequential" in e]
        assert len(seq_errors) == 1

    @pytest.mark.unit
    def test_gap_in_c_ids(self):
        """C2 only (missing C1) should fail."""
        belief = _make_belief()
        belief["claims"][0]["id"] = "C2"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "C# IDs must be sequential" in e]
        assert len(seq_errors) == 1

    @pytest.mark.unit
    def test_gap_in_e_ids(self):
        """E3 only (missing E1) should fail."""
        belief = _make_belief()
        belief["evidence"][0]["id"] = "E3"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "E# IDs must be sequential" in e]
        assert len(seq_errors) == 1

    @pytest.mark.unit
    def test_gap_in_u_ids(self):
        """U2 only (missing U1) should fail."""
        belief = _make_belief()
        belief["uncertainties"][0]["id"] = "U2"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "U# IDs must be sequential" in e]
        assert len(seq_errors) == 1

    @pytest.mark.unit
    def test_gap_in_x_ids(self):
        """X1, X3 (missing X2) should fail."""
        belief = _make_belief()
        belief["counterpositions"][1]["id"] = "X3"
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "X# IDs must be sequential" in e]
        assert len(seq_errors) == 1

    @pytest.mark.unit
    def test_starting_from_wrong_number(self):
        """D2 as only definition (should start at D1) should fail."""
        belief = _make_belief()
        belief["definitions"] = [belief["definitions"][0]]
        belief["definitions"][0]["id"] = "D2"
        belief["definitions"][0]["used_by"] = ["A1", "E1"]
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "D# IDs must be sequential" in e]
        assert len(seq_errors) == 1
        assert "['D1']" in seq_errors[0]
        assert "['D2']" in seq_errors[0]

    @pytest.mark.unit
    def test_large_sequential_ids(self):
        """D1 through D10 (all sequential) should pass."""
        belief = _make_belief()
        belief["definitions"] = [
            {
                "id": f"D{i}", "term": f"term{i}",
                "definition": f"Definition {i}.",
                "strength": 0.9, "strength_justification": "Good",
                "status": "active", "used_by": ["A1"]
            }
            for i in range(1, 11)
        ]
        belief["assumptions"][0]["supported_by_definitions"] = ["D1"]
        belief["assumptions"][1]["supported_by_definitions"] = ["D2"]
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "D# IDs must be sequential" in e]
        assert len(seq_errors) == 0


# ========================================================
# 2. Edge-Type Constraint Tests — validate_belief()
# ========================================================

class TestEdgeTypeSchema:
    """Tests for _validate_ref_prefixes via validate_belief()."""

    # --- depends_on ---

    @pytest.mark.unit
    def test_depends_on_valid_prefixes(self):
        """depends_on with A#/E#/C# passes."""
        belief = _make_belief()
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "depends_on" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_depends_on_rejects_d(self):
        """depends_on with D# should fail."""
        belief = _make_belief()
        belief["claims"][0]["depends_on"] = ["A1", "D1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "depends_on" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1
        assert "'D'" in edge_errors[0]

    @pytest.mark.unit
    def test_depends_on_rejects_u(self):
        """depends_on with U# should fail."""
        belief = _make_belief()
        belief["claims"][0]["depends_on"] = ["A1", "U1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "depends_on" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1
        assert "'U'" in edge_errors[0]

    @pytest.mark.unit
    def test_depends_on_rejects_x(self):
        """depends_on with X# should fail."""
        belief = _make_belief()
        belief["claims"][0]["depends_on"] = ["A1", "X1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "depends_on" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1
        assert "'X'" in edge_errors[0]

    # --- supports_claims ---

    @pytest.mark.unit
    def test_supports_claims_valid(self):
        """supports_claims with C# passes."""
        belief = _make_belief()
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supports_claims" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_supports_claims_rejects_a(self):
        """supports_claims with A# should fail."""
        belief = _make_belief()
        belief["assumptions"][0]["supports_claims"] = ["A2"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supports_claims" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1

    # --- supported_by_definitions (A#) ---

    @pytest.mark.unit
    def test_assumption_supported_by_definitions_valid(self):
        """supported_by_definitions with D# passes."""
        belief = _make_belief()
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supported_by_definitions" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_assumption_supported_by_definitions_rejects_a(self):
        """A#.supported_by_definitions with A# should fail."""
        belief = _make_belief()
        belief["assumptions"][0]["supported_by_definitions"] = ["A2"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supported_by_definitions" in e and "invalid prefix" in e]
        assert len(edge_errors) >= 1

    # --- supports_claims ---

    @pytest.mark.unit
    def test_supports_claims_valid(self):
        """supports_claims with C# passes."""
        belief = _make_belief()
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supports_claims" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_supports_claims_rejects_e(self):
        """supports_claims with E# should fail."""
        belief = _make_belief()
        belief["evidence"][0]["supports_claims"] = ["E1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supports_claims" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1

    # --- evidence.supported_by_definitions ---

    @pytest.mark.unit
    def test_evidence_supported_by_definitions_rejects_a(self):
        """E#.supported_by_definitions with A# should fail."""
        belief = _make_belief()
        belief["evidence"][0]["supported_by_definitions"] = ["A1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "supported_by_definitions" in e and "invalid prefix" in e]
        assert len(edge_errors) >= 1

    # --- counterposition.targets ---

    @pytest.mark.unit
    def test_counterposition_targets_valid(self):
        """X#.targets with C#/A#/E#/D# passes."""
        belief = _make_belief()
        belief["counterpositions"][0]["targets"] = ["C1", "A1", "E1", "D1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_counterposition_targets_rejects_u(self):
        """X#.targets with U# should fail."""
        belief = _make_belief()
        belief["counterpositions"][0]["targets"] = ["U1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "X1.targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_counterposition_targets_rejects_x(self):
        """X#.targets with X# should fail (can't target another counterposition)."""
        belief = _make_belief()
        belief["counterpositions"][0]["targets"] = ["X2"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "X1.targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1

    # --- uncertainty.targets ---

    @pytest.mark.unit
    def test_uncertainty_targets_valid(self):
        """U#.targets with A#/E#/C#/D# passes."""
        belief = _make_belief()
        belief["uncertainties"][0]["targets"] = ["A1", "E1", "C1", "D1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_uncertainty_targets_rejects_u(self):
        """U#.targets with U# should fail."""
        belief = _make_belief()
        belief["uncertainties"].append({
            "id": "U2", "targets": ["U1"],
            "question": "Meta-uncertainty?",
            "status": "active", "importance": "low"
        })
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "U2.targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_uncertainty_targets_rejects_x(self):
        """U#.targets with X# should fail."""
        belief = _make_belief()
        belief["uncertainties"][0]["targets"] = ["X1"]
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "U1.targets" in e and "invalid prefix" in e]
        assert len(edge_errors) == 1


# ========================================================
# 3. Patch Validation Tests — validate_patches()
# ========================================================

class TestEdgeTypePatches:
    """Tests for edge-type prefix validation in validate_patches()."""

    @pytest.mark.unit
    def test_add_claim_valid_depends_on(self):
        """add_claim with A#/E# in depends_on accepted."""
        belief = _make_belief()
        patches = [{
            "op": "add_claim",
            "item": {
                "id": "C2", "type": "deductive",
                "statement": "New claim",
                "depends_on": ["A1", "E1"],
                "strength": 0.7, "status": "active",
                "strength_justification": "Limited by A1 (0.8)",
                "inference_chain": [{"role": "premise", "text": "A1 holds", "reference": "A1"}, {"role": "inference", "text": "Therefore C2 follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "New claim"}],
                "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only A#/E#/C# IDs allowed" in e]
        assert len(edge_errors) == 0

    @pytest.mark.unit
    def test_add_claim_rejects_d_in_depends_on(self):
        """add_claim with D# in depends_on rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_claim",
            "item": {
                "id": "C2", "type": "deductive",
                "statement": "New claim",
                "depends_on": ["A1", "D1"],
                "strength": 0.7, "status": "active",
                "strength_justification": "Test",
                "inference_chain": [{"role": "premise", "text": "A1 holds", "reference": "A1"}, {"role": "inference", "text": "Therefore C2 follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "New claim"}],
                "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only A#/E#/C# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_assumption_rejects_a_in_supports_claims(self):
        """add_assumption with A# in supports_claims rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_assumption",
            "item": {
                "id": "A3", "type": "empirical",
                "statement": "New assumption",
                "strength": 0.7, "status": "active",
                "strength_justification": "Test",
                "supports_claims": ["A1"],
                "supported_by_definitions": ["D1"]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only C# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_assumption_rejects_c_in_supported_by_definitions(self):
        """add_assumption with C# in supported_by_definitions rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_assumption",
            "item": {
                "id": "A3", "type": "empirical",
                "statement": "New assumption",
                "strength": 0.7, "status": "active",
                "strength_justification": "Test",
                "supports_claims": ["C1"],
                "supported_by_definitions": ["C1"]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only D# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_evidence_rejects_e_in_supports_claims(self):
        """add_evidence with E# in supports_claims rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_evidence",
            "item": {
                "id": "E2", "type": "empirical",
                "summary": "New evidence", "source": "Test 2026",
                "supports_claims": ["E1"],
                "strength": 0.7, "status": "active",
                "strength_justification": "Test",
                "supported_by_definitions": ["D1"]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only C# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_evidence_rejects_a_in_supported_by_definitions(self):
        """add_evidence with A# in supported_by_definitions rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_evidence",
            "item": {
                "id": "E2", "type": "empirical",
                "summary": "New evidence", "source": "Test 2026",
                "supports_claims": ["C1"],
                "strength": 0.7, "status": "active",
                "strength_justification": "Test",
                "supported_by_definitions": ["A1"]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only D# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_counterposition_rejects_u_in_targets(self):
        """add_counterposition with U# in targets rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_counterposition",
            "item": {
                "id": "X3",
                "targets": ["U1"],
                "attack_type": "undermining",
                "attack_strategy": "challenge_evidence",
                "statement": "Counter", "my_response": "Response",
                "response_sufficiency": "sufficient"
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only C#/A#/E#/D# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_uncertainty_rejects_x_in_targets(self):
        """add_uncertainty with X# in targets rejected."""
        belief = _make_belief()
        patches = [{
            "op": "add_uncertainty",
            "item": {
                "id": "U2",
                "targets": ["X1"],
                "question": "Is X1 well-formed?",
                "status": "active", "importance": "medium"
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        edge_errors = [e for e in errors if "only A#/E#/C#/D# IDs allowed" in e]
        assert len(edge_errors) == 1

    @pytest.mark.unit
    def test_add_definition_rejects_c_in_used_by(self):
        """add_definition with C# in used_by rejected (pre-existing validation)."""
        belief = _make_belief()
        patches = [{
            "op": "add_definition",
            "item": {
                "id": "D3", "term": "new term",
                "definition": "New definition.",
                "strength": 0.9, "strength_justification": "Good",
                "status": "active", "used_by": ["C1"]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        # used_by already validated as A#/E# — should get a "non-existent A#/E#" error
        assert any("non-existent A#/E#" in e for e in errors)

    @pytest.mark.unit
    def test_add_claim_with_valid_deps_accepted(self):
        """add_claim with all-valid edge types passes validation."""
        belief = _make_belief()
        patches = [{
            "op": "add_claim",
            "item": {
                "id": "C2", "type": "inductive",
                "statement": "Second claim",
                "depends_on": ["A2", "E1", "C1"],
                "strength": 0.7, "status": "active",
                "strength_justification": "Limited by A2 (0.75)",
                "inference_chain": [{"role": "premise", "text": "A2 holds", "reference": "A2"}, {"role": "inference", "text": "Therefore C2 follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "Second claim"}],
                "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}]
            }
        }]
        errors = _flat(validate_patches(patches, belief))
        assert len(errors) == 0, f"Expected no errors, got: {errors}"


# ========================================================
# 4. Fixture Verification Tests
# ========================================================

class TestFixtureVerification:
    """Verify that test helpers produce valid beliefs."""

    @pytest.mark.unit
    def test_create_sample_belief_has_sequential_ids(self):
        """create_sample_belief() output has sequential IDs."""
        belief = create_sample_belief(num_claims=3, num_assumptions=2, num_evidence=2)
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "IDs must be sequential" in e]
        assert len(seq_errors) == 0, f"Sample belief has non-sequential IDs: {seq_errors}"

    @pytest.mark.unit
    def test_create_sample_belief_has_valid_edge_types(self):
        """create_sample_belief() output has correct edge-type references."""
        belief = create_sample_belief(num_claims=2, num_assumptions=2, num_evidence=2)
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "invalid prefix" in e]
        assert len(edge_errors) == 0, f"Sample belief has invalid edge types: {edge_errors}"

    @pytest.mark.unit
    def test_make_belief_has_sequential_ids(self):
        """_make_belief() output has sequential IDs."""
        belief = _make_belief()
        errors = validate_belief(belief)
        seq_errors = [e for e in errors if "IDs must be sequential" in e]
        assert len(seq_errors) == 0, f"_make_belief has non-sequential IDs: {seq_errors}"

    @pytest.mark.unit
    def test_make_belief_has_valid_edge_types(self):
        """_make_belief() output has correct edge-type references."""
        belief = _make_belief()
        errors = validate_belief(belief)
        edge_errors = [e for e in errors if "invalid prefix" in e]
        assert len(edge_errors) == 0, f"_make_belief has invalid edge types: {edge_errors}"


# ========================================================
# 5. ALLOWED_REF_PREFIXES constant integrity
# ========================================================

class TestAllowedRefPrefixes:
    """Verify the ALLOWED_REF_PREFIXES constant is correct."""

    @pytest.mark.unit
    def test_depends_on_prefixes(self):
        assert ALLOWED_REF_PREFIXES["depends_on"] == {"A", "E", "C"}

    @pytest.mark.unit
    def test_used_by_prefixes(self):
        assert ALLOWED_REF_PREFIXES["used_by"] == {"A", "E"}

    @pytest.mark.unit
    def test_supports_claims_prefixes(self):
        assert ALLOWED_REF_PREFIXES["supports_claims"] == {"C"}

    @pytest.mark.unit
    def test_supported_by_definitions_prefixes(self):
        assert ALLOWED_REF_PREFIXES["supported_by_definitions"] == {"D"}

    @pytest.mark.unit
    def test_supports_claims_prefixes(self):
        assert ALLOWED_REF_PREFIXES["supports_claims"] == {"C"}

    @pytest.mark.unit
    def test_counterposition_targets_prefixes(self):
        assert ALLOWED_REF_PREFIXES["counterposition_targets"] == {"C", "A", "E", "D"}

    @pytest.mark.unit
    def test_uncertainty_targets_prefixes(self):
        assert ALLOWED_REF_PREFIXES["uncertainty_targets"] == {"A", "E", "C", "D"}
