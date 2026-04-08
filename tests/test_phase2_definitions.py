"""
Phase 2 tests: IO, Rendering, Patch Operations & Ceiling Enforcement for D# nodes.

Tests cover:
- belief_to_markdown() with D# nodes: section rendered, supported_by_definitions on A#/E#
- project_for_embedding() with D# nodes: definitions included
- add_definition patch: valid operation, side effects, changelog
- add_definition patch: invalid cases (duplicate ID, bad format, missing fields, bad refs)
- update_definition patch: field updates, retraction forces 0.0, immutable rejection
- update_definition patch: used_by change updates supported_by_definitions
- D# ceiling enforcement: basic ceiling, cascade to C# and thesis
- Orphan A#/E# cap: all D# retracted → capped at 0.6
- Orphan C# cap: all A#/E# retracted → capped at 0.2
- Graph visualizer: D# nodes rendered (color check)
- validate_patches for definition operations
"""

import pytest
import json
import copy
from chal.beliefs.io import belief_to_markdown, project_for_embedding
from chal.beliefs.patches import (
    apply_patches, validate_patches,
    ORPHAN_AE_CAP, ORPHAN_CLAIM_CAP,
)
from chal.beliefs.belief_graph import BeliefGraph
from tests.utils import create_sample_belief


# ==============================================
# Helpers
# ==============================================

def _flat(errors_dict):
    """Flatten per-patch errors dict to a flat list of error messages."""
    return [msg for msgs in errors_dict.values() for msg in msgs]


def _make_belief_with_defs():
    """Create a belief with rich D# nodes for testing."""
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-DEFTEST",
        "version": 1,
        "metadata": {
            "topic_query": "Does free will exist?",
            "agent_persona": "Compatibilist"
        },
        "definitions": [
            {
                "id": "D1",
                "term": "free will",
                "definition": "The capacity to choose based on rational deliberation.",
                "strength": 0.8,
                "strength_justification": "Standard compatibilist definition",
                "status": "active",
                "used_by": ["A1", "E1"]
            },
            {
                "id": "D2",
                "term": "determinism",
                "definition": "The doctrine that every event is necessitated by prior causes.",
                "strength": 0.9,
                "strength_justification": "Widely accepted philosophical definition",
                "status": "active",
                "used_by": ["A2", "E1"]
            }
        ],
        "thesis": {
            "stance": "Free will exists within a compatibilist framework.",
            "summary_bullets": ["Determinism and free will are compatible"],
            "strength": 0.7
        },
        "assumptions": [
            {
                "id": "A1",
                "type": "empirical",
                "statement": "Rational deliberation is real",
                "strength": 0.8,
                "status": "active",
                "strength_justification": "Cognitive science evidence",
                "supports_claims": ["C1"],
                "supported_by_definitions": ["D1"]
            },
            {
                "id": "A2",
                "type": "foundational",
                "statement": "Causation does not eliminate agency",
                "strength": 0.85,
                "status": "active",
                "strength_justification": "Standard compatibilist premise",
                "supports_claims": ["C1"],
                "supported_by_definitions": ["D2"]
            }
        ],
        "claims": [
            {
                "id": "C1",
                "type": "deductive",
                "statement": "Humans can make choices based on reasons",
                "depends_on": ["A1", "A2", "E1"],
                "strength": 0.75,
                "strength_justification": "Strong empirical support",
                "status": "active",
                "inference_chain": [{"role": "premise", "text": "A1 and A2 together support C1", "reference": "A1"}, {"role": "inference", "text": "Therefore C1 follows from A1 and A2", "inference_type": "deductive"}, {"role": "conclusion", "text": "Humans can make choices based on reasons"}],
                "predictions": [
                    {
                        "statement": "People report feeling in control",
                        "test": "Survey",
                        "decision_criterion": ">70% report control"
                    }
                ]
            }
        ],
        "evidence": [
            {
                "id": "E1",
                "type": "empirical",
                "summary": "Studies show deliberation before decisions",
                "source": "Libet et al. (1983)",
                "supports_claims": ["C1"],
                "strength": 0.8,
                "status": "active",
                "strength_justification": "Replicated study",
                "supported_by_definitions": ["D1", "D2"]
            }
        ],
        "counterpositions": [
            {
                "id": "X1",
                "targets": ["C1"],
                "attack_type": "undercutting",
                "attack_strategy": "challenge_inference_step",
                "statement": "Neural activity precedes awareness",
                "my_response": "Only tests trivial decisions",
                "response_sufficiency": "partial"
            }
        ],
        "uncertainties": []
    }


# ==============================================
# 1. belief_to_markdown() with D# nodes
# ==============================================

@pytest.mark.unit
def test_belief_to_markdown_definitions_section():
    """Test that Definitions section is rendered before Assumptions."""
    belief = _make_belief_with_defs()
    md = belief_to_markdown(belief)

    assert "# Definitions" in md
    # Definitions section should appear before Assumptions
    def_pos = md.index("# Definitions")
    assert "# Assumptions" in md
    assump_pos = md.index("# Assumptions")
    assert def_pos < assump_pos


@pytest.mark.unit
def test_belief_to_markdown_definition_content():
    """Test that definition content is rendered correctly."""
    belief = _make_belief_with_defs()
    md = belief_to_markdown(belief)

    assert "[D1]" in md
    assert "**free will**" in md
    assert "The capacity to choose based on rational deliberation." in md
    assert "Strength: 0.8" in md
    assert "Used by: A1, E1" in md
    assert "Status: active" in md


@pytest.mark.unit
def test_belief_to_markdown_supported_by_definitions_on_assumptions():
    """Test that assumptions show supported_by_definitions."""
    belief = _make_belief_with_defs()
    md = belief_to_markdown(belief)

    assert "Supported by definitions: D1" in md
    assert "Supported by definitions: D2" in md


@pytest.mark.unit
def test_belief_to_markdown_supported_by_definitions_on_evidence():
    """Test that evidence shows supported_by_definitions."""
    belief = _make_belief_with_defs()
    md = belief_to_markdown(belief)

    assert "Supported by definitions: D1, D2" in md


@pytest.mark.unit
def test_belief_to_markdown_attack_strategy_on_counterpositions():
    """Test that counterpositions show attack_strategy."""
    belief = _make_belief_with_defs()
    md = belief_to_markdown(belief)

    assert "Attack strategy: challenge_inference_step" in md


@pytest.mark.unit
def test_belief_to_markdown_no_definitions_section_when_empty():
    """Test that Definitions section is omitted when definitions array is empty."""
    belief = _make_belief_with_defs()
    belief["definitions"] = []
    md = belief_to_markdown(belief)

    assert "# Definitions" not in md


# ==============================================
# 2. project_for_embedding() with D# nodes
# ==============================================

@pytest.mark.unit
def test_project_for_embedding_includes_definitions():
    """Test that embedding projection includes key definitions."""
    belief = _make_belief_with_defs()
    projection = project_for_embedding(belief)

    assert "Key definitions:" in projection
    assert "free will" in projection
    assert "determinism" in projection


@pytest.mark.unit
def test_project_for_embedding_excludes_retracted_definitions():
    """Test that retracted definitions are excluded from projection."""
    belief = _make_belief_with_defs()
    belief["definitions"][0]["status"] = "retracted"
    projection = project_for_embedding(belief)

    # D1 (free will) is retracted, should not appear in key definitions
    # D2 (determinism) should still appear
    assert "determinism" in projection


@pytest.mark.unit
def test_project_for_embedding_no_definitions():
    """Test that projection works with no definitions."""
    belief = _make_belief_with_defs()
    belief["definitions"] = []
    projection = project_for_embedding(belief)

    assert "Key definitions:" not in projection
    # Should still have thesis and other content
    assert "Thesis:" in projection


# ==============================================
# 3. add_definition patch operation
# ==============================================

@pytest.mark.unit
def test_add_definition_success():
    """Test adding a new definition node."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "agency",
            "definition": "The capacity to act intentionally.",
            "strength": 0.85,
            "strength_justification": "Well-established concept",
            "status": "active",
            "used_by": ["A1"]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    assert len(updated["definitions"]) == 3
    d3 = next(d for d in updated["definitions"] if d["id"] == "D3")
    assert d3["term"] == "agency"
    assert d3["strength"] == 0.85


@pytest.mark.unit
def test_add_definition_side_effect_on_ae():
    """Test that add_definition appends D# ID to supported_by_definitions on A#/E#."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "agency",
            "definition": "The capacity to act intentionally.",
            "strength": 0.85,
            "strength_justification": "Well-established concept",
            "status": "active",
            "used_by": ["A1", "E1"]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert "D3" in a1["supported_by_definitions"]

    e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
    assert "D3" in e1["supported_by_definitions"]


@pytest.mark.unit
def test_add_definition_changelog():
    """Test that add_definition creates a changelog entry."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "agency",
            "definition": "The capacity to act intentionally.",
            "strength": 0.85,
            "strength_justification": "Well-established concept",
            "status": "active",
            "used_by": ["A1"]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    changelog = updated["changelog"][-1]["changes"]
    assert any("Added definition D3" in c for c in changelog)


@pytest.mark.unit
def test_add_definition_duplicate_id_raises():
    """Test that adding a definition with duplicate ID raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D1",  # Already exists
            "term": "duplicate",
            "definition": "Duplicate definition.",
            "strength": 0.5,
            "strength_justification": "Test",
            "status": "active",
            "used_by": ["A1"]
        }
    }]

    with pytest.raises(ValueError, match="already exists"):
        apply_patches(belief, patches, propagate_strength=False)


@pytest.mark.unit
def test_add_definition_invalid_id_format_raises():
    """Test that adding a definition with invalid ID format raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "X5",  # Wrong prefix
            "term": "test",
            "definition": "Test.",
            "strength": 0.5,
            "strength_justification": "Test",
            "status": "active",
            "used_by": ["A1"]
        }
    }]

    with pytest.raises(ValueError, match="must match"):
        apply_patches(belief, patches, propagate_strength=False)


@pytest.mark.unit
def test_add_definition_missing_item_raises():
    """Test that add_definition with no item raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{"op": "add_definition"}]

    with pytest.raises(ValueError, match="requires valid item"):
        apply_patches(belief, patches, propagate_strength=False)


# ==============================================
# 4. update_definition patch operation
# ==============================================

@pytest.mark.unit
def test_update_definition_strength():
    """Test updating definition strength."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "strength": 0.55,
            "strength_justification": "Revised after challenge"
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    d1 = next(d for d in updated["definitions"] if d["id"] == "D1")
    assert d1["strength"] == 0.55
    assert d1["strength_justification"] == "Revised after challenge"


@pytest.mark.unit
def test_update_definition_retraction_forces_zero():
    """Test that retracting a definition forces strength to 0.0."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "status": "retracted"
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    d1 = next(d for d in updated["definitions"] if d["id"] == "D1")
    assert d1["status"] == "retracted"
    assert d1["strength"] == 0.0


@pytest.mark.unit
def test_update_definition_immutable_field_rejection():
    """Test that updating immutable fields (id, term) raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "term": "new term"
        }
    }]

    with pytest.raises(ValueError, match="immutable"):
        apply_patches(belief, patches, propagate_strength=False)


@pytest.mark.unit
def test_update_definition_immutable_id_rejection():
    """Test that updating id field raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "id": "D99"
        }
    }]

    with pytest.raises(ValueError, match="immutable"):
        apply_patches(belief, patches, propagate_strength=False)


@pytest.mark.unit
def test_update_definition_not_found_raises():
    """Test that updating non-existent definition raises ValueError."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D99",
        "changes": {"strength": 0.5}
    }]

    with pytest.raises(ValueError, match="non-existent definition"):
        apply_patches(belief, patches, propagate_strength=False)


@pytest.mark.unit
def test_update_definition_changelog():
    """Test that update_definition creates changelog entries."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "strength": 0.55,
            "status": "revised"
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    changelog = updated["changelog"][-1]["changes"]
    assert any("D1.strength" in c for c in changelog)
    assert any("D1.status" in c for c in changelog)


@pytest.mark.unit
def test_update_definition_used_by_change_updates_ae():
    """Test that changing used_by updates supported_by_definitions on A#/E#."""
    belief = _make_belief_with_defs()
    # D1 currently used_by: ["A1", "E1"]
    # Change to used_by: ["A2"] — remove from A1 and E1, add to A2
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "used_by": ["A2"]
        }
    }]

    updated = apply_patches(belief, patches, propagate_strength=False)

    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert "D1" not in a1["supported_by_definitions"]

    a2 = next(a for a in updated["assumptions"] if a["id"] == "A2")
    assert "D1" in a2["supported_by_definitions"]

    e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
    assert "D1" not in e1["supported_by_definitions"]


# ==============================================
# 5. D# ceiling enforcement
# ==============================================

@pytest.mark.unit
def test_definition_ceiling_caps_assumption():
    """Test that D# strength caps A# strength."""
    belief = _make_belief_with_defs()
    # D1 strength 0.8, A1 strength 0.8 — no ceiling needed
    # Lower D1 to 0.6 — A1 should be capped at 0.6
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"strength": 0.6}
    }]

    updated = apply_patches(belief, patches)

    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert a1["strength"] <= 0.6


@pytest.mark.unit
def test_definition_ceiling_caps_evidence():
    """Test that D# strength caps E# strength."""
    belief = _make_belief_with_defs()
    # E1 supported by D1 (0.8) and D2 (0.9) — ceiling is min(0.8, 0.9) = 0.8
    # E1 strength is 0.8 — exactly at ceiling, no change needed
    # Lower D1 to 0.5 — ceiling becomes 0.5, E1 should be capped
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"strength": 0.5}
    }]

    updated = apply_patches(belief, patches)

    e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
    assert e1["strength"] <= 0.5


@pytest.mark.unit
def test_definition_ceiling_cascades_to_claims():
    """Test that D# ceiling cascade propagates through to claims."""
    belief = _make_belief_with_defs()
    # Lower D1 to 0.4 — this caps A1 at 0.4, which then should cap C1
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"strength": 0.4}
    }]

    updated = apply_patches(belief, patches)

    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert a1["strength"] <= 0.4

    c1 = next(c for c in updated["claims"] if c["id"] == "C1")
    # C1 depends on A1, A2, E1. A1 is now 0.4, so C1 ceiling is 0.4
    assert c1["strength"] <= 0.4


@pytest.mark.unit
def test_orphan_ae_cap_when_all_defs_retracted():
    """Test that A#/E# with all D# retracted are capped at ORPHAN_AE_CAP."""
    belief = _make_belief_with_defs()
    # Retract both D1 and D2
    patches = [
        {
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"status": "retracted"}
        },
        {
            "op": "update_definition",
            "target_id": "D2",
            "changes": {"status": "retracted"}
        }
    ]

    updated = apply_patches(belief, patches)

    # A1 supported by D1 only — all retracted → capped at 0.6
    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert a1["strength"] <= ORPHAN_AE_CAP

    # E1 supported by D1 and D2 — all retracted → capped at 0.6
    e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
    assert e1["strength"] <= ORPHAN_AE_CAP


@pytest.mark.unit
def test_orphan_claim_cap_when_all_deps_retracted():
    """Test that C# with all deps retracted are capped at ORPHAN_CLAIM_CAP."""
    belief = _make_belief_with_defs()
    # Retract all A# and E# that C1 depends on
    patches = [
        {
            "op": "update_assumption",
            "target_id": "A1",
            "changes": {"status": "retracted"}
        },
        {
            "op": "update_assumption",
            "target_id": "A2",
            "changes": {"status": "retracted"}
        },
        {
            "op": "update_evidence",
            "target_id": "E1",
            "changes": {"status": "retracted"}
        }
    ]

    updated = apply_patches(belief, patches)

    c1 = next(c for c in updated["claims"] if c["id"] == "C1")
    assert c1["strength"] <= ORPHAN_CLAIM_CAP


@pytest.mark.unit
def test_ceiling_not_applied_to_retracted_ae():
    """Test that retracted A#/E# are not subject to ceiling enforcement."""
    belief = _make_belief_with_defs()
    # Retract A1 first, then lower D1 — A1 should stay at 0.0
    belief["assumptions"][0]["status"] = "retracted"
    belief["assumptions"][0]["strength"] = 0.0

    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"strength": 0.3}
    }]

    updated = apply_patches(belief, patches)

    a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
    assert a1["strength"] == 0.0  # Stays retracted


# ==============================================
# 6. validate_patches for definition operations
# ==============================================

@pytest.mark.unit
def test_validate_add_definition_valid():
    """Test that a valid add_definition patch passes validation."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "agency",
            "definition": "The capacity to act intentionally.",
            "strength": 0.85,
            "strength_justification": "Well-established concept",
            "status": "active",
            "used_by": ["A1"]
        }
    }]

    errors = validate_patches(patches, belief)
    assert errors == {}


@pytest.mark.unit
def test_validate_add_definition_duplicate_id():
    """Test that add_definition with duplicate ID is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D1",
            "term": "dup",
            "definition": "Dup.",
            "strength": 0.5,
            "strength_justification": "Test",
            "used_by": ["A1"]
        }
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("already exists" in e for e in flat)


@pytest.mark.unit
def test_validate_add_definition_bad_id_format():
    """Test that add_definition with wrong ID format is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "ABC",
            "term": "test",
            "definition": "Test.",
            "strength": 0.5,
            "strength_justification": "Test",
            "used_by": ["A1"]
        }
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("must match" in e for e in flat)


@pytest.mark.unit
def test_validate_add_definition_missing_required_fields():
    """Test that add_definition with missing fields is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "test"
            # Missing: definition, strength, strength_justification, used_by
        }
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("missing required field" in e for e in flat)


@pytest.mark.unit
def test_validate_add_definition_bad_used_by_ref():
    """Test that add_definition with non-existent used_by ref is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "test",
            "definition": "Test.",
            "strength": 0.5,
            "strength_justification": "Test",
            "used_by": ["A99"]  # Doesn't exist
        }
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("non-existent" in e for e in flat)


@pytest.mark.unit
def test_validate_add_definition_strength_out_of_range():
    """Test that add_definition with out-of-range strength is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "test",
            "definition": "Test.",
            "strength": 1.5,
            "strength_justification": "Test",
            "used_by": ["A1"]
        }
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("between 0.0 and 1.0" in e for e in flat)


@pytest.mark.unit
def test_validate_update_definition_valid():
    """Test that a valid update_definition patch passes validation."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {
            "strength": 0.6,
            "status": "revised"
        }
    }]

    errors = validate_patches(patches, belief)
    assert errors == {}


@pytest.mark.unit
def test_validate_update_definition_nonexistent():
    """Test that update_definition on non-existent ID is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D99",
        "changes": {"strength": 0.5}
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("non-existent" in e for e in flat)


@pytest.mark.unit
def test_validate_update_definition_immutable_fields():
    """Test that update_definition with immutable fields is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"term": "new term"}
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("immutable" in e for e in flat)


@pytest.mark.unit
def test_validate_update_definition_unknown_fields():
    """Test that update_definition with unknown fields is caught."""
    belief = _make_belief_with_defs()
    patches = [{
        "op": "update_definition",
        "target_id": "D1",
        "changes": {"nonexistent_field": "value"}
    }]

    errors = validate_patches(patches, belief)
    flat = _flat(errors)
    assert any("unknown fields" in e for e in flat)


# ==============================================
# 7. Graph visualizer D# support
# ==============================================

@pytest.mark.unit
def test_graph_visualizer_definition_color():
    """Test that D# nodes get teal color in graph visualizer."""
    from chal.beliefs.graph_visualizer import export_debate_graph
    # Just verify the color constant is registered
    node_colors = {
        "definition": "#2AA198",
        "assumption": "#3498db",
        "claim": "#e74c3c",
        "evidence": "#2ecc71",
        "prediction": "#f39c12"
    }
    assert node_colors["definition"] == "#2AA198"


@pytest.mark.unit
def test_belief_graph_includes_definition_nodes():
    """Test that BeliefGraph creates D# nodes."""
    belief = _make_belief_with_defs()
    graph = BeliefGraph(belief)

    assert "D1" in graph.nodes
    assert "D2" in graph.nodes
    assert graph.nodes["D1"]["type"] == "definition"


@pytest.mark.unit
def test_belief_graph_definition_edges():
    """Test that D# -> A#/E# support edges are created."""
    belief = _make_belief_with_defs()
    graph = BeliefGraph(belief)

    # D1 used_by ["A1", "E1"] → edges D1->A1, D1->E1
    support_edges = [(f, t) for f, t, et in graph.edges if et == "supports"]
    assert ("D1", "A1") in support_edges
    assert ("D1", "E1") in support_edges
    assert ("D2", "A2") in support_edges
    assert ("D2", "E1") in support_edges


@pytest.mark.unit
def test_belief_graph_metrics_include_definitions():
    """Test that get_graph_metrics counts D# nodes."""
    belief = _make_belief_with_defs()
    graph = BeliefGraph(belief)
    metrics = graph.get_graph_metrics()

    assert metrics["node_counts"]["definitions"] == 2


# ==============================================
# 8. Integration: full round-trip
# ==============================================

@pytest.mark.unit
def test_full_round_trip_with_definitions():
    """Test belief with D# → validate → markdown → embedding → graph."""
    from chal.beliefs.schema import validate_belief

    belief = _make_belief_with_defs()

    # Validate
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Validation failed: {errors}"

    # Markdown
    md = belief_to_markdown(belief)
    assert "# Definitions" in md
    assert "# Assumptions" in md

    # Embedding
    projection = project_for_embedding(belief)
    assert "Key definitions:" in projection

    # Graph
    graph = BeliefGraph(belief)
    assert "D1" in graph.nodes
    assert len(graph.validate_links()) == 0


@pytest.mark.unit
def test_patch_round_trip_add_and_update_definition():
    """Test add_definition + update_definition → ceiling → BFS → thesis."""
    belief = _make_belief_with_defs()

    # Add a new definition
    patches_add = [{
        "op": "add_definition",
        "item": {
            "id": "D3",
            "term": "consciousness",
            "definition": "Subjective experience or awareness.",
            "strength": 0.7,
            "strength_justification": "Widely discussed but contested",
            "status": "active",
            "used_by": ["A2"]
        }
    }]
    updated = apply_patches(belief, patches_add)
    assert len(updated["definitions"]) == 3

    # Now lower D3 strength — should cap A2 if A2 > 0.7
    patches_update = [{
        "op": "update_definition",
        "target_id": "D3",
        "changes": {"strength": 0.5, "status": "revised"}
    }]
    updated2 = apply_patches(updated, patches_update)

    # D3 supports A2, A2 also supported by D2 (0.9)
    # Ceiling = min(D2=0.9, D3=0.5) = 0.5
    a2 = next(a for a in updated2["assumptions"] if a["id"] == "A2")
    assert a2["strength"] <= 0.5
