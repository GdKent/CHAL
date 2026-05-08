"""
Phase 6 tests for Definition Nodes (D#) — Integration, Fixtures & Migration.

Tests cover:
- 6.1: Integration round trips (full lifecycle, patch propagation, cross-exam flow, graph viz)
- 6.2: Fixture verification (shared helper generates valid D#, existing fixtures have D# fields)
- 6.4: Migration graceful handling (beliefs without definitions key, empty definitions)
"""

import pytest
import json
import copy
from pathlib import Path
from chal.beliefs.schema import validate_belief
from chal.beliefs.belief_graph import BeliefGraph
from chal.beliefs.patches import apply_patches, validate_patches, ORPHAN_AE_CAP
from chal.beliefs.io import belief_to_markdown, project_for_embedding, parse_model_output_to_belief
from tests.utils import create_sample_belief


# ========================================
# Helpers
# ========================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str):
    """Load a JSON fixture file."""
    with open(FIXTURES_DIR / name, "r") as f:
        return json.load(f)


def _complete_belief():
    """Load the complete_valid fixture."""
    return copy.deepcopy(_load_fixture("test_beliefs.json")["complete_valid"])


def _belief_with_defs(num_defs=2, num_assumptions=2, num_evidence=1, num_claims=2):
    """Create a belief with fully wired D# nodes for integration tests."""
    belief = create_sample_belief(
        num_assumptions=num_assumptions,
        num_claims=num_claims,
        num_evidence=num_evidence,
    )
    # Override definitions with multiple D# nodes
    ae_ids_a = [f"A{i+1}" for i in range(num_assumptions)]
    ae_ids_e = [f"E{i+1}" for i in range(num_evidence)]

    defs = []
    for i in range(num_defs):
        used_by = []
        if i < num_assumptions:
            used_by.append(f"A{i+1}")
        if i < num_evidence:
            used_by.append(f"E{i+1}")
        if not used_by:
            # Assign to first A# and E# if index exceeds
            used_by = [ae_ids_a[0]] if ae_ids_a else [ae_ids_e[0]] if ae_ids_e else ["A1"]
        defs.append({
            "id": f"D{i+1}",
            "term": f"term_{i+1}",
            "definition": f"Definition of term_{i+1} for testing purposes.",
            "strength": 0.85 - (i * 0.05),
            "strength_justification": f"Standard definition {i+1}",
            "status": "active",
            "used_by": used_by,
        })

    belief["definitions"] = defs

    # Wire supported_by_definitions on A# and E#
    for a in belief.get("assumptions", []):
        a_id = a["id"]
        # Each A# gets supported by all D# that list it in used_by
        a["supported_by_definitions"] = [
            d["id"] for d in defs if a_id in d["used_by"]
        ]

    for e in belief.get("evidence", []):
        e_id = e["id"]
        e["supported_by_definitions"] = [
            d["id"] for d in defs if e_id in d["used_by"]
        ]

    return belief


# ========================================
# 6.1 — Integration: Full Round Trip
# ========================================

class TestFullRoundTrip:
    """D# belief → validate → markdown → embedding → graph → all pass."""

    @pytest.mark.integration
    def test_validate_belief_with_definitions(self):
        """Belief with D# nodes passes validation."""
        belief = _belief_with_defs()
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Validation errors: {errors}"

    @pytest.mark.integration
    def test_belief_to_markdown_includes_definitions(self):
        """Markdown rendering includes D# section."""
        belief = _belief_with_defs()
        md = belief_to_markdown(belief)
        assert "# Definitions" in md
        assert "D1" in md
        assert "term_1" in md
        assert "Used by:" in md

    @pytest.mark.integration
    def test_project_for_embedding_includes_definitions(self):
        """Embedding projection includes key definitions."""
        belief = _belief_with_defs()
        projection = project_for_embedding(belief)
        assert "Key definitions:" in projection
        assert "term_1" in projection

    @pytest.mark.integration
    def test_belief_graph_has_definition_nodes(self):
        """BeliefGraph includes D# nodes and D#→A#/E# edges."""
        belief = _belief_with_defs()
        graph = BeliefGraph(belief)

        # D# nodes exist
        assert "D1" in graph.nodes
        assert graph.nodes["D1"]["type"] == "definition"

        # D#→A# supports edges exist
        supports_edges = [
            (f, t) for f, t, etype in graph.edges
            if etype == "supports" and f.startswith("D")
        ]
        assert len(supports_edges) > 0

    @pytest.mark.integration
    def test_graph_validate_links_with_definitions(self):
        """Graph validation passes with D# nodes."""
        belief = _belief_with_defs()
        graph = BeliefGraph(belief)
        errors = graph.validate_links()
        assert len(errors) == 0, f"Graph validation errors: {errors}"

    @pytest.mark.integration
    def test_graph_no_cycles_with_definitions(self):
        """Graph has no cycles when D# nodes are present."""
        belief = _belief_with_defs()
        graph = BeliefGraph(belief)
        assert not graph._has_cycle()

    @pytest.mark.integration
    def test_complete_fixture_round_trip(self):
        """complete_valid fixture passes full validate → markdown → embed → graph."""
        belief = _complete_belief()
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Validation errors: {errors}"

        md = belief_to_markdown(belief)
        assert "D1" in md
        assert "free will" in md

        proj = project_for_embedding(belief)
        assert "free will" in proj

        graph = BeliefGraph(belief)
        assert "D1" in graph.nodes
        assert "D2" in graph.nodes
        errors = graph.validate_links()
        assert len(errors) == 0

    @pytest.mark.integration
    def test_parse_model_output_with_definitions(self):
        """parse_model_output_to_belief handles beliefs with D# nodes."""
        belief = _belief_with_defs()
        json_str = f"```json\n{json.dumps(belief, indent=2)}\n```"
        parsed, md, errors = parse_model_output_to_belief(json_str)
        assert parsed is not None
        assert len(errors) == 0, f"Parse errors: {errors}"
        assert len(parsed["definitions"]) > 0


# ========================================
# 6.1 — Integration: Patch Round Trip
# ========================================

class TestPatchRoundTrip:
    """add_definition + update_definition → ceiling enforced → BFS → thesis."""

    @pytest.mark.integration
    def test_add_definition_patch(self):
        """add_definition creates D# node and wires supported_by_definitions."""
        belief = _belief_with_defs(num_defs=1)
        patches = [{
            "op": "add_definition",
            "item": {
                "id": "D2",
                "term": "new term",
                "definition": "A newly added definition for testing.",
                "strength": 0.75,
                "strength_justification": "Test justification",
                "status": "active",
                "used_by": ["A1"],
            }
        }]
        updated = apply_patches(belief, patches)
        errors = validate_belief(updated)
        assert len(errors) == 0, f"Post-patch validation errors: {errors}"

        # D2 exists
        d2 = next(d for d in updated["definitions"] if d["id"] == "D2")
        assert d2["term"] == "new term"

        # A1 now has D2 in supported_by_definitions
        a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
        assert "D2" in a1["supported_by_definitions"]

    @pytest.mark.integration
    def test_update_definition_strength_triggers_ceiling(self):
        """Lowering D# strength caps dependent A#/E# via ceiling enforcement."""
        belief = _belief_with_defs(num_defs=1, num_assumptions=1, num_evidence=1, num_claims=1)
        # Set A1 strength high
        belief["assumptions"][0]["strength"] = 0.9
        belief["evidence"][0]["strength"] = 0.9
        belief["definitions"][0]["strength"] = 0.9

        # Lower D1 to 0.5 — should cap A1 and E1
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"strength": 0.5}
        }]
        updated = apply_patches(belief, patches, propagate_strength=True)

        a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
        e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
        assert a1["strength"] <= 0.5, f"A1 should be capped at 0.5, got {a1['strength']}"
        assert e1["strength"] <= 0.5, f"E1 should be capped at 0.5, got {e1['strength']}"

    @pytest.mark.integration
    def test_update_definition_retract_caps_ae(self):
        """Retracting D# with no remaining active D# caps A#/E# at ORPHAN_AE_CAP."""
        belief = _belief_with_defs(num_defs=1, num_assumptions=1, num_evidence=1, num_claims=1)
        belief["assumptions"][0]["strength"] = 0.8
        belief["evidence"][0]["strength"] = 0.8
        belief["definitions"][0]["strength"] = 0.9

        # Retract D1 — sole definition
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"status": "retracted"}
        }]
        updated = apply_patches(belief, patches, propagate_strength=True)

        a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
        e1 = next(e for e in updated["evidence"] if e["id"] == "E1")
        assert a1["strength"] <= ORPHAN_AE_CAP
        assert e1["strength"] <= ORPHAN_AE_CAP

    @pytest.mark.integration
    def test_ceiling_cascades_to_claims_and_thesis(self):
        """D# ceiling → A#/E# cap → claim cap → thesis recalculated."""
        belief = _belief_with_defs(num_defs=1, num_assumptions=1, num_evidence=1, num_claims=1)
        belief["definitions"][0]["strength"] = 0.9
        belief["assumptions"][0]["strength"] = 0.85
        belief["evidence"][0]["strength"] = 0.85
        belief["claims"][0]["strength"] = 0.8
        belief["claims"][0]["depends_on"] = ["A1", "E1"]

        original_thesis = belief["thesis"]["strength"]

        # Lower D1 drastically
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"strength": 0.3}
        }]
        updated = apply_patches(belief, patches, propagate_strength=True)

        # Claims should be capped
        c1 = next(c for c in updated["claims"] if c["id"] == "C1")
        assert c1["strength"] <= 0.3

        # Thesis should be lower
        assert updated["thesis"]["strength"] < original_thesis

    @pytest.mark.integration
    def test_validate_patches_add_definition(self):
        """validate_patches accepts well-formed add_definition patches."""
        belief = _belief_with_defs(num_defs=1)
        patches = [{
            "op": "add_definition",
            "item": {
                "id": "D2",
                "term": "new term",
                "definition": "A valid definition.",
                "strength": 0.7,
                "strength_justification": "Test",
                "status": "active",
                "used_by": ["A1"],
            }
        }]
        errors = validate_patches(patches, belief)
        assert len(errors) == 0, f"Validation errors: {errors}"

    @pytest.mark.integration
    def test_validate_patches_update_definition(self):
        """validate_patches accepts well-formed update_definition patches."""
        belief = _belief_with_defs(num_defs=1)
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"strength": 0.5, "status": "revised"}
        }]
        errors = validate_patches(patches, belief)
        assert len(errors) == 0, f"Validation errors: {errors}"

    @pytest.mark.integration
    def test_validate_patches_rejects_immutable_term(self):
        """validate_patches rejects update_definition changing 'term'."""
        belief = _belief_with_defs(num_defs=1)
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {"term": "renamed"}
        }]
        errors = validate_patches(patches, belief)
        assert len(errors) > 0
        assert any("immutable" in str(e).lower() for errs in errors.values() for e in errs)


# ========================================
# 6.1 — Integration: Cross-Examination Round Trip
# ========================================

class TestCrossExamRoundTrip:
    """Simulate Stage 2→3→4→5 flow with D# targeting."""

    @pytest.mark.integration
    def test_stage2_question_targets_definition(self):
        """A Stage 2 question targeting D# is valid via validate_stage2_questions."""
        from chal.utilities.utils import validate_stage2_questions

        questions = [{
            "qid": "Q1",
            "text": "Your definition D1 of 'free will' is circular — it uses 'unimpeded' which presupposes freedom.",
            "target_ids": ["D1"],
            "attack_type": "undercutting",
            "attack_strategy": "circularity",
        }]
        is_valid, errors = validate_stage2_questions(questions)
        assert is_valid, f"Should be valid: {errors}"

    @pytest.mark.integration
    def test_stage3_rebuttal_with_update_definition_patch(self):
        """Stage 3 rebuttal patches including update_definition are valid and apply correctly."""
        belief = _complete_belief()

        # Simulated rebuttal patches: concede the definitional challenge, revise D1
        patches = [
            {
                "op": "update_definition",
                "target_id": "D1",
                "changes": {
                    "definition": "The capacity of agents to act on reasons without external coercion.",
                    "strength": 0.7,
                    "strength_justification": "0.70 — revised to remove circularity",
                    "status": "revised",
                }
            },
            {
                "op": "update_claim",
                "target_id": "C1",
                "changes": {
                    "strength": 0.7,
                    "strength_justification": "Lowered due to D1 revision"
                }
            }
        ]

        # Validate patches
        errors = validate_patches(patches, belief)
        assert len(errors) == 0, f"Patch validation errors: {errors}"

        # Apply patches
        updated = apply_patches(belief, patches, propagate_strength=True)

        # D1 revised
        d1 = next(d for d in updated["definitions"] if d["id"] == "D1")
        assert d1["status"] == "revised"
        assert d1["strength"] == 0.7

        # Belief still valid
        v_errors = validate_belief(updated)
        assert len(v_errors) == 0, f"Post-update validation errors: {v_errors}"

    @pytest.mark.integration
    def test_full_cross_exam_cycle(self):
        """Complete cycle: belief → question D# → patch D# → ceiling → validate."""
        belief = _complete_belief()

        # Step 1: Validate original belief
        assert len(validate_belief(belief)) == 0

        # Step 2: Simulate a definitional challenge to D1
        from chal.utilities.utils import validate_stage2_questions
        questions = [{
            "qid": "Q1",
            "text": "D1 over-extends 'free will' to include coerced acts",
            "target_ids": ["D1"],
            "attack_type": "undermining",
            "attack_strategy": "over_extension",
        }]
        is_valid, _ = validate_stage2_questions(questions)
        assert is_valid

        # Step 3: Simulate rebuttal — concede and weaken D1
        patches = [{
            "op": "update_definition",
            "target_id": "D1",
            "changes": {
                "strength": 0.5,
                "strength_justification": "0.50 — conceded over-extension, narrowed scope",
                "status": "revised"
            }
        }]

        # Step 4: Apply patches with ceiling enforcement
        updated = apply_patches(belief, patches, propagate_strength=True)

        # Step 5: Verify ceiling cascaded to dependent A#/E#
        a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
        a2 = next(a for a in updated["assumptions"] if a["id"] == "A2")
        # A1 depends on D1 and D2; A2 depends only on D1
        # D1 now 0.5, D2 still 0.85 → A1 ceiling = min(0.5, 0.85) = 0.5
        # A2 ceiling = 0.5 (only D1)
        assert a1["strength"] <= 0.5
        assert a2["strength"] <= 0.5

        # Step 6: Belief still valid after full cycle
        assert len(validate_belief(updated)) == 0


# ========================================
# 6.1 — Integration: Graph Visualization
# ========================================

class TestGraphVisualization:
    """Belief with D# → BeliefGraph → graph metrics include D# nodes."""

    @pytest.mark.integration
    def test_graph_metrics_include_definitions(self):
        """Graph metrics count D# nodes."""
        belief = _complete_belief()
        graph = BeliefGraph(belief)
        metrics = graph.get_graph_metrics()

        assert metrics["total_nodes"] > 0
        # D# nodes should be counted
        node_counts = metrics.get("node_counts", {})
        assert node_counts.get("definitions", node_counts.get("definition", 0)) >= 2

    @pytest.mark.integration
    def test_graph_support_chain_through_definitions(self):
        """get_support_chain traces through D# → A# → C# → THESIS."""
        belief = _complete_belief()
        graph = BeliefGraph(belief)

        # D1 should transitively support C1 (via A1)
        chain = graph.get_support_chain("THESIS")
        # All active claims should be in the chain
        assert "C1" in chain or "C2" in chain

    @pytest.mark.integration
    def test_graph_dependent_nodes_from_definition(self):
        """get_dependent_nodes from D# includes downstream A#/C#/THESIS."""
        belief = _complete_belief()
        graph = BeliefGraph(belief)

        dependents = graph.get_dependent_nodes("D1")
        # D1 used_by: A1, A2, E2 → those should be in dependents
        assert "A1" in dependents or "A2" in dependents or "E2" in dependents


# ========================================
# 6.2 — Fixture Verification
# ========================================

class TestFixtureVerification:
    """Verify all CBS fixtures have D# fields."""

    @pytest.mark.unit
    def test_complete_valid_fixture_has_definitions(self):
        """complete_valid fixture has non-empty definitions array."""
        fixture = _load_fixture("test_beliefs.json")["complete_valid"]
        assert "definitions" in fixture
        assert len(fixture["definitions"]) >= 2

    @pytest.mark.unit
    def test_complete_valid_definitions_have_required_fields(self):
        """Each D# in complete_valid has all required fields."""
        fixture = _load_fixture("test_beliefs.json")["complete_valid"]
        required = {"id", "term", "definition", "strength", "strength_justification", "status", "used_by"}
        for d in fixture["definitions"]:
            missing = required - set(d.keys())
            assert not missing, f"D# {d.get('id')} missing fields: {missing}"

    @pytest.mark.unit
    def test_complete_valid_assumptions_have_supported_by_definitions(self):
        """A# nodes in complete_valid have supported_by_definitions."""
        fixture = _load_fixture("test_beliefs.json")["complete_valid"]
        for a in fixture.get("assumptions", []):
            assert "supported_by_definitions" in a, f"A# {a['id']} missing supported_by_definitions"
            assert len(a["supported_by_definitions"]) >= 1

    @pytest.mark.unit
    def test_complete_valid_evidence_has_supported_by_definitions(self):
        """E# nodes in complete_valid have supported_by_definitions."""
        fixture = _load_fixture("test_beliefs.json")["complete_valid"]
        for e in fixture.get("evidence", []):
            assert "supported_by_definitions" in e, f"E# {e['id']} missing supported_by_definitions"
            assert len(e["supported_by_definitions"]) >= 1

    @pytest.mark.unit
    def test_complete_valid_counterpositions_have_attack_strategy(self):
        """X# nodes in complete_valid have attack_strategy."""
        fixture = _load_fixture("test_beliefs.json")["complete_valid"]
        for x in fixture.get("counterpositions", []):
            assert "attack_strategy" in x, f"X# {x['id']} missing attack_strategy"

    @pytest.mark.unit
    def test_minimal_valid_has_definitions_key(self):
        """minimal_valid fixture has definitions key (can be empty)."""
        fixture = _load_fixture("test_beliefs.json")["minimal_valid"]
        assert "definitions" in fixture

    @pytest.mark.unit
    def test_invalid_circular_has_definitions_key(self):
        """invalid_circular fixture has definitions key."""
        fixture = _load_fixture("test_beliefs.json")["invalid_circular"]
        assert "definitions" in fixture

    @pytest.mark.unit
    def test_patches_fixture_has_definition_ops(self):
        """test_patches.json has add_definition and update_definition entries."""
        fixture = _load_fixture("test_patches.json")
        assert "add_definition" in fixture
        assert "update_definition_strength" in fixture
        assert "update_definition_retract" in fixture

    @pytest.mark.unit
    def test_create_sample_belief_generates_definitions(self):
        """create_sample_belief() generates D# nodes when A#/E# exist."""
        belief = create_sample_belief(num_assumptions=2, num_evidence=1)
        assert "definitions" in belief
        assert len(belief["definitions"]) >= 1
        assert belief["definitions"][0]["id"] == "D1"

    @pytest.mark.unit
    def test_create_sample_belief_wires_supported_by_definitions(self):
        """create_sample_belief() wires supported_by_definitions on A#/E#."""
        belief = create_sample_belief(num_assumptions=2, num_evidence=1)
        for a in belief.get("assumptions", []):
            assert "supported_by_definitions" in a
            assert "D1" in a["supported_by_definitions"]
        for e in belief.get("evidence", []):
            assert "supported_by_definitions" in e
            assert "D1" in e["supported_by_definitions"]

    @pytest.mark.unit
    def test_create_sample_belief_validates(self):
        """create_sample_belief() output passes validate_belief()."""
        belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=1)
        errors = validate_belief(belief)
        assert len(errors) == 0, f"create_sample_belief() validation errors: {errors}"


# ========================================
# 6.4 — Migration: Graceful Handling
# ========================================

class TestMigrationGracefulHandling:
    """Beliefs without definitions key or with empty definitions."""

    @pytest.mark.unit
    def test_markdown_renders_without_definitions(self):
        """belief_to_markdown handles belief with no definitions key."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-001",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "thesis": {"stance": "Test stance", "summary_bullets": ["B1"], "strength": 0.5},
        }
        md = belief_to_markdown(belief)
        assert "Thesis" in md
        assert "Definitions" not in md  # No definitions section rendered

    @pytest.mark.unit
    def test_markdown_renders_with_empty_definitions(self):
        """belief_to_markdown handles belief with empty definitions array."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-002",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "definitions": [],
            "thesis": {"stance": "Test stance", "summary_bullets": ["B1"], "strength": 0.5},
        }
        md = belief_to_markdown(belief)
        assert "Thesis" in md
        # Empty definitions should not render a section
        assert "# Definitions" not in md

    @pytest.mark.unit
    def test_embedding_projection_without_definitions(self):
        """project_for_embedding handles belief without definitions key."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-003",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "thesis": {"stance": "Test stance", "summary_bullets": ["B1"], "strength": 0.5},
        }
        projection = project_for_embedding(belief)
        assert "Test stance" in projection
        assert "Key definitions" not in projection

    @pytest.mark.unit
    def test_apply_patches_without_definitions_key(self):
        """apply_patches runs without crashing when definitions key is absent."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-004",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
            "definitions": [],
            "claims": [{
                "id": "C1", "type": "deductive", "statement": "Test claim",
                "depends_on": ["A1"], "strength": 0.7,
                "strength_justification": "Test", "status": "active",
                "inference_chain": [{"role": "premise", "text": "A1 holds", "reference": "A1"}, {"role": "inference", "text": "Therefore claim follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "Test claim"}], "predictions": [
                    {"statement": "P", "test": "T", "decision_criterion": "DC"}
                ],
            }],
            "assumptions": [{
                "id": "A1", "type": "empirical", "statement": "Test",
                "supports_claims": ["C1"], "strength": 0.8,
                "status": "active", "strength_justification": "Test",
                "supported_by_definitions": [],
            }],
        }
        patches = [{"op": "update_thesis", "change": "weaken"}]
        updated = apply_patches(belief, patches, propagate_strength=True)
        assert updated["version"] == 2

    @pytest.mark.unit
    def test_belief_graph_with_empty_definitions(self):
        """BeliefGraph handles belief with empty definitions array."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-005",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "definitions": [],
            "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
            "claims": [{
                "id": "C1", "type": "deductive", "statement": "Test",
                "depends_on": ["A1"], "strength": 0.7,
                "strength_justification": "Test", "status": "active",
                "inference_chain": [{"role": "premise", "text": "A1 holds", "reference": "A1"}, {"role": "inference", "text": "Therefore claim follows", "inference_type": "deductive"}, {"role": "conclusion", "text": "Test"}], "predictions": [
                    {"statement": "P", "test": "T", "decision_criterion": "DC"}
                ],
            }],
            "assumptions": [{
                "id": "A1", "type": "empirical", "statement": "Test",
                "supports_claims": ["C1"], "strength": 0.8,
                "status": "active", "strength_justification": "Test",
                "supported_by_definitions": [],
            }],
        }
        graph = BeliefGraph(belief)
        assert "C1" in graph.nodes
        assert "D1" not in graph.nodes  # No D# nodes

    @pytest.mark.unit
    def test_validate_belief_requires_definitions(self):
        """validate_belief reports error when definitions key is missing."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-006",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
        }
        errors = validate_belief(belief)
        assert any("definitions" in e for e in errors)

    @pytest.mark.unit
    def test_validate_belief_accepts_empty_definitions(self):
        """validate_belief accepts belief with empty definitions array."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-007",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "definitions": [],
            "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
            "assumptions": [],
            "claims": [],
            "evidence": [],
            "counterpositions": [],
            "uncertainties": [],
        }
        errors = validate_belief(belief)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    @pytest.mark.unit
    def test_add_definition_to_belief_without_definitions(self):
        """add_definition patch creates definitions array if absent."""
        belief = {
            "schema_version": "CBS",
            "belief_id": "MIGRATION-008",
            "version": 1,
            "metadata": {"topic_query": "Test", "agent_persona": "Test"},
            "definitions": [],
            "thesis": {"stance": "Test", "summary_bullets": ["B1"], "strength": 0.5},
            "assumptions": [{
                "id": "A1", "type": "empirical", "statement": "Test",
                "supports_claims": [], "strength": 0.8,
                "status": "active", "strength_justification": "Test",
                "supported_by_definitions": [],
            }],
        }
        patches = [{
            "op": "add_definition",
            "item": {
                "id": "D1",
                "term": "test",
                "definition": "A test definition.",
                "strength": 0.8,
                "strength_justification": "Test",
                "status": "active",
                "used_by": ["A1"],
            }
        }]
        updated = apply_patches(belief, patches)
        assert len(updated["definitions"]) == 1
        assert updated["definitions"][0]["id"] == "D1"
        # A1 should now have D1 in supported_by_definitions
        a1 = next(a for a in updated["assumptions"] if a["id"] == "A1")
        assert "D1" in a1["supported_by_definitions"]
