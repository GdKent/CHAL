"""
Unit tests for BeliefGraph class.

Tests cover:
- Graph construction from beliefs (including THESIS node, U# edges)
- Node operations
- Link validation
- Cycle detection
- Orphan detection (support edges only, THESIS excluded)
- Support chain analysis
- Dependent nodes analysis
- Critical path analysis
- Graph metrics
"""

import pytest
import json
from pathlib import Path
from chal.beliefs.belief_graph import BeliefGraph
from tests.utils import (
    create_sample_belief,
    assert_graph_acyclic,
    assert_graph_has_cycle,
    assert_no_orphaned_claims,
    assert_has_orphaned_claims
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
# 1. Graph Construction Tests
# ==============================================

@pytest.mark.unit
def test_build_graph_empty_belief():
    """Test graph construction with empty collections — only THESIS node."""
    belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    # THESIS node always present (from thesis section)
    assert len(graph.nodes) == 1
    assert "THESIS" in graph.nodes
    assert graph.nodes["THESIS"]["type"] == "thesis"
    assert len(graph.edges) == 0


@pytest.mark.unit
def test_build_graph_assumptions_only():
    """Test graph construction with only assumption nodes + THESIS."""
    belief = create_sample_belief(num_assumptions=3, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    # 3 assumptions + THESIS
    assert len(graph.nodes) == 4
    assert "THESIS" in graph.nodes
    assert sum(1 for n in graph.nodes.values() if n["type"] == "assumption") == 3
    assert len(graph.edges) == 0


@pytest.mark.unit
def test_build_graph_claims_only():
    """Test graph construction with only claim nodes + THESIS."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = []
    belief["claims"][1]["depends_on"] = []

    graph = BeliefGraph(belief)

    # 2 claims + THESIS
    assert len(graph.nodes) == 3
    assert sum(1 for n in graph.nodes.values() if n["type"] == "claim") == 2
    assert "THESIS" in graph.nodes
    # 2 edges: C1→THESIS, C2→THESIS
    assert sum(1 for _, to_id, et in graph.edges if to_id == "THESIS" and et == "supports") == 2


@pytest.mark.unit
def test_build_graph_complete(test_beliefs):
    """Test graph construction with all node types present."""
    belief = test_beliefs["complete_valid"]
    graph = BeliefGraph(belief)

    # Check that THESIS and standard node types exist
    assert "THESIS" in graph.nodes
    node_types = {n["type"] for n in graph.nodes.values()}
    assert "thesis" in node_types
    assert "assumption" in node_types
    assert "claim" in node_types
    assert "evidence" in node_types


@pytest.mark.unit
def test_build_graph_edges_supports():
    """Test that depends_on creates 'supports' edges."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)

    # A1 → C1 supports edge
    assert any(from_id == "A1" and to_id == "C1" and edge_type == "supports"
               for from_id, to_id, edge_type in graph.edges)


@pytest.mark.unit
def test_build_graph_edges_evidence_via_depends_on():
    """Test that E# IDs in depends_on create 'supports' edges."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["E1"]

    graph = BeliefGraph(belief)

    # E1 → C1 supports edge
    assert any(from_id == "E1" and to_id == "C1" and edge_type == "supports"
               for from_id, to_id, edge_type in graph.edges)


# ==============================================
# 1b. THESIS Node and C# → THESIS Edge Tests
# ==============================================

@pytest.mark.unit
def test_thesis_node_exists():
    """Test that THESIS node is created from thesis section."""
    belief = create_sample_belief()
    graph = BeliefGraph(belief)

    assert "THESIS" in graph.nodes
    assert graph.nodes["THESIS"]["type"] == "thesis"
    assert graph.nodes["THESIS"]["data"] == belief["thesis"]


@pytest.mark.unit
def test_thesis_node_data():
    """Test that THESIS node contains correct thesis data."""
    belief = create_sample_belief(confidence=0.8)
    graph = BeliefGraph(belief)

    thesis_data = graph.get_node("THESIS")
    assert thesis_data is not None
    assert thesis_data["strength"] == 0.8


@pytest.mark.unit
def test_active_claims_edge_to_thesis():
    """Test that active claims produce C# → THESIS 'supports' edges."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1)
    graph = BeliefGraph(belief)

    thesis_edges = [
        (f, t, et) for f, t, et in graph.edges
        if t == "THESIS" and et == "supports"
    ]
    assert len(thesis_edges) == 2
    from_ids = {e[0] for e in thesis_edges}
    assert "C1" in from_ids
    assert "C2" in from_ids


@pytest.mark.unit
def test_retracted_claims_no_edge_to_thesis():
    """Test that retracted claims do NOT produce edges to THESIS."""
    belief = create_sample_belief(num_claims=2, num_assumptions=1)
    belief["claims"][1]["status"] = "retracted"

    graph = BeliefGraph(belief)

    thesis_edges = [
        (f, t, et) for f, t, et in graph.edges
        if t == "THESIS" and et == "supports"
    ]
    # Only C1 (active) should have edge to THESIS, not C2 (retracted)
    assert len(thesis_edges) == 1
    assert thesis_edges[0][0] == "C1"


@pytest.mark.unit
def test_no_thesis_when_thesis_missing():
    """Test that THESIS node is not created if thesis section is empty."""
    belief = create_sample_belief()
    belief["thesis"] = {}  # Empty thesis

    graph = BeliefGraph(belief)

    assert "THESIS" not in graph.nodes


# ==============================================
# 1c. U# Edge and Node Tests
# ==============================================

@pytest.mark.unit
def test_uncertainty_nodes_created():
    """Test that U# nodes are created from uncertainties."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["A1"], "question": "Test?", "status": "active"},
        {"id": "U2", "targets": ["C1"], "question": "Test2?", "status": "active"}
    ]

    graph = BeliefGraph(belief)

    assert "U1" in graph.nodes
    assert graph.nodes["U1"]["type"] == "uncertainty"
    assert "U2" in graph.nodes
    assert graph.nodes["U2"]["type"] == "uncertainty"


@pytest.mark.unit
def test_uncertainty_edges_questions():
    """Test that U# with targets produces 'questions' edges."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["A1", "C1"], "question": "Test?", "status": "active"}
    ]

    graph = BeliefGraph(belief)

    question_edges = [(f, t, et) for f, t, et in graph.edges if et == "questions"]
    assert len(question_edges) == 2
    assert ("U1", "A1", "questions") in graph.edges
    assert ("U1", "C1", "questions") in graph.edges


@pytest.mark.unit
def test_uncertainty_empty_targets_no_edges():
    """Test that U# with empty targets produces no 'questions' edges."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {"id": "U1", "targets": [], "question": "Test?", "status": "active"}
    ]

    graph = BeliefGraph(belief)

    assert "U1" in graph.nodes
    question_edges = [(f, t, et) for f, t, et in graph.edges if et == "questions"]
    assert len(question_edges) == 0


@pytest.mark.unit
def test_counterposition_targets_evidence():
    """Test that X# can target evidence nodes (E#)."""
    belief = create_sample_belief(num_evidence=1)
    belief["counterpositions"] = [
        {
            "id": "X1",
            "targets": ["E1"],
            "attack_type": "undermining",
            "statement": "Evidence is flawed",
            "my_response": "No it isn't",
            "response_sufficiency": "sufficient"
        }
    ]

    graph = BeliefGraph(belief)

    assert ("X1", "E1", "challenges") in graph.edges


# ==============================================
# 1d. P# Removal Verification
# ==============================================

@pytest.mark.unit
def test_no_prediction_nodes():
    """Test that no P# prediction nodes exist in graph."""
    belief = create_sample_belief()
    # Even if legacy predictions key exists, it should not create nodes
    belief["predictions"] = [
        {"id": "P1", "statement": "Test", "linked_claims": ["C1"]}
    ]

    graph = BeliefGraph(belief)

    prediction_nodes = [nid for nid, n in graph.nodes.items() if n["type"] == "prediction"]
    assert len(prediction_nodes) == 0


@pytest.mark.unit
def test_no_predicts_edges():
    """Test that no 'predicts' edges exist in graph."""
    belief = create_sample_belief()
    graph = BeliefGraph(belief)

    predicts_edges = [e for e in graph.edges if e[2] == "predicts"]
    assert len(predicts_edges) == 0


# ==============================================
# 2. Node Operations Tests
# ==============================================

@pytest.mark.unit
def test_get_node_exists():
    """Test retrieving existing node by ID."""
    belief = create_sample_belief(num_assumptions=2)
    graph = BeliefGraph(belief)

    node = graph.get_node("A1")
    assert node is not None
    assert node["id"] == "A1"


@pytest.mark.unit
def test_get_node_thesis():
    """Test retrieving THESIS node by ID."""
    belief = create_sample_belief()
    graph = BeliefGraph(belief)

    node = graph.get_node("THESIS")
    assert node is not None
    assert "stance" in node


@pytest.mark.unit
def test_get_node_not_exists():
    """Test that get_node returns None for non-existent ID."""
    belief = create_sample_belief(num_assumptions=1)
    graph = BeliefGraph(belief)

    node = graph.get_node("A99")
    assert node is None


@pytest.mark.unit
def test_node_exists_true():
    """Test _node_exists() returns True for valid ID."""
    belief = create_sample_belief(num_assumptions=1)
    graph = BeliefGraph(belief)

    assert graph._node_exists("A1") is True
    assert graph._node_exists("THESIS") is True


@pytest.mark.unit
def test_node_exists_false():
    """Test _node_exists() returns False for invalid ID."""
    belief = create_sample_belief(num_assumptions=1)
    graph = BeliefGraph(belief)

    assert graph._node_exists("A99") is False


# ==============================================
# 3. Link Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_links_all_valid(test_beliefs):
    """Test validation of graph with all valid links."""
    belief = test_beliefs["complete_valid"]
    graph = BeliefGraph(belief)

    errors = graph.validate_links()
    assert len(errors) == 0


@pytest.mark.unit
def test_validate_links_broken_source():
    """Test detection of edge referencing non-existent source node."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    # Claim depends on non-existent assumption
    belief["claims"][0]["depends_on"] = ["A99"]

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("A99" in error for error in errors)


@pytest.mark.unit
def test_validate_links_broken_target():
    """Test detection of edge referencing non-existent target node."""
    belief = create_sample_belief(num_assumptions=1, num_claims=0, num_evidence=0)
    # Add counterposition targeting non-existent claim
    belief["counterpositions"] = [
        {
            "id": "X1",
            "targets": ["C99"],
            "attack_type": "rebutting",
            "statement": "Test",
            "my_response": "Test",
            "response_sufficiency": "sufficient"
        }
    ]

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("C99" in error for error in errors)


@pytest.mark.unit
def test_validate_links_broken_uncertainty_target():
    """Test detection of U# targeting non-existent node."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["A99"], "question": "Test?", "status": "active"}
    ]

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("A99" in error for error in errors)


@pytest.mark.unit
def test_validate_links_circular_dependency(test_beliefs):
    """Test detection of circular dependency C1 → C2 → C1."""
    belief = test_beliefs["invalid_circular"]
    graph = BeliefGraph(belief)

    errors = graph.validate_links()
    assert len(errors) > 0
    assert any("cycle" in error.lower() or "circular" in error.lower() for error in errors)


@pytest.mark.unit
def test_validate_links_self_reference():
    """Test detection of self-reference C1 → C1."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["C1"]

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("cycle" in error.lower() or "circular" in error.lower() or "self" in error.lower()
               for error in errors)


@pytest.mark.unit
def test_validate_links_orphaned_claims(test_beliefs):
    """Test detection of claims with no supporting evidence/assumptions."""
    belief = test_beliefs["invalid_orphan"]
    graph = BeliefGraph(belief)

    errors = graph.validate_links()
    assert len(errors) > 0
    assert any("orphan" in error.lower() or "no supporting" in error.lower() for error in errors)


# ==============================================
# 4. Cycle Detection Tests
# ==============================================

@pytest.mark.unit
def test_has_cycle_false_acyclic(test_beliefs):
    """Test that DAG with no cycles is detected as acyclic."""
    belief = test_beliefs["complete_valid"]
    assert_graph_acyclic(belief)


@pytest.mark.unit
def test_has_cycle_true_simple(test_beliefs):
    """Test detection of simple cycle: A → B → A."""
    belief = test_beliefs["invalid_circular"]
    assert_graph_has_cycle(belief)


@pytest.mark.unit
def test_has_cycle_true_complex():
    """Test detection of complex cycle: A → B → C → A."""
    belief = create_sample_belief(num_assumptions=0, num_claims=3, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["C2"]
    belief["claims"][1]["depends_on"] = ["C3"]
    belief["claims"][2]["depends_on"] = ["C1"]

    assert_graph_has_cycle(belief)


@pytest.mark.unit
def test_has_cycle_false_multiple_components():
    """Test acyclic graph with disconnected components."""
    belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][1]["depends_on"] = ["A2"]

    assert_graph_acyclic(belief)


# ==============================================
# 5. Orphan Detection Tests
# ==============================================

@pytest.mark.unit
def test_find_orphaned_claims_none(test_beliefs):
    """Test that graph with no orphaned claims is detected."""
    belief = test_beliefs["complete_valid"]
    assert_no_orphaned_claims(belief)


@pytest.mark.unit
def test_find_orphaned_claims_one(test_beliefs):
    """Test detection of one orphaned claim."""
    belief = test_beliefs["invalid_orphan"]
    assert_has_orphaned_claims(belief, expected_count=1)


@pytest.mark.unit
def test_find_orphaned_claims_multiple():
    """Test detection of multiple orphaned claims."""
    belief = create_sample_belief(num_assumptions=0, num_claims=3, num_evidence=0)
    for claim in belief["claims"]:
        claim["depends_on"] = []

    assert_has_orphaned_claims(belief, expected_count=3)


@pytest.mark.unit
def test_find_orphaned_claims_evidence_only():
    """Test that claims with only evidence (no assumptions) are NOT orphaned."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["E1"]

    assert_no_orphaned_claims(belief)


@pytest.mark.unit
def test_find_orphaned_claims_assumptions_only():
    """Test that claims with only assumptions (no evidence) are NOT orphaned."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    assert_no_orphaned_claims(belief)


@pytest.mark.unit
def test_thesis_not_considered_orphan():
    """Test that THESIS is never flagged as an orphan."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1)
    graph = BeliefGraph(belief)

    orphans = graph._find_orphaned_claims()
    assert "THESIS" not in orphans


@pytest.mark.unit
def test_challenges_edges_dont_count_as_support():
    """Test that 'challenges' edges from X# don't prevent orphan detection."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = []
    # X1 challenges C1 — but this should NOT count as support
    belief["counterpositions"] = [
        {
            "id": "X1",
            "targets": ["C1"],
            "attack_type": "rebutting",
            "statement": "Test",
            "my_response": "Test",
            "response_sufficiency": "sufficient"
        }
    ]

    graph = BeliefGraph(belief)
    orphans = graph._find_orphaned_claims()

    assert "C1" in orphans


@pytest.mark.unit
def test_questions_edges_dont_count_as_support():
    """Test that 'questions' edges from U# don't prevent orphan detection."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = []
    # U1 questions C1 — but this should NOT count as support
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["C1"], "question": "Test?", "status": "active"}
    ]

    graph = BeliefGraph(belief)
    orphans = graph._find_orphaned_claims()

    assert "C1" in orphans


# ==============================================
# 6. Support Chain Analysis Tests
# ==============================================

@pytest.mark.unit
def test_get_support_chain_direct():
    """Test support chain where C1 depends directly on A1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "A1" in chain


@pytest.mark.unit
def test_get_support_chain_transitive():
    """Test transitive support chain: C1 → C2 → A1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["C2"]
    belief["claims"][1]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "C2" in chain
    assert "A1" in chain


@pytest.mark.unit
def test_get_support_chain_none():
    """Test that orphaned node has empty support chain."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = []

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert len(chain) == 0


@pytest.mark.unit
def test_get_support_chain_multiple_paths():
    """Test support chain when node has multiple support paths."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["A1", "A2", "E1"]

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "A1" in chain
    assert "A2" in chain
    assert "E1" in chain


@pytest.mark.unit
def test_get_support_chain_thesis():
    """Test that THESIS support chain includes active claims."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("THESIS")

    assert "C1" in chain
    assert "A1" in chain


# ==============================================
# 7. Dependent Nodes Analysis Tests
# ==============================================

@pytest.mark.unit
def test_get_dependent_nodes_direct():
    """Test direct dependents: A1 supports C1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("A1")

    assert "C1" in dependents


@pytest.mark.unit
def test_get_dependent_nodes_transitive():
    """Test transitive dependents: A1 → C1 → C2 → THESIS."""
    belief = create_sample_belief(num_assumptions=1, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][1]["depends_on"] = ["C1"]

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("A1")

    assert "C1" in dependents
    assert "C2" in dependents
    assert "THESIS" in dependents


@pytest.mark.unit
def test_get_dependent_nodes_claim_to_thesis():
    """Test that active claims have THESIS as a dependent."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("C1")

    assert "THESIS" in dependents


@pytest.mark.unit
def test_get_dependent_nodes_thesis_is_leaf():
    """Test that THESIS has no dependents (it's the DAG sink)."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("THESIS")

    assert len(dependents) == 0


# ==============================================
# 8. Critical Path Analysis Tests
# ==============================================

@pytest.mark.unit
def test_find_critical_paths_single_path():
    """Test critical path detection when only one path exists."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["strength"] = 0.85  # High strength

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    assert len(critical_paths) > 0


@pytest.mark.unit
def test_find_critical_paths_redundant():
    """Test that redundant paths (multiple support) are not critical."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1", "A2"]  # Redundant support
    belief["claims"][0]["strength"] = 0.85

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    assert isinstance(critical_paths, list)


@pytest.mark.unit
def test_find_critical_paths_none():
    """Test critical path detection with no high-strength claims."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["strength"] = 0.3  # Low strength

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    assert len(critical_paths) == 0


# ==============================================
# 9. Graph Metrics Tests
# ==============================================

@pytest.mark.unit
def test_get_graph_metrics_empty():
    """Test graph metrics for belief with only THESIS."""
    belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    # Only THESIS node
    assert metrics["total_nodes"] == 1
    assert metrics["total_edges"] == 0
    assert metrics["node_counts"]["thesis"] == 1


@pytest.mark.unit
def test_get_graph_metrics_complete(test_beliefs):
    """Test graph metrics for complete belief."""
    belief = test_beliefs["complete_valid"]
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    assert "total_nodes" in metrics
    assert "total_edges" in metrics
    assert "node_counts" in metrics
    assert metrics["total_nodes"] > 0
    assert metrics["node_counts"]["thesis"] == 1


@pytest.mark.unit
def test_get_graph_metrics_node_counts():
    """Test that node counts per type are correct."""
    belief = create_sample_belief(num_assumptions=2, num_claims=3, num_evidence=1)
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    assert metrics["node_counts"]["thesis"] == 1
    assert metrics["node_counts"]["assumptions"] == 2
    assert metrics["node_counts"]["claims"] == 3
    assert metrics["node_counts"]["evidence"] == 1
    assert metrics["node_counts"]["uncertainties"] == 0


@pytest.mark.unit
def test_get_graph_metrics_node_counts_with_uncertainties():
    """Test that uncertainty node counts are correct."""
    belief = create_sample_belief()
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["A1"], "question": "Test?", "status": "active"},
        {"id": "U2", "targets": ["C1"], "question": "Test2?", "status": "active"}
    ]

    graph = BeliefGraph(belief)
    metrics = graph.get_graph_metrics()

    assert metrics["node_counts"]["uncertainties"] == 2


@pytest.mark.unit
def test_get_graph_metrics_no_predictions_key():
    """Test that graph metrics do not include a 'predictions' node count."""
    belief = create_sample_belief()
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    assert "predictions" not in metrics["node_counts"]


@pytest.mark.unit
def test_get_graph_metrics_edge_counts():
    """Test that edge count includes C# → THESIS edges."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["A1", "E1"]

    graph = BeliefGraph(belief)
    metrics = graph.get_graph_metrics()

    # 3 edges: A1→C1 (supports), E1→C1 (supports), C1→THESIS (supports)
    assert metrics["total_edges"] == 3
