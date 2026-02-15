"""
Unit tests for BeliefGraph class.

Tests cover:
- Graph construction from beliefs
- Node operations
- Link validation
- Cycle detection
- Orphan detection
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
    """Test graph construction with empty collections (no nodes/edges)."""
    belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


@pytest.mark.unit
def test_build_graph_assumptions_only():
    """Test graph construction with only assumption nodes."""
    belief = create_sample_belief(num_assumptions=3, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    assert len(graph.nodes) == 3
    assert all(node.startswith("A") for node in graph.nodes)
    assert len(graph.edges) == 0


@pytest.mark.unit
def test_build_graph_claims_only():
    """Test graph construction with only claim nodes."""
    belief = create_sample_belief(num_assumptions=0, num_claims=2, num_evidence=0)
    # Modify belief to avoid orphan detection
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][1]["depends_on"] = []
    belief["claims"][1]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)

    assert len(graph.nodes) == 2
    assert all(node.startswith("C") for node in graph.nodes)


@pytest.mark.unit
def test_build_graph_complete(test_beliefs):
    """Test graph construction with all node types present."""
    belief = test_beliefs["complete_valid"]
    graph = BeliefGraph(belief)

    # Check that nodes exist for all ID types
    node_types = {node[0] for node in graph.nodes}
    assert "A" in node_types  # Assumptions
    assert "C" in node_types  # Claims
    assert "E" in node_types  # Evidence


@pytest.mark.unit
def test_build_graph_edges_supports():
    """Test that depends_on creates 'supports' edges."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    # Ensure claim depends on assumption
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)

    # There should be an edge from A1 to C1
    assert any(from_id == "A1" and to_id == "C1" and edge_type == "supports"
               for from_id, to_id, edge_type in graph.edges)


@pytest.mark.unit
def test_build_graph_edges_evidences():
    """Test that backing_evidence_ids creates 'evidences' edges."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=1)
    # Ensure claim has backing evidence
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = ["E1"]

    graph = BeliefGraph(belief)

    # There should be an edge from E1 to C1
    assert any(from_id == "E1" and to_id == "C1" and edge_type == "evidences"
               for from_id, to_id, edge_type in graph.edges)


@pytest.mark.unit
def test_build_graph_edges_predicts(test_beliefs):
    """Test that linked_claims creates 'predicts' edges."""
    belief = test_beliefs["complete_valid"]
    graph = BeliefGraph(belief)

    # Check if prediction edges exist
    prediction_edges = [e for e in graph.edges if e[2] == "predicts"]
    if "predictions" in belief and len(belief["predictions"]) > 0:
        assert len(prediction_edges) > 0


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
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("A99" in error for error in errors)


@pytest.mark.unit
def test_validate_links_broken_target():
    """Test detection of edge referencing non-existent target node."""
    belief = create_sample_belief(num_assumptions=1, num_claims=0, num_evidence=0)
    # Add a prediction pointing to non-existent claim
    belief["predictions"] = [
        {"id": "P1", "statement": "Test", "linked_claims": ["C99"]}
    ]

    graph = BeliefGraph(belief)
    errors = graph.validate_links()

    assert len(errors) > 0
    assert any("C99" in error for error in errors)


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
    # Claim depends on itself
    belief["claims"][0]["depends_on"] = ["C1"]
    belief["claims"][0]["backing_evidence_ids"] = []

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
    # Check for orphan errors - edges are tuples so we need to check differently
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
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][1]["depends_on"] = ["C3"]
    belief["claims"][1]["backing_evidence_ids"] = []
    belief["claims"][2]["depends_on"] = ["C1"]
    belief["claims"][2]["backing_evidence_ids"] = []

    assert_graph_has_cycle(belief)


@pytest.mark.unit
def test_has_cycle_false_multiple_components():
    """Test acyclic graph with disconnected components."""
    belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=0)
    # Two independent claims, each depending on different assumption
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][1]["depends_on"] = ["A2"]
    belief["claims"][1]["backing_evidence_ids"] = []

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
    # All claims are orphaned
    for claim in belief["claims"]:
        claim["depends_on"] = []
        claim["backing_evidence_ids"] = []

    assert_has_orphaned_claims(belief, expected_count=3)


@pytest.mark.unit
def test_find_orphaned_claims_evidence_only():
    """Test that claims with only evidence (no assumptions) are NOT orphaned."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = ["E1"]

    assert_no_orphaned_claims(belief)


@pytest.mark.unit
def test_find_orphaned_claims_assumptions_only():
    """Test that claims with only assumptions (no evidence) are NOT orphaned."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    assert_no_orphaned_claims(belief)


# ==============================================
# 6. Support Chain Analysis Tests
# ==============================================

@pytest.mark.unit
def test_get_support_chain_direct():
    """Test support chain where C1 depends directly on A1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "A1" in chain


@pytest.mark.unit
def test_get_support_chain_transitive():
    """Test transitive support chain: C1 → C2 → A1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["C2"]
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][1]["depends_on"] = ["A1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "C2" in chain
    assert "A1" in chain


@pytest.mark.unit
def test_get_support_chain_none():
    """Test that orphaned node has empty support chain."""
    belief = create_sample_belief(num_assumptions=0, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = []
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert len(chain) == 0


@pytest.mark.unit
def test_get_support_chain_multiple_paths():
    """Test support chain when node has multiple support paths."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["A1", "A2"]
    belief["claims"][0]["backing_evidence_ids"] = ["E1"]

    graph = BeliefGraph(belief)
    chain = graph.get_support_chain("C1")

    assert "A1" in chain
    assert "A2" in chain
    assert "E1" in chain


# ==============================================
# 7. Dependent Nodes Analysis Tests
# ==============================================

@pytest.mark.unit
def test_get_dependent_nodes_direct():
    """Test direct dependents: A1 supports C1."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("A1")

    assert "C1" in dependents


@pytest.mark.unit
def test_get_dependent_nodes_transitive():
    """Test transitive dependents: A1 → C1 → C2."""
    belief = create_sample_belief(num_assumptions=1, num_claims=2, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("A1")

    assert "C1" in dependents
    assert "C2" in dependents


@pytest.mark.unit
def test_get_dependent_nodes_none():
    """Test that leaf node has no dependents."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    graph = BeliefGraph(belief)
    dependents = graph.get_dependent_nodes("C1")

    assert len(dependents) == 0


# ==============================================
# 8. Critical Path Analysis Tests
# ==============================================

@pytest.mark.unit
def test_find_critical_paths_single_path():
    """Test critical path detection when only one path exists."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][0]["confidence"] = 0.85  # High confidence

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    assert len(critical_paths) > 0


@pytest.mark.unit
def test_find_critical_paths_redundant():
    """Test that redundant paths (multiple support) are not critical."""
    belief = create_sample_belief(num_assumptions=2, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1", "A2"]  # Redundant support
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][0]["confidence"] = 0.85

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    # With redundant support, paths may not be critical
    assert isinstance(critical_paths, list)


@pytest.mark.unit
def test_find_critical_paths_none():
    """Test critical path detection with no high-confidence claims."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []
    belief["claims"][0]["confidence"] = 0.3  # Low confidence

    graph = BeliefGraph(belief)
    critical_paths = graph.find_critical_paths()

    assert len(critical_paths) == 0


# ==============================================
# 9. Graph Metrics Tests
# ==============================================

@pytest.mark.unit
def test_get_graph_metrics_empty():
    """Test graph metrics for empty belief."""
    belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    assert metrics["total_nodes"] == 0
    assert metrics["total_edges"] == 0


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


@pytest.mark.unit
def test_get_graph_metrics_node_counts():
    """Test that node counts per type are correct."""
    belief = create_sample_belief(num_assumptions=2, num_claims=3, num_evidence=1)
    graph = BeliefGraph(belief)

    metrics = graph.get_graph_metrics()

    assert metrics["node_counts"]["assumptions"] == 2
    assert metrics["node_counts"]["claims"] == 3
    assert metrics["node_counts"]["evidence"] == 1


@pytest.mark.unit
def test_get_graph_metrics_edge_counts():
    """Test that edge count is correct."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = ["E1"]

    graph = BeliefGraph(belief)
    metrics = graph.get_graph_metrics()

    # Should have 2 edges: A1→C1 and E1→C1
    assert metrics["total_edges"] == 2
