"""
Integration tests for belief system components.

Tests cover:
- Complete belief lifecycle
- Patch propagation through graph
- I/O roundtrip
- Graph visualization integration
"""

import pytest
import json
from pathlib import Path
from chal.beliefs.schema import validate_belief
from chal.beliefs.belief_graph import BeliefGraph
from chal.beliefs.patches import apply_patches
from chal.beliefs.io import (
    parse_model_output_to_belief,
    belief_to_markdown,
    project_for_embedding
)
from tests.utils import create_sample_belief, create_mock_belief_response


# ==============================================
# 1. Belief Lifecycle Integration Test
# ==============================================

@pytest.mark.integration
def test_belief_lifecycle():
    """Test Create → Validate → Graph → Patch → Validate workflow."""
    # 1. Create belief
    belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=1)

    # 2. Validate initial belief
    errors = validate_belief(belief)
    assert len(errors) == 0, f"Initial belief invalid: {errors}"

    # 3. Build graph
    graph = BeliefGraph(belief)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    # 4. Apply patches
    patches = [
        {"op": "update_thesis", "change": "weaken"},
        {
            "op": "update_claim",
            "target_id": "C1",
            "changes": {"confidence": 0.6}
        }
    ]
    updated_belief = apply_patches(belief, patches)

    # 5. Validate updated belief
    errors = validate_belief(updated_belief)
    assert len(errors) == 0, f"Updated belief invalid: {errors}"

    # 6. Verify changes applied
    assert updated_belief["thesis"]["confidence"] < belief["thesis"]["confidence"]
    assert updated_belief["version"] > belief["version"]


# ==============================================
# 2. Patch Propagation Integration Test
# ==============================================

@pytest.mark.integration
def test_belief_patch_propagation_integration():
    """Test that patches propagate through dependency graph correctly."""
    # Create belief with dependency chain: C3 → C2 → C1 → A1
    belief = create_sample_belief(num_assumptions=1, num_claims=3, num_evidence=0)

    belief["claims"][0]["id"] = "C1"
    belief["claims"][0]["confidence"] = 0.9
    belief["claims"][0]["depends_on"] = ["A1"]
    belief["claims"][0]["backing_evidence_ids"] = []

    belief["claims"][1]["id"] = "C2"
    belief["claims"][1]["confidence"] = 0.9
    belief["claims"][1]["depends_on"] = ["C1"]
    belief["claims"][1]["backing_evidence_ids"] = []

    belief["claims"][2]["id"] = "C3"
    belief["claims"][2]["confidence"] = 0.9
    belief["claims"][2]["depends_on"] = ["C2"]
    belief["claims"][2]["backing_evidence_ids"] = []

    # Weaken C1
    patches = [{
        "op": "update_claim",
        "target_id": "C1",
        "changes": {"confidence": 0.5}
    }]

    updated = apply_patches(belief, patches, propagate_confidence=True)

    # Verify propagation
    c1 = next(c for c in updated["claims"] if c["id"] == "C1")
    c2 = next(c for c in updated["claims"] if c["id"] == "C2")
    c3 = next(c for c in updated["claims"] if c["id"] == "C3")

    assert c1["confidence"] == 0.5
    assert c2["confidence"] < 0.9  # Should be weakened
    assert c3["confidence"] < 0.9  # Should be weakened transitively


# ==============================================
# 3. I/O Roundtrip Integration Test
# ==============================================

@pytest.mark.integration
def test_belief_io_roundtrip():
    """Test Parse → Render Markdown → Parse again preserves structure."""
    # 1. Create initial belief
    original_belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=1)

    # 2. Convert to markdown
    markdown = belief_to_markdown(original_belief)
    assert len(markdown) > 0

    # 3. Simulate model response with JSON
    json_response = create_mock_belief_response(original_belief)

    # 4. Parse model output
    parsed_belief, markdown, errors = parse_model_output_to_belief(json_response)
    assert parsed_belief is not None

    # 5. Verify structure preserved
    assert parsed_belief["schema_version"] == original_belief["schema_version"]
    assert parsed_belief["belief_id"] == original_belief["belief_id"]
    assert parsed_belief["thesis"]["stance"] == original_belief["thesis"]["stance"]


# ==============================================
# 4. Graph Visualization Integration Test
# ==============================================

@pytest.mark.integration
def test_belief_graph_visualization_integration():
    """Test BeliefGraph → export_debate_graph workflow."""
    # Create complex belief
    belief = create_sample_belief(num_assumptions=3, num_claims=4, num_evidence=2)

    # Build graph
    graph = BeliefGraph(belief)

    # Get graph metrics
    metrics = graph.get_graph_metrics()

    assert metrics["total_nodes"] > 0
    assert metrics["total_edges"] > 0
    assert "node_types" in metrics

    # Verify graph structure
    errors = graph.validate_links()
    assert len(errors) == 0

    # Verify no cycles
    assert not graph._has_cycle()


# ==============================================
# 5. Embedding Integration Test
# ==============================================

@pytest.mark.integration
def test_belief_embedding_projection():
    """Test that belief can be projected and embedded."""
    belief = create_sample_belief(num_assumptions=2, num_claims=2, num_evidence=1)

    # Project for embedding
    projection = project_for_embedding(belief)

    assert isinstance(projection, str)
    assert len(projection) > 50  # Should have substantial content

    # Verify key content is included
    assert belief["thesis"]["stance"] in projection


# ==============================================
# 6. Multi-Belief Comparison Integration
# ==============================================

@pytest.mark.integration
def test_multi_belief_comparison():
    """Test comparing multiple beliefs for convergence analysis."""
    belief1 = create_sample_belief(belief_id="BELIEF-A", num_claims=3)
    belief2 = create_sample_belief(belief_id="BELIEF-B", num_claims=3)

    # Both should be valid
    assert len(validate_belief(belief1)) == 0
    assert len(validate_belief(belief2)) == 0

    # Both should have graphs
    graph1 = BeliefGraph(belief1)
    graph2 = BeliefGraph(belief2)

    assert len(graph1.nodes) > 0
    assert len(graph2.nodes) > 0

    # Both should project for embeddings
    proj1 = project_for_embedding(belief1)
    proj2 = project_for_embedding(belief2)

    assert len(proj1) > 0
    assert len(proj2) > 0
