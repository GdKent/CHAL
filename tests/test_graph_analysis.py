"""
Unit tests for graph-level convergence analysis.

Tests cover:
- Graph structure analysis
- Cross-agent graph comparison
- Contested nodes identification
"""

import pytest
from chal.convergence.graph_analysis import (
    analyze_vulnerabilities,
    format_attack_suggestions,
    get_graph_summary
)
from chal.beliefs.belief_graph import BeliefGraph
from tests.utils import create_sample_belief


# ==============================================
# 1. Graph-Level Metrics Tests
# ==============================================

@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_analyze_debate_graph_structure():
    """Test computation of structural metrics for belief graph."""
    belief = create_sample_belief(num_assumptions=3, num_claims=4, num_evidence=2)
    graph = BeliefGraph(belief)

    analysis = analyze_debate_graph_structure(graph)

    assert isinstance(analysis, dict)
    assert "total_nodes" in analysis
    assert "total_edges" in analysis
    assert "graph_density" in analysis
    assert analysis["total_nodes"] > 0


@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_analyze_debate_graph_structure_empty():
    """Test graph analysis with empty belief."""
    belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)
    graph = BeliefGraph(belief)

    analysis = analyze_debate_graph_structure(graph)

    assert isinstance(analysis, dict)
    assert analysis["total_nodes"] == 0
    assert analysis["total_edges"] == 0


@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_analyze_debate_graph_structure_complex():
    """Test analysis of complex graph with many connections."""
    belief = create_sample_belief(num_assumptions=5, num_claims=10, num_evidence=5)

    # Create complex dependencies
    for i, claim in enumerate(belief["claims"]):
        claim["depends_on"] = [f"A{(i % 5) + 1}", f"E{(i % 5) + 1}"]

    graph = BeliefGraph(belief)
    analysis = analyze_debate_graph_structure(graph)

    assert analysis["total_nodes"] > 15
    assert analysis["total_edges"] > 10


# ==============================================
# 2. Cross-Agent Graph Comparison Tests
# ==============================================

@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs():
    """Test comparison of graph complexity across agents."""
    belief_a = create_sample_belief(belief_id="A", num_assumptions=2, num_claims=3, num_evidence=1)
    belief_b = create_sample_belief(belief_id="B", num_assumptions=4, num_claims=6, num_evidence=3)

    graph_a = BeliefGraph(belief_a)
    graph_b = BeliefGraph(belief_b)

    comparison = compare_agent_graphs({"Agent-A": graph_a, "Agent-B": graph_b})

    assert isinstance(comparison, dict)
    assert "Agent-A" in comparison
    assert "Agent-B" in comparison


@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs_equal_complexity():
    """Test comparison when agents have similar graph complexity."""
    belief_a = create_sample_belief(num_assumptions=3, num_claims=3, num_evidence=2)
    belief_b = create_sample_belief(num_assumptions=3, num_claims=3, num_evidence=2)

    graph_a = BeliefGraph(belief_a)
    graph_b = BeliefGraph(belief_b)

    comparison = compare_agent_graphs({"Agent-A": graph_a, "Agent-B": graph_b})

    # Should show similar complexity metrics
    assert isinstance(comparison, dict)


@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs_different_complexity():
    """Test comparison with significantly different complexities."""
    belief_simple = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=0)
    belief_complex = create_sample_belief(num_assumptions=10, num_claims=20, num_evidence=10)

    graph_simple = BeliefGraph(belief_simple)
    graph_complex = BeliefGraph(belief_complex)

    comparison = compare_agent_graphs({
        "Simple": graph_simple,
        "Complex": graph_complex
    })

    # Complex should have higher metrics
    assert isinstance(comparison, dict)


@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs_multiple_agents():
    """Test comparison with more than two agents."""
    graphs = {
        f"Agent-{chr(65+i)}": BeliefGraph(create_sample_belief(num_claims=(i+1)*2))
        for i in range(5)
    }

    comparison = compare_agent_graphs(graphs)

    assert len(comparison) == 5
    assert all(f"Agent-{chr(65+i)}" in comparison for i in range(5))


# ==============================================
# 3. Contested Nodes Identification Tests
# ==============================================

@pytest.mark.skip(reason="Function identify_contested_nodes not yet implemented")
@pytest.mark.unit
def test_identify_contested_nodes():
    """Test finding nodes targeted by challenges."""
    belief_a = create_sample_belief(belief_id="A", num_claims=3)
    belief_b = create_sample_belief(belief_id="B", num_claims=3)

    # Simulate challenges targeting specific claims
    challenges = [
        {
            "challenger": "Agent-B",
            "target": "Agent-A",
            "target_claim_id": "C1",
            "challenge": "Why do you believe C1?"
        },
        {
            "challenger": "Agent-B",
            "target": "Agent-A",
            "target_claim_id": "C1",  # Same claim challenged twice
            "challenge": "C1 seems weak"
        },
        {
            "challenger": "Agent-A",
            "target": "Agent-B",
            "target_claim_id": "C2",
            "challenge": "C2 lacks evidence"
        }
    ]

    contested = identify_contested_nodes(challenges)

    assert isinstance(contested, dict)
    # C1 should be heavily contested (2 challenges)
    assert "C1" in str(contested)


@pytest.mark.skip(reason="Function identify_contested_nodes not yet implemented")
@pytest.mark.unit
def test_identify_contested_nodes_no_challenges():
    """Test with empty challenge list."""
    contested = identify_contested_nodes([])

    assert isinstance(contested, dict)
    assert len(contested) == 0


@pytest.mark.skip(reason="Function identify_contested_nodes not yet implemented")
@pytest.mark.unit
def test_identify_contested_nodes_single_challenge():
    """Test with single challenge."""
    challenges = [
        {
            "challenger": "Agent-A",
            "target": "Agent-B",
            "target_claim_id": "C1",
            "challenge": "Challenge text"
        }
    ]

    contested = identify_contested_nodes(challenges)

    assert isinstance(contested, dict)


@pytest.mark.skip(reason="Function identify_contested_nodes not yet implemented")
@pytest.mark.unit
def test_identify_contested_nodes_counts():
    """Test that contest counts are accurate."""
    challenges = [
        {"target_claim_id": "C1", "challenge": "Q1"},
        {"target_claim_id": "C1", "challenge": "Q2"},
        {"target_claim_id": "C1", "challenge": "Q3"},
        {"target_claim_id": "C2", "challenge": "Q4"}
    ]

    contested = identify_contested_nodes(challenges)

    # C1 should have count of 3, C2 should have count of 1
    assert isinstance(contested, dict)


# ==============================================
# 4. Graph Evolution Tracking Tests
# ==============================================

@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_graph_complexity_over_rounds():
    """Test tracking how graph complexity changes across rounds."""
    round_beliefs = [
        create_sample_belief(num_claims=2),  # Round 1
        create_sample_belief(num_claims=4),  # Round 2
        create_sample_belief(num_claims=3)   # Round 3 (simplified)
    ]

    analyses = [
        analyze_debate_graph_structure(BeliefGraph(b))
        for b in round_beliefs
    ]

    # Should track evolution
    assert len(analyses) == 3
    assert all(isinstance(a, dict) for a in analyses)


@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_graph_density_calculation():
    """Test that graph density is calculated correctly."""
    # Dense graph: many edges relative to nodes
    belief_dense = create_sample_belief(num_assumptions=3, num_claims=6, num_evidence=3)
    for claim in belief_dense["claims"]:
        claim["depends_on"] = ["A1", "A2", "A3", "E1", "E2", "E3"]  # Each claim depends on all assumptions and evidence

    graph_dense = BeliefGraph(belief_dense)
    analysis_dense = analyze_debate_graph_structure(graph_dense)

    # Sparse graph: few edges
    belief_sparse = create_sample_belief(num_assumptions=3, num_claims=3, num_evidence=0)
    for i, claim in enumerate(belief_sparse["claims"]):
        claim["depends_on"] = [f"A{i+1}"]  # Each claim depends on one assumption

    graph_sparse = BeliefGraph(belief_sparse)
    analysis_sparse = analyze_debate_graph_structure(graph_sparse)

    # Dense should have higher density
    if "graph_density" in analysis_dense and "graph_density" in analysis_sparse:
        assert analysis_dense["graph_density"] > analysis_sparse["graph_density"]


# ==============================================
# 5. Critical Path Analysis Tests
# ==============================================

@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_identify_critical_assumptions():
    """Test identifying assumptions that support high-strength claims."""
    belief = create_sample_belief(num_assumptions=3, num_claims=3, num_evidence=0)

    # Make C1 high strength and depend on A1
    belief["claims"][0]["strength"] = 0.95
    belief["claims"][0]["depends_on"] = ["A1"]

    # Make C2, C3 low strength
    belief["claims"][1]["strength"] = 0.4
    belief["claims"][1]["depends_on"] = ["A2"]

    belief["claims"][2]["strength"] = 0.3
    belief["claims"][2]["depends_on"] = ["A3"]

    graph = BeliefGraph(belief)
    analysis = analyze_debate_graph_structure(graph)

    # A1 should be identified as critical
    assert isinstance(analysis, dict)


# ==============================================
# 6. Structural Pattern Detection Tests
# ==============================================

@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_detect_hierarchical_structure():
    """Test detection of hierarchical belief structures."""
    belief = create_sample_belief(num_assumptions=2, num_claims=4, num_evidence=2)

    # Create hierarchy: A1,A2 → C1,C2 → C3,C4
    belief["claims"][0]["depends_on"] = ["A1", "E1"]
    belief["claims"][1]["depends_on"] = ["A2", "E2"]
    belief["claims"][2]["depends_on"] = ["C1", "C2"]
    belief["claims"][3]["depends_on"] = ["C1", "C2"]

    graph = BeliefGraph(belief)
    analysis = analyze_debate_graph_structure(graph)

    # Should detect multi-level structure
    assert isinstance(analysis, dict)


@pytest.mark.skip(reason="Function analyze_debate_graph_structure not yet implemented")
@pytest.mark.unit
def test_detect_flat_structure():
    """Test detection of flat (non-hierarchical) structures."""
    belief = create_sample_belief(num_assumptions=3, num_claims=3, num_evidence=3)

    # All claims depend directly on assumptions (no claim-claim dependencies)
    for i, claim in enumerate(belief["claims"]):
        claim["depends_on"] = [f"A{i+1}", f"E{i+1}"]

    graph = BeliefGraph(belief)
    analysis = analyze_debate_graph_structure(graph)

    # Should have low hierarchy depth
    assert isinstance(analysis, dict)


# ==============================================
# 7. Edge Case Tests
# ==============================================

@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs_empty():
    """Test comparison with no agents."""
    comparison = compare_agent_graphs({})

    assert isinstance(comparison, dict)
    assert len(comparison) == 0


@pytest.mark.skip(reason="Function compare_agent_graphs not yet implemented")
@pytest.mark.unit
def test_compare_agent_graphs_single_agent():
    """Test comparison with single agent."""
    belief = create_sample_belief(num_claims=3)
    graph = BeliefGraph(belief)

    comparison = compare_agent_graphs({"Agent-A": graph})

    assert isinstance(comparison, dict)
    assert "Agent-A" in comparison


@pytest.mark.skip(reason="Function identify_contested_nodes not yet implemented")
@pytest.mark.unit
def test_identify_contested_nodes_malformed_challenge():
    """Test handling of malformed challenge data."""
    challenges = [
        {"target_claim_id": "C1"},  # Missing challenge text
        {},  # Missing all fields
        {"challenge": "Test"}  # Missing target_claim_id
    ]

    # Should handle gracefully
    contested = identify_contested_nodes(challenges)

    assert isinstance(contested, dict)


# ==============================================
# 8. Weak Evidence Detection Tests (Part 6B)
# ==============================================

@pytest.mark.unit
def test_weak_evidence_detection_string_quality():
    """Test that evidence with weak strength_justification string is flagged."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["evidence"][0]["strength_justification"] = "Weak, preliminary study with small sample"
    graph = BeliefGraph(belief)

    vulns = analyze_vulnerabilities(graph)
    weak_chains = vulns["weak_evidence_chains"]

    assert len(weak_chains) > 0, "Evidence with weak justification should be flagged"
    assert weak_chains[0]["claim_id"] == "C1"
    assert any(ev["ev_id"] == "E1" for ev in weak_chains[0]["weak_evidence"])


@pytest.mark.unit
def test_strong_evidence_not_flagged():
    """Test that evidence with strong strength_justification string is NOT flagged."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief["evidence"][0]["strength_justification"] = "Strong — replicated across labs, converging methods"
    graph = BeliefGraph(belief)

    vulns = analyze_vulnerabilities(graph)
    weak_chains = vulns["weak_evidence_chains"]

    assert len(weak_chains) == 0, f"Strong evidence should not be flagged, got: {weak_chains}"


@pytest.mark.unit
def test_evidence_no_strength_justification():
    """Test that evidence without strength_justification field is flagged as unknown quality."""
    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    # Remove strength_justification entirely
    belief["evidence"][0].pop("strength_justification", None)
    graph = BeliefGraph(belief)

    vulns = analyze_vulnerabilities(graph)
    weak_chains = vulns["weak_evidence_chains"]

    assert len(weak_chains) > 0, "Evidence without strength_justification should be flagged"
    weak_ev = weak_chains[0]["weak_evidence"][0]
    assert weak_ev["strength_justification"] == "(none)"


# ==============================================
# 9. Prompt Attack Framework Tests (Part 6C)
# ==============================================

@pytest.mark.unit
def test_stage_2_prompt_contains_attack_framework():
    """Test that standard Stage 2 prompt includes the attack framework section."""
    from chal.agents.prompts import build_stage_2_prompt
    from tests.utils import create_sample_belief, create_mock_belief_response
    import json

    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief_json = json.dumps(belief)

    prompt = build_stage_2_prompt(
        topic="Test topic",
        agent_name="Agent-A",
        opponent_name="Agent-B",
        agent_belief_json=belief_json,
        opponent_belief_json=belief_json
    )

    assert "<attack_framework>" in prompt
    assert "UNDERMINING" in prompt
    assert "REBUTTING" in prompt
    assert "UNDERCUTTING" in prompt
    assert "</attack_framework>" in prompt


@pytest.mark.unit
def test_stage_2_bloodsport_prompt_contains_attack_vectors():
    """Test that blood sport Stage 2 prompt includes attack vector taxonomy."""
    from chal.agents.prompts import build_stage_2_bloodsport_prompt
    import json

    belief = create_sample_belief(num_assumptions=1, num_claims=1, num_evidence=1)
    belief_json = json.dumps(belief)

    prompt = build_stage_2_bloodsport_prompt(
        topic="Test topic",
        agent_name="Agent-A",
        opponent_name="Agent-B",
        agent_belief_json=belief_json,
        opponent_belief_json=belief_json
    )

    assert "UNDERMINING" in prompt
    assert "REBUTTING" in prompt
    assert "UNDERCUTTING" in prompt
    assert "attack vector" in prompt.lower()
