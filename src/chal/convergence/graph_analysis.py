"""
graph_analysis.py

Provides vulnerability detection and attack suggestion generation based on belief graph analysis.

This module enables agents to identify structural weaknesses in opponent beliefs:
- Critical paths (single points of failure)
- Orphaned claims (unsupported assertions)
- Overcalibrated claims (confidence exceeds dependency support)
- Circular dependencies (invalid reasoning)

These vulnerabilities are formatted as strategic attack suggestions for Stage 2 (Cross-Examination).
"""

from typing import Dict, List, Any, Optional
from chal.beliefs.belief_graph import BeliefGraph


def analyze_vulnerabilities(belief_graph: BeliefGraph) -> Dict[str, Any]:
    """
    Analyze a belief graph for structural vulnerabilities.

    Args:
        belief_graph: BeliefGraph object to analyze

    Returns:
        Dict containing:
        - critical_paths: List of single-point-of-failure inference chains
        - orphaned_claims: List of claim IDs with no support
        - weak_evidence_chains: Claims backed by low-quality or unreplicated evidence
        - circular_dependencies: Boolean indicating if cycles exist
        - weak_foundations: Claims depending on low-confidence (<0.5) dependencies
    """
    vulnerabilities = {
        "critical_paths": [],
        "orphaned_claims": [],
        "weak_evidence_chains": [],
        "circular_dependencies": False,
        "weak_foundations": [],
    }

    # 1. Find critical paths (single points of failure)
    try:
        critical_paths = belief_graph.find_critical_paths()
        vulnerabilities["critical_paths"] = [
            {
                "path": path,
                "length": len(path),
                "risk": "HIGH" if len(path) > 3 else "MEDIUM"
            }
            for path in critical_paths[:5]  # Top 5 most critical
        ]
    except Exception as e:
        # If critical path analysis fails, skip
        pass

    # 2. Find orphaned claims (should be blocked by validation, but check anyway)
    try:
        orphans = belief_graph._find_orphaned_claims()
        vulnerabilities["orphaned_claims"] = orphans
    except Exception:
        pass

    # 3. Detect circular dependencies (should be blocked by validation)
    try:
        has_cycle = belief_graph._has_cycle()
        vulnerabilities["circular_dependencies"] = has_cycle
    except Exception:
        pass

    # 4. Find weak evidence chains (low-quality or unreplicated evidence)
    try:
        weak_evidence = _find_weak_evidence_chains(belief_graph)
        vulnerabilities["weak_evidence_chains"] = weak_evidence
    except Exception:
        pass

    # 5. Find claims with weak foundations (depending on low-confidence claims)
    try:
        weak_foundations = _find_weak_foundations(belief_graph)
        vulnerabilities["weak_foundations"] = weak_foundations
    except Exception:
        pass

    return vulnerabilities


def _find_weak_evidence_chains(belief_graph: BeliefGraph) -> List[Dict[str, Any]]:
    """
    Find claims backed by low-quality or unreplicated evidence.

    Focuses on evidence substance rather than numeric confidence scores.
    Low-quality evidence indicators:
    - rigor: "low" or "medium"
    - replication_status: "unreplicated", "failed", "contested"

    Returns:
        List of dicts with weak evidence chain info
    """
    weak_chains = []

    for node_id, node in belief_graph.nodes.items():
        if node["type"] != "claim":
            continue

        claim_data = node["data"]
        evidence_ids = claim_data.get("backing_evidence_ids", [])

        if not evidence_ids:
            continue  # No evidence to analyze (handled by orphaned claims check)

        weak_evidence = []
        for ev_id in evidence_ids:
            ev_node = belief_graph.get_node(ev_id)
            if ev_node and ev_node.get("type") == "evidence":
                ev_data = ev_node.get("data", {})
                rigor = ev_data.get("rigor", "medium").lower()
                replication = ev_data.get("replication_status", "unreplicated").lower()

                # Flag as weak if low/medium rigor OR problematic replication
                is_weak_rigor = rigor in ["low", "medium"]
                is_weak_replication = any(keyword in replication for keyword in ["unreplicated", "failed", "contested", "preliminary"])

                if is_weak_rigor or is_weak_replication:
                    weak_evidence.append({
                        "ev_id": ev_id,
                        "rigor": rigor,
                        "replication_status": replication,
                        "description": ev_data.get("description", "")[:80]
                    })

        if weak_evidence:
            weak_chains.append({
                "claim_id": node_id,
                "weak_evidence": weak_evidence,
                "statement": claim_data.get("statement", "")[:100]
            })

    return weak_chains


def _find_weak_foundations(belief_graph: BeliefGraph) -> List[Dict[str, Any]]:
    """
    Find claims that depend on low-confidence (<0.5) dependencies.

    Returns:
        List of dicts with weak foundation info
    """
    weak_foundations = []

    for node_id, node in belief_graph.nodes.items():
        if node["type"] != "claim":
            continue

        claim_data = node["data"]
        claim_confidence = claim_data.get("confidence", 0.5)

        # Get all dependencies
        depends_on = claim_data.get("depends_on", [])
        all_deps = depends_on  # Only check claim dependencies, not evidence

        weak_deps = []
        for dep_id in all_deps:
            dep_node = belief_graph.get_node(dep_id)
            if dep_node and dep_node.get("type") == "claim":
                dep_confidence = dep_node.get("confidence", 0.5)
                if dep_confidence < 0.5:
                    weak_deps.append({
                        "dep_id": dep_id,
                        "dep_confidence": dep_confidence,
                        "dep_statement": dep_node.get("statement", "")[:100]
                    })

        if weak_deps:
            weak_foundations.append({
                "claim_id": node_id,
                "claim_confidence": claim_confidence,
                "weak_dependencies": weak_deps,
                "statement": claim_data.get("statement", "")[:100]
            })

    return weak_foundations


def format_attack_suggestions(vulnerabilities: Dict[str, Any], opponent_name: str) -> str:
    """
    Format vulnerability analysis as strategic attack suggestions for Stage 2 prompt.

    Args:
        vulnerabilities: Dict from analyze_vulnerabilities()
        opponent_name: Name of opponent agent

    Returns:
        Formatted string with attack suggestions
    """
    if not vulnerabilities:
        return ""

    sections = []

    # Header
    sections.append(f"STRUCTURAL VULNERABILITY ANALYSIS OF {opponent_name.upper()}'S BELIEF GRAPH:\n")

    # 1. Critical paths
    critical_paths = vulnerabilities.get("critical_paths", [])
    if critical_paths:
        sections.append("⚠️  CRITICAL PATHS (Single Points of Failure):")
        sections.append(f"   {opponent_name} has {len(critical_paths)} critical inference chain(s) where removing a single node breaks the entire argument.\n")
        for i, path_info in enumerate(critical_paths[:3], 1):  # Show top 3
            path = path_info["path"]
            risk = path_info["risk"]
            sections.append(f"   Path {i} [{risk} RISK]: {' → '.join(path)}")
        sections.append(f"   💡 ATTACK STRATEGY: Challenge any node in these chains to collapse dependent claims.\n")

    # 2. Weak evidence chains
    weak_evidence_chains = vulnerabilities.get("weak_evidence_chains", [])
    if weak_evidence_chains:
        sections.append("⚠️  WEAK EVIDENCE CHAINS (Low-Quality or Unreplicated Evidence):")
        sections.append(f"   {opponent_name} has {len(weak_evidence_chains)} claim(s) backed by questionable evidence.\n")
        for i, chain_info in enumerate(weak_evidence_chains[:3], 1):
            claim_id = chain_info["claim_id"]
            weak_ev = chain_info["weak_evidence"]
            sections.append(f"   {claim_id} relies on {len(weak_ev)} weak evidence source(s):")
            for ev in weak_ev[:2]:  # Show first 2 weak evidence items
                ev_id = ev["ev_id"]
                rigor = ev["rigor"]
                replication = ev["replication_status"]
                sections.append(f"      - {ev_id}: rigor={rigor}, replication={replication}")
        sections.append(f"   💡 ATTACK STRATEGY: Challenge evidence quality, demand stronger studies, replication, or more rigorous methodology.\n")

    # 3. Weak foundations
    weak_foundations = vulnerabilities.get("weak_foundations", [])
    if weak_foundations:
        sections.append("⚠️  WEAK FOUNDATIONS (Built on Low-Confidence Claims):")
        sections.append(f"   {opponent_name} has {len(weak_foundations)} claim(s) depending on claims with confidence <0.5.\n")
        for i, claim_info in enumerate(weak_foundations[:3], 1):
            claim_id = claim_info["claim_id"]
            weak_deps = claim_info["weak_dependencies"]
            sections.append(f"   {claim_id} depends on {len(weak_deps)} weak claim(s):")
            for dep in weak_deps[:2]:
                sections.append(f"      - {dep['dep_id']} (confidence={dep['dep_confidence']:.2f})")
        sections.append(f"   💡 ATTACK STRATEGY: Challenge the foundation—if the base is weak, the structure is unstable.\n")

    # 4. Orphaned claims (should be rare due to validation)
    orphaned = vulnerabilities.get("orphaned_claims", [])
    if orphaned:
        sections.append("⚠️  ORPHANED CLAIMS (No Supporting Evidence or Assumptions):")
        sections.append(f"   {opponent_name} has {len(orphaned)} unsupported claim(s): {', '.join(orphaned)}")
        sections.append(f"   💡 ATTACK STRATEGY: Demand evidence or justification for these unsupported assertions.\n")

    # 5. Circular dependencies (should be blocked by validation)
    if vulnerabilities.get("circular_dependencies", False):
        sections.append("⚠️  CIRCULAR DEPENDENCIES DETECTED:")
        sections.append(f"   {opponent_name}'s belief contains circular reasoning (claims depending on themselves).")
        sections.append(f"   💡 ATTACK STRATEGY: Identify and expose the circular logic.\n")

    # Summary
    if not any([critical_paths, weak_evidence_chains, weak_foundations, orphaned]):
        return ""  # No vulnerabilities found

    sections.append("NOTE: These are structural vulnerabilities. Consider targeting the weakest points for maximum impact.")

    return "\n".join(sections)


def get_graph_summary(belief_graph: BeliefGraph) -> str:
    """
    Generate a brief summary of belief graph structure.

    Args:
        belief_graph: BeliefGraph object

    Returns:
        String summary of graph metrics
    """
    try:
        metrics = belief_graph.get_graph_metrics()
        return (
            f"Graph structure: {metrics['total_nodes']} nodes "
            f"({metrics['node_counts']['assumptions']} assumptions, "
            f"{metrics['node_counts']['claims']} claims, "
            f"{metrics['node_counts']['evidence']} evidence), "
            f"{metrics['total_edges']} edges, "
            f"{metrics['critical_path_count']} critical paths"
        )
    except Exception as e:
        return f"Graph summary unavailable: {e}"
