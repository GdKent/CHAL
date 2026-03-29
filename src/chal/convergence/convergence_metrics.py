"""
convergence_metrics.py

Measures semantic convergence between agent beliefs by analyzing claim-level agreement.
Reuses existing embedding infrastructure from embedding_tracker to avoid duplication.

This module complements trajectory visualization by providing quantitative convergence metrics:
- Trajectory visualization: Shows holistic belief movement in semantic space (visual)
- Convergence metrics: Measures specific claim-level agreement (quantitative)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_claim_agreement(
    agent_beliefs: List[Dict[str, Any]],
    embedding_model,
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Calculate convergence by comparing individual claims across agents.

    Reuses the existing SentenceTransformer from BeliefEmbeddingTracker to avoid
    duplicating embedding infrastructure.

    Args:
        agent_beliefs: List of CBS belief dicts from all agents
        embedding_model: SentenceTransformer instance from embedding_tracker.model
        similarity_threshold: Cosine similarity threshold for claim matching (0.0-1.0)
                            0.75 = claims must be quite similar to match (recommended)

    Returns:
        Dict containing:
        - convergence_score: Float 0.0-1.0 (higher = more agreement)
        - total_claims: Total number of accepted claims across all agents
        - shared_claim_pairs: Number of matching claim pairs from different agents
        - shared_claims: List of dicts with details about shared claims
        - unique_claims: List of dicts with details about unique claims
    """
    if len(agent_beliefs) < 2:
        return {
            "convergence_score": 1.0,  # Single agent = perfect "agreement" with self
            "total_claims": 0,
            "shared_claim_pairs": 0,
            "shared_claims": [],
            "unique_claims": []
        }

    # Extract all accepted claims with metadata
    all_claims = []
    for belief in agent_beliefs:
        agent_id = belief.get("belief_id", "unknown")
        for claim in belief.get("claims", []):
            if claim.get("status") == "accepted":
                all_claims.append({
                    "agent_id": agent_id,
                    "claim_id": claim.get("id"),
                    "statement": claim.get("statement", ""),
                    "strength": claim.get("strength", 0.5)
                })

    if not all_claims:
        return {
            "convergence_score": 0.0,
            "total_claims": 0,
            "shared_claim_pairs": 0,
            "shared_claims": [],
            "unique_claims": []
        }

    # Embed all claim statements using provided model
    statements = [c["statement"] for c in all_claims]
    embeddings = embedding_model.encode(statements, convert_to_numpy=True)

    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Find shared claims (claims from different agents that semantically match)
    shared_pairs = []
    matched_indices = set()

    for i in range(len(all_claims)):
        for j in range(i + 1, len(all_claims)):
            # Only count as shared if from different agents AND semantically similar
            if (all_claims[i]["agent_id"] != all_claims[j]["agent_id"] and
                sim_matrix[i][j] >= similarity_threshold):

                shared_pairs.append({
                    "claim_1": all_claims[i],
                    "claim_2": all_claims[j],
                    "similarity": round(float(sim_matrix[i][j]), 3)
                })
                matched_indices.add(i)
                matched_indices.add(j)

    # Find unique claims (no semantic matches from other agents)
    unique_claims = []
    for idx, claim in enumerate(all_claims):
        if idx not in matched_indices:
            unique_claims.append({
                "agent_id": claim["agent_id"],
                "claim_id": claim["claim_id"],
                "statement": claim["statement"][:100],  # Truncate for display
                "strength": claim["strength"]
            })

    # Group shared pairs into claim groups
    shared_claim_groups = _group_shared_claims(shared_pairs)

    # Calculate convergence score
    # Formula: (claims with cross-agent agreement) / (total claims)
    claims_with_agreement = len(matched_indices)
    total_claims = len(all_claims)
    convergence_score = claims_with_agreement / total_claims if total_claims > 0 else 0.0

    return {
        "convergence_score": round(convergence_score, 3),
        "total_claims": total_claims,
        "shared_claim_pairs": len(shared_pairs),
        "shared_claims": shared_claim_groups,
        "unique_claims": unique_claims
    }


def _group_shared_claims(shared_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group shared claim pairs into unified claim groups.

    Args:
        shared_pairs: List of matched claim pairs

    Returns:
        List of claim groups showing which agents agree
    """
    if not shared_pairs:
        return []

    # Simple grouping: each pair becomes a group
    # (Could be enhanced with transitive clustering for complex cases)
    groups = []
    for pair in shared_pairs[:5]:  # Limit to top 5 for display
        claim_1 = pair["claim_1"]
        claim_2 = pair["claim_2"]

        groups.append({
            "agent_ids": [claim_1["agent_id"], claim_2["agent_id"]],
            "statements": [claim_1["statement"], claim_2["statement"]],
            "similarity": pair["similarity"],
            "avg_strength": round((claim_1["strength"] + claim_2["strength"]) / 2, 2)
        })

    return groups


def format_convergence_summary(
    convergence_data: Dict[str, Any],
    agent_names: Optional[List[str]] = None,
    round_number: Optional[int] = None
) -> str:
    """
    Format convergence analysis as human-readable summary for logging/display.

    Args:
        convergence_data: Output from calculate_claim_agreement()
        agent_names: Optional list of agent names for display
        round_number: Optional round number for display

    Returns:
        Formatted string suitable for console output and logging
    """
    score = convergence_data["convergence_score"]
    total_claims = convergence_data["total_claims"]
    shared_pairs = convergence_data["shared_claim_pairs"]
    shared = convergence_data.get("shared_claims", [])
    unique = convergence_data.get("unique_claims", [])

    lines = ["=" * 70]

    if round_number is not None:
        lines.append(f"CONVERGENCE ANALYSIS - ROUND {round_number}")
    else:
        lines.append("CONVERGENCE ANALYSIS")

    lines.extend([
        "=" * 70,
        f"Convergence Score: {score:.1%}",
        f"  Total Claims: {total_claims}",
        f"  Shared Claim Pairs: {shared_pairs}",
        f"  Unique Claims: {len(unique)}",
        ""
    ])

    # Show shared claims
    if shared:
        lines.append("SHARED CLAIMS (Agents Agree):")
        for i, group in enumerate(shared[:3], 1):  # Show top 3
            agents = ", ".join(group["agent_ids"])
            similarity = group["similarity"]
            lines.append(f"  {i}. Agents: {agents} (similarity: {similarity:.2f})")
            lines.append(f"     Statement: {group['statements'][0][:80]}...")
            lines.append(f"     Avg Strength: {group['avg_strength']:.2f}")

        if len(shared) > 3:
            lines.append(f"     ... and {len(shared) - 3} more shared claim group(s)")
        lines.append("")

    # Show unique claims by agent
    if unique:
        lines.append(f"UNIQUE CLAIMS (No Agreement): {len(unique)} total")
        for agent_name in agent_names:
            agent_unique = [u for u in unique if u["agent_id"] == agent_name]
            if agent_unique:
                lines.append(f"  {agent_name}: {len(agent_unique)} unique claim(s)")
        lines.append("")

    # Interpretation
    if score >= 0.7:
        lines.append("INTERPRETATION: High convergence - agents strongly agree on claims")
    elif score >= 0.4:
        lines.append("INTERPRETATION: Moderate convergence - partial agreement on claims")
    elif score >= 0.15:
        lines.append("INTERPRETATION: Low convergence - limited agreement on claims")
    else:
        lines.append("INTERPRETATION: Minimal convergence - agents have distinct positions")

    lines.append("=" * 70)

    return "\n".join(lines)


def get_convergence_trajectory_summary(convergence_history: List[Dict[str, Any]]) -> str:
    """
    Generate summary of convergence evolution over multiple rounds.

    Args:
        convergence_history: List of convergence data from each round

    Returns:
        Formatted string showing convergence trajectory
    """
    if not convergence_history:
        return ""

    lines = [
        "=" * 70,
        "CONVERGENCE TRAJECTORY",
        "=" * 70
    ]

    for entry in convergence_history:
        round_num = entry.get("round", "?")
        score = entry.get("convergence_score", 0.0)
        shared = entry.get("shared_claim_pairs", 0)
        unique = entry.get("unique_claims_count", 0)

        lines.append(
            f"Round {round_num}: {score:.1%} convergence "
            f"({shared} shared pairs, {unique} unique)"
        )

    # Calculate trend
    if len(convergence_history) > 1:
        initial_score = convergence_history[0].get("convergence_score", 0.0)
        final_score = convergence_history[-1].get("convergence_score", 0.0)
        change = final_score - initial_score

        if change > 0.1:
            trend = "CONVERGING"
        elif change < -0.1:
            trend = "DIVERGING"
        else:
            trend = "STABLE"

        lines.append("")
        lines.append(f"Overall Trend: {trend} ({change:+.1%})")

    lines.append("=" * 70)

    return "\n".join(lines)
