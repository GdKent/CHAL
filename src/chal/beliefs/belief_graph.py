"""
belief_graph.py

Represents belief structure as a directed acyclic graph (DAG) for validation and analysis.

This module transforms CBS belief objects into graph structures where:
- Nodes: THESIS, assumptions (A#), claims (C#), evidence (E#), counterpositions (X#), uncertainties (U#)
- Edges: dependency relationships (depends_on, targets)

DAG structure:
    X# ──challenges──→ A#/E#/C#
    U# ──questions───→ A#/E#/C#
    A# ──supports────→ C#
    E# ──supports────→ C#
    C# ──supports────→ THESIS

The graph enables:
1. Structural validation (broken links, circular dependencies)
2. Critical path analysis (single points of failure)
3. Strength propagation (ensuring consistency with dependencies)
4. Argument robustness metrics
"""

from __future__ import annotations
from typing import Dict, Any, List, Set, Tuple, Optional


class BeliefGraph:
    """
    Directed acyclic graph representation of a CBS belief object.

    Attributes:
        nodes: Dict mapping node IDs to their full data
        edges: List of (source_id, target_id, edge_type) tuples
        belief_id: Unique identifier for the source belief
        version: Version number of the source belief
    """

    def __init__(self, belief: Dict[str, Any]):
        """
        Construct graph from a CBS belief object.

        Args:
            belief: Valid CBS belief dict with schema_version, belief_id, etc.
        """
        self.belief = belief
        self.belief_id = belief.get("belief_id", "unknown")
        self.version = belief.get("version", 1)

        # Build node and edge structures
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (from_id, to_id, edge_type)

        self._build_graph()

    def _build_graph(self):
        """Extract nodes and edges from belief object."""
        # Add THESIS as a formal DAG node
        thesis = self.belief.get("thesis", {})
        if thesis:
            self.nodes["THESIS"] = {"type": "thesis", "data": thesis}

        # Add all nodes by category
        for assumption in self.belief.get("assumptions", []):
            if "id" in assumption:
                self.nodes[assumption["id"]] = {
                    "type": "assumption",
                    "data": assumption
                }

        for claim in self.belief.get("claims", []):
            if "id" in claim:
                self.nodes[claim["id"]] = {
                    "type": "claim",
                    "data": claim
                }

        for evidence in self.belief.get("evidence", []):
            if "id" in evidence:
                self.nodes[evidence["id"]] = {
                    "type": "evidence",
                    "data": evidence
                }

        for cp in self.belief.get("counterpositions", []):
            if "id" in cp:
                self.nodes[cp["id"]] = {
                    "type": "counterposition",
                    "data": cp
                }

        for uncertainty in self.belief.get("uncertainties", []):
            if "id" in uncertainty:
                self.nodes[uncertainty["id"]] = {
                    "type": "uncertainty",
                    "data": uncertainty
                }

        # Build edges from dependency relationships
        for claim in self.belief.get("claims", []):
            claim_id = claim.get("id")
            if not claim_id:
                continue

            # depends_on: claim depends on assumptions, evidence, or other claims
            for dep_id in claim.get("depends_on", []):
                self.edges.append((dep_id, claim_id, "supports"))

            # Active claims support THESIS
            if claim.get("status") != "retracted" and thesis:
                self.edges.append((claim_id, "THESIS", "supports"))

        # counterpositions target claims/assumptions/evidence
        for cp in self.belief.get("counterpositions", []):
            cp_id = cp.get("id")
            if not cp_id:
                continue

            for target_id in cp.get("targets", []):
                self.edges.append((cp_id, target_id, "challenges"))

        # uncertainties target claims/assumptions/evidence
        for uncertainty in self.belief.get("uncertainties", []):
            u_id = uncertainty.get("id")
            if not u_id:
                continue

            for target_id in uncertainty.get("targets", []):
                self.edges.append((u_id, target_id, "questions"))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        node = self.nodes.get(node_id)
        return node["data"] if node else None

    def _node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes

    def validate_links(self) -> List[str]:
        """
        Validate all dependency links in the belief graph.

        Returns:
            List of human-readable validation errors (empty if valid)

        Checks:
        - All depends_on IDs exist
        - All counterposition/uncertainty targets exist
        - No circular dependencies (BLOCKING)
        - No orphaned claims (BLOCKING)
        """
        errors: List[str] = []

        # Check that all edges point to existing nodes
        for from_id, to_id, edge_type in self.edges:
            if not self._node_exists(from_id):
                errors.append(f"BLOCKING ERROR: Edge references non-existent source node: {from_id}")
            if not self._node_exists(to_id):
                errors.append(f"BLOCKING ERROR: Edge references non-existent target node: {to_id}")

        # Check for circular dependencies (BLOCKING - logically invalid)
        if self._has_cycle():
            errors.append("BLOCKING ERROR: Circular dependency detected in belief graph. Claims cannot depend on themselves directly or indirectly.")

        # Check for orphaned claims (BLOCKING - claims must have support)
        orphans = self._find_orphaned_claims()
        for orphan_id in orphans:
            errors.append(f"BLOCKING ERROR: Claim {orphan_id} has no supporting evidence or assumptions. All claims must be grounded in evidence or assumptions.")

        return errors

    def _has_cycle(self) -> bool:
        """
        Detect cycles in the directed graph using DFS.

        Returns:
            True if a cycle exists, False otherwise
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for from_id, to_id, _ in self.edges:
            if from_id in adj:
                adj[from_id].append(to_id)

        # DFS with recursion stack tracking
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # Cycle found

            rec_stack.remove(node_id)
            return False

        # Check all components
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def _find_orphaned_claims(self) -> List[str]:
        """
        Find claims that have no incoming support edges (no evidence or assumptions).

        Only "supports" edges count as support.
        "challenges" and "questions" edges do not provide support.
        THESIS is excluded from orphan detection.

        Returns:
            List of claim IDs with no support
        """
        SUPPORT_EDGE_TYPES = {"supports"}

        # Get all claim IDs (THESIS is not a claim)
        claim_ids = {node_id for node_id, node in self.nodes.items() if node["type"] == "claim"}

        # Get all claims that have incoming support edges
        supported_claims = {
            to_id for from_id, to_id, edge_type in self.edges
            if to_id in claim_ids and edge_type in SUPPORT_EDGE_TYPES
        }

        # Orphans are claims with no incoming support edges
        orphans = claim_ids - supported_claims

        return list(orphans)

    def get_support_chain(self, node_id: str) -> List[str]:
        """
        Get all nodes that support this node (recursive traversal backwards).

        Args:
            node_id: The node to find support for

        Returns:
            List of node IDs that transitively support this node
        """
        support = set()

        def backtrack(current_id: str):
            for from_id, to_id, _ in self.edges:
                if to_id == current_id and from_id not in support:
                    support.add(from_id)
                    backtrack(from_id)

        backtrack(node_id)
        return list(support)

    def get_dependent_nodes(self, node_id: str) -> List[str]:
        """
        Get all nodes that depend on this node (recursive traversal forwards).

        Args:
            node_id: The node to find dependents for

        Returns:
            List of node IDs that transitively depend on this node
        """
        dependents = set()

        def forward_track(current_id: str):
            for from_id, to_id, _ in self.edges:
                if from_id == current_id and to_id not in dependents:
                    dependents.add(to_id)
                    forward_track(to_id)

        forward_track(node_id)
        return list(dependents)

    def find_critical_paths(self) -> List[List[str]]:
        """
        Identify single-point-of-failure chains from assumptions to thesis.

        A critical path is a sequence of nodes where:
        - Removing any single node breaks the entire chain
        - The chain connects foundational assumptions to key claims

        Returns:
            List of paths, where each path is a list of node IDs
        """
        critical_paths: List[List[str]] = []

        # Find all assumption nodes
        assumptions = [node_id for node_id, node in self.nodes.items() if node["type"] == "assumption"]

        # Find all high-strength claims (>0.7) as targets
        key_claims = [
            node_id for node_id, node in self.nodes.items()
            if node["type"] == "claim" and node["data"].get("strength", 0) > 0.7
        ]

        # For each assumption, find paths to key claims
        for assumption_id in assumptions:
            for claim_id in key_claims:
                paths = self._find_all_paths(assumption_id, claim_id)
                # A path is critical if it's the ONLY path between these nodes
                if len(paths) == 1:
                    critical_paths.append(paths[0])

        return critical_paths

    def _find_all_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """
        Find all paths from start_id to end_id.

        Args:
            start_id: Starting node ID
            end_id: Target node ID

        Returns:
            List of paths, where each path is a list of node IDs
        """
        all_paths: List[List[str]] = []

        def dfs(current_id: str, path: List[str], visited: Set[str]):
            if current_id == end_id:
                all_paths.append(path.copy())
                return

            for from_id, to_id, _ in self.edges:
                if from_id == current_id and to_id not in visited:
                    visited.add(to_id)
                    path.append(to_id)
                    dfs(to_id, path, visited)
                    path.pop()
                    visited.remove(to_id)

        dfs(start_id, [start_id], {start_id})
        return all_paths

    def get_graph_metrics(self) -> Dict[str, Any]:
        """
        Calculate structural metrics for the belief graph.

        Returns:
            Dict with metrics like node counts, edge counts, critical paths, etc.
        """
        metrics = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_counts": {
                "thesis": 1 if "THESIS" in self.nodes else 0,
                "assumptions": sum(1 for n in self.nodes.values() if n["type"] == "assumption"),
                "claims": sum(1 for n in self.nodes.values() if n["type"] == "claim"),
                "evidence": sum(1 for n in self.nodes.values() if n["type"] == "evidence"),
                "counterpositions": sum(1 for n in self.nodes.values() if n["type"] == "counterposition"),
                "uncertainties": sum(1 for n in self.nodes.values() if n["type"] == "uncertainty")
            },
            "critical_path_count": len(self.find_critical_paths()),
            "orphaned_claims": self._find_orphaned_claims(),
            "has_cycles": self._has_cycle()
        }

        return metrics
