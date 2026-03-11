"""
patches.py

Deterministic patch application for CBS belief objects.
Ensures that belief updates are auditable, graph-consistent, and propagate confidence correctly.

Supported patch operations:
- update_thesis: Weaken or strengthen thesis confidence
- update_claim: Modify claim properties (confidence, status, etc.)
- retire_claim: Mark claim as retracted with 0 confidence
- add_evidence: Add new evidence item
- update_assumption: Refine assumption statement or change type
- add_counterposition: Add new X# counterposition item
- update_counterposition: Modify counterposition properties
- add_uncertainty: Add new U# uncertainty item
"""

from __future__ import annotations
from typing import Dict, Any, List
import json
from datetime import datetime
from chal.beliefs.belief_graph import BeliefGraph


def apply_patches(
    prior_belief: Dict[str, Any],
    patches: List[Dict],
    propagate_confidence: bool = True
) -> Dict[str, Any]:
    """
    Apply a list of patch operations to a belief object.

    This function:
    1. Deep copies the prior belief to avoid mutation
    2. Applies each patch operation sequentially
    3. Propagates confidence changes through the dependency graph
    4. Auto-generates changelog entries
    5. Increments version number

    Args:
        prior_belief: The current CBS belief object
        patches: List of patch operation dicts
        propagate_confidence: If True, propagate confidence changes through graph (default: True)

    Returns:
        Updated belief object with version incremented and changelog added

    Raises:
        ValueError: If patch operation is invalid or references non-existent IDs
    """
    # Deep copy to avoid mutation
    updated = json.loads(json.dumps(prior_belief))

    # Increment version
    updated["version"] = prior_belief.get("version", 1) + 1

    # Track changes for changelog
    changes: List[str] = []
    confidence_changes: Dict[str, float] = {}  # {node_id: new_confidence}

    # Apply each patch
    for patch in patches:
        op = patch.get("op")

        if op == "update_thesis":
            change = patch.get("change")  # "weaken" or "strengthen"
            current_conf = updated["thesis"]["confidence"]

            if change == "weaken":
                new_conf = max(0.0, current_conf - 0.1)
            elif change == "strengthen":
                new_conf = min(1.0, current_conf + 0.1)
            else:
                continue

            updated["thesis"]["confidence"] = new_conf
            changes.append(f"Thesis confidence: {current_conf:.2f} → {new_conf:.2f}")

        elif op == "update_claim":
            target_id = patch.get("target_id")
            patch_changes = patch.get("changes", {})

            claim_found = False
            for claim in updated.get("claims", []):
                if claim["id"] == target_id:
                    claim_found = True
                    for key, value in patch_changes.items():
                        old_value = claim.get(key)
                        claim[key] = value

                        if key == "confidence":
                            confidence_changes[target_id] = value

                        changes.append(f"{target_id}.{key}: {old_value} → {value}")
                    break

            if not claim_found:
                raise ValueError(f"Patch references non-existent claim: {target_id}")

        elif op == "retire_claim":
            target_id = patch.get("target_id")

            claim_found = False
            for claim in updated.get("claims", []):
                if claim["id"] == target_id:
                    claim_found = True
                    claim["status"] = "retracted"
                    claim["confidence"] = 0.0
                    confidence_changes[target_id] = 0.0
                    changes.append(f"Retracted {target_id}")
                    break

            if not claim_found:
                raise ValueError(f"Patch references non-existent claim: {target_id}")

        elif op == "add_evidence":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_evidence patch requires valid item with 'id'")

            if "evidence" not in updated:
                updated["evidence"] = []

            updated["evidence"].append(item)
            changes.append(f"Added evidence {item.get('id')}")

        elif op == "update_assumption":
            target_id = patch.get("target_id")
            new_statement = patch.get("new_statement")
            new_type = patch.get("new_type")

            assumption_found = False
            for assumption in updated.get("assumptions", []):
                if assumption["id"] == target_id:
                    assumption_found = True
                    if new_statement:
                        assumption["statement"] = new_statement
                        changes.append(f"Refined {target_id} statement")
                    if new_type:
                        assumption["type"] = new_type
                        changes.append(f"Changed {target_id} type to {new_type}")
                    break

            if not assumption_found:
                raise ValueError(f"Patch references non-existent assumption: {target_id}")

        elif op == "add_counterposition":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_counterposition patch requires valid item with 'id'")
            if "counterpositions" not in updated:
                updated["counterpositions"] = []
            updated["counterpositions"].append(item)
            changes.append(f"Added counterposition {item.get('id')}")

        elif op == "update_counterposition":
            target_id = patch.get("target_id")
            patch_changes = patch.get("changes", {})
            cp_found = False
            for cp in updated.get("counterpositions", []):
                if cp["id"] == target_id:
                    cp_found = True
                    for key, value in patch_changes.items():
                        old_value = cp.get(key)
                        cp[key] = value
                        changes.append(f"{target_id}.{key}: {old_value} → {value}")
                    break
            if not cp_found:
                raise ValueError(f"Patch references non-existent counterposition: {target_id}")

        elif op == "add_uncertainty":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_uncertainty patch requires valid item with 'id'")
            if "uncertainties" not in updated:
                updated["uncertainties"] = []
            updated["uncertainties"].append(item)
            changes.append(f"Added uncertainty {item.get('id')}")

        else:
            # Unknown operation - log warning but don't fail
            changes.append(f"Warning: Unknown patch operation '{op}' skipped")

    # PROPAGATE CONFIDENCE CHANGES through dependency graph (BFS level-by-level)
    # v3 rule: A claim's confidence must not exceed the confidence of its weakest
    # *claim* dependency. Assumptions are tracked via counterpositions, not confidence scores,
    # so they are intentionally excluded from the propagation cap (filtered at line type=="claim").
    if propagate_confidence and confidence_changes:
        try:
            from collections import deque
            graph = BeliefGraph(updated)

            # Track live confidences so multi-hop propagation sees updated values
            current_confidences: Dict[str, float] = {
                claim["id"]: claim.get("confidence", 0.5)
                for claim in updated.get("claims", [])
            }

            # BFS worklist: process one hop at a time so each level's
            # updated confidence is visible to the next level
            worklist = list(confidence_changes.keys())
            processed: set = set()

            while worklist:
                changed_id = worklist.pop(0)
                if changed_id in processed:
                    continue
                processed.add(changed_id)

                # Find direct dependents only (one level at a time)
                direct_dependents = [
                    to_id for from_id, to_id, _ in graph.edges
                    if from_id == changed_id
                ]

                for dep_id in direct_dependents:
                    node_info = graph.nodes.get(dep_id)
                    if not node_info or node_info.get("type") != "claim":
                        continue
                    dep_claim = node_info.get("data", {})

                    # Get all claim dependencies of this dependent
                    all_deps = dep_claim.get("depends_on", []) + dep_claim.get("backing_evidence_ids", [])
                    dep_confidences = []

                    for dep_node_id in all_deps:
                        dep_node_info = graph.nodes.get(dep_node_id)
                        if dep_node_info and dep_node_info.get("type") == "claim":
                            # Use live confidence, not the stale graph snapshot
                            dep_confidences.append(
                                current_confidences.get(dep_node_id,
                                    dep_node_info["data"].get("confidence", 0.5))
                            )
                            # Evidence has no confidence field — skip

                    if dep_confidences:
                        min_dep_conf = min(dep_confidences)
                        current_conf = current_confidences.get(dep_id, dep_claim.get("confidence", 0.5))

                        # Propagation rule: a claim cannot be more confident than its weakest dependency
                        if min_dep_conf < current_conf:
                            for claim in updated.get("claims", []):
                                if claim["id"] == dep_id:
                                    claim["confidence"] = min_dep_conf
                                    changes.append(
                                        f"Propagated: {dep_id} confidence → {min_dep_conf:.2f} "
                                        f"(limited by {changed_id})"
                                    )
                                    break

                            # Update live tracking and queue dep_id for its own downstream
                            current_confidences[dep_id] = min_dep_conf
                            if dep_id not in processed:
                                worklist.append(dep_id)

        except Exception as e:
            # If propagation fails, log it but don't fail the entire patch operation
            changes.append(f"Warning: Confidence propagation failed: {e}")

    # Add changelog entry
    if "changelog" not in updated:
        updated["changelog"] = []

    updated["changelog"].append({
        "version": updated["version"],
        "changes": changes,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

    # Update metadata timestamp
    if "metadata" not in updated:
        updated["metadata"] = {}
    updated["metadata"]["last_updated"] = datetime.utcnow().isoformat() + "Z"

    return updated


def validate_patches(patches: List[Dict], belief: Dict[str, Any]) -> List[str]:
    """
    Validate patch operations before applying them.

    Args:
        patches: List of patch operations
        belief: Belief object to validate against

    Returns:
        List of validation errors (empty if valid)
    """
    errors: List[str] = []

    # Build ID sets for fast lookup
    assumption_ids = {a["id"] for a in belief.get("assumptions", []) if "id" in a}
    claim_ids = {c["id"] for c in belief.get("claims", []) if "id" in c}
    evidence_ids = {e["id"] for e in belief.get("evidence", []) if "id" in e}
    counterposition_ids = {x["id"] for x in belief.get("counterpositions", []) if "id" in x}
    uncertainty_ids = {u["id"] for u in belief.get("uncertainties", []) if "id" in u}

    for i, patch in enumerate(patches):
        op = patch.get("op")

        if not op:
            errors.append(f"Patch {i}: Missing 'op' field")
            continue

        if op == "update_thesis":
            change = patch.get("change")
            if change not in ["weaken", "strengthen"]:
                errors.append(f"Patch {i}: update_thesis requires change='weaken' or 'strengthen'")

        elif op == "update_claim":
            target_id = patch.get("target_id")
            if not target_id:
                errors.append(f"Patch {i}: update_claim missing target_id")
            elif target_id not in claim_ids:
                errors.append(f"Patch {i}: update_claim references non-existent claim '{target_id}'")

        elif op == "retire_claim":
            target_id = patch.get("target_id")
            if not target_id:
                errors.append(f"Patch {i}: retire_claim missing target_id")
            elif target_id not in claim_ids:
                errors.append(f"Patch {i}: retire_claim references non-existent claim '{target_id}'")

        elif op == "add_evidence":
            item = patch.get("item")
            if not item:
                errors.append(f"Patch {i}: add_evidence missing item")
            elif "id" not in item:
                errors.append(f"Patch {i}: add_evidence item missing 'id' field")
            elif item["id"] in evidence_ids:
                errors.append(f"Patch {i}: add_evidence item ID '{item['id']}' already exists")

        elif op == "update_assumption":
            target_id = patch.get("target_id")
            if not target_id:
                errors.append(f"Patch {i}: update_assumption missing target_id")
            elif target_id not in assumption_ids:
                errors.append(f"Patch {i}: update_assumption references non-existent assumption '{target_id}'")

        elif op == "add_counterposition":
            item = patch.get("item")
            if not item:
                errors.append(f"Patch {i}: add_counterposition missing item")
            elif "id" not in item:
                errors.append(f"Patch {i}: add_counterposition item missing 'id' field")
            else:
                if item["id"] in counterposition_ids:
                    errors.append(f"Patch {i}: add_counterposition item ID '{item['id']}' already exists")
                required_fields = ["targets", "attack_type", "statement", "strength", "my_response", "response_sufficiency"]
                for field in required_fields:
                    if field not in item:
                        errors.append(f"Patch {i}: add_counterposition item missing required field '{field}'")

        elif op == "update_counterposition":
            target_id = patch.get("target_id")
            if not target_id:
                errors.append(f"Patch {i}: update_counterposition missing target_id")
            elif target_id not in counterposition_ids:
                errors.append(f"Patch {i}: update_counterposition references non-existent counterposition '{target_id}'")

        elif op == "add_uncertainty":
            item = patch.get("item")
            if not item:
                errors.append(f"Patch {i}: add_uncertainty missing item")
            elif "id" not in item:
                errors.append(f"Patch {i}: add_uncertainty item missing 'id' field")
            elif item["id"] in uncertainty_ids:
                errors.append(f"Patch {i}: add_uncertainty item ID '{item['id']}' already exists")

        else:
            errors.append(f"Patch {i}: Unknown operation '{op}'")

    return errors
