"""
patches.py

Deterministic patch application for CBS belief objects.
Ensures that belief updates are auditable, graph-consistent, and propagate confidence correctly.

Supported patch operations:
- update_thesis: Weaken or strengthen thesis confidence
- update_claim: Modify claim properties (confidence, status, etc.)
- retire_claim: Mark claim as retracted with 0 confidence
- add_evidence: Add new evidence item
- update_assumption: Refine assumption statement
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

            assumption_found = False
            for assumption in updated.get("assumptions", []):
                if assumption["id"] == target_id:
                    assumption_found = True
                    if new_statement:
                        old_statement = assumption.get("statement", "")
                        assumption["statement"] = new_statement
                        changes.append(f"Refined {target_id}")
                    break

            if not assumption_found:
                raise ValueError(f"Patch references non-existent assumption: {target_id}")

        else:
            # Unknown operation - log warning but don't fail
            changes.append(f"Warning: Unknown patch operation '{op}' skipped")

    # PROPAGATE CONFIDENCE CHANGES through dependency graph
    if propagate_confidence and confidence_changes:
        try:
            graph = BeliefGraph(updated)

            for changed_id, new_conf in confidence_changes.items():
                # Find all claims that depend on this one
                dependent_ids = graph.get_dependent_nodes(changed_id)

                for dep_id in dependent_ids:
                    # Get the dependent claim (check graph node type, not data type)
                    node_info = graph.nodes.get(dep_id)
                    if not node_info or node_info.get("type") != "claim":
                        continue
                    dep_claim = node_info.get("data", {})

                    # Get all dependencies of this dependent claim
                    all_deps = dep_claim.get("depends_on", []) + dep_claim.get("backing_evidence_ids", [])
                    dep_confidences = []

                    for dep_node_id in all_deps:
                        dep_node_info = graph.nodes.get(dep_node_id)
                        if dep_node_info and dep_node_info.get("type") == "claim":
                            dep_data = dep_node_info.get("data", {})
                            dep_confidences.append(dep_data.get("confidence", 0.5))
                            # Evidence doesn't have confidence, skip

                    if dep_confidences:
                        min_dep_conf = min(dep_confidences)
                        current_conf = dep_claim.get("confidence", 0.5)

                        # Propagation rule: dependent claims cannot be more confident than weakest dependency
                        if min_dep_conf < current_conf:
                            # Find the claim in updated belief and modify it
                            for claim in updated.get("claims", []):
                                if claim["id"] == dep_id:
                                    claim["confidence"] = min_dep_conf
                                    changes.append(
                                        f"Propagated: {dep_id} confidence → {min_dep_conf:.2f} "
                                        f"(limited by {changed_id})"
                                    )
                                    break
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

        else:
            errors.append(f"Patch {i}: Unknown operation '{op}'")

    return errors
