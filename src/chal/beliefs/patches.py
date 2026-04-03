"""
patches.py

Deterministic patch application for CBS belief objects.
Ensures that belief updates are auditable, graph-consistent, and propagate strength correctly.

Supported patch operations:
- update_thesis: Set thesis strength, stance text, and/or summary bullets
- update_claim: Modify claim properties (strength, status, predictions, etc.)
- retire_claim: Mark claim as retracted with 0 strength
- add_claim: Add new C# claim item
- add_evidence: Add new evidence item
- update_evidence: Modify evidence properties (strength, summary, etc.)
- update_assumption: Refine assumption statement, type, or strength
- add_assumption: Add new A# assumption item
- add_counterposition: Add new X# counterposition item
- update_counterposition: Modify counterposition properties
- add_uncertainty: Add new U# uncertainty item
- resolve_uncertainty: Mark U# as resolved with a resolution note
"""

from __future__ import annotations
from typing import Dict, Any, List, Set
import json
from datetime import datetime
from chal.beliefs.belief_graph import BeliefGraph

BREADTH_SENSITIVITY = 1.5


def apply_patches(
    prior_belief: Dict[str, Any],
    patches: List[Dict],
    propagate_strength: bool = True,
    breadth_sensitivity: float = BREADTH_SENSITIVITY
) -> Dict[str, Any]:
    """
    Apply a list of patch operations to a belief object.

    This function:
    1. Deep copies the prior belief to avoid mutation
    2. Applies each patch operation sequentially
    3. Propagates strength changes through the dependency graph
    4. Sets thesis strength: avg(active claim strengths) x breadth multiplier
    5. Auto-generates changelog entries
    6. Increments version number

    Args:
        prior_belief: The current CBS belief object
        patches: List of patch operation dicts
        propagate_strength: If True, propagate strength changes through graph (default: True)
        breadth_sensitivity: Exponent p in breadth formula n^p / (n^p + 1) (default: 1.5)

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
    strength_changes: Dict[str, float] = {}  # {node_id: new_strength}

    # Apply each patch
    for patch in patches:
        op = patch.get("op")

        if op == "update_thesis":
            new_strength = patch.get("new_strength")
            change = patch.get("change")  # "weaken" or "strengthen"
            new_stance = patch.get("stance")
            new_bullets = patch.get("summary_bullets")
            current_str = updated["thesis"]["strength"]

            # Strength update (existing logic)
            if new_strength is not None:
                new_str = max(0.0, min(1.0, new_strength))
                updated["thesis"]["strength"] = new_str
                changes.append(f"Thesis strength: {current_str:.2f} → {new_str:.2f}")
            elif change == "weaken":
                new_str = max(0.0, current_str - 0.1)
                updated["thesis"]["strength"] = new_str
                changes.append(f"Thesis strength: {current_str:.2f} → {new_str:.2f}")
            elif change == "strengthen":
                new_str = min(1.0, current_str + 0.1)
                updated["thesis"]["strength"] = new_str
                changes.append(f"Thesis strength: {current_str:.2f} → {new_str:.2f}")

            # Stance text update
            if new_stance is not None:
                updated["thesis"]["stance"] = new_stance
                changes.append("Thesis stance text updated")

            # Summary bullets update
            if new_bullets is not None and isinstance(new_bullets, list):
                updated["thesis"]["summary_bullets"] = new_bullets
                changes.append("Thesis summary bullets updated")

            # Strength reasoning update (agent's reasoning; may be overwritten
            # by the authoritative formula enforcement step below)
            new_reasoning = patch.get("strength_reasoning")
            if new_reasoning is not None and isinstance(new_reasoning, str):
                updated["thesis"]["strength_reasoning"] = new_reasoning

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

                        if key == "strength":
                            strength_changes[target_id] = value

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
                    claim["strength"] = 0.0
                    strength_changes[target_id] = 0.0
                    changes.append(f"Retracted {target_id}")
                    break

            if not claim_found:
                raise ValueError(f"Patch references non-existent claim: {target_id}")

        elif op == "add_claim":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_claim patch requires valid item with 'id'")
            if "claims" not in updated:
                updated["claims"] = []
            updated["claims"].append(item)
            changes.append(f"Added claim {item.get('id')}")

        elif op == "add_evidence":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_evidence patch requires valid item with 'id'")

            if "evidence" not in updated:
                updated["evidence"] = []

            # Default status to "active" if not provided
            if "status" not in item:
                item["status"] = "active"

            updated["evidence"].append(item)
            changes.append(f"Added evidence {item.get('id')}")

        elif op == "update_evidence":
            target_id = patch.get("target_id")
            patch_changes = patch.get("changes", {})

            ev_found = False
            for ev in updated.get("evidence", []):
                if ev["id"] == target_id:
                    ev_found = True
                    for key, value in patch_changes.items():
                        old_value = ev.get(key)
                        ev[key] = value
                        if key == "strength":
                            strength_changes[target_id] = value
                        changes.append(f"{target_id}.{key}: {old_value} → {value}")
                    # Retraction enforcement: retracted evidence gets strength 0.0
                    if ev.get("status") == "retracted" and ev.get("strength", 0.0) != 0.0:
                        ev["strength"] = 0.0
                        strength_changes[target_id] = 0.0
                        changes.append(f"{target_id}.strength forced to 0.0 (retracted)")
                    break

            if not ev_found:
                raise ValueError(f"Patch references non-existent evidence: {target_id}")

        elif op == "update_assumption":
            target_id = patch.get("target_id")
            new_statement = patch.get("new_statement")
            new_type = patch.get("new_type")
            patch_changes = patch.get("changes", {})

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
                    for key, value in patch_changes.items():
                        old_value = assumption.get(key)
                        assumption[key] = value
                        if key == "strength":
                            strength_changes[target_id] = value
                        changes.append(f"{target_id}.{key}: {old_value} → {value}")
                    # Retraction enforcement: retracted assumptions get strength 0.0
                    if assumption.get("status") == "retracted" and assumption.get("strength", 0.0) != 0.0:
                        assumption["strength"] = 0.0
                        strength_changes[target_id] = 0.0
                        changes.append(f"{target_id}.strength forced to 0.0 (retracted)")
                    break

            if not assumption_found:
                raise ValueError(f"Patch references non-existent assumption: {target_id}")

        elif op == "add_assumption":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_assumption patch requires valid item with 'id'")
            if "assumptions" not in updated:
                updated["assumptions"] = []
            # Default status to "active" if not provided
            if "status" not in item:
                item["status"] = "active"
            if "supports_claims" not in item:
                item["supports_claims"] = []
            updated["assumptions"].append(item)
            changes.append(f"Added assumption {item.get('id')}")

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

        elif op == "resolve_uncertainty":
            target_id = patch.get("target_id")
            resolution_note = patch.get("resolution_note", "")
            u_found = False
            for u in updated.get("uncertainties", []):
                if u["id"] == target_id:
                    u_found = True
                    u["status"] = "resolved"
                    u["resolution_note"] = resolution_note
                    changes.append(f"Resolved uncertainty {target_id}: {resolution_note}")
                    break
            if not u_found:
                raise ValueError(f"Patch references non-existent uncertainty: {target_id}")

        else:
            # Unknown operation - log warning but don't fail
            changes.append(f"Warning: Unknown patch operation '{op}' skipped")

    # PROPAGATE STRENGTH CHANGES through dependency graph (BFS level-by-level)
    # Rule: A claim's strength must not exceed the LOWEST strength among its
    # active/revised dependencies (C#, A#, or E#). Retracted dependencies are
    # excluded — they don't drag down dependent claims.
    if propagate_strength and strength_changes:
        try:
            graph = BeliefGraph(updated)

            # Track live strengths for all node types so multi-hop propagation
            # sees updated values
            current_strengths: Dict[str, float] = {}
            for claim in updated.get("claims", []):
                current_strengths[claim["id"]] = claim.get("strength", 0.5)
            for assumption in updated.get("assumptions", []):
                current_strengths[assumption["id"]] = assumption.get("strength", 0.5)
            for ev in updated.get("evidence", []):
                current_strengths[ev["id"]] = ev.get("strength", 0.5)

            # Build a lookup for node status (to exclude retracted nodes from cap)
            node_statuses: Dict[str, str] = {}
            for claim in updated.get("claims", []):
                node_statuses[claim["id"]] = claim.get("status", "active")
            for assumption in updated.get("assumptions", []):
                node_statuses[assumption["id"]] = assumption.get("status", "active")
            for ev in updated.get("evidence", []):
                node_statuses[ev["id"]] = ev.get("status", "active")

            # BFS worklist: process one hop at a time so each level's
            # updated strength is visible to the next level
            worklist = list(strength_changes.keys())
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

                    # Get all dependencies of this claim (A#, C#, E#)
                    all_deps = dep_claim.get("depends_on", [])
                    dep_strengths = []

                    for dep_node_id in all_deps:
                        dep_node_info = graph.nodes.get(dep_node_id)
                        if dep_node_info and dep_node_info.get("type") in ("claim", "assumption", "evidence"):
                            # Skip retracted nodes — they don't limit dependent claims
                            if node_statuses.get(dep_node_id) == "retracted":
                                continue
                            # Use live strength, not the stale graph snapshot
                            dep_strengths.append(
                                current_strengths.get(dep_node_id,
                                    dep_node_info["data"].get("strength", 0.5))
                            )

                    if dep_strengths:
                        min_dep_str = min(dep_strengths)
                        current_str = current_strengths.get(dep_id, dep_claim.get("strength", 0.5))

                        # Propagation rule: a claim cannot be stronger than the
                        # lowest-strength active/revised dependency
                        if min_dep_str < current_str:
                            for claim in updated.get("claims", []):
                                if claim["id"] == dep_id:
                                    claim["strength"] = min_dep_str
                                    changes.append(
                                        f"Propagated: {dep_id} strength → {min_dep_str:.2f} "
                                        f"(limited by {changed_id})"
                                    )
                                    break

                            # Update live tracking and queue dep_id for its own downstream
                            current_strengths[dep_id] = min_dep_str
                            if dep_id not in processed:
                                worklist.append(dep_id)

        except Exception as e:
            # If propagation fails, log it but don't fail the entire patch operation
            changes.append(f"Warning: Strength propagation failed: {e}")

    # THESIS STRENGTH: always equals avg(active claim strengths) x breadth multiplier
    # breadth = n^p / (n^p + 1) where n = number of active claims, p = breadth_sensitivity
    if propagate_strength:
        active_claim_strengths = [
            c.get("strength", 0.5)
            for c in updated.get("claims", [])
            if c.get("status") != "retracted"
        ]
        if active_claim_strengths:
            n = len(active_claim_strengths)
            p = breadth_sensitivity
            avg_str = sum(active_claim_strengths) / n
            n_p = n ** p
            breadth = n_p / (n_p + 1)
            thesis_strength = round(avg_str * breadth, 4)
            old_str = updated["thesis"].get("strength", 0.5)

            # Build strength_reasoning showing the formula with actual numbers
            strengths_str = ", ".join(f"{s:.2f}" for s in active_claim_strengths)
            reasoning = (
                f"avg({strengths_str}) × ({n}^{p} / ({n}^{p} + 1)) "
                f"= {avg_str:.2f} × {breadth:.2f} = {thesis_strength}"
            )
            updated["thesis"]["strength"] = thesis_strength
            updated["thesis"]["strength_reasoning"] = reasoning

            if old_str != thesis_strength:
                changes.append(
                    f"Thesis strength set to {thesis_strength} "
                    f"({reasoning})"
                )

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


# --- Validation constants ---
_STATUS_ENUM: Set[str] = {"active", "revised", "retracted"}
_UNCERTAINTY_STATUS_ENUM: Set[str] = {"active", "resolved"}
_IMPORTANCE_ENUM: Set[str] = {"high", "medium", "low"}
_ASSUMPTION_TYPE_ENUM: Set[str] = {"foundational", "empirical", "methodological"}
_EVIDENCE_TYPE_ENUM: Set[str] = {"empirical", "conceptual", "expert_consensus"}
_ATTACK_TYPE_ENUM: Set[str] = {"undermining", "rebutting", "undercutting"}
_SUFFICIENCY_ENUM: Set[str] = {"sufficient", "partial", "unaddressed"}
_CHANGE_ENUM: Set[str] = {"weaken", "strengthen"}

_UPDATE_CLAIM_WHITELIST: Set[str] = {
    "strength", "strength_justification", "statement", "status",
    "depends_on", "predictions", "inference_chain", "type",
}
_UPDATE_EVIDENCE_WHITELIST: Set[str] = {
    "strength", "strength_justification", "summary", "source",
    "status", "relevance_to_claims", "type",
}
_UPDATE_ASSUMPTION_WHITELIST: Set[str] = {
    "strength", "strength_justification", "statement", "status",
    "type", "supports_claims",
}
_UPDATE_COUNTERPOSITION_WHITELIST: Set[str] = {
    "my_response", "response_sufficiency", "statement",
    "attack_type", "targets",
}


def validate_patches(patches: List[Dict], belief: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Validate patch operations before applying them.

    Args:
        patches: List of patch operations
        belief: Belief object to validate against

    Returns:
        Dict mapping patch index to list of validation errors.
        Empty dict means all patches are valid.
    """
    errors: Dict[int, List[str]] = {}

    def err(idx: int, msg: str) -> None:
        errors.setdefault(idx, []).append(msg)

    # Build ID sets for fast lookup
    assumption_ids = {a["id"] for a in belief.get("assumptions", []) if "id" in a}
    claim_ids = {c["id"] for c in belief.get("claims", []) if "id" in c}
    evidence_ids = {e["id"] for e in belief.get("evidence", []) if "id" in e}
    counterposition_ids = {x["id"] for x in belief.get("counterpositions", []) if "id" in x}
    uncertainty_ids = {u["id"] for u in belief.get("uncertainties", []) if "id" in u}
    all_ref_ids = assumption_ids | claim_ids | evidence_ids

    for i, patch in enumerate(patches):
        op = patch.get("op")

        if not op:
            err(i, "Missing 'op' field")
            continue

        if op == "update_thesis":
            new_strength = patch.get("new_strength")
            change = patch.get("change")
            stance = patch.get("stance")
            summary_bullets = patch.get("summary_bullets")

            # At least one field must be present
            has_strength_op = new_strength is not None or change is not None
            has_text_op = stance is not None or summary_bullets is not None
            if not has_strength_op and not has_text_op:
                err(i, "update_thesis requires at least one of: new_strength, change, stance, summary_bullets")

            # Mutual exclusivity
            if new_strength is not None and change is not None:
                err(i, "update_thesis cannot have both new_strength and change (mutually exclusive)")

            # Validate strength fields
            if new_strength is not None:
                if not isinstance(new_strength, (int, float)):
                    err(i, "update_thesis new_strength must be a number")
                elif not (0.0 <= new_strength <= 1.0):
                    err(i, "update_thesis new_strength must be between 0.0 and 1.0")

            if change is not None and change not in _CHANGE_ENUM:
                err(i, "update_thesis change must be 'weaken' or 'strengthen'")

            # Validate stance
            if stance is not None:
                if not isinstance(stance, str) or not stance.strip():
                    err(i, "update_thesis stance must be a non-empty string")

            # Validate summary_bullets
            if summary_bullets is not None:
                if not isinstance(summary_bullets, list) or len(summary_bullets) == 0:
                    err(i, "update_thesis summary_bullets must be a non-empty list")
                elif not all(isinstance(b, str) for b in summary_bullets):
                    err(i, "update_thesis summary_bullets must contain only strings")

        elif op == "update_claim":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "update_claim missing target_id")
            elif target_id not in claim_ids:
                err(i, f"update_claim references non-existent claim '{target_id}'")

            changes = patch.get("changes")
            if not changes or not isinstance(changes, dict):
                err(i, "update_claim requires non-empty changes dict")
            else:
                # Whitelist check
                unknown = set(changes.keys()) - _UPDATE_CLAIM_WHITELIST
                if unknown:
                    err(i, f"update_claim changes contains unknown fields: {sorted(unknown)}")
                if "strength" in changes:
                    s = changes["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "update_claim strength must be between 0.0 and 1.0")
                if "status" in changes and changes["status"] not in _STATUS_ENUM:
                    err(i, f"update_claim status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                if "strength_justification" in changes:
                    if not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip():
                        err(i, "update_claim strength_justification must be a non-empty string")
                if "statement" in changes:
                    if not isinstance(changes["statement"], str) or not changes["statement"].strip():
                        err(i, "update_claim statement must be a non-empty string")

        elif op == "retire_claim":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "retire_claim missing target_id")
            elif target_id not in claim_ids:
                err(i, f"retire_claim references non-existent claim '{target_id}'")

        elif op == "add_claim":
            item = patch.get("item")
            if not item:
                err(i, "add_claim missing item")
            elif "id" not in item:
                err(i, "add_claim item missing 'id' field")
            else:
                if item["id"] in claim_ids:
                    err(i, f"add_claim item ID '{item['id']}' already exists")
                required_fields = ["id", "type", "statement", "depends_on", "strength", "status", "predictions", "inference_chain", "strength_justification"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_claim item missing required field '{field}'")
                # Validate depends_on references (A#, E#, C# all valid)
                if "depends_on" in item and isinstance(item["depends_on"], list):
                    for dep in item["depends_on"]:
                        if dep not in all_ref_ids:
                            err(i, f"add_claim depends_on references non-existent ID '{dep}'")
                # Validate strength
                if "strength" in item:
                    s = item["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "add_claim strength must be between 0.0 and 1.0")
                # Validate status
                if "status" in item and item["status"] not in _STATUS_ENUM:
                    err(i, f"add_claim status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # Validate predictions
                if "predictions" in item:
                    preds = item["predictions"]
                    if not isinstance(preds, list) or len(preds) == 0:
                        err(i, "add_claim predictions must be a non-empty array")
                    elif isinstance(preds, list):
                        for j, pred in enumerate(preds):
                            if not isinstance(pred, dict):
                                err(i, f"add_claim predictions[{j}] must be a dict")
                            else:
                                for pf in ("statement", "test", "decision_criterion"):
                                    if pf not in pred:
                                        err(i, f"add_claim predictions[{j}] missing required field '{pf}'")
                # Validate inference_chain
                if "inference_chain" in item:
                    ic = item["inference_chain"]
                    if not isinstance(ic, list) or len(ic) == 0:
                        err(i, "add_claim inference_chain must be a non-empty array")
                # Validate strength_justification
                if "strength_justification" in item:
                    if not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip():
                        err(i, "add_claim strength_justification must be a non-empty string")

        elif op == "add_evidence":
            item = patch.get("item")
            if not item:
                err(i, "add_evidence missing item")
            elif "id" not in item:
                err(i, "add_evidence item missing 'id' field")
            else:
                if item["id"] in evidence_ids:
                    err(i, f"add_evidence item ID '{item['id']}' already exists")
                # Required fields
                required_fields = ["id", "type", "summary", "source", "relevance_to_claims", "strength", "strength_justification"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_evidence item missing required field '{field}'")
                # Type enum
                if "type" in item and item["type"] not in _EVIDENCE_TYPE_ENUM:
                    err(i, f"add_evidence type must be one of: {', '.join(sorted(_EVIDENCE_TYPE_ENUM))}")
                # Non-empty strings
                if "summary" in item:
                    if not isinstance(item["summary"], str) or not item["summary"].strip():
                        err(i, "add_evidence summary must be a non-empty string")
                if "source" in item:
                    if not isinstance(item["source"], str) or not item["source"].strip():
                        err(i, "add_evidence source must be a non-empty string")
                if "strength_justification" in item:
                    if not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip():
                        err(i, "add_evidence strength_justification must be a non-empty string")
                # relevance_to_claims validation
                if "relevance_to_claims" in item:
                    rtc = item["relevance_to_claims"]
                    if not isinstance(rtc, list) or len(rtc) == 0:
                        err(i, "add_evidence relevance_to_claims must be a non-empty list")
                    elif isinstance(rtc, list):
                        for ref in rtc:
                            if ref not in claim_ids:
                                err(i, f"add_evidence relevance_to_claims references non-existent claim '{ref}'")
                # Strength validation
                if "strength" in item:
                    s = item["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "add_evidence strength must be between 0.0 and 1.0")
                # Status enum
                if "status" in item and item["status"] not in _STATUS_ENUM:
                    err(i, f"add_evidence status must be one of: {', '.join(sorted(_STATUS_ENUM))}")

        elif op == "update_evidence":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "update_evidence missing target_id")
            elif target_id not in evidence_ids:
                err(i, f"update_evidence references non-existent evidence '{target_id}'")

            changes = patch.get("changes")
            if not changes or not isinstance(changes, dict):
                err(i, "update_evidence requires non-empty changes dict")
            else:
                # Whitelist check
                unknown = set(changes.keys()) - _UPDATE_EVIDENCE_WHITELIST
                if unknown:
                    err(i, f"update_evidence changes contains unknown fields: {sorted(unknown)}")
                if "strength" in changes:
                    s = changes["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "update_evidence strength must be between 0.0 and 1.0")
                if "status" in changes and changes["status"] not in _STATUS_ENUM:
                    err(i, f"update_evidence status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # Type enum
                if "type" in changes and changes["type"] not in _EVIDENCE_TYPE_ENUM:
                    err(i, f"update_evidence type must be one of: {', '.join(sorted(_EVIDENCE_TYPE_ENUM))}")
                # Non-empty strings
                if "strength_justification" in changes:
                    if not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip():
                        err(i, "update_evidence strength_justification must be a non-empty string")
                if "summary" in changes:
                    if not isinstance(changes["summary"], str) or not changes["summary"].strip():
                        err(i, "update_evidence summary must be a non-empty string")
                if "source" in changes:
                    if not isinstance(changes["source"], str) or not changes["source"].strip():
                        err(i, "update_evidence source must be a non-empty string")

        elif op == "update_assumption":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "update_assumption missing target_id")
            elif target_id not in assumption_ids:
                err(i, f"update_assumption references non-existent assumption '{target_id}'")

            changes = patch.get("changes", {})
            new_statement = patch.get("new_statement")
            new_type = patch.get("new_type")

            # At least one content field required
            has_changes = isinstance(changes, dict) and len(changes) > 0
            if not new_statement and not new_type and not has_changes:
                err(i, "update_assumption requires at least one of: new_statement, new_type, changes")

            # new_type enum
            if new_type is not None and new_type not in _ASSUMPTION_TYPE_ENUM:
                err(i, f"update_assumption new_type must be one of: {', '.join(sorted(_ASSUMPTION_TYPE_ENUM))}")

            # new_statement non-empty
            if new_statement is not None:
                if not isinstance(new_statement, str) or not new_statement.strip():
                    err(i, "update_assumption new_statement must be a non-empty string")

            if has_changes:
                # Whitelist check
                unknown = set(changes.keys()) - _UPDATE_ASSUMPTION_WHITELIST
                if unknown:
                    err(i, f"update_assumption changes contains unknown fields: {sorted(unknown)}")
                if "strength" in changes:
                    s = changes["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "update_assumption strength must be between 0.0 and 1.0")
                if "status" in changes and changes["status"] not in _STATUS_ENUM:
                    err(i, f"update_assumption status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # Type enum in changes
                if "type" in changes and changes["type"] not in _ASSUMPTION_TYPE_ENUM:
                    err(i, f"update_assumption type must be one of: {', '.join(sorted(_ASSUMPTION_TYPE_ENUM))}")
                # Non-empty strings
                if "strength_justification" in changes:
                    if not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip():
                        err(i, "update_assumption strength_justification must be a non-empty string")
                if "statement" in changes:
                    if not isinstance(changes["statement"], str) or not changes["statement"].strip():
                        err(i, "update_assumption statement must be a non-empty string")

        elif op == "add_assumption":
            item = patch.get("item")
            if not item:
                err(i, "add_assumption missing item")
            elif "id" not in item:
                err(i, "add_assumption item missing 'id' field")
            else:
                if item["id"] in assumption_ids:
                    err(i, f"add_assumption item ID '{item['id']}' already exists")
                # Default supports_claims to empty array if missing
                if "supports_claims" not in item:
                    item["supports_claims"] = []
                required_fields = ["id", "type", "statement", "strength", "strength_justification"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_assumption item missing required field '{field}'")
                if "type" in item and item["type"] not in _ASSUMPTION_TYPE_ENUM:
                    err(i, f"add_assumption type must be one of: {', '.join(sorted(_ASSUMPTION_TYPE_ENUM))}")
                if "strength" in item:
                    s = item["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "add_assumption strength must be between 0.0 and 1.0")
                if "status" in item and item["status"] not in _STATUS_ENUM:
                    err(i, f"add_assumption status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # strength_justification non-empty
                if "strength_justification" in item:
                    if not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip():
                        err(i, "add_assumption strength_justification must be a non-empty string")

        elif op == "add_counterposition":
            item = patch.get("item")
            if not item:
                err(i, "add_counterposition missing item")
            elif "id" not in item:
                err(i, "add_counterposition item missing 'id' field")
            else:
                if item["id"] in counterposition_ids:
                    err(i, f"add_counterposition item ID '{item['id']}' already exists")
                required_fields = ["targets", "attack_type", "statement", "my_response", "response_sufficiency"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_counterposition item missing required field '{field}'")
                # Enum validations
                if "attack_type" in item and item["attack_type"] not in _ATTACK_TYPE_ENUM:
                    err(i, f"add_counterposition attack_type must be one of: {', '.join(sorted(_ATTACK_TYPE_ENUM))}")
                if "response_sufficiency" in item and item["response_sufficiency"] not in _SUFFICIENCY_ENUM:
                    err(i, f"add_counterposition response_sufficiency must be one of: {', '.join(sorted(_SUFFICIENCY_ENUM))}")
                # Targets validation
                if "targets" in item:
                    targets = item["targets"]
                    if not isinstance(targets, list) or len(targets) == 0:
                        err(i, "add_counterposition targets must be a non-empty list")
                    elif isinstance(targets, list):
                        for ref in targets:
                            if ref not in all_ref_ids:
                                err(i, f"add_counterposition targets references non-existent ID '{ref}'")
                # Non-empty strings
                if "statement" in item:
                    if not isinstance(item["statement"], str) or not item["statement"].strip():
                        err(i, "add_counterposition statement must be a non-empty string")
                if "my_response" in item:
                    if not isinstance(item["my_response"], str) or not item["my_response"].strip():
                        err(i, "add_counterposition my_response must be a non-empty string")

        elif op == "update_counterposition":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "update_counterposition missing target_id")
            elif target_id not in counterposition_ids:
                err(i, f"update_counterposition references non-existent counterposition '{target_id}'")

            changes = patch.get("changes")
            if not changes or not isinstance(changes, dict):
                err(i, "update_counterposition requires non-empty changes dict")
            else:
                # Whitelist check
                unknown = set(changes.keys()) - _UPDATE_COUNTERPOSITION_WHITELIST
                if unknown:
                    err(i, f"update_counterposition changes contains unknown fields: {sorted(unknown)}")
                # Enum validations
                if "response_sufficiency" in changes and changes["response_sufficiency"] not in _SUFFICIENCY_ENUM:
                    err(i, f"update_counterposition response_sufficiency must be one of: {', '.join(sorted(_SUFFICIENCY_ENUM))}")
                if "attack_type" in changes and changes["attack_type"] not in _ATTACK_TYPE_ENUM:
                    err(i, f"update_counterposition attack_type must be one of: {', '.join(sorted(_ATTACK_TYPE_ENUM))}")
                # Non-empty strings
                if "statement" in changes:
                    if not isinstance(changes["statement"], str) or not changes["statement"].strip():
                        err(i, "update_counterposition statement must be a non-empty string")
                if "my_response" in changes:
                    if not isinstance(changes["my_response"], str) or not changes["my_response"].strip():
                        err(i, "update_counterposition my_response must be a non-empty string")

        elif op == "add_uncertainty":
            item = patch.get("item")
            if not item:
                err(i, "add_uncertainty missing item")
            elif "id" not in item:
                err(i, "add_uncertainty item missing 'id' field")
            else:
                if item["id"] in uncertainty_ids:
                    err(i, f"add_uncertainty item ID '{item['id']}' already exists")
                # Required fields
                required_fields = ["id", "targets", "question", "status", "importance"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_uncertainty item missing required field '{field}'")
                # Targets validation
                if "targets" in item:
                    targets = item["targets"]
                    if not isinstance(targets, list) or len(targets) == 0:
                        err(i, "add_uncertainty targets must be a non-empty list")
                    elif isinstance(targets, list):
                        for ref in targets:
                            if ref not in all_ref_ids:
                                err(i, f"add_uncertainty targets references non-existent ID '{ref}'")
                # Question non-empty
                if "question" in item:
                    if not isinstance(item["question"], str) or not item["question"].strip():
                        err(i, "add_uncertainty question must be a non-empty string")
                # Status enum
                if "status" in item and item["status"] not in _UNCERTAINTY_STATUS_ENUM:
                    err(i, f"add_uncertainty status must be one of: {', '.join(sorted(_UNCERTAINTY_STATUS_ENUM))}")
                # Importance enum
                if "importance" in item and item["importance"] not in _IMPORTANCE_ENUM:
                    err(i, f"add_uncertainty importance must be one of: {', '.join(sorted(_IMPORTANCE_ENUM))}")

        elif op == "resolve_uncertainty":
            target_id = patch.get("target_id")
            resolution_note = patch.get("resolution_note", "")
            if not target_id:
                err(i, "resolve_uncertainty missing target_id")
            elif target_id not in uncertainty_ids:
                err(i, f"resolve_uncertainty references non-existent uncertainty '{target_id}'")
            if not resolution_note:
                err(i, "resolve_uncertainty requires non-empty resolution_note")

        else:
            err(i, f"Unknown operation '{op}'")

    return errors
