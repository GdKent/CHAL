"""
patches.py

Deterministic patch application for CBS belief objects.
Ensures that belief updates are auditable, graph-consistent, and propagate strength correctly.

Supported patch operations:
- update_thesis: Set thesis strength, stance text, and/or summary bullets
- update_claim: Modify claim properties (strength, status, predictions, etc.). Retraction enforcement: setting status to "retracted" forces strength to 0.0.
- add_claim: Add new C# claim item
- add_evidence: Add new evidence item
- update_evidence: Modify evidence properties (strength, summary, etc.)
- update_assumption: Refine assumption statement, type, or strength
- add_assumption: Add new A# assumption item
- add_counterposition: Add new X# counterposition item
- update_counterposition: Modify counterposition properties
- add_uncertainty: Add new U# uncertainty item
- resolve_uncertainty: Mark U# as resolved with a resolution note
- add_definition: Add new D# definition item
- update_definition: Modify definition properties (definition, strength, status, used_by)
"""

from __future__ import annotations

import json
import re
from typing import Any

from chal.beliefs.belief_graph import BeliefGraph
from chal.beliefs.schema import ALLOWED_REF_PREFIXES, validate_inference_chain

BREADTH_SENSITIVITY = 1.0
ORPHAN_AE_CAP = 0.6     # A#/E# with no active D# support — lenient
ORPHAN_CLAIM_CAP = 0.2   # C# with no active A#/E# support — strict


def _summarise_ic_diff(old_ic, new_ic) -> str:
    """Produce a concise human-readable summary of an inference_chain change."""

    def _ic_stats(ic):
        if not isinstance(ic, list):
            return 0, None
        premises = sum(1 for s in ic if isinstance(s, dict) and s.get("role") == "premise")
        inf_type = None
        for s in ic:
            if isinstance(s, dict) and s.get("role") == "inference":
                inf_type = s.get("inference_type")
                break
        return premises, inf_type

    old_p, old_t = _ic_stats(old_ic)
    new_p, new_t = _ic_stats(new_ic)

    parts = []
    if old_p != new_p:
        parts.append(f"{old_p} premises → {new_p} premises")
    else:
        parts.append(f"{new_p} premises")
    if old_t != new_t:
        parts.append(f"{old_t} → {new_t}")
    elif new_t:
        parts.append(new_t)
    return ", ".join(parts) if parts else "replaced"


def initialize_defense_tracking(belief: dict) -> dict:
    """Ensure all strength-bearing nodes have original_strength and consecutive_defenses.

    Call this after parsing a Stage 1 belief to ensure tracking fields exist.
    Modifies the belief dict in-place and returns it for convenience.
    """
    for key in ("definitions", "assumptions", "evidence", "claims"):
        for node in belief.get(key, []):
            if "strength" in node and "original_strength" not in node:
                node["original_strength"] = node["strength"]
            if "consecutive_defenses" not in node:
                node["consecutive_defenses"] = 0
    return belief


def apply_patches(
    prior_belief: dict[str, Any],
    patches: list[dict],
    propagate_strength: bool = True,
    breadth_sensitivity: float = BREADTH_SENSITIVITY
) -> dict[str, Any]:
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
        breadth_sensitivity: Exponent p in breadth formula n^p / (n^p + 1) (default: 1.0)

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
    changes: list[str] = []
    strength_changes: dict[str, float] = {}  # {node_id: new_strength}

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
            # Preserve system-managed defense tracking fields
            patch_changes.pop("original_strength", None)
            patch_changes.pop("consecutive_defenses", None)

            claim_found = False
            for claim in updated.get("claims", []):
                if claim["id"] == target_id:
                    claim_found = True
                    for key, value in patch_changes.items():
                        old_value = claim.get(key)
                        # inference_chain is always a full replacement (not a delta).
                        # The new array completely replaces the old one.
                        claim[key] = value

                        if key == "strength":
                            strength_changes[target_id] = value

                        if key == "inference_chain":
                            changes.append(
                                f"{target_id}.inference_chain updated "
                                f"({_summarise_ic_diff(old_value, value)})"
                            )
                        else:
                            changes.append(f"{target_id}.{key}: {old_value} → {value}")
                    # Retraction enforcement: retracted claims get strength 0.0
                    if claim.get("status") == "retracted" and claim.get("strength", 0.0) != 0.0:
                        claim["strength"] = 0.0
                        strength_changes[target_id] = 0.0
                        changes.append(f"{target_id}.strength forced to 0.0 (retracted)")
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
            # Initialize defense tracking fields
            if "original_strength" not in item and "strength" in item:
                item["original_strength"] = item["strength"]
            if "consecutive_defenses" not in item:
                item["consecutive_defenses"] = 0
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
            # Initialize defense tracking fields
            if "original_strength" not in item and "strength" in item:
                item["original_strength"] = item["strength"]
            if "consecutive_defenses" not in item:
                item["consecutive_defenses"] = 0
            changes.append(f"Added evidence {item.get('id')}")

        elif op == "update_evidence":
            target_id = patch.get("target_id")
            patch_changes = patch.get("changes", {})
            # Preserve system-managed defense tracking fields
            patch_changes.pop("original_strength", None)
            patch_changes.pop("consecutive_defenses", None)

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
            # Preserve system-managed defense tracking fields
            patch_changes.pop("original_strength", None)
            patch_changes.pop("consecutive_defenses", None)

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
            # Initialize defense tracking fields
            if "original_strength" not in item and "strength" in item:
                item["original_strength"] = item["strength"]
            if "consecutive_defenses" not in item:
                item["consecutive_defenses"] = 0
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

        elif op == "add_definition":
            item = patch.get("item")
            if not item or "id" not in item:
                raise ValueError("add_definition patch requires valid item with 'id'")
            if "definitions" not in updated:
                updated["definitions"] = []

            # Validate ID unique and format
            new_id = item["id"]
            if not re.match(r"^D\d+$", new_id):
                raise ValueError(f"add_definition ID '{new_id}' must match D<number> format")
            existing_ids = {d["id"] for d in updated.get("definitions", [])}
            if new_id in existing_ids:
                raise ValueError(f"add_definition ID '{new_id}' already exists")

            # Default status to "active" if not provided
            if "status" not in item:
                item["status"] = "active"

            updated["definitions"].append(item)
            # Initialize defense tracking fields
            if "original_strength" not in item and "strength" in item:
                item["original_strength"] = item["strength"]
            if "consecutive_defenses" not in item:
                item["consecutive_defenses"] = 0

            # Side effect: append D# ID to supported_by_definitions on each referenced A#/E#
            for used_id in item.get("used_by", []):
                for collection_key in ("assumptions", "evidence"):
                    for node in updated.get(collection_key, []):
                        if node["id"] == used_id:
                            sbd = node.get("supported_by_definitions", [])
                            if new_id not in sbd:
                                sbd.append(new_id)
                                node["supported_by_definitions"] = sbd

            changes.append(f"Added definition {new_id}: '{item.get('term', '')}'")

        elif op == "update_definition":
            target_id = patch.get("target_id")
            patch_changes = patch.get("changes", {})
            # Preserve system-managed defense tracking fields
            patch_changes.pop("original_strength", None)
            patch_changes.pop("consecutive_defenses", None)

            # Reject immutable fields
            immutable = {"id", "term"}
            attempted_immutable = immutable & set(patch_changes.keys())
            if attempted_immutable:
                raise ValueError(
                    f"update_definition cannot modify immutable fields: {sorted(attempted_immutable)}"
                )

            def_found = False
            for defn in updated.get("definitions", []):
                if defn["id"] == target_id:
                    def_found = True
                    old_used_by = list(defn.get("used_by", []))

                    for key, value in patch_changes.items():
                        old_value = defn.get(key)
                        defn[key] = value

                        if key == "strength":
                            strength_changes[target_id] = value

                        changes.append(f"{target_id}.{key}: {old_value} → {value}")

                    # Retraction enforcement: retracted definitions get strength 0.0
                    if defn.get("status") == "retracted" and defn.get("strength", 0.0) != 0.0:
                        defn["strength"] = 0.0
                        strength_changes[target_id] = 0.0
                        changes.append(f"{target_id}.strength forced to 0.0 (retracted)")

                    # Side effect: if used_by changed, update supported_by_definitions
                    new_used_by = list(defn.get("used_by", []))
                    if "used_by" in patch_changes:
                        removed = set(old_used_by) - set(new_used_by)
                        added = set(new_used_by) - set(old_used_by)
                        for collection_key in ("assumptions", "evidence"):
                            for node in updated.get(collection_key, []):
                                if node["id"] in removed:
                                    sbd = node.get("supported_by_definitions", [])
                                    if target_id in sbd:
                                        sbd.remove(target_id)
                                        node["supported_by_definitions"] = sbd
                                if node["id"] in added:
                                    sbd = node.get("supported_by_definitions", [])
                                    if target_id not in sbd:
                                        sbd.append(target_id)
                                        node["supported_by_definitions"] = sbd
                    break

            if not def_found:
                raise ValueError(f"Patch references non-existent definition: {target_id}")

        else:
            # Unknown operation - log warning but don't fail
            changes.append(f"Warning: Unknown patch operation '{op}' skipped")

    # --- D# ceiling enforcement (pre-step before BFS) ---
    # A#/E# strength cannot exceed the LOWEST non-retracted D# strength from
    # supported_by_definitions. If all D# are retracted, cap at ORPHAN_AE_CAP.
    if propagate_strength:
        definitions = updated.get("definitions", [])
        for collection_key in ("assumptions", "evidence"):
            for node in updated.get(collection_key, []):
                if node.get("status") == "retracted":
                    continue
                supported_defs = node.get("supported_by_definitions", [])
                active_def_strengths = [
                    d["strength"] for d in definitions
                    if d["id"] in supported_defs and d.get("status") != "retracted"
                ] if definitions else []

                if active_def_strengths:
                    ceiling = min(active_def_strengths)
                    if node["strength"] > ceiling:
                        old_str = node["strength"]
                        node["strength"] = round(ceiling, 4)
                        changes.append(
                            f"Propagated: {node['id']} strength → {ceiling} "
                            f"(limited by definition ceiling)"
                        )
                        strength_changes[node["id"]] = ceiling
                elif supported_defs:
                    # Had D# support but all are now retracted — apply orphan cap
                    if node["strength"] > ORPHAN_AE_CAP:
                        node["strength"] = ORPHAN_AE_CAP
                        changes.append(
                            f"Capped: {node['id']} strength → {ORPHAN_AE_CAP} "
                            f"(no active definitional support)"
                        )
                        strength_changes[node["id"]] = ORPHAN_AE_CAP

    # PROPAGATE STRENGTH CHANGES through dependency graph (BFS level-by-level)
    # Rule: A claim's strength must not exceed the LOWEST strength among its
    # active/revised dependencies (C#, A#, or E#). Retracted dependencies are
    # excluded — they don't drag down dependent claims.
    if propagate_strength and strength_changes:
        try:
            graph = BeliefGraph(updated)

            # Track live strengths for all node types so multi-hop propagation
            # sees updated values
            current_strengths: dict[str, float] = {}
            for claim in updated.get("claims", []):
                current_strengths[claim["id"]] = claim.get("strength", 0.5)
            for assumption in updated.get("assumptions", []):
                current_strengths[assumption["id"]] = assumption.get("strength", 0.5)
            for ev in updated.get("evidence", []):
                current_strengths[ev["id"]] = ev.get("strength", 0.5)

            # Build a lookup for node status (to exclude retracted nodes from cap)
            node_statuses: dict[str, str] = {}
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
                    elif all_deps:
                        # All dependencies retracted — orphaned claim, cap at ORPHAN_CLAIM_CAP
                        current_str = current_strengths.get(dep_id, dep_claim.get("strength", 0.5))
                        if current_str > ORPHAN_CLAIM_CAP:
                            for claim in updated.get("claims", []):
                                if claim["id"] == dep_id:
                                    claim["strength"] = ORPHAN_CLAIM_CAP
                                    changes.append(
                                        f"Capped: {dep_id} strength → {ORPHAN_CLAIM_CAP} "
                                        f"(no active dependencies — unfounded claim)"
                                    )
                                    break
                            current_strengths[dep_id] = ORPHAN_CLAIM_CAP
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
    })

    return updated  # type: ignore[no-any-return]


# --- Validation constants ---
_STATUS_ENUM: set[str] = {"active", "revised", "retracted"}
_UNCERTAINTY_STATUS_ENUM: set[str] = {"active", "resolved"}
_IMPORTANCE_ENUM: set[str] = {"high", "medium", "low"}
_ASSUMPTION_TYPE_ENUM: set[str] = {"foundational", "empirical", "methodological", "scoping"}
_EVIDENCE_TYPE_ENUM: set[str] = {"empirical", "conceptual", "expert_consensus"}
_ATTACK_TYPE_ENUM: set[str] = {"undermining", "rebutting", "undercutting"}
_SUFFICIENCY_ENUM: set[str] = {"sufficient", "partial", "unaddressed", "moot"}
_CHANGE_ENUM: set[str] = {"weaken", "strengthen"}

_UPDATE_CLAIM_WHITELIST: set[str] = {
    "strength", "strength_justification", "statement", "status",
    "depends_on", "predictions", "inference_chain", "type",
}
_UPDATE_EVIDENCE_WHITELIST: set[str] = {
    "strength", "strength_justification", "summary", "source",
    "status", "supports_claims", "type", "supported_by_definitions",
}
_UPDATE_ASSUMPTION_WHITELIST: set[str] = {
    "strength", "strength_justification", "statement", "status",
    "type", "supports_claims", "supported_by_definitions",
}
_UPDATE_COUNTERPOSITION_WHITELIST: set[str] = {
    "my_response", "response_sufficiency", "statement",
    "attack_type", "targets",
}
_UPDATE_DEFINITION_WHITELIST: set[str] = {
    "definition", "strength", "strength_justification", "status", "used_by",
}
_DEFINITION_ID_RE = re.compile(r"^D\d+$")


def validate_patches(patches: list[dict], belief: dict[str, Any]) -> dict[int, list[str]]:
    """
    Validate patch operations before applying them.

    Args:
        patches: List of patch operations
        belief: Belief object to validate against

    Returns:
        Dict mapping patch index to list of validation errors.
        Empty dict means all patches are valid.
    """
    errors: dict[int, list[str]] = {}

    def err(idx: int, msg: str) -> None:
        errors.setdefault(idx, []).append(msg)

    # Build ID sets for fast lookup
    assumption_ids = {a["id"] for a in belief.get("assumptions", []) if "id" in a}
    claim_ids = {c["id"] for c in belief.get("claims", []) if "id" in c}
    evidence_ids = {e["id"] for e in belief.get("evidence", []) if "id" in e}
    definition_ids = {d["id"] for d in belief.get("definitions", []) if "id" in d}
    counterposition_ids = {x["id"] for x in belief.get("counterpositions", []) if "id" in x}
    uncertainty_ids = {u["id"] for u in belief.get("uncertainties", []) if "id" in u}
    all_ref_ids = assumption_ids | claim_ids | evidence_ids | definition_ids

    # --- Projection pass: pre-register IDs from all add_* patches ---
    # This allows cross-references within a batch (e.g., add_definition D5
    # with used_by: ["A5"] where A5 is added in a later patch).
    # Save original belief IDs before projection (for duplicate detection).
    _belief_definition_ids = set(definition_ids)
    _belief_assumption_ids = set(assumption_ids)
    _belief_evidence_ids = set(evidence_ids)
    _belief_claim_ids = set(claim_ids)

    projected_ids: dict[str, str] = {}  # id -> collection type
    for patch in patches:
        op = patch.get("op", "")
        if op.startswith("add_"):
            item = patch.get("item", {})
            pid = item.get("id", "")
            if not pid:
                continue
            if op == "add_definition":
                projected_ids[pid] = "definition"
            elif op == "add_assumption":
                projected_ids[pid] = "assumption"
            elif op == "add_evidence":
                projected_ids[pid] = "evidence"
            elif op == "add_claim":
                projected_ids[pid] = "claim"

    # Inject projected IDs into tracking sets (for reference validation)
    for pid, ptype in projected_ids.items():
        if ptype == "definition":
            definition_ids.add(pid)
        elif ptype == "assumption":
            assumption_ids.add(pid)
            all_ref_ids.add(pid)
        elif ptype == "evidence":
            evidence_ids.add(pid)
            all_ref_ids.add(pid)
        elif ptype == "claim":
            claim_ids.add(pid)
            all_ref_ids.add(pid)

    # Track IDs added within the batch (for within-batch duplicate detection)
    batch_add_ids: set[str] = set()

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
            if stance is not None and (not isinstance(stance, str) or not stance.strip()):
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
                if "strength_justification" in changes and (not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip()):
                    err(i, "update_claim strength_justification must be a non-empty string")
                if "statement" in changes and (not isinstance(changes["statement"], str) or not changes["statement"].strip()):
                    err(i, "update_claim statement must be a non-empty string")
                if "inference_chain" in changes:
                    ic = changes["inference_chain"]
                    if not isinstance(ic, list) or len(ic) == 0:
                        err(i, "update_claim inference_chain must be a non-empty array")
                    elif isinstance(ic, list) and len(ic) > 0:
                        ic_errors: list[str] = []
                        validate_inference_chain(ic, target_id or "?", ic_errors)
                        for ic_err in ic_errors:
                            err(i, ic_err)

        elif op == "add_claim":
            item = patch.get("item")
            if not item:
                err(i, "add_claim missing item")
            elif "id" not in item:
                err(i, "add_claim item missing 'id' field")
            else:
                if item["id"] in _belief_claim_ids or item["id"] in batch_add_ids:
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
                        if isinstance(dep, str) and len(dep) >= 2 and dep[0] not in ALLOWED_REF_PREFIXES["depends_on"]:
                            err(i, f"add_claim depends_on contains '{dep}' — only A#/E#/C# IDs allowed")
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
                # Validate inference_chain (structural)
                if "inference_chain" in item:
                    ic = item["inference_chain"]
                    if not isinstance(ic, list) or len(ic) == 0:
                        err(i, "add_claim inference_chain must be a non-empty array")
                    elif isinstance(ic, list) and len(ic) > 0:
                        ic_errors_2: list[str] = []
                        validate_inference_chain(ic, item.get("id", "?"), ic_errors_2)
                        for ic_err in ic_errors_2:
                            err(i, ic_err)
                # Validate strength_justification
                if "strength_justification" in item and (not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip()):
                    err(i, "add_claim strength_justification must be a non-empty string")
                # Track new ID for forward references within the batch
                if i not in errors:
                    claim_ids.add(item["id"])
                    all_ref_ids.add(item["id"])
                    batch_add_ids.add(item["id"])

        elif op == "add_evidence":
            item = patch.get("item")
            if not item:
                err(i, "add_evidence missing item")
            elif "id" not in item:
                err(i, "add_evidence item missing 'id' field")
            else:
                if item["id"] in _belief_evidence_ids or item["id"] in batch_add_ids:
                    err(i, f"add_evidence item ID '{item['id']}' already exists")
                # Required fields
                required_fields = ["id", "type", "summary", "source", "supports_claims", "strength", "strength_justification"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_evidence item missing required field '{field}'")
                # Type enum
                if "type" in item and item["type"] not in _EVIDENCE_TYPE_ENUM:
                    err(i, f"add_evidence type must be one of: {', '.join(sorted(_EVIDENCE_TYPE_ENUM))}")
                # Non-empty strings
                if "summary" in item and (not isinstance(item["summary"], str) or not item["summary"].strip()):
                    err(i, "add_evidence summary must be a non-empty string")
                if "source" in item and (not isinstance(item["source"], str) or not item["source"].strip()):
                    err(i, "add_evidence source must be a non-empty string")
                if "strength_justification" in item and (not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip()):
                    err(i, "add_evidence strength_justification must be a non-empty string")
                # supports_claims validation
                if "supports_claims" in item:
                    rtc = item["supports_claims"]
                    if not isinstance(rtc, list) or len(rtc) == 0:
                        err(i, "add_evidence supports_claims must be a non-empty list")
                    elif isinstance(rtc, list):
                        for ref in rtc:
                            if ref not in claim_ids:
                                err(i, f"add_evidence supports_claims references non-existent claim '{ref}'")
                            if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["supports_claims"]:
                                err(i, f"add_evidence supports_claims contains '{ref}' — only C# IDs allowed")
                # supported_by_definitions validation
                if "supported_by_definitions" in item:
                    sbd = item["supported_by_definitions"]
                    if isinstance(sbd, list):
                        for ref in sbd:
                            if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["supported_by_definitions"]:
                                err(i, f"add_evidence supported_by_definitions contains '{ref}' — only D# IDs allowed")
                            elif isinstance(ref, str) and ref not in definition_ids:
                                err(i, f"add_evidence supported_by_definitions references non-existent definition '{ref}'")
                # Strength validation
                if "strength" in item:
                    s = item["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "add_evidence strength must be between 0.0 and 1.0")
                # Status enum
                if "status" in item and item["status"] not in _STATUS_ENUM:
                    err(i, f"add_evidence status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # Track new ID for forward references within the batch
                if i not in errors:
                    evidence_ids.add(item["id"])
                    all_ref_ids.add(item["id"])
                    batch_add_ids.add(item["id"])

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
                if "strength_justification" in changes and (not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip()):
                    err(i, "update_evidence strength_justification must be a non-empty string")
                if "summary" in changes and (not isinstance(changes["summary"], str) or not changes["summary"].strip()):
                    err(i, "update_evidence summary must be a non-empty string")
                if "source" in changes and (not isinstance(changes["source"], str) or not changes["source"].strip()):
                    err(i, "update_evidence source must be a non-empty string")
                # supported_by_definitions validation
                if "supported_by_definitions" in changes:
                    sbd = changes["supported_by_definitions"]
                    if not isinstance(sbd, list):
                        err(i, "update_evidence supported_by_definitions must be a list")
                    elif isinstance(sbd, list):
                        for ref in sbd:
                            if isinstance(ref, str) and ref not in definition_ids:
                                err(i, f"update_evidence supported_by_definitions references non-existent definition '{ref}'")

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
            if new_statement is not None and (not isinstance(new_statement, str) or not new_statement.strip()):
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
                if "strength_justification" in changes and (not isinstance(changes["strength_justification"], str) or not changes["strength_justification"].strip()):
                    err(i, "update_assumption strength_justification must be a non-empty string")
                if "statement" in changes and (not isinstance(changes["statement"], str) or not changes["statement"].strip()):
                    err(i, "update_assumption statement must be a non-empty string")
                # supported_by_definitions validation
                if "supported_by_definitions" in changes:
                    sbd = changes["supported_by_definitions"]
                    if not isinstance(sbd, list):
                        err(i, "update_assumption supported_by_definitions must be a list")
                    elif isinstance(sbd, list):
                        for ref in sbd:
                            if isinstance(ref, str) and ref not in definition_ids:
                                err(i, f"update_assumption supported_by_definitions references non-existent definition '{ref}'")

        elif op == "add_assumption":
            item = patch.get("item")
            if not item:
                err(i, "add_assumption missing item")
            elif "id" not in item:
                err(i, "add_assumption item missing 'id' field")
            else:
                if item["id"] in _belief_assumption_ids or item["id"] in batch_add_ids:
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
                if "strength_justification" in item and (not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip()):
                    err(i, "add_assumption strength_justification must be a non-empty string")
                # supports_claims validation
                if "supports_claims" in item and isinstance(item["supports_claims"], list):
                    for ref in item["supports_claims"]:
                        if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["supports_claims"]:
                            err(i, f"add_assumption supports_claims contains '{ref}' — only C# IDs allowed")
                        elif ref not in claim_ids:
                            err(i, f"add_assumption supports_claims references non-existent claim '{ref}'")
                # supported_by_definitions validation
                if "supported_by_definitions" in item and isinstance(item["supported_by_definitions"], list):
                    for ref in item["supported_by_definitions"]:
                        if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["supported_by_definitions"]:
                            err(i, f"add_assumption supported_by_definitions contains '{ref}' — only D# IDs allowed")
                        elif isinstance(ref, str) and ref not in definition_ids:
                            err(i, f"add_assumption supported_by_definitions references non-existent definition '{ref}'")
                # Track new ID for forward references within the batch
                if i not in errors:
                    assumption_ids.add(item["id"])
                    all_ref_ids.add(item["id"])
                    batch_add_ids.add(item["id"])

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
                            if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["counterposition_targets"]:
                                err(i, f"add_counterposition targets contains '{ref}' — only C#/A#/E#/D# IDs allowed")
                # Non-empty strings
                if "statement" in item and (not isinstance(item["statement"], str) or not item["statement"].strip()):
                    err(i, "add_counterposition statement must be a non-empty string")
                if "my_response" in item:
                    sufficiency = item.get("response_sufficiency", "")
                    if sufficiency not in ("unaddressed", "moot") and (not isinstance(item["my_response"], str) or not item["my_response"].strip()):
                        err(i, "add_counterposition my_response must be a non-empty string (required when response_sufficiency is not 'unaddressed' or 'moot')")
                # Track new ID for forward references within the batch
                if i not in errors:
                    counterposition_ids.add(item["id"])

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
                # Guard: "moot" is terminal — cannot be changed to another value
                if "response_sufficiency" in changes and target_id:
                    for cp in belief.get("counterpositions", []):
                        if cp["id"] == target_id and cp.get("response_sufficiency") == "moot":
                            if changes["response_sufficiency"] != "moot":
                                err(i, f"update_counterposition cannot change response_sufficiency from 'moot' — "
                                       f"'moot' is terminal (counterposition {target_id} targets a retracted claim)")
                            break
                if "attack_type" in changes and changes["attack_type"] not in _ATTACK_TYPE_ENUM:
                    err(i, f"update_counterposition attack_type must be one of: {', '.join(sorted(_ATTACK_TYPE_ENUM))}")
                # Non-empty strings
                if "statement" in changes and (not isinstance(changes["statement"], str) or not changes["statement"].strip()):
                    err(i, "update_counterposition statement must be a non-empty string")
                if "my_response" in changes and (not isinstance(changes["my_response"], str) or not changes["my_response"].strip()):
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
                            if isinstance(ref, str) and len(ref) >= 2 and ref[0] not in ALLOWED_REF_PREFIXES["uncertainty_targets"]:
                                err(i, f"add_uncertainty targets contains '{ref}' — only A#/E#/C#/D# IDs allowed")
                # Question non-empty
                if "question" in item and (not isinstance(item["question"], str) or not item["question"].strip()):
                    err(i, "add_uncertainty question must be a non-empty string")
                # Status enum
                if "status" in item and item["status"] not in _UNCERTAINTY_STATUS_ENUM:
                    err(i, f"add_uncertainty status must be one of: {', '.join(sorted(_UNCERTAINTY_STATUS_ENUM))}")
                # Importance enum
                if "importance" in item and item["importance"] not in _IMPORTANCE_ENUM:
                    err(i, f"add_uncertainty importance must be one of: {', '.join(sorted(_IMPORTANCE_ENUM))}")
                # Track new ID for forward references within the batch
                if i not in errors:
                    uncertainty_ids.add(item["id"])

        elif op == "resolve_uncertainty":
            target_id = patch.get("target_id")
            resolution_note = patch.get("resolution_note", "")
            if not target_id:
                err(i, "resolve_uncertainty missing target_id")
            elif target_id not in uncertainty_ids:
                err(i, f"resolve_uncertainty references non-existent uncertainty '{target_id}'")
            if not resolution_note:
                err(i, "resolve_uncertainty requires non-empty resolution_note")

        elif op == "add_definition":
            item = patch.get("item")
            if not item:
                err(i, "add_definition missing item")
            elif "id" not in item:
                err(i, "add_definition item missing 'id' field")
            else:
                # ID format and uniqueness
                d_id = item["id"]
                if not _DEFINITION_ID_RE.match(d_id):
                    err(i, f"add_definition ID '{d_id}' must match D<number> format")
                if d_id in _belief_definition_ids or d_id in batch_add_ids:
                    err(i, f"add_definition item ID '{d_id}' already exists")
                # Required fields
                required_fields = ["id", "term", "definition", "strength",
                                   "strength_justification", "used_by"]
                for field in required_fields:
                    if field not in item:
                        err(i, f"add_definition item missing required field '{field}'")
                # Non-empty strings
                if "term" in item and (not isinstance(item["term"], str) or not item["term"].strip()):
                    err(i, "add_definition term must be a non-empty string")
                if "definition" in item and (not isinstance(item["definition"], str) or not item["definition"].strip()):
                    err(i, "add_definition definition must be a non-empty string")
                if "strength_justification" in item and (not isinstance(item["strength_justification"], str) or not item["strength_justification"].strip()):
                    err(i, "add_definition strength_justification must be a non-empty string")
                # Strength validation
                if "strength" in item:
                    s = item["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "add_definition strength must be between 0.0 and 1.0")
                # Status enum
                if "status" in item and item["status"] not in _STATUS_ENUM:
                    err(i, f"add_definition status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # used_by validation
                if "used_by" in item:
                    used_by = item["used_by"]
                    if not isinstance(used_by, list) or len(used_by) == 0:
                        err(i, "add_definition used_by must be a non-empty list")
                    elif isinstance(used_by, list):
                        ae_ids = assumption_ids | evidence_ids
                        for ref in used_by:
                            if ref not in ae_ids:
                                err(i, f"add_definition used_by references non-existent A#/E# '{ref}'")
                # Track new ID for forward references within the batch
                if i not in errors:
                    definition_ids.add(item["id"])
                    all_ref_ids.add(item["id"])
                    batch_add_ids.add(item["id"])

        elif op == "update_definition":
            target_id = patch.get("target_id")
            if not target_id:
                err(i, "update_definition missing target_id")
            elif target_id not in definition_ids:
                err(i, f"update_definition references non-existent definition '{target_id}'")

            patch_changes = patch.get("changes")
            if not patch_changes or not isinstance(patch_changes, dict):
                err(i, "update_definition requires non-empty changes dict")
            else:
                # Whitelist check
                unknown = set(patch_changes.keys()) - _UPDATE_DEFINITION_WHITELIST
                if unknown:
                    err(i, f"update_definition changes contains unknown fields: {sorted(unknown)}")
                # Reject immutable fields
                immutable_attempted = {"id", "term"} & set(patch_changes.keys())
                if immutable_attempted:
                    err(i, f"update_definition cannot modify immutable fields: {sorted(immutable_attempted)}")
                # Strength validation
                if "strength" in patch_changes:
                    s = patch_changes["strength"]
                    if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0):
                        err(i, "update_definition strength must be between 0.0 and 1.0")
                # Status enum
                if "status" in patch_changes and patch_changes["status"] not in _STATUS_ENUM:
                    err(i, f"update_definition status must be one of: {', '.join(sorted(_STATUS_ENUM))}")
                # Non-empty strings
                if "definition" in patch_changes and (not isinstance(patch_changes["definition"], str) or not patch_changes["definition"].strip()):
                    err(i, "update_definition definition must be a non-empty string")
                if "strength_justification" in patch_changes and (not isinstance(patch_changes["strength_justification"], str) or not patch_changes["strength_justification"].strip()):
                    err(i, "update_definition strength_justification must be a non-empty string")
                # used_by validation
                if "used_by" in patch_changes:
                    used_by = patch_changes["used_by"]
                    if not isinstance(used_by, list) or len(used_by) == 0:
                        err(i, "update_definition used_by must be a non-empty list")
                    elif isinstance(used_by, list):
                        ae_ids = assumption_ids | evidence_ids
                        for ref in used_by:
                            if ref not in ae_ids:
                                err(i, f"update_definition used_by references non-existent A#/E# '{ref}'")

        else:
            err(i, f"Unknown operation '{op}'")

    # --- Cascade removal: if a projected add_* patch failed validation,
    # transitively flag other patches that reference its ID ---
    # Loop until no new failures are discovered (handles multi-hop chains
    # like C3 fail → A5/E5 cascade → D6 cascade).
    while True:
        failed_ids = set()
        for idx in errors:
            patch = patches[idx]
            op = patch.get("op", "")
            if op.startswith("add_"):
                item = patch.get("item", {})
                pid = item.get("id", "")
                if pid:
                    failed_ids.add(pid)

        if not failed_ids:
            break

        new_failures = False
        for i, patch in enumerate(patches):
            if i in errors:
                continue  # Already failed
            op = patch.get("op", "")
            item = patch.get("item", {})
            refs_to_check = set()
            if op == "add_claim":
                refs_to_check.update(item.get("depends_on", []))
            elif op in ("add_assumption", "add_evidence"):
                refs_to_check.update(item.get("supports_claims", []))
                refs_to_check.update(item.get("supported_by_definitions", []))
            elif op == "add_definition":
                refs_to_check.update(item.get("used_by", []))
            # Also check update_* ops that target a failed add_* ID
            elif op.startswith("update_"):
                target_id = patch.get("target_id", "")
                if target_id in failed_ids:
                    refs_to_check.add(target_id)
            if refs_to_check & failed_ids:
                err(i, f"depends on failed patch ID(s): {sorted(refs_to_check & failed_ids)}")
                new_failures = True

        if not new_failures:
            break

    return errors
