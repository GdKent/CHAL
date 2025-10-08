"""
patch.py

Applies patch operations to a CBS-v1 belief.
Operations:
- add_*       (e.g., add_claim)
- update_*    (e.g., update_claim)
- retire_*    (e.g., retire_claim)  -> sets status='retracted' or removes, configurable

All ops return the modified belief in-place for convenience.

Acronym:
- ID = Identifier
"""

from __future__ import annotations
from typing import Dict, Any, List

def _find_by_id(items: List[Dict[str, Any]], _id: str) -> int:
    for i, it in enumerate(items):
        if it.get("id") == _id:
            return i
    return -1

def apply_patch(belief: Dict[str, Any], op: Dict[str, Any]) -> None:
    """
    Example:
    {"op": "update_claim", "target_id": "C1", "changes": {"confidence": 0.55}}
    {"op": "add_evidence", "item": {...}}
    {"op": "retire_claim", "target_id": "C3", "mode": "retract"}  # or mode='remove'
    """
    op_name: str = op.get("op","")
    if not op_name:
        return

    def ensure_list(key: str) -> List[Dict[str, Any]]:
        if key not in belief or belief[key] is None:
            belief[key] = []
        return belief[key]

    mapping = {
        "claim": "claims",
        "assumption": "assumptions",
        "evidence": "evidence",
        "prediction": "predictions",
        "normative": "normative_implications",
        "uncertainty": "uncertainties",
        "counterposition": "counterposition_map"
    }

    # Determine target collection from op name
    target_kind = None
    for k in mapping:
        if k in op_name:
            target_kind = mapping[k]
            break
    if not target_kind:
        return

    coll = ensure_list(target_kind)

    if op_name.startswith("add_"):
        item = op.get("item")
        if item:
            coll.append(item)
        return

    target_id = op.get("target_id")
    if not target_id:
        return

    idx = _find_by_id(coll, target_id)
    if idx < 0:
        return

    if op_name.startswith("update_"):
        changes = op.get("changes") or {}
        coll[idx].update(changes)
        return

    if op_name.startswith("retire_"):
        mode = op.get("mode", "retract")
        if mode == "remove":
            coll.pop(idx)
        else:
            coll[idx]["status"] = "retracted"
        return