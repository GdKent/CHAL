"""
scoring.py

Computes simple quality scores from CBS-v1 beliefs to guide optimization.
"""

from __future__ import annotations
from typing import Dict, Any

def consistency_score(belief: Dict[str, Any]) -> float:
    """
    Very simple heuristic: claims with at least one backing evidence and no self-cycles.
    """
    claims = belief.get("claims") or []
    if not claims: return 0.0
    ok = 0
    for c in claims:
        if c.get("backing_evidence_ids"):
            ok += 1
    return ok / max(1, len(claims))

def falsifiability_score(belief: Dict[str, Any]) -> float:
    """
    Fraction of claims that have at least one linked prediction.
    (If you want strict linkage, you can add 'linked_claims' in predictions and check it.)
    """
    claims = belief.get("claims") or []
    preds = belief.get("predictions") or []
    if not claims: return 0.0
    linked = set()
    for p in preds:
        for cid in p.get("linked_claims") or []:
            linked.add(cid)
    return len(linked) / max(1, len(claims))