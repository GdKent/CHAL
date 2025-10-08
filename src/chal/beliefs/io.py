"""
io.py

Utilities to:
- Parse model output into (json_belief, markdown_view)
- Render a belief into Markdown (human view)
- Project a belief to a compact text string for embeddings

Acronyms:
- CBS = CHAL Belief Schema
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import json
import re
from chal.beliefs.schema import validate_belief, SCHEMA_VERSION

FALLBACK_MIN_CONF = 0.0
FALLBACK_MAX_CONF = 1.0


def parse_model_output_to_belief(output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """
    Extract a JSON code block (```json ... ```) and an accompanying Markdown block.
    Returns: (belief_dict_or_None, markdown_or_None, errors)
    """
    errors: List[str] = []

    # Extract JSON fenced block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", output, flags=re.DOTALL)
    belief_obj = None
    if json_match:
        try:
            belief_obj = json.loads(json_match.group(1))
        except Exception as e:
            errors.append(f"JSON parse error: {e}")

    # Extract Markdown view (look for a second fenced block or the remainder)
    md_match = re.search(r"```(?:json).*?```(.*)", output, flags=re.DOTALL)
    markdown_view = (md_match.group(1).strip() if md_match else None) or None

    # Validate if we got JSON
    if belief_obj is not None:
        v_errors = validate_belief(belief_obj)
        errors.extend(v_errors)

    return belief_obj, markdown_view, errors


def belief_to_markdown(belief: Dict[str, Any]) -> str:
    """
    Generate a stable Markdown view using the same IDs to ensure round-trip readability.
    """
    md = []
    th = belief.get("thesis", {})
    md.append("# Thesis")
    md.append(f"- Stance: {th.get('stance','')}")
    bullets = th.get("summary_bullets") or []
    if bullets:
        md.append("- Top bullets:")
        for b in bullets:
            md.append(f"  - {b}")
    md.append(f"- Confidence: {th.get('confidence', '')}")

    md.append("\n# Scope & Definitions")
    meta = belief.get("metadata", {})
    if meta.get("scope_conditions"): md.append(f"- Scope conditions: {meta['scope_conditions']}")
    defs = meta.get("definitions") or []
    if defs:
        md.append("- Definitions:")
        for d in defs:
            md.append(f"  - {d.get('term')}: {d.get('definition')}")

    def list_block(title: str, key: str, formatter):
        items = belief.get(key) or []
        if not items: return
        md.append(f"\n# {title}")
        for item in items:
            md.append(formatter(item))

    list_block("Assumptions", "assumptions", lambda a: f"- [{a.get('id','')}] {a.get('type','')}: {a.get('statement','')}")
    def claim_fmt(c):
        parts = [f"- [{c.get('id','')}] {c.get('type','')}: {c.get('statement','')}"]
        if c.get("depends_on"): parts.append(f"  - Depends on: {', '.join(c['depends_on'])}")
        if c.get("warrants"): parts.append(f"  - Warrants: {', '.join(c['warrants'])}")
        if c.get("backing_evidence_ids"): parts.append(f"  - Evidence: {', '.join(c['backing_evidence_ids'])}")
        if c.get("qualifiers"): parts.append(f"  - Qualifiers: {', '.join(c['qualifiers'])}")
        if c.get("known_rebuttals"):
            parts.append("  - Known rebuttals:")
            for r in c["known_rebuttals"]:
                parts.append(f"    - [{r.get('id','')}] steelman: {r.get('steelman','')} | status: {r.get('status','')}")
        parts.append(f"  - Confidence: {c.get('confidence','')} | Status: {c.get('status','')}")
        return "\n".join(parts)
    list_block("Claims (with dependencies, evidence, rebuttals)", "claims", claim_fmt)
    def ev_fmt(e):
        src = e.get("source") or {}
        src_str = ", ".join([f"{k}: {v}" for k,v in src.items()])
        return f"- [{e.get('id','')}] {e.get('type','')}: {e.get('summary','')} (source: {src_str}) → supports: {', '.join(e.get('relevance_to_claims') or [])}"
    list_block("Evidence", "evidence", ev_fmt)
    list_block("Predictions (falsifiable)", "predictions",
               lambda p: f"- [{p.get('id','')}] {p.get('statement','')} | timeframe: {p.get('timeframe','')} | test: {p.get('test','')} | falsifiers: {', '.join(p.get('potential_falsifiers') or [])} | likelihood: {p.get('expected_likelihood','')} | importance: {p.get('importance','')}")
    list_block("Normative Implications (prescriptive)", "normative_implications",
               lambda n: f"- [{n.get('id','')}] {n.get('statement','')} | linked claims: {', '.join(n.get('linked_claims') or [])} | strength: {n.get('strength','')}")
    list_block("Uncertainties", "uncertainties",
               lambda u: f"- [{u.get('id','')}] {u.get('question','')} | cruciality: {u.get('cruciality','')} | VOI: {u.get('voi_hint','')}")
    def x_fmt(x):
        lines = [f"- [{x.get('id','')}] Thesis: {x.get('thesis','')}"]
        if x.get("steelman"): lines.append(f"  - Steelman: {x.get('steelman')}")
        if x.get("strongest_arguments"): lines.append(f"  - Strongest arguments: {', '.join(x['strongest_arguments'])}")
        if x.get("our_responses"):
            lines.append("  - Our responses:")
            for r in x["our_responses"]:
                lines.append(f"    - To {r.get('to','')}: {r.get('response','')}")
        if x.get("comparative_assessment"): lines.append(f"  - When theirs might win: {x['comparative_assessment']}")
        return "\n".join(lines)
    list_block("Counterposition Map", "counterposition_map", x_fmt)

    if belief.get("update_policy"):
        up = belief["update_policy"]
        md.append("\n# Update Policy")
        if up.get("revision_triggers"): md.append(f"- Triggers: {', '.join(up['revision_triggers'])}")
        if up.get("confidence_update_rule"): md.append(f"- Confidence update rule: {up['confidence_update_rule']}")
        if up.get("retirement_criteria"): md.append(f"- Retirement criteria: {', '.join(up['retirement_criteria'])}")

    if belief.get("changelog"):
        md.append("\n# Changelog")
        for ch in belief["changelog"]:
            md.append(f"- v{ch.get('version')} ({ch.get('timestamp')}): " + "; ".join(ch.get("changes") or []))

    return "\n".join(md).strip()


def project_for_embedding(belief: Dict[str, Any]) -> str:
    """
    Create a concise, deterministic text summary for embedding.
    This avoids embedding huge JSON or verbose prose and keeps semantically stable signals.
    """
    th = belief.get("thesis", {})
    lines = [f"Thesis: {th.get('stance','')}",
             "Bullets: " + " | ".join(th.get("summary_bullets") or []),
             f"Confidence: {th.get('confidence','')}"]

    # Top 3 claims by confidence if available
    claims = sorted((belief.get("claims") or []), key=lambda c: -(c.get("confidence", 0.0)))[:3]
    for c in claims:
        lines.append(f"Claim {c.get('id')}: {c.get('statement')} (type={c.get('type')}, depends_on={','.join(c.get('depends_on') or [])})")

    # One falsifiable prediction (highest importance if present)
    preds = belief.get("predictions") or []
    if preds:
        best = sorted(preds, key=lambda p: (p.get("importance","medium") == "high", p.get("expected_likelihood",0.0)), reverse=True)[0]
        lines.append(f"Prediction {best.get('id')}: {best.get('statement')} (test={best.get('test')}, timeframe={best.get('timeframe')})")

    # One uncertainty
    if belief.get("uncertainties"):
        u = belief["uncertainties"][0]
        lines.append(f"Uncertainty {u.get('id')}: {u.get('question')} (cruciality={u.get('cruciality')})")

    return "\n".join(lines)