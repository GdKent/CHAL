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

FALLBACK_MIN_STR = 0.0
FALLBACK_MAX_STR = 1.0


def parse_model_output_to_belief(output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """
    Extract a JSON code block (```json ... ```) and an accompanying Markdown block.
    Returns: (belief_dict_or_None, markdown_or_None, errors)
    """
    errors: List[str] = []

    # Extract JSON — try fenced block first, then fall back to raw JSON
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", output, flags=re.DOTALL)
    belief_obj = None
    if json_match:
        raw_json_str = json_match.group(1)
    else:
        # No fenced block — try to extract a top-level JSON object directly
        raw_match = re.search(r"(\{.*\})", output, flags=re.DOTALL)
        raw_json_str = raw_match.group(1) if raw_match else None

    if raw_json_str:
        try:
            # Normalize IDs: some models use "A#1" instead of "A1"
            raw_json_str = re.sub(r'"([ACDEUX])#(\d+)"', r'"\1\2"', raw_json_str)
            belief_obj = json.loads(raw_json_str)
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
    if not isinstance(th, dict):
        th = {"stance": str(th)}
    md.append("# Thesis")
    md.append(f"- Stance: {th.get('stance','')}")
    bullets = th.get("summary_bullets") or []
    if bullets:
        md.append("- Top bullets:")
        for b in bullets:
            md.append(f"  - {b}")
    md.append(f"- Strength: {th.get('strength', '')}")
    if th.get("strength_reasoning"):
        md.append(f"- Reasoning: {th['strength_reasoning']}")

    def list_block(title: str, key: str, formatter):
        items = belief.get(key) or []
        if not items: return
        md.append(f"\n# {title}")
        for item in items:
            md.append(formatter(item))

    def definition_fmt(d):
        lines = [f"- [{d.get('id','')}] **{d.get('term','')}**: {d.get('definition','')}"]
        lines.append(f"  - Strength: {d.get('strength', '')} ({d.get('strength_justification', '')})")
        lines.append(f"  - Used by: {', '.join(d.get('used_by', []))}")
        lines.append(f"  - Status: {d.get('status', 'active')}")
        if d.get("consecutive_defenses", 0) > 0:
            lines.append(f"  - Defenses: {d['consecutive_defenses']} consecutive (original strength: {d.get('original_strength', '?')})")
        return "\n".join(lines)
    list_block("Definitions", "definitions", definition_fmt)

    def assumption_fmt(a):
        lines = [f"- [{a.get('id','')}] ({a.get('type','')}) {a.get('statement','')}"]
        lines.append(f"  - Supports: {', '.join(a.get('supports_claims') or [])}")
        lines.append(f"  - Strength: {a.get('strength', '')} ({a.get('strength_justification', '')})")
        lines.append(f"  - Supported by definitions: {', '.join(a.get('supported_by_definitions', []))}")
        lines.append(f"  - Status: {a.get('status', 'active')}")
        if a.get("consecutive_defenses", 0) > 0:
            lines.append(f"  - Defenses: {a['consecutive_defenses']} consecutive (original strength: {a.get('original_strength', '?')})")
        return "\n".join(lines)
    list_block("Assumptions", "assumptions", assumption_fmt)

    def claim_fmt(c):
        parts = [f"- [{c.get('id','')}] ({c.get('type','')}) {c.get('statement','')}"]
        if c.get("depends_on"): parts.append(f"  - Depends on: {', '.join(c['depends_on'])}")
        if c.get("inference_chain"):
            parts.append("  - Inference chain:")
            for step in c["inference_chain"]:
                if isinstance(step, dict) and "role" in step:
                    # New structured format: role-based steps
                    role = step.get("role", "")
                    text = step.get("text", "")
                    if role == "premise":
                        ref = step.get("reference", "")
                        parts.append(f"    - Premise ({ref}): {text}")
                    elif role == "inference":
                        inf_type = step.get("inference_type", "")
                        parts.append(f"    - Inference ({inf_type}): {text}")
                    elif role == "conclusion":
                        parts.append(f"    - Conclusion: {text}")
                    else:
                        parts.append(f"    - {role}: {text}")
                elif isinstance(step, dict):
                    # Legacy dict format: {step, justification}
                    parts.append(f"    - [legacy] {step.get('step','')} | {step.get('justification','')}")
                else:
                    # Legacy string format
                    parts.append(f"    - [legacy] {step}")
        parts.append(f"  - Strength: {c.get('strength','')} ({c.get('strength_justification','')})")
        parts.append(f"  - Status: {c.get('status','')}")
        if c.get("consecutive_defenses", 0) > 0:
            parts.append(f"  - Defenses: {c['consecutive_defenses']} consecutive (original strength: {c.get('original_strength', '?')})")
        # Inline predictions under the claim
        preds = c.get("predictions") or []
        if preds:
            parts.append("  - Predictions:")
            for pred in preds:
                parts.append(f"    - {pred.get('statement','')}")
                parts.append(f"      Test: {pred.get('test','')}")
                parts.append(f"      Criterion: {pred.get('decision_criterion','')}")
                falsifiers = pred.get("potential_falsifiers") or []
                if falsifiers:
                    parts.append(f"      Falsifiers: {', '.join(falsifiers)}")
        return "\n".join(parts)
    def ev_fmt(e):
        src = e.get("source", "")
        if isinstance(src, dict):
            src_str = ", ".join([f"{k}: {v}" for k, v in src.items()])
        else:
            src_str = str(src)
        lines = [f"- [{e.get('id','')}] ({e.get('type','')}) {e.get('summary','')}"]
        lines.append(f"  - Strength: {e.get('strength', '')} ({e.get('strength_justification', '')})")
        lines.append(f"  - Source: {src_str}")
        lines.append(f"  - Supports: {', '.join(e.get('supports_claims') or [])}")
        lines.append(f"  - Supported by definitions: {', '.join(e.get('supported_by_definitions', []))}")
        lines.append(f"  - Status: {e.get('status', 'active')}")
        if e.get("consecutive_defenses", 0) > 0:
            lines.append(f"  - Defenses: {e['consecutive_defenses']} consecutive (original strength: {e.get('original_strength', '?')})")
        return "\n".join(lines)
    list_block("Evidence", "evidence", ev_fmt)

    list_block("Claims", "claims", claim_fmt)

    def u_fmt(u):
        status = u.get("status", "active")
        lines = [f"- [{u.get('id','')}] {u.get('question','')}"]
        lines.append(f"  - Targets: {', '.join(u.get('targets', []))}")
        lines.append(f"  - Status: {status}")
        lines.append(f"  - Importance: {u.get('importance','')}")
        if status == "resolved" and u.get("resolution_note"):
            lines.append(f"  - Resolution: {u['resolution_note']}")
        return "\n".join(lines)
    list_block("Uncertainties", "uncertainties", u_fmt)

    def x_fmt(x):
        lines = [f"- [{x.get('id','')}] Statement: {x.get('statement','')}"]
        lines.append(f"  - Attack type: {x.get('attack_type','')}")
        lines.append(f"  - Attack strategy: {x.get('attack_strategy','')}")
        lines.append(f"  - Targets: {', '.join(x.get('targets', []))}")
        if x.get("my_response"):
            lines.append(f"  - My response: {x['my_response']}")
        lines.append(f"  - Sufficiency: {x.get('response_sufficiency','')}")
        return "\n".join(lines)
    list_block("Counterpositions", "counterpositions", x_fmt)

    if belief.get("changelog"):
        md.append("\n# Changelog")
        for ch in belief["changelog"]:
            md.append(f"- v{ch.get('version')}: " + "; ".join(ch.get("changes") or []))

    return "\n".join(md).strip()


def project_for_embedding(belief: Dict[str, Any]) -> str:
    """
    Create a concise, deterministic text summary for embedding.
    This avoids embedding huge JSON or verbose prose and keeps semantically stable signals.
    """
    th = belief.get("thesis", {})
    lines = [f"Thesis: {th.get('stance','')}",
             "Bullets: " + " | ".join(th.get("summary_bullets") or []),
             f"Strength: {th.get('strength','')}"]

    # Top 3 definitions by strength
    defs = sorted(
        [d for d in belief.get("definitions", []) if d.get("status") != "retracted"],
        key=lambda d: d.get("strength", 0), reverse=True
    )[:3]
    if defs:
        def_parts = [f"{d['term']}={d['definition']}" for d in defs]
        lines.append("Key definitions: " + "; ".join(def_parts))

    # Top 3 claims by strength if available
    claims = sorted((belief.get("claims") or []), key=lambda c: -(c.get("strength", 0.0)))[:3]
    for c in claims:
        lines.append(f"Claim {c.get('id')}: {c.get('statement')} (type={c.get('type')}, depends_on={','.join(c.get('depends_on') or [])})")

    # One uncertainty (first one)
    if belief.get("uncertainties"):
        u = belief["uncertainties"][0]
        lines.append(
            f"Uncertainty {u.get('id')}: {u.get('question')} "
            f"(targets={','.join(u.get('targets', []))}, status={u.get('status', 'active')})"
        )

    # Top 2 counterpositions
    counterpositions = (belief.get("counterpositions") or [])[:2]
    for x in counterpositions:
        lines.append(
            f"Counterposition {x.get('id')}: {x.get('statement')} "
            f"(attack={x.get('attack_type')}, sufficiency={x.get('response_sufficiency')})"
        )

    return "\n".join(lines)
