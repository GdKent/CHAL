"""
io.py

Utilities to:
- Parse model output into (json_belief, markdown_view)
- Render a belief into Markdown (human view)
- Project a belief to a compact text string for embeddings
- Project a belief to per-component text lists for rich embeddings

Acronyms:
- CBS = CHAL Belief Schema
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chal.beliefs.schema import validate_belief

FALLBACK_MIN_STR = 0.0
FALLBACK_MAX_STR = 1.0


def load_belief_from_file(path: str | Path) -> dict[str, Any]:
    """Load and validate a CBS belief object from a JSON file.

    Args:
        path: Path to a ``.json`` file containing a complete CBS belief object.

    Returns:
        The validated belief dict, ready for use.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or CBS schema validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Belief file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            belief_obj = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in belief file {path}: {exc}") from exc

    if not isinstance(belief_obj, dict):
        raise ValueError(
            f"Belief file must contain a JSON object, got {type(belief_obj).__name__}"
        )

    errors = validate_belief(belief_obj)
    if errors:
        error_list = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"Belief file {path} failed CBS schema validation:\n{error_list}"
        )

    return belief_obj


def parse_model_output_to_belief(output: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """
    Extract a JSON code block (```json ... ```) and an accompanying Markdown block.
    Returns: (belief_dict_or_None, markdown_or_None, errors)
    """
    errors: list[str] = []

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


def belief_to_markdown(belief: dict[str, Any]) -> str:
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
        if not items:
            return
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
        if c.get("depends_on"):
            parts.append(f"  - Depends on: {', '.join(c['depends_on'])}")
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


def project_for_embedding(belief: dict[str, Any]) -> str:
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


def project_for_component_embedding(belief: dict[str, Any]) -> dict[str, Any]:
    """
    Extract per-component text lists and scalar features from a CBS belief
    for rich, component-wise embedding.

    Returns a dict with:
      - "definitions": list of {"text": str, "strength": float}
      - "assumptions": list of {"text": str, "strength": float}
      - "evidence": list of {"text": str, "strength": float}
      - "claims": list of {"text": str, "strength": float}
      - "thesis_text": str (stance + bullets concatenated)
      - "uncertainties": list of str (questions for open uncertainties only)
      - "counterpositions": {"partial": [str], "sufficient": [str], "unaddressed": [str]}
      - "scalars": dict with 11 scalar features
    """
    # --- Strength-bearing components: filter out retracted nodes ---
    active_defs = [
        d for d in (belief.get("definitions") or [])
        if d.get("status") != "retracted"
    ]
    active_assumptions = [
        a for a in (belief.get("assumptions") or [])
        if a.get("status") != "retracted"
    ]
    active_evidence = [
        e for e in (belief.get("evidence") or [])
        if e.get("status") != "retracted"
    ]
    active_claims = [
        c for c in (belief.get("claims") or [])
        if c.get("status") != "retracted"
    ]

    # --- Text extraction per component ---
    def_items = [
        {"text": f"{d.get('term', '')}: {d.get('definition', '')}", "strength": d.get("strength", 0.0)}
        for d in active_defs
    ]
    assumption_items = [
        {"text": a.get("statement", ""), "strength": a.get("strength", 0.0)}
        for a in active_assumptions
    ]
    evidence_items = [
        {"text": e.get("summary", ""), "strength": e.get("strength", 0.0)}
        for e in active_evidence
    ]
    claim_items = [
        {"text": c.get("statement", ""), "strength": c.get("strength", 0.0)}
        for c in active_claims
    ]

    # --- Thesis: concatenate stance + bullet points ---
    th = belief.get("thesis") or {}
    stance = th.get("stance", "")
    bullets = th.get("summary_bullets") or []
    thesis_text = f"{stance}. {'. '.join(bullets)}" if bullets else stance

    # --- Uncertainties: only open (not resolved) ---
    open_uncertainties = [
        u.get("question", "")
        for u in (belief.get("uncertainties") or [])
        if u.get("status") != "resolved"
    ]

    # --- Counterpositions: group by response_sufficiency ---
    counter_by_sufficiency: dict[str, list[str]] = {
        "partial": [],
        "sufficient": [],
        "unaddressed": [],
        "moot": [],
    }
    for x in (belief.get("counterpositions") or []):
        suff = x.get("response_sufficiency", "unaddressed")
        if suff in counter_by_sufficiency:
            counter_by_sufficiency[suff].append(x.get("statement", ""))

    # --- Scalar features ---
    def _avg_strength(items: list) -> float:
        if not items:
            return 0.0
        return sum(i.get("strength", 0.0) for i in items) / len(items)  # type: ignore[no-any-return]

    all_counterpositions = belief.get("counterpositions") or []
    all_uncertainties = belief.get("uncertainties") or []

    scalars = {
        "n_definitions": len(active_defs),
        "n_assumptions": len(active_assumptions),
        "n_evidence": len(active_evidence),
        "n_claims": len(active_claims),
        "avg_strength_definitions": _avg_strength(active_defs),
        "avg_strength_assumptions": _avg_strength(active_assumptions),
        "avg_strength_evidence": _avg_strength(active_evidence),
        "avg_strength_claims": _avg_strength(active_claims),
        "n_counterpositions": len(all_counterpositions),
        "n_uncertainties": len(all_uncertainties),
        "thesis_strength": th.get("strength", 0.0),
    }

    return {
        "definitions": def_items,
        "assumptions": assumption_items,
        "evidence": evidence_items,
        "claims": claim_items,
        "thesis_text": thesis_text,
        "uncertainties": open_uncertainties,
        "counterpositions": counter_by_sufficiency,
        "scalars": scalars,
    }
