"""
validators.py

Per-stage output validation functions for the structured-output retry system.

Each validator accepts the raw LLM response text and returns a
:class:`~chal.utilities.retry.ValidationResult` indicating whether the
response is usable and, if not, a list of human-readable error strings
that will be fed back to the model on retry.

Validators are intentionally *structural* — they check that the output
can be parsed and contains the required fields/types.  Semantic checks
(e.g. graph-level cycle detection) are left to downstream code.
"""

from __future__ import annotations

import json
import re

from chal.utilities.retry import ValidationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_fenced_json(text: str) -> str | None:
    """Return the first fenced ```json ... ``` block body, or *None*."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    return m.group(1) if m else None


def _extract_any_json_object(text: str) -> str | None:
    """Return the first top-level ``{ ... }`` substring, or *None*."""
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1) if m else None


def _try_parse_json(raw: str) -> tuple[dict | None, str | None]:
    """Attempt ``json.loads``; return ``(obj, None)`` or ``(None, error)``."""
    # Normalize IDs: some models use "A#1" instead of "A1"
    raw = re.sub(r'"([ACDEUX])#(\d+)"', r'"\1\2"', raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
        return None, "Parsed JSON is not an object (expected { ... })"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


# ---------------------------------------------------------------------------
# Stage 1 — Initial Belief (CBS) Validator
# ---------------------------------------------------------------------------

_CBS_REQUIRED_KEYS = (
    "schema_version", "belief_id", "version", "metadata", "thesis",
    "definitions", "assumptions", "claims", "evidence",
    "counterpositions", "uncertainties",
)

STAGE1_REMEDIATION_HINTS = (
    "Output ONLY a single fenced ```json ... ``` code block containing "
    "your complete CBS belief object.\n"
    "The JSON must be valid (double quotes, no trailing commas, no comments).\n"
    "All required top-level keys must be present: schema_version, belief_id, "
    "version, metadata, thesis, definitions, assumptions, claims, evidence, "
    "counterpositions, uncertainties.\n"
    "Every claim must include: type, statement, depends_on, strength, "
    "strength_justification, status, inference_chain, and predictions.\n"
    "Every assumption must include: type, statement, supports_claims, "
    "strength, strength_justification.\n"
    "Every evidence item must include: type, summary, source, "
    "supports_claims, strength, strength_justification.\n"
    "Every definition must include: term, definition, strength, "
    "strength_justification, used_by."
)


def validate_stage1_output(raw_response: str) -> ValidationResult:
    """Validate that a Stage 1 response contains a well-formed CBS belief.

    Checks (in order):
    1. A JSON block is present (fenced or raw object).
    2. The JSON is valid.
    3. All required top-level CBS keys are present.
    4. ``thesis`` has required sub-fields.
    5. ``definitions``, ``assumptions``, ``claims`` are non-empty.
    6. Node ID format (``^[ACDEUX]\\d+$``).
    7. Claim required fields.
    8. Assumption required fields.
    9. Evidence required fields.
    10. Definition required fields.
    11. Cross-reference validity via ``validate_belief()``.

    Graph-level checks (cycles, orphans) are **not** included — those are
    handled by the existing ``BeliefGraph.validate_links()`` retry loop.
    """
    errors: list[str] = []

    # 1. JSON extraction
    json_str = _extract_fenced_json(raw_response)
    if json_str is None:
        json_str = _extract_any_json_object(raw_response)
    if json_str is None:
        errors.append(
            "No JSON block found. You must output exactly one fenced "
            "```json ... ``` block containing your CBS belief object."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # 2. JSON parsing
    belief, parse_err = _try_parse_json(json_str)
    if belief is None:
        errors.append(f"JSON block is not valid JSON: {parse_err}")
        return ValidationResult(is_valid=False, errors=errors)

    # 3. Required top-level keys
    for key in _CBS_REQUIRED_KEYS:
        if key not in belief:
            errors.append(f"Missing required top-level key '{key}'.")

    # Early exit if too many top-level keys are missing — further checks
    # would just produce noise.
    if len(errors) >= 4:
        return ValidationResult(is_valid=False, errors=errors, parsed_data=belief)

    # 4. Thesis sub-fields
    thesis = belief.get("thesis")
    if isinstance(thesis, dict):
        if not thesis.get("stance"):
            errors.append("thesis.stance is missing or empty.")
        bullets = thesis.get("summary_bullets")
        if not isinstance(bullets, list) or len(bullets) == 0:
            errors.append("thesis.summary_bullets must be a non-empty list.")
        t_str = thesis.get("strength")
        if t_str is None:
            errors.append("thesis.strength is missing.")
        elif not isinstance(t_str, (int, float)) or not (0.0 <= t_str <= 1.0):
            errors.append(f"thesis.strength {t_str} out of range [0.0, 1.0].")
        if not thesis.get("strength_reasoning"):
            errors.append("thesis.strength_reasoning is missing or empty.")
    elif thesis is not None:
        errors.append("'thesis' must be an object.")

    # 5. Collection minimums
    for collection, label in (
        ("definitions", "definition"),
        ("assumptions", "assumption"),
        ("claims", "claim"),
    ):
        items = belief.get(collection)
        if not isinstance(items, list) or len(items) == 0:
            errors.append(
                f"'{collection}' array is empty or missing. "
                f"You must provide at least one {label}."
            )

    # 6–10. Per-node field checks
    _id_re = re.compile(r"^[ACDEUX]\d+$")

    # Helper: check a node has an ID matching the expected format
    def _check_id(node: dict, collection: str) -> str | None:
        nid = node.get("id")
        if not nid or not isinstance(nid, str):
            errors.append(f"A node in '{collection}' is missing 'id'.")
            return None
        if not _id_re.match(nid):
            errors.append(
                f"Invalid ID format '{nid}' in '{collection}'. "
                f"Expected format like 'D1', 'A2', 'C3', etc."
            )
        return nid  # type: ignore[no-any-return]

    # 7. Claims
    for node in (belief.get("claims") or []):
        if not isinstance(node, dict):
            continue
        cid = _check_id(node, "claims") or "?"
        for field in ("type", "statement", "status"):
            if not node.get(field):
                errors.append(f"Claim '{cid}' is missing required field '{field}'.")
        if not node.get("depends_on") or not isinstance(node.get("depends_on"), list):
            errors.append(f"Claim '{cid}' must have a non-empty 'depends_on' list.")
        c_str = node.get("strength")
        if c_str is None:
            errors.append(f"Claim '{cid}' is missing 'strength'.")
        elif not isinstance(c_str, (int, float)) or not (0.0 <= c_str <= 1.0):
            errors.append(f"Claim '{cid}' strength {c_str} out of range [0.0, 1.0].")
        if "strength_justification" not in node:
            errors.append(f"Claim '{cid}' is missing 'strength_justification'.")
        ic = node.get("inference_chain")
        if not isinstance(ic, list) or len(ic) == 0:
            errors.append(f"Claim '{cid}' must have a non-empty 'inference_chain'.")
        preds = node.get("predictions")
        if not isinstance(preds, list) or len(preds) == 0:
            errors.append(f"Claim '{cid}' must have a non-empty 'predictions' list.")

    # 8. Assumptions
    for node in (belief.get("assumptions") or []):
        if not isinstance(node, dict):
            continue
        aid = _check_id(node, "assumptions") or "?"
        for field in ("type", "statement"):
            if not node.get(field):
                errors.append(f"Assumption '{aid}' is missing required field '{field}'.")
        if not node.get("supports_claims") or not isinstance(node.get("supports_claims"), list):
            errors.append(f"Assumption '{aid}' must have a non-empty 'supports_claims' list.")
        a_str = node.get("strength")
        if a_str is None:
            errors.append(f"Assumption '{aid}' is missing 'strength'.")
        elif not isinstance(a_str, (int, float)) or not (0.0 <= a_str <= 1.0):
            errors.append(f"Assumption '{aid}' strength {a_str} out of range [0.0, 1.0].")
        if "strength_justification" not in node:
            errors.append(f"Assumption '{aid}' is missing 'strength_justification'.")

    # 9. Evidence
    for node in (belief.get("evidence") or []):
        if not isinstance(node, dict):
            continue
        eid = _check_id(node, "evidence") or "?"
        for field in ("type", "summary", "source"):
            if not node.get(field):
                errors.append(f"Evidence '{eid}' is missing required field '{field}'.")
        if not node.get("supports_claims") or not isinstance(node.get("supports_claims"), list):
            errors.append(f"Evidence '{eid}' must have a non-empty 'supports_claims' list.")
        e_str = node.get("strength")
        if e_str is None:
            errors.append(f"Evidence '{eid}' is missing 'strength'.")
        elif not isinstance(e_str, (int, float)) or not (0.0 <= e_str <= 1.0):
            errors.append(f"Evidence '{eid}' strength {e_str} out of range [0.0, 1.0].")
        if "strength_justification" not in node:
            errors.append(f"Evidence '{eid}' is missing 'strength_justification'.")

    # 10. Definitions
    for node in (belief.get("definitions") or []):
        if not isinstance(node, dict):
            continue
        did = _check_id(node, "definitions") or "?"
        if not node.get("term"):
            errors.append(f"Definition '{did}' is missing required field 'term'.")
        if not node.get("definition"):
            errors.append(f"Definition '{did}' is missing required field 'definition'.")
        d_str = node.get("strength")
        if d_str is None:
            errors.append(f"Definition '{did}' is missing 'strength'.")
        elif not isinstance(d_str, (int, float)) or not (0.0 <= d_str <= 1.0):
            errors.append(f"Definition '{did}' strength {d_str} out of range [0.0, 1.0].")
        if "strength_justification" not in node:
            errors.append(f"Definition '{did}' is missing 'strength_justification'.")
        if not node.get("used_by") or not isinstance(node.get("used_by"), list):
            errors.append(f"Definition '{did}' must have a non-empty 'used_by' list.")

    # 11. Cross-reference validity via validate_belief()
    #     (only if we have a structurally plausible object — no point
    #     running schema validation on something missing half its keys)
    if len(errors) == 0:
        from chal.beliefs.schema import validate_belief
        schema_errors = validate_belief(belief)
        errors.extend(schema_errors)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=belief,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Cross-Examination Validator
# ---------------------------------------------------------------------------

STAGE2_REMEDIATION_HINTS = (
    "Output exactly ONE fenced ```json ... ``` block containing:\n"
    '{"questions": [{"qid": "Q1", "text": "...", "target_ids": ["C1"], '
    '"attack_type": "undermining|rebutting|undercutting", '
    '"attack_strategy": "..."}]}\n\n'
    "Each question MUST have all 5 fields: qid, text, target_ids, "
    "attack_type, attack_strategy.\n"
    "target_ids must be a list of 1-2 CBS node IDs "
    '(e.g., ["C1"], ["A2", "E3"]).\n'
    "attack_strategy must be valid for the chosen attack_type."
)


def validate_stage2_output(raw_response: str) -> ValidationResult:
    """Validate that a Stage 2 response contains well-formed cross-exam questions.

    Checks (in order):
    1. A fenced JSON block is present.
    2. The JSON is valid.
    3. ``questions`` key is present with a non-empty list.
    4. Per-question field validation via ``validate_stage2_questions()``.
    """
    errors: list[str] = []

    # 1. JSON extraction (fenced block preferred, raw fallback)
    json_str = _extract_fenced_json(raw_response)
    if json_str is None:
        json_str = _extract_any_json_object(raw_response)
    if json_str is None:
        errors.append(
            "No fenced ```json ... ``` block found. You must output exactly "
            "one fenced JSON block containing your questions."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # 2. JSON parsing
    obj, parse_err = _try_parse_json(json_str)
    if obj is None:
        errors.append(f"JSON block is not valid JSON: {parse_err}")
        return ValidationResult(is_valid=False, errors=errors)

    # 3. questions key
    questions = obj.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        errors.append(
            "JSON block is missing 'questions' key or questions list is empty."
        )
        return ValidationResult(is_valid=False, errors=errors, parsed_data=obj)

    # 4. Per-question field validation (delegate to existing utility)
    from chal.utilities.utils import validate_stage2_questions

    _is_valid, q_errors = validate_stage2_questions(questions)
    errors.extend(q_errors)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=obj,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Rebuttal Validator
# ---------------------------------------------------------------------------

_VALID_REBUTTAL_ACTIONS = {"refute", "concede", "defer"}
_QID_RE = re.compile(r"^Q\d+$")

STAGE3_REMEDIATION_HINTS = (
    "Output exactly ONE fenced ```json ... ``` block containing BOTH "
    "rebuttals and patches:\n"
    '{"rebuttals": [{"qid": "Q1", "answer": "...", '
    '"action": "refute|concede|defer"}], "patches": [...]}\n\n'
    "Each rebuttal MUST have: qid (matching the question), "
    "answer (your response text), and action (refute, concede, or defer).\n"
    "You must provide a rebuttal for EVERY question you received."
)


def validate_stage3_output(
    raw_response: str,
    expected_qids: list[str] | None = None,
) -> ValidationResult:
    """Validate that a Stage 3 response contains well-formed rebuttals.

    Checks (in order):
    1. A JSON block is present (fenced or raw object).
    2. The JSON is valid.
    3. ``rebuttals`` key is present with a non-empty list.
    4. Per-rebuttal required fields (qid, answer, action).
    5. qid coverage (if *expected_qids* provided).
    6. Patches structure (if ``patches`` key present).
    """
    errors: list[str] = []

    # 1. JSON extraction (fenced block preferred, raw fallback)
    json_str = _extract_fenced_json(raw_response)
    if json_str is None:
        json_str = _extract_any_json_object(raw_response)
    if json_str is None:
        errors.append(
            "No fenced ```json ... ``` block found. You must output exactly "
            "one fenced JSON block containing your rebuttals."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # 2. JSON parsing
    obj, parse_err = _try_parse_json(json_str)
    if obj is None:
        errors.append(f"JSON block is not valid JSON: {parse_err}")
        return ValidationResult(is_valid=False, errors=errors)

    # 3. rebuttals key
    rebuttals = obj.get("rebuttals")
    if not isinstance(rebuttals, list) or len(rebuttals) == 0:
        errors.append(
            "JSON block is missing 'rebuttals' key or rebuttals list is empty."
        )
        return ValidationResult(is_valid=False, errors=errors, parsed_data=obj)

    # 4. Per-rebuttal required fields
    seen_qids: set[str] = set()
    for idx, rb in enumerate(rebuttals):
        if not isinstance(rb, dict):
            errors.append(f"Rebuttal at index {idx} is not an object.")
            continue
        label = rb.get("qid", f"index {idx}")

        # qid
        qid = rb.get("qid")
        if not qid or not isinstance(qid, str):
            errors.append(f"Rebuttal at index {idx} is missing 'qid'.")
        elif not _QID_RE.match(qid):
            errors.append(
                f"Rebuttal '{qid}' has invalid qid format. Expected Q1, Q2, etc."
            )
        else:
            seen_qids.add(qid)

        # answer
        answer = rb.get("answer")
        if not answer or not isinstance(answer, str) or not answer.strip():
            errors.append(f"Rebuttal '{label}' is missing or has empty 'answer'.")

        # action
        action = rb.get("action")
        if not action or not isinstance(action, str):
            errors.append(f"Rebuttal '{label}' is missing 'action'.")
        elif action.strip().lower() not in _VALID_REBUTTAL_ACTIONS:
            errors.append(
                f"Rebuttal '{label}' has invalid action '{action}'. "
                f"Must be one of: refute, concede, defer."
            )

    # 5. qid coverage
    if expected_qids is not None:
        for qid in expected_qids:
            if qid not in seen_qids:
                errors.append(f"Missing rebuttal for question {qid}.")

    # 6. Patches structure (if present)
    patches = obj.get("patches")
    if patches is not None:
        if not isinstance(patches, list):
            errors.append("'patches' must be a list.")
        else:
            for i, patch in enumerate(patches):
                if not isinstance(patch, dict):
                    errors.append(f"Patch at index {i} is not an object.")
                elif not patch.get("op"):
                    errors.append(
                        f"Patch at index {i} is missing required 'op' field."
                    )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=obj,
    )


# ---------------------------------------------------------------------------
# Stage 5 — Belief Update Validators
# ---------------------------------------------------------------------------

STAGE5_PHASE1_REMEDIATION_HINTS = (
    "Output a <reasoning>...</reasoning> block followed by exactly ONE "
    "fenced ```json ... ``` block:\n"
    '{"patches": [{"op": "update_claim", "target_id": "C1", '
    '"changes": {"strength": 0.5, "strength_justification": "..."}}, ...]}\n\n'
    "For each CRITIQUE_VALID outcome, you MUST include at least one patch "
    "that weakens the targeted node.\n"
    'Every patch must have an "op" field. update_* patches need '
    '"target_id" and "changes". add_* patches need "item" with "id".'
)

STAGE5_PHASE2_REMEDIATION_HINTS = (
    "Output a <reasoning>...</reasoning> block followed by exactly ONE "
    "fenced ```json ... ``` block:\n"
    '{"patches": [{"op": "...", ...}, ...]}\n\n'
    'Even if you have no changes to make, output {"patches": []}.\n'
    'Every patch must have an "op" field. update_* patches need '
    '"target_id" and "changes". add_* patches need "item" with "id".'
)


def _validate_stage5_patches(
    raw_response: str,
    critique_valid_count: int = 0,
    enforce_non_empty: bool = True,
) -> ValidationResult:
    """Shared validation logic for Stage 5 Phase 1 and Phase 2.

    Checks (in order):
    1. A fenced JSON block is present.
    2. The JSON is valid.
    3. ``patches`` key is present.
    4. Enforcement compliance (Phase 1 only, when *enforce_non_empty*).
    5. Per-patch ``op`` field present.
    6. Op-specific required fields.
    """
    errors: list[str] = []

    # 1. JSON extraction (fenced block preferred, raw fallback)
    json_str = _extract_fenced_json(raw_response)
    if json_str is None:
        json_str = _extract_any_json_object(raw_response)
    if json_str is None:
        errors.append(
            "No fenced ```json ... ``` block found. You must output a "
            "```json ... ``` block containing your patches."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # 2. JSON parsing
    obj, parse_err = _try_parse_json(json_str)
    if obj is None:
        errors.append(f"JSON block is not valid JSON: {parse_err}")
        return ValidationResult(is_valid=False, errors=errors)

    # 3. patches key present
    patches = obj.get("patches")
    if patches is None:
        if enforce_non_empty:
            errors.append("JSON block is missing 'patches' key.")
        else:
            errors.append(
                "JSON block is missing 'patches' key. Even if you have no "
                'changes, output {"patches": []}.'
            )
        return ValidationResult(is_valid=False, errors=errors, parsed_data=obj)

    if not isinstance(patches, list):
        errors.append("'patches' must be a list.")
        return ValidationResult(is_valid=False, errors=errors, parsed_data=obj)

    # 4. Enforcement compliance (Phase 1 only)
    if enforce_non_empty and critique_valid_count > 0 and len(patches) == 0:
        errors.append(
            f"You received {critique_valid_count} CRITIQUE_VALID outcome(s) "
            "but returned zero patches. Each CRITIQUE_VALID requires at least "
            "one weakening patch (lower strength, retract, or refine the "
            "targeted node)."
        )

    # 5 & 6. Per-patch structural validation
    for i, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"Patch at index {i} is not an object.")
            continue

        op = patch.get("op")
        if not op or not isinstance(op, str):
            errors.append(
                f"Patch at index {i} is missing required 'op' field."
            )
            continue

        op_clean = op.strip()

        # Op-specific required fields
        if op_clean.startswith("update_") and op_clean != "update_thesis":
            if not patch.get("target_id"):
                errors.append(
                    f"Patch at index {i} (op='{op_clean}') is missing "
                    "required 'target_id' field."
                )
            if not isinstance(patch.get("changes"), dict):
                errors.append(
                    f"Patch at index {i} (op='{op_clean}') is missing "
                    "required 'changes' dict."
                )

        elif op_clean.startswith("add_"):
            item = patch.get("item")
            if not isinstance(item, dict):
                errors.append(
                    f"Patch at index {i} (op='{op_clean}') is missing "
                    "required 'item' dict."
                )
            elif not item.get("id"):
                errors.append(
                    f"Patch at index {i} (op='{op_clean}') 'item' is "
                    "missing required 'id' field."
                )

        elif op_clean == "update_thesis":
            has_content = any(
                k in patch
                for k in ("new_strength", "stance", "summary_bullets")
            )
            if not has_content:
                errors.append(
                    f"Patch at index {i} (op='update_thesis') must have at "
                    "least one of: 'new_strength', 'stance', 'summary_bullets'."
                )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=obj,
    )


def validate_stage5_phase1_output(
    raw_response: str,
    critique_valid_count: int = 0,
) -> ValidationResult:
    """Validate that a Stage 5 Phase 1 (enforcement) response has well-formed patches.

    Checks:
    1. A JSON block is present (fenced or raw object).
    2. The JSON is valid.
    3. ``patches`` key is present with a list value.
    4. Enforcement: if *critique_valid_count* > 0, patches list must be non-empty.
    5. Per-patch ``op`` field required.
    6. Op-specific required fields (target_id/changes, item/id, thesis fields).
    """
    return _validate_stage5_patches(
        raw_response,
        critique_valid_count=critique_valid_count,
        enforce_non_empty=True,
    )


def validate_stage5_phase2_output(raw_response: str) -> ValidationResult:
    """Validate that a Stage 5 Phase 2 (introspection) response has well-formed patches.

    Same as Phase 1 but without enforcement compliance — an empty patches
    array is valid in Phase 2 (introspective, not enforcement-bound).
    """
    return _validate_stage5_patches(
        raw_response,
        critique_valid_count=0,
        enforce_non_empty=False,
    )
