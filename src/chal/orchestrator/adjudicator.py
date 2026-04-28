"""
adjudicator.py

Defines the Adjudicator class for evaluating challenge-rebuttal pairs in debates.

The adjudicator performs logical evaluation in a single API call, returning:
- Restatement of the disagreement
- Formalized logical structures for both sides
- Final outcome (rebuttal_valid, critique_valid, or unresolved)
- Reasoning for the decision

This replaces the previous 3-call approach, reducing API latency by ~66%.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from chal.agents.base import Agent, Message
from chal.agents.prompts import build_adjudicator_per_pair_prompt
from chal.utilities.retry import ValidationResult, generate_with_retry

VALID_VERDICTS = {"critique_valid", "rebuttal_valid", "unresolved"}

ADJUDICATOR_REMEDIATION_HINTS = (
    "MOST COMMON ERROR: Your previous response placed the JSON inside the "
    "<reasoning> tags. The JSON block MUST appear AFTER the closing </reasoning> tag.\n\n"
    "Remember: You MUST output BOTH parts:\n"
    "1. A <reasoning>...</reasoning> block with your analysis\n"
    "2. A SEPARATE fenced ```json ... ``` block with the structured verdict\n\n"
    "The JSON block must appear OUTSIDE the reasoning tags. "
    "Do NOT nest JSON inside <reasoning>.\n"
    'The JSON must contain at minimum: "outcome", "reasoning", and "restatement" fields.\n\n'
    "Correct output structure:\n\n"
    "<reasoning>Your analysis here...</reasoning>\n\n"
    "```json\n"
    '{"outcome": "...", "reasoning": "...", "restatement": "...", ...}\n'
    "```"
)


def _normalize_verdict(raw: str) -> str:
    """Normalize a parsed verdict to one of the three valid values.

    Returns 'unresolved' for any unrecognized value.
    """
    cleaned = raw.strip().lower()
    return cleaned if cleaned in VALID_VERDICTS else "unresolved"


def _extract_json_from_response(text: str) -> dict | None:
    """Extract and parse JSON from an LLM response string.

    Handles JSON wrapped in markdown code fences or embedded in prose.
    Uses ``json.JSONDecoder.raw_decode`` instead of hand-rolled brace scanning.

    Args:
        text: Raw LLM response that may contain JSON.

    Returns:
        Parsed JSON as a dictionary, or ``None`` if no valid JSON object is found.
    """
    # 1. Try markdown code fences first
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try raw_decode to find first JSON object in text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in '{[':
            try:
                result, _ = decoder.raw_decode(text, i)
                return result
            except json.JSONDecodeError:
                continue

    return None


def validate_adjudicator_output(raw_response: str) -> ValidationResult:
    """Validate that an adjudicator response contains a well-formed verdict.

    Checks:
    1. A JSON block is present (fenced or brace-scanned).
    2. The JSON is valid.
    3. Required field ``outcome`` exists and is a recognized verdict.
    4. Required field ``reasoning`` exists and is non-empty.
    5. Required field ``restatement`` exists and is non-empty.
    6. Required field ``formalization_challenger`` exists and is non-empty.
    7. Required field ``formalization_target`` exists and is non-empty.
    8. Required field ``scores`` exists with all 6 numeric keys in [0.0, 1.0].
    """
    errors: list[str] = []
    data = _extract_json_from_response(raw_response)

    if data is None:
        errors.append(
            "No JSON block found in response. You must output a fenced "
            "```json ... ``` block OUTSIDE the <reasoning> tags."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # Required: outcome
    outcome = data.get("outcome")
    if not outcome or not isinstance(outcome, str) or not outcome.strip():
        errors.append("JSON block is missing required field 'outcome'.")
    else:
        normalized = outcome.strip().lower()
        if normalized not in VALID_VERDICTS:
            errors.append(
                f"Unrecognized outcome value '{outcome}'. "
                f"Must be one of: rebuttal_valid, critique_valid, unresolved."
            )

    # Required: reasoning
    reasoning = data.get("reasoning")
    if not reasoning or not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("JSON block is missing required field 'reasoning'.")

    # Required: restatement
    restatement = data.get("restatement")
    if not restatement or not isinstance(restatement, str) or not restatement.strip():
        errors.append("JSON block is missing required field 'restatement'.")

    # Required: formalization_challenger
    fc = data.get("formalization_challenger")
    if not fc or not isinstance(fc, str) or not fc.strip():
        errors.append("JSON block is missing required field 'formalization_challenger'.")

    # Required: formalization_target
    ft = data.get("formalization_target")
    if not ft or not isinstance(ft, str) or not ft.strip():
        errors.append("JSON block is missing required field 'formalization_target'.")

    # Required: scores (dict with 6 numeric keys)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        errors.append("JSON block is missing required field 'scores' (must be a dict).")
    else:
        required_score_keys = [
            "challenger_logic", "challenger_ethics",
            "defender_logic", "defender_ethics",
            "challenger_combined", "defender_combined",
        ]
        for key in required_score_keys:
            val = scores.get(key)
            if val is None:
                errors.append(f"scores is missing required key '{key}'.")
            elif not isinstance(val, (int, float)):
                errors.append(f"scores['{key}'] must be a number, got {type(val).__name__}.")
            elif not (0.0 <= val <= 1.0):
                errors.append(f"scores['{key}'] = {val} is out of range [0.0, 1.0].")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=data,
    )


def enforce_verdict(scores: dict, logic_weight: float, ethics_weight: float, threshold: float = 0.15) -> dict:
    """Recompute combined scores and determine verdict from math.

    Applies the scoring formula: combined = logic_weight * logic + ethics_weight * ethics.
    If challenger_combined - defender_combined >= threshold -> critique_valid.
    If defender_combined - challenger_combined >= threshold -> rebuttal_valid.
    Otherwise -> unresolved.

    Args:
        scores: Dict with challenger_logic, challenger_ethics, defender_logic, defender_ethics.
        logic_weight: Weight for logic scores (0.0-1.0).
        ethics_weight: Weight for ethics scores (0.0-1.0).
        threshold: Minimum gap for a decisive verdict (default 0.15).

    Returns:
        Dict with computed_verdict, challenger_combined, defender_combined, gap.
    """
    cl = scores.get("challenger_logic", 0.5)
    ce = scores.get("challenger_ethics", 0.5)
    dl = scores.get("defender_logic", 0.5)
    de = scores.get("defender_ethics", 0.5)

    challenger_combined = logic_weight * cl + ethics_weight * ce
    defender_combined = logic_weight * dl + ethics_weight * de

    # Round to 4 decimal places before comparison to eliminate IEEE 754
    # floating-point noise (e.g. 0.14999999999999991 instead of 0.15).
    gap = round(challenger_combined - defender_combined, 4)

    if gap >= threshold:
        computed_verdict = "critique_valid"
    elif gap <= -threshold:
        computed_verdict = "rebuttal_valid"
    else:
        computed_verdict = "unresolved"

    return {
        "computed_verdict": computed_verdict,
        "challenger_combined": round(challenger_combined, 4),
        "defender_combined": round(defender_combined, 4),
        "gap": gap,
    }


class Adjudicator:
    """
    Evaluates challenge-rebuttal pairs using logical and ethical frameworks.

    Uses a single consolidated API call to:
    1. Restate the core disagreement
    2. Formalize both arguments
    3. Adjudicate the outcome
    """

    _MODE_LABELS = {
        "logic_only": "Pure Logic",
        "balanced": "Balanced",
        "ethics_only": "Pure Ethics",
    }

    def __init__(self, adjudicator_agent: Agent, logic_weight: float = 1.0, ethics_weight: float = 0.0,
                 logic_sys: str = "", ethics_sys: str = "", threshold: float = 0.15) -> None:
        """
        Initialize the Adjudicator with evaluation frameworks and weights.

        Args:
            adjudicator_agent: An LLM agent instance for logical evaluation.
            logic_weight: Weight for logical rigor (0.0-1.0).
            ethics_weight: Weight for ethical considerations (0.0-1.0).
            logic_sys: Logic system description string (for the per-pair prompt).
            ethics_sys: Ethics system description string (for the per-pair prompt).
        """
        self.agent = adjudicator_agent
        self.logic_weight = logic_weight
        self.ethics_weight = ethics_weight
        self.logic_sys = logic_sys
        self.ethics_sys = ethics_sys
        self.threshold = threshold
        # Determine mode label from weights
        if ethics_weight < 0.01:
            self._mode = "logic_only"
        elif logic_weight < 0.01:
            self._mode = "ethics_only"
        else:
            self._mode = "balanced"
        self.mode_label = self._MODE_LABELS[self._mode]

    def run(
        self,
        challenge: str,
        rebuttal: str,
        challenger: str,
        target: str,
        challenger_belief_excerpt_json: str = "",
        target_belief_excerpt_json: str = "",
        max_retries: int = 3,
        log_fn: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a challenge-rebuttal pair, retrying on malformed output.

        Args:
            challenge: The original critique.
            rebuttal: The rebuttal issued by the target agent.
            challenger: Name of the agent issuing the critique.
            target: Name of the agent issuing the rebuttal.
            challenger_belief_excerpt_json: Optional JSON excerpt of challenger's belief.
            target_belief_excerpt_json: Optional JSON excerpt of target's belief.
            max_retries: Maximum retry attempts on validation failure.
            log_fn: Optional ``(message, level) -> None`` logging callback.

        Returns:
            dict: Resolution record with keys:
                - status: "rebuttal_valid", "critique_valid", or "unresolved"
                - reasoning: Justification for the decision
                - restatement: Clarified summary of the disagreement
                - formalizations: Dict with "challenger" and "target" logical structures
                - scores: Dict with 6 score fields (if available)
                - _retry_records: list of retry record dicts (only if retries occurred)
        """
        prompt = build_adjudicator_per_pair_prompt(
            challenge=challenge,
            rebuttal=rebuttal,
            challenger=challenger,
            target=target,
            mode_label=self.mode_label,
            logic_sys_description=self.logic_sys,
            ethics_sys_description=self.ethics_sys,
            challenger_belief_excerpt_json=challenger_belief_excerpt_json,
            target_belief_excerpt_json=target_belief_excerpt_json,
        )

        response, retry_records = generate_with_retry(
            agent=self.agent,
            messages=[Message(role="user", content=prompt)],
            validator_fn=validate_adjudicator_output,
            max_retries=max_retries,
            stage_label="Stage 4 Adjudication",
            log_fn=log_fn,
            remediation_hints=ADJUDICATOR_REMEDIATION_HINTS,
        )
        raw_response = response.content

        # Extract usage and retry count for operational metrics
        _usage = {}
        if response.metadata:
            _usage = response.metadata.get("usage", {})
        _retry_count = len(retry_records)

        # Parse JSON — use the shared extractor (same logic the validator uses)
        result = _extract_json_from_response(raw_response)

        _debug: dict[str, Any] = {
            "_debug_prompt": prompt,
            "_debug_raw_response": raw_response,
        }

        if retry_records:
            _debug["_retry_records"] = [
                {"attempt": r.attempt, "errors": r.errors}
                for r in retry_records
            ]

        # Prefer the full <reasoning> block over the short JSON summary
        reasoning_match = re.search(
            r'<reasoning>(.*?)</reasoning>', raw_response, flags=re.DOTALL
        )
        full_reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        if result is not None:
            llm_verdict = _normalize_verdict(result.get("outcome", ""))
            scores = result.get("scores", {})

            # Programmatic verdict enforcement
            enforcement = enforce_verdict(scores, self.logic_weight, self.ethics_weight, self.threshold)
            computed_verdict = enforcement["computed_verdict"]

            override_occurred = (computed_verdict != llm_verdict)
            if override_occurred and log_fn:
                log_fn(
                    f"Verdict override: LLM said '{llm_verdict}' but math says "
                    f"'{computed_verdict}' (gap={enforcement['gap']})",
                    "WARNING",
                )

            return {
                "status": computed_verdict,
                "reasoning": full_reasoning or result.get("reasoning", ""),
                "restatement": result.get("restatement", ""),
                "formalizations": {
                    "challenger": result.get("formalization_challenger", ""),
                    "target": result.get("formalization_target", "")
                },
                "scores": scores,
                "override_occurred": override_occurred,
                "llm_verdict": llm_verdict,
                "challenger_combined": enforcement["challenger_combined"],
                "defender_combined": enforcement["defender_combined"],
                "gap": enforcement["gap"],
                "_usage": _usage,
                "_retry_count": _retry_count,
                **_debug,
            }

        # Fallback: try to parse old plain-text format
        outcome_match = re.search(r'(?:Outcome|outcome):\s*(\w+)', raw_response)
        raw_status = outcome_match.group(1).strip().lower() if outcome_match else ""
        status = _normalize_verdict(raw_status)

        reason_match = re.search(r'(?:Reasoning|reasoning):\s*(.+)', raw_response, re.DOTALL)
        fallback_reasoning = reason_match.group(1).strip() if reason_match else raw_response.strip()

        return {
            "status": status,
            "reasoning": full_reasoning or fallback_reasoning,
            "restatement": "Unable to parse restatement",
            "formalizations": {"challenger": "", "target": ""},
            "scores": {},
            "_usage": _usage,
            "_retry_count": _retry_count,
            **_debug,
        }
