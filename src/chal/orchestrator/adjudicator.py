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

from typing import Any, Callable, Dict

from chal.agents.base import Agent, Message
from chal.agents.prompts import build_adjudicator_per_pair_prompt
from chal.utilities.retry import ValidationResult, generate_with_retry
import json
import re

VALID_VERDICTS = {"critique_valid", "rebuttal_valid", "unresolved"}

ADJUDICATOR_REMEDIATION_HINTS = (
    "Remember: You MUST output BOTH parts:\n"
    "1. A <reasoning>...</reasoning> block with your analysis\n"
    "2. A SEPARATE fenced ```json ... ``` block with the structured verdict\n\n"
    "The JSON block must appear OUTSIDE the reasoning tags. "
    "Do NOT nest JSON inside <reasoning>.\n"
    'The JSON must contain at minimum: "outcome", "reasoning", and "restatement" fields.'
)


def _normalize_verdict(raw: str) -> str:
    """Normalize a parsed verdict to one of the three valid values.

    Returns 'unresolved' for any unrecognized value.
    """
    cleaned = raw.strip().lower()
    return cleaned if cleaned in VALID_VERDICTS else "unresolved"


def _extract_json_from_response(text: str) -> dict | None:
    """Try to extract a JSON object from *text*.

    Attempts fenced ```json block first, then brace-depth scanning.
    Returns the parsed dict, or None if nothing was found.
    """
    # 1. Fenced block
    fenced = re.search(r'```json\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Brace-depth scanning
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth, in_string, escape_next = 0, False, False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return None


def validate_adjudicator_output(raw_response: str) -> ValidationResult:
    """Validate that an adjudicator response contains a well-formed verdict.

    Checks:
    1. A JSON block is present (fenced or brace-scanned).
    2. The JSON is valid.
    3. Required field ``outcome`` exists and is a recognized verdict.
    4. Required field ``reasoning`` exists and is non-empty.
    5. Required field ``restatement`` exists and is non-empty.
    6. ``scores`` is optional (warning only, not a failure).
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

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_data=data,
    )


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
                 logic_sys: str = "", ethics_sys: str = "") -> None:
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
    ) -> Dict[str, Any]:
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
            return {
                "status": _normalize_verdict(result.get("outcome", "")),
                "reasoning": full_reasoning or result.get("reasoning", ""),
                "restatement": result.get("restatement", ""),
                "formalizations": {
                    "challenger": result.get("formalization_challenger", ""),
                    "target": result.get("formalization_target", "")
                },
                "scores": result.get("scores", {}),
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
            **_debug,
        }