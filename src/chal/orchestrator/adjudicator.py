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

from typing import Dict, Any
from chal.agents.base import Agent, Message
import json
import re


class Adjudicator:
    """
    Evaluates challenge-rebuttal pairs using logical and ethical frameworks.

    Uses a single consolidated API call to:
    1. Restate the core disagreement
    2. Formalize both arguments
    3. Adjudicate the outcome
    """

    def __init__(self, adjudicator_agent: Agent, logic_weight: float = 1.0, ethics_weight: float = 0.0,
                 logic_sys: str = "", ethics_sys: str = "") -> None:
        """
        Initialize the Adjudicator with evaluation frameworks and weights.

        Args:
            adjudicator_agent: An LLM agent instance for logical evaluation.
            logic_weight: Weight for logical rigor (0.0-1.0).
            ethics_weight: Weight for ethical considerations (0.0-1.0).
            logic_sys: Logical framework description.
            ethics_sys: Ethical framework description.
        """
        self.agent = adjudicator_agent
        self.logic_weight = logic_weight
        self.ethics_weight = ethics_weight
        self.logic_sys = logic_sys
        self.ethics_sys = ethics_sys

    def run(self, challenge: str, rebuttal: str, challenger: str, target: str) -> Dict[str, Any]:
        """
        Evaluate a challenge-rebuttal pair in a single API call.

        Args:
            challenge: The original critique.
            rebuttal: The rebuttal issued by the target agent.
            challenger: Name of the agent issuing the critique.
            target: Name of the agent issuing the rebuttal.

        Returns:
            dict: Resolution record with keys:
                - status: "rebuttal_valid", "critique_valid", or "unresolved"
                - reasoning: Justification for the decision
                - restatement: Clarified summary of the disagreement
                - formalizations: Dict with "challenger" and "target" logical structures
        """
        prompt = f"""
You are a neutral adjudicator evaluating a philosophical debate exchange.

CHALLENGE from {challenger}:
\"\"\"{challenge}\"\"\"

REBUTTAL from {target}:
\"\"\"{rebuttal}\"\"\"

EVALUATION FRAMEWORK:
- Logic weight: {self.logic_weight}
- Ethics weight: {self.ethics_weight}
- Logic system: {self.logic_sys or 'Classical logic with Bayesian inference'}
- Ethics system: {self.ethics_sys or 'Not applicable'}

TASKS (complete all in one response):

1. RESTATE the core disagreement neutrally and clearly.

2. FORMALIZE both arguments:
   - Express the challenge as a syllogism or inference chain
   - Express the rebuttal similarly, indicating which premises it rejects/modifies

3. ADJUDICATE using the weighted framework (combined_score = LOGIC*{self.logic_weight} + ETHICS*{self.ethics_weight}):
   - Outcome must be ONE of: rebuttal_valid, critique_valid, unresolved
   - Use "unresolved" sparingly (only for genuine logical standoffs)
   - Provide clear reasoning

OUTPUT FORMAT (strict JSON):
```json
{{
  "restatement": "<neutral summary of disagreement>",
  "formalization_challenger": "<formal logical structure of challenge>",
  "formalization_target": "<formal logical structure of rebuttal>",
  "outcome": "rebuttal_valid|critique_valid|unresolved",
  "reasoning": "<justification for outcome, referencing logical principles>"
}}
```
"""
        response = self.agent.generate([Message(role="user", content=prompt)])

        # Parse JSON response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, flags=re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return {
                    "status": result.get("outcome", "unknown").lower(),
                    "reasoning": result.get("reasoning", ""),
                    "restatement": result.get("restatement", ""),
                    "formalizations": {
                        "challenger": result.get("formalization_challenger", ""),
                        "target": result.get("formalization_target", "")
                    }
                }
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse old format
        outcome_match = re.search(r'(?:Outcome|outcome):\s*(\w+)', response.content)
        status = outcome_match.group(1).strip().lower() if outcome_match else "unknown"

        reason_match = re.search(r'(?:Reasoning|reasoning):\s*(.+)', response.content, re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else response.content.strip()

        return {
            "status": status,
            "reasoning": reasoning,
            "restatement": "Unable to parse restatement",
            "formalizations": {"challenger": "", "target": ""}
        }