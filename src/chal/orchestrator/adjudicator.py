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

    def run(self, challenge: str, rebuttal: str, challenger: str, target: str,
            challenger_belief_excerpt_json: str = "", target_belief_excerpt_json: str = "") -> Dict[str, Any]:
        """
        Evaluate a challenge-rebuttal pair in a single API call.

        Args:
            challenge: The original critique.
            rebuttal: The rebuttal issued by the target agent.
            challenger: Name of the agent issuing the critique.
            target: Name of the agent issuing the rebuttal.
            challenger_belief_excerpt_json: Optional JSON excerpt of challenger's belief.
            target_belief_excerpt_json: Optional JSON excerpt of target's belief.

        Returns:
            dict: Resolution record with keys:
                - status: "rebuttal_valid", "critique_valid", or "unresolved"
                - reasoning: Justification for the decision
                - restatement: Clarified summary of the disagreement
                - formalizations: Dict with "challenger" and "target" logical structures
                - scores: Dict with 6 score fields (if available)
        """
        # Build optional belief excerpt sections
        challenger_excerpt_section = ""
        if challenger_belief_excerpt_json:
            challenger_excerpt_section = (
                f"<challenger_belief_excerpt>\n"
                f"```json\n{challenger_belief_excerpt_json}\n```\n"
                f"</challenger_belief_excerpt>\n\n"
            )

        target_excerpt_section = ""
        if target_belief_excerpt_json:
            target_excerpt_section = (
                f"<target_belief_excerpt>\n"
                f"```json\n{target_belief_excerpt_json}\n```\n"
                f"</target_belief_excerpt>\n"
            )

        prompt = (
            "<context>\n"
            f"<challenge from=\"{challenger}\">\n"
            f"{challenge}\n"
            "</challenge>\n\n"
            f"<rebuttal from=\"{target}\">\n"
            f"{rebuttal}\n"
            "</rebuttal>\n\n"
            + challenger_excerpt_section
            + target_excerpt_section
            + "</context>\n\n"
            "<instructions>\n"
            f"Evaluate this exchange. Logic weight: {self.logic_weight}, Ethics weight: {self.ethics_weight}, "
            f"System: {self.logic_sys or 'Classical logic with Bayesian inference'} / "
            f"{self.ethics_sys or 'Not applicable'}.\n\n"
            "Execute your three-step protocol. Verify cited evidence against the belief excerpts above. "
            "If the challenge targets a counterposition (X#) the defender already rated as \"partial\" or "
            "\"unaddressed,\" factor this into your assessment. Inside <reasoning> tags, analyze step by "
            "step before rendering your verdict.\n"
            "</instructions>\n\n"
            "<output_format>\n"
            "1. <reasoning>...</reasoning> tags\n"
            "2. One fenced JSON code block:\n\n"
            "```json\n"
            "{\n"
            "  \"restatement\": \"\",\n"
            "  \"formalization_challenger\": \"\",\n"
            "  \"formalization_target\": \"\",\n"
            "  \"scores\": {\n"
            "    \"challenger_logic\": 0.0,\n"
            "    \"challenger_ethics\": 0.0,\n"
            "    \"defender_logic\": 0.0,\n"
            "    \"defender_ethics\": 0.0,\n"
            "    \"challenger_combined\": 0.0,\n"
            "    \"defender_combined\": 0.0\n"
            "  },\n"
            "  \"outcome\": \"rebuttal_valid|critique_valid|unresolved\",\n"
            "  \"reasoning\": \"\"\n"
            "}\n"
            "```\n"
            "</output_format>\n"
        )
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
                    },
                    "scores": result.get("scores", {})
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
            "formalizations": {"challenger": "", "target": ""},
            "scores": {}
        }