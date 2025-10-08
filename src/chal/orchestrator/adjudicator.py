"""
adjudicator.py

This module defines the Adjudicator class, which implements a multi-stage
truth-seeking protocol to evaluate and resolve critique-rebuttal pairs.
It is used in Stage 4 of the dialectical debate controller.
"""

from typing import Dict
from chal.agents.base import Message
import re

class Adjudicator:
    def __init__(self, adjudicator_agent, logic_weight: float = 1.0, ethics_weight: float = 0.0, logic_sys: str = "", ethics_sys: str = ""):
        """
        Initializes the Adjudicator with logical and ethical framework systems as well as their respective weights when adjudicating a conflict.

        Args:
            adjudicator_agent: An LLM agent instance capable of logical evaluation.
        """
        self.agent = adjudicator_agent
        self.logic_weight = logic_weight
        self.ethics_weight = ethics_weight
        self.logic_sys = logic_sys
        self.ethics_sys = ethics_sys


    def restate_disagreement(self, challenge: str, rebuttal: str, challenger: str, target: str) -> str:
        """
        Subprotocol 1: Requests the logic agent to restate the disagreement.

        Returns:
            str: A clarified summary of the disagreement.
        """
        prompt = f"""
                Agent {challenger} has issued the following critique:
                \"\"\"{challenge}\"\"\"

                Agent {target} has issued the following rebuttal:
                \"\"\"{rebuttal}\"\"\"

                Restate the core point of disagreement as clearly and neutrally as possible.
                Avoid introducing new arguments. Just clarify what the disagreement is. Think about this step-by-step.
                """
        response = self.agent.generate([Message(role="user", content=prompt)])
        return response.content.strip()


    def formalize_both_sides(self, challenge: str, rebuttal: str, challenger: str, target: str) -> Dict[str, str]:
        """
        Subprotocol 2: Formalizes both the challenge and the rebuttal into logical structure.

        Returns:
            dict: {"challenger": formalized_challenge, "target": formalized_rebuttal}
        """
        prompt = f"""
                Below is a debate between two agents.

                Agent {challenger}'s critique:
                \"\"\"{challenge}\"\"\"

                Agent {target}'s rebuttal:
                \"\"\"{rebuttal}\"\"\"

                Formalize both sides using logical structure:
                - For the critique, write it as a syllogism or clear inference chain.
                - For the rebuttal, do the same and indicate which premise or inference it rejects or modifies.
                Think about this step-by-step.

                Return your response in the following format:
                ---
                [Challenger - {challenger}]:
                <formalized critique>

                [Target - {target}]:
                <formalized rebuttal>
                ---
                """
        response = self.agent.generate([Message(role="user", content=prompt)])

        # Simple parser for expected format
        split = response.content.split("[Target -")
        challenger_formal = split[0].split("[Challenger -")[-1].strip()
        target_formal = split[1].split("]:", 1)[-1].strip() if len(split) > 1 else ""

        return {
            "challenger": challenger_formal,
            "target": target_formal
        }


    def adjudicate(self, challenger: str, target: str, formalizations: Dict[str, str]) -> dict:
        """
        Subprotocol 3: Adjudicates the challenge-rebuttal pair based on logical structure.

        Args:
            challenger (str): Name of the challenger.
            target (str): Name of the target agent.
            formalizations (dict): Dictionary with formal logic strings.

        Returns:
            dict: {"status": ..., "reasoning": ...}
        """
        prompt = f"""
        Evaluate the following debate with a weighted policy over LOGIC and ETHICS.

        Weights (documentary; the system prompt already encodes them):
        - LOGIC weight: {getattr(self, 'logic_weight', 1.0)}
        - ETHICS weight: {getattr(self, 'ethics_weight', 0.0)}

        [Challenger - {challenger}]:
        {formalizations['challenger']}

        [Target - {target}]:
        {formalizations['target']}

        Decide whether the rebuttal resolves the challenge, using the weighted rule
        (combined_score = LOGIC*w_L + ETHICS*w_E). Use the three outcome labels (use *unresolved* sparingly):
        - rebuttal_valid
        - critique_valid
        - unresolved

        Return in this format:
        Outcome: <label>
        Reasoning: <justification>
        """
        response = self.agent.generate([Message(role="user", content=prompt)])

        match = re.search(r'Outcome:\s*(\w+)', response.content)
        status = match.group(1).strip() if match else "unknown"

        reason_match = re.search(r'Reasoning:\s*(.+)', response.content, re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else response.content.strip()

        return {"status": status.lower(), "reasoning": reasoning}


    def run(self, challenge: str, rebuttal: str, challenger: str, target: str) -> dict:
        """
        Runs all four subprotocols to evaluate a single critique-rebuttal pair.

        Args:
            challenge (str): The original critique.
            rebuttal (str): The rebuttal issued by the target agent.
            challenger (str): Name of the agent issuing the critique.
            target (str): Name of the agent issuing the rebuttal.

        Returns:
            dict: A resolution record with status, reasoning, restatement, and formalizations.
        """
        restatement = self.restate_disagreement(challenge, rebuttal, challenger, target)
        formalizations = self.formalize_both_sides(challenge, rebuttal, challenger, target)
        resolution = self.adjudicate(challenger, target, formalizations)

        return {
            "status": resolution["status"],
            "reasoning": resolution["reasoning"],
            "restatement": restatement,
            "formalizations": formalizations
        }