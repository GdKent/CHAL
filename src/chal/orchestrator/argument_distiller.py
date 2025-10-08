"""
argument_distiller.py

Defines a distillation agent that analyzes the full final transcript
of a debate and extracts the clearest, most logically sound arguments
from each participant.

It then synthesizes the core unresolved disagreement or conclusion of the debate.
"""

from chal.agents.base import Agent, Message
from typing import List


class ArgumentDistillerAgent:
    """
    Uses a language model agent to distill the final positions from a completed debate.

    Attributes:
        agent (Agent): An LLM-based agent to generate distilled output.
        format (str): Optional format style — e.g., 'bullet', 'logical', 'narrative'.
    """

    def __init__(self, agent: Agent, format: str = "bullet"):
        """
        Initializes the distiller with a backing agent and optional style.

        Args:
            agent (Agent): The LLM used to perform the distillation.
            format (str): Output format (default: 'bullet').
        """
        self.agent = agent
        self.format = format

    def distill(self, transcript: List[Message]) -> Message:
        """
        Distills the most logically sound arguments and unresolved issues
        from the full debate transcript.

        Args:
            transcript (List[Message]): The entire (or summarized) conversation history.

        Returns:
            Message: A single message containing the distilled positions of each participant
                     and the core debate tension or insight.
        """
        system_prompt = f"""
                        You are an impartial debate distiller.

                        Your job is to read the final transcript of a structured philosophical debate,
                        and extract each agent's strongest, clearest argument from the discussion.

                        You must also identify:
                        - The single most important unresolved disagreement or question
                        - Whether any points of convergence emerged
                        - Any implicit assumptions or logical gaps still present

                        Format your answer using a {self.format} style like this:

                        - Agent A: <strongest formal argument or chain of reasoning>
                        - Agent B: <strongest formal argument or chain of reasoning>
                        - Core Unresolved Tension: <short synthesis of main point of divergence>
                        - Points of Agreement (if any): <shared conclusions>
                        - Outstanding Philosophical Question: <the question both sides must address next>
                        """.strip()

        # Join messages into a transcript
        debate_lines = [
            f"{msg.role.upper()}: {msg.content}" for msg in transcript
        ]
        full_text = "\n".join(debate_lines)

        request = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=full_text),
        ]

        return self.agent.generate(request)
