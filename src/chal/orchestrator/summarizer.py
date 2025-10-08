"""
summarizer.py

Defines a summarization agent that takes the full transcript of a debate
and generates a concise summary of the key points raised by each participant.

This is useful for logging, analysis, or feeding into downstream argument evaluation tools.
"""

from chal.agents.base import Agent, Message
from chal.agents.openai_agent import OpenAIAgent
from typing import List


class SummarizerAgent:
    """
    Uses a backing LLM agent to summarize the key points from a debate transcript.

    Attributes:
        agent (Agent): The summarizing agent (e.g., OpenAI or Claude).
        style (str): Output style — 'bullet', 'narrative', etc.
    """

    def __init__(self, agent: Agent, style: str = "bullet"):
        self.agent = agent
        self.style = style

    def summarize(self, transcript: List[Message], logic_notes: str = "", agent_names: List[str] = None) -> Message:
        """
        Summarizes the key points from a round, optionally incorporating
        logic enforcement results.

        Args:
            transcript (List[Message]): Messages from the round.
            logic_notes (str): (Optional) Summary of logical inconsistencies.

        Returns:
            Message: A single summarization message.
        """
        # Build agent-specific headers
        claims_section = "\n".join([f"- {name} key claims:" for name in agent_names])
        next_steps_section = "\n".join([f"- {name} should address next:" for name in agent_names])

        system_prompt = f"""
                        You are a neutral summarizer of a philosophical debate round.

                        Your job is to:
                        - Extract each participant's most important arguments and responses
                        - Summarize outstanding points of tension
                        - Integrate insights from logic enforcement (if any)
                        - Suggest what each agent should focus on next

                        Logic issues to consider (if any):
                        {logic_notes.strip() if logic_notes else "None reported."}

                        Return your output in the following format:

                        {claims_section}
                        - Outstanding points of tension:
                        - Issues raised by logic enforcement:
                        {next_steps_section}
                        """.strip()

        debate_lines = [f"{msg.role.upper()}: {msg.content}" for msg in transcript]
        full_text = "\n".join(debate_lines)

        summary_request = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=full_text),
        ]
        return self.agent.generate(summary_request)
