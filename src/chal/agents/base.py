"""
base.py

Defines the foundational interface and data structures for all agent implementations
in the chal framework. This includes:

- Message: A reusable dataclass for structured communication between agents.
- Agent (abstract base class): Establishes the required interface that all
  language model-backed agents must implement, including the `generate()` method.

This module ensures consistent behavior across different agent types and supports
clean integration into the broader debate orchestration system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from chal.beliefs.belief_graph import BeliefGraph
from chal.log import logger


@dataclass
class Message:
    """
    Represents a single message exchanged in the dialogue between agents.

    Attributes:
        role (str): The role of the speaker. Common values include:
            - "system": typically used for setup/context.
            - "user": an initiating prompt or question.
            - "assistant": a model's response.
        content (str): The raw message text.
        metadata (dict, optional): Optional metadata about the message,
            such as:
            - token count
            - model version
            - timestamp
            - strength scores
            Defaults to None.
    """
    role: str
    content: str
    metadata: dict | None = None


class Agent(ABC):
    """
    Abstract base class (ABC) for all language model-based agents.

    All concrete agent subclasses (e.g., GPTAgent, ClaudeAgent) must:
        - Inherit from this base class
        - Implement the `generate()` method

    Attributes:
        name (str): A human-readable identifier for the agent (e.g., "GPT-4o").
        model (str): The model identifier string (e.g., "gpt-4o", "claude-sonnet-4-6").
        system_prompt (str): A persistent instruction given to the agent to
            define its behavior or persona.
        temperature (float): Sampling temperature for generation.
        key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
        internal_belief (str): The agent's current internal belief text.
        internal_belief_obj (dict | None): Structured CBS belief object.
        belief_graph (BeliefGraph | None): Persistent belief graph built from belief_obj.
        persona_label (str): Short label derived from agent name (e.g., "Empiricist").
        all_beliefs_held (list): History of all beliefs the agent has held.
    """

    def __init__(self, name: str, model: str, system_prompt: str = "",
                 temperature: float = 0.7, key_pool=None):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.key_pool = key_pool
        self.internal_belief: str = ""
        self.internal_belief_obj: dict | None = None
        self.belief_graph: BeliefGraph | None = None
        self.persona_label: str = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held: list = []

    @abstractmethod
    def generate(self, history: list[Message]) -> Message:
        """
        Generates the next message from the agent, based on the full conversation history.

        Args:
            history (list[Message]): The entire sequence of messages exchanged
            so far (in chronological order). This includes both user and assistant turns.

        Returns:
            Message: A new message representing the agent's response.
        """
        pass

    def set_internal_belief(self, belief_text: str) -> None:
        """
        Sets the agent's internal belief, typically after Stages 1 or 5.
        """
        self.internal_belief = belief_text

    def get_internal_belief(self) -> str:
        """
        Retrieves the agent's internal belief for use in prompting.
        """
        return self.internal_belief

    def set_internal_belief_obj(self, belief_obj: dict | None) -> None:
        """
        Stores the structured CBS belief object (JSON as Python dict).
        Auto-rebuilds the persistent belief graph when the belief object changes.
        """
        self.internal_belief_obj = belief_obj

        if belief_obj:
            try:
                self.belief_graph = BeliefGraph(belief_obj)
            except Exception as e:
                logger.warning(f"Could not build belief graph for {self.name}: {e}")
                self.belief_graph = None
        else:
            self.belief_graph = None

    def get_internal_belief_obj(self) -> dict | None:
        """
        Returns the structured belief object if available.
        """
        return self.internal_belief_obj

    def get_belief_graph(self):
        """
        Returns the persistent BeliefGraph object if available.
        The graph is automatically rebuilt when set_internal_belief_obj() is called.

        Returns:
            BeliefGraph object or None if no belief object is set or graph construction failed.
        """
        return self.belief_graph

    def receive_system_prompt(self, prompt: str) -> None:
        """
        Assigns the agent's system-level behavior rules and norms.

        Args:
            prompt (str): The shared, universal system prompt applied to all agents.
        """
        self.system_prompt = prompt

    def receive_role_card(self, prompt: str) -> None:
        """
        Appends the agent's unique worldview or stance to their system prompt.

        Args:
            prompt (str): A string defining the specific position the agent must uphold.

        Side Effect:
            Updates `self.system_prompt` to combine position with prior system-level behavior.
        """
        self.system_prompt = self.system_prompt + "\n\n" + prompt
