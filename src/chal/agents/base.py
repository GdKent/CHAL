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

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


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
            - confidence scores
            Defaults to None.
    """
    role: str
    content: str
    metadata: dict = None


class Agent(ABC):
    """
    Abstract base class (ABC) for all language model-based agents.

    All concrete agent subclasses (e.g., GPTAgent, ClaudeAgent) must:
        - Inherit from this base class
        - Implement the `generate()` method

    Attributes:
        name (str): A human-readable identifier for the agent (e.g., "GPT-4o").
        system_prompt (str): A persistent instruction given to the agent to
            define its behavior or persona.
    """

    name: str
    system_prompt: str

    @abstractmethod
    def generate(self, history: List[Message]) -> Message:
        """
        Generates the next message from the agent, based on the full conversation history.

        Args:
            history (List[Message]): The entire sequence of messages exchanged
            so far (in chronological order). This includes both user and assistant turns.

        Returns:
            Message: A new message representing the agent's response.
        """
        pass
    
    @abstractmethod
    def receive_system_prompt(self, prompt: str) -> None:
        """
        Set the agent's universal system instructions.

        Args:
            prompt (str): A string defining global rules and expectations for behavior.
        """
        pass

    @abstractmethod
    def receive_role_card(self, prompt: str) -> None:
        """
        Set the agent's role/persona-specific instructions (e.g., "You are a rationalist").

        Args:
            prompt (str): A string defining the agent's position or perspective in the debate.
        """
        pass