"""
anthropic_agent.py

Defines an LLM-powered agent that uses Anthropic's Messages API
(e.g., Claude Opus, Sonnet, Haiku) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an Anthropic API key to be available in the environment as `ANTHROPIC_API_KEY`.
- Can be instantiated with any supported Anthropic chat model (e.g., "claude-sonnet-4-6").
"""

import os
import time
import anthropic
from chal.agents.base import Agent, Message
from typing import List
from dotenv import load_dotenv
load_dotenv()


class AnthropicAgent(Agent):
    """
    An agent that interacts with Anthropic's Claude models.

    Attributes:
        model (str): Anthropic model name (e.g., "claude-sonnet-4-6").
        api_key (str): Anthropic API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Rationalist".
    """

    def __init__(self, model: str, name: str, api_key: str = None, system_prompt: str = ""):
        """
        Initializes the AnthropicAgent with model and optional prompt/key.

        Args:
            model (str): The name of the Anthropic model to use (e.g., "claude-sonnet-4-6").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
        """
        self.model = model
        self.name = name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.system_prompt = system_prompt
        self.internal_belief = ""
        self.internal_belief_obj = None
        self.belief_graph = None
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = []
        self._client = anthropic.Anthropic(api_key=self.api_key)

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
                from chal.beliefs.belief_graph import BeliefGraph
                self.belief_graph = BeliefGraph(belief_obj)
            except Exception as e:
                print(f"Warning: Could not build belief graph for {self.name}: {e}")
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

    def generate(self, history: List[Message], temperature: float = 0.7) -> Message:
        """
        Constructs a prompt from conversation history, sends it to Anthropic,
        and returns the model's reply wrapped in a Message object.

        Note: The Anthropic Messages API accepts only "user" and "assistant" roles
        in the messages list. Any "system" messages in history are filtered out;
        the system prompt is passed as a separate top-level parameter.
        """
        # Filter to only user/assistant roles — Anthropic does not accept "system" in messages
        messages = [
            {"role": m.role, "content": m.content}
            for m in history
            if m.role in ("user", "assistant")
        ]

        try:
            response = retry_anthropic_message(
                client=self._client,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                temperature=temperature,
            )

            return Message(
                role="assistant",
                content=response.content[0].text,
                metadata={"model": response.model, "usage": dict(response.usage)}
            )

        except Exception as e:
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_anthropic_message(client, model, system_prompt, messages, temperature,
                             max_retries=5, base_delay=60.0):
    """
    Wrapper to retry Anthropic message creation with exponential backoff.

    Args:
        client: Instantiated anthropic.Anthropic client
        model (str): Anthropic model name
        system_prompt (str): System-level instruction (passed as top-level parameter)
        messages (list): List of {"role": ..., "content": ...} dicts (user/assistant only)
        temperature (float): Sampling temperature
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds

    Returns:
        anthropic.types.Message: The raw Anthropic response object
    """
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
                "temperature": temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            return client.messages.create(**kwargs)

        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            wait = base_delay * (2 ** attempt)
            print(f"[Retry {attempt+1}/{max_retries}] Anthropic API call failed: {e}. Retrying in {wait:.1f} seconds.")
            time.sleep(wait)

    raise RuntimeError("Exceeded max retries for Anthropic API call.")
