"""
xai_agent.py

Defines an LLM-powered agent that uses xAI's Chat Completions API
(e.g., Grok-2, Grok-beta) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an xAI API key to be available in the environment as `XAI_API_KEY`.
- Can be instantiated with any supported xAI chat model (e.g., "grok-2").
- Obtain your API key at https://console.x.ai/
"""

import os
import httpx
from chal.agents.base import Agent, Message
from typing import List
import time
import json
from httpx import HTTPStatusError, TimeoutException, RequestError
from dotenv import load_dotenv
load_dotenv()


class XAIAgent(Agent):
    """
    An agent that interacts with xAI's Grok chat models.

    Attributes:
        model (str): xAI model name (e.g., "grok-2", "grok-beta").
        api_key (str): xAI API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
    """

    def __init__(self, model: str, name: str, api_key: str = None, system_prompt: str = ""):
        """
        Initializes the XAIAgent with model and optional prompt/key.

        Args:
            model (str): The name of the xAI model to use (e.g., "grok-2", "grok-beta").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
        """
        self.model = model
        self.name = name
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.system_prompt = system_prompt
        self.internal_belief = ""
        self.internal_belief_obj = None
        self.belief_graph = None
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = []

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
        """
        self.system_prompt = self.system_prompt + "\n\n" + prompt

    def generate(self, history: List[Message], temperature: float = 0.7) -> Message:
        """
        Constructs a prompt from conversation history, sends it to xAI,
        and returns the model's reply wrapped in a Message object.
        """

        messages = [{"role": m.role, "content": m.content} for m in history]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            data = retry_xai_chat_completion(httpx.post, payload, headers)

            raw_msg = data["choices"][0]["message"]
            return Message(
                role=raw_msg["role"],
                content=raw_msg["content"],
                metadata=data
            )

        except Exception as e:
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_xai_chat_completion(post_func, payload, headers, max_retries=5, base_delay=60.0):
    """
    Wrapper to retry xAI chat completions with exponential backoff.

    Args:
        post_func (callable): Function to call, e.g. httpx.post
        payload (dict): JSON payload for xAI API
        headers (dict): HTTP headers including API key
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds

    Returns:
        dict: JSON-decoded response from xAI
    """
    for attempt in range(max_retries):
        try:
            response = post_func(
                "https://api.x.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=300
            )
            response.raise_for_status()
            return response.json()

        except (HTTPStatusError, TimeoutException, RequestError) as e:
            wait = base_delay * (2 ** attempt)
            print(f"[Retry {attempt+1}/{max_retries}] API call failed: {e}. Retrying in {wait:.1f} seconds.")
            time.sleep(wait)

    raise RuntimeError("Exceeded max retries for xAI API call.")
