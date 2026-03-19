"""
xai_agent.py

Defines an LLM-powered agent that uses xAI's Chat Completions API
(e.g., Grok-3-mini, Grok-2) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an xAI API key to be available in the environment as `XAI_API_KEY`.
- Can be instantiated with any supported xAI chat model (e.g., "grok-3-mini").
- Obtain your API key at https://console.x.ai/

SDK: xai-sdk (v1+). Import path: `from xai_sdk import Client`.
The SDK uses gRPC to communicate with xAI's API.
"""

import os
import time
import grpc
from xai_sdk import Client
from xai_sdk.chat import user as xai_user, system as xai_system, assistant as xai_assistant
from chal.agents.base import Agent, Message
from typing import List
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
        self._client = None  # Lazy init: created on first generate() call

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
        messages = []
        if self.system_prompt:
            messages.append(xai_system(self.system_prompt))

        for m in history:
            if m.role == "user":
                messages.append(xai_user(m.content))
            elif m.role == "assistant":
                messages.append(xai_assistant(m.content))

        # Lazy-init client on first use (avoids requiring API key at construction time)
        if self._client is None:
            self._client = Client(api_key=self.api_key)

        try:
            response = retry_xai_chat_completion(
                client=self._client,
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return Message(
                role="assistant",
                content=response.content,
                metadata={"model": self.model, "usage": usage}
            )

        except Exception as e:
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_xai_chat_completion(client, model, messages, temperature,
                               max_retries=5, base_delay=60.0):
    """
    Wrapper to retry xAI chat completions with exponential backoff.

    Args:
        client: Instantiated xai_sdk.Client
        model (str): xAI model name
        messages (list): List of xai_sdk.chat message objects
        temperature (float): Sampling temperature
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds

    Returns:
        xai_sdk.chat.Response: The xAI SDK response object
    """
    for attempt in range(max_retries):
        try:
            chat = client.chat.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return chat.sample()

        except grpc.RpcError as e:
            code = e.code()
            if code in (grpc.StatusCode.RESOURCE_EXHAUSTED,
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED):
                wait = base_delay * (2 ** attempt)
                print(f"[Retry {attempt+1}/{max_retries}] xAI API call failed: {e.details()}. Retrying in {wait:.1f} seconds.")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Exceeded max retries for xAI API call.")
