"""
openai_agent.py

Defines an LLM-powered agent that uses OpenAI's Chat Completions API
(e.g., GPT-4o, GPT-3.5-turbo) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an OpenAI API key to be available in the environment as `OPENAI_API_KEY`.
- Can be instantiated with any supported OpenAI chat model (e.g., "gpt-4o").
"""

import os # For reading environment variables (like API keys)
import httpx # For making HTTP requests to OpenAI's API
from chal.agents.base import Agent, Message # Base class and message type
from typing import List # For type annotations
import time
import json
from httpx import HTTPStatusError, TimeoutException, RequestError
from dotenv import load_dotenv
load_dotenv() # Loads .env file from project root


class OpenAIAgent(Agent):
    """
    An agent that interacts with OpenAI's chat models.

    Attributes:
        model (str): OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo").
        api_key (str): OpenAI API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "OpenAI-gpt-4o".
    """

    def __init__(self, model: str, name: str, api_key: str = None, system_prompt: str = ""):
        """
        Initializes the OpenAIAgent with model and optional prompt/key.

        Args:
            model (str): The name of the OpenAI model to use (e.g., "gpt-4o", "gpt-3.5-turbo").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
        """
        self.model = model
        self.name = name  # Used for display/debugging
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")  # Pull from env if not passed
        self.system_prompt = system_prompt
        self.internal_belief = ""  # Tracks agent's evolving belief over time
        self.internal_belief_obj = None   # dict for json-structured belief
        self.belief_graph = None  # Persistent BeliefGraph object (derived from internal_belief_obj)
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = [] # A list of internal_belief objects that the agent has held. This is to help enable the agent to form a solid conclusion when the debat ends
    
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

        # Auto-rebuild persistent graph when belief changes
        if belief_obj:
            try:
                from chal.beliefs.belief_graph import BeliefGraph
                self.belief_graph = BeliefGraph(belief_obj)
            except Exception as e:
                # If graph construction fails, log but don't crash
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
        Constructs a prompt from conversation history, sends it to OpenAI,
        and returns the model's reply wrapped in a Message object.
        """

        # Convert internal Message objects to OpenAI's required format
        messages = [{"role": m.role, "content": m.content} for m in history]

        # Optionally insert a system message at the beginning
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Prepare the API request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # HTTP headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            # Use retry wrapper instead of direct call
            data = retry_openai_chat_completion(httpx.post, payload, headers)

            # Extract the model's reply
            raw_msg = data["choices"][0]["message"]
            return Message(
                role=raw_msg["role"],
                content=raw_msg["content"],
                metadata=data
            )

        except Exception as e:
            # Return a dummy Message if all retries fail
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )



# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_openai_chat_completion(post_func, payload, headers, max_retries=5, base_delay=60.0):
    """
    Wrapper to retry OpenAI chat completions with exponential backoff.

    Args:
        post_func (callable): Function to call, e.g. httpx.post
        payload (dict): JSON payload for OpenAI API
        headers (dict): HTTP headers including API key
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds

    Returns:
        dict: JSON-decoded response from OpenAI
    """
    for attempt in range(max_retries):
        try:
            response = post_func(
                "https://api.openai.com/v1/chat/completions",
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

    raise RuntimeError("Exceeded max retries for OpenAI API call.")