"""
google_agent.py

Defines an LLM-powered agent that uses Google's Generative AI API
(e.g., Gemini 1.5 Pro, Gemini 2.0 Flash) to generate responses. This agent
implements the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires a Google API key to be available in the environment as `GOOGLE_API_KEY`.
- Can be instantiated with any supported Gemini model (e.g., "gemini-2.0-flash").

SDK: google-genai (v1+). Import path: `from google import genai`.
System instruction is passed per-request inside `types.GenerateContentConfig`,
not at model construction time.
"""

import os
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from chal.agents.base import Agent, Message
from typing import List
from dotenv import load_dotenv
load_dotenv()


class GoogleAgent(Agent):
    """
    An agent that interacts with Google's Gemini models.

    Attributes:
        model (str): Gemini model name (e.g., "gemini-2.0-flash").
        api_key (str): Google API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
    """

    def __init__(self, model: str, name: str, api_key: str = None,
                 system_prompt: str = "", key_pool=None):
        """
        Initializes the GoogleAgent with model and optional prompt/key.

        Args:
            model (str): The name of the Gemini model to use (e.g., "gemini-2.0-flash").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
            key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
        """
        self.model = model
        self.name = name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.system_prompt = system_prompt
        self.internal_belief = ""
        self.internal_belief_obj = None
        self.belief_graph = None
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = []
        self.key_pool = key_pool
        self._client = genai.Client(api_key=self.api_key)

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
        Constructs a prompt from conversation history, sends it to Google Gemini,
        and returns the model's reply wrapped in a Message object.

        Note: The google-genai SDK uses "model" (not "assistant") as the role name
        for model responses. The history mapping converts "assistant" → "model".
        "system" messages in history are filtered out; the system prompt is passed
        per-request inside GenerateContentConfig.
        """
        # Build contents list: map "assistant" → "model", drop "system" messages
        contents = []
        for m in history:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else m.role
            contents.append(
                types.Content(role=role, parts=[types.Part(text=m.content)])
            )

        # Refresh key from pool (picks a non-cooling-down key)
        if self.key_pool is not None:
            fresh_key = self.key_pool.get_key("google")
            if fresh_key != self.api_key:
                self.api_key = fresh_key
                self._client = genai.Client(api_key=fresh_key)

        try:
            response = retry_google_generate(
                client=self._client,
                model=self.model,
                contents=contents,
                system_prompt=self.system_prompt,
                temperature=temperature,
                key_pool=self.key_pool,
                current_key=self.api_key,
            )

            return Message(
                role="assistant",
                content=response.text,
                metadata={"model": self.model}
            )

        except Exception as e:
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_google_generate(client, model, contents, system_prompt, temperature,
                           max_retries=5, base_delay=60.0,
                           key_pool=None, current_key=""):
    """
    Wrapper to retry Google Gemini content generation with exponential backoff.

    When a KeyPool is provided, rate-limit errors (HTTP 429) trigger key
    rotation: the current key is marked as cooling down, a fresh key is
    drawn from the pool, the client is rebuilt, and the request is retried
    immediately.

    Args:
        client: Instantiated genai.Client
        model (str): Gemini model name
        contents (list): List of types.Content objects
        system_prompt (str): System-level instruction (passed per-request in config)
        temperature (float): Sampling temperature
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds
        key_pool: Optional KeyPool for rate-limit-aware key rotation.
        current_key (str): The API key currently in use.

    Returns:
        genai response object with a .text attribute
    """
    config_kwargs = {"temperature": temperature}
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        except genai_errors.APIError as e:
            # Detect rate limit (HTTP 429) for key rotation
            is_rate_limit = getattr(e, 'code', None) == 429 or '429' in str(e)
            if is_rate_limit and key_pool is not None:
                key_pool.mark_rate_limited("google", current_key, cooldown_seconds=60)
                current_key = key_pool.get_key("google")
                client = genai.Client(api_key=current_key)
                print(f"[KeyPool] Rotated google API key after rate limit (attempt {attempt+1}/{max_retries}).")
                continue

            wait = base_delay * (2 ** attempt)
            print(f"[Retry {attempt+1}/{max_retries}] Google API call failed: {e}. Retrying in {wait:.1f} seconds.")
            time.sleep(wait)

    raise RuntimeError("Exceeded max retries for Google Gemini API call.")
