"""
perplexity_agent.py

Defines an LLM-powered agent that uses Perplexity's Chat Completions API
(e.g., sonar-pro, sonar-reasoning-pro) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires a Perplexity API key to be available in the environment as `PERPLEXITY_API_KEY`.
- Can be instantiated with any supported Perplexity chat model (e.g., "sonar-pro").
- Obtain your API key at https://www.perplexity.ai/settings/api

SDK: perplexityai (v0.30+). Import path: `from perplexity import Perplexity`.
The SDK uses httpx and follows the OpenAI client pattern.
"""

import os
import time
import perplexity as perplexity_module
from perplexity import Perplexity
from chal.agents.base import Agent, Message
from typing import List
from dotenv import load_dotenv
load_dotenv()


class PerplexityAgent(Agent):
    """
    An agent that interacts with Perplexity's chat models.

    Attributes:
        model (str): Perplexity model name (e.g., "sonar-pro", "sonar-reasoning").
        api_key (str): Perplexity API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
    """

    def __init__(self, model: str, name: str, api_key: str = None,
                 system_prompt: str = "", key_pool=None):
        """
        Initializes the PerplexityAgent with model and optional prompt/key.

        Args:
            model (str): The name of the Perplexity model to use (e.g., "sonar-pro", "sonar-reasoning").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
            key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
        """
        self.model = model
        self.name = name
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.system_prompt = system_prompt
        self.internal_belief = ""
        self.internal_belief_obj = None
        self.belief_graph = None
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = []
        self.key_pool = key_pool
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
        Constructs a prompt from conversation history, sends it to Perplexity,
        and returns the model's reply wrapped in a Message object.
        """
        messages = [
            {"role": m.role, "content": m.content}
            for m in history
            if m.role in ("user", "assistant")
        ]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Refresh key from pool (picks a non-cooling-down key)
        if self.key_pool is not None:
            fresh_key = self.key_pool.get_key("perplexity")
            if fresh_key != self.api_key:
                self.api_key = fresh_key
                self._client = None  # Force client rebuild with new key

        # Lazy-init client on first use (avoids requiring API key at construction time)
        if self._client is None:
            self._client = Perplexity(api_key=self.api_key)

        try:
            response = retry_perplexity_chat_completion(
                client=self._client,
                model=self.model,
                messages=messages,
                temperature=temperature,
                key_pool=self.key_pool,
                current_key=self.api_key,
                on_rate_limit=getattr(self, '_on_rate_limit', None),
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
                content=response.choices[0].message.content,
                metadata={"model": response.model, "usage": usage}
            )

        except Exception as e:
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


# --- Utility Function for Retry Calls to the API if Rate Limits are Exceeded ---
def retry_perplexity_chat_completion(client, model, messages, temperature,
                                      max_retries=5, base_delay=60.0,
                                      key_pool=None, current_key="",
                                      on_rate_limit=None):
    """
    Wrapper to retry Perplexity chat completions with exponential backoff.

    When a KeyPool is provided, rate-limit errors trigger key rotation:
    the current key is marked as cooling down, a fresh key is drawn from
    the pool, the client is rebuilt, and the request is retried immediately.

    Args:
        client: Instantiated perplexity.Perplexity client
        model (str): Perplexity model name
        messages (list): List of {"role": ..., "content": ...} dicts
        temperature (float): Sampling temperature
        max_retries (int): Max retry attempts
        base_delay (float): Delay factor in seconds
        key_pool: Optional KeyPool for rate-limit-aware key rotation.
        current_key (str): The API key currently in use.

    Returns:
        perplexity.types.StreamChunk: The Perplexity SDK response object
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

        except (perplexity_module.RateLimitError,
                perplexity_module.APIStatusError,
                perplexity_module.APIConnectionError) as e:
            # On rate limit: fire callback, then rotate key and retry
            if isinstance(e, perplexity_module.RateLimitError):
                if on_rate_limit:
                    on_rate_limit()
            if isinstance(e, perplexity_module.RateLimitError) and key_pool is not None:
                key_pool.mark_rate_limited("perplexity", current_key, cooldown_seconds=60)
                current_key = key_pool.get_key("perplexity")
                client = Perplexity(api_key=current_key)
                print(f"[KeyPool] Rotated perplexity API key after rate limit (attempt {attempt+1}/{max_retries}).")
                continue

            wait = base_delay * (2 ** attempt)
            print(f"[Retry {attempt+1}/{max_retries}] Perplexity API call failed: {e}. Retrying in {wait:.1f} seconds.")
            time.sleep(wait)

    raise RuntimeError("Exceeded max retries for Perplexity API call.")
