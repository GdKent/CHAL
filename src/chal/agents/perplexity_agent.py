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

from __future__ import annotations

import os

import perplexity as perplexity_module
from perplexity import Perplexity

from chal.agents.base import Agent, Message
from chal.log import logger
from chal.utilities.retry import retry_api_call


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
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7, key_pool=key_pool)
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._client = None  # Lazy init: created on first generate() call

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
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
            def _make_call(rotated_client):
                c = rotated_client if rotated_client is not None else self._client
                return c.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

            response = retry_api_call(
                call_fn=_make_call,
                provider="perplexity",
                rate_limit_errors=(perplexity_module.RateLimitError,),
                retryable_errors=(perplexity_module.InternalServerError, perplexity_module.APIConnectionError),
                key_pool=self.key_pool,
                current_key=self.api_key,
                rebuild_client_fn=lambda key: Perplexity(api_key=key),
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
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise
