"""
openai_agent.py

Defines an LLM-powered agent that uses OpenAI's Chat Completions API
(e.g., GPT-4o, o4-mini) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an OpenAI API key to be available in the environment as `OPENAI_API_KEY`.
- Can be instantiated with any supported OpenAI chat model (e.g., "gpt-4o").

SDK: openai (v1+). Import path: `from openai import OpenAI`.
The SDK automatically handles model-specific requirements such as reasoning
models that don't support temperature or use 'developer' instead of 'system' role.
"""

from __future__ import annotations

import os

import openai
from openai import OpenAI

from chal.agents.base import Agent, Message
from chal.log import logger
from chal.utilities.retry import retry_api_call


class OpenAIAgent(Agent):
    """
    An agent that interacts with OpenAI's chat models.

    Attributes:
        model (str): OpenAI model name (e.g., "gpt-4o", "o4-mini").
        api_key (str): OpenAI API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
    """

    def __init__(self, model: str, name: str, api_key: str = None,
                 system_prompt: str = "", key_pool=None,
                 max_tokens: int = 65536):
        """
        Initializes the OpenAIAgent with model and optional prompt/key.

        Args:
            model (str): The name of the OpenAI model to use (e.g., "gpt-4o", "o4-mini").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
            key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
            max_tokens (int): Maximum tokens for the API response. Default 65536.
        """
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7, key_pool=key_pool)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self._client = None  # Lazy init: created on first generate() call

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
        """
        Constructs a prompt from conversation history, sends it to OpenAI,
        and returns the model's reply wrapped in a Message object.

        Note: The SDK handles model-specific requirements automatically.
        For reasoning models (o-series), temperature is omitted and the
        system role is mapped to 'developer' by the SDK as needed.
        """
        # Filter out system messages from history — system prompt is passed separately
        messages = [
            {"role": m.role, "content": m.content}
            for m in history
            if m.role in ("user", "assistant")
        ]

        # Insert system prompt at the beginning
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Refresh key from pool (picks a non-cooling-down key)
        if self.key_pool is not None:
            fresh_key = self.key_pool.get_key("openai")
            if fresh_key != self.api_key:
                self.api_key = fresh_key
                self._client = None  # Force client rebuild with new key

        # Lazy-init client on first use (avoids requiring API key at construction time)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        # Reasoning models (o-series) don't support the temperature parameter
        is_reasoning_model = self.model.startswith(("o1", "o3", "o4"))

        try:
            def _make_call(rotated_client):
                c = rotated_client if rotated_client is not None else self._client
                kwargs = {"model": self.model, "messages": messages}
                if is_reasoning_model:
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["temperature"] = temperature
                    kwargs["max_tokens"] = self.max_tokens
                return c.chat.completions.create(**kwargs)

            response = retry_api_call(
                call_fn=_make_call,
                provider="openai",
                rate_limit_errors=(openai.RateLimitError,),
                retryable_errors=(openai.InternalServerError, openai.APIConnectionError),
                key_pool=self.key_pool,
                current_key=self.api_key,
                rebuild_client_fn=lambda key: OpenAI(api_key=key),
                on_rate_limit=getattr(self, '_on_rate_limit', None),
            )

            return Message(
                role="assistant",
                content=response.choices[0].message.content,
                metadata={"model": response.model, "usage": dict(response.usage)}
            )

        except Exception as e:
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise
