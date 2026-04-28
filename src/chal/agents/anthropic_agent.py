"""
anthropic_agent.py

Defines an LLM-powered agent that uses Anthropic's Messages API
(e.g., Claude Opus, Sonnet, Haiku) to generate responses. This agent implements
the abstract `Agent` interface defined in `base.py`.

Usage:
- Requires an Anthropic API key to be available in the environment as `ANTHROPIC_API_KEY`.
- Can be instantiated with any supported Anthropic chat model (e.g., "claude-sonnet-4-6").
"""

from __future__ import annotations

import os

import anthropic

from chal.agents.base import Agent, Message
from chal.log import logger
from chal.utilities.retry import retry_api_call


class AnthropicAgent(Agent):
    """
    An agent that interacts with Anthropic's Claude models.

    Attributes:
        model (str): Anthropic model name (e.g., "claude-sonnet-4-6").
        api_key (str): Anthropic API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Rationalist".
    """

    def __init__(self, model: str, name: str, api_key: str = None,
                 system_prompt: str = "", key_pool=None, max_tokens: int = 65536):
        """
        Initializes the AnthropicAgent with model and optional prompt/key.

        Args:
            model (str): The name of the Anthropic model to use (e.g., "claude-sonnet-4-6").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
            key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
            max_tokens (int): Maximum tokens for the Anthropic API response. Default 65536.
        """
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7, key_pool=key_pool)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
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

        # Refresh key from pool (picks a non-cooling-down key)
        if self.key_pool is not None:
            fresh_key = self.key_pool.get_key("anthropic")
            if fresh_key != self.api_key:
                self.api_key = fresh_key
                self._client = anthropic.Anthropic(api_key=fresh_key)

        try:
            def _make_call(rotated_client):
                c = rotated_client if rotated_client is not None else self._client
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                    "temperature": temperature,
                }
                if self.system_prompt:
                    kwargs["system"] = self.system_prompt
                return c.messages.create(**kwargs)

            response = retry_api_call(
                call_fn=_make_call,
                provider="anthropic",
                rate_limit_errors=(anthropic.RateLimitError,),
                retryable_errors=(anthropic.InternalServerError, anthropic.APIConnectionError),
                key_pool=self.key_pool,
                current_key=self.api_key,
                rebuild_client_fn=lambda key: anthropic.Anthropic(api_key=key),
                on_rate_limit=getattr(self, '_on_rate_limit', None),
            )

            return Message(
                role="assistant",
                content=response.content[0].text,
                metadata={"model": response.model, "usage": dict(response.usage)}
            )

        except Exception as e:
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise
