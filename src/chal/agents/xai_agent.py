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

from __future__ import annotations

import os

import grpc
from xai_sdk import Client
from xai_sdk.chat import assistant as xai_assistant
from xai_sdk.chat import system as xai_system
from xai_sdk.chat import user as xai_user

from chal.agents.base import Agent, Message
from chal.log import logger
from chal.utilities.retry import retry_api_call


class XAIAgent(Agent):
    """
    An agent that interacts with xAI's Grok chat models.

    Attributes:
        model (str): xAI model name (e.g., "grok-2", "grok-beta").
        api_key (str): xAI API key used for authentication.
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
    """

    def __init__(self, model: str, name: str, api_key: str | None = None,
                 system_prompt: str = "", key_pool=None):
        """
        Initializes the XAIAgent with model and optional prompt/key.

        Args:
            model (str): The name of the xAI model to use (e.g., "grok-2", "grok-beta").
            name (str): Display name for the agent, e.g., "Agent-Skeptic".
            api_key (str, optional): Explicit API key override (fallback is env var).
            system_prompt (str, optional): Optional system message to set agent behavior.
            key_pool: Optional KeyPool instance for rate-limit-aware key rotation.
        """
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7, key_pool=key_pool)
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self._client = None  # Lazy init: created on first generate() call

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
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

        # Refresh key from pool (picks a non-cooling-down key)
        if self.key_pool is not None:
            fresh_key = self.key_pool.get_key("xai")
            if fresh_key != self.api_key:
                self.api_key = fresh_key
                self._client = None  # Force client rebuild with new key

        # Lazy-init client on first use (avoids requiring API key at construction time)
        if self._client is None:
            self._client = Client(api_key=self.api_key)

        try:
            # xAI uses gRPC errors; we detect rate limits (RESOURCE_EXHAUSTED)
            # and re-raise as _XAIRateLimitError so retry_api_call can distinguish
            # them. UNAVAILABLE/DEADLINE_EXCEEDED are retryable; other codes
            # are re-raised immediately.
            def _make_call(rotated_client):
                c = rotated_client if rotated_client is not None else self._client
                try:
                    chat = c.chat.create(  # type: ignore[union-attr]
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                    )
                    return chat.sample()
                except grpc.RpcError as e:
                    code = e.code()
                    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        raise _XAIRateLimitError(e.details()) from e
                    if code in (grpc.StatusCode.UNAVAILABLE,
                                grpc.StatusCode.DEADLINE_EXCEEDED):
                        raise _XAIRetryableError(e.details()) from e
                    raise

            response = retry_api_call(
                call_fn=_make_call,
                provider="xai",
                rate_limit_errors=(_XAIRateLimitError,),
                retryable_errors=(_XAIRetryableError,),
                key_pool=self.key_pool,
                current_key=self.api_key,  # type: ignore[arg-type]
                rebuild_client_fn=lambda key: Client(api_key=key),
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
                content=response.content,  # type: ignore[arg-type]
                metadata={"model": self.model, "usage": usage}
            )

        except Exception as e:
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise


class _XAIRateLimitError(Exception):
    """Internal sentinel for xAI RESOURCE_EXHAUSTED (rate limit) errors."""
    pass


class _XAIRetryableError(Exception):
    """Internal sentinel for xAI retryable gRPC errors (UNAVAILABLE, DEADLINE_EXCEEDED)."""
    pass
