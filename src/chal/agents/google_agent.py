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

from __future__ import annotations

import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from chal.agents.base import Agent, Message
from chal.log import logger
from chal.utilities.retry import retry_api_call


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
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7, key_pool=key_pool)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
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
            # Google uses a single APIError for all errors; we detect rate limits
            # by checking the error code and re-raise as _GoogleRateLimitError
            # so retry_api_call can distinguish rate limits from other errors.
            def _make_call(rotated_client):
                c = rotated_client if rotated_client is not None else self._client
                config_kwargs = {"temperature": temperature}
                if self.system_prompt:
                    config_kwargs["system_instruction"] = self.system_prompt
                try:
                    return c.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                except genai_errors.APIError as e:
                    is_rate_limit = getattr(e, 'code', None) == 429 or '429' in str(e)
                    if is_rate_limit:
                        raise _GoogleRateLimitError(str(e)) from e
                    raise

            response = retry_api_call(
                call_fn=_make_call,
                provider="google",
                rate_limit_errors=(_GoogleRateLimitError,),
                retryable_errors=(genai_errors.APIError,),
                key_pool=self.key_pool,
                current_key=self.api_key,
                rebuild_client_fn=lambda key: genai.Client(api_key=key),
                on_rate_limit=getattr(self, '_on_rate_limit', None),
            )

            return Message(
                role="assistant",
                content=response.text,
                metadata={"model": self.model}
            )

        except Exception as e:
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise


class _GoogleRateLimitError(Exception):
    """Internal sentinel for Google 429 rate-limit errors."""
    pass
