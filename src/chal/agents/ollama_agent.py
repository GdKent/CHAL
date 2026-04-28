"""
ollama_agent.py

Defines an LLM-powered agent that uses the native Ollama Python library
to run inference against locally-hosted models (e.g., deepseek-r1:14b, phi4).
This agent implements the abstract Agent interface defined in base.py.

Usage:
- Requires Ollama to be installed and running locally (ollama serve).
- Requires the target model to be pulled first (ollama pull <model>).
- No API key is needed -- all inference runs on localhost, free of charge.
"""

from __future__ import annotations

import time

import ollama

from chal.agents.base import Agent, Message
from chal.log import logger


class OllamaAgent(Agent):
    """
    An agent that runs inference against locally-hosted models via Ollama.

    Attributes:
        model (str): Ollama model tag (e.g., "deepseek-r1:14b", "phi4").
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
    """

    def __init__(self, model: str, name: str, api_key: str | None = None, system_prompt: str = ""):
        # api_key is accepted but ignored -- Ollama runs locally and requires no authentication
        super().__init__(name=name, model=model, system_prompt=system_prompt,
                         temperature=0.7)

    def generate(self, history: list[Message], temperature: float = 0.7) -> Message:
        """Send conversation history to the local Ollama server and return the reply."""
        messages = [{"role": m.role, "content": m.content} for m in history]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        try:
            response = retry_ollama_chat(self.model, messages, temperature)
            return Message(
                role="assistant",
                content=response.message.content,
            )

        except Exception as e:
            logger.error(f"[API Error] {self.name} ({self.model}): {e}")
            raise


def retry_ollama_chat(model: str, messages: list, temperature: float,
                      max_retries: int = 5, base_delay: float = 10.0):
    """
    Wrapper to call Ollama with retry logic for transient server errors.

    - 404 (model not pulled): fails immediately.
    - Connection refused (server not running): fails immediately.
    - 5xx transient errors: retried with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature}
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' not found locally. Run: ollama pull {model}"
                ) from e
            wait = base_delay * (2 ** attempt)
            logger.debug(f"[Retry {attempt+1}/{max_retries}] Ollama error: {e}. Retrying in {wait:.1f}s.")
            time.sleep(wait)
        except (ConnectionRefusedError, OSError) as e:
            raise RuntimeError(
                "Cannot connect to Ollama. Is the server running? "
                "Start it with: ollama serve"
            ) from e

    raise RuntimeError(f"Exceeded max retries for Ollama model '{model}'.")
