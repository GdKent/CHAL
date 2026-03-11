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

import time
import ollama
from chal.agents.base import Agent, Message
from typing import List


class OllamaAgent(Agent):
    """
    An agent that runs inference against locally-hosted models via Ollama.

    Attributes:
        model (str): Ollama model tag (e.g., "deepseek-r1:14b", "phi4").
        name (str): Display name for the agent, e.g., "Agent-Empiricist".
        system_prompt (str): Optional instruction that defines the agent's persona or behavior.
    """

    def __init__(self, model: str, name: str, api_key: str = None, system_prompt: str = ""):
        self.model = model
        self.name = name
        # api_key is ignored -- Ollama runs locally and requires no authentication
        self.system_prompt = system_prompt
        self.internal_belief = ""
        self.internal_belief_obj = None
        self.belief_graph = None
        self.persona_label = name.split("Agent-", 1)[-1] if "Agent-" in name else name
        self.all_beliefs_held = []

    def set_internal_belief(self, belief_text: str) -> None:
        self.internal_belief = belief_text

    def get_internal_belief(self) -> str:
        return self.internal_belief

    def set_internal_belief_obj(self, belief_obj: dict | None) -> None:
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
        return self.internal_belief_obj

    def get_belief_graph(self):
        return self.belief_graph

    def receive_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def receive_role_card(self, prompt: str) -> None:
        self.system_prompt = self.system_prompt + "\n\n" + prompt

    def generate(self, history: List[Message], temperature: float = 0.7) -> Message:
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
            return Message(
                role="assistant",
                content=f"[Error from {self.name}]: {str(e)}"
            )


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
            print(f"[Retry {attempt+1}/{max_retries}] Ollama error: {e}. Retrying in {wait:.1f}s.")
            time.sleep(wait)
        except (ConnectionRefusedError, OSError) as e:
            raise RuntimeError(
                "Cannot connect to Ollama. Is the server running? "
                "Start it with: ollama serve"
            ) from e

    raise RuntimeError(f"Exceeded max retries for Ollama model '{model}'.")
