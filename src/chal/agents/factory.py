"""
factory.py

Agent factory: maps a provider string to the correct Agent subclass and
returns an instantiated agent. This is the single place in the codebase that
knows which provider name maps to which class.

A registry dict + importlib.import_module() keep imports lazy so that only
the SDK for the provider actually used needs to be installed. A project
using only OpenAI does not need `anthropic` or `google-genai` present.
"""

from __future__ import annotations

import importlib

from chal.agents.base import Agent

# ---------------------------------------------------------------------------
# Registry: provider name -> (module_path, class_name)
# ---------------------------------------------------------------------------
_AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("chal.agents.openai_agent", "OpenAIAgent"),
    "anthropic": ("chal.agents.anthropic_agent", "AnthropicAgent"),
    "google": ("chal.agents.google_agent", "GoogleAgent"),
    "xai": ("chal.agents.xai_agent", "XAIAgent"),
    "perplexity": ("chal.agents.perplexity_agent", "PerplexityAgent"),
    "ollama": ("chal.agents.ollama_agent", "OllamaAgent"),
}


def create_agent(name: str, model: str, provider: str = "openai",
                 system_prompt: str = "", key_pool=None,
                 max_tokens: int = 65536) -> Agent:
    """
    Instantiate an agent for the given provider.

    Args:
        name (str): Display name for the agent, e.g. "Agent-Skeptic".
        model (str): Model identifier, e.g. "gpt-4o", "claude-sonnet-4-6".
        provider (str): One of "openai", "anthropic", "google", "ollama",
            "xai", "perplexity". Default "openai".
        system_prompt (str): Optional initial system prompt.
        key_pool: Optional KeyPool instance for multi-key rotation.
            When provided, the agent's initial API key is drawn from the pool
            and the pool is stored on the agent for rate-limit-aware rotation.
        max_tokens (int): Maximum tokens for API responses (currently used by
            Anthropic only). Default 65536.

    Returns:
        Agent: A fully initialised agent instance.

    Raises:
        ValueError: If provider is not one of the supported values.
    """
    provider_lower = provider.lower().strip()

    # --- Validate provider against registry ---
    if provider_lower not in _AGENT_REGISTRY:
        available = ", ".join(sorted(_AGENT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider '{provider_lower}'. "
            f"Available: {available}"
        )

    # --- Lazily import the agent class ---
    module_path, class_name = _AGENT_REGISTRY[provider_lower]
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)

    # --- Build kwargs ---
    kwargs: dict = {"model": model, "name": name, "system_prompt": system_prompt}

    # When a key pool is available and has keys for this provider,
    # draw the initial key from it and pass the pool for rotation.
    # Ollama is local — no API key or key pool needed.
    if provider_lower != "ollama" and key_pool is not None and key_pool.has_keys(provider_lower):
        kwargs["api_key"] = key_pool.get_key(provider_lower)
        kwargs["key_pool"] = key_pool

    # Provider-specific kwargs
    if provider_lower in ("openai", "anthropic"):
        kwargs["max_tokens"] = max_tokens

    return agent_class(**kwargs)  # type: ignore[no-any-return]


def create_agent_from_config(agent_cfg, key_pool=None) -> Agent:
    """
    Convenience wrapper: instantiate an agent directly from an AgentConfig object.

    Args:
        agent_cfg (AgentConfig): Populated agent configuration dataclass.
        key_pool: Optional KeyPool instance for multi-key rotation.

    Returns:
        Agent: A fully initialised agent instance.
    """
    return create_agent(
        name=agent_cfg.name,
        model=agent_cfg.model,
        provider=agent_cfg.provider,
        key_pool=key_pool,
        max_tokens=agent_cfg.max_tokens,
    )
