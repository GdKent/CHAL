"""
factory.py

Agent factory: maps a provider string to the correct Agent subclass and
returns an instantiated agent. This is the single place in the codebase that
knows which provider name maps to which class.

Imports for each provider are deferred (inside the if-blocks) so that only
the SDK that is actually used needs to be installed. A project using only
OpenAI does not need `anthropic` or `google-genai` present.
"""

from chal.agents.base import Agent


def create_agent(name: str, model: str, provider: str = "openai",
                 system_prompt: str = "", key_pool=None) -> Agent:
    """
    Instantiate an agent for the given provider.

    Args:
        name (str): Display name for the agent, e.g. "Agent-Skeptic".
        model (str): Model identifier, e.g. "gpt-4o", "claude-sonnet-4-6".
        provider (str): One of "openai", "anthropic", "google", "ollama", "xai", "perplexity". Default "openai".
        system_prompt (str): Optional initial system prompt.
        key_pool: Optional KeyPool instance for multi-key rotation.
            When provided, the agent's initial API key is drawn from the pool
            and the pool is stored on the agent for rate-limit-aware rotation.

    Returns:
        Agent: A fully initialised agent instance.

    Raises:
        ValueError: If provider is not one of the supported values.
    """
    provider = provider.lower().strip()

    # Build common kwargs shared by all providers
    kwargs = {"model": model, "name": name, "system_prompt": system_prompt}

    # When a key pool is available and has keys for this provider,
    # draw the initial key from it and pass the pool for rotation.
    if key_pool is not None and key_pool.has_keys(provider):
        kwargs["api_key"] = key_pool.get_key(provider)
        kwargs["key_pool"] = key_pool

    if provider == "openai":
        from chal.agents.openai_agent import OpenAIAgent
        return OpenAIAgent(**kwargs)

    elif provider == "anthropic":
        from chal.agents.anthropic_agent import AnthropicAgent
        return AnthropicAgent(**kwargs)

    elif provider == "google":
        from chal.agents.google_agent import GoogleAgent
        return GoogleAgent(**kwargs)

    elif provider == "ollama":
        from chal.agents.ollama_agent import OllamaAgent
        # Ollama is local — no API key or key pool needed
        return OllamaAgent(model=model, name=name, system_prompt=system_prompt)

    elif provider == "xai":
        from chal.agents.xai_agent import XAIAgent
        return XAIAgent(**kwargs)

    elif provider == "perplexity":
        from chal.agents.perplexity_agent import PerplexityAgent
        return PerplexityAgent(**kwargs)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Must be one of: openai, anthropic, google, ollama, xai, perplexity"
        )


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
    )
