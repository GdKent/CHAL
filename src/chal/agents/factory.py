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
                 system_prompt: str = "") -> Agent:
    """
    Instantiate an agent for the given provider.

    Args:
        name (str): Display name for the agent, e.g. "Agent-Skeptic".
        model (str): Model identifier, e.g. "gpt-4o", "claude-sonnet-4-6".
        provider (str): One of "openai", "anthropic", "google". Default "openai".
        system_prompt (str): Optional initial system prompt.

    Returns:
        Agent: A fully initialised agent instance.

    Raises:
        ValueError: If provider is not one of the supported values.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from chal.agents.openai_agent import OpenAIAgent
        return OpenAIAgent(model=model, name=name, system_prompt=system_prompt)

    elif provider == "anthropic":
        from chal.agents.anthropic_agent import AnthropicAgent
        return AnthropicAgent(model=model, name=name, system_prompt=system_prompt)

    elif provider == "google":
        from chal.agents.google_agent import GoogleAgent
        return GoogleAgent(model=model, name=name, system_prompt=system_prompt)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Must be one of: openai, anthropic, google"
        )


def create_agent_from_config(agent_cfg) -> Agent:
    """
    Convenience wrapper: instantiate an agent directly from an AgentConfig object.

    Args:
        agent_cfg (AgentConfig): Populated agent configuration dataclass.

    Returns:
        Agent: A fully initialised agent instance.
    """
    return create_agent(
        name=agent_cfg.name,
        model=agent_cfg.model,
        provider=agent_cfg.provider,
    )
