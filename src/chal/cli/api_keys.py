"""
api_keys.py

API key validation and prompting for CHAL debates.

Checks that required API keys are present in the environment based on
the providers used in the debate configuration.  In interactive mode,
prompts the user for missing keys; in headless mode, only warns.
"""

from __future__ import annotations

import os
from typing import Dict, Set

import questionary
from dotenv import load_dotenv
from rich.console import Console

from chal.config import DebateConfig

# Provider → environment variable mapping
PROVIDER_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _collect_providers(config: DebateConfig) -> Set[str]:
    """Collect the set of providers used in a debate configuration."""
    providers: Set[str] = set()
    for agent in config.agents:
        providers.add(agent.provider)
    providers.add(config.adjudication.provider)
    if config.stage2_mode == "moderated":
        providers.add(config.moderator.provider)
    return providers


def check_api_keys(config: DebateConfig) -> Dict[str, bool]:
    """Check which required API keys are present in the environment.

    Loads the .env file first so keys defined there are visible.

    Returns:
        Dict mapping provider name to whether its key is set.
    """
    load_dotenv()
    providers = _collect_providers(config)
    result: Dict[str, bool] = {}
    for provider in providers:
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            result[provider] = bool(os.environ.get(env_var))
        else:
            # Unknown provider — skip validation
            result[provider] = True
    return result


def prompt_missing_keys(config: DebateConfig, console: Console) -> None:
    """Interactively prompt for missing API keys and set them in os.environ.

    Keys are set for the current process only (not persisted).
    """
    key_status = check_api_keys(config)
    for provider, present in key_status.items():
        if present:
            continue
        env_var = PROVIDER_ENV_VARS.get(provider, "")
        if not env_var:
            continue
        console.print(
            f"[yellow]![/yellow] {env_var} is not set."
        )
        answer = questionary.text(
            f"Enter your {provider.capitalize()} API key (or press Enter to skip):",
        ).ask()
        if answer is None:
            # Ctrl+C
            raise KeyboardInterrupt
        if answer.strip():
            os.environ[env_var] = answer.strip()
            console.print(f"  [green]>[/green] {env_var} set for this session.")
        else:
            console.print(
                f"  [dim]Skipped. The debate may fail if {provider} calls are needed.[/dim]"
            )


def warn_missing_keys(config: DebateConfig, console: Console) -> None:
    """Non-interactive: warn about missing API keys without prompting."""
    key_status = check_api_keys(config)
    for provider, present in key_status.items():
        if present:
            continue
        env_var = PROVIDER_ENV_VARS.get(provider, "")
        if not env_var:
            continue
        console.print(
            f"[yellow]Warning:[/yellow] {env_var} is not set. "
            f"API calls to {provider} may fail."
        )


def validate_api_keys(
    config: DebateConfig,
    console: Console,
    interactive: bool = True,
) -> None:
    """Validate API keys: prompt interactively or warn in headless mode.

    Args:
        config: Debate configuration to check providers from.
        console: Rich console for output.
        interactive: If True, prompt for missing keys. If False, just warn.
    """
    if interactive:
        prompt_missing_keys(config, console)
    else:
        warn_missing_keys(config, console)
