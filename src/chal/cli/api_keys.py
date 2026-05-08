"""
api_keys.py

API key validation and prompting for CHAL debates.

Checks that required API keys are present in the environment based on
the providers used in the debate configuration.  In interactive mode,
prompts the user for missing keys; in headless mode, only warns.
"""

from __future__ import annotations

import os

import questionary
from rich.console import Console

from chal.config import DebateConfig
from chal.constants import PROVIDER_ENV_VARS


def _collect_providers(config: DebateConfig) -> set[str]:
    """Collect the set of providers used in a debate configuration."""
    providers: set[str] = set()
    for agent in config.agents:
        providers.add(agent.provider)
    providers.add(config.adjudication.provider)
    return providers


def check_api_keys(config: DebateConfig) -> dict[str, bool]:
    """Check which required API keys are present in the environment.

    Returns:
        Dict mapping provider name to whether its key is set.
    """
    providers = _collect_providers(config)
    result: dict[str, bool] = {}
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

    Supports entering multiple keys per provider for rate-limit rotation.
    Keys are joined with commas and set for the current process only (not
    persisted).
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
        keys: list[str] = []
        key_num = 1
        while True:
            prompt = (
                f"Enter your {provider.capitalize()} API key (or press Enter to skip):"
                if key_num == 1
                else f"Enter your {provider.capitalize()} API key #{key_num} (or press Enter to finish):"
            )
            answer = questionary.text(prompt).ask()
            if answer is None:
                raise KeyboardInterrupt
            if not answer.strip():
                break
            keys.append(answer.strip())
            console.print("  [green]>[/green] Key set.")
            add_more = questionary.confirm(
                f"Add another {provider.capitalize()} key for rate-limit rotation?",
                default=False,
            ).ask()
            if add_more is None:
                raise KeyboardInterrupt
            if not add_more:
                break
            key_num += 1

        if keys:
            os.environ[env_var] = ",".join(keys)
            count_note = f" ({len(keys)} keys for rotation)" if len(keys) > 1 else ""
            console.print(f"  [green]>[/green] {env_var} set for this session{count_note}.")
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


def create_key_pool(config: DebateConfig):
    """Create and populate a KeyPool from the environment.

    Loads .env first, then reads all provider keys (supports comma-separated
    lists for multi-key rotation under parallel mode).

    Args:
        config: Debate configuration (used to identify required providers).

    Returns:
        A populated KeyPool instance.
    """
    from chal.utilities.key_pool import KeyPool
    pool = KeyPool()
    pool.load_from_env()
    return pool


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
