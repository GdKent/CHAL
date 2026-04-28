"""Shared constants used across the CHAL package."""

from __future__ import annotations

PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}
