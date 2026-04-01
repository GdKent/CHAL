"""
convergence module

Provides tools for analyzing belief convergence
and unified belief synthesis in multi-agent debates.
"""

from .convergence_metrics import (
    calculate_claim_agreement,
    format_convergence_summary,
    get_convergence_trajectory_summary
)

__all__ = [
    "calculate_claim_agreement",
    "format_convergence_summary",
    "get_convergence_trajectory_summary",
]
