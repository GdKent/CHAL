"""
convergence module

Provides tools for analyzing belief convergence, graph-based vulnerability detection,
and unified belief synthesis in multi-agent debates.
"""

from .graph_analysis import analyze_vulnerabilities, format_attack_suggestions
from .convergence_metrics import (
    calculate_claim_agreement,
    format_convergence_summary,
    get_convergence_trajectory_summary
)

__all__ = [
    "analyze_vulnerabilities",
    "format_attack_suggestions",
    "calculate_claim_agreement",
    "format_convergence_summary",
    "get_convergence_trajectory_summary",
]
