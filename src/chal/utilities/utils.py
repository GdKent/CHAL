"""
utils.py

Utility functions for parsing structured agent responses
used across different stages of the CHAL debate controller.
"""

import re
from typing import List


def parse_challenges(challenge_text: str) -> list[str]:
    """
    Parses a multi-part critique (e.g., '1. ... 2. ... 3. ...') into a list of individual critiques.

    Assumes challenges are numbered using a pattern like:
        1. [text]
        2. [text]
        ...

    Args:
        challenge_text (str): The raw text from the agent's output.

    Returns:
        list[str]: A list of individual critique strings.
    """
    # Use a regular expression to split the challenge_text into parts wherever a numbered item appears.
    # It looks for an optional newline, followed by optional spaces, followed by a digit and a period (e.g., "1. ", "2. ")
    parts = re.split(r'\n?\s*\d\.\s+', challenge_text.strip())

    # Remove any empty strings from the resulting list and trim whitespace from each part
    # Ensures that each item is a clean, non-empty critique
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def parse_structured_rebuttals_numbered(text: str) -> list[str]:
    """
    Parses numbered rebuttal format with flexible spacing:
        Critique 1:
        ...
        Response 1:
        <text>
        ...
        Response N:
        <text>

    Returns:
        list[str]: All rebuttal texts in order.
    """
    pattern = r'Response\s+(\d+):\s*\n(.*?)(?=\n\s*Critique\s+\d+:|\Z)'
    matches = re.findall(pattern, text.strip(), re.DOTALL)

    # Sort by number just in case and return the text portion only
    return [resp.strip() for _, resp in sorted(matches, key=lambda x: int(x[0]))]


def initialize_agent_stats(agent_names: list[str]) -> dict:
    """
    Initializes a statistics dictionary for each agent.

    Args:
        agent_names (list[str]): List of agent identifiers (e.g., ['Agent-A', 'Agent-B']).

    Returns:
        dict: Dictionary mapping each agent to their stat counters.
    """
    return {
        name: {
            'successful_critiques': 0,      # Critiques that won (critique_valid)
            'failed_critiques': 0,          # Critiques that lost (rebuttal_valid)
            'successful_rebuttals': 0,      # Rebuttals that won (rebuttal_valid)
            'failed_rebuttals': 0,          # Rebuttals that lost (critique_valid)
            'unresolved_arguments': 0,      # Arguments that were unresolved
            'total_arguments': 0,           # Total arguments (as challenger or target)
            'performance_score': 0.0        # Weighted performance score
        } for name in agent_names
    }


def update_agent_stats(agent_stats: dict, record: dict):
    """
    Updates agent statistics based on adjudication record.

    Args:
        agent_stats (dict): The agent statistics dictionary.
        record (dict): A resolution record with keys:
            - 'challenger': the agent who posed the critique,
            - 'target': the agent who responded,
            - 'resolution': one of ['critique_valid', 'rebuttal_valid', 'unresolved']
    """
    resolution = record.get('resolution')['status']
    challenger = record.get('challenger')
    target = record.get('target')

    # Update total argument counts
    agent_stats[challenger]['total_arguments'] += 1
    agent_stats[target]['total_arguments'] += 1

    if resolution == "critique_valid":
        # Challenger wins, target loses
        agent_stats[challenger]['successful_critiques'] += 1
        agent_stats[target]['failed_rebuttals'] += 1
    elif resolution == "rebuttal_valid":
        # Target wins, challenger loses
        agent_stats[target]['successful_rebuttals'] += 1
        agent_stats[challenger]['failed_critiques'] += 1
    elif resolution == "unresolved":
        # Both get unresolved
        agent_stats[challenger]['unresolved_arguments'] += 1
        agent_stats[target]['unresolved_arguments'] += 1

    return agent_stats


def calculate_performance_scores(agent_stats: dict, weights: dict = None) -> dict:
    """
    Calculates performance scores for all agents based on debate outcomes.

    Performance Score Formula:
        APS = (successful_critiques × W_crit)
            + (successful_rebuttals × W_reb)
            - (failed_rebuttals × W_fail)
            - (unresolved_arguments × W_unres)

    Args:
        agent_stats (dict): The agent statistics dictionary.
        weights (dict, optional): Custom weights for scoring. Defaults to standard weights.

    Returns:
        dict: Updated agent_stats with performance_score calculated.
    """
    # Default weights (can be overridden via config)
    default_weights = {
        'successful_critique': 3.0,      # Highest reward (breaking opponent's claim)
        'successful_rebuttal': 2.0,      # Moderate reward (defending own claim)
        'failed_rebuttal': -2.0,         # Moderate penalty (claim didn't survive)
        'unresolved_argument': -0.5      # Minor penalty (unclear outcome)
    }

    w = weights if weights else default_weights

    for agent_name, stats in agent_stats.items():
        score = (
            stats['successful_critiques'] * w['successful_critique'] +
            stats['successful_rebuttals'] * w['successful_rebuttal'] +
            stats['failed_rebuttals'] * w['failed_rebuttal'] +
            stats['unresolved_arguments'] * w['unresolved_argument']
        )
        stats['performance_score'] = round(score, 2)

    return agent_stats


def display_agent_stats(agent_stats: dict, show_performance_ranking: bool = True):
    """
    Pretty-prints the statistics for each agent.

    Args:
        agent_stats (dict): The statistics dictionary.
        show_performance_ranking (bool): If True, displays agents ranked by performance score.
    """
    print("\n=== Debate Statistics ===")

    # Sort agents by performance score for ranking display
    if show_performance_ranking:
        sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['performance_score'], reverse=True)
    else:
        sorted_agents = list(agent_stats.items())

    for rank, (agent, stats) in enumerate(sorted_agents, 1):
        # Show rank indicator
        rank_indicator = ""
        if show_performance_ranking:
            if rank == 1:
                rank_indicator = " [WINNING]"
            elif rank == 2:
                rank_indicator = " [2nd]"
            elif rank == 3:
                rank_indicator = " [3rd]"

        print(f"\nAgent: {agent}{rank_indicator}")
        print(f"  Performance Score: {stats['performance_score']}")
        print(f"  ------------------------------")
        print(f"  Successful Critiques: {stats['successful_critiques']}")
        print(f"  Failed Critiques: {stats['failed_critiques']}")
        print(f"  Successful Rebuttals: {stats['successful_rebuttals']}")
        print(f"  Failed Rebuttals: {stats['failed_rebuttals']}")
        print(f"  Unresolved Arguments: {stats['unresolved_arguments']}")
        print(f"  Total Arguments: {stats['total_arguments']}")

        # Show bloodsport-specific stats if present
        if stats.get('bloodsport_exchanges', 0) > 0:
            print(f"  --- Blood Sport ---")
            print(f"  Exchanges: {stats['bloodsport_exchanges']}")
            print(f"  Turns: {stats['bloodsport_turns']}")

    # Show performance gap if multiple agents
    if show_performance_ranking and len(sorted_agents) > 1:
        leader = sorted_agents[0]
        second = sorted_agents[1]
        gap = leader[1]['performance_score'] - second[1]['performance_score']
        print(f"\nPerformance Gap: {leader[0]} leads by {gap:+.2f} points")


def get_performance_summary(agent_stats: dict) -> str:
    """
    Generates a concise performance summary string for logging.

    Args:
        agent_stats (dict): The agent statistics dictionary.

    Returns:
        str: Formatted performance summary.
    """
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['performance_score'], reverse=True)

    summary_lines = ["AGENT PERFORMANCE SCORES:"]
    for agent_name, stats in sorted_agents:
        summary_lines.append(
            f"  {agent_name}: {stats['performance_score']:.2f} "
            f"({stats['successful_critiques']} critiques, "
            f"{stats['successful_rebuttals']} rebuttals, "
            f"{stats['failed_rebuttals']} failed, "
            f"{stats['unresolved_arguments']} unresolved)"
        )

    if len(sorted_agents) > 1:
        leader = sorted_agents[0]
        gap = leader[1]['performance_score'] - sorted_agents[1][1]['performance_score']
        summary_lines.append(f"\nCURRENT LEADER: {leader[0]} (+{gap:.2f})")

    return "\n".join(summary_lines)