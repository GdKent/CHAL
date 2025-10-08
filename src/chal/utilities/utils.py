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
            'successful_critiques': 0,
            'successful_rebuttals': 0,
            'unresolved_rebuttals': 0
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
            - 'resolution': one of ['challenge_correct', 'rebuttal_correct', 'unresolved']
    """
    resolution = record.get('resolution')['status']
    challenger = record.get('challenger')
    target = record.get('target')

    if resolution == "critique_valid":
        agent_stats[challenger]['successful_critiques'] += 1
    elif resolution == "rebuttal_valid":
        agent_stats[target]['successful_rebuttals'] += 1
    elif resolution == "unresolved":
        agent_stats[target]['unresolved_rebuttals'] += 1

    return agent_stats


def display_agent_stats(agent_stats: dict):
    """
    Pretty-prints the statistics for each agent.

    Args:
        agent_stats (dict): The statistics dictionary.
    """
    print("\n🧮 Debate Statistics:")
    for agent, stats in agent_stats.items():
        print(f"\nAgent: {agent}")
        for key, val in stats.items():
            label = key.replace('_', ' ').title()
            print(f"  {label}: {val}")