"""
utils.py

Utility functions for parsing structured agent responses
used across different stages of the CHAL debate controller.
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Stage 2 attack taxonomy constants
# ---------------------------------------------------------------------------

VALID_ATTACK_STRATEGIES = {
    "undermining": {
        "challenge_evidence",
        "challenge_assumption",
        "expose_weak_foundation",
        "demand_falsifiability",
        "challenge_strength_calibration",
        "press_uncertainty",
        "over_extension",           # D# definition is too broad — weakens the premise foundation
        "under_extension",          # D# definition is too narrow — premise doesn't cover key cases
    },
    "rebutting": {
        "present_counter_evidence",
        "present_counter_example",
        "exploit_counterposition",
        "offer_alternative_explanation",
    },
    "undercutting": {
        "challenge_inference_step",
        "identify_circularity",
        "expose_inconsistency",
        "identify_equivocation",
        "challenge_scope",
        "circularity",              # D# definition is circular — breaks the inference chain
        "stipulative_bias",         # D# definition begs the question — smuggles conclusion into premises
        "conceptual_conflation",    # D# definition conflates distinct concepts — equivocation
    },
}

VALID_TARGET_ID_PREFIXES = ("A", "C", "D", "E", "X", "U")

_TARGET_ID_RE = re.compile(r"^[ACDEXU]\d+$")


# Flat, sorted list of every permissible attack_strategy across all attack_types.
# Used by histogram helpers to pre-populate zeroed per-strategy counters.
ALL_STRATEGIES = sorted({
    strategy
    for strategies in VALID_ATTACK_STRATEGIES.values()
    for strategy in strategies
})

# Valid adjudicator verdict labels (tracked in agent_stats.adjudication_outcomes).
VALID_ADJUDICATION_VERDICTS = ("critique_valid", "rebuttal_valid", "unresolved")

# CBS component array names and which ones carry a `status` field whose value
# `"retracted"` excludes a node from the non-retracted count. X# (counterpositions)
# has no status field; U# (uncertainties) has status in {"active","resolved"} —
# we count all U# entries regardless (an uncertainty is never "retracted").
_COMPONENT_FIELDS = (
    "definitions",
    "assumptions",
    "claims",
    "evidence",
    "counterpositions",
    "uncertainties",
)
_STATUS_FILTERED_FIELDS = {"definitions", "assumptions", "claims", "evidence"}


def validate_stage2_questions(questions: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate parsed Stage 2 cross-examination questions.

    Checks each question for:
    - Required fields present and non-empty: qid, text, target_ids, attack_type, attack_strategy
    - qid matches pattern Q1, Q2, ...
    - target_ids is a non-empty list of 1-2 valid CBS node IDs (A#, C#, D#, E#, X#, U#)
    - attack_type is one of: undermining, rebutting, undercutting
    - attack_strategy is a valid strategy for the chosen attack_type

    Args:
        questions: List of parsed question dicts from model output.

    Returns:
        Tuple of (is_valid, errors) where is_valid is True if all questions pass,
        and errors is a list of human-readable error strings (empty if valid).
    """
    errors: list[str] = []

    if not questions:
        errors.append("questions list is empty")
        return False, errors

    for i, q in enumerate(questions):
        label = q.get("qid", f"question[{i}]")

        # --- required fields ---
        for field in ("qid", "text", "target_ids", "attack_type", "attack_strategy"):
            val = q.get(field)
            if not val:
                errors.append(f"{label}: missing or empty required field '{field}'")

        # --- qid format ---
        qid = q.get("qid", "")
        if qid and not re.fullmatch(r"Q\d+", qid):
            errors.append(f"{label}: qid '{qid}' does not match pattern Q1, Q2, ...")

        # --- target_ids ---
        target_ids = q.get("target_ids")
        if isinstance(target_ids, list):
            if len(target_ids) == 0:
                errors.append(f"{label}: target_ids is empty")
            elif len(target_ids) > 2:
                errors.append(f"{label}: target_ids has {len(target_ids)} entries (max 2)")
            for tid in target_ids:
                if not isinstance(tid, str) or not _TARGET_ID_RE.match(tid):
                    errors.append(
                        f"{label}: invalid target_ids entry '{tid}' "
                        f"(must match [ACDEXU]<number>)"
                    )

        # --- attack_type ---
        attack_type = q.get("attack_type", "")
        if attack_type and attack_type not in VALID_ATTACK_STRATEGIES:
            errors.append(
                f"{label}: invalid attack_type '{attack_type}' "
                f"(must be one of: {', '.join(sorted(VALID_ATTACK_STRATEGIES))})"
            )

        # --- attack_strategy ---
        attack_strategy = q.get("attack_strategy", "")
        if attack_type in VALID_ATTACK_STRATEGIES and attack_strategy:
            if attack_strategy not in VALID_ATTACK_STRATEGIES[attack_type]:
                errors.append(
                    f"{label}: attack_strategy '{attack_strategy}' is not valid "
                    f"for attack_type '{attack_type}' "
                    f"(valid: {', '.join(sorted(VALID_ATTACK_STRATEGIES[attack_type]))})"
                )

    return (len(errors) == 0), errors


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

    Includes the expanded tracking fields populated during the debate and at
    finalization:
      - ``initial_snapshot`` / ``final_snapshot``: thesis_strength + component
        counts (populated by the controller from snapshot_belief()).
      - ``per_round``: keyed by ``"round_N"``, same shape as snapshots.
      - ``cross_examination_attacks``: cumulative Stage 2 attack histogram
        authored BY this agent as challenger (filled by
        compute_attack_histograms()).
      - ``adjudication_outcomes``: Stage 4 verdict counts split by whether
        this agent was the challenger or the target.

    Args:
        agent_names (list[str]): List of agent identifiers (e.g., ['Agent-A', 'Agent-B']).

    Returns:
        dict: Dictionary mapping each agent to their stat counters.
    """
    def _empty_attack_histogram() -> dict:
        return {
            "total": 0,
            "by_type": {t: 0 for t in VALID_ATTACK_STRATEGIES},
            "by_strategy": {s: 0 for s in ALL_STRATEGIES},
        }

    def _empty_verdict_histogram() -> dict:
        return {v: 0 for v in VALID_ADJUDICATION_VERDICTS}

    return {
        name: {
            'successful_critiques': 0,      # Critiques that won (critique_valid)
            'failed_critiques': 0,          # Critiques that lost (rebuttal_valid)
            'successful_rebuttals': 0,      # Rebuttals that won (rebuttal_valid)
            'failed_rebuttals': 0,          # Rebuttals that lost (critique_valid)
            'unresolved_arguments': 0,      # Arguments that were unresolved
            'total_arguments': 0,           # Total arguments (as challenger or target)
            'performance_score': 0.0,       # Weighted performance score
            # Expanded tracking (populated by controller + finalize helpers)
            'initial_snapshot': None,       # Snapshot after Stage 1 (pre-round 1)
            'per_round': {},                # {"round_N": snapshot_dict}
            'final_snapshot': None,         # Snapshot after final round's Stage 5
            'cross_examination_attacks': _empty_attack_histogram(),
            'adjudication_outcomes': {
                'as_challenger': _empty_verdict_histogram(),
                'as_target': _empty_verdict_histogram(),
            },
        } for name in agent_names
    }


def update_agent_stats(agent_stats: dict, record: dict):
    """
    Updates agent statistics based on adjudication record.

    Also records the verdict in each participant's ``adjudication_outcomes``
    split (``as_challenger`` / ``as_target``). These role-split counters are a
    cumulative reflection of every Stage 4 verdict the agent participated in.

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

    # Role-split verdict histogram (only for recognised verdict labels).
    if resolution in VALID_ADJUDICATION_VERDICTS:
        agent_stats[challenger]['adjudication_outcomes']['as_challenger'][resolution] += 1
        agent_stats[target]['adjudication_outcomes']['as_target'][resolution] += 1

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
        # Skip the "_debate_aggregate" sentinel key (only added post-finalize).
        if agent_name.startswith("_") or not isinstance(stats, dict):
            continue
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

    # Filter out the "_debate_aggregate" sentinel before ranking/display.
    per_agent_items = [
        (name, stats) for name, stats in agent_stats.items()
        if not name.startswith("_") and isinstance(stats, dict)
    ]

    # Sort agents by performance score for ranking display
    if show_performance_ranking:
        sorted_agents = sorted(per_agent_items, key=lambda x: x[1]['performance_score'], reverse=True)
    else:
        sorted_agents = list(per_agent_items)

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
    # Filter out the "_debate_aggregate" sentinel inserted by finalize_agent_stats.
    per_agent_items = [
        (name, stats) for name, stats in agent_stats.items()
        if not name.startswith("_") and isinstance(stats, dict)
    ]
    sorted_agents = sorted(per_agent_items, key=lambda x: x[1]['performance_score'], reverse=True)

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


# ---------------------------------------------------------------------------
# Expanded agent_stats helpers (Phase 2 of the stats-expansion roadmap)
# ---------------------------------------------------------------------------

def compute_attack_histograms(
    agent_stats: dict,
    challenge_rebuttal_pairs: list,
    agent_names: list[str],
) -> dict:
    """Populate per-agent ``cross_examination_attacks`` and return the aggregate.

    Iterates every entry in ``challenge_rebuttal_pairs`` and bumps the
    challenger's ``total`` / ``by_type`` / ``by_strategy`` counters. Entries
    missing a recognised ``attack_type`` or ``attack_strategy`` still count
    toward ``total`` but are skipped for the categorical breakdowns so unknown
    labels never raise ``KeyError``.

    Args:
        agent_stats: Existing stats dict (mutated in place).
        challenge_rebuttal_pairs: Every Stage 2 entry recorded during the
            debate. Each entry must have at least a ``challenger`` key and
            optionally ``attack_type`` / ``attack_strategy``.
        agent_names: Ordered agent names; used to seed aggregation defaults.

    Returns:
        Aggregate dict with keys ``attacks_total``, ``attacks_by_type``,
        ``attacks_by_strategy`` (suitable for inclusion in the debate-wide
        ``_debate_aggregate`` block).
    """
    # Ensure every agent has a zeroed histogram (idempotent: initialize_agent_stats
    # already seeds this, but we defensively re-seed any missing keys).
    for name in agent_names:
        per_agent = agent_stats.get(name)
        if per_agent is None:
            continue
        hist = per_agent.setdefault("cross_examination_attacks", {
            "total": 0,
            "by_type": {t: 0 for t in VALID_ATTACK_STRATEGIES},
            "by_strategy": {s: 0 for s in ALL_STRATEGIES},
        })
        hist.setdefault("total", 0)
        hist.setdefault("by_type", {t: 0 for t in VALID_ATTACK_STRATEGIES})
        hist.setdefault("by_strategy", {s: 0 for s in ALL_STRATEGIES})

    aggregate = {
        "attacks_total": 0,
        "attacks_by_type": {t: 0 for t in VALID_ATTACK_STRATEGIES},
        "attacks_by_strategy": {s: 0 for s in ALL_STRATEGIES},
    }

    for pair in challenge_rebuttal_pairs:
        challenger = pair.get("challenger")
        if challenger is None or challenger not in agent_stats:
            continue
        hist = agent_stats[challenger]["cross_examination_attacks"]
        hist["total"] += 1
        aggregate["attacks_total"] += 1

        attack_type = pair.get("attack_type", "")
        if attack_type in hist["by_type"]:
            hist["by_type"][attack_type] += 1
        if attack_type in aggregate["attacks_by_type"]:
            aggregate["attacks_by_type"][attack_type] += 1

        attack_strategy = pair.get("attack_strategy", "")
        if attack_strategy in hist["by_strategy"]:
            hist["by_strategy"][attack_strategy] += 1
        if attack_strategy in aggregate["attacks_by_strategy"]:
            aggregate["attacks_by_strategy"][attack_strategy] += 1

    return aggregate


def snapshot_belief(belief_obj: dict) -> dict:
    """Extract a ``{thesis_strength, component_counts}`` snapshot from a CBS belief.

    ``thesis_strength`` is ``belief["thesis"]["strength"]`` if numeric, else
    ``None``. ``component_counts`` tallies each CBS component array using
    non-retracted semantics for D/A/C/E (``status`` ≠ ``"retracted"``), and
    raw lengths for X# (no status) / U# (never "retracted").

    Defensive: tolerates missing keys and non-dict beliefs by returning a
    null-valued snapshot, so the caller can preserve schema shape across
    degraded Stage 5 updates.
    """
    null_snapshot = {
        "thesis_strength": None,
        "component_counts": {field: 0 for field in _COMPONENT_FIELDS},
    }
    if not isinstance(belief_obj, dict):
        return null_snapshot

    thesis = belief_obj.get("thesis") or {}
    thesis_strength = thesis.get("strength") if isinstance(thesis, dict) else None
    if not isinstance(thesis_strength, (int, float)):
        thesis_strength = None

    component_counts = {}
    for field in _COMPONENT_FIELDS:
        items = belief_obj.get(field) or []
        if not isinstance(items, list):
            component_counts[field] = 0
            continue
        if field in _STATUS_FILTERED_FIELDS:
            component_counts[field] = sum(
                1 for item in items
                if isinstance(item, dict) and item.get("status") != "retracted"
            )
        else:
            # X# has no status; U# uses {"active","resolved"} — count all.
            component_counts[field] = len(items)

    return {
        "thesis_strength": thesis_strength,
        "component_counts": component_counts,
    }


def finalize_agent_stats(
    agent_stats: dict,
    challenge_rebuttal_pairs: list,
    agents,
    max_rounds: int,
) -> dict:
    """Assemble ``_debate_aggregate`` and derive each agent's ``final_snapshot``.

    Mutates ``agent_stats`` in place and returns it. Adds the top-level
    ``"_debate_aggregate"`` sentinel key which summarises debate-wide attack
    histograms + adjudicator verdict totals. Copies the most recent per-round
    snapshot into each agent's ``final_snapshot`` (falling back to a live
    recomputation from the agent's current belief if that round's snapshot is
    missing).

    Invariants enforced (cf. Phase 0.1 of the roadmap):
      - ``aggregate.attacks_by_type[t] == sum(agent.cross_examination_attacks.by_type[t])``
      - ``aggregate.adjudication_verdicts[v] == sum(agent.adjudication_outcomes.as_challenger[v])``

    Args:
        agent_stats: Per-agent stats dict.
        challenge_rebuttal_pairs: All Stage 2/4 entries.
        agents: Iterable of Agent objects (used for final_snapshot fallback).
        max_rounds: Number of completed rounds (used to resolve
            ``per_round[round_{max_rounds}]``).

    Returns:
        The mutated ``agent_stats`` dict with the new top-level
        ``"_debate_aggregate"`` key.
    """
    agent_names = [agent.name for agent in agents]

    # 1. Attack histograms (fills each agent's cross_examination_attacks + aggregate).
    attack_agg = compute_attack_histograms(agent_stats, challenge_rebuttal_pairs, agent_names)

    # 2. Verdict aggregate: sum over each agent's as_challenger counts (each pair
    #    has exactly one challenger, so this equals the total verdict count).
    verdict_aggregate = {v: 0 for v in VALID_ADJUDICATION_VERDICTS}
    for name in agent_names:
        per_agent = agent_stats.get(name)
        if not per_agent:
            continue
        as_chal = per_agent.get("adjudication_outcomes", {}).get("as_challenger", {})
        for verdict, count in as_chal.items():
            if verdict in verdict_aggregate:
                verdict_aggregate[verdict] += count

    # 3. Final snapshot per agent: prefer per_round[round_{max_rounds}], else
    #    live-recompute from agent.get_internal_belief_obj(). Degraded beliefs
    #    produce a null-valued snapshot rather than a missing key.
    final_round_key = f"round_{max_rounds}"
    agents_by_name = {a.name: a for a in agents}
    for name, stats in agent_stats.items():
        per_round = stats.get("per_round") or {}
        snapshot = per_round.get(final_round_key)
        if snapshot is None:
            agent = agents_by_name.get(name)
            belief = None
            if agent is not None and hasattr(agent, "get_internal_belief_obj"):
                try:
                    belief = agent.get_internal_belief_obj()
                except Exception:
                    belief = None
            snapshot = snapshot_belief(belief if isinstance(belief, dict) else {})
        stats["final_snapshot"] = snapshot

    agent_stats["_debate_aggregate"] = {
        "attacks_total": attack_agg["attacks_total"],
        "attacks_by_type": attack_agg["attacks_by_type"],
        "attacks_by_strategy": attack_agg["attacks_by_strategy"],
        "adjudication_verdicts": verdict_aggregate,
    }

    return agent_stats


def select_best_agent(agent_stats: dict, agent_order: list[str]) -> str:
    """Return the agent with the highest ``performance_score``.

    Ties are broken by first occurrence in ``agent_order`` (typically the
    ordering of ``config.agents``). The ``"_debate_aggregate"`` sentinel is
    filtered out automatically if present.

    Raises:
        ValueError: If no agents from ``agent_order`` appear in ``agent_stats``.
    """
    candidates = [n for n in agent_order if n in agent_stats and n != "_debate_aggregate"]
    if not candidates:
        raise ValueError("select_best_agent: no candidate agents found in agent_stats")
    # Sort-stable max: higher performance_score wins; ties prefer earlier
    # agent_order index (ascending index = preferred).
    return max(
        candidates,
        key=lambda n: (agent_stats[n].get("performance_score", 0.0),
                       -agent_order.index(n)),
    )