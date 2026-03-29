"""
reporting.py

Mode-agnostic post-debate analysis report generator.

Produces a structured Markdown report and/or JSON analysis from debate results.
Works with all stage3_mode options: rebuttal, collaborative, bloodsport.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_analysis_report(
    config,
    agents: list,
    challenge_rebuttal_pairs: list,
    agent_stats: dict,
    convergence_history: list = None,
    opening_positions: list = None,
    roadmap: dict = None,
    roadmap_revisions: list = None,
) -> str:
    """
    Generate a Markdown analysis report for a completed debate.

    Args:
        config: DebateConfig instance.
        agents: List of Agent instances (post-debate, with final beliefs).
        challenge_rebuttal_pairs: All challenge-rebuttal exchange records.
        agent_stats: Final agent statistics dict.
        convergence_history: Optional list of per-round convergence snapshots.
        opening_positions: Optional list of initial belief strings.
        roadmap: Optional roadmap dict with sub_topics, overall_rationale, sufficiency_note.
        roadmap_revisions: Optional list of revision dicts with round_num,
            revision_rationale, new_subtopics.

    Returns:
        str: Complete Markdown report.
    """
    sections: List[str] = []
    mode = config.stage3_mode if config else "rebuttal"
    stage2_mode = config.stage2_mode if config and hasattr(config, 'stage2_mode') else "open"
    topic = config.topic if config else "Unknown"

    # === Header ===
    sections.append("# CHAL Debate Analysis Report\n")
    sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # === 1. Metadata ===
    sections.append("## 1. Debate Metadata\n")
    sections.append(f"- **Topic**: {topic}")
    sections.append(f"- **Stage 2 Mode**: {stage2_mode}")
    sections.append(f"- **Stage 3 Mode**: {mode}")
    sections.append(f"- **Rounds**: {config.max_rounds if config else 'N/A'}")
    sections.append(f"- **Agents**: {len(agents)}")

    for agent in agents:
        model = getattr(agent, 'model', 'unknown')
        provider = getattr(agent, 'provider', 'unknown')
        persona = getattr(agent, 'persona_label', 'unknown')
        sections.append(f"  - **{agent.name}**: {model} ({provider}), persona: {persona}")

    if config:
        adj = config.adjudication
        sections.append(f"- **Adjudicator**: {adj.model} ({adj.provider})")
        sections.append(f"  - Logic weight: {adj.logic_weight}, Ethics weight: {adj.ethics_weight}")

    # Mode-specific metadata
    if mode == "bloodsport" and config:
        sections.append(f"- **Blood Sport Intensity**: {config.bloodsport.intensity}")
        sections.append(f"- **Max Exchanges**: {config.bloodsport.max_exchanges}")
    elif mode == "collaborative" and config:
        sections.append(f"- **Max Turns/Question**: {config.collaborative.max_turns_per_question}")
        sections.append(f"- **Early Termination**: {config.collaborative.early_termination_on_agreement}")

    sections.append("")

    # === Debate Roadmap (moderated mode only) ===
    if roadmap and stage2_mode == "moderated":
        sections.append("## 2. Debate Roadmap\n")
        sub_topics = roadmap.get("sub_topics", [])
        if roadmap.get("overall_rationale"):
            sections.append(f"**Overall Rationale:** {roadmap['overall_rationale']}\n")
        if roadmap.get("sufficiency_note"):
            sections.append(f"**Sufficiency Assessment:** {roadmap['sufficiency_note']}\n")

        sections.append("| Round | Sub-Topic | Description |")
        sections.append("|-------|-----------|-------------|")
        for i, st in enumerate(sub_topics, 1):
            title = st.get("title", "Untitled")
            desc = st.get("description", "")[:200]
            sections.append(f"| {i} | {title} | {desc} |")
        sections.append("")

        for i, st in enumerate(sub_topics, 1):
            sections.append(f"### Round {i}: {st.get('title', 'Untitled')}")
            sections.append(f"{st.get('description', '')}")
            if st.get("rationale"):
                sections.append(f"- **Rationale**: {st['rationale']}")
            guiding_qs = st.get("guiding_questions", [])
            if guiding_qs:
                sections.append("- **Guiding Questions**:")
                for gq in guiding_qs:
                    sections.append(f"  - {gq}")
            sections.append("")

        # Roadmap revisions sub-section (adaptive moderator)
        moderator_mode = "static"
        if config and hasattr(config, "moderator"):
            moderator_mode = getattr(config.moderator, "moderator_mode", "static")

        if roadmap_revisions and moderator_mode == "adaptive":
            sections.append("### Roadmap Revisions\n")
            for rev in roadmap_revisions:
                rnd = rev.get("round_num", "?")
                rationale = rev.get("revision_rationale", "")
                new_sts = rev.get("new_subtopics", [])
                sections.append(f"**After round {rnd}:** {rationale}\n")
                if new_sts:
                    for st in new_sts:
                        sections.append(f"- **{st.get('title', 'Untitled')}**: {st.get('description', '')}")
                sections.append("")

    # === Verdict Distribution ===
    # Section number shifts if roadmap section is present
    s = 3 if (roadmap and stage2_mode == "moderated") else 2
    sections.append(f"## {s}. Adjudicator Verdict Distribution\n")
    verdict_counts = {"critique_valid": 0, "rebuttal_valid": 0, "unresolved": 0}
    for entry in challenge_rebuttal_pairs:
        resolution = entry.get("resolution", {})
        status = resolution.get("status", "unknown") if isinstance(resolution, dict) else "unknown"
        if status in verdict_counts:
            verdict_counts[status] += 1

    total_verdicts = sum(verdict_counts.values())
    sections.append(f"| Verdict | Count | Percentage |")
    sections.append(f"|---------|-------|------------|")
    for verdict, count in verdict_counts.items():
        pct = (count / total_verdicts * 100) if total_verdicts > 0 else 0
        sections.append(f"| {verdict} | {count} | {pct:.1f}% |")
    sections.append(f"| **Total** | **{total_verdicts}** | **100%** |")
    sections.append("")

    # === Adjudicator Reasoning ===
    sections.append(f"## {s+1}. Adjudication Details\n")
    for i, entry in enumerate(challenge_rebuttal_pairs, 1):
        challenger = entry.get("challenger", "?")
        target = entry.get("target", "?")
        qid = entry.get("qid", f"Q{i}")
        resolution = entry.get("resolution", {})
        status = resolution.get("status", "unknown") if isinstance(resolution, dict) else "unknown"
        reasoning = resolution.get("reasoning", "N/A") if isinstance(resolution, dict) else "N/A"

        sections.append(f"### {qid}: {challenger} → {target}")
        sections.append(f"- **Verdict**: {status.upper()}")
        sections.append(f"- **Challenge**: {entry.get('challenge', 'N/A')[:500]}")
        sections.append(f"- **Rebuttal**: {str(entry.get('rebuttal', 'N/A'))[:500]}")
        sections.append(f"- **Reasoning**: {reasoning}")
        sections.append("")

    # === Agent Performance ===
    sections.append(f"## {s+2}. Agent Performance Summary\n")
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1].get('performance_score', 0), reverse=True)

    sections.append("| Agent | Score | Critiques Won | Rebuttals Won | Failed | Unresolved |")
    sections.append("|-------|-------|---------------|---------------|--------|------------|")
    for agent_name, stats in sorted_agents:
        sections.append(
            f"| {agent_name} | {stats.get('performance_score', 0):.2f} "
            f"| {stats.get('successful_critiques', 0)} "
            f"| {stats.get('successful_rebuttals', 0)} "
            f"| {stats.get('failed_rebuttals', 0)} "
            f"| {stats.get('unresolved_arguments', 0)} |"
        )
    sections.append("")

    # Mode-specific performance details
    if mode == "bloodsport":
        bs_agents = [(n, st) for n, st in sorted_agents if st.get('bloodsport_exchanges', 0) > 0]
        if bs_agents:
            sections.append("### Blood Sport Statistics\n")
            sections.append("| Agent | Exchanges | Turns |")
            sections.append("|-------|-----------|-------|")
            for agent_name, stats in bs_agents:
                sections.append(
                    f"| {agent_name} "
                    f"| {stats.get('bloodsport_exchanges', 0)} "
                    f"| {stats.get('bloodsport_turns', 0)} |"
                )
            sections.append("")

    # === Belief Evolution ===
    sections.append(f"## {s+3}. Belief Evolution Summary\n")
    for agent in agents:
        sections.append(f"### {agent.name}\n")
        all_beliefs = getattr(agent, 'all_beliefs_held', [])
        sections.append(f"- **Belief snapshots**: {len(all_beliefs)}")

        # Compare initial vs final belief if we have structured data
        initial_obj = None
        final_obj = agent.get_internal_belief_obj() if hasattr(agent, 'get_internal_belief_obj') else None

        if all_beliefs and len(all_beliefs) > 0:
            try:
                initial_obj = json.loads(all_beliefs[0])
            except (json.JSONDecodeError, TypeError):
                pass

        if initial_obj and final_obj:
            # Thesis comparison
            initial_thesis = initial_obj.get("thesis", {}).get("statement", "N/A")
            final_thesis = final_obj.get("thesis", {}).get("statement", "N/A")
            thesis_changed = initial_thesis != final_thesis
            sections.append(f"- **Thesis changed**: {'Yes' if thesis_changed else 'No'}")

            # Claim count comparison
            initial_claims = len(initial_obj.get("claims", []))
            final_claims = len(final_obj.get("claims", []))
            sections.append(f"- **Claims**: {initial_claims} → {final_claims}")

            # Strength drift
            initial_strengths = {
                c.get("id", ""): c.get("strength", 0)
                for c in initial_obj.get("claims", [])
            }
            final_strengths = {
                c.get("id", ""): c.get("strength", 0)
                for c in final_obj.get("claims", [])
            }
            shared_ids = set(initial_strengths.keys()) & set(final_strengths.keys())
            if shared_ids:
                drifts = [final_strengths[cid] - initial_strengths[cid] for cid in shared_ids]
                avg_drift = sum(drifts) / len(drifts)
                sections.append(f"- **Avg strength drift**: {avg_drift:+.3f} (across {len(shared_ids)} shared claims)")
        else:
            sections.append("- *(Structured belief comparison unavailable)*")
        sections.append("")

    # === Convergence ===
    if convergence_history:
        sections.append(f"## {s+4}. Convergence Trajectory\n")
        sections.append("| Round | Score | Shared Pairs | Unique Claims |")
        sections.append("|-------|-------|--------------|---------------|")
        for entry in convergence_history:
            sections.append(
                f"| {entry.get('round', '?')} "
                f"| {entry.get('convergence_score', 0):.3f} "
                f"| {entry.get('shared_claim_pairs', 0)} "
                f"| {entry.get('unique_claims_count', 0)} |"
            )
        sections.append("")

    return "\n".join(sections)


def generate_analysis_json(
    config,
    agents: list,
    challenge_rebuttal_pairs: list,
    agent_stats: dict,
    convergence_history: list = None,
    roadmap: dict = None,
    roadmap_revisions: list = None,
) -> dict:
    """
    Generate a structured JSON analysis report.

    Args:
        config: DebateConfig instance.
        agents: List of Agent instances.
        challenge_rebuttal_pairs: All challenge-rebuttal exchange records.
        agent_stats: Final agent statistics dict.
        convergence_history: Optional list of per-round convergence snapshots.
        roadmap: Optional roadmap dict with sub_topics, overall_rationale, sufficiency_note.
        roadmap_revisions: Optional list of revision dicts with round_num,
            revision_rationale, new_subtopics.

    Returns:
        dict: Machine-readable analysis data.
    """
    mode = config.stage3_mode if config else "rebuttal"
    stage2_mode = config.stage2_mode if config and hasattr(config, 'stage2_mode') else "open"

    # Verdict distribution
    verdict_counts = {"critique_valid": 0, "rebuttal_valid": 0, "unresolved": 0}
    for entry in challenge_rebuttal_pairs:
        resolution = entry.get("resolution", {})
        status = resolution.get("status", "unknown") if isinstance(resolution, dict) else "unknown"
        if status in verdict_counts:
            verdict_counts[status] += 1

    # Agent summaries
    agent_summaries = {}
    for agent in agents:
        name = agent.name
        stats = agent_stats.get(name, {})
        final_belief = agent.get_internal_belief_obj() if hasattr(agent, 'get_internal_belief_obj') else None

        summary = {
            "performance_score": stats.get("performance_score", 0),
            "successful_critiques": stats.get("successful_critiques", 0),
            "successful_rebuttals": stats.get("successful_rebuttals", 0),
            "failed_rebuttals": stats.get("failed_rebuttals", 0),
            "unresolved_arguments": stats.get("unresolved_arguments", 0),
            "total_arguments": stats.get("total_arguments", 0),
            "final_claim_count": len(final_belief.get("claims", [])) if final_belief else None,
        }

        if mode == "bloodsport":
            summary["bloodsport_exchanges"] = stats.get("bloodsport_exchanges", 0)
            summary["bloodsport_turns"] = stats.get("bloodsport_turns", 0)

        agent_summaries[name] = summary

    # Exchange details
    exchanges = []
    for entry in challenge_rebuttal_pairs:
        resolution = entry.get("resolution", {})
        exchange = {
            "challenger": entry.get("challenger"),
            "target": entry.get("target"),
            "qid": entry.get("qid"),
            "verdict": resolution.get("status") if isinstance(resolution, dict) else None,
            "reasoning": resolution.get("reasoning") if isinstance(resolution, dict) else None,
        }
        exchanges.append(exchange)

    report = {
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "topic": config.topic if config else "Unknown",
            "stage2_mode": stage2_mode,
            "stage3_mode": mode,
            "max_rounds": config.max_rounds if config else None,
            "num_agents": len(agents),
        },
        "verdict_distribution": verdict_counts,
        "agent_summaries": agent_summaries,
        "exchanges": exchanges,
    }

    if roadmap and stage2_mode == "moderated":
        report["roadmap"] = roadmap

    if roadmap_revisions:
        report["roadmap_revisions"] = roadmap_revisions

    if convergence_history:
        report["convergence_history"] = convergence_history

    if mode == "bloodsport" and config:
        report["metadata"]["bloodsport_intensity"] = config.bloodsport.intensity
        report["metadata"]["bloodsport_max_exchanges"] = config.bloodsport.max_exchanges

    return report
