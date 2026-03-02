"""
roadmap_review.py

Interactive roadmap review for moderated debates.

When stage2_mode == "moderated", the moderator generates a roadmap of
sub-topics before the debate begins.  This module displays the roadmap
and lets the user approve, edit, reorder, or regenerate it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import questionary
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chal.orchestrator.moderator import Moderator, SubTopic


def show_roadmap(
    console: Console,
    subtopics: List[SubTopic],
    sufficiency: str = "",
    overall_rationale: str = "",
) -> None:
    """Display the moderator roadmap as a Rich table in a panel.

    Args:
        console: Rich Console for output.
        subtopics: Ordered list of SubTopic objects.
        sufficiency: Moderator's sufficiency assessment string.
        overall_rationale: Moderator's rationale for the decomposition.
    """
    table = Table(
        show_header=True,
        header_style="bold",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("Round", justify="center", style="bold")
    table.add_column("Sub-topic", style="cyan")
    table.add_column("Description")

    for i, st in enumerate(subtopics, 1):
        desc = st.description
        if len(desc) > 80:
            desc = desc[:77] + "..."
        table.add_row(str(i), st.title, desc)

    # Build panel content: optional rationale paragraph above the table
    panel_items: list = []
    if overall_rationale:
        rationale_text = Text(f"Rationale: {overall_rationale[:200]}", style="dim")
        panel_items.append(rationale_text)
        panel_items.append(Text(""))  # blank line separator
    panel_items.append(table)

    subtitle = f"[dim]{sufficiency[:120]}[/dim]" if sufficiency else None
    console.print(
        Panel(
            Group(*panel_items),
            title="[bold]Moderator Roadmap[/bold]",
            subtitle=subtitle,
            border_style="blue",
            expand=False,
        )
    )

    # Show guiding questions if any
    for i, st in enumerate(subtopics, 1):
        if st.guiding_questions:
            console.print(f"  [dim]Round {i} guiding questions:[/dim]")
            for q in st.guiding_questions:
                console.print(f"    [dim]- {q}[/dim]")


def ask_roadmap_action() -> str:
    """Prompt the user for a roadmap action.

    Returns:
        One of: "approve", "reorder", "add", "remove", "edit",
        "regenerate", "adjust_rounds".
    """
    choices = [
        questionary.Choice("Approve and continue", value="approve"),
        questionary.Choice("Reorder sub-topics", value="reorder"),
        questionary.Choice("Add a sub-topic", value="add"),
        questionary.Choice("Remove a sub-topic", value="remove"),
        questionary.Choice("Edit a sub-topic", value="edit"),
        questionary.Choice("Regenerate roadmap", value="regenerate"),
        questionary.Choice("Adjust rounds to match", value="adjust_rounds"),
    ]
    result = questionary.select(
        "Roadmap actions:",
        choices=choices,
    ).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def _ask_reorder(subtopics: List[SubTopic]) -> List[SubTopic]:
    """Prompt the user to specify a new ordering."""
    console_msg = "Current order: " + ", ".join(
        f"{i}={st.title}" for i, st in enumerate(subtopics, 1)
    )
    new_order_str = questionary.text(
        f"Enter new order as comma-separated numbers (e.g. 2,1,3). {console_msg}",
    ).ask()
    if new_order_str is None:
        raise KeyboardInterrupt
    try:
        indices = [int(x.strip()) - 1 for x in new_order_str.split(",")]
        reordered = [subtopics[i] for i in indices if 0 <= i < len(subtopics)]
        if len(reordered) == len(subtopics):
            return reordered
    except (ValueError, IndexError):
        pass
    # If parsing failed, return unchanged
    return subtopics


def _ask_add(subtopics: List[SubTopic]) -> List[SubTopic]:
    """Prompt the user to add a new sub-topic."""
    title = questionary.text("New sub-topic title:").ask()
    if title is None:
        raise KeyboardInterrupt
    description = questionary.text("Description (optional):").ask() or ""
    subtopics.append(SubTopic(title=title, description=description))
    return subtopics


def _ask_remove(subtopics: List[SubTopic]) -> List[SubTopic]:
    """Prompt the user to remove a sub-topic."""
    if len(subtopics) <= 1:
        return subtopics  # Can't remove the last one
    choices = [
        questionary.Choice(f"{i}. {st.title}", value=i - 1)
        for i, st in enumerate(subtopics, 1)
    ]
    idx = questionary.select("Remove which sub-topic?", choices=choices).ask()
    if idx is None:
        raise KeyboardInterrupt
    subtopics.pop(idx)
    return subtopics


def _ask_edit(subtopics: List[SubTopic]) -> List[SubTopic]:
    """Prompt the user to edit one sub-topic's title and description."""
    choices = [
        questionary.Choice(f"{i}. {st.title}", value=i - 1)
        for i, st in enumerate(subtopics, 1)
    ]
    idx = questionary.select("Edit which sub-topic?", choices=choices).ask()
    if idx is None:
        raise KeyboardInterrupt
    st = subtopics[idx]
    new_title = questionary.text("Title:", default=st.title).ask()
    if new_title is None:
        raise KeyboardInterrupt
    new_desc = questionary.text("Description:", default=st.description).ask()
    if new_desc is None:
        raise KeyboardInterrupt
    subtopics[idx] = SubTopic(
        title=new_title,
        description=new_desc,
        rationale=st.rationale,
        guiding_questions=st.guiding_questions,
    )
    return subtopics


def _ask_adjust_rounds(subtopics: List[SubTopic], current_rounds: int) -> int:
    """Prompt the user to adjust the number of rounds.

    Suggests matching the sub-topic count, but allows a custom number.

    Args:
        subtopics: Current list of sub-topics.
        current_rounds: Current max_rounds value.

    Returns:
        The new max_rounds value.
    """
    suggested = len(subtopics)
    result = questionary.text(
        f"Current rounds: {current_rounds}, sub-topics: {suggested}. "
        f"New max_rounds (Enter for {suggested}):",
        default=str(suggested),
    ).ask()
    if result is None:
        raise KeyboardInterrupt
    try:
        value = int(result.strip())
        if value >= 1:
            return value
    except ValueError:
        pass
    return current_rounds


def run_roadmap_review(
    console: Console,
    moderator: Moderator,
    topic: str,
    num_rounds: int,
    agent_personas: List[str],
) -> Tuple[List[SubTopic], int, bool]:
    """Interactive loop: display roadmap, let user edit, until approved.

    Args:
        console: Rich Console for output.
        moderator: The Moderator instance (with roadmap already generated).
        topic: Debate topic (needed if regenerating).
        num_rounds: Number of rounds (needed if regenerating).
        agent_personas: Persona labels (needed if regenerating).

    Returns:
        Tuple of (approved_subtopics, num_rounds, was_modified).
    """
    if moderator.roadmap is None:
        return ([], num_rounds, False)

    subtopics = list(moderator.roadmap.sub_topics)
    sufficiency = moderator.roadmap.sufficiency_note
    overall_rationale = moderator.roadmap.overall_rationale
    was_modified = False

    while True:
        show_roadmap(console, subtopics, sufficiency, overall_rationale)
        action = ask_roadmap_action()

        if action == "approve":
            # Update the moderator's roadmap with any edits
            moderator.roadmap.sub_topics = subtopics
            return (subtopics, num_rounds, was_modified)

        was_modified = True

        if action == "reorder":
            subtopics = _ask_reorder(subtopics)

        elif action == "add":
            subtopics = _ask_add(subtopics)

        elif action == "remove":
            subtopics = _ask_remove(subtopics)

        elif action == "edit":
            subtopics = _ask_edit(subtopics)

        elif action == "regenerate":
            console.print("[dim]Regenerating roadmap...[/dim]")
            roadmap = moderator.generate_roadmap(
                topic=topic,
                num_rounds=num_rounds,
                agent_personas=agent_personas,
            )
            subtopics = list(roadmap.sub_topics)
            sufficiency = roadmap.sufficiency_note
            overall_rationale = roadmap.overall_rationale

        elif action == "adjust_rounds":
            num_rounds = _ask_adjust_rounds(subtopics, num_rounds)
