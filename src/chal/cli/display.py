"""
display.py

Rich-powered runtime display for CHAL debates.

Provides the DebateDisplay class whose handle_event() method is passed as
progress_callback to DebateController.run().  All terminal output during a
debate flows through this class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
from rich.text import Text


# ── Stage name lookup ────────────────────────────────────────────────
_STAGE_NAMES: Dict[int, str] = {
    0: "Briefing",
    1: "Opening Positions",
    2: "Cross-Examination",
    3: "Rebuttals",
    4: "Adjudication",
    5: "Belief Updates",
    6: "Concluding Remarks",
    7: "Scribed Narrative",
}

_STAGE_ICONS: Dict[int, str] = {
    0: "\U0001f9e0",   # 🧠
    1: "\U0001f4d6",   # 📖
    2: "\u2694\ufe0f",  # ⚔️
    3: "\U0001f6e1\ufe0f",  # 🛡️
    4: "\u2696\ufe0f",  # ⚖️
    5: "\U0001f504",   # 🔄
    6: "\U0001f3a4",   # 🎤
    7: "\U0001f4dd",   # 📝
}


def _convergence_label(score: float) -> str:
    """Return a human-readable label for a convergence score (0-1)."""
    if score >= 0.85:
        return "strong agreement"
    elif score >= 0.6:
        return "moderate agreement"
    elif score >= 0.35:
        return "partial agreement"
    else:
        return "divergent"


class DebateDisplay:
    """Renders debate progress to the terminal using Rich.

    Args:
        console: Rich Console instance for output.
        num_rounds: Total number of debate rounds.
        num_agents: Total number of participating agents.
        verbose: If True, show per-agent detail and adjudication tables.
            If False, show only stage headers and round progress.
    """

    def __init__(
        self,
        console: Console,
        num_rounds: int,
        num_agents: int,
        verbose: bool = False,
        interactive: bool = True,
    ) -> None:
        self.console = console
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.verbose = verbose
        self._interactive = interactive

        # Round progress bar (created once, updated per round)
        self._progress: Optional[Progress] = None
        self._round_task_id: Optional[int] = None

        # Collect adjudication results within a round for table display
        self._round_adjudications: List[Dict[str, str]] = []

    # ── Public callback ──────────────────────────────────────────────

    def handle_event(self, event: str, data: Dict[str, Any]) -> None:
        """Dispatch a progress event to the appropriate renderer.

        This method is designed to be passed as ``progress_callback`` to
        :meth:`DebateController.run`.
        """
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(data)

    # ── Event handlers ───────────────────────────────────────────────

    def _on_debate_start(self, data: Dict[str, Any]) -> None:
        topic = data.get("topic", "")
        n_agents = data.get("num_agents", self.num_agents)
        n_rounds = data.get("num_rounds", self.num_rounds)
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{topic}[/bold]\n"
                f"[dim]{n_agents} agents  {n_rounds} round(s)[/dim]",
                title="[bold #9B1B30]Debate Starting[/bold #9B1B30]",
                border_style="#9B1B30",
                expand=False,
            )
        )

        # Start the persistent round progress bar
        self._progress = Progress(
            TextColumn("[bold #9B1B30]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            console=self.console,
        )
        self._progress.start()
        self._round_task_id = self._progress.add_task(
            "Rounds", total=self.num_rounds, completed=0,
        )

    def _on_stage_start(self, data: Dict[str, Any]) -> None:
        stage = data.get("stage", "?")
        name = data.get("name") or _STAGE_NAMES.get(stage, "")
        icon = _STAGE_ICONS.get(stage, "")
        extra = data.get("extra", "")
        subtitle = f"  {extra}" if extra else ""
        self.console.print(
            f"\n  {icon}  [bold]Stage {stage}: {name}[/bold]{subtitle}"
        )

    def _on_stage_complete(self, data: Dict[str, Any]) -> None:
        stage = data.get("stage", "?")
        name = data.get("name") or _STAGE_NAMES.get(stage, "")
        self.console.print(f"  [dim]Stage {stage} ({name}) complete[/dim]")

    def _on_agent_start(self, data: Dict[str, Any]) -> None:
        if self.verbose:
            agent = data.get("agent_name", "?")
            action = data.get("action", "")
            self.console.print(f"    [dim]{agent}[/dim]  {action}")

    def _on_agent_complete(self, data: Dict[str, Any]) -> None:
        agent = data.get("agent_name", "?")
        action = data.get("action", "")
        if self.verbose:
            self.console.print(f"    [green]>[/green] {agent}  {action}")
        else:
            self.console.print(f"    [green]>[/green] {agent}  {action}")

    def _on_adjudication_result(self, data: Dict[str, Any]) -> None:
        self._round_adjudications.append(data)
        if self.verbose:
            challenger = data.get("challenger", "?")
            target = data.get("target", "?")
            outcome = data.get("outcome", "?")
            qid = data.get("qid", "")
            prefix = f"{qid}: " if qid else ""
            self.console.print(
                f"    [yellow]|[/yellow] {prefix}{challenger} -> {target}: "
                f"[bold]{outcome}[/bold]"
            )

    def _on_round_start(self, data: Dict[str, Any]) -> None:
        round_num = data.get("round", "?")
        total = data.get("total_rounds", self.num_rounds)
        self.console.print(
            f"\n[bold #9B1B30]{'=' * 50}[/bold #9B1B30]"
        )
        self.console.print(
            f"[bold #9B1B30]  Round {round_num} of {total}[/bold #9B1B30]"
        )
        self.console.print(
            f"[bold #9B1B30]{'=' * 50}[/bold #9B1B30]"
        )
        # Reset per-round adjudication collector
        self._round_adjudications = []

    def _on_round_complete(self, data: Dict[str, Any]) -> None:
        round_num = data.get("round", 0)

        # Advance the progress bar
        if self._progress and self._round_task_id is not None:
            self._progress.update(self._round_task_id, completed=round_num)

        # Show adjudication results table (if any collected this round)
        if self._round_adjudications:
            self._show_adjudication_table(self._round_adjudications)

        # Show performance scores
        scores = data.get("scores")
        if scores:
            self._show_performance_table(scores)

        # Show convergence info
        convergence = data.get("convergence")
        if convergence and convergence.get("convergence_score") is not None:
            score = convergence["convergence_score"]
            self.console.print(
                f"  [dim]Convergence: {score:.2f}[/dim]"
            )

    def _on_debate_complete(self, data: Dict[str, Any]) -> None:
        # Stop progress bar
        if self._progress:
            self._progress.stop()
            self._progress = None

        self.console.print()

        # Build rich summary content
        parts: List[str] = ["[bold green]Debate complete![/bold green]"]

        # Duration
        duration_s = data.get("total_duration_s")
        if duration_s is not None:
            mins, secs = divmod(int(duration_s), 60)
            parts.append(f"Duration: {mins}m {secs}s")

        # Topic
        topic = data.get("topic")
        if topic:
            parts.append(f"Topic: {topic}")

        self.console.print(
            Panel(
                "\n".join(parts),
                border_style="green",
                expand=False,
            )
        )

        # Final performance table from agent_stats
        agent_stats = data.get("agent_stats")
        if agent_stats:
            self._show_performance_table(agent_stats)

        # Convergence from convergence_history
        convergence_history = data.get("convergence_history")
        if convergence_history and len(convergence_history) > 0:
            last = convergence_history[-1]
            score = last.get("convergence_score")
            if score is not None:
                label = _convergence_label(score)
                self.console.print(
                    f"  Convergence: {score:.2f} ({label})"
                )

    def _on_output_files_saved(self, data: Dict[str, Any]) -> None:
        """Show a summary of saved output files."""
        files = data.get("files", [])
        if files:
            file_names = "  ".join(files)
            self.console.print(f"\n  Output files: {file_names}")

    # ── Table helpers ────────────────────────────────────────────────

    def _show_adjudication_table(self, results: List[Dict[str, str]]) -> None:
        """Render a compact adjudication results table."""
        table = Table(
            title="Adjudication Results",
            show_header=True,
            header_style="bold",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Challenger", style="#A82545")
        table.add_column("Target", style="#A82545")
        table.add_column("Outcome", style="bold")

        for r in results:
            outcome = r.get("outcome", "?")
            style = "green" if outcome.lower() == "sustained" else (
                "red" if outcome.lower() == "overruled" else "yellow"
            )
            table.add_row(
                r.get("challenger", "?"),
                r.get("target", "?"),
                Text(outcome, style=style),
            )

        self.console.print()
        self.console.print(table)

    def _show_performance_table(self, agent_stats: Dict[str, Any]) -> None:
        """Render a compact performance leaderboard."""
        table = Table(
            title="Performance Scores",
            show_header=True,
            header_style="bold",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Agent", style="#A82545")
        table.add_column("Score", justify="right")
        table.add_column("Wins", justify="right")
        table.add_column("Losses", justify="right")

        for name, stats in agent_stats.items():
            score = stats.get("performance_score", 0)
            wins = stats.get("sustained", 0)
            losses = stats.get("overruled", 0)
            table.add_row(
                name,
                f"{score:.1f}" if isinstance(score, (int, float)) else str(score),
                str(wins),
                str(losses),
            )

        self.console.print(table)

    # ── Error handler ────────────────────────────────────────────────

    def handle_error(self, agent_name: str, error: Exception, retry_count: int) -> str:
        """Handle an LLM error during the debate.

        In interactive mode, shows a Rich panel and prompts the user.
        In headless mode (verbose=False), retries once then aborts.

        Args:
            agent_name: Name of the agent that failed.
            error: The exception that occurred.
            retry_count: How many retries have already been attempted.

        Returns:
            One of "retry", "skip", "abort".
        """
        self.console.print(
            Panel(
                f"[red]API call failed for {agent_name}:[/red]\n"
                f"  {type(error).__name__}: {error}\n"
                f"  [dim]Retry count: {retry_count}[/dim]",
                title="[bold red]Error[/bold red]",
                border_style="red",
                expand=False,
            )
        )

        if not self._interactive:
            # Headless: retry once then abort
            if retry_count < 1:
                self.console.print("  [dim]Retrying automatically...[/dim]")
                return "retry"
            return "abort"

        # Interactive: prompt the user
        try:
            import questionary
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("Retry (try again)", value="retry"),
                    questionary.Choice("Skip this agent's turn", value="skip"),
                    questionary.Choice("Abort debate (save partial results)", value="abort"),
                ],
            ).ask()
            if action is None:
                return "abort"
            return action
        except Exception:
            return "abort"
