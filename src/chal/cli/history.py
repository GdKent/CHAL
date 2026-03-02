"""
history.py

Debate history logging and replay for CHAL.

Logs a summary entry after each debate to ~/.chal/history.json and saves
a config snapshot as YAML.  Provides functions to list past debates and
reload a config by debate ID.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from chal.config import DebateConfig

HISTORY_DIR = Path.home() / ".chal" / "history"
HISTORY_FILE = Path.home() / ".chal" / "history.json"


def _ensure_history_dir() -> None:
    """Create the history directory if it doesn't exist."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _read_history() -> List[Dict[str, Any]]:
    """Read the history file and return the debates list."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("debates", [])
    except (json.JSONDecodeError, OSError):
        return []


def _write_history(debates: List[Dict[str, Any]]) -> None:
    """Write the debates list to the history file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"debates": debates}, f, indent=2, default=str)


def log_debate(
    config: DebateConfig,
    results: Dict[str, Any],
    duration_s: float = 0,
) -> str:
    """Log a completed debate to the history file and save a config snapshot.

    Args:
        config: The debate configuration.
        results: Results dict from controller.run().
        duration_s: Total debate duration in seconds.

    Returns:
        The debate ID (short UUID).
    """
    _ensure_history_dir()
    debate_id = uuid.uuid4().hex[:8]

    # Save config snapshot
    snapshot_path = HISTORY_DIR / f"{debate_id}.yaml"
    config.to_yaml(snapshot_path)

    # Extract summary data
    agent_stats = results.get("agent_stats", {})
    convergence_history = results.get("convergence_history", [])
    last_convergence = (
        convergence_history[-1].get("convergence_score")
        if convergence_history
        else None
    )

    # Determine winner (agent with highest performance_score)
    winner = None
    best_score = -float("inf")
    for name, stats in agent_stats.items():
        score = stats.get("performance_score", 0)
        if score > best_score:
            best_score = score
            winner = name

    entry = {
        "id": debate_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topic": config.topic,
        "agents": [a.name for a in config.agents],
        "rounds": config.max_rounds,
        "duration_s": round(duration_s, 1),
        "convergence": last_convergence,
        "winner": winner,
        "config_snapshot": str(snapshot_path),
        "output_dir": str(config.outputs.storage_dir),
    }

    debates = _read_history()
    debates.append(entry)
    _write_history(debates)

    return debate_id


def list_debates() -> List[Dict[str, Any]]:
    """Return all debate entries from the history file."""
    return _read_history()


def load_debate_config(debate_id: str) -> DebateConfig:
    """Load a debate's config snapshot by its ID.

    Args:
        debate_id: Short UUID of the debate.

    Returns:
        The loaded DebateConfig.

    Raises:
        FileNotFoundError: If the snapshot doesn't exist.
    """
    snapshot_path = HISTORY_DIR / f"{debate_id}.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Config snapshot for debate '{debate_id}' not found at {snapshot_path}"
        )
    return DebateConfig.from_yaml(snapshot_path)


def format_history_table(debates: List[Dict[str, Any]], console: Console) -> None:
    """Render the debate history as a Rich table.

    Args:
        debates: List of debate entries from list_debates().
        console: Rich console for output.
    """
    if not debates:
        console.print("[dim]No debate history found.[/dim]")
        return

    table = Table(
        title="Debate History",
        show_header=True,
        header_style="bold",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("ID", style="cyan")
    table.add_column("Date", style="dim")
    table.add_column("Topic")
    table.add_column("Agents", justify="right")
    table.add_column("Rounds", justify="right")
    table.add_column("Duration")
    table.add_column("Winner", style="green")
    table.add_column("Conv.", justify="right")

    for d in debates:
        timestamp = d.get("timestamp", "")
        # Show just the date part
        date_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
        convergence = d.get("convergence")
        conv_str = f"{convergence:.2f}" if convergence is not None else "-"
        duration_s = d.get("duration_s", 0)
        mins, secs = divmod(int(duration_s), 60)
        dur_str = f"{mins}m {secs}s" if duration_s else "-"
        agents = d.get("agents", [])

        table.add_row(
            d.get("id", "?"),
            date_str,
            (d.get("topic", "")[:40] + "...") if len(d.get("topic", "")) > 40 else d.get("topic", ""),
            str(len(agents)),
            str(d.get("rounds", "?")),
            dur_str,
            d.get("winner", "-") or "-",
            conv_str,
        )

    console.print(table)
