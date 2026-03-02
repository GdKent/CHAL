"""
wizard.py

Interactive configuration wizard for CHAL debates.
Walks the user through 10 steps to build a DebateConfig, with a review/edit
loop before launching.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chal.config import (
    AgentConfig,
    AdjudicationConfig,
    BloodSportConfig,
    CollaborativeConfig,
    DebateConfig,
    ModeratorConfig,
    OutputConfig,
    CONFIG_DIR,
    DEFAULT_STORAGE_DIR,
)


# =========================================================================
# Constants
# =========================================================================

PERSONA_CHOICES = [
    Choice("EMPIRICIST      - Demands empirical evidence for all claims", value="EMPIRICIST"),
    Choice("RATIONALIST     - Trusts logical deduction over observation", value="RATIONALIST"),
    Choice("SKEPTIC         - Challenges all claims, exposes assumptions", value="SKEPTIC"),
    Choice("SUPERNATURALIST - Accepts truths beyond empirical observation", value="SUPERNATURALIST"),
    Choice("PHENOMENOLOGIST - Grounds truth in lived experience", value="PHENOMENOLOGIST"),
    Choice("PRAGMATIST      - Defines truth as what works in practice", value="PRAGMATIST"),
    Choice("CONSTRUCTIVIST  - Truth is socially constructed", value="CONSTRUCTIVIST"),
    Choice("NIHILIST        - No inherent meaning or objective truth", value="NIHILIST"),
    Choice("BAYESIAN        - Models knowledge as probabilistic inference", value="BAYESIAN"),
    Choice("PANPSYCHIST     - Consciousness is fundamental to all matter", value="PANPSYCHIST"),
    Choice("SIMULATIONIST   - Evaluates claims via simulation hypothesis", value="SIMULATIONIST"),
    Choice("SYNTHESIST      - Integrates science, spirituality, and systems", value="SYNTHESIST"),
]

PROVIDER_CHOICES = ["openai", "anthropic", "google"]

MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1", "o3-mini"],
    "anthropic": ["claude-sonnet-4-5-20250929", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
    "google": ["gemini-2.0-flash", "gemini-2.0-pro"],
}

# (display label, OutputConfig attribute, default value)
OUTPUT_TOGGLES: list[tuple[str, str, bool]] = [
    ("Debate transcript", "save_transcript", True),
    ("Narrative synthesis (scribe)", "save_synthesis", True),
    ("Belief trajectories plot", "plot_trajectories", True),
    ("Agent statistics", "save_agent_stats", True),
    ("Initial beliefs", "save_initial_beliefs", True),
    ("Final beliefs", "save_final_beliefs", True),
    ("Graph visualization", "generate_graph_visualization", True),
    ("Embeddings", "generate_embeddings", True),
    ("Training data export", "save_training_data", False),
    ("Analysis report", "save_analysis_report", False),
    ("Debug log", "save_debug_log", True),
]


# =========================================================================
# Helpers
# =========================================================================

def _require(value):
    """Raise KeyboardInterrupt if questionary returned None (Ctrl+C)."""
    if value is None:
        raise KeyboardInterrupt
    return value


def _validate_float_range(text: str, lo: float = 0.0, hi: float = 1.0) -> bool | str:
    """Validator for float inputs within a range."""
    try:
        val = float(text)
        if lo <= val <= hi:
            return True
        return f"Must be between {lo} and {hi}"
    except ValueError:
        return "Must be a number"


def _validate_int_range(text: str, lo: int, hi: int) -> bool | str:
    """Validator for integer inputs within a range."""
    try:
        val = int(text)
        if lo <= val <= hi:
            return True
        return f"Must be between {lo} and {hi}"
    except ValueError:
        return "Must be a whole number"


# =========================================================================
# Preset selection
# =========================================================================

def _scan_presets() -> list[tuple[str, str, Path]]:
    """Scan configurations/ directory for YAML presets.

    Returns:
        List of (display_name, config_name, path) tuples sorted by name.
    """
    presets: list[tuple[str, str, Path]] = []
    if not CONFIG_DIR.is_dir():
        return presets
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data.get("metadata", {}).get("name", p.stem)
            desc = data.get("metadata", {}).get("description", "")
            label = f"{name}  [dim]({desc})[/dim]" if desc else name
            presets.append((label, p.stem, p))
        except Exception:
            continue
    return presets


def ask_preset() -> DebateConfig | None:
    """Ask user whether to start from a preset configuration.

    Returns:
        A loaded DebateConfig if a preset was chosen, or None for custom.
    """
    presets = _scan_presets()
    choices = [Choice("Custom (configure from scratch)", value="__custom__")]
    for label, config_name, path in presets:
        choices.append(Choice(label, value=str(path)))

    result = questionary.select(
        "Start from a preset?",
        choices=choices,
    ).ask()
    if result is None:
        raise KeyboardInterrupt

    if result == "__custom__":
        return None

    return DebateConfig.from_yaml(Path(result))


# =========================================================================
# Step functions — each returns user-chosen value(s) for one config section
# =========================================================================

def ask_topic(default: str = "") -> str:
    """Step 1: Ask for the debate topic."""
    return _require(questionary.text(
        "What topic should the agents debate?",
        default=default,
        instruction="(free text — e.g. 'Does free will exist?')",
    ).ask())


def ask_num_agents(default: int = 2) -> int:
    """Step 2: Ask how many agents."""
    answer = _require(questionary.text(
        "How many agents should participate? (2-6)",
        default=str(default),
        validate=lambda t: _validate_int_range(t, 2, 6),
    ).ask())
    return int(answer)


def ask_agent_config(index: int, default: AgentConfig | None = None) -> AgentConfig:
    """Step 3: Configure a single agent (persona, provider, model, temperature)."""
    console = Console()
    console.print(f"\n[bold]Configure Agent {index + 1}:[/bold]")

    # Persona
    default_persona = default.persona if default else None
    persona = _require(questionary.select(
        "Persona:",
        choices=PERSONA_CHOICES,
        default=default_persona,
    ).ask())

    # Provider
    default_provider = default.provider if default else "openai"
    provider = _require(questionary.select(
        "Provider:",
        choices=PROVIDER_CHOICES,
        default=default_provider,
    ).ask())

    # Model — show suggestions for the chosen provider
    suggestions = MODEL_SUGGESTIONS.get(provider, [])
    default_model = default.model if default else (suggestions[0] if suggestions else "gpt-4o")
    model = _require(questionary.autocomplete(
        "Model:",
        choices=suggestions,
        default=default_model,
    ).ask())

    # Temperature
    default_temp = default.temperature if default else 0.7
    temp_str = _require(questionary.text(
        "Temperature (0.0-1.0):",
        default=str(default_temp),
        validate=lambda t: _validate_float_range(t, 0.0, 1.0),
    ).ask())

    # Auto-generate name from persona
    name = default.name if default else f"Agent-{persona.capitalize()}"

    return AgentConfig(
        name=name,
        persona=persona,
        model=model,
        temperature=float(temp_str),
        provider=provider,
    )


def ask_stage2_mode(default: str = "open") -> str:
    """Step 4: Cross-examination style."""
    return _require(questionary.select(
        "Cross-examination style:",
        choices=[
            Choice("Open (agents freely challenge each other)", value="open"),
            Choice("Moderated (guided by a moderator roadmap)", value="moderated"),
        ],
        default=default,
    ).ask())


def ask_stage3_mode(
    default_mode: str = "rebuttal",
    default_bloodsport: BloodSportConfig | None = None,
    default_collaborative: CollaborativeConfig | None = None,
) -> tuple[str, dict]:
    """Step 5: Debate mode + sub-options.

    Returns:
        (mode_str, sub_options_dict) where sub_options contains any mode-specific
        settings (bloodsport intensity/exchanges, collaborative params).
    """
    mode = _require(questionary.select(
        "Debate mode:",
        choices=[
            Choice("Rebuttal (single-shot responses)", value="rebuttal"),
            Choice("Collaborative (multi-turn truth-seeking)", value="collaborative"),
            Choice("Blood Sport (adversarial multi-turn)", value="bloodsport"),
        ],
        default=default_mode,
    ).ask())

    sub_options: dict = {}

    if mode == "bloodsport":
        bs = default_bloodsport or BloodSportConfig()
        intensity = _require(questionary.select(
            "Blood Sport intensity:",
            choices=[
                Choice("Mild", value="mild"),
                Choice("Moderate", value="moderate"),
                Choice("Extreme", value="extreme"),
            ],
            default=bs.intensity,
        ).ask())

        max_exchanges = _require(questionary.text(
            "Max exchanges per agent pair (1-20):",
            default=str(bs.max_exchanges),
            validate=lambda t: _validate_int_range(t, 1, 20),
        ).ask())

        sub_options["bloodsport"] = BloodSportConfig(
            intensity=intensity,
            max_exchanges=int(max_exchanges),
        )

    elif mode == "collaborative":
        collab = default_collaborative or CollaborativeConfig()
        max_turns = _require(questionary.text(
            "Max turns per question (3-30):",
            default=str(collab.max_turns_per_question),
            validate=lambda t: _validate_int_range(t, 3, 30),
        ).ask())

        early_term = _require(questionary.confirm(
            "Enable early termination on agreement?",
            default=collab.early_termination_on_agreement,
        ).ask())

        sub_options["collaborative"] = CollaborativeConfig(
            max_turns_per_question=int(max_turns),
            min_turns_per_question=collab.min_turns_per_question,
            adjudicator_check_interval=collab.adjudicator_check_interval,
            early_termination_on_agreement=early_term,
        )

    return mode, sub_options


def ask_num_rounds(default: int = 1) -> int:
    """Step 6: Number of debate rounds."""
    answer = _require(questionary.text(
        "Number of debate rounds:",
        default=str(default),
        validate=lambda t: _validate_int_range(t, 1, 10),
    ).ask())
    return int(answer)


def ask_adjudicator_config(default: AdjudicationConfig | None = None) -> AdjudicationConfig:
    """Step 7: Adjudicator model and weights."""
    adj = default or AdjudicationConfig()

    console = Console()
    console.print("\n[bold]Adjudicator Configuration:[/bold]")

    provider = _require(questionary.select(
        "Adjudicator provider:",
        choices=PROVIDER_CHOICES,
        default=adj.provider,
    ).ask())

    suggestions = MODEL_SUGGESTIONS.get(provider, [])
    model = _require(questionary.autocomplete(
        "Adjudicator model:",
        choices=suggestions,
        default=adj.model,
    ).ask())

    logic_weight = _require(questionary.text(
        "Logic weight (0.0-1.0):",
        default=str(adj.logic_weight),
        validate=lambda t: _validate_float_range(t, 0.0, 1.0),
    ).ask())

    ethics_weight = _require(questionary.text(
        "Ethics weight (0.0-1.0):",
        default=str(adj.ethics_weight),
        validate=lambda t: _validate_float_range(t, 0.0, 1.0),
    ).ask())

    return AdjudicationConfig(
        model=model,
        provider=provider,
        logic_weight=float(logic_weight),
        ethics_weight=float(ethics_weight),
        logic_system=adj.logic_system,
        ethics_system=adj.ethics_system,
    )


def ask_moderator_config(default: ModeratorConfig | None = None) -> ModeratorConfig:
    """Step 8: Moderator configuration (only shown if stage2 == 'moderated')."""
    mod = default or ModeratorConfig()

    console = Console()
    console.print("\n[bold]Moderator Configuration:[/bold]")

    provider = _require(questionary.select(
        "Moderator provider:",
        choices=PROVIDER_CHOICES,
        default=mod.provider,
    ).ask())

    suggestions = MODEL_SUGGESTIONS.get(provider, [])
    model = _require(questionary.autocomplete(
        "Moderator model:",
        choices=suggestions,
        default=mod.model,
    ).ask())

    moderator_mode = _require(questionary.select(
        "Moderator mode:",
        choices=[
            Choice("Static (fixed roadmap for all rounds)", value="static"),
            Choice("Adaptive (revises roadmap between rounds)", value="adaptive"),
        ],
        default=mod.moderator_mode,
    ).ask())

    return ModeratorConfig(
        model=model,
        provider=provider,
        temperature=mod.temperature,
        context=mod.context,
        moderator_mode=moderator_mode,
    )


def ask_output_toggles(default: OutputConfig | None = None) -> dict[str, bool]:
    """Step 9: Which outputs to generate (multi-select checkbox).

    Returns:
        Dict mapping OutputConfig attribute names to bool values.
    """
    # Build choices with current defaults
    choices = []
    for label, attr, fallback in OUTPUT_TOGGLES:
        checked = getattr(default, attr, fallback) if default else fallback
        choices.append(Choice(label, value=attr, checked=checked))

    selected = _require(questionary.checkbox(
        "Which outputs would you like?",
        choices=choices,
    ).ask())

    # Build result: selected attrs are True, others False
    all_attrs = {attr for _, attr, _ in OUTPUT_TOGGLES}
    return {attr: (attr in selected) for attr in all_attrs}


# =========================================================================
# Review panel
# =========================================================================

def show_review_panel(config: DebateConfig, console: Console) -> None:
    """Render a rich Panel summarizing the full debate configuration."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold", width=16)
    table.add_column("Value")

    table.add_row("Topic", config.topic)

    agent_lines = []
    for a in config.agents:
        agent_lines.append(f"{a.name} ({a.persona}, {a.provider}/{a.model}, t={a.temperature})")
    table.add_row("Agents", "\n".join(agent_lines))

    table.add_row("Stage 2", config.stage2_mode)
    table.add_row("Stage 3", config.stage3_mode)
    table.add_row("Rounds", str(config.max_rounds))
    table.add_row(
        "Adjudicator",
        f"{config.adjudication.model} ({config.adjudication.provider}), "
        f"logic={config.adjudication.logic_weight}, ethics={config.adjudication.ethics_weight}",
    )

    if config.stage2_mode == "moderated":
        table.add_row(
            "Moderator",
            f"{config.moderator.model} ({config.moderator.provider}), "
            f"mode={config.moderator.moderator_mode}",
        )

    if config.stage3_mode == "bloodsport":
        table.add_row(
            "Blood Sport",
            f"intensity={config.bloodsport.intensity}, "
            f"max_exchanges={config.bloodsport.max_exchanges}",
        )
    elif config.stage3_mode == "collaborative":
        table.add_row(
            "Collaborative",
            f"max_turns={config.collaborative.max_turns_per_question}, "
            f"early_term={config.collaborative.early_termination_on_agreement}",
        )

    # Summarize enabled outputs
    enabled = []
    for label, attr, _ in OUTPUT_TOGGLES:
        if getattr(config.outputs, attr, False):
            enabled.append(label.split(" (")[0])  # strip parenthetical
    table.add_row("Outputs", ", ".join(enabled) if enabled else "None")

    panel = Panel(table, title="Debate Configuration", border_style="cyan")
    console.print(panel)


def ask_review_action() -> str:
    """Ask user what to do after reviewing the configuration."""
    return _require(questionary.select(
        "Proceed with this configuration?",
        choices=[
            Choice("Launch debate", value="launch"),
            Choice("Edit a setting", value="edit"),
            Choice("Save config to YAML", value="save"),
            Choice("Save config and launch", value="save_and_launch"),
            Choice("Cancel", value="cancel"),
        ],
    ).ask())


def ask_edit_section(show_moderator: bool = False) -> str:
    """Ask which config section to re-edit."""
    choices = [
        Choice("Topic", value="topic"),
        Choice("Number of agents", value="num_agents"),
        Choice("Agent configurations", value="agents"),
        Choice("Stage 2 mode (cross-examination)", value="stage2"),
        Choice("Stage 3 mode (debate style)", value="stage3"),
        Choice("Number of rounds", value="rounds"),
        Choice("Adjudicator", value="adjudicator"),
    ]
    if show_moderator:
        choices.append(Choice("Moderator", value="moderator"))
    choices.append(Choice("Output toggles", value="outputs"))

    return _require(questionary.select(
        "Which section would you like to edit?",
        choices=choices,
    ).ask())


def _ask_save_path() -> Path:
    """Prompt for a YAML file path to save the config."""
    path_str = _require(questionary.text(
        "Save config to (YAML path):",
        default="my_debate.yaml",
    ).ask())
    path = Path(path_str)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    return path


# =========================================================================
# Main wizard orchestrator
# =========================================================================

def run_wizard(
    console: Console,
    prefill: DebateConfig | None = None,
) -> tuple[DebateConfig | None, str]:
    """Run the full interactive configuration wizard.

    Args:
        console: Rich console for output.
        prefill: Optional pre-existing config to use as defaults.

    Returns:
        (config, action) where action is one of:
        "launch", "save", "save_and_launch", "cancel".
        config is None if cancelled.
    """
    console.print("[bold]Configuration Wizard[/bold]\n")

    # ----- Preset selection (only when no prefill) -----
    if prefill is None:
        preset = ask_preset()
        if preset is not None:
            prefill = preset

    # ----- Step 1: Topic -----
    topic = ask_topic(default=prefill.topic if prefill else "")

    # ----- Step 2: Number of agents -----
    num_agents = ask_num_agents(default=len(prefill.agents) if prefill else 2)

    # ----- Step 3: Agent configs -----
    agent_configs: list[AgentConfig] = []
    for i in range(num_agents):
        default_agent = prefill.agents[i] if prefill and i < len(prefill.agents) else None
        agent_configs.append(ask_agent_config(i, default=default_agent))

    # ----- Step 4: Stage 2 mode -----
    stage2_mode = ask_stage2_mode(
        default=prefill.stage2_mode if prefill else "open"
    )

    # ----- Step 5: Stage 3 mode + sub-options -----
    stage3_mode, sub_options = ask_stage3_mode(
        default_mode=prefill.stage3_mode if prefill else "rebuttal",
        default_bloodsport=prefill.bloodsport if prefill else None,
        default_collaborative=prefill.collaborative if prefill else None,
    )

    # ----- Step 6: Number of rounds -----
    max_rounds = ask_num_rounds(
        default=prefill.max_rounds if prefill else 1
    )

    # ----- Step 7: Adjudicator -----
    adjudication = ask_adjudicator_config(
        default=prefill.adjudication if prefill else None
    )

    # ----- Step 8: Moderator (conditional) -----
    moderator = ModeratorConfig()
    if stage2_mode == "moderated":
        moderator = ask_moderator_config(
            default=prefill.moderator if prefill else None
        )

    # ----- Step 9: Output toggles -----
    output_flags = ask_output_toggles(
        default=prefill.outputs if prefill else None
    )

    # ----- Build the DebateConfig -----
    def _build_config() -> DebateConfig:
        outputs = OutputConfig(storage_dir=DEFAULT_STORAGE_DIR)
        for attr, val in output_flags.items():
            setattr(outputs, attr, val)

        bloodsport = sub_options.get("bloodsport", prefill.bloodsport if prefill else BloodSportConfig())
        collaborative = sub_options.get("collaborative", prefill.collaborative if prefill else CollaborativeConfig())

        return DebateConfig(
            name=f"Debate: {topic[:50]}",
            topic=topic,
            max_rounds=max_rounds,
            stage2_mode=stage2_mode,
            stage3_mode=stage3_mode,
            agents=agent_configs,
            adjudication=adjudication,
            outputs=outputs,
            collaborative=collaborative,
            bloodsport=bloodsport,
            moderator=moderator,
        )

    config = _build_config()

    # ----- Step 10: Review & action loop -----
    while True:
        console.print()
        show_review_panel(config, console)
        action = ask_review_action()

        if action == "launch":
            return config, "launch"

        elif action == "cancel":
            return None, "cancel"

        elif action == "save":
            path = _ask_save_path()
            config.to_yaml(path)
            console.print(f"[green]Config saved to {path}[/green]")
            # Stay in loop so user can launch or keep editing

        elif action == "save_and_launch":
            path = _ask_save_path()
            config.to_yaml(path)
            console.print(f"[green]Config saved to {path}[/green]")
            return config, "save_and_launch"

        elif action == "edit":
            section = ask_edit_section(show_moderator=(config.stage2_mode == "moderated"))
            config = _apply_edit(config, section, console)


def _apply_edit(config: DebateConfig, section: str, console: Console) -> DebateConfig:
    """Re-run a single wizard step and update the config in place.

    Returns:
        Updated DebateConfig (may be a new instance for some fields).
    """
    if section == "topic":
        config.topic = ask_topic(default=config.topic)
        config.name = f"Debate: {config.topic[:50]}"

    elif section == "num_agents":
        new_count = ask_num_agents(default=len(config.agents))
        if new_count < len(config.agents):
            config.agents = config.agents[:new_count]
        elif new_count > len(config.agents):
            for i in range(len(config.agents), new_count):
                config.agents.append(ask_agent_config(i))

    elif section == "agents":
        # Let user pick which agent to edit
        agent_choices = [
            Choice(f"{a.name} ({a.persona})", value=i)
            for i, a in enumerate(config.agents)
        ]
        idx = _require(questionary.select(
            "Which agent to edit?",
            choices=agent_choices,
        ).ask())
        config.agents[idx] = ask_agent_config(idx, default=config.agents[idx])

    elif section == "stage2":
        config.stage2_mode = ask_stage2_mode(default=config.stage2_mode)
        if config.stage2_mode == "moderated" and config.moderator.model == ModeratorConfig().model:
            # First time selecting moderated — prompt for moderator config
            config.moderator = ask_moderator_config()

    elif section == "stage3":
        mode, sub_opts = ask_stage3_mode(
            default_mode=config.stage3_mode,
            default_bloodsport=config.bloodsport,
            default_collaborative=config.collaborative,
        )
        config.stage3_mode = mode
        if "bloodsport" in sub_opts:
            config.bloodsport = sub_opts["bloodsport"]
        if "collaborative" in sub_opts:
            config.collaborative = sub_opts["collaborative"]

    elif section == "rounds":
        config.max_rounds = ask_num_rounds(default=config.max_rounds)

    elif section == "adjudicator":
        config.adjudication = ask_adjudicator_config(default=config.adjudication)

    elif section == "moderator":
        config.moderator = ask_moderator_config(default=config.moderator)

    elif section == "outputs":
        output_flags = ask_output_toggles(default=config.outputs)
        for attr, val in output_flags.items():
            setattr(config.outputs, attr, val)

    return config
