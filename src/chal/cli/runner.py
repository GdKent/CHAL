"""
runner.py

Shared debate execution and output-saving logic used by both the interactive
wizard and headless CLI mode.
"""

from __future__ import annotations

import json
import time
import traceback

from rich.console import Console

from chal.agents.epistemic_personas import get_persona
from chal.agents.factory import create_agent_from_config
from chal.beliefs.io import belief_to_markdown, load_belief_from_file
from chal.beliefs.patches import initialize_defense_tracking
from chal.cli.api_keys import create_key_pool, validate_api_keys
from chal.cli.display import DebateDisplay
from chal.config import DebateConfig
from chal.orchestrator.debate_controller import DebateController
from chal.utilities.utils import sanitize_filename, select_best_agent


def run_debate(
    config: DebateConfig,
    console: Console,
    verbose: bool = False,
    interactive: bool = True,
) -> int:
    """Execute a full debate from a DebateConfig and save outputs.

    Args:
        config: Fully populated debate configuration.
        console: Rich console for styled output.
        verbose: Whether to show verbose/debug output.
        interactive: Whether the session is interactive.

    Returns:
        Exit code (0 = success, 1 = failure).
    """

    console.print(f"\n[bold]Debate:[/bold] {config.name}")
    console.print(f"  Topic: {config.topic}")
    console.print(f"  Rounds: {config.max_rounds}")
    console.print(f"  Agents: {len(config.agents)}")

    # Ensure storage directory exists
    config.outputs.ensure_storage_dir()
    console.print(f"  Storage: {config.outputs.storage_dir}")

    # Validate API keys before creating agents
    validate_api_keys(config, console, interactive=interactive)

    # Create key pool for multi-key rotation (supports comma-separated keys in .env)
    key_pool = create_key_pool(config)

    # Create agents from config
    agents = []
    personas = {}

    console.print("\n[bold]Initializing agents:[/bold]")
    for agent_cfg in config.agents:
        try:
            persona_obj = get_persona(agent_cfg.persona)
        except KeyError:
            console.print(f"[red]Error: Unknown persona '{agent_cfg.persona}'[/red]")
            return 1

        agent = create_agent_from_config(agent_cfg, key_pool=key_pool)
        agents.append(agent)
        personas[agent_cfg.name] = persona_obj

        # Pre-load custom belief from file (skips Stage 1 for this agent)
        if agent_cfg.belief_file:
            try:
                belief_obj = load_belief_from_file(agent_cfg.belief_file)
                initialize_defense_tracking(belief_obj)
                agent.set_internal_belief_obj(belief_obj)
                agent.set_internal_belief(belief_to_markdown(belief_obj))
                console.print(
                    f"  [green]>[/green] {agent_cfg.name} "
                    f"({agent_cfg.persona}, {agent_cfg.provider}/{agent_cfg.model}) "
                    f"[cyan](custom belief loaded from {agent_cfg.belief_file})[/cyan]"
                )
            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]Error loading custom belief for {agent_cfg.name}:[/red] {e}")
                return 1
        else:
            console.print(
                f"  [green]>[/green] {agent_cfg.name} "
                f"({agent_cfg.persona}, {agent_cfg.provider}/{agent_cfg.model})"
            )

    # Determine real-time log path (streams to disk during debate)
    log_file_path = None
    if config.outputs.save_debug_log:
        log_file_path = config.outputs.storage_dir / config.outputs.debug_log_file

    # Create controller
    controller = DebateController(
        agents=agents,
        max_rounds=config.max_rounds,
        config=config,
        key_pool=key_pool,
        log_file_path=log_file_path,
    )

    # Create display and wire callback
    display = DebateDisplay(
        console=console,
        num_rounds=config.max_rounds,
        num_agents=len(config.agents),
        verbose=verbose,
        interactive=interactive,
    )

    start_time = time.time()

    try:
        results = controller.run(
            topic=config.topic,
            personas=personas,
            progress_callback=display.handle_event,
            on_error=display.handle_error,
        )
    except Exception as e:
        display.stop()
        # Write the full traceback into the debug log before closing
        controller.debug_log.write(f"\n{'=' * 80}")
        controller.debug_log.write("FATAL EXCEPTION — debate aborted")
        controller.debug_log.write(f"{'=' * 80}")
        controller.debug_log.write(traceback.format_exc())
        controller._close_debug_log()
        console.print(f"\n[red]Error during debate execution:[/red] {e}")
        if log_file_path:
            console.print(f"  [dim]Debug log (up to crash): {log_file_path}[/dim]")
        if verbose:
            console.print(traceback.format_exc())
        return 1

    duration_s = time.time() - start_time

    # Save outputs
    saved_files = save_debate_outputs(config, results, controller, console, verbose)

    # Show output files summary
    if saved_files:
        display.handle_event("output_files_saved", {
            "files": saved_files,
            "storage_dir": str(config.outputs.storage_dir),
        })

    # Log to debate history
    try:
        from chal.cli.history import log_debate
        debate_id = log_debate(config, results, duration_s=duration_s)
        console.print(f"  [dim]History ID: {debate_id}[/dim]")
    except Exception:
        pass  # History logging is best-effort

    console.print(f"\nResults saved to: {config.outputs.storage_dir}")

    return 0


def save_debate_outputs(
    config: DebateConfig,
    results: dict,
    controller,
    console: Console,
    verbose: bool = False,
) -> list[str]:
    """Save all debate outputs based on config settings.

    Args:
        config: Debate configuration with output settings.
        results: Dict returned by controller.run().
        controller: The DebateController instance (needed for graph viz).
        console: Rich console for styled output.
        verbose: Whether to show verbose output on errors.

    Returns:
        List of saved file names.
    """
    saved_files: list[str] = []
    console.print("\n[bold]Saving outputs...[/bold]")

    if config.outputs.save_transcript:
        path = config.outputs.storage_dir / config.outputs.transcript_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results["markdown_transcript"])
        console.print(f"  [green]>[/green] Transcript: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_debug_log:
        path = config.outputs.storage_dir / config.outputs.debug_log_file
        if path.exists():
            # File was already streamed to disk in real time by DebugLogWriter.
            console.print(f"  [green]>[/green] Debug log: {path.name} (real-time)")
        else:
            # Fallback: write from results dict (memory-only mode)
            with open(path, "w", encoding="utf-8") as f:
                f.write(results.get("debug_log", "No debug log available"))
            console.print(f"  [green]>[/green] Debug log: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_initial_beliefs:
        init_dir = config.outputs.storage_dir / config.outputs.initial_beliefs_dir
        init_dir.mkdir(parents=True, exist_ok=True)
        for agent in controller.agents:
            fname = sanitize_filename(agent.name) + ".json"
            fpath = init_dir / fname
            belief_json = _get_initial_belief_obj(agent)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(belief_json, f, indent=2, ensure_ascii=False)
        console.print(f"  [green]>[/green] Initial beliefs: {config.outputs.initial_beliefs_dir}/")
        saved_files.append(config.outputs.initial_beliefs_dir + "/")

    if config.outputs.save_final_beliefs:
        final_dir = config.outputs.storage_dir / config.outputs.final_beliefs_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        for agent in controller.agents:
            fname = sanitize_filename(agent.name) + ".json"
            fpath = final_dir / fname
            belief_json = agent.get_internal_belief_obj()
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(belief_json, f, indent=2, ensure_ascii=False)
        console.print(f"  [green]>[/green] Final beliefs: {config.outputs.final_beliefs_dir}/")
        saved_files.append(config.outputs.final_beliefs_dir + "/")

    if config.outputs.save_agent_stats:
        path = config.outputs.storage_dir / config.outputs.stats_file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results["agent_stats"], f, indent=2)
        console.print(f"  [green]>[/green] Agent stats: {path.name}")
        saved_files.append(path.name)

    # Always-on: best-agent initial + final belief outputs (JSON + markdown).
    # Produced after agent_stats so we can reference the same performance_score.
    try:
        best_files = _write_best_agent_beliefs(config, results, controller)
        for fname in best_files:
            console.print(f"  [green]>[/green] Best-agent beliefs: {fname}")
            saved_files.append(fname)
    except Exception as e:
        console.print(f"  [yellow]Warning:[/yellow] Could not write best-agent beliefs: {e}")
        if verbose:
            console.print(traceback.format_exc())

    # Generate belief trajectory plot if enabled
    if config.outputs.plot_trajectories:
        try:
            from chal.embeddings.embedding_visualizer import (
                generate_belief_trajectory_plot,
            )
            plot_path = generate_belief_trajectory_plot(config)
            console.print(
                f"  [green]>[/green] Belief trajectory plot: {plot_path.name}"
            )
            saved_files.append(plot_path.name)
        except FileNotFoundError:
            console.print(
                "  [yellow]Warning:[/yellow] Embeddings file not found, skipping visualization"
            )
        except Exception as e:
            console.print(f"  [yellow]Warning:[/yellow] Could not generate plot: {e}")

    # Generate PCA belief trajectory plot if enabled
    if config.outputs.plot_trajectories:
        try:
            from chal.embeddings.embedding_visualizer import (
                generate_pca_trajectory_plot,
            )
            pca_plot_path = generate_pca_trajectory_plot(config)
            console.print(
                f"  [green]>[/green] PCA trajectory plot: {pca_plot_path.name}"
            )
            saved_files.append(pca_plot_path.name)
        except FileNotFoundError:
            console.print(
                "  [yellow]Warning:[/yellow] Embeddings file not found, skipping PCA visualization"
            )
        except Exception as e:
            console.print(f"  [yellow]Warning:[/yellow] Could not generate PCA plot: {e}")

    # Generate interactive graph visualization if enabled
    if config.outputs.generate_graph_visualization:
        try:
            from chal.beliefs.graph_visualizer import export_debate_graph

            graph_path = config.outputs.storage_dir / config.outputs.graph_file
            console.print("  Generating interactive graph...")
            export_debate_graph(
                agents=controller.agents,
                topic=config.topic,
                current_round_pairs=controller.current_round_pairs,
                output_path=graph_path,
            )
            console.print(f"  [green]>[/green] Interactive graph: {graph_path.name}")
            saved_files.append(graph_path.name)
        except Exception as e:
            console.print(
                f"  [yellow]Warning:[/yellow] Could not generate graph visualization: {e}"
            )
            if verbose:
                console.print(traceback.format_exc())

    return saved_files


def _write_best_agent_beliefs(config: DebateConfig, results: dict, controller) -> list[str]:
    """Write ``best_initial_final_beliefs.{json,txt}`` for the top-scoring agent.

    Selection: highest ``performance_score`` in ``results["agent_stats"]``; ties
    broken by first occurrence in ``config.agents`` order.

    JSON payload: topic, best_agent, performance_score, selection_rule,
    initial_belief (parsed from ``agent.all_beliefs_held[0]``), and final_belief
    (from ``agent.get_internal_belief_obj()``). If the initial belief JSON is
    unparseable, the payload's ``initial_belief`` field falls back to
    ``{"error": ..., "raw": <string>}`` so downstream tooling still sees a
    well-shaped document.

    Text payload: reuses the same markdown blocks that populate
    ``initial_beliefs.txt`` / ``final_beliefs.txt`` (i.e.,
    ``results["initial_positions"][idx]`` and
    ``results["final_positions"][idx]``) so the per-agent sections are
    byte-identical to the relevant slice of those files.

    Returns:
        List of output filenames written (relative names, not full paths).
    """
    agent_stats = results.get("agent_stats", {})
    agent_order = [a.name for a in config.agents]
    best_name = select_best_agent(agent_stats, agent_order)

    agents = controller.agents
    best_agent = next((a for a in agents if a.name == best_name), None)
    if best_agent is None:
        raise ValueError(f"best agent '{best_name}' not found among controller.agents")

    # Initial belief: parse the JSON captured immediately after Stage 1.
    all_beliefs_held = getattr(best_agent, "all_beliefs_held", []) or []
    if not all_beliefs_held:
        raise ValueError(f"best_agent '{best_name}' missing initial belief (all_beliefs_held empty)")
    initial_raw = all_beliefs_held[0]
    try:
        initial_belief = json.loads(initial_raw) if isinstance(initial_raw, str) else initial_raw
    except (json.JSONDecodeError, TypeError):
        initial_belief = {"error": "initial belief unparseable", "raw": str(initial_raw)}

    # Final belief: already a dict (the controller stores structured JSON).
    if hasattr(best_agent, "get_internal_belief_obj"):
        final_belief = best_agent.get_internal_belief_obj()
    else:
        final_belief = None

    performance_score = agent_stats.get(best_name, {}).get("performance_score", 0.0)

    json_payload = {
        "topic": config.topic,
        "best_agent": best_name,
        "performance_score": performance_score,
        "selection_rule": "max performance_score; tiebreaker = config.agents order",
        "initial_belief": initial_belief,
        "final_belief": final_belief,
    }

    storage_dir = config.outputs.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    json_path = storage_dir / config.outputs.best_beliefs_json_file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    # Markdown rendering: reuse the exact per-agent text blocks saved into
    # initial_beliefs.txt / final_beliefs.txt (results["initial_positions"][i]
    # and results["final_positions"][i], respectively).
    try:
        best_idx = agent_order.index(best_name)
    except ValueError:
        best_idx = None

    def _get_position(positions_key: str) -> str:
        positions = results.get(positions_key) or []
        if best_idx is not None and 0 <= best_idx < len(positions):
            return positions[best_idx]
        return "(position unavailable)"

    initial_text = _get_position("initial_positions")
    final_text = _get_position("final_positions")

    md_lines = [
        "# Best Agent Beliefs: Initial vs Final",
        "",
        f"**Topic:** {config.topic}",
        f"**Best agent:** {best_name}  (performance score: {performance_score})",
        "",
        "---",
        "",
        "## Initial Belief (Stage 1)",
        "",
        initial_text,
        "",
        "---",
        "",
        f"## Final Belief (after Round {config.max_rounds} Stage 5)",
        "",
        final_text,
        "",
    ]

    text_path = storage_dir / config.outputs.best_beliefs_text_file
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return [json_path.name, text_path.name]


def _get_initial_belief_obj(agent) -> dict:
    """Extract the initial CBS belief dict from an agent's belief history.

    Parses ``agent.all_beliefs_held[0]`` (a JSON string) back to a dict.
    Returns an error-shaped dict if the history is empty or unparseable.
    """
    all_beliefs = getattr(agent, "all_beliefs_held", []) or []
    if not all_beliefs:
        return {"error": "no initial belief recorded"}
    raw = all_beliefs[0]
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"error": "initial belief unparseable", "raw": str(raw)}
