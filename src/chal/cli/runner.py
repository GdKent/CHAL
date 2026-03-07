"""
runner.py

Shared debate execution and output-saving logic used by both the interactive
wizard and headless CLI mode.
"""

import json
import time
import traceback

from rich.console import Console

from chal.config import DebateConfig
from chal.agents.factory import create_agent_from_config
from chal.agents.epistemic_personas import get_persona
from chal.orchestrator.debate_controller import DebateController
from chal.cli.display import DebateDisplay
from chal.cli.api_keys import validate_api_keys


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
        interactive: Whether the session is interactive (enables roadmap
            review for moderated debates).

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

        agent = create_agent_from_config(agent_cfg)
        agents.append(agent)
        personas[agent_cfg.name] = persona_obj
        console.print(
            f"  [green]>[/green] {agent_cfg.name} "
            f"({agent_cfg.persona}, {agent_cfg.provider}/{agent_cfg.model})"
        )

    # Create controller
    controller = DebateController(
        agents=agents,
        max_rounds=config.max_rounds,
        config=config,
    )

    # Interactive roadmap review (moderated mode only)
    if (
        interactive
        and config.stage2_mode == "moderated"
        and controller.moderator is not None
        and controller.roadmap is not None
    ):
        from chal.cli.roadmap_review import run_roadmap_review

        try:
            agent_persona_labels = [
                getattr(a, "persona_label", a.name) for a in agents
            ]
            _, new_rounds, was_modified = run_roadmap_review(
                console=console,
                moderator=controller.moderator,
                topic=config.topic,
                num_rounds=config.max_rounds,
                agent_personas=agent_persona_labels,
            )
            if new_rounds != config.max_rounds:
                config.max_rounds = new_rounds
                controller.max_rounds = new_rounds
            controller.roadmap_user_modified = was_modified
        except KeyboardInterrupt:
            console.print("\n[dim]Roadmap review cancelled.[/dim]")

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
        console.print(f"\n[red]Error during debate execution:[/red] {e}")
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

    if config.outputs.save_synthesis and config.scribe.enabled:
        path = config.outputs.storage_dir / config.outputs.synthesis_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results["synthesis"])
        console.print(f"  [green]>[/green] Synthesis: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_transcript:
        path = config.outputs.storage_dir / config.outputs.transcript_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results.get("markdown_transcript", results["full_transcript"]))
        console.print(f"  [green]>[/green] Transcript: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_debug_log:
        path = config.outputs.storage_dir / config.outputs.debug_log_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results.get("debug_log", "No debug log available"))
        console.print(f"  [green]>[/green] Debug log: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_initial_beliefs:
        path = config.outputs.storage_dir / config.outputs.initial_beliefs_file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(results["initial_positions"]))
        console.print(f"  [green]>[/green] Initial beliefs: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_final_beliefs:
        path = config.outputs.storage_dir / config.outputs.final_beliefs_file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(results["final_positions"]))
        console.print(f"  [green]>[/green] Final beliefs: {path.name}")
        saved_files.append(path.name)

    if config.outputs.save_agent_stats:
        path = config.outputs.storage_dir / config.outputs.stats_file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results["agent_stats"], f, indent=2)
        console.print(f"  [green]>[/green] Agent stats: {path.name}")
        saved_files.append(path.name)

    return saved_files

    # Generate embeddings and plot if enabled
    if config.outputs.generate_embeddings or config.outputs.plot_trajectories:
        embeddings_path = config.outputs.storage_dir / config.outputs.embeddings_file
        if embeddings_path.exists():
            from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
            from chal.embeddings.embedding_visualizer import BeliefTrajectoryPlotter

            tracker = BeliefEmbeddingTracker()
            tracker.load_embeddings(embeddings_path)

            if config.outputs.plot_trajectories:
                try:
                    plotter = BeliefTrajectoryPlotter(n_components=2)
                    reduced = plotter.reduce_embeddings(tracker.get_all_embeddings())
                    trajectory_path = (
                        config.outputs.storage_dir / config.outputs.trajectory_plot_file
                    )
                    plotter.plot_trajectories(reduced, output_path=trajectory_path)
                    console.print(
                        f"  [green]>[/green] Belief trajectory plot: {trajectory_path.name}"
                    )
                except Exception as e:
                    console.print(f"  [yellow]Warning:[/yellow] Could not generate plot: {e}")
        else:
            console.print(
                "  [yellow]Warning:[/yellow] Embeddings file not found, skipping visualization"
            )

    # Generate interactive graph visualization if enabled
    if config.outputs.generate_graph_visualization:
        try:
            from chal.beliefs.graph_visualizer import export_debate_graph

            graph_path = config.outputs.storage_dir / config.outputs.graph_file
            console.print("  Generating interactive graph...")
            export_debate_graph(
                agents=controller.agents,
                topic=config.topic,
                challenge_rebuttal_pairs=controller.challenge_rebuttal_pairs,
                output_path=graph_path,
            )
            console.print(f"  [green]>[/green] Interactive graph: {graph_path.name}")
        except Exception as e:
            console.print(
                f"  [yellow]Warning:[/yellow] Could not generate graph visualization: {e}"
            )
            if verbose:
                console.print(traceback.format_exc())
