#!/usr/bin/env python
"""
run_debate.py

CLI entry point for running CHAL debates with YAML configuration.

Usage:
    python run_debate.py                           # Uses src/chal/configurations/default.yaml
    python run_debate.py --config quick_test       # Uses src/chal/configurations/quick_test.yaml
    python run_debate.py --config path/to/my.yaml  # Uses custom path
"""

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env before any other imports so API keys are available
load_dotenv(Path(__file__).parent / ".env")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chal.config import load_config
from chal.agents.factory import create_agent_from_config
from chal.agents import prompts
from chal.orchestrator.debate_controller import DebateController
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.embeddings.embedding_visualizer import BeliefTrajectoryPlotter


def main():
    parser = argparse.ArgumentParser(
        description="Run a CHAL philosophical debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_debate.py                     # Use default configuration
  python run_debate.py -c quick_test       # Use quick_test configuration
  python run_debate.py -c my_debate.yaml   # Use custom YAML file
        """
    )
    parser.add_argument(
        '--config', '-c',
        default='default',
        help='Configuration name or path (default: configurations/default.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    args = parser.parse_args()

    # Load configuration
    print(f"📋 Loading configuration: {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nAvailable configurations in 'src/chal/configurations/':")
        config_dir = Path(__file__).parent / "src" / "chal" / "configurations"
        if config_dir.exists():
            for yaml_file in sorted(config_dir.glob("*.yaml")):
                print(f"  - {yaml_file.stem}")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    print(f"\n🎯 Debate: {config.name}")
    print(f"   Topic: {config.topic}")
    print(f"   Rounds: {config.max_rounds}")
    print(f"   Agents: {len(config.agents)}")

    # Ensure storage directory exists
    config.outputs.ensure_storage_dir()
    print(f"   Storage: {config.outputs.storage_dir}")

    # Create agents from config
    agents = []
    personas = {}

    print(f"\n👥 Initializing agents:")
    for agent_cfg in config.agents:
        # Map persona string to actual persona constant
        try:
            persona_obj = getattr(prompts, agent_cfg.persona)
        except AttributeError:
            print(f"❌ Error: Unknown persona '{agent_cfg.persona}'")
            print(f"   Available personas: EMPIRICIST, RATIONALIST, SUPERNATURALIST, SKEPTIC, etc.")
            return 1

        agent = create_agent_from_config(agent_cfg)
        agents.append(agent)
        personas[agent_cfg.name] = persona_obj
        print(f"   ✓ {agent_cfg.name} ({agent_cfg.persona}, {agent_cfg.provider}/{agent_cfg.model})")

    # Create controller
    controller = DebateController(
        agents=agents,
        max_rounds=config.max_rounds,
        config=config
    )

    # Run debate
    print(f"\n🚀 Starting debate...")
    print("=" * 60)

    try:
        results = controller.run(topic=config.topic, personas=personas)
    except Exception as e:
        print(f"\n❌ Error during debate execution: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    # Save outputs based on config
    print(f"\n💾 Saving outputs...")

    if config.outputs.save_synthesis and config.scribe.enabled:
        path = config.outputs.storage_dir / config.outputs.synthesis_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results["synthesis"])
        print(f"   ✓ Synthesis: {path.name}")

    if config.outputs.save_transcript:
        path = config.outputs.storage_dir / config.outputs.transcript_file
        with open(path, "w", encoding="utf-8") as f:
            # Use new markdown_transcript (clean markdown-only output)
            f.write(results.get("markdown_transcript", results["full_transcript"]))
        print(f"   ✓ Transcript: {path.name}")

    if config.outputs.save_debug_log:
        path = config.outputs.storage_dir / config.outputs.debug_log_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(results.get("debug_log", "No debug log available"))
        print(f"   ✓ Debug log: {path.name}")

    if config.outputs.save_initial_beliefs:
        path = config.outputs.storage_dir / config.outputs.initial_beliefs_file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(results["initial_positions"]))
        print(f"   ✓ Initial beliefs: {path.name}")

    if config.outputs.save_final_beliefs:
        path = config.outputs.storage_dir / config.outputs.final_beliefs_file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(results["final_positions"]))
        print(f"   ✓ Final beliefs: {path.name}")

    if config.outputs.save_agent_stats:
        path = config.outputs.storage_dir / config.outputs.stats_file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results["agent_stats"], f, indent=2)
        print(f"   ✓ Agent stats: {path.name}")

    # Generate embeddings and plot if enabled
    if config.outputs.generate_embeddings or config.outputs.plot_trajectories:
        embeddings_path = config.outputs.storage_dir / config.outputs.embeddings_file

        if embeddings_path.exists():
            tracker = BeliefEmbeddingTracker()
            tracker.load_embeddings(embeddings_path)

            if config.outputs.plot_trajectories:
                try:
                    plotter = BeliefTrajectoryPlotter(n_components=2)
                    reduced = plotter.reduce_embeddings(tracker.get_all_embeddings())

                    # Save plot to file instead of showing interactively
                    trajectory_path = config.outputs.storage_dir / config.outputs.trajectory_plot_file
                    plotter.plot_trajectories(reduced, output_path=trajectory_path)
                    print(f"   ✓ Belief trajectory plot saved to {trajectory_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not generate plot: {e}")
        else:
            print(f"   ⚠️  Embeddings file not found, skipping visualization")

    # Generate interactive graph visualization if enabled
    if config.outputs.generate_graph_visualization:
        try:
            from chal.beliefs.graph_visualizer import export_debate_graph

            graph_path = config.outputs.storage_dir / config.outputs.graph_file

            print(f"   Generating interactive graph...")
            export_debate_graph(
                agents=controller.agents,
                topic=config.topic,
                challenge_rebuttal_pairs=controller.challenge_rebuttal_pairs,
                output_path=graph_path
            )

            print(f"   ✓ Interactive graph: {graph_path.name}")
        except Exception as e:
            print(f"   ⚠️  Could not generate graph visualization: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Debate complete!")
    print(f"📊 Results saved to: {config.outputs.storage_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
