# embedding_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from typing import Dict, List, Optional
from itertools import cycle
from pathlib import Path

class BeliefTrajectoryPlotter:
    """
    Reduces high-dimensional belief embeddings to 2D/3D and plots agent belief trajectories
    with arrows and labels for semantic movement visualization.
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 42,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ):
        self.reducer = UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        self.n_components = n_components

    def flatten_embeddings(self, embeddings: Dict[str, List[np.ndarray]]) -> tuple:
        all_vectors = []
        agent_labels = []
        round_indices = []

        for agent_name, vectors in embeddings.items():
            for round_index, vec in enumerate(vectors):
                all_vectors.append(vec)
                agent_labels.append(agent_name)
                round_indices.append(round_index)

        if not all_vectors:
            raise ValueError("No embeddings to process. Provide at least one agent with embeddings.")

        return np.stack(all_vectors), agent_labels, round_indices

    def reduce_embeddings(self, embeddings: Dict[str, List[np.ndarray]]) -> dict:
        all_vecs, agent_labels, _ = self.flatten_embeddings(embeddings)
        reduced = self.reducer.fit_transform(all_vecs)

        reduced_by_agent: Dict[str, List[np.ndarray]] = {}
        for coord, agent in zip(reduced, agent_labels):
            if agent not in reduced_by_agent:
                reduced_by_agent[agent] = []
            reduced_by_agent[agent].append(coord)

        return reduced_by_agent

    def plot_belief_trajectories(
        self,
        embeddings: Dict[str, List[np.ndarray]],
        output_path: Optional[Path] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize: tuple = (10, 7),
    ):
        """
        Reduces high-dimensional embeddings and plots 2D trajectories.

        Combines reduction and plotting into a single call for convenience.

        Args:
            embeddings: Raw high-dimensional embeddings dict {agent_name: [vectors]}.
            output_path: Path to save the plot. If None, displays interactively.
            title: Optional custom title for the plot.
            colors: Optional dict mapping agent names to color strings.
            figsize: Figure size tuple (width, height).
        """
        reduced_by_agent = self.reduce_embeddings(embeddings)
        self._plot(reduced_by_agent, output_path, title=title, colors=colors, figsize=figsize)

    def plot_trajectories(self, reduced_by_agent: Dict[str, List[np.ndarray]], output_path: Optional[Path] = None):
        """
        Plots 2D trajectories from already-reduced embeddings.

        Args:
            reduced_by_agent: Dictionary mapping agent names to their reduced coordinate trajectories.
            output_path: Optional path to save the plot. If None, displays interactively.
        """
        self._plot(reduced_by_agent, output_path)

    def _plot(
        self,
        reduced_by_agent: Dict[str, List[np.ndarray]],
        output_path: Optional[Path] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize: tuple = (10, 7),
    ):
        """Internal plot implementation."""
        if self.n_components != 2:
            raise ValueError("Arrowed plotting is currently only supported in 2D.")

        fig, ax = plt.subplots(figsize=figsize)

        color_cycle = cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ])
        default_colors = {}

        for agent, coords in reduced_by_agent.items():
            coords = np.array(coords)

            # Use custom color if provided, otherwise use default cycle
            if colors and agent in colors:
                color = colors[agent]
            else:
                if agent not in default_colors:
                    default_colors[agent] = next(color_cycle)
                color = default_colors[agent]

            # Plot the line and points
            ax.plot(coords[:, 0], coords[:, 1], marker='o', label=agent, color=color)

            # Draw arrows between embeddings
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                ax.annotate(
                    '', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                    annotation_clip=False
                )

            # Add "Start" and "End" labels
            ax.text(coords[0, 0], coords[0, 1], f"{agent} Start", fontsize=9, ha='right', va='bottom', color=color)
            ax.text(coords[-1, 0], coords[-1, 1], f"{agent} End", fontsize=9, ha='left', va='top', color=color)

        ax.set_title(title or "Agent Belief Trajectories")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend()
        plt.tight_layout()

        # Save to file or show interactively
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()
