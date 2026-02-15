# embedding_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
import umap
from typing import Dict, List, Optional
from itertools import cycle
from pathlib import Path

class BeliefTrajectoryPlotter:
    """
    Reduces high-dimensional belief embeddings to 2D/3D and plots agent belief trajectories
    with arrows and labels for semantic movement visualization.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        from umap import UMAP
        self.reducer = UMAP(n_components=n_components, random_state=random_state)
        self.n_components = n_components

    def flatten_embeddings(self, embeddings: Dict[str, List[np.ndarray]]) -> tuple[np.ndarray, List[str], List[int]]:
        all_vectors = []
        agent_labels = []
        round_indices = []

        for agent_name, vectors in embeddings.items():
            for round_index, vec in enumerate(vectors):
                all_vectors.append(vec)
                agent_labels.append(agent_name)
                round_indices.append(round_index)

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

    def plot_trajectories(self, reduced_by_agent: Dict[str, List[np.ndarray]], output_path: Optional[Path] = None):
        """
        Plots 2D trajectories with color-matched arrows and labeled start/end points.

        Args:
            reduced_by_agent: Dictionary mapping agent names to their reduced coordinate trajectories
            output_path: Optional path to save the plot. If None, displays interactively (not recommended for CLI)
        """
        if self.n_components != 2:
            raise ValueError("Arrowed plotting is currently only supported in 2D.")

        fig, ax = plt.subplots(figsize=(10, 7))

        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        agent_colors = {}

        for agent, coords in reduced_by_agent.items():
            coords = np.array(coords)
            color = next(color_cycle)
            agent_colors[agent] = color

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

        ax.set_title("Agent Belief Trajectories")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend()
        plt.tight_layout()

        # Save to file or show interactively
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            print(f"      [Trajectory] Plot saved to {output_path}")
        else:
            plt.show()