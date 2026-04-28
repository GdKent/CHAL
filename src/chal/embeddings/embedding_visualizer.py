# embedding_visualizer.py

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
    VPacker,
)
from umap import UMAP

PROVIDER_LOGOS: dict[str, str] = {
    "openai": "openai.png",
    "anthropic": "claude.png",
    "google": "gemini.png",
    "ollama": "ollama.png",
    "xai": "grok.png",
    "perplexity": "perplexity.png",
}

# Persona-to-colormap mapping for trajectory plots.
# Single-run plots sample the darkest shade (position 0.95).
# Future multi-run overlays sample different hues across [0.35, 0.95].
PERSONA_COLORMAPS: dict[str, str] = {
    "EMPIRICIST": "RdPu",
    "SUPERNATURALIST": "Purples",
    "SKEPTIC": "Reds",
    "RATIONALIST": "Greens",
    "PHENOMENOLOGIST": "Oranges",
    "PRAGMATIST": "YlOrBr",
    "CONSTRUCTIVIST": "Blues",
    "NIHILIST": "Greys",
    "BAYESIAN": "GnBu",
    "PANPSYCHIST": "YlGn",
    "SIMULATIONIST": "autumn",
    "SYNTHESIST": "cool",
    "NONE": "Greys",
}


def _persona_color(persona: str, position: float = 0.95) -> str | None:
    """Sample a color from a persona's colormap at the given position.

    Args:
        persona: Persona key (e.g. "EMPIRICIST"). Case-insensitive.
        position: Position along the colormap [0, 1]. 0.95 = darkest shade.

    Returns:
        Hex color string, or None if the persona has no mapped colormap.
    """
    cmap_name = PERSONA_COLORMAPS.get(persona.upper())
    if cmap_name is None:
        return None
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(position)
    return f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'


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
        metric: str = "euclidean",
    ):
        """Initialize the plotter with UMAP dimensionality reduction settings.

        Args:
            n_components: Number of output dimensions (2 for 2D plots).
            random_state: Seed for reproducible UMAP embeddings.
            umap_n_neighbors: UMAP n_neighbors parameter (local neighborhood size).
            umap_min_dist: UMAP min_dist parameter (minimum distance in output space).
            metric: Distance metric for UMAP (e.g., "euclidean", "cosine").
        """
        self.reducer = UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
            metric=metric,
        )
        self.n_components = n_components
        self._umap_params = {
            "n_components": n_components,
            "n_neighbors": umap_n_neighbors,
            "min_dist": umap_min_dist,
            "random_state": random_state,
            "metric": metric,
        }

    def flatten_embeddings(self, embeddings: dict[str, list[np.ndarray]]) -> tuple:
        """Flatten per-agent embedding lists into a single matrix with labels.

        Args:
            embeddings: Dict mapping agent names to lists of embedding vectors.

        Returns:
            Tuple of (stacked_vectors, agent_labels, round_indices).

        Raises:
            ValueError: If no embeddings are provided.
        """
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

    def reduce_embeddings(self, embeddings: dict[str, list[np.ndarray]]) -> dict:
        """Reduce high-dimensional embeddings to 2D/3D using UMAP.

        Auto-clamps n_neighbors when the dataset is very small to prevent
        UMAP from crashing.

        Args:
            embeddings: Dict mapping agent names to lists of embedding vectors.

        Returns:
            Dict mapping agent names to lists of reduced coordinate arrays.
        """
        all_vecs, agent_labels, _ = self.flatten_embeddings(embeddings)

        # Auto-clamp n_neighbors when there are very few data points to prevent UMAP crash
        reducer = self.reducer
        if all_vecs.shape[0] <= self.reducer.n_neighbors:
            clamped_neighbors = max(2, all_vecs.shape[0] - 1)
            params = dict(self._umap_params, n_neighbors=clamped_neighbors)
            reducer = UMAP(**params)

        reduced = reducer.fit_transform(all_vecs)

        reduced_by_agent: dict[str, list[np.ndarray]] = {}
        for coord, agent in zip(reduced, agent_labels):
            if agent not in reduced_by_agent:
                reduced_by_agent[agent] = []
            reduced_by_agent[agent].append(coord)

        return reduced_by_agent

    def reduce_embeddings_pca(self, embeddings: dict[str, list[np.ndarray]]) -> dict:
        """Reduce embeddings to 2D using PCA.

        Unlike UMAP, PCA is a linear projection. The same input vector
        always maps to the same 2D point given the same set of principal
        components.
        """
        from sklearn.decomposition import PCA

        all_vecs, agent_labels, _ = self.flatten_embeddings(embeddings)
        pca = PCA(n_components=self.n_components, random_state=self._umap_params["random_state"])
        reduced = pca.fit_transform(all_vecs)

        reduced_by_agent: dict[str, list[np.ndarray]] = {}
        for coord, agent in zip(reduced, agent_labels):
            if agent not in reduced_by_agent:
                reduced_by_agent[agent] = []
            reduced_by_agent[agent].append(coord)

        return reduced_by_agent

    def plot_belief_trajectories(
        self,
        embeddings: dict[str, list[np.ndarray]],
        output_path: Path | None = None,
        title: str | None = None,
        colors: dict[str, str] | None = None,
        figsize: tuple = (10, 7),
        agent_info: dict[str, dict[str, str]] | None = None,
        topic: str | None = None,
        xlabel: str = "UMAP Dimension 1",
        ylabel: str = "UMAP Dimension 2",
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
            agent_info: Optional dict mapping agent names to dicts with 'provider' and 'model' keys.
            topic: Optional debate topic string displayed as legend title.
            xlabel: X-axis label (default: "UMAP Dimension 1").
            ylabel: Y-axis label (default: "UMAP Dimension 2").
        """
        reduced_by_agent = self.reduce_embeddings(embeddings)
        self._plot(reduced_by_agent, output_path, title=title, colors=colors, figsize=figsize, agent_info=agent_info, topic=topic, xlabel=xlabel, ylabel=ylabel)

    def plot_trajectories(self, reduced_by_agent: dict[str, list[np.ndarray]], output_path: Path | None = None, agent_info: dict[str, dict[str, str]] | None = None, topic: str | None = None, xlabel: str = "UMAP Dimension 1", ylabel: str = "UMAP Dimension 2"):
        """
        Plots 2D trajectories from already-reduced embeddings.

        Args:
            reduced_by_agent: Dictionary mapping agent names to their reduced coordinate trajectories.
            output_path: Optional path to save the plot. If None, displays interactively.
            agent_info: Optional dict mapping agent names to dicts with 'provider' and 'model' keys.
            topic: Optional debate topic string displayed as legend title.
            xlabel: X-axis label (default: "UMAP Dimension 1").
            ylabel: Y-axis label (default: "UMAP Dimension 2").
        """
        self._plot(reduced_by_agent, output_path, agent_info=agent_info, topic=topic, xlabel=xlabel, ylabel=ylabel)

    def _load_logo(self, provider: str) -> np.ndarray | None:
        """Load a provider logo PNG from the assets directory."""
        logo_dir = Path(__file__).parent.parent / "assets" / "logos"
        filename = PROVIDER_LOGOS.get(provider)
        if not filename:
            return None
        logo_path = logo_dir / filename
        if not logo_path.exists():
            return None
        return plt.imread(str(logo_path))

    def _plot(
        self,
        reduced_by_agent: dict[str, list[np.ndarray]],
        output_path: Path | None = None,
        title: str | None = None,
        colors: dict[str, str] | None = None,
        figsize: tuple = (10, 7),
        agent_info: dict[str, dict[str, str]] | None = None,
        topic: str | None = None,
        xlabel: str = "UMAP Dimension 1",
        ylabel: str = "UMAP Dimension 2",
    ):
        """Internal plot implementation."""
        if self.n_components != 2:
            raise ValueError("Arrowed plotting is currently only supported in 2D.")

        with plt.rc_context({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm',
        }):
            self._plot_inner(reduced_by_agent, output_path, title, colors, figsize, agent_info, topic, xlabel, ylabel)

    def _plot_inner(
        self,
        reduced_by_agent: dict[str, list[np.ndarray]],
        output_path: Path | None,
        title: str | None,
        colors: dict[str, str] | None,
        figsize: tuple,
        agent_info: dict[str, dict[str, str]] | None,
        topic: str | None = None,
        xlabel: str = "UMAP Dimension 1",
        ylabel: str = "UMAP Dimension 2",
    ):
        """Render the plot (called within an rc_context for font settings)."""
        fig, ax = plt.subplots(figsize=figsize)

        color_cycle = cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ])
        default_colors = {}
        agent_colors = {}

        for agent, coords in reduced_by_agent.items():
            coords = np.array(coords)

            # Use custom color if provided, otherwise try persona colormap,
            # then fall back to the default color cycle.
            if colors and agent in colors:
                color = colors[agent]
            else:
                persona_color = None
                if agent_info and agent in agent_info:
                    persona = agent_info[agent].get("persona", "")
                    if persona:
                        persona_color = _persona_color(persona)
                if persona_color:
                    color = persona_color
                elif agent not in default_colors:
                    default_colors[agent] = next(color_cycle)
                    color = default_colors[agent]
                else:
                    color = default_colors[agent]
            agent_colors[agent] = color

            # Plot the line and points
            if agent_info:
                ax.plot(coords[:, 0], coords[:, 1], marker='o', color=color)
            else:
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

            # Add "Start" and "End" labels, and place provider logo at start
            if agent_info and agent in agent_info:
                # Logo present: put "Start" label above the point, logo to the left
                ax.text(coords[0, 0], coords[0, 1], f"{agent} Start",
                        fontsize=13, fontweight='bold', ha='center', va='bottom', color=color)
                ax.text(coords[-1, 0], coords[-1, 1], f"{agent} End",
                        fontsize=13, fontweight='bold', ha='left', va='top', color=color)

                provider = agent_info[agent].get("provider", "")
                logo_img = self._load_logo(provider)
                if logo_img is not None:
                    im = OffsetImage(logo_img, zoom=0.04)
                    ab = AnnotationBbox(
                        im, (coords[0, 0], coords[0, 1]),
                        xybox=(-30, -15),
                        xycoords='data',
                        boxcoords="offset points",
                        frameon=False,
                    )
                    ax.add_artist(ab)
                elif provider:
                    # Fallback: colored circle with letter
                    ax.plot(coords[0, 0], coords[0, 1], 'o', color=color, markersize=15, zorder=5)
                    ax.text(coords[0, 0], coords[0, 1], provider[0].upper(),
                            fontsize=10, ha='center', va='center', color='white',
                            fontweight='bold', zorder=6)
            else:
                ax.text(coords[0, 0], coords[0, 1], f"{agent} Start",
                        fontsize=11, fontweight='bold', ha='right', va='bottom', color=color)
                ax.text(coords[-1, 0], coords[-1, 1], f"{agent} End",
                        fontsize=11, fontweight='bold', ha='left', va='top', color=color)

        ax.set_title(title or "Agent Belief Trajectories", fontsize=18, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        if agent_info:
            # Build custom legend with logos
            legend_rows = []

            # Add topic title row at top of legend if provided
            if topic:
                topic_text = TextArea(
                    f'Topic: "{topic}"',
                    textprops=dict(fontsize=12, fontweight='bold', color='black'),
                )
                legend_rows.append(topic_text)

            for agent in reduced_by_agent:
                color = agent_colors[agent]
                info = agent_info.get(agent, {})
                provider = info.get("provider", "")
                model = info.get("model", "")
                row_children = []

                # Add logo if available
                logo_img = self._load_logo(provider)
                if logo_img is not None:
                    logo_box = OffsetImage(logo_img, zoom=0.04)
                    row_children.append(logo_box)

                # Add text label
                label_text = f"{agent} ({model})" if model else agent
                text_box = TextArea(label_text, textprops=dict(color=color, fontsize=11, fontweight='bold'))
                row_children.append(text_box)

                row = HPacker(children=row_children, align="center", pad=2, sep=4)
                legend_rows.append(row)

            legend_box = VPacker(children=legend_rows, align="left", pad=4, sep=4)
            anchored = AnchoredOffsetbox(
                loc='upper right',
                child=legend_box,
                pad=0.5,
                frameon=True,
                borderpad=0.5,
                prop=dict(size=9),
            )
            anchored.patch.set_facecolor('white')
            anchored.patch.set_alpha(0.9)
            anchored.patch.set_boxstyle("round,pad=0.3")
            ax.add_artist(anchored)
        else:
            ax.legend()
        plt.tight_layout()

        # Save to file or show interactively
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()


def generate_belief_trajectory_plot(
    config,
    output_path: Path | None = None,
) -> Path:
    """Generate a belief trajectory plot from saved embeddings.

    Loads embeddings from disk, reduces them with UMAP, and renders a
    2D trajectory plot with provider logos and agent metadata.

    Args:
        config: A DebateConfig instance with agents, topic, and output settings.
        output_path: Optional override for the output file path.
            Defaults to config.outputs.storage_dir / config.outputs.trajectory_plot_file.

    Returns:
        The Path where the plot was saved.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker

    embeddings_path = config.outputs.storage_dir / config.outputs.embeddings_file
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    tracker = BeliefEmbeddingTracker()
    tracker.load_embeddings(embeddings_path)

    plotter = BeliefTrajectoryPlotter(n_components=2)
    reduced = plotter.reduce_embeddings(tracker.get_all_embeddings())

    # Prefer metadata saved with embeddings; fall back to config
    saved_meta = getattr(tracker, 'metadata', {})
    agent_info = saved_meta.get("agent_info") or {
        a.name: {"model": a.model, "provider": a.provider, "persona": a.persona}
        for a in config.agents
    }
    topic = saved_meta.get("topic") or config.topic

    dest = output_path or (config.outputs.storage_dir / config.outputs.trajectory_plot_file)
    plotter.plot_trajectories(reduced, output_path=dest, agent_info=agent_info, topic=topic)

    return dest


def generate_pca_trajectory_plot(
    config,
    output_path: Path | None = None,
) -> Path:
    """Generate a PCA belief trajectory plot from saved embeddings.

    Identical to generate_belief_trajectory_plot() but uses PCA instead
    of UMAP for dimensionality reduction.

    Args:
        config: A DebateConfig instance with agents, topic, and output settings.
        output_path: Optional override for the output file path.
            Defaults to config.outputs.storage_dir / config.outputs.pca_plot_file.

    Returns:
        The Path where the plot was saved.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker

    embeddings_path = config.outputs.storage_dir / config.outputs.embeddings_file
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    tracker = BeliefEmbeddingTracker()
    tracker.load_embeddings(embeddings_path)

    plotter = BeliefTrajectoryPlotter(n_components=2)
    reduced = plotter.reduce_embeddings_pca(tracker.get_all_embeddings())

    saved_meta = getattr(tracker, 'metadata', {})
    agent_info = saved_meta.get("agent_info") or {
        a.name: {"model": a.model, "provider": a.provider, "persona": a.persona}
        for a in config.agents
    }
    topic = saved_meta.get("topic") or config.topic

    dest = output_path or (config.outputs.storage_dir / config.outputs.pca_plot_file)
    plotter.plot_trajectories(
        reduced,
        output_path=dest,
        agent_info=agent_info,
        topic=topic,
        xlabel="Principal Component 1",
        ylabel="Principal Component 2",
    )

    return dest
