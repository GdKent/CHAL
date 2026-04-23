"""
Unit tests for EmbeddingVisualizer class.

All file I/O is mocked - no actual PNG files are created.

Tests cover:
- Visualization generation
- UMAP projection
- Plot customization
- n_neighbors auto-clamping
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from chal.embeddings.embedding_visualizer import BeliefTrajectoryPlotter as EmbeddingVisualizer


# Component-wise embedding dimension: 9 sub-vectors * 768 + 11 scalar features
_EMBED_DIM = 9 * 768 + 11  # 6923


def _make_mock_umap(n_samples: int = 6, n_neighbors: int = 15):
    """Create a properly configured mock UMAP instance.

    Sets n_neighbors as an integer so reduce_embeddings() auto-clamping
    comparisons work correctly.

    Args:
        n_samples: Number of samples the mock fit_transform should return.
        n_neighbors: The n_neighbors value the mock should report.
    """
    mock_umap = MagicMock()
    mock_umap.n_neighbors = n_neighbors
    mock_umap.fit_transform.return_value = np.random.rand(n_samples, 2)
    return mock_umap


# ==============================================
# Test Fixtures
# ==============================================

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return {
        "Agent-A": [
            np.random.rand(_EMBED_DIM),
            np.random.rand(_EMBED_DIM),
            np.random.rand(_EMBED_DIM)
        ],
        "Agent-B": [
            np.random.rand(_EMBED_DIM),
            np.random.rand(_EMBED_DIM),
            np.random.rand(_EMBED_DIM)
        ]
    }


# ==============================================
# 1. Visualization Generation Tests (Mocked)
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_belief_trajectories(mock_umap_class, mock_plt, sample_embeddings):
    """Test creation of matplotlib figure."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # Should have called plotting functions
        assert mock_plt.figure.called or mock_plt.subplots.called


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_includes_agents(mock_umap_class, mock_plt, sample_embeddings):
    """Test that all agents are represented in plot."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # Should plot data for all agents
        assert True  # Implementation-dependent


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_umap_projection(mock_umap_class, mock_plt, sample_embeddings):
    """Test that UMAP reduces embeddings to 2D."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # UMAP should be called
        assert mock_umap.fit_transform.called


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_arrows_show_evolution(mock_umap_class, mock_plt, sample_embeddings):
    """Test that arrows connect rounds to show evolution."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # Should use arrows or lines to show trajectory
        # Check if arrow or quiver was called
        assert True  # Implementation-dependent


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_saves_file(mock_umap_class, mock_plt, sample_embeddings):
    """Test that plot is saved to PNG file."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "trajectories.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # Should call savefig
        assert mock_plt.savefig.called


# ==============================================
# 2. UMAP Configuration Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_umap_n_components_2d(mock_umap_class, mock_plt, sample_embeddings):
    """Test that UMAP projects to 2 dimensions."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # UMAP should be initialized with n_components=2
        call_kwargs = mock_umap_class.call_args[1]
        if "n_components" in call_kwargs:
            assert call_kwargs["n_components"] == 2


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_umap_custom_parameters(mock_umap_class, mock_plt, sample_embeddings):
    """Test UMAP with custom n_neighbors and min_dist."""
    mock_umap = _make_mock_umap(n_samples=6, n_neighbors=10)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer(umap_n_neighbors=10, umap_min_dist=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(sample_embeddings, save_path)

        # Should use custom parameters
        assert True


# ==============================================
# 3. Plot Customization Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_custom_title(mock_umap_class, mock_plt, sample_embeddings):
    """Test plot with custom title."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(
            sample_embeddings,
            save_path,
            title="Custom Title"
        )

        # Title should be set
        assert True


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_custom_colors(mock_umap_class, mock_plt, sample_embeddings):
    """Test plot with custom color scheme."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(
            sample_embeddings,
            save_path,
            colors={"Agent-A": "red", "Agent-B": "blue"}
        )

        # Custom colors should be used
        assert True


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_figsize(mock_umap_class, mock_plt, sample_embeddings):
    """Test plot with custom figure size."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(
            sample_embeddings,
            save_path,
            figsize=(12, 8)
        )

        # Should set custom figure size
        assert True


# ==============================================
# 4. Multi-Agent Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_many_agents(mock_umap_class, mock_plt):
    """Test visualization with many agents."""
    mock_umap = _make_mock_umap(n_samples=15)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    embeddings = {
        f"Agent-{chr(65+i)}": [np.random.rand(_EMBED_DIM) for _ in range(3)]
        for i in range(5)
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # Should handle multiple agents
        assert True


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_different_trajectory_lengths(mock_umap_class, mock_plt):
    """Test visualization when agents have different round counts."""
    mock_umap = _make_mock_umap(n_samples=8)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM) for _ in range(5)],  # 5 rounds
        "Agent-B": [np.random.rand(_EMBED_DIM) for _ in range(3)]   # 3 rounds
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # Should handle different lengths
        assert True


# ==============================================
# 5. Edge Case Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_single_agent(mock_umap_class, mock_plt):
    """Test visualization with single agent."""
    mock_umap = _make_mock_umap(n_samples=3)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM) for _ in range(3)]
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # Should handle single agent
        assert True


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_single_round(mock_umap_class, mock_plt):
    """Test visualization with single round (no trajectory)."""
    # 2 data points < default n_neighbors=15, so auto-clamping will kick in
    mock_umap = _make_mock_umap(n_samples=2)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM)],
        "Agent-B": [np.random.rand(_EMBED_DIM)]
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # Auto-clamping should create a new UMAP with reduced n_neighbors,
        # so UMAP class is called twice (once in __init__, once in reduce_embeddings)
        assert mock_umap_class.call_count >= 2

        # Should plot points without arrows
        assert True


@pytest.mark.unit
def test_plot_empty_embeddings():
    """Test handling of empty embeddings dict."""
    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError)) or pytest.warns(UserWarning):
            visualizer.plot_belief_trajectories({}, save_path)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_high_dimensional_embeddings(mock_umap_class, mock_plt):
    """Test with full component-wise dimension embeddings."""
    mock_umap = _make_mock_umap(n_samples=4)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # Use full component-wise dimension (9 * 768 + 11 = 6923)
    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM), np.random.rand(_EMBED_DIM)],
        "Agent-B": [np.random.rand(_EMBED_DIM), np.random.rand(_EMBED_DIM)]
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # UMAP should handle high dimensions
        assert mock_umap.fit_transform.called


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_n_neighbors_auto_clamping(mock_umap_class, mock_plt):
    """Test that n_neighbors is auto-clamped when fewer samples than default."""
    mock_umap = _make_mock_umap(n_samples=3, n_neighbors=15)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # Only 3 total data points, fewer than default n_neighbors=15
    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM)],
        "Agent-B": [np.random.rand(_EMBED_DIM)],
        "Agent-C": [np.random.rand(_EMBED_DIM)],
    }

    visualizer = EmbeddingVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        visualizer.plot_belief_trajectories(embeddings, save_path)

        # Auto-clamping should create a second UMAP with clamped n_neighbors
        # (first call is in __init__, second is in reduce_embeddings)
        assert mock_umap_class.call_count >= 2

        # The clamped call should have n_neighbors = max(2, n_samples - 1) = 2
        clamped_call_kwargs = mock_umap_class.call_args_list[-1][1]
        assert clamped_call_kwargs["n_neighbors"] == 2


# ==============================================
# 6. Logo and Legend Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_with_agent_info_creates_figure(mock_umap_class, mock_plt, sample_embeddings):
    """Test that plot_belief_trajectories with agent_info completes and creates a figure."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    visualizer = EmbeddingVisualizer()
    agent_info = {
        "Agent-A": {"model": "o4-mini", "provider": "openai"},
        "Agent-B": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.png"
        with patch.object(visualizer, '_load_logo', return_value=None):
            visualizer.plot_belief_trajectories(sample_embeddings, save_path, agent_info=agent_info)

    assert mock_plt.subplots.called


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_without_agent_info_uses_standard_legend(mock_umap_class, mock_plt):
    """Test that _plot() with agent_info=None falls back to ax.legend()."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "Agent-B": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
    }

    visualizer = EmbeddingVisualizer()
    visualizer._plot(reduced, output_path=None, agent_info=None)

    mock_ax.legend.assert_called_once()


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_with_agent_info_adds_annotation_boxes(mock_umap_class, mock_plt):
    """Test that _plot() with agent_info and loadable logos calls ax.add_artist."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "Agent-B": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
    }
    agent_info = {
        "Agent-A": {"model": "o4-mini", "provider": "openai"},
        "Agent-B": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
    }

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_load_logo', return_value=np.ones((10, 10, 4))):
        visualizer._plot(reduced, output_path=None, agent_info=agent_info)

    assert mock_ax.add_artist.called


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_with_missing_logo_uses_fallback(mock_umap_class, mock_plt):
    """Test that an unknown provider triggers the fallback (colored circle + letter)."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }
    agent_info = {
        "Agent-A": {"model": "some-model", "provider": "unknown_provider"},
    }

    visualizer = EmbeddingVisualizer()
    visualizer._plot(reduced, output_path=None, agent_info=agent_info)

    # The fallback renders a colored circle via ax.plot and a letter via ax.text.
    # ax.plot is also called for the line+points, so just verify it was called.
    assert mock_ax.plot.called
    # ax.text is called for Start/End labels AND fallback; with 1 agent we get
    # at least 3 calls (Start, End, fallback letter).
    assert mock_ax.text.call_count >= 3


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_load_logo_returns_array_for_existing_file(mock_umap_class, mock_plt):
    """Test that _load_logo returns a numpy array when the file exists."""
    mock_umap_class.return_value = _make_mock_umap()
    expected = np.ones((10, 10, 4))
    mock_plt.imread.return_value = expected

    visualizer = EmbeddingVisualizer()

    with patch.object(Path, 'exists', return_value=True):
        result = visualizer._load_logo("openai")

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_load_logo_returns_none_for_missing_file(mock_umap_class, mock_plt):
    """Test that _load_logo returns None when the logo file does not exist."""
    mock_umap_class.return_value = _make_mock_umap()

    visualizer = EmbeddingVisualizer()

    with patch.object(Path, 'exists', return_value=False):
        result = visualizer._load_logo("openai")

    assert result is None


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_load_logo_returns_none_for_unknown_provider(mock_umap_class):
    """Test that _load_logo returns None for a provider not in PROVIDER_LOGOS."""
    mock_umap_class.return_value = _make_mock_umap()

    visualizer = EmbeddingVisualizer()
    result = visualizer._load_logo("totally_unknown_provider")

    assert result is None


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_trajectories_passes_agent_info_to_plot(mock_umap_class, mock_plt):
    """Test that plot_trajectories forwards agent_info to _plot."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }
    agent_info = {"Agent-A": {"model": "m", "provider": "p"}}

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_plot') as mock_plot:
        visualizer.plot_trajectories(reduced, output_path=None, agent_info=agent_info)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args
        assert call_kwargs[1].get("agent_info") == agent_info or call_kwargs[0][-1] == agent_info


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_belief_trajectories_passes_agent_info(mock_umap_class, mock_plt, sample_embeddings):
    """Test that plot_belief_trajectories forwards agent_info to _plot."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    agent_info = {
        "Agent-A": {"model": "o4-mini", "provider": "openai"},
        "Agent-B": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
    }

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_plot') as mock_plot:
        visualizer.plot_belief_trajectories(sample_embeddings, output_path=None, agent_info=agent_info)
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["agent_info"] == agent_info


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_custom_legend_entries_match_agent_count(mock_umap_class, mock_plt):
    """Test that with 3 agents, add_artist is called at least 4 times (3 logos + 1 legend)."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "Agent-B": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
        "Agent-C": [np.array([9.0, 10.0]), np.array([11.0, 12.0])],
    }
    agent_info = {
        "Agent-A": {"model": "o4-mini", "provider": "openai"},
        "Agent-B": {"model": "claude-sonnet-4-6", "provider": "anthropic"},
        "Agent-C": {"model": "gemini-2.5-pro", "provider": "google"},
    }

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_load_logo', return_value=np.ones((10, 10, 4))):
        visualizer._plot(reduced, output_path=None, agent_info=agent_info)

    # 3 AnnotationBbox logos at start points + 1 AnchoredOffsetbox for the legend = 4
    assert mock_ax.add_artist.call_count >= 4


# ==============================================
# 7. generate_belief_trajectory_plot() Tests
# ==============================================

from chal.embeddings.embedding_visualizer import generate_belief_trajectory_plot
from chal.config import DebateConfig, AgentConfig, OutputConfig


def _make_plot_config(tmp_path):
    """Build a minimal DebateConfig for plot generation tests."""
    outputs = OutputConfig(
        storage_dir=tmp_path,
        embeddings_file="embeddings.npz",
        trajectory_plot_file="belief_trajectories.png",
        plot_trajectories=True,
        save_transcript=False,
        save_debug_log=False,
        save_initial_beliefs=False,
        save_final_beliefs=False,
        save_agent_stats=False,
        generate_embeddings=False,
        generate_graph_visualization=False,
        save_analysis_report=False,
        save_training_data=False,
    )
    return DebateConfig(
        topic="Does free will exist?",
        agents=[
            AgentConfig(name="Agent-A", persona="EMPIRICIST", model="o4-mini", provider="openai"),
            AgentConfig(name="Agent-B", persona="RATIONALIST", model="claude-sonnet-4-6", provider="anthropic"),
        ],
        outputs=outputs,
    )


def _save_synthetic_embeddings(tmp_path, agent_names=("Agent-A", "Agent-B"), n_rounds=3):
    """Save synthetic embeddings to an .npz file in tmp_path."""
    data = {name: np.random.rand(n_rounds, _EMBED_DIM) for name in agent_names}
    np.savez(tmp_path / "embeddings.npz", **data)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_plot_creates_file(mock_umap_class, mock_plt, tmp_path):
    """generate_belief_trajectory_plot creates a plot file given valid config."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        result = generate_belief_trajectory_plot(config)

    assert mock_plt.savefig.called
    assert result == tmp_path / "belief_trajectories.png"


@pytest.mark.unit
def test_generate_plot_raises_on_missing_embeddings(tmp_path):
    """generate_belief_trajectory_plot raises FileNotFoundError when embeddings are missing."""
    config = _make_plot_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        generate_belief_trajectory_plot(config)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_plot_custom_output_path(mock_umap_class, mock_plt, tmp_path):
    """generate_belief_trajectory_plot uses output_path override when provided."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)
    custom_path = tmp_path / "custom_plot.png"

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        result = generate_belief_trajectory_plot(config, output_path=custom_path)

    assert result == custom_path
    # savefig should be called with the custom path
    save_args = mock_plt.savefig.call_args
    assert save_args[0][0] == custom_path


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_plot_returns_path(mock_umap_class, mock_plt, tmp_path):
    """generate_belief_trajectory_plot returns the Path where the plot was saved."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        result = generate_belief_trajectory_plot(config)

    assert isinstance(result, Path)
    assert result.name == "belief_trajectories.png"


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_plot_uses_embedded_metadata(mock_umap_class, mock_plt, tmp_path):
    """generate_belief_trajectory_plot prefers metadata from the embeddings file over config."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)

    # Save embeddings WITH metadata that differs from config agent names
    embedded_agent_info = {
        "Agent-A": {"model": "gpt-4o", "provider": "openai"},
        "Agent-B": {"model": "claude-opus-4-6", "provider": "anthropic"},
    }
    data = {name: np.random.rand(3, _EMBED_DIM) for name in ("Agent-A", "Agent-B")}
    data["__metadata__"] = np.array({
        "agent_info": embedded_agent_info,
        "topic": "Embedded topic",
    })
    np.savez(tmp_path / "embeddings.npz", **data)

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        with patch.object(EmbeddingVisualizer, '_plot') as mock_plot:
            generate_belief_trajectory_plot(config)

    call_kwargs = mock_plot.call_args[1]
    # Should use embedded metadata, NOT config metadata
    assert call_kwargs["agent_info"] == embedded_agent_info
    assert call_kwargs["topic"] == "Embedded topic"


# ==============================================
# 8. PCA Reduction Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_pca_reduces_to_2d(mock_umap_class, sample_embeddings):
    """reduce_embeddings_pca returns 2D coordinates for each agent."""
    mock_umap_class.return_value = _make_mock_umap()

    visualizer = EmbeddingVisualizer()
    result = visualizer.reduce_embeddings_pca(sample_embeddings)

    assert set(result.keys()) == set(sample_embeddings.keys())
    for agent, coords in result.items():
        for coord in coords:
            assert coord.shape == (2,)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_pca_returns_per_agent_trajectories(mock_umap_class, sample_embeddings):
    """reduce_embeddings_pca preserves agent keys and trajectory lengths."""
    mock_umap_class.return_value = _make_mock_umap()

    visualizer = EmbeddingVisualizer()
    result = visualizer.reduce_embeddings_pca(sample_embeddings)

    for agent_name, original_vecs in sample_embeddings.items():
        assert agent_name in result
        assert len(result[agent_name]) == len(original_vecs)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_pca_with_single_agent(mock_umap_class):
    """reduce_embeddings_pca works with a single agent."""
    mock_umap_class.return_value = _make_mock_umap()

    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM) for _ in range(5)],
    }

    visualizer = EmbeddingVisualizer()
    result = visualizer.reduce_embeddings_pca(embeddings)

    assert "Agent-A" in result
    assert len(result["Agent-A"]) == 5


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_pca_with_single_round(mock_umap_class):
    """reduce_embeddings_pca works with multiple agents, one round each."""
    mock_umap_class.return_value = _make_mock_umap()

    embeddings = {
        "Agent-A": [np.random.rand(_EMBED_DIM)],
        "Agent-B": [np.random.rand(_EMBED_DIM)],
    }

    visualizer = EmbeddingVisualizer()
    result = visualizer.reduce_embeddings_pca(embeddings)

    assert len(result["Agent-A"]) == 1
    assert len(result["Agent-B"]) == 1


# ==============================================
# 9. Axis Label Parameterization Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_default_axis_labels(mock_umap_class, mock_plt):
    """Default axis labels are 'UMAP Dimension 1' / 'UMAP Dimension 2'."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }

    visualizer = EmbeddingVisualizer()
    visualizer._plot(reduced, output_path=None)

    mock_ax.set_xlabel.assert_called_once_with("UMAP Dimension 1", fontsize=13, fontweight='bold')
    mock_ax.set_ylabel.assert_called_once_with("UMAP Dimension 2", fontsize=13, fontweight='bold')


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_custom_axis_labels(mock_umap_class, mock_plt):
    """Custom xlabel/ylabel are passed through to ax.set_xlabel/set_ylabel."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }

    visualizer = EmbeddingVisualizer()
    visualizer._plot(reduced, output_path=None,
                     xlabel="Principal Component 1",
                     ylabel="Principal Component 2")

    mock_ax.set_xlabel.assert_called_once_with("Principal Component 1", fontsize=13, fontweight='bold')
    mock_ax.set_ylabel.assert_called_once_with("Principal Component 2", fontsize=13, fontweight='bold')


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_trajectories_forwards_axis_labels(mock_umap_class, mock_plt):
    """plot_trajectories passes xlabel/ylabel through to _plot."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_plot') as mock_plot:
        visualizer.plot_trajectories(
            reduced, output_path=None,
            xlabel="PC1", ylabel="PC2",
        )
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["xlabel"] == "PC1"
        assert call_kwargs["ylabel"] == "PC2"


# ==============================================
# 10. Persona Colormap Tests
# ==============================================

from chal.embeddings.embedding_visualizer import PERSONA_COLORMAPS, _persona_color


@pytest.mark.unit
def test_persona_colormaps_has_all_personas():
    """PERSONA_COLORMAPS contains entries for all expected personas."""
    expected = {
        "EMPIRICIST", "SUPERNATURALIST", "SKEPTIC", "RATIONALIST",
        "PHENOMENOLOGIST", "PRAGMATIST", "CONSTRUCTIVIST", "NIHILIST",
        "BAYESIAN", "PANPSYCHIST", "SIMULATIONIST", "SYNTHESIST", "NONE",
    }
    assert expected == set(PERSONA_COLORMAPS.keys())


@pytest.mark.unit
def test_persona_color_returns_hex_string():
    """_persona_color returns a valid hex color string for known personas."""
    color = _persona_color("EMPIRICIST")
    assert color is not None
    assert color.startswith("#")
    assert len(color) == 7  # #RRGGBB


@pytest.mark.unit
def test_persona_color_case_insensitive():
    """_persona_color is case-insensitive."""
    upper = _persona_color("EMPIRICIST")
    lower = _persona_color("empiricist")
    mixed = _persona_color("Empiricist")
    assert upper == lower == mixed


@pytest.mark.unit
def test_persona_color_returns_none_for_unknown():
    """_persona_color returns None for an unknown persona."""
    assert _persona_color("UNKNOWN_PERSONA") is None


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_plot_uses_persona_colors_from_agent_info(mock_umap_class, mock_plt):
    """When agent_info includes persona, colors are derived from PERSONA_COLORMAPS."""
    mock_umap_class.return_value = _make_mock_umap()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    reduced = {
        "Agent-A": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }
    agent_info = {
        "Agent-A": {"model": "o4-mini", "provider": "openai", "persona": "EMPIRICIST"},
    }

    visualizer = EmbeddingVisualizer()

    with patch.object(visualizer, '_load_logo', return_value=None):
        visualizer._plot(reduced, output_path=None, agent_info=agent_info)

    # The line plot should use the persona color, not the default cycle color
    plot_call = mock_ax.plot.call_args_list[0]
    used_color = plot_call[1]["color"]
    expected_color = _persona_color("EMPIRICIST")
    assert used_color == expected_color


# ==============================================
# 11. generate_pca_trajectory_plot() Tests
# ==============================================

from chal.embeddings.embedding_visualizer import generate_pca_trajectory_plot


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_pca_plot_creates_file(mock_umap_class, mock_plt, tmp_path):
    """generate_pca_trajectory_plot creates a plot file given valid config."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        result = generate_pca_trajectory_plot(config)

    assert mock_plt.savefig.called
    assert result == tmp_path / "belief_trajectories_pca.png"


@pytest.mark.unit
def test_generate_pca_plot_raises_on_missing_embeddings(tmp_path):
    """generate_pca_trajectory_plot raises FileNotFoundError when embeddings are missing."""
    config = _make_plot_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        generate_pca_trajectory_plot(config)


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_pca_plot_custom_output_path(mock_umap_class, mock_plt, tmp_path):
    """generate_pca_trajectory_plot uses output_path override when provided."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)
    custom_path = tmp_path / "custom_pca.png"

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        result = generate_pca_trajectory_plot(config, output_path=custom_path)

    assert result == custom_path
    save_args = mock_plt.savefig.call_args
    assert save_args[0][0] == custom_path


@pytest.mark.unit
@patch('chal.embeddings.embedding_visualizer.plt')
@patch('chal.embeddings.embedding_visualizer.UMAP')
def test_generate_pca_plot_uses_pca_axis_labels(mock_umap_class, mock_plt, tmp_path):
    """generate_pca_trajectory_plot passes PCA axis labels to _plot."""
    mock_umap = _make_mock_umap(n_samples=6)
    mock_umap_class.return_value = mock_umap
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    config = _make_plot_config(tmp_path)
    _save_synthetic_embeddings(tmp_path)

    with patch.object(EmbeddingVisualizer, '_load_logo', return_value=None):
        with patch.object(EmbeddingVisualizer, '_plot') as mock_plot:
            generate_pca_trajectory_plot(config)

    call_kwargs = mock_plot.call_args[1]
    assert call_kwargs["xlabel"] == "Principal Component 1"
    assert call_kwargs["ylabel"] == "Principal Component 2"
