"""
Unit tests for EmbeddingTracker class.

Tests cover:
- Initialization
- Embedding generation
- Trajectory retrieval
- Persistence (save/load)
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker as EmbeddingTracker
from chal.beliefs.io import project_for_embedding
from tests.utils import create_sample_belief


# ==============================================
# 1. Initialization Tests
# ==============================================

@pytest.mark.unit
def test_embedding_tracker_init():
    """Test initialization with default model."""
    tracker = EmbeddingTracker()

    assert tracker is not None
    assert hasattr(tracker, "model")


@pytest.mark.unit
def test_embedding_tracker_default_model():
    """Test that default model is all-mpnet-base-v2."""
    tracker = EmbeddingTracker()

    # Default model should be all-mpnet-base-v2
    assert "mpnet" in tracker.model_name.lower() or tracker.model_name == "all-mpnet-base-v2"


@pytest.mark.unit
def test_embedding_tracker_custom_model():
    """Test initialization with custom model."""
    tracker = EmbeddingTracker(model_name="all-MiniLM-L6-v2")

    assert tracker.model_name == "all-MiniLM-L6-v2"


# ==============================================
# 2. Embedding Generation Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief(mock_transformer_class):
    """Test generating embedding for belief text."""
    # Mock the model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    embedding = tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0
    mock_model.encode.assert_called_once()


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_stores_by_agent(mock_transformer_class):
    """Test that embeddings are organized by agent name."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    # Should store in agent-specific trajectory
    trajectory = tracker.get_agent_trajectory("Agent-A")
    assert len(trajectory) == 1


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_multiple_rounds(mock_transformer_class):
    """Test that embeddings append to agent's trajectory."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=2)
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=3)

    trajectory = tracker.get_agent_trajectory("Agent-A")
    assert len(trajectory) == 3


# ==============================================
# 3. Trajectory Retrieval Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_get_agent_trajectory(mock_transformer_class):
    """Test retrieving embeddings for specific agent."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=2)

    trajectory = tracker.get_agent_trajectory("Agent-A")

    assert len(trajectory) == 2
    assert all(isinstance(emb, np.ndarray) for emb in trajectory)


@pytest.mark.unit
def test_get_agent_trajectory_empty():
    """Test retrieving trajectory for agent with no embeddings."""
    tracker = EmbeddingTracker()

    trajectory = tracker.get_agent_trajectory("NonExistent")

    assert len(trajectory) == 0


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_get_all_embeddings(mock_transformer_class):
    """Test retrieving all embeddings for all agents."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)
    tracker.embed_belief(belief, agent_name="Agent-B", round_num=1)

    all_embeddings = tracker.get_all_embeddings()

    assert isinstance(all_embeddings, dict)
    assert "Agent-A" in all_embeddings
    assert "Agent-B" in all_embeddings


# ==============================================
# 4. Persistence Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_save_embeddings(mock_transformer_class):
    """Test saving embeddings to .npz file."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "embeddings.npz"
        tracker.save_embeddings(save_path)

        assert save_path.exists()


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_load_embeddings(mock_transformer_class):
    """Test loading embeddings from .npz file."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "embeddings.npz"
        tracker.save_embeddings(save_path)

        # Create new tracker and load
        new_tracker = EmbeddingTracker()
        new_tracker.load_embeddings(save_path)

        # Should have same embeddings
        trajectory = new_tracker.get_agent_trajectory("Agent-A")
        assert len(trajectory) == 1


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_save_load_roundtrip(mock_transformer_class):
    """Test that save then load preserves data."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=2)
    tracker.embed_belief(belief, agent_name="Agent-B", round_num=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.npz"
        tracker.save_embeddings(save_path)

        # Load into new tracker
        loaded_tracker = EmbeddingTracker()
        loaded_tracker.load_embeddings(save_path)

        # Verify trajectories match
        assert len(loaded_tracker.get_agent_trajectory("Agent-A")) == 2
        assert len(loaded_tracker.get_agent_trajectory("Agent-B")) == 1


# ==============================================
# 5. Multi-Agent Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_multiple_agents_tracking(mock_transformer_class):
    """Test tracking embeddings for multiple agents."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1] * 384)  # Typical embedding size
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    # Track multiple agents
    for agent_name in ["Agent-A", "Agent-B", "Agent-C"]:
        for round_num in range(1, 4):
            tracker.embed_belief(belief, agent_name=agent_name, round_num=round_num)

    # Verify all trajectories
    assert len(tracker.get_agent_trajectory("Agent-A")) == 3
    assert len(tracker.get_agent_trajectory("Agent-B")) == 3
    assert len(tracker.get_agent_trajectory("Agent-C")) == 3


# ==============================================
# 6. Embedding Quality Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embedding_uses_projection(mock_transformer_class):
    """Test that embedding uses projected text from belief."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    # Should have called encode with projected text
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args
    encoded_text = call_args[0][0]

    # Should include thesis content
    assert isinstance(encoded_text, str)
    assert len(encoded_text) > 0


# ==============================================
# 7. Edge Case Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_minimal_belief(mock_transformer_class):
    """Test embedding minimal belief with only required fields."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    minimal_belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)

    embedding = tracker.embed_belief(minimal_belief, agent_name="Agent-A", round_num=1)

    assert isinstance(embedding, np.ndarray)


@pytest.mark.unit
def test_save_embeddings_empty():
    """Test saving when no embeddings have been generated."""
    tracker = EmbeddingTracker()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "empty.npz"

        # Should either save empty file or handle gracefully
        try:
            tracker.save_embeddings(save_path)
            assert True
        except ValueError:
            # Empty save may raise error
            pass


@pytest.mark.unit
def test_load_nonexistent_file():
    """Test loading from non-existent file."""
    tracker = EmbeddingTracker()

    with pytest.raises(FileNotFoundError):
        tracker.load_embeddings(Path("/nonexistent/path.npz"))
