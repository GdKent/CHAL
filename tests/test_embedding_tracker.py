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


# Model dimension for all-mpnet-base-v2
_MODEL_DIM = 768
# Component-wise embedding: 10 sub-vectors * model_dim + 11 scalar features
_COMPONENT_DIM = 10 * _MODEL_DIM + 11  # 7691


def _make_mock_model():
    """Create a properly configured mock SentenceTransformer.

    The mock handles both single-string and list-of-strings calls to encode(),
    and exposes get_sentence_embedding_dimension() returning _MODEL_DIM.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    def _mock_encode(input_text, **kwargs):
        """Return a (_MODEL_DIM,) vector for a string, or (N, _MODEL_DIM) for a list."""
        if isinstance(input_text, str):
            return np.random.RandomState(42).rand(_MODEL_DIM).astype(np.float32)
        # list of strings
        n = len(input_text)
        return np.random.RandomState(42).rand(n, _MODEL_DIM).astype(np.float32)

    mock_model.encode.side_effect = _mock_encode
    return mock_model


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
    """Test generating embedding for a CBS belief dict (component-wise path)."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    embedding = tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == _COMPONENT_DIM
    # Component-wise path calls encode multiple times (once per component group)
    assert mock_model.encode.call_count >= 1


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_string_fallback(mock_transformer_class):
    """Test that a plain text string input uses the legacy 768-dim path."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    text = "This is a pre-projected belief string."

    embedding = tracker.embed_belief(text, agent_name="Agent-A", round_num=1)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == _MODEL_DIM
    mock_model.encode.assert_called_once()


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_stores_by_agent(mock_transformer_class):
    """Test that embeddings are organized by agent name."""
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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
    mock_model = _make_mock_model()
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

        # Verify vector dimensionality is preserved
        for emb in loaded_tracker.get_agent_trajectory("Agent-A"):
            assert emb.shape == (_COMPONENT_DIM,)


# ==============================================
# 5. Multi-Agent Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_multiple_agents_tracking(mock_transformer_class):
    """Test tracking embeddings for multiple agents."""
    mock_model = _make_mock_model()
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
def test_embedding_uses_component_encoding(mock_transformer_class):
    """Test that dict beliefs go through component-wise encoding."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()

    embedding = tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    # Component-wise path calls encode multiple times for different components
    assert mock_model.encode.call_count >= 1
    # Result should be the full component-wise vector
    assert embedding.shape == (_COMPONENT_DIM,)


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embedding_string_uses_single_encode(mock_transformer_class):
    """Test that plain-string beliefs use single encode call (legacy path)."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    text_belief = "Some pre-projected text"

    tracker.embed_belief(text_belief, agent_name="Agent-A", round_num=1)

    # Legacy text path calls encode exactly once with the string
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args
    encoded_text = call_args[0][0]

    assert isinstance(encoded_text, str)
    assert len(encoded_text) > 0


# ==============================================
# 7. Edge Case Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_minimal_belief(mock_transformer_class):
    """Test embedding minimal belief with only required fields."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    minimal_belief = create_sample_belief(num_assumptions=0, num_claims=0, num_evidence=0)

    embedding = tracker.embed_belief(minimal_belief, agent_name="Agent-A", round_num=1)

    assert isinstance(embedding, np.ndarray)
    # Even minimal beliefs produce the full component-wise vector
    assert embedding.shape == (_COMPONENT_DIM,)


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


# ==============================================
# 8. Weighted Average Embedding Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_weighted_average_embedding_varied_strengths(mock_transformer_class):
    """Weighted average with varied strengths produces expected result."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    vec_a = np.ones(_MODEL_DIM, dtype=np.float32) * 2.0
    vec_b = np.ones(_MODEL_DIM, dtype=np.float32) * 4.0
    mock_model.encode.return_value = np.stack([vec_a, vec_b])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()

    items = [
        {"text": "text A", "strength": 0.25},
        {"text": "text B", "strength": 0.75},
    ]
    result = tracker._weighted_average_embedding(items)

    # weighted avg = (0.25*2 + 0.75*4) / (0.25+0.75) = (0.5+3.0)/1.0 = 3.5
    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_allclose(result, 3.5, atol=1e-5)


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_weighted_average_embedding_zero_strengths(mock_transformer_class):
    """All-zero strengths falls back to unweighted mean."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    vec_a = np.ones(_MODEL_DIM, dtype=np.float32) * 2.0
    vec_b = np.ones(_MODEL_DIM, dtype=np.float32) * 6.0
    mock_model.encode.return_value = np.stack([vec_a, vec_b])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()

    items = [
        {"text": "text A", "strength": 0.0},
        {"text": "text B", "strength": 0.0},
    ]
    result = tracker._weighted_average_embedding(items)

    # unweighted mean = (2+6)/2 = 4
    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_allclose(result, 4.0, atol=1e-5)


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_weighted_average_embedding_empty(mock_transformer_class):
    """Empty input returns zero vector of correct dimension."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    result = tracker._weighted_average_embedding([])

    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_array_equal(result, np.zeros(_MODEL_DIM, dtype=np.float32))


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_weighted_average_embedding_single_item(mock_transformer_class):
    """Single item returns that item's embedding (weight normalization for one = itself)."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    vec = np.arange(_MODEL_DIM, dtype=np.float32)
    mock_model.encode.return_value = vec.reshape(1, _MODEL_DIM)
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()

    items = [{"text": "only item", "strength": 0.6}]
    result = tracker._weighted_average_embedding(items)

    # With one item: weighted = (0.6 * vec) / 0.6 = vec
    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_allclose(result, vec, atol=1e-5)


# ==============================================
# 9. Simple Average Embedding Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_simple_average_embedding_multiple_texts(mock_transformer_class):
    """Average of multiple texts produces expected mean vector."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    vec_a = np.ones(_MODEL_DIM, dtype=np.float32) * 3.0
    vec_b = np.ones(_MODEL_DIM, dtype=np.float32) * 7.0
    mock_model.encode.return_value = np.stack([vec_a, vec_b])
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    result = tracker._simple_average_embedding(["text A", "text B"])

    # mean = (3+7)/2 = 5
    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_allclose(result, 5.0, atol=1e-5)


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_simple_average_embedding_empty(mock_transformer_class):
    """Empty input returns zero vector."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    result = tracker._simple_average_embedding([])

    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_array_equal(result, np.zeros(_MODEL_DIM, dtype=np.float32))


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_simple_average_embedding_single_item(mock_transformer_class):
    """Single item returns that item's embedding exactly."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    vec = np.arange(_MODEL_DIM, dtype=np.float32)
    mock_model.encode.return_value = vec.reshape(1, _MODEL_DIM)
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    result = tracker._simple_average_embedding(["only text"])

    assert result.shape == (_MODEL_DIM,)
    np.testing.assert_allclose(result, vec, atol=1e-5)


# ==============================================
# 10. Normalize Scalars Tests
# ==============================================

@pytest.mark.unit
def test_normalize_scalars_shape_and_values():
    """Output shape is (11,), counts are normalized, strengths pass through."""
    scalars = {
        "n_definitions": 5,
        "n_assumptions": 3,
        "n_evidence": 10,
        "n_claims": 2,
        "avg_strength_definitions": 0.85,
        "avg_strength_assumptions": 0.7,
        "avg_strength_evidence": 0.9,
        "avg_strength_claims": 0.65,
        "n_counterpositions": 4,
        "n_uncertainties": 6,
        "thesis_strength": 0.8,
    }
    result = EmbeddingTracker._normalize_scalars(scalars)

    assert result.shape == (11,)
    assert result.dtype == np.float32

    # Counts normalized: divided by 20
    np.testing.assert_allclose(result[0], 5 / 20)      # n_definitions
    np.testing.assert_allclose(result[1], 3 / 20)      # n_assumptions
    np.testing.assert_allclose(result[2], 10 / 20)     # n_evidence
    np.testing.assert_allclose(result[3], 2 / 20)      # n_claims

    # Strengths pass through
    np.testing.assert_allclose(result[4], 0.85)         # avg_strength_definitions
    np.testing.assert_allclose(result[5], 0.7)          # avg_strength_assumptions
    np.testing.assert_allclose(result[6], 0.9)          # avg_strength_evidence
    np.testing.assert_allclose(result[7], 0.65)         # avg_strength_claims

    # Other counts normalized
    np.testing.assert_allclose(result[8], 4 / 20)      # n_counterpositions
    np.testing.assert_allclose(result[9], 6 / 20)      # n_uncertainties

    # Thesis strength passes through
    np.testing.assert_allclose(result[10], 0.8)         # thesis_strength


@pytest.mark.unit
def test_normalize_scalars_counts_capped():
    """Counts above cap (20) are clamped to 1.0."""
    scalars = {
        "n_definitions": 25,
        "n_assumptions": 100,
        "n_evidence": 20,
        "n_claims": 0,
        "avg_strength_definitions": 0.5,
        "avg_strength_assumptions": 0.5,
        "avg_strength_evidence": 0.5,
        "avg_strength_claims": 0.5,
        "n_counterpositions": 50,
        "n_uncertainties": 30,
        "thesis_strength": 0.5,
    }
    result = EmbeddingTracker._normalize_scalars(scalars)

    # Counts above 20 are clamped to 1.0
    np.testing.assert_allclose(result[0], 1.0)    # 25 -> clamped
    np.testing.assert_allclose(result[1], 1.0)    # 100 -> clamped
    np.testing.assert_allclose(result[2], 1.0)    # 20/20 = 1.0 (exactly at cap)
    np.testing.assert_allclose(result[3], 0.0)    # 0/20 = 0.0

    np.testing.assert_allclose(result[8], 1.0)    # 50 -> clamped
    np.testing.assert_allclose(result[9], 1.0)    # 30 -> clamped


# ==============================================
# 11. embed_belief_components End-to-End Test
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_components_end_to_end(mock_transformer_class):
    """Produces vector of dimension 7691 (10*768+11) and stores it correctly."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    def mock_encode(input_data, convert_to_numpy=True):
        if isinstance(input_data, list):
            n = len(input_data)
            return np.ones((n, _MODEL_DIM), dtype=np.float32) * 0.1
        else:
            # Single string (thesis)
            return np.ones(_MODEL_DIM, dtype=np.float32) * 0.5

    mock_model.encode.side_effect = mock_encode
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()

    belief = {
        "schema_version": "CBS",
        "belief_id": "E2E",
        "version": 1,
        "metadata": {"topic_query": "Test", "agent_persona": "Test"},
        "thesis": {
            "stance": "Test thesis",
            "summary_bullets": ["bullet 1"],
            "strength": 0.75,
        },
        "definitions": [
            {"id": "D1", "term": "t", "definition": "d", "strength": 0.8, "status": "active",
             "strength_justification": "ok", "used_by": ["A1"]},
        ],
        "assumptions": [
            {"id": "A1", "type": "empirical", "statement": "s", "strength": 0.7,
             "status": "active", "strength_justification": "ok", "supported_by_definitions": ["D1"]},
        ],
        "evidence": [
            {"id": "E1", "type": "empirical", "summary": "ev", "source": "src",
             "supports_claims": ["C1"], "strength": 0.9, "status": "active",
             "strength_justification": "ok", "supported_by_definitions": ["D1"]},
        ],
        "claims": [
            {"id": "C1", "type": "deductive", "statement": "cl", "depends_on": ["A1", "E1"],
             "strength": 0.85, "status": "active", "strength_justification": "ok",
             "inference_chain": [], "predictions": []},
        ],
        "uncertainties": [
            {"id": "U1", "targets": ["C1"], "question": "q?", "status": "active", "importance": "high"},
        ],
        "counterpositions": [
            {"id": "X1", "targets": ["C1"], "attack_type": "rebutting",
             "attack_strategy": "strat", "statement": "counter",
             "my_response": "resp", "response_sufficiency": "partial"},
            {"id": "X2", "targets": ["C1"], "attack_type": "rebutting",
             "attack_strategy": "strat", "statement": "moot counter",
             "my_response": "", "response_sufficiency": "moot"},
        ],
    }

    result = tracker.embed_belief_components(belief, agent_name="Agent-A", round_num=1)

    expected_dim = 10 * _MODEL_DIM + 11  # 7691
    assert result.shape == (expected_dim,)
    assert result.dtype == np.float32

    # Verify it was stored in the agent's trajectory
    trajectory = tracker.get_agent_trajectory("Agent-A")
    assert len(trajectory) == 1
    np.testing.assert_array_equal(trajectory[0], result)


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_embed_belief_components_multi_round_trajectory(mock_transformer_class):
    """Trajectory tracking works across multiple rounds."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _MODEL_DIM

    call_count = [0]

    def mock_encode(input_data, convert_to_numpy=True):
        call_count[0] += 1
        if isinstance(input_data, list):
            n = len(input_data)
            return np.ones((n, _MODEL_DIM), dtype=np.float32) * call_count[0]
        else:
            return np.ones(_MODEL_DIM, dtype=np.float32) * call_count[0]

    mock_model.encode.side_effect = mock_encode
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()

    belief = {
        "schema_version": "CBS", "belief_id": "MULTI", "version": 1,
        "metadata": {"topic_query": "T", "agent_persona": "P"},
        "thesis": {"stance": "s", "summary_bullets": ["b"], "strength": 0.5},
        "definitions": [], "assumptions": [], "evidence": [],
        "claims": [], "counterpositions": [], "uncertainties": [],
    }

    r1 = tracker.embed_belief_components(belief, agent_name="Agent-A", round_num=1)
    r2 = tracker.embed_belief_components(belief, agent_name="Agent-A", round_num=2)
    r3 = tracker.embed_belief_components(belief, agent_name="Agent-A", round_num=3)

    trajectory = tracker.get_agent_trajectory("Agent-A")
    assert len(trajectory) == 3
    for emb in trajectory:
        assert emb.shape == (_COMPONENT_DIM,)

    # Each round should produce a different vector (due to incrementing mock)
    assert not np.array_equal(r1, r2)
    assert not np.array_equal(r2, r3)


# ==============================================
# 12. Metadata Save/Load Tests
# ==============================================

@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_save_embeddings_with_metadata(mock_transformer_class):
    """save_embeddings stores agent_info and topic in the .npz file."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    agent_info = {"Agent-A": {"model": "o4-mini", "provider": "openai"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "embeddings.npz"
        tracker.save_embeddings(save_path, agent_info=agent_info, topic="Free will")

        with np.load(str(save_path), allow_pickle=True) as data:
            assert "__metadata__" in data.files
            meta = data["__metadata__"].item()
            assert meta["agent_info"] == agent_info
            assert meta["topic"] == "Free will"


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_load_embeddings_with_metadata(mock_transformer_class):
    """load_embeddings populates tracker.metadata with saved agent_info and topic."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    agent_info = {"Agent-A": {"model": "o4-mini", "provider": "openai"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "embeddings.npz"
        tracker.save_embeddings(save_path, agent_info=agent_info, topic="Free will")

        new_tracker = EmbeddingTracker()
        new_tracker.load_embeddings(save_path)

        assert new_tracker.metadata["agent_info"] == agent_info
        assert new_tracker.metadata["topic"] == "Free will"
        # Embeddings should still load correctly (no __metadata__ key)
        assert "Agent-A" in new_tracker.embeddings
        assert "__metadata__" not in new_tracker.embeddings


@pytest.mark.unit
@patch('chal.embeddings.embedding_tracker.SentenceTransformer')
def test_load_embeddings_without_metadata_backward_compat(mock_transformer_class):
    """Loading old-format files (no __metadata__) sets tracker.metadata to empty dict."""
    mock_model = _make_mock_model()
    mock_transformer_class.return_value = mock_model

    tracker = EmbeddingTracker()
    belief = create_sample_belief()
    tracker.embed_belief(belief, agent_name="Agent-A", round_num=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "embeddings.npz"
        # Save without metadata (old format)
        tracker.save_embeddings(save_path)

        new_tracker = EmbeddingTracker()
        new_tracker.load_embeddings(save_path)

        assert new_tracker.metadata == {}
        assert "Agent-A" in new_tracker.embeddings
