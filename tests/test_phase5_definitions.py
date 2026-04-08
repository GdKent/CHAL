"""
Phase 5 tests for Definition Nodes (D#) — Analysis, Visualization & Convergence.

Tests:
- calculate_definitional_alignment(): aligned, divergent, unique terms, edge cases
- format_convergence_summary(): includes D# alignment data
- get_convergence_trajectory_summary(): tracks D# alignment over rounds
- compute_position_analysis(): D# vulnerability detection (weak defs, bottlenecks)
- _log_definition_statistics(): D# stats output
"""

import pytest
import numpy as np
from unittest.mock import Mock
from chal.convergence.convergence_metrics import (
    calculate_definitional_alignment,
    format_convergence_summary,
    get_convergence_trajectory_summary,
)
from chal.agents.prompts import compute_position_analysis


# ========================================
# Helpers
# ========================================

def _make_mock_model(similarity_matrix=None):
    """Create a mock embedding model.

    If similarity_matrix is given, encode returns vectors whose cosine
    similarity matches the matrix (for 2 items). Otherwise returns
    normalized random vectors.
    """
    model = Mock()

    if similarity_matrix is not None:
        call_count = [0]

        def mock_encode(texts, convert_to_numpy=True):
            call_count[0] += 1
            n = len(texts)
            # Return identity-like vectors for simplicity — tests
            # that need specific similarity should use the matrix helper
            embeddings = np.eye(max(n, 2))[:n, :max(n, 2)]
            return embeddings.astype(np.float32)

        model.encode = Mock(side_effect=mock_encode)
    else:
        def mock_encode(texts, convert_to_numpy=True):
            n = len(texts)
            embeddings = np.random.randn(n, 384)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return (embeddings / norms).astype(np.float32)

        model.encode = Mock(side_effect=mock_encode)

    return model


def _make_high_sim_model():
    """Model that returns identical embeddings so all items are similar."""
    model = Mock()

    def mock_encode(texts, convert_to_numpy=True):
        n = len(texts)
        # All vectors are identical → cosine similarity = 1.0
        return np.ones((n, 384), dtype=np.float32)

    model.encode = Mock(side_effect=mock_encode)
    return model


def _make_belief_with_defs(belief_id, definitions, **kwargs):
    """Helper to create a minimal belief dict with definitions."""
    belief = {
        "belief_id": belief_id,
        "schema_version": "CBS",
        "version": 1,
        "metadata": {"topic_query": "test", "agent_persona": "test"},
        "definitions": definitions,
        "thesis": {"stance": "test", "summary_bullets": ["t"], "strength": 0.5},
    }
    belief.update(kwargs)
    return belief


def _make_position_belief(**kwargs):
    """Helper to create a belief dict for compute_position_analysis() tests."""
    belief = {
        "claims": kwargs.get("claims", [
            {"id": "C1", "strength": 0.7, "status": "active",
             "depends_on": ["A1", "E1"]},
        ]),
        "assumptions": kwargs.get("assumptions", [
            {"id": "A1", "strength": 0.8, "status": "active",
             "supported_by_definitions": ["D1"]},
        ]),
        "evidence": kwargs.get("evidence", [
            {"id": "E1", "strength": 0.75, "status": "active",
             "supported_by_definitions": ["D1"]},
        ]),
        "definitions": kwargs.get("definitions", [
            {"id": "D1", "term": "test", "definition": "test def",
             "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
        ]),
        "counterpositions": kwargs.get("counterpositions", []),
    }
    return belief


# ========================================
# 1. calculate_definitional_alignment()
# ========================================

class TestDefinitionalAlignment:
    """Tests for calculate_definitional_alignment()."""

    @pytest.mark.unit
    def test_single_agent_returns_perfect(self):
        """Single agent returns alignment score 1.0."""
        beliefs = [_make_belief_with_defs("A1", [
            {"id": "D1", "term": "free will", "definition": "capacity to choose",
             "strength": 0.8, "status": "active"},
        ])]
        model = _make_mock_model()
        result = calculate_definitional_alignment(beliefs, model)
        assert result["definitional_alignment_score"] == 1.0

    @pytest.mark.unit
    def test_no_definitions_returns_zero(self):
        """Agents with no definitions return 0.0 alignment."""
        beliefs = [
            _make_belief_with_defs("A1", []),
            _make_belief_with_defs("A2", []),
        ]
        model = _make_mock_model()
        result = calculate_definitional_alignment(beliefs, model)
        assert result["definitional_alignment_score"] == 0.0

    @pytest.mark.unit
    def test_aligned_terms_detected(self):
        """Same term with similar definitions → aligned."""
        beliefs = [
            _make_belief_with_defs("A1", [
                {"id": "D1", "term": "consciousness",
                 "definition": "subjective experience of awareness",
                 "strength": 0.8, "status": "active"},
            ]),
            _make_belief_with_defs("A2", [
                {"id": "D1", "term": "consciousness",
                 "definition": "subjective experience of awareness",
                 "strength": 0.85, "status": "active"},
            ]),
        ]
        # Use model that returns identical vectors → similarity 1.0
        model = _make_high_sim_model()
        result = calculate_definitional_alignment(beliefs, model)
        assert result["definitional_alignment_score"] > 0.0
        assert len(result["aligned_terms"]) >= 1
        assert result["aligned_terms"][0]["term"] == "consciousness"

    @pytest.mark.unit
    def test_divergent_terms_detected(self):
        """Same term with different definitions → divergent."""
        beliefs = [
            _make_belief_with_defs("A1", [
                {"id": "D1", "term": "justice",
                 "definition": "fairness in distribution of resources",
                 "strength": 0.8, "status": "active"},
            ]),
            _make_belief_with_defs("A2", [
                {"id": "D1", "term": "justice",
                 "definition": "adherence to law and order",
                 "strength": 0.8, "status": "active"},
            ]),
        ]
        # Use model where terms are identical (same string) but definitions
        # are orthogonal (random embeddings → low similarity)
        model = Mock()
        call_count = [0]

        def mock_encode(texts, convert_to_numpy=True):
            call_count[0] += 1
            n = len(texts)
            if call_count[0] == 1:
                # Terms: identical → high similarity
                return np.ones((n, 4), dtype=np.float32)
            else:
                # Definitions: orthogonal → low similarity
                return np.eye(max(n, 2))[:n, :max(n, 2)].astype(np.float32)

        model.encode = Mock(side_effect=mock_encode)

        result = calculate_definitional_alignment(beliefs, model)
        # Terms match (exact string) but definitions don't align
        assert len(result["divergent_terms"]) >= 1
        assert result["divergent_terms"][0]["term"] == "justice"

    @pytest.mark.unit
    def test_unique_terms_detected(self):
        """Terms unique to one agent appear in unique_terms."""
        beliefs = [
            _make_belief_with_defs("A1", [
                {"id": "D1", "term": "consciousness",
                 "definition": "subjective awareness",
                 "strength": 0.8, "status": "active"},
            ]),
            _make_belief_with_defs("A2", [
                {"id": "D1", "term": "qualia",
                 "definition": "individual subjective experiences",
                 "strength": 0.8, "status": "active"},
            ]),
        ]
        # Orthogonal embeddings → no term match
        model = Mock()

        def mock_encode(texts, convert_to_numpy=True):
            n = len(texts)
            return np.eye(max(n, 2))[:n, :max(n, 2)].astype(np.float32)

        model.encode = Mock(side_effect=mock_encode)

        result = calculate_definitional_alignment(beliefs, model)
        assert len(result["unique_terms"]) == 2

    @pytest.mark.unit
    def test_retracted_definitions_excluded(self):
        """Retracted definitions are not counted."""
        beliefs = [
            _make_belief_with_defs("A1", [
                {"id": "D1", "term": "test", "definition": "retracted one",
                 "strength": 0.0, "status": "retracted"},
            ]),
            _make_belief_with_defs("A2", [
                {"id": "D1", "term": "test", "definition": "active one",
                 "strength": 0.8, "status": "active"},
            ]),
        ]
        model = _make_high_sim_model()
        result = calculate_definitional_alignment(beliefs, model)
        # Only one active def → no cross-agent match possible
        assert len(result["aligned_terms"]) == 0
        assert len(result["unique_terms"]) == 1

    @pytest.mark.unit
    def test_result_structure(self):
        """Result contains all required keys."""
        beliefs = [
            _make_belief_with_defs("A1", []),
            _make_belief_with_defs("A2", []),
        ]
        model = _make_mock_model()
        result = calculate_definitional_alignment(beliefs, model)
        assert "definitional_alignment_score" in result
        assert "aligned_terms" in result
        assert "divergent_terms" in result
        assert "unique_terms" in result


# ========================================
# 2. format_convergence_summary() with D#
# ========================================

class TestFormatConvergenceSummaryDefinitional:
    """Tests that format_convergence_summary includes D# alignment data."""

    @pytest.mark.unit
    def test_includes_definitional_section(self):
        """Summary includes DEFINITIONAL ALIGNMENT section when data provided."""
        conv_data = {
            "convergence_score": 0.5,
            "total_claims": 4,
            "shared_claim_pairs": 1,
            "shared_claims": [],
            "unique_claims": [],
        }
        def_data = {
            "definitional_alignment_score": 0.75,
            "aligned_terms": [
                {"term": "consciousness", "agents": ["A1", "A2"], "similarity": 0.92},
            ],
            "divergent_terms": [],
            "unique_terms": [],
        }
        result = format_convergence_summary(
            conv_data, agent_names=["A1", "A2"], definitional_data=def_data,
        )
        assert "DEFINITIONAL ALIGNMENT" in result
        assert "75.0%" in result
        assert "consciousness" in result

    @pytest.mark.unit
    def test_no_definitional_section_when_none(self):
        """No D# section when definitional_data is None."""
        conv_data = {
            "convergence_score": 0.5,
            "total_claims": 4,
            "shared_claim_pairs": 1,
            "shared_claims": [],
            "unique_claims": [],
        }
        result = format_convergence_summary(
            conv_data, agent_names=["A1", "A2"],
        )
        assert "DEFINITIONAL ALIGNMENT" not in result

    @pytest.mark.unit
    def test_shows_divergent_terms(self):
        """Summary shows divergent terms when present."""
        conv_data = {
            "convergence_score": 0.5,
            "total_claims": 4,
            "shared_claim_pairs": 0,
            "shared_claims": [],
            "unique_claims": [],
        }
        def_data = {
            "definitional_alignment_score": 0.0,
            "aligned_terms": [],
            "divergent_terms": [
                {"term": "justice", "agents": ["A1", "A2"],
                 "definitions": ["def1", "def2"], "similarity": 0.3},
            ],
            "unique_terms": [],
        }
        result = format_convergence_summary(
            conv_data, agent_names=["A1", "A2"], definitional_data=def_data,
        )
        assert "DIVERGENT" in result
        assert "justice" in result


# ========================================
# 3. get_convergence_trajectory_summary()
# ========================================

class TestTrajectoryWithDefinitional:
    """Tests that trajectory summary tracks D# alignment over rounds."""

    @pytest.mark.unit
    def test_includes_definitional_score_per_round(self):
        """Each round shows definitional alignment score."""
        history = [
            {"round": 1, "convergence_score": 0.3, "shared_claim_pairs": 1,
             "unique_claims_count": 3, "definitional_alignment_score": 0.2},
            {"round": 2, "convergence_score": 0.5, "shared_claim_pairs": 2,
             "unique_claims_count": 2, "definitional_alignment_score": 0.6},
        ]
        result = get_convergence_trajectory_summary(history)
        assert "def. alignment: 20.0%" in result
        assert "def. alignment: 60.0%" in result

    @pytest.mark.unit
    def test_definitional_trend(self):
        """Trajectory shows definitional alignment trend."""
        history = [
            {"round": 1, "convergence_score": 0.3, "shared_claim_pairs": 1,
             "unique_claims_count": 3, "definitional_alignment_score": 0.2},
            {"round": 2, "convergence_score": 0.5, "shared_claim_pairs": 2,
             "unique_claims_count": 2, "definitional_alignment_score": 0.6},
        ]
        result = get_convergence_trajectory_summary(history)
        assert "Definitional Trend: ALIGNING" in result

    @pytest.mark.unit
    def test_no_definitional_trend_without_data(self):
        """No definitional trend when data not present."""
        history = [
            {"round": 1, "convergence_score": 0.3, "shared_claim_pairs": 1,
             "unique_claims_count": 3},
            {"round": 2, "convergence_score": 0.5, "shared_claim_pairs": 2,
             "unique_claims_count": 2},
        ]
        result = get_convergence_trajectory_summary(history)
        assert "Definitional Trend" not in result

    @pytest.mark.unit
    def test_stable_definitional_trend(self):
        """Stable definitional trend when change < 0.1."""
        history = [
            {"round": 1, "convergence_score": 0.3, "shared_claim_pairs": 1,
             "unique_claims_count": 3, "definitional_alignment_score": 0.5},
            {"round": 2, "convergence_score": 0.5, "shared_claim_pairs": 2,
             "unique_claims_count": 2, "definitional_alignment_score": 0.55},
        ]
        result = get_convergence_trajectory_summary(history)
        assert "Definitional Trend: STABLE" in result


# ========================================
# 4. compute_position_analysis() — D# vulnerabilities
# ========================================

class TestPositionAnalysisDefinitionalVulnerabilities:
    """Tests for D# vulnerability detection in compute_position_analysis()."""

    @pytest.mark.unit
    def test_weak_definition_detected(self):
        """Low-strength D# supporting >=2 nodes triggers WEAK DEFINITION."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "weak def",
                 "strength": 0.5, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.5, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.5, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "WEAK DEFINITION" in result
        assert "D1" in result

    @pytest.mark.unit
    def test_no_weak_definition_when_strong(self):
        """High-strength D# does not trigger WEAK DEFINITION."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "strong def",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "WEAK DEFINITION" not in result

    @pytest.mark.unit
    def test_bottleneck_detected(self):
        """A#/E# with single active D# support triggers BOTTLENECK."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "only def",
                 "strength": 0.7, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.7, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "BOTTLENECK" in result
        assert "A1" in result
        assert "D1" in result

    @pytest.mark.unit
    def test_no_bottleneck_with_multiple_defs(self):
        """A# with multiple active D# support does not trigger BOTTLENECK."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test1", "definition": "def1",
                 "strength": 0.8, "status": "active", "used_by": ["A1", "E1"]},
                {"id": "D2", "term": "test2", "definition": "def2",
                 "strength": 0.8, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "BOTTLENECK" not in result

    @pytest.mark.unit
    def test_no_vulnerability_section_when_none(self):
        """No DEFINITIONAL VULNERABILITIES section when none found."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "strong def",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
                {"id": "D2", "term": "test2", "definition": "strong def2",
                 "strength": 0.85, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" not in result

    @pytest.mark.unit
    def test_vulnerability_section_present_when_found(self):
        """DEFINITIONAL VULNERABILITIES section present when issues found."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "weak def",
                 "strength": 0.4, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.4, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.4, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" in result

    @pytest.mark.unit
    def test_retracted_def_not_counted_as_support(self):
        """Retracted D# not counted for bottleneck analysis."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "retracted",
                 "strength": 0.0, "status": "retracted", "used_by": ["A1"]},
                {"id": "D2", "term": "test2", "definition": "active",
                 "strength": 0.8, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.7, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        # D1 retracted → only D2 active → bottleneck on A1
        assert "BOTTLENECK" in result
        assert "A1" in result

    @pytest.mark.unit
    def test_no_definitions_no_vulnerability(self):
        """Belief with no definitions produces no vulnerability section."""
        belief = _make_position_belief(
            definitions=[],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": []},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": []},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" not in result

    @pytest.mark.unit
    def test_standard_sections_preserved(self):
        """Standard position analysis sections still present."""
        belief = _make_position_belief()
        result = compute_position_analysis(belief)
        assert "YOUR CURRENT POSITION" in result
        assert "SENSITIVITY AT YOUR POSITION" in result
        assert "SCENARIO PROJECTIONS" in result
        assert "LOWEST-STRENGTH DEPENDENCIES" in result
        assert "STRATEGIC RECOMMENDATION" in result
        assert "INTEGRITY REMINDER" in result
