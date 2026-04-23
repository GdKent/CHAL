"""
Unit tests for post-debate analysis reporting.

Tests cover:
- generate_analysis_report() with mock data from each mode
- Report output formatting (Markdown)
- generate_analysis_json() structured output
- Edge cases (no exchanges, single exchange, many exchanges)
- Mode-specific sections appear only when relevant
"""

import pytest
import json
from unittest.mock import Mock
from chal.utilities.reporting import generate_analysis_report, generate_analysis_json
from tests.utils import create_sample_belief


# ========================================
# Helpers
# ========================================

def _make_mock_config(mode="rebuttal", topic="Free will", max_rounds=2):
    """Create a mock config for reporting tests."""
    config = Mock()
    config.stage3_mode = mode
    config.topic = topic
    config.max_rounds = max_rounds
    config.adjudication = Mock()
    config.adjudication.model = "gpt-4o"
    config.adjudication.provider = "openai"
    config.adjudication.logic_weight = 1.0
    config.adjudication.ethics_weight = 0.0

    return config


def _make_mock_agent(name, belief_obj=None):
    """Create a mock agent for reporting tests."""
    agent = Mock()
    agent.name = name
    agent.model = "gpt-4o"
    agent.provider = "openai"
    agent.persona_label = "EMPIRICIST"
    agent.get_internal_belief_obj = Mock(return_value=belief_obj or create_sample_belief())
    agent.all_beliefs_held = [json.dumps(create_sample_belief())]
    return agent


def _make_challenge_rebuttal_pairs(n=3, status="rebuttal_valid"):
    """Create sample challenge-rebuttal pairs."""
    return [
        {
            "challenger": "Agent-A",
            "target": "Agent-B",
            "challenge": f"Challenge {i+1}",
            "rebuttal": f"Rebuttal {i+1}",
            "qid": f"Q{i+1}",
            "target_ids": ["C1"],
            "attack_type": "undermining",
            "attack_strategy": "challenge_evidence",
            "resolution": {
                "status": status if i < n - 1 else "critique_valid",
                "reasoning": f"Reasoning for Q{i+1}",
            },
        }
        for i in range(n)
    ]


def _make_agent_stats():
    """Create sample agent stats."""
    return {
        "Agent-A": {
            "performance_score": 0.45,
            "successful_critiques": 2,
            "successful_rebuttals": 1,
            "failed_rebuttals": 0,
            "unresolved_arguments": 0,
            "total_arguments": 3,
        },
        "Agent-B": {
            "performance_score": 0.20,
            "successful_critiques": 0,
            "successful_rebuttals": 1,
            "failed_rebuttals": 1,
            "unresolved_arguments": 1,
            "total_arguments": 3,
        },
    }


# ============================================================
# 1. generate_analysis_report() Tests
# ============================================================

class TestGenerateAnalysisReport:
    """Tests for Markdown analysis report generation."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        pairs = _make_challenge_rebuttal_pairs()
        stats = _make_agent_stats()

        report = generate_analysis_report(config, agents, pairs, stats)

        assert isinstance(report, str)
        assert len(report) > 100

    @pytest.mark.unit
    def test_contains_header(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats())

        assert "CHAL Debate Analysis Report" in report

    @pytest.mark.unit
    def test_contains_metadata_section(self):
        config = _make_mock_config(topic="AI Ethics")
        agents = [_make_mock_agent("Agent-A")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats())

        assert "AI Ethics" in report
        assert "rebuttal" in report

    @pytest.mark.unit
    def test_contains_verdict_distribution(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        pairs = _make_challenge_rebuttal_pairs(3, "rebuttal_valid")
        report = generate_analysis_report(config, agents, pairs, _make_agent_stats())

        assert "Verdict" in report
        assert "critique_valid" in report or "rebuttal_valid" in report

    @pytest.mark.unit
    def test_contains_agent_performance(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        stats = _make_agent_stats()
        report = generate_analysis_report(config, agents, [], stats)

        assert "Agent-A" in report
        assert "Agent-B" in report
        assert "+0.4500" in report  # Agent-A's APS score

    @pytest.mark.unit
    def test_contains_belief_evolution(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats())

        assert "Belief Evolution" in report

    @pytest.mark.unit
    def test_convergence_section_when_provided(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        convergence = [
            {"round": 1, "convergence_score": 0.3, "shared_claim_pairs": 2, "unique_claims_count": 5},
            {"round": 2, "convergence_score": 0.6, "shared_claim_pairs": 4, "unique_claims_count": 3},
        ]
        report = generate_analysis_report(config, agents, [], _make_agent_stats(), convergence_history=convergence)

        assert "Convergence" in report
        assert "0.300" in report or "0.3" in report

    @pytest.mark.unit
    def test_no_convergence_section_when_absent(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats(), convergence_history=None)

        assert "Convergence Trajectory" not in report

    @pytest.mark.unit
    def test_rebuttal_mode_no_mode_specific_sections(self):
        config = _make_mock_config(mode="rebuttal")
        agents = [_make_mock_agent("Agent-A")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats())

        assert "Max Turns/Question" not in report

    @pytest.mark.unit
    def test_empty_exchanges(self):
        """No exchanges should not crash."""
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        report = generate_analysis_report(config, agents, [], _make_agent_stats())

        assert isinstance(report, str)
        assert "Total" in report

    @pytest.mark.unit
    def test_many_exchanges(self):
        """Large number of exchanges renders correctly."""
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        pairs = _make_challenge_rebuttal_pairs(20)
        report = generate_analysis_report(config, agents, pairs, _make_agent_stats())

        assert "Q20" in report


# ============================================================
# 2. generate_analysis_json() Tests
# ============================================================

class TestGenerateAnalysisJSON:
    """Tests for structured JSON analysis report."""

    @pytest.mark.unit
    def test_returns_dict(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        result = generate_analysis_json(config, agents, [], _make_agent_stats())

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_has_required_keys(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        result = generate_analysis_json(config, agents, [], _make_agent_stats())

        assert "generated_at" in result
        assert "metadata" in result
        assert "verdict_distribution" in result
        assert "agent_summaries" in result
        assert "exchanges" in result

    @pytest.mark.unit
    def test_metadata_fields(self):
        config = _make_mock_config(topic="Test Topic", mode="rebuttal")
        agents = [_make_mock_agent("Agent-A")]
        result = generate_analysis_json(config, agents, [], _make_agent_stats())

        assert result["metadata"]["topic"] == "Test Topic"
        assert result["metadata"]["stage3_mode"] == "rebuttal"
        assert result["metadata"]["num_agents"] == 1

    @pytest.mark.unit
    def test_verdict_distribution_counts(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        pairs = _make_challenge_rebuttal_pairs(3, "rebuttal_valid")
        result = generate_analysis_json(config, agents, pairs, _make_agent_stats())

        verdicts = result["verdict_distribution"]
        assert "rebuttal_valid" in verdicts
        assert "critique_valid" in verdicts
        assert "unresolved" in verdicts

    @pytest.mark.unit
    def test_agent_summaries_populated(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        stats = _make_agent_stats()
        result = generate_analysis_json(config, agents, [], stats)

        assert "Agent-A" in result["agent_summaries"]
        assert result["agent_summaries"]["Agent-A"]["performance_score"] == 0.45

    @pytest.mark.unit
    def test_convergence_history_included_when_provided(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        conv = [{"round": 1, "convergence_score": 0.5}]
        result = generate_analysis_json(config, agents, [], _make_agent_stats(), convergence_history=conv)

        assert "convergence_history" in result

    @pytest.mark.unit
    def test_convergence_history_absent_when_none(self):
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A")]
        result = generate_analysis_json(config, agents, [], _make_agent_stats(), convergence_history=None)

        assert "convergence_history" not in result

    @pytest.mark.unit
    def test_is_json_serializable(self):
        """Full report dict should be JSON-serializable."""
        config = _make_mock_config()
        agents = [_make_mock_agent("Agent-A"), _make_mock_agent("Agent-B")]
        pairs = _make_challenge_rebuttal_pairs(3)
        result = generate_analysis_json(config, agents, pairs, _make_agent_stats())

        serialized = json.dumps(result)
        assert len(serialized) > 0
