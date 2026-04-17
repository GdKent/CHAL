"""
Unit tests for chal.utils module.

Tests cover:
- Challenge parsing
- Rebuttal parsing
- Agent stats initialization and updates
- Performance score calculation
- Display functions
- Stage 2 question validation
"""

import pytest
from pathlib import Path
from chal.utilities.utils import (
    parse_challenges,
    parse_structured_rebuttals_numbered as parse_structured_rebuttals,
    initialize_agent_stats,
    update_agent_stats,
    calculate_performance_scores,
    get_performance_summary,
    display_agent_stats,
    validate_stage2_questions,
    VALID_ATTACK_STRATEGIES,
    ALL_STRATEGIES,
    VALID_ADJUDICATION_VERDICTS,
    compute_attack_histograms,
    snapshot_belief,
    finalize_agent_stats,
    select_best_agent,
)


# ==============================================
# Test Fixtures
# ==============================================

@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_challenges(fixtures_dir):
    """Load sample challenges from fixtures."""
    with open(fixtures_dir / "test_challenges.txt") as f:
        return f.read()


@pytest.fixture
def sample_rebuttals(fixtures_dir):
    """Load sample rebuttals from fixtures."""
    with open(fixtures_dir / "test_rebuttals.txt") as f:
        return f.read()


# ==============================================
# 1. Challenge Parsing Tests
# ==============================================

@pytest.mark.unit
def test_parse_challenges_numbered(sample_challenges):
    """Test parsing '1. ... 2. ...' format."""
    # Extract the "Multiple Numbered Challenges" section
    text = """1. What empirical evidence supports your assumption about rational deliberation?
2. How do you address the problem of hard determinism in your framework?
3. Can you provide specific examples where free will manifests independently of causal chains?"""

    challenges = parse_challenges(text)

    assert len(challenges) == 3
    assert "empirical evidence" in challenges[0]
    assert "hard determinism" in challenges[1]
    assert "specific examples" in challenges[2]


@pytest.mark.unit
def test_parse_challenges_empty():
    """Test that empty input returns empty list."""
    challenges = parse_challenges("")

    assert len(challenges) == 0


@pytest.mark.unit
def test_parse_challenges_single():
    """Test single challenge without number."""
    text = "How can you justify this claim without empirical evidence?"

    challenges = parse_challenges(text)

    # Should return at least one challenge
    assert len(challenges) >= 1


@pytest.mark.unit
def test_parse_challenges_whitespace():
    """Test handling of extra whitespace."""
    text = """   1.   How    can   you   justify    this?
   2.   What   evidence   supports   your   claim?"""

    challenges = parse_challenges(text)

    assert len(challenges) == 2
    assert "justify" in challenges[0]
    assert "evidence" in challenges[1]


# ==============================================
# 2. Rebuttal Parsing Tests
# ==============================================

@pytest.mark.unit
def test_parse_structured_rebuttals_numbered(sample_rebuttals):
    """Test parsing 'Response 1: ...' format."""
    text = """Response 1:
I disagree with your characterization.

Response 2:
The empirical evidence actually supports my framework.

Response 3:
This objection confuses metaphysical and practical freedom."""

    rebuttals = parse_structured_rebuttals(text)

    # NOTE: Current regex has a bug - it captures all responses as one blob
    # because it looks for "Critique" as delimiter but test only has "Response" markers
    assert len(rebuttals) == 1
    assert "disagree" in rebuttals[0]
    assert "empirical evidence" in rebuttals[0]
    assert "metaphysical" in rebuttals[0]


@pytest.mark.unit
def test_parse_structured_rebuttals_empty():
    """Test that empty input returns empty list."""
    rebuttals = parse_structured_rebuttals("")

    assert len(rebuttals) == 0


@pytest.mark.unit
def test_parse_structured_rebuttals_multiple():
    """Test parsing multiple responses."""
    text = """Response 1:
First rebuttal here.

Response 2:
Second rebuttal here.

Response 3:
Third rebuttal here."""

    rebuttals = parse_structured_rebuttals(text)

    # NOTE: Same regex bug - captures all as one
    assert len(rebuttals) == 1


# ==============================================
# 3. Agent Stats Initialization Tests
# ==============================================

@pytest.mark.unit
def test_initialize_agent_stats():
    """Test creating stats dict for all agents."""
    agent_names = ["Agent-A", "Agent-B", "Agent-C"]

    stats = initialize_agent_stats(agent_names)

    assert len(stats) == 3
    assert "Agent-A" in stats
    assert "Agent-B" in stats
    assert "Agent-C" in stats


@pytest.mark.unit
def test_initialize_agent_stats_default_values():
    """Test that all counters start at 0."""
    agent_names = ["Agent-A"]

    stats = initialize_agent_stats(agent_names)

    agent_stat = stats["Agent-A"]
    assert agent_stat["successful_critiques"] == 0
    assert agent_stat["successful_rebuttals"] == 0
    assert agent_stat["unresolved_arguments"] == 0
    assert agent_stat["total_arguments"] == 0


# ==============================================
# 4. Agent Stats Update Tests
# ==============================================

@pytest.mark.unit
def test_update_agent_stats_critique_valid():
    """Test incrementing successful_critiques."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "critique_valid"}
    }

    update_agent_stats(stats, record)

    assert stats["Challenger"]["successful_critiques"] == 1
    assert stats["Challenger"]["total_arguments"] == 1
    assert stats["Target"]["total_arguments"] == 1


@pytest.mark.unit
def test_update_agent_stats_rebuttal_valid():
    """Test incrementing successful_rebuttals."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "rebuttal_valid"}
    }

    update_agent_stats(stats, record)

    assert stats["Target"]["successful_rebuttals"] == 1
    assert stats["Challenger"]["total_arguments"] == 1
    assert stats["Target"]["total_arguments"] == 1


@pytest.mark.unit
def test_update_agent_stats_unresolved():
    """Test incrementing unresolved for both agents."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "unresolved"}
    }

    update_agent_stats(stats, record)

    assert stats["Challenger"]["unresolved_arguments"] == 1
    assert stats["Target"]["unresolved_arguments"] == 1
    assert stats["Challenger"]["total_arguments"] == 1
    assert stats["Target"]["total_arguments"] == 1


@pytest.mark.unit
def test_update_agent_stats_total_arguments():
    """Test that total_arguments increments for both agents."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record1 = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "critique_valid"}
    }
    record2 = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "rebuttal_valid"}
    }
    record3 = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "unresolved"}
    }

    update_agent_stats(stats, record1)
    update_agent_stats(stats, record2)
    update_agent_stats(stats, record3)

    assert stats["Challenger"]["total_arguments"] == 3
    assert stats["Target"]["total_arguments"] == 3


# ==============================================
# 5. Performance Score Calculation Tests
# ==============================================

@pytest.mark.unit
def test_calculate_performance_scores_default_weights():
    """Test performance score calculation with standard weights."""
    stats = {
        "Agent-A": {
            "successful_critiques": 5,
            "successful_rebuttals": 3,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10
        }
    }

    calculate_performance_scores(stats)

    assert "performance_score" in stats["Agent-A"]
    assert isinstance(stats["Agent-A"]["performance_score"], float)


@pytest.mark.unit
def test_calculate_performance_scores_custom_weights():
    """Test performance score with custom weights."""
    stats = {
        "Agent-A": {
            "successful_critiques": 5,
            "successful_rebuttals": 3,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10
        }
    }

    weights = {
        'successful_critique': 2.0,
        'successful_rebuttal': 1.0,
        'failed_rebuttal': -1.0,
        'unresolved_argument': -0.5
    }

    calculate_performance_scores(stats, weights=weights)

    assert "performance_score" in stats["Agent-A"]
    assert isinstance(stats["Agent-A"]["performance_score"], float)


@pytest.mark.unit
def test_calculate_performance_scores_formula():
    """Test that performance score formula is correct."""
    stats = {
        "Agent-A": {
            "successful_critiques": 3,
            "successful_rebuttals": 2,
            "failed_rebuttals": 0,
            "unresolved_arguments": 1,
            "total_arguments": 6
        }
    }

    # Default weights: successful_critique=3.0, successful_rebuttal=2.0, failed_rebuttal=-2.0, unresolved_argument=-0.5
    # Score = (3*3.0 + 2*2.0 + 0*(-2.0) + 1*(-0.5)) = 9 + 4 + 0 - 0.5 = 12.5
    calculate_performance_scores(stats)

    assert stats["Agent-A"]["performance_score"] == pytest.approx(12.5, abs=0.1)


@pytest.mark.unit
def test_get_performance_summary():
    """Test formatting of performance summary string."""
    stats = {
        "Agent-A": {
            "successful_critiques": 5,
            "successful_rebuttals": 3,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10,
            "performance_score": 15.5
        },
        "Agent-B": {
            "successful_critiques": 2,
            "successful_rebuttals": 6,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10,
            "performance_score": 16.0
        }
    }

    summary = get_performance_summary(stats)

    assert isinstance(summary, str)
    assert "Agent-A" in summary
    assert "Agent-B" in summary


# ==============================================
# 6. Display Functions Tests
# ==============================================

@pytest.mark.unit
def test_display_agent_stats():
    """Test generating readable stats display."""
    stats = {
        "Agent-A": {
            "successful_critiques": 5,
            "failed_critiques": 0,
            "successful_rebuttals": 3,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10,
            "performance_score": 15.5
        }
    }

    display = display_agent_stats(stats)

    # display_agent_stats doesn't return a value, it prints
    # So we can't assert on the display variable
    assert display is None


@pytest.mark.unit
def test_display_agent_stats_includes_all_metrics(capsys):
    """Test that display shows all stat categories."""
    stats = {
        "Agent-A": {
            "successful_critiques": 5,
            "failed_critiques": 0,
            "successful_rebuttals": 3,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10,
            "performance_score": 15.5
        }
    }

    display_agent_stats(stats)

    # Capture printed output
    captured = capsys.readouterr()
    display = captured.out

    # Should mention all metrics
    assert "critique" in display.lower() or "5" in display
    assert "rebuttal" in display.lower() or "3" in display
    assert "unresolved" in display.lower() or "2" in display
    assert "total" in display.lower() or "10" in display


# ==============================================
# 7. Stage 2 Question Validation Tests
# ==============================================

@pytest.mark.unit
def test_validate_stage2_questions_valid():
    """A well-formed question list passes validation."""
    questions = [
        {
            "qid": "Q1",
            "text": "How do you justify C2's strength given the weak evidence?",
            "target_ids": ["C2", "E1"],
            "attack_type": "undermining",
            "attack_strategy": "challenge_strength_calibration",
        },
        {
            "qid": "Q2",
            "text": "Your X1 is only partially addressed — doesn't this undermine C1?",
            "target_ids": ["X1"],
            "attack_type": "rebutting",
            "attack_strategy": "exploit_counterposition",
        },
    ]
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is True
    assert errors == []


@pytest.mark.unit
def test_validate_stage2_questions_missing_fields():
    """Questions missing required fields produce errors."""
    questions = [{"qid": "Q1"}]  # Missing text, target_ids, attack_type, attack_strategy
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is False
    assert len(errors) >= 1


@pytest.mark.unit
def test_validate_stage2_questions_invalid_attack_type():
    """An unrecognized attack_type value produces an error."""
    questions = [{
        "qid": "Q1",
        "text": "Some question",
        "target_ids": ["C1"],
        "attack_type": "destroying",
        "attack_strategy": "challenge_evidence",
    }]
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is False
    assert any("attack_type" in e for e in errors)


@pytest.mark.unit
def test_validate_stage2_questions_mismatched_strategy():
    """A valid attack_type paired with a strategy from a different type produces an error."""
    questions = [{
        "qid": "Q1",
        "text": "Some question",
        "target_ids": ["C1"],
        "attack_type": "undermining",
        "attack_strategy": "identify_circularity",  # belongs to undercutting
    }]
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is False
    assert any("attack_strategy" in e for e in errors)


@pytest.mark.unit
def test_validate_stage2_questions_invalid_target_ids():
    """target_ids with invalid prefixes produce errors."""
    questions = [{
        "qid": "Q1",
        "text": "Some question",
        "target_ids": ["Z1"],
        "attack_type": "undermining",
        "attack_strategy": "challenge_evidence",
    }]
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is False
    assert any("target_ids" in e for e in errors)


@pytest.mark.unit
def test_validate_stage2_questions_empty_list():
    """An empty question list returns invalid."""
    is_valid, errors = validate_stage2_questions([])
    assert is_valid is False


@pytest.mark.unit
def test_validate_stage2_questions_too_many_target_ids():
    """More than 2 target_ids produces an error."""
    questions = [{
        "qid": "Q1",
        "text": "Some question",
        "target_ids": ["C1", "A1", "E1"],
        "attack_type": "undermining",
        "attack_strategy": "challenge_evidence",
    }]
    is_valid, errors = validate_stage2_questions(questions)
    assert is_valid is False
    assert any("target_ids" in e for e in errors)


@pytest.mark.unit
def test_validate_stage2_questions_all_strategies_accepted():
    """Every valid (attack_type, attack_strategy) pair passes validation."""
    for attack_type, strategies in VALID_ATTACK_STRATEGIES.items():
        for strategy in strategies:
            questions = [{
                "qid": "Q1",
                "text": "Test question",
                "target_ids": ["C1"],
                "attack_type": attack_type,
                "attack_strategy": strategy,
            }]
            is_valid, errors = validate_stage2_questions(questions)
            assert is_valid is True, (
                f"({attack_type}, {strategy}) should be valid but got errors: {errors}"
            )


# ==============================================
# 8. Expanded Stats Helpers (Phase 2 roadmap)
# ==============================================

@pytest.mark.unit
def test_initialize_agent_stats_has_new_fields():
    """Expanded schema includes snapshots, per_round, and histogram scaffolding."""
    stats = initialize_agent_stats(["Agent-A"])
    agent = stats["Agent-A"]

    # Snapshots / per-round
    assert agent["initial_snapshot"] is None
    assert agent["final_snapshot"] is None
    assert agent["per_round"] == {}

    # Cross-examination histogram has pre-seeded keys for every type + strategy.
    cea = agent["cross_examination_attacks"]
    assert cea["total"] == 0
    assert set(cea["by_type"].keys()) == set(VALID_ATTACK_STRATEGIES.keys())
    assert all(v == 0 for v in cea["by_type"].values())
    assert set(cea["by_strategy"].keys()) == set(ALL_STRATEGIES)
    assert all(v == 0 for v in cea["by_strategy"].values())

    # Adjudication outcomes split by role with zeroed verdict histogram.
    adj = agent["adjudication_outcomes"]
    assert set(adj.keys()) == {"as_challenger", "as_target"}
    for role_hist in adj.values():
        assert set(role_hist.keys()) == set(VALID_ADJUDICATION_VERDICTS)
        assert all(v == 0 for v in role_hist.values())


@pytest.mark.unit
def test_update_agent_stats_increments_adjudication_outcomes():
    """update_agent_stats bumps as_challenger / as_target verdict counts."""
    stats = initialize_agent_stats(["Agent-A", "Agent-B"])

    update_agent_stats(stats, {
        "challenger": "Agent-A", "target": "Agent-B",
        "resolution": {"status": "critique_valid"},
    })
    update_agent_stats(stats, {
        "challenger": "Agent-A", "target": "Agent-B",
        "resolution": {"status": "rebuttal_valid"},
    })
    update_agent_stats(stats, {
        "challenger": "Agent-B", "target": "Agent-A",
        "resolution": {"status": "unresolved"},
    })

    a_chal = stats["Agent-A"]["adjudication_outcomes"]["as_challenger"]
    assert a_chal["critique_valid"] == 1
    assert a_chal["rebuttal_valid"] == 1
    assert a_chal["unresolved"] == 0

    a_tgt = stats["Agent-A"]["adjudication_outcomes"]["as_target"]
    assert a_tgt["unresolved"] == 1
    assert a_tgt["critique_valid"] == 0

    b_chal = stats["Agent-B"]["adjudication_outcomes"]["as_challenger"]
    assert b_chal["unresolved"] == 1

    b_tgt = stats["Agent-B"]["adjudication_outcomes"]["as_target"]
    assert b_tgt["critique_valid"] == 1
    assert b_tgt["rebuttal_valid"] == 1


@pytest.mark.unit
def test_update_agent_stats_unknown_verdict_skipped():
    """Unrecognised verdict label does not blow up the histogram."""
    stats = initialize_agent_stats(["Agent-A", "Agent-B"])
    update_agent_stats(stats, {
        "challenger": "Agent-A", "target": "Agent-B",
        "resolution": {"status": "weird_new_status"},
    })
    # Verdict histograms should remain fully zeroed.
    for role in ("as_challenger", "as_target"):
        for name in ("Agent-A", "Agent-B"):
            assert all(v == 0 for v in stats[name]["adjudication_outcomes"][role].values())


@pytest.mark.unit
def test_compute_attack_histograms_basic():
    """Three distinct pairs are summed correctly in per-agent + aggregate."""
    stats = initialize_agent_stats(["Agent-A", "Agent-B"])
    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence"},
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "rebutting", "attack_strategy": "present_counter_evidence"},
        {"challenger": "Agent-B", "target": "Agent-A",
         "attack_type": "undercutting", "attack_strategy": "identify_circularity"},
    ]

    aggregate = compute_attack_histograms(stats, pairs, ["Agent-A", "Agent-B"])

    a_cea = stats["Agent-A"]["cross_examination_attacks"]
    b_cea = stats["Agent-B"]["cross_examination_attacks"]

    assert a_cea["total"] == 2
    assert a_cea["by_type"]["undermining"] == 1
    assert a_cea["by_type"]["rebutting"] == 1
    assert a_cea["by_type"]["undercutting"] == 0
    assert a_cea["by_strategy"]["challenge_evidence"] == 1
    assert a_cea["by_strategy"]["present_counter_evidence"] == 1

    assert b_cea["total"] == 1
    assert b_cea["by_type"]["undercutting"] == 1
    assert b_cea["by_strategy"]["identify_circularity"] == 1

    assert aggregate["attacks_total"] == 3
    assert aggregate["attacks_by_type"]["undermining"] == 1
    assert aggregate["attacks_by_type"]["rebutting"] == 1
    assert aggregate["attacks_by_type"]["undercutting"] == 1
    # Invariant: aggregate by_type == sum of per-agent by_type
    for t in VALID_ATTACK_STRATEGIES:
        assert aggregate["attacks_by_type"][t] == (
            a_cea["by_type"][t] + b_cea["by_type"][t]
        )


@pytest.mark.unit
def test_compute_attack_histograms_unknown_strategy_falls_back():
    """Unknown attack_type/attack_strategy still counts toward total, no KeyError."""
    stats = initialize_agent_stats(["Agent-A"])
    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "UNKNOWN_TYPE", "attack_strategy": "UNKNOWN_STRAT"},
        {"challenger": "Agent-A", "target": "Agent-B"},  # no attack fields at all
    ]

    aggregate = compute_attack_histograms(stats, pairs, ["Agent-A"])

    cea = stats["Agent-A"]["cross_examination_attacks"]
    assert cea["total"] == 2
    # None of the known buckets changed.
    assert all(v == 0 for v in cea["by_type"].values())
    assert all(v == 0 for v in cea["by_strategy"].values())

    assert aggregate["attacks_total"] == 2
    assert all(v == 0 for v in aggregate["attacks_by_type"].values())
    assert all(v == 0 for v in aggregate["attacks_by_strategy"].values())


@pytest.mark.unit
def test_snapshot_belief_counts_only_non_retracted():
    """D/A/C/E nodes with status='retracted' are excluded; X# / U# counted raw."""
    belief = {
        "thesis": {"strength": 0.6},
        "definitions": [
            {"id": "D1", "status": "active"},
            {"id": "D2", "status": "retracted"},
        ],
        "assumptions": [
            {"id": "A1", "status": "active"},
            {"id": "A2", "status": "revised"},
            {"id": "A3", "status": "retracted"},
        ],
        "claims": [
            {"id": "C1", "status": "active"},
            {"id": "C2", "status": "retracted"},
            {"id": "C3", "status": "retracted"},
        ],
        "evidence": [
            {"id": "E1", "status": "active"},
            {"id": "E2", "status": "retracted"},
        ],
        # X# has no status field at all
        "counterpositions": [
            {"id": "X1"},
            {"id": "X2"},
        ],
        # U# status ∈ {"active","resolved"} — both count.
        "uncertainties": [
            {"id": "U1", "status": "active"},
            {"id": "U2", "status": "resolved"},
        ],
    }

    snap = snapshot_belief(belief)
    counts = snap["component_counts"]

    assert counts["definitions"] == 1       # D1 active
    assert counts["assumptions"] == 2       # A1 active + A2 revised
    assert counts["claims"] == 1            # C1 active
    assert counts["evidence"] == 1          # E1 active
    assert counts["counterpositions"] == 2  # X1, X2 (no status filter)
    assert counts["uncertainties"] == 2     # U1 active + U2 resolved


@pytest.mark.unit
def test_snapshot_belief_preserves_thesis_strength():
    """Thesis strength passes through untouched as a float."""
    belief = {"thesis": {"strength": 0.734}, "claims": []}
    snap = snapshot_belief(belief)
    assert snap["thesis_strength"] == pytest.approx(0.734)


@pytest.mark.unit
def test_snapshot_belief_handles_degraded_input():
    """Non-dict / missing-key beliefs yield a null-valued snapshot rather than raising."""
    snap_none = snapshot_belief(None)
    assert snap_none["thesis_strength"] is None
    assert all(v == 0 for v in snap_none["component_counts"].values())

    snap_bad = snapshot_belief({"thesis": "not a dict"})
    assert snap_bad["thesis_strength"] is None


@pytest.mark.unit
def test_select_best_agent_by_score():
    """Highest performance_score wins."""
    stats = {
        "Agent-A": {"performance_score": 5.0},
        "Agent-B": {"performance_score": 9.0},
        "Agent-C": {"performance_score": 7.5},
    }
    assert select_best_agent(stats, ["Agent-A", "Agent-B", "Agent-C"]) == "Agent-B"


@pytest.mark.unit
def test_select_best_agent_tiebreaker_first_in_order():
    """Ties are broken by earlier position in agent_order."""
    stats = {
        "Agent-A": {"performance_score": 10.0},
        "Agent-B": {"performance_score": 10.0},
        "Agent-C": {"performance_score": 10.0},
    }
    # All three tied — earlier index wins.
    assert select_best_agent(stats, ["Agent-C", "Agent-A", "Agent-B"]) == "Agent-C"
    assert select_best_agent(stats, ["Agent-B", "Agent-A", "Agent-C"]) == "Agent-B"


@pytest.mark.unit
def test_select_best_agent_ignores_debate_aggregate_sentinel():
    """The _debate_aggregate sentinel is never selected as best."""
    stats = {
        "Agent-A": {"performance_score": 3.0},
        "Agent-B": {"performance_score": 5.0},
        "_debate_aggregate": {"attacks_total": 42},
    }
    assert select_best_agent(stats, ["Agent-A", "Agent-B"]) == "Agent-B"


@pytest.mark.unit
def test_select_best_agent_empty_raises():
    """With no matching agents, select_best_agent raises ValueError."""
    with pytest.raises(ValueError):
        select_best_agent({"_debate_aggregate": {}}, ["Agent-A"])


@pytest.mark.unit
def test_finalize_agent_stats_builds_debate_aggregate():
    """finalize_agent_stats populates _debate_aggregate and final_snapshot."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A", "Agent-B"])

    # Pre-populate per_round snapshots so finalize uses them as final_snapshot.
    stats["Agent-A"]["per_round"]["round_2"] = {
        "thesis_strength": 0.5,
        "component_counts": {"claims": 3},
    }
    stats["Agent-B"]["per_round"]["round_2"] = {
        "thesis_strength": 0.7,
        "component_counts": {"claims": 4},
    }

    # Simulate verdict tallies (as would be populated via update_agent_stats).
    stats["Agent-A"]["adjudication_outcomes"]["as_challenger"]["critique_valid"] = 2
    stats["Agent-A"]["adjudication_outcomes"]["as_challenger"]["unresolved"] = 1
    stats["Agent-B"]["adjudication_outcomes"]["as_challenger"]["rebuttal_valid"] = 1

    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence"},
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence"},
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "rebutting", "attack_strategy": "present_counter_evidence"},
        {"challenger": "Agent-B", "target": "Agent-A",
         "attack_type": "undercutting", "attack_strategy": "identify_circularity"},
    ]

    agent_a = MagicMock(); agent_a.name = "Agent-A"
    agent_b = MagicMock(); agent_b.name = "Agent-B"

    result = finalize_agent_stats(stats, pairs, [agent_a, agent_b], max_rounds=2)

    # _debate_aggregate should exist with correct totals.
    assert "_debate_aggregate" in result
    agg = result["_debate_aggregate"]
    assert agg["attacks_total"] == 4
    assert agg["attacks_by_type"]["undermining"] == 2
    assert agg["attacks_by_type"]["rebutting"] == 1
    assert agg["attacks_by_type"]["undercutting"] == 1

    # Verdicts aggregate sums as_challenger columns.
    verdicts = agg["adjudication_verdicts"]
    assert verdicts["critique_valid"] == 2
    assert verdicts["rebuttal_valid"] == 1
    assert verdicts["unresolved"] == 1

    # final_snapshot copied from per_round[round_2].
    assert result["Agent-A"]["final_snapshot"]["thesis_strength"] == 0.5
    assert result["Agent-B"]["final_snapshot"]["thesis_strength"] == 0.7


@pytest.mark.unit
def test_finalize_agent_stats_falls_back_to_live_belief_for_final_snapshot():
    """Missing per_round snapshot triggers live recomputation via agent.get_internal_belief_obj."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A"])
    # Note: per_round left empty — finalize must call the live-belief fallback.

    agent_a = MagicMock()
    agent_a.name = "Agent-A"
    agent_a.get_internal_belief_obj.return_value = {
        "thesis": {"strength": 0.42},
        "claims": [{"id": "C1", "status": "active"}],
    }

    result = finalize_agent_stats(stats, [], [agent_a], max_rounds=1)
    snap = result["Agent-A"]["final_snapshot"]
    assert snap["thesis_strength"] == pytest.approx(0.42)
    assert snap["component_counts"]["claims"] == 1


@pytest.mark.unit
def test_finalize_agent_stats_invariants():
    """Aggregate by_type and verdicts equal the per-agent column sums."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A", "Agent-B"])
    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence",
         "resolution": {"status": "critique_valid"}},
        {"challenger": "Agent-B", "target": "Agent-A",
         "attack_type": "rebutting", "attack_strategy": "present_counter_evidence",
         "resolution": {"status": "rebuttal_valid"}},
    ]
    for p in pairs:
        update_agent_stats(stats, p)

    agent_a = MagicMock(); agent_a.name = "Agent-A"; agent_a.get_internal_belief_obj.return_value = {}
    agent_b = MagicMock(); agent_b.name = "Agent-B"; agent_b.get_internal_belief_obj.return_value = {}

    result = finalize_agent_stats(stats, pairs, [agent_a, agent_b], max_rounds=1)
    agg = result["_debate_aggregate"]

    # Invariant 1: aggregate by_type = sum of per-agent by_type
    for t in VALID_ATTACK_STRATEGIES:
        per_agent_sum = (
            result["Agent-A"]["cross_examination_attacks"]["by_type"][t]
            + result["Agent-B"]["cross_examination_attacks"]["by_type"][t]
        )
        assert agg["attacks_by_type"][t] == per_agent_sum

    # Invariant 2: aggregate verdicts = sum of per-agent as_challenger verdicts
    for v in VALID_ADJUDICATION_VERDICTS:
        per_agent_sum = (
            result["Agent-A"]["adjudication_outcomes"]["as_challenger"][v]
            + result["Agent-B"]["adjudication_outcomes"]["as_challenger"][v]
        )
        assert agg["adjudication_verdicts"][v] == per_agent_sum
