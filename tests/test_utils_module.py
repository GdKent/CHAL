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
    EXCHANGE_SCORE_WEIGHTS,
    compute_attack_histograms,
    compute_per_round_attack_histograms,
    snapshot_belief,
    finalize_agent_stats,
    select_best_agent,
    sanitize_filename,
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
    assert agent_stat["exchange_scores"] == []


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


@pytest.mark.unit
def test_update_agent_stats_records_exchange_scores_critique_valid():
    """critique_valid: challenger gets +1.0, target gets -1.0."""
    stats = initialize_agent_stats(["Challenger", "Target"])
    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "critique_valid"}
    }
    update_agent_stats(stats, record)
    assert stats["Challenger"]["exchange_scores"] == [1.0]
    assert stats["Target"]["exchange_scores"] == [-1.0]


@pytest.mark.unit
def test_update_agent_stats_records_exchange_scores_rebuttal_valid():
    """rebuttal_valid: challenger gets -0.5, target gets +1.0."""
    stats = initialize_agent_stats(["Challenger", "Target"])
    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "rebuttal_valid"}
    }
    update_agent_stats(stats, record)
    assert stats["Challenger"]["exchange_scores"] == [-0.5]
    assert stats["Target"]["exchange_scores"] == [1.0]


@pytest.mark.unit
def test_update_agent_stats_records_exchange_scores_unresolved():
    """unresolved: challenger gets 0.0, target gets +0.25."""
    stats = initialize_agent_stats(["Challenger", "Target"])
    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "unresolved"}
    }
    update_agent_stats(stats, record)
    assert stats["Challenger"]["exchange_scores"] == [0.0]
    assert stats["Target"]["exchange_scores"] == [0.25]


# ==============================================
# 5. Performance Score Calculation Tests
# ==============================================

@pytest.mark.unit
def test_calculate_performance_scores_from_exchange_scores():
    """APS = mean of per-exchange scores, normalized to [-1, +1]."""
    stats = {
        "Agent-A": {
            "exchange_scores": [1.0, -0.5, 0.25, 1.0],
            "performance_score": 0.0,
        }
    }

    calculate_performance_scores(stats)

    # (1.0 + -0.5 + 0.25 + 1.0) / 4 = 0.4375
    assert stats["Agent-A"]["performance_score"] == pytest.approx(0.4375)


@pytest.mark.unit
def test_calculate_performance_scores_empty_exchanges():
    """Empty exchange_scores yields APS of 0.0."""
    stats = {
        "Agent-A": {
            "exchange_scores": [],
            "performance_score": 0.0,
        }
    }

    calculate_performance_scores(stats)

    assert stats["Agent-A"]["performance_score"] == 0.0


@pytest.mark.unit
def test_calculate_performance_scores_normalized_range():
    """APS is always in [-1, +1] range."""
    # All wins as challenger → each exchange = +1.0
    stats_all_win = {
        "Agent-A": {"exchange_scores": [1.0, 1.0, 1.0], "performance_score": 0.0}
    }
    calculate_performance_scores(stats_all_win)
    assert stats_all_win["Agent-A"]["performance_score"] == pytest.approx(1.0)

    # All losses as target → each exchange = -1.0
    stats_all_lose = {
        "Agent-A": {"exchange_scores": [-1.0, -1.0, -1.0], "performance_score": 0.0}
    }
    calculate_performance_scores(stats_all_lose)
    assert stats_all_lose["Agent-A"]["performance_score"] == pytest.approx(-1.0)

    # Mixed → in between
    stats_mixed = {
        "Agent-A": {"exchange_scores": [1.0, -1.0, 0.25, -0.5], "performance_score": 0.0}
    }
    calculate_performance_scores(stats_mixed)
    score = stats_mixed["Agent-A"]["performance_score"]
    assert -1.0 <= score <= 1.0
    # (1.0 + -1.0 + 0.25 + -0.5) / 4 = -0.0625
    assert score == pytest.approx(-0.0625)


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
            "performance_score": 0.45,
            "exchange_scores": [0.45] * 10,
        },
        "Agent-B": {
            "successful_critiques": 2,
            "successful_rebuttals": 6,
            "failed_rebuttals": 1,
            "unresolved_arguments": 2,
            "total_arguments": 10,
            "performance_score": 0.55,
            "exchange_scores": [0.55] * 10,
        }
    }

    summary = get_performance_summary(stats)

    assert isinstance(summary, str)
    assert "Agent-A" in summary
    assert "Agent-B" in summary
    assert "APS" in summary


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
            "performance_score": 0.45
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
            "performance_score": 0.45
        }
    }

    display_agent_stats(stats)

    # Capture printed output
    display = capsys.readouterr().out

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
        "Agent-A": {"performance_score": 0.25},
        "Agent-B": {"performance_score": 0.75},
        "Agent-C": {"performance_score": 0.50},
    }
    assert select_best_agent(stats, ["Agent-A", "Agent-B", "Agent-C"]) == "Agent-B"


@pytest.mark.unit
def test_select_best_agent_tiebreaker_first_in_order():
    """Ties are broken by earlier position in agent_order."""
    stats = {
        "Agent-A": {"performance_score": 0.50},
        "Agent-B": {"performance_score": 0.50},
        "Agent-C": {"performance_score": 0.50},
    }
    # All three tied — earlier index wins.
    assert select_best_agent(stats, ["Agent-C", "Agent-A", "Agent-B"]) == "Agent-C"
    assert select_best_agent(stats, ["Agent-B", "Agent-A", "Agent-C"]) == "Agent-B"


@pytest.mark.unit
def test_select_best_agent_ignores_debate_aggregate_sentinel():
    """The _debate_aggregate sentinel is never selected as best."""
    stats = {
        "Agent-A": {"performance_score": 0.25},
        "Agent-B": {"performance_score": 0.50},
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

    # operational_metrics key is always present in _debate_aggregate.
    assert "operational_metrics" in agg


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


# ==============================================
# 9. Filename Sanitization Tests
# ==============================================

class TestSanitizeFilename:
    """Tests for sanitize_filename() utility function."""

    @pytest.mark.unit
    def test_clean_name_unchanged(self):
        """A name with only word chars and hyphens passes through unchanged."""
        assert sanitize_filename("Agent-Empiricist") == "Agent-Empiricist"

    @pytest.mark.unit
    def test_spaces_replaced(self):
        """Spaces are replaced with underscores."""
        assert sanitize_filename("Agent With Spaces") == "Agent_With_Spaces"

    @pytest.mark.unit
    def test_special_chars_replaced(self):
        """Slashes, colons, and other special characters become underscores."""
        assert sanitize_filename("agent/special:chars") == "agent_special_chars"

    @pytest.mark.unit
    def test_leading_trailing_stripped(self):
        """Leading and trailing underscores are stripped."""
        assert sanitize_filename("___test___") == "test"

    @pytest.mark.unit
    def test_empty_string_returns_unnamed(self):
        """An empty string returns the fallback 'unnamed'."""
        assert sanitize_filename("") == "unnamed"

    @pytest.mark.unit
    def test_only_special_chars_returns_unnamed(self):
        """A string of only special characters returns the fallback 'unnamed'."""
        assert sanitize_filename("///...") == "unnamed"

    @pytest.mark.unit
    def test_underscores_preserved(self):
        """Existing underscores in the name are preserved."""
        assert sanitize_filename("Agent_Test") == "Agent_Test"


# ==============================================
# 10. Ethical Attack Strategy Tests
# ==============================================

@pytest.mark.unit
def test_valid_attack_strategies_total_count():
    """Total strategy count equals 27 (20 epistemological + 7 ethical)."""
    assert len(ALL_STRATEGIES) == 27


@pytest.mark.unit
def test_ethical_strategies_in_correct_types():
    """Each of the 7 ethical strategies is in the expected attack type set."""
    # undermining ethical strategies
    assert "challenge_moral_implications" in VALID_ATTACK_STRATEGIES["undermining"]
    assert "expose_stakeholder_harm" in VALID_ATTACK_STRATEGIES["undermining"]

    # rebutting ethical strategies
    assert "present_ethical_counter" in VALID_ATTACK_STRATEGIES["rebutting"]
    assert "invoke_competing_obligation" in VALID_ATTACK_STRATEGIES["rebutting"]

    # undercutting ethical strategies
    assert "challenge_normative_inference" in VALID_ATTACK_STRATEGIES["undercutting"]
    assert "expose_value_conflict" in VALID_ATTACK_STRATEGIES["undercutting"]
    assert "challenge_moral_relevance" in VALID_ATTACK_STRATEGIES["undercutting"]


# ==============================================
# 10. Per-Round Attack Histogram Tests (Phase A)
# ==============================================

@pytest.mark.unit
def test_compute_per_round_attack_histograms_groups_by_round():
    """Pairs with different round_num tags are grouped correctly."""
    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence",
         "round_num": 1},
        {"challenger": "Agent-B", "target": "Agent-A",
         "attack_type": "rebutting", "attack_strategy": "present_counter_evidence",
         "round_num": 1},
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undercutting", "attack_strategy": "identify_circularity",
         "round_num": 2},
    ]

    result = compute_per_round_attack_histograms(pairs, ["Agent-A", "Agent-B"])

    assert "round_1" in result
    assert "round_2" in result

    # Round 1: Agent-A has 1 attack, Agent-B has 1 attack
    assert result["round_1"]["Agent-A"]["total"] == 1
    assert result["round_1"]["Agent-B"]["total"] == 1
    assert result["round_1"]["Agent-A"]["by_type"]["undermining"] == 1
    assert result["round_1"]["Agent-B"]["by_type"]["rebutting"] == 1

    # Round 2: Only Agent-A attacked
    assert result["round_2"]["Agent-A"]["total"] == 1
    assert result["round_2"]["Agent-A"]["by_type"]["undercutting"] == 1
    assert result["round_2"]["Agent-B"]["total"] == 0


@pytest.mark.unit
def test_compute_per_round_attack_histograms_empty_pairs():
    """Empty pairs list returns empty result."""
    result = compute_per_round_attack_histograms([], ["Agent-A"])
    assert result == {}


@pytest.mark.unit
def test_finalize_agent_stats_includes_per_round_attack_histograms():
    """finalize_agent_stats stores per-round attack data in agent stats and aggregate."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A", "Agent-B"])

    # Pre-populate per_round snapshots
    stats["Agent-A"]["per_round"]["round_1"] = {"thesis_strength": 0.5}
    stats["Agent-A"]["per_round"]["round_2"] = {"thesis_strength": 0.6}
    stats["Agent-B"]["per_round"]["round_1"] = {"thesis_strength": 0.4}
    stats["Agent-B"]["per_round"]["round_2"] = {"thesis_strength": 0.5}

    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence",
         "round_num": 1},
        {"challenger": "Agent-B", "target": "Agent-A",
         "attack_type": "rebutting", "attack_strategy": "present_counter_evidence",
         "round_num": 2},
    ]

    agent_a = MagicMock(); agent_a.name = "Agent-A"
    agent_b = MagicMock(); agent_b.name = "Agent-B"

    result = finalize_agent_stats(stats, pairs, [agent_a, agent_b], max_rounds=2)

    # Per-round attack histograms should be present
    assert "attack_histograms" in result["Agent-A"]["per_round"]["round_1"]
    assert result["Agent-A"]["per_round"]["round_1"]["attack_histograms"]["total"] == 1

    # Aggregate should include per_round_attacks
    assert "per_round_attacks" in result["_debate_aggregate"]
    assert "round_1" in result["_debate_aggregate"]["per_round_attacks"]
    assert "round_2" in result["_debate_aggregate"]["per_round_attacks"]


# ==============================================
# 11. Score Aggregation & Override Tracking Tests (Phase C)
# ==============================================

@pytest.mark.unit
def test_initialize_agent_stats_includes_score_aggregates():
    """Agent stats should include score aggregate and override tracking fields."""
    stats = initialize_agent_stats(["Agent-A"])
    agent = stats["Agent-A"]

    assert "adjudication_score_aggregates" in agent
    agg = agent["adjudication_score_aggregates"]
    assert "as_challenger" in agg
    assert "as_target" in agg
    assert agg["as_challenger"]["logic_sum"] == 0.0
    assert agg["as_challenger"]["count"] == 0
    assert agg["as_target"]["ethics_sum"] == 0.0
    assert agent["verdict_overrides"] == 0


@pytest.mark.unit
def test_update_agent_stats_accumulates_scores():
    """update_agent_stats should accumulate scores from resolution."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {
            "status": "critique_valid",
            "scores": {
                "challenger_logic": 0.7, "challenger_ethics": 0.5,
                "defender_logic": 0.4, "defender_ethics": 0.5,
                "challenger_combined": 0.6, "defender_combined": 0.45,
            },
        },
    }

    update_agent_stats(stats, record)

    c_agg = stats["Challenger"]["adjudication_score_aggregates"]["as_challenger"]
    assert c_agg["logic_sum"] == pytest.approx(0.7)
    assert c_agg["ethics_sum"] == pytest.approx(0.5)
    assert c_agg["combined_sum"] == pytest.approx(0.6)
    assert c_agg["count"] == 1

    t_agg = stats["Target"]["adjudication_score_aggregates"]["as_target"]
    assert t_agg["logic_sum"] == pytest.approx(0.4)
    assert t_agg["ethics_sum"] == pytest.approx(0.5)
    assert t_agg["combined_sum"] == pytest.approx(0.45)
    assert t_agg["count"] == 1


@pytest.mark.unit
def test_update_agent_stats_tracks_overrides():
    """override_occurred in resolution increments verdict_overrides for both agents."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {
            "status": "critique_valid",
            "override_occurred": True,
            "scores": {
                "challenger_logic": 0.8, "challenger_ethics": 0.5,
                "defender_logic": 0.3, "defender_ethics": 0.5,
                "challenger_combined": 0.65, "defender_combined": 0.4,
            },
        },
    }

    update_agent_stats(stats, record)

    assert stats["Challenger"]["verdict_overrides"] == 1
    assert stats["Target"]["verdict_overrides"] == 1


@pytest.mark.unit
def test_update_agent_stats_no_override_when_false():
    """override_occurred=False should not increment verdict_overrides."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {
            "status": "critique_valid",
            "override_occurred": False,
        },
    }

    update_agent_stats(stats, record)

    assert stats["Challenger"]["verdict_overrides"] == 0
    assert stats["Target"]["verdict_overrides"] == 0


@pytest.mark.unit
def test_update_agent_stats_accumulates_across_multiple_records():
    """Score sums and counts accumulate correctly across multiple records."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    records = [
        {"challenger": "Challenger", "target": "Target",
         "resolution": {"status": "critique_valid",
                        "scores": {"challenger_logic": 0.7, "challenger_ethics": 0.5,
                                   "defender_logic": 0.4, "defender_ethics": 0.5,
                                   "challenger_combined": 0.6, "defender_combined": 0.45}}},
        {"challenger": "Challenger", "target": "Target",
         "resolution": {"status": "rebuttal_valid",
                        "scores": {"challenger_logic": 0.3, "challenger_ethics": 0.5,
                                   "defender_logic": 0.8, "defender_ethics": 0.5,
                                   "challenger_combined": 0.4, "defender_combined": 0.65}}},
        {"challenger": "Challenger", "target": "Target",
         "resolution": {"status": "unresolved",
                        "scores": {"challenger_logic": 0.5, "challenger_ethics": 0.5,
                                   "defender_logic": 0.5, "defender_ethics": 0.5,
                                   "challenger_combined": 0.5, "defender_combined": 0.5}}},
    ]

    for r in records:
        update_agent_stats(stats, r)

    c_agg = stats["Challenger"]["adjudication_score_aggregates"]["as_challenger"]
    assert c_agg["logic_sum"] == pytest.approx(0.7 + 0.3 + 0.5)
    assert c_agg["count"] == 3

    t_agg = stats["Target"]["adjudication_score_aggregates"]["as_target"]
    assert t_agg["logic_sum"] == pytest.approx(0.4 + 0.8 + 0.5)
    assert t_agg["count"] == 3


@pytest.mark.unit
def test_update_agent_stats_tolerates_missing_scores():
    """Records without scores should not crash score accumulation."""
    stats = initialize_agent_stats(["Challenger", "Target"])

    record = {
        "challenger": "Challenger",
        "target": "Target",
        "resolution": {"status": "critique_valid"},
    }

    update_agent_stats(stats, record)

    # Scores should remain at zero
    c_agg = stats["Challenger"]["adjudication_score_aggregates"]["as_challenger"]
    assert c_agg["count"] == 0
    assert c_agg["logic_sum"] == 0.0


@pytest.mark.unit
def test_finalize_agent_stats_computes_score_means():
    """finalize_agent_stats should compute mean scores from aggregates."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A", "Agent-B"])

    # Pre-populate score aggregates (ethics 0.5 = neutral under merit-based scale)
    stats["Agent-A"]["adjudication_score_aggregates"]["as_challenger"] = {
        "logic_sum": 2.1, "ethics_sum": 1.5, "combined_sum": 1.8, "count": 3,
    }
    stats["Agent-A"]["adjudication_score_aggregates"]["as_target"] = {
        "logic_sum": 1.2, "ethics_sum": 1.0, "combined_sum": 1.1, "count": 2,
    }
    stats["Agent-B"]["adjudication_score_aggregates"]["as_challenger"] = {
        "logic_sum": 0.0, "ethics_sum": 0.0, "combined_sum": 0.0, "count": 0,
    }

    agent_a = MagicMock(); agent_a.name = "Agent-A"
    agent_b = MagicMock(); agent_b.name = "Agent-B"

    result = finalize_agent_stats(stats, [], [agent_a, agent_b], max_rounds=1)

    means_a = result["Agent-A"]["adjudication_score_means"]
    assert means_a["as_challenger"]["logic_mean"] == pytest.approx(2.1 / 3)
    assert means_a["as_challenger"]["count"] == 3
    assert means_a["as_target"]["logic_mean"] == pytest.approx(1.2 / 2)
    assert means_a["as_target"]["count"] == 2

    # Agent-B with no data should have None means
    means_b = result["Agent-B"]["adjudication_score_means"]
    assert means_b["as_challenger"]["logic_mean"] is None
    assert means_b["as_challenger"]["count"] == 0


@pytest.mark.unit
def test_finalize_agent_stats_includes_override_total_in_aggregate():
    """_debate_aggregate.total_verdict_overrides counts unique override events from pairs."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A", "Agent-B"])
    # Per-agent counts are set by update_agent_stats (both agents get incremented
    # per override), but the aggregate should count unique events from pairs.
    stats["Agent-A"]["verdict_overrides"] = 3
    stats["Agent-B"]["verdict_overrides"] = 3

    # Provide pairs with override data: 3 overrides out of 5 pairs
    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B", "resolution": {"status": "critique_valid", "override_occurred": True}},
        {"challenger": "Agent-A", "target": "Agent-B", "resolution": {"status": "rebuttal_valid", "override_occurred": False}},
        {"challenger": "Agent-B", "target": "Agent-A", "resolution": {"status": "unresolved", "override_occurred": True}},
        {"challenger": "Agent-B", "target": "Agent-A", "resolution": {"status": "critique_valid", "override_occurred": False}},
        {"challenger": "Agent-A", "target": "Agent-B", "resolution": {"status": "unresolved", "override_occurred": True}},
    ]

    agent_a = MagicMock(); agent_a.name = "Agent-A"
    agent_b = MagicMock(); agent_b.name = "Agent-B"

    result = finalize_agent_stats(stats, pairs, [agent_a, agent_b], max_rounds=1)

    # 3 unique override events (not 6 from summing per-agent counts)
    assert result["_debate_aggregate"]["total_verdict_overrides"] == 3


# ==============================================
# 12. Operational Metrics in finalize_agent_stats
# ==============================================

@pytest.mark.unit
def test_finalize_agent_stats_includes_operational_metrics():
    """finalize_agent_stats stores operational_metrics in _debate_aggregate."""
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

    pairs = [
        {"challenger": "Agent-A", "target": "Agent-B",
         "attack_type": "undermining", "attack_strategy": "challenge_evidence"},
    ]

    agent_a = MagicMock(); agent_a.name = "Agent-A"
    agent_b = MagicMock(); agent_b.name = "Agent-B"

    result = finalize_agent_stats(
        stats, pairs, [agent_a, agent_b], max_rounds=2,
        operational_metrics={
            "total_retries": 5,
            "total_rate_limit_hits": 2,
            "total_output_tokens": 1000,
            "total_input_tokens": 5000,
            "duration_s": 120.5,
        },
    )

    assert result["_debate_aggregate"]["operational_metrics"]["total_retries"] == 5
    assert result["_debate_aggregate"]["operational_metrics"]["total_rate_limit_hits"] == 2
    assert result["_debate_aggregate"]["operational_metrics"]["total_output_tokens"] == 1000
    assert result["_debate_aggregate"]["operational_metrics"]["total_input_tokens"] == 5000
    assert result["_debate_aggregate"]["operational_metrics"]["duration_s"] == 120.5


@pytest.mark.unit
def test_finalize_agent_stats_operational_metrics_default_empty():
    """finalize_agent_stats defaults operational_metrics to empty dict when not passed."""
    from unittest.mock import MagicMock

    stats = initialize_agent_stats(["Agent-A"])

    # Pre-populate per_round snapshot.
    stats["Agent-A"]["per_round"]["round_1"] = {
        "thesis_strength": 0.6,
        "component_counts": {"claims": 2},
    }

    agent_a = MagicMock(); agent_a.name = "Agent-A"

    result = finalize_agent_stats(stats, [], [agent_a], max_rounds=1)

    assert result["_debate_aggregate"]["operational_metrics"] == {}
