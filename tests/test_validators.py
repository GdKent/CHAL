"""
Tests for per-stage output validators.

Covers:
- Stage 4 (Adjudicator) validator: 8 tests
- Stage 1 (Initial Belief / CBS) validator: 8 tests
- Stage 2 (Cross-Examination) validator: 6 tests
- Stage 3 (Rebuttal) validator: 7 tests
- Stage 5 (Belief Update / Patches) validators: 10 tests
"""

import json
import pytest

from chal.orchestrator.adjudicator import validate_adjudicator_output
from chal.utilities.validators import (
    validate_stage1_output,
    validate_stage2_output,
    validate_stage3_output,
    validate_stage5_phase1_output,
    validate_stage5_phase2_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_json(obj: dict) -> str:
    """Wrap a dict in a fenced ```json block."""
    return f"```json\n{json.dumps(obj)}\n```"


def _make_valid_adjudicator_response(
    outcome="rebuttal_valid",
    reasoning="Test reasoning",
    restatement="Test restatement",
    formalization_challenger="P1: X\nP2: Y\nC: Z",
    formalization_target="P1: A\nP2: B\nC: C",
    scores=None,
) -> str:
    if scores is None:
        scores = {
            "challenger_logic": 0.5, "challenger_ethics": 0.5,
            "defender_logic": 0.5, "defender_ethics": 0.5,
            "challenger_combined": 0.5, "defender_combined": 0.5,
        }
    return (
        "<reasoning>Analysis here</reasoning>\n\n"
        + _wrap_json({
            "outcome": outcome,
            "reasoning": reasoning,
            "restatement": restatement,
            "formalization_challenger": formalization_challenger,
            "formalization_target": formalization_target,
            "scores": scores,
        })
    )


def _make_valid_belief() -> dict:
    """Return a minimal valid CBS belief dict."""
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-TEST-001",
        "version": 1,
        "metadata": {"topic_query": "Test topic", "agent_persona": "Test"},
        "thesis": {
            "stance": "Test stance",
            "summary_bullets": ["Bullet 1", "Bullet 2"],
            "strength": 0.75,
            "strength_reasoning": "Test reasoning",
        },
        "definitions": [
            {
                "id": "D1", "term": "test", "definition": "A test term.",
                "strength": 0.9, "strength_justification": "Well-established",
                "used_by": ["A1", "E1"], "status": "active",
            },
        ],
        "assumptions": [
            {
                "id": "A1", "type": "empirical", "statement": "Test assumption",
                "supports_claims": ["C1"], "strength": 0.85,
                "strength_justification": "Strong support",
                "supported_by_definitions": ["D1"], "status": "active",
            },
        ],
        "claims": [
            {
                "id": "C1", "type": "deductive",
                "statement": "Test claim", "depends_on": ["A1", "E1"],
                "strength": 0.8,
                "strength_justification": "Follows from A1 and E1",
                "status": "active",
                "inference_chain": [
                    {"role": "premise", "text": "A1 holds", "reference": "A1"},
                    {"role": "premise", "text": "E1 supports", "reference": "E1"},
                    {"role": "inference", "text": "Therefore C1", "inference_type": "deductive"},
                    {"role": "conclusion", "text": "Test claim"},
                ],
                "predictions": [
                    {"statement": "P1", "test": "T1", "decision_criterion": "DC1"},
                ],
            },
        ],
        "evidence": [
            {
                "id": "E1", "type": "empirical",
                "summary": "Test evidence", "source": "Test (2026)",
                "supports_claims": ["C1"], "strength": 0.8,
                "strength_justification": "Peer-reviewed",
                "supported_by_definitions": ["D1"], "status": "active",
            },
        ],
        "counterpositions": [],
        "uncertainties": [],
    }


def _make_valid_questions(n=3) -> dict:
    """Return a valid Stage 2 questions dict."""
    strategies = {
        "undermining": "challenge_evidence",
        "rebutting": "present_counter_evidence",
        "undercutting": "challenge_inference_step",
    }
    types = list(strategies.keys())
    return {
        "questions": [
            {
                "qid": f"Q{i+1}",
                "text": f"Question {i+1}?",
                "target_ids": ["C1"],
                "attack_type": types[i % 3],
                "attack_strategy": strategies[types[i % 3]],
            }
            for i in range(n)
        ]
    }


def _make_valid_rebuttals(qids=None) -> dict:
    """Return a valid Stage 3 rebuttals dict."""
    if qids is None:
        qids = ["Q1", "Q2", "Q3"]
    actions = ["refute", "concede", "defer"]
    return {
        "rebuttals": [
            {"qid": qid, "answer": f"Answer for {qid}", "action": actions[i % 3]}
            for i, qid in enumerate(qids)
        ],
        "patches": [],
    }


def _make_valid_patches() -> dict:
    """Return a valid Stage 5 patches dict."""
    return {
        "patches": [
            {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.5}},
            {"op": "update_thesis", "new_strength": 0.6},
        ]
    }


# ===========================================================================
# 6B: Stage 4 — Adjudicator Validator Tests
# ===========================================================================

@pytest.mark.unit
class TestAdjudicatorValidator:

    def test_valid_response(self):
        raw = _make_valid_adjudicator_response()
        result = validate_adjudicator_output(raw)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_json_block(self):
        raw = "<reasoning>Some analysis only</reasoning>"
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("no json block" in e.lower() for e in result.errors)

    def test_json_inside_reasoning_tags(self):
        """JSON nested inside <reasoning> — should still be extracted by brace scanner."""
        inner_json = json.dumps({
            "outcome": "rebuttal_valid",
            "reasoning": "ok",
            "restatement": "ok",
            "formalization_challenger": "P1: X\nC: Y",
            "formalization_target": "P1: A\nC: B",
            "scores": {
                "challenger_logic": 0.5, "challenger_ethics": 1.0,
                "defender_logic": 0.5, "defender_ethics": 1.0,
                "challenger_combined": 0.75, "defender_combined": 0.75,
            },
        })
        raw = f"<reasoning>{inner_json}</reasoning>"
        # The brace-depth scanner will find it even inside reasoning tags,
        # so this should actually pass. Verify the extractor behavior.
        result = validate_adjudicator_output(raw)
        # The adjudicator's _extract_json_from_response uses brace scanning,
        # so JSON inside reasoning tags IS found and valid.
        assert result.is_valid is True

    def test_missing_outcome(self):
        raw = _wrap_json({"reasoning": "ok", "restatement": "ok"})
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("outcome" in e.lower() for e in result.errors)

    def test_invalid_outcome_value(self):
        raw = _wrap_json({
            "outcome": "maybe",
            "reasoning": "ok",
            "restatement": "ok",
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("unrecognized" in e.lower() or "maybe" in e for e in result.errors)

    def test_missing_reasoning(self):
        raw = _wrap_json({
            "outcome": "rebuttal_valid",
            "restatement": "ok",
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("reasoning" in e.lower() for e in result.errors)

    def test_missing_restatement(self):
        raw = _wrap_json({
            "outcome": "rebuttal_valid",
            "reasoning": "ok",
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("restatement" in e.lower() for e in result.errors)

    def test_missing_scores_is_failure(self):
        """Scores are required — missing scores should fail validation."""
        raw = _wrap_json({
            "outcome": "critique_valid",
            "reasoning": "The critique was valid.",
            "restatement": "Whether X holds.",
            "formalization_challenger": "P1: X\nC: Y",
            "formalization_target": "P1: A\nC: B",
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("scores" in e.lower() for e in result.errors)

    def test_missing_formalization_challenger(self):
        """Missing formalization_challenger should fail validation."""
        raw = _wrap_json({
            "outcome": "critique_valid",
            "reasoning": "ok",
            "restatement": "ok",
            "formalization_target": "P1: A\nC: B",
            "scores": {
                "challenger_logic": 0.5, "challenger_ethics": 1.0,
                "defender_logic": 0.5, "defender_ethics": 1.0,
                "challenger_combined": 0.75, "defender_combined": 0.75,
            },
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("formalization_challenger" in e for e in result.errors)

    def test_missing_formalization_target(self):
        """Missing formalization_target should fail validation."""
        raw = _wrap_json({
            "outcome": "critique_valid",
            "reasoning": "ok",
            "restatement": "ok",
            "formalization_challenger": "P1: X\nC: Y",
            "scores": {
                "challenger_logic": 0.5, "challenger_ethics": 1.0,
                "defender_logic": 0.5, "defender_ethics": 1.0,
                "challenger_combined": 0.75, "defender_combined": 0.75,
            },
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("formalization_target" in e for e in result.errors)

    def test_missing_individual_score_key(self):
        """Missing a single score key should fail validation."""
        raw = _wrap_json({
            "outcome": "critique_valid",
            "reasoning": "ok",
            "restatement": "ok",
            "formalization_challenger": "P1: X\nC: Y",
            "formalization_target": "P1: A\nC: B",
            "scores": {
                "challenger_logic": 0.5, "challenger_ethics": 1.0,
                "defender_logic": 0.5, "defender_ethics": 1.0,
                "challenger_combined": 0.75,
                # missing defender_combined
            },
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("defender_combined" in e for e in result.errors)

    def test_score_out_of_range(self):
        """Score values outside [0.0, 1.0] should fail validation."""
        raw = _wrap_json({
            "outcome": "critique_valid",
            "reasoning": "ok",
            "restatement": "ok",
            "formalization_challenger": "P1: X\nC: Y",
            "formalization_target": "P1: A\nC: B",
            "scores": {
                "challenger_logic": 1.5,  # out of range
                "challenger_ethics": 1.0,
                "defender_logic": 0.5, "defender_ethics": 1.0,
                "challenger_combined": 0.75, "defender_combined": 0.75,
            },
        })
        result = validate_adjudicator_output(raw)
        assert result.is_valid is False
        assert any("out of range" in e.lower() for e in result.errors)


# ===========================================================================
# 6C: Stage 1 — Initial Belief (CBS) Validator Tests
# ===========================================================================

@pytest.mark.unit
class TestStage1Validator:

    def test_valid_belief(self):
        belief = _make_valid_belief()
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is True, f"Unexpected errors: {result.errors}"
        assert len(result.errors) == 0

    def test_no_json_block(self):
        raw = "I believe free will exists but here is no JSON."
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("no json block" in e.lower() for e in result.errors)

    def test_missing_top_level_keys(self):
        belief = {"schema_version": "CBS"}
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("thesis" in e.lower() for e in result.errors)
        assert any("claims" in e.lower() for e in result.errors)

    def test_empty_claims_array(self):
        belief = _make_valid_belief()
        belief["claims"] = []
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("claims" in e.lower() and "empty" in e.lower() for e in result.errors)

    def test_claim_missing_inference_chain(self):
        belief = _make_valid_belief()
        del belief["claims"][0]["inference_chain"]
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("inference_chain" in e for e in result.errors)

    def test_claim_missing_predictions(self):
        belief = _make_valid_belief()
        del belief["claims"][0]["predictions"]
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("predictions" in e for e in result.errors)

    def test_strength_out_of_range(self):
        belief = _make_valid_belief()
        belief["claims"][0]["strength"] = 1.5
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("out of range" in e.lower() for e in result.errors)

    def test_invalid_node_id(self):
        belief = _make_valid_belief()
        belief["assumptions"][0]["id"] = "assumption1"
        raw = _wrap_json(belief)
        result = validate_stage1_output(raw)
        assert result.is_valid is False
        assert any("invalid id format" in e.lower() or "assumption1" in e for e in result.errors)


# ===========================================================================
# 6D: Stage 2 — Cross-Examination Validator Tests
# ===========================================================================

@pytest.mark.unit
class TestStage2Validator:

    def test_valid_questions(self):
        raw = _wrap_json(_make_valid_questions(3))
        result = validate_stage2_output(raw)
        assert result.is_valid is True, f"Unexpected errors: {result.errors}"

    def test_no_json_block(self):
        raw = "Just some text about questions, no JSON here."
        result = validate_stage2_output(raw)
        assert result.is_valid is False
        assert any("no fenced" in e.lower() or "no json" in e.lower() for e in result.errors)

    def test_missing_questions_key(self):
        raw = _wrap_json({"data": "no questions key"})
        result = validate_stage2_output(raw)
        assert result.is_valid is False
        assert any("questions" in e.lower() for e in result.errors)

    def test_question_missing_attack_type(self):
        obj = _make_valid_questions(1)
        del obj["questions"][0]["attack_type"]
        raw = _wrap_json(obj)
        result = validate_stage2_output(raw)
        assert result.is_valid is False
        assert any("attack_type" in e for e in result.errors)

    def test_invalid_attack_strategy_for_type(self):
        """undermining + challenge_inference_step is invalid (that's an undercutting strategy)."""
        obj = _make_valid_questions(1)
        obj["questions"][0]["attack_type"] = "undermining"
        obj["questions"][0]["attack_strategy"] = "challenge_inference_step"
        raw = _wrap_json(obj)
        result = validate_stage2_output(raw)
        assert result.is_valid is False
        assert any("attack_strategy" in e and "challenge_inference_step" in e for e in result.errors)

    def test_invalid_target_id(self):
        obj = _make_valid_questions(1)
        obj["questions"][0]["target_ids"] = ["claim1"]
        raw = _wrap_json(obj)
        result = validate_stage2_output(raw)
        assert result.is_valid is False
        assert any("claim1" in e for e in result.errors)


# ===========================================================================
# 6E: Stage 3 — Rebuttal Validator Tests
# ===========================================================================

@pytest.mark.unit
class TestStage3Validator:

    def test_valid_rebuttals(self):
        raw = _wrap_json(_make_valid_rebuttals())
        result = validate_stage3_output(raw)
        assert result.is_valid is True, f"Unexpected errors: {result.errors}"

    def test_no_json_block(self):
        raw = "My rebuttal is that you are wrong. No JSON here."
        result = validate_stage3_output(raw)
        assert result.is_valid is False

    def test_missing_rebuttals_key(self):
        raw = _wrap_json({"data": "no rebuttals key"})
        result = validate_stage3_output(raw)
        assert result.is_valid is False
        assert any("rebuttals" in e.lower() for e in result.errors)

    def test_rebuttal_missing_action(self):
        obj = _make_valid_rebuttals(["Q1"])
        del obj["rebuttals"][0]["action"]
        raw = _wrap_json(obj)
        result = validate_stage3_output(raw)
        assert result.is_valid is False
        assert any("action" in e.lower() for e in result.errors)

    def test_invalid_action_value(self):
        obj = _make_valid_rebuttals(["Q1"])
        obj["rebuttals"][0]["action"] = "ignore"
        raw = _wrap_json(obj)
        result = validate_stage3_output(raw)
        assert result.is_valid is False
        assert any("ignore" in e for e in result.errors)

    def test_missing_rebuttal_for_qid(self):
        """expected_qids=["Q1","Q2"] but only Q1 answered."""
        obj = _make_valid_rebuttals(["Q1"])
        raw = _wrap_json(obj)
        result = validate_stage3_output(raw, expected_qids=["Q1", "Q2"])
        assert result.is_valid is False
        assert any("Q2" in e for e in result.errors)

    def test_no_expected_qids_skips_coverage_check(self):
        """Without expected_qids, partial coverage is fine."""
        obj = _make_valid_rebuttals(["Q1"])
        raw = _wrap_json(obj)
        result = validate_stage3_output(raw, expected_qids=None)
        assert result.is_valid is True


# ===========================================================================
# 6F: Stage 5 — Belief Update (Patches) Validator Tests
# ===========================================================================

@pytest.mark.unit
class TestStage5Phase1Validator:

    def test_valid_patches(self):
        raw = _wrap_json(_make_valid_patches())
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is True, f"Unexpected errors: {result.errors}"

    def test_no_json_block(self):
        raw = "I will update my beliefs. No JSON here."
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is False

    def test_missing_patches_key(self):
        raw = _wrap_json({"changes": []})
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is False
        assert any("patches" in e.lower() for e in result.errors)

    def test_empty_patches_with_critique_valid(self):
        """Empty patches when critique_valid_count > 0 should fail enforcement."""
        raw = _wrap_json({"patches": []})
        result = validate_stage5_phase1_output(raw, critique_valid_count=3)
        assert result.is_valid is False
        assert any("critique_valid" in e.lower() or "zero patches" in e.lower() for e in result.errors)

    def test_empty_patches_no_critique_valid(self):
        """Empty patches with critique_valid_count=0 is acceptable in Phase 1."""
        raw = _wrap_json({"patches": []})
        result = validate_stage5_phase1_output(raw, critique_valid_count=0)
        assert result.is_valid is True

    def test_patch_missing_op(self):
        raw = _wrap_json({"patches": [{"target_id": "C1"}]})
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is False
        assert any("op" in e.lower() for e in result.errors)

    def test_update_patch_missing_target_id(self):
        raw = _wrap_json({
            "patches": [{"op": "update_claim", "changes": {"strength": 0.5}}]
        })
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is False
        assert any("target_id" in e for e in result.errors)

    def test_add_patch_missing_item(self):
        raw = _wrap_json({
            "patches": [{"op": "add_uncertainty"}]
        })
        result = validate_stage5_phase1_output(raw)
        assert result.is_valid is False
        assert any("item" in e.lower() for e in result.errors)


@pytest.mark.unit
class TestStage5Phase2Validator:

    def test_valid_empty_patches(self):
        """Empty patches array is valid in Phase 2 (introspective)."""
        raw = _wrap_json({"patches": []})
        result = validate_stage5_phase2_output(raw)
        assert result.is_valid is True

    def test_missing_patches_key(self):
        raw = _wrap_json({"changes": []})
        result = validate_stage5_phase2_output(raw)
        assert result.is_valid is False
        assert any("patches" in e.lower() for e in result.errors)
