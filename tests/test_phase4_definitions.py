"""
Phase 4 tests for Definition Nodes (D#) — Combat Prompts.

Tests that Stage 2 (cross-examination), Stage 3 (rebuttals), and Stage 4
(adjudication) prompts correctly handle definitional attacks, D# defense
guidance, D# patch operations, and definitional evaluation criteria.
Also tests that validate_stage2_questions() accepts/rejects definitional attacks.
"""

import pytest
from chal.agents.prompts import (
    build_stage_2_prompt,
    build_stage_3_structured_rebuttal_prompt,
    build_adjudicator_per_pair_prompt,
    build_adjudicator_prompt,
)
from chal.utilities.utils import validate_stage2_questions


# ========================================
# Helpers
# ========================================

def _stage2(topic="test topic", agent="A1", opponent="B1",
            agent_belief='{}', opponent_belief='{}'):
    return build_stage_2_prompt(
        topic, agent, opponent, agent_belief, opponent_belief,
    )


def _stage3(topic="test topic", agent="A1", opponent="B1",
            questions='[]', belief='{}'):
    return build_stage_3_structured_rebuttal_prompt(
        topic, agent, opponent, questions, belief,
    )


def _per_pair(challenge="test challenge", rebuttal="test rebuttal"):
    return build_adjudicator_per_pair_prompt(
        challenge=challenge,
        rebuttal=rebuttal,
        challenger="A1",
        target="B1",
        mode_label="Pure Logic",
    )


def _system_adjudicator():
    logic_sys = {
        "label": "Classical Logic",
        "description": "Standard formal logic",
        "criteria": {
            "critique_valid": ["Identifies logical fallacy"],
            "rebuttal_valid": ["Provides valid counter-argument"],
            "unresolved": ["Neither side conclusively wins"],
        },
    }
    ethics_sys = {
        "label": "Utilitarianism",
        "description": "Greatest good",
        "criteria": {
            "critique_valid": ["Shows harm"],
            "rebuttal_valid": ["Shows benefit"],
            "unresolved": ["Neither side conclusively wins"],
        },
    }
    return build_adjudicator_prompt(1.0, 0.0, logic_sys, ethics_sys)


# ========================================
# 1. Stage 2: Definition-targeting attack strategies
# ========================================

class TestStage2Definitional:
    """Stage 2 cross-examination includes definition-targeting strategies under undermining/undercutting."""

    @pytest.mark.unit
    def test_definition_strategies_under_undermining(self):
        """over_extension and under_extension appear under UNDERMINING."""
        prompt = _stage2()
        idx = prompt.index("UNDERMINING")
        # Find the next section heading
        end = prompt.index("REBUTTING", idx)
        section = prompt[idx:end]
        assert "over_extension" in section
        assert "under_extension" in section

    @pytest.mark.unit
    def test_definition_strategies_under_undercutting(self):
        """circularity, stipulative_bias, conceptual_conflation appear under UNDERCUTTING."""
        prompt = _stage2()
        idx = prompt.index("UNDERCUTTING")
        section = prompt[idx:]
        assert "circularity" in section
        assert "stipulative_bias" in section
        assert "conceptual_conflation" in section

    @pytest.mark.unit
    def test_all_five_definition_strategies(self):
        """All 5 definition-targeting sub-strategies are listed."""
        prompt = _stage2()
        for strategy in ("circularity", "over_extension", "under_extension",
                         "stipulative_bias", "conceptual_conflation"):
            assert strategy in prompt, f"Missing strategy: {strategy}"

    @pytest.mark.unit
    def test_definitional_targets_d_nodes(self):
        """Definition attack instructions mention D# targeting."""
        prompt = _stage2()
        assert "D#" in prompt
        assert '["D2"]' in prompt or "D1" in prompt

    @pytest.mark.unit
    def test_definition_example_uses_undermining(self):
        """Example shows a definition attack question under undermining."""
        prompt = _stage2()
        assert "over_extension" in prompt
        assert '"undermining"' in prompt

    @pytest.mark.unit
    def test_target_ids_includes_d(self):
        """target_ids field description includes D#."""
        prompt = _stage2()
        idx = prompt.index("**target_ids**")
        snippet = prompt[idx:idx+200]
        assert "D#" in snippet

    @pytest.mark.unit
    def test_attack_type_three_types(self):
        """attack_type field lists 3 types (no definitional)."""
        prompt = _stage2()
        idx = prompt.index("**attack_type**")
        snippet = prompt[idx:idx+200]
        assert "undermining" in snippet
        assert "undercutting" in snippet

    @pytest.mark.unit
    def test_output_format_attack_type_three_types(self):
        """Output format JSON template lists 3 attack types."""
        prompt = _stage2()
        assert "undermining | rebutting | undercutting" in prompt

    @pytest.mark.unit
    def test_cascading_effect_mentioned(self):
        """Definition targeting section explains cascading A#/E# impact."""
        prompt = _stage2()
        assert "A#/E#" in prompt


# ========================================
# 2. Stage 3: D# defense and patch ops
# ========================================

class TestStage3Definitional:
    """Stage 3 rebuttal includes D# defense guidance and D# patch ops."""

    @pytest.mark.unit
    def test_definitional_defense_guidance(self):
        """D# defense guidance is present in instructions."""
        prompt = _stage3()
        assert "definitions (D# nodes)" in prompt or "D# nodes" in prompt

    @pytest.mark.unit
    def test_defense_refute_guidance(self):
        """Refute guidance for D# challenges is present."""
        prompt = _stage3()
        assert "precise" in prompt or "non-circular" in prompt

    @pytest.mark.unit
    def test_defense_concede_guidance(self):
        """Concede guidance mentions update_definition patch."""
        prompt = _stage3()
        assert "update_definition" in prompt

    @pytest.mark.unit
    def test_defense_defer_guidance(self):
        """Defer guidance mentions U# targeting D#."""
        prompt = _stage3()
        # Check that defer for D# mentions adding uncertainty
        assert "U#" in prompt

    @pytest.mark.unit
    def test_add_definition_in_patches(self):
        """add_definition operation appears in patch examples."""
        prompt = _stage3()
        assert "add_definition" in prompt

    @pytest.mark.unit
    def test_update_definition_in_patches(self):
        """update_definition operation appears in patch examples."""
        prompt = _stage3()
        assert "update_definition" in prompt

    @pytest.mark.unit
    def test_attack_strategy_on_add_counterposition(self):
        """add_counterposition example includes attack_strategy."""
        prompt = _stage3()
        idx = prompt.index("add_counterposition")
        snippet = prompt[idx:idx+300]
        assert "attack_strategy" in snippet

    @pytest.mark.unit
    def test_supported_by_definitions_on_add_evidence(self):
        """add_evidence example includes supported_by_definitions."""
        prompt = _stage3()
        idx = prompt.index("add_evidence")
        snippet = prompt[idx:idx+400]
        assert "supported_by_definitions" in snippet

    @pytest.mark.unit
    def test_attack_strategy_requirement_stated(self):
        """Instructions require attack_strategy alongside attack_type."""
        prompt = _stage3()
        assert "attack_strategy" in prompt
        assert "attack_type" in prompt


# ========================================
# 3. Stage 4 Per-Pair: Definitional evaluation
# ========================================

class TestAdjudicatorPerPairDefinitional:
    """Per-pair adjudicator includes definition-targeting evaluation criteria."""

    @pytest.mark.unit
    def test_definition_targeting_evaluation_criteria(self):
        """Definition-targeting challenge evaluation criteria present."""
        prompt = _per_pair()
        assert "definition-targeting" in prompt.lower() or "definition challenge" in prompt.lower()

    @pytest.mark.unit
    def test_restate_step_mentions_d(self):
        """RESTATE step mentions D# node identification."""
        prompt = _per_pair()
        assert "D#" in prompt

    @pytest.mark.unit
    def test_formalize_step_present(self):
        """FORMALIZE step for definitional challenges present."""
        prompt = _per_pair()
        assert "FORMALIZE" in prompt

    @pytest.mark.unit
    def test_adjudicate_mentions_downstream(self):
        """ADJUDICATE step mentions downstream A#/E# impact."""
        prompt = _per_pair()
        assert "A#/E#" in prompt

    @pytest.mark.unit
    def test_cascading_effect(self):
        """Criteria mention cascading effect on dependent nodes."""
        prompt = _per_pair()
        assert "cascading" in prompt

    @pytest.mark.unit
    def test_verdict_format_preserved(self):
        """Standard verdict format referenced."""
        prompt = _per_pair()
        assert "REBUTTAL_VALID" in prompt or "rebuttal_valid" in prompt

    @pytest.mark.unit
    def test_definitional_flaw_types(self):
        """Evaluation mentions specific flaw types."""
        prompt = _per_pair()
        assert "circularity" in prompt or "over-extension" in prompt


# ========================================
# 4. Stage 4 System: Definition-targeting recognition
# ========================================

class TestAdjudicatorSystemDefinitional:
    """System-level adjudicator recognizes definition-targeting challenges."""

    @pytest.mark.unit
    def test_definition_targeting_mentioned(self):
        """Definition-targeting challenges mentioned in system prompt."""
        prompt = _system_adjudicator()
        assert "definition-targeting" in prompt.lower() or "definition" in prompt.lower()

    @pytest.mark.unit
    def test_d_nodes_in_formalization(self):
        """D# mentioned in the FORMALIZE step."""
        prompt = _system_adjudicator()
        assert "D#" in prompt

    @pytest.mark.unit
    def test_downstream_impact(self):
        """Downstream A#/E# impact mentioned."""
        prompt = _system_adjudicator()
        assert "A#/E#" in prompt


# ========================================
# 5. validate_stage2_questions: definition-targeting strategies
# ========================================

class TestValidateStage2QuestionsDefinitional:
    """validate_stage2_questions accepts definition-targeting strategies under their new types."""

    @pytest.mark.unit
    def test_valid_definition_question_undercutting(self):
        """Definition attack with circularity under undercutting passes."""
        questions = [{
            "qid": "Q1",
            "text": "Your D1 is circular",
            "target_ids": ["D1"],
            "attack_type": "undercutting",
            "attack_strategy": "circularity",
        }]
        is_valid, errors = validate_stage2_questions(questions)
        assert is_valid, f"Should be valid, got errors: {errors}"

    @pytest.mark.unit
    def test_all_definition_strategies_valid_under_new_types(self):
        """All 5 definition-targeting strategies are accepted under their redistributed types."""
        strategy_to_type = {
            "over_extension": "undermining",
            "under_extension": "undermining",
            "circularity": "undercutting",
            "stipulative_bias": "undercutting",
            "conceptual_conflation": "undercutting",
        }
        for strategy, attack_type in strategy_to_type.items():
            questions = [{
                "qid": "Q1",
                "text": f"Testing {strategy}",
                "target_ids": ["D1"],
                "attack_type": attack_type,
                "attack_strategy": strategy,
            }]
            is_valid, errors = validate_stage2_questions(questions)
            assert is_valid, f"Strategy '{strategy}' under '{attack_type}' should be valid: {errors}"

    @pytest.mark.unit
    def test_definitional_attack_type_rejected(self):
        """'definitional' is no longer a valid attack_type."""
        questions = [{
            "qid": "Q1",
            "text": "Bad type",
            "target_ids": ["D1"],
            "attack_type": "definitional",
            "attack_strategy": "circularity",
        }]
        is_valid, errors = validate_stage2_questions(questions)
        assert not is_valid
        assert any("attack_type" in e for e in errors)

    @pytest.mark.unit
    def test_d_target_id_accepted(self):
        """D# target IDs are accepted with undermining type."""
        questions = [{
            "qid": "Q1",
            "text": "Test",
            "target_ids": ["D2"],
            "attack_type": "undermining",
            "attack_strategy": "over_extension",
        }]
        is_valid, errors = validate_stage2_questions(questions)
        assert is_valid, f"D2 target should be valid: {errors}"

    @pytest.mark.unit
    def test_mixed_definition_and_other(self):
        """Mix of definition-targeting and other attack types all valid."""
        questions = [
            {
                "qid": "Q1",
                "text": "Definition circularity",
                "target_ids": ["D1"],
                "attack_type": "undercutting",
                "attack_strategy": "circularity",
            },
            {
                "qid": "Q2",
                "text": "Undermining",
                "target_ids": ["A1"],
                "attack_type": "undermining",
                "attack_strategy": "challenge_assumption",
            },
        ]
        is_valid, errors = validate_stage2_questions(questions)
        assert is_valid, f"Mixed questions should be valid: {errors}"
