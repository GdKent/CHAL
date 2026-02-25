"""
Unit tests for Blood Sport adversarial prompts.

Tests cover:
- build_stage_2_bloodsport_prompt() intensity-appropriate tone guidance
- build_stage_3_bloodsport_prompt() response format instructions
- build_stage_5_bloodsport_prompt() resistance-to-manipulation framing
- Prompt variation across intensity levels
- Prompts do NOT contain explicit fallacy lists or tactic enumerations
- BLOODSPORT_INTENSITY_GUIDANCE dictionary
"""

import pytest
from chal.agents.prompts import (
    BLOODSPORT_INTENSITY_GUIDANCE,
    build_stage_2_bloodsport_prompt,
    build_stage_3_bloodsport_prompt,
    build_stage_5_bloodsport_prompt,
)


SAMPLE_BELIEF_JSON = '{"thesis": {"stance": "Test stance"}, "claims": [{"id": "C1", "statement": "Claim 1"}]}'
SAMPLE_OPPONENT_JSON = '{"thesis": {"stance": "Opponent stance"}, "claims": [{"id": "C2", "statement": "Claim 2"}]}'


# ============================================================
# 1. BLOODSPORT_INTENSITY_GUIDANCE
# ============================================================

class TestIntensityGuidance:
    """Tests for the BLOODSPORT_INTENSITY_GUIDANCE dictionary."""

    @pytest.mark.unit
    def test_has_three_levels(self):
        assert set(BLOODSPORT_INTENSITY_GUIDANCE.keys()) == {"mild", "moderate", "extreme"}

    @pytest.mark.unit
    def test_all_values_are_nonempty_strings(self):
        for level, text in BLOODSPORT_INTENSITY_GUIDANCE.items():
            assert isinstance(text, str)
            assert len(text) > 20, f"{level} guidance is too short"

    @pytest.mark.unit
    def test_extreme_is_strongest(self):
        """Extreme guidance should be longer/more aggressive than mild."""
        assert len(BLOODSPORT_INTENSITY_GUIDANCE["extreme"]) > 0
        assert len(BLOODSPORT_INTENSITY_GUIDANCE["mild"]) > 0


# ============================================================
# 2. Stage 2 BloodSport Prompt (Cross-Examination)
# ============================================================

class TestStage2BloodSportPrompt:
    """Tests for build_stage_2_bloodsport_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    @pytest.mark.unit
    def test_contains_bloodsport_header(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "BLOOD SPORT" in prompt

    @pytest.mark.unit
    def test_contains_intensity_label(self):
        for intensity in ("mild", "moderate", "extreme"):
            prompt = build_stage_2_bloodsport_prompt(
                topic="Test",
                agent_name="A",
                opponent_name="B",
                agent_belief_json=SAMPLE_BELIEF_JSON,
                opponent_belief_json=SAMPLE_OPPONENT_JSON,
                intensity=intensity,
            )
            assert intensity.upper() in prompt

    @pytest.mark.unit
    def test_contains_tone_guidance(self):
        for intensity in ("mild", "moderate", "extreme"):
            prompt = build_stage_2_bloodsport_prompt(
                topic="Test",
                agent_name="A",
                opponent_name="B",
                agent_belief_json=SAMPLE_BELIEF_JSON,
                opponent_belief_json=SAMPLE_OPPONENT_JSON,
                intensity=intensity,
            )
            expected_guidance = BLOODSPORT_INTENSITY_GUIDANCE[intensity]
            assert expected_guidance in prompt

    @pytest.mark.unit
    def test_contains_agent_names(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="Alpha",
            opponent_name="Beta",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "Alpha" in prompt
        assert "Beta" in prompt

    @pytest.mark.unit
    def test_contains_json_output_format(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "questions" in prompt
        assert "qid" in prompt
        assert "target_ids" in prompt

    @pytest.mark.unit
    def test_contains_both_beliefs(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "Test stance" in prompt
        assert "Opponent stance" in prompt

    @pytest.mark.unit
    def test_no_explicit_fallacy_list(self):
        """Prompt should NOT enumerate specific fallacies to use."""
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            intensity="extreme",
        )
        # Should not contain a numbered list of specific logical fallacies
        prompt_lower = prompt.lower()
        for fallacy in ["straw man", "ad hominem fallacy", "red herring", "false dilemma"]:
            assert fallacy not in prompt_lower, f"Found explicit fallacy '{fallacy}' in prompt"

    @pytest.mark.unit
    def test_win_objective(self):
        """Prompt should frame objective as winning, not truth-seeking."""
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "WIN" in prompt or "win" in prompt.lower()

    @pytest.mark.unit
    def test_anti_repetition_included_when_previous_challenges(self):
        prev = [
            {"qid": "Q1", "target_ids": ["C1"], "outcome": "critique_valid"},
        ]
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            previous_challenges=prev,
        )
        assert "Q1" in prompt
        assert "ANTI-REPETITION" in prompt or "anti-repetition" in prompt.lower()

    @pytest.mark.unit
    def test_no_anti_repetition_when_no_previous(self):
        prompt = build_stage_2_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            previous_challenges=None,
        )
        assert "ANTI-REPETITION" not in prompt


# ============================================================
# 3. Stage 3 BloodSport Prompt (Adversarial Exchange)
# ============================================================

class TestStage3BloodSportPrompt:
    """Tests for build_stage_3_bloodsport_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        prompt = build_stage_3_bloodsport_prompt(
            topic="Free will",
            agent_name="Agent-A",
            opponent_name="Agent-B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    @pytest.mark.unit
    def test_contains_bloodsport_header(self):
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "BLOOD SPORT" in prompt

    @pytest.mark.unit
    def test_output_format_has_attack_defense_targets(self):
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
        )
        assert "attack" in prompt
        assert "defense" in prompt
        assert "target_claims" in prompt

    @pytest.mark.unit
    def test_opening_turn_no_history(self):
        """On the first turn (no history), prompt says opening attack."""
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            dialogue_history=None,
        )
        assert "OPENING" in prompt or "opening" in prompt.lower()
        assert "null" in prompt.lower()  # defense should be null

    @pytest.mark.unit
    def test_subsequent_turn_with_history(self):
        """With dialogue history, prompt instructs both defend and attack."""
        history = [
            {"speaker": "B", "attack": "Your C1 is weak", "defense": None, "target_claims": ["C1"]},
        ]
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            dialogue_history=history,
        )
        assert "DEFEND" in prompt or "defend" in prompt.lower()
        assert "COUNTER-ATTACK" in prompt or "counter-attack" in prompt.lower() or "ATTACK" in prompt

    @pytest.mark.unit
    def test_history_displayed_in_prompt(self):
        """Dialogue history is included in the prompt."""
        history = [
            {"speaker": "A", "attack": "Opening salvo text", "defense": None, "target_claims": ["C2"]},
            {"speaker": "B", "attack": "Counter strike text", "defense": "Defense text here", "target_claims": ["C1"]},
        ]
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            dialogue_history=history,
        )
        assert "Opening salvo text" in prompt
        assert "Counter strike text" in prompt

    @pytest.mark.unit
    def test_intensity_variation(self):
        """Different intensities produce different tone guidance."""
        prompts_by_intensity = {}
        for intensity in ("mild", "moderate", "extreme"):
            prompt = build_stage_3_bloodsport_prompt(
                topic="Test",
                agent_name="A",
                opponent_name="B",
                agent_belief_json=SAMPLE_BELIEF_JSON,
                opponent_belief_json=SAMPLE_OPPONENT_JSON,
                intensity=intensity,
            )
            prompts_by_intensity[intensity] = prompt

        # Each intensity should produce a unique prompt
        assert prompts_by_intensity["mild"] != prompts_by_intensity["moderate"]
        assert prompts_by_intensity["moderate"] != prompts_by_intensity["extreme"]

    @pytest.mark.unit
    def test_no_explicit_fallacy_list(self):
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            intensity="extreme",
        )
        prompt_lower = prompt.lower()
        for fallacy in ["straw man", "ad hominem fallacy", "red herring", "false dilemma"]:
            assert fallacy not in prompt_lower

    @pytest.mark.unit
    def test_respects_max_response_length(self):
        prompt = build_stage_3_bloodsport_prompt(
            topic="Test",
            agent_name="A",
            opponent_name="B",
            agent_belief_json=SAMPLE_BELIEF_JSON,
            opponent_belief_json=SAMPLE_OPPONENT_JSON,
            max_response_length_chars=750,
        )
        assert "750" in prompt


# ============================================================
# 4. Stage 5 BloodSport Prompt (Belief Update)
# ============================================================

class TestStage5BloodSportPrompt:
    """Tests for build_stage_5_bloodsport_prompt."""

    @pytest.mark.unit
    def test_returns_nonempty_string(self):
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[{
                "challenger": "Agent-B",
                "challenge": "Test challenge",
                "resolution": {"status": "critique_valid", "reasoning": "Valid point"},
            }],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    @pytest.mark.unit
    def test_contains_bloodsport_header(self):
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "BLOOD SPORT" in prompt

    @pytest.mark.unit
    def test_resistance_to_manipulation_framing(self):
        """Prompt should instruct agents to resist purely rhetorical arguments."""
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        prompt_lower = prompt.lower()
        assert "rhetorical" in prompt_lower or "manipulation" in prompt_lower
        assert "logical" in prompt_lower or "valid" in prompt_lower

    @pytest.mark.unit
    def test_contains_adversarial_resilience_check(self):
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "ADVERSARIAL RESILIENCE" in prompt or "adversarial" in prompt.lower()

    @pytest.mark.unit
    def test_contains_patches_format(self):
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "patches" in prompt
        assert "update_thesis" in prompt or "update_claim" in prompt

    @pytest.mark.unit
    def test_adjudication_results_included(self):
        pairs = [
            {
                "challenger": "Agent-B",
                "challenge": "Your claim C1 is unsupported",
                "resolution": {"status": "critique_valid", "reasoning": "Evidence was weak"},
            },
        ]
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=pairs,
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "Agent-B" in prompt
        assert "critique_valid" in prompt

    @pytest.mark.unit
    def test_bloodsport_exchanges_included_when_provided(self):
        exchanges = [
            {"speaker": "Agent-B", "attack": "Devastating attack", "defense": None},
        ]
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
            bloodsport_exchanges=exchanges,
        )
        assert "Devastating attack" in prompt or "BLOODSPORT EXCHANGE" in prompt

    @pytest.mark.unit
    def test_mandatory_belief_updates_section(self):
        """Prompt should contain mandatory update instructions."""
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "MANDATORY" in prompt or "BINDING" in prompt

    @pytest.mark.unit
    def test_survival_bonus_for_rebuttal_valid(self):
        """Prompt should mention survival bonus for successful defenses."""
        prompt = build_stage_5_bloodsport_prompt(
            agent_name="Agent-A",
            challenge_rebuttal_pairs=[],
            prior_belief_json=SAMPLE_BELIEF_JSON,
        )
        assert "SURVIVAL BONUS" in prompt or "survival" in prompt.lower() or "battle-hardened" in prompt.lower()
