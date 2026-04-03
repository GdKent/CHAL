"""
Unit tests for agent prompts and prompt builders.

Tests cover:
- Persona prompts validation
- Universal debate rules
- Adjudicator prompt builder
- CBS schema terminology (strength, not confidence; no P#/N#)
- Stage-specific prompt content verification
"""

import re
import pytest
from chal.agents.prompts import (
    EMPIRICIST,
    RATIONALIST,
    SKEPTIC,
    PRAGMATIST,
    SUPERNATURALIST,
    PHENOMENOLOGIST,
    CONSTRUCTIVIST,
    NIHILIST,
    BAYESIAN,
    PANPSYCHIST,
    SIMULATIONIST,
    SYNTHESIST,
    build_adjudicator_prompt,
    build_debate_context,
    compute_position_analysis,
    build_universal_prompt,
    build_stage_1_belief_prompt_cbs,
    build_stage_2_prompt,
    build_stage_3_structured_rebuttal_prompt,
    build_stage_5_belief_update_prompt_cbs,
    build_stage_5_phase1_enforcement_prompt,
    build_stage_5_phase2_introspection_prompt,
    build_stage_6_conclusion_prompt,
    build_stage_2_bloodsport_prompt,
    build_stage_3_bloodsport_prompt,
    build_stage_5_bloodsport_prompt,
    build_adjudicator_per_pair_prompt,
)


# ==============================================
# 1. Persona Prompts Tests
# ==============================================

@pytest.mark.unit
def test_all_personas_defined():
    """Test that all 12 personas are defined as non-empty strings."""
    personas = {
        "EMPIRICIST": EMPIRICIST,
        "RATIONALIST": RATIONALIST,
        "SKEPTIC": SKEPTIC,
        "PRAGMATIST": PRAGMATIST,
        "SUPERNATURALIST": SUPERNATURALIST,
        "PHENOMENOLOGIST": PHENOMENOLOGIST,
        "CONSTRUCTIVIST": CONSTRUCTIVIST,
        "NIHILIST": NIHILIST,
        "BAYESIAN": BAYESIAN,
        "PANPSYCHIST": PANPSYCHIST,
        "SIMULATIONIST": SIMULATIONIST,
        "SYNTHESIST": SYNTHESIST
    }

    for persona_name, persona_prompt in personas.items():
        assert isinstance(persona_prompt, str)
        assert len(persona_prompt) > 0


@pytest.mark.unit
def test_persona_uniqueness():
    """Test that each persona has distinct content."""
    persona_values = [
        EMPIRICIST, RATIONALIST, SKEPTIC, PRAGMATIST,
        SUPERNATURALIST, PHENOMENOLOGIST, CONSTRUCTIVIST,
        NIHILIST, BAYESIAN, PANPSYCHIST, SIMULATIONIST, SYNTHESIST
    ]

    # Check that all persona prompts are unique
    assert len(persona_values) == len(set(persona_values))


@pytest.mark.unit
def test_persona_length():
    """Test that personas are non-empty and reasonable length."""
    personas = {
        "EMPIRICIST": EMPIRICIST,
        "RATIONALIST": RATIONALIST,
        "SKEPTIC": SKEPTIC,
        "PRAGMATIST": PRAGMATIST,
        "SUPERNATURALIST": SUPERNATURALIST,
        "PHENOMENOLOGIST": PHENOMENOLOGIST,
        "CONSTRUCTIVIST": CONSTRUCTIVIST,
        "NIHILIST": NIHILIST,
        "BAYESIAN": BAYESIAN,
        "PANPSYCHIST": PANPSYCHIST,
        "SIMULATIONIST": SIMULATIONIST,
        "SYNTHESIST": SYNTHESIST
    }

    for persona_name, persona_prompt in personas.items():
        assert len(persona_prompt) > 50, f"{persona_name} persona is too short"
        assert len(persona_prompt) < 2000, f"{persona_name} persona is too long"


# ==============================================
# 2. Adjudicator Prompt Builder Tests
# ==============================================

# --- Test fixtures for adjudicator prompt tests ---

_TEST_LOGIC_SYS = {
    "label": "Test Logic System",
    "description": "A test logic system for unit tests.",
    "criteria": {
        "critique_valid": [
            "Logic critique alpha",
            "Logic critique beta",
        ],
        "rebuttal_valid": [
            "Logic rebuttal alpha",
            "Logic rebuttal beta",
        ],
        "unresolved": [
            "Logic unresolved alpha",
        ],
    },
}

_TEST_ETHICS_SYS = {
    "label": "Test Ethics System",
    "description": "A test ethics system for unit tests.",
    "criteria": {
        "critique_valid": [
            "Ethics critique gamma",
            "Ethics critique delta",
        ],
        "rebuttal_valid": [
            "Ethics rebuttal gamma",
        ],
        "unresolved": [
            "Ethics unresolved gamma",
        ],
    },
}

_TEST_NONE_ETHICS = {
    "label": "None (Pure Logic)",
    "description": "No ethical framework applied.",
    "criteria": {
        "critique_valid": [],
        "rebuttal_valid": [],
        "unresolved": [],
    },
}


@pytest.mark.unit
def test_adjudicator_prompt_no_defaults():
    """Calling build_adjudicator_prompt with missing args raises TypeError."""
    with pytest.raises(TypeError):
        build_adjudicator_prompt()


@pytest.mark.unit
def test_adjudicator_prompt_logic_only_mode_instruction():
    """Logic-only prompt contains instruction to disregard ethical appeals."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    assert "Disregard any ethical arguments" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_ethics_only_mode_instruction():
    """Ethics-only prompt contains instruction that ethical merit overrides logical validity."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.0, ethics_weight=1.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "Logical validity is irrelevant" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_balanced_mode_instruction():
    """Balanced prompt contains instruction about logical baseline with ethical tiebreaker."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.5, ethics_weight=0.5,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "Logical soundness is the baseline" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_logic_only_no_ethics_criteria():
    """Logic-only prompt contains no (ethical) prefixed criteria."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    assert "(ethical)" not in prompt
    assert "Logic critique alpha" in prompt
    assert "Ethics critique gamma" not in prompt


@pytest.mark.unit
def test_adjudicator_prompt_ethics_only_no_logic_criteria():
    """Ethics-only prompt contains no (logical) prefixed criteria."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.0, ethics_weight=1.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "(logical)" not in prompt
    assert "Ethics critique gamma" in prompt
    assert "Logic critique alpha" not in prompt


@pytest.mark.unit
def test_adjudicator_prompt_balanced_has_both_prefixes():
    """Balanced prompt contains both (logical) and (ethical) prefixed criteria."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.5, ethics_weight=0.5,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "(logical)" in prompt
    assert "(ethical)" in prompt
    assert "Logic critique alpha" in prompt
    assert "Ethics critique gamma" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_logic_only_scoring():
    """Logic-only scoring section says ethics scores must be 0."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    assert "Set ethics scores to 0.0" in prompt
    assert "combined = logic" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_ethics_only_scoring():
    """Ethics-only scoring section says logic scores must be 0."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.0, ethics_weight=1.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "Set logic scores to 0.0" in prompt
    assert "combined = ethics" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_balanced_scoring():
    """Balanced scoring section references both axes."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.5, ethics_weight=0.5,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "combined = 0.5 * logic + 0.5 * ethics" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_anti_bias_present():
    """Anti-bias section is present in all three modes."""
    for lw, ew in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
        prompt = build_adjudicator_prompt(
            logic_weight=lw, ethics_weight=ew,
            logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
        )
        assert "<anti_bias>" in prompt
        assert "Explicit concession" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_universal_base_present():
    """Universal base flaws are present in all modes."""
    for lw, ew in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
        prompt = build_adjudicator_prompt(
            logic_weight=lw, ethics_weight=ew,
            logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
        )
        assert "circular reasoning" in prompt
        assert "misrepresentation of the opposing position" in prompt
        assert "self-defeating argument" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_criteria_from_system():
    """Criteria in the prompt match the criteria lists from the provided system dicts."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    # All logic criteria should appear
    for item in _TEST_LOGIC_SYS["criteria"]["critique_valid"]:
        assert item in prompt
    for item in _TEST_LOGIC_SYS["criteria"]["rebuttal_valid"]:
        assert item in prompt
    for item in _TEST_LOGIC_SYS["criteria"]["unresolved"]:
        assert item in prompt


@pytest.mark.unit
def test_adjudicator_prompt_logic_only_label_in_preamble():
    """Logic-only prompt includes the logic system label in the criteria preamble."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    assert "under Test Logic System:" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_ethics_only_label_in_preamble():
    """Ethics-only prompt includes the ethics system label in the criteria preamble."""
    prompt = build_adjudicator_prompt(
        logic_weight=0.0, ethics_weight=1.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    assert "under Test Ethics System:" in prompt


# ==============================================
# Stage Prompt Tests
# ==============================================

@pytest.mark.unit
def test_build_universal_prompt():
    """Test building universal system prompt."""
    from chal.agents.prompts import build_universal_prompt

    prompt = build_universal_prompt(topic="AI Ethics")

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "AI Ethics" in prompt


@pytest.mark.unit
def test_build_position_prompt():
    """Test building position-specific prompt."""
    from chal.agents.prompts import build_position_prompt, SKEPTIC

    agent_name = "Agent-Skeptic"
    persona = SKEPTIC

    prompt = build_position_prompt(agent_name, persona)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "skeptic" in prompt.lower()


@pytest.mark.unit
def test_build_stage_1_belief_prompt():
    """Test building Stage 1 initial belief prompt."""
    from chal.agents.prompts import build_stage_1_belief_prompt_cbs

    prompt = build_stage_1_belief_prompt_cbs(
        topic="Climate Change",
        agent_name="Agent-Scientist",
        persona_label="Scientist"
    )

    assert isinstance(prompt, str)
    assert "Climate Change" in prompt
    assert "CBS" in prompt or "belief" in prompt.lower()


@pytest.mark.unit
def test_build_stage_2_prompt():
    """Test building Stage 2 question generation prompt."""
    from chal.agents.prompts import build_stage_2_prompt

    agent_belief_json = '{"thesis": {"statement": "Test thesis"}}'
    opponent_belief_json = '{"thesis": {"statement": "Counter thesis"}}'

    prompt = build_stage_2_prompt(
        topic="Philosophy",
        agent_name="Agent-A",
        opponent_name="Agent-B",
        agent_belief_json=agent_belief_json,
        opponent_belief_json=opponent_belief_json
    )

    assert isinstance(prompt, str)
    assert "Agent-B" in prompt
    assert len(prompt) > 100


@pytest.mark.unit
def test_build_stage_2_prompt_with_previous_challenges():
    """Test Stage 2 prompt with previous challenges."""
    from chal.agents.prompts import build_stage_2_prompt

    previous_challenges = [
        {"qid": "Q1", "target_ids": ["C1"], "outcome": "critique_valid"},
        {"qid": "Q2", "target_ids": ["A1"], "outcome": "rebuttal_valid"}
    ]

    prompt = build_stage_2_prompt(
        topic="Ethics",
        agent_name="Agent-A",
        opponent_name="Agent-B",
        agent_belief_json='{"thesis": {"statement": "Test"}}',
        opponent_belief_json='{"thesis": {"statement": "Test2"}}',
        previous_challenges=previous_challenges,
    )

    assert isinstance(prompt, str)
    assert "Q1" in prompt or "Q2" in prompt


@pytest.mark.unit
def test_build_stage_3_structured_rebuttal_prompt():
    """Test building Stage 3 rebuttal prompt."""
    from chal.agents.prompts import build_stage_3_structured_rebuttal_prompt

    questions_json = '["Question 1?", "Question 2?"]'
    agent_belief_json = '{"thesis": {"statement": "My position"}}'

    prompt = build_stage_3_structured_rebuttal_prompt(
        topic="Science",
        agent_name="Agent-Defender",
        opponent_name="Agent-Challenger",
        received_questions_json=questions_json,
        agent_belief_json=agent_belief_json
    )

    assert isinstance(prompt, str)
    assert "Question 1?" in prompt or "rebuttal" in prompt.lower()


@pytest.mark.unit
def test_build_stage_5_belief_update_prompt():
    """Test building Stage 5 belief update prompt."""
    from chal.agents.prompts import build_stage_5_belief_update_prompt_cbs

    challenge_rebuttal_pairs = [{
        "challenger": "Agent-B",
        "challenge": "You are wrong",
        "qid": "Q1",
        "target_ids": ["C1"],
        "attack_type": "undermining",
        "attack_strategy": "challenge_evidence",
        "resolution": {
            "status": "critique_valid",
            "reasoning": "The critique was valid"
        }
    }]

    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=challenge_rebuttal_pairs,
        prior_belief_json='{"thesis": {"statement": "Original"}}'
    )

    assert isinstance(prompt, str)
    assert "Agent-B" in prompt or "critique" in prompt.lower()


@pytest.mark.unit
def test_build_stage_6_conclusion_prompt():
    """Test building Stage 6 conclusion prompt."""
    from chal.agents.prompts import build_stage_6_conclusion_prompt

    belief_changelog_summary = "v1: Initial belief formation\nv2: Lowered C1 confidence to 0.7"

    prompt = build_stage_6_conclusion_prompt(
        topic="Future of AI",
        agent_name="Agent-Futurist",
        agent_belief_json='{"thesis": {"statement": "Final position"}}',
        belief_changelog_summary=belief_changelog_summary,
        num_rounds=2,
        persona_label="EMPIRICIST"
    )

    assert isinstance(prompt, str)
    assert "Future of AI" in prompt
    assert len(prompt) > 100


@pytest.mark.unit
def test_build_stage_7_scribe_prompt_map():
    """Test building Stage 7 map phase scribe prompt."""
    from chal.agents.prompts import build_stage_7_scribe_prompt_map

    prompt = build_stage_7_scribe_prompt_map(
        topic="Technology",
        agent_names=["Agent-A", "Agent-B"],
        transcript_chunk="Agent-A said X. Agent-B said Y.",
        continuity_state_json='{"themes": []}'
    )

    assert isinstance(prompt, str)
    assert "Technology" in prompt or "Agent-A" in prompt


@pytest.mark.unit
def test_build_stage_7_scribe_prompt_reduce():
    """Test building Stage 7 reduce phase scribe prompt."""
    from chal.agents.prompts import build_stage_7_scribe_prompt_reduce

    narrative_slices = ["## Section 1\nContent 1", "## Section 2\nContent 2"]

    prompt = build_stage_7_scribe_prompt_reduce(
        topic="Philosophy",
        agent_names=["Agent-A", "Agent-B"],
        all_narrative_slices_markdown=narrative_slices,
        final_continuity_state_json='{"themes": ["theme1"]}'
    )

    assert isinstance(prompt, str)
    assert "Philosophy" in prompt or "Section 1" in prompt


# ==============================================
# CBS Terminology Tests
# ==============================================

@pytest.mark.unit
def test_universal_prompt_no_pn_ids():
    """Universal prompt should not reference P# or N# IDs."""
    prompt = build_universal_prompt(topic="Test")
    # Match standalone P# or N# references (not inside words)
    assert not re.search(r'\bP#\b', prompt), "Universal prompt still references P#"
    assert not re.search(r'\bN#\b', prompt), "Universal prompt still references N#"


@pytest.mark.unit
def test_universal_prompt_uses_strength():
    """Universal prompt should use 'strength' not 'confidence' for calibration."""
    prompt = build_universal_prompt(topic="Test")
    assert "strength" in prompt.lower()


@pytest.mark.unit
def test_stage_1_no_pn_ids():
    """Stage 1 prompt should not reference P# or N# IDs."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert not re.search(r'\bP#\b', prompt), "Stage 1 prompt still references P#"
    assert not re.search(r'\bN#\b', prompt), "Stage 1 prompt still references N#"


@pytest.mark.unit
def test_stage_1_uses_strength_terminology():
    """Stage 1 prompt should use 'strength' not 'confidence' for numeric scores."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    # The schema description should use "strength" for the thesis field
    assert '"strength"' in prompt
    # Should not contain confidence as a field name (natural language uses are fine)
    assert '"confidence"' not in prompt


@pytest.mark.unit
def test_stage_1_includes_strength_scale():
    """Stage 1 prompt should include the strength scale calibration table."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "strength_scale" in prompt or "Vacuous" in prompt
    assert "Definitive" in prompt


@pytest.mark.unit
def test_stage_1_includes_thesis_strength_formula():
    """Stage 1 prompt should include the thesis strength formula."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "thesis_strength" in prompt or "avg(active_claim_strengths)" in prompt


@pytest.mark.unit
def test_stage_1_predictions_in_claims():
    """Stage 1 prompt should describe predictions as claim sub-fields, not standalone."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    # Should mention predictions in the claims description
    assert "predictions" in prompt
    # The example JSON should have predictions inside claims, not as top-level
    assert '"predictions"' in prompt


@pytest.mark.unit
def test_stage_1_no_known_weaknesses():
    """Stage 1 prompt should not reference known_weaknesses."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "known_weaknesses" not in prompt


@pytest.mark.unit
def test_stage_1_counterpositions_no_strength():
    """Stage 1 X# description should not include a strength field."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    # Example JSON should not have "strength" inside counterpositions
    # Check that X# objects in the example don't have strength
    # Find the counterpositions section in the example
    x_section = prompt[prompt.find('"counterpositions"'):]
    x_section = x_section[:x_section.find('"uncertainties"')]
    assert '"strength"' not in x_section


@pytest.mark.unit
def test_stage_1_uncertainties_have_targets():
    """Stage 1 U# description should include targets field."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "targets" in prompt


@pytest.mark.unit
def test_stage_2_no_pn_ids():
    """Stage 2 prompt should not reference P# or N# IDs."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert not re.search(r'\bP#\b', prompt), "Stage 2 prompt still references P#"
    assert not re.search(r'\bN#\b', prompt), "Stage 2 prompt still references N#"


@pytest.mark.unit
def test_stage_2_uses_strength():
    """Stage 2 questioning strategies should reference 'strength' not 'confidence'."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "strength" in prompt.lower()


@pytest.mark.unit
def test_stage_2_internal_inconsistency_attack():
    """Stage 2 prompt should include internal inconsistency as an attack strategy."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "internal inconsistenc" in prompt.lower()


@pytest.mark.unit
def test_stage_3_uses_strength():
    """Stage 3 rebuttal prompt should use 'strength' not 'confidence'."""
    prompt = build_stage_3_structured_rebuttal_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        received_questions_json='[]', agent_belief_json='{}'
    )
    # Patch examples should use strength
    assert "strength" in prompt


# Helper to build a standard Stage 3 prompt for the tests below
def _build_stage_3():
    return build_stage_3_structured_rebuttal_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        received_questions_json='[]', agent_belief_json='{}'
    )


@pytest.mark.unit
def test_stage_3_no_update_thesis():
    """Prompt does NOT contain update_thesis."""
    prompt = _build_stage_3()
    assert "update_thesis" not in prompt


@pytest.mark.unit
def test_stage_3_has_all_patch_operations():
    """Prompt contains all 11 non-thesis patch operations."""
    prompt = _build_stage_3()
    expected_ops = [
        "update_claim", "retire_claim", "add_claim",
        "add_evidence", "update_evidence",
        "update_assumption", "add_assumption",
        "add_counterposition", "update_counterposition",
        "add_uncertainty", "resolve_uncertainty",
    ]
    for op in expected_ops:
        assert op in prompt, f"Missing operation: {op}"


@pytest.mark.unit
def test_stage_3_no_e_new_ids():
    """Prompt does NOT contain E_NEW or X_NEW — only numbered IDs."""
    prompt = _build_stage_3()
    assert "E_NEW" not in prompt
    assert "X_NEW" not in prompt


@pytest.mark.unit
def test_stage_3_add_evidence_has_required_fields():
    """The add_evidence example includes all expected fields."""
    prompt = _build_stage_3()
    # Extract the section around add_evidence
    assert '"id": "E4"' in prompt
    assert '"type": "empirical"' in prompt
    assert '"summary"' in prompt
    assert '"source"' in prompt
    assert '"relevance_to_claims"' in prompt
    assert '"strength_justification"' in prompt


@pytest.mark.unit
def test_stage_3_add_claim_has_required_fields():
    """The add_claim example includes all required fields per validate_patches."""
    prompt = _build_stage_3()
    # All required fields for add_claim
    for field in ["id", "type", "statement", "depends_on", "strength",
                  "status", "predictions", "inference_chain"]:
        assert f'"{field}"' in prompt, f"add_claim example missing field: {field}"
    # Prediction sub-fields
    for field in ["decision_criterion"]:
        assert f'"{field}"' in prompt, f"add_claim prediction missing field: {field}"


@pytest.mark.unit
def test_stage_3_add_assumption_has_required_fields():
    """The add_assumption example includes id, type, statement, strength."""
    prompt = _build_stage_3()
    assert '"id": "A3"' in prompt
    assert '"type": "empirical"' in prompt
    assert '"statement"' in prompt
    assert '"strength": 0.75' in prompt


@pytest.mark.unit
def test_stage_3_add_counterposition_has_required_fields():
    """The add_counterposition example includes all required fields."""
    prompt = _build_stage_3()
    assert '"id": "X3"' in prompt
    assert '"targets"' in prompt
    assert '"attack_type"' in prompt
    assert '"my_response"' in prompt
    assert '"response_sufficiency"' in prompt


@pytest.mark.unit
def test_stage_3_add_uncertainty_has_required_fields():
    """The add_uncertainty example includes id, targets, question, status."""
    prompt = _build_stage_3()
    assert '"id": "U2"' in prompt
    assert '"targets"' in prompt
    assert '"question"' in prompt
    assert '"status": "active"' in prompt


@pytest.mark.unit
def test_stage_3_resolve_uncertainty_has_required_fields():
    """The resolve_uncertainty example includes target_id and resolution_note."""
    prompt = _build_stage_3()
    assert "resolve_uncertainty" in prompt
    assert '"target_id": "U1"' in prompt
    assert '"resolution_note"' in prompt


@pytest.mark.unit
def test_stage_3_update_claim_has_strength_justification():
    """The update_claim example includes strength_justification in changes."""
    prompt = _build_stage_3()
    # Find the update_claim line and check it has strength_justification
    assert "update_claim" in prompt
    assert "strength_justification" in prompt


@pytest.mark.unit
def test_stage_3_defer_links_to_add_uncertainty():
    """Prompt mentions that defer SHOULD include add_uncertainty."""
    prompt = _build_stage_3()
    assert "defer" in prompt
    assert "add_uncertainty" in prompt
    # Check the specific linkage instruction
    assert "SHOULD" in prompt


@pytest.mark.unit
def test_stage_3_concede_requires_weakening_patch():
    """Prompt states that concede MUST include a weakening patch."""
    prompt = _build_stage_3()
    assert 'concede' in prompt
    assert 'MUST' in prompt
    assert 'weakening patch' in prompt


@pytest.mark.unit
def test_stage_3_id_convention_note():
    """Prompt contains guidance about using next available numbered IDs."""
    prompt = _build_stage_3()
    assert "next available number" in prompt


@pytest.mark.unit
def test_stage_3_single_json_block_format():
    """Prompt specifies one fenced JSON code block containing both rebuttals and patches."""
    prompt = _build_stage_3()
    assert "One fenced JSON code block" in prompt
    assert '"rebuttals"' in prompt
    assert '"patches"' in prompt


@pytest.mark.unit
def test_stage_3_example_patches_pass_validate_patches():
    """All example patch operations from the Stage 3 prompt pass validate_patches against a sample belief."""
    from chal.beliefs.patches import validate_patches
    from tests.utils import create_sample_belief

    # Build a belief that has the IDs referenced in the prompt examples
    belief = create_sample_belief(
        belief_id="B1", confidence=0.75,
        num_claims=4, num_assumptions=3, num_evidence=4
    )
    # Add a counterposition and an uncertainty so update/resolve targets exist
    belief["counterpositions"] = [
        {"id": "X1", "targets": ["C1"], "attack_type": "undercutting",
         "statement": "Test", "my_response": "Test", "response_sufficiency": "partial"}
    ]
    belief["uncertainties"] = [
        {"id": "U1", "targets": ["C1"], "question": "Test?", "status": "active"}
    ]

    # These mirror the exact examples from the rewritten Stage 3 prompt
    example_patches = [
        # Modify existing
        {"op": "update_claim", "target_id": "C1", "changes": {"strength": 0.6, "strength_justification": "Lowered"}},
        {"op": "retire_claim", "target_id": "C3"},
        {"op": "update_evidence", "target_id": "E1", "changes": {"strength": 0.5, "strength_justification": "Downgraded"}},
        {"op": "update_assumption", "target_id": "A2", "changes": {"strength": 0.6, "status": "revised", "strength_justification": "Weakened"}},
        {"op": "update_counterposition", "target_id": "X1", "changes": {"my_response": "Updated", "response_sufficiency": "sufficient"}},
        {"op": "resolve_uncertainty", "target_id": "U1", "resolution_note": "Resolved by new evidence E4"},
        # Add new
        {"op": "add_claim", "item": {"id": "C5", "type": "deductive", "statement": "New claim",
         "depends_on": ["A1", "E2"], "strength": 0.7, "status": "active", "strength_justification": "Test",
         "predictions": [{"statement": "P", "test": "T", "decision_criterion": "DC"}],
         "inference_chain": ["Step 1"]}},
        {"op": "add_evidence", "item": {"id": "E5", "type": "empirical", "summary": "New evidence",
         "source": "Test", "relevance_to_claims": ["C1"], "strength": 0.7, "status": "active",
         "strength_justification": "Test"}},
        {"op": "add_assumption", "item": {"id": "A4", "type": "empirical", "statement": "New assumption",
         "supports_claims": ["C1"], "strength": 0.75, "status": "active", "strength_justification": "Test"}},
        {"op": "add_counterposition", "item": {"id": "X3", "targets": ["C2"], "attack_type": "undermining",
         "statement": "Test", "my_response": "Test", "response_sufficiency": "partial"}},
        {"op": "add_uncertainty", "item": {"id": "U2", "targets": ["C1", "E1"], "question": "Test?", "status": "active", "importance": "high"}},
    ]

    errors = validate_patches(example_patches, belief)
    assert errors == {}, f"Example patches failed validation: {errors}"


@pytest.mark.unit
def test_stage_5_includes_rebuttal_text():
    """Stage 5 prompt should include rebuttal text in outcome formatting."""
    pairs = [{
        "challenger": "Agent-B",
        "challenge": "Your claim is weak",
        "rebuttal": "I defended with evidence E1",
        "resolution": {"status": "critique_valid", "reasoning": "Valid critique"}
    }]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=pairs,
        prior_belief_json='{"thesis": {"stance": "test", "strength": 0.7}}'
    )
    assert "I defended with evidence E1" in prompt
    assert "Your rebuttal:" in prompt


@pytest.mark.unit
def test_stage_5_no_pn_ids():
    """Stage 5 prompt should not reference P# or N# IDs."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert not re.search(r'\bP#\b', prompt), "Stage 5 prompt still references P#"
    assert not re.search(r'\bN#\b', prompt), "Stage 5 prompt still references N#"


@pytest.mark.unit
def test_stage_5_uses_strength():
    """Stage 5 prompt operations should use 'strength' not 'confidence'."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert '"strength"' in prompt
    assert '"confidence"' not in prompt


@pytest.mark.unit
def test_stage_5_includes_strength_scale():
    """Stage 5 prompt should include the strength scale table."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert "Vacuous" in prompt
    assert "Definitive" in prompt


@pytest.mark.unit
def test_stage_5_includes_thesis_ceiling():
    """Stage 5 prompt should include the thesis ceiling formula."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert "thesis_ceiling" in prompt or "avg(active_claim_strengths)" in prompt


@pytest.mark.unit
def test_stage_5_includes_resolve_uncertainty():
    """Stage 5 prompt should include resolve_uncertainty as a supported operation."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert "resolve_uncertainty" in prompt


@pytest.mark.unit
def test_stage_5_includes_update_evidence():
    """Stage 5 prompt should include update_evidence as a supported operation."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert "update_evidence" in prompt


@pytest.mark.unit
def test_stage_5_self_check_includes_thesis_strength():
    """Stage 5 self-check should reference thesis strength formula."""
    pairs = [{"challenger": "B", "challenge": "test", "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    # The self-check section should mention the thesis strength formula
    self_check_section = prompt[prompt.find("Self-check"):]
    assert "thesis strength" in self_check_section or "avg(active claim strengths)" in self_check_section


@pytest.mark.unit
def test_stage_6_uses_strength():
    """Stage 6 conclusion prompt should use 'strength' not 'confidence'."""
    prompt = build_stage_6_conclusion_prompt(
        topic="Test", agent_name="A",
        agent_belief_json='{}',
        belief_changelog_summary="v1: initial"
    )
    assert '"strength"' in prompt
    assert '"confidence"' not in prompt


@pytest.mark.unit
def test_adjudicator_prompt_threshold_in_scoring():
    """Adjudicator prompt scoring section includes the threshold value."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
        threshold=0.2,
    )
    assert ">= 0.2" in prompt


# ==============================================
# Stage 1 Thesis-Last Tests
# ==============================================

@pytest.mark.unit
def test_stage_1_thesis_last_instruction():
    """Stage 1 prompt contains generation_order block instructing bottom-up with thesis LAST."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "generation_order" in prompt
    assert "LAST" in prompt
    # Thesis should be explicitly instructed as last
    assert "Thesis" in prompt[prompt.find("generation_order"):]


@pytest.mark.unit
def test_stage_1_thesis_schema_after_other_components():
    """Thesis schema description appears after assumptions/claims/evidence/counterpositions/uncertainties."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    # In the STRUCTURED SECTIONS area, thesis should appear after other sections
    sections_start = prompt.find("STRUCTURED SECTIONS")
    if sections_start == -1:
        sections_start = 0
    synthesized_pos = prompt.find("SYNTHESIZED LAST", sections_start)
    assumptions_pos = prompt.find('"assumptions"', sections_start)
    claims_pos = prompt.find('"claims"', sections_start)

    # Thesis in SYNTHESIZED LAST block should come after assumptions and claims definitions
    assert assumptions_pos < synthesized_pos
    assert claims_pos < synthesized_pos


# ==============================================
# Phase 1 Enforcement Prompt Tests
# ==============================================

SAMPLE_PAIRS = [{
    "challenger": "Agent-B",
    "challenge": "Your C1 is weak because E1 is outdated",
    "rebuttal": "E1 was replicated in 2025 with consistent results",
    "resolution": {
        "status": "critique_valid",
        "reasoning": "The replication evidence was insufficient to counter the original critique"
    }
}]

SAMPLE_BELIEF_JSON = '{"thesis": {"stance": "test", "strength": 0.7}, "claims": []}'


@pytest.mark.unit
def test_phase1_prompt_includes_adjudication_outcomes():
    """Phase 1 prompt formats challenge-rebuttal pairs with outcomes."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "Agent-B" in prompt
    assert "critique_valid" in prompt


@pytest.mark.unit
def test_phase1_prompt_includes_rebuttal_text():
    """Rebuttal text appears alongside challenge text and verdict."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "E1 was replicated in 2025" in prompt
    assert "Your rebuttal:" in prompt


@pytest.mark.unit
def test_phase1_prompt_no_thesis_ceiling():
    """Phase 1 should NOT include the thesis ceiling formula (that's Phase 2)."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "thesis_ceiling" not in prompt
    assert "avg(active_claim_strengths)" not in prompt


@pytest.mark.unit
def test_phase1_prompt_scope_restriction():
    """Prompt contains text limiting scope to adjudication response only."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "Do NOT rewrite your thesis" in prompt or "do NOT rewrite your thesis" in prompt


@pytest.mark.unit
def test_phase1_prompt_mandatory_rules():
    """CRITIQUE_VALID, REBUTTAL_VALID, UNRESOLVED rules are present."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "CRITIQUE_VALID" in prompt
    assert "REBUTTAL_VALID" in prompt
    assert "UNRESOLVED" in prompt


@pytest.mark.unit
def test_phase1_prompt_uses_strength():
    """Phase 1 uses CBS strength terminology."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert '"strength"' in prompt
    assert '"confidence"' not in prompt


@pytest.mark.unit
def test_phase1_prompt_no_pn_ids():
    """Phase 1 should not reference P# or N# IDs."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert not re.search(r'\bP#\b', prompt), "Phase 1 prompt still references P#"
    assert not re.search(r'\bN#\b', prompt), "Phase 1 prompt still references N#"


# ==============================================
# Phase 2 Introspection Prompt Tests
# ==============================================

INTERMEDIATE_BELIEF_JSON = '{"thesis": {"stance": "intermediate", "strength": 0.6}, "claims": [{"id": "C1", "strength": 0.5}]}'
PHASE1_SUMMARY = "- Updated C1: strength->0.5\n- Thesis strength: weaken"


@pytest.mark.unit
def test_phase2_prompt_includes_intermediate_belief():
    """Post-Phase-1 belief JSON appears in prompt."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "intermediate" in prompt
    assert "current_belief" in prompt.lower() or "phase 1" in prompt.lower()


@pytest.mark.unit
def test_phase2_prompt_excludes_original_belief():
    """Original belief block should not be present (removed in Change 4)."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "<original_belief>" not in prompt


@pytest.mark.unit
def test_phase2_prompt_includes_phase1_summary():
    """Human-readable summary of Phase 1 changes appears."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "strength->0.5" in prompt or "strength→0.5" in prompt


@pytest.mark.unit
def test_phase2_prompt_counterposition_audit():
    """Prompt contains instructions for counterposition audit (Step 1)."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "COUNTERPOSITION AUDIT" in prompt or "counterposition" in prompt.lower()


@pytest.mark.unit
def test_phase2_prompt_strategic_evaluation():
    """Prompt contains thesis strength formula and strategic evaluation instructions."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "thesis_strength" in prompt or "thesis strength" in prompt
    assert "avg(active_claim_strengths)" in prompt or "avg(claim" in prompt


@pytest.mark.unit
def test_phase2_prompt_thesis_rewrite():
    """Prompt instructs thesis to be generated LAST with stance/bullets/strength."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "THESIS REWRITE" in prompt or "thesis" in prompt.lower()
    assert "stance" in prompt
    assert "summary_bullets" in prompt


@pytest.mark.unit
def test_phase2_prompt_guardrails():
    """Prompt contains the 'cannot reverse Phase 1 changes' guardrail."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "CANNOT reverse Phase 1" in prompt or "cannot reverse" in prompt.lower()


@pytest.mark.unit
def test_phase2_prompt_update_thesis_stance_bullets():
    """Prompt documents expanded update_thesis supporting stance and summary_bullets."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "stance" in prompt
    assert "summary_bullets" in prompt
    assert "update_thesis" in prompt


@pytest.mark.unit
def test_phase2_prompt_uses_strength():
    """Phase 2 uses CBS strength terminology."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert '"strength"' in prompt
    assert '"confidence"' not in prompt


# ==============================================
# 9. Debate Context Tests
# ==============================================

@pytest.mark.unit
def test_build_debate_context_helper():
    """Test that build_debate_context returns correct XML block."""
    result = build_debate_context("Opening — forming your initial belief")
    assert "<debate_context>" in result
    assert "</debate_context>" in result
    assert "Opening — forming your initial belief" in result
    assert "multi-round structured debate" in result
    assert "Cross-examination" in result
    assert "Rebuttal" in result
    assert "Adjudication" in result
    assert "Belief update" in result


@pytest.mark.unit
@pytest.mark.parametrize("builder,kwargs,expected_description", [
    (
        build_stage_1_belief_prompt_cbs,
        {"topic": "Test", "agent_name": "A", "persona_label": "EMPIRICIST"},
        "Opening — forming your initial belief",
    ),
    (
        build_stage_2_prompt,
        {"topic": "Test", "agent_name": "A", "opponent_name": "B",
         "agent_belief_json": "{}", "opponent_belief_json": "{}"},
        "Cross-examination — challenging your opponent",
    ),
    (
        build_stage_3_structured_rebuttal_prompt,
        {"topic": "Test", "agent_name": "A", "opponent_name": "B",
         "received_questions_json": "[]", "agent_belief_json": "{}"},
        "Rebuttal — defending against your opponent's challenges",
    ),
    (
        build_stage_5_belief_update_prompt_cbs,
        {"agent_name": "A", "challenge_rebuttal_pairs": [],
         "prior_belief_json": "{}"},
        "Belief update — incorporating adjudication outcomes",
    ),
    (
        build_stage_5_phase1_enforcement_prompt,
        {"agent_name": "A", "challenge_rebuttal_pairs": [],
         "prior_belief_json": "{}"},
        "Belief update (enforcement) — incorporating adjudication outcomes",
    ),
    (
        build_stage_5_phase2_introspection_prompt,
        {"agent_name": "A", "intermediate_belief_json": "{}",
         "phase1_changes_summary": "none"},
        "Belief update (strategic) — strengthening your position",
    ),
    (
        build_stage_6_conclusion_prompt,
        {"topic": "Test", "agent_name": "A", "agent_belief_json": "{}",
         "belief_changelog_summary": "none"},
        "Conclusion — producing your final synthesis",
    ),
])
def test_debate_context_in_standard_prompts(builder, kwargs, expected_description):
    """Each standard stage prompt contains <debate_context> with the correct stage description."""
    prompt = builder(**kwargs)
    assert "<debate_context>" in prompt
    assert "</debate_context>" in prompt
    assert f"You are currently in: {expected_description}" in prompt


@pytest.mark.unit
@pytest.mark.parametrize("builder,kwargs,expected_description", [
    (
        build_stage_2_bloodsport_prompt,
        {"topic": "Test", "agent_name": "A", "opponent_name": "B",
         "agent_belief_json": "{}", "opponent_belief_json": "{}"},
        "Cross-examination — challenging your opponent",
    ),
    (
        build_stage_3_bloodsport_prompt,
        {"topic": "Test", "agent_name": "A", "opponent_name": "B",
         "agent_belief_json": "{}", "opponent_belief_json": "{}"},
        "Rebuttal — defending against your opponent's challenges",
    ),
    (
        build_stage_5_bloodsport_prompt,
        {"agent_name": "A", "challenge_rebuttal_pairs": [],
         "prior_belief_json": "{}"},
        "Belief update (enforcement) — incorporating adjudication outcomes",
    ),
])
def test_debate_context_in_bloodsport_prompts(builder, kwargs, expected_description):
    """Each bloodsport stage prompt contains <debate_context> with the correct stage description."""
    prompt = builder(**kwargs)
    assert "<debate_context>" in prompt
    assert "</debate_context>" in prompt
    assert f"You are currently in: {expected_description}" in prompt


@pytest.mark.unit
def test_adjudicator_prompt_no_debate_context():
    """Adjudicator prompt should NOT contain <debate_context> — it's not a debate participant."""
    prompt = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_NONE_ETHICS,
    )
    assert "<debate_context>" not in prompt


# ==============================================
# 10. Position Analysis Tests
# ==============================================

# Reusable belief dict for position analysis tests
_PA_BELIEF = {
    "assumptions": [
        {"id": "A1", "type": "empirical", "statement": "test", "strength": 0.85, "status": "active", "strength_justification": "test"},
        {"id": "A2", "type": "foundational", "statement": "test", "strength": 0.40, "status": "active", "strength_justification": "test"},
        {"id": "A3", "type": "methodological", "statement": "test", "strength": 0.70, "status": "active", "strength_justification": "test"},
    ],
    "claims": [
        {"id": "C1", "type": "descriptive", "statement": "test", "depends_on": ["A1", "E1"],
         "strength": 0.75, "status": "active",
         "strength_justification": "test", "inference_chain": ["A1 supports C1"],
         "predictions": [{"statement": "x", "test": "y", "decision_criterion": "z"}]},
        {"id": "C2", "type": "descriptive", "statement": "test", "depends_on": ["A2", "E2"],
         "strength": 0.55, "status": "active",
         "strength_justification": "test", "inference_chain": ["A2 supports C2"],
         "predictions": [{"statement": "x", "test": "y", "decision_criterion": "z"}]},
        {"id": "C3", "type": "descriptive", "statement": "test", "depends_on": ["A3", "E1"],
         "strength": 0.65, "status": "active",
         "strength_justification": "test", "inference_chain": ["A3 supports C3"],
         "predictions": [{"statement": "x", "test": "y", "decision_criterion": "z"}]},
    ],
    "evidence": [
        {"id": "E1", "type": "empirical", "summary": "test", "source": "test",
         "relevance_to_claims": ["C1", "C3"], "strength": 0.30, "status": "active", "strength_justification": "weak"},
        {"id": "E2", "type": "conceptual", "summary": "test", "source": "test",
         "relevance_to_claims": ["C2"], "strength": 0.60, "status": "active", "strength_justification": "moderate"},
    ],
    "thesis": {"stance": "test", "strength": 0.5},
}


@pytest.mark.unit
def test_position_analysis_contains_sections():
    """Position analysis block contains all required sections."""
    result = compute_position_analysis(_PA_BELIEF)
    assert "<position_analysis>" in result
    assert "</position_analysis>" in result
    assert "YOUR CURRENT POSITION" in result
    assert "SENSITIVITY AT YOUR POSITION" in result
    assert "SCENARIO PROJECTIONS" in result
    assert "LOWEST-STRENGTH DEPENDENCIES" in result
    assert "STRATEGIC RECOMMENDATION" in result
    assert "INTEGRITY REMINDER" in result


@pytest.mark.unit
def test_position_analysis_correct_claim_count():
    """Position analysis reports correct number of active claims."""
    result = compute_position_analysis(_PA_BELIEF)
    assert "Active claims: 3" in result


@pytest.mark.unit
def test_position_analysis_correct_average_strength():
    """Position analysis computes correct average claim strength."""
    # avg(0.75, 0.55, 0.65) = 0.65
    result = compute_position_analysis(_PA_BELIEF)
    assert "average claim strength: 0.65" in result


@pytest.mark.unit
def test_position_analysis_partial_derivatives():
    """Position analysis contains partial derivative notation."""
    result = compute_position_analysis(_PA_BELIEF)
    assert "∂T/∂s" in result
    assert "∂T/∂n" in result


@pytest.mark.unit
def test_position_analysis_scenario_projections():
    """Position analysis shows scenario projections."""
    result = compute_position_analysis(_PA_BELIEF)
    assert "Raise avg claim strength by 0.10" in result
    assert "Add a claim at current average" in result
    assert "above avg" in result


@pytest.mark.unit
def test_position_analysis_retract_scenario():
    """With >=2 claims, retraction scenario is shown."""
    result = compute_position_analysis(_PA_BELIEF)
    assert "Retract weakest claim" in result
    # C2 at 0.55 is the weakest
    assert "C2" in result


@pytest.mark.unit
def test_position_analysis_no_retract_with_single_claim():
    """With only 1 claim, retraction scenario is not shown."""
    belief = {
        "claims": [
            {"id": "C1", "strength": 0.70, "status": "active",
             "depends_on": []},
        ],
    }
    result = compute_position_analysis(belief)
    assert "Retract weakest claim" not in result


@pytest.mark.unit
def test_position_analysis_weakest_dependencies():
    """Position analysis identifies weakest dependencies."""
    result = compute_position_analysis(_PA_BELIEF)
    # E1 at 0.30 is the weakest dependency, followed by A2 at 0.40
    assert "E1" in result
    assert "A2" in result


@pytest.mark.unit
def test_position_analysis_empty_belief():
    """Position analysis handles empty/no-claim belief gracefully."""
    result = compute_position_analysis({})
    assert "<position_analysis>" in result
    assert "No active claims" in result


@pytest.mark.unit
def test_position_analysis_retracted_claims_excluded():
    """Retracted claims should not be counted in position analysis."""
    belief = {
        "claims": [
            {"id": "C1", "strength": 0.70, "status": "active",
             "depends_on": []},
            {"id": "C2", "strength": 0.30, "status": "retracted",
             "depends_on": []},
        ],
    }
    result = compute_position_analysis(belief)
    assert "Active claims: 1" in result


@pytest.mark.unit
def test_position_analysis_breadth_sensitivity_override():
    """Position analysis respects custom breadth_sensitivity."""
    # With p=1.0, breadth for 3 claims = 3/4 = 0.75
    result = compute_position_analysis(_PA_BELIEF, breadth_sensitivity=1.0)
    assert "Breadth multiplier: 0.75" in result


@pytest.mark.unit
def test_position_analysis_recommendation_options():
    """Recommendation is one of the three expected options."""
    result = compute_position_analysis(_PA_BELIEF)
    has_option = (
        "Raising average claim strength" in result
        or "Adding more claims" in result
        or "Both levers are roughly equally valuable" in result
    )
    assert has_option, "Recommendation should be one of the three options"


@pytest.mark.unit
def test_phase2_prompt_contains_position_analysis():
    """Phase 2 prompt includes the dynamically computed <position_analysis> block."""
    import json
    belief = {
        "thesis": {"stance": "test", "strength": 0.5},
        "claims": [
            {"id": "C1", "strength": 0.70, "status": "active",
             "depends_on": []},
            {"id": "C2", "strength": 0.60, "status": "active",
             "depends_on": []},
        ],
    }
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=json.dumps(belief),
        phase1_changes_summary="none"
    )
    assert "<position_analysis>" in prompt
    assert "</position_analysis>" in prompt
    assert "Active claims: 2" in prompt
    assert "SCENARIO PROJECTIONS" in prompt


@pytest.mark.unit
def test_phase2_prompt_position_analysis_with_invalid_json():
    """Phase 2 prompt handles invalid JSON gracefully (empty position analysis)."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json="not-valid-json",
        phase1_changes_summary="none"
    )
    assert "<position_analysis>" in prompt
    assert "No active claims" in prompt


# ==============================================
# 11. Thesis Strength Guide & Restructured Instructions Tests
# ==============================================

@pytest.mark.unit
def test_phase2_thesis_strength_guide_present():
    """Phase 2 prompt contains <thesis_strength_guide> instead of <thesis_strength>."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "<thesis_strength_guide>" in prompt
    assert "</thesis_strength_guide>" in prompt
    # Old block should not be present
    assert "<thesis_strength>" not in prompt.replace("<thesis_strength_guide>", "").replace("</thesis_strength_guide>", "")


@pytest.mark.unit
def test_phase2_thesis_strength_guide_formula():
    """Thesis strength guide shows the n^p formula, not the old 1-1/(n+1)."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "n^p / (n^p + 1)" in prompt or "n^p" in prompt
    # Old formula should not appear
    assert "1 - 1/(num_active_claims + 1)" not in prompt
    assert "1 - 1/(num_claims + 1)" not in prompt


@pytest.mark.unit
def test_phase2_thesis_strength_guide_dependency_graph():
    """Thesis strength guide contains the dependency graph explanation."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "DEPENDENCY GRAPH" in prompt
    assert "Assumptions (A#)" in prompt
    assert "Claims (C#)" in prompt


@pytest.mark.unit
def test_phase2_thesis_strength_guide_two_levers():
    """Thesis strength guide contains both lever descriptions."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "Lever 1" in prompt
    assert "Lever 2" in prompt
    assert "TWO LEVERS" in prompt


@pytest.mark.unit
def test_phase2_thesis_strength_guide_breadth_table():
    """Thesis strength guide contains a dynamically computed breadth table."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    # Default p=1.5: 1 claim → 0.50, 2 claims → 0.74
    assert "1 claim" in prompt
    assert "2 claims" in prompt
    assert "0.50" in prompt


@pytest.mark.unit
def test_phase2_instructions_restructured():
    """Phase 2 instructions use roadmap structure: ADDRESS OPEN ISSUES, STRATEGIC POSITION BUILDING."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "ADDRESS OPEN ISSUES" in prompt
    assert "STRATEGIC POSITION BUILDING" in prompt
    assert "THESIS REWRITE" in prompt


@pytest.mark.unit
def test_phase2_instructions_uncertainty_review():
    """Restructured instructions contain an explicit Uncertainty Review sub-step."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "Uncertainty Review" in prompt
    assert "resolve_uncertainty" in prompt


@pytest.mark.unit
def test_phase2_instructions_add_claim_add_assumption():
    """Phase 2 SUPPORTED OPERATIONS includes add_claim and add_assumption."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "add_claim" in prompt
    assert "add_assumption" in prompt


@pytest.mark.unit
def test_phase2_instructions_patch_ordering():
    """Phase 2 mentions that supporting A#/E# must come BEFORE add_claim."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "BEFORE" in prompt
    assert "add_claim" in prompt


@pytest.mark.unit
def test_phase2_primary_goal_explicit():
    """Phase 2 instructions state the primary goal explicitly."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "strengthen your position" in prompt


@pytest.mark.unit
def test_stage1_formula_updated():
    """Stage 1 thesis_strength block uses the n^p formula."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="Test", agent_name="A", persona_label="Test"
    )
    assert "n^p / (n^p + 1)" in prompt or "n^p" in prompt
    assert "1 - 1/(num_claims + 1)" not in prompt


@pytest.mark.unit
def test_stage5_legacy_formula_updated():
    """Stage 5 legacy thesis_strength block uses the n^p formula."""
    pairs = [{"challenger": "B", "challenge": "test",
              "resolution": {"status": "critique_valid"}}]
    prompt = build_stage_5_belief_update_prompt_cbs(
        agent_name="A", challenge_rebuttal_pairs=pairs, prior_belief_json='{}'
    )
    assert "n^p / (n^p + 1)" in prompt or "n^p" in prompt
    assert "1 - 1/(num_active_claims + 1)" not in prompt
    assert "1 - 1/(num_claims + 1)" not in prompt


# ==============================================
# 12. Status and Dependency Rule Prompt Tests
# ==============================================

@pytest.mark.unit
def test_assumption_status_described_in_stage1():
    """Stage 1 prompt must describe assumption status field."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="test", agent_name="TEST", persona_label="TEST"
    )
    assert "status" in prompt.lower()
    # Check that assumption status values are mentioned
    assert '"active"' in prompt or "'active'" in prompt


@pytest.mark.unit
def test_evidence_status_described_in_stage1():
    """Stage 1 prompt must describe evidence status field."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="test", agent_name="TEST", persona_label="TEST"
    )
    # Evidence section should mention status
    assert "strength_justification" in prompt


@pytest.mark.unit
def test_dependency_rule_includes_all_types():
    """Dependency rule text should reference A#, E#, and C# — not just claims."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="test", agent_name="TEST", persona_label="TEST"
    )
    # The old exclusionary text should be gone
    assert "Assumptions and evidence are not capped this way" not in prompt
    # The new inclusive text should be present
    assert "active/revised" in prompt


@pytest.mark.unit
def test_quality_assessment_removed_from_prompts():
    """quality_assessment should not appear in Stage 1 prompt."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="test", agent_name="TEST", persona_label="TEST"
    )
    assert "quality_assessment" not in prompt


@pytest.mark.unit
def test_strength_justification_in_evidence_description():
    """Stage 1 prompt should mention strength_justification for evidence."""
    prompt = build_stage_1_belief_prompt_cbs(
        topic="test", agent_name="TEST", persona_label="TEST"
    )
    assert "strength_justification" in prompt


# ==============================================
# Stage 2 Attack Taxonomy Tests
# ==============================================

@pytest.mark.unit
def test_stage_2_prompt_contains_attack_taxonomy():
    """New prompt contains <attack_taxonomy> with all three types and all 15 strategies."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "<attack_taxonomy>" in prompt
    assert "</attack_taxonomy>" in prompt
    # All three attack types present
    assert "UNDERMINING" in prompt
    assert "REBUTTING" in prompt
    assert "UNDERCUTTING" in prompt
    # Spot-check strategies from each type
    assert "challenge_evidence" in prompt
    assert "exploit_counterposition" in prompt
    assert "identify_circularity" in prompt
    assert "press_uncertainty" in prompt


@pytest.mark.unit
def test_stage_2_prompt_no_old_attack_framework():
    """Old <attack_framework> tag and QUESTIONING STRATEGIES header are gone."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "<attack_framework>" not in prompt
    assert "QUESTIONING STRATEGIES" not in prompt


@pytest.mark.unit
def test_stage_2_prompt_output_format_has_attack_fields():
    """Output format references attack_type and attack_strategy, not old strategy field."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "attack_type" in prompt
    assert "attack_strategy" in prompt
    # Old field should not appear in output_format section
    output_section = prompt.split("<output_format>")[1].split("</output_format>")[0]
    assert '"strategy"' not in output_section


@pytest.mark.unit
def test_stage_2_prompt_no_max_question_length():
    """Prompt does not contain any character limit instruction."""
    prompt = build_stage_2_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "max_question_length" not in prompt


@pytest.mark.unit
def test_stage_2_bloodsport_prompt_has_attack_fields():
    """Bloodsport prompt includes attack_type and attack_strategy in its output format."""
    prompt = build_stage_2_bloodsport_prompt(
        topic="Test", agent_name="A", opponent_name="B",
        agent_belief_json='{}', opponent_belief_json='{}'
    )
    assert "attack_type" in prompt
    assert "attack_strategy" in prompt


# ==============================================
# 13. Adjudicator Prompt Mode Differentiation
#     (Phase 4 smoke test)
# ==============================================

@pytest.mark.unit
def test_adjudicator_prompt_mode_differentiation():
    """Build all three mode prompts with the same systems; verify structural differences.

    - All three prompts must be distinct strings.
    - Logic-only prompt must NOT contain any ethics criteria text.
    - Ethics-only prompt must NOT contain any logic criteria text.
    - Balanced prompt must contain criteria from BOTH systems.
    """
    logic_only = build_adjudicator_prompt(
        logic_weight=1.0, ethics_weight=0.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    ethics_only = build_adjudicator_prompt(
        logic_weight=0.0, ethics_weight=1.0,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )
    balanced = build_adjudicator_prompt(
        logic_weight=0.5, ethics_weight=0.5,
        logic_sys=_TEST_LOGIC_SYS, ethics_sys=_TEST_ETHICS_SYS,
    )

    # --- All three prompts must be distinct ---
    assert logic_only != ethics_only, "Logic-only and ethics-only prompts are identical"
    assert logic_only != balanced, "Logic-only and balanced prompts are identical"
    assert ethics_only != balanced, "Ethics-only and balanced prompts are identical"

    # --- Logic-only: has logic criteria, no ethics criteria ---
    assert "Logic critique alpha" in logic_only
    assert "Logic rebuttal alpha" in logic_only
    assert "Ethics critique gamma" not in logic_only
    assert "Ethics rebuttal gamma" not in logic_only
    assert "(ethical)" not in logic_only

    # --- Ethics-only: has ethics criteria, no logic criteria ---
    assert "Ethics critique gamma" in ethics_only
    assert "Ethics rebuttal gamma" in ethics_only
    assert "Logic critique alpha" not in ethics_only
    assert "Logic rebuttal alpha" not in ethics_only
    assert "(logical)" not in ethics_only

    # --- Balanced: has BOTH systems' criteria with prefixes ---
    assert "Logic critique alpha" in balanced
    assert "Ethics critique gamma" in balanced
    assert "Logic rebuttal alpha" in balanced
    assert "Ethics rebuttal gamma" in balanced
    assert "(logical)" in balanced
    assert "(ethical)" in balanced

    # --- Mode-specific instructions differ ---
    assert "Disregard any ethical arguments" in logic_only
    assert "Logical validity is irrelevant" in ethics_only
    assert "Logical soundness is the baseline" in balanced


# ==============================================
# Adjudicator Per-Pair Prompt Tests
# ==============================================

@pytest.mark.unit
def test_adjudicator_per_pair_prompt_contains_challenge():
    """Challenge text appears inside <challenge> tags."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="Your claim lacks evidence",
        rebuttal="I provided three sources",
        challenger="Agent-A",
        target="Agent-B",
        mode_label="Pure Logic",
    )
    assert "<challenge" in prompt
    assert "Your claim lacks evidence" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_contains_rebuttal():
    """Rebuttal text appears inside <rebuttal> tags."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="Your claim lacks evidence",
        rebuttal="I provided three sources",
        challenger="Agent-A",
        target="Agent-B",
        mode_label="Pure Logic",
    )
    assert "<rebuttal" in prompt
    assert "I provided three sources" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_contains_challenger_name():
    """Challenger name appears in from= attribute."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="Your claim lacks evidence",
        rebuttal="I provided three sources",
        challenger="Agent-Empiricist",
        target="Agent-Rationalist",
        mode_label="Pure Logic",
    )
    assert 'from="Agent-Empiricist"' in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_contains_target_name():
    """Target name appears in from= attribute on the rebuttal tag."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="Your claim lacks evidence",
        rebuttal="I provided three sources",
        challenger="Agent-Empiricist",
        target="Agent-Rationalist",
        mode_label="Pure Logic",
    )
    assert 'from="Agent-Rationalist"' in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_mode_label():
    """Mode label appears in the instructions section."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Balanced",
    )
    assert "Balanced" in prompt
    assert "Evaluate this exchange under Balanced mode" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_logic_sys_description():
    """Logic system description included when provided."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
        logic_sys_description="Classical Informal + Bayesian reasoning",
    )
    assert "Logic system: Classical Informal + Bayesian reasoning" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_ethics_sys_description():
    """Ethics system description included when provided."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Balanced",
        ethics_sys_description="Utilitarian framework",
    )
    assert "Ethics system: Utilitarian framework" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_no_sys_when_empty():
    """No 'Logic system:' or 'Ethics system:' lines when descriptions are empty."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
        logic_sys_description="",
        ethics_sys_description="",
    )
    assert "Logic system:" not in prompt
    assert "Ethics system:" not in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_belief_excerpts_included():
    """Belief excerpt XML blocks present when provided."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
        challenger_belief_excerpt_json='{"belief": "test_challenger"}',
        target_belief_excerpt_json='{"belief": "test_target"}',
    )
    assert "<challenger_belief_excerpt>" in prompt
    assert "test_challenger" in prompt
    assert "<target_belief_excerpt>" in prompt
    assert "test_target" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_no_excerpts_when_empty():
    """No excerpt XML blocks when not provided."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
    )
    assert "<challenger_belief_excerpt>" not in prompt
    assert "<target_belief_excerpt>" not in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_output_format():
    """Output format section includes required JSON fields."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
    )
    assert "<output_format>" in prompt
    assert '"restatement"' in prompt
    assert '"formalization_challenger"' in prompt
    assert '"formalization_target"' in prompt
    assert '"challenger_logic"' in prompt
    assert '"challenger_ethics"' in prompt
    assert '"defender_logic"' in prompt
    assert '"defender_ethics"' in prompt
    assert '"challenger_combined"' in prompt
    assert '"defender_combined"' in prompt
    assert '"outcome"' in prompt
    assert '"reasoning"' in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_three_step_protocol():
    """Instructions reference the three-step protocol."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
    )
    assert "three-step protocol" in prompt


@pytest.mark.unit
def test_adjudicator_per_pair_prompt_no_debate_context():
    """Per-pair prompt should NOT contain <debate_context> (it's not a stage prompt)."""
    prompt = build_adjudicator_per_pair_prompt(
        challenge="X",
        rebuttal="Y",
        challenger="A",
        target="B",
        mode_label="Pure Logic",
        logic_sys_description="Some logic",
        ethics_sys_description="Some ethics",
        challenger_belief_excerpt_json='{"b": 1}',
        target_belief_excerpt_json='{"b": 2}',
    )
    assert "<debate_context>" not in prompt


# ==============================================
# 3A. Phase 1 update_thesis Removal Tests
# ==============================================

@pytest.mark.unit
def test_phase1_prompt_no_update_thesis():
    """Assert update_thesis does NOT appear in Phase 1 enforcement prompt output."""
    prompt = build_stage_5_phase1_enforcement_prompt(
        agent_name="Agent-A",
        challenge_rebuttal_pairs=SAMPLE_PAIRS,
        prior_belief_json=SAMPLE_BELIEF_JSON
    )
    assert "update_thesis" not in prompt


@pytest.mark.unit
def test_phase2_prompt_has_update_thesis():
    """Assert update_thesis DOES appear in Phase 2 introspection prompt output."""
    prompt = build_stage_5_phase2_introspection_prompt(
        agent_name="Agent-A",
        intermediate_belief_json=INTERMEDIATE_BELIEF_JSON,
        phase1_changes_summary=PHASE1_SUMMARY
    )
    assert "update_thesis" in prompt
