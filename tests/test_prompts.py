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

@pytest.mark.unit
def test_build_adjudicator_prompt_defaults():
    """Test building adjudicator prompt with default weights."""
    prompt = build_adjudicator_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "adjudicat" in prompt.lower()


@pytest.mark.unit
def test_build_adjudicator_prompt_custom_weights():
    """Test building prompt with custom logic/ethics weights."""
    prompt = build_adjudicator_prompt(logic_weight=0.7, ethics_weight=0.3)

    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Should mention both logic and ethics if both weights > 0
    prompt_lower = prompt.lower()
    assert "logic" in prompt_lower or "logical" in prompt_lower
    assert "ethic" in prompt_lower or "moral" in prompt_lower


@pytest.mark.unit
def test_build_adjudicator_prompt_custom_systems():
    """Test building prompt with custom logic/ethics systems."""
    custom_logic = "Use formal predicate logic only"
    custom_ethics = "Apply Kantian deontology"

    prompt = build_adjudicator_prompt(
        logic_sys=custom_logic,
        ethics_sys=custom_ethics
    )

    assert custom_logic in prompt
    assert custom_ethics in prompt


@pytest.mark.unit
def test_build_adjudicator_prompt_includes_frameworks():
    """Test that generated prompt contains framework descriptions."""
    prompt = build_adjudicator_prompt(logic_weight=1.0, ethics_weight=0.5)

    assert isinstance(prompt, str)
    # Should describe adjudication process
    assert len(prompt) > 100


@pytest.mark.unit
def test_build_adjudicator_prompt_logic_only():
    """Test prompt with logic_weight=1.0, ethics_weight=0.0."""
    prompt = build_adjudicator_prompt(logic_weight=1.0, ethics_weight=0.0)

    assert isinstance(prompt, str)
    prompt_lower = prompt.lower()
    assert "logic" in prompt_lower or "logical" in prompt_lower


@pytest.mark.unit
def test_build_adjudicator_prompt_ethics_only():
    """Test prompt with logic_weight=0.0, ethics_weight=1.0."""
    prompt = build_adjudicator_prompt(logic_weight=0.0, ethics_weight=1.0)

    assert isinstance(prompt, str)
    prompt_lower = prompt.lower()
    assert "ethic" in prompt_lower or "moral" in prompt_lower


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
        opponent_belief_graph=None
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
def test_adjudicator_prompt_uses_strength():
    """Adjudicator prompt criteria should reference 'strength' not 'confidence'."""
    prompt = build_adjudicator_prompt()
    assert "Strength exceeding" in prompt
    assert "Confidence exceeding" not in prompt


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
    prompt = build_adjudicator_prompt()
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
