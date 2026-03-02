"""
Unit tests for agent prompts and prompt builders.

Tests cover:
- Persona prompts validation
- Universal debate rules
- Adjudicator prompt builder
"""

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
    build_adjudicator_prompt
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

    all_past_beliefs = [
        '{"version": 1, "thesis": {"statement": "Initial"}}',
        '{"version": 2, "thesis": {"statement": "Updated"}}'
    ]

    prompt = build_stage_6_conclusion_prompt(
        topic="Future of AI",
        agent_name="Agent-Futurist",
        agent_belief_json='{"thesis": {"statement": "Final position"}}',
        all_past_beliefs=all_past_beliefs
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
