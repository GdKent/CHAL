"""
Unit tests for the epistemic_personas module.

Tests cover:
- All personas exist in the PERSONAS dict
- All values are non-empty strings
- Lookup via get_persona() (valid, case-insensitive, unknown key)
- Labels and descriptions dicts match PERSONAS keys
- Backward-compatible re-export through prompts.py
"""

import pytest
from chal.agents.epistemic_personas import (
    PERSONAS,
    PERSONA_LABELS,
    PERSONA_DESCRIPTIONS,
    get_persona,
)

EXPECTED_KEYS = [
    "EMPIRICIST",
    "SUPERNATURALIST",
    "SKEPTIC",
    "RATIONALIST",
    "PHENOMENOLOGIST",
    "PRAGMATIST",
    "CONSTRUCTIVIST",
    "NIHILIST",
    "BAYESIAN",
    "PANPSYCHIST",
    "SIMULATIONIST",
    "SYNTHESIST",
    "NONE",
]


@pytest.mark.unit
def test_all_personas_in_dict():
    """All 12 persona keys exist in the PERSONAS dict."""
    for key in EXPECTED_KEYS:
        assert key in PERSONAS, f"Missing persona key: {key}"
    assert len(PERSONAS) == 13


@pytest.mark.unit
def test_none_persona_is_empty_string():
    """NONE persona returns empty string (no worldview)."""
    result = get_persona("none")
    assert result == ""


@pytest.mark.unit
def test_persona_values_are_strings():
    """Every persona value is a string; all except NONE are non-empty."""
    for key, value in PERSONAS.items():
        assert isinstance(value, str), f"{key} value is not a string"
        if key != "NONE":
            assert len(value) > 0, f"{key} value is empty"


@pytest.mark.unit
def test_get_persona_valid_key():
    """get_persona('EMPIRICIST') returns the correct text."""
    result = get_persona("EMPIRICIST")
    assert isinstance(result, str)
    assert "empiricist" in result.lower()


@pytest.mark.unit
def test_get_persona_case_insensitive():
    """get_persona('empiricist') works (uppercased internally)."""
    lower = get_persona("empiricist")
    upper = get_persona("EMPIRICIST")
    assert lower == upper


@pytest.mark.unit
def test_get_persona_unknown_key():
    """get_persona('UNKNOWN') raises KeyError."""
    with pytest.raises(KeyError):
        get_persona("UNKNOWN")


@pytest.mark.unit
def test_labels_match_personas():
    """Every key in PERSONAS has a corresponding entry in PERSONA_LABELS."""
    for key in PERSONAS:
        assert key in PERSONA_LABELS, f"Missing label for {key}"


@pytest.mark.unit
def test_descriptions_match_personas():
    """Every key in PERSONAS has a corresponding entry in PERSONA_DESCRIPTIONS."""
    for key in PERSONAS:
        assert key in PERSONA_DESCRIPTIONS, f"Missing description for {key}"


@pytest.mark.unit
def test_prompts_reexport():
    """getattr(prompts, 'EMPIRICIST') still works via re-export."""
    from chal.agents import prompts

    for key in EXPECTED_KEYS:
        val = getattr(prompts, key, None)
        assert val is not None, f"prompts.{key} missing after re-export"
        assert val == PERSONAS[key], f"prompts.{key} differs from PERSONAS[{key}]"
