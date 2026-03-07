"""
Unit tests for the logic_systems module.

Tests cover:
- All logic system keys exist in the LOGIC_SYSTEMS dict
- Labels and descriptions dicts match LOGIC_SYSTEMS keys
- All values are non-empty strings
- Lookup via get_logic_system() (valid, case-insensitive, unknown key)
"""

import pytest
from chal.agents.logic_systems import (
    LOGIC_SYSTEMS,
    LOGIC_LABELS,
    LOGIC_DESCRIPTIONS,
    get_logic_system,
)

EXPECTED_KEYS = [
    "CLASSICAL_BAYESIAN",
    "FORMAL_DEDUCTIVE",
    "DIALECTICAL",
    "INFORMAL_CRITICAL",
    "FUZZY_MULTIVALUED",
    "PARACONSISTENT",
]


@pytest.mark.unit
def test_all_systems_in_dict():
    """All 6 logic system keys exist in LOGIC_SYSTEMS."""
    for key in EXPECTED_KEYS:
        assert key in LOGIC_SYSTEMS, f"Missing logic system key: {key}"
    assert len(LOGIC_SYSTEMS) == 6


@pytest.mark.unit
def test_labels_match_systems():
    """Every key in LOGIC_SYSTEMS has a corresponding entry in LOGIC_LABELS."""
    for key in LOGIC_SYSTEMS:
        assert key in LOGIC_LABELS, f"Missing label for {key}"


@pytest.mark.unit
def test_descriptions_match_systems():
    """Every key in LOGIC_SYSTEMS has a corresponding entry in LOGIC_DESCRIPTIONS."""
    for key in LOGIC_SYSTEMS:
        assert key in LOGIC_DESCRIPTIONS, f"Missing description for {key}"


@pytest.mark.unit
def test_system_values_are_strings():
    """Every value in LOGIC_SYSTEMS is a non-empty string."""
    for key, value in LOGIC_SYSTEMS.items():
        assert isinstance(value, str), f"{key} value is not a string"
        assert len(value) > 0, f"{key} value is empty"


@pytest.mark.unit
def test_get_logic_system_valid_key():
    """get_logic_system('CLASSICAL_BAYESIAN') returns correct text."""
    result = get_logic_system("CLASSICAL_BAYESIAN")
    assert isinstance(result, str)
    assert "bayesian" in result.lower()


@pytest.mark.unit
def test_get_logic_system_case_insensitive():
    """get_logic_system('classical_bayesian') works (uppercased internally)."""
    lower = get_logic_system("classical_bayesian")
    upper = get_logic_system("CLASSICAL_BAYESIAN")
    assert lower == upper


@pytest.mark.unit
def test_get_logic_system_unknown_key():
    """get_logic_system('UNKNOWN') raises KeyError."""
    with pytest.raises(KeyError):
        get_logic_system("UNKNOWN")
