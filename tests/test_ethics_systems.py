"""
Unit tests for the ethics_systems module.

Tests cover:
- All ethics system keys exist in the ETHICS_SYSTEMS dict
- Labels and descriptions dicts match ETHICS_SYSTEMS keys
- All values are non-empty strings
- Lookup via get_ethics_system() (valid, case-insensitive, unknown key)
"""

import pytest
from chal.agents.ethics_systems import (
    ETHICS_SYSTEMS,
    ETHICS_LABELS,
    ETHICS_DESCRIPTIONS,
    get_ethics_system,
)

EXPECTED_KEYS = [
    "NONE",
    "UTILITARIAN",
    "DEONTOLOGICAL",
    "VIRTUE_ETHICS",
    "CARE_ETHICS",
    "BALANCED",
]


@pytest.mark.unit
def test_all_systems_in_dict():
    """All 6 ethics system keys exist in ETHICS_SYSTEMS."""
    for key in EXPECTED_KEYS:
        assert key in ETHICS_SYSTEMS, f"Missing ethics system key: {key}"
    assert len(ETHICS_SYSTEMS) == 6


@pytest.mark.unit
def test_labels_match_systems():
    """Every key in ETHICS_SYSTEMS has a corresponding entry in ETHICS_LABELS."""
    for key in ETHICS_SYSTEMS:
        assert key in ETHICS_LABELS, f"Missing label for {key}"


@pytest.mark.unit
def test_descriptions_match_systems():
    """Every key in ETHICS_SYSTEMS has a corresponding entry in ETHICS_DESCRIPTIONS."""
    for key in ETHICS_SYSTEMS:
        assert key in ETHICS_DESCRIPTIONS, f"Missing description for {key}"


@pytest.mark.unit
def test_system_values_are_strings():
    """Every value in ETHICS_SYSTEMS is a non-empty string."""
    for key, value in ETHICS_SYSTEMS.items():
        assert isinstance(value, str), f"{key} value is not a string"
        assert len(value) > 0, f"{key} value is empty"


@pytest.mark.unit
def test_get_ethics_system_valid_key():
    """get_ethics_system('NONE') returns correct text."""
    result = get_ethics_system("NONE")
    assert isinstance(result, str)
    assert "ethical" in result.lower() or "logic" in result.lower()


@pytest.mark.unit
def test_get_ethics_system_case_insensitive():
    """get_ethics_system('none') works (uppercased internally)."""
    lower = get_ethics_system("none")
    upper = get_ethics_system("NONE")
    assert lower == upper


@pytest.mark.unit
def test_get_ethics_system_unknown_key():
    """get_ethics_system('UNKNOWN') raises KeyError."""
    with pytest.raises(KeyError):
        get_ethics_system("UNKNOWN")
