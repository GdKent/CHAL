"""
Unit tests for the ethics_systems module.

Tests cover:
- All 6 ethics system keys exist in ETHICS_SYSTEMS
- Every system dict has required keys (label, description, criteria)
- Every criteria dict has the three required sub-keys
- Each criteria list is non-empty (except NONE)
- NONE system has empty criteria lists
- get_ethics_system() returns a dict (not a string)
- get_ethics_system() raises KeyError for unknown keys
- get_ethics_system_description() returns a non-empty string
- get_ethics_system_label() returns a non-empty string
"""

import pytest
from chal.agents.ethics_systems import (
    ETHICS_SYSTEMS,
    get_ethics_system,
    get_ethics_system_description,
    get_ethics_system_label,
)

EXPECTED_KEYS = [
    "NONE",
    "UTILITARIAN",
    "DEONTOLOGICAL",
    "VIRTUE_ETHICS",
    "CARE_ETHICS",
    "BALANCED",
]

# Systems that should have non-empty criteria lists
SYSTEMS_WITH_CRITERIA = [k for k in EXPECTED_KEYS if k != "NONE"]

REQUIRED_DICT_KEYS = {"label", "description", "criteria"}
REQUIRED_CRITERIA_KEYS = {"critique_valid", "rebuttal_valid", "unresolved"}

# Expected criteria counts: (critique_valid, rebuttal_valid, unresolved)
EXPECTED_COUNTS = {
    "NONE": (0, 0, 0),
    "UTILITARIAN": (6, 6, 3),
    "DEONTOLOGICAL": (6, 6, 3),
    "VIRTUE_ETHICS": (6, 6, 3),
    "CARE_ETHICS": (6, 6, 3),
    "BALANCED": (6, 6, 3),
}


# ==============================================
# 1. System Registry Tests
# ==============================================

@pytest.mark.unit
def test_all_systems_in_dict():
    """All 6 ethics system keys exist in ETHICS_SYSTEMS."""
    for key in EXPECTED_KEYS:
        assert key in ETHICS_SYSTEMS, f"Missing ethics system key: {key}"
    assert len(ETHICS_SYSTEMS) == 6


# ==============================================
# 2. Dict Structure Tests
# ==============================================

@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_system_has_required_keys(key):
    """Every system dict has label, description, and criteria keys."""
    system = ETHICS_SYSTEMS[key]
    assert isinstance(system, dict), f"{key} is not a dict"
    missing = REQUIRED_DICT_KEYS - set(system.keys())
    assert not missing, f"{key} missing keys: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_has_required_keys(key):
    """Every criteria dict has critique_valid, rebuttal_valid, unresolved."""
    criteria = ETHICS_SYSTEMS[key]["criteria"]
    assert isinstance(criteria, dict), f"{key} criteria is not a dict"
    missing = REQUIRED_CRITERIA_KEYS - set(criteria.keys())
    assert not missing, f"{key} criteria missing keys: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("key", SYSTEMS_WITH_CRITERIA)
def test_criteria_lists_are_non_empty(key):
    """Each non-NONE system's criteria lists have at least one item."""
    criteria = ETHICS_SYSTEMS[key]["criteria"]
    for outcome in REQUIRED_CRITERIA_KEYS:
        items = criteria[outcome]
        assert isinstance(items, list), f"{key}.{outcome} is not a list"
        assert len(items) > 0, f"{key}.{outcome} is empty"


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_items_are_strings(key):
    """Every criterion is a non-empty string (if any exist)."""
    criteria = ETHICS_SYSTEMS[key]["criteria"]
    for outcome in REQUIRED_CRITERIA_KEYS:
        for i, item in enumerate(criteria[outcome]):
            assert isinstance(item, str), (
                f"{key}.{outcome}[{i}] is not a string"
            )
            assert len(item) > 0, f"{key}.{outcome}[{i}] is empty"


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_label_is_non_empty_string(key):
    """Every system has a non-empty label string."""
    label = ETHICS_SYSTEMS[key]["label"]
    assert isinstance(label, str) and len(label) > 0


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_description_is_non_empty_string(key):
    """Every system has a non-empty description string."""
    desc = ETHICS_SYSTEMS[key]["description"]
    assert isinstance(desc, str) and len(desc) > 0


# ==============================================
# 3. NONE System Special Cases
# ==============================================

@pytest.mark.unit
def test_none_has_empty_criteria():
    """NONE system has empty criteria lists (no ethical evaluation)."""
    criteria = ETHICS_SYSTEMS["NONE"]["criteria"]
    assert criteria["critique_valid"] == []
    assert criteria["rebuttal_valid"] == []
    assert criteria["unresolved"] == []


@pytest.mark.unit
def test_none_has_description():
    """NONE system still has a non-empty description."""
    desc = ETHICS_SYSTEMS["NONE"]["description"]
    assert isinstance(desc, str) and len(desc) > 0


# ==============================================
# 4. Criteria Count Tests
# ==============================================

@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_counts(key):
    """Each system has the expected number of criteria."""
    criteria = ETHICS_SYSTEMS[key]["criteria"]
    expected = EXPECTED_COUNTS[key]
    actual = (
        len(criteria["critique_valid"]),
        len(criteria["rebuttal_valid"]),
        len(criteria["unresolved"]),
    )
    assert actual == expected, (
        f"{key}: expected {expected}, got {actual}"
    )


# ==============================================
# 5. Lookup Function Tests
# ==============================================

@pytest.mark.unit
def test_get_ethics_system_returns_dict():
    """get_ethics_system() returns a dict, not a string."""
    result = get_ethics_system("UTILITARIAN")
    assert isinstance(result, dict)
    assert "label" in result
    assert "description" in result
    assert "criteria" in result


@pytest.mark.unit
def test_get_ethics_system_case_insensitive():
    """get_ethics_system() is case-insensitive."""
    lower = get_ethics_system("none")
    upper = get_ethics_system("NONE")
    assert lower is upper


@pytest.mark.unit
def test_get_ethics_system_unknown_key():
    """get_ethics_system() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_ethics_system("UNKNOWN")


@pytest.mark.unit
def test_get_ethics_system_description():
    """get_ethics_system_description() returns the description string."""
    desc = get_ethics_system_description("DEONTOLOGICAL")
    assert isinstance(desc, str)
    assert "Kantian" in desc


@pytest.mark.unit
def test_get_ethics_system_description_case_insensitive():
    """get_ethics_system_description() is case-insensitive."""
    lower = get_ethics_system_description("utilitarian")
    upper = get_ethics_system_description("UTILITARIAN")
    assert lower == upper


@pytest.mark.unit
def test_get_ethics_system_description_unknown_key():
    """get_ethics_system_description() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_ethics_system_description("NONEXISTENT")


@pytest.mark.unit
def test_get_ethics_system_label():
    """get_ethics_system_label() returns the label string."""
    label = get_ethics_system_label("VIRTUE_ETHICS")
    assert isinstance(label, str)
    assert len(label) > 0


@pytest.mark.unit
def test_get_ethics_system_label_unknown_key():
    """get_ethics_system_label() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_ethics_system_label("NONEXISTENT")
