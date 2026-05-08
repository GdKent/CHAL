"""
Unit tests for the logic_systems module.

Tests cover:
- All 8 logic system keys exist in LOGIC_SYSTEMS
- Every system dict has required keys (label, description, criteria)
- Every criteria dict has the three required sub-keys
- Each criteria list is non-empty (except NONE)
- NONE system has empty criteria lists
- get_logic_system() returns a dict (not a string)
- get_logic_system() raises KeyError for unknown keys
- get_logic_system_description() returns a non-empty string
- get_logic_system_label() returns a non-empty string
- CLASSICAL_INFORMAL_BAYESIAN is a genuine superset of its components
"""

import pytest
from chal.agents.logic_systems import (
    LOGIC_SYSTEMS,
    get_logic_system,
    get_logic_system_description,
    get_logic_system_label,
)

EXPECTED_KEYS = [
    "CLASSICAL_INFORMAL_BAYESIAN",
    "FORMAL_DEDUCTIVE",
    "BAYESIAN",
    "INFORMAL_CRITICAL",
    "DIALECTICAL",
    "FUZZY_MULTIVALUED",
    "PARACONSISTENT",
    "NONE",
]

# Systems that should have non-empty criteria lists
SYSTEMS_WITH_CRITERIA = [k for k in EXPECTED_KEYS if k != "NONE"]

REQUIRED_DICT_KEYS = {"label", "description", "criteria"}
REQUIRED_CRITERIA_KEYS = {"critique_valid", "rebuttal_valid", "unresolved"}

# Expected criteria counts per system: (critique_valid, rebuttal_valid, unresolved)
EXPECTED_COUNTS = {
    "CLASSICAL_INFORMAL_BAYESIAN": (22, 22, 8),
    "FORMAL_DEDUCTIVE": (8, 6, 3),
    "BAYESIAN": (7, 7, 3),
    "INFORMAL_CRITICAL": (8, 7, 3),
    "DIALECTICAL": (6, 5, 3),
    "FUZZY_MULTIVALUED": (6, 6, 3),
    "PARACONSISTENT": (5, 5, 3),
    "NONE": (0, 0, 0),
}


# ==============================================
# 1. System Registry Tests
# ==============================================

@pytest.mark.unit
def test_all_systems_in_dict():
    """All 8 logic system keys exist in LOGIC_SYSTEMS."""
    for key in EXPECTED_KEYS:
        assert key in LOGIC_SYSTEMS, f"Missing logic system key: {key}"
    assert len(LOGIC_SYSTEMS) == 8


@pytest.mark.unit
def test_no_classical_bayesian():
    """CLASSICAL_BAYESIAN has been removed (replaced by CLASSICAL_INFORMAL_BAYESIAN)."""
    assert "CLASSICAL_BAYESIAN" not in LOGIC_SYSTEMS


# ==============================================
# 2. Dict Structure Tests
# ==============================================

@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_system_has_required_keys(key):
    """Every system dict has label, description, and criteria keys."""
    system = LOGIC_SYSTEMS[key]
    assert isinstance(system, dict), f"{key} is not a dict"
    missing = REQUIRED_DICT_KEYS - set(system.keys())
    assert not missing, f"{key} missing keys: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_has_required_keys(key):
    """Every criteria dict has critique_valid, rebuttal_valid, unresolved."""
    criteria = LOGIC_SYSTEMS[key]["criteria"]
    assert isinstance(criteria, dict), f"{key} criteria is not a dict"
    missing = REQUIRED_CRITERIA_KEYS - set(criteria.keys())
    assert not missing, f"{key} criteria missing keys: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("key", SYSTEMS_WITH_CRITERIA)
def test_criteria_lists_are_non_empty(key):
    """Each non-NONE system's criteria lists have at least one item."""
    criteria = LOGIC_SYSTEMS[key]["criteria"]
    for outcome in REQUIRED_CRITERIA_KEYS:
        items = criteria[outcome]
        assert isinstance(items, list), f"{key}.{outcome} is not a list"
        assert len(items) > 0, f"{key}.{outcome} is empty"


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_items_are_non_empty_strings(key):
    """Every criterion is a non-empty string."""
    criteria = LOGIC_SYSTEMS[key]["criteria"]
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
    label = LOGIC_SYSTEMS[key]["label"]
    assert isinstance(label, str) and len(label) > 0


@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_description_is_non_empty_string(key):
    """Every system has a non-empty description string."""
    desc = LOGIC_SYSTEMS[key]["description"]
    assert isinstance(desc, str) and len(desc) > 0


# ==============================================
# 3. NONE System Special Cases
# ==============================================

@pytest.mark.unit
def test_none_has_empty_criteria():
    """NONE system has empty criteria lists (no logical evaluation)."""
    criteria = LOGIC_SYSTEMS["NONE"]["criteria"]
    assert criteria["critique_valid"] == []
    assert criteria["rebuttal_valid"] == []
    assert criteria["unresolved"] == []


@pytest.mark.unit
def test_none_has_description():
    """NONE system still has a non-empty description."""
    desc = LOGIC_SYSTEMS["NONE"]["description"]
    assert isinstance(desc, str) and len(desc) > 0


# ==============================================
# 4. Criteria Count Tests
# ==============================================

@pytest.mark.unit
@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_criteria_counts(key):
    """Each system has the expected number of criteria."""
    criteria = LOGIC_SYSTEMS[key]["criteria"]
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
def test_get_logic_system_returns_dict():
    """get_logic_system() returns a dict, not a string."""
    result = get_logic_system("CLASSICAL_INFORMAL_BAYESIAN")
    assert isinstance(result, dict)
    assert "label" in result
    assert "description" in result
    assert "criteria" in result


@pytest.mark.unit
def test_get_logic_system_case_insensitive():
    """get_logic_system() is case-insensitive."""
    lower = get_logic_system("formal_deductive")
    upper = get_logic_system("FORMAL_DEDUCTIVE")
    assert lower is upper


@pytest.mark.unit
def test_get_logic_system_unknown_key():
    """get_logic_system() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_logic_system("UNKNOWN")


@pytest.mark.unit
def test_get_logic_system_description():
    """get_logic_system_description() returns the description string."""
    desc = get_logic_system_description("BAYESIAN")
    assert isinstance(desc, str)
    assert "Bayesian" in desc


@pytest.mark.unit
def test_get_logic_system_description_case_insensitive():
    """get_logic_system_description() is case-insensitive."""
    lower = get_logic_system_description("bayesian")
    upper = get_logic_system_description("BAYESIAN")
    assert lower == upper


@pytest.mark.unit
def test_get_logic_system_description_unknown_key():
    """get_logic_system_description() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_logic_system_description("NONEXISTENT")


@pytest.mark.unit
def test_get_logic_system_label():
    """get_logic_system_label() returns the label string."""
    label = get_logic_system_label("DIALECTICAL")
    assert isinstance(label, str)
    assert len(label) > 0


@pytest.mark.unit
def test_get_logic_system_label_unknown_key():
    """get_logic_system_label() raises KeyError for unknown keys."""
    with pytest.raises(KeyError):
        get_logic_system_label("NONEXISTENT")


# ==============================================
# 6. Superset Verification
# ==============================================

@pytest.mark.unit
def test_hybrid_is_larger_than_components():
    """CLASSICAL_INFORMAL_BAYESIAN has more criteria than any single component."""
    hybrid = LOGIC_SYSTEMS["CLASSICAL_INFORMAL_BAYESIAN"]["criteria"]
    for component_key in ["FORMAL_DEDUCTIVE", "BAYESIAN", "INFORMAL_CRITICAL"]:
        component = LOGIC_SYSTEMS[component_key]["criteria"]
        for outcome in REQUIRED_CRITERIA_KEYS:
            assert len(hybrid[outcome]) > len(component[outcome]), (
                f"Hybrid {outcome} ({len(hybrid[outcome])}) should be larger "
                f"than {component_key} {outcome} ({len(component[outcome])})"
            )
