"""
Phase 5 tests for Definition Nodes (D#) — Vulnerability Detection.

Tests:
- compute_position_analysis(): D# vulnerability detection (weak defs, bottlenecks)
"""

import pytest
from chal.agents.prompts import compute_position_analysis


# ========================================
# Helpers
# ========================================

def _make_position_belief(**kwargs):
    """Helper to create a belief dict for compute_position_analysis() tests."""
    belief = {
        "claims": kwargs.get("claims", [
            {"id": "C1", "strength": 0.7, "status": "active",
             "depends_on": ["A1", "E1"]},
        ]),
        "assumptions": kwargs.get("assumptions", [
            {"id": "A1", "strength": 0.8, "status": "active",
             "supported_by_definitions": ["D1"]},
        ]),
        "evidence": kwargs.get("evidence", [
            {"id": "E1", "strength": 0.75, "status": "active",
             "supported_by_definitions": ["D1"]},
        ]),
        "definitions": kwargs.get("definitions", [
            {"id": "D1", "term": "test", "definition": "test def",
             "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
        ]),
        "counterpositions": kwargs.get("counterpositions", []),
    }
    return belief


# ========================================
# 1. compute_position_analysis() — D# vulnerabilities
# ========================================

class TestPositionAnalysisDefinitionalVulnerabilities:
    """Tests for D# vulnerability detection in compute_position_analysis()."""

    @pytest.mark.unit
    def test_weak_definition_detected(self):
        """Low-strength D# supporting >=2 nodes triggers WEAK DEFINITION."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "weak def",
                 "strength": 0.5, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.5, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.5, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "WEAK DEFINITION" in result
        assert "D1" in result

    @pytest.mark.unit
    def test_no_weak_definition_when_strong(self):
        """High-strength D# does not trigger WEAK DEFINITION."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "strong def",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "WEAK DEFINITION" not in result

    @pytest.mark.unit
    def test_bottleneck_detected(self):
        """A#/E# with single active D# support triggers BOTTLENECK."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "only def",
                 "strength": 0.7, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.7, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "BOTTLENECK" in result
        assert "A1" in result
        assert "D1" in result

    @pytest.mark.unit
    def test_no_bottleneck_with_multiple_defs(self):
        """A# with multiple active D# support does not trigger BOTTLENECK."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test1", "definition": "def1",
                 "strength": 0.8, "status": "active", "used_by": ["A1", "E1"]},
                {"id": "D2", "term": "test2", "definition": "def2",
                 "strength": 0.8, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "BOTTLENECK" not in result

    @pytest.mark.unit
    def test_no_vulnerability_section_when_none(self):
        """No DEFINITIONAL VULNERABILITIES section when none found."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "strong def",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
                {"id": "D2", "term": "test2", "definition": "strong def2",
                 "strength": 0.85, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" not in result

    @pytest.mark.unit
    def test_vulnerability_section_present_when_found(self):
        """DEFINITIONAL VULNERABILITIES section present when issues found."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "weak def",
                 "strength": 0.4, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.4, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.4, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" in result

    @pytest.mark.unit
    def test_retracted_def_not_counted_as_support(self):
        """Retracted D# not counted for bottleneck analysis."""
        belief = _make_position_belief(
            definitions=[
                {"id": "D1", "term": "test", "definition": "retracted",
                 "strength": 0.0, "status": "retracted", "used_by": ["A1"]},
                {"id": "D2", "term": "test2", "definition": "active",
                 "strength": 0.8, "status": "active", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.7, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        # D1 retracted → only D2 active → bottleneck on A1
        assert "BOTTLENECK" in result
        assert "A1" in result

    @pytest.mark.unit
    def test_no_definitions_no_vulnerability(self):
        """Belief with no definitions produces no vulnerability section."""
        belief = _make_position_belief(
            definitions=[],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": []},
            ],
            evidence=[
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": []},
            ],
        )
        result = compute_position_analysis(belief)
        assert "DEFINITIONAL VULNERABILITIES" not in result

    @pytest.mark.unit
    def test_standard_sections_preserved(self):
        """Standard position analysis sections still present."""
        belief = _make_position_belief()
        result = compute_position_analysis(belief)
        assert "YOUR CURRENT POSITION" in result
        assert "SENSITIVITY AT YOUR POSITION" in result
        assert "SCENARIO PROJECTIONS" in result
        assert "LOWEST-STRENGTH DEPENDENCIES" in result
        assert "STRATEGIC RECOMMENDATION" in result
        assert "INTEGRITY REMINDER" in result
