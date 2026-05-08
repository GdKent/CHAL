"""
Phase 3 tests for Definition Nodes (D#) — Prompt Engineering.

Tests that Stage 1, Stage 5, Phase 1, and Phase 2 prompts correctly reference
D# nodes, supported_by_definitions, attack_strategy, scoping type, generation
order, ceiling rules, and orphan detection in compute_position_analysis().
"""

import json
import pytest
from chal.agents.prompts import (
    build_stage_1_belief_prompt_cbs,
    build_stage_5_belief_update_prompt_cbs,
    build_stage_5_phase1_enforcement_prompt,
    build_stage_5_phase2_introspection_prompt,
    compute_position_analysis,
)


# ========================================
# Helpers
# ========================================

def _stage1(topic="test topic", agent="A1", persona="Empiricist"):
    return build_stage_1_belief_prompt_cbs(topic, agent, persona)


def _stage5(pairs=None, belief=None, patches=""):
    if pairs is None:
        pairs = [{"challenger": "B", "challenge": "Q1", "rebuttal": "R1",
                  "resolution": {"status": "CRITIQUE_VALID", "reasoning": "test"}}]
    if belief is None:
        belief = '{"thesis": {"strength": 0.5}}'
    return build_stage_5_belief_update_prompt_cbs("A1", pairs, belief, patches)


def _phase1(pairs=None, belief=None, patches=""):
    if pairs is None:
        pairs = [{"challenger": "B", "challenge": "Q1", "rebuttal": "R1",
                  "resolution": {"status": "CRITIQUE_VALID", "reasoning": "test"}}]
    if belief is None:
        belief = '{"thesis": {"strength": 0.5}}'
    return build_stage_5_phase1_enforcement_prompt("A1", pairs, belief, patches)


def _phase2(belief_dict=None, summary="Phase 1 changes"):
    if belief_dict is None:
        belief_dict = {
            "claims": [
                {"id": "C1", "strength": 0.7, "status": "active", "depends_on": ["A1", "E1"]},
            ],
            "assumptions": [
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            "evidence": [
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            "definitions": [
                {"id": "D1", "term": "test", "definition": "test def",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
            ],
        }
    return build_stage_5_phase2_introspection_prompt(
        "A1", json.dumps(belief_dict), summary
    )


# ========================================
# 1. Stage 1: D# instructions present
# ========================================

class TestStage1Definitions:

    @pytest.mark.unit
    def test_stage1_definitions_section_present(self):
        """D# section is described in Stage 1 prompt."""
        prompt = _stage1()
        assert '"definitions" [D#]' in prompt

    @pytest.mark.unit
    def test_stage1_definition_fields_described(self):
        """All D# fields are described."""
        prompt = _stage1()
        for field in ("term", "definition", "strength", "strength_justification",
                       "status", "used_by"):
            assert field in prompt

    @pytest.mark.unit
    def test_stage1_supported_by_definitions_on_assumptions(self):
        """A# description includes supported_by_definitions."""
        prompt = _stage1()
        assert "supported_by_definitions" in prompt
        # Check it appears in the A# section context
        a_section_start = prompt.index('"assumptions" [A#]')
        a_section_end = prompt.index('"claims" [C#]')
        a_section = prompt[a_section_start:a_section_end]
        assert "supported_by_definitions" in a_section

    @pytest.mark.unit
    def test_stage1_supported_by_definitions_on_evidence(self):
        """E# description includes supported_by_definitions."""
        prompt = _stage1()
        e_section_start = prompt.index('"evidence" [E#]')
        e_section_end = prompt.index('"counterpositions" [X#]')
        e_section = prompt[e_section_start:e_section_end]
        assert "supported_by_definitions" in e_section

    @pytest.mark.unit
    def test_stage1_scoping_type_described(self):
        """Assumption type 'scoping' is described."""
        prompt = _stage1()
        assert '"scoping"' in prompt
        assert "boundary condition" in prompt.lower() or "scope of analysis" in prompt.lower()

    @pytest.mark.unit
    def test_stage1_definition_attack_strategies(self):
        """Definition-targeting strategies are described under undermining/undercutting."""
        prompt = _stage1()
        assert "over_extension" in prompt
        assert "circularity" in prompt

    @pytest.mark.unit
    def test_stage1_attack_strategy_required(self):
        """attack_strategy is described as required on X# nodes."""
        prompt = _stage1()
        assert "attack_strategy" in prompt

    @pytest.mark.unit
    def test_stage1_generation_order_d_after_e_before_c(self):
        """Generation order: A# → E# → D# → C#."""
        prompt = _stage1()
        # Find generation order section
        gen_start = prompt.index("<generation_order>")
        gen_end = prompt.index("</generation_order>")
        gen_section = prompt[gen_start:gen_end]
        # D# should come after E# and before C#
        assert gen_section.index("Evidence (E#)") < gen_section.index("Definitions (D#)")
        assert gen_section.index("Definitions (D#)") < gen_section.index("Claims (C#)")

    @pytest.mark.unit
    def test_stage1_definition_ceiling_rule(self):
        """DEFINITION CEILING rule is in dependency rules."""
        prompt = _stage1()
        assert "DEFINITION CEILING" in prompt

    @pytest.mark.unit
    def test_stage1_d_strength_guidance(self):
        """D# strength guidance: highest strengths, 0.7-1.0 range."""
        prompt = _stage1()
        assert "HIGHEST strengths" in prompt or "highest strengths" in prompt.lower()
        assert "0.7" in prompt

    @pytest.mark.unit
    def test_stage1_example_has_definitions(self):
        """Condensed example includes D# nodes."""
        prompt = _stage1()
        assert '"D1"' in prompt
        assert '"D2"' in prompt

    @pytest.mark.unit
    def test_stage1_example_has_supported_by_definitions(self):
        """Condensed example A#/E# have supported_by_definitions."""
        prompt = _stage1()
        # The example JSON should contain supported_by_definitions
        example_start = prompt.index("<example>")
        example_end = prompt.index("</example>")
        example = prompt[example_start:example_end]
        assert '"supported_by_definitions"' in example

    @pytest.mark.unit
    def test_stage1_example_has_attack_strategy(self):
        """Condensed example X# nodes have attack_strategy."""
        prompt = _stage1()
        example_start = prompt.index("<example>")
        example_end = prompt.index("</example>")
        example = prompt[example_start:example_end]
        assert '"attack_strategy"' in example

    @pytest.mark.unit
    def test_stage1_x_targets_include_d(self):
        """X# targets description includes D# IDs."""
        prompt = _stage1()
        # Check counterposition targets mention D#
        x_section_start = prompt.index('"counterpositions" [X#]')
        x_section_end = prompt.index('"uncertainties" [U#]')
        x_section = prompt[x_section_start:x_section_end]
        assert "D#" in x_section

    @pytest.mark.unit
    def test_stage1_u_targets_include_d(self):
        """U# targets description includes D# IDs."""
        prompt = _stage1()
        u_section_start = prompt.index('"uncertainties" [U#]')
        u_section_end = prompt.index("SYNTHESIZED LAST")
        u_section = prompt[u_section_start:u_section_end]
        assert "D#" in u_section


# ========================================
# 2. Stage 5: D# patch operations
# ========================================

class TestStage5Definitions:

    @pytest.mark.unit
    def test_stage5_add_definition_described(self):
        """add_definition operation is in Stage 5 prompt."""
        prompt = _stage5()
        assert "add_definition" in prompt

    @pytest.mark.unit
    def test_stage5_update_definition_described(self):
        """update_definition operation is in Stage 5 prompt."""
        prompt = _stage5()
        assert "update_definition" in prompt

    @pytest.mark.unit
    def test_stage5_definition_ceiling_rule(self):
        """DEFINITION CEILING rule is in Stage 5 mandatory rules."""
        prompt = _stage5()
        assert "DEFINITION CEILING" in prompt

    @pytest.mark.unit
    def test_stage5_definitional_critique_rule(self):
        """Definitional challenge handling is in mandatory rules."""
        prompt = _stage5()
        assert "definitional" in prompt.lower()

    @pytest.mark.unit
    def test_stage5_immutable_fields_mentioned(self):
        """Immutable fields (id, term) are mentioned."""
        prompt = _stage5()
        assert "Immutable" in prompt
        assert "id" in prompt
        assert "term" in prompt

    @pytest.mark.unit
    def test_stage5_attack_strategy_in_add_counterposition(self):
        """add_counterposition format includes attack_strategy."""
        prompt = _stage5()
        # Find add_counterposition in SUPPORTED OPERATIONS
        idx = prompt.index("add_counterposition")
        snippet = prompt[idx:idx+300]
        assert "attack_strategy" in snippet

    @pytest.mark.unit
    def test_stage5_supported_by_definitions_in_add_evidence(self):
        """add_evidence format includes supported_by_definitions."""
        prompt = _stage5()
        ops_start = prompt.index("SUPPORTED OPERATIONS")
        idx = prompt.index('"add_evidence"', ops_start)
        snippet = prompt[idx:idx+300]
        assert "supported_by_definitions" in snippet

    @pytest.mark.unit
    def test_stage5_example_has_update_definition(self):
        """Stage 5 example includes an update_definition patch."""
        prompt = _stage5()
        example_start = prompt.index("<example>")
        example_end = prompt.index("</example>")
        example = prompt[example_start:example_end]
        assert "update_definition" in example

    @pytest.mark.unit
    def test_stage5_self_check_includes_d(self):
        """Self-check mentions D# nodes."""
        prompt = _stage5()
        assert "D#" in prompt[prompt.index("Self-check"):]


# ========================================
# 3. Phase 1: D# awareness
# ========================================

class TestPhase1Definitions:

    @pytest.mark.unit
    def test_phase1_add_definition_described(self):
        """add_definition operation is in Phase 1 prompt."""
        prompt = _phase1()
        assert "add_definition" in prompt

    @pytest.mark.unit
    def test_phase1_update_definition_described(self):
        """update_definition operation is in Phase 1 prompt."""
        prompt = _phase1()
        assert "update_definition" in prompt

    @pytest.mark.unit
    def test_phase1_definition_ceiling_rule(self):
        """DEFINITION CEILING rule is in Phase 1 mandatory rules."""
        prompt = _phase1()
        assert "DEFINITION CEILING" in prompt

    @pytest.mark.unit
    def test_phase1_definitional_critique_rule(self):
        """Definitional challenge handling is in Phase 1 mandatory rules."""
        prompt = _phase1()
        assert "definitional" in prompt.lower()

    @pytest.mark.unit
    def test_phase1_attack_strategy_in_add_counterposition(self):
        """add_counterposition format includes attack_strategy."""
        prompt = _phase1()
        idx = prompt.index("add_counterposition")
        snippet = prompt[idx:idx+300]
        assert "attack_strategy" in snippet

    @pytest.mark.unit
    def test_phase1_supported_by_definitions_in_add_evidence(self):
        """add_evidence format includes supported_by_definitions."""
        prompt = _phase1()
        ops_start = prompt.index("SUPPORTED OPERATIONS")
        idx = prompt.index('"add_evidence"', ops_start)
        snippet = prompt[idx:idx+300]
        assert "supported_by_definitions" in snippet


# ========================================
# 4. Phase 2: D# in introspection
# ========================================

class TestPhase2Definitions:

    @pytest.mark.unit
    def test_phase2_add_definition_described(self):
        """add_definition operation is in Phase 2 prompt."""
        prompt = _phase2()
        assert "add_definition" in prompt

    @pytest.mark.unit
    def test_phase2_update_definition_described(self):
        """update_definition operation is in Phase 2 prompt."""
        prompt = _phase2()
        assert "update_definition" in prompt

    @pytest.mark.unit
    def test_phase2_definition_ceiling_in_dependency_graph(self):
        """DEPENDENCY GRAPH includes D# ceiling layer."""
        prompt = _phase2()
        dep_start = prompt.index("DEPENDENCY GRAPH")
        dep_end = prompt.index("A claim cannot be stronger")
        dep_section = prompt[dep_start:dep_end]
        assert "Definitions (D#)" in dep_section
        assert "ceiling" in dep_section.lower()
        assert "0.6" in dep_section
        assert "0.2" in dep_section

    @pytest.mark.unit
    def test_phase2_definition_ceiling_in_guardrails(self):
        """D# ceiling is in guardrails."""
        prompt = _phase2()
        guard_start = prompt.index("<guardrails>")
        guard_end = prompt.index("</guardrails>")
        guardrails = prompt[guard_start:guard_end]
        assert "DEFINITION CEILING" in guardrails

    @pytest.mark.unit
    def test_phase2_attack_strategy_in_add_counterposition(self):
        """add_counterposition format includes attack_strategy."""
        prompt = _phase2()
        idx = prompt.index("add_counterposition")
        snippet = prompt[idx:idx+300]
        assert "attack_strategy" in snippet

    @pytest.mark.unit
    def test_phase2_supported_by_definitions_in_add_evidence(self):
        """add_evidence format includes supported_by_definitions."""
        prompt = _phase2()
        ops_start = prompt.index("SUPPORTED OPERATIONS")
        idx = prompt.index('"add_evidence"', ops_start)
        snippet = prompt[idx:idx+300]
        assert "supported_by_definitions" in snippet

    @pytest.mark.unit
    def test_phase2_supported_by_definitions_in_add_assumption(self):
        """add_assumption format includes supported_by_definitions."""
        prompt = _phase2()
        ops_start = prompt.index("SUPPORTED OPERATIONS")
        idx = prompt.index('"add_assumption"', ops_start)
        snippet = prompt[idx:idx+300]
        assert "supported_by_definitions" in snippet

    @pytest.mark.unit
    def test_phase2_scoping_type_in_add_assumption(self):
        """add_assumption type list includes scoping."""
        prompt = _phase2()
        ops_start = prompt.index("SUPPORTED OPERATIONS")
        idx = prompt.index('"add_assumption"', ops_start)
        snippet = prompt[idx:idx+300]
        assert "scoping" in snippet

    @pytest.mark.unit
    def test_phase2_self_check_structural_gaps(self):
        """Self-check mentions STRUCTURAL GAPS."""
        prompt = _phase2()
        assert "STRUCTURAL GAPS" in prompt[prompt.index("Self-check"):]


# ========================================
# 5. compute_position_analysis — orphan detection
# ========================================

class TestPositionAnalysisOrphans:

    def _make_belief(self, **overrides):
        """Create a baseline belief with D# support for orphan testing."""
        belief = {
            "definitions": [
                {"id": "D1", "term": "t", "definition": "d",
                 "strength": 0.9, "status": "active", "used_by": ["A1", "E1"]},
            ],
            "assumptions": [
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            "evidence": [
                {"id": "E1", "strength": 0.75, "status": "active",
                 "supported_by_definitions": ["D1"]},
            ],
            "claims": [
                {"id": "C1", "strength": 0.7, "status": "active",
                 "depends_on": ["A1", "E1"]},
            ],
        }
        belief.update(overrides)
        return belief

    @pytest.mark.unit
    def test_no_orphans_no_structural_gaps(self):
        """Fully supported belief has no STRUCTURAL GAPS section."""
        belief = self._make_belief()
        result = compute_position_analysis(belief)
        assert "STRUCTURAL GAPS" not in result

    @pytest.mark.unit
    def test_orphaned_ae_detected(self):
        """A# with all D# retracted triggers STRUCTURAL GAPS."""
        belief = self._make_belief(
            definitions=[
                {"id": "D1", "term": "t", "definition": "d",
                 "strength": 0.0, "status": "retracted", "used_by": ["A1", "E1"]},
            ]
        )
        result = compute_position_analysis(belief)
        assert "STRUCTURAL GAPS" in result
        assert "A1" in result[result.index("STRUCTURAL GAPS"):]
        assert "capped at 0.6" in result

    @pytest.mark.unit
    def test_orphaned_ae_only_when_had_support(self):
        """A# with empty supported_by_definitions does NOT trigger orphan."""
        belief = self._make_belief(
            definitions=[],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": []},
            ],
        )
        result = compute_position_analysis(belief)
        # No A#/E# orphan because supported_defs is empty (never had support)
        if "STRUCTURAL GAPS" in result:
            gaps = result[result.index("STRUCTURAL GAPS"):]
            assert "A1 has NO active definitional" not in gaps

    @pytest.mark.unit
    def test_orphaned_claim_no_active_deps(self):
        """C# with all deps retracted triggers STRUCTURAL GAPS."""
        belief = self._make_belief(
            assumptions=[
                {"id": "A1", "strength": 0.0, "status": "retracted",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[
                {"id": "E1", "strength": 0.0, "status": "retracted",
                 "supported_by_definitions": ["D1"]},
            ],
        )
        result = compute_position_analysis(belief)
        assert "STRUCTURAL GAPS" in result
        assert "C1" in result[result.index("STRUCTURAL GAPS"):]
        assert "capped at 0.2" in result

    @pytest.mark.unit
    def test_orphaned_claim_empty_depends_on(self):
        """C# with empty depends_on triggers STRUCTURAL GAPS."""
        belief = self._make_belief(
            claims=[
                {"id": "C1", "strength": 0.7, "status": "active",
                 "depends_on": []},
            ],
        )
        result = compute_position_analysis(belief)
        assert "STRUCTURAL GAPS" in result
        assert "C1" in result[result.index("STRUCTURAL GAPS"):]
        assert "no depends_on" in result

    @pytest.mark.unit
    def test_retracted_ae_not_flagged(self):
        """Retracted A#/E# are not reported as orphans."""
        belief = self._make_belief(
            definitions=[
                {"id": "D1", "term": "t", "definition": "d",
                 "strength": 0.0, "status": "retracted", "used_by": ["A1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.0, "status": "retracted",
                 "supported_by_definitions": ["D1"]},
            ],
            evidence=[],
            claims=[
                {"id": "C1", "strength": 0.7, "status": "active",
                 "depends_on": ["A1"]},
            ],
        )
        result = compute_position_analysis(belief)
        # A1 is retracted, so it should not appear in orphan A#/E# section
        # But C1 should appear as orphaned claim since its only dep is retracted
        if "STRUCTURAL GAPS" in result:
            gaps = result[result.index("STRUCTURAL GAPS"):]
            # A1 should NOT be listed as orphaned A# (it's retracted)
            assert "A1 has NO active definitional" not in gaps
            # C1 SHOULD be listed as orphaned claim
            assert "C1" in gaps

    @pytest.mark.unit
    def test_mixed_active_retracted_defs(self):
        """A# with one active and one retracted D# is NOT orphaned."""
        belief = self._make_belief(
            definitions=[
                {"id": "D1", "term": "t", "definition": "d",
                 "strength": 0.0, "status": "retracted", "used_by": ["A1"]},
                {"id": "D2", "term": "t2", "definition": "d2",
                 "strength": 0.8, "status": "active", "used_by": ["A1", "E1"]},
            ],
            assumptions=[
                {"id": "A1", "strength": 0.8, "status": "active",
                 "supported_by_definitions": ["D1", "D2"]},
            ],
        )
        result = compute_position_analysis(belief)
        # A1 still has D2 active, so no orphan
        if "STRUCTURAL GAPS" in result:
            gaps = result[result.index("STRUCTURAL GAPS"):]
            assert "A1" not in gaps

    @pytest.mark.unit
    def test_position_analysis_still_has_standard_sections(self):
        """Orphan detection doesn't break standard sections."""
        belief = self._make_belief()
        result = compute_position_analysis(belief)
        assert "YOUR CURRENT POSITION" in result
        assert "SENSITIVITY AT YOUR POSITION" in result
        assert "SCENARIO PROJECTIONS" in result
        assert "LOWEST-STRENGTH DEPENDENCIES" in result
        assert "STRATEGIC RECOMMENDATION" in result
        assert "INTEGRITY REMINDER" in result
