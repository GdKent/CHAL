"""
Unit tests for project_for_component_embedding in beliefs/io.py.

Tests cover:
- Definition text extraction as "term: definition"
- Filtering retracted nodes from definitions, assumptions, evidence, claims
- Filtering resolved uncertainties (only active/open included)
- Grouping counterpositions by response_sufficiency
- Empty components (empty lists, zero counts, 0.0 avg strengths)
- Scalar counts and averages for a populated belief
- Thesis text concatenation with bullets
- Thesis with no bullets
"""

import pytest
from chal.beliefs.io import project_for_component_embedding
from tests.utils import create_sample_belief


# ==============================================
# Helpers
# ==============================================

def _make_populated_belief():
    """Build a belief dict with a rich set of components for testing."""
    return {
        "schema_version": "CBS",
        "belief_id": "BELIEF-COMP-TEST",
        "version": 1,
        "metadata": {"topic_query": "Test", "agent_persona": "Test"},
        "thesis": {
            "stance": "Climate change is anthropogenic",
            "summary_bullets": ["CO2 levels are rising", "Temperatures correlate with emissions"],
            "strength": 0.85,
        },
        "definitions": [
            {
                "id": "D1", "term": "anthropogenic",
                "definition": "Caused by human activity",
                "strength": 0.9, "status": "active",
                "strength_justification": "Standard scientific usage",
                "used_by": ["A1"],
            },
            {
                "id": "D2", "term": "greenhouse effect",
                "definition": "Warming due to atmospheric gases",
                "strength": 0.85, "status": "active",
                "strength_justification": "Well-established physics",
                "used_by": ["A1"],
            },
            {
                "id": "D3", "term": "obsolete term",
                "definition": "No longer used",
                "strength": 0.5, "status": "retracted",
                "strength_justification": "Superseded",
                "used_by": [],
            },
        ],
        "assumptions": [
            {
                "id": "A1", "type": "empirical",
                "statement": "CO2 measurements are accurate",
                "strength": 0.8, "status": "active",
                "strength_justification": "Calibrated instruments",
                "supported_by_definitions": ["D1"],
            },
            {
                "id": "A2", "type": "foundational",
                "statement": "Retracted assumption",
                "strength": 0.3, "status": "retracted",
                "strength_justification": "Withdrawn",
                "supported_by_definitions": [],
            },
        ],
        "evidence": [
            {
                "id": "E1", "type": "empirical",
                "summary": "Ice core data shows CO2 correlation",
                "source": "Petit et al. (1999)",
                "supports_claims": ["C1"],
                "strength": 0.9, "status": "active",
                "strength_justification": "Replicated",
                "supported_by_definitions": ["D1"],
            },
            {
                "id": "E2", "type": "empirical",
                "summary": "Retracted evidence",
                "source": "Dubious (2020)",
                "supports_claims": [],
                "strength": 0.2, "status": "retracted",
                "strength_justification": "Methodological flaws",
                "supported_by_definitions": [],
            },
        ],
        "claims": [
            {
                "id": "C1", "type": "deductive",
                "statement": "Human emissions drive warming",
                "depends_on": ["A1", "E1"],
                "strength": 0.85, "status": "active",
                "strength_justification": "Strong support",
                "inference_chain": [],
                "predictions": [],
            },
            {
                "id": "C2", "type": "inductive",
                "statement": "Retracted claim",
                "depends_on": [],
                "strength": 0.4, "status": "retracted",
                "strength_justification": "Withdrawn",
                "inference_chain": [],
                "predictions": [],
            },
        ],
        "uncertainties": [
            {
                "id": "U1", "targets": ["C1"],
                "question": "How sensitive is climate to CO2?",
                "status": "active", "importance": "high",
            },
            {
                "id": "U2", "targets": ["A1"],
                "question": "Are older CO2 records reliable?",
                "status": "resolved",
                "resolution_note": "Confirmed by multiple proxies",
                "importance": "medium",
            },
            {
                "id": "U3", "targets": ["C1"],
                "question": "What about aerosol cooling?",
                "status": "active", "importance": "medium",
            },
        ],
        "counterpositions": [
            {
                "id": "X1", "targets": ["C1"],
                "attack_type": "rebutting",
                "attack_strategy": "Solar activity",
                "statement": "Solar cycles explain warming",
                "my_response": "Solar output has been flat since 1980",
                "response_sufficiency": "sufficient",
            },
            {
                "id": "X2", "targets": ["A1"],
                "attack_type": "undermining",
                "attack_strategy": "Measurement error",
                "statement": "CO2 instruments may be biased",
                "my_response": "Partial calibration response",
                "response_sufficiency": "partial",
            },
            {
                "id": "X3", "targets": ["C1"],
                "attack_type": "undercutting",
                "attack_strategy": "Natural variability",
                "statement": "Natural variability is sufficient",
                "my_response": "",
                "response_sufficiency": "unaddressed",
            },
            {
                "id": "X4", "targets": ["C2"],
                "attack_type": "rebutting",
                "attack_strategy": "present_counter_evidence",
                "statement": "Counter to retracted claim",
                "my_response": "",
                "response_sufficiency": "moot",
            },
        ],
    }


# ==============================================
# 1. Definition text extraction
# ==============================================

@pytest.mark.unit
def test_definition_text_format():
    """Definitions are extracted as 'term: definition' strings."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    def_texts = [item["text"] for item in result["definitions"]]
    assert "anthropogenic: Caused by human activity" in def_texts
    assert "greenhouse effect: Warming due to atmospheric gases" in def_texts


# ==============================================
# 2. Retracted nodes are filtered out
# ==============================================

@pytest.mark.unit
def test_filters_retracted_definitions():
    """Retracted definitions are excluded from the result."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    def_texts = [item["text"] for item in result["definitions"]]
    # D3 (retracted) should not appear
    assert not any("obsolete term" in t for t in def_texts)
    assert len(result["definitions"]) == 2


@pytest.mark.unit
def test_filters_retracted_assumptions():
    """Retracted assumptions are excluded from the result."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    assumption_texts = [item["text"] for item in result["assumptions"]]
    assert not any("Retracted assumption" in t for t in assumption_texts)
    assert len(result["assumptions"]) == 1


@pytest.mark.unit
def test_filters_retracted_evidence():
    """Retracted evidence is excluded from the result."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    evidence_texts = [item["text"] for item in result["evidence"]]
    assert not any("Retracted evidence" in t for t in evidence_texts)
    assert len(result["evidence"]) == 1


@pytest.mark.unit
def test_filters_retracted_claims():
    """Retracted claims are excluded from the result."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    claim_texts = [item["text"] for item in result["claims"]]
    assert not any("Retracted claim" in t for t in claim_texts)
    assert len(result["claims"]) == 1


# ==============================================
# 3. Resolved uncertainties are filtered
# ==============================================

@pytest.mark.unit
def test_filters_resolved_uncertainties():
    """Only active (non-resolved) uncertainties are included."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    # U1 (active) and U3 (active) should be present; U2 (resolved) should not
    assert len(result["uncertainties"]) == 2
    assert "How sensitive is climate to CO2?" in result["uncertainties"]
    assert "What about aerosol cooling?" in result["uncertainties"]
    assert "Are older CO2 records reliable?" not in result["uncertainties"]


# ==============================================
# 4. Counterpositions grouped by response_sufficiency
# ==============================================

@pytest.mark.unit
def test_counterpositions_grouped_by_sufficiency():
    """Counterpositions are grouped into partial, sufficient, unaddressed lists."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    cp = result["counterpositions"]
    assert isinstance(cp, dict)
    assert set(cp.keys()) == {"partial", "sufficient", "unaddressed", "moot"}

    assert "Solar cycles explain warming" in cp["sufficient"]
    assert "CO2 instruments may be biased" in cp["partial"]
    assert "Natural variability is sufficient" in cp["unaddressed"]
    assert "Counter to retracted claim" in cp["moot"]

    assert len(cp["sufficient"]) == 1
    assert len(cp["partial"]) == 1
    assert len(cp["unaddressed"]) == 1
    assert len(cp["moot"]) == 1


# ==============================================
# 5. Empty components
# ==============================================

@pytest.mark.unit
def test_empty_components():
    """An empty belief returns empty lists, zero counts, and 0.0 avg strengths."""
    belief = {
        "schema_version": "CBS",
        "belief_id": "EMPTY",
        "version": 1,
        "metadata": {"topic_query": "Test", "agent_persona": "Test"},
        "thesis": {"stance": "", "summary_bullets": [], "strength": 0.0},
        "definitions": [],
        "assumptions": [],
        "evidence": [],
        "claims": [],
        "counterpositions": [],
        "uncertainties": [],
    }
    result = project_for_component_embedding(belief)

    assert result["definitions"] == []
    assert result["assumptions"] == []
    assert result["evidence"] == []
    assert result["claims"] == []
    assert result["uncertainties"] == []
    assert result["counterpositions"] == {"partial": [], "sufficient": [], "unaddressed": [], "moot": []}

    scalars = result["scalars"]
    assert scalars["n_definitions"] == 0
    assert scalars["n_assumptions"] == 0
    assert scalars["n_evidence"] == 0
    assert scalars["n_claims"] == 0
    assert scalars["n_counterpositions"] == 0
    assert scalars["n_uncertainties"] == 0
    assert scalars["avg_strength_definitions"] == 0.0
    assert scalars["avg_strength_assumptions"] == 0.0
    assert scalars["avg_strength_evidence"] == 0.0
    assert scalars["avg_strength_claims"] == 0.0
    assert scalars["thesis_strength"] == 0.0


# ==============================================
# 6. Scalar counts and averages for populated belief
# ==============================================

@pytest.mark.unit
def test_scalars_populated_belief():
    """Scalar counts and average strengths are correct for a populated belief."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)
    scalars = result["scalars"]

    # Counts (active nodes only for defs/assumptions/evidence/claims)
    assert scalars["n_definitions"] == 2       # D1, D2 (D3 retracted)
    assert scalars["n_assumptions"] == 1       # A1 (A2 retracted)
    assert scalars["n_evidence"] == 1          # E1 (E2 retracted)
    assert scalars["n_claims"] == 1            # C1 (C2 retracted)

    # Counterpositions and uncertainties count ALL (not filtered)
    assert scalars["n_counterpositions"] == 4  # X1, X2, X3, X4
    assert scalars["n_uncertainties"] == 3     # U1, U2, U3

    # Average strengths of active nodes
    assert scalars["avg_strength_definitions"] == pytest.approx((0.9 + 0.85) / 2)
    assert scalars["avg_strength_assumptions"] == pytest.approx(0.8)
    assert scalars["avg_strength_evidence"] == pytest.approx(0.9)
    assert scalars["avg_strength_claims"] == pytest.approx(0.85)

    # Thesis strength
    assert scalars["thesis_strength"] == pytest.approx(0.85)


# ==============================================
# 7. Thesis text concatenation with bullets
# ==============================================

@pytest.mark.unit
def test_thesis_text_with_bullets():
    """Thesis text is 'stance. bullet1. bullet2'."""
    belief = _make_populated_belief()
    result = project_for_component_embedding(belief)

    expected = "Climate change is anthropogenic. CO2 levels are rising. Temperatures correlate with emissions"
    assert result["thesis_text"] == expected


# ==============================================
# 8. Thesis with no bullets
# ==============================================

@pytest.mark.unit
def test_thesis_text_no_bullets():
    """Thesis with no bullets is just the stance string."""
    belief = {
        "schema_version": "CBS",
        "belief_id": "NO-BULLETS",
        "version": 1,
        "metadata": {"topic_query": "Test", "agent_persona": "Test"},
        "thesis": {"stance": "A simple stance", "summary_bullets": [], "strength": 0.5},
        "definitions": [],
        "assumptions": [],
        "evidence": [],
        "claims": [],
        "counterpositions": [],
        "uncertainties": [],
    }
    result = project_for_component_embedding(belief)

    assert result["thesis_text"] == "A simple stance"


# ==============================================
# load_belief_from_file tests
# ==============================================

class TestLoadBeliefFromFile:
    """Tests for load_belief_from_file()."""

    def test_load_valid_belief(self, tmp_path):
        """Loading a valid CBS belief JSON file returns the belief dict."""
        from chal.beliefs.io import load_belief_from_file
        from tests.utils import create_sample_belief
        import json

        belief = create_sample_belief()
        path = tmp_path / "belief.json"
        path.write_text(json.dumps(belief), encoding="utf-8")

        result = load_belief_from_file(path)
        assert isinstance(result, dict)
        assert result["belief_id"] == belief["belief_id"]
        assert result["schema_version"] == "CBS"

    def test_load_nonexistent_file(self):
        """Loading from a non-existent path raises FileNotFoundError."""
        from chal.beliefs.io import load_belief_from_file

        with pytest.raises(FileNotFoundError, match="not found"):
            load_belief_from_file("/nonexistent/path/belief.json")

    def test_load_invalid_json(self, tmp_path):
        """Loading a file with invalid JSON raises ValueError."""
        from chal.beliefs.io import load_belief_from_file

        path = tmp_path / "bad.json"
        path.write_text("{ this is not valid json }", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_belief_from_file(path)

    def test_load_non_object_json(self, tmp_path):
        """Loading a file with a JSON array raises ValueError."""
        from chal.beliefs.io import load_belief_from_file

        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            load_belief_from_file(path)

    def test_load_invalid_schema(self, tmp_path):
        """Loading valid JSON that fails CBS validation raises ValueError."""
        from chal.beliefs.io import load_belief_from_file
        import json

        invalid_belief = {"not": "a valid belief"}
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(invalid_belief), encoding="utf-8")

        with pytest.raises(ValueError, match="failed CBS schema validation"):
            load_belief_from_file(path)

    def test_load_string_path(self, tmp_path):
        """load_belief_from_file accepts a string path."""
        from chal.beliefs.io import load_belief_from_file
        from tests.utils import create_sample_belief
        import json

        belief = create_sample_belief()
        path = tmp_path / "belief.json"
        path.write_text(json.dumps(belief), encoding="utf-8")

        result = load_belief_from_file(str(path))
        assert result["belief_id"] == belief["belief_id"]
