# CBS (CHAL Belief Schema) — Comprehensive Reference

This document provides a complete specification of the CBS belief object: every field, its constraints, where it is defined, how it flows through the debate pipeline, and which downstream systems consume it.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Top-Level Structure](#2-top-level-structure)
3. [Identity & Versioning Fields](#3-identity--versioning-fields)
4. [Metadata Block](#4-metadata-block)
5. [Thesis Block](#5-thesis-block)
6. [Definitions (D#)](#6-definitions-d)
7. [Assumptions (A#)](#7-assumptions-a)
8. [Claims (C#)](#8-claims-c)
9. [Evidence (E#)](#9-evidence-e)
10. [Uncertainties (U#)](#10-uncertainties-u)
11. [Counterpositions (X#)](#11-counterpositions-x)
12. [Changelog](#12-changelog)
13. [Strength System](#13-strength-system)
14. [Belief Graph (DAG)](#14-belief-graph-dag)
15. [Graph Visualization](#15-graph-visualization)
16. [Patch Operations](#16-patch-operations)
17. [Validation Pipeline Summary](#17-validation-pipeline-summary)
18. [Validation Enums & Constants Reference](#18-validation-enums--constants-reference)
19. [Field Whitelists Reference](#19-field-whitelists-reference)
20. [CBS Parsing and I/O](#20-cbs-parsing-and-io)
21. [Embedding Projection Functions](#21-embedding-projection-functions)
22. [Pipeline Lifecycle](#22-pipeline-lifecycle)
23. [Component Interactions & Truth-Directed Design](#23-component-interactions--truth-directed-design)
24. [Unused & Write-Only Fields](#24-unused--write-only-fields)

---

## 1. Overview

The CBS (CHAL Belief Schema) is the canonical JSON structure that represents an agent's epistemic position during a CHAL debate. Every agent maintains one CBS object that evolves across debate rounds through a deterministic patch system.

**Schema version**: `"CBS"` (constant, defined as `SCHEMA_VERSION` in `schema.py`)

**Key design principles**:
- Every node has a unique prefixed ID: D# (definitions), A# (assumptions), C# (claims), E# (evidence), U# (uncertainties), X# (counterpositions)
- Strength values are calibrated on a [0.0, 1.0] scale
- Dependencies form a directed acyclic graph (DAG) — no circular references
- Updates are applied as atomic patch operations with full audit trail
- Thesis strength is always a deterministic formula result, never agent-chosen

**Source files**:
| File | Role |
|------|------|
| `beliefs/schema.py` | JSON Schema definition (`CBS_JSON_SCHEMA`) + `validate_belief()` with 16-phase validation |
| `beliefs/patches.py` | Patch application (`apply_patches()`), `validate_patches()` pre-flight validation, strength propagation, field whitelists, validation enums, `initialize_defense_tracking()` |
| `beliefs/belief_graph.py` | DAG construction (`BeliefGraph`) + structural validation (`validate_links()`) |
| `beliefs/io.py` | JSON parsing (`parse_model_output_to_belief()`), Markdown rendering (`belief_to_markdown()`), embedding projection (`project_for_embedding()`, `project_for_component_embedding()`) |
| `beliefs/graph_visualizer.py` | Interactive Cytoscape.js visualization |
| `agents/prompts.py` | Prompt templates that instruct agents to generate/update CBS |
| `orchestrator/debate_controller.py` | Orchestrates CBS creation, patching, defense boosts (`apply_defense_boosts()`), `filter_strength_increases()`, `cap_phase1_counterposition_sufficiency()`, and storage across stages |
| `utilities/utils.py` | Attack taxonomy constants (`VALID_ATTACK_STRATEGIES`), `EXCHANGE_SCORE_WEIGHTS`, `snapshot_belief()`, `finalize_agent_stats()`, Stage 2 validation |
| `constants.py` | Provider-to-environment-variable mapping (`PROVIDER_ENV_VARS`) |
| `config.py` | Configuration dataclasses including `DefenseBoostConfig` |

**Note**: `filter_strength_increases()` and `cap_phase1_counterposition_sufficiency()` are in `debate_controller.py`, NOT in `patches.py`.

---

## 2. Top-Level Structure

```json
{
  "schema_version": "CBS",
  "belief_id": "BELIEF-AgentA-001",
  "version": 1,
  "metadata": { ... },
  "definitions": [ ... ],
  "thesis": { ... },
  "assumptions": [ ... ],
  "claims": [ ... ],
  "evidence": [ ... ],
  "uncertainties": [ ... ],
  "counterpositions": [ ... ],
  "changelog": [ ... ]
}
```

**Required top-level keys** (validated by `validate_belief()`):
- `schema_version`, `belief_id`, `version`, `metadata`, `thesis`, `definitions`, `assumptions`, `claims`, `evidence`, `counterpositions`, `uncertainties`

**Optional top-level keys**:
- `changelog`

**Additional properties**: Allowed (`additionalProperties: true` in `CBS_JSON_SCHEMA`) for forward compatibility.

**Key ordering**: The CBS schema does NOT enforce key ordering — JSON objects are unordered by specification. The example order shown above is the conventional/recommended order.

---

## 3. Identity & Versioning Fields

| Field | Type | Required | Defined | Description |
|-------|------|----------|---------|-------------|
| `schema_version` | string | Yes | `CBS_JSON_SCHEMA` in `schema.py` | Must equal `"CBS"`. Fixed label for downstream tooling. |
| `belief_id` | string | Yes | `CBS_JSON_SCHEMA` in `schema.py` | Stable unique ID across updates. Format: `"BELIEF-{agent_name}-001"`. |
| `version` | integer | Yes | `CBS_JSON_SCHEMA` in `schema.py` | Monotone-increasing revision counter. Starts at 1, auto-incremented by `apply_patches()`. |

**Pipeline usage**:
- `belief_id`: Used as key in `BeliefGraph`, embedding tracker, reporting, and `debate_controller.py` (used in `agent_stats` and file naming via `sanitize_filename()`).
- `version`: Incremented by `apply_patches()` on every patch application (by 1, defaulting to 2 if no version exists). Displayed in changelog entries and Markdown rendering.
- `schema_version`: Validated by `validate_belief()`. Checked to equal `"CBS"` exactly.

### Node ID Sequencing

All node IDs within each collection must be **sequential integers starting from 1 with no gaps**. For example, if a belief has 3 definitions, their IDs must be `D1`, `D2`, `D3` — not `D1`, `D3`, `D5`. Starting from any number other than 1 (e.g., `D2` with no `D1`) is also rejected.

This applies to all six node types: D#, A#, C#, E#, U#, X#. Enforced by `_validate_sequential_ids()` in `schema.py`.

---

## 4. Metadata Block

Defined in `CBS_JSON_SCHEMA` in `schema.py`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic_query` | string | Yes | The debate question (e.g., "Does free will exist?"). |
| `agent_persona` | string | Yes | Persona label (e.g., "Empiricist", "Skeptic"). |

**Pipeline usage**:
- `topic_query` + `agent_persona`: Set during Stage 1 belief formation. Never modified after creation.

**Removed fields**: `last_updated` (written but never read), `scope_conditions` (display only, replaced by scoping assumptions), `definitions` (display only, replaced by top-level D# nodes).

---

## 5. Thesis Block

Defined in `CBS_JSON_SCHEMA` in `schema.py`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `stance` | string | Yes | — | The agent's core position statement, referencing supporting IDs parenthetically. |
| `summary_bullets` | array of strings | Yes | minItems: 1 (prompt asks for 3-10) | Key themes of the position. |
| `strength` | number | Yes | [0.0, 1.0] | **Always equals formula result** — not agent-chosen. See [Strength System](#13-strength-system). |
| `strength_reasoning` | string | No | — | Formula with actual numbers plugged in. Optional in agent-generated input; always present after patch application (auto-generated by `apply_patches()`). |

**Pipeline usage**:
- `stance`: Generated in Stage 1 (last, after all other components). Rewritten in Stage 5 Phase 2 introspection with updated ID references. Rendered in Markdown and analysis reports.
- `summary_bullets`: Same lifecycle as stance. Updated via `update_thesis` patch.
- `strength`: Computed deterministically by `apply_patches()`. Formula: `avg(active_claim_strengths) × (n^p / (n^p + 1))`. Compared before/after in `debate_controller.py` (thesis strength drift detection). Displayed in reports via `generate_analysis_report()`.
- `strength_reasoning`: Auto-generated by `apply_patches()`. Overwrites any agent-supplied value. Rendered in Markdown by `belief_to_markdown()`. Not consumed programmatically.

---

## 6. Definitions (D#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `D`.

Definitions are the semantic bedrock of a CBS belief. Each D# node defines a key term used by assumptions and evidence. D# strengths act as ceilings on the A#/E# nodes that depend on them.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `D\d+` | Unique identifier (e.g., "D1", "D2"). |
| `term` | string | Yes | Non-empty, **immutable** | The term being defined. Cannot be changed after creation. |
| `definition` | string | Yes | Non-empty | The definition text. Mutable (can be revised via `update_definition`). |
| `strength` | number | Yes | [0.0, 1.0] | Calibrated strength. Acts as ceiling for dependent A#/E# nodes. |
| `strength_justification` | string | Yes | — | Rationale for the strength number. |
| `status` | string | Yes | Enum: `active`, `revised`, `retracted` | Lifecycle status. Retracted definitions get strength 0.0. |
| `used_by` | array of strings | Yes | A#/E# IDs, minItems: 1 | Which A#/E# nodes depend on this definition. Bidirectionally consistent with `supported_by_definitions`. |
| `original_strength` | number | No (system-managed) | [0.0, 1.0] | Immutable snapshot of initial strength, set by `initialize_defense_tracking()`. Used as baseline for defense boost ceiling. |
| `consecutive_defenses` | integer | No (system-managed) | ≥ 0 | Count of consecutive REBUTTAL_VALID outcomes for this node. Reset to 0 on CRITIQUE_VALID. Drives defense boost formula. |

**Immutability**: `id` and `term` are immutable after creation. To redefine a term entirely, retract the old D# and create a new one.

**Pipeline usage**:
- **Graph**: D# nodes in the DAG. `used_by` creates "supports" edges from D# → A#/E# (built by `_build_graph()` in `belief_graph.py`).
- **D# ceiling enforcement** (pre-BFS step in `apply_patches()`): A#/E# strength cannot exceed the lowest non-retracted D# strength from their `supported_by_definitions`. If all supporting D# are retracted, A#/E# are capped at `ORPHAN_AE_CAP` (0.6).
- **Cross-examination (Stage 2)**: Opponents can target D# nodes using definitional attack strategies distributed across the existing attack types: `over_extension` and `under_extension` (under `undermining`), and `circularity`, `stipulative_bias`, `conceptual_conflation` (under `undercutting`).
- **Stage 3 rebuttal**: Agents defend challenged definitions (refute, concede via `update_definition`, or defer via U#).
- **Stage 4 adjudication**: Definitional challenges are evaluated for whether the definition exhibits the claimed flaw and downstream A#/E# impact.
- **Embedding projection (simple)**: Top 3 definitions by strength included in embedding text via `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` includes ALL non-retracted definitions as `{"text": "term: definition", "strength": float}` dicts. The text is formed by concatenating term and definition with ": ".
- **Patch operations**: `add_definition`, `update_definition`. Retracted definitions forced to strength 0.0.
- **Rendering**: Full display in Markdown via `belief_to_markdown()` showing term, definition, strength, used_by, and status.
- **Definition statistics**: At debate end, `_log_definition_statistics()` records D# count, active vs. retracted breakdown, average strength of active definitions, and most-challenged definitions.

**Bidirectional cross-reference validation** (in `validate_belief()` in `schema.py`):
- Every A#/E# ID in `D#.used_by` must list the D# ID in its `supported_by_definitions`.
- Every D# ID in `A#/E#.supported_by_definitions` must list the A#/E# ID in its `used_by`.
- Violations are reported as validation errors.

---

## 7. Assumptions (A#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `A`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `A\d+` | Unique identifier (e.g., "A1", "A2"). |
| `type` | string | Yes | Enum: `foundational`, `empirical`, `methodological`, `scoping` | Category of assumption. |
| `statement` | string | Yes | — | The assumption text. |
| `supports_claims` | array of strings | Yes | C# IDs | Which claims this assumption supports. |
| `strength` | number | Yes | [0.0, 1.0] | Calibrated strength value. Capped by D# ceiling. |
| `status` | string | Yes | Enum: `active`, `revised`, `retracted` | Lifecycle status. |
| `strength_justification` | string | Yes | — | Rationale for the strength number. |
| `supported_by_definitions` | array of strings | Yes | D# IDs, minItems: 1 | Which D# definitions ground this assumption. Bidirectionally consistent with `D#.used_by`. |
| `original_strength` | number | No (system-managed) | [0.0, 1.0] | Immutable snapshot of initial strength. Defense boost ceiling baseline. |
| `consecutive_defenses` | integer | No (system-managed) | ≥ 0 | Consecutive REBUTTAL_VALID count. Reset to 0 on CRITIQUE_VALID. |

**Type definitions** (from prompt instructions):
- `foundational`: Definitional/logical axioms — challenged only by showing incoherence.
- `empirical`: Assumed true based on evidence — challenged with counter-evidence.
- `methodological`: Adopted for analytical purposes — challenged by questioning the method.
- `scoping`: Assumptions that delimit the scope of inquiry (e.g., "We restrict 'free will' to the compatibilist sense"). Replaces the removed `metadata.scope_conditions` field.

**`supports_claims` note**: The `supports_claims` field is present in the schema but is NOT used for graph edge construction. Edges from A# → C# are derived from `claim.depends_on`, not `assumption.supports_claims`. See [Notable Asymmetry](#notable-asymmetry-supports_claims-vs-depends_on) in Section 24.

**Pipeline usage**:
- **Graph**: Nodes in the DAG. Edges from A# → C# are constructed via `claim.depends_on` (not `supports_claims`) in `_build_graph()` in `belief_graph.py`.
- **D# ceiling**: A# strength is capped by the lowest non-retracted D# strength from `supported_by_definitions` (enforced by D# ceiling step in `apply_patches()`).
- **Strength propagation**: Strength values participate in claim-limiting logic in `apply_patches()`. If an assumption's strength drops, all dependent claims are capped to the new minimum.
- **Cross-examination (Stage 2)**: Opponent agents target assumptions via `challenge_assumption` and `expose_weak_foundation` strategies.
- **Patch operations**: `update_assumption`, `add_assumption`. Retracted assumptions are forced to strength 0.0.
- **Rendering**: Full display in Markdown via `belief_to_markdown()` including `supported_by_definitions`.
- **Embedding projection (simple)**: Not included in `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` includes all non-retracted assumptions as `{"text": "statement", "strength": float}` dicts.

---

## 8. Claims (C#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `C`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `C\d+` | Unique identifier. |
| `type` | string | Yes | Free-form (e.g., "descriptive", "causal", "deductive") | Claim category. Unlike assumption and evidence types, claim types have NO enum constraint in the schema — they are truly free-form strings. |
| `statement` | string | Yes | — | The substantive assertion. |
| `depends_on` | array of strings | Yes | A#, E#, or C# IDs | Dependencies this claim builds on. |
| `strength` | number | Yes | [0.0, 1.0] | Must not exceed lowest active dependency strength. |
| `status` | string | Yes | Enum: `active`, `revised`, `retracted` | Lifecycle status. |
| `predictions` | array of objects | Yes | minItems: 1 | Falsifiable predictions (see sub-table). Each prediction must have `statement`, `test`, and `decision_criterion` fields (checked by `validate_belief()`). |
| `inference_chain` | array of objects | Yes | minItems: 3 | Structured reasoning steps (see [Inference Chain](#inference-chain) below). |
| `strength_justification` | string | Yes | — | Must name the limiting dependency. Format: `"<strength> — <rationale>; limited by <ID> (<lowest_value>)"`. |
| `original_strength` | number | No (system-managed) | [0.0, 1.0] | Immutable snapshot of initial strength. Defense boost ceiling baseline. |
| `consecutive_defenses` | integer | No (system-managed) | ≥ 0 | Consecutive REBUTTAL_VALID count. Reset to 0 on CRITIQUE_VALID. |

### Predictions (nested in Claims)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `statement` | string | Yes | The falsifiable prediction. |
| `test` | string | Yes | How the prediction can be tested. |
| `decision_criterion` | string | Yes | What outcome would confirm/deny. |
| `potential_falsifiers` | array of strings | No | Evidence that would disprove. |

### Inference Chain

A structured array of step objects (minItems: 3) showing the explicit reasoning from premises to conclusion. Each step has a `role` and `text`, plus role-specific fields. Defined in `CBS_JSON_SCHEMA` in `schema.py`, validated by `validate_inference_chain()` in `schema.py`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Enum: `premise`, `inference`, `conclusion` (validated via `VALID_IC_ROLES`). |
| `text` | string | Yes | What this step establishes or describes. |
| `reference` | string | Premise only | Must cite exactly one A#, E#, or C# ID (e.g., `"A1"`, `"E2"`). Validated via `_IC_REFERENCE_RE` (`^[ACE]\d+$`). Required for premise steps. |
| `inference_type` | string | Inference only | Enum: `deductive`, `inductive`, `abductive` (validated via `VALID_INFERENCE_TYPES`). Required for the inference step. |

**Structural rules**:
- At least one `premise` step (each must cite a dependency via `reference`).
- Exactly one `inference` step (with `inference_type`).
- Exactly one `conclusion` step (text should match the claim's `statement`).
- Ordering: all premises first, then inference, then conclusion.

**Pipeline usage**:
- **Graph**: Claims are the primary structural nodes. `depends_on` creates "supports" edges from dependencies → claim (built by `_build_graph()` in `belief_graph.py`). Active claims get a "supports" edge to THESIS.
- **Strength propagation**: Core subject of propagation. A claim's strength is capped by its lowest active/revised dependency (in `apply_patches()`). Retracted claims get strength 0.0 and are excluded from thesis calculation.
- **Thesis calculation**: Only active claims contribute to `avg(active_claim_strengths)` in the thesis formula.
- **Cross-examination (Stage 2)**: Primary targets for opponent challenges. Attack strategies target specific claims via `target_ids`.
- **Embedding projection (simple)**: Top 3 claims by strength are included in the embedding text via `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` includes ALL non-retracted claims as `{"text": "statement", "strength": float}` dicts.
- **Patch operations**: `update_claim`, `add_claim`. Retraction is via `update_claim` with `{"status": "retracted"}` (strength forced to 0.0 automatically).
- **Critical path analysis**: High-strength claims (>0.7) are identified as targets in `find_critical_paths()` in `belief_graph.py` for single-point-of-failure detection.

---

## 9. Evidence (E#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `E`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `E\d+` | Unique identifier. |
| `type` | string | Yes | Enum: `empirical`, `conceptual`, `expert_consensus` | Type of evidence. |
| `summary` | string | Yes | — | Brief description. |
| `source` | string | Yes | — | Citation or provenance. Can also be a dict — see source rendering note below. |
| `supports_claims` | array of strings | Yes | C# IDs | Which claims this evidence supports. |
| `strength` | number | Yes | [0.0, 1.0] | Calibrated evidence strength. Capped by D# ceiling. |
| `status` | string | Yes | Enum: `active`, `revised`, `retracted` | Lifecycle status. |
| `strength_justification` | string | Yes | — | Rationale for the strength number. |
| `supported_by_definitions` | array of strings | Yes | D# IDs, minItems: 1 | Which D# definitions ground this evidence. Bidirectionally consistent with `D#.used_by`. |
| `original_strength` | number | No (system-managed) | [0.0, 1.0] | Immutable snapshot of initial strength. Defense boost ceiling baseline. |
| `consecutive_defenses` | integer | No (system-managed) | ≥ 0 | Consecutive REBUTTAL_VALID count. Reset to 0 on CRITIQUE_VALID. |

**`source` field flexibility**: When `source` is a dict, `belief_to_markdown()` renders it as comma-separated `key: value` pairs on a single line (e.g., `"Author: Smith, Year: 2024"`). When it's a string, it's rendered directly.

**Pipeline usage**:
- **Graph**: Nodes in the DAG. No explicit edges are created from `supports_claims` in the graph builder — instead, evidence IDs appear in claim `depends_on`, which creates the edges.
- **D# ceiling**: E# strength is capped by the lowest non-retracted D# strength from `supported_by_definitions` (enforced by D# ceiling step in `apply_patches()`).
- **Strength propagation**: Evidence strength values participate in claim-limiting in `apply_patches()`. Weakened evidence can cascade through dependent claims.
- **Cross-examination (Stage 2)**: Targeted via `challenge_evidence` strategy.
- **Patch operations**: `add_evidence`, `update_evidence`. Retracted evidence forced to strength 0.0.
- **Rendering**: Full display in Markdown via `belief_to_markdown()` including `supported_by_definitions`. Summary included in graph visualization node labels.
- **Embedding projection (simple)**: Not mentioned in `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` includes all non-retracted evidence as `{"text": "summary", "strength": float}` dicts.

---

## 10. Uncertainties (U#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `U`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `U\d+` | Unique identifier. |
| `targets` | array of strings | Yes | A#, E#, C#, or D# IDs | What this uncertainty questions. |
| `question` | string | Yes | — | The open question. |
| `status` | string | Yes | Enum: `active`, `resolved` | Whether the uncertainty has been resolved. |
| `importance` | string | Yes | Enum: `high`, `medium`, `low` | How critical resolving this uncertainty is. |
| `resolution_note` | string | No | — | How the uncertainty was resolved. While not required by the JSON schema itself, the `resolve_uncertainty` patch operation requires a non-empty `resolution_note` — so it IS effectively required whenever status transitions to "resolved". |

**Pipeline usage**:
- **Graph**: "questions" edges from U# → targets (built by `_build_graph()` in `belief_graph.py`). These edges are informational — they don't affect strength propagation.
- **Cross-examination (Stage 2)**: Opponents use `press_uncertainty` strategy, prioritizing `high` and `medium` importance items.
- **Stage 5 (Belief Update)**: Agents review active uncertainties. High/medium importance items are prioritized for resolution. Resolved by adding supporting evidence/claims.
- **Patch operations**: `add_uncertainty`, `resolve_uncertainty` (sets status="resolved" + resolution_note in `apply_patches()`).
- **Rendering**: Full display in Markdown via `belief_to_markdown()`.
- **Embedding projection (simple)**: First uncertainty included in `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` includes ALL non-resolved uncertainties (question text only, no strength weighting).

---

## 11. Counterpositions (X#)

Defined in `CBS_JSON_SCHEMA` in `schema.py`. ID prefix: `X`.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Pattern: `X\d+` | Unique identifier. |
| `targets` | array of strings | Yes | C#, A#, E#, or D# IDs | What this counterposition attacks. |
| `attack_type` | string | Yes | Enum: `undermining`, `rebutting`, `undercutting` | Type of attack. |
| `attack_strategy` | string | Yes | Must be valid for the `attack_type` | Specific sub-strategy (see table below). Validated against `VALID_ATTACK_STRATEGIES` in both `schema.py` (via `validate_belief()`) and `patches.py` (via `validate_patches()` for `add_counterposition` operations). |
| `statement` | string | Yes | — | The objection/attack text. |
| `my_response` | string | Yes | — | The agent's prepared defense response. Always a required field (must be present), but its non-empty content validation is relaxed when `response_sufficiency` is "unaddressed" or "moot" — in those cases, an empty string is accepted. |
| `response_sufficiency` | string | Yes | Enum: `sufficient`, `partial`, `unaddressed`, `moot` | How well the response handles the objection. |

**Attack type definitions**:
- `undermining`: Challenges a premise, assumption, or definition directly. Includes definitional attacks like `over_extension` and `under_extension`.
- `rebutting`: Presents counter-evidence or counter-conclusion.
- `undercutting`: Challenges the inference step connecting premise to conclusion. Includes definitional attacks like `circularity`, `stipulative_bias`, and `conceptual_conflation`.

**Note on definitional attacks**: There is no separate `definitional` attack type. Attacks on D# nodes are distributed across the existing attack types as specific strategies. A successful definitional attack weakens all A#/E# that depend on the targeted definition via the D# ceiling mechanism.

**Attack strategies by type** (validated in `utils.py:VALID_ATTACK_STRATEGIES`):

| attack_type | Valid attack_strategy values |
|---|---|
| `undermining` | `challenge_evidence`, `challenge_assumption`, `expose_weak_foundation`, `demand_falsifiability`, `challenge_strength_calibration`, `press_uncertainty`, `over_extension` (D#), `under_extension` (D#), `challenge_moral_implications` (ethical), `expose_stakeholder_harm` (ethical) |
| `rebutting` | `present_counter_evidence`, `present_counter_example`, `exploit_counterposition`, `offer_alternative_explanation`, `present_ethical_counter` (ethical), `invoke_competing_obligation` (ethical) |
| `undercutting` | `challenge_inference_step`, `identify_circularity`, `expose_inconsistency`, `identify_equivocation`, `challenge_scope`, `circularity` (D#), `stipulative_bias` (D#), `conceptual_conflation` (D#), `challenge_normative_inference` (ethical), `expose_value_conflict` (ethical), `challenge_moral_relevance` (ethical) |

**Response sufficiency definitions**:
- `sufficient`: The response fully addresses the objection.
- `partial`: The response addresses some aspects but leaves gaps.
- `unaddressed`: No adequate response yet — a known vulnerability.
- `moot`: The targeted node has been retracted, making the counterposition irrelevant. **Terminal state** — once set to "moot", `response_sufficiency` cannot be changed to any other value (enforced by `validate_patches()`).

**Pipeline usage**:
- **Graph**: "challenges" edges from X# → targets (built by `_build_graph()` in `belief_graph.py`). These edges are informational — they don't affect strength propagation.
- **Cross-examination (Stage 2)**: Opponents exploit counterpositions rated "partial" or "unaddressed" as weaknesses. Guides opponent question strategy and attack selection.
- **Adjudication (Stage 4)**: Adjudicator checks whether challenges map to existing counterpositions.
- **Stage 5 Phase 1**: New counterpositions added via `add_counterposition` have their `response_sufficiency` automatically capped from "sufficient" to "partial" by `cap_phase1_counterposition_sufficiency()` in `debate_controller.py`. This forces agents to refine their responses in Phase 2 or later rounds before they can claim full sufficiency. Note: "moot" is intentionally NOT capped (it correctly indicates a retracted target).
- **Stage 5 Phase 2**: Unaddressed counterpositions must be resolved — either weaken the target or upgrade `response_sufficiency`.
- **Patch operations**: `add_counterposition`, `update_counterposition` (update `my_response` and `response_sufficiency` in `apply_patches()`).
- **Rendering**: Full display in Markdown via `belief_to_markdown()`.
- **Embedding projection (simple)**: Top 2 included in `project_for_embedding()`.
- **Embedding projection (component-wise)**: `project_for_component_embedding()` groups counterpositions by `response_sufficiency` into 4 buckets (partial, sufficient, unaddressed, moot) with statement text extracted for each. Missing sufficiency defaults to "unaddressed".
- **Prompt minimum**: Agents are instructed to include at least 2 counterpositions in Stage 1.

---

## 12. Changelog

Defined in `CBS_JSON_SCHEMA` in `schema.py`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | integer | Yes | The belief version this entry corresponds to. |
| `changes` | array of strings | Yes | Human-readable list of changes made. |

**Pipeline usage**:
- **Auto-generated**: `apply_patches()` appends a changelog entry on every patch application. Changes are tracked automatically from each patch operation.
- **Debate controller**: Changelog is read and passed as context in belief update prompts, giving agents awareness of their belief version history.
- **Rendering**: Full display in Markdown via `belief_to_markdown()`.

**Changelog content detail** — types of changes auto-logged:
- Thesis strength changes: `"Thesis strength: 0.50 → 0.65"`
- Node strength changes: `"C3.strength: 0.7 → 0.6"`
- Inference chain updates: `"C3.inference_chain updated (2 premises, deductive → inductive)"` (via `_summarise_ic_diff()`)
- Propagation cascades: `"Propagated: C4 strength → 0.60 (limited by C3)"`
- Orphan caps: `"Capped: C5 strength → 0.2 (no active dependencies — unfounded claim)"`
- New nodes: `"Added definition D1: 'epistemology'"`
- Status changes: `"E2.status: active → retracted"`

**Removed fields**: `timestamp` (written but never used in logic).

---

## 13. Strength System

### Scale

All strength fields (thesis, definitions, claims, assumptions, evidence) use the same calibrated scale:

| Range | Label | Meaning |
|-------|-------|---------|
| 0.0 | Vacuous | No credible support; should be retracted. |
| 0.0–0.3 | Weak | Indefensible; requires substantial strengthening. |
| 0.3–0.5 | Contested | More reasons to doubt than believe. |
| 0.5 | Threshold | Evenly balanced — could go either way. |
| 0.5–0.7 | Moderate | More reasons to believe than doubt; some gaps remain. |
| 0.7–0.9 | Strong | Well-supported; minor open questions. |
| 0.9–1.0 | Robust | Near-certain given available evidence/reasoning. |
| 1.0 | Definitive | Established beyond reasonable dispute. |

**Calibration guidance**: 0.9 means ~10% chance of being wrong. Most philosophical claims warrant 0.4–0.8.

### Thesis Strength Formula

The thesis strength `T` is always computed deterministically — never agent-chosen. Implemented in the thesis recalculation step of `apply_patches()`.

**Formula**:

```
T = s̄ × B(n)
```

Where:
- `s̄` = mean strength of active (non-retracted) claims = `(1/n) × Σᵢ sᵢ` for `i = 1..n`
- `sᵢ` = strength of the i-th active claim
- `n` = number of active (non-retracted) claims
- `B(n)` = breadth multiplier = `nᵖ / (nᵖ + 1)`
- `p` = breadth sensitivity exponent (default: `BREADTH_SENSITIVITY = 1.0`, defined in `patches.py`)

**Expanded form**:

```
T = (1/n × Σᵢ sᵢ) × (nᵖ / (nᵖ + 1))
```

**Breadth multiplier behavior** (at default `p = 1.0`):

| Active claims (n) | B(n) = nᵖ/(nᵖ+1) | Effect |
|---|---|---|
| 1 | 0.50 | Thesis halved — single claim is a fragile position |
| 2 | 0.67 | Significant improvement |
| 3 | 0.75 | Diminishing returns begin |
| 4 | 0.80 | |
| 5 | 0.83 | |
| 7 | 0.88 | Approaching asymptote |
| 10 | 0.91 | Near-saturation |

The breadth multiplier asymptotically approaches 1.0 but never reaches it. This means an argument resting on a single claim (no matter how strong) can never have thesis strength above 0.50. Broader positions with more well-supported claims are mechanically rewarded.

**Worked example**: An agent has 3 active claims with strengths 0.70, 0.55, 0.65:
```
s̄ = (0.70 + 0.55 + 0.65) / 3 = 0.633
B(3) = 3¹ / (3¹ + 1) = 3/4 = 0.75
T = 0.633 × 0.75 = 0.47
```

The result `0.47` overwrites any agent-supplied thesis strength. The `strength_reasoning` field is auto-generated showing this calculation.

### Partial Derivatives of Thesis Strength

The thesis strength function `T(s̄, n)` has two inputs an agent can influence: the average claim strength `s̄` and the number of active claims `n`. The partial derivatives tell the agent which lever is more effective at their current position. Computed in `compute_position_analysis()` in `prompts.py` and injected into the Stage 5 Phase 2 prompt.

**Partial derivative with respect to average strength** (how much thesis improves per unit increase in average claim strength):

```
∂T/∂s̄ = nᵖ / (nᵖ + 1) = B(n)
```

This is simply the breadth multiplier itself. With 3 claims at `p = 1.0`, `∂T/∂s̄ = 0.75` — each +0.10 increase in average claim strength yields +0.075 thesis increase.

**Partial derivative with respect to claim count** (marginal thesis gain from adding one more claim at current average):

```
∂T/∂n = s̄ × p × n⁽ᵖ⁻¹⁾ / (nᵖ + 1)²
```

With 3 claims, avg strength 0.633, and `p = 1.0`:
```
∂T/∂n = 0.633 × 1.0 × 3⁰ / (3 + 1)² = 0.633 / 16 = 0.040
```

This means adding one claim at the current average yields approximately +0.040 thesis increase.

### How Partial Derivatives Guide Belief Refinement

The `compute_position_analysis()` function in `prompts.py` uses these derivatives to generate a dynamic `<position_analysis>` block that is injected into the Stage 5 Phase 2 prompt. This block tells the agent:

1. **Current position**: Active claim count, average strength, breadth multiplier, thesis strength.

2. **Sensitivity at current position**: The exact values of `∂T/∂s̄` and `∂T/∂n`, showing which lever has more impact.

3. **Scenario projections** — four concrete what-if calculations:
   - **Raise avg by 0.10**: `T₁ = min(s̄ + 0.10, 1.0) × B(n)` — gain `ΔT₁ = T₁ - T`
   - **Add claim at current avg**: `T₂ = s̄ × B(n+1)` — gain `ΔT₂ = T₂ - T`
   - **Add strong claim** (0.15 above avg): `s̄₃ = (s̄ × n + min(s̄ + 0.15, 1.0)) / (n+1)`, then `T₃ = s̄₃ × B(n+1)` — gain `ΔT₃ = T₃ - T`
   - **Retract weakest claim** (if n ≥ 2): Remove lowest-strength claim, recompute with remaining — gain `ΔT₄ = T₄ - T`

4. **Strategic recommendation** — based on comparing the two levers:
   - If `∂T/∂s̄ × 0.10 > ∂T/∂n × 1.5`: "Raising average strength is your strongest lever"
   - If `∂T/∂n > ∂T/∂s̄ × 0.10 × 1.5`: "Adding more claims is your strongest lever"
   - Otherwise: "Both levers are roughly equally valuable"

5. **Lowest-strength dependencies**: Identifies the bottom 3 weakest A#/E# nodes and which claims they limit, with specific suggested patch operations.

6. **Definitional vulnerabilities**: Identifies weak D# nodes supporting many dependents and A#/E# bottlenecks with only a single D# support.

7. **Structural gaps**: Detects orphaned A#/E# (no active D# support, capped at 0.6) and orphaned C# (no active dependencies, capped at 0.2).

8. **Integrity reminder**: Explicitly states that the mathematical analysis does not authorize inflating strength values — every strength must reflect genuine epistemic assessment.

This analysis gives the LLM a quantitative understanding of its position's structure, so it can make targeted improvements rather than guessing.

### Dependency Constraint Equations

Strength propagation enforces a chain of upper-bound constraints from definitions through to the thesis. All constraints are applied automatically by `apply_patches()` after any patch operation.

#### Equation 1: D# Ceiling on A#/E#

Implemented in the D# ceiling enforcement step of `apply_patches()`. For each non-retracted A# or E# node:

```
strength(A#) ≤ min{ strength(Dⱼ) : Dⱼ ∈ supported_by_definitions(A#), status(Dⱼ) ≠ "retracted" }
```

```
strength(E#) ≤ min{ strength(Dⱼ) : Dⱼ ∈ supported_by_definitions(E#), status(Dⱼ) ≠ "retracted" }
```

If the node's strength exceeds this ceiling, it is capped to the ceiling value. If **all** supporting D# are retracted (the set is empty), the orphan cap applies instead:

```
If all Dⱼ retracted:  strength(A#) ≤ ORPHAN_AE_CAP = 0.6
                       strength(E#) ≤ ORPHAN_AE_CAP = 0.6
```

**Example**: A1 has `supported_by_definitions: ["D1", "D2"]`. D1 has strength 0.85, D2 has strength 0.70. The ceiling for A1 is `min(0.85, 0.70) = 0.70`. If A1's strength is 0.80, it gets capped to 0.70.

#### Equation 2: Dependency Floor on C#

Implemented in the BFS dependency propagation step of `apply_patches()`. BFS-based, level-by-level. For each non-retracted claim:

```
strength(C#) ≤ min{ strength(Xₖ) : Xₖ ∈ depends_on(C#), status(Xₖ) ≠ "retracted" }
```

Where `Xₖ` can be A#, E#, or other C# nodes. Retracted dependencies are excluded from the minimum — they don't drag down dependent claims.

If **all** dependencies are retracted (the set is empty), the orphan cap applies:

```
If all Xₖ retracted:  strength(C#) ≤ ORPHAN_CLAIM_CAP = 0.2
```

**Note**: This is strict. An unfounded claim with no epistemic backing is capped at 0.2 (weak/contested range).

**Example**: C1 depends on A1 (0.70), A2 (0.85), E1 (0.60). The minimum is `min(0.70, 0.85, 0.60) = 0.60`. If C1's strength is 0.75, it gets capped to 0.60. The `strength_justification` must name E1 as the limiting dependency.

**BFS worklist seeding**: The initial worklist for BFS propagation is seeded from `strength_changes` — a dict tracking all nodes whose strengths changed during patch application. This includes both direct patch changes and D# ceiling enforcement changes.

#### Equation 3: Thesis from Claims

```
T = (1/n × Σᵢ sᵢ) × (nᵖ / (nᵖ + 1))
```

Where only active (non-retracted) claims contribute. See [Thesis Strength Formula](#thesis-strength-formula) above.

#### Full Propagation Chain

When any node's strength changes (via patch or enforcement), the system applies all constraints in order:

```
Step 1: D# ceiling enforcement
          For each non-retracted A#/E#:
            strength(A#/E#) ← min(strength(A#/E#), ceiling from D#)

Step 2: BFS dependency propagation
          For each claim in BFS order from changed nodes:
            strength(C#) ← min(strength(C#), floor from depends_on)

Step 3: Orphan detection
          A#/E# with all D# retracted → capped at 0.6
          C# with all dependencies retracted → capped at 0.2

Step 4: Thesis recalculation
          T = s̄ × B(n) from active claims only
```

This means a single D# strength change can cascade: `D# → A#/E# (ceiling) → C# (dependency floor) → Thesis (formula)`.

**Strength propagation exception handling**: If the propagation pipeline encounters an exception, it logs a warning (`"Warning: Strength propagation failed: {e}"`) but does NOT fail the entire patch operation. The patches that succeeded before the error remain applied.

**`_summarise_ic_diff()` helper**: Inference chain changes during strength propagation changelog generation are summarized by `_summarise_ic_diff()` in `patches.py`, which produces human-readable strings like `"2 premises → 3 premises, deductive → inductive"`.

### Retraction Enforcement

When any node's status is set to `"retracted"`, its strength is forced to 0.0 regardless of any other value. Implemented per node type in `apply_patches()`:

```
If status(node) = "retracted":  strength(node) = 0.0
```

Retracted nodes are then excluded from all constraint calculations:
- Retracted D# are excluded from the D# ceiling minimum
- Retracted A#/E#/C# are excluded from the dependency floor minimum
- Retracted C# are excluded from the thesis strength average

### Defense Boost System

Implemented in `apply_defense_boosts()` in `debate_controller.py`. Configured via `DefenseBoostConfig` in `config.py`.

When a node survives a challenge (REBUTTAL_VALID adjudication verdict), the system automatically applies a formula-driven strength increase. This is the **only** mechanism by which existing node strengths can increase — agents cannot unilaterally raise strengths in Phase 2 (see [Phase 2 Strength Filter](#phase-2-strength-filter)).

#### Defense Boost Formula

The boost is a **flat constant per defense** — streak length only determines whether a boost occurs (count ≥ 1), not the boost amount.

**Per-defense boost amount**:

```
boost = flat_boost  (constant per defense, regardless of streak length)
```

Every successful defense adds exactly `flat_boost` (default: 0.02) to the node's strength. The `compute_defense_boost()` function in `debate_controller.py` returns `flat_boost` regardless of `consecutive_defenses` count (as long as count ≥ 1).

**New strength after boost**:

```
strength_new = min(strength_current + flat_boost, strength_original + c_max, 1.0)
```

Where:
- `strength_original` = `original_strength` field — immutable snapshot of the node's initial strength from Stage 1
- `c_max` = `max_cumulative_boost` (default: 0.15) — maximum total boost above original strength

The three-way `min()` ensures:
1. The boost amount is applied (`strength_current + flat_boost`)
2. Total cumulative boosts never exceed `c_max` above the original strength (`strength_original + c_max`)
3. Strength never exceeds 1.0

**Default parameters** (`DefenseBoostConfig`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `flat_boost` | b | 0.02 | Constant boost per successful defense |
| `max_cumulative_boost` | c_max | 0.15 | Max total boost above `original_strength` |

**Worked example**: A node with `original_strength = 0.65`, `current_strength = 0.68`, and `consecutive_defenses = 2` (about to become 3):

```
k = 3
boost = flat_boost = 0.02
ceiling  = min(0.65 + 0.15, 1.0) = 0.80
strength_new = min(0.68 + 0.02, 0.80) = min(0.70, 0.80) = 0.70
```

**Behavior by adjudication verdict**:
- **REBUTTAL_VALID**: Increment `consecutive_defenses` on each targeted node. Apply boost formula. Node strength increases up to the ceiling.
- **CRITIQUE_VALID**: Reset `consecutive_defenses` to 0 on all targeted nodes (streak broken). No strength increase.
- **UNRESOLVED**: No effect on defense tracking.

**`consecutive_defenses` persistence**: The `consecutive_defenses` counter is NOT reset between rounds — it tracks the full defense streak across the entire debate. Only a CRITIQUE_VALID verdict resets it to 0.

**Timing**: Defense boosts are applied mechanically by the system between Phase 1 and Phase 2 of Stage 5 (`apply_defense_boosts()` in `debate_controller.py`). Agents do NOT manually increase strengths for successful defenses.

**Required tracking fields**: `original_strength` and `consecutive_defenses` on all D#, A#, E#, and C# nodes. Initialized by `initialize_defense_tracking()` in `patches.py` after Stage 1 belief formation.

### Phase 2 Strength Filter

Defined in `filter_strength_increases()` in `debate_controller.py`. Phase 2 is unilateral — the agent has no opponent scrutiny — so existing node strengths can only stay the same or go down, never up. This prevents "trust me bro" self-strengthening.

The function is called by the debate controller between model output parsing and patch application during Phase 2. It operates on the raw patch list, not within `apply_patches()`.

```
For update_* patches on existing nodes:
  If patch.strength > current_strength(node):  strip strength from patch
```

The rest of the patch (semantic changes, status changes) is preserved. New nodes added via `add_*` operations are NOT affected — they can have any strength.

This creates an asymmetry by design:
- **Strength can only increase** through the defense boost system (mechanical, post-adjudication, adversarially tested)
- **Strength can only decrease** through agent patches (voluntary concession or enforcement) or automatic propagation (D# ceiling, dependency floor)

### Equations Summary

For quick reference, all strength-governing equations in one place:

```
THESIS STRENGTH
  T = s̄ × B(n)
  s̄ = (1/n) × Σᵢ sᵢ                    (avg of active claim strengths)
  B(n) = nᵖ / (nᵖ + 1)                  (breadth multiplier, p = 1.0)
  Result: round(T, 4)                    (prevents IEEE 754 floating-point noise)

PARTIAL DERIVATIVES
  ∂T/∂s̄ = B(n)                           (gain per unit avg strength increase)
  ∂T/∂n = s̄ × p × n⁽ᵖ⁻¹⁾ / (nᵖ + 1)²   (marginal gain from one more claim)

D# CEILING
  strength(A#/E#) ≤ min{ strength(Dⱼ) : Dⱼ active in supported_by_definitions }
  If all Dⱼ retracted: strength(A#/E#) ≤ 0.6
  Result: round(ceiling, 4)

DEPENDENCY FLOOR
  strength(C#) ≤ min{ strength(Xₖ) : Xₖ active in depends_on }
  If all Xₖ retracted: strength(C#) ≤ 0.2

RETRACTION
  If status = "retracted": strength = 0.0

DEFENSE BOOST
  boost = flat_boost                     (constant 0.02 per defense)
  strength_new = min(strength_current + flat_boost, strength_original + c_max, 1.0)
  Result: round(strength_new, 4)

PHASE 2 FILTER
  update_* on existing node: strength can only decrease or stay the same
  add_* on new node: no restriction
```

---

## 14. Belief Graph (DAG)

Implemented in `beliefs/belief_graph.py`. The `BeliefGraph` class transforms a CBS object into a directed acyclic graph.

### Node Types

| Type | Source | ID |
|------|--------|----|
| thesis | `belief["thesis"]` | `"THESIS"` |
| definition | `belief["definitions"]` | `D1`, `D2`, ... |
| assumption | `belief["assumptions"]` | `A1`, `A2`, ... |
| claim | `belief["claims"]` | `C1`, `C2`, ... |
| evidence | `belief["evidence"]` | `E1`, `E2`, ... |
| counterposition | `belief["counterpositions"]` | `X1`, `X2`, ... |
| uncertainty | `belief["uncertainties"]` | `U1`, `U2`, ... |

### Edge Types

| Edge Type | From → To | Source Field |
|-----------|-----------|--------------|
| `supports` | D# → A#/E# | `definition.used_by` |
| `supports` | A#/E#/C# → C# | `claim.depends_on` |
| `supports` | C# (active) → THESIS | Implicit (all active claims) |
| `challenges` | X# → A#/E#/C#/D# | `counterposition.targets` |
| `questions` | U# → A#/E#/C#/D# | `uncertainty.targets` |

### Graph Construction

The `_build_graph()` method constructs nodes and edges from the belief dict. Key construction rules:
- All nodes from all 7 collections are added regardless of status
- Only **active** (non-retracted) claims get a "supports" edge to THESIS
- All other edges (supports, challenges, questions) are constructed regardless of status
- Edge data includes the `edge_type` label for downstream analysis

### Structural Diagram

```
D# ──supports────→ A#/E#
X# ──challenges──→ A#/E#/C#/D#
U# ──questions───→ A#/E#/C#/D#
A# ──supports────→ C#
E# ──supports────→ C#
C# ──supports────→ THESIS
```

### Validation Checks (`validate_links()`)

- All edge endpoints reference existing nodes (BLOCKING)
- No circular dependencies (BLOCKING) — detected via DFS in `_has_cycle()`
- No orphaned claims — every claim must have at least one incoming "supports" edge (BLOCKING). `_find_orphaned_claims()` only counts "supports" edges as valid support — "challenges" and "questions" edges do NOT count. The THESIS node is excluded from orphan detection.

### Edge-Type Constraints (Cross-Reference Validation)

Each cross-reference field is restricted to specific ID prefixes. Enforced by `_validate_ref_prefixes()` in `schema.py` (for complete beliefs) and by `validate_patches()` in `patches.py` (for add operations). The authoritative mapping is defined in `ALLOWED_REF_PREFIXES` in `schema.py`.

| Field | Node Type | Allowed ID Prefixes | Rationale |
|---|---|---|---|
| `claim.depends_on` | C# | A#, E#, C# | Claims build on assumptions, evidence, and other claims |
| `definition.used_by` | D# | A#, E# | Definitions ground assumptions and evidence |
| `assumption.supports_claims` | A# | C# | Assumptions support claims |
| `assumption.supported_by_definitions` | A# | D# | Assumptions are grounded by definitions |
| `evidence.supports_claims` | E# | C# | Evidence supports claims |
| `evidence.supported_by_definitions` | E# | D# | Evidence is grounded by definitions |
| `counterposition.targets` | X# | C#, A#, E#, D# | Counterpositions attack any substantive node |
| `uncertainty.targets` | U# | A#, E#, C#, D# | Uncertainties question any substantive node |

**Not valid targets**: U# and X# are never valid in `depends_on`, `used_by`, `supports_claims`, `supports_claims`, or `supported_by_definitions`. They are reactive/meta nodes, not load-bearing.

### Analysis Methods

- `get_support_chain(node_id)`: Recursive backward traversal — all nodes that transitively support this node. For D# nodes, traces through A#/E# → C# → THESIS.
- `get_dependent_nodes(node_id)`: Recursive forward traversal — all nodes that transitively depend on this node. From D# nodes, finds A#/E# via `used_by`, then their dependent claims and thesis.
- `find_critical_paths()`: Finds single-point-of-failure chains from assumptions to high-strength claims (>0.7). Uses `_find_all_paths()` DFS to find ALL paths between each assumption and each high-strength claim. A path is critical if it's the ONLY path between those nodes.
- `get_graph_metrics()`: Returns `{total_nodes, total_edges, node_counts, critical_path_count, orphaned_claims, has_cycles}`. `node_counts` includes separate counts for all 7 types including `"thesis": 0 or 1`.
- `get_node(node_id)`: Returns node data by ID, or `None` if not found. Simple lookup in the internal `self.nodes` dict.
- `_node_exists(node_id)`: Returns `True` if a node ID exists in the graph.
- `_find_all_paths(source, target)`: DFS that finds all paths between two nodes. Used by `find_critical_paths()`. Uses visited set and backtracking to explore all branches.

---

## 15. Graph Visualization

Implemented in `beliefs/graph_visualizer.py`. Generates an interactive Cytoscape.js-based belief graph visualization as a standalone HTML file.

### Node Creation

Each CBS node becomes a Cytoscape node with ID `{agent_name}_{node_id}`. Labels are truncated to 50 characters.

**Label generation rules**:
- Definitions: `{id}: {term}`
- Claims/Assumptions: `{id}: {statement}`
- Evidence: `{id}: {summary}`

### Node Color Scheme

| Node Type | Color | Hex Code |
|-----------|-------|----------|
| Definition | Teal | `#2AA198` |
| Assumption | Blue | `#3498db` |
| Claim | Red | `#e74c3c` |
| Evidence | Green | `#2ecc71` |
| Prediction | Orange | `#f39c12` |
| Unrecognized | Gray | `#95a5a6` |

### Edge Styling

- Width: 2px
- Color: Dark gray (`#666`)
- Arrowheads: Triangle
- Curve style: Bezier

### Layout

- Algorithm: Breadthfirst (hierarchical tree-like)
- Directed: Yes
- Padding: 30px
- Spacing factor: 1.5

### Interactive Features

- Click a node to see its full data in the sidebar
- Q&A overlay: clicking a node shows all debate Q&A pairs targeting that node, including attack type, strategy, agent responses, and resolution
- Dark theme UI with proper contrast

---

## 16. Patch Operations

All belief updates go through `apply_patches()` in `patches.py`. Each patch is a dict with an `op` field.

| Operation | Required Fields | Description |
|-----------|----------------|-------------|
| `update_thesis` | At least one of: `new_strength`, `change`, `stance`, `summary_bullets` | Update thesis properties. `change` can be `"weaken"` or `"strengthen"` (±0.1, validated via `_CHANGE_ENUM`). `change` and `new_strength` are **mutually exclusive** — providing both fails validation. Also accepts `strength_reasoning`. |
| `update_claim` | `target_id`, `changes` (dict) | Modify any claim property. Tracks strength changes for propagation. Setting status to "retracted" forces strength to 0.0. |
| `add_claim` | `item` (full claim object) | Requires: id, type, statement, depends_on, strength, status, predictions, inference_chain. |
| `add_definition` | `item` (full D# object) | Requires: id (must match `^D\d+$` via `_DEFINITION_ID_RE`), term, definition, strength, strength_justification, status, used_by. Wires bidirectional cross-references automatically. |
| `update_definition` | `target_id`, `changes` (dict) | Modify definition/strength/status. `term` is **immutable** — cannot be changed. Retracted → strength 0.0. Triggers D# ceiling re-enforcement. Automatically maintains bidirectional `used_by`↔`supported_by_definitions` when `used_by` changes. |
| `add_evidence` | `item` (with `id`) | Defaults status to "active" if not provided. |
| `update_evidence` | `target_id`, `changes` | Enforces retracted → strength 0.0. |
| `update_assumption` | `target_id`, plus at least one of: `new_statement`, `new_type`, `changes` (dict) | Enforces retracted → strength 0.0. At least one change must be provided (the operation needs something to change). |
| `add_assumption` | `item` (with `id`, `type`, `statement`, `strength`) | Defaults status="active", supports_claims=[]. |
| `add_counterposition` | `item` (with `id`, `targets`, `attack_type`, `statement`, `my_response`, `response_sufficiency`) | Adds a new X# node. Requires valid `attack_strategy` for the given `attack_type`. `my_response` is always required but can be empty when `response_sufficiency` is "unaddressed" or "moot". |
| `update_counterposition` | `target_id`, `changes` | Modify any counterposition property. Moot terminal enforcement: cannot change `response_sufficiency` from "moot" to any other value. |
| `add_uncertainty` | `item` (with `id`) | Adds a new U# node. |
| `resolve_uncertainty` | `target_id`, `resolution_note` | Sets status="resolved" + resolution note. `resolution_note` must be non-empty. |

### Automatic Side Effects

After all patches are applied:
1. **D# ceiling enforcement**: A#/E# strengths capped by their lowest non-retracted D# support.
2. **Strength propagation**: BFS through dependency graph, capping claim strengths.
3. **Orphan caps**: A#/E# with all D# retracted capped at `ORPHAN_AE_CAP` (0.6); C# with no active support capped at `ORPHAN_CLAIM_CAP` (0.2).
4. **Thesis recalculation**: Formula applied, `strength_reasoning` auto-generated.
5. **Changelog entry**: Auto-appended with version and changes list.
6. **Version increment**: `version` field incremented by 1 (defaulting to 2 if no version exists).

**Deep copy mechanism**: `apply_patches()` creates its working copy via `json.loads(json.dumps(prior_belief))` — a full deep copy via JSON serialization to avoid mutating the original belief dict. This is important because patches mutate the working copy in-place.

**System-managed field stripping**: When applying `update_*` patches, system-managed fields (`original_strength`, `consecutive_defenses`) are automatically stripped from user-provided changes to prevent accidental mutation.

**Defense tracking initialization in `add_*` operations**: When new nodes are added via `add_claim`, `add_evidence`, `add_assumption`, or `add_definition`, defense tracking fields (`original_strength` and `consecutive_defenses`) are automatically initialized on the new node.

**Bidirectional wiring for definitions**:
- **`add_definition`**: When a new definition is added, `apply_patches()` automatically appends the D# ID to `supported_by_definitions` on all A#/E# nodes listed in the definition's `used_by` field. This maintains bidirectional consistency without requiring the agent to issue separate `update_assumption`/`update_evidence` patches.
- **`update_definition`**: When a definition's `used_by` field is changed, the system automatically updates `supported_by_definitions` on affected A#/E# nodes:
  - **Removed A#/E#**: D# ID stripped from their `supported_by_definitions`
  - **Added A#/E#**: D# ID appended to their `supported_by_definitions`

**Non-fatal error handling**: Unknown patch `op` values are logged as warnings and skipped (not fatal). Strength propagation failures are also non-fatal — logged as warnings, patch application continues.

### Patch Validation

`validate_patches()` in `patches.py` checks before applying:
- All `op` fields are recognized operations
- All `target_id` references exist in the belief
- All `item.id` values are unique (no duplicates)
- Strength values are in [0.0, 1.0]
- Status values are valid enums
- `add_claim` items have all required fields including predictions and inference_chain
- `depends_on` references exist
- `resolve_uncertainty` has a non-empty resolution_note
- `add_definition` items have all required fields (id, term, definition, strength, strength_justification, status, used_by). ID must match `_DEFINITION_ID_RE` pattern (`^D\d+$`).
- `update_definition` cannot modify `term` (immutable field)
- `add_counterposition` items have valid `attack_strategy` for their `attack_type`
- Moot terminal enforcement: `update_counterposition` cannot change `response_sufficiency` from "moot" to any other value
- Cross-reference fields in add operations contain only allowed ID prefixes, aligned with `ALLOWED_REF_PREFIXES` from `schema.py`: `depends_on` → A#/E#/C#, `supports_claims` → C#, `supported_by_definitions` → D#, `targets` (X#) → C#/A#/E#/D#, `targets` (U#) → A#/E#/C#/D#

**Projection pass**: `validate_patches()` performs a **projection pass** before validation — it pre-registers IDs from all `add_*` patches, allowing later patches in the same batch to forward-reference newly added items.

**Cascade failure detection**: If an `add_*` patch fails validation, a transitive cascade loop flags all downstream patches that reference the failed patch's ID. This handles multi-hop chains (e.g., C3 fails → A5 references C3 → D6 references A5 → all cascaded). The cascade repeats until no new failures are discovered.

---

## 17. Validation Pipeline Summary

`validate_belief()` in `schema.py` performs a complete ordered sequence of validation stages:

1. **Presence checks**: All 11 required top-level keys exist
2. **Schema version check**: `schema_version` must equal `"CBS"`
3. **Version type & range**: Must be integer ≥ 1
4. **Thesis field-level checks**: strength ∈ [0.0, 1.0], summary_bullets non-empty
5. **Type checks**: thesis/metadata are dicts, all collections are arrays
6. **Deep JSON Schema validation**: If `jsonschema` library installed (`HAVE_JSONSCHEMA` flag), full schema validation against `CBS_JSON_SCHEMA`
7. **ID hygiene**: Format must match `^[ACDEUX]\d+$`, correct collection placement, no duplicates
8. **Sequential ID ordering**: `_validate_sequential_ids()` per prefix (D, A, C, E, U, X)
9. **Definition field-level validation**: term non-empty, definition non-empty, strength range, status enum, used_by non-empty
10. **Definition cross-reference validation**: Bidirectional D#↔A#/E# consistency checks
11. **Edge-type prefix validation**: `_validate_ref_prefixes()` using `ALLOWED_REF_PREFIXES`
12. **Assumption validation**: type enum, strength range, status enum, strength_justification required
13. **Counterposition validation**: attack_type enum, response_sufficiency enum, attack_strategy valid for attack_type
14. **Claim validation**: status enum, strength range, strength_justification required, predictions non-empty with required fields, inference_chain via `validate_inference_chain()`
15. **Evidence validation**: type enum, strength range, strength_justification required, status enum
16. **Uncertainty validation**: status enum, importance enum

Stage 6 (deep JSON Schema validation) is gated behind `HAVE_JSONSCHEMA` — if the `jsonschema` library is not installed, this stage is skipped and only the manual checks (stages 1–5, 7–16) are performed.

---

## 18. Validation Enums & Constants Reference

Quick-reference table for all validation enums defined in `patches.py`:

| Constant | Values | Used By |
|----------|--------|---------|
| `_STATUS_ENUM` | `{active, revised, retracted}` | D#, A#, C#, E# |
| `_UNCERTAINTY_STATUS_ENUM` | `{active, resolved}` | U# |
| `_IMPORTANCE_ENUM` | `{high, medium, low}` | U# |
| `_ASSUMPTION_TYPE_ENUM` | `{foundational, empirical, methodological, scoping}` | A# |
| `_EVIDENCE_TYPE_ENUM` | `{empirical, conceptual, expert_consensus}` | E# |
| `_ATTACK_TYPE_ENUM` | `{undermining, rebutting, undercutting}` | X# |
| `_SUFFICIENCY_ENUM` | `{sufficient, partial, unaddressed, moot}` | X# |
| `_CHANGE_ENUM` | `{weaken, strengthen}` | update_thesis |
| `_DEFINITION_ID_RE` | `^D\d+$` | add_definition |

And from `schema.py`:

| Constant | Values | Used By |
|----------|--------|---------|
| `VALID_IC_ROLES` | `{premise, inference, conclusion}` | inference_chain |
| `VALID_INFERENCE_TYPES` | `{deductive, inductive, abductive}` | inference_chain |
| `_IC_REFERENCE_RE` | `^[ACE]\d+$` | inference_chain premise references |
| `ALLOWED_REF_PREFIXES` | (see [Edge-Type Constraints](#edge-type-constraints-cross-reference-validation)) | Cross-reference validation |

---

## 19. Field Whitelists Reference

Quick-reference for all update operation whitelists from `patches.py`. Fields not on the whitelist are silently ignored during update operations.

| Whitelist | Mutable Fields |
|-----------|----------------|
| `_UPDATE_CLAIM_WHITELIST` | strength, strength_justification, statement, status, depends_on, predictions, inference_chain, type |
| `_UPDATE_EVIDENCE_WHITELIST` | strength, strength_justification, summary, source, status, supports_claims, type, supported_by_definitions |
| `_UPDATE_ASSUMPTION_WHITELIST` | strength, strength_justification, statement, status, type, supports_claims, supported_by_definitions |
| `_UPDATE_COUNTERPOSITION_WHITELIST` | my_response, response_sufficiency, statement, attack_type, targets |
| `_UPDATE_DEFINITION_WHITELIST` | definition, strength, strength_justification, status, used_by |

**Notable absences**: `id` and `term` are absent from `_UPDATE_DEFINITION_WHITELIST` — they are immutable after creation. Similarly, `attack_strategy` is absent from `_UPDATE_COUNTERPOSITION_WHITELIST`.

---

## 20. CBS Parsing and I/O

Implemented in `beliefs/io.py`.

### `parse_model_output_to_belief()`

Extracts a CBS belief object from raw LLM output text.

**JSON extraction strategy** (in priority order):
1. Fenced JSON block: regex `r"```json\s*(\{.*?\})\s*```"`
2. Raw JSON object fallback: regex `r"(\{.*\})"`

**ID normalization**: Converts LLM output IDs like `"A#1"` to `"A1"` using regex substitution (`r'"([ACDEUX])#(\d+)"'` → `r'"\1\2"'`). This handles common LLM formatting artifacts.

**Validation**: After parsing, calls `validate_belief()` and extends the error list.

**Return**: Tuple of `(belief_dict | None, markdown_text | None, errors_list)`.

### `belief_to_markdown()`

Generates a stable Markdown rendering of all CBS components.

- Includes all 7 component sections: Thesis, Definitions, Assumptions, Claims, Evidence, Uncertainties, Counterpositions, plus Changelog
- Includes defense tracking info (`consecutive_defenses`, `original_strength`) when present
- Handles both dict and string `source` fields on evidence (dict sources rendered as comma-separated `key: value` pairs)
- Supports both role-based and legacy inference chain formats

### `load_belief_from_file()`

Loads a CBS belief from a JSON file.

- Validates using `validate_belief()`
- Raises `FileNotFoundError` or `ValueError` on problems

---

## 21. Embedding Projection Functions

Two embedding projection functions in `io.py` extract CBS data for semantic embedding.

### `project_for_embedding()` — Simple Text Projection

Creates a concise, deterministic text summary for embedding.

**Includes**:
- Thesis stance + bullets + strength
- Top 3 definitions by strength
- Top 3 claims by strength
- First uncertainty
- First 2 counterpositions

Used when only a string input is available for embedding.

### `project_for_component_embedding()` — Rich Per-Component Projection

Returns a dict with per-component text lists, strength values, and scalar features.

**Components**:
- **Definitions**: `{"text": "term: definition", "strength": float}` — retracted filtered out
- **Assumptions**: `{"text": "statement", "strength": float}` — retracted filtered out
- **Evidence**: `{"text": "summary", "strength": float}` — retracted filtered out
- **Claims**: `{"text": "statement", "strength": float}` — retracted filtered out
- **Thesis text**: `"stance. bullet1. bullet2. ..."` or just stance if no bullets
- **Uncertainties**: Non-resolved question strings only
- **Counterpositions**: Grouped by `response_sufficiency` into 4 buckets (partial/sufficient/unaddressed/moot)

**Scalars**: 11 raw numeric features (normalization/clamping happens downstream in the embedding tracker, not here):
1. n_definitions (raw count of active definitions)
2. n_assumptions (raw count of active assumptions)
3. n_evidence (raw count of active evidence)
4. n_claims (raw count of active claims)
5. avg_strength_definitions
6. avg_strength_assumptions
7. avg_strength_evidence
8. avg_strength_claims
9. n_counterpositions (raw count of ALL counterpositions)
10. n_uncertainties (raw count of ALL uncertainties)
11. thesis_strength

**Active node filtering**: Strength-bearing components (D#, A#, C#, E#) filter out retracted nodes. Counterposition and uncertainty counts include ALL items regardless of status.

---

## 22. Pipeline Lifecycle

### Stage 1: Belief Formation

- **Agent prompt**: `build_stage_1_belief_prompt_cbs()` in `prompts.py`
- **Output**: Complete CBS object with all required fields
- **Generation order**: A# assumptions → E# evidence → D# definitions → C# claims → X# counterpositions → U# uncertainties → Thesis (last). D# definitions are generated AFTER A#/E# so the agent can see which terms need defining.
- **D# requirements**: At least 1 definition per key term. Each A#/E# must list at least one D# in `supported_by_definitions`. Bidirectional cross-references validated.
- **Custom belief loading**: Agents with a `belief_file` configured in `AgentConfig` skip LLM generation entirely in Stage 1. Their pre-loaded belief object is used directly, with defense tracking initialized on it.
- **Storage**: JSON stored via `agent.set_internal_belief_obj()`, Markdown via `agent.set_internal_belief()`
- **Snapshot**: JSON string appended to `agent.all_beliefs_held[]`
- **Defense tracking initialization**: After beliefs are parsed and validated, `initialize_defense_tracking()` (from `patches.py`) sets `original_strength` and `consecutive_defenses` on ALL strength-bearing nodes (D#, A#, C#, E#).
- **Initial snapshot**: `snapshot_belief()` captures `thesis_strength` and `component_counts` for each agent, stored in `agent_stats` for tracking belief evolution across rounds.
- **Validation**: `validate_belief()` + `BeliefGraph.validate_links()`. Blocking validation errors trigger up to 3 retry attempts with a revision prompt explicitly listing the fix requirements. Non-blocking warnings are logged but don't trigger retries.

### Stage 2: Cross-Examination

- **Reads**: Opponent's belief Markdown and structured CBS object
- **Targets**: Specific D#/A#/C#/E#/X#/U# nodes via `target_ids`
- **Uses**:
  - Definition `term` and `strength` to identify definitional weaknesses
  - Claim `strength` and `strength_justification` to identify weak points
  - Counterposition `response_sufficiency` to find vulnerabilities ("partial"/"unaddressed")
  - Uncertainty `importance` to prioritize ("high"/"medium")
  - Assumption `type` to select appropriate attack strategy
- **D# targeting**: Opponents can challenge D# nodes using definitional attack strategies: `over_extension` and `under_extension` (under `undermining`), and `circularity`, `stipulative_bias`, `conceptual_conflation` (under `undercutting`).
- **Anti-repetition tracking**: Previous round challenges and their outcomes are passed to Stage 2 prompts, preventing agents from repeating failed attack strategies.
- **`targeted_claims_json` parameter**: `build_stage_2_prompt()` accepts an optional `targeted_claims_json` parameter for focused cross-examination on specific claims.
- **Does not modify**: The CBS object itself. Questions are external artifacts.

### Stage 3: Rebuttal / Response

- **Agent prompt**: `build_stage_3_rebuttal_prompt()` or mode-specific variant
- **Output**: JSON patches array
- **QID renumbering**: In `_generate_rebuttal()` in `debate_controller.py`, questions from multiple challengers are renumbered sequentially (Q1, Q2, ..., QN) before being presented to the target agent. This is necessary because each challenger independently numbers their questions Q1–Q5, causing ID collisions when combined. The renumbered QIDs are written back into the `current_round_pairs` entries so the mapping-back step (matching LLM responses to original challenges) uses consistent IDs. The rebuttal validator also uses the renumbered QIDs as `expected_qids`.
- **`last_rebuttals_patches` tracking**: Patches proposed in Stage 3 rebuttals are stored per-agent in `last_rebuttals_patches` and passed to Stage 5 Phase 1 as additional context.
- **Legacy JSON fallback parsing**: Stage 3 parsing tries a unified JSON block first (`{rebuttals: [...], patches: [...]}`), then falls back to two separate fenced JSON blocks (first = rebuttals, second = patches).
- **Typical patches**: `update_claim`, `update_evidence`, `update_assumption`, `update_definition`, `add_evidence`, `add_definition`, `add_counterposition`, `resolve_uncertainty`
- **D# patches**: Agents can revise definitions (`update_definition`), add new ones (`add_definition`), or retract via status change. D# ceiling enforcement runs automatically after patches.
- **Applied**: Via `apply_patches()` with D# ceiling enforcement + strength propagation

### Stage 4: Adjudication

- **Reads**: Challenge text, rebuttal patches, targeted belief excerpts
- **Uses**: `_extract_belief_excerpt()` to pull relevant D#/A#/C#/E#/X# items by `target_ids`
- **Incomplete pair handling**: Challenge-rebuttal pairs missing either a challenge or rebuttal are skipped entirely with a warning, not dispatched to adjudication.
- **D# evaluation**: Definitional challenges (using strategies like `circularity`, `over_extension`, `under_extension`, `stipulative_bias`, `conceptual_conflation`) are assessed for whether the definition exhibits the claimed flaw and downstream A#/E# impact.
- **Output**: Resolution verdicts (critique_valid / rebuttal_valid / unresolved)
- **`enforce_verdict()` system**: After the LLM renders its verdict, `enforce_verdict()` (in `adjudicator.py`) recomputes combined scores mathematically using the formula `combined = logic_weight × logic + ethics_weight × ethics`. If the computed verdict disagrees with the LLM's stated verdict, the computed verdict overrides it. This prevents LLM bias from affecting outcomes.
- **Does not modify**: The CBS object. Verdicts feed into Stage 5.

### Stage 5: Belief Update (Two-Phase)

**Phase 1 — Adjudication Enforcement**:
- Mandatory response to adjudication outcomes
- If `critique_valid` on a D# node: must weaken/revise the targeted definition
- D# ceiling enforcement automatically cascades weakened definitions to dependent A#/E# → C# → thesis
- Restricted: No thesis text rewrites, no strategic retractions
- Patches applied via `apply_patches()`
- **Defense boosts applied** after Phase 1 patches: `apply_defense_boosts()` mechanically increases strengths for REBUTTAL_VALID outcomes and resets defense streaks for CRITIQUE_VALID outcomes (see [Defense Boost System](#defense-boost-system))

**Phase 2 — Introspective Rewrite**:
- Free-form strengthening of position
- Must resolve unaddressed counterpositions (including definitional ones)
- Must review high/medium importance uncertainties
- May add new D# definitions or strengthen existing ones
- **Strength filter applied**: `filter_strength_increases()` in `debate_controller.py` strips strength increases from `update_*` patches on existing nodes — agents cannot unilaterally raise existing strengths (only defense boosts can). New nodes via `add_*` are not affected.
- **Example patches in prompt**: The Phase 2 prompt includes three categories of example patches to guide the agent: **Defensive** (weaken/retract under pressure), **Growth** (add supporting infrastructure + new claim), and **Refinement** (improve existing nodes textually + add support).
- **Breadth table**: A dynamically generated table showing n claims (1–7) → breadth multiplier values is injected into the Phase 2 prompt.
- `update_thesis` must be LAST operation
- Thesis strength must equal formula result
- Patches applied via `apply_patches()`

**Patch application error handling**: If patch application fails (exception), the system reverts to the prior belief and logs an error. If graph validation fails after patches, the system also reverts. Individual invalid patches are skipped (not all-or-nothing within a batch).

**Legacy single-phase Stage 5 flow**: `build_stage_5_belief_update_prompt_cbs()` still exists as a legacy fallback for when `prior_json` (the previous round's belief) is not available. It runs a single-phase update with the full thesis strength formula.

### Final Snapshot

- Updated CBS object stored via `agent.set_internal_belief_obj()`
- Markdown view regenerated via `belief_to_markdown()`
- JSON snapshot appended to `agent.all_beliefs_held[]`
- Embeddings tracked for trajectory visualization
- Belief refinement is tracked through per-round `snapshot_belief()` captures (thesis strength and component counts), attack histogram analysis, verdict override frequency, and embedding trajectory visualization.

---

## 23. Component Interactions & Truth-Directed Design

This section describes how the six node types (D#, A#, E#, C#, X#, U#) work together as an integrated system, how the LLM prompts orchestrate their use across stages, and how the overall architecture drives belief refinement toward truth rather than rhetorical victory.

### Structural Dependency Hierarchy

The CBS belief forms a layered DAG where each layer constrains the ones above it:

```
Layer 0: Definitions (D#) ─── semantic bedrock; highest strengths (0.7-1.0)
   │ ceiling
Layer 1: Assumptions (A#) + Evidence (E#) ─── grounded by D#; strength ≤ min(D#)
   │ ceiling
Layer 2: Claims (C#) ─── depend on A#/E#/C#; strength ≤ min(dependencies)
   │ determine
Layer 3: Thesis ─── computed from claim strengths; never agent-set
```

**D# grounds A# and E#**: Every assumption and evidence item must reference at least one definition via `supported_by_definitions`. D# strength acts as a ceiling — if D1 has strength 0.7, no A# or E# that depends on D1 can exceed 0.7. This forces agents to establish clear, defensible term definitions before building arguments. Weakening a definition automatically cascades through all dependent nodes.

**A# and E# support C#**: Claims list their dependencies in `depends_on`. A claim's strength cannot exceed its weakest active dependency. This means a claim is only as strong as the weakest link in its supporting chain. Agents must identify their limiting dependencies in `strength_justification`.

**C# determines Thesis**: Thesis strength is always computed by formula: `avg(active_claim_strengths) × (n^p / (n^p + 1))`. Agents cannot set thesis strength directly — they can only raise it by building stronger or more numerous claims.

**X# challenges all substantive nodes**: Counterpositions target D#, A#, E#, or C# nodes. They serve dual roles: (1) honest self-assessment during belief formation, and (2) attack surface that opponents exploit during cross-examination. Unaddressed counterpositions must be resolved in Stage 5 Phase 2.

**U# questions all substantive nodes**: Uncertainties mark open questions about D#, A#, E#, or C# nodes. High/medium importance uncertainties are prioritized for both opponent attacks (`press_uncertainty`) and agent resolution. Resolving a U# by adding supporting material strengthens the position.

### How LLM Prompts Orchestrate Component Use

Each stage prompt instructs the LLM to interact with the component system differently:

**Stage 0 (Briefing)** — `build_universal_prompt()` and `build_position_prompt()`:
- Establishes the CBS protocol: stable IDs, strength calibration, intellectual honesty norms
- Assigns the persona as "a lens for analysis, not conclusions to defend at all costs"
- Sets the expectation that concession is valued over rhetorical victory

**Stage 1 (Opening Position)** — `build_stage_1_belief_prompt_cbs()`:
- The LLM builds a complete CBS belief bottom-up: A# → E# → D# → C# → X# → U# → Thesis
- D# are generated *after* A#/E# so the agent can see which terms need defining
- The prompt enforces all dependency rules: D# ceiling, claim limiting, bidirectional cross-references
- Thesis is generated *last* — grounded in the claims actually built, not the reverse
- The prompt includes the full strength scale, thesis formula with worked examples, and a condensed CBS example

**Stage 2 (Cross-Examination)** — `build_stage_2_prompt()`:
- The LLM receives the opponent's full CBS belief and targets specific nodes via `target_ids`
- The prompt provides the complete attack taxonomy (27 strategies across 3 attack types, including epistemological and ethical strategies)
- The LLM uses node metadata to find weaknesses: low D# `strength`, "partial"/"unaddressed" `response_sufficiency` on X#, "high" `importance` on U#, weak `strength_justification` on C#
- Definitional attack strategies (e.g., `over_extension`, `circularity`) specifically target D# nodes to cascade weakness through the dependency graph
- Ethical attack strategies (e.g., `challenge_moral_implications`, `present_ethical_counter`, `challenge_normative_inference`) challenge the moral standing, normative inferences, or value coherence of a position

**Stage 3 (Rebuttal)** — `build_stage_3_structured_rebuttal_prompt()`:
- The LLM defends by proposing patches: `update_claim`, `add_evidence`, `update_definition`, etc.
- Three actions: `refute` (defend the node), `concede` (weaken with patches), `defer` (add U#)
- Concessions require corresponding weakening patches — the agent cannot just acknowledge a problem without reflecting it in the belief structure

**Stage 4 (Adjudication)** — `build_adjudicator_per_pair_prompt()`:
- The adjudicator evaluates each challenge-rebuttal pair and renders a verdict: `rebuttal_valid`, `critique_valid`, or `unresolved`
- Uses `_extract_belief_excerpt()` to pull relevant D#/A#/C#/E#/X# items by `target_ids`
- The prompt enforces anti-bias rules: "acknowledging a challenge is NOT a successful defense — it must RESOLVE the logical issue"

**Stage 5 Phase 1 (Enforcement)** — `build_stage_5_phase1_enforcement_prompt()`:
- The LLM responds to adjudication verdicts with mandatory patches
- `CRITIQUE_VALID` → at least one weakening patch; definitional critiques require `update_definition` which automatically cascades via ceiling
- `REBUTTAL_VALID` → defense boosts applied mechanically by the system (not the agent)
- `UNRESOLVED` → must add U# uncertainty targeting the disputed nodes
- The prompt explicitly tells agents NOT to manually increase strengths — the system handles this

**Stage 5 Phase 2 (Introspection)** — `build_stage_5_phase2_introspection_prompt()`:
- The LLM receives a dynamically computed `<position_analysis>` block with partial derivatives and scenario projections
- The prompt includes example patches (defensive, growth, refinement categories) and a dynamically generated breadth multiplier table
- The prompt guides the agent through: (1) counterposition audit — resolve unaddressed X#, (2) uncertainty review — prioritize high/medium U#, (3) strategic position building — add new D#/A#/E#/C# to raise thesis strength
- Two levers for raising thesis: increase average claim strength or increase breadth (more claims)
- Strength filter prevents unilateral self-strengthening of existing nodes
- Thesis rewrite is always LAST, grounded in current claims

### Truth-Directed Mechanisms

The CBS system incorporates multiple design choices that push toward truth rather than rhetorical victory:

1. **Formula-driven thesis strength**: Agents cannot set their thesis strength arbitrarily. It is always `avg(active_claim_strengths) × breadth`. This prevents gaming the top-line number independently of argument quality.

2. **D# ceiling enforcement**: Weak definitions automatically cap everything downstream. This forces agents to invest in clear, defensible semantics — vague or question-begging definitions mechanically weaken the entire position.

3. **Dependency-limited claim strength**: Claims cannot be stronger than their weakest link. The `strength_justification` field forces agents to identify their limiting dependency, creating transparency about where the argument is weakest.

4. **Defense boost rewards genuine defense**: Only nodes that survive adjudicated challenges get mechanically stronger. This incentivizes building defensible positions rather than inflating strengths.

5. **Phase 2 strength filter**: Agents cannot unilaterally raise existing node strengths during introspection. They can only add new nodes or lower existing ones. This prevents "trust me bro" self-strengthening without adversarial scrutiny.

6. **Counterposition honesty requirements**: Agents must rate their own `response_sufficiency` honestly. Rating a weak response as "sufficient" gets exposed during cross-examination and counts against them in adjudication. Unaddressed counterpositions must be resolved in Phase 2.

7. **Uncertainty as epistemic honesty**: U# nodes are explicit admissions of unknowns. High/medium importance uncertainties are prioritized for opponent attack, so agents cannot hide them. But resolving uncertainties through new evidence genuinely strengthens the position.

8. **Persona as lens, not conclusion**: The `build_position_prompt()` instructs agents to use their worldview as "a lens for analysis, not a set of conclusions to defend at all costs." When the worldview conflicts with strong evidence, agents are instructed to update.

9. **Concession norms**: The universal prompt states "Intellectual honesty is valued over rhetorical victory." Conceding genuinely when a critique lands is encouraged, not penalized.

10. **Bidirectional cross-reference validation**: D#↔A#/E# cross-references are validated in both directions, preventing structural inconsistencies that could mask logical gaps.

### Belief Refinement Cycle

Across multiple rounds, the system drives belief refinement through an adversarial-cooperative cycle:

```
Round N:
  Cross-examination targets weak D#/A#/E#/C# nodes → exposes vulnerabilities
  Rebuttal defends or concedes → patches proposed
  Adjudication rules on each challenge → REBUTTAL_VALID / CRITIQUE_VALID / UNRESOLVED
  Enforcement applies mandatory changes → weaknesses corrected
  Defense boosts reward survived challenges → strong nodes get stronger
  Introspection adds new material, resolves U#, retires weak claims → position refined

Result: Each round culls weak arguments, strengthens defended ones, and
        adds new supporting material — converging toward a more robust,
        better-calibrated, truth-tracking belief.
```

Belief refinement is tracked through per-round `snapshot_belief()` captures (thesis strength and component counts), attack histogram analysis, verdict override frequency, and embedding trajectory visualization.

---

## 24. Unused & Write-Only Fields

This section identifies fields that are generated but have limited or no downstream consumption beyond display.

### Removed Fields

The following fields were removed as part of the D# definition nodes preparation:

| Field | Former Location | Reason for Removal |
|-------|----------------|-------------------|
| `update_policy` (and sub-fields) | `schema.py` (commented out) | Completely dead code — never generated, never consumed. |
| `metadata.last_updated` | `schema.py`, `patches.py` | Written by `apply_patches()` but never read or displayed. |
| `metadata.scope_conditions` | `schema.py`, `io.py` | Display only. Replaced by scoping assumptions (A# type: "scoping"). |
| `metadata.definitions[]` | `schema.py`, `io.py` | Display only. Replaced by top-level D# definition nodes. |
| `changelog[].timestamp` | `schema.py`, `patches.py`, `io.py` | Written and displayed but never used in logic. |

### Generated But Display-Only (Not Consumed Programmatically)

These fields are generated by agents or the patch system and rendered in transcript Markdown, but no downstream logic reads or acts on their values.

| Field | Generated By | Displayed In | Programmatic Consumer |
|-------|-------------|-------------|----------------------|
| `thesis.strength_reasoning` | `apply_patches()` auto-generation | `belief_to_markdown()` | None — always overwritten by formula |
| `definition.strength_justification` | Agent | `belief_to_markdown()` | None |
| `assumption.strength_justification` | Agent | `belief_to_markdown()` | None |
| `claim.strength_justification` | Agent | `belief_to_markdown()` | None |
| `evidence.strength_justification` | Agent | `belief_to_markdown()` | None |
| `claim.inference_chain` | Agent | `belief_to_markdown()` | `validate_inference_chain()` — structurally validated (roles, ordering, references, inference_type) |
| `claim.predictions[]` | Agent | `belief_to_markdown()` | Validated for structural completeness: each prediction must have `statement`, `test`, and `decision_criterion` fields (checked by `validate_belief()`) |
| `claim.predictions[].decision_criterion` | Agent | `belief_to_markdown()` | None |
| `claim.predictions[].potential_falsifiers` | Agent | `belief_to_markdown()` | None |
| `evidence.source` | Agent | `belief_to_markdown()` | None |

### Generated and Actively Consumed (For Reference)

These fields are read programmatically by downstream systems — they are NOT write-only.

| Field | Key Consumer | What It Does |
|-------|-------------|-------------|
| `definition.term` | `graph_visualizer.py` (node labels), `schema.py` (immutability enforcement), `io.py` (Markdown rendering and embedding projection) | Display; identity; embedding |
| `definition.definition` | `io.py` (embedding projection in `project_for_component_embedding()`, Markdown rendering) | Embedding and display |
| `definition.strength` | `patches.py` (D# ceiling enforcement), `io.py` (embedding ranking) | Caps dependent A#/E# strengths |
| `definition.status` | `patches.py` (retraction → 0.0, orphan cap logic) | Controls ceiling enforcement and orphan detection |
| `definition.used_by` | `belief_graph.py` (D# → A#/E# edges), `patches.py` (ceiling lookup), `schema.py` (cross-ref validation) | Graph structure; strength propagation; validation |
| `thesis.stance` | `io.py` (`project_for_embedding()`, `project_for_component_embedding()`, `belief_to_markdown()`), `reporting.py` (analysis reports) | Embedding text; display |
| `claim.strength` | `patches.py` (propagation), `belief_graph.py` (critical paths), `reporting.py` (drift tracking), `io.py` (embedding projection), `debate_controller.py` (`snapshot_belief()`) | Core of strength propagation; ranking |
| `claim.status` | `patches.py` (retraction enforcement, thesis formula exclusion), `belief_graph.py` (THESIS edge) | Controls which claims participate in calculations |
| `claim.depends_on` | `patches.py` (propagation), `belief_graph.py` (edge construction), `schema.py` (edge-type validation: A#/E#/C# only), `debate_controller.py` (excerpt extraction) | Core graph structure |
| `claim.statement` | `graph_visualizer.py` (node labels), `io.py` (Markdown rendering and embedding projection) | Display and embedding |
| `assumption.strength` | `patches.py` (propagation) | Limits dependent claim strengths |
| `assumption.status` | `patches.py` (retraction enforcement) | Excluded from limiting when retracted |
| `assumption.supported_by_definitions` | `patches.py` (D# ceiling lookup), `schema.py` (cross-ref validation) | Links A# to its grounding D# definitions |
| `assumption.supports_claims` | `schema.py` (edge-type validation: C# only), `patches.py` (add_assumption prefix check) | Validated for correct prefixes. NOT used for graph edge construction — see below. |
| `evidence.strength` | `patches.py` (propagation) | Limits dependent claim strengths |
| `evidence.status` | `patches.py` (retraction enforcement) | Excluded from limiting when retracted |
| `evidence.supported_by_definitions` | `patches.py` (D# ceiling lookup), `schema.py` (cross-ref validation) | Links E# to its grounding D# definitions |
| `evidence.supports_claims` | `schema.py` (edge-type validation: C# only), `patches.py` (add_evidence prefix check) | Validated for correct prefixes. NOT used for graph edge construction — see Notable Asymmetry below. |
| `counterposition.response_sufficiency` | `prompts.py` (Stage 2 attack, Stage 5 Phase 2 enforcement) | Guides opponent strategy; mandatory resolution trigger |
| `counterposition.attack_type` | `prompts.py` (Stage 2), `graph_visualizer.py` | Guides attack strategy selection |
| `counterposition.attack_strategy` | `schema.py` (validation against VALID_ATTACK_STRATEGIES), `prompts.py` (Stage 2) | Validated sub-strategy; guides specific attack formulation. Valid values include both epistemological strategies (e.g., `challenge_evidence`, `identify_circularity`) and ethical strategies (e.g., `challenge_moral_implications`, `present_ethical_counter`, `challenge_normative_inference`). |
| `counterposition.targets` | `belief_graph.py` (edge construction), `debate_controller.py` (excerpt extraction) | Graph structure |
| `uncertainty.status` | `patches.py` (resolve operation), `io.py` (rendering) | Resolution tracking |
| `uncertainty.importance` | `prompts.py` (Stage 2 prioritization, Stage 5 review) | Prioritizes which uncertainties to press/resolve |
| `uncertainty.targets` | `belief_graph.py` (edge construction) | Graph structure |
| `uncertainty.resolution_note` | `patches.py` (resolve operation), `io.py` (rendering) | Records how uncertainty was resolved |
| `changelog[].changes` | `debate_controller.py` | Passed as context in belief update prompts |
| `thesis.strength` | `patches.py`, `reporting.py`, `debate_controller.py` | Formula enforcement, change detection, reporting |
| `*.original_strength` (D#/A#/E#/C#) | `debate_controller.py:apply_defense_boosts()` | Defense boost ceiling calculation: `original_strength + max_cumulative_boost` |
| `*.consecutive_defenses` (D#/A#/E#/C#) | `debate_controller.py:apply_defense_boosts()` | Defense boost tracking: each successful defense adds `flat_boost` to strength |

### Summary: Remaining Display-Only Fields

The following fields could theoretically be removed without breaking any logic, but they provide crucial transparency and auditability for human reviewers:

1. **`strength_justification`** (all node types including D#) — display only
2. **`thesis.strength_reasoning`** — auto-generated display only (always overwritten by formula)
3. **`inference_chain`** — structurally validated (roles, ordering, references, inference_type checked by `validate_inference_chain()`) but content not consumed programmatically beyond validation
4. **`predictions`** — validated for structural completeness (required fields checked) but content never parsed
5. **`evidence.source`** — display only

**Important**: "Display only" does not mean "useless." These fields provide crucial transparency and auditability for human reviewers examining debate transcripts. The distinction is between *programmatic consumption* (affects debate behavior) and *informational consumption* (affects human understanding).
