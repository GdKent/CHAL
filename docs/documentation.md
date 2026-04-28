# CHAL (Council of Hierarchical Agentic Language) — Complete Technical Documentation

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [The CHAL Belief Schema (CBS)](#3-the-chal-belief-schema-cbs)
4. [Agent System](#4-agent-system)
5. [Epistemic Personas](#5-epistemic-personas)
6. [Logic Systems](#6-logic-systems)
7. [Ethics Systems](#7-ethics-systems)
8. [The 6-Stage Debate Pipeline](#8-the-6-stage-debate-pipeline)
   - [Stage 0: Briefing](#stage-0-briefing)
   - [Stage 1: Opening Positions](#stage-1-opening-positions)
   - [Stage 2: Cross-Examination](#stage-2-cross-examination)
   - [Stage 3: Rebuttals](#stage-3-rebuttals)
   - [Stage 4: Adjudication](#stage-4-adjudication)
   - [Stage 5: Belief Updates (Two-Phase)](#stage-5-belief-updates-two-phase)
9. [Debate Mode](#9-debate-mode)
10. [Cross-Examination & Attack Taxonomy](#10-cross-examination--attack-taxonomy)
11. [Belief Graph & Patch System](#11-belief-graph--patch-system)
12. [Adjudicator System](#12-adjudicator-system)
13. [Defense Boost System](#13-defense-boost-system)
14. [Embedding & Trajectory Analysis](#14-embedding--trajectory-analysis)
15. [Performance Scoring](#15-performance-scoring)
16. [Configuration System](#16-configuration-system)
17. [CLI & Interactive Wizard](#17-cli--interactive-wizard)
18. [Utilities](#18-utilities)
19. [Debug Logging System](#19-debug-logging-system)
20. [Output Artifacts](#20-output-artifacts)
21. [Complete Prompt Reference](#21-complete-prompt-reference)
22. [Dependencies](#22-dependencies)
23. [Test Suite](#23-test-suite)
24. [Primary Goal: Belief Refinement](#24-primary-goal-belief-refinement)

---

## 1. Executive Summary

CHAL is a framework for orchestrating structured philosophical debates between multiple LLM agents. The system implements a **6-stage debate pipeline** where agents representing distinct epistemological positions engage in multi-round dialectical exchanges with formal belief tracking, independent adjudication, and comprehensive output generation.

**One of the primary goals of the CHAL pipeline is to refine belief objects so that they are better — more logically coherent, better supported, and more accurately calibrated — by the end of the debate than they were at the beginning.** The entire pipeline is designed around this goal: agents start with initial beliefs, those beliefs are challenged through cross-examination, defended or conceded through rebuttals, evaluated by an independent adjudicator, and then deterministically updated through a two-phase patch system that propagates strength changes through the dependency graph. By the end of a debate, each agent's belief object should reflect the outcomes of the dialectical process — weakened where critiques landed, strengthened where defenses held, and refined where new understanding emerged.

### Key Capabilities

- **Multi-provider LLM support**: OpenAI (GPT-4o, o4-mini, o3-mini, o3), Anthropic (Claude Opus 4.6, Sonnet 4.5, Haiku 4.5), Google (Gemini 2.5 Pro, Gemini 2.5 Flash), Ollama (local models), xAI (Grok 3, Grok 3 Mini), Perplexity (Sonar Reasoning Pro, Sonar Reasoning)
- **Rebuttal debate mode**: Single-shot structured rebuttals with independent adjudication
- **Cross-examination**: Open free-form questioning with 27 targeted attack strategies across three attack types (undermining, rebutting, undercutting)
- **13 epistemic personas**: Philosophically grounded worldviews from Empiricist to Synthesist, plus a NONE option for persona-free argumentation — each rooted in specific intellectual traditions
- **8 logic systems**: From Classical Informal Bayesian (hybrid) to Paraconsistent, plus a NONE option for pure-ethics mode — configuring how the adjudicator evaluates argument validity
- **6 ethics systems**: From pure-logic (None) to Care Ethics, configuring the ethical dimension of adjudication
- **Formal belief tracking**: JSON-structured belief objects (CBS) with dependency graphs, definitions, counterpositions, strength propagation, and patch-based updates
- **Independent adjudication**: Neutral adjudicator with configurable logic/ethics weighting, explicit logical criteria, and mathematical verdict enforcement that can override LLM bias
- **Two-phase belief updates**: Phase 1 (Enforcement) applies mandatory patches from adjudication outcomes; Phase 2 (Introspection) enables strategic position building with strength-increase filtering
- **Defense boost system**: Formula-driven strength increases for nodes that survive adversarial challenges, with cascade propagation through dependency graphs
- **Component-wise embedding**: 7,691-dimensional belief vectors combining strength-weighted semantic averages with normalized scalar features
- **Trajectory visualization**: Track belief evolution through semantic space using UMAP and PCA
- **Custom belief loading**: Agents can load pre-defined CBS beliefs from JSON files, skipping Stage 1
- **Performance scoring**: Agent Performance Score (APS) based on normalized exchange-level role-weighted scores in [-1.0, +1.0]
- **Training data export**: JSONL export of belief formation and belief update pairs for fine-tuning
- **Interactive CLI**: Configuration wizard with preset system, debate history tracking, and 23 context-sensitive help panels
- **Parallel execution**: ThreadPoolExecutor-based concurrent API calls with thread-safe key rotation

---

## 2. Architecture Overview

### Directory Structure

```
CHAL/
├── src/chal/
│   ├── agents/                    # LLM agent implementations & framework definitions
│   │   ├── base.py                # Abstract Agent class and Message dataclass
│   │   ├── openai_agent.py        # OpenAI provider (GPT-4o, o-series)
│   │   ├── anthropic_agent.py     # Anthropic provider (Claude models)
│   │   ├── google_agent.py        # Google provider (Gemini models)
│   │   ├── ollama_agent.py        # Ollama provider (local models)
│   │   ├── xai_agent.py           # xAI provider (Grok models, gRPC-based)
│   │   ├── perplexity_agent.py    # Perplexity provider
│   │   ├── factory.py             # Agent creation factory (lazy imports per provider)
│   │   ├── prompts.py             # ALL stage prompt builders + position analysis + re-exports personas
│   │   ├── epistemic_personas.py  # 13 epistemic persona definitions & lookup
│   │   ├── logic_systems.py       # 8 logic system definitions for adjudicator
│   │   └── ethics_systems.py      # 6 ethics system definitions for adjudicator
│   ├── beliefs/                   # CBS belief system
│   │   ├── schema.py              # CBS_JSON_SCHEMA, ALLOWED_REF_PREFIXES, validate_belief(), validate_inference_chain()
│   │   ├── belief_graph.py        # DAG representation, structural analysis, critical path detection
│   │   ├── patches.py             # Deterministic patch application, validate_patches(), strength propagation pipeline
│   │   ├── io.py                  # Parse/render beliefs (JSON ↔ Markdown ↔ Embedding), project_for_component_embedding()
│   │   └── graph_visualizer.py    # Interactive Cytoscape.js visualization
│   ├── orchestrator/              # Debate pipeline
│   │   ├── debate_controller.py   # 6-stage orchestrator (CORE), DebateMetrics, two-phase Stage 5, defense boosts, anti-repetition
│   │   └── adjudicator.py         # Independent argument evaluator, enforce_verdict(), validate_adjudicator_output()
│   ├── cli/                       # Interactive CLI
│   │   ├── main.py                # Entry point, argument parsing, 5 CLI modes
│   │   ├── wizard.py              # Interactive configuration wizard (9 steps, 23 help panels)
│   │   ├── runner.py              # Debate execution and output saving
│   │   ├── display.py             # Rich terminal UI, event-driven display system
│   │   ├── api_keys.py            # API key validation and key pool creation
│   │   └── history.py             # Debate history tracking (~/.chal/)
│   ├── embeddings/                # Belief trajectory tracking
│   │   ├── embedding_tracker.py   # Component-wise 7,691-dim embeddings via SentenceTransformer
│   │   └── embedding_visualizer.py # UMAP/PCA reduction, persona colormaps, provider logos
│   ├── configurations/            # YAML debate presets
│   │   └── default.yaml           # Standard configuration
│   ├── utilities/                 # Shared utilities
│   │   ├── parallel.py            # ParallelDispatcher (ThreadPoolExecutor wrapper)
│   │   ├── retry.py               # retry_api_call(), generate_with_retry(), ValidationResult, RetryRecord
│   │   ├── key_pool.py            # Thread-safe API key rotation with rate-limit cooldown
│   │   ├── validators.py          # Stage-specific output validators (stages 1, 2, 3, 5p1, 5p2)
│   │   ├── training_data.py       # DebateRecorder for JSONL training data export
│   │   ├── reporting.py           # Analysis report generation (Markdown + JSON)
│   │   ├── debug_log_writer.py    # Thread-safe real-time debug log writer + Python logging bridge
│   │   └── utils.py               # VALID_ATTACK_STRATEGIES, EXCHANGE_SCORE_WEIGHTS, stats, parsing helpers
│   ├── config.py                  # Configuration dataclasses & YAML loader
│   ├── constants.py               # PROVIDER_ENV_VARS mapping
│   ├── log.py                     # Logging setup
│   └── assets/logos/              # Provider logo PNGs for trajectory visualization
├── tests/                         # Comprehensive test suite (46 files, 250+ tests)
├── docs/                          # Documentation
│   └── documentation.md           # This file
└── pyproject.toml                 # Package configuration (Poetry)
```

### Data Flow

```
User Input (topic, personas, config)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                   DebateController.run()                  │
│                                                          │
│  Stage 0: Briefing ────────────────────────────────────► │
│       │  (universal prompt + persona role cards)          │
│       ▼                                                  │
│  Stage 1: Opening Positions ───────────────────────────► │
│       │  (agents generate CBS beliefs OR load from file) │
│       │  (defense tracking initialized)                  │
│       ▼                                                  │
│  ┌─── Round Loop (1..N) ──────────────────────────────┐  │
│  │  Stage 2: Cross-Examination                        │  │
│  │    │  (targeted questions with 27 attack strategies)│  │
│  │    │  (anti-repetition from prior rounds) ◄────┐   │  │
│  │    ▼                                           │   │  │
│  │  Stage 3: Rebuttals                            │   │  │
│  │    │  (refute / concede / defer + patches)     │   │  │
│  │    ▼                                           │   │  │
│  │  Stage 4: Adjudication                         │   │  │
│  │    │  (independent evaluation)                 │   │  │
│  │    │  (mathematical verdict enforcement)       │   │  │
│  │    ▼                                           │   │  │
│  │  Stage 5: Belief Updates                       │   │  │
│  │    │  Phase 1: Enforcement patches ──────────► │   │  │
│  │    │  Defense Boosts (mechanical)              │   │  │
│  │    │  Phase 2: Introspection ──────────────────┘   │  │
│  │    │  (strength-increase filtering applied)        │  │
│  └────────────────────────────────────────────────────┘  │
│       │                                                  │
│       ▼                                                  │
│  Output Artifacts                                        │
└──────────────────────────────────────────────────────────┘
```

---

## 3. The CHAL Belief Schema (CBS)

The CBS is the formal JSON structure used to represent agent beliefs throughout the debate. It is the central data structure that gets refined through the pipeline.

### Required Top-Level Keys

Every CBS object must contain **11 required top-level keys**:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be exactly `"CBS"` (single-value enum) |
| `belief_id` | string | Unique identifier (e.g., `"BELIEF-<agent>-001"`) |
| `version` | integer | Monotonically increasing revision number (starts at 1) |
| `metadata` | object | Contains `topic_query`, `agent_persona` |
| `thesis` | object | Contains `stance` (string), `summary_bullets` (array, minItems: 1), `strength` (0.0-1.0), `strength_reasoning` (string, auto-generated formula breakdown) |
| `definitions` | array | D# nodes — semantic bedrock of the belief |
| `assumptions` | array | A# nodes — typed premises |
| `claims` | array | C# nodes — primary assertions with inline predictions |
| `evidence` | array | E# nodes — supporting evidence |
| `counterpositions` | array | X# nodes — strongest known arguments against the position |
| `uncertainties` | array | U# nodes — unresolved questions |

### Optional Top-Level Key

| Field | Type | Description |
|-------|------|-------------|
| `changelog` | array | Array of `{version: int, changes: [str]}` entries. Auto-generated by the patch system. |

### Definitions (D#)

Definitions are the **semantic bedrock** of the CBS. Each definition pins down the meaning of a key term used elsewhere in the belief, preventing equivocation.

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (D1, D2, etc.) |
| `term` | Yes | The term being defined (immutable after creation) |
| `definition` | Yes | The definition text |
| `strength` | Yes | Calibrated strength (0.0-1.0) — how well-supported this definition is |
| `strength_justification` | Yes | Rationale for the strength value |
| `status` | Yes | One of: `active`, `revised`, `retracted` |
| `used_by` | Yes | Array of A# or E# IDs that depend on this definition (minItems: 1) |
| `original_strength` | No | Immutable snapshot of initial strength (system-managed for defense boost) |
| `consecutive_defenses` | No | Count of consecutive successful defenses (system-managed for defense boost) |

Definitions are linked to assumptions and evidence through the `supported_by_definitions` cross-reference field, creating a semantic foundation layer in the dependency graph. The system enforces **bidirectional consistency**: if D1 lists A1 in `used_by`, then A1 must list D1 in `supported_by_definitions`, and vice versa.

### Assumptions (A#)

Each assumption must declare its type, which determines how it can be challenged:

| Type | Meaning | Can Be Challenged By |
|------|---------|---------------------|
| `foundational` | Definitional or logical axioms | Showing incoherence only |
| `empirical` | Assumed true based on evidence | Counter-evidence |
| `methodological` | Adopted for analytical purposes | Questioning the method |
| `scoping` | Scope-limiting commitments | Questioning the scope boundaries |

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (A1, A2, etc.) |
| `type` | Yes | One of: `foundational`, `empirical`, `methodological`, `scoping` |
| `statement` | Yes | The assumption text |
| `supports_claims` | Yes | Array of C# IDs this assumption supports |
| `strength` | Yes | Calibrated strength (0.0-1.0) |
| `status` | Yes | One of: `active`, `revised`, `retracted` |
| `strength_justification` | Yes | Rationale for the strength value |
| `supported_by_definitions` | Yes | Array of D# IDs that define key terms (minItems: 1) |
| `original_strength` | No | Immutable snapshot of initial strength (system-managed) |
| `consecutive_defenses` | No | Count of consecutive successful defenses (system-managed) |

This typing matters for cross-examination: an assumption labeled "foundational" that is actually empirical can be challenged with evidence, and agents are prompted to identify such misclassifications.

### Claims (C#)

Claims are the primary assertions in a belief, forming the backbone of the dependency graph. Each claim declares what it depends on and what evidence backs it, creating the DAG structure that enables strength propagation and vulnerability analysis.

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (C1, C2, etc.) |
| `type` | Yes | Free-form claim category (e.g., "deductive", "inductive", "abductive") — no enum restriction |
| `statement` | Yes | The substantive assertion |
| `depends_on` | Yes | Array of A#, E#, or C# IDs this claim builds on (graph edges) |
| `backing_evidence_ids` | Yes | Array of E# IDs supporting this claim (graph edges) |
| `strength` | Yes | Calibrated strength in [0.0, 1.0] — must not exceed weakest dependency strength |
| `strength_justification` | Yes | Rationale for the strength value |
| `status` | Yes | One of: `active` (current), `revised` (modified during debate), `retracted` (abandoned) |
| `inference_chain` | Yes | Step-by-step reasoning from premises to conclusion (minItems: 3). Each step has `role` (premise/inference/conclusion), `text`, and optionally `reference` (for premises, matching `^[ACE]\d+$`) or `inference_type` (for inference steps: deductive/inductive/abductive). Ordering enforced: all premises → exactly 1 inference → exactly 1 conclusion. Validated by `validate_inference_chain()` |
| `predictions` | Yes | Array of inline predictions (minItems: 1). Each has: `statement`, `test`, `decision_criterion`, and optional `potential_falsifiers`. Predictions operationalize the claim by specifying what should be true if the claim holds |

The `status` field is critical for the patch system: setting status to `"retracted"` via `update_claim` forces strength to 0.0 automatically (retraction enforcement), triggering strength propagation through all dependent claims.

### Evidence (E#)

Evidence items support claims through the `backing_evidence_ids` relationship. Each evidence item is typed to distinguish its epistemic weight.

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (E1, E2, etc.) |
| `type` | Yes | One of: `empirical` (observational/experimental data), `conceptual` (philosophical/theoretical arguments), `expert_consensus` (authoritative agreement) |
| `summary` | Yes | Description of the evidence content |
| `source` | Yes | Citation or provenance string |
| `supports_claims` | Yes | Array of C# IDs this evidence supports |
| `strength` | Yes | Calibrated strength (0.0-1.0) |
| `strength_justification` | Yes | Rationale for the strength value |
| `status` | Yes | One of: `active`, `revised`, `retracted` |
| `supported_by_definitions` | Yes | Array of D# IDs that define key terms (minItems: 1) |
| `quality_assessment` | No | Free-text quality evaluation |
| `limitations` | No | Caveats about the evidence |
| `original_strength` | No | Immutable snapshot of initial strength (system-managed) |
| `consecutive_defenses` | No | Count of consecutive successful defenses (system-managed) |

### Counterpositions (X#)

Counterpositions are a critical CBS component that distinguishes CHAL beliefs from simple argument structures. Each counterposition represents the **strongest known argument against** the agent's own position:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (X1, X2, etc.) |
| `targets` | Yes | Array of C#, A#, E#, or D# IDs this counterposition attacks |
| `attack_type` | Yes | One of: `undermining` (challenges premise), `rebutting` (presents counter-evidence), `undercutting` (challenges inference step) |
| `attack_strategy` | Yes | Specific strategy from the 27-strategy taxonomy (e.g., `challenge_evidence`, `exploit_counterposition`, `identify_circularity`) validated against `VALID_ATTACK_STRATEGIES` |
| `statement` | Yes | The counterposition argument |
| `my_response` | Yes | The agent's current response to this counterposition |
| `response_sufficiency` | Yes | One of: `sufficient` (fully addressed), `partial` (partially addressed), `unaddressed` (no response yet), `moot` (target retracted — terminal state, cannot be changed once set) |

Agents must include at least 2 counterpositions. Counterpositions rated "partial" or "unaddressed" become strategic targets during cross-examination — agents are prompted to exploit these self-identified vulnerabilities.

### Uncertainties (U#)

Uncertainties capture unresolved questions that the agent recognizes but cannot currently answer. They serve as honest signals of epistemic limits and help prioritize future inquiry.

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable identifier (U1, U2, etc.) |
| `question` | Yes | The unresolved question |
| `targets` | Yes | Array of A#, E#, C#, or D# IDs this uncertainty pertains to |
| `importance` | Yes | One of: `high`, `medium`, `low` — how much resolving this would affect the overall belief |
| `status` | Yes | One of: `active`, `resolved` |
| `resolution_note` | Conditional | Required if `status` is `"resolved"` — explanation of how the uncertainty was resolved |

In Stage 5, `UNRESOLVED` adjudication outcomes must be recorded as new uncertainties, ensuring the belief honestly reflects where the debate could not reach resolution.

### Key Design Principles

1. **Stable IDs**: Every item has a stable ID (D1, A1, C3, E5, X2, etc.) that persists across belief versions, enabling precise cross-referencing during cross-examination and adjudication.
2. **Dependency tracking**: Claims declare what they `depends_on` (other claims, assumptions, or evidence) and what `backing_evidence_ids` support them, forming a directed acyclic graph (DAG). Definitions provide the semantic foundation through `supported_by_definitions` links.
3. **Strength calibration**: Every strength-bearing node (D#, A#, C#, E#) has a strength value in [0,1] that respects dependencies — a claim's strength cannot exceed its weakest dependency. The thesis strength is computed from active claims via the formula `avg(strengths) × breadth`.
4. **Required vs optional fields**: The schema distinguishes between structurally required fields (used by graph construction, strength propagation, and patch operations) and optional enrichment fields (used only for display). This design makes the system robust to imperfect LLM output while still encouraging rich responses.
5. **Self-critical honesty**: Counterpositions (X#) require agents to engage with the strongest arguments against their position. A belief that honestly rates its counterposition responses as "partial" is stronger than one that claims all responses are "sufficient."
6. **JSON-first**: All LLM outputs are JSON. Human-readable Markdown is generated programmatically by `belief_to_markdown()` to minimize token usage (~40-50% savings).

### Validation

The `schema.py` module validates CBS objects at two levels:

**Structural validation** (JSON Schema + custom checks):
- All 11 required top-level keys present (`schema_version`, `belief_id`, `version`, `metadata`, `thesis`, `definitions`, `assumptions`, `claims`, `evidence`, `counterpositions`, `uncertainties`)
- `schema_version` must equal exactly `"CBS"` (single-value enum)
- `version` must be an integer ≥ 1
- `thesis.strength` must be in [0.0, 1.0]
- `thesis.summary_bullets` must be non-empty (minItems: 1)
- Type checks for fields that must be dicts/arrays
- ID format (prefix + digits, e.g., `C1`, `E12`, `D3`)
- ID prefix matches collection (C# in claims, E# in evidence, D# in definitions, etc.)
- Sequential ID ordering within each collection (`_validate_sequential_ids()`)
- No duplicate IDs across collections
- Cross-reference integrity via `_validate_ref_prefixes()` — validates that all IDs in a reference list have allowed prefix letters (e.g., `depends_on` only allows A#, E#, C#)
- Bidirectional consistency between `D#.used_by` and `A#/E#.supported_by_definitions`
- If `jsonschema` library is installed (`HAVE_JSONSCHEMA` flag), full JSON Schema validation of all item-level schemas runs in addition to manual checks. Graceful degradation if not installed
- `validate_inference_chain()` performs deep structural validation of claim inference chains: premise ordering, reference format matching `^[ACE]\d+$`, inference_type validation

**Enum and bounds validation** (works without jsonschema):
- **Definition status**: must be one of `active`, `revised`, `retracted`
- **Assumption types**: must be one of `foundational`, `empirical`, `methodological`, `scoping`
- **Claim status**: must be one of `active`, `revised`, `retracted`
- **Claim strength**: must be in [0.0, 1.0]
- **Evidence type**: must be one of `empirical`, `conceptual`, `expert_consensus`
- **Evidence status**: must be one of `active`, `revised`, `retracted`
- **Counterposition attack_type**: must be one of `undermining`, `rebutting`, `undercutting`
- **Counterposition response_sufficiency**: must be one of `sufficient`, `partial`, `unaddressed`, `moot`
- **Uncertainty importance**: must be one of `high`, `medium`, `low`
- **Uncertainty status**: must be one of `active`, `resolved`

The `belief_graph.py` module performs graph-level structural validation:
- **Broken links**: All `depends_on` and `backing_evidence_ids` reference existing nodes
- **Circular dependencies** (BLOCKING): Claims cannot depend on themselves directly or indirectly (detected via DFS recursion stack tracking)
- **Orphaned claims** (BLOCKING): Claims must have at least one supporting edge (evidence or assumption)

Blocking errors trigger a retry loop (up to 3 attempts) where the agent is asked to revise its belief structure with a revision prompt explicitly listing the fix requirements. Non-blocking warnings are logged but don't trigger retries.

---

## 4. Agent System

### Abstract Base Class

All agents inherit from `Agent` (in `base.py`) and must implement:

```python
class Agent(ABC):
    def generate(self, history: List[Message]) -> Message: ...
    def receive_system_prompt(self, prompt: str) -> None: ...
    def receive_role_card(self, prompt: str) -> None: ...
```

The `Message` dataclass carries `role` ("system"/"user"/"assistant"), `content` (text), and optional `metadata`.

The `receive_role_card()` method **appends** to the agent's system prompt (does not replace it), allowing both universal debate rules and persona-specific guidance to coexist.

### Provider Implementations

| Provider | Class | File | Supported Models | SDK | Client Init |
|----------|-------|------|-----------------|-----|-------------|
| OpenAI | `OpenAIAgent` | `openai_agent.py` | gpt-4o, gpt-4o-mini, o4-mini, o3-mini, o3 | `openai` (v1+) | Lazy |
| Anthropic | `AnthropicAgent` | `anthropic_agent.py` | claude-opus-4-6, claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001 | `anthropic` | Eager |
| Google | `GoogleAgent` | `google_agent.py` | gemini-2.5-pro, gemini-2.5-flash | `google-genai` | Eager |
| Ollama | `OllamaAgent` | `ollama_agent.py` | Any local model (e.g., deepseek-r1:14b) | `ollama` | — |
| xAI | `XAIAgent` | `xai_agent.py` | grok-3, grok-3-mini | `xai-sdk` | Lazy |
| Perplexity | `PerplexityAgent` | `perplexity_agent.py` | sonar-reasoning-pro, sonar-reasoning | `perplexityai` | Lazy |

**Client initialization patterns**:
- **Lazy** (OpenAI, xAI, Perplexity): `self._client = None`, created on first `generate()` call. Lazy init allows key rotation to simply reset `self._client = None`.
- **Eager** (Anthropic, Google): Client created immediately in constructor.

**System prompt handling** varies by provider:
- OpenAI/Perplexity: System prompt inserted as first message in messages list
- Anthropic: System prompt passed as separate `system=` API parameter (NOT in messages)
- Google: System prompt passed as `system_instruction=` in `GenerateContentConfig`; role mapping converts `assistant` → `model`
- Ollama: System prompt inserted as first message
- xAI: Uses `xai_system()` wrapper function

Provider selection is handled by the `factory.py` module with **lazy imports** — only the SDK for the selected provider needs to be installed. The factory provides two entry points:

```python
# By explicit parameters
agent = create_agent(name="Agent-Skeptic", model="gpt-4o", provider="openai")

# From an AgentConfig dataclass
agent = create_agent_from_config(agent_config)
```

The factory also accepts an optional `key_pool` parameter (`KeyPool` instance) for multi-key rotation and an optional `max_tokens` parameter (default 65536, passed to OpenAI and Anthropic agents only).

### Agent Internal State

Each agent maintains:
- `internal_belief`: Human-readable Markdown string of current belief
- `internal_belief_obj`: Parsed CBS JSON dict
- `all_beliefs_held`: Chronological list of all belief snapshots (JSON strings) for trajectory tracking
- `belief_graph`: `BeliefGraph` instance (auto-rebuilt when `set_internal_belief_obj()` is called)
- `persona_label`: String label of the persona, automatically extracted from agent name: `name.split("Agent-", 1)[-1]` if name contains "Agent-", otherwise uses full name

### Reasoning Models

The OpenAI agent automatically handles model-specific requirements. For reasoning models (o-series: o1, o3, o4), the `temperature` parameter is omitted from API calls, and the SDK handles mapping the `system` role to `developer` as needed. This is detected via `model.startswith(("o1", "o3", "o4"))`.

### Retry Logic

All providers (except Ollama) use the generic `retry_api_call()` function from `retry.py`, which provides:
- Exponential backoff retry (up to 5 attempts, base delay 60 seconds, doubling each attempt)
- Configurable error types per provider
- Key rotation on rate limits (resets lazy client via `self._client = None`)
- Rate limit marking on the `KeyPool`

**Provider-specific error handling**:
- Google: `_GoogleRateLimitError` sentinel wraps 429 status codes detected from `genai_errors.APIError`
- xAI: `_XAIRateLimitError` (gRPC RESOURCE_EXHAUSTED) and `_XAIRetryableError` (UNAVAILABLE, DEADLINE_EXCEEDED)

**Ollama** does NOT use `retry_api_call()`. It has its own `retry_ollama_chat()` with exponential backoff (base delay 10s, doubles each attempt, max 5 retries) and immediate failure for 404 (model not pulled) and connection refused (server not running).

---

## 5. Epistemic Personas

**File**: `src/chal/agents/epistemic_personas.py`

Epistemic personas define the philosophical worldviews that shape how agents evaluate evidence, construct arguments, and engage with opposing positions. Each persona is a system prompt string grounded in specific philosophical traditions.

### Module Structure

The module provides:
- **13 persona prompt strings**: Each as a triple-quoted text block defining the agent's epistemological worldview
- **`PERSONAS` dict**: Maps uppercase keys (e.g., `"EMPIRICIST"`) to prompt strings
- **`PERSONA_LABELS` dict**: Maps keys to human-readable display names (e.g., `"Empiricist"`)
- **`PERSONA_DESCRIPTIONS` dict**: Maps keys to one-line summaries for CLI display
- **`get_persona(key)` function**: Case-insensitive lookup by key

### The 13 Personas

| Key | Display Label | Philosophical Tradition | Core Epistemological Commitment |
|-----|--------------|------------------------|-------------------------------|
| `EMPIRICIST` | Empiricist | Hume, logical positivists, modern scientific methodology | All substantive knowledge traces back to sensory observation, experiment, or measurable evidence. Favors inductive/abductive reasoning. Treats unfalsifiable claims as meaningless. |
| `SUPERNATURALIST` | Supernaturalist | Classical theistic arguments (cosmological, teleological, moral) | Reality encompasses dimensions beyond empirical science. Faith and reason are complementary. Takes spiritual experience as a genuine epistemic source. |
| `SKEPTIC` | Skeptic | Pyrrhonian tradition | For any claim, equally compelling arguments exist for and against. Presses the regress problem. Treats certainty as a psychological state with no necessary connection to truth. |
| `RATIONALIST` | Rationalist | Descartes, Leibniz, Spinoza | Reason and logical deduction are the primary knowledge sources. Certain truths are knowable a priori. Trusts clear and distinct rational intuition. |
| `PHENOMENOLOGIST` | Phenomenologist | Husserl, Merleau-Ponty | Philosophy begins with structures of conscious experience prior to theoretical interpretation. First-person experience is an irreducible knowledge source. |
| `PRAGMATIST` | Pragmatist | Peirce, James, Dewey | Meaning lies in practical consequences. Truth emerges through inquiry and is validated by outcomes. All beliefs are provisional hypotheses. |
| `CONSTRUCTIVIST` | Constructivist | Kuhn, Berger & Luckmann, social epistemology | Knowledge is actively constituted through social practices, linguistic categories, and power relations. Claims to 'objectivity' can mask particular perspectives. |
| `NIHILIST` | Nihilist | Nietzsche's critique of metaphysical foundations | No inherent meaning, purpose, or objective moral truths. Distinguishes ontological, moral, and epistemological nihilism. Human meaning-making is a psychological coping mechanism. |
| `BAYESIAN` | Bayesian | Bayesian epistemology, probability theory | Rational belief is probabilistic inference governed by Bayes' theorem. All beliefs should carry explicit uncertainty estimates. Dutch book arguments demand probabilistic coherence. |
| `PANPSYCHIST` | Panpsychist | Chalmers, Goff, Integrated Information Theory, Russellian monism | Consciousness is a fundamental and ubiquitous feature of reality. Motivated by the hard problem of consciousness. Distinguishes from animism or mysticism. |
| `SIMULATIONIST` | Simulationist | Bostrom's trilemma, simulation hypothesis | Takes seriously the possibility we live in a simulation. Evaluates claims through the lens of what would be true in a simulated reality. |
| `SYNTHESIST` | Synthesist | Ken Wilber's integral theory, metamodern philosophy, systems thinking | Seeks to integrate partial truths from each tradition. Uses the principle of non-exclusion. Resists both reductionism and relativism. |
| `NONE` | None | — | No epistemic lens applied. Agent argues strictly from content, logic, and evidence without a philosophical worldview shaping its analysis. |

### Integration Points

Personas integrate into the debate pipeline at multiple points:

1. **Stage 0 (Briefing)**: The persona prompt is applied to each agent via `receive_role_card()`, which appends it to the agent's system prompt. When persona is empty (NONE), the agent is instructed to "argue strictly from content, logic, evidence."
2. **Stage 1 (Opening Positions)**: The persona label is included in the belief formation prompt, and the `metadata.agent_persona` field records it in the CBS object.
3. **CLI Wizard**: Personas appear as selectable choices with one-line descriptions from `PERSONA_DESCRIPTIONS`.
4. **Agent name auto-generation**: Agent names are auto-generated as `"Agent-{Persona.capitalize()}"` if not explicitly provided.
5. **Backward Compatibility**: `prompts.py` re-exports all persona symbols via `from chal.agents.epistemic_personas import *`, so existing code that does `from chal.agents.prompts import EMPIRICIST` continues to work.

### Design Philosophy

Each persona is designed to be:
- **Philosophically specific**: Grounded in named traditions and thinkers, not vague attitudes
- **Epistemologically distinct**: Each has genuinely different standards for what counts as evidence and justification
- **Internally coherent**: The worldview doesn't contradict itself
- **Debate-relevant**: The persona shapes how agents form beliefs, challenge opponents, and evaluate concessions
- **Non-dogmatic**: Agents are instructed (via the position prompt) to use their worldview as a lens, not a set of conclusions to defend at all costs

---

## 6. Logic Systems

**File**: `src/chal/agents/logic_systems.py`

Logic systems configure the logical framework the adjudicator uses to evaluate argument validity. The selected logic system is injected into the adjudicator's system prompt, shaping how it evaluates deductive validity, inductive strength, and the handling of contradictions.

### Module Structure

Each logic system is a dict with:
- **`label`**: Human-readable display name
- **`description`**: Full description of the reasoning framework
- **`criteria`**: Dict with `critique_valid`, `rebuttal_valid`, `unresolved` lists — each containing specific conditions under which that verdict applies

The module provides:
- **`LOGIC_SYSTEMS` dict**: Maps uppercase keys to logic system dicts
- **`get_logic_system(key)`**: Case-insensitive lookup returning the full dict
- **`get_logic_system_description(key)`**: Returns just the description string
- **`get_logic_system_label(key)`**: Returns the human-readable label

### Available Logic Systems

| Key | Display Label | Description | Criteria Scale |
|-----|--------------|-------------|----------------|
| `CLASSICAL_INFORMAL_BAYESIAN` | Classical + Informal + Bayesian (Hybrid) | Comprehensive hybrid: formal deductive validity as the foundation, informal fallacy detection as the practical layer, and Bayesian reasoning as the scientific evidence-evaluation framework. Evaluates deductive validity, inductive strength, abductive coherence, evidence quality, and reasoning hygiene. **This is the default and recommended system.** | 21 critique / extensive rebuttal / 3 unresolved |
| `FORMAL_DEDUCTIVE` | Formal Deductive | Strict formal deductive logic only. Only valid syllogisms and formally sound inferences accepted. Rejects inductive and abductive reasoning entirely. | Fewer criteria |
| `BAYESIAN` | Bayesian | Pure Bayesian reasoning. Evaluates arguments through prior probabilities, likelihood ratios, and posterior updates. Focuses on evidence quality and probabilistic coherence. | — |
| `INFORMAL_CRITICAL` | Informal / Critical Thinking | Informal logic and critical thinking. Evaluates via fallacy identification, relevance, and sufficiency of evidence. Accepts inductive and abductive reasoning when well-supported. | — |
| `DIALECTICAL` | Dialectical (Hegelian) | Hegelian thesis-antithesis-synthesis. Contradictions are productive and drive toward higher understanding. Evaluates whether opposing positions can be sublated into a more comprehensive synthesis. | — |
| `FUZZY_MULTIVALUED` | Fuzzy / Multi-valued | Truth admits degrees between 0 and 1. Avoids binary true/false judgments. Evaluates the degree to which premises support conclusions. | — |
| `PARACONSISTENT` | Paraconsistent | Tolerates local contradictions without global explosion (ex contradictione quodlibet does not apply). Evaluates whether contradictions are contained and productive vs. undermining. | 5 critique criteria |
| `NONE` | None (Pure Ethics) | No logical framework applied. Used when `logic_weight` is 0 and only ethical evaluation applies. Has empty criteria lists. | Empty criteria |

### Integration

The logic system is configured in `AdjudicationConfig.logic_system` (default: `"CLASSICAL_INFORMAL_BAYESIAN"`). At debate startup, the `DebateController` calls `get_logic_system()` to resolve the key to the full system dict, whose description is then passed to `build_adjudicator_prompt()` and injected into the adjudicator's evaluation framework. The criteria lists are used to build the specific conditions for each verdict type.

The choice of logic system materially affects adjudication outcomes:
- **FORMAL_DEDUCTIVE** will reject arguments that rely on inductive generalizations, even if well-supported
- **DIALECTICAL** evaluates whether opposing positions can be sublated into a more comprehensive synthesis
- **PARACONSISTENT** will not automatically rule against a position that contains local contradictions
- **FUZZY_MULTIVALUED** will produce more "unresolved" outcomes when positions are partially supported

---

## 7. Ethics Systems

**File**: `src/chal/agents/ethics_systems.py`

Ethics systems configure the ethical framework the adjudicator uses alongside the logic system. When `ethics_weight > 0`, the adjudicator evaluates arguments on both logical rigor and ethical merit, combining scores using the configured weights.

### Module Structure

Each ethics system is a dict with:
- **`label`**: Human-readable display name
- **`description`**: Full description of the ethical evaluation approach
- **`criteria`**: Dict with `critique_valid`, `rebuttal_valid`, `unresolved` lists — each containing specific ethical conditions under which that verdict applies

The module provides:
- **`ETHICS_SYSTEMS` dict**: Maps uppercase keys to ethics system dicts
- **`get_ethics_system(key)`**: Case-insensitive lookup returning the full dict
- **`get_ethics_system_description(key)`**: Returns just the description string
- **`get_ethics_system_label(key)`**: Returns the human-readable label

### Available Ethics Systems

| Key | Display Label | Description |
|-----|--------------|-------------|
| `NONE` | None (Pure Logic) | No ethical framework applied. Evaluate arguments solely on logical rigor and soundness. **This is the default** (ethics_weight = 0.0). Has empty criteria lists. |
| `UTILITARIAN` | Utilitarian | Consequentialist utilitarianism: evaluate by whether conclusions maximize well-being and minimize suffering for the greatest number. Criteria include net harm, stakeholder neglect, utility miscalculation, short-termism, distributional blindness, and generalization failure. |
| `DEONTOLOGICAL` | Deontological (Kantian) | Kantian ethics: evaluate whether arguments respect universal moral duties, autonomy, and the categorical imperative regardless of consequences. Criteria include universalizability failure, instrumentalization, duty violation, autonomy infringement, rights violation, and consequentialist smuggling. |
| `VIRTUE_ETHICS` | Virtue Ethics (Aristotelian) | Aristotelian: evaluate whether arguments promote human flourishing (eudaimonia), practical wisdom, courage, and temperance. Criteria include vice promotion, phronesis failure, flourishing undermined, akrasia, character blindness, and inverted goods. |
| `CARE_ETHICS` | Care Ethics | Evaluate through the lens of relationships, responsibility, and attentiveness to context-dependent needs. Criteria include vulnerability neglect, abstraction over context, relational damage, unmet dependency, interchangeability assumption, and power asymmetry ignored. |
| `BALANCED` | Balanced (Rule-Utilitarian) | Weigh both outcomes/welfare and autonomy/rights. Neither consequences nor duties alone are sufficient. The most comprehensive system with 22 critique_valid criteria, 21 rebuttal_valid criteria, and 10 unresolved criteria. |

### Integration

The ethics system is configured in `AdjudicationConfig.ethics_system` (default: `"NONE"`) alongside `AdjudicationConfig.ethics_weight` (default: `0.0`). When `ethics_weight = 0.0` and `ethics_system = "NONE"`, the adjudicator operates in pure-logic mode. When ethics is enabled, both scores are computed and combined:

```
combined_score = logic_weight × logic_score + ethics_weight × ethics_score
```

### How Logic and Ethics Systems Work Together

The logic system and ethics system operate as **independent evaluation axes** within the adjudicator:

1. The adjudicator scores each side (challenger and defender) on both axes (0.0-1.0 each)
2. The combined score is computed mathematically using the configured weights
3. The outcome is determined by the gap between combined scores:
   - `critique_valid` if challenger combined - defender combined >= threshold (default 0.15)
   - `rebuttal_valid` if defender combined - challenger combined >= threshold
   - `unresolved` otherwise

This means a logically weak but ethically compelling argument can still prevail if ethics_weight is high enough, and vice versa. The system is fully transparent — all six individual scores are included in the adjudicator's output, and a mathematical verdict enforcement system ensures the LLM's stated verdict matches the computed scores.

---

## 8. The 6-Stage Debate Pipeline

The `DebateController` class in `debate_controller.py` orchestrates the entire pipeline. The `run()` method executes all stages sequentially, with Stages 2-5 repeating for each round. The controller tracks debate metrics via the `DebateMetrics` dataclass, including `total_input_tokens` and `total_output_tokens` across the entire debate.

All prompts use **XML-style tags** (`<context>`, `<instructions>`, `<output_format>`, etc.) for clear structural boundaries, and most include **`<reasoning>` tags** that prompt the agent to think through its response before producing structured output.

### Stage 0: Briefing

**Purpose**: Initialize each agent with shared rules and a unique persona.

**What happens**:
1. A **universal debate prompt** is generated and applied to all agents via `receive_system_prompt()`
2. A **debate context block** is generated by `build_debate_context()`, describing the 4-stage loop (cross-exam → rebuttal → adjudication → belief update)
3. A **persona-specific role card** is generated and applied to each agent via `receive_role_card()`

**Prompt: Universal Debate Prompt** (`build_universal_prompt`):
```
You are a philosophical agent participating in a structured debate on the topic:

  "{topic}"

<debate_protocol>
You are in the CHAL debate framework. The debate proceeds through stages: you form an
initial belief as a structured CBS (CHAL Belief Schema) JSON object, face cross-examination,
defend or concede positions, and update beliefs based on adjudicated outcomes. You will work
with stable IDs — D# (definitions), A# (assumptions), C# (claims), E# (evidence),
X# (counterpositions), U# (uncertainties) — that persist across rounds for precise
cross-referencing.
</debate_protocol>

<intellectual_standards>
1. Distinguish descriptive claims (what IS) from normative claims (what OUGHT). Label them
   explicitly.
2. Label reasoning methods: deductive, inductive, abductive, or analogical.
3. Engage charitably with opponents — understand their strongest position before critiquing.
4. Calibrate strength honestly. 0.9 means ~10% chance you're wrong — that requires very
   strong evidence. Most philosophical claims warrant 0.4-0.8.
5. Concede genuinely when a critique lands.
6. Take counterpositions seriously. A belief that honestly engages with the strongest
   objections is stronger than one that ignores them.
7. When convergence is reached, name the shared ground explicitly.
</intellectual_standards>

You will receive stage-specific instructions for each phase.
```

**Prompt: Position/Role Card** (`build_position_prompt`):
```
You are Agent {agent_name}. Your epistemological worldview:

<persona>
{persona}
</persona>

<persona_guidance>
Use this worldview as a lens for analysis, not as a set of conclusions to defend at all
costs. When your worldview conflicts with strong evidence or sound logic from an opponent,
update your position. Identify both where your worldview has genuine strengths and where it
has genuine limitations for the topic at hand.
</persona_guidance>
```

When persona is empty (NONE), the agent is instructed to "argue strictly from content, logic, evidence."

### Stage 1: Opening Positions

**Purpose**: Each agent generates their initial CBS belief object.

**What happens**:
1. Each agent receives the Stage 1 prompt requesting a complete CBS JSON belief
2. Agents with a `belief_file` configured skip LLM generation entirely — their pre-loaded belief object is used directly
3. For LLM-generated beliefs: the agent first reasons inside `<reasoning>` tags, then outputs a JSON code block
4. The response is parsed to extract the JSON code block
5. The JSON is validated against the CBS schema (all 11 required keys)
6. The belief graph is constructed and validated (broken links, cycles, orphans)
7. If blocking validation errors are found, the agent is asked to revise (up to 3 retries with a revision prompt explicitly listing the fix requirements)
8. Defense tracking is initialized via `initialize_defense_tracking()` — sets `original_strength` and `consecutive_defenses` on ALL strength-bearing nodes (D#, A#, C#, E#)
9. The belief is stored on the agent and Markdown is generated programmatically
10. `snapshot_belief()` captures thesis_strength and component_counts for each agent, stored in agent_stats

Stage 1 uses the **gather-then-apply pattern**: parallel LLM calls via `ParallelDispatcher`, then sequential application of results.

**Key prompt features** (`build_stage_1_belief_prompt_cbs`):
- Full CBS schema specification with all field definitions inside `<cbs_schema>` tags
- **7-step bottom-up generation order**: (1) A# assumptions → (2) E# evidence → (3) D# definitions → (4) C# claims → (5) X# counterpositions → (6) U# uncertainties → (7) Thesis
- Thesis strength formula: `avg(active_claim_strengths) × (n^p / (n^p + 1))`
- **Recommendation for 2-3+ independent claims** for breadth, as this affects the thesis strength formula
- Definition requirements (D# nodes for key terms)
- Assumption type taxonomy (foundational/empirical/methodological/scoping)
- Counterposition requirements (at least 2, with attack_type, attack_strategy, and response_sufficiency)
- Strength scale block mapping 0.0-1.0 to calibration labels
- A comprehensive `<example>` CBS object demonstrating expected quality
- `<instructions>` tag requiring the agent to think through its position inside `<reasoning>` tags before outputting JSON
- Dependency rules: no circular deps, strength respects weakest dependency, every claim needs support

### Stage 2: Cross-Examination

**Purpose**: Each agent generates targeted questions challenging every other agent's beliefs.

**What happens**:
1. For each agent pair (challenger → target), the challenger receives both beliefs as JSON
2. Optional vulnerability analysis from the target's belief graph is included
3. Anti-repetition context from previous rounds is included — prior challenges with their outcomes are passed to prevent agents from repeating failed attack strategies
4. The agent reasons inside `<reasoning>` tags about the opponent's strongest position and genuine vulnerabilities
5. The challenger generates up to N questions in JSON format, each with required fields
6. Questions are stored as individual challenge-rebuttal pair entries
7. Optional `targeted_claims_json` parameter enables focused cross-examination on specific claims

**Question output structure**: Each question includes:
- `qid`: Sequential identifier (Q1, Q2, ...)
- `text`: The question text
- `target_ids`: 1-2 node IDs being challenged
- `attack_type`: One of `undermining`, `rebutting`, `undercutting`
- `attack_strategy`: Specific strategy from the 27-strategy taxonomy

**Key prompt features** (`build_stage_2_prompt`):
- Both beliefs provided as JSON in `<your_belief>` and `<opponent_belief>` tags
- `<previous_round_questions>` section for anti-repetition
- `<attack_framework>` section providing the formal taxonomy of 27 attack strategies (see [Section 10](#10-cross-examination--attack-taxonomy))
- Questioning strategies with priority on exploiting partial counterpositions, challenging assumption types, questioning strength calibration, exposing dependency vulnerabilities
- `<example>` question demonstrating counterposition exploitation

**Fallback parsing**: Stage 2 has a legacy fallback parser `parse_challenges()` for backward compatibility with unstructured challenge text when JSON parsing fails.

### Stage 3: Rebuttals

**Purpose**: Each agent provides single-shot structured rebuttals to all questions they received.

**What happens**:
1. Questions from multiple challengers are **renumbered sequentially** (Q1, Q2, ..., QN) before being presented to the target agent, ensuring unique IDs
2. The agent receives all questions + their own belief as JSON
3. The agent reasons inside `<reasoning>` tags, honestly assessing each question
4. The agent produces a **single JSON block** containing both rebuttals and optional patches
5. Rebuttals are mapped back to challenge-rebuttal pair entries
6. Patches from Stage 3 are stored per-agent in `last_rebuttals_patches` and passed to Stage 5 Phase 1 as additional context

**Key prompt features** (`build_stage_3_structured_rebuttal_prompt`):
- Three binding action types with strict semantic requirements in `<instructions>`:
  - `refute`: Must argue AGAINST the challenge with evidence or logic
  - `concede`: Must acknowledge the weakness AND include a weakening patch. Defending your position while claiming to concede is a protocol violation
  - `defer`: Must explain what would resolve the uncertainty
- Good/bad `<examples>` showing the difference between genuine concession and "disguised refute"
- Combined `<output_format>` with both `rebuttals` and `patches` in a single JSON block
- Expanded patch operations including `add_counterposition` and `update_counterposition`
- Mandatory patches for concessions
- **Definition-specific challenge handling**: When challenges target D# nodes, the prompt provides specific guidance: refute, concede with `update_definition`, or defer with `add_uncertainty`
- **Legacy JSON fallback**: Parsing tries unified JSON block first (`{rebuttals: [...], patches: [...]}`), then falls back to two separate fenced JSON blocks (first = rebuttals, second = patches)

### Stage 4: Adjudication

**Purpose**: An independent neutral adjudicator evaluates each challenge-rebuttal pair and renders a verdict.

**What happens**:
1. **Incomplete pair handling**: Pairs missing either a challenge or rebuttal are skipped entirely with a warning, not dispatched to adjudication
2. For each complete challenge-rebuttal pair, the adjudicator receives both arguments plus belief excerpts extracted via `_extract_belief_excerpt()` — includes both directly targeted nodes AND nodes that depend on targets
3. The adjudicator reasons inside `<reasoning>` tags, analyzing step by step
4. Both arguments are formalized as syllogisms/inference chains
5. Six scores are computed (challenger/defender × logic/ethics/combined)
6. All scores are rounded to 4 decimal places via `round(x, 4)` to eliminate IEEE 754 floating-point noise
7. **Verdict enforcement**: `enforce_verdict()` recomputes combined scores mathematically and determines the "correct" verdict. If the LLM's verdict disagrees with the computed verdict, the computed verdict is used and an `override_occurred` flag is set
8. Verdicts are stored on the challenge-rebuttal pair entries

See [Section 12: Adjudicator System](#12-adjudicator-system) for the full adjudicator architecture.

### Stage 5: Belief Updates (Two-Phase)

**Purpose**: Each agent updates their CBS belief object based on adjudication outcomes. This is where **belief refinement** happens — the core goal of the pipeline. Stage 5 uses a **two-phase architecture** separating enforcement from strategic introspection.

#### Phase 1 — Enforcement

**Scope**: Respond ONLY to adjudication outcomes. No thesis rewrite, no strategic additions.

**What happens**:
1. Adjudication results are grouped by target agent
2. Each agent receives a prompt with their prior belief + all adjudication outcomes + Stage 3 patches from `last_rebuttals_patches`
3. The agent produces patch operations responding to each outcome
4. New counterposition `response_sufficiency` is capped at `"partial"` via `cap_phase1_counterposition_sufficiency()` (in `debate_controller.py`) — prevents agents from immediately claiming full resolution
5. Patches are validated via `validate_patches()` and applied deterministically with strength propagation through the dependency graph

**Key enforcement mechanisms** (in `<mandatory_rules>` tags):
- **CRITIQUE_VALID outcomes are mandatory**: At least one weakening patch per outcome. Empty patches after CRITIQUE_VALID is a protocol violation
- **New counterpositions for revealed vulnerabilities**: If a critique exposed a new weakness, the agent must add a counterposition (X#) recording it
- **UNRESOLVED outcomes**: Must add to uncertainties (U#)

#### Defense Boosts (Between Phases)

Applied mechanically after Phase 1 patches:
- **REBUTTAL_VALID**: increment `consecutive_defenses`, apply flat boost with cascade to dependency nodes
- **CRITIQUE_VALID**: reset `consecutive_defenses` to 0
- The `consecutive_defenses` counter persists across rounds — it tracks the full defense streak across the entire debate

See [Section 13: Defense Boost System](#13-defense-boost-system) for the full algorithm.

#### Phase 2 — Introspection

**Scope**: Strategic position building with the intermediate belief (post-Phase-1 + defense boosts).

**What happens**:
1. The agent receives the intermediate belief + Phase 1 change summary
2. **Dynamic position analysis** is computed from the intermediate belief via `compute_position_analysis()` and injected into the prompt as a `<position_analysis>` block containing:
   - Active claim strengths and thesis strength formula
   - Partial derivatives (∂T/∂s for strength gain, ∂T/∂n for breadth gain)
   - Scenario projections (raise avg strength, add claim, retract weakest)
   - Lowest-strength dependency identification
   - D# vulnerability analysis (weak definitions, bottleneck definitions)
   - Orphan detection
   - Strategic recommendation
3. A **breadth table** showing n claims (1-7) → breadth multiplier values is dynamically generated
4. The agent follows a three-step protocol: (1A) Counterposition Audit, (1B) Uncertainty Review, (2) Strategic Position Building, (3) Thesis Rewrite
5. The prompt includes three categories of **example patches**: Defensive (weaken and retract under pressure), Growth (add supporting infrastructure + new claim), Refinement (improve existing nodes textually + add support)
6. **Guardrail**: `filter_strength_increases()` (in `debate_controller.py`) strips strength increases on existing nodes from Phase 2 patches — prevents unilateral self-strengthening
7. Can add new nodes, revise text, retract weak positions
8. Thesis rewrite happens last, after all patches

**Patch application error handling**: If patch application fails (exception), the system reverts to the prior belief and logs an error. If graph validation fails after patches, the system also reverts. Individual invalid patches are skipped (not all-or-nothing).

**Legacy fallback**: `build_stage_5_belief_update_prompt_cbs()` still exists as a legacy single-phase flow for when `prior_json` is not available.

#### Supported Patch Operations (13 total)

| Operation | Target | Effect |
|-----------|--------|--------|
| `update_thesis` | thesis | Modify thesis strength and/or summary |
| `update_claim` | C# | Modify claim properties (strength, status, statement, depends_on, predictions, inference_chain, type). Setting status to "retracted" forces strength to 0.0 |
| `add_claim` | C# | Add new claim to the claims array |
| `add_evidence` | E# | Add new evidence item |
| `update_evidence` | E# | Modify evidence properties (strength, summary, source, status, supports_claims, type, supported_by_definitions) |
| `update_assumption` | A# | Modify assumption properties (strength, statement, status, type, supports_claims, supported_by_definitions) |
| `add_assumption` | A# | Add new assumption |
| `add_counterposition` | X# | Add new counterposition recording a vulnerability |
| `update_counterposition` | X# | Modify counterposition properties (my_response, response_sufficiency, statement, attack_type, targets) |
| `add_uncertainty` | U# | Add new uncertainty item |
| `resolve_uncertainty` | U# | Mark an uncertainty as resolved |
| `add_definition` | D# | Add new definition item |
| `update_definition` | D# | Modify definition properties (definition, strength, status, used_by). `id` and `term` are immutable |

---

## 9. Debate Mode

CHAL uses the **rebuttal** debate mode for Stage 3 argumentation.

| Mode | `stage3_mode` | Stage 3 Behavior | Stage 4 | Use Case |
|------|--------------|------------------|---------|----------|
| **Rebuttal** | `"rebuttal"` | Single-shot rebuttals with patch proposals | Runs separately | Default. Traditional debate format. |

Rebuttal mode is efficient (one API call per agent per round for rebuttals) and uses independent adjudication in Stage 4.

---

## 10. Cross-Examination & Attack Taxonomy

CHAL uses open cross-examination for Stage 2. Agents autonomously analyze opponent belief structures to identify structural and epistemic vulnerabilities, then generate targeted questions. There is no pre-planned topic structure — agents use their own analysis of the opponent's belief graph to determine where to focus their challenges.

### Attack Taxonomy — 27 Strategies

The cross-examination prompt includes a comprehensive attack taxonomy with **27 named strategies** across three attack types, defined in `VALID_ATTACK_STRATEGIES` in `utils.py`:

#### Undermining (10 strategies) — Challenge a premise or assumption directly

| Strategy | Description |
|----------|-------------|
| `challenge_evidence` | Challenge the quality, relevance, or interpretation of evidence |
| `challenge_assumption` | Challenge an assumption's validity or type classification |
| `expose_weak_foundation` | Target assumptions or evidence with low strength that support critical claims |
| `demand_falsifiability` | Require testable predictions or falsification criteria |
| `challenge_strength_calibration` | Argue that strength values are miscalibrated |
| `press_uncertainty` | Press on acknowledged uncertainties (U# nodes) |
| `over_extension` | Argue the claim extends beyond what the evidence supports |
| `under_extension` | Argue the claim fails to account for relevant cases |
| `challenge_moral_implications` | Challenge ethical consequences of the position |
| `expose_stakeholder_harm` | Identify harmed stakeholders not considered |

#### Rebutting (6 strategies) — Present counter-evidence or counter-conclusions

| Strategy | Description |
|----------|-------------|
| `present_counter_evidence` | Offer evidence that directly contradicts a claim |
| `present_counter_example` | Provide a specific counterexample |
| `exploit_counterposition` | Exploit self-identified weaknesses (X# with "partial"/"unaddressed" response_sufficiency) |
| `offer_alternative_explanation` | Propose a different explanation for the same evidence |
| `present_ethical_counter` | Present an ethical argument against the position |
| `invoke_competing_obligation` | Argue competing moral obligations outweigh the position |

#### Undercutting (11 strategies) — Challenge the inference step

| Strategy | Description |
|----------|-------------|
| `challenge_inference_step` | Challenge a specific step in the inference chain |
| `identify_circularity` | Identify circular reasoning (conclusion used as premise) |
| `expose_inconsistency` | Expose logical contradictions between claims |
| `identify_equivocation` | Identify a term used with different meanings |
| `challenge_scope` | Challenge whether the claim's scope is warranted |
| `circularity` | Identify circular dependencies in the belief graph |
| `stipulative_bias` | Challenge biased or loaded definitions (D# nodes) |
| `conceptual_conflation` | Argue distinct concepts are being conflated |
| `challenge_normative_inference` | Challenge the move from descriptive to normative claims |
| `expose_value_conflict` | Expose conflicts between stated values |
| `challenge_moral_relevance` | Challenge the moral relevance of cited considerations |

---

## 11. Belief Graph & Patch System

### Belief Graph (`belief_graph.py`)

The `BeliefGraph` class constructs a directed acyclic graph (DAG) from a CBS belief object:

**Nodes**:
- Definitions (D#)
- Assumptions (A#)
- Claims (C#)
- Evidence (E#)
- Counterpositions (X#)
- Uncertainties (U#)

**Edges** (derived from dependency declarations):
- `supports`: assumption/evidence/claim → dependent claim (from `depends_on`)
- `evidences`: evidence → supported claim (from `backing_evidence_ids`)
- `defines`: definition → assumption/evidence (from `used_by` / `supported_by_definitions`)
- `challenges`: counterposition → targets (from X# `targets`)
- `questions`: uncertainty → targets (from U# `targets`)

**Analysis Capabilities**:
- `validate_links()`: Check for broken links, circular dependencies (via DFS recursion stack tracking in `_has_cycle()`), orphaned claims
- `get_support_chain(node_id)`: Recursive backward traversal to find all nodes that transitively support a given node
- `get_dependent_nodes(node_id)`: Recursive forward traversal to find all nodes that transitively depend on a given node
- `find_critical_paths()`: Identify single-point-of-failure chains — nodes whose removal would break the support chain to the thesis. Returns list of critical path nodes
- `get_graph_metrics()`: Returns comprehensive structural metrics including `total_nodes`, `total_edges`, `critical_path_count`

### Graph Visualization (`graph_visualizer.py`)

The `export_debate_graph()` function generates an interactive standalone HTML visualization using Cytoscape.js:
- Nodes are color-coded by type:
  - Definition: #2AA198 (Teal)
  - Assumption: #3498db (Blue)
  - Claim: #e74c3c (Red)
  - Evidence: #2ecc71 (Green)
  - Prediction: #f39c12 (Orange)
- Edges show dependency relationships
- Click-to-inspect: clicking a node shows its full details
- Q&A overlay mapping debate question/answer pairs to graph nodes, showing attack type, strategy, agent responses, and resolution
- Fully self-contained (no external dependencies at runtime)

### Patch System (`patches.py`)

The `apply_patches()` function deterministically updates a belief object. It creates its working copy via `json.loads(json.dumps(prior_belief))` — full deep copy to avoid mutating the original.

#### Patch Validation (`validate_patches()`)

Before applying patches, comprehensive pre-flight validation runs:
1. **Build ID tracking** sets from existing belief
2. **Projection pass**: Pre-register IDs from all `add_*` patches before validation, allowing later patches in the same batch to reference newly added items (forward references)
3. **Per-patch validation** with detailed field-level checks:
   - All referenced target IDs exist in the belief
   - New IDs follow the correct format (prefix + number)
   - No duplicate IDs would be created
   - Required fields are present on new items
   - Strength values are within [0.0, 1.0]
   - Cross-references point to existing nodes
   - Immutable fields (like `id`, `term` on definitions) are not modified
   - System-managed fields (`original_strength`, `consecutive_defenses`) are automatically stripped from user-provided patch changes
4. **Cascade failure detection**: If an `add_*` patch fails validation, a transitive cascade loop flags all downstream patches that reference the failed patch's ID. Handles multi-hop chains (e.g., C3 fail → A5 cascade → D6 cascade)
5. **Non-fatal error handling**: Unknown patch operations are logged as warnings and skipped (not fatal)

**Field whitelists** — each update operation has a strict whitelist of modifiable fields:
- `_UPDATE_CLAIM_WHITELIST`: strength, strength_justification, statement, status, depends_on, predictions, inference_chain, type
- `_UPDATE_EVIDENCE_WHITELIST`: strength, strength_justification, summary, source, status, supports_claims, type, supported_by_definitions
- `_UPDATE_ASSUMPTION_WHITELIST`: strength, strength_justification, statement, status, type, supports_claims, supported_by_definitions
- `_UPDATE_COUNTERPOSITION_WHITELIST`: my_response, response_sufficiency, statement, attack_type, targets
- `_UPDATE_DEFINITION_WHITELIST`: definition, strength, strength_justification, status, used_by

**Validation enum constants** used throughout:
- `_STATUS_ENUM = {"active", "revised", "retracted"}`
- `_UNCERTAINTY_STATUS_ENUM = {"active", "resolved"}`
- `_IMPORTANCE_ENUM = {"high", "medium", "low"}`
- `_ASSUMPTION_TYPE_ENUM = {"foundational", "empirical", "methodological", "scoping"}`
- `_EVIDENCE_TYPE_ENUM = {"empirical", "conceptual", "expert_consensus"}`
- `_ATTACK_TYPE_ENUM = {"undermining", "rebutting", "undercutting"}`
- `_SUFFICIENCY_ENUM = {"sufficient", "partial", "unaddressed", "moot"}`

#### Strength Propagation Pipeline

When patches are applied, a three-step propagation pipeline runs:

**Step 1 — D# Ceiling Enforcement**:
For each non-retracted A# or E# with `supported_by_definitions`:
- Collect strengths of non-retracted D# nodes from `supported_by_definitions`
- If active D# exist: `ceiling = min(active_def_strengths)`; cap node strength at ceiling
- If D# referenced but all retracted: cap at `ORPHAN_AE_CAP = 0.6`
- Rounding: `round(ceiling, 4)`

**Step 2 — Claim BFS Propagation**:
1. Build worklist from all nodes with strength changes
2. For each changed node, find direct dependent claims via graph edges
3. For each dependent claim:
   - Collect strengths of active (non-retracted) dependencies
   - Skip retracted dependencies — they don't drag down dependent claims
   - If active deps exist: `claim.strength = min(active_dep_strengths)` (if current > min)
   - If all deps retracted: `claim.strength = ORPHAN_CLAIM_CAP = 0.2`
4. Queue affected claim for downstream propagation
5. Process until worklist empty (handles multi-hop cascades)

**Step 3 — Thesis Strength Calculation**:
```
avg_str = sum(active_claim_strengths) / n
breadth = n^p / (n^p + 1)
thesis_strength = round(avg_str × breadth, 4)
```
- `n` = number of active (non-retracted) claims
- `p` = `BREADTH_SENSITIVITY` (default 1.0, defined in `patches.py`, imported by `prompts.py` and `debate_controller.py`)

Auto-generated strength reasoning string example:
```
"avg(0.70, 0.55, 0.65) × (3^1.0 / (3^1.0 + 1)) = 0.63 × 0.75 = 0.47"
```

#### Additional Patch System Behaviors

- **Retraction enforcement** on all node types: Setting status to "retracted" on any D#, A#, C#, or E# node automatically forces its strength to 0.0, cascading through all dependents
- **Moot counterposition guard**: If `response_sufficiency` is "moot" (target retracted), it cannot be changed to another value — this is a terminal state
- **Definition update side effects**: When a definition's `used_by` field changes, `supported_by_definitions` on affected A#/E# nodes is automatically updated (bidirectional consistency maintenance)
- **Defense tracking initialization** (`initialize_defense_tracking()`): After Stage 1, sets `original_strength` and `consecutive_defenses` on all strength-bearing nodes
- **Changelog auto-generation**: Every patch application increments the belief version number and generates a changelog entry listing all changes (including propagated ones). `_summarise_ic_diff()` creates human-readable summaries of inference_chain changes
- **Strength propagation failures** are non-fatal — logged as warnings, patch application continues

---

## 12. Adjudicator System

**File**: `src/chal/orchestrator/adjudicator.py`

The adjudicator is a neutral, independent LLM agent that evaluates challenge-rebuttal pairs. It uses a detailed system prompt built by `build_adjudicator_prompt()` that integrates the configured logic system and ethics system.

### Adjudicator Class

The `Adjudicator` class wraps an LLM agent and operates in one of three modes determined by `_determine_mode()`:
- **`logic_only`** (ethics_weight < 0.01): Pure Logic evaluation — `_MODE_LABELS["logic_only"]` = "Pure Logic"
- **`ethics_only`** (logic_weight < 0.01): Pure Ethics evaluation — `_MODE_LABELS["ethics_only"]` = "Pure Ethics"
- **`balanced`** (both weights > 0): Balanced evaluation — `_MODE_LABELS["balanced"]` = "Balanced"

### Adjudicator System Prompt Structure

The adjudicator prompt is organized into XML-tagged sections and includes several scale/reference blocks:

**`_STRENGTH_SCALE_BLOCK`**: Maps 0.0-1.0 to calibration labels (Vacuous → Definitive)

**`_LOGIC_SCALE_BLOCK`**: Logic scoring guide (0.0 = No reasoning → 1.0 = Rigorous)

**`_ETHICS_SCALE_BLOCK`**: Ethics scoring guide (0.0 = Ethically untenable → 1.0 = Ethically exemplary)

```
<protocol>
For each critique-rebuttal pair:
1. RESTATE the core disagreement neutrally. Identify specific belief IDs under dispute.
2. FORMALIZE both sides as explicit inference chains. Verify cited evidence. Check whether
   the challenge maps to an existing counterposition (X#).
3. ADJUDICATE using the weighted framework.
</protocol>

<evaluation_framework>
WEIGHTS: LOGIC {logic_weight} | ETHICS {ethics_weight}
LOGIC SYSTEM: {logic_system_description}
ETHICS SYSTEM: {ethics_system_description}
MODE: {mode_instructions}
</evaluation_framework>

<criteria>
{criteria_section}
</criteria>
```

**Mode-specific scoring instructions** (from `_build_mode_scoring()`):
- **Logic-only**: `combined = logic` (ethics scores forced to 0.0)
- **Ethics-only**: `combined = ethics` (logic scores forced to 0.0); must cite criterion number for scores >0.6 or <0.4
- **Balanced**: `combined = logic_weight × logic + ethics_weight × ethics`

**Mode-specific evaluation instructions** (from `_MODE_INSTRUCTIONS`):
- `"logic_only"`: "Evaluate using ONLY the logical criteria below. Disregard any ethical arguments..."
- `"balanced"`: "Evaluate using both logical and ethical criteria... Both dimensions contribute..."
- `"ethics_only"`: "Evaluate using ONLY the ethical criteria below... Logical validity is irrelevant..."

**Criteria interleaving** (from `_build_criteria_section()`):
- Logic-only / Ethics-only: criteria listed under single system label
- Balanced: criteria from both systems interleaved with `"(logical)"` / `"(ethical)"` prefixes, numbered sequentially

**Anti-bias guardrails** (`_ANTI_BIAS` block):
1. A response that merely acknowledges a challenge is NOT a successful defense
2. Explicit concession = CRITIQUE_VALID
3. Successful reframing only applies if the defender AVOIDS the critique with substance, not just weakening patches
4. Evaluate substance over rhetorical polish
5. An assumption labeled "foundational" that is actually empirical does not shield it from evidential challenge

### Per-Pair Evaluation

For each challenge-rebuttal pair, the adjudicator receives (via `build_adjudicator_per_pair_prompt()`):
- A `<context>` section containing the challenge/rebuttal text + optional belief excerpts
- An `<instructions>` section with a 3-step protocol
- Special handling for definition-targeting challenges

The adjudicator outputs:
1. A `<reasoning>` block with step-by-step analysis
2. A separate fenced ```json block with the structured verdict:

```json
{
  "restatement": "",
  "formalization_challenger": "",
  "formalization_target": "",
  "scores": {
    "challenger_logic": 0.0,
    "challenger_ethics": 0.0,
    "defender_logic": 0.0,
    "defender_ethics": 0.0,
    "challenger_combined": 0.0,
    "defender_combined": 0.0
  },
  "outcome": "rebuttal_valid|critique_valid|unresolved",
  "reasoning": ""
}
```

### Verdict Enforcement — Exact Algorithm

The `enforce_verdict()` function provides a mathematical override of LLM verdicts when computed scores disagree. This is a critical anti-bias mechanism.

**Step 1 — Compute combined scores**:
```
challenger_combined = logic_weight × challenger_logic + ethics_weight × challenger_ethics
defender_combined   = logic_weight × defender_logic   + ethics_weight × defender_ethics
```
Missing scores default to 0.5. All values rounded to 4 decimal places.

**Step 2 — Determine verdict**:
```
gap = round(challenger_combined - defender_combined, 4)

if gap >= threshold:        → critique_valid
elif gap <= -threshold:     → rebuttal_valid
else:                       → unresolved
```
Default threshold: 0.15 (configurable via `AdjudicationConfig.threshold`).

**Step 3 — Override detection**:
```
override_occurred = (computed_verdict != llm_verdict)
```
If the LLM says "rebuttal_valid" but the math says "critique_valid", the math verdict is used and the override is logged.

### Response Parsing

The adjudicator module includes robust response parsing:
1. `_extract_json_from_response()`: Tries markdown code fences first, then falls back to `raw_decode` scanning from first `{` or `[`. Returns parsed dict or None
2. `validate_adjudicator_output()`: Checks for required fields (outcome, reasoning, restatement, scores with 6 sub-keys)
3. `_normalize_verdict()`: Normalizes raw verdict strings to valid values; unrecognized values map to "unresolved"
4. `ADJUDICATOR_REMEDIATION_HINTS`: Detailed hints sent when output fails validation (JSON must appear after `</reasoning>`, separate fenced block, required fields)

If the response fails validation, the retry system (`generate_with_retry`) sends remediation hints and requests a corrected response.

### Configurable Parameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Logic weight | `adjudication.logic_weight` | 1.0 | Weight for logical evaluation |
| Ethics weight | `adjudication.ethics_weight` | 0.0 | Weight for ethical evaluation |
| Logic system | `adjudication.logic_system` | `"CLASSICAL_INFORMAL_BAYESIAN"` | Key from `logic_systems.py` |
| Ethics system | `adjudication.ethics_system` | `"NONE"` | Key from `ethics_systems.py` |
| Model | `adjudication.model` | `"o4-mini"` | LLM model for the adjudicator |
| Provider | `adjudication.provider` | `"openai"` | LLM provider |
| Threshold | `adjudication.threshold` | 0.15 | Score difference for decisive outcome |

---

## 13. Defense Boost System

**File**: Configuration in `config.py` (`DefenseBoostConfig`), logic in `debate_controller.py`

The defense boost system provides mechanical strength increases for nodes that survive adversarial challenges. Defense boosts are applied **between Phase 1 and Phase 2** of Stage 5, not as a separate post-processing step.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Master switch for defense boosts |
| `flat_boost` | `0.02` | Constant boost per successful defense |
| `max_cumulative_boost` | `0.15` | Maximum total boost above `original_strength` |

### Defense Boost — Exact Algorithm

**Per REBUTTAL_VALID verdict** (from `apply_defense_boosts()` in `debate_controller.py`):
1. Increment: `consecutive_defenses += 1`
2. Boost amount: `compute_defense_boost(count, flat_boost)` → returns `flat_boost` (constant 0.02)
3. Ceiling: `min(original_strength + max_cumulative_boost, 1.0)` where `max_cumulative_boost = 0.15`
4. New strength: `min(current + boost, ceiling)`
5. Rounding: `round(new_strength, 4)`

**Cascade to dependencies** (via `_gather_dependency_nodes()` BFS):
- All dependency nodes of directly boosted targets receive a cascade boost
- Cascade boost amount: `max(directly_boosted.values())` — maximum boost from direct targets
- Already-boosted nodes are excluded from cascade
- Same ceiling formula applies to cascade targets
- After cascade, claim strength is capped at the minimum of its dependency strengths, preventing boosts from violating the dependency constraint

**Per CRITIQUE_VALID verdict**:
- `consecutive_defenses = 0` (streak reset)
- No boost applied

**X#/U# target resolution**: When targets are X# or U# nodes (which don't have strength), the system resolves to their underlying D#/A#/E#/C# targets via `indirect_targets` lookup.

**Key behavior notes**:
- Nodes not attacked in a round are unchanged
- The `original_strength` field (set during defense tracking initialization after Stage 1) serves as the baseline for cumulative boost calculations
- The `consecutive_defenses` counter persists across rounds — it is NOT reset between rounds, tracking the full defense streak across the entire debate

---

## 14. Embedding & Trajectory Analysis

### Component-Wise Embedding System (`embeddings/embedding_tracker.py`)

The `BeliefEmbeddingTracker` uses `sentence-transformers` to compute **7,691-dimensional** belief embeddings that capture both semantic content and structural properties.

**Model**: `SentenceTransformer('all-mpnet-base-v2')` — produces 768-dimensional vectors. The dimension is cached via `self.model.get_sentence_embedding_dimension()`.

#### Embedding Pipeline — Exact Algorithm

**1. Input Projection** (`project_for_component_embedding()` in `io.py`):
- Filters out retracted nodes (`status != "retracted"`) for D#, A#, E#, C#
- Definitions textualized as: `"{term}: {definition}"`
- Assumptions/Claims: uses `statement` field
- Evidence: uses `summary` field
- Thesis text: `"{stance}. {bullet1}. {bullet2}. ..."` (or just stance if no bullets)
- Uncertainties: only non-resolved (`status != "resolved"`) question strings
- Counterpositions: grouped by `response_sufficiency` into 4 buckets (partial/sufficient/unaddressed/moot); missing sufficiency defaults to "unaddressed"
- Scalar avg_strength computed as: `sum(node.strength for node in active_nodes) / len(active_nodes)` (0.0 if empty)

**2. Strength-Weighted Average** (`_weighted_average_embedding()`):
- Formula: `result = (vectors × strengths[:, newaxis]).sum(axis=0) / total_weight`
- If all strengths are 0.0: falls back to unweighted mean `vectors.mean(axis=0)`
- If items list is empty: returns zero vector of shape `(768,)`

**3. Simple Average** (`_simple_average_embedding()`):
- Formula: `vectors.mean(axis=0)`
- If texts list is empty: returns zero vector of shape `(768,)`

**4. Scalar Normalization** (`_normalize_scalars()`, 11 values):
- `_COUNT_CAP = 20`
- Counts (6 features): `min(count, 20) / 20` → clamped to [0, 1]
- Strengths (4 features): passed through unchanged (already [0, 1])
- thesis_strength (1 feature): passed through unchanged
- Exact order: n_definitions, n_assumptions, n_evidence, n_claims, avg_strength_definitions, avg_strength_assumptions, avg_strength_evidence, avg_strength_claims, n_counterpositions, n_uncertainties, thesis_strength

**5. Final Vector Concatenation** (exact order):
1. definitions_vec (768-dim, strength-weighted)
2. assumptions_vec (768-dim, strength-weighted)
3. evidence_vec (768-dim, strength-weighted)
4. claims_vec (768-dim, strength-weighted)
5. thesis_vec (768-dim, single encoding)
6. uncertainties_vec (768-dim, simple average)
7. counter_partial_vec (768-dim, simple average)
8. counter_sufficient_vec (768-dim, simple average)
9. counter_unaddressed_vec (768-dim, simple average)
10. counter_moot_vec (768-dim, simple average)
11. normalized_scalars (11 values)

**Total: 10 × 768 + 11 = 7,691 dimensions**

#### Dual-Path Embedding

The `embed_belief()` method dispatches based on input type:
- **Dict inputs**: Go through component-wise 7,691-dim embedding pipeline
- **String inputs**: Use simple single-text 768-dim encoding via the legacy `project_for_embedding()` function (thesis stance + top 3 definitions + top 3 claims + first uncertainty + first 2 counterpositions)

#### Storage

- Embeddings are saved as `.npz` files (NumPy compressed archives) via `save_embeddings()`
- Loaded via `load_embeddings()`
- Optional `__metadata__` entry preserves agent_info (model, provider, persona) and topic

### Trajectory Visualization (`embeddings/embedding_visualizer.py`)

The visualization system supports both **UMAP** and **PCA** dimensionality reduction:

**UMAP trajectory plots** (`generate_belief_trajectory_plot()`):
- Each agent's belief trajectory over rounds is plotted as a connected path
- Shows how beliefs evolve and whether agents converge in semantic space
- UMAP auto-clamping: when sample count < default n_neighbors (15), n_neighbors is automatically clamped to `max(2, n_samples - 1)`
- Output: PNG plot saved to storage directory

**PCA trajectory plots** (`generate_pca_trajectory_plot()`):
- Alternative visualization using PCA reduction via `reduce_embeddings_pca()`
- Same connected-path style as UMAP plots

**Visual features**:
- 13 **persona colormaps** (e.g., EMPIRICIST → RdPu, SKEPTIC → Reds) — persona-specific colors resolved via the `_persona_color()` helper (case-insensitive persona lookup → hex color string from colormap sampling)
- **Provider logos**: Provider logo PNGs loaded from `assets/logos/` and placed at trajectory start positions. Fallback: colored circle with provider initial letter
- **Arrow annotations**: Consecutive trajectory points connected with `matplotlib.annotate()` arrowprops (`arrowstyle='->'`, `lw=1.5`)
- **Plot styling**: Serif font context (Times New Roman, DejaVu Serif), title fontsize=18 bold, axis labels fontsize=13 bold, DPI=150 for saved files

---

## 15. Performance Scoring

### Agent Performance Score (APS)

A normalized scoring system evaluates agent performance during debate based on adjudication outcomes. The APS is computed from per-exchange role-weighted scores, producing a value in [-1.0, +1.0].

#### Per-Exchange Scoring (`update_agent_stats()` in `utils.py`)

Each exchange appends a score to both participants' `exchange_scores` lists, based on `EXCHANGE_SCORE_WEIGHTS`:

| Role | Verdict | Score |
|------|---------|-------|
| Challenger | critique_valid | +1.0 |
| Challenger | rebuttal_valid | -0.5 |
| Challenger | unresolved | 0.0 |
| Target | critique_valid | -1.0 |
| Target | rebuttal_valid | +1.0 |
| Target | unresolved | +0.25 |

#### APS Calculation (`calculate_performance_scores()`)

```
APS = round(sum(exchange_scores) / len(exchange_scores), 4)
```
- Range: [-1.0, +1.0] (+1.0 = won every exchange, 0.0 = break-even, -1.0 = lost every)
- Returns 0.0 if no exchanges

#### Score Aggregation

Per-agent, per-role (as_challenger / as_target) accumulation of adjudication scores in `adjudication_score_aggregates`:
- `logic_sum`, `ethics_sum`, `combined_sum`, `count`

Score mean calculation (`finalize_agent_stats()`):
```
logic_mean    = round(logic_sum / count, 4)
ethics_mean   = round(ethics_sum / count, 4)
combined_mean = round(combined_sum / count, 4)
```
Returns None if count == 0.

#### Override Counting

Total override count derived from pairs (not agent stats) to avoid double-counting:
```
total_overrides = sum(1 for pair in pairs if pair.resolution.override_occurred)
```
Each agent's stats include a `verdict_overrides` counter tracking how many times the mathematical verdict override affected their exchanges.

#### Attack Histograms

`compute_attack_histograms()` and `compute_per_round_attack_histograms()` populate per-agent and debate-wide attack statistics, initialized from `ALL_STRATEGIES` — a pre-computed sorted flat list of all 27 attack strategies.

#### Belief Snapshots

`snapshot_belief()` captures per-round belief state:
- `thesis_strength` and `component_counts` for tracking belief evolution
- D#/A#/C#/E# nodes counted excluding retracted items
- X#/U# nodes counted including all items (different semantics by node type)

#### Debate Aggregate

The finalized `agent_stats` includes a `_debate_aggregate` key containing debate-wide summary statistics (total attacks, verdicts, per-round attacks, operational metrics).

Best agent is selected by highest APS via `select_best_agent()`, with ties broken by `config.agents` order.

---

## 16. Configuration System

### Configuration Hierarchy

All configuration is managed through Python dataclasses in `config.py`:

```
DebateConfig
├── name, description, version        # Metadata
├── topic                             # Debate question
├── max_rounds                        # Number of rounds
├── stage3_mode                       # "rebuttal"
├── agents: List[AgentConfig]         # Agent definitions
│   └── name, persona, model, temperature, provider, belief_file, max_tokens
├── adjudication: AdjudicationConfig  # Adjudicator settings
│   └── model, provider, logic_weight, ethics_weight, logic_system,
│       ethics_system, threshold
├── stages: StageConfig               # Stage parameters
│   └── max_questions_per_cross_exam, max_rebuttals_per_response,
│       max_rebuttal_length_chars, generation_temperature,
│       short_note_max_chars, parse_retries
├── outputs: OutputConfig             # Output file settings
│   └── storage_dir, save_transcript, save_initial_beliefs,
│       save_final_beliefs, best_beliefs_json_file, best_beliefs_text_file,
│       generate_embeddings, plot_trajectories, pca_plot_file,
│       generate_graph_visualization, save_agent_stats, save_debug_log,
│       save_analysis_report, save_training_data
├── parallel: ParallelConfig          # Parallelization settings
│   └── enabled, max_workers
└── defense_boost: DefenseBoostConfig  # Defense boost settings
    └── enabled, flat_boost, max_cumulative_boost
```

### Key Configuration Details

**AgentConfig**:
- `name`: Agent display name (auto-generated as `"Agent-{Persona.capitalize()}"` if not provided)
- `persona` (e.g., `"EMPIRICIST"`): String key mapping to a persona in `epistemic_personas.py`. Resolved via `get_persona()`
- `provider`: One of `"openai"`, `"anthropic"`, `"google"`, `"ollama"`, `"xai"`, `"perplexity"`
- `model`: Model identifier string
- `temperature`: Generation temperature (default 1.0)
- `belief_file: str | None = None`: Path to pre-defined CBS belief JSON (agent skips Stage 1 when set)
- `max_tokens: int = 65536`: Max output tokens (passed to OpenAI and Anthropic agents only)

**AdjudicationConfig**:
- `logic_system` (default: `"CLASSICAL_INFORMAL_BAYESIAN"`): String key mapping to a logic system in `logic_systems.py`
- `ethics_system` (default: `"NONE"`): String key mapping to an ethics system in `ethics_systems.py`
- `logic_weight` (default: `1.0`) and `ethics_weight` (default: `0.0`): Must be one of three valid combinations: `{(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)}`. Enforced via `__post_init__` validation
- `threshold` (default: `0.15`): Score difference required for a decisive verdict

**StageConfig**:
- `max_questions_per_cross_exam: int = 5`
- `max_rebuttals_per_response: int = 5`
- `max_rebuttal_length_chars: int = 500`
- `generation_temperature: float = 0.2`
- `short_note_max_chars: int = 140`
- `parse_retries: int = 3`

**DefenseBoostConfig**:
- `enabled: bool = True` — master switch for mechanical strength boosts
- `flat_boost: float = 0.02` — constant strength increase per successful defense
- `max_cumulative_boost: float = 0.15` — ceiling above `original_strength`

**ParallelConfig**:
- `enabled: bool = False` (default is False; `default.yaml` overrides to True)
- `max_workers: int = 4`

**OutputConfig**:
- `pca_plot_file: str = "belief_trajectories_pca.png"` — PCA trajectory visualization output path
- `save_analysis_report` and `save_training_data` default to `false`

### Module-Level Path Constants

`config.py` defines path constants used throughout:
- `PROJECT_ROOT`: Root of the CHAL project
- `CHAL_PACKAGE_DIR`: `src/chal/` directory
- `CONFIG_DIR`: `src/chal/configurations/` directory
- `DEFAULT_STORAGE_DIR`: Default output directory (`src/chal/storage/`)

### YAML Configuration Files

Configuration is loaded from YAML files in `src/chal/configurations/`:
- `default.yaml`: Standard rebuttal mode, Empiricist vs Supernaturalist, 1 round

**Notable discrepancies** between config.py defaults and `default.yaml`:
- `max_rebuttal_length_chars`: config.py default 500, default.yaml 1000
- `parallel.enabled`: config.py default False, default.yaml True
- `generate_graph_visualization`: config.py default True, default.yaml False

### Loading Configurations

```python
# By name (resolves to configurations/{name}.yaml via from_name())
config = load_config("default")

# By path
config = load_config("/path/to/my_config.yaml")

# Programmatically
config = DebateConfig(
    topic="Does free will exist?",
    max_rounds=3,
    agents=[
        AgentConfig(name="Agent-A", persona="EMPIRICIST", model="gpt-4o"),
        AgentConfig(name="Agent-B", persona="SKEPTIC", model="gpt-4o"),
    ],
    adjudication=AdjudicationConfig(
        logic_system="DIALECTICAL",
        ethics_system="VIRTUE_ETHICS",
        logic_weight=0.5,
        ethics_weight=0.5,
    ),
    ...
)
```

`from_yaml()` resolves `belief_file` paths relative to the YAML file directory.

### Serialization

The `DebateConfig` class supports:
- `from_yaml(path)`: Load from a YAML file
- `to_yaml(path)`: Save to a YAML file
- `to_dict()`: Convert to a plain dict (converts storage_dir Path to relative string with forward slashes)
- `from_name(name)`: Load config by name from the `configurations/` directory

---

## 17. CLI & Interactive Wizard

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-c` | Configuration name or YAML file path (headless mode) |
| `--edit` | `-e` | Load config into wizard for editing before launch |
| `--verbose` | `-v` | Enable verbose output |
| `--history` | — | Display past debate history table |
| `--replay <id>` | — | Re-run a past debate by its history ID |

### 5 CLI Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Interactive wizard** | Default (no args) | Full wizard → launch |
| **Headless** | `--config` only | Load config, run without interaction |
| **Edit** | `--config --edit` | Load config into wizard for modification |
| **History** | `--history` | Display past debate history table |
| **Replay** | `--replay <id>` | Reload config from history, run headless |

**Return codes**: 0 for success/cancellation, 1 for errors.

### Interactive Wizard (`wizard.py`)

A configuration wizard using `questionary` and `rich`. The wizard starts with a **main menu**: About CHAL, Debate, Gauntlet, Exit.

The **ABOUT_CHAL** panel displays framework philosophy (belief search, structured debate, AI safety), three implications (AI Safety & Reasoning, Human Knowledge Extension, Argumentation Literacy), and a 6-step wizard usage summary.

#### Wizard Steps

The wizard runs through **9 steps** (`_WIZARD_STEPS`):

1. **Preset Selection** (`_step_preset`): `_scan_presets()` scans `CONFIG_DIR` for `*.yaml` files, loads metadata (name, description), presents with "Custom" option first and `_PRESET_ORDER = ["default"]` preferred ordering
2. **Topic** (`_step_topic`): Enter the debate topic/question
3. **Number of Agents** (`_step_num_agents`): Select 2-6 agents
4. **Agent Configuration** (`_step_agents`): For each agent — persona (from `PERSONA_DESCRIPTIONS`), provider, model (from `MODEL_SUGGESTIONS`), and optional belief file loading
5. **Number of Rounds** (`_step_rounds`): Select 1-10 rounds
6. **Adjudicator Configuration** (`_step_adjudicator`): Provider, model, logic system (with descriptive tags like "Recommended", "Strict deduction", "Degrees of truth"), ethics system, logic/ethics weight balance via 3 presets: Pure Logic (1.0/0.0), Balanced (0.5/0.5), Pure Ethics (0.0/1.0). `_detect_balance_preset()` auto-detects from weights
7. **Output Settings** (`_step_outputs`): 10 checkboxes: save_transcript, plot_trajectories, save_agent_stats, save_initial_beliefs, save_final_beliefs, generate_graph_visualization, generate_embeddings, save_training_data, save_analysis_report, save_debug_log
8. **Parallelization Settings** (`_step_parallelization`): Enable/disable concurrent API calls, configure max workers
9. **API Key Collection** (`_step_api_keys`): Prompt for any missing API keys required by the selected providers

**Model suggestions** (`MODEL_SUGGESTIONS`):
- openai: o4-mini, o3, o3-mini
- anthropic: claude-opus-4-6, claude-sonnet-4-5-20250929
- google: gemini-2.5-pro, gemini-2.5-flash
- ollama: deepseek-r1:14b, deepseek-r1:32b, qwq
- xai: grok-3-mini, grok-3
- perplexity: sonar-reasoning-pro, sonar-reasoning

**After all steps**, a **review panel** shows a summary of all settings. From there the user can:
- **Launch**: Start the debate
- **Edit**: Re-run specific wizard steps via `_apply_edit()` to modify individual config sections
- **Save**: Save the configuration to YAML
- **Cancel**: Exit

**Navigation**:
- `Esc`: Exit wizard
- `Ctrl+Z`: Go back one step
- `Ctrl+F1`: Show context-sensitive help for current prompt

**23 context-sensitive help texts**: Every wizard step has a dedicated help panel, including: HELP_MAIN_MENU, HELP_PRESET, HELP_TOPIC, HELP_NUM_AGENTS, HELP_PERSONA (all 12+NONE with descriptions), HELP_PROVIDER, HELP_MODEL, HELP_TEMPERATURE, HELP_STAGE2, HELP_STAGE3, HELP_NUM_ROUNDS, HELP_ADJ_BALANCE, HELP_ADJ_CUSTOM_WEIGHTS, HELP_ADJ_LOGIC_SYSTEM, HELP_ADJ_ETHICS_SYSTEM, HELP_OUTPUTS, HELP_REVIEW_ACTION, HELP_EDIT_SECTION, HELP_SAVE_PATH, HELP_PARALLELIZATION, HELP_MAX_WORKERS, HELP_API_KEYS, HELP_GAUNTLET.

### Display System (`display.py`)

Uses `rich` library for styled terminal output with an **event-driven display system**:
- 10 event handlers: debate_start, stage_start, stage_complete, agent_start, agent_complete, adjudication_result, round_start, round_complete, debate_complete, output_files_saved
- Progress bars for debate stages
- Formatted tables for agent stats
- Color-coded adjudication outcomes
- `handle_error()`: Interactive retry/skip/abort prompts in interactive mode, auto-retry-once in headless mode

### Debate History (`history.py`)

Tracks past debates in `~/.chal/history.json` with full configs stored in `~/.chal/history/<id>.yaml`:

**Complete history lifecycle**:
- `log_debate()`: Generates 8-char hex ID, saves config snapshot YAML, records 11 data fields: id, timestamp, topic, agents, rounds, duration_s, convergence, winner, config_snapshot, output_dir, operational_metrics
- `list_debates()`: Returns all debate entries
- `load_debate_config()`: Reconstructs `DebateConfig` from history ID
- `format_history_table()`: Rich table with columns: ID, Date, Topic (truncated 40 chars), Agents, Rounds, Duration, Winner, Conv.
- Enables replay with same configuration

### API Key Management (`api_keys.py`)

Provider-to-environment-variable mapping (defined in `constants.py` as `PROVIDER_ENV_VARS`):

| Provider | Environment Variable |
|----------|---------------------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `perplexity` | `PERPLEXITY_API_KEY` |

Ollama does not require an API key (local models).

`validate_api_keys()` checks for required keys based on selected providers. `create_key_pool()` initializes a `KeyPool` for multi-key rotation.

### Logger Configuration

The `chal` logger is set to DEBUG level. Console handler level is DEBUG if `--verbose`, else INFO. Third-party noise suppressed for: httpx, httpcore, sentence_transformers, torch, transformers.

---

## 18. Utilities

### Parallel Dispatch (`utilities/parallel.py`)

The `ParallelDispatcher` wraps `concurrent.futures.ThreadPoolExecutor` for the gather-then-apply pattern:

- **`WorkItem`** dataclass: `key` (unique identifier), `callable` (zero-argument callable)
- **`WorkResult`** dataclass: `key`, `result` (return value or exception), `elapsed` (seconds), `success` (bool)
- **Deterministic ordering**: Results are returned in submission order via `OrderedDict`, regardless of completion order
- **Graceful fallback**: When `enabled=False`, runs items sequentially in a plain for-loop — producing identical behavior to the pre-parallelization codebase

### Retry Logic (`utilities/retry.py`)

**`retry_api_call()`**: Generic retry wrapper for all provider API calls with:
- Exponential backoff (configurable base delay, doubling each attempt)
- Configurable error types per provider
- Key rotation on rate limits
- Rate limit marking on the `KeyPool`

**`generate_with_retry()`**: Structured LLM output validation with targeted correction:
- **`ValidationResult`** dataclass: `is_valid: bool`, `errors: list[str]`, `parsed_data: Any | None`
- **`RetryRecord`** dataclass: `attempt: int`, `errors: list[str]`, `raw_response_preview: str` (first 500 chars), `timestamp: float`
- **Flow**: Generate response → validate with caller-supplied validator → if invalid, append correction message (listing errors + remediation hints) to the conversation history so the model sees its own errors → retry up to `max_retries` times
- Each stage in the pipeline supplies its own validator function (see `validators.py`)

### API Key Pool (`utilities/key_pool.py`)

The `KeyPool` class provides thread-safe, rate-limit-aware API key rotation:

- **Resource Pool + Circuit Breaker** pattern
- Supports multiple API keys per provider (comma-separated in environment variables)
- **`load_from_env()`**: Reads environment variables and splits on commas. Example: `OPENAI_API_KEY="sk-a,sk-b,sk-c"` registers 3 keys for round-robin rotation
- **`get_key(provider)`**: Returns the next available key via round-robin rotation; blocks if all keys are cooling down (sleeps until the soonest key recovers)
- **`mark_rate_limited(provider, key, cooldown)`**: Marks a key as cooling down for `cooldown_seconds` (default 60s). Key is skipped until cooldown expires
- Thread-safe via `threading.Lock`

### Stage Validators (`utilities/validators.py`)

Stage-specific output validators called by the retry system:

| Validator | Stage | Key Checks |
|-----------|-------|------------|
| `validate_stage1_output()` | Stage 1 | CBS structure, all 11 required keys, inference_chain, predictions |
| `validate_stage2_output()` | Stage 2 | Question format, attack_type/strategy validation against `VALID_ATTACK_STRATEGIES` |
| `validate_stage3_output()` | Stage 3 | Rebuttal format, action enum (refute/concede/defer), qid coverage, patches structure |
| `validate_stage5_phase1_output()` | Stage 5 Phase 1 | Enforcement compliance (non-empty patches if critique_valid) |
| `validate_stage5_phase2_output()` | Stage 5 Phase 2 | Introspection format (empty patches allowed) |

**ID normalization**: Both `parse_model_output_to_belief()` (in `io.py`) and `_try_parse_json()` (in `validators.py`) normalize IDs like `"A#1"` → `"A1"` before JSON parsing, handling common LLM formatting artifacts.

### Training Data Export (`utilities/training_data.py`)

The `DebateRecorder` class passively records all debate events for training data export:

**Recording methods**:
- `record_belief_formation()`: Stage 1 initial beliefs
- `record_cross_examination()`: Stage 2 questions
- `record_rebuttal()`: Stage 3 rebuttals
- `record_adjudication()`: Stage 4 verdicts
- `record_belief_update()`: Stage 5 patches

**Export formats**:
- `export_jsonl()`: All events as JSONL (one JSON object per line)
- `export_belief_training_pairs()`: Before/after belief pairs for fine-tuning

**Training data pair types** (`_extract_belief_pairs()`):
- **`belief_formation`**: (topic + persona + system_prompt) → belief object
- **`belief_update`**: (belief_before + adjudication_results + debate_context) → belief_after + patches_applied

### Reporting (`utilities/reporting.py`)

**`generate_analysis_report()`**: Produces Markdown analysis with:
- Header, Metadata, Adjudicator Details
- Verdict Distribution Table
- Adjudication Details (per exchange, truncated to 500 chars)
- Agent Performance Table (sorted by APS)
- Belief Evolution (per agent with strength drift)

**`generate_analysis_json()`**: Returns structured JSON with:
- `generated_at`, `metadata`, `verdict_distribution`
- `agent_summaries` (per agent with scores and counts)
- `exchanges` (per pair with verdict/reasoning)

### Utility Functions (`utilities/utils.py`)

Key functions and constants:
- `VALID_ATTACK_STRATEGIES`: Full 27-strategy taxonomy dict
- `ALL_STRATEGIES`: Pre-computed sorted flat list of all 27 strategies
- `EXCHANGE_SCORE_WEIGHTS`: Per-exchange role-weighted scores for APS
- `initialize_agent_stats()`: Set up per-agent tracking structures
- `update_agent_stats()`: Record exchange outcomes
- `calculate_performance_scores()`: Compute APS values
- `finalize_agent_stats()`: Assemble `_debate_aggregate` and derive final snapshots
- `snapshot_belief()`: Extract thesis_strength and component_counts
- `compute_attack_histograms()` / `compute_per_round_attack_histograms()`: Attack statistics
- `select_best_agent()`: Returns agent with highest APS
- `sanitize_filename()`: Converts agent names to filesystem-safe filenames (non-alphanumeric → `_`, strip leading/trailing underscores, default "unnamed")
- `parse_challenges()` / `parse_structured_rebuttals_numbered()`: Legacy fallback parsers for unstructured text

---

## 19. Debug Logging System

**File**: `src/chal/utilities/debug_log_writer.py`

### DebugLogWriter

Thread-safe debug log writer with real-time file streaming:
- Supports both **file-backed** (writes to disk in real-time) and **memory-only** modes
- Captures all prompts, responses, retries, parsing details, and validation errors
- Thread-safe for concurrent access during parallel LLM dispatch

### DebugLogHandler

A Python `logging.Handler` subclass that bridges standard Python logging records into the `DebugLogWriter`:
- Allows `logger.debug()`, `logger.info()`, etc. to be captured in the debug log file
- Integrates with the existing Python logging infrastructure

### Log Levels

The `chal` logger is set to DEBUG level. Console handler level is DEBUG if `--verbose` flag is provided, else INFO. Third-party noise is suppressed for: httpx, httpcore, sentence_transformers, torch, transformers.

---

## 20. Output Artifacts

| File | Config Key | Default | Description |
|------|-----------|---------|-------------|
| `debate_transcript.txt` | `save_transcript` | `true` | Human-readable Markdown transcript generated via `_add_to_markdown()` |
| `initial_beliefs/` | `save_initial_beliefs` | `true` | Individual JSON files per agent with Stage 1 beliefs |
| `final_beliefs/` | `save_final_beliefs` | `true` | Individual JSON files per agent with final beliefs |
| `best_initial_final_beliefs.json` | `best_beliefs_json_file` | always | Raw CBS JSON for the highest-APS agent (initial + final). Always generated regardless of output config |
| `best_initial_final_beliefs.txt` | `best_beliefs_text_file` | always | Markdown rendering of the highest-APS agent's initial and final belief. Always generated |
| `embeddings.npz` | `generate_embeddings` | `true` | NumPy compressed archive of belief embeddings per round |
| `belief_trajectories.png` | `plot_trajectories` | `true` | UMAP 2D visualization of belief evolution |
| `belief_trajectories_pca.png` | `pca_plot_file` | `true` | PCA 2D visualization of belief evolution |
| `belief_graph.html` | `generate_graph_visualization` | `true` | Interactive Cytoscape.js graph visualization |
| `agent_stats.json` | `save_agent_stats` | `true` | Performance metrics (APS, wins, losses, attack histograms) |
| `log.txt` | `save_debug_log` | `true` | Comprehensive debug log with all prompts, responses, and parsing details |
| `debate_analysis_report.md` | `save_analysis_report` | `false` | Structured Markdown analysis report |
| `debate_analysis_report.json` | (paired with above) | `false` | Structured JSON analysis report |
| `debate_training_data.jsonl` | `save_training_data` | `false` | JSONL training data for fine-tuning |
| `debate_belief_pairs.jsonl` | (paired with above) | `false` | Before/after belief pairs (belief_formation + belief_update types) |

All output files are saved to the configured `storage_dir` (default: `src/chal/storage/`).

**Token accumulation**: `DebateMetrics` tracks `total_input_tokens` and `total_output_tokens` across the entire debate, handling both Anthropic format (`input_tokens`/`output_tokens`) and OpenAI format (`prompt_tokens`/`completion_tokens`).

**Definition statistics log**: At debate end, `_log_definition_statistics()` records D# count, active vs. retracted breakdown, average strength of active definitions, and most-challenged definitions.

---

## 21. Complete Prompt Reference

This section indexes every prompt function in `prompts.py` by stage and mode:

### Stage 0: Briefing
| Function | Purpose |
|----------|---------|
| `build_universal_prompt(topic)` | Shared debate protocol, intellectual standards (7 principles), and debate context for all agents |
| `build_debate_context()` | Generates shared `<debate_context>` block describing the 4-stage loop (cross-exam → rebuttal → adjudication → belief update) |
| `build_position_prompt(agent_name, persona)` | Agent-specific persona/role card with `<persona>` and `<persona_guidance>` blocks. Empty persona instructs agent to argue from content/logic/evidence |

### Stage 1: Opening Positions
| Function | Purpose |
|----------|---------|
| `build_stage_1_belief_prompt_cbs(topic, agent_name, persona_label)` | Elicit complete CBS JSON belief with 7-step generation order, thesis strength formula, strength scale block, typed assumptions, definitions, counterpositions, and example |

### Stage 2: Cross-Examination
| Function | Purpose |
|----------|---------|
| `build_stage_2_prompt(...)` | Cross-examination with 27 attack strategies, anti-repetition context, and vulnerability analysis. Accepts optional `targeted_claims_json` |

### Stage 3: Rebuttals
| Function | Purpose |
|----------|---------|
| `build_stage_3_structured_rebuttal_prompt(...)` | Structured rebuttals (refute/concede/defer) + patches in single JSON. Definition-specific challenge handling |

### Stage 4: Adjudication
| Function | Purpose |
|----------|---------|
| `build_adjudicator_prompt(logic_weight, ethics_weight, logic_sys, ethics_sys)` | Adjudicator system prompt with XML-structured criteria, mode-specific scoring, anti-bias guardrails, and strength/scoring scale blocks |
| `build_adjudicator_per_pair_prompt(...)` | Per-pair evaluation with `<context>` (challenge/rebuttal + belief excerpts) and `<instructions>` (3-step protocol) |

### Stage 5: Belief Updates
| Function | Purpose |
|----------|---------|
| `build_stage_5_phase1_enforcement_prompt(...)` | Phase 1 enforcement-only patching with mandatory rules for CRITIQUE_VALID outcomes |
| `build_stage_5_phase2_introspection_prompt(...)` | Phase 2 strategic position building with dynamic position analysis, breadth table, and example patches |
| `build_stage_5_belief_update_prompt_cbs(...)` | Legacy single-phase belief update (fallback when `prior_json` unavailable) |

### Helper Prompt Functions
| Function | Purpose |
|----------|---------|
| `compute_position_analysis()` | Dynamic position analysis: partial derivatives (∂T/∂s, ∂T/∂n), scenario projections, D# vulnerability analysis, strategic recommendations |
| `_build_supported_ops_block()` | Parameterized patch operations reference block |
| `_build_mandatory_rules_block()` | Phase-context-aware mandatory rules for Stage 5 |
| `_build_criteria_section()` | Mode-aware criteria merging — interleaves logic/ethics criteria with prefixes in balanced mode |
| `_build_mode_scoring()` | Mode-specific scoring instructions (logic_only, ethics_only, balanced) |
| `_format_adjudication_outcomes()` | Formats adjudication outcomes for Stage 5 injection |
| `_determine_mode()` | Converts weight pair to mode string |

### Prompt Constants
| Constant | Purpose |
|----------|---------|
| `_STRENGTH_SCALE_BLOCK` | Strength calibration table mapping 0.0-1.0 to labels (Vacuous → Definitive) |
| `_LOGIC_SCALE_BLOCK` | Logic scoring guide (0.0 = No reasoning → 1.0 = Rigorous) |
| `_ETHICS_SCALE_BLOCK` | Ethics scoring guide (0.0 = Ethically untenable → 1.0 = Ethically exemplary) |
| `_VALID_DEPENDENCIES_BLOCK` | Allowed dependency edges including key rule: "D# support claims INDIRECTLY through A#/E# layer only" |
| `_ANTI_BIAS` | 5 anti-bias principles for adjudicator |
| `_UNIVERSAL_BASE` | Disqualifying flaws list |
| `_MODE_INSTRUCTIONS` | Three distinct evaluation mode instruction sets (logic_only, balanced, ethics_only) |

### Framework Lookup Functions
| Module | Function | Purpose |
|--------|----------|---------|
| `epistemic_personas.py` | `get_persona(key)` | Look up persona prompt by key (case-insensitive) |
| `logic_systems.py` | `get_logic_system(key)` | Look up logic system dict by key (case-insensitive) |
| `logic_systems.py` | `get_logic_system_description(key)` | Look up logic system description string |
| `logic_systems.py` | `get_logic_system_label(key)` | Look up logic system display label |
| `ethics_systems.py` | `get_ethics_system(key)` | Look up ethics system dict by key (case-insensitive) |
| `ethics_systems.py` | `get_ethics_system_description(key)` | Look up ethics system description string |
| `ethics_systems.py` | `get_ethics_system_label(key)` | Look up ethics system display label |

---

## 22. Dependencies

### Core Dependencies (from `pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >=1.0.0,<2.0.0 | OpenAI API client |
| `anthropic` | >=0.40.0,<1.0.0 | Anthropic API client |
| `google-genai` | — | Google Generative AI client |
| `ollama` | — | Ollama local model client |
| `xai-sdk` | — | xAI API client (gRPC-based) |
| `perplexityai` | — | Perplexity API client |
| `httpx` | — | HTTP client (used by agent implementations) |
| `pydantic` | — | Data validation |
| `sentence-transformers` | >=5.0.0,<6.0.0 | Embedding generation (all-mpnet-base-v2) |
| `numpy` | ==2.2.0 | Numerical operations, embedding storage |
| `umap-learn` | >=0.5.7,<0.6.0 | UMAP dimensionality reduction for trajectory visualization |
| `pandas` | — | Data analysis and tabular operations |
| `matplotlib` | — | Plotting belief trajectories |
| `pyyaml` | — | YAML configuration parsing |
| `tiktoken` | — | Token counting for OpenAI models |
| `python-dotenv` | — | Environment variable loading from `.env` files |
| `tqdm` | — | Progress bars |
| `rich` | — | Terminal formatting and progress display |
| `questionary` | — | Interactive CLI prompts |

### Transitive / Optional Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Cosine similarity (transitive via `sentence-transformers`) |
| `jsonschema` | Optional deep CBS schema validation (enables full item-level validation when installed; graceful degradation via `HAVE_JSONSCHEMA` flag) |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ^8.4.1 | Testing framework |
| `pytest-asyncio` | — | Async test support |
| `pytest-cov` | — | Coverage reporting |
| `pytest-mock` | — | Mock fixtures |
| `black` | ^25.1.0 | Code formatting |
| `isort` | ^6.0.1 | Import sorting |
| `pre-commit` | ^4.2.0 | Pre-commit hooks |

### Build System

Poetry-based: `poetry-core>=2.0.0`

### Tool Configuration

- **Ruff**: Linting
- **Mypy**: Type checking
- **Pytest markers**: `unit`, `integration`, `e2e`, `slow`
- **Coverage requirement**: 85%

### Project Metadata

- **Name**: `chal`
- **Version**: `0.1.0`
- **Python**: `>=3.10`
- **License**: MIT
- **Author**: Griffin Dean Kent
- **Entry point**: `chal = "chal.cli.main:main"`

---

## 23. Test Suite

The CHAL project includes a comprehensive test suite:

- **46 test files** across all modules
- **250+ test functions** covering unit, integration, and end-to-end scenarios

### Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Isolated unit tests with no external dependencies |
| `integration` | Tests requiring multiple components working together |
| `e2e` | End-to-end tests running the full pipeline |
| `slow` | Tests that take significant time (e.g., embedding generation) |

### Coverage

85% coverage requirement enforced via `pytest-cov`.

### Key Test Utilities

- `create_sample_belief()`: Generate valid CBS belief objects for testing
- `create_mock_agent()`: Create mock agent instances with configurable behavior
- Custom assertions for belief validation

### Test Patterns

- **Mock SentenceTransformer**: Embedding tests mock the transformer model to avoid downloading weights
- **Mock providers**: Agent tests mock API calls to avoid hitting real endpoints
- **Temporary directories**: Output tests use `tmp_path` fixtures for isolated file operations

---

## 24. Primary Goal: Belief Refinement

**The central purpose of the CHAL pipeline is to produce better belief objects through structured dialectical exchange.** "Better" means:

1. **More logically coherent**: Circular dependencies, contradictions, and inference chain breaks are identified and eliminated through cross-examination and adjudication.

2. **Better supported**: Claims that lack evidence or depend on weak assumptions are identified as vulnerabilities. The patch system forces strength adjustments that reflect the actual strength of support.

3. **More accurately calibrated**: Strength levels are challenged when they exceed what the evidence warrants. The mandatory patch rules ensure that lost arguments result in actual strength decreases, and the strength propagation system ensures dependent claims are adjusted accordingly.

4. **More honestly self-aware**: Known weaknesses, counterpositions (X#), uncertainties (U#), and limitations are surfaced through the cross-examination process. Beliefs that survive the debate have been tested and are annotated with the vulnerabilities they overcame. Counterpositions with "partial" or "unaddressed" response_sufficiency honestly signal where the position remains vulnerable.

5. **Semantically grounded**: Definitions (D#) pin down the meaning of key terms, preventing equivocation. Definition strength provides a ceiling on the strength of nodes that reference them.

6. **Auditable**: Every change to a belief is recorded through the patch system with a versioned changelog, making it possible to trace exactly how and why a belief evolved.

### How Refinement Happens at Each Stage

| Stage | Refinement Mechanism |
|-------|---------------------|
| **Stage 1** | Initial belief formation with required definitions, inference chains (minItems: 3), inline predictions, typed assumptions (foundational/empirical/methodological/scoping), and at least 2 honest counterpositions with response_sufficiency ratings |
| **Stage 2** | Vulnerability identification through 27 attack strategies across three types (undermining, rebutting, undercutting), including exploiting self-identified "partial"/"unaddressed" counterpositions and challenging assumption type classifications |
| **Stage 3** | Defense or concession — agents must commit to refute/concede/defer with semantic alignment and binding patch requirements for concessions |
| **Stage 4** | Independent evaluation with 6-score weighted framework, mathematical verdict enforcement to prevent LLM bias, and anti-bias guardrails |
| **Stage 5 Phase 1** | Mandatory enforcement patches for lost arguments; new counterpositions for revealed vulnerabilities; strength propagation through dependency graph |
| **Defense Boosts** | Mechanical strength increases for nodes that survive challenges, cascading through dependency graph |
| **Stage 5 Phase 2** | Strategic introspection with dynamic position analysis (partial derivatives, scenario projections); counterposition audit; uncertainty review; thesis rewrite. Guardrail: strength increases on existing nodes are stripped |

### Measuring Refinement

The pipeline provides multiple signals for whether beliefs actually improved:

- **Strength trajectory**: Did strength levels move in response to evidence and argument quality?
- **Patch count and type**: How many weakening vs. strengthening patches were applied?
- **Counterposition evolution**: Did counterpositions get resolved ("sufficient") or remain vulnerable ("partial"/"unaddressed")?
- **Embedding trajectories**: Did beliefs move in semantic space in response to the debate?
- **Adjudication outcomes**: What fraction of challenges were successfully defended vs. conceded?
- **Verdict override frequency**: How often did the mathematical verdict enforcement override the LLM's stated verdict? (System health signal)
- **Attack histogram analysis**: What strategies were most effective? Which nodes were most frequently targeted?
- **Performance scores (APS)**: How did agents perform across the debate in the [-1.0, +1.0] range?
- **Changelog analysis**: The versioned changelog in each belief tracks exactly what changed and why.

The belief refinement goal is what distinguishes CHAL from a simple chatbot debate — the structured CBS schema with typed assumptions, definitions, and honest counterpositions, the mandatory two-phase patch system, the strength propagation through dependency graphs, the defense boost for survived challenges, the mathematical verdict enforcement, the configurable logic and ethics systems for adjudication, and the 13 philosophically grounded personas all work together to ensure that beliefs are genuinely tested and improved through the debate process, not just defended or abandoned.