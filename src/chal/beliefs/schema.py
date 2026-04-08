"""
schema.py

Defines the canonical CBS (CHAL Belief Schema) JSON structure and a validator.

Acronyms:
- JSON = JavaScript Object Notation
- ID   = Identifier
- DOI  = Digital Object Identifier
"""

from __future__ import annotations
from typing import Any, Dict, List
import json
import re

try:
    # 'jsonschema' is a small validation library; if it's not present, we degrade gracefully.
    import jsonschema  # type: ignore
    HAVE_JSONSCHEMA = True
except Exception:
    HAVE_JSONSCHEMA = False


# ---- Canonical key names (kept stable across versions) ----
SCHEMA_VERSION = "CBS"  # CBS = CHAL Belief Schema


# Minimal JSON Schema (JSON = JavaScript Object Notation).
CBS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",  # Top-level artifact must be a JSON object (i.e., key/value map)
    "required": [      # Keys that MUST be present for a valid belief object
        "schema_version",
        "belief_id",
        "version",
        "metadata",
        "thesis",
        "definitions",
        "assumptions",
        "claims",
        "evidence",
        "counterpositions",
        "uncertainties"
    ],
    "properties": {    # Definitions for each permissible top-level property
        # --- Identity & versioning ---
        "schema_version": {                     # Fixed schema label so downstream tooling knows how to parse
            "type": "string",
            "enum": [SCHEMA_VERSION]            # Must equal "CBS"
        },
        "belief_id": {                          # Stable unique ID for the belief object across updates
            "type": "string"                    # e.g., "BELIEF-<uuid>", not enforced by regex here
        },
        "version": {                            # Monotone-increasing integer for belief revisions
            "type": "integer",
            "minimum": 1                        # Version numbers start at 1
        },

        # --- Metadata block: who/what ---
        "metadata": {
            "type": "object",                   # A nested object containing descriptive metadata
            "required": [                       # Minimum metadata to track provenance and persona
                "topic_query",                  # The debate question (e.g., "Does free will exist?")
                "agent_persona"                 # Persona label (e.g., "Empiricist", "Skeptic")
            ],
            "properties": {
                "topic_query": { "type": "string" },     # Query or topic string
                "agent_persona": { "type": "string" },   # Agent persona or role name
            }
        },

        # --- Thesis block: concise position summary ---
        "thesis": {
            "type": "object",                      # Core stance of the belief
            "required": [
                "stance",                          # One-sentence position statement
                "summary_bullets",                 # 2–3 short bullets highlighting key points
                "strength"                         # Author's calibrated strength in [0, 1]
            ],
            "properties": {
                "stance": { "type": "string" },    # Single-sentence thesis
                "summary_bullets": {               # Short bullet points expanding the stance
                    "type": "array",
                    "items": { "type": "string" }, # Each bullet is a short string
                    "minItems": 1                  # At least one bullet (recommend 2–3)
                },
                "strength": {                      # Calibrated strength (0 = no support, 1 = definitive)
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "strength_reasoning": { "type": "string" }  # Auto-generated formula breakdown (optional, set by apply_patches)
            }
        },

        # --- Definitions: semantic bedrock (D# nodes) ---

        "definitions": {                             # List of D# items defining key terms
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "term", "definition", "strength",
                              "strength_justification", "status", "used_by"],
                "properties": {
                    "id": { "type": "string" },              # D1, D2, etc.
                    "term": { "type": "string" },            # The term being defined
                    "definition": { "type": "string" },      # The definition text
                    "strength": {                            # Calibrated strength (0.0 - 1.0)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "strength_justification": { "type": "string" },  # Rationale for the strength
                    "status": {                              # Lifecycle status
                        "type": "string",
                        "enum": ["active", "revised", "retracted"]
                    },
                    "used_by": {                             # A#/E# IDs that depend on this definition
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    },
                    "original_strength": {                   # Immutable snapshot of initial strength (system-managed)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "consecutive_defenses": {                # Consecutive successful defenses (system-managed)
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        },

        # --- Required collection sections (lists of structured nodes) ---

        "assumptions": {                             # List of A# items with typed categories, each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "statement", "supports_claims", "strength", "status",
                             "strength_justification", "supported_by_definitions"],
                "properties": {
                    "id": { "type": "string" },
                    "type": {
                        "type": "string",
                        "enum": ["foundational", "empirical", "methodological", "scoping"]
                    },
                    "statement": { "type": "string" },
                    "supports_claims": {                 # Cross-references to C# IDs this assumption supports
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "strength": {                        # Calibrated strength (0.0 - 1.0)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "status": {                          # Lifecycle status
                        "type": "string",
                        "enum": ["active", "revised", "retracted"]
                    },
                    "strength_justification": { "type": "string" },  # Rationale for the strength number
                    "supported_by_definitions": {        # D# IDs that define key terms in this assumption
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    },
                    "original_strength": {                   # Immutable snapshot of initial strength (system-managed)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "consecutive_defenses": {                # Consecutive successful defenses (system-managed)
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        },
        "claims": {                                  # List of C# items (claims with type/strength/status), each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "statement", "depends_on",
                             "strength", "status",
                             "predictions", "strength_justification",
                             "inference_chain"],
                "properties": {
                    "id": { "type": "string" },
                    "type": { "type": "string" },              # Free-form claim category (e.g., "descriptive", "causal")
                    "statement": { "type": "string" },
                    "depends_on": {                            # References to A#/E#/C# IDs this claim builds on
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "strength": {                              # Calibrated strength in [0, 1]
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "revised", "retracted"]
                    },
                    "predictions": {                           # Falsifiable predictions (at least one per claim)
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["statement", "test", "decision_criterion"],
                            "properties": {
                                "statement": { "type": "string" },
                                "test": { "type": "string" },
                                "decision_criterion": { "type": "string" },
                                "potential_falsifiers": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            }
                        },
                        "minItems": 1
                    },
                    "strength_justification": { "type": "string" },  # Rationale for the strength number
                    "inference_chain": {                        # Structured reasoning steps leading to this claim
                        "type": "array",
                        "minItems": 3,                          # At least: 1 premise + 1 inference + 1 conclusion
                        "items": {
                            "type": "object",
                            "required": ["role", "text"],
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["premise", "inference", "conclusion"]
                                },
                                "text": {
                                    "type": "string",
                                    "minLength": 1
                                },
                                "reference": {                  # Required for premises — cites a single A#, E#, or C# ID
                                    "type": "string"
                                },
                                "inference_type": {             # Required for inference role
                                    "type": "string",
                                    "enum": ["deductive", "inductive", "abductive"]
                                }
                            }
                        }
                    },
                    "original_strength": {                      # Immutable snapshot of initial strength (system-managed)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "consecutive_defenses": {                   # Consecutive successful defenses (system-managed)
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        },
        "evidence": {                                # List of E# items, each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "summary", "source", "supports_claims",
                             "strength", "status", "strength_justification",
                             "supported_by_definitions"],
                "properties": {
                    "id": { "type": "string" },
                    "type": {
                        "type": "string",
                        "enum": ["empirical", "conceptual", "expert_consensus"]
                    },
                    "summary": { "type": "string" },
                    "source": { "type": "string" },            # Citation or provenance string
                    "supports_claims": {                   # Cross-references to C# IDs
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "strength": {                              # Calibrated strength (0.0 - 1.0)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "status": {                                # Lifecycle status
                        "type": "string",
                        "enum": ["active", "revised", "retracted"]
                    },
                    "strength_justification": { "type": "string" },  # Rationale for the strength number
                    "supported_by_definitions": {              # D# IDs that define key terms in this evidence
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    },
                    "original_strength": {                     # Immutable snapshot of initial strength (system-managed)
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "consecutive_defenses": {                  # Consecutive successful defenses (system-managed)
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        },
        "uncertainties": {                           # List of U# items (key unknowns), each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "targets", "question", "status", "importance"],
                "properties": {
                    "id": { "type": "string" },
                    "targets": {                               # References to A#/E#/C# IDs this uncertainty questions
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "question": { "type": "string" },
                    "status": {                                # Whether the uncertainty has been resolved
                        "type": "string",
                        "enum": ["active", "resolved"]
                    },
                    "importance": {                              # How critical resolving this uncertainty is
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "resolution_note": { "type": "string" }    # How the uncertainty was resolved (required when resolving)
                }
            }
        },
        "counterpositions": {                        # List of X# items (prepared defenses + responses), each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "targets", "attack_type", "attack_strategy",
                             "statement", "my_response", "response_sufficiency"],
                "properties": {
                    "id": { "type": "string" },
                    "targets": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "attack_type": {
                        "type": "string",
                        "enum": ["undermining", "rebutting", "undercutting"]
                    },
                    "attack_strategy": { "type": "string" },  # Must be valid for the attack_type
                    "statement": { "type": "string" },
                    "my_response": { "type": "string" },
                    "response_sufficiency": {
                        "type": "string",
                        "enum": ["sufficient", "partial", "unaddressed"]
                    }
                }
            }
        },

        # --- Audit trail ---

        "changelog": {                               # Optional list of versioned change records (what changed)
            "type": "array",
            "items": {
                "type": "object",
                "required": ["version", "changes"],
                "properties": {
                    "version": { "type": "integer", "minimum": 1 },
                    "changes": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                }
            }
        }
    },
    "additionalProperties": True  # Allow forward-compatible keys; safer during research iteration
}


# --- Authoritative allowed-prefix mapping for cross-reference fields ---
ALLOWED_REF_PREFIXES = {
    "depends_on": {"A", "E", "C"},
    "used_by": {"A", "E"},
    "supports_claims": {"C"},
    "supported_by_definitions": {"D"},
    "counterposition_targets": {"C", "A", "E", "D"},
    "uncertainty_targets": {"A", "E", "C", "D"},
    "inference_chain_reference": {"A", "E", "C"},
}


def _validate_ref_prefixes(
    refs: list, field_name: str, allowed_prefixes: set,
    node_id: str, errors: List[str]
) -> None:
    """Validate that all IDs in *refs* have an allowed prefix letter."""
    for ref in refs:
        if not isinstance(ref, str) or len(ref) < 2:
            continue
        prefix = ref[0]
        if prefix not in allowed_prefixes:
            sorted_prefixes = sorted(allowed_prefixes)
            errors.append(
                f"{node_id}.{field_name} contains '{ref}' with invalid prefix "
                f"'{prefix}'; allowed: {sorted_prefixes}"
            )


def _validate_sequential_ids(
    nodes: list, prefix: str, errors: List[str]
) -> None:
    """Validate that IDs for a given prefix are sequential starting from 1 with no gaps.

    Extracts all IDs matching ``prefix`` from the node list, sorts them
    numerically, and checks they form the sequence prefix1, prefix2, ...
    """
    ids = []
    for item in nodes:
        _id = item.get("id", "")
        if isinstance(_id, str) and _id.startswith(prefix) and _id[len(prefix):].isdigit():
            ids.append(int(_id[len(prefix):]))
    if not ids:
        return
    ids.sort()
    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        actual = [f"{prefix}{n}" for n in ids]
        exp = [f"{prefix}{n}" for n in expected]
        errors.append(
            f"{prefix}# IDs must be sequential starting from 1: "
            f"expected {exp}, got {actual}"
        )


VALID_IC_ROLES = {"premise", "inference", "conclusion"}
VALID_INFERENCE_TYPES = {"deductive", "inductive", "abductive"}
_IC_REFERENCE_RE = re.compile(r"^[ACE]\d+$")


def validate_inference_chain(
    ic: list,
    claim_id: str,
    errors: List[str],
) -> None:
    """Validate the structural integrity of a claim's inference_chain.

    Checks:
    1. Each step is a dict with required ``role`` and ``text`` keys.
    2. ``role`` is one of ``premise``, ``inference``, ``conclusion``.
    3. Premise steps must have a ``reference`` matching ``^[ACE]\\d+$``.
    4. Exactly one ``inference`` step, with a valid ``inference_type``.
    5. Exactly one ``conclusion`` step.
    6. At least one ``premise`` step.
    7. Ordering: all premises → inference → conclusion.
    """
    premise_count = 0
    inference_count = 0
    conclusion_count = 0
    last_role_rank = -1  # premise=0, inference=1, conclusion=2
    role_rank = {"premise": 0, "inference": 1, "conclusion": 2}

    for idx, step in enumerate(ic):
        step_label = f"inference_chain[{idx}]"

        if not isinstance(step, dict):
            errors.append(
                f"Claim '{claim_id}' {step_label} must be an object, "
                f"got {type(step).__name__}"
            )
            continue

        # Required keys
        role = step.get("role")
        text = step.get("text")

        if role is None:
            errors.append(
                f"Claim '{claim_id}' {step_label} is missing required 'role' field"
            )
        elif role not in VALID_IC_ROLES:
            errors.append(
                f"Claim '{claim_id}' {step_label} has invalid role '{role}'; "
                f"must be one of: {sorted(VALID_IC_ROLES)}"
            )

        if text is None or (isinstance(text, str) and not text.strip()):
            errors.append(
                f"Claim '{claim_id}' {step_label} is missing or has empty 'text' field"
            )

        if role not in VALID_IC_ROLES:
            continue  # Can't do further checks without a valid role

        # Ordering check
        rank = role_rank[role]
        if rank < last_role_rank:
            errors.append(
                f"Claim '{claim_id}' {step_label} role '{role}' is out of order; "
                f"expected order: premises → inference → conclusion"
            )
        last_role_rank = rank

        # Role-specific checks
        if role == "premise":
            premise_count += 1
            ref = step.get("reference")
            if ref is None:
                errors.append(
                    f"Claim '{claim_id}' {step_label} (premise) is missing "
                    f"required 'reference' field (must cite an A#, E#, or C# ID)"
                )
            elif not _IC_REFERENCE_RE.match(ref):
                errors.append(
                    f"Claim '{claim_id}' {step_label} reference '{ref}' "
                    f"must match A#/E#/C# format (e.g. 'A1', 'E2', 'C3')"
                )

        elif role == "inference":
            inference_count += 1
            inf_type = step.get("inference_type")
            if inf_type is None:
                errors.append(
                    f"Claim '{claim_id}' {step_label} (inference) is missing "
                    f"required 'inference_type' field"
                )
            elif inf_type not in VALID_INFERENCE_TYPES:
                errors.append(
                    f"Claim '{claim_id}' {step_label} inference_type '{inf_type}' "
                    f"is invalid; must be one of: {sorted(VALID_INFERENCE_TYPES)}"
                )

        elif role == "conclusion":
            conclusion_count += 1

    # Cardinality checks
    if premise_count == 0:
        errors.append(
            f"Claim '{claim_id}' inference_chain must have at least one premise step"
        )
    if inference_count == 0:
        errors.append(
            f"Claim '{claim_id}' inference_chain must have exactly one inference step"
        )
    elif inference_count > 1:
        errors.append(
            f"Claim '{claim_id}' inference_chain has {inference_count} inference steps; "
            f"exactly one is required"
        )
    if conclusion_count == 0:
        errors.append(
            f"Claim '{claim_id}' inference_chain must have exactly one conclusion step"
        )
    elif conclusion_count > 1:
        errors.append(
            f"Claim '{claim_id}' inference_chain has {conclusion_count} conclusion steps; "
            f"exactly one is required"
        )


def validate_belief(belief: Dict[str, Any]) -> List[str]:
    """
    Validate a CBS belief object.
    Returns a list of human-readable validation errors (empty list means OK).

    Notes:
    - If 'jsonschema' (JSON Schema validator) is not available, we perform essential checks only.
    - This validator intentionally focuses on structural validity, not semantic argument quality.
    """
    errors: List[str] = []

    # --- Minimal presence checks (robust even without jsonschema installed) ---
    for k in ("schema_version", "belief_id", "version", "metadata", "thesis",
              "definitions", "assumptions", "claims", "evidence",
              "counterpositions", "uncertainties"):
        if k not in belief:
            errors.append(f"Missing required top-level key: '{k}'")

    # Enforce the exact schema label (useful for migrations later)
    if belief.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be '{SCHEMA_VERSION}'")

    # --- Version type and range checks ---
    ver = belief.get("version")
    if ver is not None and not isinstance(ver, int):
        errors.append(f"'version' must be an integer, not {type(ver).__name__}")
    elif isinstance(ver, int) and ver < 1:
        errors.append(f"'version' must be >= 1, got {ver}")

    # --- Thesis field-level checks ---
    thesis = belief.get("thesis")
    if isinstance(thesis, dict):
        t_str = thesis.get("strength")
        if t_str is not None and not (0.0 <= t_str <= 1.0):
            errors.append(
                f"thesis.strength {t_str} out of range [0.0, 1.0]"
            )
        bullets = thesis.get("summary_bullets")
        if isinstance(bullets, list) and len(bullets) == 0:
            errors.append("thesis.summary_bullets must not be empty")

    # --- Type checks for fields that must be dicts/arrays ---
    # These are critical even without jsonschema, since downstream code
    # calls .get() on these values and will crash if they are strings.
    if "thesis" in belief and not isinstance(belief["thesis"], dict):
        errors.append("'thesis' must be an object, not a " + type(belief["thesis"]).__name__)
    if "metadata" in belief and not isinstance(belief["metadata"], dict):
        errors.append("'metadata' must be an object, not a " + type(belief["metadata"]).__name__)
    for arr_key in ("definitions", "assumptions", "claims", "evidence",
                     "uncertainties", "counterpositions", "changelog"):
        if arr_key in belief and not isinstance(belief[arr_key], list):
            errors.append(f"'{arr_key}' must be an array, not a " + type(belief[arr_key]).__name__)

    # --- Optional: deep validation via jsonschema (if installed) ---
    if HAVE_JSONSCHEMA:
        try:
            jsonschema.validate(instance=belief, schema=CBS_JSON_SCHEMA)
        except Exception as e:
            errors.append(str(e))

    # --- ID hygiene checks for known collections ---
    # id_fields lists every array that should contain items with prefixed IDs
    id_fields = (
        "definitions",
        "assumptions",
        "claims",
        "evidence",
        "uncertainties",
        "counterpositions"
    )

    # prefix_ok maps the first letter of an ID to the expected collection
    # D#: definitions, A#: assumptions, C#: claims, E#: evidence, U#: uncertainties, X#: counterpositions
    prefix_ok = {
        "D": "definitions",
        "A": "assumptions",
        "C": "claims",
        "E": "evidence",
        "U": "uncertainties",
        "X": "counterpositions"
    }

    # Track all IDs across all collections for duplicate detection
    all_ids: set[str] = set()

    for field in id_fields:
        for item in belief.get(field, []) or []:
            _id = item.get("id")
            if not _id:
                # If there's no 'id' key, we skip here—jsonschema (if present) can catch detailed structure issues.
                continue
            # Enforce simple 'Prefix + digits' format, e.g., "D1", "C1", "E12"
            if not re.match(r"^[ACDEUX]\d+$", _id):
                errors.append(
                    f"Invalid ID format '{_id}' in '{field}'. Expected like 'D1', 'C1', 'E2', etc."
                )
            # Ensure IDs live in the correct collection by their prefix
            exp_field = prefix_ok.get(_id[0])
            if exp_field and exp_field != field:
                errors.append(
                    f"ID '{_id}' has wrong prefix for field '{field}'"
                )
            # Check for duplicate IDs
            if _id in all_ids:
                errors.append(
                    f"Duplicate ID '{_id}' found in '{field}'. IDs must be unique across all collections."
                )
            all_ids.add(_id)

    # --- Sequential ID checks (no gaps, starting from 1) ---
    _validate_sequential_ids(belief.get("definitions", []) or [], "D", errors)
    _validate_sequential_ids(belief.get("assumptions", []) or [], "A", errors)
    _validate_sequential_ids(belief.get("claims", []) or [], "C", errors)
    _validate_sequential_ids(belief.get("evidence", []) or [], "E", errors)
    _validate_sequential_ids(belief.get("uncertainties", []) or [], "U", errors)
    _validate_sequential_ids(belief.get("counterpositions", []) or [], "X", errors)

    # --- D# definition field-level validation ---
    VALID_DEF_STATUSES = {"active", "revised", "retracted"}
    def_ids: set[str] = set()
    for item in belief.get("definitions", []) or []:
        d_id = item.get("id", "?")
        def_ids.add(d_id)
        # term: non-empty string
        term = item.get("term")
        if not term or not isinstance(term, str) or not term.strip():
            errors.append(f"Definition '{d_id}' has empty or missing 'term'")
        # definition: non-empty string
        defn = item.get("definition")
        if not defn or not isinstance(defn, str) or not defn.strip():
            errors.append(f"Definition '{d_id}' has empty or missing 'definition'")
        # strength range
        d_str = item.get("strength")
        if d_str is not None and not (0.0 <= d_str <= 1.0):
            errors.append(f"Definition '{d_id}' strength {d_str} out of range [0.0, 1.0]")
        # strength_justification required
        if "strength_justification" not in item:
            errors.append(f"Definition '{d_id}' is missing required 'strength_justification' field")
        # status enum
        d_status = item.get("status")
        if d_status is not None and d_status not in VALID_DEF_STATUSES:
            errors.append(
                f"Invalid definition status '{d_status}'. "
                f"Must be one of: {sorted(VALID_DEF_STATUSES)}"
            )
        # used_by: non-empty array
        used_by = item.get("used_by")
        if not used_by or not isinstance(used_by, list) or len(used_by) == 0:
            errors.append(f"Definition '{d_id}' must have non-empty 'used_by' array")

    # --- D# cross-reference validation ---
    # Collect A# and E# IDs for cross-reference checks
    assumption_ids = {item.get("id") for item in (belief.get("assumptions", []) or []) if item.get("id")}
    evidence_ids = {item.get("id") for item in (belief.get("evidence", []) or []) if item.get("id")}
    ae_ids = assumption_ids | evidence_ids

    # D#.used_by must reference existing A# or E# IDs
    for item in belief.get("definitions", []) or []:
        d_id = item.get("id", "?")
        for ref in (item.get("used_by") or []):
            if ref not in ae_ids:
                errors.append(f"Definition '{d_id}' used_by references non-existent node '{ref}'")

    # A#.supported_by_definitions must reference existing D# IDs
    for item in belief.get("assumptions", []) or []:
        a_id = item.get("id", "?")
        sbd = item.get("supported_by_definitions")
        if sbd is not None:
            for ref in sbd:
                if ref not in def_ids:
                    errors.append(f"Assumption '{a_id}' supported_by_definitions references non-existent definition '{ref}'")

    # E#.supported_by_definitions must reference existing D# IDs
    for item in belief.get("evidence", []) or []:
        e_id = item.get("id", "?")
        sbd = item.get("supported_by_definitions")
        if sbd is not None:
            for ref in sbd:
                if ref not in def_ids:
                    errors.append(f"Evidence '{e_id}' supported_by_definitions references non-existent definition '{ref}'")

    # Bidirectional consistency: D#.used_by ↔ A#/E#.supported_by_definitions
    # Build reverse map: for each D#, which A#/E# should reference it
    for d_item in belief.get("definitions", []) or []:
        d_id = d_item.get("id", "?")
        for ae_ref in (d_item.get("used_by") or []):
            # Find the A# or E# node and check it references back
            ae_node = None
            for a_item in (belief.get("assumptions", []) or []):
                if a_item.get("id") == ae_ref:
                    ae_node = a_item
                    break
            if ae_node is None:
                for e_item in (belief.get("evidence", []) or []):
                    if e_item.get("id") == ae_ref:
                        ae_node = e_item
                        break
            if ae_node is not None:
                sbd = ae_node.get("supported_by_definitions") or []
                if d_id not in sbd:
                    errors.append(
                        f"Bidirectional inconsistency: D#{d_id} lists '{ae_ref}' in used_by, "
                        f"but '{ae_ref}' does not list '{d_id}' in supported_by_definitions"
                    )

    # Reverse direction: A#/E# references D# but D# doesn't list the A#/E#
    for collection_key in ("assumptions", "evidence"):
        for ae_item in (belief.get(collection_key, []) or []):
            ae_id = ae_item.get("id", "?")
            for d_ref in (ae_item.get("supported_by_definitions") or []):
                d_node = None
                for d_item in (belief.get("definitions", []) or []):
                    if d_item.get("id") == d_ref:
                        d_node = d_item
                        break
                if d_node is not None:
                    used_by = d_node.get("used_by") or []
                    if ae_id not in used_by:
                        errors.append(
                            f"Bidirectional inconsistency: '{ae_id}' lists '{d_ref}' in supported_by_definitions, "
                            f"but D#{d_ref} does not list '{ae_id}' in used_by"
                        )

    # --- Edge-type prefix validation for cross-reference fields ---
    for item in belief.get("claims", []) or []:
        c_id = item.get("id", "?")
        _validate_ref_prefixes(
            item.get("depends_on") or [], "depends_on",
            ALLOWED_REF_PREFIXES["depends_on"], c_id, errors
        )

    for item in belief.get("assumptions", []) or []:
        a_id = item.get("id", "?")
        _validate_ref_prefixes(
            item.get("supports_claims") or [], "supports_claims",
            ALLOWED_REF_PREFIXES["supports_claims"], a_id, errors
        )
        _validate_ref_prefixes(
            item.get("supported_by_definitions") or [], "supported_by_definitions",
            ALLOWED_REF_PREFIXES["supported_by_definitions"], a_id, errors
        )

    for item in belief.get("evidence", []) or []:
        e_id = item.get("id", "?")
        _validate_ref_prefixes(
            item.get("supports_claims") or [], "supports_claims",
            ALLOWED_REF_PREFIXES["supports_claims"], e_id, errors
        )
        _validate_ref_prefixes(
            item.get("supported_by_definitions") or [], "supported_by_definitions",
            ALLOWED_REF_PREFIXES["supported_by_definitions"], e_id, errors
        )

    for item in belief.get("counterpositions", []) or []:
        x_id = item.get("id", "?")
        _validate_ref_prefixes(
            item.get("targets") or [], "targets",
            ALLOWED_REF_PREFIXES["counterposition_targets"], x_id, errors
        )

    for item in belief.get("uncertainties", []) or []:
        u_id = item.get("id", "?")
        _validate_ref_prefixes(
            item.get("targets") or [], "targets",
            ALLOWED_REF_PREFIXES["uncertainty_targets"], u_id, errors
        )

    # --- Enum validation for v3 typed fields (works without jsonschema) ---
    VALID_ASSUMPTION_TYPES = {"foundational", "empirical", "methodological", "scoping"}
    for item in belief.get("assumptions", []) or []:
        atype = item.get("type")
        if atype is not None and atype not in VALID_ASSUMPTION_TYPES:
            errors.append(
                f"Invalid assumption type '{atype}'. "
                f"Must be one of: {sorted(VALID_ASSUMPTION_TYPES)}"
            )
        elif "type" not in item and item.get("id"):
            errors.append(
                f"Assumption '{item.get('id')}' is missing required 'type' field"
            )

    from chal.utilities.utils import VALID_ATTACK_STRATEGIES
    VALID_ATTACK_TYPES = {"undermining", "rebutting", "undercutting"}
    VALID_SUFFICIENCY = {"sufficient", "partial", "unaddressed"}
    for item in belief.get("counterpositions", []) or []:
        x_id = item.get("id", "?")
        at = item.get("attack_type")
        if at is not None and at not in VALID_ATTACK_TYPES:
            errors.append(
                f"Invalid attack_type '{at}'. "
                f"Must be one of: {sorted(VALID_ATTACK_TYPES)}"
            )
        rs = item.get("response_sufficiency")
        if rs is not None and rs not in VALID_SUFFICIENCY:
            errors.append(
                f"Invalid response_sufficiency '{rs}'. "
                f"Must be one of: {sorted(VALID_SUFFICIENCY)}"
            )
        # attack_strategy is required and must be valid for the attack_type
        a_strat = item.get("attack_strategy")
        if a_strat is None or a_strat == "":
            errors.append(f"Counterposition '{x_id}' is missing required 'attack_strategy' field")
        elif at in VALID_ATTACK_STRATEGIES:
            if a_strat not in VALID_ATTACK_STRATEGIES[at]:
                errors.append(
                    f"Counterposition '{x_id}' attack_strategy '{a_strat}' is not valid "
                    f"for attack_type '{at}' "
                    f"(valid: {', '.join(sorted(VALID_ATTACK_STRATEGIES[at]))})"
                )

    VALID_CLAIM_STATUSES = {"active", "revised", "retracted"}
    for item in belief.get("claims", []) or []:
        status = item.get("status")
        if status is not None and status not in VALID_CLAIM_STATUSES:
            errors.append(
                f"Invalid claim status '{status}'. "
                f"Must be one of: {sorted(VALID_CLAIM_STATUSES)}"
            )
        c_str = item.get("strength")
        if c_str is not None and not (0.0 <= c_str <= 1.0):
            errors.append(
                f"Claim '{item.get('id')}' strength {c_str} out of range [0.0, 1.0]"
            )
        # Predictions are required on claims (at least one per claim)
        preds = item.get("predictions")
        claim_id = item.get("id", "?")
        if preds is None:
            errors.append(
                f"Claim '{claim_id}' is missing required 'predictions' field"
            )
        elif not isinstance(preds, list):
            errors.append(
                f"Claim '{claim_id}' predictions must be an array"
            )
        elif len(preds) == 0:
            errors.append(
                f"Claim '{claim_id}' predictions must have at least one item"
            )
        else:
            for j, pred in enumerate(preds):
                if not isinstance(pred, dict):
                    errors.append(
                        f"Claim '{claim_id}' prediction {j} must be an object"
                    )
                    continue
                for req_field in ("statement", "test", "decision_criterion"):
                    if req_field not in pred:
                        errors.append(
                            f"Claim '{claim_id}' prediction {j} missing required field '{req_field}'"
                        )
        # strength_justification is required on claims
        if "strength_justification" not in item:
            errors.append(
                f"Claim '{claim_id}' is missing required 'strength_justification' field"
            )
        # inference_chain is required on claims — structured validation
        ic = item.get("inference_chain")
        if ic is None:
            errors.append(
                f"Claim '{claim_id}' is missing required 'inference_chain' field"
            )
        elif not isinstance(ic, list):
            errors.append(
                f"Claim '{claim_id}' inference_chain must be an array"
            )
        elif len(ic) == 0:
            errors.append(
                f"Claim '{claim_id}' inference_chain must have at least one step"
            )
        else:
            validate_inference_chain(ic, claim_id, errors)

    VALID_EVIDENCE_TYPES = {"empirical", "conceptual", "expert_consensus"}
    for item in belief.get("evidence", []) or []:
        etype = item.get("type")
        if etype is not None and etype not in VALID_EVIDENCE_TYPES:
            errors.append(
                f"Invalid evidence type '{etype}'. "
                f"Must be one of: {sorted(VALID_EVIDENCE_TYPES)}"
            )
        e_str = item.get("strength")
        if e_str is not None and not (0.0 <= e_str <= 1.0):
            errors.append(
                f"Evidence '{item.get('id')}' strength {e_str} out of range [0.0, 1.0]"
            )
        # strength_justification is required on evidence
        if "strength_justification" not in item:
            errors.append(
                f"Evidence '{item.get('id', '?')}' is missing required 'strength_justification' field"
            )

    # --- Assumption strength range and status checks ---
    VALID_ASSUMPTION_STATUSES = {"active", "revised", "retracted"}
    for item in belief.get("assumptions", []) or []:
        a_str = item.get("strength")
        if a_str is not None and not (0.0 <= a_str <= 1.0):
            errors.append(
                f"Assumption '{item.get('id')}' strength {a_str} out of range [0.0, 1.0]"
            )
        a_status = item.get("status")
        if a_status is not None and a_status not in VALID_ASSUMPTION_STATUSES:
            errors.append(
                f"Invalid assumption status '{a_status}'. "
                f"Must be one of: {sorted(VALID_ASSUMPTION_STATUSES)}"
            )
        # strength_justification is required on assumptions
        if "strength_justification" not in item:
            errors.append(
                f"Assumption '{item.get('id', '?')}' is missing required 'strength_justification' field"
            )

    # --- Evidence status checks ---
    VALID_EVIDENCE_STATUSES = {"active", "revised", "retracted"}
    for item in belief.get("evidence", []) or []:
        e_status = item.get("status")
        if e_status is not None and e_status not in VALID_EVIDENCE_STATUSES:
            errors.append(
                f"Invalid evidence status '{e_status}'. "
                f"Must be one of: {sorted(VALID_EVIDENCE_STATUSES)}"
            )

    # --- Uncertainty status and importance validation ---
    VALID_UNCERTAINTY_STATUS = {"active", "resolved"}
    VALID_UNCERTAINTY_IMPORTANCE = {"high", "medium", "low"}
    for item in belief.get("uncertainties", []) or []:
        u_id = item.get("id", "?")
        u_status = item.get("status")
        if u_status is not None and u_status not in VALID_UNCERTAINTY_STATUS:
            errors.append(
                f"Invalid uncertainty status '{u_status}'. "
                f"Must be one of: {sorted(VALID_UNCERTAINTY_STATUS)}"
            )
        u_importance = item.get("importance")
        if u_importance is None:
            errors.append(
                f"Uncertainty '{u_id}' is missing required 'importance' field"
            )
        elif u_importance not in VALID_UNCERTAINTY_IMPORTANCE:
            errors.append(
                f"Invalid uncertainty importance '{u_importance}'. "
                f"Must be one of: {sorted(VALID_UNCERTAINTY_IMPORTANCE)}"
            )

    return errors