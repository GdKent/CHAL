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
        "thesis"
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

        # --- Metadata block: who/what/when/scope/definitions ---
        "metadata": {
            "type": "object",                   # A nested object containing descriptive metadata
            "required": [                       # Minimum metadata to track provenance and persona
                "topic_query",                  # The debate question (e.g., "Does free will exist?")
                "agent_persona",                # Persona label (e.g., "Empiricist", "Skeptic")
                "created_at"                    # Creation date/time (ISO-8601 preferred)
            ],
            "properties": {
                "topic_query": { "type": "string" },     # Query or topic string
                "agent_persona": { "type": "string" },   # Agent persona or role name
                "created_at": { "type": "string" },      # ISO-8601 timestamp recommended, but not enforced
                "last_updated": { "type": "string" },    # Optional ISO-8601 timestamp for last modification
                "scope_conditions": { "type": "string" },# Optional scoping statement (assumptions about domain/scope)
                "definitions": {                         # Optional glossary for key terms used in the belief
                    "type": "array",                     # A list of term-definition objects
                    "items": {
                        "type": "object",                # Each glossary entry is an object with a term and definition
                        "required": ["term", "definition"],
                        "properties": {
                            "term": { "type": "string" },       # The technical term being defined
                            "definition": { "type": "string" }  # The meaning of the term as used in this belief
                        }
                    }
                }
            }
        },

        # --- Thesis block: concise position summary ---
        "thesis": {
            "type": "object",                      # Core stance of the belief
            "required": [
                "stance",                          # One-sentence position statement
                "summary_bullets",                 # 2–3 short bullets highlighting key points
                "confidence"                       # Author's calibrated confidence in [0, 1]
            ],
            "properties": {
                "stance": { "type": "string" },    # Single-sentence thesis
                "summary_bullets": {               # Short bullet points expanding the stance
                    "type": "array",
                    "items": { "type": "string" }, # Each bullet is a short string
                    "minItems": 1                  # At least one bullet (recommend 2–3)
                },
                "confidence": {                    # Self-reported confidence (0 = no confidence, 1 = certain)
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            }
        },

        # --- Optional but recommended sections (lists of structured nodes) ---

        "assumptions": {                             # List of A# items with typed categories, each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "statement"],
                "properties": {
                    "id": { "type": "string" },
                    "type": {
                        "type": "string",
                        "enum": ["foundational", "empirical", "methodological", "normative"]
                    },
                    "statement": { "type": "string" }
                }
            }
        },
        "claims": { "type": "array" },            # List of C# items (deductive/inductive/abductive claims), each with 'id'
        "evidence": { "type": "array" },          # List of E# items (empirical/conceptual/expert_consensus), each with 'id'
        "predictions": { "type": "array" },       # List of P# items (falsifiable consequences/tests), each with 'id'
        "normative_implications": { "type": "array" },  # List of N# items (ethical/policy implications), each with 'id'
        "uncertainties": { "type": "array" },     # List of U# items (key unknowns + VOI notes), each with 'id'
        "counterpositions": {                        # List of X# items (opponent objections + responses), each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "targets", "attack_type", "statement", "strength", "my_response", "response_sufficiency"],
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
                    "statement": { "type": "string" },
                    "strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "my_response": { "type": "string" },
                    "response_sufficiency": {
                        "type": "string",
                        "enum": ["sufficient", "partial", "unaddressed"]
                    }
                }
            }
        },

        # --- Governance of updates & audit trail ---

        #"update_policy": { "type": "object" },    # Optional rules: revision triggers, confidence update rule, retirement criteria
        "changelog": { "type": "array" }          # Optional list of versioned change records (who/when/what changed)
    },
    "additionalProperties": True  # Allow forward-compatible keys; safer during research iteration
}


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
    for k in ("schema_version", "belief_id", "version", "metadata", "thesis"):
        if k not in belief:
            errors.append(f"Missing required top-level key: '{k}'")

    # Enforce the exact schema label (useful for migrations later)
    if belief.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be '{SCHEMA_VERSION}'")

    # --- Type checks for fields that must be dicts/arrays ---
    # These are critical even without jsonschema, since downstream code
    # calls .get() on these values and will crash if they are strings.
    if "thesis" in belief and not isinstance(belief["thesis"], dict):
        errors.append("'thesis' must be an object, not a " + type(belief["thesis"]).__name__)
    if "metadata" in belief and not isinstance(belief["metadata"], dict):
        errors.append("'metadata' must be an object, not a " + type(belief["metadata"]).__name__)
    for arr_key in ("assumptions", "claims", "evidence", "predictions",
                     "normative_implications", "uncertainties", "counterpositions", "changelog"):
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
        "assumptions",
        "claims",
        "evidence",
        "predictions",
        "normative_implications",
        "uncertainties",
        "counterpositions"
    )

    # prefix_ok maps the first letter of an ID to the expected collection
    # A#: assumptions, C#: claims, E#: evidence, P#: predictions,
    # N#: normative_implications, U#: uncertainties, X#: counterpositions
    prefix_ok = {
        "A": "assumptions",
        "C": "claims",
        "E": "evidence",
        "P": "predictions",
        "N": "normative_implications",
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
            # Enforce simple 'Prefix + digits' format, e.g., "C1", "E12"
            if not re.match(r"^[ACEPNUX]\d+$", _id):
                errors.append(
                    f"Invalid ID format '{_id}' in '{field}'. Expected like 'C1', 'E2', etc."
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

    # --- Enum validation for v3 typed fields (works without jsonschema) ---
    VALID_ASSUMPTION_TYPES = {"foundational", "empirical", "methodological", "normative"}
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

    VALID_ATTACK_TYPES = {"undermining", "rebutting", "undercutting"}
    VALID_SUFFICIENCY = {"sufficient", "partial", "unaddressed"}
    for item in belief.get("counterpositions", []) or []:
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

    return errors