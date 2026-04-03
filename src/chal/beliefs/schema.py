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
                "agent_persona"                 # Persona label (e.g., "Empiricist", "Skeptic")
            ],
            "properties": {
                "topic_query": { "type": "string" },     # Query or topic string
                "agent_persona": { "type": "string" },   # Agent persona or role name
                "last_updated": { "type": "string" },    # Optional ISO-8601 timestamp (set by patch system)
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
                }
            }
        },

        # --- Optional but recommended sections (lists of structured nodes) ---

        "assumptions": {                             # List of A# items with typed categories, each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "statement", "supports_claims", "strength", "status", "strength_justification"],
                "properties": {
                    "id": { "type": "string" },
                    "type": {
                        "type": "string",
                        "enum": ["foundational", "empirical", "methodological"]
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
                    "strength_justification": { "type": "string" }  # Rationale for the strength number
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
                    "inference_chain": {                        # Steps of reasoning leading to this claim
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1
                    }
                }
            }
        },
        "evidence": {                                # List of E# items, each with 'id'
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "summary", "source", "relevance_to_claims",
                             "strength", "status", "strength_justification"],
                "properties": {
                    "id": { "type": "string" },
                    "type": {
                        "type": "string",
                        "enum": ["empirical", "conceptual", "expert_consensus"]
                    },
                    "summary": { "type": "string" },
                    "source": { "type": "string" },            # Citation or provenance string
                    "relevance_to_claims": {                   # Cross-references to C# IDs
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
                    "strength_justification": { "type": "string" }  # Rationale for the strength number
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
                "required": ["id", "targets", "attack_type", "statement", "my_response", "response_sufficiency"],
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
                    },
                    "timestamp": { "type": "string" }    # Optional — added by patch system, not by LLM
                }
            }
        }
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
    for arr_key in ("assumptions", "claims", "evidence",
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
        "assumptions",
        "claims",
        "evidence",
        "uncertainties",
        "counterpositions"
    )

    # prefix_ok maps the first letter of an ID to the expected collection
    # A#: assumptions, C#: claims, E#: evidence, U#: uncertainties, X#: counterpositions
    prefix_ok = {
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
            # Enforce simple 'Prefix + digits' format, e.g., "C1", "E12"
            if not re.match(r"^[ACEUX]\d+$", _id):
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
    VALID_ASSUMPTION_TYPES = {"foundational", "empirical", "methodological"}
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
        # inference_chain is required on claims (at least one step)
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