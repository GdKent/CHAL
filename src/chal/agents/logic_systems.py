"""
logic_systems.py

Defines the logic and reasoning frameworks available for the adjudicator.
Each system describes a formal or informal approach to evaluating the
logical validity of arguments during debate adjudication.

Usage:
    from chal.agents.logic_systems import get_logic_system, LOGIC_SYSTEMS

    description = get_logic_system("CLASSICAL_BAYESIAN")
    all_keys = list(LOGIC_SYSTEMS.keys())
"""


# === Logic System Definitions ===

CLASSICAL_BAYESIAN = (
    "Classical logic + Bayesian reasoning for inductive support; reject "
    "contradictions; prefer simpler hypotheses (Occam's Razor). Evaluate "
    "deductive validity, inductive strength, abductive coherence, and "
    "internal consistency across claims."
)

FORMAL_DEDUCTIVE = (
    "Strict formal deductive logic: only valid syllogisms and formally "
    "sound inferences accepted. Reject inductive and abductive reasoning "
    "entirely. An argument is valid only if its conclusion follows "
    "necessarily from its premises."
)

DIALECTICAL = (
    "Hegelian dialectical logic: thesis-antithesis-synthesis. "
    "Contradictions are productive and drive toward higher understanding. "
    "Evaluate whether opposing positions can be sublated into a more "
    "comprehensive synthesis rather than simply declaring a winner."
)

INFORMAL_CRITICAL = (
    "Informal logic and critical thinking: evaluate argument strength via "
    "fallacy identification, relevance, and sufficiency of evidence. "
    "Accept inductive and abductive reasoning when well-supported. Focus "
    "on practical reasoning quality over formal validity."
)

FUZZY_MULTIVALUED = (
    "Fuzzy and multi-valued logic: truth admits degrees between 0 and 1; "
    "partial truth is acceptable. Avoid binary true/false judgments. "
    "Evaluate the degree to which premises support conclusions rather "
    "than demanding all-or-nothing validity."
)

PARACONSISTENT = (
    "Paraconsistent logic: tolerate local contradictions without global "
    "explosion (ex contradictione quodlibet does not apply). Evaluate "
    "whether contradictions are contained and productive or whether they "
    "undermine the overall argument structure."
)


# === Lookup Structures ===

LOGIC_SYSTEMS = {
    "CLASSICAL_BAYESIAN": CLASSICAL_BAYESIAN,
    "FORMAL_DEDUCTIVE": FORMAL_DEDUCTIVE,
    "DIALECTICAL": DIALECTICAL,
    "INFORMAL_CRITICAL": INFORMAL_CRITICAL,
    "FUZZY_MULTIVALUED": FUZZY_MULTIVALUED,
    "PARACONSISTENT": PARACONSISTENT,
}

LOGIC_LABELS = {
    "CLASSICAL_BAYESIAN": "Classical + Bayesian",
    "FORMAL_DEDUCTIVE": "Formal Deductive",
    "DIALECTICAL": "Dialectical (Hegelian)",
    "INFORMAL_CRITICAL": "Informal / Critical Thinking",
    "FUZZY_MULTIVALUED": "Fuzzy / Multi-valued",
    "PARACONSISTENT": "Paraconsistent",
}

LOGIC_DESCRIPTIONS = {
    "CLASSICAL_BAYESIAN": (
        "Standard logic with Bayesian induction and Occam's Razor"
    ),
    "FORMAL_DEDUCTIVE": (
        "Only formally valid syllogisms; no inductive reasoning"
    ),
    "DIALECTICAL": (
        "Thesis-antithesis-synthesis; contradictions drive progress"
    ),
    "INFORMAL_CRITICAL": (
        "Fallacy detection, relevance, and evidence sufficiency"
    ),
    "FUZZY_MULTIVALUED": (
        "Degrees of truth between 0 and 1; no binary judgments"
    ),
    "PARACONSISTENT": (
        "Tolerates local contradictions without global explosion"
    ),
}


def get_logic_system(key: str) -> str:
    """
    Look up a logic system description by key.

    Args:
        key: Logic system identifier (e.g., "CLASSICAL_BAYESIAN").
             Case-insensitive.

    Returns:
        The full logic system description text.

    Raises:
        KeyError: If the key does not match any known logic system.
    """
    return LOGIC_SYSTEMS[key.upper()]
