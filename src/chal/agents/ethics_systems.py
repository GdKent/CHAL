"""
ethics_systems.py

Defines the ethical frameworks available for the adjudicator.
Each system describes an approach to evaluating the ethical dimensions
of arguments during debate adjudication.

Usage:
    from chal.agents.ethics_systems import get_ethics_system, ETHICS_SYSTEMS

    description = get_ethics_system("UTILITARIAN")
    all_keys = list(ETHICS_SYSTEMS.keys())
"""


# === Ethics System Definitions ===

NONE = (
    "No ethical framework applied. Evaluate arguments solely on logical "
    "rigor and soundness, not ethical implications."
)

UTILITARIAN = (
    "Consequentialist utilitarianism: evaluate arguments by whether their "
    "conclusions maximize well-being and minimize suffering for the "
    "greatest number. Prefer positions that produce the best overall "
    "outcomes regardless of the means."
)

DEONTOLOGICAL = (
    "Kantian deontological ethics: evaluate whether arguments respect "
    "universal moral duties, autonomy, and the categorical imperative "
    "regardless of consequences. An argument that violates moral duties "
    "is ethically deficient even if it leads to good outcomes."
)

VIRTUE_ETHICS = (
    "Aristotelian virtue ethics: evaluate whether arguments promote human "
    "flourishing (eudaimonia) and reflect practical wisdom, courage, and "
    "temperance. Favor positions that cultivate excellence of character "
    "and contribute to the good life."
)

CARE_ETHICS = (
    "Care ethics: evaluate arguments through the lens of relationships, "
    "responsibility, and attentiveness to context-dependent needs of "
    "affected parties. Prioritize responsiveness to vulnerability and "
    "the maintenance of caring relationships."
)

BALANCED = (
    "Balanced consequentialist-deontological: weigh both outcomes/welfare "
    "and autonomy/rights. Neither consequences nor duties alone are "
    "sufficient; evaluate arguments for both their practical impact and "
    "their respect for moral principles."
)


# === Lookup Structures ===

ETHICS_SYSTEMS = {
    "NONE": NONE,
    "UTILITARIAN": UTILITARIAN,
    "DEONTOLOGICAL": DEONTOLOGICAL,
    "VIRTUE_ETHICS": VIRTUE_ETHICS,
    "CARE_ETHICS": CARE_ETHICS,
    "BALANCED": BALANCED,
}

ETHICS_LABELS = {
    "NONE": "None (Pure Logic)",
    "UTILITARIAN": "Utilitarian",
    "DEONTOLOGICAL": "Deontological (Kantian)",
    "VIRTUE_ETHICS": "Virtue Ethics (Aristotelian)",
    "CARE_ETHICS": "Care Ethics",
    "BALANCED": "Balanced (Consequentialist-Deontological)",
}

ETHICS_DESCRIPTIONS = {
    "NONE": (
        "No ethical evaluation; judge only logical soundness"
    ),
    "UTILITARIAN": (
        "Maximize well-being and minimize suffering for the greatest number"
    ),
    "DEONTOLOGICAL": (
        "Respect universal moral duties and the categorical imperative"
    ),
    "VIRTUE_ETHICS": (
        "Promote human flourishing, practical wisdom, and excellence"
    ),
    "CARE_ETHICS": (
        "Prioritize relationships, responsibility, and vulnerability"
    ),
    "BALANCED": (
        "Weigh both outcomes/welfare and autonomy/rights equally"
    ),
}


def get_ethics_system(key: str) -> str:
    """
    Look up an ethics system description by key.

    Args:
        key: Ethics system identifier (e.g., "UTILITARIAN").
             Case-insensitive.

    Returns:
        The full ethics system description text.

    Raises:
        KeyError: If the key does not match any known ethics system.
    """
    return ETHICS_SYSTEMS[key.upper()]
