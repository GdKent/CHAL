"""
ethics_systems.py

Defines the ethical frameworks available for the adjudicator.
Each system is a dict with:
  - label: Human-readable display name
  - description: Full description of the ethical framework
  - criteria: Dict with critique_valid, rebuttal_valid, unresolved lists

The NONE system has empty criteria lists (no ethical evaluation).

Usage:
    from chal.agents.ethics_systems import get_ethics_system, ETHICS_SYSTEMS

    system = get_ethics_system("UTILITARIAN")
    description = get_ethics_system_description("DEONTOLOGICAL")
    all_keys = list(ETHICS_SYSTEMS.keys())
"""


# === Ethics System Definitions ===

NONE = {
    "label": "None (Pure Logic)",
    "description": (
        "No ethical framework applied. Evaluate arguments solely on logical "
        "rigor and soundness, not ethical implications."
    ),
    "criteria": {
        "critique_valid": [],
        "rebuttal_valid": [],
        "unresolved": [],
    },
}

UTILITARIAN = {
    "label": "Utilitarian",
    "description": (
        "Consequentialist utilitarianism: evaluate arguments by whether their "
        "conclusions maximize well-being and minimize suffering for the "
        "greatest number. Prefer positions that produce the best overall "
        "outcomes regardless of the means."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Net harm — the position leads to outcomes that demonstrably "
                "increase net suffering or decrease net well-being for affected "
                "parties"
            ),
            (
                "Stakeholder neglect — the position ignores or underweights "
                "significant negative consequences for a substantial number of "
                "affected people"
            ),
            (
                "Utility miscalculation — the position miscounts, double-counts, "
                "or omits relevant stakeholders in its assessment of consequences"
            ),
            (
                "Short-termism — the position favors immediate benefits while "
                "ignoring foreseeable long-term harms that would outweigh them"
            ),
            (
                "Distributional blindness — the position fails to account for "
                "how harms and benefits are distributed (e.g., concentrated "
                "severe harm to a few vs. diffuse mild benefit to many)"
            ),
            (
                "Generalization failure — the position’s approach, if adopted as "
                "a general policy, would produce worse aggregate outcomes than "
                "available alternatives"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the challenged position actually "
                "produces better aggregate outcomes than the alternative the "
                "challenger proposes or implies"
            ),
            (
                "The critique’s proposed alternative would itself cause greater "
                "net suffering when all consequences are considered"
            ),
            (
                "The critique exaggerates or fabricates harms without evidence "
                "of actual negative consequences"
            ),
            (
                "The defender accounts for long-term consequences that reverse "
                "the critique’s short-term harm analysis"
            ),
            (
                "The defender shows the identified harm is outweighed by "
                "proportionally greater benefits across the full set of affected "
                "stakeholders"
            ),
            (
                "The critique applies utilitarian calculus inconsistently — "
                "holding the defender to a standard it does not apply to its own "
                "position"
            ),
        ],
        "unresolved": [
            (
                "Both positions would produce significant benefits and "
                "significant harms, with no clear method to determine which has "
                "better net outcomes given available information"
            ),
            (
                "The utility comparison requires empirical data about "
                "consequences that neither side can provide"
            ),
            (
                "The disagreement turns on how to weigh incommensurable goods "
                "(e.g., liberty vs. safety, equality vs. efficiency) where "
                "reasonable aggregation methods diverge"
            ),
        ],
    },
}

DEONTOLOGICAL = {
    "label": "Deontological (Kantian)",
    "description": (
        "Kantian deontological ethics: evaluate whether arguments respect "
        "universal moral duties, autonomy, and the categorical imperative "
        "regardless of consequences. An argument that violates moral duties "
        "is ethically deficient even if it leads to good outcomes."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Universalizability failure — the maxim behind the endorsed "
                "action, if universalized as a law for all rational agents, leads "
                "to a contradiction (either logical or practical)"
            ),
            (
                "Instrumentalization — the position treats persons merely as "
                "means to an end rather than as ends in themselves, violating the "
                "humanity formulation of the categorical imperative"
            ),
            (
                "Duty violation — the position requires violating a recognized "
                "moral duty (truth-telling, promise-keeping, non-maleficence) "
                "without adequate deontological justification from a competing "
                "duty"
            ),
            (
                "Autonomy infringement — the position endorses an action that "
                "would override or undermine the rational autonomy of affected "
                "agents"
            ),
            (
                "Rights violation — the position requires infringing a "
                "fundamental right (bodily autonomy, freedom of conscience, due "
                "process) that deontological ethics treats as inviolable"
            ),
            (
                "Consequentialist smuggling — the position appeals solely to "
                "good outcomes to justify an action that is intrinsically "
                "impermissible under duty-based reasoning"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the maxim is universalizable without "
                "contradiction when properly formulated"
            ),
            (
                "The defender shows the position respects all persons as ends "
                "and the critique mischaracterizes the relationship between "
                "agents"
            ),
            (
                "No moral duty is actually violated, or the defender shows that "
                "competing duties are properly prioritized using established "
                "deontological frameworks (e.g., perfect duties override "
                "imperfect duties)"
            ),
            (
                "The critique itself advocates a course of action that would "
                "violate a stronger or more fundamental deontological duty"
            ),
            (
                "The defender demonstrates that the autonomy of rational agents "
                "is preserved or enhanced by the defended position"
            ),
            (
                "The apparent rights conflict resolves in favor of the defended "
                "position under established deontological priority rules (e.g., "
                "negative duties before positive duties)"
            ),
        ],
        "unresolved": [
            (
                "Genuine duty conflict — two legitimate moral duties point in "
                "opposite directions and no clear priority rule from "
                "deontological theory resolves which takes precedence"
            ),
            (
                "The universalizability test yields ambiguous results because "
                "the outcome depends on how the maxim is formulated, and both "
                "formulations are reasonable"
            ),
            (
                "Both positions respect moral duties but disagree about which "
                "duty applies to the situation or which takes priority"
            ),
        ],
    },
}

VIRTUE_ETHICS = {
    "label": "Virtue Ethics (Aristotelian)",
    "description": (
        "Aristotelian virtue ethics: evaluate whether arguments promote human "
        "flourishing (eudaimonia) and reflect practical wisdom, courage, and "
        "temperance. Favor positions that cultivate excellence of character "
        "and contribute to the good life."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Vice promotion — the position promotes a vice (excess or "
                "deficiency of a relevant trait) rather than the virtuous mean "
                "(e.g., recklessness rather than courage, or cowardice rather "
                "than prudent caution)"
            ),
            (
                "Phronesis failure — the position reflects a failure of practical "
                "wisdom by applying a rule or principle mechanically without "
                "attending to the particular circumstances of the situation"
            ),
            (
                "Flourishing undermined — the position would undermine eudaimonia "
                "(human flourishing) for the agents or communities involved if "
                "adopted"
            ),
            (
                "Akrasia — the position acknowledges or implies awareness of the "
                "better course of action but advocates for a worse one (weakness "
                "of will elevated to principle)"
            ),
            (
                "Character blindness — the position fails to model the kind of "
                "reasoning a person of excellent character (the phronimos) would "
                "employ in the situation"
            ),
            (
                "Inverted goods — the position prioritizes external or "
                "instrumental goods (wealth, status, power) over intrinsic goods "
                "(knowledge, virtue, friendship, meaningful activity)"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the position reflects the virtuous "
                "mean between the relevant extremes for the situation at hand"
            ),
            (
                "The defender shows practical wisdom by attending to contextual "
                "particulars and the specific features of the case, rather than "
                "mechanical rule application"
            ),
            (
                "The defender demonstrates the position contributes to human "
                "flourishing for the affected parties"
            ),
            (
                "The critique itself reflects a deficiency of a relevant "
                "virtue — for example, a failure of courage in facing difficult "
                "truths, or a failure of justice in distributing burdens"
            ),
            (
                "The defender shows the position models the kind of deliberation "
                "and judgment a phronimos would endorse"
            ),
            (
                "The defender demonstrates proper ordering of goods — intrinsic "
                "goods are not sacrificed for merely instrumental ones"
            ),
        ],
        "unresolved": [
            (
                "Both positions reflect different but legitimate conceptions of "
                "the virtuous response to the situation, corresponding to "
                "different but defensible virtue priorities"
            ),
            (
                "Practical wisdom (phronesis) could reasonably support either "
                "course of action given the particular circumstances, and no "
                "clearly superior character exemplar resolves the tie"
            ),
            (
                "The disagreement reflects a genuine tension between virtues "
                "(e.g., honesty vs. compassion, justice vs. mercy) where the "
                "tradition provides no clear hierarchy"
            ),
        ],
    },
}

CARE_ETHICS = {
    "label": "Care Ethics",
    "description": (
        "Care ethics: evaluate arguments through the lens of relationships, "
        "responsibility, and attentiveness to context-dependent needs of "
        "affected parties. Prioritize responsiveness to vulnerability and "
        "the maintenance of caring relationships."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Vulnerability neglect — the position ignores or would harm "
                "vulnerable parties who would be directly affected by the outcome"
            ),
            (
                "Abstraction over context — the position applies abstract "
                "principles or rules while ignoring the concrete relational "
                "context and the particular circumstances of those involved"
            ),
            (
                "Relational damage — the position would damage or neglect "
                "important caring relationships between affected parties"
            ),
            (
                "Unmet dependency — the position fails to respond to a clear "
                "dependency or need that creates an obligation of care (a party "
                "who depends on the outcome is left unattended)"
            ),
            (
                "Interchangeability assumption — the position treats affected "
                "persons as fungible or interchangeable rather than attending to "
                "their particular situations, histories, and needs"
            ),
            (
                "Power asymmetry ignored — the position fails to account for "
                "asymmetric power dynamics between the parties involved, treating "
                "all parties as if they were equally situated"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates attentiveness to the specific needs "
                "of vulnerable or dependent parties affected by the situation"
            ),
            (
                "The defender shows the position strengthens or preserves the "
                "web of caring relationships rather than damaging it"
            ),
            (
                "The defender accounts for the relational context and particular "
                "circumstances that the critique abstracts away or ignores"
            ),
            (
                "The critique itself applies abstract rules or principles "
                "without attending to the concrete needs and situations of "
                "affected parties"
            ),
            (
                "The defender shows responsible engagement with the power "
                "dynamics and dependencies in the situation"
            ),
            (
                "The defender demonstrates the position responds with "
                "appropriate specificity to the particular needs of those who "
                "depend on the outcome"
            ),
        ],
        "unresolved": [
            (
                "Both positions reflect genuine concern for affected parties but "
                "prioritize different relationships, dependencies, or "
                "vulnerabilities that cannot all be served simultaneously"
            ),
            (
                "Caring for one set of affected parties unavoidably comes at the "
                "expense of another, and no clear relational priority determines "
                "which caring obligation takes precedence"
            ),
            (
                "The particular context is too ambiguous or complex to determine "
                "which position better serves the web of caring relationships"
            ),
        ],
    },
}

BALANCED = {
    "label": "Balanced (Consequentialist-Deontological)",
    "description": (
        "Balanced consequentialist-deontological: weigh both outcomes/welfare "
        "and autonomy/rights. Neither consequences nor duties alone are "
        "sufficient; evaluate arguments for both their practical impact and "
        "their respect for moral principles."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Dual-axis failure — the position violates moral duties AND "
                "produces worse aggregate outcomes, failing on both the "
                "deontological and consequentialist dimensions"
            ),
            (
                "One-sided ethical reasoning — the position considers only "
                "consequences while ignoring duties, or only duties while "
                "ignoring consequences, when both are relevant"
            ),
            (
                "Disproportionate duty — the position respects a moral duty but "
                "at a disproportionate cost to overall well-being that a balanced "
                "framework cannot justify (the duty-consequence trade-off is "
                "unreasonable)"
            ),
            (
                "Disproportionate consequence — the position produces good "
                "outcomes but through means that violate duties in a way the good "
                "consequences cannot adequately redeem"
            ),
            (
                "Opaque trade-offs — the position fails to make its trade-offs "
                "between consequences and duties explicit, obscuring the ethical "
                "reasoning and preventing proper evaluation"
            ),
            (
                "Rights-welfare blindspot — the position attends to either the "
                "rights of individuals or the welfare of the broader group, but "
                "not both"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the position satisfies both "
                "consequentialist and deontological considerations — it produces "
                "good outcomes without violating moral duties"
            ),
            (
                "The defender shows that the trade-off between duties and "
                "consequences is explicitly acknowledged and proportionally "
                "balanced"
            ),
            (
                "The critique applies only one ethical lens (purely "
                "consequentialist or purely deontological) where the balanced "
                "framework requires both"
            ),
            (
                "The defender demonstrates good outcomes are achieved WITHOUT "
                "violating moral duties, so no trade-off is actually required"
            ),
            (
                "The defender shows the apparent duty violation is a justified "
                "exception given the severity of consequences, with explicit "
                "reasoning for why consequences override in this case"
            ),
            (
                "The defender demonstrates attention to both individual rights "
                "and aggregate welfare, addressing the interests of both"
            ),
        ],
        "unresolved": [
            (
                "The position performs well on one ethical axis (consequences or "
                "duties) but poorly on the other, and neither the framework nor "
                "the evidence provides a clear method for weighing them against "
                "each other"
            ),
            (
                "Both positions make defensible but different trade-offs between "
                "consequentialist and deontological considerations, each "
                "sacrificing something the other preserves"
            ),
            (
                "The disagreement turns on the relative weight of duties vs. "
                "consequences in this particular situation, which the balanced "
                "framework intentionally leaves as a matter of judgment rather "
                "than a fixed rule"
            ),
        ],
    },
}


# === Lookup Structures ===

ETHICS_SYSTEMS = {
    "NONE": NONE,
    "UTILITARIAN": UTILITARIAN,
    "DEONTOLOGICAL": DEONTOLOGICAL,
    "VIRTUE_ETHICS": VIRTUE_ETHICS,
    "CARE_ETHICS": CARE_ETHICS,
    "BALANCED": BALANCED,
}


# === Lookup Functions ===

def get_ethics_system(key: str) -> dict:
    """
    Look up an ethics system by key.

    Args:
        key: Ethics system identifier (e.g., "UTILITARIAN").
             Case-insensitive.

    Returns:
        The full ethics system dict with label, description, and criteria.

    Raises:
        KeyError: If the key does not match any known ethics system.
    """
    return ETHICS_SYSTEMS[key.upper()]


def get_ethics_system_description(key: str) -> str:
    """
    Look up an ethics system's description string by key.

    Convenience function for callers that only need the description text.

    Args:
        key: Ethics system identifier. Case-insensitive.

    Returns:
        The description string for the ethics system.

    Raises:
        KeyError: If the key does not match any known ethics system.
    """
    return ETHICS_SYSTEMS[key.upper()]["description"]


def get_ethics_system_label(key: str) -> str:
    """
    Look up an ethics system's human-readable label by key.

    Args:
        key: Ethics system identifier. Case-insensitive.

    Returns:
        The human-readable label string.

    Raises:
        KeyError: If the key does not match any known ethics system.
    """
    return ETHICS_SYSTEMS[key.upper()]["label"]
