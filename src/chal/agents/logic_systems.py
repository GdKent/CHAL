"""
logic_systems.py

Defines the logic and reasoning frameworks available for the adjudicator.
Each system is a dict with:
  - label: Human-readable display name
  - description: Full description of the reasoning framework
  - criteria: Dict with critique_valid, rebuttal_valid, unresolved lists

Usage:
    from chal.agents.logic_systems import get_logic_system, LOGIC_SYSTEMS

    system = get_logic_system("CLASSICAL_INFORMAL_BAYESIAN")
    description = get_logic_system_description("FORMAL_DEDUCTIVE")
    all_keys = list(LOGIC_SYSTEMS.keys())
"""


# === Logic System Definitions ===

CLASSICAL_INFORMAL_BAYESIAN = {
    "label": "Classical + Informal + Bayesian (Hybrid)",
    "description": (
        "Comprehensive hybrid: formal deductive validity as the foundation, "
        "informal fallacy detection as the practical layer, and Bayesian reasoning as "
        "the scientific evidence-evaluation framework. This is the most thorough "
        "general-purpose logic system — it evaluates deductive validity, inductive "
        "strength, abductive coherence, evidence quality, and reasoning hygiene."
    ),
    "criteria": {
        "critique_valid": [
            # Deductive structure (from FORMAL_DEDUCTIVE)
            (
                "Deductive invalidity — the conclusion does not follow necessarily "
                "from the premises"
            ),
            (
                "Named formal fallacy — the argument commits a recognized formal "
                "fallacy (undistributed middle, illicit major or minor, affirming the "
                "consequent, denying the antecedent, etc.) that invalidates the "
                "deductive structure"
            ),
            (
                "Hidden premise — the argument requires an unstated premise to be "
                "deductively valid, and that premise is not self-evident"
            ),
            (
                "Equivocation — a key term shifts meaning between uses, "
                "invalidating the reasoning"
            ),
            (
                "Reliance on inductive or probabilistic reasoning where deductive "
                "proof is required — the argument uses “likely,” “probably,” or "
                "statistical evidence instead of deductive entailment"
            ),
            # Evidence and probability (from BAYESIAN)
            (
                "Evidence misuse — the evidence does not support the claim as "
                "stated, or a correlation is asserted as causation without "
                "controlling for confounders"
            ),
            (
                "Prior probability neglected — an extraordinary claim is adopted "
                "without proportionally strong evidence to overcome a low prior "
                "(violation of Bayesian updating)"
            ),
            (
                "Base rate neglect — the argument ignores the base rate of the "
                "phenomenon when evaluating the significance of evidence (e.g., "
                "prosecutor’s fallacy, treating a rare positive test result as "
                "near-certain)"
            ),
            (
                "Asymmetric updating — confirming evidence is accepted while "
                "equally valid disconfirming evidence is dismissed or downweighted "
                "without justification"
            ),
            (
                "Likelihood neglect — the argument fails to consider how probable "
                "the observed evidence would be under competing hypotheses"
            ),
            (
                "Unfalsifiable claim treated as empirical — a non-testable "
                "assertion is used as though it were evidence for a descriptive "
                "conclusion"
            ),
            (
                "Occam’s Razor violation — an unnecessarily complex hypothesis is "
                "maintained when a simpler one accounts for the same evidence "
                "equally well"
            ),
            # Practical reasoning (from INFORMAL_CRITICAL)
            (
                "Named informal fallacy — the argument commits a recognized "
                "informal fallacy (ad hominem, red herring, false dilemma, slippery "
                "slope, appeal to emotion, tu quoque, appeal to authority, etc.) "
                "that materially weakens the conclusion"
            ),
            (
                "Insufficient evidence — claims are not adequately supported by "
                "the evidence provided, given the strength of the conclusion drawn"
            ),
            (
                "Irrelevance — a premise or piece of evidence does not actually "
                "bear on the conclusion it is meant to support"
            ),
            (
                "Hasty generalization — the conclusion is drawn from too few "
                "cases, unrepresentative samples, or anecdotal evidence"
            ),
            (
                "False analogy — a comparison relies on superficial similarity "
                "while ignoring relevant differences between the cases"
            ),
            (
                "Ambiguity or vagueness — key terms are too imprecise to "
                "evaluate, making the argument’s truth conditions unclear"
            ),
            (
                "Burden of proof violation — a claim is asserted without meeting "
                "the appropriate evidentiary standard for its type (empirical, "
                "normative, existential)"
            ),
            (
                "Suppressed evidence — relevant counterevidence is ignored, "
                "minimized, or selectively omitted to strengthen the appearance "
                "of the argument"
            ),
            # General reasoning (hybrid-original)
            (
                "Inference chain break — a step in the reasoning does not follow "
                "from the previous step (applies to deductive, inductive, and "
                "abductive chains alike)"
            ),
            (
                "Dependency failure — a claim rests on a premise that is false, "
                "unjustified, or unsupported by the cited evidence"
            ),
        ],
        "rebuttal_valid": [
            # Deductive defense (from FORMAL_DEDUCTIVE)
            (
                "The defender demonstrates the challenged inference is valid — by "
                "exhibiting the rule of inference, providing a truth-table, formal "
                "derivation, or proof"
            ),
            "The critique misidentifies the logical form of the argument",
            (
                "The defender supplies a missing premise and demonstrates it is "
                "true or self-evident, restoring deductive validity"
            ),
            (
                "The critique attacks an inductive auxiliary element rather than "
                "the core deductive structure"
            ),
            # Evidence/probability defense (from BAYESIAN)
            (
                "Bayesian counter — the defender shows the evidence has a higher "
                "likelihood under the defended hypothesis than under the "
                "challenger’s alternative"
            ),
            (
                "The defender demonstrates proper evidence updating with "
                "appropriate prior-to-posterior reasoning"
            ),
            (
                "The critique neglects the base rate in a way that distorts the "
                "probability assessment"
            ),
            (
                "The defender shows the challenged claim is falsifiable and "
                "specifies what evidence would refute it"
            ),
            (
                "Occam defense — the defender shows the hypothesis is the simplest "
                "one consistent with all available evidence"
            ),
            (
                "The critique demands certainty where probabilistic reasoning is "
                "appropriate — the defender shows the conclusion follows with "
                "adequate posterior probability"
            ),
            (
                "The defender shows the critique selectively weighs evidence, and "
                "that symmetric updating supports the defended position"
            ),
            # Practical reasoning defense (from INFORMAL_CRITICAL)
            (
                "The alleged fallacy is misidentified — the reasoning pattern is "
                "actually legitimate in the specific context (e.g., a legitimate "
                "appeal to expert authority is not ad verecundiam)"
            ),
            (
                "The defender provides sufficient additional evidence to meet the "
                "challenged evidentiary standard"
            ),
            (
                "The defender demonstrates the relevance of the challenged premise "
                "to the conclusion with an explicit connection"
            ),
            (
                "The defender shows the generalization is adequately supported by "
                "the scope, quality, and representativeness of the evidence"
            ),
            (
                "Principle of charity — the defender clarifies that the strongest "
                "reasonable interpretation of their argument avoids the identified "
                "weakness"
            ),
            (
                "The defender demonstrates the analogy holds on the relevant "
                "structural dimensions despite surface-level differences"
            ),
            # General defense (merged cross-system + hybrid-original)
            (
                "The critique itself contains a logical flaw the defender "
                "exposes — formal fallacy, informal fallacy, unsupported premise, "
                "or false dichotomy in the challenge"
            ),
            "Direct evidence refutes the challenger’s factual premise",
            (
                "The critique depends on a hidden false assumption the defender "
                "exposes"
            ),
            (
                "The defender provides a complete inference chain resolving the "
                "specific concern"
            ),
            (
                "Successful reframing that avoids the critique without losing "
                "substance or weakening the position"
            ),
        ],
        "unresolved": [
            # Deductive deadlock (from FORMAL_DEDUCTIVE)
            (
                "Both sides use formally valid deductions from incompatible axioms "
                "or definitional starting points"
            ),
            (
                "The dispute reduces to a disagreement about the truth of a "
                "foundational premise that cannot be derived from common ground"
            ),
            # Evidential deadlock (from BAYESIAN)
            (
                "Both sides present coherent arguments grounded in different but "
                "defensible prior probability assignments"
            ),
            (
                "The disagreement hinges on an empirical question that neither "
                "side can resolve with the available evidence"
            ),
            (
                "The evidence is genuinely ambiguous — it has comparable "
                "likelihood under both competing hypotheses"
            ),
            # Practical reasoning deadlock (from INFORMAL_CRITICAL)
            (
                "Both sides make arguments of roughly equal practical reasoning "
                "quality with different but defensible evidentiary standards"
            ),
            (
                "The dispute hinges on how to weigh competing but individually "
                "legitimate considerations, and no logical principle settles the "
                "priority"
            ),
            # General (merged cross-system)
            (
                "Both arguments contain flaws of comparable severity with no "
                "clear net advantage to either side"
            ),
        ],
    },
}

FORMAL_DEDUCTIVE = {
    "label": "Formal Deductive",
    "description": (
        "Strict formal deductive logic: only valid syllogisms and formally "
        "sound inferences accepted. Reject inductive and abductive reasoning "
        "entirely. An argument is valid only if its conclusion follows "
        "necessarily from its premises."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Formal invalidity — the conclusion does not follow necessarily "
                "from the premises by any valid rule of inference (modus ponens, "
                "modus tollens, hypothetical syllogism, etc.)"
            ),
            (
                "Undistributed middle term — the middle term in a categorical "
                "syllogism is not distributed in at least one premise"
            ),
            (
                "Illicit major or minor — a term is distributed in the conclusion "
                "but not in the premise where it appears"
            ),
            (
                "Affirming the consequent — inferring P from P→Q and Q"
            ),
            (
                "Denying the antecedent — inferring ¬Q from P→Q and ¬P"
            ),
            (
                "Reliance on inductive or probabilistic reasoning where deductive "
                "proof is required — the argument uses “likely,” “probably,” or "
                "statistical evidence instead of deductive entailment"
            ),
            (
                "Equivocation — a key term shifts meaning between premises, "
                "destroying the formal structure of the argument"
            ),
            (
                "Hidden premise — the argument requires an unstated premise to "
                "be deductively valid, and that premise is not self-evident"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the challenged inference is a valid "
                "syllogistic or propositional form (e.g., by exhibiting the rule "
                "of inference used)"
            ),
            "The critique misidentifies the logical form of the argument",
            "The critique itself commits a formal fallacy",
            (
                "The defender supplies a missing premise and demonstrates it is "
                "true or self-evident, restoring deductive validity"
            ),
            (
                "The critique attacks an inductive auxiliary element rather than "
                "the core deductive structure"
            ),
            (
                "A formal proof, truth-table, or derivation demonstrates the "
                "inference is valid"
            ),
        ],
        "unresolved": [
            (
                "Both sides use formally valid deductions from incompatible "
                "axioms or definitional starting points"
            ),
            (
                "The dispute reduces to a disagreement about the truth of a "
                "foundational premise that cannot be derived deductively from "
                "common ground"
            ),
            "Both arguments contain formal fallacies of comparable severity",
        ],
    },
}

BAYESIAN = {
    "label": "Bayesian Probabilistic",
    "description": (
        "Bayesian probabilistic reasoning: evaluate arguments by how well "
        "they update on evidence, respect prior probabilities, and apply "
        "parsimony. Conclusions should follow from the evidence with "
        "appropriate posterior probability, not from deductive necessity or "
        "rhetorical force."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Prior probability neglected — an extraordinary claim is adopted "
                "without proportionally strong evidence to overcome a low prior"
            ),
            (
                "Base rate neglect — the argument ignores the base rate of the "
                "phenomenon when evaluating the significance of evidence (e.g., "
                "prosecutor’s fallacy, treating a rare positive test result as "
                "near-certain)"
            ),
            (
                "Unfalsifiable claim treated as empirical — a non-testable "
                "assertion is used as though it were evidence for a descriptive "
                "conclusion"
            ),
            (
                "Occam’s Razor violation — an unnecessarily complex hypothesis "
                "is maintained when a simpler one accounts for the same evidence "
                "equally well"
            ),
            (
                "Evidence misuse — the evidence does not support the claim as "
                "stated, or a correlation is asserted as causation without "
                "controlling for confounders"
            ),
            (
                "Asymmetric updating — confirming evidence is accepted while "
                "equally valid disconfirming evidence is dismissed or downweighted "
                "without justification"
            ),
            (
                "Likelihood neglect — the argument fails to consider how probable "
                "the observed evidence would be under competing hypotheses"
            ),
        ],
        "rebuttal_valid": [
            (
                "Bayesian counter — the defender shows the evidence has a higher "
                "likelihood under the defended hypothesis than under the "
                "challenger’s alternative"
            ),
            (
                "The defender demonstrates proper evidence updating with "
                "appropriate prior-to-posterior reasoning"
            ),
            (
                "The critique neglects the base rate in a way that distorts the "
                "probability assessment"
            ),
            (
                "The defender shows the challenged claim is falsifiable and "
                "specifies what evidence would refute it"
            ),
            (
                "The defender demonstrates the hypothesis is the simplest one "
                "consistent with all available evidence"
            ),
            (
                "The critique demands certainty where probabilistic reasoning is "
                "appropriate — the defender shows the conclusion follows with "
                "adequate posterior probability"
            ),
            (
                "The defender shows the critique selectively weighs evidence, and "
                "that symmetric updating supports the defended position"
            ),
        ],
        "unresolved": [
            (
                "Both sides present coherent arguments grounded in different but "
                "defensible prior probability assignments"
            ),
            (
                "The disagreement hinges on an empirical question that neither "
                "side can resolve with the available evidence"
            ),
            (
                "The evidence is genuinely ambiguous — it has comparable "
                "likelihood under both competing hypotheses"
            ),
        ],
    },
}

DIALECTICAL = {
    "label": "Dialectical (Hegelian)",
    "description": (
        "Hegelian dialectical logic: thesis-antithesis-synthesis. "
        "Contradictions are productive and drive toward higher understanding. "
        "Evaluate whether opposing positions can be sublated into a more "
        "comprehensive synthesis rather than simply declaring a winner."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Arrested dialectic — the position presents itself as final or "
                "absolute rather than as one moment in an ongoing dialectical "
                "process"
            ),
            (
                "Unacknowledged internal contradiction — the position fails to "
                "account for tensions or contradictions within its own framework"
            ),
            (
                "Bare reassertion — the rebuttal merely restates the thesis "
                "without engaging the antithesis on its own terms"
            ),
            (
                "Suppressed negation — the position ignores or dismisses a "
                "legitimate opposing moment that should be incorporated through "
                "sublation"
            ),
            (
                "Pseudo-synthesis — the position claims to have achieved "
                "synthesis but actually just restates the original thesis while "
                "ignoring the antithesis"
            ),
            (
                "Destructive negation only — the rebuttal treats contradiction "
                "as purely destructive rather than as a potentially productive "
                "moment that could drive toward higher understanding"
            ),
        ],
        "rebuttal_valid": [
            (
                "Genuine sublation (Aufhebung) — the defender preserves valid "
                "elements of both thesis and antithesis in a higher synthesis "
                "that transcends the original opposition"
            ),
            (
                "The critique is a one-sided negation that doesn’t advance the "
                "dialectic beyond bare opposition"
            ),
            (
                "The defender demonstrates their position already incorporates "
                "and accounts for the tension the challenger raises"
            ),
            (
                "The defender’s position represents a more comprehensive moment "
                "that subsumes the challenger’s concerns as a partial truth "
                "within a richer whole"
            ),
            (
                "The defender shows the alleged contradiction is a productive "
                "tension already accounted for in their synthesis"
            ),
        ],
        "unresolved": [
            (
                "Both positions represent equally valid but partial moments that "
                "require a higher synthesis neither side has achieved"
            ),
            (
                "The contradiction between thesis and antithesis is genuinely "
                "productive and premature resolution would impoverish the "
                "discourse"
            ),
            (
                "Both sides fail to move beyond mere negation toward "
                "constructive synthesis"
            ),
        ],
    },
}

INFORMAL_CRITICAL = {
    "label": "Informal / Critical Thinking",
    "description": (
        "Informal logic and critical thinking: evaluate argument strength "
        "via fallacy identification, relevance, and sufficiency of evidence. "
        "Accept inductive and abductive reasoning when well-supported. Focus "
        "on practical reasoning quality over formal validity."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Named informal fallacy — the argument commits a recognized "
                "informal fallacy (ad hominem, strawman, appeal to authority, red "
                "herring, false dilemma, slippery slope, tu quoque, appeal to "
                "emotion, etc.) that materially weakens the conclusion"
            ),
            (
                "Insufficient evidence — claims are not adequately supported by "
                "the evidence provided, given the strength of the conclusion drawn"
            ),
            (
                "Irrelevance — a premise or piece of evidence does not actually "
                "bear on the conclusion it is meant to support"
            ),
            (
                "Hasty generalization — the conclusion is drawn from too few "
                "cases, unrepresentative samples, or anecdotal evidence"
            ),
            (
                "False analogy — a comparison relies on superficial similarity "
                "while ignoring relevant differences between the cases"
            ),
            (
                "Burden of proof violation — a claim is asserted without meeting "
                "the appropriate evidentiary standard for its type (empirical, "
                "normative, existential)"
            ),
            (
                "Ambiguity or vagueness — key terms are too imprecise to "
                "evaluate, making the argument’s truth conditions unclear"
            ),
            (
                "Suppressed evidence — relevant counterevidence is ignored, "
                "minimized, or selectively omitted to strengthen the appearance "
                "of the argument"
            ),
        ],
        "rebuttal_valid": [
            (
                "The alleged fallacy is misidentified — the reasoning pattern is "
                "actually legitimate in the specific context (e.g., a legitimate "
                "appeal to expert authority is not an ad verecundiam)"
            ),
            (
                "The defender provides sufficient additional evidence to meet the "
                "challenged evidentiary standard"
            ),
            "The critique itself commits a significant informal fallacy",
            (
                "The defender demonstrates the relevance of the challenged "
                "premise to the conclusion with an explicit connection"
            ),
            (
                "The defender shows the generalization is adequately supported by "
                "the scope, quality, and representativeness of the evidence"
            ),
            (
                "Principle of charity — the defender clarifies that the strongest "
                "reasonable interpretation of their argument avoids the identified "
                "weakness"
            ),
            (
                "The defender demonstrates the analogy holds on the relevant "
                "structural dimensions despite surface-level differences"
            ),
        ],
        "unresolved": [
            (
                "Both sides make arguments of roughly equal practical reasoning "
                "quality with different but defensible evidentiary standards"
            ),
            (
                "The dispute hinges on how to weigh competing but individually "
                "legitimate considerations, and no logical principle settles the "
                "priority"
            ),
            (
                "Both sides have identified genuine weaknesses in the other’s "
                "reasoning, with no clear net advantage to either"
            ),
        ],
    },
}

FUZZY_MULTIVALUED = {
    "label": "Fuzzy / Multi-valued",
    "description": (
        "Fuzzy and multi-valued logic: truth admits degrees between 0 and 1; "
        "partial truth is acceptable. Avoid binary true/false judgments. "
        "Evaluate the degree to which premises support conclusions rather "
        "than demanding all-or-nothing validity."
    ),
    "criteria": {
        "critique_valid": [
            (
                "False binarism — the position forces a true/false dichotomy "
                "where the subject matter admits degrees of truth (e.g., "
                "“either X is safe or it isn’t” when safety is a spectrum)"
            ),
            (
                "Threshold abuse — a gradual spectrum is treated as having a "
                "sharp cutoff without justification for where the boundary is "
                "drawn"
            ),
            (
                "Degree of support overstated — the premises support the "
                "conclusion to a lesser degree than claimed; the argument treats "
                "partial support as if it were decisive"
            ),
            (
                "Composition fallacy — partial truth of individual components is "
                "aggregated into full truth of the whole without accounting for "
                "how partial truths combine"
            ),
            (
                "Failure to propagate uncertainty — uncertainty in the premises "
                "is not reflected in the confidence level of the conclusion"
            ),
            (
                "Sorites vulnerability — the position relies on a vague "
                "predicate (heap, tall, rich) and the argument’s force depends on "
                "ignoring boundary cases where the predicate’s application is "
                "genuinely indeterminate"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender demonstrates the appropriate truth degree was "
                "assigned and the critique demands a level of precision the "
                "subject matter does not support"
            ),
            (
                "The defender’s position already accounts for degrees of truth "
                "and the critique imposes a false binary that the multi-valued "
                "approach correctly avoids"
            ),
            (
                "The defender provides a principled justification for a threshold "
                "or boundary in a graded concept (e.g., a regulatory standard, a "
                "statistical significance level)"
            ),
            (
                "The critique itself imposes binary thinking where the defender’s "
                "multi-valued approach is more appropriate to the domain"
            ),
            (
                "The defender shows the uncertainty propagation is calibrated "
                "appropriately given the available evidence and the nature of the "
                "claims"
            ),
            (
                "The vagueness is a genuine feature of the domain (not a flaw of "
                "the argument), and the defender handles it by acknowledging the "
                "indeterminacy rather than papering over it"
            ),
        ],
        "unresolved": [
            (
                "Both sides assign different but defensible degrees of truth to "
                "the same proposition, and the evidence does not clearly favor "
                "one calibration over the other"
            ),
            (
                "The dispute turns on where to place a threshold in a genuinely "
                "continuous spectrum, and no principled method settles the "
                "boundary"
            ),
            (
                "Both positions appropriately acknowledge degrees of truth but "
                "reach different conclusions because they weight partial truths "
                "differently in their aggregation"
            ),
        ],
    },
}

PARACONSISTENT = {
    "label": "Paraconsistent",
    "description": (
        "Paraconsistent logic: tolerate local contradictions without global "
        "explosion (ex contradictione quodlibet does not apply). Evaluate "
        "whether contradictions are contained and productive or whether they "
        "undermine the overall argument structure."
    ),
    "criteria": {
        "critique_valid": [
            (
                "Global explosion — a local contradiction has been allowed to "
                "propagate beyond its scope, making the entire argument trivially "
                "derivable (the very thing paraconsistent logic is meant to "
                "prevent)"
            ),
            (
                "Uncontained contradiction — a contradiction genuinely undermines "
                "the argument’s core structure rather than being locally bounded "
                "to a non-essential component"
            ),
            (
                "Ad hoc inconsistency — a contradiction is tolerated not because "
                "the domain requires it but simply to avoid conceding a point the "
                "defender would otherwise have to give up"
            ),
            (
                "Selective rigor — classical logic is applied where it supports "
                "the position and paraconsistent tolerance is invoked where "
                "classical logic would defeat it, without a principled "
                "justification for the switch"
            ),
            (
                "False resolution — a contradiction is presented as resolved or "
                "dissolved when it actually remains active and unaddressed within "
                "the argument"
            ),
        ],
        "rebuttal_valid": [
            (
                "The defender shows the contradiction is genuinely local — it "
                "does not propagate to affect any claims outside its immediate "
                "scope"
            ),
            (
                "The contradiction is productive — it reveals a genuine tension "
                "in the domain that forcing a binary resolution would distort or "
                "oversimplify"
            ),
            (
                "The critique assumes the principle of explosion (from a "
                "contradiction, anything follows), which does not hold under "
                "paraconsistent logic"
            ),
            (
                "The apparent contradiction dissolves when the relevant terms are "
                "properly distinguished (different senses, contexts, levels of "
                "description, or time indices)"
            ),
            (
                "The defender demonstrates principled application of "
                "paraconsistent reasoning with explicit scope boundaries for the "
                "tolerated inconsistency"
            ),
        ],
        "unresolved": [
            (
                "Both sides contain locally contained contradictions, and neither "
                "has achieved a clearly superior containment or resolution "
                "strategy"
            ),
            (
                "It is genuinely unclear whether the contradiction in question is "
                "productive (revealing real domain complexity) or destructive "
                "(undermining argument integrity)"
            ),
            (
                "The disagreement turns on whether to tolerate or resolve a "
                "particular inconsistency, with defensible philosophical "
                "arguments for both approaches"
            ),
        ],
    },
}


# === Lookup Structures ===

LOGIC_SYSTEMS = {
    "CLASSICAL_INFORMAL_BAYESIAN": CLASSICAL_INFORMAL_BAYESIAN,
    "FORMAL_DEDUCTIVE": FORMAL_DEDUCTIVE,
    "BAYESIAN": BAYESIAN,
    "INFORMAL_CRITICAL": INFORMAL_CRITICAL,
    "DIALECTICAL": DIALECTICAL,
    "FUZZY_MULTIVALUED": FUZZY_MULTIVALUED,
    "PARACONSISTENT": PARACONSISTENT,
}


# === Lookup Functions ===

def get_logic_system(key: str) -> dict:
    """
    Look up a logic system by key.

    Args:
        key: Logic system identifier (e.g., "CLASSICAL_INFORMAL_BAYESIAN").
             Case-insensitive.

    Returns:
        The full logic system dict with label, description, and criteria.

    Raises:
        KeyError: If the key does not match any known logic system.
    """
    return LOGIC_SYSTEMS[key.upper()]


def get_logic_system_description(key: str) -> str:
    """
    Look up a logic system's description string by key.

    Convenience function for callers that only need the description text
    (e.g., the per-pair prompt in Adjudicator.run()).

    Args:
        key: Logic system identifier. Case-insensitive.

    Returns:
        The description string for the logic system.

    Raises:
        KeyError: If the key does not match any known logic system.
    """
    return LOGIC_SYSTEMS[key.upper()]["description"]


def get_logic_system_label(key: str) -> str:
    """
    Look up a logic system's human-readable label by key.

    Args:
        key: Logic system identifier. Case-insensitive.

    Returns:
        The human-readable label string.

    Raises:
        KeyError: If the key does not match any known logic system.
    """
    return LOGIC_SYSTEMS[key.upper()]["label"]
