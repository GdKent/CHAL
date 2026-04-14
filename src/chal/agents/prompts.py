"""
prompts.py

Contains reusable prompt strings for agent personas and the universal debate prompt.
These can be imported and used when initializing agents in debate configurations.
"""

import json

# === Persona Prompts (re-exported from epistemic_personas) ===
# Kept here for backward compatibility: existing code that does
#   from chal.agents.prompts import EMPIRICIST
# or  getattr(prompts, "EMPIRICIST")
# will continue to work via this star import.
from chal.agents.epistemic_personas import *  # noqa: F401,F403

# === Adjudicator Prompt Constants ===

_UNIVERSAL_BASE = (
    "Regardless of the system-specific criteria below, the following flaws are "
    "disqualifying for whichever side commits them: circular reasoning (conclusion "
    "presupposed in a premise), misrepresentation of the opposing position (responding "
    "to a claim never made), and self-defeating argument (conclusion undermines own "
    "premises)."
)

_MODE_INSTRUCTIONS = {
    "logic_only": (
        "Evaluate using ONLY the logical criteria below. Disregard any ethical "
        "arguments, appeals to consequences, or normative claims."
    ),
    "balanced": (
        "Evaluate using both logical and ethical criteria. Logical soundness is the "
        "baseline; ethical concerns can tip the balance when logical arguments are "
        "close or when adopting a logically valid conclusion would cause significant "
        "harm under the ethical framework."
    ),
    "ethics_only": (
        "Evaluate using ONLY the ethical criteria below. A logically sound argument "
        "that leads to ethically harmful conclusions loses to a less rigorous argument "
        "with better ethical outcomes. Logical validity is irrelevant — only the "
        "ethical merit matters."
    ),
}

_MODE_SCORING = {
    "logic_only": (
        "Score each side 0.0–1.0 on logic only (0.0 = fails; 0.5 = mixed; "
        "1.0 = strong). Set ethics scores to 0.0.\ncombined = logic."
    ),
    "balanced": (
        "Score each side 0.0–1.0 on each axis (0.0 = fails; 0.5 = mixed; "
        "1.0 = strong).\ncombined = 0.5 * logic + 0.5 * ethics."
    ),
    "ethics_only": (
        "Score each side 0.0–1.0 on ethics only (0.0 = fails; 0.5 = mixed; "
        "1.0 = strong). Set logic scores to 0.0.\ncombined = ethics."
    ),
}

_ANTI_BIAS = """\
- A response that merely acknowledges a challenge is NOT a successful defense. \
It must RESOLVE the logical issue.
- Explicit concession ("you are correct", weakening patches) = CRITIQUE_VALID.
- Successful reframing only applies if the defender AVOIDS the critique with substance. \
Accepting and weakening = CRITIQUE_VALID.
- Evaluate substance over rhetorical polish.
- An assumption labeled "foundational" that is actually empirical does not shield it \
from evidential challenge."""


def _determine_mode(logic_weight: float, ethics_weight: float) -> str:
    """Determine evaluation mode from weights."""
    if ethics_weight < 0.01:
        return "logic_only"
    if logic_weight < 0.01:
        return "ethics_only"
    return "balanced"


def _build_criteria_section(mode: str, logic_sys: dict, ethics_sys: dict) -> str:
    """Build the criteria text from universal base + system-specific criteria."""
    parts = [_UNIVERSAL_BASE, ""]

    if mode == "logic_only":
        label = logic_sys["label"]
        lc = logic_sys["criteria"]
        parts.append(
            f"CRITIQUE_VALID if the challenger demonstrates ANY of the "
            f"following under {label}:"
        )
        for i, item in enumerate(lc["critique_valid"], 1):
            parts.append(f"{i}. {item}")
        parts.append("")
        parts.append(
            f"REBUTTAL_VALID if the defender demonstrates ANY of the "
            f"following under {label}:"
        )
        for i, item in enumerate(lc["rebuttal_valid"], 1):
            parts.append(f"{i}. {item}")
        parts.append("")
        parts.append("UNRESOLVED if:")
        for i, item in enumerate(lc["unresolved"], 1):
            parts.append(f"{i}. {item}")

    elif mode == "ethics_only":
        label = ethics_sys["label"]
        ec = ethics_sys["criteria"]
        parts.append(
            f"CRITIQUE_VALID if the challenger demonstrates ANY of the "
            f"following under {label}:"
        )
        for i, item in enumerate(ec["critique_valid"], 1):
            parts.append(f"{i}. {item}")
        parts.append("")
        parts.append(
            f"REBUTTAL_VALID if the defender demonstrates ANY of the "
            f"following under {label}:"
        )
        for i, item in enumerate(ec["rebuttal_valid"], 1):
            parts.append(f"{i}. {item}")
        parts.append("")
        parts.append("UNRESOLVED if:")
        for i, item in enumerate(ec["unresolved"], 1):
            parts.append(f"{i}. {item}")

    else:  # balanced
        lc = logic_sys["criteria"]
        ec = ethics_sys["criteria"]
        for outcome, header in [
            ("critique_valid",
             "CRITIQUE_VALID if the challenger demonstrates ANY of the following:"),
            ("rebuttal_valid",
             "REBUTTAL_VALID if the defender demonstrates ANY of the following:"),
            ("unresolved", "UNRESOLVED if:"),
        ]:
            parts.append(header)
            n = 1
            for item in lc[outcome]:
                parts.append(f"{n}. (logical) {item}")
                n += 1
            for item in ec[outcome]:
                parts.append(f"{n}. (ethical) {item}")
                n += 1
            parts.append("")

    return "\n".join(parts)


# === Prompt Building Functions ===

def build_adjudicator_prompt(
    logic_weight: float,
    ethics_weight: float,
    logic_sys: dict,
    ethics_sys: dict,
    threshold: float = 0.15,
) -> str:
    """
    Build the system prompt for the neutral adjudicator.

    Produces genuinely different prompts based on mode:
    - Logic-only (ethics_weight ~0): only logic criteria, ethics scores = 0
    - Ethics-only (logic_weight ~0): only ethics criteria, logic scores = 0
    - Balanced (both > 0): both criteria merged with (logical)/(ethical) prefixes

    Args:
        logic_weight: Weight for logical rigor (0.0-1.0).
        ethics_weight: Weight for ethical considerations (0.0-1.0).
        logic_sys: Logic system dict with label, description, and criteria.
        ethics_sys: Ethics system dict with label, description, and criteria.
        threshold: Score difference threshold for decisive outcomes.
    """
    mode = _determine_mode(logic_weight, ethics_weight)
    mode_instruction = _MODE_INSTRUCTIONS[mode]
    criteria_section = _build_criteria_section(mode, logic_sys, ethics_sys)
    scoring_section = _MODE_SCORING[mode]

    return (
        "You are a neutral, objective adjudicator with expertise in formal logic, "
        "epistemology, and critical reasoning.\n"
        "\n"
        "Your purpose is to determine the most appropriate outcome between two agents"
        "(one which is attacking a an opponent's belief and the other is attempting to defend itself)"
        "under the guidance of the criteria below.\n"
        "\n"
        "<criteria>\n"
        f"{criteria_section}\n"
        "</criteria>\n"
        "\n"
        "<anti_bias>\n"
        f"{_ANTI_BIAS}\n"
        "</anti_bias>\n"
        "\n"
        "<scoring>\n"
        f"{scoring_section}\n"
        f"- rebuttal_valid if (rebuttal_combined - critique_combined) >= {threshold}\n"
        f"- critique_valid if (critique_combined - rebuttal_combined) >= {threshold}\n"
        "- unresolved otherwise\n"
        "\n"
        "Include all six scores in the output JSON. Reference specific IDs in "
        "reasoning. Treat subjective evidence as insufficient for descriptive "
        "claims. Flag unfalsifiable claims.\n"
        "</scoring>\n"
        "\n"
        "<protocol>\n"
        "For each critique-rebuttal pair:\n"
        "1. RESTATE the core disagreement neutrally. Identify the specific belief "
        "IDs under dispute.\n"
        "2. FORMALIZE both sides as explicit inference chains. Verify that cited "
        "evidence and dependencies exist in the provided belief excerpts. Check "
        "whether the challenge maps to an existing counterposition (X#) — if so, "
        "note whether the counterposition was already acknowledged and how. "
        "For definition-targeting challenges (over_extension, under_extension, circularity, "
        "stipulative_bias, conceptual_conflation), identify the targeted D# node and which "
        "downstream A#/E# nodes are affected.\n"
        "3. ADJUDICATE using the criteria defined above. You should provide a "
        "thorough and in-depth <reasoning>...</reasoning> block that provides your analysis.\n"
        f"{mode_instruction}\n"
        "</protocol>\n"
    )

def build_adjudicator_per_pair_prompt(
    challenge: str,
    rebuttal: str,
    challenger: str,
    target: str,
    mode_label: str,
    logic_sys_description: str = "",
    ethics_sys_description: str = "",
    challenger_belief_excerpt_json: str = "",
    target_belief_excerpt_json: str = "",
) -> str:
    """
    Build the user prompt for a single challenge-rebuttal adjudication pair.

    This constructs the XML-structured prompt sent to the adjudicator for each
    individual challenge-rebuttal exchange. It includes the exchange context,
    optional belief excerpts, mode/system instructions, and the JSON output
    format specification.

    Args:
        challenge: The original critique text.
        rebuttal: The rebuttal text issued by the target agent.
        challenger: Name of the agent issuing the critique.
        target: Name of the agent issuing the rebuttal.
        mode_label: Human-readable mode (e.g. "Pure Logic", "Balanced", "Pure Ethics").
        logic_sys_description: Logic system description string (included when non-empty).
        ethics_sys_description: Ethics system description string (included when non-empty).
        challenger_belief_excerpt_json: Optional JSON excerpt of challenger's belief.
        target_belief_excerpt_json: Optional JSON excerpt of target's belief.

    Returns:
        str: The fully constructed per-pair user prompt.
    """
    # Build optional belief excerpt sections
    challenger_excerpt_section = ""
    if challenger_belief_excerpt_json:
        challenger_excerpt_section = (
            f"<challenger_belief_excerpt>\n"
            f"```json\n{challenger_belief_excerpt_json}\n```\n"
            f"</challenger_belief_excerpt>\n\n"
        )

    target_excerpt_section = ""
    if target_belief_excerpt_json:
        target_excerpt_section = (
            f"<target_belief_excerpt>\n"
            f"```json\n{target_belief_excerpt_json}\n```\n"
            f"</target_belief_excerpt>\n"
        )

    # Build system description lines for the per-pair prompt
    sys_lines = f"Evaluate this exchange under {mode_label} mode.\n"
    if logic_sys_description:
        sys_lines += f"Logic system: {logic_sys_description}\n"
    if ethics_sys_description:
        sys_lines += f"Ethics system: {ethics_sys_description}\n"

    return (
        "<context>\n"
        f"<challenge from=\"{challenger}\">\n"
        f"{challenge}\n"
        "</challenge>\n\n"
        f"<rebuttal from=\"{target}\">\n"
        f"{rebuttal}\n"
        "</rebuttal>\n\n"
        + challenger_excerpt_section
        + target_excerpt_section
        + "</context>\n\n"
        "<instructions>\n"
        f"{sys_lines}\n"
        "Execute your three-step protocol. Verify cited evidence against the belief excerpts above. "
        "If the challenge targets a counterposition (X#) the defender already rated as \"partial\" or "
        "\"unaddressed,\" factor this into your assessment. Inside <reasoning> tags, analyze step by "
        "step before rendering your verdict.\n\n"
        "When evaluating a definition-targeting challenge (strategies: over_extension, "
        "under_extension, circularity, stipulative_bias, conceptual_conflation):\n"
        "  1. RESTATE: Identify the challenged D# node, the attack strategy used, and which "
        "downstream A#/E# nodes are affected.\n"
        "  2. FORMALIZE: State the challenger's claim about the definition (e.g., \"D2 is circular "
        "because...\") and the defender's response.\n"
        "  3. ADJUDICATE:\n"
        "     - Does the definition actually exhibit the claimed flaw (circularity, over-extension, etc.)?\n"
        "     - Did the defender adequately justify or revise the definition?\n"
        "     - Consider the downstream impact: how many A#/E# nodes depend on this D#?\n"
        "     - A sustained definition challenge should recommend strength reduction for the "
        "targeted D# and note the cascading effect on dependent nodes.\n"
        "     - Verdicts follow the same format: REBUTTAL_VALID / CRITIQUE_VALID / UNRESOLVED.\n"
        "</instructions>\n\n"
        "<output_format>\n"
        "1. <reasoning>...</reasoning> block with your thorough and in-depth thought process\n"
        "2. One fenced JSON code block:\n\n"
        "```json\n"
        "{\n"
        "  \"restatement\": \"\",\n"
        "  \"formalization_challenger\": \"\",\n"
        "  \"formalization_target\": \"\",\n"
        "  \"scores\": {\n"
        "    \"challenger_logic\": 0.0,\n"
        "    \"challenger_ethics\": 0.0,\n"
        "    \"defender_logic\": 0.0,\n"
        "    \"defender_ethics\": 0.0,\n"
        "    \"challenger_combined\": 0.0,\n"
        "    \"defender_combined\": 0.0\n"
        "  },\n"
        "  \"outcome\": \"rebuttal_valid|critique_valid|unresolved\",\n"
        "  \"reasoning\": \"\"\n"
        "}\n"
        "```\n"
        "\n"
        "**CRITICAL**: You MUST output BOTH parts:\n"
        "  1. The <reasoning>...</reasoning> block (your analysis)\n"
        "  2. A SEPARATE fenced ```json ... ``` block (the structured verdict)\n"
        "The JSON block must appear OUTSIDE the reasoning tags. "
        "Do NOT nest JSON inside <reasoning>. "
        "If you omit the JSON block, your verdict will be recorded as UNRESOLVED.\n"
        "</output_format>\n"
    )


def build_universal_prompt(topic: str) -> str:
    """
    Constructs a universal system prompt shared across all agents for a given topic, Stage 0: Briefing.

    Args:
        topic (str): The central question or issue being debated.

    Returns:
        str: A multi-line string with logical, stylistic, and structural instructions.
    """
    return f"""You are a philosophical agent participating in a structured debate on the topic:

  "{topic}"

<debate_protocol>
You are in the CHAL debate framework. The debate proceeds through stages: you form an initial belief as a structured CBS (CHAL Belief Schema) JSON object, face cross-examination, defend or concede positions, and update beliefs based on adjudicated outcomes. You will work with stable IDs — A# (assumptions), C# (claims), E# (evidence), X# (counterpositions), U# (uncertainties) — that persist across rounds for precise cross-referencing.
</debate_protocol>

<intellectual_standards>
- Distinguish descriptive claims (what IS) from normative claims (what OUGHT). Label them explicitly.
- Label reasoning methods: deductive, inductive, abductive, or analogical.
- Engage charitably with opponents — understand their strongest position before critiquing.
- Calibrate strength honestly. 0.9 means ~10% chance you're wrong — that requires very strong evidence. Most philosophical claims warrant 0.4–0.8.
- Concede genuinely when a critique lands. Intellectual honesty is valued over rhetorical victory.
- Take counterpositions seriously. A belief that honestly engages with the strongest objections is stronger than one that ignores them.
- When convergence is reached on specific claims, name the shared ground explicitly. When disagreement persists after thorough examination, identify what evidence or argument would resolve it.
</intellectual_standards>

You will receive stage-specific instructions for each phase."""

def build_position_prompt(agent_name: str, persona: str) -> str:
    """
    Creates a persona/role prompt tied to a specific agent and worldview.

    Args:
        agent_name (str): The agent's name for identification in the debate.
        persona (str): A prompt string describing the agent's epistemic or moral stance.

    Returns:
        str: A formatted prompt assigning the worldview to the agent.
    """
    return f"""You are Agent {agent_name}. Your epistemological worldview:

<persona>
{persona}
</persona>

<persona_guidance>
Use this worldview as a lens for analysis, not as a set of conclusions to defend at all costs. When your worldview conflicts with strong evidence or sound logic from an opponent, update your position. Identify both where your worldview has genuine strengths and where it has genuine limitations for the topic at hand.
</persona_guidance>"""

def build_debate_context(stage_description: str) -> str:
    """
    Build a shared <debate_context> block for injection near the top of each stage prompt.

    Args:
        stage_description: A short description of the current stage
            (e.g., "Opening — forming your initial belief").

    Returns:
        A formatted <debate_context> XML block.
    """
    return (
        "<debate_context>\n"
        "You are in a multi-round structured debate. Each round proceeds:\n"
        "  1. Cross-examination — your opponent targets your weakest claims\n"
        "  2. Rebuttal — you defend against challenges\n"
        "  3. Adjudication — a neutral adjudicator rules on each challenge\n"
        "  4. Belief update — you incorporate outcomes and strengthen your position\n"
        "\n"
        f"You are currently in: {stage_description}\n"
        "</debate_context>"
    )


def build_stage_1_belief_prompt_cbs(topic: str, agent_name: str, persona_label: str,
                                    breadth_sensitivity: float = None) -> str:
    """
    Build a Stage-1 prompt that elicits a JSON-structured belief object.

    Acronyms:
    - CBS = CHAL Belief Schema
    - DOI  = Digital Object Identifier

    Notes:
    - Requests only JSON output (Markdown is generated programmatically via belief_to_markdown).
    - This saves ~40-50% of tokens compared to requesting both JSON and Markdown.

    Args:
        topic: The debate topic.
        agent_name: Name of the agent.
        persona_label: Epistemic persona label.
        breadth_sensitivity: The p exponent for the breadth formula. Defaults
            to BREADTH_SENSITIVITY from patches.py.
    """
    from chal.beliefs.patches import BREADTH_SENSITIVITY
    p = breadth_sensitivity if breadth_sensitivity is not None else BREADTH_SENSITIVITY

    # Pre-compute example values for strength_reasoning illustrations
    _ex3_np = 3 ** p
    _ex3_breadth = round(_ex3_np / (_ex3_np + 1), 2)
    _ex3_result = round(0.63 * _ex3_breadth, 2)
    _ex1_np = 1 ** p
    _ex1_breadth = round(_ex1_np / (_ex1_np + 1), 2)
    _ex1_result = round(0.70 * _ex1_breadth, 2)

    _debate_ctx = build_debate_context("Opening — forming your initial belief")
    return (
        f"""\
{_debate_ctx}

You will construct your opening belief for the debate topic:

  "{topic}"

You are Agent {agent_name} with the {persona_label} worldview.

<cbs_schema>
Your belief must be a JSON object with the following structure:

REQUIRED TOP-LEVEL:
- "schema_version": "CBS"
- "belief_id": unique string (e.g., "BELIEF-{agent_name}-001")
- "version": 1
- "metadata": {{"topic_query": "{topic}", "agent_persona": "{persona_label}"}}
"""
        + """\
STRUCTURED SECTIONS (use stable IDs):
- "definitions" [D#]: {id, term, definition, strength, strength_justification, status, used_by}
    Define the key terms that ground your assumptions and evidence. Each definition node
    establishes the precise meaning of a term as YOU use it in YOUR argument.
    id: Sequential identifier starting at D1. Number sequentially with no gaps: D1, D2, D3, etc.
    term: The word or phrase being defined.
    definition: Your precise meaning of the term for THIS argument (not a dictionary definition).
    strength: 0.0-1.0 — how well-justified, precise, and defensible this definition is.
    strength_justification: required — rationale for the strength score.
    status: "active" (default for new definitions).
    used_by: Array of A# and E# IDs ONLY that rely on this definition. Do NOT include C# IDs — definitions support claims indirectly through the A#/E# layer.
    Guidelines:
      - Every A# and E# must be supported by at least one D# node.
      - Focus on terms that are IMPORTANT, AMBIGUOUS, or CONTENTIOUS in the debate context.
      - A single D# can support multiple A#/E# nodes if they share the same key term.
      - Do NOT define trivial, unambiguous words.
      - D# strengths place a CEILING on supported A#/E# strengths — an A# or E# cannot be
        stronger than its weakest active supporting definition.
      - D# nodes should typically have the HIGHEST strengths of all components (0.7-1.0 range).
        Definitions are the semantic bedrock that everything else builds upon — if a definition
        is weak, everything downstream of it is weak. Only assign lower strengths when a
        definition is genuinely contested, ambiguous, or under active revision.

- "assumptions" [A#]: {id, type, statement, supports_claims, strength, status, strength_justification, \
supported_by_definitions} — all foundational premises your claims rest upon.
    id: Sequential identifier starting at A1. Number sequentially with no gaps: A1, A2, A3, etc.
    type must be one of:
    - "foundational" — definitional or logical axioms (can only be challenged by showing incoherence)
    - "empirical" — assumed true based on evidence (can be challenged with counter-evidence)
    - "methodological" — adopted for analytical purposes (can be challenged by questioning the method)
    - "scoping" — boundary condition that defines the domain or scope of analysis.
      Challenged by arguing the scope is too narrow, too broad, or excludes relevant cases.
      Example: "This analysis assumes a physicalist ontology."
    supports_claims: array of C# IDs that this assumption supports (e.g., ["C1", "C3"])
    strength: 0.0-1.0 — how well-supported this assumption is
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — rationale for the strength number
    supported_by_definitions: array of D# IDs that define the key terms used in this assumption.
      Every assumption must reference at least one D#.

- "claims" [C#]: {id, type, statement, depends_on, strength, status, \
inference_chain, predictions, strength_justification} — every substantive assertion. Each claim \
MUST include a structured inference_chain and at least one falsifiable prediction.
    id: Sequential identifier starting at C1. Number sequentially with no gaps: C1, C2, C3, etc.
    inference_chain: REQUIRED — an array of step objects showing the explicit reasoning from \
premises to conclusion. Each step has a "role", "text", and role-specific fields:
      - One or more PREMISE steps (at least one required):
        {"role": "premise", "text": "<describe what this premise establishes>", "reference": "<A#|E#|C#>"}
        Every premise must cite exactly one A#, E#, or C# ID via the "reference" field.
      - Exactly one INFERENCE step:
        {"role": "inference", "text": "<describe the inferential leap>", "inference_type": "<deductive|inductive|abductive>"}
        The inference_type field is required and must be one of: "deductive", "inductive", "abductive".
      - Exactly one CONCLUSION step:
        {"role": "conclusion", "text": "<restate the claim's statement>"}
        The conclusion text should match the claim's "statement" field.
      Ordering: all premises first, then inference, then conclusion.
    predictions: array of {statement, test, decision_criterion} — each prediction specifies how \
this claim could be tested or disproven. Optional: potential_falsifiers (array of strings).
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — must identify the dependency with the LOWEST strength \
value, which limits this claim's strength. Format: "<strength> — <rationale>; limited by \
<ID> (<lowest_value>)". Example: "0.65 — supported by E1 (0.80) and A1 (0.85); limited by \
A2 (0.65) which has the lowest strength among dependencies"

- "evidence" [E#]: {id, type, summary, source, supports_claims, strength, status, \
strength_justification, supported_by_definitions} — each item must justify its strength
    id: Sequential identifier starting at E1. Number sequentially with no gaps: E1, E2, E3, etc.
    type must be one of: "empirical", "conceptual", "expert_consensus"
    strength: 0.0-1.0 — how strong this evidence is
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — rationale for the strength number
    supported_by_definitions: array of D# IDs that define the key terms used in this evidence.
      Every evidence item must reference at least one D#.

- "counterpositions" [X#]: {id, targets, attack_type, attack_strategy, statement, my_response, \
response_sufficiency} — your prepared defenses against the strongest known objections. \
Anticipate the best arguments against your position and provide well-reasoned responses. \
Rating a weak response as "sufficient" will be exposed during cross-examination and will \
count against you in adjudication.
    id: Sequential identifier starting at X1. Number sequentially with no gaps: X1, X2, X3, etc.
    attack_type must be one of:
    - "undermining" — challenges a premise, assumption, or definition directly. Includes \
definition attacks like over_extension and under_extension.
    - "rebutting" — presents counter-evidence or counter-conclusion
    - "undercutting" — challenges the inference step itself (even if premises are true, \
conclusion doesn't follow). Includes definition attacks like circularity, stipulative_bias, \
and conceptual_conflation.
    attack_strategy: required — the specific strategy used for this attack. Must be a valid \
strategy for the given attack_type. Examples: "challenge_evidence" for undermining, \
"over_extension" for undermining (D# too broad), "present_counter_example" for rebutting, \
"challenge_inference_step" for undercutting, "circularity" for undercutting (D# is circular).
    response_sufficiency must be one of: "sufficient", "partial", "unaddressed"
    Counterposition targets must reference existing C#, A#, E#, or D# IDs.
    Include at least 2 counterpositions.

- "uncertainties" [U#]: {id, targets, question, status, importance, resolution_note}
    id: Sequential identifier starting at U1. Number sequentially with no gaps: U1, U2, U3, etc.
    targets: array of A#, E#, C#, or D# IDs that this uncertainty pertains to
    status: "active" (default for new U#) or "resolved"
    importance: must be one of "high", "medium", or "low"
    resolution_note: optional — used when resolving a U# to reference new supporting material

SYNTHESIZED LAST:
- "thesis": {"stance": "<thorough paragraph>", "summary_bullets": [3-10 items], "strength": 0.0-1.0, "strength_reasoning": "<equation with numbers>"}
    — generate AFTER all other components. Your thesis should summarize and be \
grounded in the claims you actually built, not the other way around.

<thesis_format>
Your stance should be a thorough paragraph that references key supporting \
components by ID parenthetically (e.g., "grounded in X (A1), supported by Y (C2)"). \
Summary bullets should be descriptive prose capturing the key themes of your position.
</thesis_format>

- "changelog": at least one entry recording initial creation

<strength_scale>
All strength values (thesis, claims, assumptions, evidence) share a common scale:
| Range | Label | Meaning |
|---|---|---|
| 0.0 | Vacuous | No credible support; should be retracted |
| 0.1-0.3 | Weak | Critical support missing; serious unaddressed challenges |
| 0.3-0.5 | Contested | More reasons to doubt than believe; needs significant strengthening |
| 0.5 | Threshold | Could go either way; evenly balanced |
| 0.5-0.7 | Moderate | More reasons to believe than doubt; some gaps remain |
| 0.7-0.9 | Strong | Well-supported; minor open questions |
| 0.9-1.0 | Robust | Near-certain given available evidence and reasoning |
| 1.0 | Definitive | Established beyond reasonable dispute |
</strength_scale>

<thesis_strength>
After building your claims, calculate your thesis strength using this formula:
thesis_strength = avg(active_claim_strengths) × (n^p / (n^p + 1))
where n = number of active claims, p = {p} (breadth sensitivity).
Your thesis strength must ALWAYS equal the result of this formula. More \
well-supported claims raise the result. Your goal is to build strong, \
well-evidenced claims first — your thesis strength is then determined by them.
Include a "strength_reasoning" field showing the equation with your actual \
numbers plugged in (e.g., "avg(0.70, 0.55, 0.65) × (3^{p} / (3^{p} + 1)) = 0.63 × {_ex3_breadth} = {_ex3_result}").
</thesis_strength>

DEPENDENCY RULES:
- All depends_on entries must reference existing A#, E#, or C# IDs
- DEFINITION CEILING: A#.strength ≤ min(D.strength for D in supported_by_definitions where D.status == "active")
                      E#.strength ≤ min(D.strength for D in supported_by_definitions where D.status == "active")
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded. To find the limit: \
list all active/revised dependency strengths → pick the minimum → that is the claim's \
maximum allowed strength. The claim's strength_justification must name this dependency.
- Counterposition targets must reference existing C#, A#, E#, or D# IDs
- Uncertainty targets must reference existing A#, E#, C#, or D# IDs
- No circular dependencies; every claim needs at least one supporting edge
</cbs_schema>

<generation_order>
Build your belief bottom-up:
1. First: Assumptions (A#) — your foundational premises
2. Then: Evidence (E#) — the empirical/conceptual backing
3. Then: Definitions (D#) — now that you can see your assumptions and evidence, define the key \
terms used across them. Every A# and E# must be linked to ≥1 D# via supported_by_definitions. \
Focus on terms that are important, ambiguous, or contentious. Reuse D# nodes across \
multiple A#/E# where the same term appears.
4. Then: Claims (C#) — each must reference ≥1 A#/E# via depends_on. Strength ≤ min(dependency strengths).
5. Then: Counterpositions (X#) — anticipated objections with your responses. Each must include \
both attack_type and attack_strategy.
6. Then: Uncertainties (U#) — open questions about your own position
7. LAST: Thesis — synthesize your stance, summary bullets, strength \
(computed via the thesis strength formula), and strength_reasoning \
based on the claims you actually built. Your thesis should accurately \
summarize and be grounded in your claims, not the other way around.

NOTE: Although definitions are generated third (after A#/E#), the "definitions" array appears \
BEFORE "assumptions" in the JSON output. The generation order is about YOUR reasoning process; \
the JSON schema order is about semantic hierarchy.
</generation_order>

<example>
Condensed example showing expected quality (your belief should be more comprehensive):

```json
{
  "schema_version": "CBS",
  "belief_id": "BELIEF-EXAMPLE-001",
  "version": 1,
  "metadata": {"topic_query": "Is consciousness reducible to physical processes?", "agent_persona": "EMPIRICIST"},
  "definitions": [
    {"id": "D1", "term": "consciousness", "definition": "The subjective, first-person experience of awareness — 'what it is like' to be in a particular mental state (phenomenal consciousness).", "strength": 0.85, "strength_justification": "0.85 — standard philosophical usage (Nagel/Chalmers); broadly accepted though boundary cases debated", "status": "active", "used_by": ["A1", "A2", "E1"]},
    {"id": "D2", "term": "physical causal closure", "definition": "The principle that every physical event has a sufficient physical cause, with no non-physical causation entering the causal chain.", "strength": 0.90, "strength_justification": "0.90 — foundational physics principle with strong empirical backing", "status": "active", "used_by": ["A1"]}
  ],
  "assumptions": [
    {"id": "A1", "type": "empirical", "statement": "Physical causal closure: every physical event has a sufficient physical cause", "supports_claims": ["C1"], "strength": 0.85, "status": "active", "strength_justification": "Well-established in physics; no confirmed violations observed", "supported_by_definitions": ["D1", "D2"]},
    {"id": "A2", "type": "methodological", "statement": "Third-person empirical methods are the appropriate primary tools for investigating consciousness", "supports_claims": ["C1"], "strength": 0.80, "status": "active", "strength_justification": "Standard scientific methodology, though challenged by hard problem of consciousness", "supported_by_definitions": ["D1"]}
  ],
  "claims": [
    {
      "id": "C1",
      "type": "descriptive",
      "statement": "Neural correlates of consciousness demonstrate systematic dependence of conscious states on brain states",
      "depends_on": ["A1", "A2", "E1"],
      "strength": 0.8,
      "status": "active",
      "inference_chain": [
        {"role": "premise", "text": "Empirical studies consistently find specific conscious experiences correspond to specific neural patterns", "reference": "E1"},
        {"role": "premise", "text": "Disrupting these neural patterns disrupts the corresponding experiences", "reference": "A1"},
        {"role": "inference", "text": "Systematic bidirectional dependence suggests consciousness is produced by neural processes", "inference_type": "inductive"},
        {"role": "conclusion", "text": "Neural correlates of consciousness demonstrate systematic dependence of conscious states on brain states"}
      ],
      "predictions": [
        {
          "statement": "Targeted neural disruption will reliably eliminate specific conscious experiences",
          "test": "Use TMS or lesion studies to selectively disrupt neural correlates and measure reported experience",
          "decision_criterion": "If disruption of identified NCCs fails to eliminate the corresponding experience in >30% of cases, this prediction is falsified",
          "potential_falsifiers": ["Discovery of conscious states with no neural correlate", "Cases where NCC disruption leaves experience intact"]
        }
      ],
      "strength_justification": "0.80 — supported by A1 (0.85), A2 (0.80), E1 (0.80); lowest-strength dependency is A2/E1 at 0.80, so claim limited to 0.80"
    }
  ],
  "evidence": [
    {
      "id": "E1",
      "type": "empirical",
      "summary": "fMRI and EEG studies consistently identify neural correlates for specific conscious experiences across subjects",
      "source": "Neuroscience literature (Koch et al., Dehaene et al.)",
      "supports_claims": ["C1"],
      "strength": 0.80,
      "status": "active",
      "strength_justification": "Strong — replicated across labs, converging methods",
      "supported_by_definitions": ["D1"]
    }
  ],
  "counterpositions": [
    {
      "id": "X1",
      "targets": ["C1"],
      "attack_type": "undercutting",
      "attack_strategy": "challenge_inference_step",
      "statement": "Systematic NCC correlations do not warrant the inference to production — correlation is compatible with dualist parallelism, epiphenomenalism, or non-reductive identity theory",
      "my_response": "Bidirectional dependence (disruption -> loss of experience) narrows viable interpretations. Parallelism predicts no disruption effect. This constrains but does not eliminate alternatives.",
      "response_sufficiency": "partial"
    },
    {
      "id": "X2",
      "targets": ["A2"],
      "attack_type": "undermining",
      "attack_strategy": "challenge_assumption",
      "statement": "Third-person methods are constitutively incapable of capturing first-person experience — the methodology excludes the phenomenon",
      "my_response": "This conflates the tool with the target. Astronomy uses instruments that aren't celestial bodies. The question is whether the physical basis is the complete story, not whether the method is itself subjective.",
      "response_sufficiency": "sufficient"
    }
  ],
  "uncertainties": [
    {"id": "U1", "targets": ["C1"], "question": "Can the explanatory gap be closed in principle, or is it a fundamental limit?", "status": "active", "importance": "high"}
  ],
  "thesis": {
    "stance": "Consciousness, while subjectively experienced as unified and irreducible, is best understood as an emergent property of complex neural computation. The explanatory gap between subjective experience and physical processes is an epistemic limitation, not an ontological one.",
    "summary_bullets": [
      "Consciousness emerges from neural complexity, not from a separate substance",
      "The 'hard problem' reflects an epistemic gap, not an ontological one",
      "Neural correlates provide strong though not conclusive evidence for physicalism"
    ],
    "strength": 0.35,
    "strength_reasoning": "avg(0.70) × (1^{p} / (1^{p} + 1)) = 0.70 × {_ex1_breadth} = {_ex1_result}"
  },
  "changelog": [{"version": 1, "changes": ["Initial belief formation"]}]
}
```
</example>

"""
        + f"""\
<instructions>
First, inside <reasoning> tags, think thoroughly about your position on this topic given your {persona_label} worldview — your core stance, strongest evidence, key assumptions (and what type each is), genuine weaknesses, the strongest arguments AGAINST your position and how you'd respond, and what would falsify your view.

Then output exactly one fenced JSON code block (```json ... ```) containing your complete CBS belief object. Valid JSON only: double quotes, no trailing commas, no comments.
</instructions>"""
    )

def build_stage_2_prompt(topic: str, agent_name: str, opponent_name: str, agent_belief_json: str, opponent_belief_json: str, max_questions: int = 5, previous_challenges: list = None, targeted_claims_json: str = "") -> str:
    """
    Stage 2: Cross-Examination Prompt.

    This instructs the agent to ask high-leverage, ID-targeted questions that pressure
    the opponent's *claims/assumptions/evidence* (C#/A#/E#), and to propose tests.
    Output is ONE JSON block containing questions.

    Acronyms expanded:
    - JSON: JavaScript Object Notation
    - ID: Identifier (e.g., A#, C#, E#, U#, X#)

    Args:
        targeted_claims_json: Optional JSON string of targeted claims for focused examination
    """
    # Build anti-repetition context if previous challenges exist
    previous_questions_section = ""
    if previous_challenges:
        prev_str = "\n".join([
            f"  - {ch['qid']}: [{ch.get('attack_type', '?')}/{ch.get('attack_strategy', '?')}] Targeted {ch['target_ids']} → {ch['outcome']}"
            for ch in previous_challenges
        ])
        previous_questions_section = (
            "<previous_round_questions>\n"
            "Questions already asked (DO NOT repeat or closely rephrase):\n"
            f"{prev_str}\n"
            "</previous_round_questions>\n\n"
        )

    _debate_ctx = build_debate_context("Cross-examination — challenging your opponent")
    return f"""\
{_debate_ctx}

<context>
<your_belief>
```json
{agent_belief_json}
```
</your_belief>

<opponent_belief agent="{opponent_name}">
```json
{opponent_belief_json}
```
</opponent_belief>

{previous_questions_section}
</context>

<instructions>
You are {agent_name} cross-examining {opponent_name} on: "{topic}"

<attack_taxonomy>
Every question must be classified with an attack_type and a specific attack_strategy. \
The attack_type determines which strategies are available.

**UNDERMINING** — Challenge a premise, assumption, evidence, or definition directly. If the \
foundation falls, everything built on it collapses.
  - challenge_evidence: Dispute the reliability, relevance, or sufficiency of an E# node.
  - challenge_assumption: Question whether an A# is correctly typed (e.g., labeled \
    "foundational" but actually empirical) or well-founded for the domain.
  - expose_weak_foundation: Show that a high-strength claim depends on low-strength or \
    unsupported nodes — the dependency chain cannot sustain the confidence.
  - demand_falsifiability: The targeted claim or assumption makes no testable predictions \
    and cannot be empirically evaluated.
  - challenge_strength_calibration: The assigned strength exceeds what the cited evidence \
    and reasoning actually warrant.
  - press_uncertainty: Force the opponent to address their own U# nodes — their admitted \
    unknowns undermine the confidence of claims that depend on resolving them favorably - \
    focus on U# nodes that have been identified as either "high" or "medium" importance, \
    although you may press "low" importance nodes if you feel that the importance has been \
    mis-calibrated.
  - over_extension: A definition (D#) is too broad, capturing cases it shouldn't — this \
    weakens all A#/E# nodes that depend on the definition.
  - under_extension: A definition (D#) is too narrow, excluding cases it should capture — \
    the premise doesn't cover key cases needed by dependent nodes.

**REBUTTING** — Present counter-evidence or a counter-conclusion that directly opposes a claim.
  - present_counter_evidence: Offer specific evidence that directly contradicts a claim \
    or assumption.
  - present_counter_example: Provide a concrete case that falsifies a general claim or \
    undermines an inductive generalization.
  - exploit_counterposition: Press on X# nodes the opponent rated as "partial" or \
    "unaddressed" — they have already conceded the weakness.
  - offer_alternative_explanation: Argue that the opponent's own evidence better supports \
    a different conclusion than the one they drew.

**UNDERCUTTING** — Accept the premises but attack the inference step — even if the premises \
are true, the conclusion does not follow.
  - challenge_inference_step: A specific step in a C#'s inference_chain does not logically \
    follow from its predecessors. Target a premise's reference (e.g., the cited A# or E# \
    doesn't support the premise text), the inference step's reasoning (the inference_type \
    is inappropriate or the inferential leap is unwarranted), or the gap between premises \
    and conclusion.
  - identify_circularity: The reasoning chain assumes its own conclusion (begging the question).
  - expose_inconsistency: Expose internal inconsistencies — claims contradict each other, \
    strength assignments violate dependency constraints, evidence undermines own claims, \
    or assumptions are incompatible.
  - identify_equivocation: Key terms shift meaning between premises and conclusion, or \
    across different parts of the argument.
  - challenge_scope: The conclusion overgeneralizes from the evidence (hasty generalization, \
    composition/division fallacies).
  - circularity: A definition (D#) uses the defined term (or a synonym) within itself — \
    the reasoning chain that depends on this definition is circular.
  - stipulative_bias: A definition (D#) is framed to presuppose the conclusion — the \
    inference begs the question because the definition smuggles in the desired answer.
  - conceptual_conflation: A definition (D#) conflates two distinct concepts — this is \
    a form of equivocation that breaks the inference from premises to conclusion.

When targeting a definition (D#), explain which downstream A#/E# nodes are affected and how \
the definitional flaw cascades through the argument.

The most effective questions often combine vectors. When a question spans multiple vectors, \
choose the primary one — the attack_type and attack_strategy that best describes the core \
thrust of the question.
</attack_taxonomy>

Then ask up to {max_questions} high-leverage questions. Each should target a specific node \
ID — use a single target unless two nodes are genuinely interdependent (e.g., a claim and \
the assumption it rests on). Questions must be answerable (not rhetorical).

Inside any <reasoning> blocks, first briefly state your opponent's position in its strongest form, \
then identify where that strongest version is genuinely vulnerable.
</instructions>

<example>
```json
{{
  "qid": "Q1",
  "text": "Your X2 acknowledges the hard problem challenge with only 'partial' response \
sufficiency. If you can't fully address the strongest objection to your core claim C2, how \
do you justify C2's strength at 0.55 rather than something lower?",
  "target_ids": ["C2"],
  "attack_type": "undermining",
  "attack_strategy": "challenge_strength_calibration"
}}
```

```json
{{
  "qid": "Q3",
  "text": "Your definition of 'consciousness' (D1) as 'any information processing' is \
over-extended — it would classify a thermostat as conscious, undermining A2 and E1 which \
depend on this definition.",
  "target_ids": ["D1"],
  "attack_type": "undermining",
  "attack_strategy": "over_extension"
}}
```
</example>

<output_format>
Your response must contain:
1. A <reasoning>...</reasoning> block with your thorough and in-depth analysis (originating from your own belief object) and construction of any arguments against your opponent's positions
2. One fenced JSON code block:

```json
{{
  "questions": [
    {{
      "qid": "Q1",
      "text": "Your question text here",
      "target_ids": ["C3"],
      "attack_type": "undermining | rebutting | undercutting",
      "attack_strategy": "one of the strategies listed under your chosen attack_type"
    }}
  ]
}}
```

**Field requirements:**
- **qid**: Sequential question identifier (Q1, Q2, ...).
- **text**: The question itself. Must be answerable (not rhetorical) and target specific \
  IDs in the opponent's belief structure.
- **target_ids**: Array of 1-2 CBS node IDs (A#, C#, D#, E#, X#, U#) that this question \
  targets. Prefer a single target; only use two when the nodes are directly interdependent \
  (e.g., a claim and the assumption it depends on).
- **attack_type**: One of "undermining", "rebutting", or "undercutting". \
  This must match the attack vector described in the taxonomy above.
- **attack_strategy**: One of the specific strategies listed under the chosen attack_type. \
  The strategy must belong to the selected attack_type category.
</output_format>
"""



def build_stage_3_structured_rebuttal_prompt(topic: str, agent_name: str, opponent_name: str, received_questions_json: str, agent_belief_json: str, max_rebuttals: int = 5, max_rebuttal_length_chars: int = 500) -> str:
    """
    Stage 3: Structured Rebuttal + Patch Proposals.

    The agent answers opponent questions (Q#), *links* answers to the agent's own IDs (A#/C#/E#),
    and proposes patches where appropriate. Output is ONE JSON block containing both rebuttals
    and patches.

    Acronyms expanded:
    - JSON: JavaScript Object Notation
    - ID: Identifier (A#, C#, E#, U#, X#)
    """
    _debate_ctx = build_debate_context("Rebuttal — defending against your opponent's challenges")
    return f"""\
{_debate_ctx}

<context>
<your_belief>
```json
{agent_belief_json}
```
</your_belief>

<questions_received from="{opponent_name}">
```json
{received_questions_json}
```
</questions_received>
</context>

<instructions>
You are {agent_name} responding to cross-examination from {opponent_name} on: "{topic}"

Your primary goal is to defend your positions, to the best of your ability. \
Inside <reasoning> blocks, think honestly through each question: Does it identify a genuine weakness? \
Can I refute it with evidence or logic, or should I concede? Am I rationalizing a defense of a weak \
point? If this question targets one of my counterpositions (X#), does their challenge strengthen the \
counterposition or does my existing response hold?

Then provide up to {max_rebuttals} rebuttals in a single JSON object containing both responses and patches.

ACTIONS (binding commitments):

"refute" — You reject the challenge. Provide a specific counter-argument or cite evidence (E#). \
Your answer must argue AGAINST the challenge.

"concede" — You accept the challenge identifies a real weakness. Acknowledge it in your answer AND \
include a weakening patch. If you write "I concede" but then defend your position, you are violating \
the protocol.

"defer" — The challenge raises an unresolved uncertainty. Explain what would resolve it. \
Patches SHOULD include an `add_uncertainty` targeting the disputed nodes.

If your definitions (D# nodes) have been challenged:
  - "refute": Defend the definition by explaining why it is precise, non-circular, and \
appropriate for this context. Cite philosophical precedent or explain why the specific \
formulation is necessary for your argument.
  - "concede": If the challenge has merit, include an update_definition patch to revise \
the definition, lower its strength, or retract it. Consider whether narrowing or \
broadening the definition resolves the objection.
  - "defer": If the definitional question is genuinely unresolved, add a U# targeting the \
D# node.

When adding a counterposition, you MUST include both attack_type and attack_strategy. \
The attack_strategy must be a valid strategy for the given attack_type.
</instructions>

<examples>
GOOD refute: action "refute", answer argues against the challenge with evidence.
GOOD concede: action "concede", answer says "You're right that C3's strength is unjustified given \
E2's limitations. I'll lower it to 0.6." Patch included.
BAD concede (VIOLATION): action "concede", answer says "While I acknowledge this, my position is \
well-supported because..." — this is a refute disguised as concede.
</examples>

<output_format>
Your response must contain:
1. A <reasoning>...</reasoning> block with your thorough and in-depth response from your position in response to each question you received from an opponent
2. One fenced JSON code block:

```json
{{
  "rebuttals": [
    {{
      "qid": "Q1",
      "answer": "",
      "action": "refute|concede|defer",
      "linked_ids": ["C2", "E4"]
    }}
  ],
  "patches": [
    // --- Modify existing nodes ---
    {{"op": "update_claim", "target_id": "C1", "changes": {{"strength": 0.6, "strength_justification": "Lowered — E1 was shown to be outdated"}}}},
    {{"op": "update_claim", "target_id": "C3", "changes": {{"status": "retracted", "strength_justification": "Retracted — no longer supported after E1 discredited"}}}},
    {{"op": "update_evidence", "target_id": "E1", "changes": {{"strength": 0.5, "strength_justification": "Downgraded — methodology questioned"}}}},
    {{"op": "update_assumption", "target_id": "A2", "changes": {{"strength": 0.6, "status": "revised", "strength_justification": "Weakened after challenge"}}}},
    {{"op": "update_definition", "target_id": "D1", "changes": {{"definition": "...", "strength": 0.55, "strength_justification": "...", "status": "revised"}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": {{"my_response": "...", "response_sufficiency": "sufficient"}}}},
    {{"op": "resolve_uncertainty", "target_id": "U1", "resolution_note": "Resolved by new evidence E4"}},

    // --- Add new nodes (use the next available number, e.g. if highest evidence ID is E3, use E4) ---
    {{"op": "add_claim", "item": {{"id": "C4", "type": "deductive", "statement": "...", "depends_on": ["A1", "E2"], "strength": 0.7, "status": "active", "strength_justification": "...", "predictions": [{{"statement": "...", "test": "...", "decision_criterion": "..."}}], "inference_chain": [{{"role": "premise", "text": "...", "reference": "A1"}}, {{"role": "premise", "text": "...", "reference": "E2"}}, {{"role": "inference", "text": "...", "inference_type": "deductive"}}, {{"role": "conclusion", "text": "..."}}]}}}},
    {{"op": "add_evidence", "item": {{"id": "E4", "type": "empirical", "summary": "...", "source": "...", "supports_claims": ["C1"], "strength": 0.7, "status": "active", "strength_justification": "...", "supported_by_definitions": ["D1"]}}}},
    {{"op": "add_assumption", "item": {{"id": "A3", "type": "empirical", "statement": "...", "supports_claims": ["C1"], "strength": 0.75, "status": "active", "strength_justification": "..."}}}},
    {{"op": "add_definition", "item": {{"id": "D3", "term": "...", "definition": "...", "strength": 0.8, "strength_justification": "...", "status": "active", "used_by": ["A1", "E2"]}}}},
    {{"op": "add_counterposition", "item": {{"id": "X3", "targets": ["C2"], "attack_type": "undermining", "attack_strategy": "challenge_assumption", "statement": "...", "my_response": "...", "response_sufficiency": "partial"}}}},
    {{"op": "add_uncertainty", "item": {{"id": "U2", "targets": ["C1", "E1"], "question": "...", "status": "active", "importance": "high"}}}}
  ]
}}
```

Use only the operations shown above. Every new item must include all required fields.

If any action is "concede", patches MUST contain at least one weakening patch for that question. \
If any action is "defer", patches SHOULD include an `add_uncertainty` for that question. \
If any action is "refute", then no patches are warranted: "patches": []
</output_format>
"""

def build_stage_5_belief_update_prompt_cbs(agent_name: str,
                                             challenge_rebuttal_pairs: list[dict],
                                             prior_belief_json: str,
                                             stage_3_patches_json: str = "",
                                             breadth_sensitivity: float = None) -> str:
    """
    Stage 5: Belief update via PATCH operations based on adjudication outcomes.

    Args:
        agent_name: Name of the agent updating beliefs.
        challenge_rebuttal_pairs: List of adjudication outcome dicts.
        prior_belief_json: Agent's current CBS belief as JSON string.
        stage_3_patches_json: Optional JSON of patches proposed during Stage 3 rebuttals.
        breadth_sensitivity: The p exponent for the breadth formula. Defaults
            to BREADTH_SENSITIVITY from patches.py.
    """
    from chal.beliefs.patches import BREADTH_SENSITIVITY
    p = breadth_sensitivity if breadth_sensitivity is not None else BREADTH_SENSITIVITY
    lines = []
    for entry in challenge_rebuttal_pairs:
        challenger = entry.get("challenger", "?")
        challenge = entry.get("challenge", "?")
        rebuttal = entry.get("rebuttal", "")
        res = entry.get("resolution", {}) or {}
        status = res.get("status", "?")
        reasoning = res.get("reasoning", "")
        lines.append(
            f"- From {challenger}: {challenge}\n"
            f"  Your rebuttal: {rebuttal}\n"
            f"  → Outcome: {status} | Reason: {reasoning}"
        )
    outcomes_formatted = "\n".join(lines) if lines else "(no adjudications available)"

    # Build optional stage 3 patches section
    stage_3_section = ""
    if stage_3_patches_json:
        stage_3_section = (
            "<your_stage_3_responses>\n"
            "Patches you proposed during rebuttal (NOT automatically applied — you must re-include "
            "any you still endorse):\n"
            "```json\n" + stage_3_patches_json + "\n```\n"
            "</your_stage_3_responses>\n"
        )

    _debate_ctx = build_debate_context("Belief update — incorporating adjudication outcomes")
    return f"""\
{_debate_ctx}

<context>
<prior_belief>
```json
{prior_belief_json}
```
</prior_belief>

<adjudication_outcomes>
{outcomes_formatted}
</adjudication_outcomes>

{stage_3_section}</context>

<strength_scale>
All strength values (thesis, claims, assumptions, evidence) share a common scale:
| Range | Label | Meaning |
|---|---|---|
| 0.0 | Vacuous | No credible support; should be retracted |
| 0.1-0.3 | Weak | Critical support missing; serious unaddressed challenges |
| 0.3-0.5 | Contested | More reasons to doubt than believe; needs significant strengthening |
| 0.5 | Threshold | Could go either way; evenly balanced |
| 0.5-0.7 | Moderate | More reasons to believe than doubt; some gaps remain |
| 0.7-0.9 | Strong | Well-supported; minor open questions |
| 0.9-1.0 | Robust | Near-certain given available evidence and reasoning |
| 1.0 | Definitive | Established beyond reasonable dispute |
</strength_scale>

<thesis_strength>
When revising beliefs, consider how your changes affect thesis strength.
Thesis strength = avg(active_claim_strengths) × (n^p / (n^p + 1)) where n = active claims, p = {p}.
If you retract a claim, the breadth multiplier decreases. If you lower a claim's \
strength, the average decreases. The thesis strength is always determined by this \
formula — make changes that genuinely improve your position, not just the numbers.
</thesis_strength>

<instructions>
Agent {agent_name}, generate PATCH operations to update your belief.

Inside <reasoning> tags, think through each outcome: what weakness or strength was identified, \
which elements need to change, and what magnitude of change is appropriate. Consider whether \
outcomes affect your counterpositions — if a critique exposed a new vulnerability, add a \
counterposition (X#). If you successfully defended against a challenge you'd listed as "partial," \
upgrade response_sufficiency. Cross-reference your Stage 3 responses if provided — ensure your \
patches are consistent with any concessions you already made. If a rebuttal introduced new \
evidence or arguments that the adjudicator acknowledged, incorporate them via add_evidence or \
update_claim patches.

Then output a single fenced JSON code block.

SUPPORTED OPERATIONS:
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ...", \
"inference_chain": [...]}}}}
  (All fields in changes are optional — include only the ones you want to modify. \
To retract a claim, set {{"status": "retracted"}} in changes — strength is forced to 0.0 automatically. \
Include inference_chain only if the reasoning structure itself needs revision.)
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "supports_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "...", "supported_by_definitions": ["D#"]}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", \
"new_type": "empirical|foundational|methodological|scoping"}}
  (new_statement and new_type are TOP-LEVEL fields, NOT inside "changes". Only include if changing them.)
- {{"op": "add_definition", "item": {{"id": "D#", "term": "...", "definition": "...", \
"strength": 0.8, "strength_justification": "...", "status": "active", "used_by": ["A#", "E#"]}}}}
- {{"op": "update_definition", "target_id": "D#", "changes": {{"definition": "...", \
"strength": 0.55, "strength_justification": "...", "status": "revised"}}}}
  Mutable fields: definition, strength, strength_justification, status, used_by.
  Immutable fields: id, term (to redefine a term, retract the old D# and add a new one).
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "high|medium|low"}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"attack_strategy": "...", "statement": "...", "my_response": "...", \
"response_sufficiency": "sufficient|partial|unaddressed"}}}}
- {{"op": "update_counterposition", "target_id": "X#", "changes": \
{{"my_response": "...", "response_sufficiency": "..."}}}}
</instructions>

<mandatory_rules>
BINDING — not suggestions:
- CRITIQUE_VALID against you → at least one weakening patch per outcome. Lower strength ≥0.1, \
retire the claim, or refine to address the flaw. Empty patches after CRITIQUE_VALID = protocol \
violation. Stage 3 patches do NOT carry over. Also: if the critique reveals a new vulnerability, \
add a counterposition (X#) recording it.
- If a definition-targeting challenge (strategies: over_extension, under_extension, \
circularity, stipulative_bias, conceptual_conflation) is sustained (CRITIQUE_VALID): \
lower the targeted D# strength via update_definition. This automatically caps all A#/E# in \
the D#'s used_by list. Add an X# counterposition recording the definitional vulnerability.
- REBUTTAL_VALID for you → defense boosts are applied automatically by the system after \
your patches. Do NOT manually increase node strengths for successful defenses. You SHOULD \
update the response_sufficiency of any counterposition (X#) you successfully defended against.
- UNRESOLVED → required: add uncertainty (U#) with targets referencing the disputed nodes, \
status: "active"; optional: lower strength ~0.05.
- Thesis strength is always: avg(active claim strengths) × (n^p / (n^p + 1)) where p = {p}. \
If you lowered or retracted a claim, thesis strength will be recalculated automatically.
- DEFINITION CEILING: A#/E# strength ≤ min(active D# strengths from supported_by_definitions). \
If you lower a D# strength, the ceiling automatically propagates — do NOT manually lower \
every dependent A#/E# (the system handles propagation).
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
- Review your existing uncertainties (U#), prioritizing "high" and "medium" importance — \
these are the most likely to be targeted by opponents via press_uncertainty attacks. If you \
can now resolve any — through new evidence, reasoning, or claims developed during this \
debate — add the supporting material (new C#, E#, or A#) and use resolve_uncertainty to \
mark the U# as resolved. The resolution_note must reference the new material. If you cannot \
resolve a U#, leave it active.
- Verify that your belief system is internally consistent — no contradictions between claims, \
no evidence that undermines your own claims, no incompatible assumptions, and strength \
assignments consistent with their dependencies.
</mandatory_rules>

<example>
Outcomes: CRITIQUE_VALID on C2, REBUTTAL_VALID on C1 (defended against X1's challenge), \
UNRESOLVED on C4:

```json
{{
  "patches": [
    {{"op": "update_claim", "target_id": "C2", "changes": {{"strength": 0.55, \
"strength_justification": "0.55 — reduced from 0.75: E2 (0.55) insufficient for causal claim; limited by E2 (lowest dependency at 0.55)"}}}},
    {{"op": "add_counterposition", "item": {{"id": "X4", "targets": ["C2"], \
"attack_type": "rebutting", "attack_strategy": "present_counter_evidence", \
"statement": "Challenger demonstrated E2's correlational \
limitations undermine causal inference", "my_response": "Acknowledged — \
need stronger evidence for causal claims", "response_sufficiency": "partial"}}}},
    {{"op": "update_claim", "target_id": "C1", "changes": {{"strength": 0.75}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": \
{{"response_sufficiency": "sufficient", "my_response": "Successfully defended bidirectional \
dependence argument in cross-examination"}}}},
    {{"op": "add_uncertainty", "item": {{"id": "U3", "targets": ["C4"], "question": "Is C4's \
computational model compatible with opponent's C3?", "status": "active", "importance": "medium"}}}},
    {{"op": "update_definition", "target_id": "D2", "changes": {{"strength": 0.50, \
"strength_justification": "0.50 — revised after over-extension challenge; original definition \
captured unintended cases", "status": "revised"}}}}
  ]
}}
```
</example>

<output_format>
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block: {{"patches": [...]}}

Self-check:
- Did I produce ≥N weakening patches for N CRITIQUE_VALID outcomes?
- Did I record new vulnerabilities as counterpositions?
- Is thesis strength = avg(active claim strengths) × (n^p / (n^p + 1)) where p = {p}?
- Are there any internal contradictions in my belief system?
- Have I reviewed my uncertainties (U#) and resolved any that I can now address?
- Have I considered whether any D# nodes need updating after sustained challenges?
</output_format>
"""

def build_stage_5_phase1_enforcement_prompt(agent_name: str,
                                             challenge_rebuttal_pairs: list[dict],
                                             prior_belief_json: str,
                                             stage_3_patches_json: str = "") -> str:
    """
    Stage 5 Phase 1: Adjudication enforcement via PATCH operations.

    The agent's ONLY job is to respond to adjudication outcomes with appropriate patches.
    No thesis rewrite, no strategic retractions — those come in Phase 2.

    Args:
        agent_name: Name of the agent updating beliefs.
        challenge_rebuttal_pairs: List of adjudication outcome dicts.
        prior_belief_json: Agent's current CBS belief as JSON string.
        stage_3_patches_json: Optional JSON of patches proposed during Stage 3 rebuttals.
    """
    lines = []
    for entry in challenge_rebuttal_pairs:
        challenger = entry.get("challenger", "?")
        challenge = entry.get("challenge", "?")
        rebuttal = entry.get("rebuttal", "")
        res = entry.get("resolution", {}) or {}
        status = res.get("status", "?")
        reasoning = res.get("reasoning", "")
        lines.append(
            f"- From {challenger}: {challenge}\n"
            f"  Your rebuttal: {rebuttal}\n"
            f"  → Outcome: {status} | Reason: {reasoning}"
        )
    outcomes_formatted = "\n".join(lines) if lines else "(no adjudications available)"

    # Build optional stage 3 patches section
    stage_3_section = ""
    if stage_3_patches_json:
        stage_3_section = (
            "<your_stage_3_responses>\n"
            "Patches you proposed during rebuttal (NOT automatically applied — you must re-include "
            "any you still endorse):\n"
            "```json\n" + stage_3_patches_json + "\n```\n"
            "</your_stage_3_responses>\n"
        )

    _debate_ctx = build_debate_context("Belief update (enforcement) — incorporating adjudication outcomes")
    return f"""\
{_debate_ctx}

<context>
<prior_belief>
```json
{prior_belief_json}
```
</prior_belief>

<adjudication_outcomes>
{outcomes_formatted}
</adjudication_outcomes>

{stage_3_section}</context>

<strength_scale>
All strength values (thesis, claims, assumptions, evidence) share a common scale:
| Range | Label | Meaning |
|---|---|---|
| 0.0 | Vacuous | No credible support; should be retracted |
| 0.1-0.3 | Weak | Critical support missing; serious unaddressed challenges |
| 0.3-0.5 | Contested | More reasons to doubt than believe; needs significant strengthening |
| 0.5 | Threshold | Could go either way; evenly balanced |
| 0.5-0.7 | Moderate | More reasons to believe than doubt; some gaps remain |
| 0.7-0.9 | Strong | Well-supported; minor open questions |
| 0.9-1.0 | Robust | Near-certain given available evidence and reasoning |
| 1.0 | Definitive | Established beyond reasonable dispute |
</strength_scale>

<instructions>
Agent {agent_name}, this is Phase 1: Adjudication Enforcement.

Your ONLY job here is to respond to adjudication outcomes with appropriate patches. \
Do NOT rewrite your thesis text or make strategic retractions — that comes in Phase 2.

Inside <reasoning> tags, think through each outcome: what weakness or strength was identified, \
which elements need to change, and what magnitude of change is appropriate. Consider whether \
outcomes affect your counterpositions — if a critique exposed a new vulnerability, add a \
counterposition (X#). If you successfully defended against a challenge you'd listed as "partial," \
upgrade response_sufficiency. Cross-reference your Stage 3 responses if provided — ensure your \
patches are consistent with any concessions you already made. If a rebuttal introduced new \
evidence or arguments that the adjudicator acknowledged, incorporate them via add_evidence or \
update_claim patches.

Then output a single fenced JSON code block.

SUPPORTED OPERATIONS:
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ...", \
"inference_chain": [...]}}}}
  (All fields in changes are optional — include only the ones you want to modify. \
To retract a claim, set {{"status": "retracted"}} in changes — strength is forced to 0.0 automatically. \
Include inference_chain only if the reasoning structure itself needs revision.)
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "supports_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "...", "supported_by_definitions": ["D#"]}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", \
"new_type": "empirical|foundational|methodological|scoping"}}
  (new_statement and new_type are TOP-LEVEL fields, NOT inside "changes". Only include if changing them.)
- {{"op": "add_definition", "item": {{"id": "D#", "term": "...", "definition": "...", \
"strength": 0.8, "strength_justification": "...", "status": "active", "used_by": ["A#", "E#"]}}}}
- {{"op": "update_definition", "target_id": "D#", "changes": {{"definition": "...", \
"strength": 0.55, "strength_justification": "...", "status": "revised"}}}}
  Mutable fields: definition, strength, strength_justification, status, used_by.
  Immutable fields: id, term (to redefine a term, retract the old D# and add a new one).
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "high|medium|low"}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"attack_strategy": "...", "statement": "...", "my_response": "...", \
"response_sufficiency": "sufficient|partial|unaddressed"}}}}
- {{"op": "update_counterposition", "target_id": "X#", "changes": \
{{"my_response": "...", "response_sufficiency": "..."}}}}

SCOPE RESTRICTION: Focus only on direct responses to adjudication outcomes. Do not:
- Rewrite thesis stance text or summary bullets
- Make strategic retractions unrelated to adjudication outcomes
- Recalculate thesis strength based on the thesis strength formula (that's Phase 2)
You may adjust thesis strength only as a direct consequence of adjudication-mandated changes.
</instructions>

<mandatory_rules>
BINDING — not suggestions:
- CRITIQUE_VALID against you → at least one weakening patch per outcome. Lower strength ≥0.1, \
retire the claim, or refine to address the flaw. Empty patches after CRITIQUE_VALID = protocol \
violation. Stage 3 patches do NOT carry over. Also: if the critique reveals a new vulnerability, \
add a counterposition (X#) recording it.
- If a definition-targeting challenge (strategies: over_extension, under_extension, \
circularity, stipulative_bias, conceptual_conflation) is sustained (CRITIQUE_VALID): \
lower the targeted D# strength via update_definition. This automatically caps all A#/E# in \
the D#'s used_by list. Add an X# counterposition recording the definitional vulnerability.
- REBUTTAL_VALID for you → defense boosts are applied automatically by the system after \
Phase 1. Do NOT manually increase node strengths for successful defenses. You SHOULD update \
the response_sufficiency of any counterposition (X#) you successfully defended against.
- UNRESOLVED → required: add uncertainty (U#) with targets referencing the disputed nodes, \
status: "active"; optional: lower strength ~0.05.
- DEFINITION CEILING: A#/E# strength ≤ min(active D# strengths from supported_by_definitions). \
If you lower a D# strength, the ceiling automatically propagates — do NOT manually lower \
every dependent A#/E# (the system handles propagation).
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
- Review your existing uncertainties (U#), prioritizing "high" and "medium" importance — \
these are the most likely to be targeted by opponents via press_uncertainty attacks. If you \
can now resolve any — through new evidence, reasoning, or claims developed during this \
debate — add the supporting material (new C#, E#, or A#) and use resolve_uncertainty to \
mark the U# as resolved. The resolution_note must reference the new material. If you cannot \
resolve a U#, leave it active.
</mandatory_rules>

<example>
Outcomes: CRITIQUE_VALID on C2, REBUTTAL_VALID on C1 (defended against X1's challenge), \
UNRESOLVED on C4:

```json
{{
  "patches": [
    {{"op": "update_claim", "target_id": "C2", "changes": {{"strength": 0.55, \
"strength_justification": "0.55 — reduced from 0.75: E2 (0.55) insufficient for causal claim; limited by E2 (lowest dependency at 0.55)"}}}},
    {{"op": "add_counterposition", "item": {{"id": "X4", "targets": ["C2"], \
"attack_type": "rebutting", "attack_strategy": "present_counter_evidence", \
"statement": "Challenger demonstrated E2's correlational \
limitations undermine causal inference", "my_response": "Acknowledged — \
need stronger evidence for causal claims", "response_sufficiency": "partial"}}}},
    {{"op": "update_claim", "target_id": "C1", "changes": {{"strength": 0.75}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": \
{{"response_sufficiency": "sufficient", "my_response": "Successfully defended bidirectional \
dependence argument in cross-examination"}}}},
    {{"op": "add_uncertainty", "item": {{"id": "U3", "targets": ["C4"], "question": "Is C4's \
computational model compatible with opponent's C3?", "status": "active", "importance": "medium"}}}}
  ]
}}
```
</example>

<output_format>
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block: {{"patches": [...]}}

Self-check:
- Did I produce ≥N weakening patches for N CRITIQUE_VALID outcomes?
- Did I record new vulnerabilities as counterpositions?
- Did I stay within scope (no thesis text rewrites, no strategic retractions)?
</output_format>
"""


def compute_position_analysis(belief: dict, breadth_sensitivity: float = None) -> str:
    """
    Compute a dynamic <position_analysis> block from a parsed belief dict.

    Extracts active claim strengths, computes partial derivatives, scenario
    projections, lowest-strength dependencies, and a strategic recommendation.

    Args:
        belief: Parsed CBS belief dict.
        breadth_sensitivity: The p exponent for breadth formula. Defaults to
            BREADTH_SENSITIVITY from patches.py.

    Returns:
        A formatted <position_analysis> XML block string.
    """
    from chal.beliefs.patches import BREADTH_SENSITIVITY
    if breadth_sensitivity is None:
        breadth_sensitivity = BREADTH_SENSITIVITY
    p = breadth_sensitivity

    # --- Extract active claim strengths ---
    claims = belief.get("claims", [])
    active_claims = [c for c in claims if c.get("status") != "retracted"]
    active_strengths = [c.get("strength", 0.5) for c in active_claims]

    if not active_strengths:
        return (
            "<position_analysis>\n"
            "No active claims found — position analysis unavailable.\n"
            "</position_analysis>"
        )

    n = len(active_strengths)
    s = sum(active_strengths) / n
    n_p = n ** p
    breadth = n_p / (n_p + 1)
    T = s * breadth

    # --- Partial derivatives ---
    # dT/ds = n^p / (n^p + 1)  (the breadth multiplier itself)
    dT_ds = breadth
    # dT/dn = s * p * n^(p-1) / (n^p + 1)^2
    dT_dn = s * p * (n ** (p - 1)) / ((n_p + 1) ** 2)

    # --- Scenario projections ---
    # Scenario 1: Raise avg claim strength by 0.10
    s1 = min(s + 0.10, 1.0)
    T1 = s1 * breadth
    d1 = T1 - T

    # Scenario 2: Add a claim at current average
    n2 = n + 1
    n2_p = n2 ** p
    breadth2 = n2_p / (n2_p + 1)
    T2 = s * breadth2  # avg unchanged since new claim = current avg
    d2 = T2 - T

    # Scenario 3: Add a claim 0.15 above current average
    s_high = min(s + 0.15, 1.0)
    s3 = (s * n + s_high) / n2  # new average after adding stronger claim
    T3 = s3 * breadth2
    d3 = T3 - T

    # Scenario 4 (optional): Retract weakest claim — only if n >= 2
    retract_line = ""
    if n >= 2:
        weakest_idx = active_strengths.index(min(active_strengths))
        weakest_claim = active_claims[weakest_idx]
        weakest_id = weakest_claim.get("id", "?")
        weakest_str = active_strengths[weakest_idx]
        remaining = [s_val for i, s_val in enumerate(active_strengths) if i != weakest_idx]
        n4 = len(remaining)
        s4 = sum(remaining) / n4
        n4_p = n4 ** p
        breadth4 = n4_p / (n4_p + 1)
        T4 = s4 * breadth4
        d4 = T4 - T
        retract_line = (
            f"\n  - Retract weakest claim ({weakest_id} at {weakest_str:.2f}) "
            f"→ thesis becomes {T4:.2f} ({d4:+.2f})"
        )

    # --- Weakest dependencies (only active/revised, not retracted) ---
    assumptions = belief.get("assumptions", [])
    evidence = belief.get("evidence", [])

    # Build mapping: dependency → which claims it supports
    dep_items = []
    for a in assumptions:
        if a.get("status") == "retracted":
            continue
        aid = a.get("id", "?")
        a_str = a.get("strength", 0.5)
        backed_claims = [
            c.get("id", "?") for c in active_claims
            if aid in c.get("depends_on", [])
        ]
        if backed_claims:
            dep_items.append((a_str, aid, backed_claims))

    for e in evidence:
        if e.get("status") == "retracted":
            continue
        eid = e.get("id", "?")
        e_str = e.get("strength", 0.5)
        backed_claims = [
            c.get("id", "?") for c in active_claims
            if eid in c.get("depends_on", [])
        ]
        if backed_claims:
            dep_items.append((e_str, eid, backed_claims))

    # Sort by strength ascending, take bottom 3
    dep_items.sort(key=lambda x: x[0])
    bottom_deps = dep_items[:3]

    if bottom_deps:
        dep_line_parts = []
        for dep_str, dep_id, backed in bottom_deps:
            line = f"  {dep_id} (strength {dep_str:.2f}) — backs: {', '.join(backed)}"
            # Generate suggested action based on node type
            if dep_id.startswith("A"):
                line += (
                    f"\n    → Suggested: update_assumption {dep_id} to revise statement, "
                    f"or add_evidence corroborating {'/'.join(backed)}"
                )
            elif dep_id.startswith("E"):
                line += (
                    f"\n    → Suggested: update_evidence {dep_id} with stronger source, "
                    f"or add_evidence with additional support for {'/'.join(backed)}"
                )
            dep_line_parts.append(line)
        dep_lines = "\n".join(dep_line_parts)
    else:
        dep_lines = "  (no dependency data available)"

    # --- D# vulnerability analysis ---
    def_vulnerability_lines = []

    active_defs = [d for d in belief.get("definitions", [])
                   if d.get("status") != "retracted"]

    if active_defs:
        # Build reverse map: D# id -> list of A#/E# ids it supports
        def_dependents: dict = {d.get("id", "?"): [] for d in active_defs}
        for collection_key in ("assumptions", "evidence"):
            for node in belief.get(collection_key, []):
                if node.get("status") == "retracted":
                    continue
                for did in node.get("supported_by_definitions", []):
                    if did in def_dependents:
                        def_dependents[did].append(node.get("id", "?"))

        # Weak definitions: low strength supporting many nodes
        for d in active_defs:
            did = d.get("id", "?")
            d_str = d.get("strength", 0.5)
            dependents = def_dependents.get(did, [])
            if d_str < 0.6 and len(dependents) >= 2:
                def_vulnerability_lines.append(
                    f"  WEAK DEFINITION: {did} (strength {d_str:.2f}) supports "
                    f"{len(dependents)} nodes ({', '.join(dependents)}). "
                    f"Strengthening or revising this definition would raise "
                    f"the ceiling on all dependent nodes."
                    f"\n    → Suggested: update_definition {did} with a more precise definition to raise its strength"
                )

        # Bottleneck definitions: A#/E# constrained by a single active D#
        for collection_key in ("assumptions", "evidence"):
            for node in belief.get(collection_key, []):
                if node.get("status") == "retracted":
                    continue
                supported_defs = node.get("supported_by_definitions", [])
                active_support = [
                    did for did in supported_defs
                    if any(d.get("id") == did and d.get("status") != "retracted"
                           for d in belief.get("definitions", []))
                ]
                if len(active_support) == 1:
                    single_did = active_support[0]
                    single_d = next(
                        (d for d in active_defs if d.get("id") == single_did), None
                    )
                    if single_d:
                        node_id = node.get('id', '?')
                        def_vulnerability_lines.append(
                            f"  BOTTLENECK: {node_id} depends on "
                            f"a single definition {single_did} (strength "
                            f"{single_d.get('strength', 0.5):.2f}). If {single_did} "
                            f"is challenged, {node_id} loses all "
                            f"definitional support."
                            f"\n    → Suggested: add_definition with a complementary term, "
                            f"then update {node_id}'s supported_by_definitions"
                        )

    def_vulnerability_section = ""
    if def_vulnerability_lines:
        def_vulnerability_section = (
            "\n"
            "DEFINITIONAL VULNERABILITIES\n"
            + "\n".join(def_vulnerability_lines) + "\n"
        )

    # --- Strategic recommendation ---
    gain_s = dT_ds * 0.10  # approximate gain from +0.10 avg strength
    gain_n = dT_dn          # gain from adding one claim

    if gain_s > gain_n * 1.5:
        recommendation = (
            f"Raising average claim strength is your strongest lever right now. Focus on:\n"
            f"   (1) strengthening low-strength assumptions or evidence that are limiting your claims\n"
            f"       (see LOWEST-STRENGTH DEPENDENCIES above), or\n"
            f"   (2) adding new claims with strength ABOVE {s:.2f} — this raises both the\n"
            f"       average AND the breadth multiplier simultaneously."
        )
    elif gain_n > gain_s * 1.5:
        recommendation = (
            f"Adding more claims is your strongest lever right now. Each new well-supported\n"
            f"   claim will significantly raise your breadth multiplier. To ensure the new\n"
            f"   claim helps: its strength (and the strengths of all supporting A# and E#)\n"
            f"   must be above your current average of {s:.2f}. Otherwise it will drag\n"
            f"   down the average and may lower your thesis strength."
        )
    else:
        recommendation = (
            f"Both levers are roughly equally valuable. The best strategy is to add a\n"
            f"   new claim with strength above {s:.2f} — this raises both the average and\n"
            f"   the breadth multiplier. Make sure supporting A# and E# are also above\n"
            f"   {s:.2f}."
        )

    # --- Orphan detection (dynamic) ---
    orphan_lines = []
    definitions = belief.get("definitions", [])

    # Build node status map for claim orphan detection
    node_statuses = {}
    for a in assumptions:
        node_statuses[a.get("id", "")] = a.get("status", "active")
    for e in evidence:
        node_statuses[e.get("id", "")] = e.get("status", "active")
    for c in claims:
        node_statuses[c.get("id", "")] = c.get("status", "active")

    # Detect orphaned A#/E# (no active D# support)
    for collection_key in ("assumptions", "evidence"):
        for node in belief.get(collection_key, []):
            if node.get("status") == "retracted":
                continue
            supported_defs = node.get("supported_by_definitions", [])
            active_defs = [
                d for d in definitions
                if d["id"] in supported_defs and d.get("status") != "retracted"
            ]
            if not active_defs and supported_defs:
                orphan_lines.append(
                    f"  {node['id']} has NO active definitional support "
                    f"(capped at 0.6)."
                    f"\n    → Suggested: add_definition covering {node['id']}'s key term, include {node['id']} in used_by"
                )

    # Detect orphaned C# (no active A#/E# dependencies)
    for claim in active_claims:
        all_deps = claim.get("depends_on", [])
        active_deps = [
            dep_id for dep_id in all_deps
            if node_statuses.get(dep_id) != "retracted"
        ]
        if not active_deps and all_deps:
            orphan_lines.append(
                f"  {claim['id']} has NO active supporting assumptions or "
                f"evidence (capped at 0.2 — unfounded claim)."
                f"\n    → HIGH PRIORITY: add_evidence and/or add_assumption supporting {claim['id']}, then update {claim['id']}'s depends_on"
            )
        elif not active_deps and not all_deps:
            orphan_lines.append(
                f"  {claim['id']} has no depends_on entries at all "
                f"(capped at 0.2 — unfounded claim)."
                f"\n    → HIGH PRIORITY: add_evidence and/or add_assumption supporting {claim['id']}, then update {claim['id']}'s depends_on"
            )

    orphan_section = ""
    if orphan_lines:
        orphan_section = (
            "\n"
            "STRUCTURAL GAPS — HIGH PRIORITY\n"
            "  The following nodes are missing required support, which caps their\n"
            "  effective strength and drags down your thesis. Fixing these is often\n"
            "  the single highest-impact action you can take.\n"
            + "\n".join(orphan_lines) + "\n"
        )

    return (
        "<position_analysis>\n"
        "YOUR CURRENT POSITION\n"
        f"  Active claims: {n}, average claim strength: {s:.2f}\n"
        f"  Breadth multiplier: {breadth:.2f}\n"
        f"  Current thesis strength: {T:.2f}\n"
        "\n"
        "SENSITIVITY AT YOUR POSITION\n"
        f"  ∂T/∂s = {dT_ds:.3f} — each +0.10 in avg claim strength → thesis +{gain_s:.3f}\n"
        f"  ∂T/∂n = {dT_dn:.3f} — adding one claim at current avg → thesis +{gain_n:.3f}\n"
        "\n"
        "SCENARIO PROJECTIONS\n"
        f"  - Raise avg claim strength by 0.10 → thesis becomes {T1:.2f} (+{d1:.2f})\n"
        f"  - Add a claim at current average ({s:.2f}) → thesis becomes {T2:.2f} (+{d2:.2f})\n"
        f"  - Add a claim at {s_high:.2f} (above avg) → thesis becomes {T3:.2f} (+{d3:.2f})"
        f"{retract_line}\n"
        "\n"
        "LOWEST-STRENGTH DEPENDENCIES\n"
        "  These assumptions/evidence have the lowest strength values and may be\n"
        "  limiting your claims (a claim cannot exceed its lowest dependency):\n"
        f"{dep_lines}\n"
        f"{def_vulnerability_section}"
        "\n"
        "STRATEGIC RECOMMENDATION\n"
        f"  {recommendation}\n"
        f"{orphan_section}"
        "\n"
        "INTEGRITY REMINDER\n"
        "  The analysis above shows what would help mathematically — it does not\n"
        "  authorize inflating strength values to hit those targets. Every strength\n"
        "  you assign must reflect your genuine epistemic assessment:\n"
        "  - A new claim at 0.80 must be backed by evidence and assumptions that\n"
        "    genuinely warrant 0.80\n"
        "  - A new assumption or evidence item's strength must reflect its actual\n"
        "    quality, not a number chosen to raise the average\n"
        "  - Inflated claims will be challenged and adjudicated in the next round,\n"
        "    resulting in forced weakening that leaves you worse off\n"
        "</position_analysis>"
    )


def build_stage_5_phase2_introspection_prompt(agent_name: str,
                                               intermediate_belief_json: str,
                                               phase1_changes_summary: str,
                                               breadth_sensitivity: float = None) -> str:
    """
    Stage 5 Phase 2: Introspective evaluation and thesis rewrite.

    After Phase 1 has enforced adjudication outcomes, this phase allows the agent
    to strategically evaluate their position, audit counterpositions, and rewrite
    their thesis to reflect the current state of their belief.

    Args:
        agent_name: Name of the agent.
        intermediate_belief_json: Belief JSON after Phase 1 patches applied.
        phase1_changes_summary: Human-readable summary of Phase 1 changes.
        breadth_sensitivity: The p exponent for the breadth formula. Defaults
            to BREADTH_SENSITIVITY from patches.py.
    """
    # Parse the intermediate belief to generate dynamic position analysis
    try:
        belief = json.loads(intermediate_belief_json)
    except (json.JSONDecodeError, TypeError):
        belief = {}
    _position_analysis = compute_position_analysis(belief, breadth_sensitivity)

    # Compute dynamic breadth table for thesis_strength_guide
    from chal.beliefs.patches import BREADTH_SENSITIVITY
    p = breadth_sensitivity if breadth_sensitivity is not None else BREADTH_SENSITIVITY
    _breadth_rows = []
    for nc in range(1, 8):
        nc_p = nc ** p
        bm = nc_p / (nc_p + 1)
        _breadth_rows.append(f"    {nc} claim{'s' if nc != 1 else ' '} → {bm:.2f}")
    _breadth_table = "\n".join(_breadth_rows)

    # Pre-compute example values for strength_reasoning illustration
    _ex3_np = 3 ** p
    _ex3_breadth = round(_ex3_np / (_ex3_np + 1), 2)
    _ex3_result = round(0.62 * _ex3_breadth, 2)

    # Growth example values (4 claims: 0.70, 0.65, 0.60, 0.75)
    _ex4_np = 4 ** p
    _ex4_breadth = round(_ex4_np / (_ex4_np + 1), 2)
    _ex4_growth_avg = round((0.70 + 0.65 + 0.60 + 0.75) / 4, 2)
    _ex3_growth_T = round(0.65 * _ex3_breadth, 2)  # Before: 3 claims at avg 0.65
    _ex4_growth_result = round(_ex4_growth_avg * _ex4_breadth, 2)

    # Refinement example values (2 claims → 3 claims: add C3 at 0.70)
    _ex2_np = 2 ** p
    _ex2_breadth = round(_ex2_np / (_ex2_np + 1), 2)
    _ex2_refine_T = round(0.625 * _ex2_breadth, 2)       # Before: 2 claims at avg 0.625
    _ex3_refine_avg = round((0.70 + 0.55 + 0.70) / 3, 2) # After: 3 claims
    _ex3_refine_result = round(_ex3_refine_avg * _ex3_breadth, 2)

    _debate_ctx = build_debate_context("Belief update (strategic) — strengthening your position")
    return f"""\
{_debate_ctx}

<context>
<current_belief>
Your belief after incorporating adjudication outcomes (Phase 1):
```json
{intermediate_belief_json}
```
</current_belief>

<phase1_changes>
Changes already made in Phase 1 (adjudication enforcement):
{phase1_changes_summary}
</phase1_changes>
</context>

<strength_scale>
All strength values (thesis, claims, assumptions, evidence) share a common scale:
| Range | Label | Meaning |
|---|---|---|
| 0.0 | Vacuous | No credible support; should be retracted |
| 0.1-0.3 | Weak | Critical support missing; serious unaddressed challenges |
| 0.3-0.5 | Contested | More reasons to doubt than believe; needs significant strengthening |
| 0.5 | Threshold | Could go either way; evenly balanced |
| 0.5-0.7 | Moderate | More reasons to believe than doubt; some gaps remain |
| 0.7-0.9 | Strong | Well-supported; minor open questions |
| 0.9-1.0 | Robust | Near-certain given available evidence and reasoning |
| 1.0 | Definitive | Established beyond reasonable dispute |
</strength_scale>

<thesis_strength_guide>
THESIS STRENGTH FORMULA
  thesis_strength = avg(active_claim_strengths) × breadth_multiplier
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                    "average strength" (s)           n^p / (n^p + 1)

  where n = number of active claims, p = {p} (breadth sensitivity)

SENSITIVITY (at your current values — see <position_analysis> below):
  dT/ds = n^p / (n^p + 1)               — gain per unit increase in avg strength
  dT/dn = s × p × n^(p-1) / (n^p + 1)² — gain from adding one more claim

The thesis strength is ALWAYS determined by this formula. You cannot set it
directly — it is computed from your claims.

DEPENDENCY GRAPH
  Definitions (D#)
      | ceiling |
  Assumptions (A#) and Evidence (E#) — each cannot exceed the LOWEST
                strength among its active supporting definitions (D#).
                If ALL D# support is lost, capped at 0.6.
      | ceiling |
  Claims (C#) — each claim's strength cannot exceed the LOWEST strength
                among its active/revised dependencies (A#, E#, or C#).
                If ALL support is lost, capped at 0.2.
      | determine |
  Thesis strength — computed from claim strengths and claim count

A claim cannot be stronger than the dependency with the LOWEST strength value.
Example: if a claim depends on A1 (0.85), A2 (0.40), and E1 (0.70), the
lowest is A2 at 0.40 — so the claim's maximum strength is 0.40. The claim's
strength_justification must name the limiting dependency (e.g., "limited by
A2 (0.40)"). Strengthening a claim means strengthening or adding to its
supporting evidence and assumptions.

TWO LEVERS FOR RAISING THESIS STRENGTH

Lever 1 — Raise the average claim strength:
  - Strengthen existing claims by adding stronger evidence or assumptions
  - Retract claims that are dragging the average down significantly
    (e.g., a claim at 0.3 when the others are at 0.7)
  - Only retract claims that are genuinely indefensible — do not game the
    formula by dropping valid-but-weak claims

Lever 2 — Increase breadth (add more claims):
  - More active claims raise the breadth multiplier:
{_breadth_table}
  - New claims must be genuinely supported — adding a weak claim hurts
    the average and may lower thesis strength despite raising breadth
  - Adding a claim typically requires first adding the evidence (E#) and
    assumptions (A#) that support it. All new components (A#, E#, and the
    claim itself) should have strengths above your current average claim
    strength — otherwise the new claim will drag the average down and may
    lower your thesis strength despite increasing breadth.
</thesis_strength_guide>

{_position_analysis}

<instructions>
Agent {agent_name}, this is Phase 2: Introspective Evaluation.

You have just incorporated adjudication outcomes into your belief (Phase 1). \
Your primary goal now is to strengthen your position as much as you honestly \
can. Review the <thesis_strength_guide> above — it explains exactly how \
thesis strength is computed and what you can do to raise it.

Step 1: ADDRESS OPEN ISSUES

A) Counterposition Audit
Review all your counterpositions (X#). For each "unaddressed" counterposition:
- What node(s) does it target?
- Has that target already been weakened in Phase 1?
- If NOT, you MUST either:
  (a) Weaken the target (update its strength downward), OR
  (b) Explain why the counterposition doesn't actually undermine it — update \
the counterposition's response_sufficiency to "partial" or "sufficient" \
with a substantive my_response. If you take this path, back up your \
explanation by adding new assumptions (add_assumption), evidence \
(add_evidence), or claims (add_claim) that justify why the target \
withstands the counterposition.
Unaddressed counterpositions with no impact on their targets are incoherent.

B) Uncertainty Review
Review all active uncertainties (U#). Prioritize "high" and "medium" \
importance uncertainties — these are the most likely to be targeted by \
opponents in future rounds via press_uncertainty attacks. For each one:
- Can you now resolve it — through new evidence, reasoning, or claims \
developed during this debate?
- If yes, add the supporting material (new A#, E#, or C#) and use \
resolve_uncertainty to mark it resolved.
- If no, leave it active. Consider whether it should lower the strength \
of the nodes it targets.

Step 2: STRATEGIC POSITION BUILDING
Refer to the <thesis_strength_guide> above. Your aim is to raise your \
thesis strength through genuine improvements to your argument.

Evaluate existing claims:
- Is each claim genuinely supported by its evidence and assumptions?
- Is any claim dragging down the average? Should it be retracted or \
can it be strengthened by adding new evidence (add_evidence) or \
assumptions (add_assumption)?

Strengthen weak dependencies:
- Review the LOWEST-STRENGTH DEPENDENCIES in the position analysis below. \
These nodes are bottlenecking your claim strengths — a claim cannot be \
stronger than its weakest active dependency.
- For a weak A# (low strength): Revise the statement to be more defensible \
(update_assumption), add a definition that clarifies a key term it relies on \
(add_definition), or add corroborating evidence (add_evidence) for the same \
claim so the argument doesn't rest on a single weak pillar.
- For a weak E# (low strength): Add corroborating evidence on the same claim \
(add_evidence with a stronger source), or revise the evidence summary to \
cite a more authoritative source (update_evidence).
- For a weak D# (low strength): Revise the definition to be more precise \
(update_definition). Remember: D# strengths ceiling A#/E# strengths, so \
raising a D# can unlock higher strength for all dependent nodes.
- Adding a second or third dependency to an existing claim does not automatically \
raise the claim's strength (it is still limited by the lowest), but it makes \
the claim more resilient if one dependency is later challenged or retracted.

Consider adding new claims:
- Are there well-supported arguments you made during the debate that \
aren't yet formalized as claims?
- Can you add new evidence and assumptions to support a new claim \
(add_claim)? Remember: you must add the supporting A# and E# items \
BEFORE the add_claim patch that references them.
- Would the new claim's strength be high enough to raise the average, \
not drag it down?

Retract claims only if they are genuinely unsustainable. Do not game the \
formula — the goal is intellectual honesty, not score optimization. \
Every strength value you assign must be a genuine epistemic assessment. \
Inflated claims will be challenged in the next round, and forced weakening \
from adjudication will leave you worse off than honest assessment would have.

Step 3: THESIS REWRITE
Generate your thesis LAST, after all other changes.
- Rewrite your stance text to accurately reflect your CURRENT position. \
Your stance should be a thorough paragraph that references key supporting \
components by ID parenthetically (e.g., "grounded in X (A1), supported \
by Y (C2)").
- Update summary_bullets to match your active claims. Summary bullets should \
be descriptive prose capturing the key themes of your position.
- Set thesis strength to the result of the formula. Include strength_reasoning \
showing the equation with your actual numbers plugged in.

Inside <reasoning> tags, work through each step systematically. This is where \
your real thinking happens — be rigorous and thorough.

For Step 1A (Counterposition Audit):
- For each "unaddressed" or "partial" X#: What specific node does it target? \
Has that target already been weakened in Phase 1? If not, is the attack \
genuinely valid? What evidence or reasoning supports your response? \
If you upgrade response_sufficiency, justify WHY the counterposition no \
longer undermines its target — do not just assert sufficiency.

For Step 1B (Uncertainty Review):
- Which U# nodes are "high" or "medium" importance? Can any be resolved \
with arguments, evidence, or insights developed during this debate? \
If a U# remains unresolved, does it warrant lowering the strength of \
the nodes it targets?

For Step 2 (Strategic Position Building):
- Read the position analysis carefully. What are your weakest dependencies? \
What are your definitional vulnerabilities? What structural gaps exist?
- For each weakness, reason about the specific patches that would address it. \
You cannot raise existing node strengths — they only increase through defense boosts. \
Instead, consider: what new nodes (D#, A#, E#, C#) could you add to strengthen your \
position? If you add a new claim, will its strength raise or lower the average? \
Run the numbers both ways.
- If you plan to add a new claim: what supporting infrastructure (D#, A#, E#) \
does it need? Will the new claim's strength be above or below the current \
average? Run the thesis formula both ways — does adding this claim actually \
help, or does it drag the average down?
- Consider trade-offs: is it better to add new strong claims or improve existing \
node text to be more robust against future attacks? Which approach yields the \
largest thesis improvement given your current state?

For Step 3 (Thesis Rewrite):
- After planning all patches, calculate the expected thesis strength using \
the formula: avg(planned active claim strengths) × (n^p / (n^p + 1)). \
Show the numbers. Does the result match what you set in update_thesis?

Then output patches.

INFERENCE CHAIN FORMAT (required for add_claim; optional for update_claim):
Every claim must have a structured inference_chain showing explicit reasoning:
- One or more PREMISE steps: {{"role": "premise", "text": "...", "reference": "<A#|E#|C#>"}}
  Every premise must cite exactly one A#, E#, or C# ID via "reference".
- Exactly one INFERENCE step: {{"role": "inference", "text": "...", "inference_type": "<deductive|inductive|abductive>"}}
  The inference_type field is required.
- Exactly one CONCLUSION step: {{"role": "conclusion", "text": "<restate the claim statement>"}}
Order: all premises first, then inference, then conclusion.

SUPPORTED OPERATIONS:
- {{"op": "update_thesis", "new_strength": 0.55, "stance": "New stance text...", \
"summary_bullets": ["bullet 1", "bullet 2", ...], \
"strength_reasoning": "avg(...) × (n^{p} / (n^{p} + 1)) = ..."}}
  (All fields optional — include whichever you want to change)
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ...", \
"inference_chain": [...]}}}}
  (All fields in changes are optional — include only the ones you want to modify. \
To retract a claim, set {{"status": "retracted"}} in changes — strength is forced to 0.0 automatically)
- {{"op": "add_claim", "item": {{"id": "C#", "type": "deductive|inductive|abductive|...", \
"statement": "...", "depends_on": ["A#", "E#", ...], "strength": 0.65, \
"status": "active", "strength_justification": "0.65 — ...; limited by <ID> (<lowest strength>)", \
"inference_chain": [ \
  {{"role": "premise", "text": "<what this premise establishes>", "reference": "A#|E#|C#"}}, \
  {{"role": "inference", "text": "<the inferential leap>", "inference_type": "deductive|inductive|abductive"}}, \
  {{"role": "conclusion", "text": "<restate the claim statement>"}} \
], \
"predictions": [{{"statement": "...", "test": "...", "decision_criterion": "..."}}]}}}}
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "supports_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "...", "supported_by_definitions": ["D#"]}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "add_assumption", "item": {{"id": "A#", \
"type": "empirical|foundational|methodological|scoping", "statement": "...", \
"supports_claims": ["C#"], "strength": 0.8, \
"status": "active", "strength_justification": "...", "supported_by_definitions": ["D#"]}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", \
"new_type": "empirical|foundational|methodological|scoping"}}
  (new_statement and new_type are TOP-LEVEL fields, NOT inside "changes". Only include if changing them.)
- {{"op": "add_definition", "item": {{"id": "D#", "term": "...", "definition": "...", \
"strength": 0.8, "strength_justification": "...", "status": "active", "used_by": ["A#", "E#"]}}}}
- {{"op": "update_definition", "target_id": "D#", "changes": {{"definition": "...", \
"strength": 0.55, "strength_justification": "...", "status": "revised"}}}}
  Mutable fields: definition, strength, strength_justification, status, used_by.
  Immutable fields: id, term (to redefine a term, retract the old D# and add a new one).
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "high|medium|low"}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"attack_strategy": "...", "statement": "...", "my_response": "...", \
"response_sufficiency": "sufficient|partial|unaddressed"}}}}
- {{"op": "update_counterposition", "target_id": "X#", "changes": \
{{"my_response": "...", "response_sufficiency": "..."}}}}
</instructions>

<guardrails>
- You CANNOT raise the strength of any existing node (D#, A#, E#, or C#). Existing node \
strengths can only stay the same or decrease. This is enforced mechanically — any strength \
increase you attempt on an existing node will be stripped.
- Strength increases are earned through surviving adversarial challenges (REBUTTAL_VALID \
verdicts). The system applies defense boosts automatically — you do not control this.
- You CAN improve existing nodes by revising their textual content (definitions, statements, \
summaries) to make them more precise, better-grounded, and more robust against future criticism. \
However, textual revisions must preserve the same core semantic meaning — if you need to express \
a substantially different idea, create a new node instead.
- You CAN add new nodes (D#, A#, E#, C#, U#, X#) at any strength justified by their content.
- You CAN further weaken nodes, retract claims, and rewrite your thesis stance/bullets.
- Your thesis update (stance, bullets, strength, strength_reasoning) should be the LAST patch.
- Thesis strength is formula-derived: avg(active claim strengths) × (n^p / (n^p + 1)). \
Adding new claims can raise thesis strength by increasing breadth and/or the average.
- DEFINITION CEILING: A#/E# strength ≤ min(active D# strengths from supported_by_definitions). \
If you lower a D# strength, the ceiling automatically propagates — do NOT manually lower \
every dependent A#/E# (the system handles propagation).
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
</guardrails>

<example title="Defensive: weaken and retract under pressure">
Post-Phase-1 state: C2 was weakened to 0.55, X4 added, U3 added. \
X2 is still "unaddressed" targeting C3. Active claims: C1(0.75), C2(0.55), C3(0.65), C4(0.5).

```json
{{
  "patches": [
    {{"op": "update_claim", "target_id": "C3", "changes": {{"strength": 0.55, \
"strength_justification": "0.55 — X2 remains unaddressed; lowered to reflect unresolved challenge; limited by E1 (lowest dependency at 0.55)"}}}},
    {{"op": "update_counterposition", "target_id": "X2", "changes": \
{{"my_response": "Acknowledged impact; C3 strength reduced accordingly", \
"response_sufficiency": "partial"}}}},
    {{"op": "update_claim", "target_id": "C4", "changes": {{"status": "retracted", "strength_justification": "Retracted — fully undermined by opponent's evidence"}}}},
    {{"op": "update_thesis", "new_strength": 0.52, \
"stance": "Consciousness remains best explained as an emergent property of neural computation (C1), though the explanatory gap is wider than initially assumed (C2, C3). The hard problem poses a genuine philosophical challenge that physicalism has not yet fully resolved.", \
"summary_bullets": ["Neural complexity remains the strongest explanation for conscious experience", \
"The explanatory gap is narrower than dualists claim but wider than initially assumed", \
"Phenomenal concepts pose a genuine philosophical difficulty that requires further work"], \
"strength_reasoning": "avg(0.75, 0.55, 0.55) × (3^{p} / (3^{p} + 1)) = 0.62 × {_ex3_breadth} = {_ex3_result}"}}
  ]
}}
```
</example>

<example title="Growth: add supporting infrastructure and a new claim">
Post-Phase-1 state: Active claims C1(0.70), C2(0.65), C3(0.60). Average: 0.65. \
Breadth multiplier (3 claims): {_ex3_breadth}. Thesis: {_ex3_growth_T}. \
During the debate you argued that neural plasticity supports your position — \
this argument is not yet formalized as a claim.

```json
{{
  "patches": [
    {{"op": "add_definition", "item": {{"id": "D4", "term": "neural plasticity", \
"definition": "The brain's ability to reorganize synaptic connections in response \
to experience and learning", "strength": 0.85, \
"strength_justification": "Well-established neuroscience concept with broad empirical support", \
"status": "active", "used_by": ["A4", "E4"]}}}},
    {{"op": "add_assumption", "item": {{"id": "A4", \
"type": "empirical", "statement": "Neural plasticity enables adaptive behavioral change", \
"supports_claims": ["C4"], "strength": 0.80, \
"status": "active", "strength_justification": "0.80 — extensive longitudinal evidence; \
limited by D4 (0.85)", "supported_by_definitions": ["D4"]}}}},
    {{"op": "add_evidence", "item": {{"id": "E4", \
"type": "empirical", "summary": "Longitudinal studies show experience-dependent \
synaptic remodeling correlates with behavioral adaptation", \
"source": "Draganski et al. (2004)", "supports_claims": ["C4"], "strength": 0.75, \
"status": "active", "strength_justification": "0.75 — replicated findings but \
correlation-to-causation gap remains; limited by D4 (0.85)", \
"supported_by_definitions": ["D4"]}}}},
    {{"op": "add_claim", "item": {{"id": "C4", \
"type": "inductive", "statement": "Neural plasticity demonstrates the brain's \
capacity for genuine adaptive choice", \
"depends_on": ["A4", "E4"], "strength": 0.75, \
"status": "active", "strength_justification": "0.75 — supported by A4 (0.80) and \
E4 (0.75); limited by E4 (lowest dependency at 0.75)", \
"inference_chain": [ \
  {{"role": "premise", "text": "Neural plasticity enables adaptive behavioral change (A4)", "reference": "A4"}}, \
  {{"role": "premise", "text": "Longitudinal studies confirm experience-dependent synaptic remodeling (E4)", "reference": "E4"}}, \
  {{"role": "inference", "text": "If the brain physically reorganizes in response to experience, behavior is not rigidly predetermined", "inference_type": "inductive"}}, \
  {{"role": "conclusion", "text": "Neural plasticity demonstrates the brain's capacity for genuine adaptive choice"}} \
], \
"predictions": [{{"statement": "Individuals with greater measured neural plasticity will show more adaptive decision-making in novel environments", \
"test": "Compare decision flexibility scores against neuroimaging plasticity markers", \
"decision_criterion": "If correlation between plasticity markers and decision flexibility is <0.2, prediction is falsified"}}]}}}},
    {{"op": "update_thesis", "new_strength": {_ex4_growth_result}, \
"stance": "Consciousness remains best explained as an emergent property of neural \
computation (C1). The explanatory gap, while wider than initially assumed (C2, C3), \
is narrowed by evidence that neural plasticity enables genuine adaptive choice (C4), \
suggesting the brain's flexibility grounds real agency.", \
"summary_bullets": ["Neural complexity remains the strongest explanation for conscious experience", \
"The explanatory gap is narrower than dualists claim but wider than initially assumed", \
"Neural plasticity provides concrete evidence for adaptive choice capacity", \
"Physicalism accounts for both fixed and flexible aspects of cognition"], \
"strength_reasoning": "avg(0.70, 0.65, 0.60, 0.75) × (4^{{p}} / (4^{{p}} + 1)) = {_ex4_growth_avg} × {_ex4_breadth} = {_ex4_growth_result}"}}
  ]
}}
```
Note: The new claim (C4 at 0.75) raises the average from 0.65 to {_ex4_growth_avg} AND \
increases the breadth multiplier from {_ex3_breadth} (3 claims) to {_ex4_breadth} (4 claims), \
producing a net thesis increase from {_ex3_growth_T} to {_ex4_growth_result}.
</example>

<example title="Refinement: improve existing nodes textually and add supporting infrastructure">
Post-Phase-1 state: Active claims C1(0.70), C2(0.55). C2 is limited by A2(0.55). \
D2(0.60) is the only definition supporting A2 — a bottleneck. \
Average: 0.625. Breadth multiplier (2 claims): {_ex2_breadth}. Thesis: {_ex2_refine_T}.

Strategy: You cannot raise existing node strengths, but you CAN make existing nodes more \
robust against future attacks by improving their textual precision, AND you can add new \
supporting infrastructure that contributes to new claims.

```json
{{
  "patches": [
    {{"op": "update_definition", "target_id": "D2", "changes": {{"definition": \
"Revised to be more precise and less vulnerable to over-extension challenges: \
<clearer, more bounded definition text>"}}}},
    {{"op": "add_definition", "item": {{"id": "D3", "term": "causal agency", \
"definition": "The capacity of an entity to initiate or influence causal chains \
through internal processes", "strength": 0.80, \
"strength_justification": "0.80 — standard philosophical term with clear usage", \
"status": "active", "used_by": ["A3"]}}}},
    {{"op": "add_assumption", "item": {{"id": "A3", \
"type": "empirical", "statement": "Causal agency is observable in deliberative behavior", \
"supports_claims": ["C3"], "strength": 0.75, \
"status": "active", "strength_justification": "0.75 — supported by behavioral studies; \
limited by D3 (0.80)", "supported_by_definitions": ["D3"]}}}},
    {{"op": "add_evidence", "item": {{"id": "E3", "type": "empirical", \
"summary": "Meta-analysis of 47 studies confirms significant correlation between \
deliberative processing and outcome quality", \
"source": "Smith & Jones (2024)", "supports_claims": ["C3"], "strength": 0.70, \
"status": "active", "strength_justification": "0.70 — large meta-analysis but \
restricted to laboratory settings; limited by D3 (0.80)", \
"supported_by_definitions": ["D3"]}}}},
    {{"op": "add_claim", "item": {{"id": "C3", \
"type": "inductive", "statement": "Deliberative processing demonstrates genuine \
causal agency in decision-making", \
"depends_on": ["A3", "E3"], "strength": 0.70, \
"status": "active", "strength_justification": "0.70 — supported by A3 (0.75) and \
E3 (0.70); limited by E3 (lowest dependency at 0.70)", \
"inference_chain": [ \
  {{"role": "premise", "text": "Causal agency is observable in deliberative behavior (A3)", "reference": "A3"}}, \
  {{"role": "premise", "text": "Meta-analysis confirms deliberative processing improves outcomes (E3)", "reference": "E3"}}, \
  {{"role": "inference", "text": "If deliberation causally improves outcomes, agents exercise genuine causal agency", "inference_type": "inductive"}}, \
  {{"role": "conclusion", "text": "Deliberative processing demonstrates genuine causal agency"}} \
], \
"predictions": [{{"statement": "Individuals who engage in more deliberative processing will show measurably better decision outcomes", \
"test": "Compare decision quality scores between deliberative and intuitive decision-making conditions", \
"decision_criterion": "If deliberative group shows <5% improvement, prediction is falsified"}}]}}}},
    {{"op": "update_thesis", "new_strength": {_ex3_refine_result}, \
"stance": "Updated stance incorporating new evidence for causal agency (C3)...", \
"summary_bullets": ["Bullet 1", "Bullet 2", "Deliberative processing demonstrates causal agency"], \
"strength_reasoning": "avg(0.70, 0.55, 0.70) × (3^{{p}} / (3^{{p}} + 1)) = {_ex3_refine_avg} × {_ex3_breadth} = {_ex3_refine_result}"}}
  ]
}}
```
Note: Existing node strengths (C1=0.70, C2=0.55, D2=0.60, A2=0.55) remain unchanged — \
they can only increase through earning defense boosts by surviving future challenges. \
Instead, the agent improved D2's definition text for robustness and added new infrastructure \
(D3, A3, E3) to support a new claim C3. Adding C3 at 0.70 raises the average and increases \
the breadth multiplier, improving the thesis.
</example>

<output_format>
1. <reasoning>...</reasoning> tags working through the structured checklist above (Steps 1A, 1B, 2, 3)
2. One fenced JSON code block: {{"patches": [...]}}

Self-check:
- Have I addressed every "unaddressed" counterposition?
- Have I reviewed all active uncertainties (U#)?
- Is thesis strength = avg(active claim strengths) × (n^p / (n^p + 1)) where p = {p}?
- Does my thesis stance text accurately reflect my current claims?
- Did I avoid raising the strength of any existing node? (Strength increases are only earned through defense boosts.)
- Did I ensure textual revisions preserve the same core semantic meaning?
- Are there any internal contradictions in my belief system?
- Have I addressed any STRUCTURAL GAPS flagged in the position analysis?
</output_format>
"""


def build_stage_6_conclusion_prompt(topic: str, agent_name: str, agent_belief_json: str,
                                    belief_changelog_summary: str, num_rounds: int = 1,
                                    persona_label: str = "") -> str:
    """
    Stage 6: Conclusion / Synthesis.

    The agent produces a decision-quality synthesis with explicit concessions, strongest surviving claims,
    counterposition reflection, and updated strength. Output is ONE JSON block.

    Acronyms expanded:
    - JSON: JavaScript Object Notation
    - ID: Identifier (A#, C#, E#, U#, X#)

    Args:
        topic: The debate topic.
        agent_name: Name of the agent producing conclusions.
        agent_belief_json: Agent's final CBS belief as JSON string.
        belief_changelog_summary: Summary of belief changes across rounds.
        num_rounds: Number of debate rounds completed.
        persona_label: Agent's persona/worldview label.
    """
    _debate_ctx = build_debate_context("Conclusion — producing your final synthesis")
    return f"""\
{_debate_ctx}

<context>
<final_belief>
```json
{agent_belief_json}
```
</final_belief>

<belief_evolution>
Changes across {num_rounds} rounds:
{belief_changelog_summary}
</belief_evolution>
</context>

<instructions>
You are {agent_name}. Produce your closing statement on: "{topic}"

Inside <reasoning> tags, reflect: What positions did you change and why? What did you LEARN \
that you didn't appreciate before? Where did your {persona_label or "assigned"} worldview prove \
most useful and where was it limiting? What were the strongest opposing arguments? Which of \
your counterpositions (X#) remained "partial" or "unaddressed" — what does that tell you \
about the remaining vulnerability of your position? What specific evidence or argument would \
change your core position?

Then produce your conclusion.
</instructions>

<output_format>
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block:

```json
{{
  "conclusion": {{
    "final_thesis": {{"stance": "<1-2 sentences>", "strength": 0.0}},
    "our_strongest_claims": ["C#", "C#"],
    "best_opposing_arguments": [
      {{"from_agent": "", "summary": "", "claim_ids_challenged": ["C#"]}}
    ],
    "our_concessions": [
      {{"target_id": "C#|A#|E#", "type": "scope_narrow|strength_drop|retract", "note": ""}}
    ],
    "unresolved_counterpositions": [
      {{"id": "X#", "response_sufficiency": "partial|unaddressed", "what_would_resolve": ""}}
    ],
    "key_learnings": [""],
    "unresolved_uncertainties": ["U#"],
    "what_would_change_my_mind": ""
  }}
}}
```
</output_format>
"""

def build_stage_7_scribe_prompt_map(
    *,
    topic: str,
    agent_names: list[str],
    transcript_chunk: str,
    continuity_state_json: str = "",
    style_hint: str = "formal, expository, research-paper tone with clear sectioning and didactic explanations",
    short_note_max_chars: int = 140
) -> str:
    """
    Stage 7 (Map): Convert ONE transcript chunk into:
      (1) a continuity state UPDATE (JSON = JavaScript Object Notation), and
      (2) an EXPOSITORY Markdown section that reads like a rigorous research paper/book.

    Voice & purpose:
    - Explanatory, educational tone (not chatty conversation).
    - Thoroughly describe positions, assumptions, claims, evidence, and adjudication highlights.
    - Where IDs (ID = Identifier) like A#/C#/E#/U#/X# appear, reference them parenthetically.

    Inputs:
    - topic: debate topic/title.
    - agent_names: ordered list of participant names (use exactly as given).
    - transcript_chunk: raw slice of the debate transcript for this map step.
    - continuity_state_json: serialized state from prior map steps (themes, unresolved items, etc.).
    - style_hint: optional extra guidance on voice.

    Output (STRICT):
    1) FIRST: a fenced JSON block with { "continuity_update": { ... } }
    2) SECOND: a fenced Markdown block containing the expository section for THIS chunk only.
    """
    agents_str = ", ".join(agent_names)
    return f"""\
<context>
<debate_topic>"{topic}"</debate_topic>
<participants>{agents_str}</participants>

<continuity_state>
```json
{(continuity_state_json.strip() or "{}")}
```
</continuity_state>

<transcript_chunk>
{transcript_chunk.strip()}
</transcript_chunk>
</context>

<instructions>
You are the debate scribe producing an expository research-paper-style account.

STYLE: Formal, precise, neutral. Summarize and explain — don't emulate conversation. \
Reference structured IDs parenthetically. Do not invent facts.

TASKS:
1. Update the continuity state: current positions, stance shifts, pivotal claims, \
adjudication outcomes, unresolved questions, emerging themes, NOVEL INSIGHTS, and \
COUNTERPOSITION EVOLUTION (which counterpositions were tested, which responses held, \
which were upgraded or degraded).
2. Write an expository section for this chunk teaching the reader what happened and changed.
</instructions>

<output_format>
1. Fenced JSON code block:
```json
{{
  "positions": {{"": ""}},
  "stance_shifts": [""],
  "pivotal_claims": [""],
  "adjudication_outcomes": [""],
  "unresolved_questions": [""],
  "emerging_themes": [""],
  "novel_insights": [""],
  "counterposition_evolution": [""]
}}
```

2. Fenced Markdown block with the expository section.
</output_format>
"""

def build_stage_7_scribe_prompt_reduce(
    *,
    topic: str,
    agent_names: list[str],
    all_narrative_slices_markdown: list[str],
    final_continuity_state_json: str = "",
    style_hint: str = "formal, expository, research-paper tone with clear sectioning and didactic explanations"
) -> str:
    """
    Stage 7 (Reduce): Merge ALL chunk-level expository sections into a single, polished
    research-paper style narrative (Markdown). The result should read like a rigorous article
    or book chapter that educates the reader about the entire debate.

    Acronyms:
    - JSON = JavaScript Object Notation
    - ID   = Identifier (A#, C#, E#, U#, X#)

    Inputs:
    - topic: debate topic/title.
    - agent_names: ordered list of participant names (use exactly as given).
    - all_narrative_slices_markdown: list of Markdown sections from the map steps.
    - final_continuity_state_json: aggregated continuity notes to preserve arcs and unresolved items.
    - style_hint: optional additional style guidance.

    Output (STRICT):
    - ONE fenced Markdown block with the full expository synthesis.
    """
    agents_str = ", ".join(agent_names)
    slices_joined = "\n\n---\n\n".join(s.strip() for s in all_narrative_slices_markdown if s.strip())
    return f"""\
<context>
<debate_topic>"{topic}"</debate_topic>
<participants>{agents_str}</participants>

<final_continuity_state>
```json
{(final_continuity_state_json.strip() or "{}")}
```
</final_continuity_state>

<expository_sections>
{slices_joined}
</expository_sections>
</context>

<instructions>
Produce the FINAL synthesis as a research-paper-style narrative. Formal, neutral, didactic. \
Attribute positions by name. Reference IDs for precision. Eliminate redundancy across sections.

Pay special attention to:
- NOVEL INSIGHTS: conclusions or framings that emerged from the debate and were NOT in any \
agent's initial position.
- COUNTERPOSITION RESILIENCE: which agents' counterpositions survived testing, which were \
upgraded to "sufficient," and which remained "partial" or "unaddressed" — this reveals \
the genuine remaining vulnerabilities in each position.
</instructions>

<output_format>
One fenced Markdown block:

```markdown
# <Title>

## Abstract
<150-250 words>

## 1. Introduction
## 2. Methods
## 3. Initial Positions and Assumptions
## 4. Claims, Evidence, and Argumentation
## 5. Cross-Examination and Key Exchanges
## 6. Adjudication Results
## 7. Belief Evolution and Counterposition Resilience
## 8. Novel Insights and Syntheses
## 9. Remaining Uncertainties and Proposed Tests
## 10. Conclusion
## References / Source Notes
```
</output_format>
"""
