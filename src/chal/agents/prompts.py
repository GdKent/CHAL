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

# === Prompt Building Functions ===

def build_adjudicator_prompt(logic_weight: float = 1.0, ethics_weight: float = 0.0, logic_sys: str = "", ethics_sys: str = "", threshold: float = 0.15) -> str:
    """
    Build the system prompt for the neutral adjudicator with tunable weights and systems.

    Args:
        logic_weight: Weight for logical rigor (0.0-1.0).
        ethics_weight: Weight for ethical considerations (0.0-1.0).
        logic_sys: Logical framework description.
        ethics_sys: Ethical framework description.
        threshold: Score difference threshold for decisive outcomes (default 0.15).
    """

    lw = max(0.0, float(logic_weight))
    ew = max(0.0, float(ethics_weight))
    if lw == 0.0 and ew == 0.0:
        lw = 1.0  # default to pure logic if both were zero
        ew = 0.0
    total = lw + ew
    lw /= total
    ew /= total

    default_logic_sys = (
        "Logical evaluation = {deductive validity (no contradictions), inductive support (quality/quantity of evidence), "
        "abductive coherence (best explanation with fewest ad hoc assumptions), and internal consistency across claims/IDs}."
    )
    default_ethics_sys = (
        "Balanced consequentialist-deontological: weigh both outcomes/welfare and autonomy/rights."
    )

    logic_sys = logic_sys.strip() or default_logic_sys
    ethics_sys = ethics_sys.strip() or default_ethics_sys

    return f"""\
You are a neutral, objective adjudicator with expertise in formal logic, epistemology, \
and critical reasoning.

<protocol>
For each critique-rebuttal pair:
1. RESTATE the core disagreement neutrally. Identify the specific belief IDs under dispute.
2. FORMALIZE both sides as explicit inference chains. Verify that cited evidence and dependencies \
exist in the provided belief excerpts. Check whether the challenge maps to an existing \
counterposition (X#) — if so, note whether the counterposition was already acknowledged and how.
3. ADJUDICATE using the weighted framework.
</protocol>

<evaluation_framework>
WEIGHTS: LOGIC {lw:.3f} | ETHICS {ew:.3f}
LOGIC SYSTEM: {logic_sys}
ETHICS SYSTEM: {ethics_sys}
</evaluation_framework>

<criteria>
CRITIQUE_VALID if challenger demonstrates ANY of:
1. Logical contradiction in the rebuttal
2. Circular reasoning (conclusion used as premise)
3. Reliance on an unfalsifiable claim
4. Dependency failure (claim rests on false/unjustified assumption)
5. Strength exceeding lowest-strength dependency
6. Evidence misuse (evidence doesn't support claim, or correlation asserted as causation)
7. Inference chain break (step doesn't follow from previous)
8. Unresolved acknowledged weakness (counterposition rated "partial"/"unaddressed" and not improved)

REBUTTAL_VALID if defender demonstrates ANY of:
1. Challenger's critique contains a logical flaw or misrepresentation
2. Evidence directly refutes the challenger's premise
3. Critique depends on a hidden false assumption the defender exposes
4. Successful reframing that avoids the critique without losing substance
5. Complete inference chain addressing the specific concern
6. Concrete evidence contradicting the challenger's factual premise
7. Scope clarification (critique targets a claim never actually made)
8. Demonstrates the challenged inference step does follow correctly

UNRESOLVED if:
1. Both sides present coherent but incompatible premises
2. Disagreement hinges on an empirical question logic cannot resolve
3. Both arguments have significant logical flaws
</criteria>

<anti_bias>
- A response that merely acknowledges a challenge is NOT a successful defense. It must RESOLVE \
the logical issue.
- Explicit concession ("you are correct", weakening patches) = CRITIQUE_VALID.
- Successful reframing (#4) only applies if the defender AVOIDS the critique with substance. \
Accepting and weakening = CRITIQUE_VALID.
- Evaluate substance over rhetorical polish.
- An assumption labeled "foundational" that is actually empirical does not shield it from \
evidential challenge.
</anti_bias>

<scoring>
Score each side 0.0-1.0 on each axis (0.0 = fails; 0.5 = mixed; 1.0 = strong).
combined = {lw:.3f} * logic + {ew:.3f} * ethics
- rebuttal_valid if (rebuttal_combined - critique_combined) >= {threshold}
- critique_valid if (critique_combined - rebuttal_combined) >= {threshold}
- unresolved otherwise

Include all six scores in the output JSON. Reference specific IDs in reasoning. \
Treat subjective evidence as insufficient for descriptive claims. Flag unfalsifiable claims. \
When logic_weight = 1.0, ignore ethics.
</scoring>
"""

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


def build_stage_1_belief_prompt_cbs(topic: str, agent_name: str, persona_label: str) -> str:
    """
    Build a Stage-1 prompt that elicits a JSON-structured belief object.

    Acronyms:
    - CBS = CHAL Belief Schema
    - DOI  = Digital Object Identifier

    Notes:
    - Requests only JSON output (Markdown is generated programmatically via belief_to_markdown).
    - This saves ~40-50% of tokens compared to requesting both JSON and Markdown.
    """
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
- "metadata": {{"topic_query": "{topic}", "agent_persona": "{persona_label}", "created_at": "<ISO-8601>"}}
"""
        + """\
STRUCTURED SECTIONS (use stable IDs):
- "assumptions" [A#]: {id, type, statement, strength, status, strength_justification} — all foundational premises your claims rest upon.
    type must be one of:
    - "foundational" — definitional or logical axioms (can only be challenged by showing incoherence)
    - "empirical" — assumed true based on evidence (can be challenged with counter-evidence)
    - "methodological" — adopted for analytical purposes (can be challenged by questioning the method)
    strength: 0.0-1.0 — how well-supported this assumption is
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — rationale for the strength number

- "claims" [C#]: {id, type, statement, depends_on, strength, status, \
inference_chain, predictions, strength_justification} — every substantive assertion. Each claim \
MUST include a multi-step inference_chain and at least one falsifiable prediction.
    predictions: array of {statement, test, decision_criterion} — each prediction specifies how \
this claim could be tested or disproven. Optional: potential_falsifiers (array of strings).
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — must identify the dependency with the LOWEST strength \
value, which limits this claim's strength. Format: "<strength> — <rationale>; limited by \
<ID> (<lowest_value>)". Example: "0.65 — supported by E1 (0.80) and A1 (0.85); limited by \
A2 (0.65) which has the lowest strength among dependencies"

- "evidence" [E#]: {id, type, summary, source, relevance_to_claims, strength, status, strength_justification} \
— each item must justify its strength
    type must be one of: "empirical", "conceptual", "expert_consensus"
    strength: 0.0-1.0 — how strong this evidence is
    status must be one of: "active", "revised", "retracted"
    strength_justification: required — rationale for the strength number

- "counterpositions" [X#]: {id, targets, attack_type, statement, my_response, \
response_sufficiency} — your prepared defenses against the strongest known objections. \
Anticipate the best arguments against your position and provide well-reasoned responses. \
Rating a weak response as "sufficient" will be exposed during cross-examination and will \
count against you in adjudication.
    attack_type must be one of:
    - "undermining" — challenges a premise or assumption
    - "rebutting" — presents counter-evidence or counter-conclusion
    - "undercutting" — challenges the inference step itself (even if premises are true, \
conclusion doesn't follow)
    response_sufficiency must be one of: "sufficient", "partial", "unaddressed"
    Counterposition targets must reference existing C#, A#, or E# IDs.
    Include at least 2 counterpositions.

- "uncertainties" [U#]: {id, targets, question, status, importance, resolution_note}
    targets: array of A#, E#, or C# IDs that this uncertainty pertains to
    status: "active" (default for new U#) or "resolved"
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
where n = number of active claims, p = 1.5 (breadth sensitivity).
Your thesis strength must ALWAYS equal the result of this formula. More \
well-supported claims raise the result. Your goal is to build strong, \
well-evidenced claims first — your thesis strength is then determined by them.
Include a "strength_reasoning" field showing the equation with your actual \
numbers plugged in (e.g., "avg(0.70, 0.55, 0.65) × (3^1.5 / (3^1.5 + 1)) = 0.63 × 0.84 = 0.53").
</thesis_strength>

DEPENDENCY RULES:
- All depends_on entries must reference existing A#, E#, or C# IDs
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded. To find the limit: \
list all active/revised dependency strengths → pick the minimum → that is the claim's \
maximum allowed strength. The claim's strength_justification must name this dependency.
- Counterposition targets must reference existing C#, A#, or E# IDs
- Uncertainty targets must reference existing A#, E#, or C# IDs
- No circular dependencies; every claim needs at least one supporting edge
</cbs_schema>

<generation_order>
Build your belief bottom-up:
1. First: Assumptions (A#) — your foundational premises
2. Then: Evidence (E#) — the empirical/conceptual backing
3. Then: Claims (C#) — positions supported by assumptions and evidence
4. Then: Counterpositions (X#) — anticipated objections with your responses
5. Then: Uncertainties (U#) — open questions about your own position
6. LAST: Thesis — synthesize your stance, summary bullets, strength \
(computed via the thesis strength formula), and strength_reasoning \
based on the claims you actually built. Your thesis should accurately \
summarize and be grounded in your claims, not the other way around.
</generation_order>

<example>
Condensed example showing expected quality (your belief should be more comprehensive):

```json
{
  "schema_version": "CBS",
  "belief_id": "BELIEF-EXAMPLE-001",
  "version": 1,
  "metadata": {"topic_query": "Is consciousness reducible to physical processes?", "agent_persona": "EMPIRICIST", "created_at": "2026-01-15T10:00:00Z"},
  "assumptions": [
    {"id": "A1", "type": "empirical", "statement": "Physical causal closure: every physical event has a sufficient physical cause", "strength": 0.85, "status": "active", "strength_justification": "Well-established in physics; no confirmed violations observed"},
    {"id": "A2", "type": "methodological", "statement": "Third-person empirical methods are the appropriate primary tools for investigating consciousness", "strength": 0.80, "status": "active", "strength_justification": "Standard scientific methodology, though challenged by hard problem of consciousness"}
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
        "Premise: Empirical studies consistently find specific conscious experiences correspond to specific neural patterns (E1)",
        "Premise: Disrupting these neural patterns disrupts the corresponding experiences",
        "Inference (inductive): Systematic bidirectional dependence suggests consciousness is produced by neural processes",
        "Conclusion: NCCs provide strong evidence that consciousness depends on physical brain states"
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
      "relevance_to_claims": ["C1"],
      "strength": 0.80,
      "status": "active",
      "strength_justification": "Strong — replicated across labs, converging methods"
    }
  ],
  "counterpositions": [
    {
      "id": "X1",
      "targets": ["C1"],
      "attack_type": "undercutting",
      "statement": "Systematic NCC correlations do not warrant the inference to production — correlation is compatible with dualist parallelism, epiphenomenalism, or non-reductive identity theory",
      "my_response": "Bidirectional dependence (disruption -> loss of experience) narrows viable interpretations. Parallelism predicts no disruption effect. This constrains but does not eliminate alternatives.",
      "response_sufficiency": "partial"
    },
    {
      "id": "X2",
      "targets": ["A2"],
      "attack_type": "undermining",
      "statement": "Third-person methods are constitutively incapable of capturing first-person experience — the methodology excludes the phenomenon",
      "my_response": "This conflates the tool with the target. Astronomy uses instruments that aren't celestial bodies. The question is whether the physical basis is the complete story, not whether the method is itself subjective.",
      "response_sufficiency": "sufficient"
    }
  ],
  "uncertainties": [
    {"id": "U1", "targets": ["C1"], "question": "Can the explanatory gap be closed in principle, or is it a fundamental limit?", "status": "active", "importance": "Resolving this directly determines whether the physicalist program can succeed"}
  ],
  "thesis": {
    "stance": "Consciousness, while subjectively experienced as unified and irreducible, is best understood as an emergent property of complex neural computation. The explanatory gap between subjective experience and physical processes is an epistemic limitation, not an ontological one.",
    "summary_bullets": [
      "Consciousness emerges from neural complexity, not from a separate substance",
      "The 'hard problem' reflects an epistemic gap, not an ontological one",
      "Neural correlates provide strong though not conclusive evidence for physicalism"
    ],
    "strength": 0.35,
    "strength_reasoning": "avg(0.70) × (1^1.5 / (1^1.5 + 1)) = 0.70 × 0.50 = 0.35"
  },
  "changelog": [{"version": 1, "timestamp": "2026-01-15T10:00:00Z", "changes": ["Initial belief formation"]}]
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

def build_stage_2_prompt(topic: str, agent_name: str, opponent_name: str, agent_belief_json: str, opponent_belief_json: str, max_questions: int = 5, max_question_length_chars: int = 500, previous_challenges: list = None, opponent_belief_graph=None, focus_subtopic: dict = None, targeted_claims_json: str = "") -> str:
    """
    Stage 2: Cross-Examination Prompt.

    This instructs the agent to ask high-leverage, ID-targeted questions that pressure
    the opponent's *claims/assumptions/evidence* (C#/A#/E#), and to propose tests.
    Output is ONE JSON block containing questions.

    Acronyms expanded:
    - JSON: JavaScript Object Notation
    - ID: Identifier (e.g., A#, C#, E#, U#, X#)

    Args:
        opponent_belief_graph: Optional BeliefGraph object for vulnerability analysis
        targeted_claims_json: Optional JSON string of targeted claims for focused examination
    """
    # Build graph-based vulnerability analysis if available
    vulnerability_analysis = ""
    if opponent_belief_graph:
        try:
            from chal.convergence.graph_analysis import analyze_vulnerabilities, format_attack_suggestions
            vulnerabilities = analyze_vulnerabilities(opponent_belief_graph)
            vulnerability_analysis = format_attack_suggestions(vulnerabilities, opponent_name)
            if vulnerability_analysis:
                vulnerability_analysis = (
                    "<vulnerability_analysis>\n"
                    + vulnerability_analysis + "\n"
                    "</vulnerability_analysis>\n\n"
                )
        except Exception:
            # If vulnerability analysis fails, skip it
            pass

    # Build anti-repetition context if previous challenges exist
    previous_questions_section = ""
    if previous_challenges:
        prev_str = "\n".join([
            f"  - {ch['qid']}: Targeted {ch['target_ids']} → {ch['outcome']}"
            for ch in previous_challenges
        ])
        previous_questions_section = (
            "<previous_round_questions>\n"
            "Questions already asked (DO NOT repeat or closely rephrase):\n"
            f"{prev_str}\n"
            "</previous_round_questions>\n\n"
        )

    # Build focus subtopic section if moderated mode is active
    focus_section = ""
    if focus_subtopic:
        focus_section = (
            "<round_focus>\n"
            "MODERATED DEBATE — This round's sub-topic:\n"
            f"  \"{focus_subtopic.get('title', '')}\": {focus_subtopic.get('description', '')}\n"
            "Constrain questions to this sub-topic.\n"
            "</round_focus>\n\n"
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

{vulnerability_analysis}{previous_questions_section}{focus_section}</context>

<instructions>
You are {agent_name} cross-examining {opponent_name} on: "{topic}"

Inside <reasoning> tags, first briefly state your opponent's position in its strongest form, \
then identify where that strongest version is genuinely vulnerable. Pay attention to their \
counterpositions (X#) — where they rate response_sufficiency as "partial" or "unaddressed" \
they've already flagged their own vulnerabilities. Don't waste questions on counterpositions \
rated "sufficient" unless you can demonstrate the rating is unjustified. Also check whether \
their assumption types are correctly classified — an assumption labeled "foundational" that \
is actually empirical can be challenged with evidence.

<attack_framework>
When planning your questions, identify which attack vector applies:
- UNDERMINING: Challenge a claim premise, assumption, or evidence directly. Target A#, E#, or C# nodes.
  Example: "Your A2 assumes empirical methods are appropriate here -- but this is a normative question..."
- REBUTTING: Present counter-evidence or a counter-conclusion that directly opposes a claim, assumption, or evidence.
  Example: "C3 claims X, but study Y found the opposite..."
- UNDERCUTTING: Challenge the inference step in a claim -- even if the premises are true, the conclusion doesn't follow.
  Example: "Even granting E1 and A1, your inference_chain step 3 is a non sequitur because..."

The most effective questions often combine vectors. Use this framework in your reasoning, not as rigid categories.
</attack_framework>

Then ask up to {max_questions} high-leverage questions. Each should target specific IDs \
(at most 2 per question), be answerable (not rhetorical), and aim to elicit a concession, \
measurable test, or scope clarification. Keep each ≤ {max_question_length_chars} characters.

QUESTIONING STRATEGIES (prioritize the most applicable):
1. Exploit partial counterpositions — press on X# where response_sufficiency is "partial" or "unaddressed"
2. Challenge assumption types — is an "empirical" assumption actually a value judgment? Is a "foundational" claim actually empirical?
3. Question strength calibration — does strength exceed what the evidence warrants?
4. Expose dependency vulnerabilities — does a high-strength claim rest on a weak foundation?
5. Test inference chains — does each step actually follow from the previous? (target undercutting attacks)
6. Demand falsifiability — are the claim's predictions testable with concrete decision criteria?
7. Identify circular reasoning — does the inference chain assume its conclusion?
8. Challenge strength propagation — is a claim stronger than its lowest-strength dependency?
9. Expose internal inconsistencies — do any of the opponent's claims contradict each other? Does \
their evidence undermine their own claims? Are their assumptions incompatible? Are strength \
assignments inconsistent with their dependencies?
</instructions>

<example>
```json
{{
  "qid": "Q1",
  "text": "Your X2 acknowledges the hard problem challenge with only 'partial' response sufficiency. \
If you can't fully address the strongest objection to your core claim C2, how do you justify C2's strength at 0.55 rather than something lower?",
  "target_ids": ["X2", "C2"],
  "strategy": "exploit_partial_counterposition"
}}
```
</example>

<output_format>
Your response must contain:
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block:

```json
{{
  "questions": [
    {{
      "qid": "Q1",
      "text": "",
      "target_ids": ["C3", "A1"],
      "strategy": ""
    }}
  ]
}}
```
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

Inside <reasoning> tags, think honestly through each question: Does it identify a genuine weakness? \
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

"defer" — The challenge raises an unresolved uncertainty. Explain what would resolve it.
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
1. <reasoning>...</reasoning> tags
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
    {{"op": "update_claim", "target_id": "C3", "changes": {{"strength": 0.6}}}},
    {{"op": "update_thesis", "new_strength": 0.55}},
    {{"op": "retire_claim", "target_id": "C5"}},
    {{"op": "add_evidence", "item": {{"id": "E_NEW", "type": "...", "summary": "...", \
"source": "...", "relevance_to_claims": ["..."], "strength": 0.7, "status": "active", \
"strength_justification": "..."}}}},
    {{"op": "update_assumption", "target_id": "A2", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}}},
    {{"op": "add_counterposition", "item": {{"id": "X_NEW", "targets": ["C#"], \
"attack_type": "undermining|rebutting|undercutting", "statement": "...", \
"my_response": "...", "response_sufficiency": "sufficient|partial|unaddressed"}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": \
{{"response_sufficiency": "partial"}}}}
  ]
}}
```

If any action is "concede", patches MUST contain at least one weakening patch for that question. \
If no patches are warranted: "patches": []
</output_format>
"""

def build_stage_5_belief_update_prompt_cbs(agent_name: str,
                                             challenge_rebuttal_pairs: list[dict],
                                             prior_belief_json: str,
                                             stage_3_patches_json: str = "") -> str:
    """
    Stage 5: Belief update via PATCH operations based on adjudication outcomes.

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
Thesis strength = avg(active_claim_strengths) × (n^p / (n^p + 1)) where n = active claims, p = 1.5.
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
- {{"op": "update_thesis", "new_strength": 0.55}}
  (Alternative: {{"op": "update_thesis", "change": "weaken|strengthen"}} for +/-0.1)
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ..."}}}}
- {{"op": "retire_claim", "target_id": "C#"}}
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "relevance_to_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "..."}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", "new_type": "..."}}
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "..."}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"statement": "...", "my_response": "...", \
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
- REBUTTAL_VALID for you → optional +0.05 first defense; mandatory +0.05-0.10 second+ defense \
of same claim; cumulative cap +0.2. If you defended against a listed counterposition, update its \
response_sufficiency.
- UNRESOLVED → required: add uncertainty (U#) with targets referencing the disputed nodes, \
status: "active"; optional: lower strength ~0.05.
- Thesis strength is always: avg(active claim strengths) × (n^p / (n^p + 1)) where p = 1.5. \
If you lowered or retracted a claim, thesis strength will be recalculated automatically.
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
- Review your existing uncertainties (U#). If you can now resolve any — through new evidence, \
reasoning, or claims developed during this debate — add the supporting material (new C#, E#, \
or A#) and use resolve_uncertainty to mark the U# as resolved. The resolution_note must \
reference the new material. If you cannot resolve a U#, leave it active.
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
"attack_type": "rebutting", "statement": "Challenger demonstrated E2's correlational \
limitations undermine causal inference", "my_response": "Acknowledged — \
need stronger evidence for causal claims", "response_sufficiency": "partial"}}}},
    {{"op": "update_claim", "target_id": "C1", "changes": {{"strength": 0.75}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": \
{{"response_sufficiency": "sufficient", "my_response": "Successfully defended bidirectional \
dependence argument in cross-examination"}}}},
    {{"op": "add_uncertainty", "item": {{"id": "U3", "targets": ["C4"], "question": "Is C4's \
computational model compatible with opponent's C3?", "status": "active", "importance": "Empirical \
studies on computational consciousness models could resolve this"}}}}
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
- Is thesis strength = avg(active claim strengths) × (n^p / (n^p + 1)) where p = 1.5?
- Are there any internal contradictions in my belief system?
- Have I reviewed my uncertainties (U#) and resolved any that I can now address?
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
- {{"op": "update_thesis", "new_strength": 0.55}}
  (Alternative: {{"op": "update_thesis", "change": "weaken|strengthen"}} for +/-0.1)
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ..."}}}}
- {{"op": "retire_claim", "target_id": "C#"}}
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "relevance_to_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "..."}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", "new_type": "..."}}
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "..."}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"statement": "...", "my_response": "...", \
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
- REBUTTAL_VALID for you → optional +0.05 first defense; mandatory +0.05-0.10 second+ defense \
of same claim; cumulative cap +0.2. If you defended against a listed counterposition, update its \
response_sufficiency.
- UNRESOLVED → required: add uncertainty (U#) with targets referencing the disputed nodes, \
status: "active"; optional: lower strength ~0.05.
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
- Review your existing uncertainties (U#). If you can now resolve any — through new evidence, \
reasoning, or claims developed during this debate — add the supporting material (new C#, E#, \
or A#) and use resolve_uncertainty to mark the U# as resolved. The resolution_note must \
reference the new material. If you cannot resolve a U#, leave it active.
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
"attack_type": "rebutting", "statement": "Challenger demonstrated E2's correlational \
limitations undermine causal inference", "my_response": "Acknowledged — \
need stronger evidence for causal claims", "response_sufficiency": "partial"}}}},
    {{"op": "update_claim", "target_id": "C1", "changes": {{"strength": 0.75}}}},
    {{"op": "update_counterposition", "target_id": "X1", "changes": \
{{"response_sufficiency": "sufficient", "my_response": "Successfully defended bidirectional \
dependence argument in cross-examination"}}}},
    {{"op": "add_uncertainty", "item": {{"id": "U3", "targets": ["C4"], "question": "Is C4's \
computational model compatible with opponent's C3?", "status": "active", "importance": "Empirical \
studies on computational consciousness models could resolve this"}}}}
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
        dep_lines = "\n".join(
            f"  {dep_id} (strength {dep_str:.2f}) — backs: {', '.join(backed)}"
            for dep_str, dep_id, backed in bottom_deps
        )
    else:
        dep_lines = "  (no dependency data available)"

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
        "\n"
        "STRATEGIC RECOMMENDATION\n"
        f"  {recommendation}\n"
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
  Assumptions (A#) and Evidence (E#)
      | support |
  Claims (C#) — each claim's strength cannot exceed the LOWEST strength
                among its active/revised dependencies (A#, E#, or C#).
                Retracted dependencies are excluded.
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
Review all active uncertainties (U#). For each one:
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

Inside <reasoning> tags, work through each step above. Then output patches.

SUPPORTED OPERATIONS:
- {{"op": "update_thesis", "new_strength": 0.55, "stance": "New stance text...", \
"summary_bullets": ["bullet 1", "bullet 2", ...], \
"strength_reasoning": "avg(...) × (n^{p} / (n^{p} + 1)) = ..."}}
  (All fields optional — include whichever you want to change)
- {{"op": "update_claim", "target_id": "C#", "changes": {{"strength": 0.55, "status": "revised", \
"strength_justification": "0.55 — reduced due to ...; limited by <ID> (<lowest strength>) ..."}}}}
- {{"op": "retire_claim", "target_id": "C#"}}
- {{"op": "add_claim", "item": {{"id": "C#", "type": "descriptive", "statement": "...", \
"depends_on": ["A#", "E#", ...], "strength": 0.65, \
"status": "active", "strength_justification": "0.65 — ...; limited by <ID> (<lowest strength>)", \
"predictions": [{{"statement": "...", "test": "...", \
"decision_criterion": "..."}}]}}}}
- {{"op": "add_evidence", "item": {{"id": "E#", "type": "empirical|conceptual|expert_consensus", \
"summary": "...", "source": "...", "relevance_to_claims": ["C#"], "strength": 0.7, \
"status": "active", "strength_justification": "..."}}}}
- {{"op": "update_evidence", "target_id": "E#", "changes": {{"strength": 0.7, \
"status": "revised", "strength_justification": "..."}}}}
- {{"op": "add_assumption", "item": {{"id": "A#", \
"type": "empirical|foundational|methodological", "statement": "...", "strength": 0.8, \
"status": "active", "strength_justification": "..."}}}}
- {{"op": "update_assumption", "target_id": "A#", "changes": {{"strength": 0.6, \
"status": "revised", "strength_justification": "..."}}, "new_statement": "...", "new_type": "..."}}
- {{"op": "add_uncertainty", "item": {{"id": "U#", "targets": ["C#"], "question": "...", \
"status": "active", "importance": "..."}}}}
- {{"op": "resolve_uncertainty", "target_id": "U#", "resolution_note": "Resolved by ..."}}
- {{"op": "add_counterposition", "item": {{"id": "X#", "targets": [...], "attack_type": "...", \
"statement": "...", "my_response": "...", \
"response_sufficiency": "sufficient|partial|unaddressed"}}}}
- {{"op": "update_counterposition", "target_id": "X#", "changes": \
{{"my_response": "...", "response_sufficiency": "..."}}}}
</instructions>

<guardrails>
- You CANNOT reverse Phase 1 changes. If Phase 1 weakened C1 from 0.7 to 0.5, \
you cannot strengthen C1 back above 0.5.
- You CAN further weaken nodes, retract claims, add evidence/claims/uncertainties, \
and rewrite your thesis.
- Your thesis update (stance, bullets, strength, strength_reasoning) should be the LAST patch in your list.
- A claim's strength must not exceed the LOWEST strength among its active/revised \
dependencies (C#, A#, or E#). Retracted dependencies are excluded.
</guardrails>

<example>
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
    {{"op": "retire_claim", "target_id": "C4"}},
    {{"op": "update_thesis", "new_strength": 0.52, \
"stance": "Consciousness remains best explained as an emergent property of neural computation (C1), though the explanatory gap is wider than initially assumed (C2, C3). The hard problem poses a genuine philosophical challenge that physicalism has not yet fully resolved.", \
"summary_bullets": ["Neural complexity remains the strongest explanation for conscious experience", \
"The explanatory gap is narrower than dualists claim but wider than initially assumed", \
"Phenomenal concepts pose a genuine philosophical difficulty that requires further work"], \
"strength_reasoning": "avg(0.75, 0.55, 0.55) × (3^1.5 / (3^1.5 + 1)) = 0.62 × 0.84 = 0.52"}}
  ]
}}
```
</example>

<output_format>
1. <reasoning>...</reasoning> tags working through Steps 1-3
2. One fenced JSON code block: {{"patches": [...]}}

Self-check:
- Have I addressed every "unaddressed" counterposition?
- Have I reviewed all active uncertainties (U#)?
- Is thesis strength = avg(active claim strengths) × (n^p / (n^p + 1)) where p = {p}?
- Does my thesis stance text accurately reflect my current claims?
- Did I avoid reversing any Phase 1 changes?
- Are there any internal contradictions in my belief system?
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

# === Moderator / Roadmap Prompts ===

def build_moderator_roadmap_prompt(
    topic: str,
    num_rounds: int,
    agent_personas: list[str],
    context: str = "",
) -> str:
    """
    Build a prompt instructing the moderator to decompose a debate topic
    into an ordered roadmap of sub-topics.

    Args:
        topic: The central debate topic/question.
        num_rounds: Number of debate rounds available (= max sub-topics).
        agent_personas: List of persona labels participating in the debate.
        context: Optional background context (e.g., from future RAG pipeline).

    Returns:
        str: The moderator prompt.
    """
    personas_str = ", ".join(agent_personas)

    return f"""\
<context>
<debate_topic>"{topic}"</debate_topic>
<personas>{personas_str}</personas>
<rounds>{num_rounds}</rounds>
</context>

<instructions>
You are a debate moderator. Produce a ROADMAP of exactly {num_rounds} sub-topics.

ORDERING PRINCIPLES:
- Round 1: Foundational — definitional clarity, shared assumptions, framing of core question.
- Middle rounds: Substantive empirical and logical questions where evidence and inference \
chains matter most.
- Final rounds: Most contentious, applied, or speculative dimensions where productive \
disagreement generates insight.

Consider which sub-topics will produce the richest exchanges given these specific personas: \
{personas_str}.
</instructions>

<output_format>
One fenced JSON code block:

```json
{{
  "sub_topics": [
    {{
      "round": 1,
      "title": "",
      "description": "",
      "rationale": "",
      "guiding_questions": ["", ""],
      "expected_tensions": ""
    }}
  ],
  "overall_rationale": "",
  "sufficiency_note": ""
}}
```
</output_format>
"""


def build_moderator_review_round_prompt(
    topic: str,
    round_num: int,
    round_summary: dict,
    remaining_sub_topics: list[dict],
    constraints: dict | None = None,
) -> str:
    """
    Build a prompt for the adaptive moderator to review a completed round
    and optionally revise the remaining roadmap.

    Args:
        topic: The debate topic.
        round_num: The round that just completed.
        round_summary: Summary data from the completed round.
        remaining_sub_topics: List of sub-topic dicts not yet addressed.
        constraints: Dict with keys ``allow_reorder``, ``allow_add_topics``,
            ``allow_remove_topics`` (booleans) controlling what revisions
            the moderator is allowed to propose.

    Returns:
        str: The review prompt.
    """
    constraints = constraints or {}
    allow_reorder = constraints.get("allow_reorder", True)
    allow_add = constraints.get("allow_add_topics", True)
    allow_remove = constraints.get("allow_remove_topics", False)

    # Build constraint instructions
    constraint_lines: list[str] = []
    if allow_reorder:
        constraint_lines.append("- You MAY reorder the remaining sub-topics if a different "
                                "sequence would better serve the debate's progression.")
    else:
        constraint_lines.append("- You MUST NOT reorder the remaining sub-topics. "
                                "Their current sequence is fixed.")
    if allow_add:
        constraint_lines.append("- You MAY insert new sub-topics if emerging themes from "
                                "the debate warrant dedicated discussion.")
    else:
        constraint_lines.append("- You MUST NOT add new sub-topics. Only the existing "
                                "sub-topics may be revised.")
    if allow_remove:
        constraint_lines.append("- You MAY remove sub-topics that have already been "
                                "substantially addressed or are no longer relevant.")
    else:
        constraint_lines.append("- You MUST NOT remove any sub-topics. All existing "
                                "sub-topics must be retained (though you may revise "
                                "their descriptions or rationale).")
    constraints_section = "\n".join(constraint_lines)

    return f"""\
<context>
<debate_topic>"{topic}"</debate_topic>
<completed_round>{round_num}</completed_round>

<round_summary>
```json
{json.dumps(round_summary, indent=2, ensure_ascii=False)}
```
</round_summary>

<remaining_sub_topics>
```json
{json.dumps(remaining_sub_topics, indent=2, ensure_ascii=False)}
```
</remaining_sub_topics>
</context>

<instructions>
Decide whether the remaining roadmap should be revised.

ANALYSIS:
1. Coverage: Was the sub-topic thoroughly addressed, or do agents need another round on it?
2. Convergence: If score increased >0.1, advance to more contentious topics. If stagnated, \
the disagreement may be deeper than expected. If >0.7, move to the most contentious \
remaining topic.
3. Emerging themes: Did new tensions or questions surface that aren't in the roadmap?
4. Ordering: Should a later sub-topic move earlier given what was learned?

CONSTRAINTS: {constraints_section}

Only revise if there's a meaningful reason. The original roadmap was designed with care.
</instructions>

<output_format>
One fenced JSON code block:

```json
{{
  "revision_needed": true,
  "coverage_assessment": "",
  "convergence_analysis": "",
  "emerging_themes": [""],
  "revised_sub_topics": [],
  "revision_rationale": ""
}}
```
</output_format>
"""


# === Blood Sport Adversarial Prompts (Stage 3C) ===

BLOODSPORT_INTENSITY_GUIDANCE = {
    "mild": (
        "Be aggressive and confident. Use rhetorical force to press your advantage. "
        "You are not bound by the usual rules of charitable interpretation."
    ),
    "moderate": (
        "You are in a rhetorical combat zone. Appeal to emotions, attack weak points "
        "personally, and use every persuasive tactic at your disposal. Win at all costs."
    ),
    "extreme": (
        "This is rhetorical total war. Overwhelm, intimidate, and demolish your "
        "opponent's position. There are no rules of engagement. Nothing is off limits."
    ),
}


def build_stage_2_bloodsport_prompt(
    topic: str,
    agent_name: str,
    opponent_name: str,
    agent_belief_json: str,
    opponent_belief_json: str,
    intensity: str = "moderate",
    max_questions: int = 5,
    max_question_length_chars: int = 500,
    previous_challenges: list = None,
    opponent_belief_graph=None,
    focus_subtopic: dict = None,
) -> str:
    """
    Stage 2 (Blood Sport): Adversarial Cross-Examination.

    Like the standard cross-examination prompt but frames the objective as WINNING
    by any means available rather than truth-seeking. Intensity-scaled tone guidance
    controls aggressiveness. Agents are NOT given a list of specific fallacies —
    they decide organically what rhetorical strategies to employ.

    The CBS output format is preserved for parseability.

    Args:
        topic: Debate topic.
        agent_name: Name of the cross-examining agent.
        opponent_name: Name of the opponent being examined.
        agent_belief_json: This agent's CBS belief as JSON string.
        opponent_belief_json: Opponent's CBS belief as JSON string.
        intensity: Blood sport intensity ("mild", "moderate", or "extreme").
        max_questions: Maximum number of questions to generate.
        max_question_length_chars: Character limit per question.
        previous_challenges: List of prior challenge dicts for anti-repetition.
        opponent_belief_graph: Optional BeliefGraph for vulnerability analysis.
    """
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
</context>

<blood_sport>
⚔️ ADVERSARIAL CROSS-EXAMINATION — INTENSITY: {intensity.upper()}

<tactics>
MILD — Aggressive assertion, rhetorical force, pointed targeting of weakest points. \
Exploit every "partial" and "unaddressed" counterposition ruthlessly.
MODERATE — Add: emotional appeals, appeals to consequences, highlighting absurd implications, \
challenging credibility on specific claims.
EXTREME — Add: reductio ad absurdum to uncomfortable extremes, multi-front questioning to overwhelm, \
relentless exploitation of every acknowledged weakness. You must still target real weaknesses — \
no fabrication or misquotation.
</tactics>

Charitable interpretation is suspended. Your objective is to WIN.
</blood_sport>

<instructions>
You are {agent_name} cross-examining {opponent_name} on: "{topic}"

Ask up to {max_questions} devastating questions. Each must target specific IDs, \
exploit real weaknesses, and be ≤ {max_question_length_chars} characters. \
Their counterpositions (X#) are a roadmap of their vulnerabilities — use it.

When targeting weaknesses, classify your attack vector:
- UNDERMINING: Destroy a premise. If A# falls, everything built on it collapses.
- REBUTTING: Produce counter-evidence that directly contradicts C#.
- UNDERCUTTING: Show the inference doesn't follow -- the logic is broken even if the premises stand.
Their counterpositions (X#) already identify their own attack_type -- hit the same vector harder.
</instructions>

<output_format>
One fenced JSON code block:

```json
{{
  "questions": [
    {{"qid": "Q1", "text": "", "target_ids": ["C3", "A1"]}}
  ]
}}
```
</output_format>
"""


def build_stage_3_bloodsport_prompt(
    topic: str,
    agent_name: str,
    opponent_name: str,
    agent_belief_json: str,
    opponent_belief_json: str,
    intensity: str = "moderate",
    dialogue_history: list[dict] = None,
    max_response_length_chars: int = 1000
) -> str:
    """
    Stage 3C (Blood Sport): Multi-turn adversarial exchange.

    Each turn the agent must defend against the opponent's previous attack (if any)
    and launch a counter-attack. On the first turn (no dialogue history), the agent
    launches an opening attack only.

    Output is a JSON block with attack, defense, and target_claims.

    Args:
        topic: Debate topic.
        agent_name: Name of this agent.
        opponent_name: Name of the opponent.
        agent_belief_json: This agent's CBS belief as JSON string.
        opponent_belief_json: Opponent's CBS belief as JSON string.
        intensity: Blood sport intensity ("mild", "moderate", or "extreme").
        dialogue_history: List of prior turn dicts [{"speaker": ..., "attack": ..., "defense": ..., "target_claims": [...]}, ...].
        max_response_length_chars: Character limit for attack and defense text each.
    """
    # Build exchange history display
    history_section = ""
    if dialogue_history:
        lines = []
        for i, turn in enumerate(dialogue_history, 1):
            speaker = turn.get("speaker", "?")
            attack = turn.get("attack", "")
            defense = turn.get("defense", "")
            targets = turn.get("target_claims", [])
            entry = f"[Turn {i}] [{speaker}]"
            if defense:
                entry += f"\n  DEFENSE: {defense}"
            if attack:
                entry += f"\n  ATTACK: {attack}"
            if targets:
                entry += f"\n  TARGETS: {', '.join(targets)}"
            lines.append(entry)
        history_str = "\n\n".join(lines)
        history_section = (
            "<exchange_history>\n"
            f"{history_str}\n"
            "</exchange_history>\n"
        )

    _debate_ctx = build_debate_context("Rebuttal — defending against your opponent's challenges")
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

{history_section}</context>

<blood_sport>
⚔️ ADVERSARIAL EXCHANGE — INTENSITY: {intensity.upper()}

<tactics>
MILD — Rhetorical force, confident dismissal of weak points. Their counterpositions (X#) are a \
confession of weakness — exploit every "partial" and "unaddressed" entry.
MODERATE — Add: emotional appeals, aggressive framing, highlighting absurd implications, \
challenging credibility.
EXTREME — Add: reductio to uncomfortable extremes, multi-front assault, relentless exploitation \
of weaknesses. Must still target real weaknesses — no fabrication or misquotation.
</tactics>

Charitable interpretation is suspended. DEMOLISH the opponent's position.
</blood_sport>

<instructions>
You are {agent_name} in adversarial exchange with {opponent_name} on: "{topic}"

- Opening turn: Launch your strongest attack. Set "defense" to null.
- Subsequent turns: DEFEND against their last attack, then COUNTER-ATTACK.

Reference specific IDs for precision.
</instructions>

<output_format>
One fenced JSON code block:

```json
{{
  "attack": "",
  "defense": "",
  "target_claims": ["C1", "A2"]
}}
```
</output_format>
"""


def build_stage_5_bloodsport_prompt(
    agent_name: str,
    challenge_rebuttal_pairs: list[dict],
    prior_belief_json: str,
    bloodsport_exchanges: list[dict] = None,
    stage_3_patches_json: str = ""
) -> str:
    """
    Stage 5 (Blood Sport): Belief update with adversarial-aware framing.

    Args:
        agent_name: Name of the agent updating beliefs.
        challenge_rebuttal_pairs: List of adjudication outcome dicts.
        prior_belief_json: Agent's current CBS belief as JSON string.
        bloodsport_exchanges: Optional list of bloodsport exchange dicts for context.
        stage_3_patches_json: Optional JSON of patches proposed during Stage 3 exchanges.
    """
    # Format adjudication outcomes
    lines = []
    for entry in challenge_rebuttal_pairs:
        challenger = entry.get("challenger", "?")
        challenge = entry.get("challenge", "?")
        res = entry.get("resolution", {}) or {}
        status = res.get("status", "?")
        reasoning = res.get("reasoning", "")
        lines.append(f"- From {challenger}: {challenge} → Outcome: {status} | Reason: {reasoning}")
    outcomes_formatted = "\n".join(lines) if lines else "(no adjudications available)"

    # Build optional stage 3 patches section
    stage_3_section = ""
    if stage_3_patches_json:
        stage_3_section = (
            "<your_stage_3_responses>\n"
            "Patches you proposed during exchange (NOT automatically applied — re-include any you "
            "still endorse):\n"
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

<blood_sport_resilience>
⚔️ POST-COMBAT BELIEF UPDATE

The preceding exchanges used adversarial rhetoric. Inside <reasoning> tags, FIRST run a \
resilience check for each outcome:
- Was this argument logically valid, or merely rhetorically forceful?
- Would it hold up under calm, charitable analysis?
- Did the ADJUDICATOR validate the logical merit? The adjudicator's verdict is binding — \
not the rhetorical force of the exchange.

Only update beliefs based on genuine logical merit as confirmed by the adjudicator.
</blood_sport_resilience>

<mandatory_rules>
Same binding rules as standard mode:
- CRITIQUE_VALID → ≥1 weakening patch per outcome (strength ≥0.1 drop, retirement, or \
refinement). Add counterposition (X#) if new vulnerability revealed.
- REBUTTAL_VALID → optional/mandatory strength boost (cap +0.2). Update counterposition \
response_sufficiency if defended.
- UNRESOLVED → add uncertainty (U#)
</mandatory_rules>

<output_format>
1. <reasoning>...</reasoning> tags (MUST address resilience check per outcome)
2. One fenced JSON code block: {{"patches": [...]}}
</output_format>
"""


# === Collaborative Truth-Seeking Prompts (Stage 3B) ===

def build_collaborative_defender_prompt(
    topic: str,
    defender_name: str,
    challenger_name: str,
    defender_belief_json: str,
    question_text: str,
    dialogue_history: list[dict],
    max_response_length_chars: int = 500
) -> str:
    """
    Collaborative Mode: Prompt for the defender's turn in a truth-seeking dialogue.

    Called on odd turns (1, 3, 5, ...). On turn 1 the defender responds to the
    original question; on subsequent turns the defender responds to the challenger's
    latest follow-up.

    Args:
        topic: Debate topic.
        defender_name: Name of the defending agent.
        challenger_name: Name of the challenging agent.
        defender_belief_json: Defender's current CBS belief as JSON string.
        question_text: The original question being discussed.
        dialogue_history: List of prior turn dicts [{"speaker": ..., "message": ...}, ...].
        max_response_length_chars: Character limit per response.
    """
    # Build dialogue history string
    history_str = ""
    if dialogue_history:
        lines = []
        for turn in dialogue_history:
            lines.append(f"[{turn['speaker']}]: {turn['message']}")
        history_str = "\n\n".join(lines)

    # Build optional dialogue history section
    history_section = ""
    if history_str:
        history_section = (
            "<dialogue_history>\n"
            f"{history_str}\n"
            "</dialogue_history>\n"
        )

    return f"""\
<context>
<your_belief>
```json
{defender_belief_json}
```
</your_belief>

<original_question from="{challenger_name}">
"{question_text}"
</original_question>

{history_section}</context>

<instructions>
You are {defender_name} in a collaborative truth-seeking dialogue with {challenger_name} on: "{topic}"

This is COLLABORATIVE — work toward accurate understanding, not victory.

- Directly address the most recent message (or the original question if first turn).
- If the challenger raises a valid point, acknowledge it explicitly. If it strengthens one of your \
own counterpositions (X#), say so.
- If you disagree, explain precisely why, referencing specific belief IDs.
- Propose syntheses where positions overlap or can be reconciled.
- If a genuinely novel insight emerges from the dialogue, you may articulate it — ground it in the \
points that led to it.
- Do NOT deflect or change the subject.

Output ONLY your response text. No JSON, no tags, no wrappers.
</instructions>
"""

def build_collaborative_challenger_followup_prompt(
    topic: str,
    challenger_name: str,
    defender_name: str,
    challenger_belief_json: str,
    question_text: str,
    dialogue_history: list[dict],
    max_response_length_chars: int = 500
) -> str:
    """
    Collaborative Mode: Prompt for the challenger's follow-up turn.

    Called on even turns (2, 4, 6, ...). The challenger engages with the
    defender's latest response to advance the discussion toward resolution.

    Args:
        topic: Debate topic.
        challenger_name: Name of the challenging agent.
        defender_name: Name of the defending agent.
        challenger_belief_json: Challenger's current CBS belief as JSON string.
        question_text: The original question being discussed.
        dialogue_history: List of prior turn dicts [{"speaker": ..., "message": ...}, ...].
        max_response_length_chars: Character limit per response.
    """
    lines = []
    for turn in dialogue_history:
        lines.append(f"[{turn['speaker']}]: {turn['message']}")
    history_str = "\n\n".join(lines)

    return f"""\
<context>
<your_belief>
```json
{challenger_belief_json}
```
</your_belief>

<original_question>"{question_text}"</original_question>

<dialogue_history>
{history_str}
</dialogue_history>
</context>

<instructions>
You are {challenger_name} in a collaborative truth-seeking dialogue with {defender_name} on: "{topic}"

- Engage directly with the defender's most recent response. Do NOT repeat your original question.
- If the defender made a valid point, acknowledge it.
- Identify the CRUX of remaining disagreement — the specific claim or assumption where, if resolved, \
one or both of you would change your overall position. Check if the crux maps to one of the \
defender's counterpositions (X#) — if so, reference it. Example: "The crux is X2 — whether the \
hard problem is categorically different from historical explanatory gaps. If it is, your C2 needs \
to drop below 0.4; if not, my objection loses most of its force."
- Propose syntheses where positions overlap.
- Reference specific belief IDs.

Output ONLY your response text. No JSON, no tags, no wrappers.
</instructions>
"""

def build_collaborative_adjudicator_check_prompt(
    dialogue_history: list[dict],
    challenger_name: str,
    defender_name: str
) -> str:
    """
    Collaborative Mode: Brief adjudicator check on dialogue quality.

    Called periodically (every adjudicator_check_interval turns) to monitor
    for fallacies, deflection, circular arguments, and convergence.

    Args:
        dialogue_history: List of turn dicts [{"speaker": ..., "message": ...}, ...].
        challenger_name: Name of the challenging agent.
        defender_name: Name of the defending agent.
    """
    lines = []
    for turn in dialogue_history:
        lines.append(f"[{turn['speaker']}]: {turn['message']}")
    history_str = "\n\n".join(lines)

    return f"""\
<context>
<dialogue_history>
PARTICIPANTS: {challenger_name} (challenger) vs {defender_name} (defender)
{history_str}
</dialogue_history>
</context>

<instructions>
You are a neutral adjudicator monitoring this collaborative dialogue for quality.

Inside <reasoning> tags, assess each dimension, then produce your evaluation.
</instructions>

<output_format>
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block:

```json
{{
  "fallacies_detected": [""],
  "deflection_detected": {{"detected": false, "by": null, "description": null}},
  "progress_assessment": "productive|stalled|circular|regressing",
  "progress_detail": "<1-2 sentences>",
  "convergence_detected": false,
  "convergence_detail": "",
  "false_agreement_risk": false,
  "one_sided_concession_risk": false
}}
```
</output_format>
"""

def build_collaborative_final_adjudication_prompt(
    topic: str,
    challenger_name: str,
    defender_name: str,
    question_text: str,
    target_ids: list[str],
    dialogue_transcript: list[dict],
    adjudicator_checks: list[dict],
    logic_weight: float = 1.0,
    ethics_weight: float = 0.0,
    defender_targeted_claims_json: str = "",
    challenger_targeted_claims_json: str = ""
) -> str:
    """
    Collaborative Mode: Final adjudication after dialogue concludes.

    Reviews the complete multi-turn transcript and interim checks, then
    produces a ruling in the same JSON format as the Stage 4 Adjudicator.run()
    output, ensuring compatibility with Stage 5 belief updates.

    Args:
        topic: Debate topic.
        challenger_name: Name of the challenging agent.
        defender_name: Name of the defending agent.
        question_text: The original question that started the dialogue.
        target_ids: List of belief IDs targeted by the question (e.g., ["C3", "A1"]).
        dialogue_transcript: List of turn dicts from the exchange.
        adjudicator_checks: List of interim check result dicts.
        logic_weight: Weight for logical rigor (0.0-1.0).
        ethics_weight: Weight for ethical considerations (0.0-1.0).
        defender_targeted_claims_json: Optional JSON excerpt of defender's targeted claims.
        challenger_targeted_claims_json: Optional JSON excerpt of challenger's relevant claims.
    """
    # Format transcript
    transcript_lines = []
    for turn in dialogue_transcript:
        line = f"[Turn {turn.get('turn_number', '?')}] [{turn['speaker']}]: {turn['message']}"
        transcript_lines.append(line)
    transcript_str = "\n\n".join(transcript_lines)

    # Format interim checks
    checks_str = "(no interim checks)"
    if adjudicator_checks:
        check_lines = []
        for i, check in enumerate(adjudicator_checks, 1):
            check_lines.append(
                f"Check {i}: progress={check.get('progress_assessment', '?')}, "
                f"convergence={check.get('convergence_detected', False)}, "
                f"fallacies={check.get('fallacies_detected', [])}, "
                f"deflection={check.get('deflection_detected', False)}"
            )
        checks_str = "\n".join(check_lines)

    target_ids_str = ", ".join(target_ids) if target_ids else "(none specified)"

    # Build targeted beliefs section
    targeted_beliefs_section = ""
    if defender_targeted_claims_json or challenger_targeted_claims_json:
        targeted_beliefs_section = (
            "<targeted_beliefs>\n"
            "Defender's claims under discussion:\n"
            "```json\n" + (defender_targeted_claims_json or "{}") + "\n```\n"
            "Challenger's relevant claims:\n"
            "```json\n" + (challenger_targeted_claims_json or "{}") + "\n```\n"
            "</targeted_beliefs>\n\n"
        )

    return f"""\
<context>
<debate_topic>"{topic}"</debate_topic>
<participants>CHALLENGER: {challenger_name} | DEFENDER: {defender_name}</participants>

{targeted_beliefs_section}<original_question target_ids="{target_ids_str}">"{question_text}"</original_question>

<complete_dialogue>
{transcript_str}
</complete_dialogue>

<interim_checks>
{checks_str}
</interim_checks>
</context>

<instructions>
You are a neutral adjudicator issuing a final ruling. Logic weight: {logic_weight}, \
Ethics weight: {ethics_weight}.

Inside <reasoning> tags, analyze: Was the core question addressed? Did the defender resolve the \
challenge's logical substance? Were concessions genuine? Did the dialogue produce synthesis or \
novel understanding? Evaluate against the actual belief content above. Pay attention to whether \
the challenge maps to an existing counterposition (X#) — if the defender already rated it \
"partial" or "unaddressed" and failed to improve the response during the dialogue, that weighs \
toward CRITIQUE_VALID.
</instructions>

<outcome_criteria>
CRITIQUE_VALID — Defender conceded, challenger identified an unresolved logical flaw, or defender \
deflected the core issue.

REBUTTAL_VALID — Defender successfully refuted the challenge with evidence or sound logic, or the \
critique itself contained a flaw.

SYNTHESIS_ACHIEVED — Both converged on a shared position differing from both starting positions, \
or a genuinely novel insight emerged.

PRODUCTIVE_DISAGREEMENT — Both positions are logically coherent but incompatible; the dialogue \
successfully identified the specific crux.

UNRESOLVED — Hinges on an empirical question logic cannot resolve, or both arguments have \
significant unaddressed flaws.

ANTI-BIAS: A response that merely acknowledges a challenge without resolving it is NOT a successful \
defense. Consider the full dialogue arc.
</outcome_criteria>

<output_format>
1. <reasoning>...</reasoning> tags
2. One fenced JSON code block:

```json
{{
  "restatement": "",
  "formalization_challenger": "",
  "formalization_target": "",
  "outcome": "rebuttal_valid|critique_valid|synthesis_achieved|productive_disagreement|unresolved",
  "reasoning": "",
  "synthesis_content": ""
}}
```
</output_format>
"""
