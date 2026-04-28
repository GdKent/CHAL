"""
wizard.py

Interactive configuration wizard for CHAL debates.
Walks the user through 12 steps to build a DebateConfig, with a review/edit
loop before launching.

Navigation:
    Esc      - Exit wizard
    Ctrl+Z   - Go back one step
    Ctrl+F1  - Show help for current prompt
"""

from __future__ import annotations

import os
from pathlib import Path

import questionary
import yaml
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.styles import Style as PTKStyle
from prompt_toolkit.styles import merge_styles
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chal.agents.epistemic_personas import PERSONA_DESCRIPTIONS, PERSONAS
from chal.agents.ethics_systems import (
    get_ethics_system_label,
)
from chal.agents.logic_systems import (
    LOGIC_SYSTEMS,
    get_logic_system_label,
)
from chal.cli.api_keys import PROVIDER_ENV_VARS
from chal.config import (
    CONFIG_DIR,
    DEFAULT_STORAGE_DIR,
    AdjudicationConfig,
    AgentConfig,
    DebateConfig,
    OutputConfig,
    ParallelConfig,
)

# =========================================================================
# Constants
# =========================================================================

PERSONA_CHOICES = [
    Choice(f"{key:<16s} - {PERSONA_DESCRIPTIONS[key]}", value=key)
    for key in PERSONAS
]

PROVIDER_CHOICES = ["openai", "anthropic", "google", "ollama", "xai", "perplexity"]

MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "openai": ["o4-mini", "o3", "o3-mini"],
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-5-20250929"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "ollama": ["deepseek-r1:14b", "deepseek-r1:32b", "qwq"],
    "xai": ["grok-3-mini", "grok-3"],
    "perplexity": ["sonar-reasoning-pro", "sonar-reasoning"],
}

# Short wizard tags for logic/ethics system selectors (displayed instead of
# the full multi-sentence descriptions from logic_systems.py / ethics_systems.py)
_LOGIC_WIZARD_TAGS: dict[str, str] = {
    "CLASSICAL_INFORMAL_BAYESIAN": "Recommended",
    "FORMAL_DEDUCTIVE": "Strict deduction",
    "BAYESIAN": "Evidence-based",
    "INFORMAL_CRITICAL": "Fallacy detection",
    "DIALECTICAL": "Thesis-antithesis",
    "FUZZY_MULTIVALUED": "Degrees of truth",
    "PARACONSISTENT": "Tolerates contradictions",
    "NONE": "Not recommended",
}

_ETHICS_WIZARD_TAGS: dict[str, str] = {
    "NONE": "Recommended",
    "UTILITARIAN": "Maximize welfare",
    "DEONTOLOGICAL": "Moral duties",
    "VIRTUE_ETHICS": "Character & flourishing",
    "CARE_ETHICS": "Relationships",
    "BALANCED": "Recommended",
}

# Display order for ethics system selector (BALANCED second, after NONE)
_ETHICS_KEY_ORDER = ["NONE", "BALANCED", "UTILITARIAN", "DEONTOLOGICAL", "VIRTUE_ETHICS", "CARE_ETHICS"]

# (display label, OutputConfig attribute, default value)
OUTPUT_TOGGLES: list[tuple[str, str, bool]] = [
    ("Debate transcript", "save_transcript", False),
    ("Belief trajectories plot", "plot_trajectories", False),
    ("Agent statistics", "save_agent_stats", False),
    ("Initial beliefs", "save_initial_beliefs", False),
    ("Final beliefs", "save_final_beliefs", False),
    ("Graph visualization", "generate_graph_visualization", False),
    ("Embeddings", "generate_embeddings", False),
    ("Training data export", "save_training_data", False),
    ("Analysis report", "save_analysis_report", False),
    ("Debug log", "save_debug_log", False),
]


# =========================================================================
# Questionary style — maroon theme to match CHAL banner
# =========================================================================

WIZARD_STYLE = PTKStyle([
    ("answer", "fg:#A82545 bold"),
    ("highlighted", "noinherit fg:#A82545 bold"),
    ("pointer", "fg:#A82545 bold"),
    ("selected", "noinherit"),
    ("hint", "fg:gray italic"),
])

# =========================================================================
# About CHAL — displayed from the main menu
# =========================================================================

ABOUT_CHAL = """\
[bold]CHAL[/bold] (Council of Hierarchical Agentic Language) is a framework for \
orchestrating structured debates between multiple AI agents, each embodying a \
distinct epistemological position.

[bold]Why CHAL?[/bold]

The pursuit of truth through argumentation is one of humanity's oldest and most \
powerful intellectual traditions. Yet good-faith dialectical exchange \
— where participants genuinely seek truth rather than rhetorical victory — \
remains remarkably difficult. Cognitive biases, emotional attachments to \
positions, and social pressures routinely derail even well-intentioned discourse.

CHAL addresses this by deploying AI agents as rigorous interlocutors capable of \
[italic]belief search[/italic]: systematically exploring argument space, tracking formal belief \
structures with dependency graphs and strength scores, and converging toward \
well-supported conclusions through structured debate.

[bold]Implications[/bold]

  [bold]\u2022[/bold] [bold]AI Safety & Reasoning[/bold] \u2014 Studying how agents form, challenge, and \
revise beliefs under adversarial pressure reveals failure modes and robustness \
properties of LLM reasoning.

  [bold]\u2022[/bold] [bold]Human Knowledge Extension[/bold] \u2014 Debates surface novel arguments, \
overlooked assumptions, and unexpected convergences on questions that have \
challenged thinkers for centuries.

  [bold]\u2022[/bold] [bold]Argumentation Literacy[/bold] \u2014 Observing structured debates helps users \
distinguish sound reasoning from rhetorical fallacies, strengthening critical \
thinking skills.

[bold]Using the Wizard[/bold]

Select [bold]Run a debate[/bold] to configure a new session. The wizard guides you through:

  1. Choosing a preset or building a custom configuration
  2. Setting a debate topic
  3. Selecting agent personas, providers, and models
  4. Configuring cross-examination and debate modes
  5. Setting adjudication and round parameters
  6. Choosing which outputs to generate

After configuration, review your settings, then launch the debate, save the \
configuration to YAML, or edit individual settings before proceeding.

[bold]Navigation[/bold]

  [bold]\u2022[/bold] [bold]Esc[/bold] \u2014 Exit the wizard at any time
  [bold]\u2022[/bold] [bold]Ctrl+Z[/bold] \u2014 Go back to the previous step
  [bold]\u2022[/bold] [bold]Ctrl+F1[/bold] \u2014 Show help for the current prompt\
"""


# =========================================================================
# Help texts — context-sensitive guidance displayed via Ctrl+F1
# =========================================================================

HELP_MAIN_MENU = """\
[bold]About CHAL[/bold] \u2014 Learn what CHAL is and how the wizard works.
[bold]Run a debate[/bold] \u2014 Configure and launch a new debate session.
[bold]Run the gauntlet[/bold] \u2014 Run a battery of debates (coming soon).
[bold]Exit[/bold] \u2014 Quit the wizard.\
"""

HELP_PRESET = """\
[bold]Presets[/bold] are pre-built debate configurations stored as YAML files in the \
[italic]configurations/[/italic] directory. Selecting a preset populates all settings \
instantly and takes you straight to the review screen.

Choose [bold]Custom[/bold] to configure everything from scratch \u2014 recommended if \
you want full control over agents, modes, and outputs.\
"""

HELP_TOPIC = """\
Enter a question or thesis for the agents to debate. Good topics are:

  \u2022 [bold]Specific[/bold] \u2014 "Does free will exist?" rather than "Philosophy"
  \u2022 [bold]Debatable[/bold] \u2014 Multiple reasonable positions should be possible
  \u2022 [bold]Scoped[/bold] \u2014 Narrow enough for agents to argue meaningfully

Examples: "Is consciousness reducible to computation?", \
"Should AI systems have legal personhood?", \
"Does the simulation hypothesis have empirical consequences?"\
"""

HELP_NUM_AGENTS = """\
Choose how many agents will participate in the debate (2\u20136).

  \u2022 [bold]2 agents[/bold] \u2014 Classic point/counterpoint. Fastest and most focused.
  \u2022 [bold]3 agents[/bold] \u2014 Adds a third perspective; richer cross-examination.
  \u2022 [bold]4\u20136 agents[/bold] \u2014 Multi-perspective debates. More diverse but longer \
runs and higher API costs.\
"""

HELP_PERSONA = """\
Each persona embodies a distinct epistemological stance:

  \u2022 [bold]EMPIRICIST[/bold] \u2014 Demands empirical evidence; rejects unfalsifiable claims.
  \u2022 [bold]RATIONALIST[/bold] \u2014 Trusts logical deduction and a-priori reasoning.
  \u2022 [bold]SKEPTIC[/bold] \u2014 Challenges all claims; exposes hidden assumptions.
  \u2022 [bold]SUPERNATURALIST[/bold] \u2014 Accepts truths beyond empirical observation.
  \u2022 [bold]PHENOMENOLOGIST[/bold] \u2014 Grounds truth in lived, first-person experience.
  \u2022 [bold]PRAGMATIST[/bold] \u2014 Defines truth as what works in practice.
  \u2022 [bold]CONSTRUCTIVIST[/bold] \u2014 Truth is socially and culturally constructed.
  \u2022 [bold]NIHILIST[/bold] \u2014 Denies inherent meaning or objective truth.
  \u2022 [bold]BAYESIAN[/bold] \u2014 Models knowledge as probabilistic inference.
  \u2022 [bold]PANPSYCHIST[/bold] \u2014 Consciousness is fundamental to all matter.
  \u2022 [bold]SIMULATIONIST[/bold] \u2014 Evaluates claims through the simulation hypothesis.
  \u2022 [bold]SYNTHESIST[/bold] \u2014 Integrates science, spirituality, and systems thinking.
  \u2022 [bold]NONE[/bold] \u2014 No persona; argues purely from the belief structure.

[dim]Tip: Pair contrasting personas (e.g. Empiricist vs Supernaturalist) for the \
most productive debates.[/dim]\
"""

HELP_PROVIDER = """\
Choose which LLM provider to use for this agent:

  \u2022 [bold]openai[/bold] \u2014 o-series reasoning models (o4-mini, o3, o3-mini).
  \u2022 [bold]anthropic[/bold] \u2014 Claude Opus and Sonnet with extended thinking.
  \u2022 [bold]google[/bold] \u2014 Gemini 2.5 reasoning models with thinking mode.
  \u2022 [bold]ollama[/bold] \u2014 Local reasoning models (DeepSeek-R1, QwQ). No API key required.
  \u2022 [bold]xai[/bold] \u2014 Grok reasoning models with think mode.
  \u2022 [bold]perplexity[/bold] \u2014 Sonar reasoning models with citations.

[dim]Make sure you have the corresponding API key set in your .env file \
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, or \
PERPLEXITY_API_KEY). Ollama runs locally and needs no key.[/dim]\
"""

HELP_MODEL = """\
[bold]Available reasoning models by provider:[/bold]

[bold]OpenAI:[/bold]
  \u2022 o4-mini \u2014 Advanced reasoning. Best balance of speed and depth. [bold](Recommended)[/bold]
  \u2022 o3 \u2014 Full-size reasoning model. Maximum depth.
  \u2022 o3-mini \u2014 Compact reasoning model. Fast and cost-effective.

[bold]Anthropic:[/bold]
  \u2022 claude-opus-4-6 \u2014 Most capable Claude model. Deep extended thinking. [bold](Recommended)[/bold]
  \u2022 claude-sonnet-4-5-20250929 \u2014 Strong reasoning with extended thinking.

[bold]Google:[/bold]
  \u2022 gemini-2.5-pro \u2014 Strongest Gemini reasoning model. [bold](Recommended)[/bold]
  \u2022 gemini-2.5-flash \u2014 Fast reasoning with thinking mode.

[bold]Ollama:[/bold]
  \u2022 deepseek-r1:14b \u2014 Deep reasoning. Good balance of quality and speed. [bold](Recommended)[/bold]
  \u2022 deepseek-r1:32b \u2014 Larger reasoning model. Higher quality, more resources.
  \u2022 qwq \u2014 Qwen's reasoning model. Strong analytical capabilities.

[bold]xAI:[/bold]
  \u2022 grok-3-mini \u2014 Compact Grok reasoning model. Fast. [bold](Recommended)[/bold]
  \u2022 grok-3 \u2014 Full Grok reasoning model. Maximum depth.

[bold]Perplexity:[/bold]
  \u2022 sonar-reasoning-pro \u2014 Multi-step reasoning with citations. [bold](Recommended)[/bold]
  \u2022 sonar-reasoning \u2014 Basic reasoning with search augmentation.

[dim]All listed models are reasoning-focused. You can also type any valid \
model name not listed here.[/dim]\
"""

HELP_TEMPERATURE = """\
Temperature controls the randomness of the model's output (0.0\u20131.0):

  \u2022 [bold]0.0\u20130.3[/bold] \u2014 Deterministic and focused. Best for logical, precise arguments.
  \u2022 [bold]0.4\u20130.7[/bold] \u2014 Balanced creativity and coherence. [bold](Recommended: 0.7)[/bold]
  \u2022 [bold]0.8\u20131.0[/bold] \u2014 More creative and unpredictable. Can surface novel arguments \
but may reduce coherence.

[dim]Note: Some reasoning models (o4-mini, o3-mini) ignore the temperature setting.[/dim]\
"""

HELP_STAGE2 = """\
Stage 2 is the cross-examination phase where agents challenge each other's \
initial belief structures.

Agents freely challenge each other without guidance. Each agent generates \
questions targeting weaknesses in other agents' arguments.\
"""

HELP_STAGE3 = """\
Stage 3 is the main debate phase where agents respond to cross-examination \
findings and argue their positions.

In rebuttal mode, each agent writes one single-shot rebuttal per round. \
This is fast and concise, making it ideal for focused debates.\
"""

HELP_NUM_ROUNDS = """\
Number of full debate rounds (1\u201310). Each round includes:

  1. Initial belief formation (Stage 1)
  2. Cross-examination (Stage 2)
  3. Debate / argumentation (Stage 3)
  4. Adjudication and scoring (Stage 4)

  \u2022 [bold]1 round[/bold] \u2014 Single pass. Fast, but agents cannot revise positions.
  \u2022 [bold]2\u20133 rounds[/bold] \u2014 Agents refine beliefs based on prior feedback. \
[bold](Recommended)[/bold]
  \u2022 [bold]4\u201310 rounds[/bold] \u2014 Deep iterative refinement. High cost but maximum \
convergence.\
"""

HELP_ADJ_BALANCE = """\
Choose how the adjudicator should balance [bold]logical validity[/bold] and \
[bold]ethical reasoning[/bold] when scoring arguments.

  \u2022 [bold]Pure Logic[/bold] \u2014 Only logical structure, deductive soundness, and \
consistency matter. Best for empirical and analytical topics. \
[bold](Recommended)[/bold]
  \u2022 [bold]Pure Ethics[/bold] \u2014 Only moral implications, fairness, and human values \
matter. Best for normative and policy questions.
  \u2022 [bold]Balanced[/bold] \u2014 Equal weight to both logic and ethics. Good for topics \
that span empirical and normative dimensions.
  \u2022 [bold]Custom[/bold] \u2014 Set your own weights (they will be normalized to sum \
to 1.0). Use this for fine-grained control.\
"""

HELP_ADJ_CUSTOM_WEIGHTS = """\
Enter custom weights for [bold]logic[/bold] and [bold]ethics[/bold] scoring \
(each 0.0\u20131.0).

The two values will be [bold]normalized[/bold] so they sum to 1.0. For example:
  \u2022 logic=0.8, ethics=0.2  \u2192  logic=0.8, ethics=0.2  (already sums to 1.0)
  \u2022 logic=3.0, ethics=1.0  \u2192  logic=0.75, ethics=0.25  (scaled down)

[dim]Tip: Higher logic weight prioritizes formal reasoning; higher ethics weight \
rewards moral consideration.[/dim]\
"""

HELP_ADJ_LOGIC_SYSTEM = """\
Choose the [bold]logical framework[/bold] the adjudicator uses to evaluate \
argument validity.

  \u2022 [bold]Classical + Bayesian[/bold] \u2014 Standard logic with Bayesian induction \
and Occam's Razor. [bold](Recommended)[/bold]
  \u2022 [bold]Formal Deductive[/bold] \u2014 Only formally valid syllogisms; no inductive \
reasoning.
  \u2022 [bold]Dialectical[/bold] \u2014 Hegelian thesis-antithesis-synthesis; contradictions \
drive progress.
  \u2022 [bold]Informal / Critical Thinking[/bold] \u2014 Fallacy detection, relevance, and \
evidence sufficiency.
  \u2022 [bold]Fuzzy / Multi-valued[/bold] \u2014 Degrees of truth between 0 and 1; no \
binary judgments.
  \u2022 [bold]Paraconsistent[/bold] \u2014 Tolerates local contradictions without global \
explosion.
  \u2022 [bold]None (Pure Ethics)[/bold] \u2014 No logical evaluation; judge only ethical \
merit. [bold](Not recommended for most topics)[/bold]\
"""

HELP_ADJ_ETHICS_SYSTEM = """\
Choose the [bold]ethical framework[/bold] the adjudicator uses to evaluate \
the moral dimensions of arguments.

  \u2022 [bold]None (Pure Logic)[/bold] \u2014 No ethical evaluation; judge only logical \
soundness. [bold](Recommended for most topics)[/bold]
  \u2022 [bold]Utilitarian[/bold] \u2014 Maximize well-being and minimize suffering for \
the greatest number.
  \u2022 [bold]Deontological[/bold] \u2014 Respect universal moral duties and the \
categorical imperative.
  \u2022 [bold]Virtue Ethics[/bold] \u2014 Promote human flourishing, practical wisdom, \
and excellence.
  \u2022 [bold]Care Ethics[/bold] \u2014 Prioritize relationships, responsibility, and \
vulnerability.
  \u2022 [bold]Balanced (Rule-Utilitarian)[/bold] \u2014 Comprehensive: evaluate both \
outcomes/welfare and moral rules/rights. Most thorough ethics system. (Recommended)\
"""

HELP_OUTPUTS = """\
Select which outputs to generate after the debate:

  \u2022 [bold]Debate transcript[/bold] \u2014 Full text of all agent exchanges.
  \u2022 [bold]Belief trajectories[/bold] \u2014 Plot showing how strength scores evolved.
  \u2022 [bold]Agent statistics[/bold] \u2014 Per-agent metrics (claims, evidence, revisions).
  \u2022 [bold]Initial beliefs[/bold] \u2014 Each agent's CBS belief structure before debate.
  \u2022 [bold]Final beliefs[/bold] \u2014 Each agent's CBS belief structure after debate.
  \u2022 [bold]Graph visualization[/bold] \u2014 Dependency graph of claims and evidence.
  \u2022 [bold]Embeddings[/bold] \u2014 Sentence embeddings for semantic analysis.
  \u2022 [bold]Training data export[/bold] \u2014 Structured data for fine-tuning.
  \u2022 [bold]Analysis report[/bold] \u2014 Detailed analytical report of the debate.
  \u2022 [bold]Debug log[/bold] \u2014 Verbose log for troubleshooting.

[dim]Use <space> to toggle individual items, <a> to toggle all.[/dim]\
"""

HELP_REVIEW_ACTION = """\
  \u2022 [bold]Launch debate[/bold] \u2014 Start the debate with the current configuration.
  \u2022 [bold]Edit a setting[/bold] \u2014 Go back and change a specific setting.
  \u2022 [bold]Save config to YAML[/bold] \u2014 Save this configuration to a file for reuse.
  \u2022 [bold]Cancel[/bold] \u2014 Discard this configuration and exit.\
"""

HELP_EDIT_SECTION = """\
Choose which part of the configuration to modify. After editing, you will \
return to the review screen to verify your changes.\
"""

HELP_SAVE_PATH = """\
Enter a file path for saving the configuration as YAML.

  \u2022 Use a [bold].yaml[/bold] extension (added automatically if omitted).
  \u2022 Relative paths are saved from the current working directory.
  \u2022 Saved configs can be reloaded with [bold]chal --config path.yaml[/bold].\
"""

HELP_PARALLELIZATION = """\
CHAL's debate pipeline is sequential, but many stages contain independent API \
calls that can run concurrently:

  \u2022 [bold]Stage 1[/bold] \u2014 Opening positions (one call per agent)
  \u2022 [bold]Stage 2[/bold] \u2014 Cross-examination (one call per agent pair)
  \u2022 [bold]Stage 3[/bold] \u2014 Rebuttals (one call per agent per challenge)
  \u2022 [bold]Stage 4[/bold] \u2014 Adjudication (one call per challenge-rebuttal pair)

With parallelization enabled, these independent calls fire concurrently using \
a thread pool, significantly reducing wall-clock time.

[bold yellow]Important:[/bold yellow] Most LLM providers enforce per-key rate limits. Under \
parallel mode, multiple requests hit the API simultaneously, which can trigger \
rate-limit errors. To avoid this, [bold]use multiple API keys[/bold] for each \
provider \u2014 CHAL's key pool rotates keys automatically.

[dim]Recommendation: Enable parallelization and enter 2\u20134 API keys per provider \
in the next step.[/dim]\
"""

HELP_MAX_WORKERS = """\
The [bold]max workers[/bold] setting controls how many API calls can run \
simultaneously in the thread pool.

[bold]Recommendations:[/bold]
  • [bold]2–3 agents:[/bold] 5 threads (default) is plenty — one thread per \
agent plus headroom for adjudication calls.
  • [bold]4–6 agents:[/bold] 5–8 threads works well.
  • [bold]7+ agents:[/bold] Consider 8–12 threads if your API keys support it.

[bold yellow]Key consideration:[/bold yellow] Each concurrent thread makes an \
API call, so the number of threads should not exceed the rate limits of your \
API keys. If you have [bold]N[/bold] API keys for a provider, CHAL's key pool \
rotates through them — so N keys × provider rate limit = your effective \
concurrency ceiling.

[dim]Higher values use more system threads but won't help if the API is the \
bottleneck. The default of 5 works well for most setups.[/dim]\
"""

HELP_API_KEYS = """\
Enter API keys for each LLM provider used in this debate. Keys are set for \
this session only and are [bold]not[/bold] persisted to disk.

For each provider, you can enter [bold]multiple keys[/bold] for rate-limit \
rotation \u2014 CHAL will cycle through them automatically when making parallel \
API calls.

  \u2022 [bold]OpenAI[/bold] \u2014 OPENAI_API_KEY (required for GPT and o-series models)
  \u2022 [bold]Anthropic[/bold] \u2014 ANTHROPIC_API_KEY (required for Claude models)
  \u2022 [bold]Google[/bold] \u2014 GOOGLE_API_KEY (required for Gemini models)
  \u2022 [bold]xAI[/bold] \u2014 XAI_API_KEY (required for Grok models)
  \u2022 [bold]Perplexity[/bold] \u2014 PERPLEXITY_API_KEY (required for Sonar models)
  \u2022 [bold]Ollama[/bold] \u2014 No API key needed (local inference)

[dim]Tip: If you already have keys in your .env file, they will be detected \
automatically and you can skip this step.[/dim]\
"""


# =========================================================================
# Wizard navigation
# =========================================================================

class _Sentinel:
    """Unique sentinel for wizard navigation signals."""
    def __init__(self, name: str) -> None:
        self.name = name
    def __repr__(self) -> str:
        return f"_Sentinel({self.name})"

_EXIT = _Sentinel("EXIT")
_BACK = _Sentinel("BACK")
_HELP = _Sentinel("HELP")


class WizardExit(Exception):
    """Raised when the user presses Esc to exit the wizard."""


class WizardBack(Exception):
    """Raised when the user presses Ctrl+Z to go back one step."""


def _ask(question, help_text: str | None = None):
    """Run a questionary Question with Esc, Ctrl+Z, and Ctrl+F1 bindings.

    Injects prompt_toolkit key bindings before running the prompt:
      - Esc:      exit wizard  (raises WizardExit)
      - Ctrl+Z:   go back      (raises WizardBack)
      - Ctrl+F1:  show help    (displays help_text, then re-shows prompt)
      - Ctrl+C: cancel       (raises KeyboardInterrupt, unchanged)
    """
    try:
        app = question.application
        nav_kb = KeyBindings()

        @nav_kb.add('escape', eager=True, filter=~has_completions)
        def _handle_escape(event):
            event.app.exit(result=_EXIT)

        @nav_kb.add('c-z')
        def _handle_back(event):
            event.app.exit(result=_BACK)

        @nav_kb.add('c-f1')
        def _handle_help(event):
            event.app.exit(result=_HELP)

        existing = app.key_bindings or KeyBindings()
        app.key_bindings = merge_key_bindings([existing, nav_kb])

        # Apply maroon wizard style
        if hasattr(app, 'style') and app.style:
            app.style = merge_styles([app.style, WIZARD_STYLE])
        else:
            app.style = WIZARD_STYLE

        # Clear selected_options for select prompts so default items use
        # class:highlighted (not class:selected) when pointed at.  The
        # pointer position is already set via initial_choice/pointed_at.
        # Skip for checkbox prompts where checked choices must be preserved.
        from questionary.prompts.common import InquirerControl
        for child in getattr(app.layout.container, 'children', []):
            window = getattr(child, 'content', child)
            ic = getattr(window, 'content', None)
            if isinstance(ic, InquirerControl):
                has_prechecked = any(
                    getattr(c, 'checked', False) for c in ic.choices
                )
                if not has_prechecked:
                    ic.selected_options = []
                break
    except Exception:
        pass  # Gracefully degrade (e.g., in test environments with mocks)

    while True:
        result = question.ask()

        if result is _HELP:
            if help_text:
                Console().print(Panel(
                    help_text,
                    title="[bold]Help[/bold]",
                    border_style="#9B1B30",
                    expand=False,
                    width=80,
                ))
            continue  # re-show the prompt

        if result is None:
            raise KeyboardInterrupt
        if result is _EXIT:
            raise WizardExit
        if result is _BACK:
            raise WizardBack
        return result


# =========================================================================
# Validation helpers
# =========================================================================

def _validate_float_range(text: str, lo: float = 0.0, hi: float = 1.0) -> bool | str:
    """Validator for float inputs within a range."""
    try:
        val = float(text)
        if lo <= val <= hi:
            return True
        return f"Must be between {lo} and {hi}"
    except ValueError:
        return "Must be a number"


def _validate_int_range(text: str, lo: int, hi: int) -> bool | str:
    """Validator for integer inputs within a range."""
    try:
        val = int(text)
        if lo <= val <= hi:
            return True
        return f"Must be between {lo} and {hi}"
    except ValueError:
        return "Must be a whole number"


# =========================================================================
# Main menu
# =========================================================================

def ask_main_menu() -> str:
    """Display the main CHAL menu.

    Returns:
        One of "about", "debate", "gauntlet", "exit".
    """
    return _ask(questionary.select(
        "What would you like to do?",
        choices=[
            Choice("About CHAL", value="about"),
            Choice("Run a debate", value="debate"),
            Choice("Run the gauntlet", value="gauntlet"),
            Choice("Exit", value="exit"),
        ],
    ), help_text=HELP_MAIN_MENU)


# =========================================================================
# Preset selection
# =========================================================================

_PRESET_ORDER = ["default"]


def _scan_presets() -> list[tuple[str, str, Path]]:
    """Scan configurations/ directory for YAML presets.

    Returns:
        List of (display_name, config_name, path) tuples in preferred order.
    """
    presets: list[tuple[str, str, Path]] = []
    if not CONFIG_DIR.is_dir():
        return presets
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data.get("metadata", {}).get("name", p.stem)
            desc = data.get("metadata", {}).get("description", "")
            label = f"{name} ({desc})" if desc else name
            presets.append((label, p.stem, p))
        except Exception:
            continue

    # Sort by preferred order; unlisted presets go to the end alphabetically
    order_map = {name: i for i, name in enumerate(_PRESET_ORDER)}
    presets.sort(key=lambda t: (order_map.get(t[1], len(_PRESET_ORDER)), t[1]))
    return presets


def ask_preset() -> DebateConfig | None:
    """Ask user whether to start from a preset configuration.

    Returns:
        A loaded DebateConfig if a preset was chosen, or None for custom.
    """
    presets = _scan_presets()
    choices = [Choice("Custom (configure from scratch)", value="__custom__")]
    for label, config_name, path in presets:
        choices.append(Choice(label, value=str(path)))

    result = _ask(questionary.select(
        "Start from a preset?",
        choices=choices,
    ), help_text=HELP_PRESET)

    if result == "__custom__":
        return None

    return DebateConfig.from_yaml(Path(result))


# =========================================================================
# Step functions — each returns user-chosen value(s) for one config section
# =========================================================================

def ask_topic(default: str = "") -> str:
    """Step 1: Ask for the debate topic."""
    return _ask(questionary.text(
        "What topic should the agents debate?",
        default=default,
        instruction="(free text \u2014 e.g. 'Does free will exist?')",
    ), help_text=HELP_TOPIC)


def ask_num_agents(default: int = 2) -> int:
    """Step 2: Ask how many agents."""
    answer = _ask(questionary.text(
        "How many agents should participate? (2-6)",
        default=str(default),
        validate=lambda t: _validate_int_range(t, 2, 6),
    ), help_text=HELP_NUM_AGENTS)
    return int(answer)


def ask_agent_config(index: int, default: AgentConfig | None = None) -> AgentConfig:
    """Step 3: Configure a single agent (persona, provider, model).

    Supports internal back navigation between sub-questions via Ctrl+Z.
    Temperature is fixed at 1.0 (required/expected by reasoning models).
    """
    console = Console()
    console.print(f"\n[bold]Configure Agent {index + 1}:[/bold]")

    # Track answers for back navigation
    d_persona = default.persona if default else None
    d_provider = default.provider if default else "openai"
    d_model = default.model if default else None
    d_temp = 1.0  # Fixed — reasoning models require or expect temperature 1.0
    d_name = default.name if default else None
    d_belief_file = default.belief_file if default else None

    sub = 0
    while sub < 4:
        try:
            if sub == 0:
                d_persona = _ask(questionary.select(
                    "Persona:",
                    choices=PERSONA_CHOICES,
                    default=d_persona,
                ), help_text=HELP_PERSONA)
            elif sub == 1:
                d_provider = _ask(questionary.select(
                    "Provider:",
                    choices=PROVIDER_CHOICES,
                    default=d_provider,
                ), help_text=HELP_PROVIDER)
            elif sub == 2:
                suggestions = MODEL_SUGGESTIONS.get(d_provider, [])
                default_model = d_model or (suggestions[0] if suggestions else "o4-mini")
                d_model = _ask(questionary.autocomplete(
                    "Model:",
                    choices=suggestions,
                    default=default_model,
                ), help_text=HELP_MODEL)
            elif sub == 3:
                use_belief = _ask(questionary.select(
                    "Load a pre-existing belief file? (Stage 1 will be skipped)",
                    choices=[
                        Choice("No", value="no"),
                        Choice("Yes", value="yes"),
                    ],
                    default="yes" if d_belief_file else "no",
                ))
                if use_belief == "yes":
                    d_belief_file = _ask(questionary.text(
                        "Path to belief JSON file:",
                        default=d_belief_file or "",
                        validate=lambda t: (
                            True if t.strip().endswith(".json") and Path(t.strip()).is_file()
                            else "File must exist and end in .json"
                        ),
                    ))
                else:
                    d_belief_file = None
            sub += 1
        except WizardBack:
            if sub > 0:
                sub -= 1
            else:
                raise

    name = d_name or f"Agent-{d_persona.capitalize()}"

    return AgentConfig(
        name=name,
        persona=d_persona,
        model=d_model,
        temperature=d_temp,
        provider=d_provider,
        belief_file=d_belief_file,
    )


def ask_num_rounds(default: int = 1) -> int:
    """Step 6: Number of debate rounds."""
    answer = _ask(questionary.text(
        "Number of debate rounds:",
        default=str(default),
        validate=lambda t: _validate_int_range(t, 1, 10),
    ), help_text=HELP_NUM_ROUNDS)
    return int(answer)


def _detect_balance_preset(logic: float, ethics: float) -> str:
    """Detect which balance preset matches the given weights."""
    if (logic, ethics) == (1.0, 0.0):
        return "pure_logic"
    if (logic, ethics) == (0.0, 1.0):
        return "pure_ethics"
    if (logic, ethics) == (0.5, 0.5):
        return "balanced"
    return "pure_logic"  # fallback to pure_logic for any legacy custom values


def ask_adjudicator_config(default: AdjudicationConfig | None = None) -> AdjudicationConfig:
    """Step 7: Adjudicator model and weights with internal back navigation."""
    adj = default or AdjudicationConfig()

    console = Console()
    console.print("\n[bold]Adjudicator Configuration:[/bold]")

    d_provider = adj.provider
    d_model = adj.model
    d_logic_sys = adj.logic_system
    d_ethics_sys = adj.ethics_system
    d_logic = adj.logic_weight
    d_ethics = adj.ethics_weight
    d_balance = _detect_balance_preset(d_logic, d_ethics)

    # Build Choice lists for logic and ethics system selectors
    # CLASSICAL_INFORMAL_BAYESIAN appears first (recommended default), NONE last
    _LOGIC_KEY_ORDER = ["CLASSICAL_INFORMAL_BAYESIAN"] + [
        k for k in LOGIC_SYSTEMS if k not in ("CLASSICAL_INFORMAL_BAYESIAN", "NONE")
    ] + ["NONE"]
    logic_sys_choices = [
        Choice(f"{get_logic_system_label(k)} — {_LOGIC_WIZARD_TAGS[k]}", value=k)
        for k in _LOGIC_KEY_ORDER
    ]
    ethics_sys_choices = [
        Choice(f"{get_ethics_system_label(k)} — {_ETHICS_WIZARD_TAGS[k]}", value=k)
        for k in _ETHICS_KEY_ORDER
    ]

    sub = 0
    while sub < 5:
        try:
            if sub == 0:
                d_provider = _ask(questionary.select(
                    "Adjudicator provider:",
                    choices=PROVIDER_CHOICES,
                    default=d_provider,
                ), help_text=HELP_PROVIDER)
            elif sub == 1:
                suggestions = MODEL_SUGGESTIONS.get(d_provider, [])
                default_model = d_model or (suggestions[0] if suggestions else "o4-mini")
                d_model = _ask(questionary.autocomplete(
                    "Adjudicator model:",
                    choices=suggestions,
                    default=default_model,
                ), help_text=HELP_MODEL)
            elif sub == 2:
                d_logic_sys = _ask(questionary.select(
                    "Logic system for adjudication:",
                    choices=logic_sys_choices,
                    default=d_logic_sys,
                ), help_text=HELP_ADJ_LOGIC_SYSTEM)
            elif sub == 3:
                d_ethics_sys = _ask(questionary.select(
                    "Ethics system for adjudication:",
                    choices=ethics_sys_choices,
                    default=d_ethics_sys,
                ), help_text=HELP_ADJ_ETHICS_SYSTEM)
            elif sub == 4:
                d_balance = _ask(questionary.select(
                    "How should the adjudicator balance logic and ethics?",
                    choices=[
                        Choice("Pure Logic (logic=1.0, ethics=0.0)", value="pure_logic"),
                        Choice("Balanced (logic=0.5, ethics=0.5)", value="balanced"),
                        Choice("Pure Ethics (logic=0.0, ethics=1.0)", value="pure_ethics"),
                    ],
                    default=d_balance,
                ), help_text=HELP_ADJ_BALANCE)
                if d_balance == "pure_logic":
                    d_logic, d_ethics = 1.0, 0.0
                elif d_balance == "pure_ethics":
                    d_logic, d_ethics = 0.0, 1.0
                else:
                    d_logic, d_ethics = 0.5, 0.5
            sub += 1
        except WizardBack:
            if sub > 0:
                sub -= 1
            else:
                raise

    return AdjudicationConfig(
        model=d_model,
        provider=d_provider,
        logic_weight=d_logic,
        ethics_weight=d_ethics,
        logic_system=d_logic_sys,
        ethics_system=d_ethics_sys,
    )


def ask_output_toggles(default: OutputConfig | None = None) -> dict[str, bool]:
    """Step 9: Which outputs to generate (multi-select checkbox).

    Returns:
        Dict mapping OutputConfig attribute names to bool values.
    """
    # Build choices with current defaults
    choices = []
    for label, attr, fallback in OUTPUT_TOGGLES:
        checked = getattr(default, attr, fallback) if default else fallback
        choices.append(Choice(label, value=attr, checked=checked))

    selected = _ask(questionary.checkbox(
        "Which outputs would you like?",
        choices=choices,
        instruction="(Use arrow keys to move, <space> to select, <a> to toggle)",
    ), help_text=HELP_OUTPUTS)

    # Build result: selected attrs are True, others False
    all_attrs = {attr for _, attr, _ in OUTPUT_TOGGLES}
    return {attr: (attr in selected) for attr in all_attrs}


def ask_parallelization(default: bool = True) -> bool:
    """Step 10: Ask whether to enable parallel API dispatch."""
    return _ask(questionary.confirm(
        "Enable parallel API dispatch? (recommended)",
        default=default,
    ), help_text=HELP_PARALLELIZATION)


def ask_max_workers(default: int = 5) -> int:
    """Step 10b: Ask how many threads to use for parallel dispatch."""
    answer = _ask(questionary.text(
        "Max parallel threads:",
        default=str(default),
    ), help_text=HELP_MAX_WORKERS)
    try:
        value = int(answer)
        if value < 1:
            return 1
        return value
    except ValueError:
        return default


def ask_api_keys_for_config(state: dict) -> None:
    """Step 11: Prompt for API keys for each provider used in this config.

    Collects providers from agent configs and adjudication.
    Skips providers that already have keys set in the environment and
    providers that don't need keys (e.g. ollama).
    """
    console = Console()

    # Collect unique providers
    providers: set[str] = set()
    for ac in state.get('agent_configs', []):
        providers.add(ac.provider)
    adj = state.get('adjudication')
    if adj:
        providers.add(adj.provider)

    # Filter to providers that need API keys
    providers = {p for p in providers if p in PROVIDER_ENV_VARS}

    if not providers:
        return

    console.print("\n[bold]API Key Configuration:[/bold]")
    console.print("  [dim]Keys are set for this session only (not persisted).[/dim]")

    for provider in sorted(providers):
        env_var = PROVIDER_ENV_VARS[provider]
        if os.environ.get(env_var):
            console.print(f"  [dim]{env_var} already set — skipping.[/dim]")
            continue

        console.print(f"\n  [yellow]![/yellow] {env_var} is not set.")
        keys: list[str] = []
        key_num = 1
        while True:
            prompt = (
                f"Enter your {provider.capitalize()} API key (or press Enter to skip):"
                if key_num == 1
                else f"Enter your {provider.capitalize()} API key #{key_num} (or press Enter to finish):"
            )
            answer = _ask(questionary.text(prompt), help_text=HELP_API_KEYS)
            if not answer.strip():
                break
            keys.append(answer.strip())
            console.print("  [green]>[/green] Key set.")
            add_more = _ask(questionary.confirm(
                f"Add another {provider.capitalize()} key for rate-limit rotation?",
                default=False,
            ), help_text=HELP_API_KEYS)
            if not add_more:
                break
            key_num += 1

        if keys:
            os.environ[env_var] = ",".join(keys)
            count_note = f" ({len(keys)} keys for rotation)" if len(keys) > 1 else ""
            console.print(f"  [green]>[/green] {env_var} set for this session{count_note}.")
        else:
            console.print(
                f"  [dim]Skipped. The debate may fail if {provider} calls are needed.[/dim]"
            )

    state['api_keys_configured'] = True


# =========================================================================
# Review panel
# =========================================================================

def show_review_panel(config: DebateConfig, console: Console) -> None:
    """Render a rich Panel summarizing the full debate configuration."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold", width=16)
    table.add_column("Value")

    table.add_row("Topic", config.topic)

    agent_lines = []
    for a in config.agents:
        persona_display = "(no persona)" if a.persona.upper() == "NONE" else a.persona
        line = f"{a.name} ({persona_display}, {a.provider}/{a.model})"
        if a.belief_file:
            line += " (custom belief)"
        agent_lines.append(line)
    table.add_row("Agents", "\n".join(agent_lines))

    table.add_row("Stage 3", "rebuttal")
    table.add_row("Rounds", str(config.max_rounds))
    table.add_row(
        "Adjudicator",
        f"{config.adjudication.model} ({config.adjudication.provider}), "
        f"logic={config.adjudication.logic_weight}, ethics={config.adjudication.ethics_weight}",
    )

    # Summarize enabled outputs
    enabled = []
    for label, attr, _ in OUTPUT_TOGGLES:
        if getattr(config.outputs, attr, False):
            enabled.append(label.split(" (")[0])  # strip parenthetical
    table.add_row("Outputs", ", ".join(enabled) if enabled else "None")

    if config.parallel.enabled:
        table.add_row("Parallelization", f"Enabled ({config.parallel.max_workers} threads)")
    else:
        table.add_row("Parallelization", "Disabled")

    panel = Panel(table, title="[bold #B8405E]Debate Configuration[/bold #B8405E]", border_style="#B8405E")
    console.print(panel)


def ask_review_action() -> str:
    """Ask user what to do after reviewing the configuration."""
    return _ask(questionary.select(
        "Proceed with this configuration?",
        choices=[
            Choice("Launch debate", value="launch"),
            Choice("Edit a setting", value="edit"),
            Choice("Save config to YAML", value="save"),
            Choice("Cancel", value="cancel"),
        ],
    ), help_text=HELP_REVIEW_ACTION)


def ask_edit_section() -> str:
    """Ask which config section to re-edit."""
    choices = [
        Choice("Topic", value="topic"),
        Choice("Number of agents", value="num_agents"),
        Choice("Agent configurations", value="agents"),
        Choice("Number of rounds", value="rounds"),
        Choice("Adjudicator", value="adjudicator"),
        Choice("Output toggles", value="outputs"),
        Choice("Parallelization", value="parallelization"),
    ]

    return _ask(questionary.select(
        "Which section would you like to edit?",
        choices=choices,
    ), help_text=HELP_EDIT_SECTION)


def _ask_save_path() -> Path:
    """Prompt for a YAML file path to save the config."""
    path_str = _ask(questionary.text(
        "Save config to (YAML path):",
        default="my_debate.yaml",
    ), help_text=HELP_SAVE_PATH)
    path = Path(path_str)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    return path


# =========================================================================
# Step machine — internal step functions for wizard navigation
# =========================================================================

def _populate_state_from_config(config: DebateConfig, state: dict) -> None:
    """Fill wizard state dict from a complete DebateConfig.

    Used when a preset is selected so all intermediate steps can be skipped.
    """
    state['prefill'] = config
    state['topic'] = config.topic
    state['num_agents'] = len(config.agents)
    state['agent_configs'] = list(config.agents)
    state['sub_options'] = {}
    state['max_rounds'] = config.max_rounds
    state['adjudication'] = config.adjudication
    state['output_flags'] = {
        attr: getattr(config.outputs, attr, fallback)
        for _, attr, fallback in OUTPUT_TOGGLES
    }
    state['parallel_enabled'] = config.parallel.enabled
    state['max_workers'] = config.parallel.max_workers


def _step_preset(state: dict) -> bool:
    """Step 0: Preset selection."""
    if state.get('prefill') is not None:
        return False  # skip when editing a loaded config

    preset = ask_preset()
    if preset is not None:
        _populate_state_from_config(preset, state)
        state['_preset_selected'] = True
    return True


def _step_topic(state: dict) -> bool:
    """Step 1: Debate topic."""
    pf = state.get('prefill')
    default = state.get('topic', '') or (pf.topic if pf else '')
    state['topic'] = ask_topic(default=default)
    return True


def _step_num_agents(state: dict) -> bool:
    """Step 2: Number of agents."""
    pf = state.get('prefill')
    default = state.get('num_agents') or (len(pf.agents) if pf else 2)
    state['num_agents'] = ask_num_agents(default=default)
    return True


def _step_agents(state: dict) -> bool:
    """Step 3: Agent configurations (loops internally with back support)."""
    num = state['num_agents']
    existing = state.get('agent_configs', [])
    pf = state.get('prefill')

    configs: list[AgentConfig] = []
    i = 0
    while i < num:
        default = None
        if i < len(existing):
            default = existing[i]
        elif pf and i < len(pf.agents):
            default = pf.agents[i]

        try:
            configs.append(ask_agent_config(i, default=default))
            i += 1
        except WizardBack:
            if i > 0:
                i -= 1
                configs.pop()
            else:
                raise

    state['agent_configs'] = configs
    return True


def _step_rounds(state: dict) -> bool:
    """Step 6: Number of rounds."""
    pf = state.get('prefill')
    default = state.get('max_rounds') or (pf.max_rounds if pf else 1)
    state['max_rounds'] = ask_num_rounds(default=default)
    return True


def _step_adjudicator(state: dict) -> bool:
    """Step 7: Adjudicator configuration."""
    pf = state.get('prefill')
    default = state.get('adjudication') or (pf.adjudication if pf else None)
    state['adjudication'] = ask_adjudicator_config(default=default)
    return True


def _step_outputs(state: dict) -> bool:
    """Step 9: Output toggles."""
    pf = state.get('prefill')
    default_output = None
    if state.get('output_flags'):
        default_output = OutputConfig(storage_dir=DEFAULT_STORAGE_DIR)
        for attr, val in state['output_flags'].items():
            setattr(default_output, attr, val)
    elif pf:
        default_output = pf.outputs
    state['output_flags'] = ask_output_toggles(default=default_output)
    return True


def _step_parallelization(state: dict) -> bool:
    """Step 10: Parallelization toggle + max workers."""
    default = state.get('parallel_enabled', True)
    state['parallel_enabled'] = ask_parallelization(default=default)
    if state['parallel_enabled']:
        default_workers = state.get('max_workers', 5)
        state['max_workers'] = ask_max_workers(default=default_workers)
    return True


def _step_api_keys(state: dict) -> bool:
    """Step 11: API key collection."""
    ask_api_keys_for_config(state)
    return True


def _build_config(state: dict) -> DebateConfig:
    """Build a DebateConfig from wizard state dict."""
    pf = state.get('prefill')

    output_flags = state.get('output_flags', {})
    outputs = OutputConfig(storage_dir=DEFAULT_STORAGE_DIR)
    for attr, val in output_flags.items():
        setattr(outputs, attr, val)

    return DebateConfig(
        name=f"Debate: {state['topic'][:50]}",
        topic=state['topic'],
        max_rounds=state['max_rounds'],
        agents=state['agent_configs'],
        adjudication=state['adjudication'],
        outputs=outputs,
        parallel=ParallelConfig(enabled=state.get('parallel_enabled', True), max_workers=state.get('max_workers', 5)),
    )


# =========================================================================
# Main wizard orchestrator
# =========================================================================

_WIZARD_STEPS = [
    _step_preset,          # 0: Preset selection
    _step_topic,           # 1: Debate topic
    _step_num_agents,      # 2: Number of agents
    _step_agents,          # 3: Agent configurations
    _step_rounds,          # 4: Number of rounds
    _step_adjudicator,     # 5: Adjudicator
    _step_outputs,         # 6: Output toggles
    _step_parallelization, # 7: Parallelization toggle
    _step_api_keys,        # 8: API key collection
]


def run_wizard(
    console: Console,
    prefill: DebateConfig | None = None,
) -> tuple[DebateConfig | None, str]:
    """Run the full interactive configuration wizard.

    Args:
        console: Rich console for output.
        prefill: Optional pre-existing config to use as defaults.

    Returns:
        (config, action) where action is one of:
        "launch", "save", "cancel".
        config is None if cancelled.
    """
    console.print("  [dim]Press Esc to exit  \u2022  Ctrl+Z to go back  \u2022  Ctrl+F1 for help[/dim]\n")

    # Outer loop allows returning to the main menu via Ctrl+Z at step 0
    while True:
        # ── Main menu (only when launched without prefill) ──
        if prefill is None:
            while True:
                try:
                    choice = ask_main_menu()
                except (WizardExit, WizardBack, KeyboardInterrupt):
                    return None, "cancel"

                if choice == "exit":
                    return None, "cancel"
                elif choice == "about":
                    console.print()
                    console.print(Panel(
                        ABOUT_CHAL,
                        title="[bold]About CHAL[/bold]",
                        border_style="#9B1B30",
                        expand=False,
                        width=80,
                    ))
                    console.print()
                    continue
                elif choice == "gauntlet":
                    console.print("\n[dim]Future content coming soon![/dim]\n")
                    continue
                elif choice == "debate":
                    break

        # State dict accumulates wizard answers across steps
        state: dict = {'prefill': prefill} if prefill else {}

        start = 1 if prefill else 0  # skip preset step when editing
        idx = start
        visited: list[int] = []  # stack of step indices that showed UI
        back_to_menu = False

        while idx < len(_WIZARD_STEPS):
            try:
                showed_ui = _WIZARD_STEPS[idx](state)
                if showed_ui is not False:
                    visited.append(idx)
                # Preset selected — state is fully populated, skip to review
                if state.get('_preset_selected'):
                    break
                idx += 1
            except WizardBack:
                if visited:
                    idx = visited.pop()
                elif prefill is None:
                    back_to_menu = True
                    break
                else:
                    console.print("[dim]Already at the first step.[/dim]")
            except (WizardExit, KeyboardInterrupt):
                return None, "cancel"

        if back_to_menu:
            continue  # restart outer loop -> main menu

        config = _build_config(state)

        # ----- Review & action loop -----
        while True:
            console.print()
            show_review_panel(config, console)

            try:
                action = ask_review_action()
            except (WizardExit, WizardBack, KeyboardInterrupt):
                return None, "cancel"

            if action == "launch":
                return config, "launch"

            elif action == "cancel":
                return None, "cancel"

            elif action == "save":
                try:
                    path = _ask_save_path()
                    config.to_yaml(path)
                    console.print(f"[green]Config saved to {path}[/green]")
                except (WizardExit, WizardBack, KeyboardInterrupt):
                    continue  # back to review

            elif action == "edit":
                try:
                    section = ask_edit_section()
                    config = _apply_edit(config, section, console)
                except (WizardExit, KeyboardInterrupt):
                    return None, "cancel"
                except WizardBack:
                    continue  # back to review


def _apply_edit(config: DebateConfig, section: str, console: Console) -> DebateConfig:
    """Re-run a single wizard step and update the config in place.

    Returns:
        Updated DebateConfig (may be a new instance for some fields).
    """
    if section == "topic":
        config.topic = ask_topic(default=config.topic)
        config.name = f"Debate: {config.topic[:50]}"

    elif section == "num_agents":
        new_count = ask_num_agents(default=len(config.agents))
        if new_count < len(config.agents):
            config.agents = config.agents[:new_count]
        elif new_count > len(config.agents):
            for i in range(len(config.agents), new_count):
                config.agents.append(ask_agent_config(i))

    elif section == "agents":
        # Let user pick which agent to edit
        agent_choices = [
            Choice(f"{a.name} ({a.persona})", value=i)
            for i, a in enumerate(config.agents)
        ]
        idx = _ask(questionary.select(
            "Which agent to edit?",
            choices=agent_choices,
        ), help_text=HELP_PERSONA)
        config.agents[idx] = ask_agent_config(idx, default=config.agents[idx])

    elif section == "rounds":
        config.max_rounds = ask_num_rounds(default=config.max_rounds)

    elif section == "adjudicator":
        config.adjudication = ask_adjudicator_config(default=config.adjudication)

    elif section == "outputs":
        output_flags = ask_output_toggles(default=config.outputs)
        for attr, val in output_flags.items():
            setattr(config.outputs, attr, val)

    elif section == "parallelization":
        enabled = ask_parallelization(default=config.parallel.enabled)
        workers = config.parallel.max_workers
        if enabled:
            workers = ask_max_workers(default=config.parallel.max_workers)
        config.parallel = ParallelConfig(enabled=enabled, max_workers=workers)

    return config
