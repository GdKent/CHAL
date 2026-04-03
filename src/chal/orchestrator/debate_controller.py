"""
debate_controller.py

Orchestrates structured multi-agent philosophical debates using the CHAL Belief Schema.

The DebateController manages a 7-stage debate process:
- Stage 0: Briefing - Initialize agents with personas and universal rules
- Stage 1: Opening Positions - Agents state initial beliefs as structured JSON (CBS)
- Stage 2: Cross-Examination - Agents ask targeted questions about opponents' claims/assumptions
- Stage 3: Rebuttals - Agents respond to questions with structured answers and optional belief patches
- Stage 4: Adjudication - Independent evaluator assesses challenge-rebuttal pairs
- Stage 5: Belief Updates - Agents revise beliefs based on adjudication outcomes
- Stage 6: Concluding Remarks - Agents synthesize their positions and concessions
- Stage 7: Scribing - Generate a flowing narrative synthesis of the entire debate

Features:
- Structured belief tracking with JSON schemas (CBS)
- Embedding-based belief trajectory visualization
- Configurable adjudication weights (logic vs. ethics)
- Token-optimized prompts (JSON-only responses, Markdown generated programmatically)
- Round-robin debate structure with multiple rounds

Note: All belief outputs are JSON-first. Human-readable Markdown is generated
programmatically using belief_to_markdown() to minimize token usage.
"""

from typing import Callable, List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from chal.agents.base import Agent, Message
from chal.agents import prompts
from chal.agents.factory import create_agent
from chal.agents.logic_systems import get_logic_system, get_logic_system_description
from chal.agents.ethics_systems import get_ethics_system, get_ethics_system_description
from chal.orchestrator.adjudicator import Adjudicator
from chal.orchestrator.moderator import Moderator
from chal.utilities.utils import parse_challenges, parse_structured_rebuttals_numbered, initialize_agent_stats, update_agent_stats, display_agent_stats, calculate_performance_scores, get_performance_summary, validate_stage2_questions
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.convergence import calculate_claim_agreement, format_convergence_summary, get_convergence_trajectory_summary
from chal.beliefs.io import parse_model_output_to_belief, belief_to_markdown
from chal.beliefs.io import project_for_embedding
from chal.beliefs.graph_visualizer import export_debate_graph
from chal.config import DebateConfig, AdjudicationConfig
from chal.utilities.training_data import DebateRecorder
from chal.utilities.reporting import generate_analysis_report, generate_analysis_json
from chal.utilities.parallel import ParallelDispatcher, WorkItem
import tiktoken
import json
import re

# === Path Configuration ===
# Get the project root directory (CHAL/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "src" / "chal" / "storage"

class DebateController:
    """
    Orchestrates structured philosophical debates between multiple LLM agents.

    The controller manages the complete debate lifecycle through 7 stages, maintains
    agent beliefs using the CHAL Belief Schema (CBS), tracks embeddings for
    visualization, and coordinates adjudication of challenge-rebuttal exchanges.

    Attributes:
        agents: List of participating Agent instances
        max_rounds: Number of debate rounds (each round includes Stages 2-5)
        challenge_rebuttal_pairs: All challenge-rebuttal exchanges
        opening_positions: Initial belief statements from Stage 1
        full_transcript: Complete debate transcript as list of strings
        round_histories: Message history per round for context management
        scribe_agent: Optional agent for generating narrative synthesis
    """
    def __init__(self, agents: List[Agent], max_rounds: int = 3, config: Optional[DebateConfig] = None,
                 key_pool=None):
        """
        Initializes the DebateController with a list of agents and a number of debate rounds.

        Args:
            agents (List[Agent]): A list of LLM-powered agents participating in the debate.
            max_rounds (int): Number of complete debate rounds (each consisting of Stages 2-5).
            config (Optional[DebateConfig]): Configuration object containing all debate parameters.
                If None, a default configuration will be created.
            key_pool: Optional KeyPool instance for multi-key API key rotation.
                When provided, internal agents (adjudicator, scribe) also get the pool.
        """
        self.agents = agents
        self.max_rounds = config.max_rounds if config and hasattr(config, 'max_rounds') else max_rounds
        self.config = config  # Store config for accessing stage and scribe parameters
        self.topic = config.topic if config else ""
        self.key_pool = key_pool

        # Parallel dispatcher — uses config when available, else sequential
        par_enabled = config.parallel.enabled if config else False
        par_workers = config.parallel.max_workers if config else 5
        self.dispatcher = ParallelDispatcher(max_workers=par_workers, enabled=par_enabled)

        self.challenge_rebuttal_pairs = []
        self.opening_positions = []

        # Separate tracking for debug log vs clean markdown transcript
        self.debug_log: List[str] = []  # Comprehensive debug log with all model outputs, prompts, and processing details
        self.markdown_transcript: List[str] = []  # Clean markdown-only output for human reading
        self.full_transcript: List[str] = []  # Legacy: keep for backward compatibility, will contain markdown

        self.round_histories: dict[str, List[Message]] = {} # A dictionary of full debate rounds for tracking and memory efficient prompting
        self.current_round_key = None  # Tracks the active round name like "round-1"
        self.last_challenges: dict[str, dict[str, str]] = {} # A dictionary of challenges issued (Stage 2): {challenger: {target: challenge}}
        self.last_rebuttals: dict[str, str] = {} # A dictionary of rebuttals per agent (Stage 3): {agent_name: combined rebuttal}
        self.previous_rounds_challenges: dict[str, list] = {}  # Track challenges from previous rounds for anti-repetition: {agent_name: [challenges]}
        self.resolution_outcomes: dict[str, dict[str, str]] = {} # A dictionary of adjudicated outcomes per agent (Stage 4): {agent_name: {challenge_text: resolution_result}}

        # Instantiate the adjudicator agent
        adj_cfg = config.adjudication if config else AdjudicationConfig()
        logic_sys_dict = get_logic_system(adj_cfg.logic_system)
        ethics_sys_dict = get_ethics_system(adj_cfg.ethics_system)
        adjudicator_prompt = prompts.build_adjudicator_prompt(
            logic_weight=adj_cfg.logic_weight,
            ethics_weight=adj_cfg.ethics_weight,
            logic_sys=logic_sys_dict,
            ethics_sys=ethics_sys_dict,
        )
        adjudicator_agent = create_agent(
            name="Adjudicator",
            model=adj_cfg.model,
            provider=adj_cfg.provider,
            system_prompt=adjudicator_prompt,
            key_pool=self.key_pool,
        )
        self.adjudicator = Adjudicator(
            adjudicator_agent,
            logic_weight=adj_cfg.logic_weight,
            ethics_weight=adj_cfg.ethics_weight,
            logic_sys=get_logic_system_description(adj_cfg.logic_system),
            ethics_sys=get_ethics_system_description(adj_cfg.ethics_system),
        )

        # Instantiate the scribe agent
        scribe_model = config.scribe.model if config else "gpt-4o"
        scribe_provider = config.adjudication.provider if config else "openai"
        self.scribe_agent = create_agent(
            name="Scribe",
            model=scribe_model,
            provider=scribe_provider,
            system_prompt="",
            key_pool=self.key_pool,
        )

        # Initialize agent statistics
        self.agent_stats = initialize_agent_stats([agent.name for agent in self.agents])

        # Initialize convergence tracking
        self.convergence_history = []

        # Initialize moderator (if stage2_mode == "moderated")
        self.stage2_mode = config.stage2_mode if config else "open"
        self.moderator: Optional[Moderator] = None
        self.roadmap = None
        self.roadmap_revisions: List[Dict[str, Any]] = []
        if self.stage2_mode == "moderated" and config:
            self.moderator = Moderator(config.moderator)

        # Training data recorder (initialized in run() if save_training_data is enabled)
        self.recorder: Optional[DebateRecorder] = None

        # Progress callback (set by run() caller, e.g. DebateDisplay.handle_event)
        self._progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        # Error recovery callback (set by run() caller)
        self._on_error: Optional[Callable[[str, Exception, int], str]] = None

    def _call_agent(self, agent, messages, agent_name: str = "", **kwargs) -> Optional[str]:
        """Call agent.generate() with error recovery via on_error callback.

        Args:
            agent: The Agent instance to call.
            messages: Messages to pass to agent.generate().
            agent_name: Display name for error reporting.
            **kwargs: Additional kwargs for agent.generate().

        Returns:
            The agent response string, or None if skipped.

        Raises:
            Exception: Re-raises if on_error returns "abort" or is not set.
        """
        retry_count = 0
        name = agent_name or getattr(agent, "name", "unknown")
        while True:
            try:
                return agent.generate(messages, **kwargs)
            except Exception as e:
                if self._on_error is None:
                    raise
                action = self._on_error(name, e, retry_count)
                if action == "retry":
                    retry_count += 1
                    continue
                elif action == "skip":
                    self._log(f"Skipped {name} due to error: {e}", "WARNING")
                    return None
                else:  # "abort" or unknown
                    raise

    def _notify(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Fire a progress event if a callback is registered.

        Args:
            event: Event name (e.g. "stage_start", "agent_complete").
            data: Event-specific payload dict.
        """
        if self._progress_callback is not None:
            self._progress_callback(event, data or {})

    def _log(self, message: str, level: str = "INFO", include_timestamp: bool = True) -> None:
        """
        Add a message to the debug log with formatting.

        Args:
            message: The log message
            level: Log level (INFO, DEBUG, PROMPT, RESPONSE, PARSE, ERROR)
            include_timestamp: Whether to include timestamp
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] [{level}] {message}"
        else:
            log_entry = f"[{level}] {message}"
        self.debug_log.append(log_entry)

    def _log_header(self, title: str, char: str = "=") -> None:
        """Add a formatted header to the debug log."""
        separator = char * 80
        self.debug_log.append(f"\n{separator}")
        self.debug_log.append(f"{title}")
        self.debug_log.append(f"{separator}\n")

    def _log_prompt(self, agent_name: str, prompt: str, stage: str = "") -> None:
        """Log a prompt being sent to a model."""
        stage_info = f" - {stage}" if stage else ""
        self._log_header(f"PROMPT TO {agent_name}{stage_info}", "-")
        self.debug_log.append(prompt)
        self.debug_log.append("\n" + "-" * 80 + "\n")

    def _log_response(self, agent_name: str, response: str, stage: str = "") -> None:
        """Log a raw model response."""
        stage_info = f" - {stage}" if stage else ""
        self._log_header(f"RAW RESPONSE FROM {agent_name}{stage_info}", "-")
        self.debug_log.append(response)
        self.debug_log.append("\n" + "-" * 80 + "\n")

    def _log_parse_result(self, success: bool, details: str) -> None:
        """Log the result of parsing a model response."""
        status = "SUCCESS" if success else "FAILURE"
        self._log(f"PARSE {status}: {details}", level="PARSE")

    def _add_to_markdown(self, content: str) -> None:
        """Add content to the clean markdown transcript."""
        self.markdown_transcript.append(content)

    def _retry_on_parse_failure(self, generate_fn, is_valid_fn, stage_label, agent_name,
                                 initial_result=None):
        """Retry LLM call when output fails parse validation.

        For parallel-dispatch stages: pass initial WorkResult. If it already
        contains valid output, returns immediately with no retry.

        For sequential stages (or when initial_result is None): calls
        generate_fn() up to (1 + max_retries) times.

        Returns:
            result dict, or None if all attempts exhausted and no usable result exists.
        """
        max_retries = self.config.stages.parse_retries if self.config else 3
        last_result = None

        # Check initial parallel result
        if initial_result is not None:
            if initial_result.error is None:
                if is_valid_fn(initial_result.result):
                    return initial_result.result
                last_result = initial_result.result
                self._log(f"[{stage_label}] Output validation failed for {agent_name}, retrying", "WARN")
            else:
                self._log(f"[{stage_label}] Initial call error for {agent_name}: {initial_result.error}", "WARN")

        # Retry loop
        for attempt in range(1, max_retries + 1):
            self._log(f"[{stage_label}] Parse retry {attempt}/{max_retries} for {agent_name}", "WARN")
            try:
                result = generate_fn()
                if is_valid_fn(result):
                    self._log(f"[{stage_label}] Retry {attempt} succeeded for {agent_name}", "INFO")
                    return result
                last_result = result
                self._log(f"[{stage_label}] Retry {attempt} output still invalid for {agent_name}", "WARN")
            except Exception as e:
                self._log(f"[{stage_label}] Retry {attempt} error for {agent_name}: {e}", "ERROR")

        self._log(f"[{stage_label}] All {max_retries} retries exhausted for {agent_name}", "ERROR")
        return last_result
        self.full_transcript.append(content)  # Also add to legacy transcript

    def _positions_agree(self, agent_a: str, agent_b: str) -> bool:
        """
        Placeholder method to determine whether two agents broadly agree.

        Returns:
            bool: True if agents are considered aligned.
        """
        # TODO: In future, this could compare current_positions via embeddings or heuristics
        return False

    def _build_round_summary(
        self,
        round_num: int,
        round_idx: int,
        focus_subtopic: Optional[Dict[str, Any]],
        convergence_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a structured summary of a completed round for the adaptive moderator.

        Args:
            round_num: 1-based round number.
            round_idx: 0-based round index.
            focus_subtopic: The sub-topic dict for this round (or None in open mode).
            convergence_data: Convergence metrics dict from this round (or None).

        Returns:
            A dict with keys: round_num, focus_subtopic, adjudication,
            belief_changes, convergence, key_arguments, remaining_rounds,
            completed_subtopics.
        """
        # --- Adjudication summary ---
        # Collect verdicts from this round's challenge_rebuttal_pairs
        verdict_counts: Dict[str, int] = {}
        key_arguments: list[str] = []
        for entry in self.challenge_rebuttal_pairs:
            resolution = entry.get("resolution")
            if resolution and isinstance(resolution, dict):
                status = resolution.get("status", "unknown")
                verdict_counts[status] = verdict_counts.get(status, 0) + 1
                # Extract a brief key argument from each pair
                challenger = entry.get("challenger", "?")
                challenge_text = entry.get("challenge", "")
                if challenge_text:
                    truncated = challenge_text[:200]
                    key_arguments.append(f"{challenger} challenged: {truncated}")

        # Determine round winner from adjudication results
        winner = None
        if verdict_counts:
            # critique_valid means the challenger won; rebuttal_valid means the target defended
            critique_wins = verdict_counts.get("critique_valid", 0)
            rebuttal_wins = verdict_counts.get("rebuttal_valid", 0)
            if critique_wins > rebuttal_wins:
                winner = "challengers"
            elif rebuttal_wins > critique_wins:
                winner = "defenders"

        adjudication_summary = {
            "verdict_counts": verdict_counts,
            "winner": winner,
        }

        # --- Belief changes ---
        belief_changes: list[Dict[str, Any]] = []
        for agent in self.agents:
            beliefs_held = getattr(agent, "all_beliefs_held", [])
            changed = len(beliefs_held) > 1  # More than one belief snapshot = changed
            change_summary = ""
            if changed and len(beliefs_held) >= 2:
                change_summary = f"Updated belief after round {round_num}"
            belief_changes.append({
                "agent": agent.name,
                "changed": changed,
                "change_summary": change_summary,
            })

        # --- Convergence ---
        convergence_summary: Optional[Dict[str, Any]] = None
        if convergence_data is not None:
            score = convergence_data.get("convergence_score", 0.0)
            # Determine trend from history
            trend = "stable"
            if len(self.convergence_history) >= 2:
                prev_score = self.convergence_history[-2].get("convergence_score", 0.0)
                curr_score = self.convergence_history[-1].get("convergence_score", 0.0)
                if curr_score > prev_score + 0.05:
                    trend = "converging"
                elif curr_score < prev_score - 0.05:
                    trend = "diverging"
            convergence_summary = {
                "score": score,
                "trend": trend,
            }

        # --- Completed subtopics ---
        completed_subtopics: list[str] = []
        if self.roadmap:
            for i in range(min(round_num, len(self.roadmap.sub_topics))):
                completed_subtopics.append(self.roadmap.sub_topics[i].title)

        # --- Remaining rounds ---
        remaining_rounds = self.max_rounds - round_num

        # Truncate key_arguments to avoid bloating the prompt
        key_arguments = key_arguments[:6]

        return {
            "round_num": round_num,
            "focus_subtopic": focus_subtopic,
            "adjudication": adjudication_summary,
            "belief_changes": belief_changes,
            "convergence": convergence_summary,
            "key_arguments": key_arguments,
            "remaining_rounds": remaining_rounds,
            "completed_subtopics": completed_subtopics,
        }

    def run_stage_0_briefing(self, topic: str, personas: dict) -> None:
        """
        Stage 0: Briefing

        Instantiates each agent with:
        - A universal debate prompt defining shared reasoning rules
        - A unique persona prompt that defines the agent's worldview or position

        Args:
            topic (str): The central debate topic (e.g., "Does free will exist?")
            personas (dict[str, str]): Mapping of agent name to persona string,
                e.g., {"Agent-A": prompts.EMPIRICIST, "Agent-B": prompts.SKEPTIC}

        Side Effect:
            - Updates each agent's system prompt
            - Stores a silent briefing marker in self.history
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 0: BRIEFING")
        self._log("Initializing agents with personas and universal rules", "INFO")
        self._log(f"Topic: {topic}", "INFO")
        self._log(f"Number of agents: {len(self.agents)}", "INFO")

        # Shared system prompt
        universal = prompts.build_universal_prompt(topic)
        self._log("Generated universal debate prompt", "DEBUG")
        self.debug_log.append(f"\n--- UNIVERSAL PROMPT ---\n{universal}\n--- END UNIVERSAL PROMPT ---\n")

        for agent in self.agents:
            self._log(f"Configuring agent: {agent.name}", "INFO")

            # Apply universal reasoning rules
            agent.receive_system_prompt(universal)
            self._log(f"  ✓ Applied universal prompt to {agent.name}", "DEBUG")

            # Apply agent-specific persona card
            persona = personas.get(agent.name, "")
            self._log(f"  Persona for {agent.name}: {persona[:50]}..." if len(persona) > 50 else f"  Persona for {agent.name}: {persona}", "DEBUG")
            role_card = prompts.build_position_prompt(agent.name, persona)
            self.debug_log.append(f"\n--- ROLE CARD FOR {agent.name} ---\n{role_card}\n--- END ROLE CARD ---\n")
            agent.receive_role_card(role_card)
            self._log(f"  ✓ Applied role card to {agent.name}", "DEBUG")

        self._log("Stage 0 complete - all agents briefed", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "# 🧠 Stage 0: Briefing\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 0, "name": "Briefing"})

    def run_stage_1_opening_positions(self, topic: str) -> None:
        """
        Stage 1: Opening Positions

        Each agent states their internal beliefs, a-priori assumptions, supporting reasons,
        and relevant implications. These are stored as belief objects with a JSON structure
        and used as the foundation for later rounds.

        Args:
            topic (str): The central debate topic.

        Returns:
            opening_positions (dict[str, str]): A mapping from agent name to their full opening position text.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 1: OPENING POSITIONS")
        self._log(f"Requesting opening belief statements from {len(self.agents)} agents", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 📖 Stage 1: Opening Positions\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 1, "name": "Opening Positions"})

        self.round_histories["round-0"] = []
        self.current_round_key = "round-0"

        # --- GATHER: Fire all opening-position calls in parallel ---
        items = [
            WorkItem(
                key=agent.name,
                callable=lambda a=agent: _generate_opening_position(a, topic),
                context={"agent": agent},
            )
            for agent in self.agents
        ]
        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic agent order ---
        for agent in self.agents:
            work_result = results[agent.name]
            self._log(f"\n--- Processing agent: {agent.name} ---", "INFO")

            # Handle dispatcher-level error (e.g. unhandled exception in pure function)
            if work_result.error is not None:
                self._log(f"Stage 1 error for {agent.name}: {work_result.error}", "ERROR")
                md_view = f"[Error generating opening position for {agent.name}]: {work_result.error}"
                agent.set_internal_belief(md_view)
                agent.all_beliefs_held.append(md_view)
                markdown_content = f"\n## {agent.name} - Opening Statement\n\n{md_view}\n"
                self._add_to_markdown(markdown_content)
                self._notify("agent_complete", {"agent_name": agent.name, "stage": 1, "action": "Opening statement failed"})
                continue

            r = work_result.result  # dict from _generate_opening_position
            response = r["response"]
            belief_obj = r["belief_obj"]
            md_view = r["md_view"]
            errs = r["errs"]
            opening_prompt = r["prompt"]

            # Replay logs from the pure function
            self._log_prompt(agent.name, opening_prompt, "Stage 1 - Opening Position")
            self._log(f"Received response from {agent.name} ({len(response.content)} chars)", "INFO")
            self._log_response(agent.name, response.content, "Stage 1 - Opening Position")

            # Replay validation logs
            for log_entry in r.get("validation_logs", []):
                self._log(log_entry["message"], log_entry.get("level", "INFO"))

            if belief_obj is not None:
                if errs:
                    self._log(f"Schema validation warnings for {agent.name} (non-blocking): {errs}", "WARN")
                self._log_parse_result(True, f"Parsed CBS belief for {agent.name}")
                self.debug_log.append(f"\n--- PARSED BELIEF JSON FOR {agent.name} ---\n{json.dumps(belief_obj, ensure_ascii=False, indent=2)}\n--- END PARSED JSON ---\n")

                graph_metrics = r.get("graph_metrics")
                if graph_metrics:
                    self._log(f"Graph metrics: {graph_metrics['total_nodes']} nodes, {graph_metrics['total_edges']} edges, {graph_metrics['critical_path_count']} critical paths", "INFO")

                agent.set_internal_belief_obj(belief_obj)                   # store structured JSON (auto-rebuilds graph)
                # Generate Markdown from JSON (no longer requested from model)
                md_view = belief_to_markdown(belief_obj)
                self._log(f"Generated Markdown view from JSON ({len(md_view)} chars)", "DEBUG")
                agent.set_internal_belief(md_view)  # keep human-readable string too
                agent.all_beliefs_held.append(json.dumps(belief_obj, ensure_ascii=False, indent=2) if belief_obj else "") # track beliefs
            else:
                # True fallback: no belief object parsed at all
                self._log_parse_result(False, f"Failed to parse CBS belief for {agent.name}. No belief object returned")
                self._log("Falling back to raw response content", "WARN")
                md_view = re.sub(r"```(?:json)?\s*", "", response.content).replace("```", "").strip()
                agent.set_internal_belief(md_view)
                agent.all_beliefs_held.append(md_view) # track beliefs

            # Log result
            self.round_histories[self.current_round_key].append(response)

            # Record training data
            if self.recorder:
                self.recorder.record_belief_formation(
                    agent_id=agent.name,
                    inputs={
                        "topic": topic,
                        "persona": getattr(agent, 'persona_label', ''),
                    },
                    belief=belief_obj,
                    raw_response=response.content,
                )

            # Add to markdown transcript
            markdown_content = f"\n## {agent.name} - Opening Statement\n\n{md_view}\n"
            self._add_to_markdown(markdown_content)
            self._notify("agent_complete", {"agent_name": agent.name, "stage": 1, "action": "Opening statement received"})

        self.opening_positions = [agent.internal_belief for agent in self.agents]
        self._log(f"Stage 1 complete - {len(self.opening_positions)} opening positions captured", "INFO")

        return

    def run_stage_2_cross_examination(self, only_if_disagree: bool = False, focus_subtopic: dict = None) -> list:
        """
        Stage 2: Cross-Examination

        Each agent critiques every other agent's current position. Multiple critiques per
        agent-pair are supported, and each is recorded as a separate entry.

        Structure of each entry in `self.challenge_rebuttal_pairs`:
            {
                "challenger": "Agent-A",
                "target": "Agent-B",
                "challenge": "<individual critique text>",
                "rebuttal": None,
                "resolution": None
            }

        Args:
            only_if_disagree (bool): Skip challenge generation if agents broadly agree.
            focus_subtopic (dict, optional): Sub-topic dict with 'title', 'description',
                'guiding_questions' for moderated mode. None for open mode.

        Returns:
            list[dict]: A flat list of critique records, one per challenge.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 2: CROSS-EXAMINATION")
        self._log(f"Starting cross-examination phase with only_if_disagree={only_if_disagree}", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚔️ Stage 2: Cross-Examination\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 2, "name": "Cross-Examination"})

        self.challenge_rebuttal_pairs = []

        # Build all challenger→target pairs (excluding self, and optionally skipping agreements)
        topic = self.topic if hasattr(self, "topic") else "<topic>"
        pairs = []
        for challenger in self.agents:
            for target in self.agents:
                if challenger.name == target.name:
                    self._log(f"Skipping self-examination: {challenger.name} == {target.name}", "DEBUG")
                    continue
                if only_if_disagree and self._positions_agree(challenger.name, target.name):
                    self._log(f"Agents agree - skipping: {challenger.name} ↔ {target.name}", "INFO")
                    markdown_note = f"\n*{challenger.name} and {target.name} broadly agree - skipping cross-examination*\n"
                    self._add_to_markdown(markdown_note)
                    self._notify("agent_complete", {"agent_name": challenger.name, "stage": 2, "action": f"Agrees with {target.name} — skipping"})
                    continue
                pairs.append((challenger, target))

        # --- GATHER: Fire all cross-examination calls in parallel ---
        items = [
            WorkItem(
                key=f"{c.name}→{t.name}",
                callable=lambda c=c, t=t: _generate_cross_examination(
                    challenger=c, target=t, topic=topic, config=self.config,
                    previous_challenges=self.previous_rounds_challenges.get(f"{c.name}→{t.name}", []),
                    focus_subtopic=focus_subtopic,
                ),
                context={"challenger": c, "target": t},
            )
            for c, t in pairs
        ]
        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic pair order ---
        for challenger, target in pairs:
            pair_key = f"{challenger.name}→{target.name}"
            work_result = results[pair_key]
            challenger_name = challenger.name
            target_name = target.name

            self._log(f"\n--- Cross-examination: {challenger_name} → {target_name} ---", "INFO")

            r = self._retry_on_parse_failure(
                generate_fn=lambda c=challenger, t=target: _generate_cross_examination(
                    challenger=c, target=t, topic=topic, config=self.config,
                    previous_challenges=self.previous_rounds_challenges.get(f"{c.name}→{t.name}", []),
                    focus_subtopic=focus_subtopic,
                ),
                is_valid_fn=_validate_cross_exam_result,
                stage_label="Stage 2",
                agent_name=pair_key,
                initial_result=work_result,
            )

            if r is None:
                self._notify("agent_complete", {"agent_name": challenger_name, "stage": 2, "action": f"Error questioning {target_name}"})
                continue

            response = r["response"]
            prompt = r["prompt"]
            questions = r["questions"]
            parsed_challenges = r["parsed_challenges"]
            ch_belief_obj = r["ch_belief_obj"]
            tg_belief_obj = r["tg_belief_obj"]

            # Replay logs
            self._log_prompt(challenger_name, prompt, f"Stage 2 - Questioning {target_name}")
            self._log(f"Received response ({len(response.content)} chars)", "INFO")
            self._log_response(challenger_name, response.content, f"Stage 2 - Questions for {target_name}")

            self.round_histories[self.current_round_key].append(response)

            # Fallback to legacy parser if needed (keeps backward compat)
            if not questions:
                self._log_parse_result(False, "No structured questions found, trying legacy parser")
                self._log(f"Legacy parser found {len(parsed_challenges)} challenges", "INFO")
                for challenge in parsed_challenges:
                    self.challenge_rebuttal_pairs.append({
                        "challenger": challenger_name,
                        "target": target_name,
                        "challenge": challenge,
                        "rebuttal": None,
                        "resolution": None
                    })
            else:
                self._log_parse_result(True, f"Successfully parsed {len(questions)} structured questions")
                self.debug_log.append(f"\n--- PARSED QUESTIONS JSON ({challenger_name} → {target_name}) ---\n{json.dumps(questions, ensure_ascii=False, indent=2)}\n--- END QUESTIONS ---\n")

                # Store structured questions
                for q in questions:
                    self.challenge_rebuttal_pairs.append({
                        "challenger": challenger_name,
                        "target": target_name,
                        "challenge": q.get("text", "").strip(),       # human-readable question
                        "qid": q.get("qid"),                           # Q1, Q2, ...
                        "target_ids": q.get("target_ids", []),         # ["C3","A1"]
                        "attack_type": q.get("attack_type", ""),       # undermining|rebutting|undercutting
                        "attack_strategy": q.get("attack_strategy", ""),  # specific strategy
                        "rebuttal": None,
                        "resolution": None
                    })

            num_questions = len(questions) if questions else len(parsed_challenges)
            self._log(f"Stored {num_questions} challenge(s) from {challenger_name} to {target_name}", "INFO")

            # Record training data
            if self.recorder:
                challenge_texts = [q.get("text", "").strip() for q in questions] if questions else parsed_challenges
                self.recorder.record_cross_examination(
                    agent_id=challenger_name,
                    target_id=target_name,
                    inputs={
                        "own_belief": ch_belief_obj,
                        "opponent_belief": tg_belief_obj,
                    },
                    challenges=challenge_texts,
                    raw_response=response.content,
                )

            # Add to markdown transcript
            markdown_section = f"\n## {challenger_name} questions {target_name}\n\n"
            if questions:
                for q in questions:
                    markdown_section += f"**{q.get('qid', 'Q?')}**: {q.get('text', '').strip()}\n\n"
                    if q.get('target_ids'):
                        markdown_section += f"  *Targets: {', '.join(q.get('target_ids', []))}*\n\n"
                    attack_type = q.get('attack_type', '')
                    attack_strategy = q.get('attack_strategy', '')
                    if attack_type:
                        markdown_section += f"  *Attack: {attack_type} / {attack_strategy}*\n\n"
            else:
                markdown_section += response.content + "\n\n"

            self._add_to_markdown(markdown_section)
            self._notify("agent_complete", {"agent_name": challenger_name, "stage": 2, "action": f"Generated {num_questions} question(s) for {target_name}"})

        return self.challenge_rebuttal_pairs

    def run_stage_3_rebuttals(self) -> list:
        """
        Stage 3: Rebuttals

        Each agent is presented with all critiques directed at them,
        and asked to respond to each critique individually.

        Updates:
            Each entry in self.challenge_rebuttal_pairs has its "rebuttal" field filled.

        Returns:
            list[dict]: Updated list of challenge-rebuttal-resolution entries.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 3: REBUTTALS")
        self._log("Starting rebuttal phase - agents respond to questions", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 🛡️ Stage 3: Rebuttals\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 3, "name": "Rebuttals"})

        # Group all challenges targeting each agent
        grouped_entries = {}
        for entry in self.challenge_rebuttal_pairs:
            target = entry["target"]
            grouped_entries.setdefault(target, []).append(entry)

        self._log(f"Grouped {len(self.challenge_rebuttal_pairs)} challenges into {len(grouped_entries)} targets", "INFO")

        self.last_rebuttals = {}
        self.last_rebuttals_patches = {}

        topic = self.topic if hasattr(self, "topic") else "<topic>"

        # Identify agents that have challenges to rebut
        agents_with_entries = []
        for target_agent in self.agents:
            relevant_entries = grouped_entries.get(target_agent.name)
            if not relevant_entries:
                self._log(f"No questions for {target_agent.name}, skipping", "INFO")
                continue
            agents_with_entries.append((target_agent, relevant_entries))

        # --- GATHER: Fire all rebuttal calls in parallel ---
        items = [
            WorkItem(
                key=target_agent.name,
                callable=lambda ta=target_agent, re=relevant_entries: _generate_rebuttal(
                    target_agent=ta, relevant_entries=re, topic=topic, config=self.config,
                ),
                context={"target_agent": target_agent, "relevant_entries": relevant_entries},
            )
            for target_agent, relevant_entries in agents_with_entries
        ]
        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic agent order ---
        for target_agent, relevant_entries in agents_with_entries:
            target_name = target_agent.name
            work_result = results[target_name]

            self._log(f"\n--- Processing rebuttals for: {target_name} ---", "INFO")
            self._log(f"Agent faces {len(relevant_entries)} question(s)", "INFO")

            r = self._retry_on_parse_failure(
                generate_fn=lambda ta=target_agent, re=relevant_entries: _generate_rebuttal(
                    target_agent=ta, relevant_entries=re, topic=topic, config=self.config,
                ),
                is_valid_fn=lambda r: bool(r.get("rebuttals")),
                stage_label="Stage 3",
                agent_name=target_name,
                initial_result=work_result,
            )

            if r is None:
                self._notify("agent_complete", {"agent_name": target_name, "stage": 3, "action": "Rebuttal generation failed"})
                continue
            response = r["response"]
            prompt = r["prompt"]
            rebuttals = r["rebuttals"]
            patches = r["patches"]
            received_questions_json = r["received_questions_json"]
            tgt_belief_obj = r["tgt_belief_obj"]

            # Replay logs
            self.debug_log.append(f"\n--- QUESTIONS PAYLOAD FOR {target_name} ---\n{received_questions_json}\n--- END QUESTIONS ---\n")
            self._log_prompt(target_name, prompt, f"Stage 3 - Rebuttals to {len(relevant_entries)} question(s)")
            self._log(f"Received response ({len(response.content)} chars)", "INFO")
            self._log_response(target_name, response.content, f"Stage 3 - Rebuttals")

            self.round_histories[self.current_round_key].append(response)

            if rebuttals:
                self._log_parse_result(True, f"Parsed {len(rebuttals)} rebuttal(s) from {target_name}")
                self.debug_log.append(f"\n--- PARSED REBUTTALS ({target_name}) ---\n{json.dumps(rebuttals, ensure_ascii=False, indent=2)}\n--- END REBUTTALS ---\n")
            else:
                self._log_parse_result(False, f"No rebuttals parsed from {target_name}")

            if patches:
                self._log(f"Parsed {len(patches)} patch operation(s) from {target_name}", "INFO")
                self.debug_log.append(f"\n--- PARSED PATCHES ({target_name}) ---\n{json.dumps(patches, ensure_ascii=False, indent=2)}\n--- END PATCHES ---\n")

            # Map rebuttals back to entries by qid
            # (If model didn't echo qid, we align in order as a fallback.)
            by_qid = {rb.get("qid", f"Q{idx+1}"): rb for idx, rb in enumerate(rebuttals)}
            for idx, entry in enumerate(relevant_entries):
                qid = entry.get("qid", f"Q{idx+1}")
                rb = by_qid.get(qid)
                entry["rebuttal"] = (rb.get("answer", "") if rb else "").strip() or entry.get("rebuttal")

            # Save for Stage 5 (optional but useful)
            self.last_rebuttals[target_name] = rebuttals
            self.last_rebuttals_patches[target_name] = patches

            self._log(f"Mapped rebuttals back to challenge entries for {target_name}", "INFO")

            # Record training data
            if self.recorder:
                challenger_names = list(set(e["challenger"] for e in relevant_entries))
                self.recorder.record_rebuttal(
                    agent_id=target_name,
                    challenger_id=challenger_names[0] if len(challenger_names) == 1 else "Various",
                    inputs={
                        "own_belief": tgt_belief_obj,
                        "challenges_received": [e["challenge"] for e in relevant_entries],
                    },
                    rebuttals=rebuttals,
                    raw_response=response.content,
                )

            # Add to markdown transcript
            markdown_section = f"\n## {target_name} responds\n\n"
            for rb in rebuttals:
                qid = rb.get("qid", "Q?")
                answer = rb.get("answer", "")
                action = rb.get("action", "unknown")
                markdown_section += f"**{qid}** ({action}): {answer}\n\n"

            self._add_to_markdown(markdown_section)
            self._notify("agent_complete", {"agent_name": target_name, "stage": 3, "action": f"Provided {len(rebuttals)} rebuttal(s), {len(patches)} patch(es)"})

        return self.challenge_rebuttal_pairs

    def run_stage_3_collaborative(self) -> list:
        """
        Stage 3B: Collaborative Truth-Seeking Dialogue.

        For each question in self.challenge_rebuttal_pairs, runs a multi-turn
        dialogue between challenger and defender with periodic adjudicator checks.

        Updates each entry with:
            - "rebuttal": defender's final message from the dialogue
            - "resolution": adjudication dict (same format as Stage 4)
            - "collaborative_transcript": list of exchange turn dicts
            - "adjudicator_checks": list of intermediate check dicts

        Returns:
            list[dict]: Updated self.challenge_rebuttal_pairs
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 3B: COLLABORATIVE TRUTH-SEEKING DIALOGUE")
        self._log("Starting collaborative dialogue phase", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 🤝 Stage 3B: Collaborative Truth-Seeking Dialogue\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 3, "name": "Collaborative Truth-Seeking Dialogue"})

        # Read config values
        max_turns = self.config.collaborative.max_turns_per_question if self.config else 10
        min_turns = self.config.collaborative.min_turns_per_question if self.config else 3
        check_interval = self.config.collaborative.adjudicator_check_interval if self.config else 2
        early_term = self.config.collaborative.early_termination_on_agreement if self.config else True

        self._log(f"Config: max_turns={max_turns}, min_turns={min_turns}, "
                  f"check_interval={check_interval}, early_termination={early_term}", "INFO")

        # Build agent lookup by name
        agent_by_name = {agent.name: agent for agent in self.agents}

        # Get adjudication weights from config
        logic_weight = self.config.adjudication.logic_weight if self.config else 1.0
        ethics_weight = self.config.adjudication.ethics_weight if self.config else 0.0

        self.last_rebuttals = {}

        for entry_idx, entry in enumerate(self.challenge_rebuttal_pairs):
            challenger_name = entry["challenger"]
            defender_name = entry["target"]
            question_text = entry["challenge"]
            qid = entry.get("qid", f"Q{entry_idx + 1}")

            self._log(f"\n--- Collaborative dialogue #{entry_idx + 1}: "
                      f"{challenger_name} → {defender_name} (QID: {qid}) ---", "INFO")
            self._log(f"Question: {question_text[:150]}...", "DEBUG")

            challenger_agent = agent_by_name.get(challenger_name)
            defender_agent = agent_by_name.get(defender_name)

            if not challenger_agent or not defender_agent:
                self._log(f"ERROR: Could not find agents: challenger={challenger_name}, "
                          f"defender={defender_name}", "ERROR")
                continue

            # Get belief JSONs
            ch_belief_obj = challenger_agent.get_internal_belief_obj()
            df_belief_obj = defender_agent.get_internal_belief_obj()
            ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
            df_belief_json = json.dumps(df_belief_obj, ensure_ascii=False, indent=2) if df_belief_obj else ""

            # Initialize dialogue tracking
            entry["collaborative_transcript"] = []
            entry["adjudicator_checks"] = []
            dialogue_history = []

            # Markdown for this exchange
            markdown_section = f"\n## {challenger_name} ↔ {defender_name} — {qid}\n\n"
            markdown_section += f"**Original question**: {question_text}\n\n"
            entry_attack_type = entry.get("attack_type", "")
            entry_attack_strategy = entry.get("attack_strategy", "")
            if entry_attack_type:
                markdown_section += f"*Attack: {entry_attack_type} / {entry_attack_strategy}*\n\n"

            max_rebuttal_length = self.config.stages.max_rebuttal_length_chars if self.config else 500
            topic = self.topic if hasattr(self, "topic") else "<topic>"

            for turn_num in range(1, max_turns + 1):
                if turn_num % 2 == 1:
                    # Odd turn: defender speaks
                    speaker = defender_agent
                    prompt = prompts.build_collaborative_defender_prompt(
                        topic=topic,
                        defender_name=defender_name,
                        challenger_name=challenger_name,
                        defender_belief_json=df_belief_json,
                        question_text=question_text,
                        dialogue_history=dialogue_history,
                        max_response_length_chars=max_rebuttal_length
                    )
                else:
                    # Even turn: challenger follows up
                    speaker = challenger_agent
                    prompt = prompts.build_collaborative_challenger_followup_prompt(
                        topic=topic,
                        challenger_name=challenger_name,
                        defender_name=defender_name,
                        challenger_belief_json=ch_belief_json,
                        question_text=question_text,
                        dialogue_history=dialogue_history,
                        max_response_length_chars=max_rebuttal_length
                    )

                # Log and generate
                self._log_prompt(speaker.name, prompt, f"Stage 3B - Turn {turn_num}")
                response = speaker.generate([Message(role="user", content=prompt)])
                self._log(f"Turn {turn_num} from {speaker.name} ({len(response.content)} chars)", "INFO")
                self._log_response(speaker.name, response.content, f"Stage 3B - Turn {turn_num}")

                turn_dict = {
                    "turn_number": turn_num,
                    "speaker": speaker.name,
                    "message": response.content
                }
                dialogue_history.append(turn_dict)
                entry["collaborative_transcript"].append(turn_dict)

                # Add turn to markdown
                markdown_section += f"**Turn {turn_num} [{speaker.name}]**: {response.content}\n\n"

                # Periodic adjudicator check
                if turn_num % check_interval == 0:
                    self._log(f"Running adjudicator check at turn {turn_num}...", "INFO")
                    check_prompt = prompts.build_collaborative_adjudicator_check_prompt(
                        dialogue_history=dialogue_history,
                        challenger_name=challenger_name,
                        defender_name=defender_name
                    )
                    self._log_prompt("Adjudicator", check_prompt, f"Stage 3B - Check at turn {turn_num}")
                    check_response = self.adjudicator.agent.generate(
                        [Message(role="user", content=check_prompt)]
                    )
                    self._log_response("Adjudicator", check_response.content, f"Stage 3B - Check at turn {turn_num}")

                    # Parse check result
                    check_result = _extract_first_json_block(check_response.content)
                    if not check_result:
                        self._log("Failed to parse adjudicator check JSON, using defaults", "WARN")
                        check_result = {
                            "fallacies_detected": [],
                            "deflection_detected": False,
                            "progress_assessment": "unknown",
                            "convergence_detected": False
                        }

                    self._log(f"Check result: progress={check_result.get('progress_assessment')}, "
                              f"convergence={check_result.get('convergence_detected')}", "INFO")
                    self.debug_log.append(
                        f"\n--- ADJUDICATOR CHECK AT TURN {turn_num} ({qid}) ---\n"
                        f"{json.dumps(check_result, ensure_ascii=False, indent=2)}\n"
                        f"--- END CHECK ---\n"
                    )

                    entry["adjudicator_checks"].append(check_result)
                    markdown_section += (
                        f"*[Adjudicator check — progress: {check_result.get('progress_assessment', '?')}, "
                        f"convergence: {check_result.get('convergence_detected', False)}]*\n\n"
                    )

                    # Early termination (only after min_turns)
                    if early_term and turn_num >= min_turns:
                        if check_result.get("convergence_detected", False):
                            self._log(f"Convergence detected at turn {turn_num} — terminating early", "INFO")
                            markdown_section += f"*Early termination: convergence detected at turn {turn_num}*\n\n"
                            break

            self._log(f"Dialogue completed after {len(dialogue_history)} turn(s)", "INFO")

            # Final adjudication — extract targeted beliefs for context
            self._log("Running final adjudication...", "INFO")
            collab_target_ids = entry.get("target_ids", [])
            defender_targeted_claims_json = ""
            challenger_targeted_claims_json = ""
            if collab_target_ids:
                if df_belief_obj:
                    excerpt = _extract_belief_excerpt(df_belief_obj, collab_target_ids)
                    if excerpt:
                        defender_targeted_claims_json = json.dumps(excerpt, ensure_ascii=False, indent=2)
                if ch_belief_obj:
                    excerpt = _extract_belief_excerpt(ch_belief_obj, collab_target_ids)
                    if excerpt:
                        challenger_targeted_claims_json = json.dumps(excerpt, ensure_ascii=False, indent=2)

            final_prompt = prompts.build_collaborative_final_adjudication_prompt(
                topic=topic,
                challenger_name=challenger_name,
                defender_name=defender_name,
                question_text=question_text,
                target_ids=collab_target_ids,
                dialogue_transcript=entry["collaborative_transcript"],
                adjudicator_checks=entry["adjudicator_checks"],
                logic_weight=logic_weight,
                ethics_weight=ethics_weight,
                defender_targeted_claims_json=defender_targeted_claims_json,
                challenger_targeted_claims_json=challenger_targeted_claims_json
            )
            self._log_prompt("Adjudicator", final_prompt, f"Stage 3B - Final Adjudication ({qid})")
            final_response = self.adjudicator.agent.generate(
                [Message(role="user", content=final_prompt)]
            )
            self._log_response("Adjudicator", final_response.content, f"Stage 3B - Final Adjudication ({qid})")

            # Parse final adjudication JSON
            result = _extract_first_json_block(final_response.content)
            if result:
                resolution = {
                    "status": _normalize_collab_verdict(result.get("outcome", "")),
                    "reasoning": result.get("reasoning", ""),
                    "restatement": result.get("restatement", ""),
                    "formalizations": {
                        "challenger": result.get("formalization_challenger", ""),
                        "target": result.get("formalization_target", "")
                    }
                }
            else:
                self._log("Failed to parse final adjudication JSON, marking as unresolved", "WARN")
                resolution = {
                    "status": "unresolved",
                    "reasoning": final_response.content.strip(),
                    "restatement": "Unable to parse adjudication",
                    "formalizations": {"challenger": "", "target": ""}
                }

            entry["resolution"] = resolution

            # Set rebuttal to defender's final message
            defender_messages = [t for t in dialogue_history if t["speaker"] == defender_name]
            entry["rebuttal"] = defender_messages[-1]["message"] if defender_messages else ""

            self._log(f"Final outcome: {resolution['status'].upper()}", "INFO")
            self.debug_log.append(
                f"\n--- FINAL ADJUDICATION ({qid}: {challenger_name} → {defender_name}) ---\n"
                f"{json.dumps(resolution, ensure_ascii=False, indent=2)}\n"
                f"--- END ADJUDICATION ---\n"
            )

            # Update agent stats
            self.agent_stats = update_agent_stats(self.agent_stats, entry)

            # Finalize markdown for this exchange
            markdown_section += f"**Outcome**: {resolution['status'].upper()}\n\n"
            markdown_section += f"**Reasoning**: {resolution.get('reasoning', 'N/A')}\n\n"
            self._add_to_markdown(markdown_section)
            self._notify("adjudication_result", {
                "challenger": challenger_name, "target": defender_name,
                "outcome": resolution['status'].upper(), "qid": qid,
                "turns": len(dialogue_history),
            })

        # Anti-repetition tracking (same format as Stage 4)
        for entry in self.challenge_rebuttal_pairs:
            prev_key = f"{entry['challenger']}→{entry['target']}"
            self.previous_rounds_challenges.setdefault(prev_key, []).append({
                "qid": entry.get("qid"),
                "target_ids": entry.get("target_ids", []),
                "attack_type": entry.get("attack_type", ""),
                "attack_strategy": entry.get("attack_strategy", ""),
                "outcome": entry.get("resolution", {}).get("status", "unknown")
            })

        self._log(f"Stage 3B complete - processed {len(self.challenge_rebuttal_pairs)} exchange(s)", "INFO")

        return self.challenge_rebuttal_pairs

    def run_stage_3_bloodsport(self) -> list:
        """
        Stage 3C: Blood Sport Adversarial Exchange.

        For each agent pair, runs a multi-turn adversarial exchange where agents
        attack each other's positions with escalating rhetorical intensity.
        Standard adjudication evaluates the exchanges unchanged.

        Flow per agent pair:
            1. Agent A delivers opening attack targeting Agent B's claims
            2. Agent B responds with counter-attack + defense
            3. Continue for max_exchanges turns
            4. Standard adjudicator evaluates the exchange

        Updates each entry with:
            - "rebuttal": summary of defender's arguments
            - "resolution": adjudication dict (same format as Stage 4)
            - "bloodsport_transcript": list of exchange turn dicts
            - "bloodsport_stats": exchange metadata

        Returns:
            list[dict]: Updated self.challenge_rebuttal_pairs
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 3C: BLOOD SPORT ADVERSARIAL EXCHANGE")
        self._log("Starting blood sport adversarial exchange phase", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        intensity = self.config.bloodsport.intensity if self.config else "moderate"
        max_exchanges = self.config.bloodsport.max_exchanges if self.config else 5

        markdown_header = f"\n# ⚔️ Stage 3C: Blood Sport Exchange (Intensity: {intensity})\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 3, "name": f"Blood Sport Exchange (Intensity: {intensity})"})

        self._log(f"Config: intensity={intensity}, max_exchanges={max_exchanges}", "INFO")

        # Build agent lookup by name
        agent_by_name = {agent.name: agent for agent in self.agents}

        # Get adjudication weights from config
        logic_weight = self.config.adjudication.logic_weight if self.config else 1.0
        ethics_weight = self.config.adjudication.ethics_weight if self.config else 0.0

        self.last_rebuttals = {}

        # Group challenges by agent pair to run one multi-turn exchange per pair
        pair_entries = {}
        for entry in self.challenge_rebuttal_pairs:
            pair_key = f"{entry['challenger']}→{entry['target']}"
            pair_entries.setdefault(pair_key, []).append(entry)

        for pair_key, entries in pair_entries.items():
            challenger_name = entries[0]["challenger"]
            defender_name = entries[0]["target"]

            self._log(f"\n--- Blood sport exchange: {challenger_name} vs {defender_name} ---", "INFO")

            challenger_agent = agent_by_name.get(challenger_name)
            defender_agent = agent_by_name.get(defender_name)

            if not challenger_agent or not defender_agent:
                self._log(f"ERROR: Could not find agents: {challenger_name}, {defender_name}", "ERROR")
                continue

            # Get belief JSONs
            ch_belief_obj = challenger_agent.get_internal_belief_obj()
            df_belief_obj = defender_agent.get_internal_belief_obj()
            ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
            df_belief_json = json.dumps(df_belief_obj, ensure_ascii=False, indent=2) if df_belief_obj else ""

            # Initialize exchange tracking
            dialogue_history = []
            topic = self.topic if hasattr(self, "topic") else "<topic>"

            # Markdown for this exchange
            markdown_section = f"\n## {challenger_name} vs {defender_name}\n\n"

            max_rebuttal_length = self.config.stages.max_rebuttal_length_chars if self.config else 1000

            # Multi-turn adversarial exchange
            for turn_num in range(1, max_exchanges + 1):
                # Alternate: odd turns = challenger attacks, even turns = defender attacks
                if turn_num % 2 == 1:
                    speaker = challenger_agent
                    opponent = defender_agent
                    speaker_belief_json = ch_belief_json
                    opponent_belief_json = df_belief_json
                else:
                    speaker = defender_agent
                    opponent = challenger_agent
                    speaker_belief_json = df_belief_json
                    opponent_belief_json = ch_belief_json

                prompt = prompts.build_stage_3_bloodsport_prompt(
                    topic=topic,
                    agent_name=speaker.name,
                    opponent_name=opponent.name,
                    agent_belief_json=speaker_belief_json,
                    opponent_belief_json=opponent_belief_json,
                    intensity=intensity,
                    dialogue_history=dialogue_history if dialogue_history else None,
                    max_response_length_chars=max_rebuttal_length
                )

                # Log and generate
                self._log_prompt(speaker.name, prompt, f"Stage 3C - Turn {turn_num}")
                generation_temp = self.config.stages.generation_temperature if self.config else 0.2
                response = speaker.generate([Message(role="user", content=prompt)], temperature=generation_temp)
                self._log(f"Turn {turn_num} from {speaker.name} ({len(response.content)} chars)", "INFO")
                self._log_response(speaker.name, response.content, f"Stage 3C - Turn {turn_num}")

                # Parse JSON response
                turn_data = _extract_first_json_block(response.content)
                if turn_data:
                    attack = turn_data.get("attack", "")
                    defense = turn_data.get("defense")
                    target_claims = turn_data.get("target_claims", [])
                    self._log_parse_result(True, f"Parsed attack/defense for turn {turn_num}")
                else:
                    self._log_parse_result(False, f"Failed to parse JSON for turn {turn_num}, using raw text")
                    attack = response.content.strip()
                    defense = None
                    target_claims = []

                turn_dict = {
                    "turn_number": turn_num,
                    "speaker": speaker.name,
                    "attack": attack,
                    "defense": defense,
                    "target_claims": target_claims,
                    "raw_response": response.content
                }
                dialogue_history.append(turn_dict)

                # Add turn to markdown
                markdown_section += f"**Turn {turn_num} [{speaker.name}]**\n\n"
                if defense:
                    markdown_section += f"*Defense*: {defense}\n\n"
                markdown_section += f"*Attack*: {attack}\n\n"
                if target_claims:
                    markdown_section += f"*Targets*: {', '.join(target_claims)}\n\n"

            self._log(f"Exchange completed after {len(dialogue_history)} turn(s)", "INFO")

            # Record training data for bloodsport exchange
            if self.recorder:
                self.recorder.record_bloodsport_exchange(
                    agent_ids=[challenger_name, defender_name],
                    intensity=intensity,
                    inputs={
                        "agent_beliefs": {
                            challenger_name: ch_belief_obj,
                            defender_name: df_belief_obj,
                        },
                    },
                    turns=[
                        {
                            "turn": t["turn_number"],
                            "agent_id": t["speaker"],
                            "attack": t["attack"],
                            "defense": t["defense"],
                            "target_claims": t["target_claims"],
                            "raw_response": t["raw_response"],
                        }
                        for t in dialogue_history
                    ],
                )

            # Build a combined summary of each side's attacks for adjudication
            challenger_attacks = [t for t in dialogue_history if t["speaker"] == challenger_name]
            defender_attacks = [t for t in dialogue_history if t["speaker"] == defender_name]

            challenger_summary = " ".join([t["attack"] for t in challenger_attacks if t["attack"]])
            defender_summary = " ".join([t["attack"] for t in defender_attacks if t["attack"]])
            defender_defenses = " ".join([t["defense"] for t in defender_attacks if t.get("defense")])

            # Adjudicate each original challenge entry using exchange context
            for entry in entries:
                qid = entry.get("qid", "Q?")
                self._log(f"Adjudicating {qid}: {challenger_name} → {defender_name}...", "INFO")

                # Frame as: challenger's challenges + attacks vs defender's defenses
                combined_challenge = entry["challenge"]
                if challenger_summary:
                    combined_challenge += f"\n\n[Blood Sport Attacks]: {challenger_summary}"

                combined_rebuttal = defender_defenses if defender_defenses else defender_summary
                if not combined_rebuttal:
                    combined_rebuttal = "(No defense provided)"

                resolution = self.adjudicator.run(
                    challenge=combined_challenge,
                    rebuttal=combined_rebuttal,
                    challenger=challenger_name,
                    target=defender_name
                )

                # Log the per-pair prompt and raw response for debugging
                bs_label = f"Stage 3C - Blood Sport Adjudication ({qid}: {challenger_name} → {defender_name})"
                if resolution.get("_debug_prompt"):
                    self._log_prompt("Adjudicator", resolution["_debug_prompt"], bs_label)
                if resolution.get("_debug_raw_response"):
                    self._log_response("Adjudicator", resolution["_debug_raw_response"], bs_label)

                # Strip debug keys before storing
                resolution.pop("_debug_prompt", None)
                resolution.pop("_debug_raw_response", None)

                entry["resolution"] = resolution
                entry["rebuttal"] = combined_rebuttal
                entry["bloodsport_transcript"] = dialogue_history
                entry["bloodsport_stats"] = {
                    "intensity": intensity,
                    "num_turns": len(dialogue_history),
                    "challenger_attacks": len(challenger_attacks),
                    "defender_attacks": len(defender_attacks)
                }

                self._log(f"Adjudication outcome for {qid}: {resolution.get('status', 'unknown').upper()}", "INFO")
                self.debug_log.append(
                    f"\n--- BLOODSPORT ADJUDICATION ({qid}: {challenger_name} → {defender_name}) ---\n"
                    f"{json.dumps(resolution, ensure_ascii=False, indent=2)}\n"
                    f"--- END ADJUDICATION ---\n"
                )

                # Update agent stats
                self.agent_stats = update_agent_stats(self.agent_stats, entry)

                # Update bloodsport-specific stats
                for agent_name in [challenger_name, defender_name]:
                    if agent_name in self.agent_stats:
                        stats = self.agent_stats[agent_name]
                        stats['bloodsport_exchanges'] = stats.get('bloodsport_exchanges', 0) + 1
                        agent_turns = len([t for t in dialogue_history if t["speaker"] == agent_name])
                        stats['bloodsport_turns'] = stats.get('bloodsport_turns', 0) + agent_turns

                markdown_section += f"**{qid} Outcome**: {resolution['status'].upper()}\n\n"
                markdown_section += f"**Reasoning**: {resolution.get('reasoning', 'N/A')}\n\n"
                self._notify("adjudication_result", {
                    "challenger": challenger_name, "target": defender_name,
                    "outcome": resolution['status'].upper(), "qid": qid,
                    "turns": len(dialogue_history),
                })

            self._add_to_markdown(markdown_section)

        # Anti-repetition tracking
        for entry in self.challenge_rebuttal_pairs:
            prev_key = f"{entry['challenger']}→{entry['target']}"
            self.previous_rounds_challenges.setdefault(prev_key, []).append({
                "qid": entry.get("qid"),
                "target_ids": entry.get("target_ids", []),
                "attack_type": entry.get("attack_type", ""),
                "attack_strategy": entry.get("attack_strategy", ""),
                "outcome": entry.get("resolution", {}).get("status", "unknown")
            })

        self._log(f"Stage 3C complete - processed {len(pair_entries)} exchange pair(s)", "INFO")

        return self.challenge_rebuttal_pairs

    def run_stage_4_conflict_resolution(self) -> list:
        """
        Stage 4: Rigorous Conflict Resolution

        Iterates through all challenge-rebuttal pairs and uses the Adjudicator
        to run a structured truth-seeking protocol consisting of:
            1. Restatement of disagreement
            2. Formalization of both arguments
            3. Third-party adjudication
            4. Final resolution logging

        Returns:
            list[dict]: Updated challenge_rebuttal_pairs list with resolution entries.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 4: ADJUDICATION")
        self._log(f"Starting adjudication of {len(self.challenge_rebuttal_pairs)} challenge-rebuttal pair(s)", "INFO")
        self._log_prompt("Adjudicator", self.adjudicator.agent.system_prompt, "Stage 4 - System Prompt")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚖️ Stage 4: Adjudication\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 4, "name": "Adjudication"})

        # Build agent lookup for belief excerpt extraction
        agent_by_name = {agent.name: agent for agent in self.agents}

        # Separate complete pairs from incomplete ones
        complete_pairs = []
        for i, entry in enumerate(self.challenge_rebuttal_pairs):
            if not entry["challenge"] or not entry["rebuttal"]:
                self._log(f"Skipping incomplete pair: {entry['challenger']} → {entry['target']} (missing challenge or rebuttal)", "WARN")
                markdown_note = f"\n*Skipped incomplete pair: {entry['challenger']} → {entry['target']}*\n"
                self._add_to_markdown(markdown_note)
                self._notify("agent_complete", {"agent_name": "Adjudicator", "stage": 4, "action": f"Skipped incomplete pair: {entry['challenger']} → {entry['target']}"})
            else:
                complete_pairs.append((i, entry))

        # --- GATHER: Fire all adjudication calls in parallel ---
        items = [
            WorkItem(
                key=f"{entry['challenger']}→{entry['target']}:{entry.get('qid', f'Q{i}')}",
                callable=lambda e=entry: _run_adjudication(self.adjudicator, e, agent_by_name),
                context={"entry_index": i},
            )
            for i, entry in complete_pairs
        ]
        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic order ---
        adjudication_count = 0
        for item in items:
            work_result = results[item.key]
            entry_idx = item.context["entry_index"]
            entry = self.challenge_rebuttal_pairs[entry_idx]

            adjudication_count += 1
            self._log(f"\n--- Adjudicating pair #{adjudication_count}: {entry['challenger']} → {entry['target']} ---", "INFO")
            self._log(f"Challenge: {entry['challenge'][:100]}...", "DEBUG")
            self._log(f"Rebuttal: {entry['rebuttal'][:100]}...", "DEBUG")

            r = self._retry_on_parse_failure(
                generate_fn=lambda e=entry: _run_adjudication(self.adjudicator, e, agent_by_name),
                is_valid_fn=lambda r: True,
                stage_label="Stage 4",
                agent_name=item.key,
                initial_result=work_result,
            )

            if r is None:
                entry["resolution"] = {"status": "unresolved", "reasoning": "All retry attempts failed"}
                self._notify("adjudication_result", {
                    "challenger": entry["challenger"], "target": entry["target"],
                    "outcome": "ERROR",
                })
                continue

            resolution = r["resolution"]

            # Log the per-pair prompt and raw response for debugging
            pair_label = f"Stage 4 - Adjudication ({entry['challenger']} → {entry['target']})"
            if resolution.get("_debug_prompt"):
                self._log_prompt("Adjudicator", resolution["_debug_prompt"], pair_label)
            if resolution.get("_debug_raw_response"):
                self._log_response("Adjudicator", resolution["_debug_raw_response"], pair_label)

            # Strip debug keys before storing
            resolution.pop("_debug_prompt", None)
            resolution.pop("_debug_raw_response", None)

            self._log(f"Adjudication outcome: {resolution.get('status', 'unknown').upper()}", "INFO")
            self.debug_log.append(f"\n--- ADJUDICATION RESULT ({entry['challenger']} → {entry['target']}) ---\n{json.dumps(resolution, ensure_ascii=False, indent=2)}\n--- END ADJUDICATION ---\n")

            # Save structured result in the entry
            entry["resolution"] = resolution

            # Update agent stats
            self.agent_stats = update_agent_stats(self.agent_stats, entry)
            self._log(f"Updated agent stats for this pair", "DEBUG")

            # Record training data
            if self.recorder:
                self.recorder.record_adjudication(
                    challenger_id=entry["challenger"],
                    target_id=entry["target"],
                    inputs={
                        "challenge": entry["challenge"],
                        "rebuttal": entry["rebuttal"],
                    },
                    verdict=resolution.get("status", "unknown"),
                    reasoning=resolution.get("reasoning", ""),
                    raw_response=json.dumps(resolution, ensure_ascii=False),
                )

            # Add to markdown transcript
            markdown_section = f"\n### {entry['challenger']} → {entry['target']}\n\n"
            adj_attack_type = entry.get("attack_type", "")
            adj_attack_strategy = entry.get("attack_strategy", "")
            if adj_attack_type:
                markdown_section += f"*Attack: {adj_attack_type} / {adj_attack_strategy}*\n\n"
            if resolution.get('restatement'):
                markdown_section += f"**Disagreement**: {resolution['restatement']}\n\n"
            markdown_section += f"**Outcome**: {resolution['status'].upper()}\n\n"
            markdown_section += f"**Reasoning**: {resolution.get('reasoning', 'N/A')}\n\n"

            self._add_to_markdown(markdown_section)
            self._notify("adjudication_result", {
                "challenger": entry["challenger"], "target": entry["target"],
                "outcome": resolution["status"].upper(),
            })

        self._log(f"Stage 4 complete - adjudicated {adjudication_count} pair(s)", "INFO")

        # Save challenges for anti-repetition in next round (if multi-round debate)
        for entry in self.challenge_rebuttal_pairs:
            challenger = entry.get("challenger")
            target = entry.get("target")
            qid = entry.get("qid")
            target_ids = entry.get("target_ids", [])
            outcome = (entry.get("resolution") or {}).get("status", "unknown")

            prev_challenges_key = f"{challenger}→{target}"
            if prev_challenges_key not in self.previous_rounds_challenges:
                self.previous_rounds_challenges[prev_challenges_key] = []

            self.previous_rounds_challenges[prev_challenges_key].append({
                "qid": qid,
                "target_ids": target_ids,
                "attack_type": entry.get("attack_type", ""),
                "attack_strategy": entry.get("attack_strategy", ""),
                "outcome": outcome
            })

        return self.challenge_rebuttal_pairs

    def run_stage_5_update_positions(self) -> None:
        """
        Stage 5: Belief Updating and Position Reframing

        For each agent, gathers all resolution outcomes where they were the target.
        Uses this to synthesize an updated version of their beliefs and claims.

        Uses the gather-then-apply pattern: each agent's full Stage 5 pipeline
        (Phase 1 → Phase 2 for CBS, single-phase for bloodsport/legacy) runs
        concurrently via ParallelDispatcher.  Shared-state mutations happen in
        the sequential APPLY phase.

        Updates:
            dict[str, str]: Updated current_positions keyed by agent name.

        Returns:
            None.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 5: BELIEF UPDATES")
        self._log("Starting belief update phase based on adjudication outcomes", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 🔄 Stage 5: Belief Updates\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 5, "name": "Belief Updates"})

        # Group all adjudicated results by target agent
        grouped_by_target = {}
        for entry in self.challenge_rebuttal_pairs:
            target = entry["target"]
            grouped_by_target.setdefault(target, []).append(entry)

        self._log(f"Grouped adjudication results for {len(grouped_by_target)} agent(s)", "INFO")

        # Extract config values for the gather function (avoid self.* in threads)
        stage3_mode = self.config.stage3_mode if self.config else "rebuttal"
        generation_temp = self.config.stages.generation_temperature if self.config else 0.2
        max_retries = self.config.stages.parse_retries if self.config else 3

        # --- GATHER: Build work items for agents with adjudication results ---
        items = [
            WorkItem(
                key=agent.name,
                callable=lambda a=agent, re=grouped_by_target[agent.name]: _run_stage5_for_agent(
                    agent=a,
                    relevant_entries=re,
                    stage3_mode=stage3_mode,
                    last_rebuttals_patches=self.last_rebuttals_patches.get(a.name, []),
                    generation_temp=generation_temp,
                    max_retries=max_retries,
                ),
                context={"agent": agent},
            )
            for agent in self.agents
            if agent.name in grouped_by_target
        ]

        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic agent order ---
        for agent in self.agents:
            name = agent.name
            relevant_entries = grouped_by_target.get(name, [])

            # Agents with no adjudication results — not dispatched, handle here
            if not relevant_entries:
                self._log(f"No adjudication results for {name}, keeping current belief", "INFO")
                self.current_positions[name] = self.opening_positions.get(name, "[No change]")
                continue

            work_result = results[name]

            # Handle dispatcher-level error (unhandled exception in gather function)
            if work_result.error is not None:
                self._log(f"Stage 5 error for {name}: {work_result.error}", "ERROR")
                md_view = agent.get_internal_belief()
                markdown_section = f"\n## {name} - Updated Position\n\n{md_view}\n"
                self._add_to_markdown(markdown_section)
                self._notify("agent_complete", {"agent_name": name, "stage": 5, "action": "Belief update failed"})
                continue

            r = work_result.result

            # 1. Replay prompt/response logging
            for api_call in r["api_calls"]:
                if not api_call.get("is_retry"):
                    self._log_prompt(name, api_call["prompt"], api_call["label"])
                self._log(f"Received response ({len(api_call['response'].content)} chars)", "INFO")
                self._log_response(name, api_call["response"].content, api_call["label"])
                self.round_histories[self.current_round_key].append(api_call["response"])

            # 2. Replay deferred log entries
            for msg, level in r["log_entries"]:
                self._log(msg, level)

            # 3. Append deferred debug entries
            for entry in r["debug_entries"]:
                self.debug_log.append(entry)

            # 4. Commit agent belief state
            final_belief = r["final_belief"]
            md_view = r["md_view"]
            if final_belief is not None and not r["reverted"]:
                agent.set_internal_belief_obj(final_belief)
                md_view = belief_to_markdown(final_belief)
                agent.set_internal_belief(md_view)
                agent.all_beliefs_held.append(json.dumps(final_belief, ensure_ascii=False, indent=2))

            # 5. Record training data
            if self.recorder:
                belief_after = agent.get_internal_belief_obj()
                adjudication_results = [
                    {
                        "role": "target",
                        "opponent": e.get("challenger", "?"),
                        "verdict": (e.get("resolution") or {}).get("status", "unknown")
                                   if isinstance(e.get("resolution"), dict) else "unknown",
                        "reasoning": (e.get("resolution") or {}).get("reasoning", "")
                                     if isinstance(e.get("resolution"), dict) else "",
                    }
                    for e in relevant_entries
                ]
                last_response_content = r["api_calls"][-1]["response"].content if r["api_calls"] else ""
                self.recorder.record_belief_update(
                    agent_id=name,
                    belief_before=r["prior_json"],
                    belief_after=belief_after,
                    adjudication_results=adjudication_results,
                    patches=r["patches"],
                    raw_response=last_response_content,
                )

            # 6. Add to markdown transcript
            markdown_section = f"\n## {name} - Updated Position\n\n{md_view}\n"
            self._add_to_markdown(markdown_section)
            self._notify("agent_complete", {"agent_name": name, "stage": 5, "action": "Belief updated"})

        self._log("Stage 5 complete - all agents updated their beliefs", "INFO")

        return

    def run_stage_6_concluding_remarks(self) -> dict:
        """
        Stage 6: Concluding Remarks

        Each agent reflects on the outcome of the debate, what they learned,
        what they still believe, and any changes in their stance.

        Returns:
            dict[str, str]: A mapping from agent names to their final concluding remarks.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 6: CONCLUDING REMARKS")
        self._log(f"Requesting concluding remarks from {len(self.agents)} agent(s)", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 🎤 Stage 6: Concluding Remarks\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 6, "name": "Concluding Remarks"})

        self.conclusions = {}

        # Pre-serialize the adjudicated pairs for context
        cr_pairs_json = json.dumps(self.challenge_rebuttal_pairs, ensure_ascii=False, indent=2)
        self._log(f"Challenge-rebuttal pairs JSON size: {len(cr_pairs_json)} chars", "DEBUG")

        topic = self.topic if hasattr(self, "topic") else "<topic>"
        current_round = self.current_round if hasattr(self, "current_round") else 1

        # --- GATHER: Fire all conclusion calls in parallel ---
        items = [
            WorkItem(
                key=agent.name,
                callable=lambda a=agent: _generate_conclusion(
                    agent=a, topic=topic, config=self.config,
                    cr_pairs_json=cr_pairs_json, current_round=current_round,
                ),
                context={"agent": agent},
            )
            for agent in self.agents
        ]
        results = self.dispatcher.run(items)

        # --- APPLY: Process results sequentially in deterministic agent order ---
        for agent in self.agents:
            name = agent.name
            work_result = results[name]

            self._log(f"\n--- Processing concluding remarks for: {name} ---", "INFO")

            r = self._retry_on_parse_failure(
                generate_fn=lambda a=agent: _generate_conclusion(
                    agent=a, topic=topic, config=self.config,
                    cr_pairs_json=cr_pairs_json, current_round=current_round,
                ),
                is_valid_fn=lambda r: True,
                stage_label="Stage 6",
                agent_name=name,
                initial_result=work_result,
            )

            if r is None:
                self.conclusions[name] = f"[All retry attempts failed for {name}]"
                markdown_section = f"\n## {name} - Concluding Remarks\n\n[All retry attempts failed]\n"
                self._add_to_markdown(markdown_section)
                self._notify("agent_complete", {"agent_name": name, "stage": 6, "action": "Concluding remarks failed"})
                continue

            response = r["response"]
            prompt = r["prompt"]
            a_belief_obj = r["a_belief_obj"]

            # Replay logs
            self._log_prompt(name, prompt, "Stage 6 - Concluding Remarks")
            self._log(f"Received response ({len(response.content)} chars)", "INFO")
            self._log_response(name, response.content, "Stage 6 - Concluding Remarks")

            self.round_histories[self.current_round_key].append(response)

            # Try to parse {"conclusion": {...}} for structured logging
            self._log("Parsing conclusion JSON block...", "INFO")
            concl_obj = _extract_first_json_block(response.content)
            if concl_obj and "conclusion" in concl_obj:
                self._log_parse_result(True, f"Successfully parsed structured conclusion for {name}")
                self.debug_log.append(f"\n--- PARSED CONCLUSION JSON ({name}) ---\n{json.dumps(concl_obj['conclusion'], ensure_ascii=False, indent=2)}\n--- END CONCLUSION ---\n")
                self.conclusions[name] = concl_obj["conclusion"]
                c = concl_obj["conclusion"]
                ft = c.get("final_thesis", {})
                lines = []
                if ft.get("stance"):
                    lines.append(f"**Final stance**: {ft['stance']}")
                if ft.get("strength") is not None:
                    lines.append(f"**Final strength**: {ft['strength']}")
                if c.get("our_strongest_claims"):
                    lines.append(f"**Strongest claims**: {', '.join(c['our_strongest_claims'])}")
                if c.get("our_concessions"):
                    lines.append("**Concessions**:")
                    for con in c["our_concessions"]:
                        lines.append(f"  - {con.get('target_id','?')} ({con.get('type','?')}): {con.get('note','')}")
                if c.get("unresolved_uncertainties"):
                    lines.append(f"**Unresolved uncertainties**: {', '.join(c['unresolved_uncertainties'])}")
                markdown_content = "\n".join(lines)
            else:
                self._log_parse_result(False, f"No structured conclusion found for {name}, using raw text")
                markdown_content = re.sub(r"```(?:json)?\s*", "", response.content).replace("```", "").strip()
                self.conclusions[name] = markdown_content

            # Record training data
            if self.recorder:
                self.recorder.record_concluding_remarks(
                    agent_id=name,
                    final_belief=a_belief_obj,
                    remarks=self.conclusions[name],
                    raw_response=response.content,
                )

            # Add to markdown transcript
            markdown_section = f"\n## {name} - Concluding Remarks\n\n{markdown_content}\n"
            self._add_to_markdown(markdown_section)
            self._notify("agent_complete", {"agent_name": name, "stage": 6, "action": "Concluding remarks received"})

        return self.conclusions

    def run_stage_7_scribing(self, scribe_agent: Optional[Agent] = None, max_chars_per_chunk: Optional[int] = None, overlap_chars: Optional[int] = None) -> str:
        """
        Stage 7: Scribe a single, flowing narrative of the entire debate.

        Parameters:
        - scribe_agent: the LLM-backed agent to use for scribing. If None, we reuse the first agent.
        - max_chars_per_chunk: per-chunk size to keep prompts within model context.
        - overlap_chars: small overlap between chunks for continuity.

        Flow:
        - Map: for each chunk, call build_stage_7_scribe_prompt_map -> get continuity_update (JSON) + narrative slice (Markdown).
        - Reduce: combine all slices with build_stage_7_scribe_prompt_reduce -> final Markdown narrative.

        Side effects:
        - self.scribed_transcript_markdown: final narrative (Markdown)
        - Appends the final narrative to self.full_transcript for completeness.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 7: SCRIBING")
        self._log("Starting scribe synthesis - generating flowing narrative from debate", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# 📝 Stage 7: Scribed Narrative\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 7, "name": "Scribed Narrative"})

        scribe = scribe_agent
        if scribe is None:
            self._log("ERROR: No scribe agent available", "ERROR")
            raise RuntimeError("No agent available for Stage 7 scribing.")

        self._log(f"Using scribe agent: {scribe.name}", "INFO")

        # Get chunking parameters from config or use defaults
        if max_chars_per_chunk is None:
            max_chars_per_chunk = self.config.scribe.max_chars_per_chunk if self.config else 15000
        if overlap_chars is None:
            overlap_chars = self.config.scribe.overlap_chars if self.config else 1000

        # 1) Gather source transcript
        full_text = "\n".join(str(x) for x in self.full_transcript)
        self._log(f"Source transcript length: {len(full_text)} chars", "INFO")

        # 2) Chunk it
        chunks = _chunk_text_by_chars(full_text, max_chars=max_chars_per_chunk, overlap=overlap_chars)
        self._log(f"Split transcript into {len(chunks)} chunk(s) (max={max_chars_per_chunk}, overlap={overlap_chars})", "INFO")

        # 3) Map over chunks, maintaining a continuity state
        continuity_state = {}  # grows over chunks
        narrative_slices = []
        agent_names = [a.name for a in self.agents]

        for idx, ch in enumerate(chunks, start=1):
            self._log(f"\n--- Processing chunk {idx}/{len(chunks)} ({len(ch)} chars) ---", "INFO")

            short_note_max = self.config.stages.short_note_max_chars if self.config else 140
            prompt = prompts.build_stage_7_scribe_prompt_map(
                topic=getattr(self, "topic", "<topic>"),
                agent_names=agent_names,
                transcript_chunk=ch,
                continuity_state_json=json.dumps(continuity_state, ensure_ascii=False, indent=2),
                short_note_max_chars=short_note_max
            )

            # Log prompt
            self._log_prompt(scribe.name, prompt, f"Stage 7 Map - Chunk {idx}/{len(chunks)}")

            stage_request = [Message(role="user", content=prompt)]

            scribe_temp = self.config.scribe.scribe_temperature if self.config else 0.3
            self._log(f"Calling scribe for chunk {idx}...", "INFO")
            resp = scribe.generate(stage_request, temperature=scribe_temp)
            self._log(f"Received response ({len(resp.content)} chars)", "INFO")

            # Log raw response
            self._log_response(scribe.name, resp.content, f"Stage 7 Map - Chunk {idx}/{len(chunks)}")

            self.round_histories[self.current_round_key].append(resp)

            # Parse continuity JSON + narrative slice
            self._log("Parsing continuity update and narrative slice...", "INFO")
            first_json = _extract_first_json_block(resp.content) or {}
            cont_update = (first_json.get("continuity_update") or {})

            if cont_update:
                self._log(f"Continuity update keys: {list(cont_update.keys())}", "DEBUG")
                self.debug_log.append(f"\n--- CONTINUITY UPDATE CHUNK {idx} ---\n{json.dumps(cont_update, ensure_ascii=False, indent=2)}\n--- END CONTINUITY ---\n")

            # Merge continuity updates shallowly
            for k, v in (cont_update or {}).items():
                if isinstance(v, dict):
                    # If target is not a dict yet, replace; else shallow-merge
                    if not isinstance(continuity_state.get(k), dict):
                        continuity_state[k] = {}
                    # Shallow merge is usually enough for our fields
                    continuity_state[k].update(v)
                elif isinstance(v, list):
                    # Ensure target is a list, then dedupe-extend
                    if not isinstance(continuity_state.get(k), list):
                        continuity_state[k] = []
                    _dedupe_extend(continuity_state[k], v)
                else:
                    # Scalars: last write wins
                    continuity_state[k] = v

            narrative_md = _extract_first_markdown_block(resp.content) or ""
            narrative_slices.append(narrative_md)

            self._log(f"Extracted narrative slice ({len(narrative_md)} chars)", "INFO")
            self._log(f"Cumulative continuity state keys: {list(continuity_state.keys())}", "DEBUG")
            self._notify("agent_complete", {"agent_name": "Scribe", "stage": 7, "action": f"Chunk {idx}/{len(chunks)}: {len(narrative_md)} chars"})

        # 4) Reduce: stitch all slices into a single cohesive narrative
        self._log("\n--- REDUCE PHASE: Synthesizing final narrative ---", "INFO")
        self._log(f"Combining {len(narrative_slices)} narrative slice(s)", "INFO")

        reduce_prompt = prompts.build_stage_7_scribe_prompt_reduce(
            topic=getattr(self, "topic", "<topic>"),
            agent_names=agent_names,
            all_narrative_slices_markdown=narrative_slices,
            final_continuity_state_json=json.dumps(continuity_state, ensure_ascii=False, indent=2)
        )

        # Log prompt
        self._log_prompt(scribe.name, reduce_prompt, "Stage 7 Reduce - Final Synthesis")

        reduce_request = [Message(role="user", content=reduce_prompt)]

        scribe_temp = self.config.scribe.scribe_temperature if self.config else 0.3
        self._log("Calling scribe for final synthesis...", "INFO")
        reduce_resp = scribe.generate(reduce_request, temperature=scribe_temp)
        self._log(f"Received final synthesis ({len(reduce_resp.content)} chars)", "INFO")

        # Log raw response
        self._log_response(scribe.name, reduce_resp.content, "Stage 7 Reduce - Final Synthesis")

        self.round_histories[self.current_round_key].append(reduce_resp)

        final_md = _extract_first_markdown_block(reduce_resp.content) or reduce_resp.content.strip()
        self.scribed_transcript_markdown = final_md

        self._log(f"Final narrative length: {len(final_md)} chars", "INFO")
        self._log("Stage 7 complete - scribed narrative generated", "INFO")

        # Add to markdown transcript
        self._add_to_markdown(f"\n{final_md}\n")
        self._notify("stage_complete", {"stage": 7, "name": "Scribed Narrative"})

        return final_md

    def run(self, topic: str, personas: dict[str, str],
            progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
            on_error: Optional[Callable[[str, Exception, int], str]] = None) -> dict:
            """
            Executes a full structured debate.

            Args:
                topic (str): The topic of the debate.
                personas (dict): Mapping of agent names to their role prompts.
                progress_callback: Optional callback fired at stage boundaries and
                    key events.  Signature: ``callback(event: str, data: dict)``.
                on_error: Optional error callback fired when an LLM call fails.
                    Signature: ``on_error(agent_name: str, error: Exception, retry_count: int) -> str``
                    Return value must be one of: ``"retry"``, ``"skip"``, ``"abort"``.
                    If None, errors propagate normally.

            Returns:
                dict: Contains final positions, conclusions, and synthesis.
            """
            self._progress_callback = progress_callback
            self._on_error = on_error

            self._notify("debate_start", {
                "topic": topic,
                "num_agents": len(self.agents),
                "num_rounds": self.max_rounds,
            })

            log = "\n🚀 Debate Start: Topic →" + topic
            self.full_transcript.append(log)

            # Initialize the belief tracker
            log = "\n📦 Initializing Belief Tracker"
            self.full_transcript.append(log)
            embedding_tracker = BeliefEmbeddingTracker()

            # Initialize training data recorder if enabled
            if self.config and self.config.outputs.save_training_data:
                self.recorder = DebateRecorder(self.config, self.agents, topic)
                self._log("Training data recorder initialized", "INFO")

            # Stage 0: Briefing (agents already initialized)
            self.run_stage_0_briefing(topic, personas)

            # Agent personas loaded (debug log only — no print)
            for agent in self.agents:
                persona = personas.get(agent.name, "")
                self._log(f"Loaded persona for {agent.name}: {persona[:80]}...", "DEBUG")

            # Moderator: Generate roadmap (if stage2_mode == "moderated")
            if self.stage2_mode == "moderated" and self.moderator:
                self._log_header("MODERATOR: ROADMAP GENERATION")
                self._notify("agent_start", {"agent_name": "Moderator", "stage": "roadmap", "action": "Generating debate roadmap"})

                agent_personas = [
                    personas.get(agent.name, "").split(".")[0][:50]  # Short persona label
                    for agent in self.agents
                ]
                # Use persona_label if available (cleaner than extracting from prompt text)
                agent_persona_labels = [
                    getattr(agent, 'persona_label', agent_personas[i])
                    for i, agent in enumerate(self.agents)
                ]

                self.roadmap = self.moderator.generate_roadmap(
                    topic=topic,
                    num_rounds=self.max_rounds,
                    agent_personas=agent_persona_labels,
                )

                # Log the roadmap
                self._log(f"Roadmap generated with {len(self.roadmap.sub_topics)} sub-topics", "INFO")
                self._log(f"Overall rationale: {self.roadmap.overall_rationale}", "INFO")
                self._log(f"Sufficiency note: {self.roadmap.sufficiency_note}", "INFO")
                self._log_response("Moderator", self.roadmap.raw_response, "Roadmap Generation")

                # Add roadmap to markdown transcript
                roadmap_md = "\n# 📋 Debate Roadmap (Moderated)\n\n"
                roadmap_md += f"*Generated by the moderator agent*\n\n"
                roadmap_md += f"**Overall Rationale:** {self.roadmap.overall_rationale}\n\n"
                roadmap_md += f"**Sufficiency Assessment:** {self.roadmap.sufficiency_note}\n\n"
                for i, st in enumerate(self.roadmap.sub_topics, 1):
                    roadmap_md += f"### Round {i}: {st.title}\n"
                    roadmap_md += f"{st.description}\n"
                    if st.guiding_questions:
                        roadmap_md += "\n**Guiding questions:**\n"
                        for gq in st.guiding_questions:
                            roadmap_md += f"- {gq}\n"
                    roadmap_md += "\n"
                self._add_to_markdown(roadmap_md)

                # Notify: roadmap generated
                self._notify("roadmap_generated", {
                    "subtopics": [
                        {"round": i, "title": st.title, "description": st.description,
                         "guiding_questions": st.guiding_questions}
                        for i, st in enumerate(self.roadmap.sub_topics, 1)
                    ],
                    "sufficiency_note": self.roadmap.sufficiency_note,
                })

                # Record roadmap in training data
                if self.recorder:
                    self.recorder.record_roadmap_generation(
                        roadmap={
                            "sub_topics": [
                                {"title": st.title, "description": st.description,
                                 "rationale": st.rationale, "guiding_questions": st.guiding_questions}
                                for st in self.roadmap.sub_topics
                            ],
                            "overall_rationale": self.roadmap.overall_rationale,
                            "sufficiency_note": self.roadmap.sufficiency_note,
                        },
                        raw_response=self.roadmap.raw_response,
                    )
                    self.recorder.metadata["roadmap_user_modified"] = getattr(
                        self, "roadmap_user_modified", False
                    )

            # Stage 1: Opening Positions
            self.run_stage_1_opening_positions(topic)

            for round_idx in range(self.max_rounds):
                round_num = round_idx + 1
                self.current_round_key = f"round-{round_num}"
                self.round_histories[self.current_round_key] = []  # Initialize storage
                self._notify("round_start", {"round": round_num, "total_rounds": self.max_rounds})

                # Update recorder round
                if self.recorder:
                    self.recorder.set_round(round_num)

                # Track the beliefs of the agents
                for agent in self.agents:
                    belief_obj = agent.get_internal_belief_obj()
                    if belief_obj is not None:
                        embedding_tracker.embed_belief(belief_obj, agent_name=agent.name, round_num=round_num)
                    else:
                        text_for_embedding = agent.get_internal_belief()  # legacy fallback
                        embedding_tracker.embed_belief(text_for_embedding, agent_name=agent.name, round_num=round_num)

                # Stage 2: Cross-Examination
                # Determine focus subtopic for moderated mode
                current_focus_subtopic = None
                if self.stage2_mode == "moderated" and self.roadmap:
                    st = self.moderator.get_subtopic_for_round(round_idx)
                    if st:
                        current_focus_subtopic = {
                            "title": st.title,
                            "description": st.description,
                            "guiding_questions": st.guiding_questions,
                        }
                        self._log(f"Round {round_num} focus sub-topic: {st.title}", "INFO")

                self.run_stage_2_cross_examination(
                    only_if_disagree=False,
                    focus_subtopic=current_focus_subtopic,
                )

                # Stage 3: Response Phase (mode-dependent)
                stage3_mode = self.config.stage3_mode if self.config else "rebuttal"

                if stage3_mode == "collaborative":
                    # Stage 3B: Collaborative truth-seeking dialogue (includes embedded adjudication)
                    self.run_stage_3_collaborative()
                    # Stage 4: SKIP — adjudication already embedded in Stage 3B
                elif stage3_mode == "bloodsport":
                    # Stage 3C: Blood sport adversarial exchange (includes embedded adjudication)
                    self.run_stage_3_bloodsport()
                    # Stage 4: SKIP — adjudication already embedded in Stage 3C
                else:
                    # Stage 3: Single-shot rebuttals (default)
                    self.run_stage_3_rebuttals()
                    # Stage 4: Conflict Resolution
                    self.run_stage_4_conflict_resolution()

                # Stage 5: Belief Update (unified — works with both modes)
                self.run_stage_5_update_positions()

                # Calculate performance scores after each round
                self.agent_stats = calculate_performance_scores(self.agent_stats)

                # Log performance summary for this round
                perf_summary = get_performance_summary(self.agent_stats)
                self._log(f"\n{perf_summary}", "INFO")

                # Calculate convergence metrics (if enabled)
                convergence_data = None
                if self.config and hasattr(self.config, 'convergence') and self.config.convergence.enabled:
                    self._log("Calculating convergence metrics...", "INFO")

                    # Get agent beliefs
                    agent_beliefs = [agent.belief for agent in self.agents]

                    # Calculate claim-level agreement (reuse embedding model from tracker)
                    convergence_data = calculate_claim_agreement(
                        agent_beliefs,
                        embedding_model=embedding_tracker.model,
                        similarity_threshold=self.config.convergence.similarity_threshold
                    )

                    # Store in history if tracking enabled
                    if self.config.convergence.track_history:
                        self.convergence_history.append({
                            "round": round_num,
                            "convergence_score": convergence_data["convergence_score"],
                            "shared_claim_pairs": convergence_data["shared_claim_pairs"],
                            "unique_claims_count": len(convergence_data["unique_claims"])
                        })

                    # Log convergence summary
                    if self.config.convergence.display_in_round_summary:
                        agent_names = [a.name for a in self.agents]
                        conv_summary = format_convergence_summary(
                            convergence_data,
                            agent_names,
                            round_number=round_num
                        )
                        self._log(f"\n{conv_summary}", "INFO")

                # Notify: round complete with scores + convergence
                self._notify("round_complete", {
                    "round": round_num,
                    "scores": self.agent_stats,
                    "convergence": convergence_data,
                    "focus_subtopic": current_focus_subtopic,
                })

                # Adaptive moderator: review round and optionally revise roadmap
                if (self.stage2_mode == "moderated"
                        and self.moderator is not None
                        and self.moderator.config.moderator_mode == "adaptive"):

                    round_summary = self._build_round_summary(
                        round_num=round_num,
                        round_idx=round_idx,
                        focus_subtopic=current_focus_subtopic,
                        convergence_data=convergence_data,
                    )

                    revision = self.moderator.review_round(round_num, round_summary)

                    if revision is not None:
                        # Replace remaining sub-topics in the roadmap
                        completed = self.roadmap.sub_topics[:round_num]
                        self.roadmap.sub_topics = completed + revision.revised_sub_topics

                        self._log(
                            f"Roadmap revised after round {round_num}: "
                            f"{revision.revision_rationale}",
                            "INFO",
                        )

                        # Add revision to markdown transcript
                        revision_md = f"\n---\n\n**Roadmap Revised** (after round {round_num})\n\n"
                        revision_md += f"*Rationale:* {revision.revision_rationale}\n\n"
                        for i, st in enumerate(revision.revised_sub_topics, round_num + 1):
                            revision_md += f"- **Round {i}: {st.title}** — {st.description}\n"
                        revision_md += "\n"
                        self._add_to_markdown(revision_md)

                        # Fire display event
                        self._notify("roadmap_revised", {
                            "round_num": round_num,
                            "new_subtopics": [
                                {"title": st.title, "description": st.description}
                                for st in revision.revised_sub_topics
                            ],
                            "rationale": revision.revision_rationale,
                        })

                        # Record in training data
                        revision_record = {
                            "round_num": round_num,
                            "revision_rationale": revision.revision_rationale,
                            "new_subtopics": [
                                {"title": st.title, "description": st.description,
                                 "rationale": st.rationale,
                                 "guiding_questions": st.guiding_questions}
                                for st in revision.revised_sub_topics
                            ],
                        }
                        self.roadmap_revisions.append(revision_record)

                        if self.recorder:
                            self.recorder.record_event(
                                "roadmap_revision", revision_record,
                            )

            # Track the final beliefs of the agents
            for agent in self.agents:
                belief_obj = agent.get_internal_belief_obj()
                if belief_obj is not None:
                    embedding_tracker.embed_belief(belief_obj, agent_name=agent.name, round_num=self.max_rounds)
                else:
                    text_for_embedding = agent.get_internal_belief()  # legacy fallback
                    embedding_tracker.embed_belief(text_for_embedding, agent_name=agent.name, round_num=self.max_rounds)
            # Save the embeddings
            # Ensure storage directory exists
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            embedding_tracker.save_embeddings(STORAGE_DIR / "embeddings.npz")

            # Generate belief graph visualization (if enabled)
            if self.config and hasattr(self.config.outputs, 'generate_graph_visualization') and self.config.outputs.generate_graph_visualization:
                self._log("Generating interactive belief graph visualization...", "INFO")

                try:
                    graph_output_path = STORAGE_DIR / self.config.outputs.graph_file
                    html_content = export_debate_graph(
                        agents=self.agents,
                        topic=self.topic,
                        challenge_rebuttal_pairs=self.challenge_rebuttal_pairs,
                        output_path=graph_output_path
                    )

                    # Write HTML to file
                    with open(graph_output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    self._log(f"Belief graph visualization saved to {graph_output_path}", "INFO")
                except Exception as e:
                    self._log(f"Warning: Failed to generate belief graph visualization: {e}", "WARNING")

            # Stage 6: Concluding Reflections
            self.run_stage_6_concluding_remarks()

            # Calculate final performance scores
            self.agent_stats = calculate_performance_scores(self.agent_stats)

            # Log agent stats (display handled by callback / display layer)
            self._log(f"Final agent stats: {json.dumps(self.agent_stats, default=str)}", "INFO")

            # Log convergence trajectory (if enabled)
            if self.config and hasattr(self.config, 'convergence') and self.config.convergence.enabled:
                if self.config.convergence.display_in_final_summary and self.convergence_history:
                    conv_trajectory = get_convergence_trajectory_summary(self.convergence_history)
                    self._log(f"\n{conv_trajectory}", "INFO")

            # Stage 7: Scribe Summary
            # Instantiate Scribe with transcript
            self.final_synthesis = self.run_stage_7_scribing(self.scribe_agent)

            # Export training data (if enabled)
            if self.recorder and self.config:
                storage_dir = self.config.outputs.storage_dir
                storage_dir.mkdir(parents=True, exist_ok=True)

                training_path = storage_dir / self.config.outputs.training_data_file
                self.recorder.export_jsonl(training_path)
                self._log(f"Training data exported to {training_path}", "INFO")

                pairs_path = storage_dir / self.config.outputs.belief_pairs_file
                self.recorder.export_belief_training_pairs(pairs_path)
                self._log(f"Belief training pairs exported to {pairs_path}", "INFO")

            # Generate analysis report (if enabled)
            if self.config and self.config.outputs.save_analysis_report:
                storage_dir = self.config.outputs.storage_dir
                storage_dir.mkdir(parents=True, exist_ok=True)

                report_path = storage_dir / self.config.outputs.analysis_report_file

                # Build roadmap dict for reporting (if moderated mode was used)
                roadmap_dict = None
                if self.roadmap:
                    roadmap_dict = {
                        "sub_topics": [
                            {"title": st.title, "description": st.description,
                             "rationale": st.rationale, "guiding_questions": st.guiding_questions}
                            for st in self.roadmap.sub_topics
                        ],
                        "overall_rationale": self.roadmap.overall_rationale,
                        "sufficiency_note": self.roadmap.sufficiency_note,
                    }

                try:
                    report_md = generate_analysis_report(
                        config=self.config,
                        agents=self.agents,
                        challenge_rebuttal_pairs=self.challenge_rebuttal_pairs,
                        agent_stats=self.agent_stats,
                        convergence_history=self.convergence_history if self.convergence_history else None,
                        opening_positions=self.opening_positions,
                        roadmap=roadmap_dict,
                        roadmap_revisions=self.roadmap_revisions if self.roadmap_revisions else None,
                    )
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_md)
                    self._log(f"Analysis report saved to {report_path}", "INFO")

                    # Also save JSON version
                    json_report_path = report_path.with_suffix('.json')
                    report_json = generate_analysis_json(
                        config=self.config,
                        agents=self.agents,
                        challenge_rebuttal_pairs=self.challenge_rebuttal_pairs,
                        agent_stats=self.agent_stats,
                        convergence_history=self.convergence_history if self.convergence_history else None,
                        roadmap=roadmap_dict,
                        roadmap_revisions=self.roadmap_revisions if self.roadmap_revisions else None,
                    )
                    with open(json_report_path, 'w', encoding='utf-8') as f:
                        json.dump(report_json, f, ensure_ascii=False, indent=2)
                    self._log(f"Analysis JSON saved to {json_report_path}", "INFO")
                except Exception as e:
                    self._log(f"Warning: Failed to generate analysis report: {e}", "WARNING")

            # Final logging
            self._log_header("DEBATE COMPLETE")
            self._log(f"Total stages completed: 8 (including briefing)", "INFO")
            self._log(f"Debug log entries: {len(self.debug_log)}", "INFO")
            self._log(f"Markdown transcript entries: {len(self.markdown_transcript)}", "INFO")

            self._notify("debate_complete", {
                "agent_stats": self.agent_stats,
                "convergence_history": self.convergence_history,
            })

            return {
                "initial_positions": self.opening_positions,
                "final_positions": [agent.internal_belief for agent in self.agents],
                "conclusions": self.conclusions,
                "synthesis": self.final_synthesis,
                "full_transcript": "\n".join(self.full_transcript),  # Legacy: markdown transcript
                "markdown_transcript": "\n".join(self.markdown_transcript),  # New: clean markdown
                "debug_log": "\n".join(self.debug_log),  # New: comprehensive debug log
                "agent_stats": self.agent_stats
            }

# --- Pure Functions for Parallel Dispatch ---

def _generate_opening_position(agent, topic: str) -> dict:
    """Pure function: generate an opening position for one agent.

    Handles the full API call → parse → graph-validation retry loop.
    Returns a dict of results for the sequential apply step.

    This function is safe to call from a worker thread because it only
    reads/writes its own local state and the thread-safe agent.generate()
    method.
    """
    from chal.beliefs.belief_graph import BeliefGraph

    opening_prompt = prompts.build_stage_1_belief_prompt_cbs(
        topic=topic, agent_name=agent.name, persona_label=agent.persona_label
    )
    stage_request = [Message(role="user", content=opening_prompt)]
    response = agent.generate(stage_request)

    belief_obj, md_view, errs = parse_model_output_to_belief(response.content)

    validation_logs = []
    graph_metrics = None

    if belief_obj is not None:
        max_validation_retries = 3
        retry_count = 0
        validation_passed = False

        while retry_count < max_validation_retries and not validation_passed:
            try:
                graph = BeliefGraph(belief_obj)
                graph_errors = graph.validate_links()

                blocking_errors = [err for err in graph_errors if "BLOCKING ERROR" in err]
                warnings = [err for err in graph_errors if "BLOCKING ERROR" not in err]

                if blocking_errors:
                    validation_logs.append({
                        "message": f"BLOCKING validation errors for {agent.name} (attempt {retry_count + 1}/{max_validation_retries}): {blocking_errors}",
                        "level": "ERROR",
                    })
                    retry_count += 1
                    if retry_count < max_validation_retries:
                        error_summary = "\n".join(f"- {err}" for err in blocking_errors)
                        revision_prompt = (
                            f"Your belief structure contains validation errors that must be fixed:\n\n"
                            f"{error_summary}\n\n"
                            f"Please revise your belief to address these issues. Ensure that:\n"
                            f"1. All claims have supporting evidence or assumptions (no orphaned claims)\n"
                            f"2. Claims do not depend on themselves directly or indirectly (no circular dependencies)\n"
                            f"3. All referenced IDs exist in your belief structure\n\n"
                            f"Provide your revised CBS belief object."
                        )
                        revision_response = agent.generate([Message(role="user", content=revision_prompt)])
                        validation_logs.append({
                            "message": f"Received revision from {agent.name} ({len(revision_response.content)} chars)",
                            "level": "INFO",
                        })
                        belief_obj, md_view, errs = parse_model_output_to_belief(revision_response.content)
                        if not belief_obj:
                            validation_logs.append({
                                "message": f"Failed to parse revised belief from {agent.name}",
                                "level": "ERROR",
                            })
                            break
                        if errs:
                            validation_logs.append({
                                "message": f"Revised belief has warnings: {errs}",
                                "level": "WARN",
                            })
                    else:
                        validation_logs.append({
                            "message": f"Max retries ({max_validation_retries}) reached for {agent.name}. Using last attempt despite errors.",
                            "level": "ERROR",
                        })
                else:
                    validation_passed = True
                    if warnings:
                        for warn in warnings:
                            validation_logs.append({
                                "message": f"Graph validation warning for {agent.name}: {warn}",
                                "level": "WARN",
                            })
                    else:
                        validation_logs.append({
                            "message": f"Graph validation passed for {agent.name}",
                            "level": "INFO",
                        })
                    graph_metrics = graph.get_graph_metrics()

            except Exception as e:
                validation_logs.append({
                    "message": f"Graph validation error for {agent.name}: {e}",
                    "level": "ERROR",
                })
                retry_count += 1
                if retry_count >= max_validation_retries:
                    break

    return {
        "response": response,
        "belief_obj": belief_obj,
        "md_view": md_view,
        "errs": errs,
        "prompt": opening_prompt,
        "graph_metrics": graph_metrics,
        "validation_logs": validation_logs,
    }


def _generate_cross_examination(challenger, target, topic, config, previous_challenges,
                                 focus_subtopic) -> dict:
    """Pure function: generate cross-examination questions from challenger to target.

    Returns a dict of results for the sequential apply step.
    """
    challenger_name = challenger.name
    target_name = target.name

    ch_belief_obj = challenger.get_internal_belief_obj()
    tg_belief_obj = target.get_internal_belief_obj()
    ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
    tg_belief_json = json.dumps(tg_belief_obj, ensure_ascii=False, indent=2) if tg_belief_obj else ""

    max_questions = config.stages.max_questions_per_cross_exam if config else 5
    stage3_mode = config.stage3_mode if config else "rebuttal"

    if stage3_mode == "bloodsport":
        intensity = config.bloodsport.intensity if config else "moderate"
        prompt = prompts.build_stage_2_bloodsport_prompt(
            topic=topic,
            agent_name=challenger_name,
            opponent_name=target_name,
            agent_belief_json=ch_belief_json,
            opponent_belief_json=tg_belief_json,
            intensity=intensity,
            max_questions=max_questions,
            previous_challenges=previous_challenges if previous_challenges else None,
            focus_subtopic=focus_subtopic,
        )
    else:
        prompt = prompts.build_stage_2_prompt(
            topic=topic,
            agent_name=challenger_name,
            opponent_name=target_name,
            agent_belief_json=ch_belief_json,
            opponent_belief_json=tg_belief_json,
            max_questions=max_questions,
            previous_challenges=previous_challenges if previous_challenges else None,
            focus_subtopic=focus_subtopic,
        )

    stage_request = [Message(role="user", content=prompt)]
    generation_temp = config.stages.generation_temperature if config else 0.2
    response = challenger.generate(stage_request, temperature=generation_temp)

    # Parse the FIRST fenced JSON block -> {"questions":[...]}
    questions_obj = _extract_first_json_block(response.content)
    questions = (questions_obj or {}).get("questions", [])

    # Fallback to legacy parser
    parsed_challenges = []
    if not questions:
        parsed_challenges = parse_challenges(response.content)

    return {
        "response": response,
        "prompt": prompt,
        "questions": questions,
        "parsed_challenges": parsed_challenges,
        "ch_belief_obj": ch_belief_obj,
        "tg_belief_obj": tg_belief_obj,
    }


def _generate_rebuttal(target_agent, relevant_entries, topic, config) -> dict:
    """Pure function: generate rebuttals for one target agent.

    Returns a dict of results for the sequential apply step.
    """
    target_name = target_agent.name

    questions_payload = {
        "questions": [
            {
                "qid": e.get("qid") or f"Q{idx+1}",
                "text": e["challenge"],
                "target_ids": e.get("target_ids", []),
                "from": e["challenger"]
            }
            for idx, e in enumerate(relevant_entries)
        ]
    }
    received_questions_json = json.dumps(questions_payload, ensure_ascii=False, indent=2)

    tgt_belief_obj = target_agent.get_internal_belief_obj()
    tgt_belief_json = json.dumps(tgt_belief_obj, ensure_ascii=False, indent=2) if tgt_belief_obj else ""

    opponent_name = questions_payload["questions"][0]["from"] if questions_payload["questions"] else "Opponent"

    max_rebuttals = config.stages.max_rebuttals_per_response if config else 5
    max_rebuttal_length = config.stages.max_rebuttal_length_chars if config else 500
    prompt = prompts.build_stage_3_structured_rebuttal_prompt(
        topic=topic,
        agent_name=target_name,
        opponent_name=opponent_name,
        received_questions_json=received_questions_json,
        agent_belief_json=tgt_belief_json,
        max_rebuttals=min(max_rebuttals, len(questions_payload["questions"])),
        max_rebuttal_length_chars=max_rebuttal_length
    )

    stage_request = [Message(role="user", content=prompt)]
    generation_temp = config.stages.generation_temperature if config else 0.2
    response = target_agent.generate(stage_request, temperature=generation_temp)

    # Parse JSON: prompt requests a single block with both "rebuttals" and "patches" keys.
    # Fallback: if two separate blocks exist, treat block[0] as rebuttals, block[1] as patches.
    blocks = _extract_all_json_blocks(response.content)
    if blocks:
        first_block = json.loads(blocks[0])
        if "rebuttals" in first_block and ("patches" in first_block or len(blocks) == 1):
            # Single unified block (expected path)
            rebuttals = first_block.get("rebuttals", [])
            patches = first_block.get("patches", [])
        elif len(blocks) > 1:
            # Legacy two-block format (backward compat)
            rebuttals = first_block.get("rebuttals", [])
            patches_block = json.loads(blocks[1])
            patches = patches_block.get("patches", [])
        else:
            rebuttals = first_block.get("rebuttals", [])
            patches = first_block.get("patches", [])
    else:
        rebuttals = []
        patches = []

    return {
        "response": response,
        "prompt": prompt,
        "rebuttals": rebuttals,
        "patches": patches,
        "received_questions_json": received_questions_json,
        "tgt_belief_obj": tgt_belief_obj,
        "relevant_entries": relevant_entries,
    }


def _run_adjudication(adjudicator, entry, agent_by_name) -> dict:
    """Pure function: adjudicate a single challenge-rebuttal pair.

    Returns a dict of results for the sequential apply step.
    """
    target_ids = entry.get("target_ids", [])
    challenger_belief_excerpt_json = ""
    target_belief_excerpt_json = ""
    if target_ids:
        challenger_agent = agent_by_name.get(entry["challenger"])
        target_agent = agent_by_name.get(entry["target"])
        if challenger_agent:
            ch_belief = challenger_agent.get_internal_belief_obj()
            if ch_belief:
                excerpt = _extract_belief_excerpt(ch_belief, target_ids)
                if excerpt:
                    challenger_belief_excerpt_json = json.dumps(excerpt, ensure_ascii=False, indent=2)
        if target_agent:
            tgt_belief = target_agent.get_internal_belief_obj()
            if tgt_belief:
                excerpt = _extract_belief_excerpt(tgt_belief, target_ids)
                if excerpt:
                    target_belief_excerpt_json = json.dumps(excerpt, ensure_ascii=False, indent=2)

    resolution = adjudicator.run(
        challenge=entry["challenge"],
        rebuttal=entry["rebuttal"],
        challenger=entry["challenger"],
        target=entry["target"],
        challenger_belief_excerpt_json=challenger_belief_excerpt_json,
        target_belief_excerpt_json=target_belief_excerpt_json
    )

    return {"resolution": resolution}


def _generate_conclusion(agent, topic, config, cr_pairs_json, current_round) -> dict:
    """Pure function: generate concluding remarks for one agent.

    Returns a dict of results for the sequential apply step.
    """
    name = agent.name
    a_belief_obj = agent.get_internal_belief_obj()
    a_belief_json = json.dumps(a_belief_obj, ensure_ascii=False, indent=2) if a_belief_obj else ""

    changelog_entries = []
    for i, belief_json_str in enumerate(agent.all_beliefs_held):
        try:
            belief = json.loads(belief_json_str) if isinstance(belief_json_str, str) else belief_json_str
            for entry in belief.get("changelog", []):
                changelog_entries.append(
                    f"v{entry.get('version', i+1)}: {'; '.join(entry.get('changes', []))}"
                )
        except Exception:
            pass
    belief_changelog_summary = "\n".join(changelog_entries) if changelog_entries else "(no changelog available)"

    prompt = prompts.build_stage_6_conclusion_prompt(
        topic=topic,
        agent_name=name,
        agent_belief_json=a_belief_json,
        belief_changelog_summary=belief_changelog_summary,
        num_rounds=current_round,
        persona_label=agent.persona_label if hasattr(agent, "persona_label") else ""
    )

    stage_request = [Message(role="user", content=prompt)]
    generation_temp = config.stages.generation_temperature if config else 0.2
    response = agent.generate(stage_request, temperature=generation_temp)

    return {
        "response": response,
        "prompt": prompt,
        "a_belief_obj": a_belief_obj,
    }


def _apply_patches_and_validate(
    name: str,
    response_content: str,
    prior_json: dict,
    relevant_entries: list,
    md_view_fallback: str,
) -> dict:
    """Parse patches from response, apply to belief, validate graph.

    Pure function — no agent or controller mutations.
    Used by the bloodsport and legacy flows within _run_stage5_for_agent.

    Returns:
        dict with keys: patches, final_belief, md_view, reverted,
        log_entries, debug_entries.
    """
    from chal.beliefs.patches import apply_patches, validate_patches
    from chal.beliefs.belief_graph import BeliefGraph

    log_entries = []
    debug_entries = []
    patches = []

    blocks = _extract_all_json_blocks(response_content)

    if not blocks:
        log_entries.append((f"PARSE FAILURE: No JSON blocks found in {name} response", "PARSE"))
        log_entries.append(("Keeping prior belief unchanged", "WARN"))
        return {
            "patches": patches,
            "final_belief": prior_json,
            "md_view": md_view_fallback,
            "reverted": True,
            "log_entries": log_entries,
            "debug_entries": debug_entries,
        }

    patches_block = json.loads(blocks[0])
    patches = patches_block.get("patches", [])

    if not patches:
        critique_valid_count = sum(
            1 for entry in relevant_entries
            if (entry.get("resolution") or {}).get("status") == "critique_valid"
        )
        if critique_valid_count > 0:
            log_entries.append((
                f"PARSE FAILURE: WARNING: {name} received {critique_valid_count} CRITIQUE_VALID "
                f"outcome(s) but returned no patches.",
                "PARSE",
            ))
            log_entries.append((
                f"ENFORCEMENT FAILURE: Agent {name} ignored mandatory patch requirement",
                "ERROR",
            ))
        else:
            log_entries.append((
                f"PARSE FAILURE: No patches returned for {name}, keeping prior belief",
                "PARSE",
            ))
        return {
            "patches": patches,
            "final_belief": prior_json,
            "md_view": md_view_fallback,
            "reverted": True,
            "log_entries": log_entries,
            "debug_entries": debug_entries,
        }

    log_entries.append((
        f"PARSE SUCCESS: Successfully parsed {len(patches)} patch(es) for {name}",
        "PARSE",
    ))
    debug_entries.append(
        f"\n--- PATCHES FOR {name} ---\n"
        f"{json.dumps(patches, ensure_ascii=False, indent=2)}\n"
        f"--- END PATCHES ---\n"
    )

    if not prior_json:
        log_entries.append((f"No prior belief object for {name}, cannot apply patches", "WARN"))
        return {
            "patches": patches,
            "final_belief": None,
            "md_view": md_view_fallback,
            "reverted": True,
            "log_entries": log_entries,
            "debug_entries": debug_entries,
        }

    patch_errors = validate_patches(patches, prior_json)
    if patch_errors:
        invalid_indices = set(patch_errors.keys())
        for idx in sorted(invalid_indices):
            log_entries.append((f"SKIPPING invalid patch {idx} ({patches[idx].get('op', '?')}): {'; '.join(patch_errors[idx])}", "WARN"))
        patches = [p for i, p in enumerate(patches) if i not in invalid_indices]

    try:
        updated_belief = apply_patches(prior_json, patches, propagate_strength=True)
        debug_entries.append(
            f"\n--- UPDATED BELIEF FOR {name} ---\n"
            f"{json.dumps(updated_belief, ensure_ascii=False, indent=2)}\n"
            f"--- END UPDATED BELIEF ---\n"
        )

        try:
            validation_graph = BeliefGraph(updated_belief)
            validation_errors = validation_graph.validate_links()
            blocking_errors = [err for err in validation_errors if "BLOCKING ERROR" in err]
            warnings = [err for err in validation_errors if "BLOCKING ERROR" not in err]

            if blocking_errors:
                log_entries.append((f"BLOCKING validation errors for {name}:", "ERROR"))
                for err in blocking_errors:
                    log_entries.append((f"  - {err}", "ERROR"))
                log_entries.append((f"Reverting to prior belief for {name}", "ERROR"))
                raise Exception(f"Updated belief contains blocking validation errors: {blocking_errors}")

            if warnings:
                log_entries.append((f"Graph validation warnings for {name}:", "WARN"))
                for warn in warnings:
                    log_entries.append((f"  - {warn}", "WARN"))
            else:
                log_entries.append((f"Graph validation passed for {name}", "INFO"))

        except Exception as e:
            if "blocking validation errors" in str(e).lower():
                raise
            log_entries.append((f"Graph validation error for {name}: {e}", "ERROR"))

        # Success: patches applied, no blocking errors
        log_entries.append((f"Applied {len(patches)} patches for {name}", "INFO"))
        return {
            "patches": patches,
            "final_belief": updated_belief,
            "md_view": belief_to_markdown(updated_belief),
            "reverted": False,
            "log_entries": log_entries,
            "debug_entries": debug_entries,
        }

    except Exception as e:
        log_entries.append((f"Error applying patches for {name}: {e}", "ERROR"))
        log_entries.append(("Keeping prior belief unchanged", "WARN"))
        return {
            "patches": patches,
            "final_belief": prior_json,
            "md_view": md_view_fallback,
            "reverted": True,
            "log_entries": log_entries,
            "debug_entries": debug_entries,
        }


def _run_stage5_for_agent(
    agent,
    relevant_entries: list,
    stage3_mode: str,
    last_rebuttals_patches: list,
    generation_temp: float,
    max_retries: int,
) -> dict:
    """Pure function: run full Stage 5 belief update for one agent.

    Handles all three flow types (bloodsport, CBS two-phase, legacy).
    Returns a dict of results for the sequential apply step.

    This function is safe to call from a worker thread because it only
    reads agent state (get_internal_belief_obj), makes API calls via
    agent.generate(), and performs local computation. It does NOT mutate
    agent belief state or any shared controller state.
    """
    name = agent.name
    prior_json = agent.get_internal_belief_obj()
    md_view_fallback = agent.get_internal_belief()
    api_calls = []
    log_entries = []
    debug_entries = []

    log_entries.append((f"\n--- Updating belief for: {name} ---", "INFO"))
    log_entries.append((f"Agent received {len(relevant_entries)} adjudication result(s)", "INFO"))

    if prior_json is not None and stage3_mode == "bloodsport":
        return _run_stage5_bloodsport(
            agent, name, prior_json, md_view_fallback,
            relevant_entries, last_rebuttals_patches,
            generation_temp, max_retries,
            api_calls, log_entries, debug_entries,
        )

    elif prior_json is not None:
        return _run_stage5_cbs_two_phase(
            agent, name, prior_json, md_view_fallback,
            relevant_entries, last_rebuttals_patches,
            generation_temp, max_retries,
            api_calls, log_entries, debug_entries,
        )

    else:
        return _run_stage5_legacy(
            agent, name, prior_json, md_view_fallback,
            relevant_entries,
            generation_temp, max_retries,
            api_calls, log_entries, debug_entries,
        )


def _run_stage5_bloodsport(
    agent, name, prior_json, md_view_fallback,
    relevant_entries, last_rebuttals_patches,
    generation_temp, max_retries,
    api_calls, log_entries, debug_entries,
) -> dict:
    """Bloodsport single-phase flow for _run_stage5_for_agent."""
    log_entries.append((f"Using bloodsport belief update prompt for {name}", "DEBUG"))

    bloodsport_exchanges = []
    for e in relevant_entries:
        if "bloodsport_transcript" in e:
            bloodsport_exchanges.extend(e["bloodsport_transcript"])

    stage_3_patches_json = (
        json.dumps(last_rebuttals_patches, ensure_ascii=False, indent=2)
        if last_rebuttals_patches else ""
    )

    prompt = prompts.build_stage_5_bloodsport_prompt(
        agent_name=agent.name,
        challenge_rebuttal_pairs=relevant_entries,
        prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2),
        bloodsport_exchanges=bloodsport_exchanges if bloodsport_exchanges else None,
        stage_3_patches_json=stage_3_patches_json,
    )

    label = "Stage 5 - Belief Update (Bloodsport)"
    stage_request = [Message(role="user", content=prompt)]
    log_entries.append((f"Calling model for {name} belief update (bloodsport)...", "INFO"))
    response = agent.generate(stage_request, temperature=generation_temp)
    api_calls.append({"prompt": prompt, "response": response, "label": label, "is_retry": False})

    # Retry if no JSON blocks found
    if not _extract_all_json_blocks(response.content):
        for bs_attempt in range(1, max_retries + 1):
            log_entries.append((
                f"[Stage 5 Bloodsport] Parse retry {bs_attempt}/{max_retries} for {name}: no JSON blocks",
                "WARN",
            ))
            response = agent.generate(stage_request, temperature=generation_temp)
            api_calls.append({
                "prompt": prompt, "response": response,
                "label": f"Stage 5 - Bloodsport retry {bs_attempt}",
                "is_retry": True,
            })
            if _extract_all_json_blocks(response.content):
                log_entries.append((
                    f"[Stage 5 Bloodsport] Retry {bs_attempt} succeeded for {name}",
                    "INFO",
                ))
                break

    patch_result = _apply_patches_and_validate(
        name, response.content, prior_json, relevant_entries, md_view_fallback
    )

    return {
        "flow": "bloodsport",
        "api_calls": api_calls,
        "prior_json": prior_json,
        "final_belief": patch_result["final_belief"],
        "md_view": patch_result["md_view"],
        "patches": patch_result["patches"],
        "phase1_patches": None,
        "phase2_patches": None,
        "reverted": patch_result["reverted"],
        "log_entries": log_entries + patch_result["log_entries"],
        "debug_entries": debug_entries + patch_result["debug_entries"],
    }


def _run_stage5_cbs_two_phase(
    agent, name, prior_json, md_view_fallback,
    relevant_entries, last_rebuttals_patches,
    generation_temp, max_retries,
    api_calls, log_entries, debug_entries,
) -> dict:
    """CBS two-phase flow (Phase 1: enforcement, Phase 2: introspection)."""
    log_entries.append((f"Using two-phase CBS belief update for {name}", "DEBUG"))

    from chal.beliefs.patches import apply_patches, validate_patches
    from chal.beliefs.belief_graph import BeliefGraph

    stage_3_patches_json = (
        json.dumps(last_rebuttals_patches, ensure_ascii=False, indent=2)
        if last_rebuttals_patches else ""
    )
    prior_belief_json_str = json.dumps(prior_json, ensure_ascii=False, indent=2)

    # --- PHASE 1: Adjudication Enforcement ---
    log_entries.append((f"Phase 1: Adjudication enforcement for {name}", "INFO"))

    phase1_prompt = prompts.build_stage_5_phase1_enforcement_prompt(
        agent_name=name,
        challenge_rebuttal_pairs=relevant_entries,
        prior_belief_json=prior_belief_json_str,
        stage_3_patches_json=stage_3_patches_json,
    )

    phase1_response = agent.generate(
        [Message(role="user", content=phase1_prompt)],
        temperature=generation_temp,
    )
    api_calls.append({
        "prompt": phase1_prompt, "response": phase1_response,
        "label": "Stage 5 Phase 1 - Enforcement", "is_retry": False,
    })

    # Parse Phase 1 patches (with retry if no JSON blocks)
    phase1_patches = []
    phase1_blocks = _extract_all_json_blocks(phase1_response.content)
    if not phase1_blocks:
        for p1_attempt in range(1, max_retries + 1):
            log_entries.append((
                f"[Stage 5 Phase 1] Parse retry {p1_attempt}/{max_retries} for {name}: no JSON blocks",
                "WARN",
            ))
            phase1_response = agent.generate(
                [Message(role="user", content=phase1_prompt)],
                temperature=generation_temp,
            )
            api_calls.append({
                "prompt": phase1_prompt, "response": phase1_response,
                "label": f"Stage 5 Phase 1 - Enforcement retry {p1_attempt}",
                "is_retry": True,
            })
            phase1_blocks = _extract_all_json_blocks(phase1_response.content)
            if phase1_blocks:
                log_entries.append((
                    f"[Stage 5 Phase 1] Retry {p1_attempt} succeeded for {name}",
                    "INFO",
                ))
                break

    intermediate_belief = prior_json  # Default: unchanged

    if phase1_blocks:
        phase1_patches_block = json.loads(phase1_blocks[0])
        phase1_patches = phase1_patches_block.get("patches", [])

    if phase1_patches:
        log_entries.append((f"Phase 1: parsed {len(phase1_patches)} patch(es) for {name}", "INFO"))
        debug_entries.append(
            f"\n--- PHASE 1 PATCHES FOR {name} ---\n"
            f"{json.dumps(phase1_patches, ensure_ascii=False, indent=2)}\n"
            f"--- END PHASE 1 PATCHES ---\n"
        )

        patch_errors = validate_patches(phase1_patches, prior_json)
        if patch_errors:
            invalid_indices = set(patch_errors.keys())
            for idx in sorted(invalid_indices):
                log_entries.append((f"SKIPPING invalid Phase 1 patch {idx} ({phase1_patches[idx].get('op', '?')}): {'; '.join(patch_errors[idx])}", "WARN"))
            phase1_patches = [p for i, p in enumerate(phase1_patches) if i not in invalid_indices]

        try:
            intermediate_belief = apply_patches(prior_json, phase1_patches, propagate_strength=True)
            log_entries.append((f"Phase 1: applied {len(phase1_patches)} patches for {name}", "INFO"))
        except Exception as e:
            log_entries.append((f"Phase 1: error applying patches for {name}: {e}", "ERROR"))
            intermediate_belief = prior_json
    else:
        # Check enforcement compliance
        critique_valid_count = sum(
            1 for entry in relevant_entries
            if (entry.get("resolution") or {}).get("status") == "critique_valid"
        )
        if critique_valid_count > 0:
            log_entries.append((
                f"ENFORCEMENT FAILURE: {name} received {critique_valid_count} CRITIQUE_VALID "
                f"outcome(s) but Phase 1 returned no patches",
                "ERROR",
            ))
        else:
            log_entries.append((f"Phase 1: no patches for {name} (no enforcement needed)", "INFO"))

    # Summarize Phase 1 changes for Phase 2 context
    phase1_summary = summarize_changes(phase1_patches, prior_json, intermediate_belief)
    log_entries.append((f"Phase 1 summary for {name}:\n{phase1_summary}", "DEBUG"))

    # --- PHASE 2: Introspective Evaluation ---
    log_entries.append((f"Phase 2: Introspective evaluation for {name}", "INFO"))

    phase2_prompt = prompts.build_stage_5_phase2_introspection_prompt(
        agent_name=name,
        intermediate_belief_json=json.dumps(intermediate_belief, ensure_ascii=False, indent=2),
        phase1_changes_summary=phase1_summary,
    )

    phase2_response = agent.generate(
        [Message(role="user", content=phase2_prompt)],
        temperature=generation_temp,
    )
    api_calls.append({
        "prompt": phase2_prompt, "response": phase2_response,
        "label": "Stage 5 Phase 2 - Introspection", "is_retry": False,
    })

    # Parse Phase 2 patches (with retry if no JSON blocks)
    phase2_patches = []
    phase2_blocks = _extract_all_json_blocks(phase2_response.content)
    if not phase2_blocks:
        for p2_attempt in range(1, max_retries + 1):
            log_entries.append((
                f"[Stage 5 Phase 2] Parse retry {p2_attempt}/{max_retries} for {name}: no JSON blocks",
                "WARN",
            ))
            phase2_response = agent.generate(
                [Message(role="user", content=phase2_prompt)],
                temperature=generation_temp,
            )
            api_calls.append({
                "prompt": phase2_prompt, "response": phase2_response,
                "label": f"Stage 5 Phase 2 - Introspection retry {p2_attempt}",
                "is_retry": True,
            })
            phase2_blocks = _extract_all_json_blocks(phase2_response.content)
            if phase2_blocks:
                log_entries.append((
                    f"[Stage 5 Phase 2] Retry {p2_attempt} succeeded for {name}",
                    "INFO",
                ))
                break

    final_belief = intermediate_belief  # Default: Phase 1 result

    if phase2_blocks:
        phase2_patches_block = json.loads(phase2_blocks[0])
        phase2_patches = phase2_patches_block.get("patches", [])

    if phase2_patches:
        # GUARDRAIL: Filter reversals
        original_count = len(phase2_patches)
        phase2_patches = filter_reversal_patches(phase2_patches, phase1_patches, intermediate_belief)
        filtered_count = original_count - len(phase2_patches)
        if filtered_count > 0:
            log_entries.append((
                f"Phase 2: filtered {filtered_count} reversal patch(es) for {name}",
                "WARN",
            ))

        log_entries.append((f"Phase 2: parsed {len(phase2_patches)} patch(es) for {name}", "INFO"))
        debug_entries.append(
            f"\n--- PHASE 2 PATCHES FOR {name} ---\n"
            f"{json.dumps(phase2_patches, ensure_ascii=False, indent=2)}\n"
            f"--- END PHASE 2 PATCHES ---\n"
        )

        if phase2_patches:
            patch_errors = validate_patches(phase2_patches, intermediate_belief)
            if patch_errors:
                invalid_indices = set(patch_errors.keys())
                for idx in sorted(invalid_indices):
                    log_entries.append((f"SKIPPING invalid Phase 2 patch {idx} ({phase2_patches[idx].get('op', '?')}): {'; '.join(patch_errors[idx])}", "WARN"))
                phase2_patches = [p for i, p in enumerate(phase2_patches) if i not in invalid_indices]

            try:
                final_belief = apply_patches(intermediate_belief, phase2_patches, propagate_strength=True)
                log_entries.append((f"Phase 2: applied {len(phase2_patches)} patches for {name}", "INFO"))
            except Exception as e:
                log_entries.append((f"Phase 2: error applying patches for {name}: {e}", "ERROR"))
                final_belief = intermediate_belief
    else:
        log_entries.append((f"Phase 2: no patches for {name}", "INFO"))

    # Validate final belief structure
    md_view = md_view_fallback
    reverted = True  # Pessimistic default

    try:
        validation_graph = BeliefGraph(final_belief)
        validation_errors = validation_graph.validate_links()
        blocking_errors = [err for err in validation_errors if "BLOCKING ERROR" in err]
        warnings = [err for err in validation_errors if "BLOCKING ERROR" not in err]

        if blocking_errors:
            log_entries.append((f"BLOCKING validation errors in final belief for {name}:", "ERROR"))
            for err in blocking_errors:
                log_entries.append((f"  - {err}", "ERROR"))
            log_entries.append((f"Reverting to prior belief for {name}", "ERROR"))
        else:
            if warnings:
                log_entries.append((f"Graph validation warnings for {name}:", "WARN"))
                for warn in warnings:
                    log_entries.append((f"  - {warn}", "WARN"))
            else:
                log_entries.append((f"Graph validation passed for {name}", "INFO"))

            md_view = belief_to_markdown(final_belief)
            debug_entries.append(
                f"\n--- FINAL BELIEF FOR {name} ---\n"
                f"{json.dumps(final_belief, ensure_ascii=False, indent=2)}\n"
                f"--- END FINAL BELIEF ---\n"
            )
            log_entries.append((
                f"Two-phase update complete for {name}: "
                f"{len(phase1_patches)} P1 + {len(phase2_patches)} P2 patches",
                "INFO",
            ))
            reverted = False

    except Exception as e:
        log_entries.append((f"Graph validation error for {name}: {e}", "ERROR"))

    return {
        "flow": "cbs_two_phase",
        "api_calls": api_calls,
        "prior_json": prior_json,
        "final_belief": final_belief if not reverted else prior_json,
        "md_view": md_view,
        "patches": phase1_patches + phase2_patches,
        "phase1_patches": phase1_patches,
        "phase2_patches": phase2_patches,
        "reverted": reverted,
        "log_entries": log_entries,
        "debug_entries": debug_entries,
    }


def _run_stage5_legacy(
    agent, name, prior_json, md_view_fallback,
    relevant_entries,
    generation_temp, max_retries,
    api_calls, log_entries, debug_entries,
) -> dict:
    """Legacy single-phase flow for _run_stage5_for_agent."""
    log_entries.append((f"Using legacy belief update prompt for {name}", "DEBUG"))

    prompt = prompts.build_stage_5_belief_update_prompt_cbs(
        agent_name=agent.name,
        challenge_rebuttal_pairs=relevant_entries,
        prior_belief_json=(
            json.dumps(prior_json, ensure_ascii=False, indent=2)
            if prior_json else "{}"
        ),
    )

    label = "Stage 5 - Belief Update (Legacy)"
    stage_request = [Message(role="user", content=prompt)]
    log_entries.append((f"Calling model for {name} belief update (legacy)...", "INFO"))
    response = agent.generate(stage_request, temperature=generation_temp)
    api_calls.append({"prompt": prompt, "response": response, "label": label, "is_retry": False})

    # Retry if no JSON blocks found
    if not _extract_all_json_blocks(response.content):
        for leg_attempt in range(1, max_retries + 1):
            log_entries.append((
                f"[Stage 5 Legacy] Parse retry {leg_attempt}/{max_retries} for {name}: no JSON blocks",
                "WARN",
            ))
            response = agent.generate(stage_request, temperature=generation_temp)
            api_calls.append({
                "prompt": prompt, "response": response,
                "label": f"Stage 5 - Legacy retry {leg_attempt}",
                "is_retry": True,
            })
            if _extract_all_json_blocks(response.content):
                log_entries.append((
                    f"[Stage 5 Legacy] Retry {leg_attempt} succeeded for {name}",
                    "INFO",
                ))
                break

    patch_result = _apply_patches_and_validate(
        name, response.content, prior_json, relevant_entries, md_view_fallback
    )

    return {
        "flow": "legacy",
        "api_calls": api_calls,
        "prior_json": prior_json,
        "final_belief": patch_result["final_belief"],
        "md_view": patch_result["md_view"],
        "patches": patch_result["patches"],
        "phase1_patches": None,
        "phase2_patches": None,
        "reverted": patch_result["reverted"],
        "log_entries": log_entries + patch_result["log_entries"],
        "debug_entries": debug_entries + patch_result["debug_entries"],
    }


# --- Helper Functions ---
def _extract_belief_excerpt(belief: dict, target_ids: list) -> dict:
    """Extract claims, assumptions, evidence, and counterpositions referenced by target_ids."""
    excerpt = {}
    for section in ["assumptions", "claims", "evidence", "counterpositions"]:
        items = [item for item in belief.get(section, [])
                 if item.get("id") in target_ids or
                 any(tid in item.get("depends_on", []) for tid in target_ids) or
                 any(tid in item.get("targets", []) for tid in target_ids)]
        if items:
            excerpt[section] = items
    return excerpt

def _normalize_collab_verdict(raw: str) -> str:
    """Normalize a collaborative-mode verdict to a valid value."""
    _VALID = {"critique_valid", "rebuttal_valid", "unresolved"}
    cleaned = raw.strip().lower()
    return cleaned if cleaned in _VALID else "unresolved"


def _extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    # Try fenced block first, then fall back to raw JSON object
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    # No fenced block — try to parse the first raw JSON object
    raw = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if raw:
        try:
            return json.loads(raw.group(1))
        except Exception:
            return None
    return None


def _validate_cross_exam_result(r: dict) -> bool:
    """Check whether a parsed Stage 2 result dict is valid enough to accept.

    - If it contains a ``questions`` list, every question must pass
      ``validate_stage2_questions`` (field presence, attack_type/strategy match, etc.).
    - If it only contains ``parsed_challenges`` (legacy fallback), accept as-is.
    - Otherwise reject.
    """
    questions = r.get("questions")
    if questions:
        is_valid, _errors = validate_stage2_questions(questions)
        return is_valid
    if r.get("parsed_challenges"):
        return True
    return False


def _extract_all_json_blocks(text: str) -> List[str]:
    # Try fenced blocks first
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if blocks:
        return blocks
    # No fenced blocks — extract all top-level JSON objects using brace-depth tracking
    raw_blocks = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j + 1]
                        try:
                            json.loads(candidate)
                            raw_blocks.append(candidate)
                        except Exception:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return raw_blocks

def _extract_first_markdown_block(text: str) -> Optional[str]:
    m = re.search(r"```markdown\s*(.*?)\s*```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def summarize_changes(patches: list, before: dict, after: dict) -> str:
    """Produce a human-readable summary of what changed between two belief states.

    Summarises the patches applied and any strength changes detected, suitable
    for inclusion in the Phase 2 prompt context.
    """
    lines: list[str] = []

    # Summarise each patch operation
    for p in patches:
        op = p.get("op", "?")
        if op == "update_thesis":
            new_str = p.get("new_strength")
            change = p.get("change")
            if new_str is not None:
                lines.append(f"- Thesis strength set to {new_str}")
            elif change:
                lines.append(f"- Thesis strength: {change}")
        elif op == "update_claim":
            tid = p.get("target_id", "?")
            changes = p.get("changes", {})
            parts = []
            if "strength" in changes:
                parts.append(f"strength→{changes['strength']}")
            if "status" in changes:
                parts.append(f"status→{changes['status']}")
            lines.append(f"- Updated {tid}: {', '.join(parts) if parts else 'fields updated'}")
        elif op == "retire_claim":
            lines.append(f"- Retired {p.get('target_id', '?')}")
        elif op == "add_evidence":
            item = p.get("item", {})
            lines.append(f"- Added evidence {item.get('id', '?')}: {item.get('summary', '')[:80]}")
        elif op == "update_evidence":
            tid = p.get("target_id", "?")
            changes = p.get("changes", {})
            lines.append(f"- Updated evidence {tid}: {changes}")
        elif op == "update_assumption":
            lines.append(f"- Updated assumption {p.get('target_id', '?')}")
        elif op == "add_uncertainty":
            item = p.get("item", {})
            lines.append(f"- Added uncertainty {item.get('id', '?')}: {item.get('question', '')[:80]}")
        elif op == "resolve_uncertainty":
            lines.append(f"- Resolved uncertainty {p.get('target_id', '?')}")
        elif op == "add_counterposition":
            item = p.get("item", {})
            lines.append(f"- Added counterposition {item.get('id', '?')} targeting {item.get('targets', [])}")
        elif op == "update_counterposition":
            lines.append(f"- Updated counterposition {p.get('target_id', '?')}")
        else:
            lines.append(f"- {op}: {p}")

    # Detect propagated strength changes not directly caused by patches
    before_thesis_str = before.get("thesis", {}).get("strength")
    after_thesis_str = after.get("thesis", {}).get("strength")
    if before_thesis_str is not None and after_thesis_str is not None:
        if abs(after_thesis_str - before_thesis_str) > 0.001:
            # Check if thesis was explicitly patched
            thesis_patched = any(p.get("op") == "update_thesis" for p in patches)
            if not thesis_patched:
                lines.append(f"- [propagated] Thesis strength: {before_thesis_str:.2f} → {after_thesis_str:.2f}")

    return "\n".join(lines) if lines else "(no changes)"


def filter_reversal_patches(phase2_patches: list, phase1_patches: list,
                            intermediate_belief: dict) -> list:
    """Remove Phase 2 patches that would reverse Phase 1 enforcement.

    Specifically: if Phase 1 weakened a node (claim/thesis), Phase 2 cannot
    strengthen it back above its intermediate (post-Phase-1) value.

    Returns a filtered copy of phase2_patches with reversals removed.
    """
    # Build a set of nodes that Phase 1 weakened (target_id → post-Phase-1 strength)
    phase1_weakened: dict[str, float] = {}

    for p in phase1_patches:
        op = p.get("op")
        if op == "update_claim":
            tid = p.get("target_id", "")
            new_strength = p.get("changes", {}).get("strength")
            if new_strength is not None:
                # Find post-Phase-1 strength from intermediate belief
                for c in intermediate_belief.get("claims", []):
                    if c["id"] == tid:
                        phase1_weakened[tid] = c.get("strength", new_strength)
                        break
        elif op == "update_thesis":
            new_str = p.get("new_strength")
            change = p.get("change")
            if new_str is not None or change == "weaken":
                phase1_weakened["THESIS"] = intermediate_belief.get("thesis", {}).get("strength", 0.5)
        elif op == "retire_claim":
            phase1_weakened[p.get("target_id", "")] = 0.0

    if not phase1_weakened:
        return phase2_patches

    filtered = []
    for p in phase2_patches:
        op = p.get("op")
        keep = True

        if op == "update_claim":
            tid = p.get("target_id", "")
            if tid in phase1_weakened:
                new_strength = p.get("changes", {}).get("strength")
                if new_strength is not None and new_strength > phase1_weakened[tid]:
                    keep = False  # Would reverse Phase 1 weakening
        elif op == "update_thesis":
            if "THESIS" in phase1_weakened:
                new_str = p.get("new_strength")
                change = p.get("change")
                if new_str is not None and new_str > phase1_weakened["THESIS"]:
                    keep = False
                elif change == "strengthen":
                    keep = False  # Cannot strengthen thesis that was weakened in Phase 1

        if keep:
            filtered.append(p)

    return filtered

def _key_for_dedupe(x: Any) -> str:
    """
    Make a stable, hashable key for heterogeneous items.
    - dict/list -> JSON string with sorted keys
    - other     -> str(x)
    """
    if isinstance(x, (dict, list)):
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    return str(x)

def _dedupe_extend(base_list: List[Any], new_items: Optional[List[Any]]) -> None:
    """
    Extend base_list with new_items, preserving order and removing duplicates
    using _key_for_dedupe.
    """
    seen = {_key_for_dedupe(it) for it in base_list}
    for it in new_items or []:
        k = _key_for_dedupe(it)
        if k not in seen:
            base_list.append(it)
            seen.add(k)

# --- Helper: naive chunker by characters (keeps it dependency-free) ---
def _chunk_text_by_chars(s: str, max_chars: int = 8000, overlap: int = 500) -> list[str]:
    """
    Breaks text into chunks of ~max_chars with small overlaps.
    Overlap preserves context between chunks for coherence.
    """
    s = s or ""
    if len(s) <= max_chars: 
        return [s]
    chunks = []
    i = 0
    while i < len(s):
        j = min(i + max_chars, len(s))
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(0, j - overlap)
    return chunks

def chunk_transcript_by_tokens(text: str, max_tokens: int = 10000, model: str = "gpt-4o") -> list[str]:
    """
    Splits a long string into chunks of approximately `max_tokens` tokens using tiktoken.

    Args:
        text (str): The full transcript string.
        max_tokens (int): Desired max tokens per chunk.
        model (str): The model name to select appropriate tokenizer.

    Returns:
        list[str]: A list of text chunks.
    """
    enc = tiktoken.encoding_for_model(model)
    lines = text.split("\n")
    
    chunks = []
    current_chunk = []
    current_token_count = 0

    for line in lines:
        line_tokens = len(enc.encode(line))
        if current_token_count + line_tokens > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_token_count = line_tokens
        else:
            current_chunk.append(line)
            current_token_count += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks