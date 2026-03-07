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
from chal.orchestrator.adjudicator import Adjudicator
from chal.orchestrator.moderator import Moderator
from chal.utilities.utils import parse_challenges, parse_structured_rebuttals_numbered, initialize_agent_stats, update_agent_stats, display_agent_stats, calculate_performance_scores, get_performance_summary
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.convergence import calculate_claim_agreement, format_convergence_summary, get_convergence_trajectory_summary
from chal.beliefs.io import parse_model_output_to_belief, belief_to_markdown
from chal.beliefs.io import project_for_embedding
from chal.beliefs.graph_visualizer import export_debate_graph
from chal.config import DebateConfig
from chal.utilities.training_data import DebateRecorder
from chal.utilities.reporting import generate_analysis_report, generate_analysis_json
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
    def __init__(self, agents: List[Agent], max_rounds: int = 3, config: Optional[DebateConfig] = None):
        """
        Initializes the DebateController with a list of agents and a number of debate rounds.

        Args:
            agents (List[Agent]): A list of LLM-powered agents participating in the debate.
            max_rounds (int): Number of complete debate rounds (each consisting of Stages 2-5).
            config (Optional[DebateConfig]): Configuration object containing all debate parameters.
                If None, a default configuration will be created.
        """
        self.agents = agents
        self.max_rounds = config.max_rounds if config and hasattr(config, 'max_rounds') else max_rounds
        self.config = config  # Store config for accessing stage and scribe parameters
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
        adjudicator_prompt = prompts.build_adjudicator_prompt(
            logic_weight=1.0,
            ethics_weight=0.0,
            logic_sys="Classical logic + Bayesian reasoning for inductive support; reject contradictions; prefer simpler hypotheses (Occam's Razor).",
            ethics_sys="None. Only prioritize logical rigor and soundness, not ethical implications" # "Rule-Utilitarianism."
        )
        adj_model = config.adjudication.model if config else "gpt-4o"
        adj_provider = config.adjudication.provider if config else "openai"
        adjudicator_agent = create_agent(
            name="Adjudicator",
            model=adj_model,
            provider=adj_provider,
            system_prompt=adjudicator_prompt
        )
        # Build the adjudicator agent
        self.adjudicator = Adjudicator(adjudicator_agent)

        # Instantiate the scribe agent
        scribe_model = config.scribe.model if config else "gpt-4o"
        scribe_provider = config.adjudication.provider if config else "openai"
        self.scribe_agent = create_agent(
            name="Scribe",
            model=scribe_model,
            provider=scribe_provider,
            system_prompt=""
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

        for agent in self.agents:
            self._log(f"\n--- Processing agent: {agent.name} ---", "INFO")

            opening_prompt = prompts.build_stage_1_belief_prompt_cbs(
                            topic=topic, agent_name=agent.name, persona_label=agent.persona_label
                        )

            # Log the prompt
            self._log_prompt(agent.name, opening_prompt, "Stage 1 - Opening Position")

            # Build full prompt history
            stage_request = [Message(role="user", content=opening_prompt)]

            # Generate the opening statement
            self._log(f"Calling model for {agent.name} opening position...", "INFO")
            response = agent.generate(stage_request)
            self._log(f"Received response from {agent.name} ({len(response.content)} chars)", "INFO")

            # Log raw response
            self._log_response(agent.name, response.content, "Stage 1 - Opening Position")

            # Parse response
            self._log(f"Parsing CBS belief object for {agent.name}...", "INFO")
            belief_obj, md_view, errs = parse_model_output_to_belief(response.content)

            if belief_obj and not errs:
                self._log_parse_result(True, f"Successfully parsed CBS belief for {agent.name}")
                self.debug_log.append(f"\n--- PARSED BELIEF JSON FOR {agent.name} ---\n{json.dumps(belief_obj, ensure_ascii=False, indent=2)}\n--- END PARSED JSON ---\n")

                # Validate belief graph structure with retry loop for blocking errors
                from chal.beliefs.belief_graph import BeliefGraph
                max_validation_retries = 3
                retry_count = 0
                validation_passed = False

                while retry_count < max_validation_retries and not validation_passed:
                    try:
                        graph = BeliefGraph(belief_obj)
                        graph_errors = graph.validate_links()

                        # Separate blocking errors from warnings
                        blocking_errors = [err for err in graph_errors if "BLOCKING ERROR" in err]
                        warnings = [err for err in graph_errors if "BLOCKING ERROR" not in err]

                        if blocking_errors:
                            self._log(f"BLOCKING validation errors for {agent.name} (attempt {retry_count + 1}/{max_validation_retries}):", "ERROR")
                            for err in blocking_errors:
                                self._log(f"  - {err}", "ERROR")

                            # Request revision from agent
                            retry_count += 1
                            if retry_count < max_validation_retries:
                                self._log(f"Requesting revision from {agent.name}...", "INFO")
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

                                revision_request = [Message(role="user", content=revision_prompt)]
                                self._log(f"Calling model for {agent.name} belief revision...", "INFO")
                                revision_response = agent.generate(revision_request)
                                self._log(f"Received revision from {agent.name} ({len(revision_response.content)} chars)", "INFO")

                                # Parse revised belief
                                belief_obj, md_view, errs = parse_model_output_to_belief(revision_response.content)
                                if not belief_obj or errs:
                                    self._log(f"Failed to parse revised belief from {agent.name}", "ERROR")
                                    break
                            else:
                                self._log(f"Max retries ({max_validation_retries}) reached for {agent.name}. Using last attempt despite errors.", "ERROR")
                        else:
                            validation_passed = True
                            if warnings:
                                self._log(f"Graph validation warnings for {agent.name}:", "WARN")
                                for warn in warnings:
                                    self._log(f"  - {warn}", "WARN")
                            else:
                                self._log(f"Graph validation passed for {agent.name}", "INFO")

                            # Log graph metrics
                            metrics = graph.get_graph_metrics()
                            self._log(f"Graph metrics: {metrics['total_nodes']} nodes, {metrics['total_edges']} edges, {metrics['critical_path_count']} critical paths", "INFO")

                    except Exception as e:
                        self._log(f"Graph validation error for {agent.name}: {e}", "ERROR")
                        retry_count += 1
                        if retry_count >= max_validation_retries:
                            break

                agent.set_internal_belief_obj(belief_obj)                   # store structured JSON (auto-rebuilds graph)
                # Generate Markdown from JSON (no longer requested from model)
                md_view = belief_to_markdown(belief_obj)
                self._log(f"Generated Markdown view from JSON ({len(md_view)} chars)", "DEBUG")
                agent.set_internal_belief(md_view)  # keep human-readable string too
                agent.all_beliefs_held.append(json.dumps(belief_obj, ensure_ascii=False, indent=2) if belief_obj else "") # track beliefs
            else:
                # Fallback to legacy behavior if parsing failed
                err_details = f"Errors: {errs}" if errs else "No belief object returned"
                self._log_parse_result(False, f"Failed to parse CBS belief for {agent.name}. {err_details}")
                self._log("Falling back to raw response content", "WARN")
                agent.set_internal_belief(response.content.strip())
                agent.all_beliefs_held.append(response.content.strip()) # track beliefs
                md_view = response.content.strip()

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

        for challenger in self.agents:
            challenger_name = challenger.name

            for target in self.agents:
                target_name = target.name
                if challenger_name == target_name:
                    self._log(f"Skipping self-examination: {challenger_name} == {target_name}", "DEBUG")
                    continue

                if only_if_disagree and self._positions_agree(challenger_name, target_name):
                    self._log(f"Agents agree - skipping: {challenger_name} ↔ {target_name}", "INFO")
                    markdown_note = f"\n*{challenger_name} and {target_name} broadly agree - skipping cross-examination*\n"
                    self._add_to_markdown(markdown_note)
                    self._notify("agent_complete", {"agent_name": challenger_name, "stage": 2, "action": f"Agrees with {target_name} — skipping"})
                    continue

                self._log(f"\n--- Cross-examination: {challenger_name} → {target_name} ---", "INFO")

                # Pull beliefs as JSON strings when available, else fall back to Markdown strings
                ch_belief_obj = challenger.get_internal_belief_obj()
                tg_belief_obj = target.get_internal_belief_obj()
                ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
                tg_belief_json = json.dumps(tg_belief_obj, ensure_ascii=False, indent=2) if tg_belief_obj else ""

                self._log(f"Challenger belief size: {len(ch_belief_json)} chars", "DEBUG")
                self._log(f"Target belief size: {len(tg_belief_json)} chars", "DEBUG")

                # Gather previous challenges for anti-repetition (if multi-round debate)
                prev_challenges_key = f"{challenger_name}→{target_name}"
                previous_challenges = self.previous_rounds_challenges.get(prev_challenges_key, [])

                # Get opponent's belief graph for vulnerability analysis
                opponent_belief_graph = target.get_belief_graph() if hasattr(target, 'get_belief_graph') else None

                # Use the Stage 2 prompt (topic-aware, ID-targeting)
                max_questions = self.config.stages.max_questions_per_cross_exam if self.config else 5
                max_question_length = self.config.stages.max_question_length_chars if self.config else 500
                stage3_mode = self.config.stage3_mode if self.config else "rebuttal"

                if stage3_mode == "bloodsport":
                    # Use adversarial cross-examination prompt for bloodsport mode
                    intensity = self.config.bloodsport.intensity if self.config else "moderate"
                    prompt = prompts.build_stage_2_bloodsport_prompt(
                        topic=self.topic if hasattr(self, "topic") else "<topic>",
                        agent_name=challenger_name,
                        opponent_name=target_name,
                        agent_belief_json=ch_belief_json,
                        opponent_belief_json=tg_belief_json,
                        intensity=intensity,
                        max_questions=max_questions,
                        max_question_length_chars=max_question_length,
                        previous_challenges=previous_challenges if previous_challenges else None,
                        opponent_belief_graph=opponent_belief_graph,
                        focus_subtopic=focus_subtopic,
                    )
                else:
                    prompt = prompts.build_stage_2_prompt(
                        topic=self.topic if hasattr(self, "topic") else "<topic>",
                        agent_name=challenger_name,
                        opponent_name=target_name,
                        agent_belief_json=ch_belief_json,
                        opponent_belief_json=tg_belief_json,
                        max_questions=max_questions,
                        max_question_length_chars=max_question_length,
                        previous_challenges=previous_challenges if previous_challenges else None,
                        opponent_belief_graph=opponent_belief_graph,
                        focus_subtopic=focus_subtopic,
                    )

                # Log prompt
                self._log_prompt(challenger_name, prompt, f"Stage 2 - Questioning {target_name}")

                stage_request = [Message(role="user", content=prompt)]

                # Get critique
                generation_temp = self.config.stages.generation_temperature if self.config else 0.2
                self._log(f"Calling model for {challenger_name} to question {target_name}...", "INFO")
                response = challenger.generate(stage_request, temperature=generation_temp)
                self._log(f"Received response ({len(response.content)} chars)", "INFO")

                # Log raw response
                self._log_response(challenger_name, response.content, f"Stage 2 - Questions for {target_name}")

                self.round_histories[self.current_round_key].append(response)

                # Parse the FIRST fenced JSON block -> {"questions":[...]}
                self._log("Parsing questions JSON block...", "INFO")
                questions_obj = _extract_first_json_block(response.content)
                questions = (questions_obj or {}).get("questions", [])

                # Fallback to legacy parser if needed (keeps backward compat)
                if not questions:
                    self._log_parse_result(False, "No structured questions found, trying legacy parser")
                    parsed_challenges = parse_challenges(response.content)
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

        # Iterate through targets
        for target_agent in self.agents:
            target_name = target_agent.name
            relevant_entries = grouped_entries.get(target_name) # Grab the list of critiques that are aimed at the target agent
            # If there are no critiques of this agent, move to the next agent
            if not relevant_entries:
                self._log(f"No questions for {target_name}, skipping", "INFO")
                continue

            self._log(f"\n--- Processing rebuttals for: {target_name} ---", "INFO")
            self._log(f"Agent faces {len(relevant_entries)} question(s)", "INFO")

            # Build the questions JSON payload expected by Stage 3 prompt
            questions_payload = {
                "questions": [
                    {
                        "qid": e.get("qid") or f"Q{idx+1}",
                        "text": e["challenge"],
                        "target_ids": e.get("target_ids", []),
                        "intent": e.get("intent"),
                        "why_high_value": e.get("why_high_value"),
                        "proposed_test": e.get("proposed_test"),
                        "from": e["challenger"]
                    }
                    for idx, e in enumerate(relevant_entries)
                ]
            }
            received_questions_json = json.dumps(questions_payload, ensure_ascii=False, indent=2)
            self.debug_log.append(f"\n--- QUESTIONS PAYLOAD FOR {target_name} ---\n{received_questions_json}\n--- END QUESTIONS ---\n")

            # Include the target's CURRENT belief JSON
            tgt_belief_obj = target_agent.get_internal_belief_obj()
            tgt_belief_json = json.dumps(tgt_belief_obj, ensure_ascii=False, indent=2) if tgt_belief_obj else ""
            self._log(f"Target belief size: {len(tgt_belief_json)} chars", "DEBUG")

            # Choose one representative opponent name (or "Various" in multi-agent debates)
            opponent_name = questions_payload["questions"][0]["from"] if questions_payload["questions"] else "Opponent"

            max_rebuttals = self.config.stages.max_rebuttals_per_response if self.config else 5
            max_rebuttal_length = self.config.stages.max_rebuttal_length_chars if self.config else 500
            prompt = prompts.build_stage_3_structured_rebuttal_prompt(
                topic=self.topic if hasattr(self, "topic") else "<topic>",
                agent_name=target_name,
                opponent_name=opponent_name,
                received_questions_json=received_questions_json,
                agent_belief_json=tgt_belief_json,
                max_rebuttals=min(max_rebuttals, len(questions_payload["questions"])),
                max_rebuttal_length_chars=max_rebuttal_length
            )

            # Log prompt
            self._log_prompt(target_name, prompt, f"Stage 3 - Rebuttals to {len(relevant_entries)} question(s)")

            stage_request = [Message(role="user", content=prompt)]

            generation_temp = self.config.stages.generation_temperature if self.config else 0.2
            self._log(f"Calling model for {target_name} rebuttals...", "INFO")
            response = target_agent.generate(stage_request, temperature=generation_temp)
            self._log(f"Received response ({len(response.content)} chars)", "INFO")

            # Log raw response
            self._log_response(target_name, response.content, f"Stage 3 - Rebuttals")

            self.round_histories[self.current_round_key].append(response)

            # Parse JSON blocks: 1) rebuttals, 2) optional patches
            self._log("Parsing rebuttal JSON blocks (rebuttals + optional patches)...", "INFO")
            blocks = _extract_all_json_blocks(response.content)
            self._log(f"Found {len(blocks)} JSON block(s)", "INFO")

            rebuttals_block = json.loads(blocks[0]) if blocks else {"rebuttals": []}
            patches_block = json.loads(blocks[1]) if len(blocks) > 1 else {"patches": []}

            rebuttals = rebuttals_block.get("rebuttals", [])
            patches = patches_block.get("patches", [])

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
            by_qid = {r.get("qid", f"Q{idx+1}"): r for idx, r in enumerate(rebuttals)}
            for idx, entry in enumerate(relevant_entries):
                qid = entry.get("qid", f"Q{idx+1}")
                r = by_qid.get(qid)
                entry["rebuttal"] = (r.get("answer", "") if r else "").strip() or entry.get("rebuttal")

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
            for r in rebuttals:
                qid = r.get("qid", "Q?")
                answer = r.get("answer", "")
                action = r.get("action", "unknown")
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
                    "status": result.get("outcome", "unknown").lower(),
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

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚖️ Stage 4: Adjudication\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 4, "name": "Adjudication"})

        # Build agent lookup for belief excerpt extraction
        agent_by_name = {agent.name: agent for agent in self.agents}

        adjudication_count = 0
        for entry in self.challenge_rebuttal_pairs:
            # Skip if missing key elements
            if not entry["challenge"] or not entry["rebuttal"]:
                self._log(f"Skipping incomplete pair: {entry['challenger']} → {entry['target']} (missing challenge or rebuttal)", "WARN")
                markdown_note = f"\n*Skipped incomplete pair: {entry['challenger']} → {entry['target']}*\n"
                self._add_to_markdown(markdown_note)
                self._notify("agent_complete", {"agent_name": "Adjudicator", "stage": 4, "action": f"Skipped incomplete pair: {entry['challenger']} → {entry['target']}"})
                continue

            adjudication_count += 1
            self._log(f"\n--- Adjudicating pair #{adjudication_count}: {entry['challenger']} → {entry['target']} ---", "INFO")
            self._log(f"Challenge: {entry['challenge'][:100]}...", "DEBUG")
            self._log(f"Rebuttal: {entry['rebuttal'][:100]}...", "DEBUG")

            # Extract targeted belief excerpts for the adjudicator
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

            # Run logic enforcement pipeline
            self._log("Calling adjudicator.run()...", "INFO")
            resolution = self.adjudicator.run(
                challenge=entry["challenge"],
                rebuttal=entry["rebuttal"],
                challenger=entry["challenger"],
                target=entry["target"],
                challenger_belief_excerpt_json=challenger_belief_excerpt_json,
                target_belief_excerpt_json=target_belief_excerpt_json
            )

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
            markdown_section += f"**Outcome**: {resolution['status'].upper()}\n\n"
            markdown_section += f"**Reasoning**: {resolution.get('reasoning', 'N/A')}\n\n"
            if resolution.get('restatement'):
                markdown_section += f"**Disagreement**: {resolution['restatement']}\n\n"

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
                "outcome": outcome
            })

        return self.challenge_rebuttal_pairs

    def run_stage_5_update_positions(self) -> None:
        """
        Stage 5: Belief Updating and Position Reframing

        For each agent, gathers all resolution outcomes where they were the target.
        Uses this to synthesize an updated version of their beliefs and claims.

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

        # Ask each agent to revise their view
        for agent in self.agents:
            name = agent.name
            relevant_entries = grouped_by_target.get(name, [])

            if not relevant_entries:
                self._log(f"No adjudication results for {name}, keeping current belief", "INFO")
                self.current_positions[name] = self.opening_positions.get(name, "[No change]")
                continue

            self._log(f"\n--- Updating belief for: {name} ---", "INFO")
            self._log(f"Agent received {len(relevant_entries)} adjudication result(s)", "INFO")

            # Build prompt
            prior_json = agent.get_internal_belief_obj()
            stage3_mode = self.config.stage3_mode if self.config else "rebuttal"

            if prior_json is not None and stage3_mode == "bloodsport":
                self._log(f"Using bloodsport belief update prompt for {name}", "DEBUG")
                # Gather bloodsport exchange context for this agent
                bloodsport_exchanges = []
                for e in relevant_entries:
                    if "bloodsport_transcript" in e:
                        bloodsport_exchanges.extend(e["bloodsport_transcript"])

                stage_3_patches = self.last_rebuttals_patches.get(name, [])
                stage_3_patches_json = json.dumps(stage_3_patches, ensure_ascii=False, indent=2) if stage_3_patches else ""

                prompt = prompts.build_stage_5_bloodsport_prompt(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,
                    prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2),
                    bloodsport_exchanges=bloodsport_exchanges if bloodsport_exchanges else None,
                    stage_3_patches_json=stage_3_patches_json
                )
            elif prior_json is not None:
                self._log(f"Using CBS patch/update prompt for {name}", "DEBUG")
                # Use CBS patch/update builder
                stage_3_patches = self.last_rebuttals_patches.get(name, [])
                stage_3_patches_json = json.dumps(stage_3_patches, ensure_ascii=False, indent=2) if stage_3_patches else ""

                prompt = prompts.build_stage_5_belief_update_prompt_cbs(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,
                    prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2),
                    stage_3_patches_json=stage_3_patches_json
                )
            else:
                self._log(f"Using legacy belief update prompt for {name}", "DEBUG")
                # Fallback to cbs builder with empty prior JSON
                prior_json = agent.get_internal_belief_obj()
                prompt = prompts.build_stage_5_belief_update_prompt_cbs(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,
                    prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2) if prior_json else "{}"
                )

            # Log prompt
            self._log_prompt(name, prompt, "Stage 5 - Belief Update")

            stage_request = [Message(role="user", content=prompt)]

            generation_temp = self.config.stages.generation_temperature if self.config else 0.2
            self._log(f"Calling model for {name} belief update...", "INFO")
            response = agent.generate(stage_request, temperature=generation_temp)
            self._log(f"Received response ({len(response.content)} chars)", "INFO")

            # Log raw response
            self._log_response(name, response.content, "Stage 5 - Belief Update")

            self.round_histories[self.current_round_key].append(response)

            # Parse PATCHES from the response (not full belief)
            self._log(f"Parsing patches for {name}...", "INFO")
            patches = []  # Initialize for recorder tracking
            blocks = _extract_all_json_blocks(response.content)

            if not blocks:
                self._log_parse_result(False, f"No JSON blocks found in {name} response")
                # Keep prior belief unchanged
                self._log("Keeping prior belief unchanged", "WARN")
                md_view = agent.get_internal_belief()
            else:
                patches_block = json.loads(blocks[0])
                patches = patches_block.get("patches", [])

                if patches:
                    self._log_parse_result(True, f"Successfully parsed {len(patches)} patch(es) for {name}")
                    self.debug_log.append(f"\n--- PATCHES FOR {name} ---\n{json.dumps(patches, ensure_ascii=False, indent=2)}\n--- END PATCHES ---\n")

                    # Apply patches to prior belief
                    prior_belief = agent.get_internal_belief_obj()

                    if prior_belief:
                        from chal.beliefs.patches import apply_patches, validate_patches

                        # Validate patches first
                        patch_errors = validate_patches(patches, prior_belief)
                        if patch_errors:
                            self._log(f"Patch validation warnings for {name}:", "WARN")
                            for err in patch_errors:
                                self._log(f"  - {err}", "WARN")

                        try:
                            # Apply with confidence propagation
                            updated_belief = apply_patches(prior_belief, patches, propagate_confidence=True)

                            self.debug_log.append(f"\n--- UPDATED BELIEF FOR {name} ---\n{json.dumps(updated_belief, ensure_ascii=False, indent=2)}\n--- END UPDATED BELIEF ---\n")

                            # Validate updated belief structure
                            from chal.beliefs.belief_graph import BeliefGraph
                            try:
                                validation_graph = BeliefGraph(updated_belief)
                                validation_errors = validation_graph.validate_links()

                                # Separate blocking errors from warnings
                                blocking_errors = [err for err in validation_errors if "BLOCKING ERROR" in err]
                                warnings = [err for err in validation_errors if "BLOCKING ERROR" not in err]

                                if blocking_errors:
                                    self._log(f"BLOCKING validation errors in updated belief for {name}:", "ERROR")
                                    for err in blocking_errors:
                                        self._log(f"  - {err}", "ERROR")
                                    self._log(f"Reverting to prior belief for {name} due to validation errors", "ERROR")
                                    # Keep prior belief unchanged
                                    md_view = agent.get_internal_belief()
                                    raise Exception(f"Updated belief contains blocking validation errors: {blocking_errors}")

                                if warnings:
                                    self._log(f"Graph validation warnings for updated belief of {name}:", "WARN")
                                    for warn in warnings:
                                        self._log(f"  - {warn}", "WARN")
                                else:
                                    self._log(f"Graph validation passed for updated belief of {name}", "INFO")

                            except Exception as e:
                                if "blocking validation errors" in str(e).lower():
                                    raise  # Re-raise validation errors
                                self._log(f"Graph validation error for updated belief of {name}: {e}", "ERROR")

                            # Set updated belief
                            agent.set_internal_belief_obj(updated_belief)

                            # Generate markdown
                            md_view = belief_to_markdown(updated_belief)
                            agent.set_internal_belief(md_view)
                            agent.all_beliefs_held.append(json.dumps(updated_belief, ensure_ascii=False, indent=2))

                            self._log(f"Applied {len(patches)} patches and propagated confidence for {name}", "INFO")
                        except Exception as e:
                            self._log(f"Error applying patches for {name}: {e}", "ERROR")
                            self._log("Keeping prior belief unchanged", "WARN")
                            md_view = agent.get_internal_belief()
                    else:
                        self._log(f"No prior belief object for {name}, cannot apply patches", "WARN")
                        md_view = agent.get_internal_belief()
                else:
                    # VALIDATE: Check if agent should have generated patches
                    critique_valid_count = sum(
                        1 for entry in relevant_entries
                        if entry.get("resolution", {}).get("status") == "critique_valid"
                    )

                    if critique_valid_count > 0:
                        self._log_parse_result(False,
                            f"WARNING: {name} received {critique_valid_count} CRITIQUE_VALID outcome(s) but returned no patches. "
                            f"This violates the mandatory patch requirement."
                        )
                        self._log(f"ENFORCEMENT FAILURE: Agent {name} ignored mandatory patch requirement after {critique_valid_count} CRITIQUE_VALID outcomes", "ERROR")
                    else:
                        self._log_parse_result(False, f"No patches returned for {name}, keeping prior belief")

                    md_view = agent.get_internal_belief()

            # Record training data
            if self.recorder:
                belief_after = agent.get_internal_belief_obj()
                adjudication_results = [
                    {
                        "role": "target",
                        "opponent": e.get("challenger", "?"),
                        "verdict": e.get("resolution", {}).get("status", "unknown") if isinstance(e.get("resolution"), dict) else "unknown",
                        "reasoning": e.get("resolution", {}).get("reasoning", "") if isinstance(e.get("resolution"), dict) else "",
                    }
                    for e in relevant_entries
                ]
                self.recorder.record_belief_update(
                    agent_id=name,
                    belief_before=prior_json,
                    belief_after=belief_after,
                    adjudication_results=adjudication_results,
                    patches=patches,
                    raw_response=response.content,
                )

            # Add to markdown transcript
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

        for agent in self.agents:
            name = agent.name
            self._log(f"\n--- Processing concluding remarks for: {name} ---", "INFO")

            # Beliefs as JSON strings if available
            a_belief_obj = agent.get_internal_belief_obj()
            a_belief_json = json.dumps(a_belief_obj, ensure_ascii=False, indent=2) if a_belief_obj else ""
            self._log(f"Final belief size: {len(a_belief_json)} chars", "DEBUG")
            self._log(f"Belief history: {len(agent.all_beliefs_held)} snapshots", "DEBUG")

            # Build changelog summary from all belief versions
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
                topic=self.topic if hasattr(self, "topic") else "<topic>",
                agent_name=name,
                agent_belief_json=a_belief_json,
                belief_changelog_summary=belief_changelog_summary,
                num_rounds=self.current_round if hasattr(self, "current_round") else 1,
                persona_label=agent.persona_label if hasattr(agent, "persona_label") else ""
            )

            # Log prompt
            self._log_prompt(name, prompt, "Stage 6 - Concluding Remarks")

            stage_request = [Message(role="user", content=prompt)]

            generation_temp = self.config.stages.generation_temperature if self.config else 0.2
            self._log(f"Calling model for {name} concluding remarks...", "INFO")
            response = agent.generate(stage_request, temperature=generation_temp)
            self._log(f"Received response ({len(response.content)} chars)", "INFO")

            # Log raw response
            self._log_response(name, response.content, "Stage 6 - Concluding Remarks")

            self.round_histories[self.current_round_key].append(response)

            # Try to parse {"conclusion": {...}} for structured logging
            self._log("Parsing conclusion JSON block...", "INFO")
            concl_obj = _extract_first_json_block(response.content)
            if concl_obj and "conclusion" in concl_obj:
                self._log_parse_result(True, f"Successfully parsed structured conclusion for {name}")
                self.debug_log.append(f"\n--- PARSED CONCLUSION JSON ({name}) ---\n{json.dumps(concl_obj['conclusion'], ensure_ascii=False, indent=2)}\n--- END CONCLUSION ---\n")
                self.conclusions[name] = concl_obj["conclusion"]
                markdown_content = response.content
            else:
                self._log_parse_result(False, f"No structured conclusion found for {name}, using raw text")
                # Fallback: keep raw text
                self.conclusions[name] = response.content.strip()
                markdown_content = response.content.strip()

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

def _extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    return json.loads(m.group(1)) if m else None

def _extract_all_json_blocks(text: str) -> List[str]:
    return re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)

def _extract_first_markdown_block(text: str) -> Optional[str]:
    m = re.search(r"```markdown\s*(.*?)\s*```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

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