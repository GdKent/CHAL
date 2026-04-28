"""
debate_controller.py

Orchestrates structured multi-agent philosophical debates using the CHAL Belief Schema.

The DebateController manages a 6-stage debate process:
- Stage 0: Briefing - Initialize agents with personas and universal rules
- Stage 1: Opening Positions - Agents state initial beliefs as structured JSON (CBS)
- Stage 2: Cross-Examination - Agents ask targeted questions about opponents' claims/assumptions
- Stage 3: Rebuttals - Agents respond to questions with structured answers and optional belief patches
- Stage 4: Adjudication - Independent evaluator assesses challenge-rebuttal pairs
- Stage 5: Belief Updates - Agents revise beliefs based on adjudication outcomes

Features:
- Structured belief tracking with JSON schemas (CBS)
- Embedding-based belief trajectory visualization
- Configurable adjudication weights (logic vs. ethics)
- Token-optimized prompts (JSON-only responses, Markdown generated programmatically)
- Round-robin debate structure with multiple rounds

Note: All belief outputs are JSON-first. Human-readable Markdown is generated
programmatically using belief_to_markdown() to minimize token usage.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken

from chal.agents import prompts
from chal.agents.base import Agent, Message
from chal.agents.ethics_systems import get_ethics_system, get_ethics_system_description
from chal.agents.factory import create_agent
from chal.agents.logic_systems import get_logic_system, get_logic_system_description
from chal.beliefs.graph_visualizer import export_debate_graph
from chal.beliefs.io import (
    belief_to_markdown,
    parse_model_output_to_belief,
)
from chal.beliefs.patches import initialize_defense_tracking
from chal.config import AdjudicationConfig, DebateConfig, DefenseBoostConfig
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.orchestrator.adjudicator import Adjudicator
from chal.utilities.parallel import ParallelDispatcher, WorkItem
from chal.utilities.reporting import generate_analysis_json, generate_analysis_report
from chal.utilities.training_data import DebateRecorder
from chal.utilities.debug_log_writer import DebugLogHandler, DebugLogWriter
from chal.utilities.utils import (
    calculate_performance_scores,
    finalize_agent_stats,
    get_performance_summary,
    initialize_agent_stats,
    parse_challenges,
    snapshot_belief,
    update_agent_stats,
    validate_stage2_questions,
)


@dataclass
class DebateMetrics:
    """Accumulates operational metrics throughout a debate run."""

    total_retries: int = 0
    total_rate_limit_hits: int = 0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return round(self.end_time - self.start_time, 2) if self.end_time else 0.0

    def to_dict(self) -> dict:
        return {
            "total_retries": self.total_retries,
            "total_rate_limit_hits": self.total_rate_limit_hits,
            "total_output_tokens": self.total_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "duration_s": self.duration_s,
        }


# === Path Configuration ===
# Get the project root directory (CHAL/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "src" / "chal" / "storage"

class DebateController:
    """
    Orchestrates structured philosophical debates between multiple LLM agents.

    The controller manages the complete debate lifecycle through 6 stages, maintains
    agent beliefs using the CHAL Belief Schema (CBS), tracks embeddings for
    visualization, and coordinates adjudication of challenge-rebuttal exchanges.

    Attributes:
        agents: List of participating Agent instances
        max_rounds: Number of debate rounds (each round includes Stages 2-5)
        current_round_pairs: Current round's challenge-rebuttal exchanges
        opening_positions: Initial belief statements from Stage 1
        round_histories: Message history per round for context management
    """
    def __init__(self, agents: list[Agent], max_rounds: int = 3, config: DebateConfig | None = None,
                 key_pool=None, log_file_path: Path | None = None):
        """
        Initializes the DebateController with a list of agents and a number of debate rounds.

        Args:
            agents (list[Agent]): A list of LLM-powered agents participating in the debate.
            max_rounds (int): Number of complete debate rounds (each consisting of Stages 2-5).
            config (DebateConfig | None): Configuration object containing all debate parameters.
                If None, a default configuration will be created.
            key_pool: Optional KeyPool instance for multi-key API key rotation.
                When provided, the adjudicator also gets the pool.
        """
        self.agents = agents
        self.max_rounds = config.max_rounds if config and hasattr(config, 'max_rounds') else max_rounds
        self.config = config  # Store config for accessing stage parameters
        self.topic = config.topic if config else ""
        self.key_pool = key_pool

        # Parallel dispatcher — uses config when available, else sequential
        par_enabled = config.parallel.enabled if config else False
        par_workers = config.parallel.max_workers if config else 5
        self.dispatcher = ParallelDispatcher(max_workers=par_workers, enabled=par_enabled)

        self.current_round_pairs = []
        self.all_rounds_pairs = []
        self.opening_positions = []

        # Real-time debug log (streams to file when log_file_path is set)
        self.debug_log = DebugLogWriter(file_path=log_file_path)
        self.markdown_transcript: list[str] = []  # Clean markdown-only output for human reading

        # Bridge Python logger → debug log so retry/API messages are captured
        import logging as _logging

        self._debug_log_handler = DebugLogHandler(self.debug_log)
        self._debug_log_handler.setFormatter(
            _logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                               datefmt="%H:%M:%S")
        )
        _logging.getLogger("chal").addHandler(self._debug_log_handler)

        self.round_histories: dict[str, list[Message]] = {} # A dictionary of full debate rounds for tracking and memory efficient prompting
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
            threshold=adj_cfg.threshold,
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
            threshold=adj_cfg.threshold,
        )

        # Initialize agent statistics
        self.agent_stats = initialize_agent_stats([agent.name for agent in self.agents])

        # Operational metrics accumulator
        self.metrics = DebateMetrics()

        # Wire rate-limit callback into every agent (including adjudicator)
        def _on_rate_limit():
            self.metrics.total_rate_limit_hits += 1

        for agent in self.agents:
            agent._on_rate_limit = _on_rate_limit
        adjudicator_agent._on_rate_limit = _on_rate_limit

        # Training data recorder (initialized in run() if save_training_data is enabled)
        self.recorder: DebateRecorder | None = None

        # Progress callback (set by run() caller, e.g. DebateDisplay.handle_event)
        self._progress_callback: Callable[[str, dict[str, Any]], None] | None = None
        # Error recovery callback (set by run() caller)
        self._on_error: Callable[[str, Exception, int], str] | None = None

    def _accumulate_tokens(self, response) -> None:
        """Extract token counts from a Message's metadata and add to metrics."""
        if response is None or not hasattr(response, "metadata") or response.metadata is None:
            return
        usage = response.metadata.get("usage")
        if not usage:
            return
        # Anthropic: input_tokens/output_tokens  OpenAI/XAI/Perplexity: prompt_tokens/completion_tokens
        self.metrics.total_input_tokens += (
            usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        )
        self.metrics.total_output_tokens += (
            usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        )

    def _accumulate_tokens_from_usage(self, usage: dict) -> None:
        """Accumulate tokens from a raw usage dict."""
        if not usage:
            return
        self.metrics.total_input_tokens += (
            usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        )
        self.metrics.total_output_tokens += (
            usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        )

    def _call_agent(self, agent, messages, agent_name: str = "", **kwargs) -> str | None:
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
                response = agent.generate(messages, **kwargs)
                self._accumulate_tokens(response)
                return response
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

    def _notify(self, event: str, data: dict[str, Any] | None = None) -> None:
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
        self.debug_log.write(log_entry)

    def _log_header(self, title: str, char: str = "=") -> None:
        """Add a formatted header to the debug log."""
        separator = char * 80
        self.debug_log.write(f"\n{separator}")
        self.debug_log.write(f"{title}")
        self.debug_log.write(f"{separator}\n")

    def _log_prompt(self, agent_name: str, prompt: str, stage: str = "") -> None:
        """Log a prompt being sent to a model."""
        stage_info = f" - {stage}" if stage else ""
        self._log_header(f"PROMPT TO {agent_name}{stage_info}", "-")
        self.debug_log.write(prompt)
        self.debug_log.write("\n" + "-" * 80 + "\n")

    def _log_response(self, agent_name: str, response: str, stage: str = "") -> None:
        """Log a raw model response."""
        stage_info = f" - {stage}" if stage else ""
        self._log_header(f"RAW RESPONSE FROM {agent_name}{stage_info}", "-")
        self.debug_log.write(response)
        self.debug_log.write("\n" + "-" * 80 + "\n")

    def _log_parse_result(self, success: bool, details: str) -> None:
        """Log the result of parsing a model response."""
        status = "SUCCESS" if success else "FAILURE"
        self._log(f"PARSE {status}: {details}", level="PARSE")

    def _add_to_markdown(self, content: str) -> None:
        """Add content to the clean markdown transcript."""
        self.markdown_transcript.append(content)

    def _close_debug_log(self) -> None:
        """Detach the logging handler and close the debug log file."""
        import logging as _logging

        _logging.getLogger("chal").removeHandler(self._debug_log_handler)
        self.debug_log.close()

    def _log_definition_statistics(self) -> None:
        """Log end-of-debate D# (definition) statistics per agent."""
        lines = ["", "=" * 70, "DEFINITION STATISTICS", "=" * 70]

        for agent in self.agents:
            belief = agent.get_internal_belief_obj()
            if not isinstance(belief, dict):
                lines.append(f"  {agent.name}: no definitions")
                continue
            defs = belief.get("definitions", [])
            if not defs:
                lines.append(f"  {agent.name}: no definitions")
                continue

            total = len(defs)
            active = [d for d in defs if d.get("status") != "retracted"]
            retracted = [d for d in defs if d.get("status") == "retracted"]
            strengths = [d.get("strength", 0.0) for d in active]
            avg_str = sum(strengths) / len(strengths) if strengths else 0.0

            lines.append(f"  {agent.name}: {total} definitions "
                         f"({len(active)} active, {len(retracted)} retracted)")
            lines.append(f"    Avg D# strength (active): {avg_str:.2f}")

            # Most challenged: D# IDs that appear in X# targets across ALL agents
            challenged_counts: dict[str, int] = {}
            for other_agent in self.agents:
                other_belief = other_agent.get_internal_belief_obj()
                if not isinstance(other_belief, dict):
                    continue
                for x in other_belief.get("counterpositions", []):
                    for tid in x.get("targets", []):
                        if tid.startswith("D"):
                            challenged_counts[tid] = challenged_counts.get(tid, 0) + 1

            # Filter to this agent's definitions
            agent_def_ids = {d.get("id") for d in defs}
            agent_challenged = {
                k: v for k, v in challenged_counts.items() if k in agent_def_ids
            }
            if agent_challenged:
                top = sorted(agent_challenged.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join(f"{did} ({cnt}x)" for did, cnt in top)
                lines.append(f"    Most challenged: {top_str}")

        lines.append("=" * 70)
        self._log("\n".join(lines), "INFO")

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
            self.metrics.total_retries += 1
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
        self.debug_log.write(f"\n--- UNIVERSAL PROMPT ---\n{universal}\n--- END UNIVERSAL PROMPT ---\n")

        for agent in self.agents:
            self._log(f"Configuring agent: {agent.name}", "INFO")

            # Apply universal reasoning rules
            agent.receive_system_prompt(universal)
            self._log(f"  ✓ Applied universal prompt to {agent.name}", "DEBUG")

            # Apply agent-specific persona card
            persona = personas.get(agent.name, "")
            self._log(f"  Persona for {agent.name}: {persona[:50]}..." if len(persona) > 50 else f"  Persona for {agent.name}: {persona}", "DEBUG")
            role_card = prompts.build_position_prompt(agent.name, persona)
            self.debug_log.write(f"\n--- ROLE CARD FOR {agent.name} ---\n{role_card}\n--- END ROLE CARD ---\n")
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

        # --- Partition agents: custom (pre-loaded belief) vs generation (need LLM) ---
        custom_agents = [a for a in self.agents if a.get_internal_belief_obj() is not None]
        generation_agents = [a for a in self.agents if a.get_internal_belief_obj() is None]

        # --- Handle custom agents (no LLM call) ---
        for agent in custom_agents:
            self._log(f"\n--- Processing agent: {agent.name} (custom belief pre-loaded) ---", "INFO")
            self._log(f"[Stage 1] Skipping generation for {agent.name} — custom belief pre-loaded", "INFO")

            belief_obj = agent.get_internal_belief_obj()
            md_view = agent.get_internal_belief()

            agent.all_beliefs_held.append(json.dumps(belief_obj, ensure_ascii=False, indent=2))
            self.debug_log.write(f"\n--- PRE-LOADED BELIEF JSON FOR {agent.name} ---\n{json.dumps(belief_obj, ensure_ascii=False, indent=2)}\n--- END PRE-LOADED JSON ---\n")

            synthetic_response = Message(role="assistant", content=md_view)
            self.round_histories[self.current_round_key].append(synthetic_response)

            if self.recorder:
                self.recorder.record_belief_formation(
                    agent_id=agent.name,
                    inputs={
                        "topic": topic,
                        "persona": getattr(agent, 'persona_label', ''),
                    },
                    belief=belief_obj,
                    raw_response="[custom belief pre-loaded from file]",
                )

            markdown_content = f"\n## {agent.name} - Opening Statement (Custom Belief)\n\n{md_view}\n"
            self._add_to_markdown(markdown_content)
            self._notify("agent_complete", {"agent_name": agent.name, "stage": 1, "action": "Custom belief loaded"})

        # --- GATHER: Fire opening-position calls for generation agents in parallel ---
        if generation_agents:
            s1_max_retries = self.config.stages.parse_retries if self.config else 3
            items = [
                WorkItem(
                    key=agent.name,
                    callable=lambda a=agent: _generate_opening_position(
                        a, topic, max_retries=s1_max_retries, log_fn=self._log,
                    ),
                    context={"agent": agent},
                )
                for agent in generation_agents
            ]
            results = self.dispatcher.run(items)
        else:
            results = {}

        # --- APPLY: Process generation results sequentially in deterministic agent order ---
        for agent in generation_agents:
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
            self._accumulate_tokens_from_usage(r.get("usage", {}))
            self.metrics.total_retries += len(r.get("retry_records", []))
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
                self.debug_log.write(f"\n--- PARSED BELIEF JSON FOR {agent.name} ---\n{json.dumps(belief_obj, ensure_ascii=False, indent=2)}\n--- END PARSED JSON ---\n")

                graph_metrics = r.get("graph_metrics")
                if graph_metrics:
                    self._log(f"Graph metrics: {graph_metrics['total_nodes']} nodes, {graph_metrics['total_edges']} edges, {graph_metrics['critical_path_count']} critical paths", "INFO")

                initialize_defense_tracking(belief_obj)                      # set original_strength + consecutive_defenses
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

        # Capture each agent's initial_snapshot for expanded agent_stats tracking.
        # Defensive: try/except per agent so a single degraded belief cannot derail
        # the whole snapshot pass.
        for agent in self.agents:
            try:
                belief_obj = agent.get_internal_belief_obj()
            except Exception as e:
                self._log(f"initial_snapshot: failed to read belief for {agent.name}: {e}", "WARNING")
                belief_obj = None
            self.agent_stats[agent.name]["initial_snapshot"] = snapshot_belief(
                belief_obj if isinstance(belief_obj, dict) else {}
            )

        return

    def run_stage_2_cross_examination(self) -> list:
        """
        Stage 2: Cross-Examination

        Each agent critiques every other agent's current position. Multiple critiques per
        agent-pair are supported, and each is recorded as a separate entry.

        Structure of each entry in `self.current_round_pairs`:
            {
                "challenger": "Agent-A",
                "target": "Agent-B",
                "challenge": "<individual critique text>",
                "rebuttal": None,
                "resolution": None
            }

        Returns:
            list[dict]: A flat list of critique records, one per challenge.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 2: CROSS-EXAMINATION")
        self._log("Starting cross-examination phase", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚔️ Stage 2: Cross-Examination\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 2, "name": "Cross-Examination"})

        self.current_round_pairs = []

        # Build all challenger→target pairs (excluding self)
        topic = self.topic if hasattr(self, "topic") else "<topic>"
        pairs = []
        for challenger in self.agents:
            for target in self.agents:
                if challenger.name == target.name:
                    self._log(f"Skipping self-examination: {challenger.name} == {target.name}", "DEBUG")
                    continue
                pairs.append((challenger, target))

        # --- GATHER: Fire all cross-examination calls in parallel ---
        s2_max_retries = self.config.stages.parse_retries if self.config else 3
        items = [
            WorkItem(
                key=f"{c.name}→{t.name}",
                callable=lambda c=c, t=t: _generate_cross_examination(
                    challenger=c, target=t, topic=topic, config=self.config,
                    previous_challenges=self.previous_rounds_challenges.get(f"{c.name}→{t.name}", []),
                    max_retries=s2_max_retries,
                    log_fn=self._log,
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

            if work_result.error is not None:
                self._log(f"[Stage 2] Error for {pair_key}: {work_result.error}", "ERROR")
                self._notify("agent_complete", {"agent_name": challenger_name, "stage": 2, "action": f"Error questioning {target_name}"})
                continue

            r = work_result.result
            self._accumulate_tokens_from_usage(r.get("usage", {}))
            self.metrics.total_retries += len(r.get("retry_records", []))

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
                    self.current_round_pairs.append({
                        "challenger": challenger_name,
                        "target": target_name,
                        "challenge": challenge,
                        "rebuttal": None,
                        "resolution": None
                    })
            else:
                self._log_parse_result(True, f"Successfully parsed {len(questions)} structured questions")
                self.debug_log.write(f"\n--- PARSED QUESTIONS JSON ({challenger_name} → {target_name}) ---\n{json.dumps(questions, ensure_ascii=False, indent=2)}\n--- END QUESTIONS ---\n")

                # Store structured questions
                for q in questions:
                    self.current_round_pairs.append({
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

        return self.current_round_pairs

    def run_stage_3_rebuttals(self) -> list:
        """
        Stage 3: Rebuttals

        Each agent is presented with all critiques directed at them,
        and asked to respond to each critique individually.

        Updates:
            Each entry in self.current_round_pairs has its "rebuttal" field filled.

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
        for entry in self.current_round_pairs:
            target = entry["target"]
            grouped_entries.setdefault(target, []).append(entry)

        self._log(f"Grouped {len(self.current_round_pairs)} challenges into {len(grouped_entries)} targets", "INFO")

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
        s3_max_retries = self.config.stages.parse_retries if self.config else 3
        items = [
            WorkItem(
                key=target_agent.name,
                callable=lambda ta=target_agent, re=relevant_entries: _generate_rebuttal(
                    target_agent=ta, relevant_entries=re, topic=topic, config=self.config,
                    max_retries=s3_max_retries, log_fn=self._log,
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

            if work_result.error is not None:
                self._log(f"[Stage 3] Error for {target_name}: {work_result.error}", "ERROR")
                self._notify("agent_complete", {"agent_name": target_name, "stage": 3, "action": "Rebuttal generation failed"})
                continue

            r = work_result.result
            self._accumulate_tokens_from_usage(r.get("usage", {}))
            self.metrics.total_retries += len(r.get("retry_records", []))
            response = r["response"]
            prompt = r["prompt"]
            rebuttals = r["rebuttals"]
            patches = r["patches"]
            received_questions_json = r["received_questions_json"]
            tgt_belief_obj = r["tgt_belief_obj"]

            # Replay logs
            self.debug_log.write(f"\n--- QUESTIONS PAYLOAD FOR {target_name} ---\n{received_questions_json}\n--- END QUESTIONS ---\n")
            self._log_prompt(target_name, prompt, f"Stage 3 - Rebuttals to {len(relevant_entries)} question(s)")
            self._log(f"Received response ({len(response.content)} chars)", "INFO")
            self._log_response(target_name, response.content, "Stage 3 - Rebuttals")

            self.round_histories[self.current_round_key].append(response)

            if rebuttals:
                self._log_parse_result(True, f"Parsed {len(rebuttals)} rebuttal(s) from {target_name}")
                self.debug_log.write(f"\n--- PARSED REBUTTALS ({target_name}) ---\n{json.dumps(rebuttals, ensure_ascii=False, indent=2)}\n--- END REBUTTALS ---\n")
            else:
                self._log_parse_result(False, f"No rebuttals parsed from {target_name}")

            if patches:
                self._log(f"Parsed {len(patches)} patch operation(s) from {target_name}", "INFO")
                self.debug_log.write(f"\n--- PARSED PATCHES ({target_name}) ---\n{json.dumps(patches, ensure_ascii=False, indent=2)}\n--- END PATCHES ---\n")

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

        return self.current_round_pairs

    def run_stage_4_conflict_resolution(self) -> list:
        """
        Stage 4: Rigorous Conflict Resolution

        Iterates through all challenge-rebuttal pairs and uses the Adjudicator
        to run a structured resolution protocol consisting of:
            1. Restatement of disagreement
            2. Formalization of both arguments
            3. Third-party adjudication
            4. Final resolution logging

        Returns:
            list[dict]: Updated current_round_pairs list with resolution entries.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 4: ADJUDICATION")
        self._log(f"Starting adjudication of {len(self.current_round_pairs)} challenge-rebuttal pair(s)", "INFO")
        self._log_prompt("Adjudicator", self.adjudicator.agent.system_prompt, "Stage 4 - System Prompt")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚖️ Stage 4: Adjudication\n"
        self._add_to_markdown(markdown_header)
        self._notify("stage_start", {"stage": 4, "name": "Adjudication"})

        # Build agent lookup for belief excerpt extraction
        agent_by_name = {agent.name: agent for agent in self.agents}

        # Separate complete pairs from incomplete ones
        complete_pairs = []
        for i, entry in enumerate(self.current_round_pairs):
            if not entry["challenge"] or not entry["rebuttal"]:
                self._log(f"Skipping incomplete pair: {entry['challenger']} → {entry['target']} (missing challenge or rebuttal)", "WARN")
                markdown_note = f"\n*Skipped incomplete pair: {entry['challenger']} → {entry['target']}*\n"
                self._add_to_markdown(markdown_note)
                self._notify("agent_complete", {"agent_name": "Adjudicator", "stage": 4, "action": f"Skipped incomplete pair: {entry['challenger']} → {entry['target']}"})
            else:
                complete_pairs.append((i, entry))

        # --- GATHER: Fire all adjudication calls in parallel ---
        adj_max_retries = self.config.stages.parse_retries if self.config else 3
        items = [
            WorkItem(
                key=f"{entry['challenger']}→{entry['target']}:{entry.get('qid', f'Q{i}')}",
                callable=lambda e=entry: _run_adjudication(
                    self.adjudicator, e, agent_by_name,
                    max_retries=adj_max_retries, log_fn=self._log,
                ),
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
            entry = self.current_round_pairs[entry_idx]

            adjudication_count += 1
            self._log(f"\n--- Adjudicating pair #{adjudication_count}: {entry['challenger']} → {entry['target']} ---", "INFO")
            self._log(f"Challenge: {entry['challenge'][:100]}...", "DEBUG")
            self._log(f"Rebuttal: {entry['rebuttal'][:100]}...", "DEBUG")

            r = self._retry_on_parse_failure(
                generate_fn=lambda e=entry: _run_adjudication(
                    self.adjudicator, e, agent_by_name,
                    max_retries=adj_max_retries, log_fn=self._log,
                ),
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

            # Accumulate operational metrics from adjudicator
            self._accumulate_tokens_from_usage(resolution.pop("_usage", {}))
            self.metrics.total_retries += resolution.pop("_retry_count", 0)

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
            self.debug_log.write(f"\n--- ADJUDICATION RESULT ({entry['challenger']} → {entry['target']}) ---\n{json.dumps(resolution, ensure_ascii=False, indent=2)}\n--- END ADJUDICATION ---\n")

            # Save structured result in the entry
            entry["resolution"] = resolution

            # Update agent stats
            self.agent_stats = update_agent_stats(self.agent_stats, entry)
            self._log("Updated agent stats for this pair", "DEBUG")

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
        for entry in self.current_round_pairs:
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

        # Accumulate pairs across rounds with round tagging
        current_round = int(self.current_round_key.split("-")[1]) if self.current_round_key else 1
        for pair in self.current_round_pairs:
            pair["round_num"] = current_round
        self.all_rounds_pairs.extend(self.current_round_pairs)

        return self.current_round_pairs

    def run_stage_5_update_positions(self) -> None:
        """
        Stage 5: Belief Updating and Position Reframing

        For each agent, gathers all resolution outcomes where they were the target.
        Uses this to synthesize an updated version of their beliefs and claims.

        Uses the gather-then-apply pattern: each agent's full Stage 5 pipeline
        (Phase 1 → Phase 2 for CBS, single-phase for legacy) runs
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
        for entry in self.current_round_pairs:
            target = entry["target"]
            grouped_by_target.setdefault(target, []).append(entry)

        self._log(f"Grouped adjudication results for {len(grouped_by_target)} agent(s)", "INFO")

        # Extract config values for the gather function (avoid self.* in threads)
        generation_temp = self.config.stages.generation_temperature if self.config else 0.2
        max_retries = self.config.stages.parse_retries if self.config else 3
        defense_boost_config = self.config.defense_boost if self.config else None

        # --- GATHER: Build work items for agents with adjudication results ---
        items = [
            WorkItem(
                key=agent.name,
                callable=lambda a=agent, re=grouped_by_target[agent.name]: _run_stage5_for_agent(
                    agent=a,
                    relevant_entries=re,
                    last_rebuttals_patches=self.last_rebuttals_patches.get(a.name, []),
                    generation_temp=generation_temp,
                    max_retries=max_retries,
                    defense_boost_config=defense_boost_config,
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

            # Accumulate operational metrics from Stage 5 pure function
            for u in r.get("usages", []):
                self._accumulate_tokens_from_usage(u)
            self.metrics.total_retries += r.get("retry_count", 0)

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
                self.debug_log.write(entry)

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

    def run(self, topic: str, personas: dict[str, str],
            progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
            on_error: Callable[[str, Exception, int], str] | None = None) -> dict:
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
                dict: Contains initial positions, final positions, transcript,
                debug log, and agent statistics.
            """
            self._progress_callback = progress_callback
            self._on_error = on_error
            self.metrics.start_time = time.time()

            self._notify("debate_start", {
                "topic": topic,
                "num_agents": len(self.agents),
                "num_rounds": self.max_rounds,
            })

            # Initialize the belief tracker
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
                self.run_stage_2_cross_examination()

                # Stage 3: Rebuttals
                self.run_stage_3_rebuttals()
                # Stage 4: Conflict Resolution
                self.run_stage_4_conflict_resolution()

                # Stage 5: Belief Update
                self.run_stage_5_update_positions()

                # Calculate performance scores after each round
                self.agent_stats = calculate_performance_scores(self.agent_stats)

                # Capture per-round snapshot (thesis_strength + component counts)
                # for each agent, after Stage 5 beliefs and this round's performance
                # scores are settled.
                round_key = f"round_{round_num}"
                for agent in self.agents:
                    try:
                        belief_obj = agent.get_internal_belief_obj()
                    except Exception as e:
                        self._log(f"{round_key} snapshot: failed to read belief for {agent.name}: {e}", "WARNING")
                        belief_obj = None
                    self.agent_stats[agent.name]["per_round"][round_key] = snapshot_belief(
                        belief_obj if isinstance(belief_obj, dict) else {}
                    )

                # Log performance summary for this round
                perf_summary = get_performance_summary(self.agent_stats)
                self._log(f"\n{perf_summary}", "INFO")

                # Notify: round complete with scores
                self._notify("round_complete", {
                    "round": round_num,
                    "scores": self.agent_stats,
                })

            # Track the final beliefs of the agents
            for agent in self.agents:
                belief_obj = agent.get_internal_belief_obj()
                if belief_obj is not None:
                    embedding_tracker.embed_belief(belief_obj, agent_name=agent.name, round_num=self.max_rounds)
                else:
                    text_for_embedding = agent.get_internal_belief()  # legacy fallback
                    embedding_tracker.embed_belief(text_for_embedding, agent_name=agent.name, round_num=self.max_rounds)
            # Save the embeddings
            output_dir = self.config.outputs.storage_dir if self.config else STORAGE_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            agent_info = {
                a.name: {"model": a.model, "provider": a.provider, "persona": a.persona}
                for a in self.config.agents
            } if self.config else None
            topic = self.config.topic if self.config else None
            embedding_tracker.save_embeddings(
                output_dir / "embeddings.npz",
                agent_info=agent_info,
                topic=topic,
            )

            # Generate belief graph visualization (if enabled)
            if self.config and hasattr(self.config.outputs, 'generate_graph_visualization') and self.config.outputs.generate_graph_visualization:
                self._log("Generating interactive belief graph visualization...", "INFO")

                try:
                    graph_output_path = self.config.outputs.storage_dir / self.config.outputs.graph_file
                    html_content = export_debate_graph(
                        agents=self.agents,
                        topic=self.topic,
                        current_round_pairs=self.all_rounds_pairs,
                        output_path=graph_output_path
                    )

                    # Write HTML to file
                    with open(graph_output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    self._log(f"Belief graph visualization saved to {graph_output_path}", "INFO")
                except Exception as e:
                    self._log(f"Warning: Failed to generate belief graph visualization: {e}", "WARNING")

            # Calculate final performance scores
            self.agent_stats = calculate_performance_scores(self.agent_stats)

            # Finalize timing
            self.metrics.end_time = time.time()

            # Assemble expanded stats: attack histograms, adjudicator verdicts,
            # final_snapshot per agent, and the top-level "_debate_aggregate"
            # sentinel summarising debate-wide totals.
            self.agent_stats = finalize_agent_stats(
                self.agent_stats,
                self.all_rounds_pairs,
                self.agents,
                self.max_rounds,
                operational_metrics=self.metrics.to_dict(),
            )

            # Log agent stats (display handled by callback / display layer)
            self._log(f"Final agent stats: {json.dumps(self.agent_stats, default=str)}", "INFO")

            # Log D# statistics per agent
            self._log_definition_statistics()

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

                try:
                    report_md = generate_analysis_report(
                        config=self.config,
                        agents=self.agents,
                        challenge_rebuttal_pairs=self.all_rounds_pairs,
                        agent_stats=self.agent_stats,
                        opening_positions=self.opening_positions,
                    )
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_md)
                    self._log(f"Analysis report saved to {report_path}", "INFO")

                    # Also save JSON version
                    json_report_path = report_path.with_suffix('.json')
                    report_json = generate_analysis_json(
                        config=self.config,
                        agents=self.agents,
                        challenge_rebuttal_pairs=self.all_rounds_pairs,
                        agent_stats=self.agent_stats,
                    )
                    with open(json_report_path, 'w', encoding='utf-8') as f:
                        json.dump(report_json, f, ensure_ascii=False, indent=2)
                    self._log(f"Analysis JSON saved to {json_report_path}", "INFO")
                except Exception as e:
                    self._log(f"Warning: Failed to generate analysis report: {e}", "WARNING")

            # Final logging
            self._log_header("DEBATE COMPLETE")
            self._log("Total stages completed: 6 (including briefing)", "INFO")
            self._log(f"Debug log entries: {len(self.debug_log)}", "INFO")
            self._log(f"Markdown transcript entries: {len(self.markdown_transcript)}", "INFO")

            self._notify("debate_complete", {
                "agent_stats": self.agent_stats,
                "total_duration_s": self.metrics.duration_s,
                "operational_metrics": self.metrics.to_dict(),
            })

            self._close_debug_log()

            return {
                "initial_positions": self.opening_positions,
                "final_positions": [agent.internal_belief for agent in self.agents],
                "markdown_transcript": "\n".join(self.markdown_transcript),
                "debug_log": self.debug_log.get_contents(),
                "agent_stats": self.agent_stats,
                "operational_metrics": self.metrics.to_dict(),
            }

# --- Pure Functions for Parallel Dispatch ---

def _generate_opening_position(agent, topic: str, max_retries: int = 3,
                               log_fn=None) -> dict:
    """Pure function: generate an opening position for one agent.

    Handles the full API call → parse → graph-validation retry loop.
    Returns a dict of results for the sequential apply step.

    This function is safe to call from a worker thread because it only
    reads/writes its own local state and the thread-safe agent.generate()
    method.
    """
    from chal.beliefs.belief_graph import BeliefGraph
    from chal.utilities.retry import generate_with_retry
    from chal.utilities.validators import (
        STAGE1_REMEDIATION_HINTS,
        validate_stage1_output,
    )

    opening_prompt = prompts.build_stage_1_belief_prompt_cbs(
        topic=topic, agent_name=agent.name, persona_label=agent.persona_label
    )
    stage_request = [Message(role="user", content=opening_prompt)]

    # Layer 1: Retry until we get parseable, schema-valid JSON
    response, retry_records = generate_with_retry(
        agent=agent,
        messages=stage_request,
        validator_fn=validate_stage1_output,
        max_retries=max_retries,
        stage_label=f"Stage 1 Opening Position ({agent.name})",
        log_fn=log_fn,
        remediation_hints=STAGE1_REMEDIATION_HINTS,
    )

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

    # Extract usage for token accumulation by the controller
    usage = {}
    if response and hasattr(response, "metadata") and response.metadata:
        usage = response.metadata.get("usage", {})

    return {
        "response": response,
        "belief_obj": belief_obj,
        "md_view": md_view,
        "errs": errs,
        "prompt": opening_prompt,
        "graph_metrics": graph_metrics,
        "validation_logs": validation_logs,
        "retry_records": retry_records,
        "usage": usage,
    }


def _generate_cross_examination(challenger, target, topic, config, previous_challenges,
                                 max_retries=3, log_fn=None) -> dict:
    """Pure function: generate cross-examination questions from challenger to target.

    Returns a dict of results for the sequential apply step.
    """
    from chal.utilities.retry import generate_with_retry
    from chal.utilities.validators import (
        STAGE2_REMEDIATION_HINTS,
        validate_stage2_output,
    )

    challenger_name = challenger.name
    target_name = target.name

    ch_belief_obj = challenger.get_internal_belief_obj()
    tg_belief_obj = target.get_internal_belief_obj()
    ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
    tg_belief_json = json.dumps(tg_belief_obj, ensure_ascii=False, indent=2) if tg_belief_obj else ""

    max_questions = config.stages.max_questions_per_cross_exam if config else 5

    prompt = prompts.build_stage_2_prompt(
        topic=topic,
        agent_name=challenger_name,
        opponent_name=target_name,
        agent_belief_json=ch_belief_json,
        opponent_belief_json=tg_belief_json,
        max_questions=max_questions,
        previous_challenges=previous_challenges if previous_challenges else None,
    )

    stage_request = [Message(role="user", content=prompt)]
    generation_temp = config.stages.generation_temperature if config else 0.2

    response, retry_records = generate_with_retry(
        agent=challenger,
        messages=stage_request,
        validator_fn=validate_stage2_output,
        max_retries=max_retries,
        stage_label="Stage 2 Cross-Examination",
        log_fn=log_fn,
        temperature=generation_temp,
        remediation_hints=STAGE2_REMEDIATION_HINTS,
    )

    # Parse the FIRST fenced JSON block -> {"questions":[...]}
    questions_obj = _extract_first_json_block(response.content)
    questions = (questions_obj or {}).get("questions", [])

    # Fallback to legacy parser
    parsed_challenges = []
    if not questions:
        parsed_challenges = parse_challenges(response.content)

    usage = {}
    if response and hasattr(response, "metadata") and response.metadata:
        usage = response.metadata.get("usage", {})

    return {
        "response": response,
        "prompt": prompt,
        "questions": questions,
        "parsed_challenges": parsed_challenges,
        "ch_belief_obj": ch_belief_obj,
        "tg_belief_obj": tg_belief_obj,
        "retry_records": retry_records,
        "usage": usage,
    }


def _generate_rebuttal(target_agent, relevant_entries, topic, config,
                       max_retries=3, log_fn=None) -> dict:
    """Pure function: generate rebuttals for one target agent.

    Returns a dict of results for the sequential apply step.
    """
    from functools import partial

    from chal.utilities.retry import generate_with_retry
    from chal.utilities.validators import (
        STAGE3_REMEDIATION_HINTS,
        validate_stage3_output,
    )

    target_name = target_agent.name

    # Assign globally unique sequential qids (Q1, Q2, ..., QN) across all
    # opponents.  Each opponent independently numbers their questions Q1-Q5,
    # so the originals collide when combined.  Sequential renumbering ensures
    # unique qids for the LLM, the validator, and the mapping-back step.
    questions_payload = {
        "questions": [
            {
                "qid": f"Q{idx+1}",
                "text": e["challenge"],
                "target_ids": e.get("target_ids", []),
                "from": e["challenger"]
            }
            for idx, e in enumerate(relevant_entries)
        ]
    }
    # Update entries so the mapping-back step (by_qid lookup) uses the same
    # renumbered qids that the LLM will echo in its response.
    for idx, e in enumerate(relevant_entries):
        e["qid"] = f"Q{idx+1}"
    received_questions_json = json.dumps(questions_payload, ensure_ascii=False, indent=2)

    # Collect expected qids for coverage validation
    expected_qids = [q["qid"] for q in questions_payload["questions"]]

    tgt_belief_obj = target_agent.get_internal_belief_obj()
    tgt_belief_json = json.dumps(tgt_belief_obj, ensure_ascii=False, indent=2) if tgt_belief_obj else ""

    challenger_names = sorted(set(q["from"] for q in questions_payload["questions"]))
    opponent_name = " & ".join(challenger_names) if challenger_names else "Opponent"

    max_rebuttals = len(questions_payload["questions"])
    max_rebuttal_length = config.stages.max_rebuttal_length_chars if config else 500
    prompt = prompts.build_stage_3_structured_rebuttal_prompt(
        topic=topic,
        agent_name=target_name,
        opponent_name=opponent_name,
        received_questions_json=received_questions_json,
        agent_belief_json=tgt_belief_json,
        max_rebuttals=max_rebuttals,
        max_rebuttal_length_chars=max_rebuttal_length
    )

    stage_request = [Message(role="user", content=prompt)]
    generation_temp = config.stages.generation_temperature if config else 0.2

    validator = partial(validate_stage3_output, expected_qids=expected_qids)
    response, retry_records = generate_with_retry(
        agent=target_agent,
        messages=stage_request,
        validator_fn=validator,
        max_retries=max_retries,
        stage_label="Stage 3 Rebuttal",
        log_fn=log_fn,
        temperature=generation_temp,
        remediation_hints=STAGE3_REMEDIATION_HINTS,
    )

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

    usage = {}
    if response and hasattr(response, "metadata") and response.metadata:
        usage = response.metadata.get("usage", {})

    return {
        "response": response,
        "prompt": prompt,
        "rebuttals": rebuttals,
        "patches": patches,
        "received_questions_json": received_questions_json,
        "tgt_belief_obj": tgt_belief_obj,
        "relevant_entries": relevant_entries,
        "retry_records": retry_records,
        "usage": usage,
    }


def _run_adjudication(adjudicator, entry, agent_by_name, max_retries=3, log_fn=None) -> dict:
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
        target_belief_excerpt_json=target_belief_excerpt_json,
        max_retries=max_retries,
        log_fn=log_fn,
    )

    return {"resolution": resolution}


def _apply_patches_and_validate(
    name: str,
    response_content: str,
    prior_json: dict,
    relevant_entries: list,
    md_view_fallback: str,
) -> dict:
    """Parse patches from response, apply to belief, validate graph.

    Pure function — no agent or controller mutations.
    Used by the legacy flow within _run_stage5_for_agent.

    Returns:
        dict with keys: patches, final_belief, md_view, reverted,
        log_entries, debug_entries.
    """
    from chal.beliefs.belief_graph import BeliefGraph
    from chal.beliefs.patches import apply_patches, validate_patches

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
    last_rebuttals_patches: list,
    generation_temp: float,
    max_retries: int,
    defense_boost_config: DefenseBoostConfig | None = None,
) -> dict:
    """Pure function: run full Stage 5 belief update for one agent.

    Handles CBS two-phase and legacy flows.
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

    if prior_json is not None:
        return _run_stage5_cbs_two_phase(
            agent, name, prior_json, md_view_fallback,
            relevant_entries, last_rebuttals_patches,
            generation_temp, max_retries,
            api_calls, log_entries, debug_entries,
            defense_boost_config=defense_boost_config,
        )

    else:
        return _run_stage5_legacy(
            agent, name, prior_json, md_view_fallback,
            relevant_entries,
            generation_temp, max_retries,
            api_calls, log_entries, debug_entries,
        )


def _run_stage5_cbs_two_phase(
    agent, name, prior_json, md_view_fallback,
    relevant_entries, last_rebuttals_patches,
    generation_temp, max_retries,
    api_calls, log_entries, debug_entries,
    defense_boost_config: DefenseBoostConfig | None = None,
) -> dict:
    """CBS two-phase flow (Phase 1: enforcement, Phase 2: introspection)."""
    log_entries.append((f"Using two-phase CBS belief update for {name}", "DEBUG"))

    from functools import partial

    from chal.beliefs.belief_graph import BeliefGraph
    from chal.beliefs.patches import apply_patches, validate_patches
    from chal.utilities.retry import generate_with_retry
    from chal.utilities.validators import (
        STAGE5_PHASE1_REMEDIATION_HINTS,
        STAGE5_PHASE2_REMEDIATION_HINTS,
        validate_stage5_phase1_output,
        validate_stage5_phase2_output,
    )

    stage_3_patches_json = (
        json.dumps(last_rebuttals_patches, ensure_ascii=False, indent=2)
        if last_rebuttals_patches else ""
    )
    prior_belief_json_str = json.dumps(prior_json, ensure_ascii=False, indent=2)

    # Count enforcement-relevant outcomes for validator
    critique_valid_count = sum(
        1 for entry in relevant_entries
        if (entry.get("resolution") or {}).get("status") == "critique_valid"
    )

    # --- PHASE 1: Adjudication Enforcement ---
    log_entries.append((f"Phase 1: Adjudication enforcement for {name}", "INFO"))

    phase1_prompt = prompts.build_stage_5_phase1_enforcement_prompt(
        agent_name=name,
        challenge_rebuttal_pairs=relevant_entries,
        prior_belief_json=prior_belief_json_str,
        stage_3_patches_json=stage_3_patches_json,
    )

    phase1_validator = partial(validate_stage5_phase1_output, critique_valid_count=critique_valid_count)
    phase1_response, phase1_retries = generate_with_retry(
        agent=agent,
        messages=[Message(role="user", content=phase1_prompt)],
        validator_fn=phase1_validator,
        max_retries=max_retries,
        stage_label="Stage 5 Phase 1",
        log_fn=lambda msg, lvl: log_entries.append((msg, lvl)),
        temperature=generation_temp,
        remediation_hints=STAGE5_PHASE1_REMEDIATION_HINTS,
    )
    api_calls.append({
        "prompt": phase1_prompt, "response": phase1_response,
        "label": "Stage 5 Phase 1 - Enforcement", "is_retry": False,
    })

    # Parse Phase 1 patches
    phase1_patches = []
    phase1_blocks = _extract_all_json_blocks(phase1_response.content)

    intermediate_belief = prior_json  # Default: unchanged

    if phase1_blocks:
        phase1_patches_block = json.loads(phase1_blocks[0])
        phase1_patches = phase1_patches_block.get("patches", [])

    if phase1_patches:
        # Cap response_sufficiency on new counterpositions before validation
        cap_phase1_counterposition_sufficiency(phase1_patches, log_entries)

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
        # Enforcement compliance fallback (validator retries may have been exhausted)
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

    # --- DEFENSE BOOSTS: Mechanical strength increases for REBUTTAL_VALID ---
    intermediate_belief = apply_defense_boosts(
        belief=intermediate_belief,
        challenge_rebuttal_pairs=relevant_entries,
        log_entries=log_entries,
        agent_name=name,
        boost_config=defense_boost_config,
    )

    # --- DEPENDENCY CEILING: Ensure defense-boosted claims don't exceed
    # the minimum strength of their active dependencies ---
    for claim in intermediate_belief.get("claims", []):
        if claim.get("status") == "retracted":
            continue
        deps = claim.get("depends_on", [])
        if not deps:
            continue
        dep_strengths = []
        for dep_id in deps:
            # Look up dependency strength from assumptions, evidence, or claims
            for collection in ("assumptions", "evidence", "claims"):
                for node in intermediate_belief.get(collection, []):
                    if node["id"] == dep_id and node.get("status") != "retracted":
                        dep_strengths.append(node.get("strength", 0.5))
                        break
        if dep_strengths:
            min_dep = min(dep_strengths)
            if claim["strength"] > min_dep:
                log_entries.append((
                    f"Dependency ceiling: {claim['id']} strength "
                    f"{claim['strength']:.4f} → {min_dep:.4f} "
                    f"(limited by weakest active dependency)",
                    "INFO",
                ))
                claim["strength"] = min_dep

    # --- PHASE 2: Introspective Evaluation ---
    log_entries.append((f"Phase 2: Introspective evaluation for {name}", "INFO"))

    phase2_prompt = prompts.build_stage_5_phase2_introspection_prompt(
        agent_name=name,
        intermediate_belief_json=json.dumps(intermediate_belief, ensure_ascii=False, indent=2),
        phase1_changes_summary=phase1_summary,
    )

    phase2_response, phase2_retries = generate_with_retry(
        agent=agent,
        messages=[Message(role="user", content=phase2_prompt)],
        validator_fn=validate_stage5_phase2_output,
        max_retries=max_retries,
        stage_label="Stage 5 Phase 2",
        log_fn=lambda msg, lvl: log_entries.append((msg, lvl)),
        temperature=generation_temp,
        remediation_hints=STAGE5_PHASE2_REMEDIATION_HINTS,
    )
    api_calls.append({
        "prompt": phase2_prompt, "response": phase2_response,
        "label": "Stage 5 Phase 2 - Introspection", "is_retry": False,
    })

    # Parse Phase 2 patches
    phase2_patches = []
    phase2_blocks = _extract_all_json_blocks(phase2_response.content)

    final_belief = intermediate_belief  # Default: Phase 1 result

    if phase2_blocks:
        phase2_patches_block = json.loads(phase2_blocks[0])
        phase2_patches = phase2_patches_block.get("patches", [])

    if phase2_patches:
        # GUARDRAIL: No unilateral strengthening
        original_count = len(phase2_patches)
        phase2_patches = filter_strength_increases(phase2_patches, intermediate_belief)
        filtered_count = original_count - len(phase2_patches)
        if filtered_count > 0:
            log_entries.append((
                f"Phase 2: stripped/filtered {filtered_count} strength increase(s) for {name}",
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

    # Collect usage from all API calls and retry records
    usages = []
    retry_count = len(phase1_retries) + len(phase2_retries)
    for call in api_calls:
        resp = call.get("response")
        if resp and hasattr(resp, "metadata") and resp.metadata:
            u = resp.metadata.get("usage", {})
            if u:
                usages.append(u)

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
        "usages": usages,
        "retry_count": retry_count,
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

    from functools import partial

    from chal.utilities.retry import generate_with_retry
    from chal.utilities.validators import (
        STAGE5_PHASE1_REMEDIATION_HINTS,
        validate_stage5_phase1_output,
    )

    critique_valid_count = sum(
        1 for entry in relevant_entries
        if (entry.get("resolution") or {}).get("status") == "critique_valid"
    )
    validator = partial(validate_stage5_phase1_output, critique_valid_count=critique_valid_count)
    response, retry_records = generate_with_retry(
        agent=agent,
        messages=stage_request,
        validator_fn=validator,
        max_retries=max_retries,
        stage_label="Stage 5 Legacy",
        log_fn=lambda msg, lvl: log_entries.append((msg, lvl)),
        temperature=generation_temp,
        remediation_hints=STAGE5_PHASE1_REMEDIATION_HINTS,
    )
    api_calls.append({"prompt": prompt, "response": response, "label": label, "is_retry": False})

    patch_result = _apply_patches_and_validate(
        name, response.content, prior_json, relevant_entries, md_view_fallback
    )

    # Collect usage from all API calls and retry records
    usages = []
    retry_count = len(retry_records)
    for call in api_calls:
        resp = call.get("response")
        if resp and hasattr(resp, "metadata") and resp.metadata:
            u = resp.metadata.get("usage", {})
            if u:
                usages.append(u)

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
        "usages": usages,
        "retry_count": retry_count,
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

def _extract_first_json_block(text: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object from *text*.

    Tries a fenced ``json`` code block first, then falls back to the first
    raw ``{...}`` substring.

    Returns:
        Parsed dict, or None if no valid JSON object is found.
    """
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


def _extract_all_json_blocks(text: str) -> list[str]:
    """Extract all top-level JSON object strings from *text*.

    Tries fenced ``json`` code blocks first.  If none are found, falls
    back to brace-depth tracking to locate raw ``{...}`` objects while
    correctly handling nested braces and quoted strings.

    Returns:
        List of raw JSON strings (not yet parsed).
    """
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


def compute_defense_boost(
    consecutive_defenses: int,
    flat_boost: float = 0.02,
) -> float:
    """Return the flat per-defense strength boost.

    Every successful defense adds the same constant boost regardless of
    streak length.

    Args:
        consecutive_defenses: Number of consecutive successful defenses (1-indexed).
        flat_boost: Constant boost amount per defense.

    Returns:
        The strength boost amount for this defense.
    """
    if consecutive_defenses < 1:
        return 0.0
    return flat_boost


def _gather_dependency_nodes(belief: dict, node_id: str) -> list[str]:
    """BFS backward through the dependency graph to collect all supporting node IDs.

    Traversal edges:
    - C# → depends_on (A#, E#, C#)
    - A# → supported_by_definitions (D#)
    - E# → supported_by_definitions (D#)
    - D# → (leaf — no further dependencies)

    Returns a deduplicated list of dependency IDs (excludes the input node
    itself).  Skips retracted nodes.
    """
    # Build lookup: node_id → node dict
    node_map: dict[str, dict] = {}
    for key in ("definitions", "assumptions", "evidence", "claims"):
        for node in belief.get(key, []):
            nid = node.get("id", "")
            if nid:
                node_map[nid] = node

    result: list[str] = []
    visited: set[str] = {node_id}
    queue: list[str] = [node_id]

    while queue:
        current_id = queue.pop(0)
        current_node = node_map.get(current_id)
        if current_node is None:
            continue

        # Determine which field holds this node's dependencies
        dep_ids: list[str] = []
        if current_id.startswith("C"):
            dep_ids = current_node.get("depends_on", [])
        elif current_id.startswith(("A", "E")):
            dep_ids = current_node.get("supported_by_definitions", [])
        # D# nodes are leaves — no further dependencies

        for dep_id in dep_ids:
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep_node = node_map.get(dep_id)
            if dep_node is None:
                continue
            if dep_node.get("status") == "retracted":
                continue
            result.append(dep_id)
            queue.append(dep_id)

    return result


def apply_defense_boosts(
    belief: dict,
    challenge_rebuttal_pairs: list,
    log_entries: list,
    agent_name: str,
    boost_config: DefenseBoostConfig | None = None,
) -> dict:
    """Mechanically apply defense boosts for REBUTTAL_VALID outcomes.

    For each REBUTTAL_VALID verdict, identify the targeted node(s) from the
    original critique's target_ids. Increment each targeted node's
    consecutive_defenses counter and apply the formula-driven strength boost.

    For each CRITIQUE_VALID verdict, reset the consecutive_defenses counter
    on all targeted nodes to 0 (streak broken).

    Nodes not attacked in this round are untouched.

    Args:
        belief: The intermediate belief dict (post-Phase-1). Deep-copied
                internally; the original is not mutated.
        challenge_rebuttal_pairs: List of {challenger, challenge, rebuttal,
                resolution: {status, ...}, target_ids: [...]} dicts.
        log_entries: Logging list for debug output.
        agent_name: Name of the agent whose belief is being boosted.
        boost_config: Defense boost configuration. If None, uses defaults.

    Returns:
        The modified belief dict with defense boosts applied.
    """
    import copy

    if boost_config is None:
        boost_config = DefenseBoostConfig()

    if not boost_config.enabled:
        return belief

    belief = copy.deepcopy(belief)

    # Build a lookup: node_id -> (collection_key, index)
    node_lookup: dict[str, tuple[str, int]] = {}
    for key in ("definitions", "assumptions", "evidence", "claims"):
        for idx, node in enumerate(belief.get(key, [])):
            nid = node.get("id", "")
            if nid:
                node_lookup[nid] = (key, idx)

    # Build resolution map for U#/X# → their underlying D#/A#/E#/C# target nodes
    indirect_targets: dict[str, list[str]] = {}
    for u in belief.get("uncertainties", []):
        uid = u.get("id", "")
        if uid:
            indirect_targets[uid] = [t for t in u.get("targets", []) if t in node_lookup]
    for x in belief.get("counterpositions", []):
        xid = x.get("id", "")
        if xid:
            indirect_targets[xid] = [t for t in x.get("targets", []) if t in node_lookup]

    # Process each adjudication outcome
    for pair in challenge_rebuttal_pairs:
        resolution = pair.get("resolution") or {}
        status = resolution.get("status", "")
        target_ids = pair.get("target_ids", [])

        # Resolve U#/X# targets to their underlying D#/A#/E#/C# nodes
        resolved_ids: list[str] = []
        seen: set[str] = set()
        for tid in target_ids:
            if tid in node_lookup:
                if tid not in seen:
                    resolved_ids.append(tid)
                    seen.add(tid)
            elif tid in indirect_targets:
                for resolved_tid in indirect_targets[tid]:
                    if resolved_tid not in seen:
                        resolved_ids.append(resolved_tid)
                        seen.add(resolved_tid)
        target_ids = resolved_ids

        if not target_ids:
            continue

        if status == "rebuttal_valid":
            # Track all nodes that receive a direct boost in this verdict
            # so we can cascade afterward without double-boosting
            directly_boosted: dict[str, float] = {}  # tid -> boost amount

            for tid in target_ids:
                if tid not in node_lookup:
                    continue
                collection_key, idx = node_lookup[tid]
                node = belief[collection_key][idx]

                # Skip retracted nodes
                if node.get("status") == "retracted":
                    continue

                # Ensure original_strength is set
                if "original_strength" not in node:
                    node["original_strength"] = node.get("strength", 0.5)

                # Increment consecutive defenses
                prev_count = node.get("consecutive_defenses", 0)
                new_count = prev_count + 1
                node["consecutive_defenses"] = new_count

                # Calculate and apply boost (using config parameters)
                boost = compute_defense_boost(
                    new_count,
                    flat_boost=boost_config.flat_boost,
                )
                original = node["original_strength"]
                current = node.get("strength", 0.5)
                ceiling = min(original + boost_config.max_cumulative_boost, 1.0)
                new_strength = min(current + boost, ceiling)

                if new_strength > current:
                    log_entries.append((
                        f"Defense boost: {tid} ({collection_key}) for {agent_name}: "
                        f"{current:.2f} → {new_strength:.2f} "
                        f"(defense #{new_count}, boost +{boost:.2f}, "
                        f"ceiling {ceiling:.2f})",
                        "INFO",
                    ))
                    node["strength"] = round(new_strength, 4)
                else:
                    log_entries.append((
                        f"Defense boost: {tid} for {agent_name}: at ceiling "
                        f"({current:.2f} >= {ceiling:.2f}), no increase",
                        "DEBUG",
                    ))
                directly_boosted[tid] = boost

            # --- CASCADE: Apply same boost to all dependency nodes ---
            cascade_ids: set[str] = set()
            for tid in directly_boosted:
                for dep_id in _gather_dependency_nodes(belief, tid):
                    cascade_ids.add(dep_id)
            # Remove any IDs already boosted as direct targets
            cascade_ids -= set(directly_boosted.keys())

            # Use the maximum boost amount from the direct targets
            if directly_boosted:
                cascade_boost = max(directly_boosted.values())
            else:
                cascade_boost = 0.0

            for dep_id in sorted(cascade_ids):
                if dep_id not in node_lookup:
                    continue
                dep_coll, dep_idx = node_lookup[dep_id]
                dep_node = belief[dep_coll][dep_idx]

                if dep_node.get("status") == "retracted":
                    continue

                # Ensure original_strength is set
                if "original_strength" not in dep_node:
                    dep_node["original_strength"] = dep_node.get("strength", 0.5)

                dep_original = dep_node["original_strength"]
                dep_current = dep_node.get("strength", 0.5)
                dep_ceiling = min(dep_original + boost_config.max_cumulative_boost, 1.0)
                dep_new = min(dep_current + cascade_boost, dep_ceiling)

                if dep_new > dep_current:
                    log_entries.append((
                        f"Defense boost (cascade): {dep_id} ({dep_coll}) for "
                        f"{agent_name}: {dep_current:.2f} → {dep_new:.2f} "
                        f"(cascade boost +{cascade_boost:.2f}, "
                        f"ceiling {dep_ceiling:.2f})",
                        "INFO",
                    ))
                    dep_node["strength"] = round(dep_new, 4)

        elif status == "critique_valid":
            # Reset consecutive defense counter for targeted nodes
            for tid in target_ids:
                if tid not in node_lookup:
                    continue
                collection_key, idx = node_lookup[tid]
                node = belief[collection_key][idx]
                prev_count = node.get("consecutive_defenses", 0)
                if prev_count > 0:
                    node["consecutive_defenses"] = 0
                    log_entries.append((
                        f"Defense streak reset: {tid} for {agent_name} "
                        f"(was {prev_count}, reset due to CRITIQUE_VALID)",
                        "DEBUG",
                    ))

    return belief


def cap_phase1_counterposition_sufficiency(
    patches: list[dict],
    log_entries: list | None = None,
) -> list[dict]:
    """Downgrade 'sufficient' → 'partial' on new counterpositions in Phase 1.

    When an agent loses an argument (CRITIQUE_VALID), Phase 1 forces them to
    add a counterposition acknowledging the vulnerability.  Allowing immediate
    "sufficient" lets the agent claim it has fully addressed an argument it
    just lost—defeating the purpose.  This function caps the maximum
    response_sufficiency at "partial" so the agent must refine its position
    in Phase 2 or later rounds before reaching "sufficient".

    Only affects ``add_counterposition`` operations; ``update_counterposition``
    is untouched (agents can always upgrade *existing* counterpositions).

    Mutates ``patches`` in-place and returns the same list for convenience.
    """
    for patch in patches:
        if patch.get("op") == "add_counterposition":
            item = patch.get("item")
            # NOTE: "moot" is intentionally not capped here — it indicates the
            # counterposition targets a retracted claim and is already obsolete.
            if isinstance(item, dict) and item.get("response_sufficiency") == "sufficient":
                item["response_sufficiency"] = "partial"
                if log_entries is not None:
                    log_entries.append((
                        f"Phase 1 sufficiency cap: {item.get('id', '?')} "
                        f"downgraded from 'sufficient' → 'partial'",
                        "INFO",
                    ))
    return patches


def filter_strength_increases(phase2_patches: list, intermediate_belief: dict) -> list:
    """Strip strength increases from Phase 2 patches on existing nodes.

    Phase 2 is unilateral — no opponent scrutiny — so existing node strengths
    can only stay the same or go down, never up. This prevents "trust me bro"
    self-strengthening.

    For update_* operations (update_claim, update_assumption, update_evidence,
    update_definition): if the patch would increase the node's strength above
    its current value in intermediate_belief, the strength field is removed
    from the patch's changes dict. The rest of the patch (semantic changes,
    status changes, etc.) is preserved.

    For update_thesis: if new_strength > current thesis strength, the
    new_strength is removed. If change == "strengthen", the patch is dropped.

    add_* operations are NOT affected — new nodes can have any strength.

    Args:
        phase2_patches: List of Phase 2 patch dicts.
        intermediate_belief: The post-Phase-1, post-defense-boost belief.

    Returns:
        Filtered copy of phase2_patches with strength increases stripped.
    """
    import copy

    # Build lookup: node_id -> current strength
    current_strengths: dict[str, float] = {}
    for key in ("definitions", "assumptions", "evidence", "claims"):
        for node in intermediate_belief.get(key, []):
            nid = node.get("id", "")
            if nid and "strength" in node:
                current_strengths[nid] = node["strength"]

    thesis_strength = intermediate_belief.get("thesis", {}).get("strength", 0.5)

    filtered = []
    for patch in phase2_patches:
        op = patch.get("op", "")

        # update_claim, update_assumption, update_evidence, update_definition
        if op in ("update_claim", "update_assumption", "update_evidence", "update_definition"):
            tid = patch.get("target_id", "")
            changes = patch.get("changes", {})
            new_strength = changes.get("strength")

            if new_strength is not None and tid in current_strengths:
                if new_strength > current_strengths[tid]:
                    # Strip the strength increase; keep everything else
                    patch = copy.deepcopy(patch)
                    del patch["changes"]["strength"]
                    # Also strip strength_justification if strength was removed
                    patch["changes"].pop("strength_justification", None)
                    # If changes dict is now empty, skip the whole patch
                    if not patch.get("changes"):
                        continue
            filtered.append(patch)

        elif op == "update_thesis":
            new_str = patch.get("new_strength")
            change = patch.get("change")

            if change == "strengthen":
                continue  # Drop the entire patch

            if new_str is not None and new_str > thesis_strength:
                # Strip the strength increase from thesis update
                patch = copy.deepcopy(patch)
                patch.pop("new_strength", None)
                # Keep stance/summary_bullets/strength_reasoning if present
                if not any(k in patch for k in ("stance", "summary_bullets", "strength_reasoning")):
                    if patch.get("op"):
                        # Only has op — skip entirely
                        continue
            filtered.append(patch)

        else:
            # add_* operations, resolve_uncertainty, etc. — pass through
            filtered.append(patch)

    return filtered

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
