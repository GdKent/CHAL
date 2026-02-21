"""
debate_controller.py

Orchestrates structured multi-agent philosophical debates using the CHAL Belief Schema.

The DebateController manages a 7-stage debate process:
- Stage 0: Briefing - Initialize agents with personas and universal rules
- Stage 1: Opening Positions - Agents state initial beliefs as structured JSON (CBS-v1)
- Stage 2: Cross-Examination - Agents ask targeted questions about opponents' claims/assumptions
- Stage 3: Rebuttals - Agents respond to questions with structured answers and optional belief patches
- Stage 4: Adjudication - Independent evaluator assesses challenge-rebuttal pairs
- Stage 5: Belief Updates - Agents revise beliefs based on adjudication outcomes
- Stage 6: Concluding Remarks - Agents synthesize their positions and concessions
- Stage 7: Scribing - Generate a flowing narrative synthesis of the entire debate

Features:
- Structured belief tracking with JSON schemas (CBS-v1)
- Embedding-based belief trajectory visualization
- Configurable adjudication weights (logic vs. ethics)
- Token-optimized prompts (JSON-only responses, Markdown generated programmatically)
- Round-robin debate structure with multiple rounds

Note: All belief outputs are JSON-first. Human-readable Markdown is generated
programmatically using belief_to_markdown() to minimize token usage.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from chal.agents.base import Agent, Message
from chal.agents import prompts
from chal.agents.factory import create_agent
from chal.orchestrator.adjudicator import Adjudicator
from chal.utilities.utils import parse_challenges, parse_structured_rebuttals_numbered, initialize_agent_stats, update_agent_stats, display_agent_stats, calculate_performance_scores, get_performance_summary
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.convergence import calculate_claim_agreement, format_convergence_summary, get_convergence_trajectory_summary
from chal.beliefs.io import parse_model_output_to_belief, belief_to_markdown
from chal.beliefs.io import project_for_embedding
from chal.beliefs.graph_visualizer import export_debate_graph
from chal.config import DebateConfig
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
    agent beliefs using the CHAL Belief Schema (CBS-v1), tracks embeddings for
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
        self.max_rounds = max_rounds
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
        print(markdown_header.strip())

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
        print(markdown_header.strip())

        self.round_histories["round-0"] = []
        self.current_round_key = "round-0"

        for agent in self.agents:
            self._log(f"\n--- Processing agent: {agent.name} ---", "INFO")

            opening_prompt = prompts.build_stage_1_belief_prompt_cbsv1(
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
            self._log(f"Parsing CBS-v1 belief object for {agent.name}...", "INFO")
            belief_obj, md_view, errs = parse_model_output_to_belief(response.content)

            if belief_obj and not errs:
                self._log_parse_result(True, f"Successfully parsed CBS-v1 belief for {agent.name}")
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
                                    f"Provide your revised CBS-v1 belief object."
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
                self._log_parse_result(False, f"Failed to parse CBS-v1 belief for {agent.name}. {err_details}")
                self._log("Falling back to raw response content", "WARN")
                agent.set_internal_belief(response.content.strip())
                agent.all_beliefs_held.append(response.content.strip()) # track beliefs
                md_view = response.content.strip()

            # Log result
            self.round_histories[self.current_round_key].append(response)

            # Add to markdown transcript
            markdown_content = f"\n## {agent.name} - Opening Statement\n\n{md_view}\n"
            self._add_to_markdown(markdown_content)
            print(f"[{agent.name}] Opening statement received")

        self.opening_positions = [agent.internal_belief for agent in self.agents]
        self._log(f"Stage 1 complete - {len(self.opening_positions)} opening positions captured", "INFO")

        return

    def run_stage_2_cross_examination(self, only_if_disagree: bool = False) -> list:
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

        Returns:
            list[dict]: A flat list of critique records, one per challenge.
        """
        # === DEBUG LOG ===
        self._log_header("STAGE 2: CROSS-EXAMINATION")
        self._log(f"Starting cross-examination phase with only_if_disagree={only_if_disagree}", "INFO")

        # === MARKDOWN TRANSCRIPT ===
        markdown_header = "\n# ⚔️ Stage 2: Cross-Examination\n"
        self._add_to_markdown(markdown_header)
        print(markdown_header.strip())

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
                    print(f"[{challenger_name}] agrees with [{target_name}] — skipping.")
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
                prompt = prompts.build_stage_2_prompt(
                    topic=self.topic if hasattr(self, "topic") else "<topic>",
                    agent_name=challenger_name,
                    opponent_name=target_name,
                    agent_belief_json=ch_belief_json,
                    opponent_belief_json=tg_belief_json,
                    max_questions=max_questions,
                    max_question_length_chars=max_question_length,
                    previous_challenges=previous_challenges if previous_challenges else None,
                    opponent_belief_graph=opponent_belief_graph
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
                print(f"[{challenger_name} → {target_name}] Generated {num_questions} question(s)")

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
        print(markdown_header.strip())

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

            # Add to markdown transcript
            markdown_section = f"\n## {target_name} responds\n\n"
            for r in rebuttals:
                qid = r.get("qid", "Q?")
                answer = r.get("answer", "")
                action = r.get("action", "unknown")
                markdown_section += f"**{qid}** ({action}): {answer}\n\n"

            self._add_to_markdown(markdown_section)
            print(f"[{target_name}] Provided {len(rebuttals)} rebuttal(s), {len(patches)} patch(es)")

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
        print(markdown_header.strip())

        adjudication_count = 0
        for entry in self.challenge_rebuttal_pairs:
            # Skip if missing key elements
            if not entry["challenge"] or not entry["rebuttal"]:
                self._log(f"Skipping incomplete pair: {entry['challenger']} → {entry['target']} (missing challenge or rebuttal)", "WARN")
                markdown_note = f"\n*Skipped incomplete pair: {entry['challenger']} → {entry['target']}*\n"
                self._add_to_markdown(markdown_note)
                print(f"⚠️ Incomplete pair: skipping {entry['challenger']} → {entry['target']}")
                continue

            adjudication_count += 1
            self._log(f"\n--- Adjudicating pair #{adjudication_count}: {entry['challenger']} → {entry['target']} ---", "INFO")
            self._log(f"Challenge: {entry['challenge'][:100]}...", "DEBUG")
            self._log(f"Rebuttal: {entry['rebuttal'][:100]}...", "DEBUG")

            # Run logic enforcement pipeline
            self._log("Calling adjudicator.run()...", "INFO")
            resolution = self.adjudicator.run(
                challenge=entry["challenge"],
                rebuttal=entry["rebuttal"],
                challenger=entry["challenger"],
                target=entry["target"]
            )

            self._log(f"Adjudication outcome: {resolution.get('status', 'unknown').upper()}", "INFO")
            self.debug_log.append(f"\n--- ADJUDICATION RESULT ({entry['challenger']} → {entry['target']}) ---\n{json.dumps(resolution, ensure_ascii=False, indent=2)}\n--- END ADJUDICATION ---\n")

            # Save structured result in the entry
            entry["resolution"] = resolution

            # Update agent stats
            self.agent_stats = update_agent_stats(self.agent_stats, entry)
            self._log(f"Updated agent stats for this pair", "DEBUG")

            # Add to markdown transcript
            markdown_section = f"\n### {entry['challenger']} → {entry['target']}\n\n"
            markdown_section += f"**Outcome**: {resolution['status'].upper()}\n\n"
            markdown_section += f"**Reasoning**: {resolution.get('reasoning', 'N/A')}\n\n"
            if resolution.get('restatement'):
                markdown_section += f"**Disagreement**: {resolution['restatement']}\n\n"

            self._add_to_markdown(markdown_section)
            print(f"🧮 [{entry['challenger']} → {entry['target']}] Resolution: {resolution['status'].upper()}")

        self._log(f"Stage 4 complete - adjudicated {adjudication_count} pair(s)", "INFO")

        # Save challenges for anti-repetition in next round (if multi-round debate)
        for entry in self.challenge_rebuttal_pairs:
            challenger = entry.get("challenger")
            target = entry.get("target")
            qid = entry.get("qid")
            target_ids = entry.get("target_ids", [])
            outcome = entry.get("resolution", {}).get("status", "unknown")

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
        print(markdown_header.strip())

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
            if prior_json is not None:
                self._log(f"Using CBS-v1 patch/update prompt for {name}", "DEBUG")
                # Use CBS patch/update builder
                prompt = prompts.build_stage_5_belief_update_prompt_cbsv1(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,  # whatever you collect from adjudication
                    prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2)
                )
            else:
                self._log(f"Using legacy belief update prompt for {name}", "DEBUG")
                # Legacy builder (unchanged)
                prompt = prompts.build_stage_5_belief_update_prompt(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,
                    original_position=agent.get_internal_belief()
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

            # Add to markdown transcript
            markdown_section = f"\n## {name} - Updated Position\n\n{md_view}\n"
            self._add_to_markdown(markdown_section)
            print(f"[{name}] Belief updated")

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
        print(markdown_header.strip())

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

            short_note_max = self.config.stages.short_note_max_chars if self.config else 140
            prompt = prompts.build_stage_6_conclusion_prompt(
                topic=self.topic if hasattr(self, "topic") else "<topic>",
                agent_name=name,
                agent_belief_json=a_belief_json,
                all_past_beliefs=agent.all_beliefs_held,
                short_note_max_chars=short_note_max
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

            # Add to markdown transcript
            markdown_section = f"\n## {name} - Concluding Remarks\n\n{markdown_content}\n"
            self._add_to_markdown(markdown_section)
            print(f"[{name}] Concluding remarks received")

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
        print(markdown_header.strip())

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
            print(f"[Scribe] Chunk {idx}/{len(chunks)}: {len(narrative_md)} chars")

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
        print("[Scribe] Final narrative complete")

        return final_md

    def run(self, topic: str, personas: dict[str, str]) -> dict:
            """
            Executes a full structured debate.

            Args:
                topic (str): The topic of the debate.
                personas (dict): Mapping of agent names to their role prompts.

            Returns:
                dict: Contains final positions, conclusions, and synthesis.
            """
            log = "\n🚀 Debate Start: Topic →" + topic
            print(log)
            self.full_transcript.append(log)

            # Initialize the belief tracker
            log = "\n📦 Initializing Belief Tracker"
            print(log)
            self.full_transcript.append(log)
            embedding_tracker = BeliefEmbeddingTracker()

            # Stage 0: Briefing (agents already initialized)
            self.run_stage_0_briefing(topic, personas)

            # Print Agent Personas
            for agent in self.agents:
                persona = personas.get(agent.name, "")
                print(f"Loaded persona for {agent.name}: {persona[:]}...")

            # Stage 1: Opening Positions
            self.run_stage_1_opening_positions(topic)

            for round_idx in range(self.max_rounds):
                round_num = round_idx + 1
                self.current_round_key = f"round-{round_num}"
                self.round_histories[self.current_round_key] = []  # Initialize storage
                print(f"\n🔁 Debate Round {round_num} of {self.max_rounds}")

                # Track the beliefs of the agents
                for agent in self.agents:
                    belief_obj = agent.get_internal_belief_obj()
                    if belief_obj is not None:
                        text_for_embedding = project_for_embedding(belief_obj)
                    else:
                        text_for_embedding = agent.get_internal_belief()  # legacy fallback

                    embedding_tracker.embed_belief(agent.name, text_for_embedding)

                # Stage 2: Cross-Examination
                self.run_stage_2_cross_examination(only_if_disagree=False)

                # Stage 3: Rebuttals
                self.run_stage_3_rebuttals()

                # Stage 4: Conflict Resolution
                self.run_stage_4_conflict_resolution()

                # Stage 5: Belief Update
                self.run_stage_5_update_positions()

                # Calculate performance scores after each round
                self.agent_stats = calculate_performance_scores(self.agent_stats)

                # Log performance summary for this round
                perf_summary = get_performance_summary(self.agent_stats)
                self._log(f"\n{perf_summary}", "INFO")
                print(f"\n{perf_summary}\n")

                # Calculate convergence metrics (if enabled)
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

                    # Display convergence summary (if enabled)
                    if self.config.convergence.display_in_round_summary:
                        agent_names = [a.name for a in self.agents]
                        conv_summary = format_convergence_summary(
                            convergence_data,
                            agent_names,
                            round_number=round_num
                        )
                        self._log(f"\n{conv_summary}", "INFO")
                        print(f"\n{conv_summary}\n")

            # Track the final beliefs of the agents
            for agent in self.agents:
                belief_obj = agent.get_internal_belief_obj()
                if belief_obj is not None:
                    text_for_embedding = project_for_embedding(belief_obj)
                else:
                    text_for_embedding = agent.get_internal_belief()  # legacy fallback

                embedding_tracker.embed_belief(agent.name, text_for_embedding)
            # Save the embeddings
            # Ensure storage directory exists
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            embedding_tracker.save_embeddings(STORAGE_DIR / "embeddings.npz")

            # Generate belief graph visualization (if enabled)
            if self.config and hasattr(self.config.outputs, 'generate_graph_visualization') and self.config.outputs.generate_graph_visualization:
                self._log("Generating interactive belief graph visualization...", "INFO")
                print("\n[Graph] Generating interactive belief graph visualization...")

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
                    print(f"      [Graph] Saved to {graph_output_path}")
                except Exception as e:
                    self._log(f"Warning: Failed to generate belief graph visualization: {e}", "WARNING")
                    print(f"      [Graph] Warning: Failed to generate visualization: {e}")

            # Stage 6: Concluding Reflections
            self.run_stage_6_concluding_remarks()

            # Calculate final performance scores
            self.agent_stats = calculate_performance_scores(self.agent_stats)

            # Print agent stats with performance ranking
            display_agent_stats(self.agent_stats)

            # Display convergence trajectory (if enabled)
            if self.config and hasattr(self.config, 'convergence') and self.config.convergence.enabled:
                if self.config.convergence.display_in_final_summary and self.convergence_history:
                    conv_trajectory = get_convergence_trajectory_summary(self.convergence_history)
                    self._log(f"\n{conv_trajectory}", "INFO")
                    print(f"\n{conv_trajectory}\n")

            # Stage 7: Scribe Summary
            # Instantiate Scribe with transcript
            self.final_synthesis = self.run_stage_7_scribing(self.scribe_agent)

            print("\n✅ Debate Complete.")

            # Final logging
            self._log_header("DEBATE COMPLETE")
            self._log(f"Total stages completed: 8 (including briefing)", "INFO")
            self._log(f"Debug log entries: {len(self.debug_log)}", "INFO")
            self._log(f"Markdown transcript entries: {len(self.markdown_transcript)}", "INFO")

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