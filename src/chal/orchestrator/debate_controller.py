"""
debate_controller.py

Coordinates a round-robin debate between multiple agents.
Maintains a shared message history and tracks responses per round.
"""

from typing import List
from chal.agents.base import Agent, Message
from chal.agents import prompts
from chal.agents.openai_agent import OpenAIAgent
from chal.orchestrator.adjudicator import Adjudicator
from chal.utilities.utils import parse_challenges, parse_structured_rebuttals_numbered, initialize_agent_stats, update_agent_stats, display_agent_stats
from chal.embeddings.embedding_tracker import BeliefEmbeddingTracker
from chal.beliefs.io import parse_model_output_to_belief, belief_to_markdown
from chal.beliefs.io import project_for_embedding
import tiktoken
import json
import re


class DebateController:
    """
    Manages a multi-agent debate over multiple turns.
    """
    def __init__(self, agents: List[Agent], max_rounds: int = 3):
        """
        Initializes the DebateController with a list of agents and a number of debate rounds.

        Args:
            agents (List[Agent]): A list of LLM-powered agents participating in the debate.
            max_rounds (int): Number of complete debate rounds (each consisting of Stages 2-5).
        """
        self.agents = agents
        self.max_rounds = max_rounds
        self.challenge_rebuttal_pairs = []
        self.opening_positions = []
        self.full_transcript: List[str] = [] # A list of consecutive strings that define the full transcript of the debate
        self.round_histories: dict[str, List[Message]] = {} # A dictionary of full debate rounds for tracking and memory efficient prompting
        self.current_round_key = None  # Tracks the active round name like "round-1"
        self.last_challenges: dict[str, dict[str, str]] = {} # A dictionary of challenges issued (Stage 2): {challenger: {target: challenge}}
        self.last_rebuttals: dict[str, str] = {} # A dictionary of rebuttals per agent (Stage 3): {agent_name: combined rebuttal}
        self.resolution_outcomes: dict[str, dict[str, str]] = {} # A dictionary of adjudicated outcomes per agent (Stage 4): {agent_name: {challenge_text: resolution_result}}

        # Instantiate the adjudicator agent
        adjudicator_prompt = prompts.build_adjudicator_prompt(
            logic_weight=1.0,
            ethics_weight=0.0,
            logic_sys="Classical logic + Bayesian reasoning for inductive support; reject contradictions; prefer simpler hypotheses (Occam's Razor).",
            ethics_sys="None. Only prioritize logical rigor and soundness, not ethical implications" # "Rule-Utilitarianism."
        )
        adjudicator_agent = OpenAIAgent(
            model="gpt-4o",
            name="Adjudicator",
            system_prompt=adjudicator_prompt
        )
        # Build the adjudicator agent
        self.adjudicator = Adjudicator(adjudicator_agent)

        # Instantiate the scribe agent
        self.scribe_agent = OpenAIAgent(
            model="gpt-4o",
            name="Scribe",
            system_prompt=""
        )

        # Initialize agent statistics
        self.agent_stats = initialize_agent_stats([agent.name for agent in self.agents])


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
        log = "🧠 Stage 0: Briefing"
        print(log)
        self.full_transcript.append(log)

        # Shared system prompt
        universal = prompts.build_universal_prompt(topic)

        for agent in self.agents:
            # Apply universal reasoning rules
            agent.receive_system_prompt(universal)
            # Apply agent-specific persona card
            persona = personas.get(agent.name, "")
            role_card = prompts.build_position_prompt(agent.name, persona)
            agent.receive_role_card(role_card)

        # Log that briefing occurred (no output expected from agents)
        #self.history.append(Message(role="system", content="[Stage 0 briefing complete.]"))


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
        log = "\n📖 Stage 1: Opening Positions"
        print(log)
        self.full_transcript.append(log)

        #opening_prompt = prompts.build_stage_1_prompt(topic)

        self.round_histories["round-0"] = []
        self.current_round_key = "round-0"
        for agent in self.agents:
            opening_prompt = prompts.build_stage_1_belief_prompt_cbsv1(
                            topic=topic, agent_name=agent.name, persona_label=agent.persona_label
                        )

            # Build full prompt history
            #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=opening_prompt)]
            stage_request = [Message(role="user", content=opening_prompt)]

            # Generate the opening statement
            response = agent.generate(stage_request)

            belief_obj, md_view, errs = parse_model_output_to_belief(response.content)

            if belief_obj and not errs:
                agent.set_internal_belief_obj(belief_obj)                   # store structured JSON
                agent.set_internal_belief(md_view or belief_to_markdown(belief_obj))  # keep human-readable string too
                agent.all_beliefs_held.append(json.dumps(belief_obj, ensure_ascii=False, indent=2) if belief_obj else "") # track beliefs
            else:
                # Fallback to legacy behavior if parsing failed
                agent.set_internal_belief(response.content.strip())
                agent.all_beliefs_held.append(response.content.strip()) # track beliefs

            # Log result
            self.round_histories[self.current_round_key].append(response)

            log = f"[{agent.name} - Opening Statement]:\n{md_view}\n"
            print(log)
            self.full_transcript.append(log)

        self.opening_positions = [agent.internal_belief for agent in self.agents]

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
        log = "\n⚔️ Stage 2: Cross-Examination"
        print(log)
        self.full_transcript.append(log)

        self.challenge_rebuttal_pairs = []

        for challenger in self.agents:
            challenger_name = challenger.name

            for target in self.agents:
                target_name = target.name
                if challenger_name == target_name:
                    continue

                if only_if_disagree and self._positions_agree(challenger_name, target_name):
                    log = f"[{challenger_name}] agrees with [{target_name}] — skipping."
                    print(log)
                    self.full_transcript.append(log)
                    continue

                # Pull beliefs as JSON strings when available, else fall back to Markdown strings
                ch_belief_obj = challenger.get_internal_belief_obj()
                tg_belief_obj = target.get_internal_belief_obj()
                ch_belief_json = json.dumps(ch_belief_obj, ensure_ascii=False, indent=2) if ch_belief_obj else ""
                tg_belief_json = json.dumps(tg_belief_obj, ensure_ascii=False, indent=2) if tg_belief_obj else ""

                # Use the Stage 2 prompt (topic-aware, ID-targeting)
                prompt = prompts.build_stage_2_prompt(
                    topic=self.topic if hasattr(self, "topic") else "<topic>",
                    agent_name=challenger_name,
                    opponent_name=target_name,
                    agent_belief_json=ch_belief_json,
                    opponent_belief_json=tg_belief_json,
                    max_questions=5
                )
                #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]
                stage_request = [Message(role="user", content=prompt)]

                # Get position and build critique prompt
                # target_position = target.get_internal_belief()
                # prompt = prompts.build_stage_2_prompt(target_position)
                # stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]

                # Get critique
                response = challenger.generate(stage_request, temperature=0.2)
                self.round_histories[self.current_round_key].append(response)

                # Parse individual challenges from response
                # parsed_challenges = parse_challenges(response.content)
                #for challenge in parsed_challenges:
                #    self.challenge_rebuttal_pairs.append({
                #        "challenger": challenger_name,
                #        "target": target_name,
                #        "challenge": challenge,
                #        "rebuttal": None,
                #        "resolution": None
                #    })

                # Parse the FIRST fenced JSON block -> {"questions":[...]}
                questions_obj = _extract_first_json_block(response.content)
                questions = (questions_obj or {}).get("questions", [])

                # Fallback to legacy parser if needed (keeps backward compat)
                if not questions:
                    parsed_challenges = parse_challenges(response.content)
                    for challenge in parsed_challenges:
                        self.challenge_rebuttal_pairs.append({
                            "challenger": challenger_name,
                            "target": target_name,
                            "challenge": challenge,
                            "rebuttal": None,
                            "resolution": None
                        })
                else:
                    # Store structured questions with rich metadata
                    for q in questions:
                        self.challenge_rebuttal_pairs.append({
                            "challenger": challenger_name,
                            "target": target_name,
                            "challenge": q.get("text", "").strip(),       # human-readable question
                            "qid": q.get("qid"),                           # Q1, Q2, ...
                            "target_ids": q.get("target_ids", []),         # ["C3","A1"]
                            "intent": q.get("intent"),                     # concession|test|clarification
                            "why_high_value": q.get("why_high_value"),     # 1-sentence rationale
                            "proposed_test": q.get("proposed_test"),       # small object, may be None
                            "rebuttal": None,
                            "resolution": None
                        })

                log = f"""[{challenger_name} → {target_name}] Generated {len(questions) if questions else len(parsed_challenges)} critique(s).
                [{challenger_name} - Critique of {target_name}]:
                {response.content}
                """
                print(log)
                self.full_transcript.append(log)

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
        log = "\n🛡️ Stage 3: Rebuttals"
        print(log)
        self.full_transcript.append(log)

        # Group all challenges targeting each agent
        grouped_entries = {}
        for entry in self.challenge_rebuttal_pairs:
            target = entry["target"]
            grouped_entries.setdefault(target, []).append(entry)

        self.last_rebuttals = {}
        self.last_rebuttals_patches = {}

        # Iterate through targets
        for target_agent in self.agents:
            target_name = target_agent.name
            relevant_entries = grouped_entries.get(target_name) # Grab the list of critiques that are aimed at the target agent
            # If there are no critiques of this agent, move to the next agent
            if not relevant_entries:
                continue

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

            # Include the target's CURRENT belief JSON
            tgt_belief_obj = target_agent.get_internal_belief_obj()
            tgt_belief_json = json.dumps(tgt_belief_obj, ensure_ascii=False, indent=2) if tgt_belief_obj else ""

            # Choose one representative opponent name (or "Various" in multi-agent debates)
            opponent_name = questions_payload["questions"][0]["from"] if questions_payload["questions"] else "Opponent"

            prompt = prompts.build_stage_3_structured_rebuttal_prompt(
                topic=self.topic if hasattr(self, "topic") else "<topic>",
                agent_name=target_name,
                opponent_name=opponent_name,
                received_questions_json=received_questions_json,
                agent_belief_json=tgt_belief_json,
                max_rebuttals=min(5, len(questions_payload["questions"]))
            )
            #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]
            stage_request = [Message(role="user", content=prompt)]

            # challenge_texts = [entry["challenge"] for entry in relevant_entries] # Creates a list of the challenges for this target agent
            # prompt = prompts.build_stage_3_structured_rebuttal_prompt(challenge_texts)
            # stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]

            response = target_agent.generate(stage_request, temperature=0.2)
            self.round_histories[self.current_round_key].append(response)

            # Parse JSON blocks: 1) rebuttals, 2) optional patches
            blocks = _extract_all_json_blocks(response.content)
            rebuttals_block = json.loads(blocks[0]) if blocks else {"rebuttals": []}
            patches_block = json.loads(blocks[1]) if len(blocks) > 1 else {"patches": []}

            rebuttals = rebuttals_block.get("rebuttals", [])
            patches = patches_block.get("patches", [])

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

            # rebuttals = parse_structured_rebuttals_numbered(response.content) # Creates a list of the rebuttals from this target agents response

            # Assign rebuttals to entries
            #for entry, rebuttal_text in zip(relevant_entries, rebuttals):
            #    entry["rebuttal"] = rebuttal_text.strip()

            log = f"""[{target_name}] returned {len(rebuttals)} rebuttals, {len(patches)} patch ops.
            [{target_name} - Rebuttal]:
            {response.content}
            """
            print(log)
            self.full_transcript.append(log)

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
        log = "\n⚖️ Stage 4: Rigorous Conflict Resolution"
        print(log)
        self.full_transcript.append(log)

        for entry in self.challenge_rebuttal_pairs:
            # Skip if missing key elements
            if not entry["challenge"] or not entry["rebuttal"]:
                log = f"⚠️ Incomplete pair: skipping {entry['challenger']} → {entry['target']}"
                print(log)
                self.full_transcript.append(log)
                continue

            # Run logic enforcement pipeline
            resolution = self.adjudicator.run(
                challenge=entry["challenge"],
                rebuttal=entry["rebuttal"],
                challenger=entry["challenger"],
                target=entry["target"]
            )

            # Save structured result in the entry
            entry["resolution"] = resolution

            # Update agent stats
            self.agent_stats = update_agent_stats(self.agent_stats, entry)

            # Display outcome
            log = f"""🧮 [{entry['challenger']} → {entry['target']}] Resolution: {resolution['status'].upper()}\nReasoning: {resolution['reasoning']}\n"""
            print(log)
            self.full_transcript.append(log)

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
        log = "\n🔄 Stage 5: Updating Internal Beliefs"
        print(log)
        self.full_transcript.append(log)

        # Group all adjudicated results by target agent
        grouped_by_target = {}
        for entry in self.challenge_rebuttal_pairs:
            target = entry["target"]
            grouped_by_target.setdefault(target, []).append(entry)

        # Ask each agent to revise their view
        for agent in self.agents:
            name = agent.name
            relevant_entries = grouped_by_target.get(name, [])

            if not relevant_entries:
                self.current_positions[name] = self.opening_positions.get(name, "[No change]")
                continue

            # Build prompt
            prior_json = agent.get_internal_belief_obj()
            if prior_json is not None:
                # Use CBS patch/update builder
                prompt = prompts.build_stage_5_belief_update_prompt_cbsv1(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,  # whatever you collect from adjudication
                    prior_belief_json=json.dumps(prior_json, ensure_ascii=False, indent=2)
                )
            else:
                # Legacy builder (unchanged)
                prompt = prompts.build_stage_5_belief_update_prompt(
                    agent_name=agent.name,
                    challenge_rebuttal_pairs=relevant_entries,
                    original_position=agent.get_internal_belief()
                )
            #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]
            stage_request = [Message(role="user", content=prompt)]
            response = agent.generate(stage_request, temperature=0.2)
            self.round_histories[self.current_round_key].append(response)

            # Try to parse an UPDATED CBS object from the response
            belief_obj, md_view, errs = parse_model_output_to_belief(response.content)
            if belief_obj and not errs:
                agent.set_internal_belief_obj(belief_obj)
                agent.set_internal_belief(md_view or belief_to_markdown(belief_obj))
                agent.all_beliefs_held.append(json.dumps(belief_obj, ensure_ascii=False, indent=2) if belief_obj else "") # track beliefs
            else:
                agent.set_internal_belief(response.content.strip())
                agent.all_beliefs_held.append(response.content.strip()) # track beliefs

            #log = f"[{name} - Updated Position]:\n{response.content}\n"
            log = f"[{name} - Updated Position]:\n{md_view}\n"
            print(log)
            self.full_transcript.append(log)

        return


    def run_stage_6_concluding_remarks(self) -> dict:
        """
        Stage 6: Concluding Remarks

        Each agent reflects on the outcome of the debate, what they learned,
        what they still believe, and any changes in their stance.

        Returns:
            dict[str, str]: A mapping from agent names to their final concluding remarks.
        """
        log = "\n🎤 Stage 6: Concluding Remarks"
        print(log)
        self.full_transcript.append(log)

        self.conclusions = {}

        # Pre-serialize the adjudicated pairs for context
        cr_pairs_json = json.dumps(self.challenge_rebuttal_pairs, ensure_ascii=False, indent=2)

        for agent in self.agents:
            name = agent.name
            # # Select an opponent label for readability
            # others = [a for a in self.agents if a.name != name]
            # opponent_name = others[0].name if others else "Opponent"

            # Beliefs as JSON strings if available
            a_belief_obj = agent.get_internal_belief_obj()
            a_belief_json = json.dumps(a_belief_obj, ensure_ascii=False, indent=2) if a_belief_obj else ""
            # o_belief_obj = others[0].get_internal_belief_obj() if others else None
            # o_belief_json = json.dumps(o_belief_obj, ensure_ascii=False, indent=2) if o_belief_obj else ""

            prompt = prompts.build_stage_6_conclusion_prompt(
                topic=self.topic if hasattr(self, "topic") else "<topic>",
                agent_name=name,
                #opponent_name=opponent_name,
                agent_belief_json=a_belief_json,
                all_past_beliefs=agent.all_beliefs_held
                #opponent_belief_json=o_belief_json,
                #challenge_rebuttal_pairs_json=cr_pairs_json
            )
            #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]
            stage_request = [Message(role="user", content=prompt)]

            response = agent.generate(stage_request, temperature=0.2)
            self.round_histories[self.current_round_key].append(response)

            # Try to parse {"conclusion": {...}} for structured logging
            concl_obj = _extract_first_json_block(response.content)
            if concl_obj and "conclusion" in concl_obj:
                self.conclusions[name] = concl_obj["conclusion"]
                # Also append the human Markdown that follows for transcript readability
                log = f"[{name} – Concluding Remarks (structured)]:\n{response.content}\n"
            else:
                # Fallback: keep raw text
                self.conclusions[name] = response.content.strip()
                log = f"[{name} – Concluding Remarks]:\n{response.content}\n"

            print(log)
            self.full_transcript.append(log)

        return self.conclusions


    # def run_stage_7_scribe_synthesis(self, scribe_agent) -> str:
    #     """
    #     Stage 7: Chunked Scribe Synthesis

    #     Breaks the full transcript into ~5000-token chunks and synthesizes each sequentially.
    #     Maintains narrative coherence by passing prior outputs as context into later chunks.

    #     Args:
    #         scribe_agent (Agent): A specialized summarizer or narrator agent.

    #     Returns:
    #         str: The final, flowing narrative essay based on the entire debate.
    #     """
    #     from chal.agents import prompts

    #     print("\n📜 Stage 7: Synthesizing Debate")

    #     # --- Step 1: Tokenize and Chunk the Transcript ---
    #     transcript_text = "\n\n".join(self.full_transcript)
    #     self.full_transcript = transcript_text
    #     chunks = chunk_transcript_by_tokens(transcript_text, max_tokens=12000)

    #     essay_parts = []
    #     previous_output = ""

    #     for idx, chunk in enumerate(chunks):
    #         if idx == 0:
    #             prompt = prompts.build_stage_7_scribe_prompt(chunk)
    #         else:
    #             prompt = f"""
    #                     You are continuing a long-form philosophical essay based on a multi-agent debate. The previous portion ended as follows:

    #                     \"\"\"{previous_output[-1000:]}\"

    #                     Now continue the narrative by integrating the following transcript excerpt:
    #                     \"\"\"{chunk}\"\"\"

    #                     Continue in the same voice and structure. Do not summarize — this is a continuous exposition. Keep the tone analytical and elegant.
    #                     """.strip()

    #         print(f"\n🧩 Scribing chunk {idx+1} of {len(chunks)} ({len(chunk.split())} words)...")
    #         stage_request = [Message(role="user", content=prompt)]
    #         response = scribe_agent.generate(stage_request)

    #         response_text = response.content.strip()
    #         essay_parts.append(response_text)
    #         previous_output = response_text

    #     # --- Step 3: Stitch Together the Full Essay ---
    #     full_essay = "\n\n".join(essay_parts)
    #     self.final_synthesis = full_essay

    #     print("🧾 Final Synthesis Generated.\n")
    #     return self.final_synthesis, self.full_transcript


    def run_stage_7_scribing(self, scribe_agent=None, max_chars_per_chunk: int = 15000, overlap_chars: int = 1000):
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
        scribe = scribe_agent
        if scribe is None:
            raise RuntimeError("No agent available for Stage 7 scribing.")

        # 1) Gather source transcript
        full_text = "\n".join(str(x) for x in self.full_transcript)

        # 2) Chunk it
        chunks = _chunk_text_by_chars(full_text, max_chars=max_chars_per_chunk, overlap=overlap_chars)

        # 3) Map over chunks, maintaining a continuity state
        continuity_state = {}  # grows over chunks
        narrative_slices = []
        agent_names = [a.name for a in self.agents]
        for idx, ch in enumerate(chunks, start=1):
            prompt = prompts.build_stage_7_scribe_prompt_map(
                topic=getattr(self, "topic", "<topic>"),
                agent_names=agent_names,
                transcript_chunk=ch,
                continuity_state_json=json.dumps(continuity_state, ensure_ascii=False, indent=2)
            )
            #stage_request = self.round_histories[self.current_round_key] + [Message(role="user", content=prompt)]
            stage_request = [Message(role="user", content=prompt)]
            resp = scribe.generate(stage_request, temperature=0.3)
            self.round_histories[self.current_round_key].append(resp)

            # Parse continuity JSON + narrative slice
            first_json = _extract_first_json_block(resp.content) or {}
            cont_update = (first_json.get("continuity_update") or {})
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

            # Optional debug
            print(f"[DEBUG] Stage 7 map chunk {idx}/{len(chunks)}: "
                f"{len(narrative_md)} chars; continuity keys={list(continuity_state.keys())}")

        # 4) Reduce: stitch all slices into a single cohesive narrative
        reduce_prompt = prompts.build_stage_7_scribe_prompt_reduce(
            topic=getattr(self, "topic", "<topic>"),
            agent_names=agent_names,
            all_narrative_slices_markdown=narrative_slices,
            final_continuity_state_json=json.dumps(continuity_state, ensure_ascii=False, indent=2)
        )
        #reduce_request = self.round_histories[self.current_round_key] + [Message(role="user", content=reduce_prompt)]
        reduce_request = [Message(role="user", content=reduce_prompt)]
        reduce_resp = scribe.generate(reduce_request, temperature=0.3)
        self.round_histories[self.current_round_key].append(reduce_resp)

        final_md = _extract_first_markdown_block(reduce_resp.content) or reduce_resp.content.strip()
        self.scribed_transcript_markdown = final_md

        # Log & append to master transcript
        log = "\n📝 Stage 7: Scribed Narrative\n" + final_md + "\n"
        print(log)
        self.full_transcript.append(log)

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

            # Track the final beliefs of the agents
            for agent in self.agents:
                belief_obj = agent.get_internal_belief_obj()
                if belief_obj is not None:
                    text_for_embedding = project_for_embedding(belief_obj)
                else:
                    text_for_embedding = agent.get_internal_belief()  # legacy fallback

                embedding_tracker.embed_belief(agent.name, text_for_embedding)
            # Save the embeddings
            embedding_tracker.save_embeddings("src/chal/storage/embeddings.npz")

            # Stage 6: Concluding Reflections
            self.run_stage_6_concluding_remarks()

            # Print agent stats
            display_agent_stats(self.agent_stats)

            # Stage 7: Scribe Summary
            # Instantiate Scribe with transcript
            #self.run_stage_7_scribe_synthesis(self.scribe_agent)
            self.final_synthesis = self.run_stage_7_scribing(self.scribe_agent)

            print("\n✅ Debate Complete.")

            return {
                "initial_positions": self.opening_positions,
                "final_positions": [agent.internal_belief for agent in self.agents],
                "conclusions": self.conclusions,
                "synthesis": self.final_synthesis,
                "full_transcript": "\n".join(self.full_transcript),
                "agent_stats": self.agent_stats
            }



# --- Helper Functions ---
def _extract_first_json_block(text: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    return json.loads(m.group(1)) if m else None

def _extract_all_json_blocks(text: str):
    return re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)

def _extract_first_markdown_block(text: str):
    m = re.search(r"```markdown\s*(.*?)\s*```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def _key_for_dedupe(x):
    """
    Make a stable, hashable key for heterogeneous items.
    - dict/list -> JSON string with sorted keys
    - other     -> str(x)
    """
    if isinstance(x, (dict, list)):
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    return str(x)

def _dedupe_extend(base_list, new_items):
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