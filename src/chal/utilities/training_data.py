"""
training_data.py

Mode-agnostic training data recorder and exporter for CHAL debates.

Captures structured debate events throughout the pipeline and exports them
as training data in two formats:
  1. Full DebateRecord (JSONL) - complete timeline with all events
  2. Belief training pairs (JSONL) - extracted input→belief target mappings
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class DebateRecorder:
    """
    Collects structured debate events for training data export.

    Instantiated once at the start of DebateController.run() and called
    at each stage to record events. Passive observer — does not affect
    debate logic.
    """

    def __init__(self, debate_config, agents: list, topic: str):
        """
        Initialize with debate metadata.

        Args:
            debate_config: DebateConfig instance.
            agents: List of Agent instances.
            topic: The debate topic string.
        """
        self.debate_id = str(uuid.uuid4())
        self.topic = topic
        self.config = debate_config
        self.current_round = 0
        self.timeline: list[dict[str, Any]] = []

        mode = "rebuttal"

        # Build metadata
        agent_meta = []
        for agent in agents:
            agent_meta.append({
                "agent_id": agent.name,
                "model": getattr(agent, 'model', 'unknown'),
                "provider": getattr(agent, 'provider', 'unknown'),
                "persona": getattr(agent, 'persona_label', 'unknown'),
                "temperature": getattr(agent, 'temperature', 0.7),
            })

        adj = debate_config.adjudication if debate_config else None
        adjudicator_meta = {
            "model": adj.model if adj else "unknown",
            "provider": adj.provider if adj else "unknown",
        } if adj else {}

        self.metadata = {
            "topic": topic,
            "mode": mode,
            "num_rounds": debate_config.max_rounds if debate_config else 1,
            "num_agents": len(agents),
            "agents": agent_meta,
            "adjudicator": adjudicator_meta,
            "config_snapshot": {
                "stage3_mode": mode,
                "max_rounds": debate_config.max_rounds if debate_config else 1,
            },
        }

    def set_round(self, round_num: int):
        """Update the current round number."""
        self.current_round = round_num

    def record_belief_formation(
        self,
        agent_id: str,
        inputs: dict,
        belief: Any,
        raw_response: str,
    ):
        """Record a Stage 1 belief formation event."""
        self.timeline.append({
            "type": "belief_formation",
            "round": self.current_round,
            "stage": 1,
            "agent_id": agent_id,
            "inputs": inputs,
            "outputs": {
                "belief": belief,
                "raw_response": raw_response,
            },
        })

    def record_cross_examination(
        self,
        agent_id: str,
        target_id: str,
        inputs: dict,
        challenges: list,
        raw_response: str,
    ):
        """Record a Stage 2 cross-examination event."""
        self.timeline.append({
            "type": "cross_examination",
            "round": self.current_round,
            "stage": 2,
            "agent_id": agent_id,
            "target_agent_id": target_id,
            "inputs": inputs,
            "outputs": {
                "challenges": challenges,
                "raw_response": raw_response,
            },
        })

    def record_rebuttal(
        self,
        agent_id: str,
        challenger_id: str,
        inputs: dict,
        rebuttals: list,
        raw_response: str,
    ):
        """Record a Stage 3 rebuttal event."""
        self.timeline.append({
            "type": "rebuttal",
            "round": self.current_round,
            "stage": 3,
            "agent_id": agent_id,
            "challenger_agent_id": challenger_id,
            "inputs": inputs,
            "outputs": {
                "rebuttals": rebuttals,
                "raw_response": raw_response,
            },
        })

    def record_adjudication(
        self,
        challenger_id: str,
        target_id: str,
        inputs: dict,
        verdict: str,
        reasoning: str,
        raw_response: str,
    ):
        """Record a Stage 4 adjudication event."""
        self.timeline.append({
            "type": "adjudication",
            "round": self.current_round,
            "stage": 4,
            "challenger_agent_id": challenger_id,
            "target_agent_id": target_id,
            "inputs": inputs,
            "outputs": {
                "verdict": verdict,
                "reasoning": reasoning,
                "raw_response": raw_response,
            },
        })

    def record_belief_update(
        self,
        agent_id: str,
        belief_before: Any,
        belief_after: Any,
        adjudication_results: list,
        patches: list,
        raw_response: str,
    ):
        """Record a Stage 5 belief update event."""
        self.timeline.append({
            "type": "belief_update",
            "round": self.current_round,
            "stage": 5,
            "agent_id": agent_id,
            "inputs": {
                "belief_before": belief_before,
                "adjudication_results": adjudication_results,
            },
            "outputs": {
                "belief_after": belief_after,
                "patches_applied": patches,
                "raw_response": raw_response,
            },
        })

    def record_event(self, event_type: str, data: dict[str, Any]):
        """Record a generic event in the timeline.

        Used for events that don't have a dedicated method.

        Args:
            event_type: The event type string.
            data: Arbitrary event data dict.
        """
        self.timeline.append({
            "type": event_type,
            "round": self.current_round,
            "data": data,
        })

    def get_debate_record(self) -> dict:
        """Return the full DebateRecord as a dict."""
        return {
            "debate_id": self.debate_id,
            "metadata": self.metadata,
            "timeline": self.timeline,
        }

    def export_jsonl(self, output_path: Path):
        """
        Export the full DebateRecord as a single JSON line in a JSONL file.

        Args:
            output_path: Path to the output .jsonl file.
        """
        record = self.get_debate_record()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def export_belief_training_pairs(self, output_path: Path):
        """
        Export belief-focused training pairs optimized for supervised fine-tuning.

        Extracts two pair types:
          - belief_formation: topic+persona → initial belief
          - belief_update: belief_before+adjudication → belief_after

        Args:
            output_path: Path to the output .jsonl file.
        """
        pairs = self._extract_belief_pairs()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'a', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    def _extract_belief_pairs(self) -> list[dict]:
        """Extract belief training pairs from the timeline."""
        pairs = []
        mode = self.metadata.get("mode", "rebuttal")

        # Build agent model lookup from metadata
        agent_models = {}
        for a in self.metadata.get("agents", []):  # type: ignore[union-attr]
            agent_models[a["agent_id"]] = a.get("model", "unknown")  # type: ignore[union-attr, index]

        for event in self.timeline:
            if event["type"] == "belief_formation":
                agent_id = event["agent_id"]
                inputs = event.get("inputs", {})
                belief = event.get("outputs", {}).get("belief")

                if belief is not None:
                    pairs.append({
                        "pair_type": "belief_formation",
                        "input": {
                            "topic": inputs.get("topic", self.topic),
                            "persona": inputs.get("persona", ""),
                            "system_prompt": inputs.get("system_prompt", ""),
                        },
                        "target": belief,
                        "metadata": {
                            "debate_id": self.debate_id,
                            "agent_id": agent_id,
                            "agent_model": agent_models.get(agent_id, "unknown"),
                            "round": event.get("round", 1),
                        },
                    })

            elif event["type"] == "belief_update":
                agent_id = event["agent_id"]
                inputs = event.get("inputs", {})
                outputs = event.get("outputs", {})

                belief_before = inputs.get("belief_before")
                belief_after = outputs.get("belief_after")

                if belief_before is not None and belief_after is not None:
                    pairs.append({
                        "pair_type": "belief_update",
                        "input": {
                            "belief_before": belief_before,
                            "adjudication_results": inputs.get("adjudication_results", []),
                            "debate_context": {
                                "mode": mode,
                                "round": event.get("round", 1),
                                "topic": self.topic,
                            },
                        },
                        "target": belief_after,
                        "metadata": {
                            "debate_id": self.debate_id,
                            "agent_id": agent_id,
                            "agent_model": agent_models.get(agent_id, "unknown"),
                            "round": event.get("round", 1),
                            "patches_applied": outputs.get("patches_applied", []),
                        },
                    })

        return pairs
