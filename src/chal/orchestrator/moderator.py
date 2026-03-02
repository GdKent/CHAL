"""
moderator.py

Defines the Moderator class for generating debate roadmaps.

The moderator analyzes a debate topic and produces an ordered roadmap of
sub-topics, each assigned to a debate round. This gives the debate structured
progression instead of unguided disagreement discovery.

Two modes:
  - static:   Roadmap is generated once before the debate begins (default).
  - adaptive: Roadmap can be revised between rounds based on progress (future).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chal.agents.base import Agent, Message
from chal.agents.factory import create_agent
from chal.agents import prompts
from chal.config import ModeratorConfig


@dataclass
class SubTopic:
    """A single sub-topic in the debate roadmap."""
    title: str
    description: str
    rationale: str = ""
    guiding_questions: List[str] = field(default_factory=list)


@dataclass
class Roadmap:
    """An ordered roadmap of sub-topics for a moderated debate."""
    sub_topics: List[SubTopic]
    overall_rationale: str = ""
    sufficiency_note: str = ""
    raw_response: str = ""


@dataclass
class RoadmapRevision:
    """A revision to the roadmap produced by the adaptive moderator (future)."""
    revised_sub_topics: List[SubTopic]
    revision_rationale: str = ""


class Moderator:
    """
    Generates and manages a debate roadmap of ordered sub-topics.

    Follows the same pattern as the Adjudicator: a lightweight wrapper
    around an LLM agent that makes targeted API calls, rather than a
    full debate participant.
    """

    def __init__(self, config: ModeratorConfig) -> None:
        """
        Initialize the Moderator with its configuration.

        Args:
            config: ModeratorConfig with model, provider, temperature, etc.
        """
        self.config = config
        self.agent: Agent = create_agent(
            name="Moderator",
            model=config.model,
            provider=config.provider,
            system_prompt="You are a debate moderator and topic analyst.",
        )
        self.roadmap: Optional[Roadmap] = None
        self._revision_count: int = 0
        self._topic: str = ""

    def generate_roadmap(
        self,
        topic: str,
        num_rounds: int,
        agent_personas: List[str],
    ) -> Roadmap:
        """
        Generate an ordered roadmap of sub-topics for the debate.

        Args:
            topic: The central debate topic/question.
            num_rounds: Number of debate rounds available.
            agent_personas: List of persona labels (e.g., ["EMPIRICIST", "SUPERNATURALIST"]).

        Returns:
            Roadmap: The generated roadmap with sub-topics.
        """
        self._topic = topic

        prompt = prompts.build_moderator_roadmap_prompt(
            topic=topic,
            num_rounds=num_rounds,
            agent_personas=agent_personas,
            context=self.config.context,
        )

        response = self.agent.generate(
            [Message(role="user", content=prompt)],
            temperature=self.config.temperature,
        )
        raw_response = response.content

        roadmap = self._parse_roadmap_response(raw_response, num_rounds)
        roadmap.raw_response = raw_response
        self.roadmap = roadmap
        return roadmap

    def review_round(
        self,
        round_num: int,
        round_summary: Dict[str, Any],
    ) -> Optional[RoadmapRevision]:
        """
        Review a completed round and optionally revise the remaining roadmap.

        Static mode: Returns None (no-op).
        Adaptive mode: Analyzes round results and may propose revisions to
        the remaining roadmap, subject to frequency, revision-count, and
        constraint settings.

        Args:
            round_num: The 1-based round that just completed.
            round_summary: Summary data from the completed round.

        Returns:
            Optional[RoadmapRevision]: None for static mode or if no revision
            is needed; a RoadmapRevision otherwise.
        """
        if self.config.moderator_mode != "adaptive":
            return None

        # Check review frequency
        if round_num % self.config.review_frequency != 0:
            return None

        # Check max revisions
        if (self.config.max_revisions >= 0
                and self._revision_count >= self.config.max_revisions):
            return None

        # Build remaining sub-topics list (everything after current round)
        if not self.roadmap:
            return None
        remaining = self.roadmap.sub_topics[round_num:]  # round_num is 1-based
        if not remaining:
            return None  # No remaining topics to revise

        remaining_dicts = [
            {"title": st.title, "description": st.description,
             "rationale": st.rationale, "guiding_questions": st.guiding_questions}
            for st in remaining
        ]

        # Build and send prompt
        prompt = prompts.build_moderator_review_round_prompt(
            topic=self._topic,
            round_num=round_num,
            round_summary=round_summary,
            remaining_sub_topics=remaining_dicts,
            constraints={
                "allow_reorder": self.config.allow_reorder,
                "allow_add_topics": self.config.allow_add_topics,
                "allow_remove_topics": self.config.allow_remove_topics,
            },
        )

        response = self.agent.generate(
            [Message(role="user", content=prompt)],
            temperature=self.config.temperature,
        )
        raw_response = response.content

        revision = self._parse_revision_response(raw_response, remaining)
        if revision is not None:
            self._revision_count += 1
        return revision

    def get_subtopic_for_round(self, round_index: int) -> Optional[SubTopic]:
        """
        Get the sub-topic assigned to a specific round.

        Args:
            round_index: Zero-based round index.

        Returns:
            Optional[SubTopic]: The sub-topic, or None if out of range.
        """
        if self.roadmap and 0 <= round_index < len(self.roadmap.sub_topics):
            return self.roadmap.sub_topics[round_index]
        return None

    def _parse_roadmap_response(self, raw_response: str, num_rounds: int) -> Roadmap:
        """
        Parse the LLM response into a Roadmap object.

        Extracts JSON from fenced code blocks, falls back to the full
        response if no fenced block is found.

        Args:
            raw_response: Raw text from the LLM.
            num_rounds: Expected number of sub-topics (for truncation).

        Returns:
            Roadmap: Parsed roadmap, truncated to num_rounds.
        """
        # Try to extract JSON from fenced code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object in the response
            json_match = re.search(r'\{.*\}', raw_response, flags=re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Return a minimal roadmap with the topic as the only sub-topic
            return Roadmap(
                sub_topics=[SubTopic(
                    title="General Discussion",
                    description="Open discussion of the debate topic.",
                )],
                overall_rationale="Failed to parse moderator response.",
                sufficiency_note="",
            )

        # Parse sub-topics
        sub_topics = []
        for item in data.get("sub_topics", []):
            sub_topics.append(SubTopic(
                title=item.get("title", "Untitled"),
                description=item.get("description", ""),
                rationale=item.get("rationale", ""),
                guiding_questions=item.get("guiding_questions", []),
            ))

        # Truncate to num_rounds
        sub_topics = sub_topics[:num_rounds]

        # Fallback if no sub-topics were parsed
        if not sub_topics:
            return Roadmap(
                sub_topics=[SubTopic(
                    title="General Discussion",
                    description="Open discussion of the debate topic.",
                )],
                overall_rationale="No sub-topics could be parsed from moderator response.",
                sufficiency_note="",
            )

        return Roadmap(
            sub_topics=sub_topics,
            overall_rationale=data.get("overall_rationale", ""),
            sufficiency_note=data.get("sufficiency_note", ""),
        )

    def _parse_revision_response(
        self,
        raw_response: str,
        current_remaining: List[SubTopic],
    ) -> Optional[RoadmapRevision]:
        """
        Parse the LLM's revision response.

        Returns None if the moderator says no revision is needed or if the
        response cannot be parsed. Returns a RoadmapRevision if changes
        are proposed.

        Args:
            raw_response: Raw text from the LLM.
            current_remaining: The current remaining sub-topics (for fallback).

        Returns:
            Optional[RoadmapRevision]: The parsed revision, or None.
        """
        # Extract JSON (same pattern as _parse_roadmap_response)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', raw_response, flags=re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None  # Can't parse → no revision

        if not data.get("revision_needed", False):
            return None

        revised = []
        for item in data.get("revised_sub_topics", []):
            revised.append(SubTopic(
                title=item.get("title", "Untitled"),
                description=item.get("description", ""),
                rationale=item.get("rationale", ""),
                guiding_questions=item.get("guiding_questions", []),
            ))

        if not revised:
            return None  # Empty revision = no change

        return RoadmapRevision(
            revised_sub_topics=revised,
            revision_rationale=data.get("revision_rationale", ""),
        )
