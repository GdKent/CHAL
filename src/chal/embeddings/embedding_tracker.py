# embedding_tracker.py

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from typing import Dict, List, Union
from pathlib import Path

# Upper bound used to normalize count features into [0, 1] range.
_COUNT_CAP = 20


class BeliefEmbeddingTracker:
    """
    Tracks the embeddings of each agent's internal beliefs over the course of a debate.

    Uses a sentence embedding model (e.g., Sentence-BERT) to convert textual internal beliefs
    into fixed-size vector embeddings. Stores embeddings for each round for later analysis.
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initializes the embedding model and the tracking structure.

        Args:
            model_name (str): The HuggingFace model name for sentence embeddings.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self.metadata: dict = {}
        self._model_dim: int = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _weighted_average_embedding(
        self, items: List[Dict[str, Union[str, float]]]
    ) -> np.ndarray:
        """Compute a strength-weighted average embedding for a list of nodes.

        Args:
            items: List of {"text": str, "strength": float} dicts.

        Returns:
            np.ndarray of shape (model_dim,).
            Zero vector if *items* is empty.
            Unweighted mean if all strengths are zero.
        """
        if not items:
            return np.zeros(self._model_dim, dtype=np.float32)

        texts = [it["text"] for it in items]
        strengths = np.array([it["strength"] for it in items], dtype=np.float32)
        vectors = self.model.encode(texts, convert_to_numpy=True)  # (N, dim)

        total_weight = strengths.sum()
        if total_weight == 0.0:
            return vectors.mean(axis=0)

        weighted = (vectors * strengths[:, np.newaxis]).sum(axis=0) / total_weight
        return weighted.astype(np.float32)

    def _simple_average_embedding(self, texts: List[str]) -> np.ndarray:
        """Compute an unweighted average embedding for a list of texts.

        Args:
            texts: List of strings to encode and average.

        Returns:
            np.ndarray of shape (model_dim,).
            Zero vector if *texts* is empty.
        """
        if not texts:
            return np.zeros(self._model_dim, dtype=np.float32)

        vectors = self.model.encode(texts, convert_to_numpy=True)  # (N, dim)
        return vectors.mean(axis=0).astype(np.float32)

    @staticmethod
    def _normalize_scalars(scalars: Dict[str, float]) -> np.ndarray:
        """Normalize the 11 scalar features into a consistent range.

        Counts are clamped to [0, _COUNT_CAP] then divided by _COUNT_CAP.
        Strengths are already in [0, 1] and passed through unchanged.

        Returns:
            np.ndarray of shape (11,).
        """
        count_keys = [
            "n_definitions", "n_assumptions", "n_evidence", "n_claims",
        ]
        strength_keys = [
            "avg_strength_definitions", "avg_strength_assumptions",
            "avg_strength_evidence", "avg_strength_claims",
        ]
        other_count_keys = ["n_counterpositions", "n_uncertainties"]
        thesis_key = "thesis_strength"

        values: List[float] = []
        for k in count_keys:
            values.append(min(scalars.get(k, 0), _COUNT_CAP) / _COUNT_CAP)
        for k in strength_keys:
            values.append(scalars.get(k, 0.0))
        for k in other_count_keys:
            values.append(min(scalars.get(k, 0), _COUNT_CAP) / _COUNT_CAP)
        values.append(scalars.get(thesis_key, 0.0))

        return np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------
    # Component-wise embedding
    # ------------------------------------------------------------------

    def embed_belief_components(
        self, belief: dict, agent_name: str, round_num: int = 0
    ) -> np.ndarray:
        """Produce a rich, component-wise embedding vector from a CBS belief dict.

        The vector is the concatenation of:
          1. definitions_vec      (strength-weighted avg, model_dim)
          2. assumptions_vec      (strength-weighted avg, model_dim)
          3. evidence_vec         (strength-weighted avg, model_dim)
          4. claims_vec           (strength-weighted avg, model_dim)
          5. thesis_vec           (single-text encoding, model_dim)
          6. uncertainties_vec    (simple avg, model_dim)
          7. counter_partial_vec  (simple avg, model_dim)
          8. counter_sufficient_vec (simple avg, model_dim)
          9. counter_unaddressed_vec (simple avg, model_dim)
          10. counter_moot_vec    (simple avg, model_dim)
          11. normalized_scalars  (11 values)

        Total dimension = 10 * model_dim + 11.

        Args:
            belief: CBS belief dict.
            agent_name: Identifier for the agent.
            round_num: The round number (for tracking purposes).

        Returns:
            np.ndarray: The concatenated embedding vector.
        """
        from chal.beliefs.io import project_for_component_embedding

        proj = project_for_component_embedding(belief)

        # Strength-weighted components
        definitions_vec = self._weighted_average_embedding(proj["definitions"])
        assumptions_vec = self._weighted_average_embedding(proj["assumptions"])
        evidence_vec = self._weighted_average_embedding(proj["evidence"])
        claims_vec = self._weighted_average_embedding(proj["claims"])

        # Thesis — single text
        thesis_text = proj["thesis_text"]
        if thesis_text:
            thesis_vec = self.model.encode(thesis_text, convert_to_numpy=True).astype(np.float32)
        else:
            thesis_vec = np.zeros(self._model_dim, dtype=np.float32)

        # Uncertainties — simple average
        uncertainties_vec = self._simple_average_embedding(proj["uncertainties"])

        # Counterpositions — simple average per sufficiency category
        counter_partial_vec = self._simple_average_embedding(proj["counterpositions"]["partial"])
        counter_sufficient_vec = self._simple_average_embedding(proj["counterpositions"]["sufficient"])
        counter_unaddressed_vec = self._simple_average_embedding(proj["counterpositions"]["unaddressed"])
        counter_moot_vec = self._simple_average_embedding(proj["counterpositions"]["moot"])

        # Scalar features
        normalized_scalars = self._normalize_scalars(proj["scalars"])

        # Concatenate everything
        full_vector = np.concatenate([
            definitions_vec,
            assumptions_vec,
            evidence_vec,
            claims_vec,
            thesis_vec,
            uncertainties_vec,
            counter_partial_vec,
            counter_sufficient_vec,
            counter_unaddressed_vec,
            counter_moot_vec,
            normalized_scalars,
        ])

        # Store
        if agent_name not in self.embeddings:
            self.embeddings[agent_name] = []
        self.embeddings[agent_name].append(full_vector)
        return full_vector

    # ------------------------------------------------------------------
    # Original public API
    # ------------------------------------------------------------------

    def embed_belief(self, belief, agent_name: str, round_num: int = 0) -> np.ndarray:
        """
        Converts a belief into an embedding and stores it under the agent's name.

        For dict inputs (CBS belief objects), delegates to embed_belief_components()
        to produce a rich component-wise vector. For pre-projected text strings,
        falls back to simple single-text encoding.

        Args:
            belief: Either a CBS belief dict or a pre-projected text string.
            agent_name (str): Identifier for the agent.
            round_num (int): The round number (for tracking purposes).

        Returns:
            np.ndarray: The generated embedding vector.
        """
        if isinstance(belief, dict):
            return self.embed_belief_components(belief, agent_name, round_num)

        # Legacy fallback: plain text string
        embedding = self.model.encode(belief, convert_to_numpy=True)
        if agent_name not in self.embeddings:
            self.embeddings[agent_name] = []
        self.embeddings[agent_name].append(embedding)
        return embedding


    def get_agent_trajectory(self, agent_name: str) -> List[np.ndarray]:
        """
        Returns all stored embeddings (over rounds) for a given agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            List[np.ndarray]: List of embeddings from each round.
        """
        return self.embeddings.get(agent_name, [])


    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Returns the full dictionary of all agent embeddings.

        Returns:
            Dict[str, List[np.ndarray]]
        """
        return self.embeddings


    def save_embeddings(self, filepath: Union[str, Path], agent_info=None, topic=None):
        """
        Saves all embeddings to a .npz file for later use.

        Args:
            filepath: Where to save the embedding data.
            agent_info: Optional dict mapping agent names to {"model": ..., "provider": ...}.
            topic: Optional debate topic string.
        """
        save_dict = {agent: np.stack(vectors) for agent, vectors in self.embeddings.items()}
        if agent_info or topic:
            save_dict["__metadata__"] = np.array({"agent_info": agent_info or {}, "topic": topic or ""})
        np.savez_compressed(str(filepath), **save_dict)


    def load_embeddings(self, filepath: Union[str, Path]):
        """
        Loads embeddings from a saved .npz file.

        Args:
            filepath: Path to the saved embedding data.
        """
        data = np.load(str(filepath), allow_pickle=True)
        self.metadata = {}
        self.embeddings = {}
        for name in data.files:
            if name == "__metadata__":
                self.metadata = data[name].item()
            else:
                self.embeddings[name] = list(data[name])
