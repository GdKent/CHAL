# embedding_tracker.py

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from typing import Dict, List, Union
from pathlib import Path

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


    def embed_belief(self, belief, agent_name: str, round_num: int = 0) -> np.ndarray:
        """
        Converts a belief into an embedding and stores it under the agent's name.

        Args:
            belief: Either a CBS belief dict or a pre-projected text string.
            agent_name (str): Identifier for the agent.
            round_num (int): The round number (for tracking purposes).

        Returns:
            np.ndarray: The generated embedding vector.
        """
        if isinstance(belief, dict):
            from chal.beliefs.io import project_for_embedding
            belief_text = project_for_embedding(belief)
        else:
            belief_text = belief

        embedding = self.model.encode(belief_text, convert_to_numpy=True)
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


    def save_embeddings(self, filepath: Union[str, Path]):
        """
        Saves all embeddings to a .npz file for later use.

        Args:
            filepath: Where to save the embedding data.
        """
        np.savez_compressed(str(filepath), **{agent: np.stack(vectors) for agent, vectors in self.embeddings.items()})


    def load_embeddings(self, filepath: Union[str, Path]):
        """
        Loads embeddings from a saved .npz file.

        Args:
            filepath: Path to the saved embedding data.
        """
        data = np.load(str(filepath), allow_pickle=True)
        self.embeddings = {agent: list(data[agent]) for agent in data.files}
