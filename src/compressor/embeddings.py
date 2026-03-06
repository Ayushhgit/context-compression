"""
Embedding Generator — produces dense vector representations.

Uses `sentence-transformers` with the lightweight *all-MiniLM-L6-v2* model
so everything runs locally without API keys.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Thin wrapper around a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``all-MiniLM-L6-v2``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Return L2-normalised embeddings of shape ``(n, dim)``."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings: np.ndarray = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings
