"""
Relevance Scorer — compute a weighted relevance score for each chunk.

The final score blends:
  • cosine similarity (semantic)
  • keyword overlap  (lexical)
  • chunk-length normalisation
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class ScoringWeights:
    """Relative weights for the composite score (will be normalised)."""

    semantic: float = 0.70
    keyword: float = 0.20
    length_norm: float = 0.10


class RelevanceScorer:
    """Score each chunk's relevance to a query.

    Parameters
    ----------
    weights : ScoringWeights | None
        Override the default score blend.
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.w = weights or ScoringWeights()
        # normalise so they always sum to 1
        total = self.w.semantic + self.w.keyword + self.w.length_norm
        self.w.semantic /= total
        self.w.keyword /= total
        self.w.length_norm /= total

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def score(
        self,
        chunks: list[Chunk],
        query: str,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
    ) -> list[Chunk]:
        """Mutate *chunks* in-place, setting ``relevance_score``."""
        query_tokens = self._tokenize(query)

        for i, chunk in enumerate(chunks):
            sem = self._cosine(query_embedding, chunk_embeddings[i])
            kw = self._keyword_overlap(query_tokens, chunk.text)
            ln = self._length_score(chunk.token_count)

            chunk.relevance_score = (
                self.w.semantic * sem
                + self.w.keyword * kw
                + self.w.length_norm * ln
            )

        return chunks

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    @classmethod
    def _keyword_overlap(cls, query_tokens: set[str], chunk_text: str) -> float:
        chunk_tokens = cls._tokenize(chunk_text)
        if not query_tokens:
            return 0.0
        return len(query_tokens & chunk_tokens) / len(query_tokens)

    @staticmethod
    def _length_score(token_count: int, ideal: int = 256) -> float:
        """Prefer medium-length chunks — too short often means noise."""
        return 1.0 - min(abs(token_count - ideal) / ideal, 1.0)
