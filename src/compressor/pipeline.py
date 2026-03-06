"""
Context Compression Pipeline — the main user-facing API.

Usage::

    from src.compressor import ContextCompressor

    compressor = ContextCompressor(max_tokens=4000)
    compressed = compressor.compress(document=long_text, query="What are the key findings?")
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from .allocation import (
    BudgetSplit,
    HierarchicalCompressor,
    TokenBudgetAllocator,
)
from .chunking import Chunk, ChunkingEngine
from .embeddings import EmbeddingGenerator
from .llm import LLMClient
from .scoring import RelevanceScorer, ScoringWeights


@dataclass
class CompressionResult:
    """Returned by :meth:`ContextCompressor.compress`."""

    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    reduction_pct: float
    num_chunks: int
    tier_counts: dict[str, int]  # {"keep": N, "summarize": M, "drop": K}


class ContextCompressor:
    """End-to-end adaptive context compression pipeline.

    Parameters
    ----------
    max_tokens : int
        Hard token budget for the compressed output.
    chunk_size : int
        Soft max tokens per chunk.
    chunk_overlap : int
        Overlap tokens between consecutive chunks.
    embedding_model : str
        SentenceTransformer model name.
    budget_split : BudgetSplit | None
        Override the default 60/30/10 budget split.
    scoring_weights : ScoringWeights | None
        Override the default relevance score weights.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "all-MiniLM-L6-v2",
        budget_split: BudgetSplit | None = None,
        scoring_weights: ScoringWeights | None = None,
        groq_api_key: str | None = None,
    ) -> None:
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.chunker = ChunkingEngine(
            max_chunk_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.scorer = RelevanceScorer(weights=scoring_weights)

        # LLM for smart summarisation (optional — falls back to extractive)
        llm = LLMClient(api_key=groq_api_key) if groq_api_key else None

        self.allocator = TokenBudgetAllocator(
            max_tokens=max_tokens, split=budget_split
        )
        self.compressor = HierarchicalCompressor(
            allocator=self.allocator, llm=llm
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def compress(self, document: str, query: str) -> CompressionResult:
        """Run the full pipeline and return a :class:`CompressionResult`."""
        original_tokens = len(self.enc.encode(document, disallowed_special=()))

        # 1. Chunk
        chunks: list[Chunk] = self.chunker.chunk(document)

        # 2. Embed
        query_emb = self.embedder.embed(query)
        chunk_texts = [c.text for c in chunks]
        chunk_embs = self.embedder.embed(chunk_texts)

        # 3. Score
        self.scorer.score(chunks, query, query_emb, chunk_embs)

        # 4. Compress
        compressed_text = self.compressor.compress(chunks, document)
        compressed_tokens = len(
            self.enc.encode(compressed_text, disallowed_special=())
        )

        # 5. Stats
        tier_counts = {"keep": 0, "summarize": 0, "drop": 0}
        for c in chunks:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1

        reduction_pct = (
            (1 - compressed_tokens / original_tokens) * 100
            if original_tokens
            else 0.0
        )

        return CompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            reduction_pct=reduction_pct,
            num_chunks=len(chunks),
            tier_counts=tier_counts,
        )
