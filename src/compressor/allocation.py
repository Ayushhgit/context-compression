"""
Token Budget Allocator & Hierarchical Compressor.

Distributes a global token budget across three tiers and produces the
final compressed context string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

from .chunking import Chunk

if TYPE_CHECKING:
    from .llm import LLMClient


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #


@dataclass
class BudgetSplit:
    """Fraction of the total budget for each tier."""

    tier1_keep: float = 0.60       # verbatim top chunks
    tier2_summarize: float = 0.30  # summarised mid chunks
    tier3_global: float = 0.10     # global document summary

    def __post_init__(self) -> None:
        total = self.tier1_keep + self.tier2_summarize + self.tier3_global
        assert abs(total - 1.0) < 1e-6, f"Budget fractions must sum to 1.0 (got {total})"


# ------------------------------------------------------------------ #
# Budget allocator                                                    #
# ------------------------------------------------------------------ #


class TokenBudgetAllocator:
    """Decide how many tokens go to each tier.

    Parameters
    ----------
    max_tokens : int
        Hard upper-bound on the final compressed context.
    split : BudgetSplit | None
        Override the default 60/30/10 split.
    """

    def __init__(
        self, max_tokens: int = 4000, split: BudgetSplit | None = None
    ) -> None:
        self.max_tokens = max_tokens
        self.split = split or BudgetSplit()

    @property
    def tier1_budget(self) -> int:
        return int(self.max_tokens * self.split.tier1_keep)

    @property
    def tier2_budget(self) -> int:
        return int(self.max_tokens * self.split.tier2_summarize)

    @property
    def tier3_budget(self) -> int:
        return int(self.max_tokens * self.split.tier3_global)


# ------------------------------------------------------------------ #
# Hierarchical compressor                                             #
# ------------------------------------------------------------------ #


class HierarchicalCompressor:
    """Assign tiers to ranked chunks and build the compressed context.

    Parameters
    ----------
    allocator : TokenBudgetAllocator
        Provides per-tier token budgets.
    tier1_fraction : float
        Fraction (0–1) of chunks assigned to Tier 1.
    tier2_fraction : float
        Fraction (0–1) of chunks assigned to Tier 2.
    model_name : str
        Tiktoken encoding used for token counting.
    """

    def __init__(
        self,
        allocator: TokenBudgetAllocator,
        tier1_fraction: float = 0.30,
        tier2_fraction: float = 0.40,
        model_name: str = "cl100k_base",
        llm: LLMClient | None = None,
    ) -> None:
        self.allocator = allocator
        self.tier1_frac = tier1_fraction
        self.tier2_frac = tier2_fraction
        self.enc = tiktoken.get_encoding(model_name)
        self.llm = llm

    # ---- public ----

    def compress(
        self,
        chunks: list[Chunk],
        document: str,
    ) -> str:
        """Return the compressed context string within the token budget."""
        ranked = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        n = len(ranked)
        n_tier1 = max(1, int(n * self.tier1_frac))
        n_tier2 = max(1, int(n * self.tier2_frac))

        tier1 = ranked[:n_tier1]
        tier2 = ranked[n_tier1 : n_tier1 + n_tier2]
        # rest is implicitly tier3 (dropped)

        for c in tier1:
            c.tier = "keep"
        for c in tier2:
            c.tier = "summarize"

        # ---- build context ----
        parts: list[str] = []
        used_tokens = 0

        # Tier 1 — verbatim
        budget1 = self.allocator.tier1_budget
        for c in sorted(tier1, key=lambda c: c.index):
            if used_tokens + c.token_count > budget1:
                break
            parts.append(c.text)
            used_tokens += c.token_count

        # Tier 2 — LLM-powered or extractive summary
        budget2 = self.allocator.tier2_budget
        tier2_sorted = sorted(tier2, key=lambda c: c.index)
        if self.llm and tier2_sorted:
            tier2_texts = [c.text for c in tier2_sorted]
            summary = self.llm.summarize_chunks(tier2_texts, max_tokens=budget2)
            parts.append("\n[Summarised sections]\n" + summary)
            used_tokens += len(self.enc.encode(summary, disallowed_special=()))
        else:
            summary_parts: list[str] = []
            tier2_used = 0
            for c in tier2_sorted:
                first_sent = self._first_sentence(c.text)
                tok_count = len(self.enc.encode(first_sent, disallowed_special=()))
                if tier2_used + tok_count > budget2:
                    break
                summary_parts.append(first_sent)
                tier2_used += tok_count
            if summary_parts:
                parts.append("\n[Summarised sections]\n" + "\n".join(summary_parts))
                used_tokens += tier2_used

        # Tier 3 — global document overview
        budget3 = self.allocator.tier3_budget
        if self.llm:
            global_summary = self.llm.global_summary(document, max_tokens=budget3)
        else:
            global_summary = self._global_summary(document, budget3)
        if global_summary:
            parts.append("\n[Document overview]\n" + global_summary)

        return "\n\n".join(parts)

    # ---- helpers ----

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Extract the first sentence (crude but fast)."""
        for end in (".", "!", "?"):
            idx = text.find(end)
            if idx != -1:
                return text[: idx + 1]
        # fallback: first 80 chars
        return text[:80]

    def _global_summary(self, document: str, budget: int) -> str:
        """Create a very short extractive overview (first N tokens)."""
        tokens = self.enc.encode(document[:2000], disallowed_special=())
        trimmed = tokens[:budget]
        return self.enc.decode(trimmed)
