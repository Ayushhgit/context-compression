"""
Chunking Engine — splits long documents into semantically meaningful chunks.

Uses paragraph/section boundaries with optional sliding-window overlap
so that no relevant context is lost at chunk edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    index: int
    token_count: int
    start_char: int
    end_char: int
    # filled later by the scorer
    relevance_score: float = 0.0
    tier: str = "drop"  # "keep" | "summarize" | "drop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTION_SPLITTER = re.compile(r"\n{2,}")  # two+ newlines = paragraph break


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChunkingEngine:
    """Break a document into token-bounded chunks along paragraph boundaries.

    Parameters
    ----------
    max_chunk_tokens : int
        Soft upper-bound on chunk size in tokens.
    overlap_tokens : int
        Number of tokens of overlap between consecutive chunks
        (achieved by repeating trailing sentences of the previous chunk).
    model_name : str
        Tiktoken model name used for the token counter.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        model_name: str = "cl100k_base",
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding(model_name)

    # ---- core ----

    def chunk(self, document: str) -> list[Chunk]:
        """Split *document* into a list of :class:`Chunk` objects."""
        paragraphs = _SECTION_SPLITTER.split(document.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[Chunk] = []
        current_text = ""
        current_start = 0
        char_offset = 0

        for para in paragraphs:
            candidate = (current_text + "\n\n" + para).strip() if current_text else para
            candidate_tokens = _count_tokens(candidate, self.enc)

            if candidate_tokens > self.max_chunk_tokens and current_text:
                # flush current chunk
                chunk = self._make_chunk(current_text, len(chunks), current_start)
                chunks.append(chunk)

                # start new chunk with overlap
                overlap_text = self._get_overlap(current_text)
                current_text = (overlap_text + "\n\n" + para).strip() if overlap_text else para
                current_start = char_offset
            else:
                current_text = candidate
                if not chunks:
                    current_start = 0

            char_offset += len(para) + 2  # account for \n\n separator

        # flush remaining
        if current_text.strip():
            chunks.append(self._make_chunk(current_text, len(chunks), current_start))

        return chunks

    # ---- internals ----

    def _make_chunk(self, text: str, index: int, start_char: int) -> Chunk:
        return Chunk(
            text=text,
            index=index,
            token_count=_count_tokens(text, self.enc),
            start_char=start_char,
            end_char=start_char + len(text),
        )

    def _get_overlap(self, text: str) -> str:
        """Return the last *overlap_tokens* tokens of *text* as a string."""
        if self.overlap_tokens <= 0:
            return ""
        tokens = self.enc.encode(text, disallowed_special=())
        overlap = tokens[-self.overlap_tokens :]
        return self.enc.decode(overlap)
