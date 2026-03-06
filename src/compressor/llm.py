"""
LLM Client — wraps the Groq API for text summarization tasks.
"""

from __future__ import annotations

import os

from groq import Groq


class LLMClient:
    """Thin wrapper around Groq for summarisation tasks.

    Parameters
    ----------
    api_key : str | None
        Groq API key.  Falls back to ``GROQ_API_KEY`` env var.
    model : str
        Model identifier on Groq (default: ``llama-3.3-70b-versatile``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

    def summarize_chunks(self, chunks_text: list[str], max_tokens: int = 300) -> str:
        """Summarise a list of text chunks into a single compressed paragraph."""
        combined = "\n---\n".join(chunks_text)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise summariser. Condense the following "
                        "text sections into a single, dense paragraph that "
                        "preserves all key facts. No preamble."
                    ),
                },
                {"role": "user", "content": combined},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def global_summary(self, document: str, max_tokens: int = 150) -> str:
        """Generate a short 'document essence' overview."""
        # Send first ~2000 chars to stay fast
        snippet = document[:3000]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Produce a one-paragraph overview of this document's "
                        "topic, scope, and key findings. Be maximally concise."
                    ),
                },
                {"role": "user", "content": snippet},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
