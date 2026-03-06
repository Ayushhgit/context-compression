# Adaptive Context Compression (V1)

A smart preprocessing layer for LLMs that **compresses long documents** against a query, fitting them into a strict token budget while preserving task-relevant information.

> **Not** naive summarization — it's **query-aware, tier-based, budget-controlled** compression.

## Architecture

```
Long Document
      ↓
Chunking Engine          (paragraph-aware, sliding-window overlap)
      ↓
Embedding Generator      (sentence-transformers: all-MiniLM-L6-v2)
      ↓
Query Similarity Scorer  (cosine + keyword overlap + length norm)
      ↓
Relevance Ranking
      ↓
Hierarchical Compression
  ├── Tier 1: Keep top chunks verbatim        (60% budget)
  ├── Tier 2: Summarise mid chunks via LLM    (30% budget)
  └── Tier 3: Drop lowest relevance           (10% → global overview)
      ↓
Token Budget Allocator
      ↓
Compressed Context → LLM
```

## Quick Start

```bash
# Install
uv sync

# Set your Groq API key
export GROQ_API_KEY="your-key-here"       # Linux/Mac
$env:GROQ_API_KEY="your-key-here"         # PowerShell

# Run evaluation
uv run python scripts/evaluate.py
```

## Usage

```python
from src.compressor import ContextCompressor

compressor = ContextCompressor(
    max_tokens=4000,
    groq_api_key="your-groq-key",   # optional — falls back to extractive
)
result = compressor.compress(
    document=long_text,
    query="What are the key findings?",
)
print(result.compressed_text)
print(f"Reduced {result.reduction_pct:.1f}% tokens")
```

## Example Results

| Metric | Value |
|--------|-------|
| Original tokens | 1,552 |
| Compressed tokens | 897 |
| **Token reduction** | **42.2%** |
| Tier distribution | keep=1, summarize=1, drop=2 |

## Tech Stack

- **Chunking**: `tiktoken` (cl100k_base)
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2, local)
- **LLM Summarisation**: Groq (`llama-3.3-70b-versatile`)
- **Scoring**: Cosine similarity + keyword overlap + length normalisation

## Project Structure

```
src/compressor/
├── chunking.py      # Paragraph-aware chunking engine
├── embeddings.py    # SentenceTransformer wrapper
├── scoring.py       # Multi-signal relevance scorer
├── allocation.py    # Budget allocator & hierarchical compressor
├── llm.py           # Groq API wrapper
└── pipeline.py      # ContextCompressor (main API)

scripts/
└── evaluate.py      # End-to-end benchmark
```
