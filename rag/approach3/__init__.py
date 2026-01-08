"""Approach 3 modules (two-stage dense recall -> reranking).

This package is intentionally structured so that heavy ML dependencies are optional:
- Builder utilities can be run to create cached assets (corpus, embeddings, ANN index).
- Retrieval/reranking modules should only import heavy libraries when explicitly used.
"""

