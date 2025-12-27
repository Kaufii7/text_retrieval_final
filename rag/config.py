"""Configuration placeholders for retrieval approaches.

This module intentionally stays minimal: it provides a structured place to keep
approach parameters without forcing a heavyweight config system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ApproachConfig:
    """Generic approach config container."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    candidates_depth: int | None = None


def default_approach2_config() -> ApproachConfig:
    """Default (placeholder) config for Approach 2."""
    return ApproachConfig(
        name="clustpsg",
        params={
            # Retrieval models used inside clustpsg:
            # - docs: BM25 + QLD(mu=1000)
            # - passages: BM25 + QLD(mu=1000) (requires a passage index)
            "doc_retrieval": {"model": "bm25", "qld_mu": 1000},
            "passage_retrieval": {"model": "bm25", "qld_mu": 1000},

            # Passage extraction (based on `rag_system/passages.py`):
            "min_sentences": 3,
            "max_sentences": 5,
            "stride_sentences": 2,
            # Soft cap: do NOT truncate sentences; allow exceeding this to keep full sentences.
            "max_chars_per_sentence_soft": 300,

            # PR4 clustering configuration (pluggable methods + hyperparameters)
            "clustering": {
                "vectorizer": "tfidf",  # tfidf | embeddings | custom
                "similarity": "cosine",  # cosine | dot | jaccard | custom
                # Default is dependency-free:
                "algorithm": "graph_threshold",  # graph_threshold | kmeans | agglomerative | dbscan
                "random_state": 42,
                # Used by graph_threshold:
                "threshold": 0.5,
                # Vectorizer params (tfidf example)
                "tfidf": {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0, "max_features": None},
                # Embeddings params (if used)
                "embeddings": {"model": None, "normalize": True},
                # Clustering params (examples)
                "kmeans": {"n_clusters": 20, "n_init": 10, "max_iter": 300},
                "agglomerative": {"n_clusters": None, "distance_threshold": 0.7, "linkage": "average"},
                "dbscan": {"eps": 0.5, "min_samples": 5},
            },

            # Feature selection / extensibility (PR5): list of enabled feature names.
            "enabled_features": [],
        },
        candidates_depth=None,
    )


def default_approach3_config() -> ApproachConfig:
    """Default (placeholder) config for Approach 3."""
    return ApproachConfig(
        name="approach3_template",
        params={
            # Put approach-specific knobs here (placeholder).
            # Example: "rrf_k": 60
        },
        # For multi-stage methods, candidates_depth is often > topk (e.g., 2000-5000).
        candidates_depth=2000,
    )


