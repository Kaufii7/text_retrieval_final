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
                # For graph_threshold, we default to KL-based similarity over unigram LMs:
                "vectorizer": "lm",  # lm | tfidf | jaccard | embeddings | custom
                "similarity": "kl",  # kl | cosine | dot | jaccard | custom
                # Default is dependency-free:
                "algorithm": "graph_threshold",  # graph_threshold | kmeans | agglomerative | dbscan
                "random_state": 42,
                # Used by graph_threshold:
                "threshold": 0.5,
                # Max number of passages in a single centered cluster (including the center passage).
                "max_cluster_size": 20,
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
            # If empty, treat as "all registered features".
            "enabled_features": [],
            # Feature extraction knobs (PR5)
            "feature_max_pairwise_sim": 2000,
            # Optional: provide your own stopword list later; for now features use a small built-in set.

            # PR6 label/aggregation settings (doc-level LTR)
            # Label is derived from qrels: label = 1 if rel >= threshold else 0
            "label_rel_threshold": 1,
            # How to aggregate cluster-level features into a per-document feature vector:
            # max | sum | mean
            "doc_feature_aggregation": "max",

            # PR7 SVM hyperparameters / persistence
            "svm": {
                # Backend: "svm_rank" (Joachims SVM^rank) or "linear_svc" (sklearn baseline)
                "backend": "svm_rank",
                "C": 1.0,
                "class_weight": "balanced",  # "balanced" | None
                "max_iter": 5000,
                "random_state": 42,
                # Metadata pickle (used by both backends)
                "model_path": "models/clustpsg_svm.pkl",

                # SVM^rank binaries + paths (must be installed on your machine and in PATH, or provide absolute paths)
                "svm_rank_learn_bin": "svm_rank_learn",
                "svm_rank_classify_bin": "svm_rank_classify",
                # Where to write svmrank train/test files and predictions
                "svm_rank_work_dir": "models/svmrank_work",
                # Where svm_rank_learn writes the external model file
                "svm_rank_model_path": "models/svmrank.model",
            },

            # PR8 pipeline limits (runtime control during development)
            "doc_content_topk": 2000,
            "clustering_max_passages": 200,
            # Candidate generation depth for clustpsg (retrieve this many docs, then rerank, then output topk=1000).
            "doc_candidates_depth": 2000,

            # Training-time: include *all* judged qrels docids as candidates, even if retrieval is top-k capped.
            "train_include_all_qrels_docs": True,
            # When split=train, how many candidate docs to fetch full content for passage extraction.
            # If None, uses doc_content_topk.
            "train_doc_content_topk": None,
            # Training doc source:
            # - "qrels": build training set from qrels docids only (fetched by docid)
            # - "retrieved": train from retrieved candidates (optionally augmented with qrels docids)
            "train_docs_source": "qrels",

            # PR8 reranking control: keep clustpsg from wrecking a strong baseline.
            # - Only rerank within the top-N retrieved documents.
            # - Blend SVM score with baseline retrieval score.
            "rerank": {
                "topn": 1000,
                # final = alpha * svm + (1-alpha) * baseline
                "alpha": 0.8,
                # Optional normalization of per-query SVM decision scores before blending:
                # none | zscore | minmax
                "svm_norm": "zscore",
                # Additional multiplier applied to the (optionally normalized) SVM score.
                "svm_scale": 1.0,
            },
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


