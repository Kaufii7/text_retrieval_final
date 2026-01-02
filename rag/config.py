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
            # - docs: BM25, BM25+RM3, or QLD(mu=1000)
            # - passages: BM25 + QLD(mu=1000) (requires a passage index)
            "doc_retrieval": {
                # model: "bm25" | "bm25+rm3" | "qld"
                "model": "bm25+rm3",
                # BM25 params (used by "bm25" and "bm25+rm3")
                "k1": 1.0, # Term frequency normalization
                "b": 0.8, # Length normalization
                # RM3 params (used by "bm25+rm3")
                "rm3_fb_terms": 10,
                "rm3_fb_docs": 10,
                "rm3_original_query_weight": 0.5,
                # QLD param (used by "qld")
                "qld_mu": 1000,
            },
            "passage_retrieval": {"model": "bm25", "qld_mu": 1000, "per_doc": True, "per_doc_filter": "overlap", "per_doc_filter_k": 10},

            # Passage extraction (based on `rag_system/passages.py`):
            "min_sentences": 3,
            "max_sentences": 5,
            "stride_sentences": 2,
            # Soft cap: do NOT truncate sentences; allow exceeding this to keep full sentences.
            "max_chars_per_sentence_soft": 300,

            # PR4 clustering configuration (pluggable methods + hyperparameters)
            "clustering": {
                # Faster defaults: TF-IDF + cosine (much cheaper than LM+KL)
                "vectorizer": "tfidf",  # lm | tfidf | jaccard | embeddings | custom
                "similarity": "cosine",  # kl | cosine | dot | jaccard | custom
                # Default is dependency-free:
                "algorithm": "graph_threshold",  # graph_threshold | kmeans | agglomerative | dbscan
                "random_state": 42,
                # Used by graph_threshold:
                "threshold": 0.2,
                # Max number of passages in a single centered cluster (including the center passage).
                "max_cluster_size": 5,
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

            # PR8 passage re-ranking and fusion (cluster -> passage -> document)
            "passage_rerank": {
                # Linear mix: combined = (1-beta)*passage + beta*cluster
                "beta": 0.7,
                # Normalize scores per topic before mixing: none | zscore | minmax
                "passage_norm": "minmax",
                "cluster_norm": "zscore",
                # Additional multipliers (after normalization)
                "passage_scale": 1.0,
                "cluster_scale": 1.0,
                # How to aggregate overlapping cluster scores for a passage: max | mean
                "cluster_score_agg": "max",
            },
            # RRF parameters for converting passage ranking -> document scores
            "rrf": {"k": 60, "depth": 200},

            # Final ClustPsg scoring (as per assignment spec):
            # - Re-rank passages by cluster RankSVM score (tie-break by original passage rank)
            # - Sum reciprocal ranks of top passages per doc (cap M)
            # - Fuse with BM25 doc reciprocal rank using adaptive lambda based on how many top passages the doc got
            "final": {
                # If true, cluster scores come from RankSVM predictions.
                # If false, cluster scores are derived from the cluster seed passage rank.
                "use_svm_cluster_scores": True,
                # Reciprocal-rank denominator offset (RRF-style): contribution = 1 / (rr_k + rank)
                "rr_k": 100,
                "max_passages_per_doc": 200,  # M
                "lambda_min": 0.5,
                "lambda_max": 1.0,
            },

            # Cache ranked passages to speed up iterative runs (especially when only tuning downstream params).
            # Cache key includes: queries + docids used + extraction params + passage_retrieval params + topk.
            "passage_cache": {
                "enabled": False,
                "dir": "cache/ranked_passages",
            },

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
                "svm_rank_learn_bin": "svm_rank/svm_rank_learn",
                "svm_rank_classify_bin": "svm_rank/svm_rank_classify",
                # Where to write svmrank train/test files and predictions
                "svm_rank_work_dir": "models/svmrank_work",
                # Where svm_rank_learn writes the external model file
                "svm_rank_model_path": "models/svmrank.model",
            },

            # PR8 pipeline limits (runtime control during development)
            "doc_content_topk": 1000,
            # Fewer passages => much faster O(n^2) clustering
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
            # - Blend clustpsg doc score (via passage-level RRF) with baseline retrieval score.
            "rerank": {
                "topn": 1000,
                # final = alpha * clustpsg + (1-alpha) * baseline
                "alpha": 0.8,
                # Optional normalization of per-query clustpsg doc scores before blending:
                # none | zscore | minmax
                "rrf_norm": "minmax",
                # Additional multiplier applied to the (optionally normalized) clustpsg score.
                "rrf_scale": 1.0,
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


