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
            # - docs: BM25, BM25+RM3, QLD(mu=1000), or QLD+RM3
            # - passages: BM25 + QLD(mu=1000) (requires a passage index)
            "doc_retrieval": {
                # model: "bm25" | "bm25+rm3" | "qld" | "qld+rm3"
                "model": "bm25+rm3",
                # BM25 params (used by "bm25" and "bm25+rm3")
                "k1": 1.0, # Term frequency normalization
                "b": 0.8, # Length normalization
                # RM3 params (used by "bm25+rm3" and "qld+rm3")
                "rm3_fb_terms": 10,
                "rm3_fb_docs": 30,
                "rm3_original_query_weight": 0.5,
                # QLD param (used by "qld" and "qld+rm3")
                "qld_mu": 1000,
            },
            "passage_retrieval": {
                # Passage ranking model: local bm25/qld OR Lucene BM25 over a per-query passage index.
                "model": "lucene_bm25+rm3",
                # BM25 params for lucene_bm25 or local bm25.
                "k1": 0.9,
                "b": 0.4,
                # Optional speed/recall knobs (used if lucene_use_all_passages=False or for local scoring).
                "per_doc": True,
                "per_doc_filter": "overlap",
                "per_doc_filter_k": 10,
                # Choose between (A) and (B):
                # - mode="temp"  => build a temporary per-query index each run (A)
                # - mode="cache" => cache per-query passage indices on disk (B, default)
                "lucene": {
                    "mode": "cache",
                    "cache_dir": "cache/passage_lucene",
                    # Default per requirement: index all extracted passages for the query.
                    "use_all_passages": True,
                    # Optional: RM3 pseudo-relevance feedback over the passage index.
                    # Enable either by setting this flag, or by using model="lucene_bm25+rm3".
                    "rm3": {
                        "enabled": True,
                        "fb_terms": 50,
                        "fb_docs": 10,
                        "original_query_weight": 0.2,
                    },
                },
            },

            # Conservative semantic query expansion (noun-only via curated synonym map).
            # Adds up to N extra terms per query to improve recall while limiting drift.
            "semantic_expansion": {
                "enabled": False,
                # "static" uses a small curated noun synonym map (no extra deps).
                # "nltk_wordnet" uses NLTK POS tagging + WordNet synonyms (noun-only).
                "backend": "nltk_wordnet",
                # Recommended: 1-2
                "max_terms": 5,
                # NLTK backend: cap how many lemma candidates we consider per noun.
                "nltk_lemma_max_per_noun": 5,
                # Static backend only: overrides/extensions to the built-in noun synonym map.
                "synonyms": {},
            },

            # Passage extraction (based on `rag_system/passages.py`):
            "min_sentences": 3,
            "max_sentences": 10,
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
                # Safety: blend RankSVM scores with the seed-heuristic cluster score
                # to reduce degradation from noisy model predictions.
                #
                # blended = alpha * norm(svm_score) + (1-alpha) * norm(seed_rr)
                "cluster_score_blend": {
                    # alpha=1.0 => pure SVM (old behavior when use_svm_cluster_scores=True)
                    # alpha=0.0 => pure seed heuristic
                    "alpha": 0.5,
                    # Score normalization per topic before blending: none | zscore | minmax
                    "svm_norm": "minmax",
                    "seed_norm": "minmax",
                },
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
                # RankSVM only: global feature normalization before writing svmrank_{train,test}.dat
                # Use this to avoid feature-scale domination (e.g., counts overpowering RR features).
                # Options: "none" | "minmax" | "zscore"
                "svm_rank_feature_norm": "minmax",
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
            "doc_content_topk": 2000,
            # Fewer passages => much faster O(n^2) clustering
            "clustering_max_passages": 200,
            # Candidate generation depth for clustpsg (retrieve this many docs, then rerank, then output topk=1000).
            "doc_candidates_depth": 5000,

            # Training-time: include *all* judged qrels docids as candidates, even if retrieval is top-k capped.
            "train_include_all_qrels_docs": True,
            # When split=train, how many candidate docs to fetch full content for passage extraction.
            # If None, uses doc_content_topk.
            "train_doc_content_topk": None,
            # Training doc source:
            # - "qrels": build training set from qrels docids only (fetched by docid)
            # - "retrieved": train from retrieved candidates (optionally augmented with qrels docids)
            "train_docs_source": "retrieved",

            # Cluster label supervision for RankSVM training (used by build_cluster_level_training_set):
            # - "any_relevant_doc": label=1 if cluster contains ANY passage from a relevant qrels doc (binary, noisy)
            # - "top_weighted_density": bucketed, top-heavy density of relevant-doc passages inside the cluster
            #   density uses weights w(r)=1/(rr_k + r) where r is global passage rank (1=best).
            "cluster_labeling": {
                # Cluster label method (training only; inference uses label=0):
                # - "any_relevant_doc": label=1 if cluster contains ANY passage from a relevant qrels doc (binary, noisy)
                # - "top_weighted_density": top-heavy density of relevant-doc passages inside the cluster (bucketed 0/1/2)
                # - "best_evidence": bucketed by the best (lowest) relevant-doc passage rank inside the cluster (0/1/2)
                # - "pseudo_passage_overlap": build pseudo-relevant passages from qrels relevant docs using Lucene BM25+RM3,
                #   then label clusters by overlap with those pseudo-relevant passages (0/1/2).
                "method": "pseudo_passage_overlap",

                # ---- top_weighted_density ----
                "rr_k": 40,
                "threshold_low": 0.10,
                "threshold_high": 0.30,

                # ---- best_evidence ----
                # label=2 if best_rank <= best_rank_high; label=1 if <= best_rank_low
                "best_rank_low": 100,
                "best_rank_high": 20,

                # ---- pseudo_passage_overlap ----
                # For each relevant qrels doc, mark its top-K passages (by Lucene BM25+RM3) as pseudo-relevant.
                "pseudo_topk_per_doc": 3,
                # How many overlaps with pseudo-relevant passages are needed for labels 1/2.
                "pseudo_overlap_low": 1,
                "pseudo_overlap_high": 2,
                # Lucene settings for pseudo passage ranking (per-doc temporary passage index).
                "pseudo_lucene": {
                    "mode": "temp",  # temp | cache
                    "cache_dir": "cache/passage_lucene",
                    # BM25 parameters for the per-doc passage index.
                    "k1": 0.9,
                    "b": 0.4,
                    # RM3 pseudo-relevance feedback over the per-doc passage index.
                    "rm3": {
                        "enabled": True,
                        "fb_terms": 50,
                        "fb_docs": 10,
                        "original_query_weight": 0.2,
                    },
                },
            },

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
    """Default config for Approach 3 (two-stage: dense recall -> optional CE rerank).

    Notes:
    - Dense assets (A3-1) are expected under `cache/approach3_dense/` by default.
    - ANN index (A3-3) is cached to disk when enabled (hnswlib).
    - Heavy ML deps are optional but required when running Approach 3:
      - sentence-transformers (for query embedding; and for cross-encoder rerank if enabled)
      - hnswlib (optional, for ANN; otherwise exact fallback is used)
    """
    # Keep defaults local and deterministic; filenames match A3-1/A3-3 helpers.
    index_name = "robust04"
    assets_dir = "cache/approach3_dense"
    bi_encoder_model = "sentence-transformers/all-mpnet-base-v2"

    safe_index = index_name.replace("/", "_")
    safe_model = bi_encoder_model.replace("/", "_")
    embeddings_path = f"{assets_dir}/embeddings_{safe_index}__{safe_model}.npy"
    docids_path = f"{assets_dir}/docids_{safe_index}.txt"
    hnsw_index_path = f"{assets_dir}/hnsw_{safe_index}__{safe_model}.bin"
    hnsw_meta_path = f"{assets_dir}/hnsw_{safe_index}__{safe_model}.meta.json"

    return ApproachConfig(
        name="approach3_dense_ce",
        params={
            "dense": {
                "index": index_name,
                "assets_dir": assets_dir,
                "embeddings_path": embeddings_path,
                "docids_path": docids_path,
                # Similarity metric used for dense retrieval (cosine recommended for normalized embeddings).
                "metric": "cosine",  # cosine | ip | l2
                # Retrieval backend: try hnswlib (fast ANN) and fall back to exact if unavailable.
                "backend": "hnswlib",  # hnswlib | exact
                # HNSW cache paths
                "hnsw_index_path": hnsw_index_path,
                "hnsw_meta_path": hnsw_meta_path,
                # If the HNSW cache is missing, build it once and cache it (requires hnswlib).
                "build_index_if_missing": True,
                "hnsw": {"ef_construction": 200, "M": 16, "seed": 42, "ef": 200},
                # Query encoder
                "model_name": bi_encoder_model,
                "device": "cpu",
                "batch_size": 1,
                "normalize_embeddings": True,
            },
            "rerank": {
                "enabled": False,
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                # If true, use `finetuned_model_dir` instead of `model_name`.
                # This should point to a directory produced by A3-7 fine-tuning.
                "use_finetuned": False,
                "finetuned_model_dir": "models/approach3_ce/best",
                "device": "cpu",
                "batch_size": 16,
                "topn": 100,
            },
            # Final scoring policy for reranked docs:
            # final = alpha * ce + (1-alpha) * dense_score
            "score_fusion": {"alpha": 1.0},
        },
        # Stage-1 candidates depth (often > topk).
        candidates_depth=2000,
    )


