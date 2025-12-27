"""PR4: Similarity utilities for clustering passages (pluggable).

This module provides:
- vectorization (tfidf / jaccard token-sets)
- similarity computation (cosine / dot / jaccard)

All choices are driven by `cfg.params["clustering"]`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from rag.types import Passage


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


SparseVec = Dict[str, float]


def vectorize_passages(passages: Sequence[Passage], clustering_cfg: Mapping[str, object]) -> List[object]:
    """Vectorize passages according to config.

    Returns a list of vectors; vector type depends on the chosen vectorizer:
    - tfidf -> SparseVec (dict[str, float])
    - jaccard -> set[str]
    """
    vectorizer = str(clustering_cfg.get("vectorizer", "tfidf")).lower()
    if vectorizer == "tfidf":
        tfidf_cfg = clustering_cfg.get("tfidf", {}) or {}
        ngram_range = tuple(tfidf_cfg.get("ngram_range", (1, 1)))
        min_df = int(tfidf_cfg.get("min_df", 1))
        max_df = float(tfidf_cfg.get("max_df", 1.0))
        max_features = tfidf_cfg.get("max_features", None)
        if max_features is not None:
            max_features = int(max_features)
        return _tfidf_vectors(
            passages,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

    if vectorizer in ("jaccard", "token_set"):
        return [set(_tokenize(p.content)) for p in passages]

    if vectorizer in ("embeddings", "custom"):
        raise NotImplementedError(
            f"vectorizer={vectorizer!r} is not implemented yet. Use 'tfidf' or 'jaccard' for now."
        )

    raise ValueError(f"Unknown vectorizer: {vectorizer!r}")


def similarity_edges(
    vectors: Sequence[object],
    clustering_cfg: Mapping[str, object],
) -> List[Tuple[int, int, float]]:
    """Compute (i, j, sim) edges for pairs above a threshold.

    This is designed for graph-threshold clustering (connected components).
    """
    similarity = str(clustering_cfg.get("similarity", "cosine")).lower()
    threshold = float(clustering_cfg.get("threshold", 0.5))
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("threshold must be in [-1, 1]")

    edges: List[Tuple[int, int, float]] = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            s = pair_similarity(vectors[i], vectors[j], similarity=similarity)
            if s >= threshold:
                edges.append((i, j, s))
    # Deterministic order
    edges.sort(key=lambda x: (-x[2], x[0], x[1]))
    return edges


def pair_similarity(a: object, b: object, *, similarity: str) -> float:
    """Compute similarity between two vectors."""
    if similarity == "cosine":
        return _cosine(a, b)
    if similarity == "dot":
        return _dot(a, b)
    if similarity == "jaccard":
        return _jaccard(a, b)
    raise ValueError(f"Unknown similarity: {similarity!r}")


def _dot(a: object, b: object) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        # sparse dot
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * float(b.get(k, 0.0)) for k, v in a.items()))
    raise TypeError("dot similarity is only implemented for sparse dict vectors")


def _cosine(a: object, b: object) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        num = _dot(a, b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(num / (na * nb))
    raise TypeError("cosine similarity is only implemented for sparse dict vectors")


def _jaccard(a: object, b: object) -> float:
    if isinstance(a, set) and isinstance(b, set):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter / union) if union else 0.0
    raise TypeError("jaccard similarity is only implemented for set vectors")


def _tfidf_vectors(
    passages: Sequence[Passage],
    *,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    max_features: int | None,
) -> List[SparseVec]:
    """Compute simple TF-IDF sparse vectors for passages (dependency-free)."""
    n_min, n_max = ngram_range
    if n_min <= 0 or n_max < n_min:
        raise ValueError("Invalid ngram_range")
    if min_df <= 0:
        raise ValueError("min_df must be >= 1")
    if not (0.0 < max_df <= 1.0):
        raise ValueError("max_df must be in (0, 1]")

    docs_ngrams: List[List[str]] = []
    for p in passages:
        toks = _tokenize(p.content)
        grams: List[str] = []
        for n in range(n_min, n_max + 1):
            for i in range(0, max(0, len(toks) - n + 1)):
                grams.append(" ".join(toks[i : i + n]))
        docs_ngrams.append(grams)

    n_docs = len(docs_ngrams)
    df: Counter = Counter()
    for grams in docs_ngrams:
        df.update(set(grams))

    # apply df filters
    max_df_count = int(math.floor(max_df * n_docs))
    vocab = [t for t, c in df.items() if c >= min_df and c <= max_df_count]
    if max_features is not None and len(vocab) > max_features:
        # keep most common by df
        vocab = sorted(vocab, key=lambda t: (-df[t], t))[:max_features]
    vocab_set = set(vocab)

    # precompute idf
    idf: Dict[str, float] = {}
    for t in vocab:
        dft = df[t]
        idf[t] = math.log((n_docs + 1.0) / (dft + 1.0)) + 1.0

    vectors: List[SparseVec] = []
    for grams in docs_ngrams:
        tf = Counter(g for g in grams if g in vocab_set)
        if not tf:
            vectors.append({})
            continue
        vec: SparseVec = {t: float(tf[t]) * idf[t] for t in tf.keys()}
        vectors.append(vec)

    return vectors


