"""Local text scoring utilities for clustpsg (no LuceneSearcher).

Provides in-memory BM25 and QLD (Dirichlet) scoring for small candidate sets,
e.g., extracted passages per query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


_TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bm25_score(
    *,
    query_terms: Sequence[str],
    doc_tf: Counter,
    doc_len: int,
    avgdl: float,
    df: Dict[str, int],
    n_docs: int,
    k1: float,
    b: float,
) -> float:
    if doc_len <= 0 or n_docs <= 0 or avgdl <= 0:
        return 0.0

    score = 0.0
    for t in query_terms:
        tf = doc_tf.get(t, 0)
        if tf <= 0:
            continue
        dft = df.get(t, 0)
        # Standard BM25 idf with +1 to keep it non-negative-ish for small sets.
        idf = math.log((n_docs - dft + 0.5) / (dft + 0.5) + 1.0)
        denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
        score += idf * (tf * (k1 + 1.0)) / denom
    return float(score)


def qld_score(
    *,
    query_terms: Sequence[str],
    doc_tf: Counter,
    doc_len: int,
    bg_prob: Dict[str, float],
    mu: float,
) -> float:
    """QLD (Dirichlet) log-likelihood score.

    bg_prob is a background model P(w|C) over the candidate collection.
    """
    if doc_len <= 0:
        return 0.0
    if mu <= 0:
        raise ValueError("mu must be > 0")

    score = 0.0
    denom = doc_len + mu
    eps = 1e-12

    for t in query_terms:
        p_c = bg_prob.get(t, 0.0)
        if p_c <= 0.0:
            p_c = eps
        tf = float(doc_tf.get(t, 0))
        score += math.log((tf + mu * p_c) / denom)

    return float(score)


def compute_df(docs_tokens: Sequence[Sequence[str]]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return df


def compute_bg_prob(docs_tokens: Sequence[Sequence[str]]) -> Dict[str, float]:
    counts: Counter = Counter()
    total = 0
    for toks in docs_tokens:
        counts.update(toks)
        total += len(toks)
    if total <= 0:
        return {}
    return {t: c / float(total) for t, c in counts.items()}


