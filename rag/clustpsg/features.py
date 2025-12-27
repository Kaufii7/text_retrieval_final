"""PR5: Feature computation for clustpsg (cluster -> feature vector).

Key goal: make feature addition easy.
- Add a new feature by writing a function and registering it in FEATURES.
- Select enabled features by name via config (cfg.params["enabled_features"]).

This module computes features for a single cluster given:
- the query
- the cluster membership (passage indices)
- the passages (with document_id + content)
- rank maps for passages and documents (for query-dependent features)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from rag.types import Passage, Query
from rag.clustpsg.cluster import Cluster
from rag.clustpsg.similarity import pair_similarity, vectorize_passages


FeatureFn = Callable[["FeatureContext"], float]


@dataclass(frozen=True)
class FeatureContext:
    query: Query
    cluster: Cluster
    passages: Sequence[Passage]
    # Query-dependent rank maps (1 = best). Missing -> treated as worst/absent.
    passage_rank: Mapping[tuple[str, int], int]  # (document_id, passage_index) -> rank
    document_rank: Mapping[str, int]  # document_id -> rank
    # Config knobs
    stopwords: frozenset[str]
    similarity_cfg: Mapping[str, object]  # reuse cfg.params["clustering"] subset
    max_pairwise: int

    def cluster_passages(self) -> List[Passage]:
        return [self.passages[i] for i in self.cluster.passage_indices]


FEATURES: Dict[str, FeatureFn] = {}


def register_feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def _wrap(fn: FeatureFn) -> FeatureFn:
        if name in FEATURES:
            raise ValueError(f"Duplicate feature name: {name}")
        FEATURES[name] = fn
        return fn

    return _wrap


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _safe_reciprocal_rank(rank: Optional[int]) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


@register_feature("qdep_rr_cluster_passages_max")
def feat_rr_cluster_passages_max(ctx: FeatureContext) -> float:
    """Max reciprocal rank over the cluster's passages in the passage ranking."""
    best = 0.0
    for p in ctx.cluster_passages():
        r = ctx.passage_rank.get((p.document_id, p.index))
        best = max(best, _safe_reciprocal_rank(r))
    return best


@register_feature("qdep_rr_cluster_docs_max")
def feat_rr_cluster_docs_max(ctx: FeatureContext) -> float:
    """Max reciprocal rank over the cluster's source documents in the doc ranking."""
    best = 0.0
    for p in ctx.cluster_passages():
        r = ctx.document_rank.get(p.document_id)
        best = max(best, _safe_reciprocal_rank(r))
    return best


@register_feature("prior_term_entropy")
def feat_term_entropy(ctx: FeatureContext) -> float:
    """Entropy of token distribution (excluding stopwords). Lower => more coherent."""
    counts: Counter = Counter()
    total = 0
    for p in ctx.cluster_passages():
        toks = [t for t in _tokens(p.content) if t not in ctx.stopwords]
        counts.update(toks)
        total += len(toks)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / float(total)
        ent -= p * math.log(p)
    return float(ent)


@register_feature("prior_stopword_ratio")
def feat_stopword_ratio(ctx: FeatureContext) -> float:
    """Fraction of tokens that are stopwords inside the cluster."""
    sw = 0
    total = 0
    for p in ctx.cluster_passages():
        toks = _tokens(p.content)
        total += len(toks)
        sw += sum(1 for t in toks if t in ctx.stopwords)
    if total <= 0:
        return 0.0
    return float(sw / float(total))


@register_feature("prior_intra_similarity_mean")
def feat_intra_similarity_mean(ctx: FeatureContext) -> float:
    """Mean pairwise similarity within the cluster (mutual support)."""
    ps = ctx.cluster_passages()
    n = len(ps)
    if n <= 1:
        return 0.0

    # Use the same vectorizer as clustering config (pluggable).
    vectors = vectorize_passages(ps, ctx.similarity_cfg)
    sim_name = str(ctx.similarity_cfg.get("similarity", "cosine")).lower()

    # Compute up to max_pairwise pairs in deterministic order.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    if ctx.max_pairwise > 0 and len(pairs) > ctx.max_pairwise:
        pairs = pairs[: ctx.max_pairwise]

    total = 0.0
    for i, j in pairs:
        total += pair_similarity(vectors[i], vectors[j], similarity=sim_name)
    return float(total / float(len(pairs))) if pairs else 0.0


@register_feature("prior_unique_docs")
def feat_unique_docs(ctx: FeatureContext) -> float:
    """Number of unique documents contributing passages to this cluster."""
    docs = {p.document_id for p in ctx.cluster_passages()}
    return float(len(docs))


def get_enabled_feature_names(enabled: Sequence[str] | None) -> List[str]:
    """Return a stable list of enabled features.

    If enabled is empty/None, returns all known features (sorted).
    """
    if not enabled:
        return sorted(FEATURES.keys())
    # keep user order, but validate
    out: List[str] = []
    for name in enabled:
        if name not in FEATURES:
            raise ValueError(f"Unknown feature: {name!r}. Known: {sorted(FEATURES.keys())}")
        out.append(name)
    return out


def compute_cluster_features(
    *,
    ctx: FeatureContext,
    enabled_features: Sequence[str] | None,
) -> Dict[str, float]:
    """Compute a dict of feature_name -> value for the given cluster."""
    names = get_enabled_feature_names(enabled_features)
    return {name: float(FEATURES[name](ctx)) for name in names}


