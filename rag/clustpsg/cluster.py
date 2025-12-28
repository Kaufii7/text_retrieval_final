"""PR4: Clustering passages (pluggable & configurable).

This module provides a dependency-free default clustering method:
- graph-threshold clustering (centered): for every passage, create its own cluster
  centered at that passage containing its most similar neighbors above threshold.

It also leaves room for sklearn-based methods (kmeans/agglomerative/dbscan) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from rag.types import Passage

from rag.clustpsg.similarity import pair_similarity, vectorize_passages


@dataclass(frozen=True)
class Cluster:
    id: int
    passage_indices: List[int]  # indices into the input passage list


def cluster_passages(passages: Sequence[Passage], *, cfg: Mapping[str, object]) -> List[Cluster]:
    """Cluster passages according to cfg['algorithm'].

    cfg is expected to be the `params["clustering"]` dict from the approach config.
    """
    algorithm = str(cfg.get("algorithm", "graph_threshold")).lower()

    if algorithm in ("graph_threshold", "threshold_graph"):
        return _cluster_graph_threshold(passages, cfg)

    if algorithm in ("kmeans", "agglomerative", "dbscan"):
        raise NotImplementedError(
            f"algorithm={algorithm!r} is not implemented yet in this repo. "
            "Use algorithm='graph_threshold' for now."
        )

    raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")


def _cluster_graph_threshold(passages: Sequence[Passage], cfg: Mapping[str, object]) -> List[Cluster]:
    """Centered threshold clustering.

    For every passage i, we create a cluster:
      {i} ∪ top neighbors j with sim(i,j) >= threshold
    capped to max_cluster_size.

    Note: clusters may overlap (this is intended).
    """
    vectors = vectorize_passages(passages, cfg)
    similarity = str(cfg.get("similarity", "cosine")).lower()
    threshold = float(cfg.get("threshold", 0.5))
    max_cluster_size = int(cfg.get("max_cluster_size", 20))
    if max_cluster_size <= 0:
        raise ValueError("max_cluster_size must be > 0")

    n = len(passages)
    clusters: List[Cluster] = []

    for i in range(n):
        sims: List[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            s = pair_similarity(vectors[i], vectors[j], similarity=similarity)
            if s >= threshold:
                sims.append((float(s), j))

        # Deterministic: sort by similarity desc then index asc.
        sims.sort(key=lambda x: (-x[0], x[1]))
        members = [i] + [j for _s, j in sims[: max(0, max_cluster_size - 1)]]
        clusters.append(Cluster(id=i, passage_indices=members))

    return clusters


