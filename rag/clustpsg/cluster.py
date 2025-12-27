"""PR4: Clustering passages (pluggable & configurable).

This module provides a dependency-free default clustering method:
- graph-threshold clustering: build similarity edges above a threshold and take connected components.

It also leaves room for sklearn-based methods (kmeans/agglomerative/dbscan) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from rag.types import Passage

from rag.clustpsg.similarity import similarity_edges, vectorize_passages


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
    vectors = vectorize_passages(passages, cfg)
    edges = similarity_edges(vectors, cfg)

    n = len(passages)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # deterministic union: attach larger root id to smaller
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, j, _s in edges:
        union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    clusters: List[Cluster] = []
    for cid, root in enumerate(sorted(groups.keys())):
        clusters.append(Cluster(id=cid, passage_indices=groups[root]))

    return clusters


