"""Approach 2 template (placeholder).

This module is a *template* for a second run. It is intentionally not a final
method yet. The goal is to establish:
- a clear function signature
- a deterministic output contract
- a configuration surface

Output contract (important):
- Return `results_by_topic` compatible with `rag.runs.write_trec_run`, i.e.
    dict[int topic_id] -> list of results
  where each result is either:
    - (docid, score) tuple, OR
    - dict-like with keys {"docid": ..., "score": ...}
- Deterministic ordering is enforced by `write_trec_run`, but your scores should
  be stable/reproducible for a given query/config.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from rag.config import ApproachConfig, default_approach2_config
from rag.io import Query


def approach2_retrieve(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int = 1000,
    config: Optional[ApproachConfig] = None,
) -> Dict[int, List[Mapping[str, object]]]:
    """Template retrieval function for Approach 2.

    Args:
        queries: list of queries (topic_id + text)
        searcher: a Pyserini LuceneSearcher (already initialized)
        topk: number of docs per query (default 1000)
        config: optional ApproachConfig with params used by the approach

    Returns:
        results_by_topic: dict[topic_id] -> list of {"docid": ..., "score": ...}

    Notes:
        This template currently raises NotImplementedError to force you to
        explicitly choose and implement an Approach 2 method.
    """
    _ = (queries, searcher, topk)  # reserved for implementation
    if config is None:
        config = default_approach2_config()

    raise NotImplementedError(
        "Approach 2 is a template only. Implement your method here. "
        f"(config.name={config.name!r}, params={config.params!r})"
    )


