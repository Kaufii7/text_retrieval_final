"""Approach 3 template (placeholder; "advanced slot").

This module is a *template* for the third run. It is intentionally not a final
method yet. The goal is to establish structure for more advanced pipelines such
as:
- multi-stage retrieval (candidates -> rerank)
- fusion of multiple ranked lists (e.g., RRF)
- reranking hooks (cross-encoder, LTR, etc.)

Output contract (important):
- Return `results_by_topic` compatible with `rag.runs.write_trec_run`, i.e.
    dict[int topic_id] -> list of results
  where each result is either:
    - (docid, score) tuple, OR
    - dict-like with keys {"docid": ..., "score": ...}
- Determinism: for a given query/config, scores should be stable/reproducible.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from rag.config import ApproachConfig, default_approach3_config
from rag.types import Query


def approach3_retrieve(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int = 1000,
    config: Optional[ApproachConfig] = None,
) -> Dict[int, List[Mapping[str, object]]]:
    """Template retrieval function for Approach 3 (advanced slot).

    Suggested design (not implemented here):
    - Stage 1: generate candidates with depth = config.candidates_depth or (topk * X)
    - Stage 2: rerank or fuse candidates with another signal
    - Output: topk results in run-writer compatible format
    """
    _ = (queries, searcher, topk)  # reserved for implementation
    if config is None:
        config = default_approach3_config()

    raise NotImplementedError(
        "Approach 3 is a template only. Implement your method here. "
        f"(config.name={config.name!r}, params={config.params!r}, candidates_depth={config.candidates_depth!r})"
    )


