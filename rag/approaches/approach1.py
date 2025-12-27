"""Approach 1 — BM25 baseline retrieval.

This module implements a thin approach layer that:
- takes a list of queries
- executes BM25 via a provided Pyserini searcher
- returns results in a structure suitable for `rag.runs.write_trec_run`
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from rag.io import Query
from rag.lucene_backend import SearchHit, search, set_bm25


def bm25_retrieve(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
) -> Dict[int, List[Mapping[str, object]]]:
    """Run BM25 retrieval for all queries.

    Returns:
      results_by_topic: dict[topic_id] -> list of dicts with at least {docid, score}
    """
    set_bm25(searcher, k1=k1, b=b)

    results_by_topic: Dict[int, List[Mapping[str, object]]] = {}
    for q in queries:
        hits: List[SearchHit] = search(searcher, q.text, topk=topk)
        results_by_topic[q.topic_id] = [{"docid": h.docid, "score": h.score} for h in hits]
    return results_by_topic


