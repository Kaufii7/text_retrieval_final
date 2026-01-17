"""Approach 1 — BM25 baseline retrieval.

This module implements a thin approach layer that:
- takes a list of queries
- executes BM25 via a provided Pyserini searcher
- returns results in a structure suitable for `rag.runs.write_trec_run`
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from rag.types import Query
from rag.lucene_backend import SearchHit, search, set_bm25, set_rm3


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


def bm25_rm3_retrieve(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
    rm3_fb_terms: int = 50,
    rm3_fb_docs: int = 50,
    rm3_original_query_weight: float = 0.2,
) -> Dict[int, List[Mapping[str, object]]]:
    """Run BM25+RM3 retrieval for all queries.

    Notes:
    - RM3 is applied inside the LuceneSearcher (Pyserini).
    - This function does not apply any query expansion (unless your query texts were pre-expanded upstream).
    """
    set_bm25(searcher, k1=k1, b=b)
    set_rm3(
        searcher,
        fb_terms=int(rm3_fb_terms),
        fb_docs=int(rm3_fb_docs),
        original_query_weight=float(rm3_original_query_weight),
    )

    results_by_topic: Dict[int, List[Mapping[str, object]]] = {}
    for q in queries:
        hits: List[SearchHit] = search(searcher, q.text, topk=topk)
        results_by_topic[q.topic_id] = [{"docid": h.docid, "score": h.score} for h in hits]
    return results_by_topic

