"""Pyserini Lucene backend helpers.

Centralizes initialization and common operations so approaches can be simple and consistent.

Expected usage:
    from rag.lucene_backend import get_searcher, set_bm25, search

    searcher = get_searcher("robust04")
    set_bm25(searcher, k1=0.9, b=0.4)
    hits = search(searcher, "international organized crime", topk=1000)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SearchHit:
    docid: str
    score: float
    rank: int


def get_searcher(index_name: str = "robust04"):
    """Create a Pyserini LuceneSearcher from a prebuilt index."""
    from pyserini.search.lucene import LuceneSearcher

    return LuceneSearcher.from_prebuilt_index(index_name)


def get_index_reader(index_name: str = "robust04"):
    """Create a Pyserini IndexReader from a prebuilt index."""
    from pyserini.index.lucene import IndexReader

    return IndexReader.from_prebuilt_index(index_name)


def set_bm25(searcher, k1: float, b: float) -> None:
    """Set BM25 parameters on a LuceneSearcher."""
    if k1 <= 0:
        raise ValueError("k1 must be > 0")
    if not (0.0 <= b <= 1.0):
        raise ValueError("b must be in [0, 1]")
    searcher.set_bm25(k1=k1, b=b)


def set_qld(searcher, mu: float) -> None:
    """Set Query Likelihood with Dirichlet smoothing (QLD) on a LuceneSearcher."""
    if mu <= 0:
        raise ValueError("mu must be > 0")
    # Pyserini exposes QLD via LuceneSearcher.set_qld(mu)
    searcher.set_qld(mu)


def search(searcher, query: str, topk: int = 1000) -> List[SearchHit]:
    """Execute a search and normalize results.

    Returns a list of SearchHit with ranks starting at 1.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")

    hits = searcher.search(query, k=topk)
    out: List[SearchHit] = []
    for i, h in enumerate(hits, start=1):
        # Pyserini returns objects with .docid and .score
        out.append(SearchHit(docid=str(h.docid), score=float(h.score), rank=i))
    return out


def to_result_dicts(hits: List[SearchHit]):
    """Convert SearchHit list to dicts compatible with run writing."""
    return [{"docid": h.docid, "score": h.score, "rank": h.rank} for h in hits]


