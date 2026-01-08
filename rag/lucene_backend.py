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
import json
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


def fetch_doc_contents(searcher, docid: str) -> str:
    """Best-effort raw content fetch for a document from a LuceneSearcher.

    Preference order:
    1) `doc.contents()` (when available)
    2) `doc.raw()` parsed as JSON with `"contents"` (or `"raw"`) field
    3) `doc.raw()` as-is
    """
    d = searcher.doc(docid)
    if d is None:
        return ""

    # Prefer actual contents if available; raw() is often JSON and may include metadata.
    try:
        if hasattr(d, "contents"):
            c = d.contents()
            if c:
                return c
    except Exception:
        pass

    raw = ""
    try:
        raw = d.raw() or ""
    except Exception:
        raw = ""

    # If raw looks like JSON with a "contents" field, extract it.
    if raw and raw.lstrip().startswith("{"):
        try:
            obj = json.loads(raw)
            c = obj.get("contents") or obj.get("raw") or ""
            if isinstance(c, str) and c:
                return c
        except Exception:
            pass

    return raw


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


def set_rm3(
    searcher,
    *,
    fb_terms: int = 10,
    fb_docs: int = 10,
    original_query_weight: float = 0.5,
) -> None:
    """Enable RM3 pseudo-relevance feedback on a LuceneSearcher.

    Notes:
    - RM3 is typically applied on top of a first-stage lexical retriever (e.g., BM25).
    - Pyserini exposes RM3 via LuceneSearcher.set_rm3(...).
    """
    if not isinstance(fb_terms, int) or fb_terms <= 0:
        raise ValueError("fb_terms must be a positive integer")
    if not isinstance(fb_docs, int) or fb_docs <= 0:
        raise ValueError("fb_docs must be a positive integer")
    if not (0.0 <= float(original_query_weight) <= 1.0):
        raise ValueError("original_query_weight must be in [0, 1]")

    searcher.set_rm3(
        fb_terms=int(fb_terms),
        fb_docs=int(fb_docs),
        original_query_weight=float(original_query_weight),
    )


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


