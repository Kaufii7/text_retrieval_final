"""PR1: Document candidate retrieval for clustpsg.

Provides a helper that produces a per-query ranked list of candidate documents
using a provided Pyserini LuceneSearcher.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from rag.config import ApproachConfig
from rag.lucene_backend import SearchHit, search, set_bm25, set_qld
from rag.types import Document, Query


def _apply_doc_retrieval_model(searcher, cfg: ApproachConfig) -> None:
    model_cfg = (cfg.params or {}).get("doc_retrieval", {}) if cfg else {}
    model = (model_cfg.get("model") or "bm25").lower()

    if model == "bm25":
        # Allow overriding within params; fall back to Approach1 defaults.
        k1 = float(model_cfg.get("k1", 0.9))
        b = float(model_cfg.get("b", 0.4))
        set_bm25(searcher, k1=k1, b=b)
        return

    if model in ("qld", "ql", "dirichlet"):
        mu = float(model_cfg.get("qld_mu", 1000))
        set_qld(searcher, mu=mu)
        return

    raise ValueError(f"Unknown doc retrieval model: {model!r} (expected 'bm25' or 'qld')")


def retrieve_doc_candidates(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int,
    config: ApproachConfig,
    logger: Optional[logging.Logger] = None,
    debug_output_path: Optional[str] = None,
) -> Dict[int, List[Document]]:
    """Retrieve top-k document candidates per query.

    Returns:
      dict[topic_id] -> ranked list of Documents (rank is the list order; score is populated).
    """
    if topk <= 0:
        raise ValueError("topk must be > 0")
    log = logger or logging.getLogger("rag.clustpsg.doc_retrieval")

    _apply_doc_retrieval_model(searcher, config)

    results_by_topic: Dict[int, List[Document]] = {}
    for q in queries:
        hits: List[SearchHit] = search(searcher, q.content, topk=topk)
        # We don't fetch full document text here; downstream stages can fetch content
        # if needed via an IndexReader. Rank is implied by list ordering.
        results_by_topic[q.id] = [Document(id=h.docid, content="", score=h.score) for h in hits]

    if debug_output_path:
        # Simple TSV: topic_id \t docid \t rank \t score
        with open(debug_output_path, "w", encoding="utf-8") as f:
            for topic_id in sorted(results_by_topic.keys()):
                for rank, d in enumerate(results_by_topic[topic_id], start=1):
                    f.write(f"{topic_id}\t{d.id}\t{rank}\t{d.score}\n")
        log.info("Wrote clustpsg doc candidates debug TSV: %s", debug_output_path)

    return results_by_topic


