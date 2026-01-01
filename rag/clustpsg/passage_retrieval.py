"""PR3: Passage retrieval / scoring (query -> passages) for clustpsg.

This PR ranks *extracted passages* in-memory (no Lucene index):
- BM25 (local)
- QLD (Dirichlet smoothing)

Two modes:
- global (default): score all passages across all documents for the query, then take top-k
- per_doc: score passages *within each document only*, keep top-N per document, then merge and take top-k

The per_doc mode avoids computing a single huge DF/background model over hundreds of thousands
of passages, which can be very slow and memory-heavy.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from rag.config import ApproachConfig
from rag.types import Passage, Query

from rag.clustpsg.text_scoring import (
    bm25_score,
    compute_bg_prob,
    compute_df,
    qld_score,
    tokenize,
)


def rank_passages(
    *,
    queries: Sequence[Query],
    passages_by_topic: Dict[int, List[Passage]],
    topk: int,
    config: ApproachConfig,
    logger: Optional[logging.Logger] = None,
    stage: str | None = None,
) -> Dict[int, List[Passage]]:
    """Rank passages per query.

    Returns:
      dict[topic_id] -> ranked list of Passage (rank implied by list order; score populated).
    """
    if topk <= 0:
        raise ValueError("topk must be > 0")
    log = logger or logging.getLogger("rag.clustpsg.passage_retrieval")
    t0 = time.perf_counter()

    model_cfg = (config.params or {}).get("passage_retrieval", {}) if config else {}
    model = (model_cfg.get("model") or "bm25").lower()

    per_doc = bool(model_cfg.get("per_doc", False))
    per_doc_topn = int(model_cfg.get("per_doc_topn", 3))
    if per_doc_topn < 0:
        per_doc_topn = 0

    results_by_topic: Dict[int, List[Passage]] = {}
    total_passages = 0

    for q in queries:
        passages = passages_by_topic.get(q.id, [])
        if not passages:
            results_by_topic[q.id] = []
            continue
        total_passages += len(passages)

        q_terms = tokenize(q.content)

        def _score_one(tf: Counter, dl: int, *, df, bg, n_docs: int, avgdl: float) -> float:
            if model == "bm25":
                k1 = float(model_cfg.get("k1", 0.9))
                b = float(model_cfg.get("b", 0.4))
                return bm25_score(
                    query_terms=q_terms,
                    doc_tf=tf,
                    doc_len=dl,
                    avgdl=avgdl,
                    df=df,
                    n_docs=n_docs,
                    k1=k1,
                    b=b,
                )
            if model in ("qld", "ql", "dirichlet"):
                mu = float(model_cfg.get("qld_mu", 1000))
                return qld_score(query_terms=q_terms, doc_tf=tf, doc_len=dl, bg_prob=bg, mu=mu)
            if model in ("bm25+qld", "bm25_qld"):
                alpha = float(model_cfg.get("alpha", 0.5))
                mu = float(model_cfg.get("qld_mu", 1000))
                k1 = float(model_cfg.get("k1", 0.9))
                b = float(model_cfg.get("b", 0.4))
                s_bm25 = bm25_score(
                    query_terms=q_terms,
                    doc_tf=tf,
                    doc_len=dl,
                    avgdl=avgdl,
                    df=df,
                    n_docs=n_docs,
                    k1=k1,
                    b=b,
                )
                s_qld = qld_score(query_terms=q_terms, doc_tf=tf, doc_len=dl, bg_prob=bg, mu=mu)
                return alpha * s_bm25 + (1.0 - alpha) * s_qld
            raise ValueError(f"Unknown passage retrieval model: {model!r}")

        if per_doc:
            # Score within each document only; keep top-N per document; merge and globally sort.
            by_doc: Dict[str, List[Passage]] = defaultdict(list)
            for p in passages:
                by_doc[p.document_id].append(p)

            kept: List[tuple[float, Passage]] = []
            if per_doc_topn > 0:
                for _docid, ps in by_doc.items():
                    if not ps:
                        continue

                    doc_tokens = [tokenize(p.content) for p in ps]
                    df = compute_df(doc_tokens)
                    bg = compute_bg_prob(doc_tokens)
                    n_docs = len(doc_tokens)
                    avgdl = sum(len(t) for t in doc_tokens) / float(n_docs) if n_docs else 0.0

                    scored_doc: List[tuple[float, Passage]] = []
                    for p, toks in zip(ps, doc_tokens):
                        tf = Counter(toks)
                        dl = len(toks)
                        s = _score_one(tf, dl, df=df, bg=bg, n_docs=n_docs, avgdl=avgdl)
                        scored_doc.append((s, Passage(document_id=p.document_id, index=p.index, content=p.content, score=s)))

                    # deterministic: score desc, then passage index asc
                    scored_doc.sort(key=lambda x: (-x[0], x[1].index))
                    kept.extend(scored_doc[:per_doc_topn])

            kept.sort(key=lambda x: (-x[0], x[1].document_id, x[1].index))
            results_by_topic[q.id] = [p for _s, p in kept[:topk]]
        else:
            # Original mode: score globally across all passages.
            docs_tokens = [tokenize(p.content) for p in passages]
            df = compute_df(docs_tokens)
            bg = compute_bg_prob(docs_tokens)
            n_docs = len(docs_tokens)
            avgdl = sum(len(t) for t in docs_tokens) / float(n_docs) if n_docs else 0.0

            scored: List[tuple[float, Passage]] = []
            for p, toks in zip(passages, docs_tokens):
                tf = Counter(toks)
                dl = len(toks)
                s = _score_one(tf, dl, df=df, bg=bg, n_docs=n_docs, avgdl=avgdl)
                scored.append((s, Passage(document_id=p.document_id, index=p.index, content=p.content, score=s)))

            scored.sort(key=lambda x: (-x[0], x[1].document_id, x[1].index))
            results_by_topic[q.id] = [p for _s, p in scored[:topk]]

    dt = time.perf_counter() - t0
    stage_str = f", stage={stage}" if stage else ""
    mode_str = ", mode=per_doc" if per_doc else ", mode=global"
    log.info(
        "Ranked passages locally for %d queries (topk=%d, model=%s, passages=%d%s%s) in %.2fs.",
        len(queries),
        topk,
        model,
        total_passages,
        stage_str,
        mode_str,
        dt,
    )
    return results_by_topic
