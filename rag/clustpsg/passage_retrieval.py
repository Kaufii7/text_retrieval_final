"""PR3: Passage retrieval / scoring (query -> passages) for clustpsg.

This PR ranks *extracted passages* in-memory (no Lucene index):
- BM25 (local)
- QLD (Dirichlet smoothing)

Two modes:
- global (default): score all passages across all documents for the query, then take top-k
- per_doc: first filter passages within each document using a cheap heuristic, then score the
  pooled candidate set globally (single DF/background model) so scores are comparable across docs.

The per_doc mode avoids scoring hundreds of thousands of passages globally, while still preserving
comparability of scores across documents by using a global DF/background model on the reduced pool.
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
    per_doc_filter = str(model_cfg.get("per_doc_filter", "overlap")).lower()  # overlap | first_k
    # Backward-compat: if older config uses per_doc_topn, treat it as per_doc_filter_k
    per_doc_filter_k = int(model_cfg.get("per_doc_filter_k", model_cfg.get("per_doc_topn", 5)))
    if per_doc_filter_k < 0:
        per_doc_filter_k = 0

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
            if per_doc_filter_k == 0:
                results_by_topic[q.id] = []
                continue

            by_doc: Dict[str, List[Passage]] = defaultdict(list)
            for p in passages:
                by_doc[p.document_id].append(p)

            q_set = set(q_terms)

            pooled: List[tuple[Passage, List[str]]] = []
            for _docid, ps in by_doc.items():
                if not ps:
                    continue

                if per_doc_filter == "first_k":
                    # Deterministic: keep earliest passages by index.
                    ps_sorted = sorted(ps, key=lambda p: p.index)
                    for p in ps_sorted[:per_doc_filter_k]:
                        pooled.append((p, tokenize(p.content)))
                elif per_doc_filter == "overlap":
                    # Cheap heuristic: query term overlap count.
                    scored_h: List[tuple[int, int, Passage, List[str]]] = []
                    for p in ps:
                        toks = tokenize(p.content)
                        overlap = sum(1 for t in toks if t in q_set)
                        scored_h.append((int(overlap), int(p.index), p, toks))
                    scored_h.sort(key=lambda x: (-x[0], x[1]))
                    for _overlap, _idx, p, toks in scored_h[:per_doc_filter_k]:
                        pooled.append((p, toks))
                else:
                    raise ValueError(f"Unknown passage_retrieval.per_doc_filter: {per_doc_filter!r} (use overlap|first_k)")

            if not pooled:
                results_by_topic[q.id] = []
                continue

            pooled_tokens = [toks for _p, toks in pooled]
            df = compute_df(pooled_tokens)
            bg = compute_bg_prob(pooled_tokens)
            n_docs = len(pooled_tokens)
            avgdl = sum(len(t) for t in pooled_tokens) / float(n_docs) if n_docs else 0.0

            scored: List[tuple[float, Passage]] = []
            for p, toks in pooled:
                tf = Counter(toks)
                dl = len(toks)
                s = _score_one(tf, dl, df=df, bg=bg, n_docs=n_docs, avgdl=avgdl)
                scored.append((s, Passage(document_id=p.document_id, index=p.index, content=p.content, score=s)))

            scored.sort(key=lambda x: (-x[0], x[1].document_id, x[1].index))
            results_by_topic[q.id] = [p for _s, p in scored[:topk]]
        else:
            # Global mode: score across all passages.
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
    if per_doc:
        mode_str = f", mode=per_doc(filter={per_doc_filter},k={per_doc_filter_k})"
    else:
        mode_str = ", mode=global"

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
