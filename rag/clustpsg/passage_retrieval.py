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

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from rag.config import ApproachConfig
from rag.lucene_backend import search as lucene_search
from rag.lucene_backend import set_bm25 as lucene_set_bm25
from rag.lucene_backend import set_rm3 as lucene_set_rm3
from rag.types import Passage, Query

from rag.clustpsg.text_scoring import (
    bm25_score,
    compute_bg_prob,
    compute_df,
    qld_score,
    tokenize,
)

from rag.query_expansion import expand_query_terms_semantic



def _passage_pid(p: Passage) -> str:
    # Unique + reversible-ish id for mapping Lucene hits back to passages.
    return f"{p.document_id}__p{int(p.index)}"


def _fingerprint_passages(passages: Sequence[Passage]) -> str:
    """Stable-ish fingerprint of a passage set (content-sensitive)."""
    h = hashlib.sha256()
    for p in passages:
        h.update(str(p.document_id).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(str(int(p.index)).encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update((p.content or "").encode("utf-8", "ignore"))
        h.update(b"\n")
    return h.hexdigest()[:20]


def _ensure_lucene_index_for_passages(
    *,
    passages: Sequence[Passage],
    cache_dir: str,
    mode: str,
) -> Tuple[object, Optional[tempfile.TemporaryDirectory]]:
    """Build a Lucene index over the given passages and return a LuceneSearcher.

    Modes:
      - \"temp\"  : build in a temporary directory (deleted after use)   [A]
      - \"cache\" : build under cache_dir using a content fingerprint     [B]
    """
    from pyserini.search.lucene import LuceneSearcher

    mode = str(mode or "cache").lower()
    if mode not in ("temp", "cache"):
        raise ValueError(f"Unknown passage_lucene.mode: {mode!r} (use 'temp' or 'cache')")

    if mode == "temp":
        tmp = tempfile.TemporaryDirectory(prefix="passage_lucene_")
        base = tmp.name
        collection_dir = os.path.join(base, "collection")
        index_dir = os.path.join(base, "index")
        os.makedirs(collection_dir, exist_ok=True)

        jsonl_path = os.path.join(collection_dir, "passages.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in passages:
                rec = {"id": _passage_pid(p), "contents": p.content}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            collection_dir,
            "--index",
            index_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storeRaw",
            "--storeDocvectors",
            "--storePositions",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return LuceneSearcher(index_dir), tmp

    # mode == "cache"
    os.makedirs(cache_dir, exist_ok=True)
    fp = _fingerprint_passages(passages)
    base = os.path.join(cache_dir, f"qpassages_v2_{fp}")
    collection_dir = os.path.join(base, "collection")
    index_dir = os.path.join(base, "index")
    meta_path = os.path.join(base, "meta.json")

    if not os.path.exists(index_dir):
        os.makedirs(collection_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        jsonl_path = os.path.join(collection_dir, "passages.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in passages:
                rec = {"id": _passage_pid(p), "contents": p.content}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            collection_dir,
            "--index",
            index_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storeRaw",
            "--storeDocvectors",
            "--storePositions",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        meta = {"fingerprint": fp, "n_passages": int(len(passages))}
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        except OSError:
            # Best-effort; index dir is the source of truth.
            pass

    return LuceneSearcher(index_dir), None


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

        q_terms = expand_query_terms_semantic(query_terms=q_terms, cfg=config)
        query_str = " ".join(q_terms)

        if model in ("lucene_bm25", "pyserini_bm25", "lucene_bm25+rm3", "lucene_bm25_rm3", "lucene_rm3"):
            # Passage ranking via Pyserini BM25 over a per-query Lucene index.
            lucene_cfg = (model_cfg.get("lucene") or {}) if isinstance(model_cfg.get("lucene"), dict) else {}
            # (A) "temp" or (B) "cache" (default)
            lucene_mode = str(lucene_cfg.get("mode", model_cfg.get("lucene_mode", "cache"))).lower()
            lucene_cache_dir = str(lucene_cfg.get("cache_dir", model_cfg.get("lucene_cache_dir", "cache/passage_lucene")))
            # Requirement-default: build the searcher over *all extracted passages per query*.
            lucene_use_all = bool(lucene_cfg.get("use_all_passages", model_cfg.get("lucene_use_all_passages", True)))

            candidate_passages = passages
            if (not lucene_use_all) and per_doc:
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
                        ps_sorted = sorted(ps, key=lambda p: p.index)
                        for p in ps_sorted[:per_doc_filter_k]:
                            pooled.append((p, tokenize(p.content)))
                    elif per_doc_filter == "overlap":
                        scored_h: List[tuple[int, int, Passage, List[str]]] = []
                        for p in ps:
                            toks = tokenize(p.content)
                            overlap = sum(1 for t in toks if t in q_set)
                            scored_h.append((int(overlap), int(p.index), p, toks))
                        scored_h.sort(key=lambda x: (-x[0], x[1]))
                        for _overlap, _idx, p, toks in scored_h[:per_doc_filter_k]:
                            pooled.append((p, toks))
                    else:
                        raise ValueError(
                            f"Unknown passage_retrieval.per_doc_filter: {per_doc_filter!r} (use overlap|first_k)"
                        )

                candidate_passages = [p for p, _toks in pooled]

            pid_to_passage = {_passage_pid(p): p for p in candidate_passages}
            if not pid_to_passage:
                results_by_topic[q.id] = []
                continue

            searcher, tmp = _ensure_lucene_index_for_passages(
                passages=candidate_passages, cache_dir=lucene_cache_dir, mode=lucene_mode
            )
            try:
                k1 = float(model_cfg.get("k1", 0.9))
                b = float(model_cfg.get("b", 0.4))
                lucene_set_bm25(searcher, k1=k1, b=b)

                # Optional: RM3 pseudo-relevance feedback over the passage index.
                rm3_cfg = (lucene_cfg.get("rm3") or {}) if isinstance(lucene_cfg.get("rm3"), dict) else {}
                rm3_enabled = bool(rm3_cfg.get("enabled", model in ("lucene_bm25+rm3", "lucene_bm25_rm3", "lucene_rm3")))
                if rm3_enabled:
                    fb_terms = int(rm3_cfg.get("fb_terms", model_cfg.get("rm3_fb_terms", 10)))
                    fb_docs = int(rm3_cfg.get("fb_docs", model_cfg.get("rm3_fb_docs", 10)))
                    oqw = float(rm3_cfg.get("original_query_weight", model_cfg.get("rm3_original_query_weight", 0.5)))
                    lucene_set_rm3(searcher, fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=oqw)

                hits = lucene_search(searcher, query_str, topk=topk)
                out: List[Passage] = []
                for h in hits:
                    p0 = pid_to_passage.get(h.docid)
                    if p0 is None:
                        continue
                    out.append(Passage(document_id=p0.document_id, index=p0.index, content=p0.content, score=float(h.score)))

                results_by_topic[q.id] = out
            finally:
                if tmp is not None:
                    tmp.cleanup()
            continue

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
        "Ranked passages for %d queries (topk=%d, model=%s, passages=%d%s%s) in %.2fs.",
        len(queries),
        topk,
        model,
        total_passages,
        stage_str,
        mode_str,
        dt,
    )
    return results_by_topic
