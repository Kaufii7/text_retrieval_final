"""PR8: End-to-end clustpsg pipeline wiring (flag-gated from main.py).

Pipeline (per query):
1) Retrieve candidate documents (PR1)
2) Fetch document contents (required for passage extraction)
3) Extract passages (PR2)
4) Rank passages locally with BM25/QLD (PR3)
5) Cluster passages (PR4)
6) Build cluster-level features / instances (PR5/PR6)
7) Train/load SVM and score clusters (PR7)
8) Re-rank clusters -> re-rank passages -> RRF over passages -> score documents
9) Produce a reranked doc run (TREC run file handled by main.py)
"""

from __future__ import annotations

import logging
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rag.config import ApproachConfig, default_approach2_config
from rag.clustpsg.cluster import Cluster, cluster_passages
from rag.clustpsg.dataset import build_cluster_level_training_set
from rag.clustpsg.doc_retrieval import retrieve_doc_candidates
from rag.clustpsg.model import load_model, save_model, score_documents, train_model_from_config
from rag.clustpsg.passage_cache import load_ranked_passages, save_ranked_passages
from rag.clustpsg.passage_extraction import passages_by_topic
from rag.clustpsg.passage_retrieval import rank_passages
from rag.io import load_qrels
from rag.training_utils import augment_candidates_with_qrels
from rag.types import Document, Passage, Query


def _norm_scores(values: List[float], mode: str) -> List[float]:
    mode = (mode or "none").lower()
    if not values:
        return []
    if mode in ("none", ""):
        return list(values)
    if mode == "zscore":
        m_ = sum(values) / float(len(values))
        v_ = sum((x - m_) ** 2 for x in values) / float(len(values))
        s_ = math.sqrt(v_)
        if s_ <= 1e-12:
            return [0.0 for _ in values]
        return [(x - m_) / s_ for x in values]
    if mode == "minmax":
        mn, mx = min(values), max(values)
        if abs(mx - mn) <= 1e-12:
            return [0.0 for _ in values]
        return [(x - mn) / (mx - mn) for x in values]
    raise ValueError(f"Unknown normalization mode: {mode!r} (use none|zscore|minmax)")


def _rrf_doc_scores(passages_ranked: Sequence[Passage], *, k: int, depth: int) -> Dict[str, float]:
    """Reciprocal Rank Fusion over a ranked list of passages -> document scores."""
    if k < 0:
        k = 0
    if depth <= 0:
        depth = len(passages_ranked)
    out: Dict[str, float] = {}
    for r, p in enumerate(passages_ranked[:depth], start=1):
        out[p.document_id] = out.get(p.document_id, 0.0) + (1.0 / float(k + r))
    return out


def _rerank_passages_by_cluster_scores(
    ranked_passages: Sequence[Passage],
    *,
    clusters: Sequence[Cluster],
    cluster_scores_by_id: Mapping[int, float],
) -> List[Passage]:
    """Re-rank passages by descending cluster score, tie-breaking by original passage rank.

    Overlap handling: if a passage appears in multiple clusters, it takes the max score
    across those clusters (i.e., the highest-ranked cluster it appears in).
    """
    # Original rank is implied by list order (1 = best).
    orig_rank: Dict[tuple[str, int], int] = {(p.document_id, p.index): i for i, p in enumerate(ranked_passages, start=1)}

    best_score: Dict[tuple[str, int], float] = {}
    for cl in clusters:
        s = float(cluster_scores_by_id.get(int(cl.id), 0.0))
        for pi in cl.passage_indices:
            if 0 <= pi < len(ranked_passages):
                p = ranked_passages[pi]
                key = (p.document_id, p.index)
                prev = best_score.get(key)
                best_score[key] = s if prev is None else (s if s > prev else prev)

    # Deduplicate passages (should already be unique, but keep it robust).
    seen: set[tuple[str, int]] = set()
    unique: List[Passage] = []
    for p in ranked_passages:
        key = (p.document_id, p.index)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    unique.sort(key=lambda p: (-float(best_score.get((p.document_id, p.index), 0.0)), orig_rank.get((p.document_id, p.index), 10**9)))
    return unique


def _doc_scores_from_ranked_passages_rr(
    passages_ranked: Sequence[Passage],
    *,
    max_passages_per_doc: int,
) -> tuple[Dict[str, float], Dict[str, int]]:
    """Sum reciprocal ranks of passages per document, capped to M passages per doc.

    Uses the *global* passage rank (position in `passages_ranked`), per the ClustPsg spec.
    """
    if max_passages_per_doc <= 0:
        return {}, {}
    score: Dict[str, float] = {}
    used: Dict[str, int] = {}
    for r, p in enumerate(passages_ranked, start=1):
        docid = p.document_id
        cnt = used.get(docid, 0)
        if cnt >= max_passages_per_doc:
            continue
        score[docid] = score.get(docid, 0.0) + (1.0 / float(r))
        used[docid] = cnt + 1
    return score, used


def _adaptive_lambda(*, used_passages: int, max_passages_per_doc: int, lambda_min: float, lambda_max: float) -> float:
    """Adaptive lambda based on how many top passages a document received (0..M).

    Simple linear schedule:
      lambda = lambda_min + (lambda_max - lambda_min) * (used / M)
    """
    if max_passages_per_doc <= 0:
        return float(lambda_min)
    x = float(used_passages) / float(max_passages_per_doc)
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    lam = float(lambda_min) + (float(lambda_max) - float(lambda_min)) * x
    if lam < 0.0:
        return 0.0
    if lam > 1.0:
        return 1.0
    return lam


def _fetch_doc_contents(searcher, docid: str) -> str:
    """Best-effort raw content fetch for passage extraction."""
    d = searcher.doc(docid)
    if d is None:
        return ""
    # Prefer actual contents if available; raw() is often JSON and is poor for sentence splitting.
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


def _with_contents(
    *,
    searcher,
    docs: Sequence[Document],
    topk: int,
) -> List[Document]:
    out: List[Document] = []
    for i, d in enumerate(docs):
        if i >= topk:
            break
        out.append(Document(id=d.id, content=_fetch_doc_contents(searcher, d.id), score=d.score))
    return out


def _docs_from_qrels(
    *,
    searcher,
    qrels_topic: Mapping[str, int],
    topk: int | None,
) -> List[Document]:
    """Fetch Document objects for all docids in a qrels topic (deterministic order).

    qrels_topic maps docid -> relevance.
    """
    docids = sorted(qrels_topic.keys())
    if topk is not None:
        docids = docids[: int(topk)]
    out: List[Document] = []
    for docid in docids:
        out.append(Document(id=docid, content=_fetch_doc_contents(searcher, docid), score=None))
    return out


def clustpsg_run(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int,
    config: Optional[ApproachConfig] = None,
    split: str,
    qrels_path: str = "qrels_50_Queries",
    train_model: bool = False,
    precompute_only: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """Run clustpsg and return per-topic reranked documents.

    Returns:
      dict[topic_id] -> list of (docid, score) tuples.
    """
    log = logger or logging.getLogger("rag.clustpsg.pipeline")
    cfg = config or default_approach2_config()
    params = cfg.params or {}

    cache_cfg = (params.get("passage_cache") or {})
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_dir = str(cache_cfg.get("dir", "cache/ranked_passages"))

    # Limits to keep runtime sane during development.
    doc_content_topk = int(params.get("doc_content_topk", 100))
    train_doc_content_topk = params.get("train_doc_content_topk", None)
    if train_doc_content_topk is None:
        train_doc_content_topk = doc_content_topk
    else:
        train_doc_content_topk = int(train_doc_content_topk)
    clustering_max_passages = int(params.get("clustering_max_passages", 200))
    train_include_all_qrels_docs = bool(params.get("train_include_all_qrels_docs", True))
    doc_candidates_depth = int(params.get("doc_candidates_depth", topk))
    if doc_candidates_depth < topk:
        doc_candidates_depth = topk
    train_docs_source = str(params.get("train_docs_source", "qrels")).lower()

    qrels = load_qrels(qrels_path) if split == "train" else {}

    # --- Training set construction (can be different from inference candidates) ---
    train_docs_by_topic: Dict[int, List[Document]] = {}
    train_ranked_passages_by_topic: Dict[int, List] | None = None
    if split == "train":
        if train_docs_source == "qrels":
            # Use judged documents only (both relevant and non-relevant).
            # If passage cache is enabled and available, we can skip fetching contents entirely.
            train_docids_by_topic: Dict[int, List[str]] = {}
            for topic_id in sorted(qrels.keys()):
                docids = sorted((qrels.get(topic_id, {}) or {}).keys())
                if train_doc_content_topk is not None:
                    docids = docids[: int(train_doc_content_topk)]
                train_docids_by_topic[topic_id] = docids

            if cache_enabled:
                cached = load_ranked_passages(
                    cache_dir=cache_dir,
                    stage="train",
                    queries=queries,
                    docids_by_topic=train_docids_by_topic,
                    topk=clustering_max_passages,
                    cfg=cfg,
                )
                if cached is not None:
                    train_ranked_passages_by_topic = cached
                    # Keep doc list for doc-rank features; contents aren't needed if passages are cached.
                    for topic_id, docids in train_docids_by_topic.items():
                        train_docs_by_topic[topic_id] = [Document(id=d, content="", score=None) for d in docids]
                    log.info("Loaded cached ranked passages (stage=train) from %s", cache_dir)
                else:
                    log.info("No cached ranked passages found (stage=train); computing and caching to %s", cache_dir)

            if train_ranked_passages_by_topic is None:
                # Cache miss or caching disabled -> fetch doc contents and compute passages.
                for topic_id in sorted(qrels.keys()):
                    train_docs_by_topic[topic_id] = _docs_from_qrels(
                        searcher=searcher,
                        qrels_topic=qrels.get(topic_id, {}),
                        topk=train_doc_content_topk,
                    )
        elif train_docs_source == "retrieved":
            # Train from retrieved candidates (optionally augmented with qrels docids).
            train_docs_by_topic = retrieve_doc_candidates(
                queries=queries, searcher=searcher, topk=doc_candidates_depth, config=cfg, logger=log
            )
            if train_include_all_qrels_docs:
                train_docs_by_topic = augment_candidates_with_qrels(candidates_by_topic=train_docs_by_topic, qrels=qrels)
            # If passages are cached, skip content fetch; otherwise fetch contents for capped subset.
            train_docids_by_topic = {tid: [d.id for d in train_docs_by_topic.get(tid, [])[: int(train_doc_content_topk)]] for tid in sorted(train_docs_by_topic.keys())}
            if cache_enabled:
                cached = load_ranked_passages(
                    cache_dir=cache_dir,
                    stage="train",
                    queries=queries,
                    docids_by_topic=train_docids_by_topic,
                    topk=clustering_max_passages,
                    cfg=cfg,
                )
                if cached is not None:
                    train_ranked_passages_by_topic = cached
                    log.info("Loaded cached ranked passages (stage=train) from %s", cache_dir)
                else:
                    log.info("No cached ranked passages found (stage=train); computing and caching to %s", cache_dir)
            if train_ranked_passages_by_topic is None:
                for topic_id in sorted(train_docs_by_topic.keys()):
                    train_docs_by_topic[topic_id] = _with_contents(
                        searcher=searcher,
                        docs=train_docs_by_topic[topic_id],
                        topk=train_doc_content_topk,
                    )
        else:
            raise ValueError(f"Unknown train_docs_source: {train_docs_source!r} (use 'qrels' or 'retrieved')")

    # --- Inference candidates for producing a run (train/test) ---
    doc_candidates_by_topic = retrieve_doc_candidates(
        queries=queries, searcher=searcher, topk=doc_candidates_depth, config=cfg, logger=log
    )

    # 2-4) Fetch contents -> extract passages -> rank passages (with optional cache)
    cap = train_doc_content_topk if split == "train" else doc_content_topk
    infer_docids_by_topic: Dict[int, List[str]] = {
        tid: [d.id for d in (doc_candidates_by_topic.get(tid, [])[: int(cap)])]
        for tid in sorted(doc_candidates_by_topic.keys())
    }
    ranked_passages_by_topic = None
    if cache_enabled:
        ranked_passages_by_topic = load_ranked_passages(
            cache_dir=cache_dir,
            stage="inference",
            queries=queries,
            docids_by_topic=infer_docids_by_topic,
            topk=clustering_max_passages,
            cfg=cfg,
        )
        if ranked_passages_by_topic is not None:
            log.info("Loaded cached ranked passages (stage=inference) from %s", cache_dir)
        else:
            log.info("No cached ranked passages found (stage=inference); computing and caching to %s", cache_dir)

    if ranked_passages_by_topic is None:
        docs_with_content_by_topic: Dict[int, List[Document]] = {}
        for topic_id in sorted(doc_candidates_by_topic.keys()):
            docs = doc_candidates_by_topic[topic_id]
            docs_with_content_by_topic[topic_id] = _with_contents(searcher=searcher, docs=docs, topk=int(cap))

        extracted_passages_by_topic = passages_by_topic(docs_with_content_by_topic, cfg=cfg)

        ranked_passages_by_topic = rank_passages(
            queries=queries,
            passages_by_topic=extracted_passages_by_topic,
            topk=clustering_max_passages,
            config=cfg,
            logger=log,
            stage="inference",
        )
        if cache_enabled:
            save_ranked_passages(
                cache_dir=cache_dir,
                stage="inference",
                queries=queries,
                docids_by_topic=infer_docids_by_topic,
                topk=clustering_max_passages,
                cfg=cfg,
                ranked_passages_by_topic=ranked_passages_by_topic,
            )

    # If we're only precomputing caches, stop here (no need for clustering/SVM).
    if precompute_only:
        if split == "train":
            # Ensure train cache is also populated
            if train_ranked_passages_by_topic is None:
                train_passages = passages_by_topic(train_docs_by_topic, cfg=cfg)
                train_ranked_passages = rank_passages(
                    queries=queries,
                    passages_by_topic=train_passages,
                    topk=clustering_max_passages,
                    config=cfg,
                    logger=log,
                    stage="train",
                )
                # docids used for extraction for train
                train_docids_for_cache = {tid: [d.id for d in train_docs_by_topic.get(tid, [])] for tid in sorted(train_docs_by_topic.keys())}
                if cache_enabled:
                    save_ranked_passages(
                        cache_dir=cache_dir,
                        stage="train",
                        queries=queries,
                        docids_by_topic=train_docids_for_cache,
                        topk=clustering_max_passages,
                        cfg=cfg,
                        ranked_passages_by_topic=train_ranked_passages,
                    )
        return {}

    # 5) Cluster passages per topic (over the ranked list)
    clustering_cfg = params.get("clustering", {}) or {}
    clusters_by_topic = {}
    for q in queries:
        plist = ranked_passages_by_topic.get(q.id, [])
        clusters_by_topic[q.id] = cluster_passages(plist, cfg=clustering_cfg)

    # 6) Build cluster-level instances:
    # - training instances come from the chosen training doc source
    # - inference instances come from retrieved candidates and do not need qrels
    train_instances: List = []
    feature_names: List[str] = []
    if split == "train":
        # Build a training pipeline over the training docs (can be cached)
        if train_ranked_passages_by_topic is None:
            train_passages = passages_by_topic(train_docs_by_topic, cfg=cfg)
            train_ranked_passages = rank_passages(
                queries=queries,
                passages_by_topic=train_passages,
                topk=clustering_max_passages,
                config=cfg,
                logger=log,
                stage="train",
            )
            train_ranked_passages_by_topic = train_ranked_passages
            if cache_enabled:
                train_docids_for_cache = {tid: [d.id for d in train_docs_by_topic.get(tid, [])] for tid in sorted(train_docs_by_topic.keys())}
                save_ranked_passages(
                    cache_dir=cache_dir,
                    stage="train",
                    queries=queries,
                    docids_by_topic=train_docids_for_cache,
                    topk=clustering_max_passages,
                    cfg=cfg,
                    ranked_passages_by_topic=train_ranked_passages_by_topic,
                )
        train_clusters_by_topic = {}
        for q in queries:
            plist = train_ranked_passages_by_topic.get(q.id, [])
            train_clusters_by_topic[q.id] = cluster_passages(plist, cfg=clustering_cfg)
        train_instances, feature_names = build_cluster_level_training_set(
            queries=queries,
            qrels=qrels,
            doc_candidates_by_topic=train_docs_by_topic,
            # Use retrieval candidates to define document rank features consistently with inference.
            doc_rank_candidates_by_topic=doc_candidates_by_topic,
            ranked_passages_by_topic=train_ranked_passages_by_topic,
            clusters_by_topic=train_clusters_by_topic,
            config=cfg,
        )

    instances, _feature_names_infer = build_cluster_level_training_set(
        queries=queries,
        qrels={},  # no labels needed for inference scoring
        doc_candidates_by_topic=doc_candidates_by_topic,
        ranked_passages_by_topic=ranked_passages_by_topic,
        clusters_by_topic=clusters_by_topic,
        config=cfg,
    )
    if not feature_names:
        feature_names = _feature_names_infer

    # 7) Train/load model
    svm_cfg = (params.get("svm") or {})
    model_path = str(svm_cfg.get("model_path", "models/clustpsg_svm.pkl"))
    if split == "train" and train_model:
        m = train_model_from_config(instances=train_instances, feature_names=feature_names, svm_cfg=svm_cfg)
        save_model(m, model_path)
        log.info("Saved clustpsg model metadata to %s (backend=%s)", model_path, m.model_type)
    m = load_model(model_path)

    # 8) Score clusters -> propagate to passages -> RRF -> doc scores
    cluster_scores = score_documents(model=m, instances=instances)  # dict[(topic_id, "cl_<id>")] -> score

    final_cfg = (params.get("final") or {})
    max_passages_per_doc = int(final_cfg.get("max_passages_per_doc", 3))
    lambda_min = float(final_cfg.get("lambda_min", 0.2))
    lambda_max = float(final_cfg.get("lambda_max", 0.8))

    rerank_cfg = (params.get("rerank") or {})
    rerank_topn = int(rerank_cfg.get("topn", 1000))
    if rerank_topn < 0:
        rerank_topn = 0

    run: Dict[int, List[Tuple[str, float]]] = {}
    for q in queries:
        topic_id = q.id
        docs = list(doc_candidates_by_topic.get(topic_id, []))
        ranked_passages = list(ranked_passages_by_topic.get(topic_id, []))
        clusters = list(clusters_by_topic.get(topic_id, []))

        orig_doc_rank = {d.id: i for i, d in enumerate(docs, start=1)}
        bm25_rr = {d.id: (1.0 / float(orig_doc_rank[d.id])) for d in docs if d.id in orig_doc_rank}

        # Cluster.id -> score (score_documents uses "cl_<id>")
        cs_by_id: Dict[int, float] = {cl.id: float(cluster_scores.get((topic_id, f"cl_{cl.id}"), 0.0)) for cl in clusters}

        reranked_passages = _rerank_passages_by_cluster_scores(
            ranked_passages,
            clusters=clusters,
            cluster_scores_by_id=cs_by_id,
        )

        passage_rr, used_counts = _doc_scores_from_ranked_passages_rr(
            reranked_passages,
            max_passages_per_doc=max_passages_per_doc,
        )

        # Score documents by adaptive fusion: lambda * passage_rr + (1-lambda) * bm25_rr
        fused: Dict[str, float] = {}
        for d in docs:
            used = int(used_counts.get(d.id, 0))
            lam = _adaptive_lambda(
                used_passages=used,
                max_passages_per_doc=max_passages_per_doc,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            )
            fused[d.id] = lam * float(passage_rr.get(d.id, 0.0)) + (1.0 - lam) * float(bm25_rr.get(d.id, 0.0))

        # Only rerank within top-N docs for safety; keep tail in original BM25 order.
        head = docs[:rerank_topn]
        tail = docs[rerank_topn:]

        head_pairs = [(d.id, float(fused.get(d.id, 0.0))) for d in head]
        head_pairs.sort(key=lambda x: (-x[1], orig_doc_rank.get(x[0], 10**9)))

        # Tail: keep original order, use BM25 reciprocal-rank as a deterministic score.
        tail_pairs = [(d.id, float(bm25_rr.get(d.id, 0.0))) for d in tail]

        run[topic_id] = (head_pairs + tail_pairs)[:topk]

    return run


