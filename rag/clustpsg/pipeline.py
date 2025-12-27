"""PR8: End-to-end clustpsg pipeline wiring (flag-gated from main.py).

Pipeline (per query):
1) Retrieve candidate documents (PR1)
2) Fetch document contents (required for passage extraction)
3) Extract passages (PR2)
4) Rank passages locally with BM25/QLD (PR3)
5) Cluster passages (PR4)
6) Compute cluster features and aggregate to documents (PR5/PR6)
7) Train/load SVM and score documents (PR7)
8) Produce a reranked doc run (TREC run file handled by main.py)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rag.config import ApproachConfig, default_approach2_config
from rag.clustpsg.cluster import cluster_passages
from rag.clustpsg.dataset import build_doc_level_training_set
from rag.clustpsg.doc_retrieval import retrieve_doc_candidates
from rag.clustpsg.model import load_model, save_model, score_documents, train_linear_svm
from rag.clustpsg.passage_extraction import passages_by_topic
from rag.clustpsg.passage_retrieval import rank_passages
from rag.io import load_qrels
from rag.types import Document, Query


def _fetch_doc_contents(searcher, docid: str) -> str:
    """Best-effort raw content fetch for passage extraction."""
    d = searcher.doc(docid)
    if d is None:
        return ""
    try:
        raw = d.raw()
    except Exception:
        raw = ""
    return raw or ""


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


def clustpsg_run(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int,
    config: Optional[ApproachConfig] = None,
    split: str,
    qrels_path: str = "qrels_50_Queries",
    train_model: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """Run clustpsg and return per-topic reranked documents.

    Returns:
      dict[topic_id] -> list of (docid, score) tuples.
    """
    log = logger or logging.getLogger("rag.clustpsg.pipeline")
    cfg = config or default_approach2_config()
    params = cfg.params or {}

    # Limits to keep runtime sane during development.
    doc_content_topk = int(params.get("doc_content_topk", 100))
    clustering_max_passages = int(params.get("clustering_max_passages", 200))

    # 1) Candidate documents
    doc_candidates_by_topic = retrieve_doc_candidates(
        queries=queries, searcher=searcher, topk=topk, config=cfg, logger=log
    )

    # 2) Fetch doc contents for passage extraction (cap by doc_content_topk)
    docs_with_content_by_topic: Dict[int, List[Document]] = {}
    for topic_id in sorted(doc_candidates_by_topic.keys()):
        docs = doc_candidates_by_topic[topic_id]
        docs_with_content_by_topic[topic_id] = _with_contents(searcher=searcher, docs=docs, topk=doc_content_topk)

    # 3) Extract passages
    extracted_passages_by_topic = passages_by_topic(docs_with_content_by_topic, cfg=cfg)

    # 4) Rank passages locally (BM25/QLD over candidate set)
    ranked_passages_by_topic = rank_passages(
        queries=queries, passages_by_topic=extracted_passages_by_topic, topk=clustering_max_passages, config=cfg, logger=log
    )

    # 5) Cluster passages per topic (over the ranked list)
    clustering_cfg = params.get("clustering", {}) or {}
    clusters_by_topic = {}
    for q in queries:
        plist = ranked_passages_by_topic.get(q.id, [])
        clusters_by_topic[q.id] = cluster_passages(plist, cfg=clustering_cfg)

    # 6) Build doc-level instances (labels only needed for training/eval)
    qrels = load_qrels(qrels_path) if split == "train" else {}
    instances, feature_names = build_doc_level_training_set(
        queries=queries,
        qrels=qrels,
        doc_candidates_by_topic=doc_candidates_by_topic,
        ranked_passages_by_topic=ranked_passages_by_topic,
        clusters_by_topic=clusters_by_topic,
        config=cfg,
    )

    # 7) Train/load model
    svm_cfg = (params.get("svm") or {})
    model_path = str(svm_cfg.get("model_path", "models/clustpsg_svm.pkl"))
    if split == "train" and train_model:
        m = train_linear_svm(
            instances=instances,
            feature_names=feature_names,
            C=float(svm_cfg.get("C", 1.0)),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            max_iter=int(svm_cfg.get("max_iter", 5000)),
            random_state=int(svm_cfg.get("random_state", 42)),
        )
        save_model(m, model_path)
        log.info("Saved clustpsg SVM model to %s", model_path)
    m = load_model(model_path)

    # 8) Score docs and produce reranked run
    scores = score_documents(model=m, instances=instances)
    run: Dict[int, List[Tuple[str, float]]] = {}
    for topic_id, docs in doc_candidates_by_topic.items():
        pairs: List[Tuple[str, float]] = []
        for d in docs:
            pairs.append((d.id, float(scores.get((topic_id, d.id), 0.0))))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        run[topic_id] = pairs[:topk]

    return run


