"""Approach 3: two-stage retrieval (dense recall -> optional cross-encoder rerank).

Stage 1 (recall):
- Embed the query with a bi-encoder and retrieve candidates from precomputed
  document embeddings using an ANN index (hnswlib) or an exact fallback.

Stage 2 (precision, optional):
- Score (query, doc_text) pairs with a cross-encoder and rerank the top-N.

Output contract (important):
- Return `results_by_topic` compatible with `rag.runs.write_trec_run`, i.e.
    dict[int topic_id] -> list of results
  where each result is either:
    - (docid, score) tuple, OR
    - dict-like with keys {"docid": ..., "score": ...}
- Determinism: for a given query/config, scores should be stable/reproducible.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rag.config import ApproachConfig, default_approach3_config
from rag.lucene_backend import fetch_doc_contents
from rag.types import Query


def _get_param(d: Mapping[str, object], key: str, default):
    v = d.get(key, default)
    return default if v is None else v


def _load_dense_index(params: Mapping[str, object]):
    """Load (or build+load) the dense retrieval index according to config params."""
    dense = params.get("dense") or {}
    if not isinstance(dense, Mapping):
        raise ValueError("approach3 params['dense'] must be a mapping")

    backend = str(_get_param(dense, "backend", "hnswlib")).lower()
    metric = str(_get_param(dense, "metric", "cosine")).lower()
    embeddings_path = str(_get_param(dense, "embeddings_path", ""))
    docids_path = str(_get_param(dense, "docids_path", ""))
    if not embeddings_path or not docids_path:
        raise ValueError("approach3 dense config must include embeddings_path and docids_path")

    from rag.approach3.ann_index import build_hnsw_index, load_exact_index, load_hnsw_index

    if backend == "exact":
        return load_exact_index(embeddings_path=embeddings_path, docids_path=docids_path, metric=metric)

    # hnswlib preferred
    hnsw_index_path = str(_get_param(dense, "hnsw_index_path", ""))
    hnsw_meta_path = str(_get_param(dense, "hnsw_meta_path", ""))
    build_if_missing = bool(_get_param(dense, "build_index_if_missing", False))
    hnsw = dense.get("hnsw") or {}
    if not isinstance(hnsw, Mapping):
        hnsw = {}
    ef_construction = int(_get_param(hnsw, "ef_construction", 200))
    M = int(_get_param(hnsw, "M", 16))
    seed = int(_get_param(hnsw, "seed", 42))

    if hnsw_index_path and hnsw_meta_path and os.path.exists(hnsw_index_path) and os.path.exists(hnsw_meta_path):
        return load_hnsw_index(hnsw_meta_path)

    if build_if_missing and hnsw_index_path and hnsw_meta_path:
        # Attempt to build the index (requires hnswlib). If missing, fall back to exact.
        try:
            os.makedirs(os.path.dirname(hnsw_index_path) or ".", exist_ok=True)
            build_hnsw_index(
                embeddings_path=embeddings_path,
                docids_path=docids_path,
                out_index_path=hnsw_index_path,
                out_meta_path=hnsw_meta_path,
                metric=metric,
                ef_construction=ef_construction,
                M=M,
                seed=seed,
                force=True,
            )
            return load_hnsw_index(hnsw_meta_path)
        except Exception:
            return load_exact_index(embeddings_path=embeddings_path, docids_path=docids_path, metric=metric)

    # No cached index and we didn't build it: use exact fallback to remain runnable.
    return load_exact_index(embeddings_path=embeddings_path, docids_path=docids_path, metric=metric)


def _embed_query(query: str, *, model_name: str, device: str, normalize_embeddings: bool):
    # Lazy import for heavy deps
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Approach 3 requires the optional dependency 'sentence-transformers' to embed queries. "
            "Install it to run Approach 3."
        ) from e

    model = SentenceTransformer(model_name, device=device)
    vec = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
    )[0]
    return vec


def _rerank_with_cross_encoder(
    *,
    query: str,
    searcher,
    candidates: Sequence[Tuple[str, float]],
    rerank_cfg: Mapping[str, object],
) -> Dict[str, float]:
    """Return CE scores for a subset of candidates (docid -> ce_score)."""
    enabled = bool(_get_param(rerank_cfg, "enabled", False))
    if not enabled:
        return {}

    topn = int(_get_param(rerank_cfg, "topn", 100))
    if topn <= 0:
        return {}
    use_finetuned = bool(_get_param(rerank_cfg, "use_finetuned", False))
    finetuned_dir = str(_get_param(rerank_cfg, "finetuned_model_dir", "models/approach3_ce/best"))
    base_model_name = str(_get_param(rerank_cfg, "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    model_name = finetuned_dir if use_finetuned else base_model_name
    device = str(_get_param(rerank_cfg, "device", "cpu"))
    batch_size = int(_get_param(rerank_cfg, "batch_size", 16))

    from rag.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name=model_name, device=device, batch_size=batch_size)

    # Build (docid, doc_text) for topn candidates
    chosen = candidates[:topn]
    pairs = []
    for docid, _s in chosen:
        text = fetch_doc_contents(searcher, docid)
        pairs.append((docid, text))

    reranked = reranker.rerank(query, pairs)
    return {r.docid: float(r.score) for r in reranked}


def approach3_retrieve(
    *,
    queries: Sequence[Query],
    searcher,
    topk: int = 1000,
    config: Optional[ApproachConfig] = None,
) -> Dict[int, List[Mapping[str, object]]]:
    """Run Approach 3 retrieval for all queries."""
    if config is None:
        config = default_approach3_config()
    params = config.params or {}

    log = logging.getLogger("rag.approaches.approach3")
    dense_cfg = params.get("dense") or {}
    if not isinstance(dense_cfg, Mapping):
        raise ValueError("approach3 params['dense'] must be a mapping")

    # Load the index once; reuse for all queries.
    index = _load_dense_index(params)

    candidates_depth = int(config.candidates_depth or max(int(topk), 1000))
    if candidates_depth < int(topk):
        candidates_depth = int(topk)

    model_name = str(_get_param(dense_cfg, "model_name", "sentence-transformers/all-mpnet-base-v2"))
    device = str(_get_param(dense_cfg, "device", "cpu"))
    normalize_embeddings = bool(_get_param(dense_cfg, "normalize_embeddings", True))
    ef = None
    hnsw_cfg = dense_cfg.get("hnsw") or {}
    if isinstance(hnsw_cfg, Mapping) and "ef" in hnsw_cfg and hnsw_cfg.get("ef") is not None:
        ef = int(hnsw_cfg.get("ef"))  # type: ignore[arg-type]

    rerank_cfg = params.get("rerank") or {}
    if not isinstance(rerank_cfg, Mapping):
        rerank_cfg = {}
    fusion_cfg = params.get("score_fusion") or {}
    if not isinstance(fusion_cfg, Mapping):
        fusion_cfg = {}
    alpha = float(_get_param(fusion_cfg, "alpha", 1.0))
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    results_by_topic: Dict[int, List[Mapping[str, object]]] = {}
    for q in queries:
        qvec = _embed_query(q.text, model_name=model_name, device=device, normalize_embeddings=normalize_embeddings)
        candidates = index.search(qvec, topk=candidates_depth, ef=ef)
        # candidates: list[(docid, dense_score)] sorted with tie-breaks

        ce_scores = _rerank_with_cross_encoder(query=q.text, searcher=searcher, candidates=candidates, rerank_cfg=rerank_cfg)

        # Final score: for reranked docs blend ce+dense; for others keep dense.
        dense_by_doc = {docid: float(s) for docid, s in candidates}
        final: List[Tuple[str, float]] = []
        seen = set()
        if ce_scores:
            for docid, ce in ce_scores.items():
                ds = dense_by_doc.get(docid, 0.0)
                final.append((docid, alpha * float(ce) + (1.0 - alpha) * float(ds)))
                seen.add(docid)

        for docid, ds in candidates:
            if docid in seen:
                continue
            final.append((docid, float(ds)))

        final.sort(key=lambda x: (-x[1], x[0]))
        final = final[: int(topk)]
        results_by_topic[q.topic_id] = [{"docid": docid, "score": float(score)} for docid, score in final]

        if not candidates:
            log.warning("No dense candidates for topic_id=%s", q.topic_id)

    return results_by_topic


