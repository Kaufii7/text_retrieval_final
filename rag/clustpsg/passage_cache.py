"""On-disk cache for ranked passages.

Goal: speed up iterative experiments by avoiding repeated:
  - document content fetch (if we can reuse cached passages with content)
  - passage extraction
  - local passage scoring (BM25/QLD) over many passages

Cache key is derived from:
  - stage ("train" | "inference")
  - query ids + query text
  - docids used for passage extraction (per topic, order preserved)
  - passage extraction params
  - passage retrieval params (model + hyperparameters)
  - topk passages requested
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict
from typing import Dict, List, Mapping, Sequence

from rag.config import ApproachConfig
from rag.types import Passage, Query


def _stable_hash(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_key(
    *,
    stage: str,
    queries: Sequence[Query],
    docids_by_topic: Mapping[int, Sequence[str]],
    topk: int,
    cfg: ApproachConfig,
) -> str:
    params = cfg.params or {}
    extraction_cfg = {
        "min_sentences": int(params.get("min_sentences", 3)),
        "max_sentences": int(params.get("max_sentences", 5)),
        "stride_sentences": int(params.get("stride_sentences", 2)),
        "max_chars_per_sentence_soft": int(params.get("max_chars_per_sentence_soft", 300)),
    }
    passage_retrieval_cfg = (params.get("passage_retrieval") or {})

    q_fingerprint = [(int(q.id), str(q.content)) for q in queries]
    docids_fingerprint = {int(t): list(map(str, docids_by_topic.get(t, []))) for t in sorted(docids_by_topic.keys())}

    payload = {
        "v": 1,
        "stage": str(stage),
        "topk": int(topk),
        "queries": q_fingerprint,
        "docids_by_topic": docids_fingerprint,
        "extraction": extraction_cfg,
        "passage_retrieval": passage_retrieval_cfg,
    }
    # shorter filename-friendly key
    return _stable_hash(payload)[:20]


def _cache_paths(cache_dir: str, key: str) -> tuple[str, str]:
    pkl = os.path.join(cache_dir, f"ranked_passages_{key}.pkl")
    meta = os.path.join(cache_dir, f"ranked_passages_{key}.meta.json")
    return pkl, meta


def load_ranked_passages(
    *,
    cache_dir: str,
    stage: str,
    queries: Sequence[Query],
    docids_by_topic: Mapping[int, Sequence[str]],
    topk: int,
    cfg: ApproachConfig,
) -> Dict[int, List[Passage]] | None:
    key = _cache_key(stage=stage, queries=queries, docids_by_topic=docids_by_topic, topk=topk, cfg=cfg)
    pkl_path, _meta_path = _cache_paths(cache_dir, key)
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return None
    return obj


def save_ranked_passages(
    *,
    cache_dir: str,
    stage: str,
    queries: Sequence[Query],
    docids_by_topic: Mapping[int, Sequence[str]],
    topk: int,
    cfg: ApproachConfig,
    ranked_passages_by_topic: Mapping[int, Sequence[Passage]],
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(stage=stage, queries=queries, docids_by_topic=docids_by_topic, topk=topk, cfg=cfg)
    pkl_path, meta_path = _cache_paths(cache_dir, key)

    payload = {int(t): list(ps) for t, ps in ranked_passages_by_topic.items()}
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "key": key,
        "stage": stage,
        "topk": int(topk),
        "n_topics": len(payload),
        "n_passages": int(sum(len(v) for v in payload.values())),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    return key


