"""ANN utilities for Approach 3 (Stage 1 dense recall).

This module provides a small abstraction over ANN backends to keep dependencies optional.

Supported backends:
- **hnswlib**: fast ANN with on-disk index caching (optional dependency)
- **exact**: debug fallback (full dot-product scan; slow; no ANN structure)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class AnnIndexPaths:
    index_path: str
    meta_json: str


def default_ann_paths(*, out_dir: str, index_name: str, model_name: str, backend: str) -> AnnIndexPaths:
    safe_index = index_name.replace("/", "_")
    safe_model = model_name.replace("/", "_")
    safe_backend = backend.replace("/", "_")
    index_path = os.path.join(out_dir, f"ann_{safe_index}__{safe_model}__{safe_backend}.bin")
    meta_json = os.path.join(out_dir, f"ann_meta_{safe_index}__{safe_model}__{safe_backend}.json")
    return AnnIndexPaths(index_path=index_path, meta_json=meta_json)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_docids(docids_txt: str) -> List[str]:
    """Load docids list (one docid per line)."""
    out: List[str] = []
    with open(docids_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            d = line.strip()
            if not d:
                continue
            out.append(d)
    return out


def _load_embeddings_mmap(embeddings_npy: str):
    import numpy as np

    return np.load(embeddings_npy, mmap_mode="r")


def build_hnswlib_index(
    *,
    embeddings_npy: str,
    out_index_path: str,
    space: str = "ip",
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 128,
    num_threads: int = 1,
    force: bool = False,
) -> Tuple[int, int]:
    """Build a HNSWlib ANN index from an embeddings .npy matrix.

    Returns: (num_docs, dim)
    """
    if (not force) and os.path.exists(out_index_path):
        # Best effort: return shape from embeddings to avoid loading index.
        arr = _load_embeddings_mmap(embeddings_npy)
        return int(arr.shape[0]), int(arr.shape[1])

    try:
        import hnswlib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for ANN backend 'hnswlib'. Install hnswlib to build the ANN index."
        ) from e

    import numpy as np

    arr = _load_embeddings_mmap(embeddings_npy)
    n = int(arr.shape[0])
    dim = int(arr.shape[1])
    if n <= 0 or dim <= 0:
        raise ValueError(f"Invalid embeddings shape: {tuple(arr.shape)}")

    _ensure_dir(os.path.dirname(out_index_path) or ".")
    tmp = out_index_path + ".tmp"

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=int(ef_construction), M=int(M))

    # Determinism best-effort: single-thread index build.
    labels = np.arange(n, dtype=np.int64)
    index.add_items(arr, labels, num_threads=int(num_threads))
    index.set_ef(int(ef_search))

    index.save_index(tmp)
    os.replace(tmp, out_index_path)
    return n, dim


def load_hnswlib_index(*, index_path: str, dim: int, space: str = "ip"):
    """Load a HNSWlib index from disk."""
    try:
        import hnswlib  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency for ANN backend 'hnswlib'.") from e

    idx = hnswlib.Index(space=space, dim=int(dim))
    idx.load_index(index_path)
    return idx


def hnswlib_knn(
    *,
    index,
    query_vec,
    topk: int,
    ef_search: Optional[int] = None,
    num_threads: int = 1,
) -> Tuple[List[int], List[float]]:
    """Query a HNSWlib index. Returns (labels, scores) for a single query."""
    if ef_search is not None:
        try:
            index.set_ef(int(ef_search))
        except Exception:
            pass
    labels, dists = index.knn_query(query_vec, k=int(topk), num_threads=int(num_threads))
    # For space='ip', hnswlib returns distance = -inner_product? (depends on version).
    # We'll treat the second output as a score-like value and pass it through; callers can interpret.
    labels0 = [int(x) for x in labels[0].tolist()]
    scores0 = [float(x) for x in dists[0].tolist()]
    return labels0, scores0


def exact_topk(
    *,
    embeddings_npy: str,
    query_vec,
    topk: int,
) -> Tuple[List[int], List[float]]:
    """Exact top-k by full dot-product scan (debug fallback; slow)."""
    import numpy as np

    arr = _load_embeddings_mmap(embeddings_npy)
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != arr.shape[1]:
        raise ValueError(f"Query dim mismatch: query={q.shape[1]} embeddings={arr.shape[1]}")
    scores = (arr @ q.T).reshape(-1)
    k = int(topk)
    if k <= 0:
        return [], []
    if k >= scores.shape[0]:
        idx = np.argsort(-scores)
    else:
        idx_part = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx_part[np.argsort(-scores[idx_part])]
    labels = [int(i) for i in idx.tolist()]
    sc = [float(scores[i]) for i in idx.tolist()]
    return labels, sc


def write_ann_meta(path: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    _atomic_write_text(path, json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj

