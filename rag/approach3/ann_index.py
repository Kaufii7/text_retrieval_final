"""PR A3-3: ANN index + dense retrieval foundations (Stage 1 recall).

This module builds and queries an ANN index over precomputed document embeddings.

Backends:
- **hnswlib** (preferred): fast ANN; extra dependency; deterministic when seeded.
- **exact** (fallback): exact dot-product / cosine via numpy (debugging only).

All heavy/optional dependencies are imported lazily so the repo remains usable
without them unless this module is explicitly used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AnnIndexMeta:
    backend: str
    metric: str
    dim: int
    num_items: int
    index_path: str
    docids_path: str
    embeddings_path: str
    params: dict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_docids(docids_path: str) -> List[str]:
    docids: List[str] = []
    with open(docids_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            docids.append(s)
    return docids


def load_embeddings(embeddings_path: str):
    import numpy as np

    return np.load(embeddings_path, mmap_mode="r")


def _l2_normalize_rows(x):
    import numpy as np

    eps = 1e-12
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def build_hnsw_index(
    *,
    embeddings_path: str,
    docids_path: str,
    out_index_path: str,
    out_meta_path: str,
    metric: str = "ip",
    ef_construction: int = 200,
    M: int = 16,
    seed: int = 42,
    force: bool = False,
) -> AnnIndexMeta:
    """Build and persist an hnswlib index aligned with docids file order."""
    if (not force) and os.path.exists(out_index_path) and os.path.exists(out_meta_path):
        return load_index_meta(out_meta_path)

    try:
        import hnswlib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing optional dependency: hnswlib. Install it to build an ANN index, "
            "or use the exact fallback."
        ) from e

    import numpy as np

    emb = load_embeddings(embeddings_path)
    docids = load_docids(docids_path)
    if int(emb.shape[0]) != len(docids):
        raise ValueError(
            f"Embeddings/docids length mismatch: embeddings={int(emb.shape[0])} docids={len(docids)}"
        )
    dim = int(emb.shape[1])
    n = int(emb.shape[0])

    _ensure_dir(os.path.dirname(out_index_path) or ".")

    # hnswlib expects contiguous float32
    data = np.asarray(emb, dtype="float32")

    # If metric is cosine, normalize and use inner product.
    m = str(metric).lower()
    if m == "cosine":
        data = _l2_normalize_rows(data)
        hnsw_metric = "ip"
    else:
        hnsw_metric = m

    p = hnswlib.Index(space=hnsw_metric, dim=dim)
    p.init_index(max_elements=n, ef_construction=int(ef_construction), M=int(M))
    if hasattr(p, "set_seed"):
        p.set_seed(int(seed))
    p.add_items(data, list(range(n)))
    # Set a conservative default ef for queries; can be overridden at query time.
    if hasattr(p, "set_ef"):
        p.set_ef(max(50, int(M) * 2))
    p.save_index(out_index_path)

    meta = AnnIndexMeta(
        backend="hnswlib",
        metric=str(metric),
        dim=dim,
        num_items=n,
        index_path=out_index_path,
        docids_path=docids_path,
        embeddings_path=embeddings_path,
        params={"ef_construction": int(ef_construction), "M": int(M), "seed": int(seed)},
    )
    _atomic_write_text(out_meta_path, json.dumps(meta.__dict__, indent=2, sort_keys=True) + "\n")
    return meta


def load_index_meta(meta_path: str) -> AnnIndexMeta:
    with open(meta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return AnnIndexMeta(
        backend=str(obj["backend"]),
        metric=str(obj["metric"]),
        dim=int(obj["dim"]),
        num_items=int(obj["num_items"]),
        index_path=str(obj["index_path"]),
        docids_path=str(obj["docids_path"]),
        embeddings_path=str(obj["embeddings_path"]),
        params=dict(obj.get("params") or {}),
    )


@dataclass
class DenseIndex:
    """Uniform interface for ANN/exact search over doc embeddings aligned with docids file."""

    backend: str
    metric: str
    docids: List[str]
    dim: int

    # backend-specific:
    _hnsw: object | None = None
    _emb: object | None = None

    def search(self, query_vec, *, topk: int, ef: Optional[int] = None) -> List[Tuple[str, float]]:
        if int(topk) <= 0:
            return []
        if self.backend == "hnswlib":
            return self._search_hnsw(query_vec, topk=topk, ef=ef)
        if self.backend == "exact":
            return self._search_exact(query_vec, topk=topk)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _search_hnsw(self, query_vec, *, topk: int, ef: Optional[int]) -> List[Tuple[str, float]]:
        import numpy as np

        if self._hnsw is None:
            raise RuntimeError("HNSW index not loaded")
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if int(q.shape[1]) != int(self.dim):
            raise ValueError(f"Query dim mismatch: got={int(q.shape[1])} expected={int(self.dim)}")
        if ef is not None and hasattr(self._hnsw, "set_ef"):
            self._hnsw.set_ef(int(ef))
        labels, distances = self._hnsw.knn_query(q, k=int(topk))
        # hnswlib returns labels and distances; for ip metric distances are actually -ip? depends on backend.
        # We'll convert to a "score" where higher is better:
        # - For 'ip' we treat distance as similarity if hnswlib returns raw IP, otherwise we still get stable ranking.
        # - For 'l2' we use negative distance so higher is better.
        idxs = [int(x) for x in labels[0].tolist()]
        dists = [float(x) for x in distances[0].tolist()]
        out: List[Tuple[str, float]] = []
        for i, d in zip(idxs, dists):
            docid = self.docids[i]
            if str(self.metric).lower() == "l2":
                out.append((docid, -d))
            else:
                out.append((docid, d))
        # Deterministic tie-break
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def _search_exact(self, query_vec, *, topk: int) -> List[Tuple[str, float]]:
        import numpy as np

        if self._emb is None:
            raise RuntimeError("Embeddings not loaded for exact search")
        emb = self._emb
        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        if int(q.shape[0]) != int(self.dim):
            raise ValueError(f"Query dim mismatch: got={int(q.shape[0])} expected={int(self.dim)}")

        # Normalize if cosine metric requested.
        m = str(self.metric).lower()
        if m == "cosine":
            qn = np.linalg.norm(q) + 1e-12
            q = q / qn
            e = emb
            # If embeddings are not normalized already, normalize on the fly (debug-only).
            # This is expensive but acceptable for fallback mode.
            en = np.linalg.norm(e, axis=1, keepdims=True)
            en = np.maximum(en, 1e-12)
            e = e / en
        else:
            e = emb

        # Inner product (works for normalized cosine too).
        scores = e @ q
        # Topk via argpartition then exact sort with deterministic tie-break by docid
        k = int(topk)
        if k >= int(scores.shape[0]):
            idx = list(range(int(scores.shape[0])))
        else:
            idx = np.argpartition(-scores, kth=k - 1)[:k].tolist()
        out = [(self.docids[int(i)], float(scores[int(i)])) for i in idx]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out


def load_hnsw_index(meta_path: str) -> DenseIndex:
    meta = load_index_meta(meta_path)
    if meta.backend != "hnswlib":
        raise ValueError(f"Meta backend is not hnswlib: {meta.backend!r}")
    try:
        import hnswlib  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing optional dependency: hnswlib (required to load ANN index).") from e

    docids = load_docids(meta.docids_path)
    p = hnswlib.Index(space=("ip" if str(meta.metric).lower() in ("ip", "cosine") else str(meta.metric).lower()), dim=int(meta.dim))
    p.load_index(meta.index_path, max_elements=int(meta.num_items))
    # default ef (can be overridden per-query)
    if hasattr(p, "set_ef"):
        p.set_ef(max(50, int(meta.params.get("M", 16)) * 2))
    return DenseIndex(backend="hnswlib", metric=str(meta.metric), docids=docids, dim=int(meta.dim), _hnsw=p)


def load_exact_index(*, embeddings_path: str, docids_path: str, metric: str = "ip") -> DenseIndex:
    docids = load_docids(docids_path)
    emb = load_embeddings(embeddings_path)
    if int(emb.shape[0]) != len(docids):
        raise ValueError(f"Embeddings/docids length mismatch: embeddings={int(emb.shape[0])} docids={len(docids)}")
    return DenseIndex(backend="exact", metric=str(metric), docids=docids, dim=int(emb.shape[1]), _emb=emb)

