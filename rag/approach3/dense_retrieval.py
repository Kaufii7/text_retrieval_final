"""PR A3-3: Dense stage-1 retrieval (query embedding + ANN search).

This module assumes you have already built A3-1 assets:
- corpus JSONL
- docids txt
- embeddings npy

Then A3-3 can (optionally) build/load an ANN index and retrieve top-K docids for a query.

Heavy dependencies are imported lazily:
- `sentence-transformers` is required only when embedding queries.
- `hnswlib` is required only when using the ANN backend.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rag.approach3.ann_index import (
    DenseIndex,
    build_hnsw_index,
    load_exact_index,
    load_hnsw_index,
)


@dataclass(frozen=True)
class DenseRetrieveConfig:
    model_name: str
    device: str
    normalize_embeddings: bool
    metric: str
    backend: str
    ef: Optional[int] = None


def embed_query(*, query: str, model_name: str, device: str, normalize_embeddings: bool):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for dense retrieval: sentence-transformers. "
            "Install it to embed queries."
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


def retrieve_dense(
    *,
    query: str,
    index: DenseIndex,
    cfg: DenseRetrieveConfig,
    topk: int,
) -> List[Tuple[str, float]]:
    qvec = embed_query(
        query=query,
        model_name=cfg.model_name,
        device=cfg.device,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    return index.search(qvec, topk=int(topk), ef=cfg.ef)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dense stage-1 retrieval (A3-3): embed query -> ANN search.")
    p.add_argument("--query", required=True, help="Query string to retrieve for.")
    p.add_argument("--topk", type=int, default=10, help="How many results to return.")

    # Assets
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy from A3-1.")
    p.add_argument("--docids", required=True, help="Path to docids .txt aligned with embeddings.")

    # Model
    p.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2", help="Bi-encoder model name.")
    p.add_argument("--device", default="cpu", help="cpu|cuda")
    p.add_argument("--no-normalize", action="store_true", help="Disable query embedding normalization.")

    # Retrieval backend
    p.add_argument("--metric", default="ip", choices=["ip", "cosine", "l2"], help="Similarity metric.")
    p.add_argument("--backend", default="hnswlib", choices=["hnswlib", "exact"], help="ANN backend.")

    # hnsw specifics
    p.add_argument("--hnsw-index", default=None, help="Path to persist/load hnsw index file.")
    p.add_argument("--hnsw-meta", default=None, help="Path to persist/load hnsw meta json.")
    p.add_argument("--ef", type=int, default=None, help="HNSW ef at query time (higher=better recall, slower).")
    p.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction.")
    p.add_argument("--M", type=int, default=16, help="HNSW M parameter.")
    p.add_argument("--seed", type=int, default=42, help="Seed for HNSW build (if supported).")
    p.add_argument("--rebuild-index", action="store_true", help="Force rebuild HNSW index.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = DenseRetrieveConfig(
        model_name=str(args.model_name),
        device=str(args.device),
        normalize_embeddings=not bool(args.no_normalize),
        metric=str(args.metric),
        backend=str(args.backend),
        ef=args.ef if args.ef is None else int(args.ef),
    )

    if cfg.backend == "exact":
        index = load_exact_index(embeddings_path=str(args.embeddings), docids_path=str(args.docids), metric=cfg.metric)
    else:
        # hnswlib: build/load a persistent index if paths are provided; otherwise fall back to exact
        if not args.hnsw_index or not args.hnsw_meta:
            # No cache paths provided; build in-memory exact fallback to keep CLI usable.
            index = load_exact_index(embeddings_path=str(args.embeddings), docids_path=str(args.docids), metric=cfg.metric)
        else:
            hnsw_index = str(args.hnsw_index)
            hnsw_meta = str(args.hnsw_meta)
            if (not args.rebuild_index) and os.path.exists(hnsw_index) and os.path.exists(hnsw_meta):
                index = load_hnsw_index(hnsw_meta)
            else:
                build_hnsw_index(
                    embeddings_path=str(args.embeddings),
                    docids_path=str(args.docids),
                    out_index_path=hnsw_index,
                    out_meta_path=hnsw_meta,
                    metric=cfg.metric,
                    ef_construction=int(args.ef_construction),
                    M=int(args.M),
                    seed=int(args.seed),
                    force=bool(args.rebuild_index),
                )
                index = load_hnsw_index(hnsw_meta)

    results = retrieve_dense(query=str(args.query), index=index, cfg=cfg, topk=int(args.topk))
    for rank, (docid, score) in enumerate(results, start=1):
        print(f"{rank}\t{docid}\t{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

