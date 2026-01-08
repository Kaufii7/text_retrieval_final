"""PR A3-1: Dense indexing foundations (corpus extraction + embedding cache).

This module provides a **builder-only** CLI that creates reusable artifacts for
bi-encoder retrieval:

- A deterministic corpus JSONL: one line per doc: {"docid": "...", "text": "..."}
- A deterministic docid list aligned with the corpus (one docid per line)
- A cached embedding matrix for the chosen bi-encoder model

Design goals:
- Deterministic outputs (stable docid traversal, stable JSON writing, fixed options)
- Idempotent (skip work when artifacts exist unless --force)
- Merge-safe (heavy deps like sentence-transformers are only imported when needed)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _best_effort_doc_text(searcher, docid: str) -> str:
    """Backward-compatible wrapper (use rag.lucene_backend.fetch_doc_contents)."""
    from rag.lucene_backend import fetch_doc_contents

    return fetch_doc_contents(searcher, docid)


def _iter_docids(index_name: str) -> Iterable[str]:
    """Yield docids from a Pyserini prebuilt index.

    Notes:
    - We import Pyserini only inside this function to keep imports lightweight.
    - Docid order should be deterministic for a fixed index build.
    """
    from rag.lucene_backend import get_index_reader, get_searcher

    def _docid_from_lucene_document(doc) -> str:
        # Common field names across Anserini/Pyserini collections
        for k in ("id", "docid", "docno", "DOCNO"):
            try:
                v = doc.get(k)
            except Exception:
                v = None
            if isinstance(v, str) and v:
                return v
        return ""

    def _iter_from_java_reader(jreader) -> Iterable[str]:
        # Lucene IndexReader-style iteration via internal doc ids.
        try:
            max_doc = int(jreader.maxDoc())
        except Exception:
            try:
                max_doc = int(jreader.numDocs())
            except Exception as e:
                raise RuntimeError("Unable to determine maxDoc/numDocs from Lucene reader") from e

        for i in range(max_doc):
            try:
                d = jreader.document(i)
            except Exception:
                # Some Lucene versions require storedFields().document(i)
                try:
                    sf = jreader.storedFields()
                    d = sf.document(i)
                except Exception:
                    continue
            docid = _docid_from_lucene_document(d)
            if docid:
                yield docid

    def _iter_from_searcher(searcher) -> Iterable[str]:
        # Anserini SimpleSearcher has numDocs() and doc(int internalId)
        j = None
        for attr in ("searcher", "_searcher", "simple_searcher", "_simple_searcher"):
            if hasattr(searcher, attr):
                j = getattr(searcher, attr)
                if j is not None:
                    break

        # Try to get count
        n = None
        for cand in ("num_docs", "numDocs", "getNumDocs"):
            try:
                v = getattr(searcher, cand)
                n = int(v() if callable(v) else v)
                break
            except Exception:
                pass
        if n is None and j is not None:
            for cand in ("numDocs", "getNumDocs"):
                try:
                    n = int(getattr(j, cand)())
                    break
                except Exception:
                    pass
        if n is None:
            # Last resort: use index reader iteration if we can obtain it
            try:
                r = j.getIndexReader() if j is not None and hasattr(j, "getIndexReader") else None
            except Exception:
                r = None
            if r is not None:
                yield from _iter_from_java_reader(r)
                return
            raise RuntimeError("Unable to iterate docids via LuceneSearcher (missing numDocs/doc APIs)")

        for internal_id in range(int(n)):
            if j is None:
                # Try LuceneSearcher.doc(int) directly if exposed
                try:
                    d = searcher.doc(int(internal_id))
                    docid = _docid_from_lucene_document(d)
                    if docid:
                        yield docid
                    continue
                except Exception:
                    continue
            try:
                d = j.doc(int(internal_id))
            except Exception:
                # Some versions: document(int)
                try:
                    d = j.document(int(internal_id))
                except Exception:
                    continue
            docid = _docid_from_lucene_document(d)
            if docid:
                yield docid

    # 1) Try IndexReader convenience APIs (newer Pyserini)
    reader = get_index_reader(index_name)
    if hasattr(reader, "docids"):
        return reader.docids()
    if hasattr(reader, "get_docids"):
        return reader.get_docids()

    # 2) Try to access underlying Lucene reader if exposed on the wrapper
    for attr in ("reader", "_reader", "index_reader", "_index_reader", "lucene_reader", "_lucene_reader"):
        jreader = getattr(reader, attr, None)
        if jreader is not None:
            return _iter_from_java_reader(jreader)

    # 3) Fallback: iterate via LuceneSearcher / SimpleSearcher internal docids (works on older installs)
    searcher = get_searcher(index_name)
    return _iter_from_searcher(searcher)


@dataclass(frozen=True)
class DenseAssetsPaths:
    corpus_jsonl: str
    docids_txt: str
    embeddings_npy: str
    meta_json: str


def default_assets_paths(*, out_dir: str, index_name: str, model_name: str) -> DenseAssetsPaths:
    safe_index = index_name.replace("/", "_")
    safe_model = model_name.replace("/", "_")
    corpus_jsonl = os.path.join(out_dir, f"corpus_{safe_index}.jsonl")
    docids_txt = os.path.join(out_dir, f"docids_{safe_index}.txt")
    embeddings_npy = os.path.join(out_dir, f"embeddings_{safe_index}__{safe_model}.npy")
    meta_json = os.path.join(out_dir, f"meta_{safe_index}__{safe_model}.json")
    return DenseAssetsPaths(
        corpus_jsonl=corpus_jsonl,
        docids_txt=docids_txt,
        embeddings_npy=embeddings_npy,
        meta_json=meta_json,
    )


def build_corpus_jsonl(
    *,
    index_name: str,
    out_corpus_jsonl: str,
    out_docids_txt: str,
    max_docs: Optional[int] = None,
    force: bool = False,
) -> Tuple[int, str]:
    """Build corpus JSONL + aligned docid list.

    Returns: (num_docs_written, sha256(docids_txt))
    """
    if (not force) and os.path.exists(out_corpus_jsonl) and os.path.exists(out_docids_txt):
        # Assume outputs are already built.
        return _count_lines(out_docids_txt), _sha256_file(out_docids_txt)

    _ensure_dir(os.path.dirname(out_corpus_jsonl) or ".")

    # Import Pyserini only when building.
    from rag.lucene_backend import get_searcher

    searcher = get_searcher(index_name)
    docids_iter = _iter_docids(index_name)

    tmp_corpus = out_corpus_jsonl + ".tmp"
    tmp_docids = out_docids_txt + ".tmp"

    n = 0
    with open(tmp_corpus, "w", encoding="utf-8") as f_corpus, open(tmp_docids, "w", encoding="utf-8") as f_docids:
        for docid in docids_iter:
            if max_docs is not None and n >= int(max_docs):
                break
            docid_s = str(docid)
            text = _best_effort_doc_text(searcher, docid_s)
            rec = {"docid": docid_s, "text": text}
            # Stable json encoding (no whitespace variability).
            f_corpus.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            f_docids.write(docid_s + "\n")
            n += 1

    os.replace(tmp_corpus, out_corpus_jsonl)
    os.replace(tmp_docids, out_docids_txt)

    return n, _sha256_file(out_docids_txt)


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _iter_corpus_texts(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (docid, text) from a corpus JSONL file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            docid = str(obj.get("docid", ""))
            text = obj.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            if not docid:
                raise ValueError(f"{path}:{line_no}: missing docid")
            yield docid, text


def build_embeddings_npy(
    *,
    corpus_jsonl: str,
    out_embeddings_npy: str,
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    force: bool = False,
    log_every: int = 5000,
    show_progress_bar: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """Compute embeddings for the corpus JSONL and write a .npy matrix.

    Returns: (num_docs, embedding_dim)
    """
    if (not force) and os.path.exists(out_embeddings_npy):
        # Best effort to infer shape without loading full array
        try:
            import numpy as np

            arr = np.load(out_embeddings_npy, mmap_mode="r")
            return int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            pass

    log = logger or logging.getLogger("rag.approach3.build_dense_assets")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for embeddings: sentence-transformers. "
            "Install it in your environment to build dense embeddings."
        ) from e

    import numpy as np
    from numpy.lib.format import open_memmap

    # First pass: count docs to allocate array deterministically.
    t_count = time.perf_counter()
    num_docs = 0
    for _docid, _text in _iter_corpus_texts(corpus_jsonl):
        num_docs += 1
    log.info("Embedding: counted %d docs in %.2fs", num_docs, time.perf_counter() - t_count)

    if num_docs <= 0:
        raise ValueError(f"No documents found in corpus: {corpus_jsonl}")

    t_model = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())
    log.info(
        "Embedding: loaded model=%r device=%r dim=%d in %.2fs",
        model_name,
        device,
        dim,
        time.perf_counter() - t_model,
    )

    _ensure_dir(os.path.dirname(out_embeddings_npy) or ".")
    tmp = out_embeddings_npy + ".tmp"

    # Write using numpy .npy memmap for large arrays.
    mmap = open_memmap(tmp, mode="w+", dtype="float32", shape=(num_docs, dim))

    buf_texts = []
    row = 0
    last_log_row = 0
    t0 = time.perf_counter()
    for _docid, text in _iter_corpus_texts(corpus_jsonl):
        buf_texts.append(text)
        if len(buf_texts) >= int(batch_size):
            emb = model.encode(
                buf_texts,
                batch_size=int(batch_size),
                show_progress_bar=bool(show_progress_bar),
                convert_to_numpy=True,
                normalize_embeddings=bool(normalize_embeddings),
            )
            mmap[row : row + emb.shape[0], :] = emb.astype("float32", copy=False)
            row += int(emb.shape[0])
            buf_texts = []

            if int(log_every) > 0 and (row - last_log_row) >= int(log_every):
                elapsed = time.perf_counter() - t0
                rate = (row / elapsed) if elapsed > 1e-9 else 0.0
                remaining = max(0, num_docs - row)
                eta = (remaining / rate) if rate > 1e-9 else float("inf")
                log.info(
                    "Embedding progress: %d/%d (%.1f%%) elapsed=%.1fs rate=%.1f docs/s eta=%.1fs",
                    row,
                    num_docs,
                    100.0 * float(row) / float(num_docs),
                    elapsed,
                    rate,
                    eta,
                )
                last_log_row = row

    if buf_texts:
        emb = model.encode(
            buf_texts,
            batch_size=int(batch_size),
            show_progress_bar=bool(show_progress_bar),
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize_embeddings),
        )
        mmap[row : row + emb.shape[0], :] = emb.astype("float32", copy=False)
        row += int(emb.shape[0])

    # Flush and atomically move into place.
    mmap.flush()
    os.replace(tmp, out_embeddings_npy)
    elapsed = time.perf_counter() - t0
    rate = (row / elapsed) if elapsed > 1e-9 else 0.0
    log.info("Embedding: finished %d docs in %.2fs (%.1f docs/s)", row, elapsed, rate)
    return num_docs, dim


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Approach 3 dense assets (corpus JSONL + embedding cache).")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name (default: robust04).")
    p.add_argument("--out-dir", default="cache/approach3_dense", help="Output directory for artifacts.")
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model name for the bi-encoder.",
    )
    p.add_argument("--device", default="cpu", help="Device for embedding model (cpu|cuda).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding.")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
    p.add_argument("--log-every", type=int, default=5000, help="Log embedding progress every N documents.")
    p.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Let sentence-transformers show its own progress bar (may require extra deps).",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization (default is normalize embeddings).",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap for debugging (build only the first N docs in index order).",
    )
    p.add_argument("--force", action="store_true", help="Rebuild artifacts even if they already exist.")
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only build corpus/docids (skip embedding computation).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        from rag.logging_utils import configure_logging

        configure_logging(args.log_level)
    except Exception:
        logging.basicConfig(level=str(args.log_level).upper())

    out_dir = str(args.out_dir)
    index_name = str(args.index)
    model_name = str(args.model_name)
    max_docs = args.max_docs if args.max_docs is None else int(args.max_docs)

    _ensure_dir(out_dir)
    paths = default_assets_paths(out_dir=out_dir, index_name=index_name, model_name=model_name)

    n_docs, docids_sha = build_corpus_jsonl(
        index_name=index_name,
        out_corpus_jsonl=paths.corpus_jsonl,
        out_docids_txt=paths.docids_txt,
        max_docs=max_docs,
        force=bool(args.force),
    )

    emb_docs = None
    emb_dim = None
    if not bool(args.skip_embeddings):
        emb_docs, emb_dim = build_embeddings_npy(
            corpus_jsonl=paths.corpus_jsonl,
            out_embeddings_npy=paths.embeddings_npy,
            model_name=model_name,
            batch_size=int(args.batch_size),
            device=str(args.device),
            normalize_embeddings=not bool(args.no_normalize),
            force=bool(args.force),
            log_every=int(args.log_every),
            show_progress_bar=bool(args.show_progress_bar),
            logger=logging.getLogger("rag.approach3.build_dense_assets"),
        )

    meta = {
        "created_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "index": index_name,
        "model_name": model_name,
        "max_docs": max_docs,
        "outputs": {
            "corpus_jsonl": paths.corpus_jsonl,
            "docids_txt": paths.docids_txt,
            "embeddings_npy": None if bool(args.skip_embeddings) else paths.embeddings_npy,
        },
        "counts": {
            "corpus_docs": int(n_docs),
            "embedding_docs": None if emb_docs is None else int(emb_docs),
            "embedding_dim": None if emb_dim is None else int(emb_dim),
        },
        "options": {
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "normalize_embeddings": not bool(args.no_normalize),
        },
        "checksums": {
            "docids_sha256": docids_sha,
        },
    }
    _atomic_write_text(paths.meta_json, json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print("Wrote:")
    print(f"- corpus: {paths.corpus_jsonl}")
    print(f"- docids: {paths.docids_txt}")
    if bool(args.skip_embeddings):
        print("- embeddings: (skipped)")
    else:
        print(f"- embeddings: {paths.embeddings_npy}")
    print(f"- meta: {paths.meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

