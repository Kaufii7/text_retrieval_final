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
from typing import Any, Iterable, Iterator, Optional, Tuple


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


def _atomic_write_json(path: str, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _try_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_device(requested: str) -> str:
    """Resolve a device string for torch-backed models.

    Supported:
    - "auto": prefer CUDA, then MPS (Apple Silicon), else CPU
    - "gpu": alias of "auto"
    - explicit: "cpu" | "cuda" | "mps" | "cuda:0" etc.
    """
    req = (requested or "auto").strip().lower()
    if req in ("", "auto", "gpu"):
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            # MPS on macOS Apple Silicon
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


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
    resume: bool = True,
    save_every: int = 5000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, str]:
    """Build corpus JSONL + aligned docid list.

    Returns: (num_docs_written, sha256(docids_txt))
    """
    log = logger or logging.getLogger("rag.approach3.build_dense_assets")
    if (not force) and os.path.exists(out_corpus_jsonl) and os.path.exists(out_docids_txt):
        # Assume outputs are already built.
        log.info("Corpus: using existing outputs corpus=%s docids=%s", out_corpus_jsonl, out_docids_txt)
        return _count_lines(out_docids_txt), _sha256_file(out_docids_txt)

    _ensure_dir(os.path.dirname(out_corpus_jsonl) or ".")
    log.info(
        "Corpus: build start index=%s out_corpus=%s out_docids=%s max_docs=%s resume=%s save_every=%d",
        str(index_name),
        str(out_corpus_jsonl),
        str(out_docids_txt),
        "None" if max_docs is None else str(int(max_docs)),
        str(bool(resume)),
        int(save_every),
    )

    # Import Pyserini only when building.
    from rag.lucene_backend import get_searcher

    searcher = get_searcher(index_name)
    docids_iter = _iter_docids(index_name)

    # Resume/safe-write:
    # - write to ".partial" files and checkpoint progress to ".progress.json"
    # - on success, atomically replace final outputs
    partial_corpus = out_corpus_jsonl + ".partial"
    partial_docids = out_docids_txt + ".partial"
    progress_path = out_docids_txt + ".progress.json"

    if force:
        _safe_unlink(partial_corpus)
        _safe_unlink(partial_docids)
        _safe_unlink(progress_path)

    n_written = 0
    if resume and os.path.exists(partial_corpus) and os.path.exists(partial_docids):
        n_written = _count_lines(partial_docids)
        log.info("Corpus resume: found %d docs already written in partial outputs.", n_written)
    else:
        _safe_unlink(partial_corpus)
        _safe_unlink(partial_docids)
        _safe_unlink(progress_path)

    mode = "a" if n_written > 0 else "w"
    n = int(n_written)
    last_save_n = int(n_written)
    t0 = time.perf_counter()

    with open(partial_corpus, mode, encoding="utf-8") as f_corpus, open(partial_docids, mode, encoding="utf-8") as f_docids:
        try:
            for i, docid in enumerate(docids_iter):
                if i < int(n_written):
                    continue
                if max_docs is not None and n >= int(max_docs):
                    break
                docid_s = str(docid)
                text = _best_effort_doc_text(searcher, docid_s)
                rec = {"docid": docid_s, "text": text}
                f_corpus.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                f_docids.write(docid_s + "\n")
                n += 1

                if int(save_every) > 0 and (n - last_save_n) >= int(save_every):
                    f_corpus.flush()
                    f_docids.flush()
                    _atomic_write_json(
                        progress_path,
                        {
                            "kind": "approach3_corpus_build",
                            "index": str(index_name),
                            "out_corpus_jsonl": str(out_corpus_jsonl),
                            "out_docids_txt": str(out_docids_txt),
                            "partial_corpus": str(partial_corpus),
                            "partial_docids": str(partial_docids),
                            "max_docs": None if max_docs is None else int(max_docs),
                            "docs_done": int(n),
                            "updated_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        },
                    )
                    last_save_n = n
                    elapsed = time.perf_counter() - t0
                    rate = (n / elapsed) if elapsed > 1e-9 else 0.0
                    log.info("Corpus progress: %d docs written (%.1f docs/s)", n, rate)
        except KeyboardInterrupt:
            # Ensure we checkpoint as close as possible to the interruption point.
            try:
                f_corpus.flush()
                f_docids.flush()
                _atomic_write_json(
                    progress_path,
                    {
                        "kind": "approach3_corpus_build",
                        "index": str(index_name),
                        "out_corpus_jsonl": str(out_corpus_jsonl),
                        "out_docids_txt": str(out_docids_txt),
                        "partial_corpus": str(partial_corpus),
                        "partial_docids": str(partial_docids),
                        "max_docs": None if max_docs is None else int(max_docs),
                        "docs_done": int(n),
                        "updated_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "interrupted": True,
                    },
                )
            except Exception:
                pass
            log.warning("Corpus build interrupted. Checkpointed at docs_done=%d. You can rerun to resume.", int(n))
            raise

    os.replace(partial_corpus, out_corpus_jsonl)
    os.replace(partial_docids, out_docids_txt)
    _safe_unlink(progress_path)
    return int(n), _sha256_file(out_docids_txt)


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
    resume: bool = True,
    save_every: int = 5000,
    log_every: int = 5000,
    show_progress_bar: bool = False,
    logger: Optional[logging.Logger] = None,
    sagemaker_config: Optional[dict] = None,
) -> Tuple[int, int]:
    """Compute embeddings for the corpus JSONL and write a .npy matrix.

    Args:
        sagemaker_config: Optional dict with SageMaker config:
            - enabled: bool
            - endpoint_name: str
            - region: str (optional)

    Returns: (num_docs, embedding_dim)
    """
    if (not force) and os.path.exists(out_embeddings_npy):
        # Best effort to infer shape without loading full array
        try:
            import numpy as np

            arr = np.load(out_embeddings_npy, mmap_mode="r")
            log = logger or logging.getLogger("rag.approach3.build_dense_assets")
            log.info(
                "Embedding: using existing output embeddings=%s shape=(%d,%d) dtype=%s",
                out_embeddings_npy,
                int(arr.shape[0]),
                int(arr.shape[1]),
                str(arr.dtype),
            )
            return int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            pass

    log = logger or logging.getLogger("rag.approach3.build_dense_assets")

    import numpy as np
    from numpy.lib.format import open_memmap

    # Import embedding adapter (handles both local and SageMaker)
    from rag.approach3.sagemaker_embeddings import get_embedding_model

    _ensure_dir(os.path.dirname(out_embeddings_npy) or ".")

    # Resume/safe-write:
    # - write to ".partial" file and checkpoint progress to ".progress.json"
    # - on success, atomically replace final output
    partial = out_embeddings_npy + ".partial"
    progress_path = out_embeddings_npy + ".progress.json"

    if force:
        _safe_unlink(partial)
        _safe_unlink(progress_path)

    progress = _try_read_json(progress_path) if (resume and os.path.exists(progress_path)) else None
    rows_done = int(progress.get("rows_done") or 0) if progress else 0
    num_docs = int(progress.get("num_docs")) if (progress and progress.get("num_docs") is not None) else None
    dim = int(progress.get("dim")) if (progress and progress.get("dim") is not None) else None
    if progress:
        log.info("Embedding resume: rows_done=%d num_docs=%s dim=%s", rows_done, str(num_docs), str(dim))

    if num_docs is None:
        t_count = time.perf_counter()
        n_count = 0
        for _docid, _text in _iter_corpus_texts(corpus_jsonl):
            n_count += 1
        num_docs = int(n_count)
        log.info("Embedding: counted %d docs in %.2fs", num_docs, time.perf_counter() - t_count)

    if int(num_docs) <= 0:
        raise ValueError(f"No documents found in corpus: {corpus_jsonl}")

    t_model = time.perf_counter()
    model = get_embedding_model(model_name=model_name, device=device, sagemaker_config=sagemaker_config)
    if dim is None:
        dim = int(model.get_sentence_embedding_dimension())

    model_desc = (
        f"SageMaker endpoint: {sagemaker_config.get('endpoint_name')}"
        if (sagemaker_config and sagemaker_config.get("enabled"))
        else f"local model={model_name} device={device}"
    )
    log.info("Embedding: loaded %s dim=%d in %.2fs", model_desc, int(dim), time.perf_counter() - t_model)
    log.info(
        "Embedding: build start corpus=%s out=%s num_docs=%d batch_size=%d normalize=%s save_every=%d log_every=%d",
        str(corpus_jsonl),
        str(out_embeddings_npy),
        int(num_docs),
        int(batch_size),
        str(bool(normalize_embeddings)),
        int(save_every),
        int(log_every),
    )

    # Open/create the partial memmap
    if resume and os.path.exists(partial):
        try:
            mmap = open_memmap(partial, mode="r+")
            if int(mmap.shape[0]) != int(num_docs) or int(mmap.shape[1]) != int(dim):
                raise ValueError(
                    f"Partial embeddings shape mismatch: got={tuple(mmap.shape)} expected={(int(num_docs), int(dim))}"
                )
        except Exception:
            _safe_unlink(partial)
            _safe_unlink(progress_path)
            mmap = open_memmap(partial, mode="w+", dtype="float32", shape=(int(num_docs), int(dim)))
            rows_done = 0
    else:
        _safe_unlink(partial)
        _safe_unlink(progress_path)
        mmap = open_memmap(partial, mode="w+", dtype="float32", shape=(int(num_docs), int(dim)))
        rows_done = 0

    buf_texts = []
    row = int(rows_done)
    last_log_row = 0
    last_save_row = int(rows_done)
    t0 = time.perf_counter()
    try:
        for i, (_docid, text) in enumerate(_iter_corpus_texts(corpus_jsonl)):
            if i < int(rows_done):
                continue
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

                if int(save_every) > 0 and (row - last_save_row) >= int(save_every):
                    mmap.flush()
                    _atomic_write_json(
                        progress_path,
                        {
                            "kind": "approach3_embeddings_build",
                            "corpus_jsonl": str(corpus_jsonl),
                            "out_embeddings_npy": str(out_embeddings_npy),
                            "partial": str(partial),
                            "model_name": str(model_name),
                            "device": str(device),
                            "normalize_embeddings": bool(normalize_embeddings),
                            "batch_size": int(batch_size),
                            "sagemaker": sagemaker_config,
                            "num_docs": int(num_docs),
                            "dim": int(dim),
                            "rows_done": int(row),
                            "updated_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        },
                    )
                    last_save_row = row

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
    except KeyboardInterrupt:
        # Ensure we checkpoint as close as possible to the interruption point.
        try:
            mmap.flush()
            _atomic_write_json(
                progress_path,
                {
                    "kind": "approach3_embeddings_build",
                    "corpus_jsonl": str(corpus_jsonl),
                    "out_embeddings_npy": str(out_embeddings_npy),
                    "partial": str(partial),
                    "model_name": str(model_name),
                    "device": str(device),
                    "normalize_embeddings": bool(normalize_embeddings),
                    "batch_size": int(batch_size),
                    "sagemaker": sagemaker_config,
                    "num_docs": int(num_docs),
                    "dim": int(dim),
                    "rows_done": int(row),
                    "updated_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    "interrupted": True,
                },
            )
        except Exception:
            pass
        log.warning("Embedding interrupted. Checkpointed at rows_done=%d. You can rerun to resume.", int(row))
        raise

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
    os.replace(partial, out_embeddings_npy)
    _safe_unlink(progress_path)
    elapsed = time.perf_counter() - t0
    rate = (row / elapsed) if elapsed > 1e-9 else 0.0
    log.info("Embedding: finished %d docs in %.2fs (%.1f docs/s)", row, elapsed, rate)
    return int(num_docs), int(dim)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Approach 3 dense assets (corpus JSONL + embedding cache).")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name (default: robust04).")
    p.add_argument("--out-dir", default="cache/approach3_dense", help="Output directory for artifacts.")
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model name for the bi-encoder.",
    )
    p.add_argument("--device", default="auto", help="Device for embedding model (auto|cpu|cuda|mps).")
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
        "--no-resume",
        action="store_true",
        help="Disable resume mode (otherwise partial outputs + progress checkpoints are used).",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="Checkpoint embeddings every N docs (flush + write *.progress.json).",
    )
    p.add_argument(
        "--corpus-save-every",
        type=int,
        default=5000,
        help="Checkpoint corpus/docids every N docs (flush + write *.progress.json).",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only build corpus/docids (skip embedding computation).",
    )
    # SageMaker options
    p.add_argument(
        "--sagemaker-endpoint",
        type=str,
        default=None,
        help="Use SageMaker endpoint instead of local model (provide endpoint name).",
    )
    p.add_argument(
        "--sagemaker-region",
        type=str,
        default="eu-north-1",
        help="AWS region for SageMaker endpoint (default: eu-north-1).",
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
    log = logging.getLogger("rag.approach3.build_dense_assets")
    paths = default_assets_paths(out_dir=out_dir, index_name=index_name, model_name=model_name)
    log.info(
        "Dense assets: start index=%s model=%s out_dir=%s max_docs=%s force=%s resume=%s",
        index_name,
        model_name,
        out_dir,
        "None" if max_docs is None else str(int(max_docs)),
        str(bool(args.force)),
        str(not bool(args.no_resume)),
    )
    log.info(
        "Dense assets: paths corpus=%s docids=%s embeddings=%s meta=%s",
        paths.corpus_jsonl,
        paths.docids_txt,
        paths.embeddings_npy,
        paths.meta_json,
    )

    n_docs, docids_sha = build_corpus_jsonl(
        index_name=index_name,
        out_corpus_jsonl=paths.corpus_jsonl,
        out_docids_txt=paths.docids_txt,
        max_docs=max_docs,
        force=bool(args.force),
        resume=not bool(args.no_resume),
        save_every=int(args.corpus_save_every),
        logger=logging.getLogger("rag.approach3.build_dense_assets"),
    )

    emb_docs = None
    emb_dim = None
    if not bool(args.skip_embeddings):
        # Build SageMaker config if endpoint is provided
        sagemaker_config = None
        if args.sagemaker_endpoint:
            sagemaker_config = {
                "enabled": True,
                "endpoint_name": str(args.sagemaker_endpoint),
                "region": str(args.sagemaker_region),
            }
        emb_docs, emb_dim = build_embeddings_npy(
            corpus_jsonl=paths.corpus_jsonl,
            out_embeddings_npy=paths.embeddings_npy,
            model_name=model_name,
            batch_size=int(args.batch_size),
            device=str(args.device),
            normalize_embeddings=not bool(args.no_normalize),
            force=bool(args.force),
            resume=not bool(args.no_resume),
            save_every=int(args.save_every),
            log_every=int(args.log_every),
            show_progress_bar=bool(args.show_progress_bar),
            logger=logging.getLogger("rag.approach3.build_dense_assets"),
            sagemaker_config=sagemaker_config,
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
            "sagemaker": sagemaker_config if not bool(args.skip_embeddings) else None,
        },
        "checksums": {
            "docids_sha256": docids_sha,
        },
    }
    _atomic_write_text(paths.meta_json, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log.info("Dense assets: wrote meta=%s", paths.meta_json)

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

