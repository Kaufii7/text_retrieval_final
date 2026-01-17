"""BM25 -> embedding mapping -> cross-encoder rerank pipeline.

Pipeline (per query):
1) Retrieve 5k docs using BM25 (Pyserini / Lucene).
2) Map each document (docid) to the correct embedding row using docids.txt.
3) Score candidates with a cross-encoder (HuggingFace seq-classification) using (query, doc_text).
4) Re-order the *embeddings* according to the cross-encoder score.
5) Re-map the reordered embeddings back to original docids.
6) Return top-1000 docids (and optionally embeddings/scores).

Notes:
- Step (3) fundamentally requires *text* (a cross-encoder scores text pairs), not raw embeddings.
  We keep embeddings aligned throughout, as requested.
- For doc text, prefer `--corpus-jsonl` to avoid extra Java calls; otherwise we fall back
  to Pyserini `fetch_doc_contents(searcher, docid)`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rag.io import load_queries
from rag.lucene_backend import fetch_doc_contents, get_searcher, search, set_bm25, set_rm3
from rag.runs import write_trec_run
from rag.types import Query


def _require_ce_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Cross-encoder scoring requires optional deps: torch + transformers (AutoTokenizer/AutoModel)."
        ) from e


def _require_monot5_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "MonoT5 scoring requires optional deps: torch + transformers (AutoTokenizer/AutoModelForSeq2SeqLM)."
        ) from e


def _configure_logging(*, level: str) -> logging.Logger:
    log = logging.getLogger("rag.approach3.bm25_ce_pipeline")
    log.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.addHandler(sh)
    return log


class _CorpusJsonlLookup:
    """Docid->text lookup for a corpus JSONL using a SQLite offset index."""

    def __init__(self, *, corpus_jsonl: str, sqlite_path: str, logger: logging.Logger):
        self.corpus_jsonl = str(corpus_jsonl)
        self.sqlite_path = str(sqlite_path)
        self._log = logger

        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("CREATE TABLE IF NOT EXISTS docs (docid TEXT PRIMARY KEY, offset INTEGER NOT NULL)")
        self._conn.commit()

        if not self._has_rows():
            self._build_index()

        self._fh = open(self.corpus_jsonl, "rb")

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            try:
                self._conn.close()
            except Exception:
                pass

    def _has_rows(self) -> bool:
        try:
            cur = self._conn.execute("SELECT 1 FROM docs LIMIT 1")
            return cur.fetchone() is not None
        except Exception:
            return False

    def _build_index(self) -> None:
        self._log.info("Building corpus offset index: corpus=%s db=%s", self.corpus_jsonl, self.sqlite_path)
        t0 = time.perf_counter()
        n = 0
        batch: List[Tuple[str, int]] = []
        with open(self.corpus_jsonl, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                    docid = str(obj.get("docid", "")).strip()
                    if not docid:
                        continue
                except Exception:
                    continue
                batch.append((docid, int(off)))
                n += 1
                if len(batch) >= 50_000:
                    self._conn.executemany("INSERT OR REPLACE INTO docs(docid, offset) VALUES (?, ?)", batch)
                    self._conn.commit()
                    batch = []
                    elapsed = time.perf_counter() - t0
                    rate = n / elapsed if elapsed > 1e-9 else 0.0
                    self._log.info("Corpus index progress: %d docs (%.1f docs/s)", n, rate)
        if batch:
            self._conn.executemany("INSERT OR REPLACE INTO docs(docid, offset) VALUES (?, ?)", batch)
            self._conn.commit()
        self._log.info("Corpus index built: docs=%d elapsed=%.2fs", n, time.perf_counter() - t0)

    def get_text(self, docid: str) -> str:
        cur = self._conn.execute("SELECT offset FROM docs WHERE docid = ?", (str(docid),))
        row = cur.fetchone()
        if row is None:
            return ""
        off = int(row[0])
        self._fh.seek(off)
        line = self._fh.readline()
        try:
            obj = json.loads(line.decode("utf-8", errors="replace"))
            txt = obj.get("text", "")
            return txt if isinstance(txt, str) else str(txt)
        except Exception:
            return ""


def _load_docid_to_row(docids_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(docids_path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f):
            docid = raw.strip()
            if not docid:
                continue
            mapping[docid] = int(i)
    return mapping


def _score_pairs_hf(
    *,
    model_name_or_dir: str,
    pairs: Sequence[Tuple[str, str]],
    device: str,
    batch_size: int,
    max_length: int,
    log: Optional[logging.Logger] = None,
    log_prefix: str = "CE",
    log_every_batches: int = 10,
) -> List[float]:
    _require_ce_deps()
    import os
    from pathlib import Path
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

    dev_str = str(device or "cpu")
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        if log is not None:
            log.warning("CUDA requested (%s) but not available; falling back to cpu.", dev_str)
        dev_str = "cpu"
    if dev_str == "mps" and not getattr(torch.backends, "mps", None):
        if log is not None:
            log.warning("MPS requested but torch.backends.mps missing; falling back to cpu.")
        dev_str = "cpu"
    if dev_str == "mps" and not torch.backends.mps.is_available():
        if log is not None:
            log.warning("MPS requested but not available; falling back to cpu.")
        dev_str = "cpu"

    # Check if it's a local path vs HuggingFace Hub ID
    model_path = str(model_name_or_dir)
    is_local = os.path.isdir(model_path)
    
    if is_local:
        # For local paths, use Path object to avoid HuggingFace repo_id validation issues
        local_path = Path(model_path).resolve()
        config = AutoConfig.from_pretrained(local_path)
        tok = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path, config=config)
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    dev = torch.device(dev_str)
    model.to(dev)

    scores: List[float] = []
    bs = max(1, int(batch_size))
    log_every = max(1, int(log_every_batches))
    t0 = time.perf_counter()
    with torch.no_grad():
        for b_idx, i in enumerate(range(0, len(pairs), bs), start=1):
            batch_pairs = pairs[i : i + bs]
            qs = [q for q, _d in batch_pairs]
            ds = [d for _q, d in batch_pairs]
            enc = tok(qs, ds, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits
            nlab = int(getattr(getattr(model, "config", None), "num_labels", 1) or 1)
            if logits.ndim == 2 and int(logits.shape[1]) >= 2 and nlab >= 2:
                s = logits[:, 1]
            elif logits.ndim == 2 and int(logits.shape[1]) >= 1:
                s = logits[:, 0]
            else:
                s = logits.view(-1)
            scores.extend([float(x) for x in s.detach().cpu().tolist()])
            if log is not None:
                done = min(len(pairs), i + len(batch_pairs))
                if (b_idx % log_every) == 0 or done == len(pairs):
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 1e-9 else 0.0
                    log.info(
                        "%s progress: pairs=%d/%d batches=%d elapsed=%.1fs rate=%.1f pairs/s",
                        str(log_prefix),
                        int(done),
                        int(len(pairs)),
                        int(b_idx),
                        float(elapsed),
                        float(rate),
                    )
    return scores


def _score_pairs_monot5_hf(
    *,
    model_name_or_dir: str,
    pairs: Sequence[Tuple[str, str]],
    device: str,
    batch_size: int,
    max_length: int,
    torch_dtype: str = "auto",
    log: Optional[logging.Logger] = None,
    log_prefix: str = "MonoT5",
    log_every_batches: int = 10,
) -> List[float]:
    """Score (query, doc_text) pairs with MonoT5 (seq2seq) using "true/false" log-prob."""
    _require_monot5_deps()
    import os
    from pathlib import Path

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dev_str = str(device or "cpu")
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        if log is not None:
            log.warning("CUDA requested (%s) but not available; falling back to cpu.", dev_str)
        dev_str = "cpu"
    if dev_str == "mps" and not getattr(torch.backends, "mps", None):
        if log is not None:
            log.warning("MPS requested but torch.backends.mps missing; falling back to cpu.")
        dev_str = "cpu"
    if dev_str == "mps" and not torch.backends.mps.is_available():
        if log is not None:
            log.warning("MPS requested but not available; falling back to cpu.")
        dev_str = "cpu"

    # Map dtype string to torch dtype / 'auto'
    td_raw = str(torch_dtype or "auto")
    if td_raw.lower() == "auto":
        td_arg = "auto"
    else:
        td_arg = getattr(torch, td_raw, None)
        if td_arg is None:
            if log is not None:
                log.warning("Unknown torch_dtype=%s; falling back to 'auto'", td_raw)
            td_arg = "auto"

    # Check if it's a local path vs HuggingFace Hub ID
    model_path = str(model_name_or_dir)
    is_local = os.path.isdir(model_path)
    if is_local:
        local_path = Path(model_path).resolve()
        tok = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_path, torch_dtype=td_arg)
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=td_arg)

    model.eval()
    dev = torch.device(dev_str)
    model.to(dev)

    true_ids = tok.encode("true", add_special_tokens=False)
    false_ids = tok.encode("false", add_special_tokens=False)
    if not true_ids or not false_ids:
        raise RuntimeError("Tokenizer could not encode 'true'/'false' token ids")
    true_token_id = int(true_ids[0])
    false_token_id = int(false_ids[0])

    decoder_start_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = getattr(tok, "pad_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = 0

    def _prompt(q: str, d: str) -> str:
        return f"Query: {q} Document: {d} Relevant:"

    scores: List[float] = []
    bs = max(1, int(batch_size))
    log_every = max(1, int(log_every_batches))
    t0 = time.perf_counter()
    with torch.no_grad():
        for b_idx, i in enumerate(range(0, len(pairs), bs), start=1):
            batch_pairs = pairs[i : i + bs]
            prompts = [_prompt(q, d) for q, d in batch_pairs]
            enc = tok(
                prompts,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            decoder_input_ids = torch.full(
                (int(len(prompts)), 1),
                int(decoder_start_id),
                dtype=torch.long,
                device=dev,
            )
            logits = model(**enc, decoder_input_ids=decoder_input_ids).logits  # [B,1,V]
            step_logits = logits[:, 0, :]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            s = log_probs[:, true_token_id] - log_probs[:, false_token_id]
            scores.extend([float(x) for x in s.detach().cpu().tolist()])

            if log is not None:
                done = min(len(pairs), i + len(batch_pairs))
                if (b_idx % log_every) == 0 or done == len(pairs):
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 1e-9 else 0.0
                    log.info(
                        "%s progress: pairs=%d/%d batches=%d elapsed=%.1fs rate=%.1f pairs/s",
                        str(log_prefix),
                        int(done),
                        int(len(pairs)),
                        int(b_idx),
                        float(elapsed),
                        float(rate),
                    )
    return scores


@dataclass(frozen=True)
class RerankedCandidate:
    docid: str
    ce_score: float
    bm25_score: float
    final_score: float
    emb_row: int


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    mn = float(min(scores))
    mx = float(max(scores))
    if mx <= mn + 1e-12:
        return [0.5 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]


def bm25_to_ce_topk(
    *,
    query: str,
    searcher,
    bm25_topk: int,
    final_topk: int,
    rerank_depth: int,
    alpha: float,
    docid_to_row: Dict[str, int],
    embeddings: np.ndarray,
    reranker_type: str,
    ce_model: str,
    monot5_model: str,
    monot5_torch_dtype: str,
    ce_device: str,
    ce_batch_size: int,
    ce_max_length: int,
    corpus_lookup: Optional[_CorpusJsonlLookup],
    log: logging.Logger,
    ce_log_every_batches: int = 10,
    ce_log_prefix: str = "CE",
) -> Tuple[List[RerankedCandidate], np.ndarray]:
    """Return (top_candidates, top_embeddings) aligned by rank."""
    # 1) BM25 retrieve
    hits = search(searcher, query, topk=int(bm25_topk))
    if not hits:
        return [], np.zeros((0, int(embeddings.shape[1])), dtype=np.float32)

    # 2) Map docids -> embedding rows (keep order)
    kept: List[Tuple[str, float, int]] = []
    missing = 0
    for h in hits:
        r = docid_to_row.get(h.docid)
        if r is None:
            missing += 1
            continue
        kept.append((h.docid, float(h.score), int(r)))
    if missing:
        log.warning("Missing embeddings for %d/%d BM25 hits (skipped).", missing, len(hits))
    if not kept:
        return [], np.zeros((0, int(embeddings.shape[1])), dtype=np.float32)

    # 3) Cross-encoder score using doc_text for the top-N candidates only.
    rd = int(rerank_depth)
    if rd <= 0:
        rd = int(final_topk)
    rd = min(rd, len(kept))
    kept_rerank = kept[:rd]

    doc_texts: List[str] = []
    for docid, _bm25, _row in kept_rerank:
        if corpus_lookup is not None:
            doc_texts.append(corpus_lookup.get_text(docid))
        else:
            doc_texts.append(fetch_doc_contents(searcher, docid))

    pairs = [(query, t) for t in doc_texts]
    rr_type = str(reranker_type or "ce").lower().strip()
    if rr_type == "monot5":
        ce_scores = _score_pairs_monot5_hf(
            model_name_or_dir=str(monot5_model),
            pairs=pairs,
            device=str(ce_device),
            batch_size=int(ce_batch_size),
            max_length=int(ce_max_length),
            torch_dtype=str(monot5_torch_dtype),
            log=log,
            log_prefix=str(ce_log_prefix).replace("CE", "MonoT5"),
            log_every_batches=int(ce_log_every_batches),
        )
    else:
        ce_scores = _score_pairs_hf(
            model_name_or_dir=str(ce_model),
            pairs=pairs,
            device=str(ce_device),
            batch_size=int(ce_batch_size),
            max_length=int(ce_max_length),
            log=log,
            log_prefix=str(ce_log_prefix),
            log_every_batches=int(ce_log_every_batches),
        )

    # 4) Compute final score:
    #    - normalize BM25 scores and CE scores separately (min-max within this query)
    #    - blend: final = (1-alpha)*bm25_norm + alpha*ce_norm
    #    This prevents CE from destroying a strong lexical ordering when CE is noisy.
    a = float(alpha)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

    bm25_scores_rerank = [float(b) for _d, b, _r in kept_rerank]
    bm25_norm = _minmax(bm25_scores_rerank)
    ce_norm = _minmax([float(s) for s in ce_scores])

    reranked: List[RerankedCandidate] = []
    for (docid, bm25_score, row), s_raw, b_n, c_n in zip(kept_rerank, ce_scores, bm25_norm, ce_norm):
        final = (1.0 - a) * float(b_n) + a * float(c_n)
        reranked.append(
            RerankedCandidate(
                docid=str(docid),
                ce_score=float(s_raw),
                bm25_score=float(bm25_score),
                final_score=float(final),
                emb_row=int(row),
            )
        )
    # Deterministic: final desc, then docid asc
    reranked.sort(key=lambda x: (-x.final_score, x.docid))

    # Append the rest (not reranked) in original BM25 order.
    # They get final_score = bm25_norm only (computed over the rerank slice) to keep them behind.
    for docid, bm25_score, row in kept[rd:]:
        reranked.append(
            RerankedCandidate(
                docid=str(docid),
                ce_score=float("nan"),
                bm25_score=float(bm25_score),
                final_score=float("-inf"),
                emb_row=int(row),
            )
        )

    # 5) Re-map embeddings in the same order
    top = reranked[: int(final_topk)]
    rows = [c.emb_row for c in top]
    top_emb = embeddings[np.array(rows, dtype=np.int64), :]

    # 6) Return top-1000 docs (+ embeddings aligned)
    return top, top_emb


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25@5k -> rerank -> top1k pipeline (with embedding alignment).")
    p.add_argument("--index", default="robust04")
    p.add_argument("--queries", default=None, help="Optional queries file (queriesROBUST.txt format).")
    p.add_argument("--query", default=None, help="Single query string (if --queries is not used).")
    p.add_argument("--output-run", default=None, help="Optional TREC run output path (if using --queries).")
    p.add_argument("--run-tag", default="bm25_ce", help="Run tag for TREC output (if --output-run is set).")
    p.add_argument("--log-level", default="INFO")

    # BM25
    p.add_argument("--bm25-topk", type=int, default=5000)
    p.add_argument("--bm25-k1", type=float, default=0.9)
    p.add_argument("--bm25-b", type=float, default=0.4)
    p.add_argument(
        "--use-rm3",
        action="store_true",
        help="Enable RM3 pseudo-relevance feedback (disabled by default; pipeline is BM25 -> CE).",
    )
    p.add_argument("--rm3-fb-terms", type=int, default=50)
    p.add_argument("--rm3-fb-docs", type=int, default=50)
    p.add_argument("--rm3-orig-weight", type=float, default=0.2)
    p.add_argument("--final-topk", type=int, default=1000)
    p.add_argument(
        "--rerank-depth",
        type=int,
        default=5000,
        help="How many of the top BM25 candidates to rerank with the reranker (default: 5000).",
    )
    p.add_argument(
        "--ce-log-every-batches",
        type=int,
        default=10,
        help="Log reranker scoring progress every N batches (default: 10).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Blend weight for reranker in final score: final=(1-alpha)*lex + alpha*rerank (default: 0.2).",
    )

    # Embeddings
    p.add_argument("--embeddings", required=True, help="Embeddings .npy (rows aligned with docids.txt).")
    p.add_argument("--docids", required=True, help="Docids .txt aligned with embeddings.")

    # Doc text source
    p.add_argument("--corpus-jsonl", default=None, help="Optional corpus JSONL to fetch doc text without Lucene doc().")
    p.add_argument("--corpus-index-db", default=None, help="Optional SQLite offset index path for corpus JSONL.")

    # Reranker
    p.add_argument(
        "--reranker-type",
        choices=["ce", "monot5"],
        default="ce",
        help="Which reranker to use: 'ce' (sequence classification) or 'monot5' (seq2seq true/false).",
    )
    p.add_argument("--ce-model", default=None, help="HF model name or local directory for CE (required when --reranker-type=ce).")
    p.add_argument(
        "--monot5-model",
        default="castorini/monot5-3b-msmarco-10k",
        help="HF model id or local dir for MonoT5 (used when --reranker-type=monot5).",
    )
    p.add_argument(
        "--monot5-torch-dtype",
        default="auto",
        help="Torch dtype for MonoT5 loading (e.g. auto/float16/bfloat16/float32). Default: auto.",
    )
    p.add_argument("--ce-device", default="cpu")
    p.add_argument("--ce-batch-size", type=int, default=16)
    p.add_argument("--ce-max-length", type=int, default=256)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    log = _configure_logging(level=str(args.log_level))

    # Common thread-safety defaults (helps on macOS setups)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    rr_type = str(getattr(args, "reranker_type", "ce") or "ce").lower().strip()
    if rr_type == "ce" and (args.ce_model is None or not str(args.ce_model).strip()):
        raise SystemExit("--ce-model is required when --reranker-type=ce")
    rr_prefix = "MonoT5" if rr_type == "monot5" else "CE"

    searcher = get_searcher(str(args.index))
    set_bm25(searcher, k1=float(args.bm25_k1), b=float(args.bm25_b))
    if bool(args.use_rm3):
        set_rm3(
            searcher,
            fb_terms=int(args.rm3_fb_terms),
            fb_docs=int(args.rm3_fb_docs),
            original_query_weight=float(args.rm3_orig_weight),
        )
        log.info(
            "Enabled RM3: fb_terms=%d fb_docs=%d original_query_weight=%.3f",
            int(args.rm3_fb_terms),
            int(args.rm3_fb_docs),
            float(args.rm3_orig_weight),
        )

    log.info("Loading embeddings=%s", str(args.embeddings))
    emb = np.load(str(args.embeddings), mmap_mode="r")
    log.info("Loading docids=%s", str(args.docids))
    docid_to_row = _load_docid_to_row(str(args.docids))
    if int(emb.shape[0]) != len(docid_to_row):
        log.warning("Embeddings rows (%d) != docids count (%d). Mapping may be incomplete.", int(emb.shape[0]), len(docid_to_row))

    corpus_lookup: Optional[_CorpusJsonlLookup] = None
    if args.corpus_jsonl:
        db = str(args.corpus_index_db or os.path.join(os.path.dirname(str(args.corpus_jsonl)), "corpus_offsets.sqlite3"))
        corpus_lookup = _CorpusJsonlLookup(corpus_jsonl=str(args.corpus_jsonl), sqlite_path=db, logger=log)

    try:
        if args.queries:
            qs: List[Query] = load_queries(str(args.queries))
            if not args.output_run:
                raise SystemExit("When using --queries, also provide --output-run to write a TREC run.")

            results_by_topic: Dict[int, List[Tuple[str, float]]] = {}
            for q in qs:
                t0 = time.perf_counter()
                log.info(
                    "topic=%d start: bm25_topk=%d rerank_depth=%d final_topk=%d",
                    int(q.topic_id),
                    int(args.bm25_topk),
                    int(args.rerank_depth),
                    int(args.final_topk),
                )
                top, _top_emb = bm25_to_ce_topk(
                    query=q.text,
                    searcher=searcher,
                    bm25_topk=int(args.bm25_topk),
                    final_topk=int(args.final_topk),
                    rerank_depth=int(args.rerank_depth),
                    alpha=float(args.alpha),
                    docid_to_row=docid_to_row,
                    embeddings=emb,
                    reranker_type=str(args.reranker_type),
                    ce_model=str(args.ce_model),
                    monot5_model=str(args.monot5_model),
                    monot5_torch_dtype=str(args.monot5_torch_dtype),
                    ce_device=str(args.ce_device),
                    ce_batch_size=int(args.ce_batch_size),
                    ce_max_length=int(args.ce_max_length),
                    corpus_lookup=corpus_lookup,
                    log=log,
                    ce_log_every_batches=int(args.ce_log_every_batches),
                    ce_log_prefix=f"{rr_prefix} topic={int(q.topic_id)}",
                )
                results_by_topic[int(q.topic_id)] = [(c.docid, float(c.final_score)) for c in top]
                log.info(
                    "topic=%d done: returned=%d elapsed=%.2fs",
                    int(q.topic_id),
                    len(top),
                    time.perf_counter() - t0,
                )

            write_trec_run(
                results_by_topic=results_by_topic,
                output_path=str(args.output_run),
                run_tag=str(args.run_tag),
                topk=int(args.final_topk),
            )
            log.info("Wrote run: %s", str(args.output_run))
            return 0

        if args.query:
            top, _top_emb = bm25_to_ce_topk(
                query=str(args.query),
                searcher=searcher,
                bm25_topk=int(args.bm25_topk),
                final_topk=int(args.final_topk),
                rerank_depth=int(args.rerank_depth),
                alpha=float(args.alpha),
                docid_to_row=docid_to_row,
                embeddings=emb,
                reranker_type=str(args.reranker_type),
                ce_model=str(args.ce_model),
                monot5_model=str(args.monot5_model),
                monot5_torch_dtype=str(args.monot5_torch_dtype),
                ce_device=str(args.ce_device),
                ce_batch_size=int(args.ce_batch_size),
                ce_max_length=int(args.ce_max_length),
                corpus_lookup=corpus_lookup,
                log=log,
            )
            for i, c in enumerate(top, start=1):
                print(
                    f"{i}\t{c.docid}\t{rr_prefix.lower()}={c.ce_score:.6f}\tbm25={c.bm25_score:.6f}\tfinal={c.final_score:.6f}\trow={c.emb_row}"
                )
            return 0

        raise SystemExit("Provide either --query or --queries.")
    finally:
        if corpus_lookup is not None:
            corpus_lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())

