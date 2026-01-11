#!/usr/bin/env python3
"""Evaluate BM25, BM25+RM3, and Cross-Encoder pipeline on all 50 topics.

Outputs a table with AP per topic for each method, plus overall MAP.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from rag.eval import average_precision, mean_average_precision
from rag.io import load_qrels, load_queries
from rag.lucene_backend import fetch_doc_contents, get_searcher, search, set_bm25, set_rm3
from rag.types import Query


def _configure_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("rag.approach3.evaluate_all_topics")
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


def recall_at_k(qrels_for_topic: Mapping[str, int], ranked_docids: Sequence[str], k: int) -> float:
    """Recall@k for a single topic (binary relevance: rel>0)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    relevant = {docid for docid, rel in qrels_for_topic.items() if int(rel) > 0}
    if not relevant:
        return 0.0
    retrieved_rel = 0
    for docid in ranked_docids[:k]:
        if docid in relevant:
            retrieved_rel += 1
    return float(retrieved_rel) / float(len(relevant))


def write_eval_markdown(
    *,
    output_path: str,
    results: Dict[str, object],
    ce_checkpoint: Optional[str],
    ce_rerank_depth: int,
    alpha: float,
) -> None:
    """Write a markdown report similar to EVAL_PER_TOPIC_V2_BM25_ONLY.md."""
    topics: Sequence[int] = results["topics"]
    no_rm3 = bool(results.get("no_rm3", False))

    bm25_map = float(results["bm25_map"])
    bm25_ap: Mapping[int, float] = results["bm25_ap"]
    bm25_recall_at_5000: Mapping[int, float] = results["bm25_recall_at_5000"]

    ce_map = float(results.get("ce_map", 0.0))
    ce_ap: Mapping[int, float] = results.get("ce_ap", {}) or {}
    ce_recall_at_1000: Mapping[int, float] = results.get("ce_recall_at_1000", {}) or {}

    # Header
    title = "## Per-topic evaluation (ROBUST04 qrels_50_Queries)"
    if ce_checkpoint:
        if no_rm3:
            title += " — Pipeline: BM25 → CE (no RM3)"
        else:
            title += " — Pipeline: BM25+RM3 → CE"

    lines: List[str] = []
    lines.append(title + "\n\n")

    lines.append("**What this doc contains**: per-topic Recall and AP (used for MAP) for:\n")
    lines.append("- **BM25**: Recall@5000 and AP@1000\n")
    if ce_checkpoint:
        if no_rm3:
            lines.append(
                f"- **BM25→CE**: Recall@1000 and AP@1000 (CE reranks BM25 candidates; rerank depth {int(ce_rerank_depth)}, blend α={float(alpha):.3f})\n"
            )
        else:
            lines.append(
                f"- **BM25+RM3→CE**: Recall@1000 and AP@1000 (CE reranks RM3 candidates; rerank depth {int(ce_rerank_depth)}, blend α={float(alpha):.3f})\n"
            )
    lines.append("\n")

    lines.append("**Important notes**:\n")
    lines.append("- “MAP per topic” is **AP**; MAP is the mean of AP over topics.\n")
    if no_rm3:
        lines.append("- This run was computed with RM3 disabled; any “RM3” columns are omitted.\n")
    lines.append("\n")

    # Overall
    lines.append("### Overall (50 topics)\n\n")
    lines.append(f"- **BM25 MAP@1000**: {bm25_map:.4f}\n")
    if ce_checkpoint:
        lines.append(f"- **BM25→CE MAP@1000**: {ce_map:.4f}\n")
        lines.append(f"- **Δ (CE - BM25)**: {ce_map - bm25_map:+.4f}\n")
    lines.append("\n")

    # Table
    lines.append("### Per-topic table\n\n")
    if ce_checkpoint:
        lines.append("| Topic | BM25 Recall@5000 | BM25 AP@1000 | CE Recall@1000 | CE AP@1000 | ΔAP |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for t in topics:
            b_rec = float(bm25_recall_at_5000.get(int(t), 0.0))
            b_ap = float(bm25_ap.get(int(t), 0.0))
            c_rec = float(ce_recall_at_1000.get(int(t), 0.0))
            c_ap = float(ce_ap.get(int(t), 0.0))
            lines.append(
                f"| {int(t)} | {b_rec:.4f} | {b_ap:.4f} | {c_rec:.4f} | {c_ap:.4f} | {c_ap - b_ap:+.4f} |\n"
            )
    else:
        lines.append("| Topic | BM25 Recall@5000 | BM25 AP@1000 |\n")
        lines.append("|---:|---:|---:|\n")
        for t in topics:
            b_rec = float(bm25_recall_at_5000.get(int(t), 0.0))
            b_ap = float(bm25_ap.get(int(t), 0.0))
            lines.append(f"| {int(t)} | {b_rec:.4f} | {b_ap:.4f} |\n")

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.writelines(lines)


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
        obj = json.loads(line.decode("utf-8", errors="replace"))
        return obj.get("text", "")


def _make_doc_text_fetcher(
    *,
    corpus_jsonl: Optional[str],
    index_name: str,
    logger: logging.Logger,
) -> Tuple[Callable[[str], str], Optional[Callable[[], None]]]:
    """Return (fetch_doc_text, close_fn)."""
    if corpus_jsonl:
        db_path = os.path.join(os.path.dirname(str(corpus_jsonl)), "corpus_offsets.sqlite3")
        lookup = _CorpusJsonlLookup(corpus_jsonl=str(corpus_jsonl), sqlite_path=db_path, logger=logger)
        return lookup.get_text, lookup.close

    # Fallback to Pyserini
    searcher = get_searcher(index_name)

    def _fetch(docid: str) -> str:
        return fetch_doc_contents(searcher, str(docid))

    return _fetch, None


def _score_pairs_with_ce(
    *,
    model,
    tokenizer,
    query: str,
    doc_texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> List[float]:
    """Score (query, doc) pairs with cross-encoder."""
    import torch

    if not doc_texts:
        return []

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    scores: List[float] = []
    with torch.no_grad():
        for i in range(0, len(doc_texts), int(batch_size)):
            batch_docs = doc_texts[i : i + int(batch_size)]
            enc = tokenizer(
                [query] * len(batch_docs),
                list(batch_docs),
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits
            # Handle regression (num_labels=1) vs classification (num_labels=2)
            nlab = int(getattr(getattr(model, "config", None), "num_labels", 1) or 1)
            if logits.ndim == 2 and int(logits.shape[1]) >= 2 and nlab >= 2:
                s = logits[:, 1]
            elif logits.ndim == 2 and int(logits.shape[1]) >= 1:
                s = logits[:, 0]
            else:
                s = logits.view(-1)
            scores.extend([float(x) for x in s.detach().cpu().tolist()])
    return scores


def _score_docids_with_ce_batched(
    *,
    model,
    tokenizer,
    query: str,
    docids: Sequence[str],
    fetch_text: Callable[[str], str],
    batch_size: int,
    max_length: int,
    device: str,
    logger: logging.Logger,
    topic_id: int,
    log_every_batches: int = 10,
) -> List[float]:
    """Score (query, doc_text(docid)) pairs with a CE, with batch-level progress logging."""
    import torch

    if not docids:
        return []

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    scores: List[float] = []
    n = int(len(docids))
    bs = max(1, int(batch_size))
    log_every = max(1, int(log_every_batches))
    t0 = time.perf_counter()

    for b_idx, start in enumerate(range(0, n, bs), start=1):
        batch_docids = docids[start : start + bs]
        batch_texts = [fetch_text(d) for d in batch_docids]

        with torch.no_grad():
            enc = tokenizer(
                [query] * len(batch_texts),
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
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

        done_docs = min(n, start + len(batch_docids))
        if (b_idx % log_every) == 0 or done_docs == n:
            elapsed = time.perf_counter() - t0
            rate = done_docs / elapsed if elapsed > 1e-9 else 0.0
            logger.info(
                "CE progress: topic=%d docs=%d/%d batches=%d elapsed=%.1fs rate=%.1f docs/s",
                int(topic_id),
                int(done_docs),
                int(n),
                int(b_idx),
                float(elapsed),
                float(rate),
            )

    return scores


def evaluate_all_topics(
    *,
    queries: Sequence[Query],
    qrels: Mapping[int, Mapping[str, int]],
    index_name: str,
    bm25_topk: int,
    rm3_fb_terms: int,
    rm3_fb_docs: int,
    rm3_orig_weight: float,
    no_rm3: bool,
    ce_checkpoint: Optional[str],
    ce_rerank_depth: int,
    ce_batch_size: int,
    ce_max_length: int,
    ce_device: str,
    ce_log_every_batches: int,
    corpus_jsonl: Optional[str],
    alpha: float,
    map_k: int,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Run evaluation for all topics.
    
    Returns dict with:
      - bm25_ap: {topic_id: ap}
      - rm3_ap: {topic_id: ap}
      - ce_ap: {topic_id: ap} (if ce_checkpoint provided)
      - bm25_map, rm3_map, ce_map
    """
    # Load CE model if provided
    ce_model = None
    ce_tokenizer = None
    if ce_checkpoint:
        try:
            import os
            from pathlib import Path

            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

            # Some HF hub/validators are strict; use Path for local dirs.
            if os.path.isdir(str(ce_checkpoint)):
                p = Path(str(ce_checkpoint)).resolve()
                cfg = AutoConfig.from_pretrained(p)
                ce_tokenizer = AutoTokenizer.from_pretrained(p)
                ce_model = AutoModelForSequenceClassification.from_pretrained(p, config=cfg)
            else:
                ce_tokenizer = AutoTokenizer.from_pretrained(ce_checkpoint)
                ce_model = AutoModelForSequenceClassification.from_pretrained(ce_checkpoint)
            logger.info("Loaded CE model from %s", ce_checkpoint)
        except Exception as e:
            logger.warning("Failed to load CE model from %s: %r", ce_checkpoint, e)
            ce_checkpoint = None

    # Doc text fetcher for CE
    fetch_text: Optional[Callable[[str], str]] = None
    close_text: Optional[Callable[[], None]] = None
    if ce_checkpoint:
        fetch_text, close_text = _make_doc_text_fetcher(
            corpus_jsonl=corpus_jsonl, index_name=index_name, logger=logger
        )

    # Get searcher
    searcher = get_searcher(index_name)

    # Build query lookup
    q_by_id = {int(q.id): q for q in queries}
    topics = sorted(int(t) for t in qrels.keys())

    bm25_run: Dict[int, List[str]] = {}
    rm3_run: Dict[int, List[str]] = {}
    ce_run: Dict[int, List[str]] = {}
    bm25_recall_at_5000: Dict[int, float] = {}
    rm3_recall_at_5000: Dict[int, float] = {}
    ce_recall_at_1000: Dict[int, float] = {}

    t0 = time.perf_counter()
    logger.info(
        "Eval start: topics=%d index=%s bm25_topk=%d map_k=%d ce_rerank_depth=%d alpha=%.3f no_rm3=%s ce=%s ce_log_every_batches=%d",
        len(topics),
        str(index_name),
        int(bm25_topk),
        int(map_k),
        int(ce_rerank_depth),
        float(alpha),
        bool(no_rm3),
        str(ce_checkpoint) if ce_checkpoint else "none",
        int(ce_log_every_batches),
    )
    try:
        for idx, topic_id in enumerate(topics):
            topic_t0 = time.perf_counter()
            logger.info("Topic start: %d (%d/%d)", int(topic_id), int(idx + 1), int(len(topics)))
            q = q_by_id.get(topic_id)
            if q is None:
                logger.warning("No query text for topic %d, skipping", topic_id)
                continue

            # BM25 (need fresh searcher to avoid RM3 state)
            bm25_searcher = get_searcher(index_name)
            set_bm25(bm25_searcher, k1=0.9, b=0.4)
            bm25_hits = search(bm25_searcher, q.text, topk=bm25_topk)
            bm25_docids_all = [h.docid for h in bm25_hits]
            bm25_docids = bm25_docids_all[:map_k]
            bm25_run[topic_id] = bm25_docids
            bm25_recall_at_5000[topic_id] = recall_at_k(qrels.get(topic_id, {}), bm25_docids_all, k=bm25_topk)

            # Lexical stage for CE: either BM25+RM3 or plain BM25 (when --no-rm3)
            if not bool(no_rm3):
                set_bm25(searcher, k1=0.9, b=0.4)
                set_rm3(searcher, fb_terms=rm3_fb_terms, fb_docs=rm3_fb_docs, original_query_weight=rm3_orig_weight)
                rm3_hits = search(searcher, q.text, topk=bm25_topk)
                lex_docids = [h.docid for h in rm3_hits]
                lex_scores = {h.docid: h.score for h in rm3_hits}
                rm3_run[topic_id] = lex_docids[:map_k]
                rm3_recall_at_5000[topic_id] = recall_at_k(qrels.get(topic_id, {}), lex_docids, k=bm25_topk)
            else:
                # Keep rm3_* keys for backward compatibility, but reflect BM25-only pipeline.
                lex_docids = bm25_docids_all
                lex_scores = {h.docid: h.score for h in bm25_hits}
                rm3_run[topic_id] = bm25_docids  # same as BM25 at AP cutoff
                rm3_recall_at_5000[topic_id] = bm25_recall_at_5000[topic_id]

            # CE reranking (on top of lexical candidates)
            if ce_model is not None and fetch_text is not None:
                cand_docids = lex_docids[:ce_rerank_depth]
                ce_scores = _score_docids_with_ce_batched(
                    model=ce_model,
                    tokenizer=ce_tokenizer,
                    query=q.text,
                    docids=cand_docids,
                    fetch_text=fetch_text,
                    batch_size=ce_batch_size,
                    max_length=ce_max_length,
                    device=ce_device,
                    logger=logger,
                    topic_id=int(topic_id),
                    log_every_batches=int(ce_log_every_batches),
                )

                # Normalize scores for blending
                lex_vals = [lex_scores.get(d, 0.0) for d in cand_docids]
                lex_min, lex_max = min(lex_vals), max(lex_vals)
                lex_range = lex_max - lex_min if lex_max > lex_min else 1.0
                lex_norm = [(v - lex_min) / lex_range for v in lex_vals]

                ce_min, ce_max = min(ce_scores), max(ce_scores)
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                ce_norm = [(v - ce_min) / ce_range for v in ce_scores]

                # Blend: final = alpha * ce + (1-alpha) * lexical
                final_scores = [alpha * c + (1 - alpha) * b for c, b in zip(ce_norm, lex_norm)]
                pairs = list(zip(cand_docids, final_scores))
                # Deterministic: final desc, then docid asc
                pairs.sort(key=lambda x: (-x[1], x[0]))
                ce_run[topic_id] = [d for d, _ in pairs][:map_k]
            else:
                ce_run[topic_id] = []
            ce_recall_at_1000[topic_id] = recall_at_k(qrels.get(topic_id, {}), ce_run.get(topic_id, []), k=map_k)

            # Progress / ETA logging
            done = idx + 1
            elapsed = time.perf_counter() - t0
            per_topic = elapsed / float(done) if done > 0 else 0.0
            remaining = (len(topics) - done) * per_topic
            logger.info(
                "Progress: %d/%d topics (topic=%d) topic_s=%.2f elapsed=%.1fs avg_topic_s=%.2f eta=%.1fs",
                done,
                len(topics),
                int(topic_id),
                time.perf_counter() - topic_t0,
                elapsed,
                per_topic,
                remaining,
            )

    finally:
        if close_text is not None:
            try:
                close_text()
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    logger.info("Evaluation complete: %d topics in %.2fs", len(topics), elapsed)

    # Compute AP per topic
    bm25_map, bm25_ap = mean_average_precision(qrels, bm25_run, k=map_k)
    rm3_map, rm3_ap = mean_average_precision(qrels, rm3_run, k=map_k)
    ce_map, ce_ap = mean_average_precision(qrels, ce_run, k=map_k) if ce_checkpoint else (0.0, {})

    return {
        "bm25_ap": bm25_ap,
        "rm3_ap": rm3_ap,
        "ce_ap": ce_ap,
        "bm25_recall_at_5000": bm25_recall_at_5000,
        "rm3_recall_at_5000": rm3_recall_at_5000,
        "ce_recall_at_1000": ce_recall_at_1000,
        "bm25_map": bm25_map,
        "rm3_map": rm3_map,
        "ce_map": ce_map,
        "no_rm3": bool(no_rm3),
        "topics": topics,
        "elapsed_s": elapsed,
    }


def print_results_table(results: Dict[str, object], *, show_ce: bool) -> None:
    """Print a formatted comparison table."""
    topics = results["topics"]
    bm25_ap = results["bm25_ap"]
    rm3_ap = results["rm3_ap"]
    ce_ap = results.get("ce_ap", {})
    no_rm3 = bool(results.get("no_rm3", False))

    if show_ce and no_rm3:
        print(f"\n{'Topic':>6} | {'BM25':>8} | {'BM25+CE':>8} | {'Δ':>8}")
        print("-" * 42)
    elif show_ce:
        print(f"\n{'Topic':>6} | {'BM25':>8} | {'BM25+RM3':>8} | {'CE':>8} | {'CE-RM3':>8}")
        print("-" * 55)
    elif no_rm3:
        print(f"\n{'Topic':>6} | {'BM25':>8}")
        print("-" * 19)
    else:
        print(f"\n{'Topic':>6} | {'BM25':>8} | {'BM25+RM3':>8} | {'Δ':>8}")
        print("-" * 42)

    for t in topics:
        b = bm25_ap.get(t, 0.0)
        r = rm3_ap.get(t, 0.0)
        c = ce_ap.get(t, 0.0) if ce_ap else 0.0
        if show_ce and no_rm3:
            delta = c - b
            print(f"{t:>6} | {b:>8.4f} | {c:>8.4f} | {delta:>+8.4f}")
        elif show_ce:
            delta = c - r
            print(f"{t:>6} | {b:>8.4f} | {r:>8.4f} | {c:>8.4f} | {delta:>+8.4f}")
        elif no_rm3:
            print(f"{t:>6} | {b:>8.4f}")
        else:
            delta = r - b
            print(f"{t:>6} | {b:>8.4f} | {r:>8.4f} | {delta:>+8.4f}")

    if show_ce and no_rm3:
        print("-" * 42)
        print(f"{'MAP':>6} | {results['bm25_map']:>8.4f} | {results['ce_map']:>8.4f} | {results['ce_map'] - results['bm25_map']:>+8.4f}")
    elif show_ce:
        print("-" * 55)
        print(
            f"{'MAP':>6} | {results['bm25_map']:>8.4f} | {results['rm3_map']:>8.4f} | {results['ce_map']:>8.4f} | {results['ce_map'] - results['rm3_map']:>+8.4f}"
        )
    elif no_rm3:
        print("-" * 19)
        print(f"{'MAP':>6} | {results['bm25_map']:>8.4f}")
    else:
        print("-" * 42)
        print(f"{'MAP':>6} | {results['bm25_map']:>8.4f} | {results['rm3_map']:>8.4f} | {results['rm3_map'] - results['bm25_map']:>+8.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate BM25, BM25+RM3, and CE pipeline on all topics.")
    p.add_argument("--queries", default="queriesROBUST.txt")
    p.add_argument("--qrels", default="qrels_50_Queries")
    p.add_argument("--index", default="robust04")

    # BM25 / RM3 settings
    p.add_argument("--bm25-topk", type=int, default=5000)
    p.add_argument("--rm3-fb-terms", type=int, default=50)
    p.add_argument("--rm3-fb-docs", type=int, default=50)
    p.add_argument("--rm3-orig-weight", type=float, default=0.2)
    p.add_argument("--no-rm3", action="store_true", help="Disable RM3 and run BM25 -> CE instead.")

    # CE settings
    p.add_argument("--ce-checkpoint", default=None, help="Path to trained CE checkpoint (optional)")
    p.add_argument("--ce-rerank-depth", type=int, default=5000)
    p.add_argument("--ce-batch-size", type=int, default=32)
    p.add_argument("--ce-max-length", type=int, default=256)
    p.add_argument("--ce-device", default="cpu")
    p.add_argument("--ce-log-every-batches", type=int, default=10, help="Log CE progress every N batches (default: 10)")
    p.add_argument("--alpha", type=float, default=0.2, help="Blend: alpha*CE + (1-alpha)*lexical")
    p.add_argument("--corpus-jsonl", default=None, help="Path to corpus JSONL for doc text lookup")

    # Eval settings
    p.add_argument("--map-k", type=int, default=1000)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--output-json", default=None, help="Optional path to save results as JSON")
    p.add_argument("--output-md", default=None, help="Optional path to save a markdown per-topic report")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = _configure_logging(args.log_level)

    # Helps avoid noisy HF tokenizer parallelism warnings / perf cliffs on some setups.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.info("Loading queries=%s qrels=%s", args.queries, args.qrels)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    results = evaluate_all_topics(
        queries=queries,
        qrels=qrels,
        index_name=args.index,
        bm25_topk=args.bm25_topk,
        rm3_fb_terms=args.rm3_fb_terms,
        rm3_fb_docs=args.rm3_fb_docs,
        rm3_orig_weight=args.rm3_orig_weight,
        no_rm3=bool(args.no_rm3),
        ce_checkpoint=args.ce_checkpoint,
        ce_rerank_depth=args.ce_rerank_depth,
        ce_batch_size=args.ce_batch_size,
        ce_max_length=args.ce_max_length,
        ce_device=args.ce_device,
        ce_log_every_batches=args.ce_log_every_batches,
        corpus_jsonl=args.corpus_jsonl,
        alpha=args.alpha,
        map_k=args.map_k,
        logger=logger,
    )

    show_ce = args.ce_checkpoint is not None
    print_results_table(results, show_ce=show_ce)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            # Convert int keys to str for JSON
            out = {
                "bm25_ap": {str(k): v for k, v in results["bm25_ap"].items()},
                "rm3_ap": {str(k): v for k, v in results["rm3_ap"].items()},
                "ce_ap": {str(k): v for k, v in results.get("ce_ap", {}).items()},
                "bm25_recall_at_5000": {str(k): v for k, v in results["bm25_recall_at_5000"].items()},
                "rm3_recall_at_5000": {str(k): v for k, v in results["rm3_recall_at_5000"].items()},
                "ce_recall_at_1000": {str(k): v for k, v in results["ce_recall_at_1000"].items()},
                "bm25_map": results["bm25_map"],
                "rm3_map": results["rm3_map"],
                "ce_map": results.get("ce_map", 0.0),
                "elapsed_s": results["elapsed_s"],
                "no_rm3": bool(results.get("no_rm3", False)),
                "topics": [int(t) for t in results.get("topics", [])],
            }
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    if args.output_md:
        write_eval_markdown(
            output_path=str(args.output_md),
            results=results,
            ce_checkpoint=args.ce_checkpoint,
            ce_rerank_depth=int(args.ce_rerank_depth),
            alpha=float(args.alpha),
        )
        print(f"\nMarkdown report saved to: {args.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
