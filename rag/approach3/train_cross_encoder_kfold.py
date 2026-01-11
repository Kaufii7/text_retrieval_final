"""Train a cross-encoder reranker with topic-level K-fold CV (20/20/10 splits).

This script fine-tunes a HuggingFace sequence-classification model as a cross-encoder
for (query, doc_text) relevance scoring, using:
- `queriesROBUST.txt` for query texts (topic_id<TAB>query_text)
- `qrels_50_Queries` for ground-truth labels (topic_id 0 doc_id relevance)
- a precomputed dense embedding matrix + aligned docids list to generate hard negatives

Split scheme (as requested):
- Total topics: 50
- K folds (default 5): each fold holds out 10 topics as TEST
- Remaining 40 topics are split into 20 TRAIN topics and 20 VALIDATION topics

Training data:
- Positives: (query, doc) where qrels relevance >= threshold
- Negatives: sampled from dense candidates not labeled relevant for that topic (hard negatives)

Evaluation:
- Builds a dense candidate list per query and reranks it with the trained cross-encoder
- Reports MAP@K on the TEST topics (and optionally on VAL topics too)

Dependencies (optional; required to run this script):
- torch
- transformers
- pyserini (already used by this repo for doc text fetching)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from rag.approaches.approach3 import _embed_query, _load_dense_index  # type: ignore
from rag.eval import mean_average_precision
from rag.io import load_qrels, load_queries
from rag.types import Query


def _require_training_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Cross-encoder fine-tuning requires optional deps: torch + transformers (AutoTokenizer/AutoModel). "
            "Install them to run this script."
        ) from e


def _has_hf_trainer() -> bool:
    """Return True if transformers.Trainer is available in this environment."""
    try:
        from transformers import Trainer  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class PairExample:
    topic_id: int
    query: str
    docid: str
    doc_text: str
    label: int  # 0/1


class _DocTextCache:
    """Small best-effort cache for docid->text to avoid repeated lookups."""

    def __init__(self, fetch: Callable[[str], str], *, max_items: int = 50_000):
        self._fetch = fetch
        self._max_items = int(max_items)
        self._cache: MutableMapping[str, str] = {}

    def get(self, docid: str) -> str:
        k = str(docid)
        if k in self._cache:
            return self._cache[k]
        txt = self._fetch(k)
        # keep cache bounded (simple random eviction; good enough)
        if len(self._cache) >= self._max_items:
            try:
                self._cache.pop(next(iter(self._cache)))
            except Exception:
                self._cache.clear()
        self._cache[k] = txt
        return txt


class _CorpusJsonlLookup:
    """Docid->text lookup for a corpus JSONL using a SQLite offset index.

    The corpus is expected to have one JSON object per line with keys:
      {"docid": "...", "text": "..."}
    """

    def __init__(self, *, corpus_jsonl: str, sqlite_path: str, logger: logging.Logger):
        self.corpus_jsonl = str(corpus_jsonl)
        self.sqlite_path = str(sqlite_path)
        self._log = logger

        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs (docid TEXT PRIMARY KEY, offset INTEGER NOT NULL)"
        )
        self._conn.commit()

        if not self._has_rows():
            self._build_index()

        # Keep file handle open for random access reads.
        self._fh = open(self.corpus_jsonl, "rb")

    def close(self) -> None:
        try:
            if hasattr(self, "_fh") and self._fh:
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
        self._log.info("Building corpus offset index (first time): corpus=%s db=%s", self.corpus_jsonl, self.sqlite_path)
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
        elapsed = time.perf_counter() - t0
        self._log.info("Corpus index built: docs=%d elapsed=%.2fs", n, elapsed)

    def get_text(self, docid: str) -> str:
        cur = self._conn.execute("SELECT offset FROM docs WHERE docid = ?", (str(docid),))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"docid not found in corpus index: {docid}")
        off = int(row[0])
        self._fh.seek(off)
        line = self._fh.readline()
        obj = json.loads(line.decode("utf-8", errors="replace"))
        txt = obj.get("text", "")
        return txt if isinstance(txt, str) else str(txt)


def _make_doc_text_fetcher(
    *,
    corpus_jsonl: Optional[str],
    corpus_index_db: Optional[str],
    index_name: str,
    logger: logging.Logger,
) -> Tuple[Callable[[str], str], Optional[Callable[[], None]]]:
    """Return (fetch_doc_text, close_fn). Prefer corpus-jsonl to avoid Java/Pyserini."""
    if corpus_jsonl:
        db_path = str(corpus_index_db or (os.path.join(os.path.dirname(str(corpus_jsonl)), "corpus_offsets.sqlite3")))
        lookup = _CorpusJsonlLookup(corpus_jsonl=str(corpus_jsonl), sqlite_path=db_path, logger=logger)
        return lookup.get_text, lookup.close

    # Fallback to Pyserini (Java) for text fetching
    from rag.lucene_backend import fetch_doc_contents, get_searcher

    searcher = get_searcher(index_name)

    def _fetch(docid: str) -> str:
        return fetch_doc_contents(searcher, str(docid))

    def _close() -> None:
        # Pyserini handles JVM lifecycle; nothing to close explicitly here.
        return None

    return _fetch, _close


def _configure_logging(*, out_dir: str, level: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("rag.approach3.train_cross_encoder_kfold")
    log.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.propagate = False

    # Idempotent: clear existing handlers if re-run in the same process.
    for h in list(log.handlers):
        log.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    log.addHandler(sh)
    log.addHandler(fh)
    return log


def _ensure_hf_config(output_dir: str, *, model, logger: logging.Logger) -> None:
    """Ensure output_dir/config.json is a valid HF config with model_type.

    Some environments may end up with a non-standard config.json in the output
    directory (e.g., due to partial writes or external tooling). AutoTokenizer /
    AutoConfig require `model_type` to be present.
    """
    try:
        cfg_path = os.path.join(str(output_dir), "config.json")
        obj = None
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                obj = None
        if not isinstance(obj, dict) or not str(obj.get("model_type") or "").strip():
            # Rewrite a valid HF config from the model we just trained/saved.
            # Prefer save_pretrained (writes config.json + extras) but keep a direct fallback too.
            try:
                model.config.save_pretrained(str(output_dir))
            except Exception:
                model.config.to_json_file(cfg_path)
            logger.info("Rewrote missing/invalid config.json with model_type at: %s", cfg_path)
    except Exception as e:
        logger.warning("Unable to validate/rewrite config.json in %s: %r", str(output_dir), e)


def _topic_queries(queries: Sequence[Query]) -> Dict[int, Query]:
    return {int(q.id): q for q in queries}


def _split_topics_20_20_10(
    topics: Sequence[int], *, k: int, seed: int
) -> List[Tuple[List[int], List[int], List[int]]]:
    """Return list of folds as (train_topics, val_topics, test_topics)."""
    t = list(sorted(set(int(x) for x in topics)))
    rng = random.Random(int(seed))
    rng.shuffle(t)

    folds: List[List[int]] = [[] for _ in range(int(k))]
    for i, topic_id in enumerate(t):
        folds[i % int(k)].append(topic_id)

    out: List[Tuple[List[int], List[int], List[int]]] = []
    for fold_idx in range(int(k)):
        test = sorted(folds[fold_idx])
        rest = [x for i, f in enumerate(folds) if i != fold_idx for x in f]
        rr = random.Random(int(seed) + 10_000 + int(fold_idx))
        rr.shuffle(rest)
        # requested fixed sizes (20/20/10) for 50 topics
        train = sorted(rest[:20])
        val = sorted(rest[20:40])
        out.append((train, val, test))
    return out


def _build_examples_for_topics(
    *,
    topic_ids: Sequence[int],
    queries_by_id: Mapping[int, Query],
    qrels: Mapping[int, Mapping[str, int]],
    dense_cfg: Mapping[str, object],
    candidates_depth: int,
    label_rel_threshold: int,
    neg_per_pos: int,
    hard_neg_ratio: float,
    seed: int,
    doc_cache: _DocTextCache,
    logger: logging.Logger,
) -> List[PairExample]:
    """Build (query, doc_text) examples for the given topics.
    
    Negative sampling strategy:
    - hard_neg_ratio (0.0 to 1.0): fraction of negatives that are "hard" 
      (top-ranked non-relevant docs from dense retrieval)
    - The remaining (1 - hard_neg_ratio) are sampled randomly from the pool
    - Negatives are deduplicated per positive to avoid redundancy
    
    Example: neg_per_pos=7, hard_neg_ratio=0.7 -> 5 hard + 2 random per positive
    """
    topic_ids = list(int(t) for t in topic_ids)
    rng = random.Random(int(seed))
    index = _load_dense_index({"dense": dict(dense_cfg)})

    model_name = str(dense_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
    device = str(dense_cfg.get("device", "cpu"))
    normalize_embeddings = bool(dense_cfg.get("normalize_embeddings", True))
    metric = str(dense_cfg.get("metric", "cosine")).lower()
    ef = None
    hnsw = dense_cfg.get("hnsw") or {}
    if isinstance(hnsw, dict) and hnsw.get("ef") is not None:
        ef = int(hnsw.get("ef"))

    # Compute hard/random split
    n_neg = int(neg_per_pos)
    n_hard = max(0, min(n_neg, int(round(float(hard_neg_ratio) * n_neg))))
    n_random = n_neg - n_hard
    logger.info(
        "Negative sampling: neg_per_pos=%d hard_neg_ratio=%.2f -> %d hard + %d random per positive",
        n_neg, float(hard_neg_ratio), n_hard, n_random,
    )

    out: List[PairExample] = []
    t0 = time.perf_counter()
    topics_used = 0
    n_topics = len(topic_ids)
    
    for topic_id in sorted(topic_ids):
        q = queries_by_id.get(topic_id)
        if q is None:
            continue
        qrels_topic = qrels.get(topic_id, {}) or {}
        rel_docids = sorted([docid for docid, rel in qrels_topic.items() if int(rel) >= int(label_rel_threshold)])
        if not rel_docids:
            continue

        qvec = _embed_query(q.text, model_name=model_name, device=device, normalize_embeddings=normalize_embeddings)
        candidates = index.search(qvec, topk=int(candidates_depth), ef=ef)
        # candidates are ordered by score (highest first) - these are "hard" negatives
        candidate_docids = [docid for docid, _s in candidates]
        rel_set = set(rel_docids)
        
        # neg_pool preserves ranking order: first items are hardest (highest-scored non-relevant)
        neg_pool = [d for d in candidate_docids if d not in rel_set]
        if not neg_pool:
            continue

        # Track negatives used globally for this topic to enable variety across positives
        hard_neg_ptr = 0  # pointer into neg_pool for round-robin hard negative assignment
        
        for docid in rel_docids:
            txt = doc_cache.get(docid)
            out.append(PairExample(topic_id=topic_id, query=q.text, docid=docid, doc_text=txt, label=1))
            
            used_negs: set = set()
            
            # 1) Hard negatives: take from top of neg_pool (round-robin across positives)
            for _ in range(n_hard):
                if hard_neg_ptr >= len(neg_pool):
                    hard_neg_ptr = 0  # wrap around
                nd = neg_pool[hard_neg_ptr]
                hard_neg_ptr += 1
                if nd in used_negs:
                    continue  # skip duplicates
                used_negs.add(nd)
                ntxt = doc_cache.get(nd)
                out.append(PairExample(topic_id=topic_id, query=q.text, docid=nd, doc_text=ntxt, label=0))
            
            # 2) Random negatives: sample from remaining pool (not already used)
            random_pool = [d for d in neg_pool if d not in used_negs]
            if random_pool and n_random > 0:
                # Sample without replacement if possible
                k_sample = min(n_random, len(random_pool))
                sampled = rng.sample(random_pool, k_sample)
                for nd in sampled:
                    ntxt = doc_cache.get(nd)
                    out.append(PairExample(topic_id=topic_id, query=q.text, docid=nd, doc_text=ntxt, label=0))

        topics_used += 1
        if topics_used % 5 == 0:
            logger.info(
                "Built examples: topics_done=%d/%d examples=%d metric=%s",
                topics_used,
                n_topics,
                len(out),
                metric,
            )

    logger.info(
        "Built examples done: topics=%d examples=%d elapsed=%.2fs",
        topics_used,
        len(out),
        time.perf_counter() - t0,
    )
    return out


def _make_hf_dataset(
    examples: Sequence[PairExample],
    *,
    tokenizer,
    max_length: int,
    label_mode: str,
):
    _require_training_deps()
    import torch

    lm = str(label_mode).lower().strip()
    if lm not in ("classification", "regression"):
        raise ValueError(f"label_mode must be classification|regression, got: {label_mode!r}")

    class _DS(torch.utils.data.Dataset):
        def __init__(self, exs: Sequence[PairExample]):
            self.exs = list(exs)

        def __len__(self) -> int:
            return len(self.exs)

        def __getitem__(self, idx: int):
            ex = self.exs[idx]
            enc = tokenizer(
                ex.query,
                ex.doc_text,
                truncation=True,
                padding="max_length",
                max_length=int(max_length),
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if lm == "classification":
                item["labels"] = torch.tensor(int(ex.label), dtype=torch.long)
            else:
                # Regression head expects float labels (shape: [batch])
                item["labels"] = torch.tensor(float(ex.label), dtype=torch.float)
            return item

    return _DS(examples)


def _train_one_fold(
    *,
    fold_idx: int,
    train_examples: Sequence[PairExample],
    dev_examples: Sequence[PairExample],
    output_dir: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    patience: int,
    seed: int,
    ce_num_labels: Optional[int],
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    logger: logging.Logger,
) -> Dict[str, object]:
    _require_training_deps()
    import math

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_cfg = AutoConfig.from_pretrained(model_name)
    desired_labels = int(ce_num_labels) if ce_num_labels is not None else int(getattr(base_cfg, "num_labels", 1) or 1)
    label_mode = "regression" if int(desired_labels) == 1 else "classification"
    logger.info(
        "Fold %d model init: base_model=%s base_num_labels=%s desired_num_labels=%d label_mode=%s",
        int(fold_idx),
        str(model_name),
        str(getattr(base_cfg, "num_labels", None)),
        int(desired_labels),
        str(label_mode),
    )
    # If overriding num_labels, allow head resize.
    if ce_num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=int(desired_labels), ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    train_ds = _make_hf_dataset(train_examples, tokenizer=tokenizer, max_length=int(max_length), label_mode=label_mode)
    dev_ds = _make_hf_dataset(dev_examples, tokenizer=tokenizer, max_length=int(max_length), label_mode=label_mode)

    logger.info(
        "Fold %d train start: train_size=%d dev_size=%d model=%s",
        int(fold_idx),
        len(train_examples),
        len(dev_examples),
        str(model_name),
    )

    # Prefer HF Trainer if present; otherwise use a minimal torch loop.
    if _has_hf_trainer():
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
        from transformers import get_linear_schedule_with_warmup

        # Warmup steps are computed from total training steps.
        # (Trainer will handle the schedule if we pass warmup_ratio/warmup_steps.)
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            num_train_epochs=float(num_epochs),
            per_device_train_batch_size=int(batch_size),
            per_device_eval_batch_size=int(batch_size),
            learning_rate=float(lr),
            weight_decay=float(weight_decay),
            warmup_ratio=float(warmup_ratio),
            max_grad_norm=float(max_grad_norm),
            logging_strategy="epoch",
            seed=int(seed),
            report_to=[],  # disable wandb/etc.
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=int(patience))],
        )

        train_out = trainer.train()
        eval_out = trainer.evaluate()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        _ensure_hf_config(output_dir, model=trainer.model, logger=logger)
        report = {
            "fold": int(fold_idx),
            "train_size": int(len(train_examples)),
            "dev_size": int(len(dev_examples)),
            "backend": "hf_trainer",
            "label_mode": str(label_mode),
            "num_labels": int(getattr(model.config, "num_labels", desired_labels)),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "max_grad_norm": float(max_grad_norm),
            "train_runtime": float(getattr(train_out, "training_time", 0.0) or 0.0),
            "eval": {k: float(v) if isinstance(v, (int, float)) else v for k, v in (eval_out or {}).items()},
            "checkpoint_dir": str(output_dir),
        }
        logger.info("Fold %d train done (hf_trainer).", int(fold_idx))
    else:
        # ---- Minimal torch training loop (no transformers.Trainer dependency) ----
        device = os.environ.get("CE_TRAIN_DEVICE", "cpu")
        dev = torch.device(device)
        model.to(dev)

        import torch.nn.functional as F

        def _collate(batch):
            keys = batch[0].keys()
            out = {}
            for k in keys:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            return out

        def _forward_and_loss(batch_tensors) -> Tuple[torch.Tensor, torch.Tensor]:
            """Return (loss, logits). Compute loss manually if model doesn't provide it."""
            labels = batch_tensors.get("labels")
            inputs = {k: v for k, v in batch_tensors.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits

            # Some checkpoints / transformer versions may not return outputs.loss reliably.
            if labels is None:
                raise RuntimeError("Missing labels in batch")

            if str(label_mode) == "classification":
                # logits: [B, C], labels: [B]
                loss = F.cross_entropy(logits, labels.long())
            else:
                # regression: logits [B,1] or [B]; labels [B]
                pred = logits.view(-1).float()
                tgt = labels.view(-1).float()
                loss = F.mse_loss(pred, tgt)
            return loss, logits

        g = torch.Generator()
        g.manual_seed(int(seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=True,
            generator=g,
            collate_fn=_collate,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=_collate,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        # Simple linear LR schedule (no warmup) if transformers scheduler available.
        sched = None
        try:
            from transformers import get_linear_schedule_with_warmup

            total_steps = max(1, int(num_epochs) * max(1, len(train_loader)))
            warmup_steps = int(float(warmup_ratio) * float(total_steps))
            sched = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(warmup_steps), num_training_steps=int(total_steps)
            )
        except Exception:
            sched = None

        best_dev = float("inf")
        best_epoch = -1
        bad_epochs = 0
        t0 = time.perf_counter()
        saved_any = False
        blew_up = False

        for epoch in range(int(num_epochs)):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = {k: v.to(dev) for k, v in batch.items()}
                loss, _logits = _forward_and_loss(batch)
                if not torch.isfinite(loss):
                    logger.error(
                        "Fold %d: non-finite loss detected (epoch=%d). Aborting training early. loss=%r",
                        int(fold_idx),
                        int(epoch) + 1,
                        float(loss.detach().cpu().item()) if loss is not None else None,
                    )
                    blew_up = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
                optimizer.step()
                if sched is not None:
                    sched.step()
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1
            if blew_up:
                break

            train_loss = epoch_loss / float(max(1, n_batches))

            # Dev loss
            model.eval()
            dev_loss_sum = 0.0
            dev_batches = 0
            with torch.no_grad():
                for batch in dev_loader:
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    loss, _logits = _forward_and_loss(batch)
                    if not torch.isfinite(loss):
                        blew_up = True
                        break
                    dev_loss_sum += float(loss.detach().cpu().item())
                    dev_batches += 1
            if blew_up:
                logger.error("Fold %d: non-finite dev loss detected. Aborting training early.", int(fold_idx))
                break
            dev_loss = dev_loss_sum / float(max(1, dev_batches))

            logger.info(
                "Fold %d epoch %d/%d: train_loss=%.6f dev_loss=%.6f",
                int(fold_idx),
                epoch + 1,
                int(num_epochs),
                float(train_loss),
                float(dev_loss),
            )

            # Early stopping
            if dev_loss + 1e-9 < best_dev:
                best_dev = dev_loss
                best_epoch = epoch
                bad_epochs = 0
                # Save best checkpoint
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                _ensure_hf_config(output_dir, model=model, logger=logger)
                saved_any = True
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    logger.info(
                        "Fold %d early stopping at epoch %d (best_epoch=%d best_dev_loss=%.6f)",
                        int(fold_idx),
                        epoch + 1,
                        best_epoch + 1,
                        float(best_dev),
                    )
                    break

        # Ensure we always persist a loadable HF checkpoint even if no "best" was found.
        if not saved_any:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            _ensure_hf_config(output_dir, model=model, logger=logger)

        train_runtime = time.perf_counter() - t0
        report = {
            "fold": int(fold_idx),
            "train_size": int(len(train_examples)),
            "dev_size": int(len(dev_examples)),
            "backend": "torch_loop",
            "label_mode": str(label_mode),
            "num_labels": int(getattr(model.config, "num_labels", desired_labels)),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "max_grad_norm": float(max_grad_norm),
            "blew_up": bool(blew_up),
            "train_runtime": float(train_runtime),
            "eval": {"eval_loss": float(best_dev), "best_epoch": int(best_epoch + 1 if best_epoch >= 0 else 1)},
            "checkpoint_dir": str(output_dir),
            "train_device": str(device),
        }
        logger.info("Fold %d train done (torch_loop).", int(fold_idx))

    logger.info("Fold %d train done: %s", int(fold_idx), json.dumps(report, sort_keys=True))
    return report


def _score_pairs_with_hf(
    *,
    model,
    tokenizer,
    query: str,
    doc_texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> List[float]:
    _require_training_deps()
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
            # If regression (num_labels==1), use logits[:,0]; else use positive-class logit.
            nlab = int(getattr(getattr(model, "config", None), "num_labels", 1) or 1)
            if logits.ndim == 2 and int(logits.shape[1]) >= 2 and nlab >= 2:
                s = logits[:, 1]
            elif logits.ndim == 2 and int(logits.shape[1]) >= 1:
                s = logits[:, 0]
            else:
                s = logits.view(-1)
            scores.extend([float(x) for x in s.detach().cpu().tolist()])
    return scores


def _rerank_and_eval_map(
    *,
    topic_ids: Sequence[int],
    queries_by_id: Mapping[int, Query],
    qrels: Mapping[int, Mapping[str, int]],
    dense_cfg: Mapping[str, object],
    ce_checkpoint_dir: str,
    candidates_depth: int,
    rerank_depth: int,
    map_k: int,
    ce_batch_size: int,
    max_length: int,
    ce_device: str,
    doc_cache: _DocTextCache,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Compute MAP@k on `topic_ids` using dense candidates reranked by the cross-encoder."""
    _require_training_deps()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    index = _load_dense_index({"dense": dict(dense_cfg)})
    model_name = str(dense_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
    be_device = str(dense_cfg.get("device", "cpu"))
    normalize_embeddings = bool(dense_cfg.get("normalize_embeddings", True))
    ef = None
    hnsw = dense_cfg.get("hnsw") or {}
    if isinstance(hnsw, dict) and hnsw.get("ef") is not None:
        ef = int(hnsw.get("ef"))

    try:
        tok = AutoTokenizer.from_pretrained(ce_checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ce_checkpoint_dir)
    except Exception as e:
        raise RuntimeError(
            "Failed to load trained checkpoint for evaluation. This usually means the checkpoint directory "
            "is missing a valid HuggingFace `config.json` with a `model_type` field.\n\n"
            f"checkpoint_dir={ce_checkpoint_dir}\n"
            f"original_error={e!r}\n\n"
            "If this is an existing run directory, delete it and rerun training.\n"
            "If it keeps happening, run with a fresh --out-dir and ensure the process has write permissions."
        ) from e

    run: Dict[int, List[str]] = {}
    t0 = time.perf_counter()
    for topic_id in sorted(int(t) for t in topic_ids):
        q = queries_by_id.get(topic_id)
        if q is None:
            continue

        qvec = _embed_query(q.text, model_name=model_name, device=be_device, normalize_embeddings=normalize_embeddings)
        candidates = index.search(qvec, topk=int(candidates_depth), ef=ef)
        cand_docids = [docid for docid, _s in candidates][: int(rerank_depth)]
        cand_texts = [doc_cache.get(d) for d in cand_docids]
        scores = _score_pairs_with_hf(
            model=model,
            tokenizer=tok,
            query=q.text,
            doc_texts=cand_texts,
            batch_size=int(ce_batch_size),
            max_length=int(max_length),
            device=str(ce_device),
        )
        pairs = list(zip(cand_docids, scores))
        pairs.sort(key=lambda x: (-float(x[1]), str(x[0])))
        run[int(topic_id)] = [docid for docid, _s in pairs][: int(map_k)]

    # Evaluate MAP@k on these topics only
    qrels_subset = {int(t): qrels[int(t)] for t in topic_ids if int(t) in qrels}
    map_value, ap_by_topic = mean_average_precision(qrels_subset, run, k=int(map_k))
    elapsed = time.perf_counter() - t0
    logger.info(
        "Eval MAP@%d: topics=%d map=%.6f elapsed=%.2fs",
        int(map_k),
        len(qrels_subset),
        float(map_value),
        float(elapsed),
    )
    return {
        "topics": sorted(int(t) for t in qrels_subset.keys()),
        "map_k": int(map_k),
        "map": float(map_value),
        "ap_by_topic": {int(k): float(v) for k, v in ap_by_topic.items()},
        "elapsed_s": float(elapsed),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a cross-encoder with 5-fold CV (20/20/10 topics).")
    p.add_argument("--queries", default="queriesROBUST.txt")
    p.add_argument("--qrels", default="qrels_50_Queries")
    p.add_argument("--index", default="robust04", help="Only used if --corpus-jsonl is not provided.")
    p.add_argument("--out-dir", default="models/approach3_ce_kfold")
    p.add_argument("--log-level", default="INFO")

    # Doc text source (prefer corpus to avoid Java on macOS)
    p.add_argument(
        "--corpus-jsonl",
        default=None,
        help="Path to corpus JSONL with {'docid','text'} per line. If set, avoids Pyserini/Java.",
    )
    p.add_argument(
        "--corpus-index-db",
        default=None,
        help="Optional path for SQLite corpus offset index (default: alongside corpus-jsonl).",
    )

    # Dense assets / bi-encoder (for candidate generation)
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy (doc embeddings).")
    p.add_argument("--docids", required=True, help="Path to docids .txt aligned with embeddings.")
    p.add_argument("--dense-backend", default="exact", help="Dense backend: exact | hnswlib (if installed).")
    p.add_argument("--dense-metric", default="cosine", help="Dense metric: cosine | ip (inner product).")
    p.add_argument("--bi-encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--bi-encoder-device", default="cpu")
    p.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable query embedding normalization (default: normalize).",
    )

    # CV / sampling
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-rel-threshold", type=int, default=1)
    p.add_argument("--candidates-depth", type=int, default=2000)
    p.add_argument("--rerank-depth", type=int, default=500, help="How many dense candidates to rerank for MAP eval.")
    p.add_argument("--neg-per-pos", type=int, default=7, help="Number of negatives per positive example (default: 7).")
    p.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.7,
        help="Fraction of negatives that are 'hard' (top-ranked non-relevant). Default: 0.7 (5 hard + 2 random for neg_per_pos=7).",
    )

    # Cross-encoder training
    p.add_argument("--ce-base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument(
        "--ce-num-labels",
        type=int,
        default=2,
        help="num_labels for CE head. Default: 2 (classification, more stable than regression).",
    )
    p.add_argument("--max-length", type=int, default=384, help="Max sequence length (default: 384).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5, lower is more stable).")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    p.add_argument("--patience", type=int, default=2, help="Early stopping patience (default: 2).")

    # Evaluation
    p.add_argument("--map-k", type=int, default=1000)
    p.add_argument("--ce-eval-device", default="cpu", help="Device for CE evaluation scoring (e.g., cpu|cuda).")
    p.add_argument("--ce-eval-batch-size", type=int, default=16)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = os.path.abspath(str(args.out_dir))
    log = _configure_logging(out_dir=out_dir, level=str(args.log_level))
    log.info("cwd=%s out_dir=%s", os.getcwd(), out_dir)

    # Reduce thread contention on macOS; also avoids tokenizer thread spam.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    log.info("Loading queries=%s qrels=%s", str(args.queries), str(args.qrels))
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    topics = sorted(int(t) for t in qrels.keys())
    log.info("Loaded topics=%d (expected 50)", len(topics))

    q_by_id = _topic_queries(queries)
    fetch_text, close_text = _make_doc_text_fetcher(
        corpus_jsonl=args.corpus_jsonl,
        corpus_index_db=args.corpus_index_db,
        index_name=str(args.index),
        logger=log,
    )
    doc_cache = _DocTextCache(fetch_text, max_items=50_000)

    dense_cfg: Dict[str, object] = {
        "backend": str(args.dense_backend),
        "metric": str(args.dense_metric),
        "embeddings_path": str(args.embeddings),
        "docids_path": str(args.docids),
        "model_name": str(args.bi_encoder_model),
        "device": str(args.bi_encoder_device),
        "normalize_embeddings": not bool(args.no_normalize_embeddings),
    }

    folds = _split_topics_20_20_10(topics, k=int(args.folds), seed=int(args.seed))
    log.info("Prepared %d folds with 20/20/10 topic split each.", len(folds))

    fold_reports: List[Dict[str, object]] = []
    try:
        for fold_idx, (train_topics, val_topics, test_topics) in enumerate(folds):
            log.info(
                "Fold %d topics: train=%d val=%d test=%d",
                int(fold_idx),
                len(train_topics),
                len(val_topics),
                len(test_topics),
            )

            train_ex = _build_examples_for_topics(
                topic_ids=train_topics,
                queries_by_id=q_by_id,
                qrels=qrels,
                dense_cfg=dense_cfg,
                candidates_depth=int(args.candidates_depth),
                label_rel_threshold=int(args.label_rel_threshold),
                neg_per_pos=int(args.neg_per_pos),
                hard_neg_ratio=float(args.hard_neg_ratio),
                seed=int(args.seed) + 1000 + int(fold_idx),
                doc_cache=doc_cache,
                logger=log,
            )
            val_ex = _build_examples_for_topics(
                topic_ids=val_topics,
                queries_by_id=q_by_id,
                qrels=qrels,
                dense_cfg=dense_cfg,
                candidates_depth=int(args.candidates_depth),
                label_rel_threshold=int(args.label_rel_threshold),
                neg_per_pos=int(args.neg_per_pos),
                hard_neg_ratio=float(args.hard_neg_ratio),
                seed=int(args.seed) + 2000 + int(fold_idx),
                doc_cache=doc_cache,
                logger=log,
            )

            fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
            report = _train_one_fold(
                fold_idx=fold_idx,
                train_examples=train_ex,
                dev_examples=val_ex,
                output_dir=fold_dir,
                model_name=str(args.ce_base_model),
                max_length=int(args.max_length),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                num_epochs=int(args.epochs),
                patience=int(args.patience),
                seed=int(args.seed) + int(fold_idx),
            ce_num_labels=args.ce_num_labels if args.ce_num_labels is None else int(args.ce_num_labels),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            max_grad_norm=float(args.max_grad_norm),
                logger=log,
            )

            test_eval = _rerank_and_eval_map(
                topic_ids=test_topics,
                queries_by_id=q_by_id,
                qrels=qrels,
                dense_cfg=dense_cfg,
                ce_checkpoint_dir=fold_dir,
                candidates_depth=int(args.candidates_depth),
                rerank_depth=int(args.rerank_depth),
                map_k=int(args.map_k),
                ce_batch_size=int(args.ce_eval_batch_size),
                max_length=int(args.max_length),
                ce_device=str(args.ce_eval_device),
                doc_cache=doc_cache,
                logger=log,
            )

            val_eval = _rerank_and_eval_map(
                topic_ids=val_topics,
                queries_by_id=q_by_id,
                qrels=qrels,
                dense_cfg=dense_cfg,
                ce_checkpoint_dir=fold_dir,
                candidates_depth=int(args.candidates_depth),
                rerank_depth=int(args.rerank_depth),
                map_k=int(args.map_k),
                ce_batch_size=int(args.ce_eval_batch_size),
                max_length=int(args.max_length),
                ce_device=str(args.ce_eval_device),
                doc_cache=doc_cache,
                logger=log,
            )

            report["topics"] = {"train": train_topics, "val": val_topics, "test": test_topics}
            report["eval_map"] = {"val": val_eval, "test": test_eval}
            fold_reports.append(report)

            # Persist fold report as we go
            with open(os.path.join(fold_dir, "fold_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
                f.write("\n")
    finally:
        if close_text is not None:
            try:
                close_text()
            except Exception:
                pass

    summary = {
        "folds": int(args.folds),
        "seed": int(args.seed),
        "data": {"queries": str(args.queries), "qrels": str(args.qrels), "index": str(args.index)},
        "dense": dense_cfg,
        "params": {
            "label_rel_threshold": int(args.label_rel_threshold),
            "candidates_depth": int(args.candidates_depth),
            "rerank_depth": int(args.rerank_depth),
            "neg_per_pos": int(args.neg_per_pos),
            "ce_base_model": str(args.ce_base_model),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_grad_norm": float(args.max_grad_norm),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "map_k": int(args.map_k),
            "ce_eval_device": str(args.ce_eval_device),
            "ce_eval_batch_size": int(args.ce_eval_batch_size),
        },
        "folds_report": fold_reports,
    }
    out_path = os.path.join(out_dir, "cv_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    log.info("Wrote CV summary: %s", out_path)
    print(f"Wrote CV summary: {out_path}")
    print(f"Fold checkpoints under: {out_dir}/fold_*")
    print(f"Logs: {os.path.join(out_dir, 'train.log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

