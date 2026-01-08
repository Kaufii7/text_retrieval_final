"""PR A3-7 (optional): Fine-tune a cross-encoder on the 50 labeled topics (CV + early stopping).

This is an **optional** training utility. It is dependency-gated:
- Requires: torch, transformers
- Uses: Pyserini (already used elsewhere) to fetch doc contents by docid

Training data (per the plan):
- Positives: (query, doc) where qrels relevance >= threshold
- Negatives: sampled from dense candidates not labeled relevant for that topic

We do **topic-level K-fold CV**:
- split topics into K folds deterministically (seeded)
- train on K-1 folds, validate on held-out fold
- early stopping on validation loss

Artifacts:
- writes per-fold checkpoints under `models/approach3_ce/fold_{i}/`
- writes a summary JSON under `models/approach3_ce/cv_summary.json`

Notes:
- This script does *not* change default inference behavior.
- To use a fine-tuned model at inference time, set in config:
    params["rerank"]["use_finetuned"]=True
    params["rerank"]["finetuned_model_dir"]="models/approach3_ce/best" (or a fold dir)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from rag.approaches.approach3 import _embed_query, _load_dense_index  # type: ignore
from rag.config import ApproachConfig, default_approach3_config
from rag.io import load_qrels, load_queries
from rag.lucene_backend import fetch_doc_contents, get_searcher
from rag.types import Query


def _require_training_deps():
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        raise RuntimeError(
            "Fine-tuning requires optional deps: torch + transformers. "
            "Install them to run A3-7 fine-tuning."
        ) from e


@dataclass(frozen=True)
class PairExample:
    topic_id: int
    query: str
    docid: str
    doc_text: str
    label: int  # 0/1


def _split_topics_kfold(topics: Sequence[int], *, k: int, seed: int) -> List[List[int]]:
    t = list(sorted(set(int(x) for x in topics)))
    rng = random.Random(int(seed))
    rng.shuffle(t)
    folds: List[List[int]] = [[] for _ in range(int(k))]
    for i, topic_id in enumerate(t):
        folds[i % int(k)].append(topic_id)
    return folds


def _topic_queries(queries: Sequence[Query]) -> Dict[int, Query]:
    return {int(q.id): q for q in queries}


def _build_examples_for_topics(
    *,
    topic_ids: Sequence[int],
    queries_by_id: Mapping[int, Query],
    qrels: Mapping[int, Mapping[str, int]],
    searcher,
    dense_cfg: Mapping[str, object],
    candidates_depth: int,
    label_rel_threshold: int,
    neg_per_pos: int,
    seed: int,
) -> List[PairExample]:
    """Build (query, doc_text) examples for the given topics."""
    rng = random.Random(int(seed))
    # Build dense index once
    cfg = ApproachConfig(name="a3_train", params={"dense": dict(dense_cfg)}, candidates_depth=int(candidates_depth))
    index = _load_dense_index(cfg.params)

    model_name = str(dense_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
    device = str(dense_cfg.get("device", "cpu"))
    normalize_embeddings = bool(dense_cfg.get("normalize_embeddings", True))
    ef = None
    hnsw = dense_cfg.get("hnsw") or {}
    if isinstance(hnsw, dict) and hnsw.get("ef") is not None:
        ef = int(hnsw.get("ef"))

    out: List[PairExample] = []
    for topic_id in sorted(int(t) for t in topic_ids):
        q = queries_by_id.get(topic_id)
        if q is None:
            continue
        qrels_topic = qrels.get(topic_id, {}) or {}
        rel_docids = sorted([docid for docid, rel in qrels_topic.items() if int(rel) >= int(label_rel_threshold)])
        if not rel_docids:
            continue

        # Dense candidates
        qvec = _embed_query(q.text, model_name=model_name, device=device, normalize_embeddings=normalize_embeddings)
        candidates = index.search(qvec, topk=int(candidates_depth), ef=ef)
        candidate_docids = [docid for docid, _s in candidates]

        rel_set = set(rel_docids)
        neg_pool = [d for d in candidate_docids if d not in rel_set]
        if not neg_pool:
            continue

        # Positives
        for docid in rel_docids:
            txt = fetch_doc_contents(searcher, docid)
            out.append(PairExample(topic_id=topic_id, query=q.text, docid=docid, doc_text=txt, label=1))
            # Negatives sampled per positive
            for _ in range(int(neg_per_pos)):
                nd = rng.choice(neg_pool)
                ntxt = fetch_doc_contents(searcher, nd)
                out.append(PairExample(topic_id=topic_id, query=q.text, docid=nd, doc_text=ntxt, label=0))

    return out


def _make_hf_dataset(examples: Sequence[PairExample], *, tokenizer, max_length: int):
    _require_training_deps()
    import torch

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
            item["labels"] = torch.tensor(int(ex.label), dtype=torch.long)
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
) -> Dict[str, object]:
    _require_training_deps()
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = _make_hf_dataset(train_examples, tokenizer=tokenizer, max_length=int(max_length))
    dev_ds = _make_hf_dataset(dev_examples, tokenizer=tokenizer, max_length=int(max_length))

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

    # Persist the best model (Trainer already keeps best in output_dir with load_best_model_at_end)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "fold": int(fold_idx),
        "train_size": int(len(train_examples)),
        "dev_size": int(len(dev_examples)),
        "train_runtime": float(getattr(train_out, "training_time", 0.0) or 0.0),
        "eval": {k: float(v) if isinstance(v, (int, float)) else v for k, v in (eval_out or {}).items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune cross-encoder with topic-level CV (A3-7).")
    p.add_argument("--queries", default="queriesROBUST.txt")
    p.add_argument("--qrels", default="qrels_50_Queries")
    p.add_argument("--index", default="robust04")
    p.add_argument("--out-dir", default="models/approach3_ce")

    # CV / sampling
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-rel-threshold", type=int, default=1)
    p.add_argument("--candidates-depth", type=int, default=2000)
    p.add_argument("--neg-per-pos", type=int, default=2)

    # Dense encoder settings (must match your A3-1 embeddings)
    p.add_argument("--bi-encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--bi-encoder-device", default="cpu")

    # Cross-encoder training
    p.add_argument("--ce-base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=1)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    searcher = get_searcher(args.index)

    # Build dense config compatible with approach3 helpers
    base_cfg = default_approach3_config()
    dense_cfg = (base_cfg.params or {}).get("dense") or {}
    if not isinstance(dense_cfg, dict):
        dense_cfg = {}
    dense_cfg = dict(dense_cfg)
    dense_cfg["model_name"] = str(args.bi_encoder_model)
    dense_cfg["device"] = str(args.bi_encoder_device)

    # Topics for CV
    topics = sorted(int(t) for t in qrels.keys())
    folds = _split_topics_kfold(topics, k=int(args.folds), seed=int(args.seed))
    q_by_id = _topic_queries(queries)

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fold_reports: List[Dict[str, object]] = []

    for fold_idx in range(int(args.folds)):
        dev_topics = set(folds[fold_idx])
        train_topics = [t for i, f in enumerate(folds) if i != fold_idx for t in f]

        train_ex = _build_examples_for_topics(
            topic_ids=train_topics,
            queries_by_id=q_by_id,
            qrels=qrels,
            searcher=searcher,
            dense_cfg=dense_cfg,
            candidates_depth=int(args.candidates_depth),
            label_rel_threshold=int(args.label_rel_threshold),
            neg_per_pos=int(args.neg_per_pos),
            seed=int(args.seed) + 1000 + fold_idx,
        )
        dev_ex = _build_examples_for_topics(
            topic_ids=sorted(dev_topics),
            queries_by_id=q_by_id,
            qrels=qrels,
            searcher=searcher,
            dense_cfg=dense_cfg,
            candidates_depth=int(args.candidates_depth),
            label_rel_threshold=int(args.label_rel_threshold),
            neg_per_pos=int(args.neg_per_pos),
            seed=int(args.seed) + 2000 + fold_idx,
        )

        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        report = _train_one_fold(
            fold_idx=fold_idx,
            train_examples=train_ex,
            dev_examples=dev_ex,
            output_dir=fold_dir,
            model_name=str(args.ce_base_model),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            num_epochs=int(args.epochs),
            patience=int(args.patience),
            seed=int(args.seed) + fold_idx,
        )
        fold_reports.append(report)

    summary = {
        "folds": int(args.folds),
        "seed": int(args.seed),
        "params": {
            "label_rel_threshold": int(args.label_rel_threshold),
            "candidates_depth": int(args.candidates_depth),
            "neg_per_pos": int(args.neg_per_pos),
            "ce_base_model": str(args.ce_base_model),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
        },
        "folds_report": fold_reports,
    }
    out_path = os.path.join(out_dir, "cv_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote CV summary: {out_path}")
    print(f"Fold checkpoints under: {out_dir}/fold_*")
    print("To use a fold checkpoint for inference, set rerank.use_finetuned=true and rerank.finetuned_model_dir accordingly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

