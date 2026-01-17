"""Optuna tuning for Approach 1 (BM25) on ROBUST04 train topics.

This tunes ONLY BM25 hyperparameters (k1, b) and explicitly does NOT use:
- Query expansion
- RM3 / pseudo-relevance feedback

It evaluates MAP@K on the train split (default: first 50 topics with qrels).

Example:
  .venv/bin/python -m rag.experiments.bm25_optuna \
    --trials 50 \
    --out-json results/bm25_optuna_best.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rag.eval import average_precision
from rag.io import load_qrels, load_queries
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher, search, set_bm25
from rag.types import Query


@dataclass(frozen=True)
class BestResult:
    k1: float
    b: float
    map_at_k: float
    k: int


def _split_train(queries: List[Query], train_topics: int = 50) -> List[Query]:
    return queries[:train_topics]


def _filter_to_qrels_topics(queries: Sequence[Query], qrels: Mapping[int, Mapping[str, int]]) -> List[Query]:
    qrels_topics = set(qrels.keys())
    return [q for q in queries if q.topic_id in qrels_topics]


def _compute_map_for_params(
    *,
    searcher,
    queries: Sequence[Query],
    qrels: Mapping[int, Mapping[str, int]],
    topk: int,
    k1: float,
    b: float,
    trial=None,
) -> float:
    """Compute MAP@topk for BM25(k1,b), optionally reporting intermediate values for pruning."""
    set_bm25(searcher, k1=float(k1), b=float(b))

    ap_sum = 0.0
    n = 0
    for i, q in enumerate(queries):
        hits = search(searcher, q.text, topk=topk)
        ranked_docids = [h.docid for h in hits]
        ap = average_precision(qrels.get(q.topic_id, {}), ranked_docids, k=topk)
        ap_sum += ap
        n += 1

        if trial is not None:
            # Report running MAP for pruning (step = topic index).
            trial.report(ap_sum / float(n), step=i)
            if trial.should_prune():
                # Import locally so the module remains importable even if optuna is missing.
                import optuna  # type: ignore

                raise optuna.TrialPruned()

    return ap_sum / float(n) if n else 0.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna tuning for Approach1 BM25 (no QE, no RM3).")
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file (train topics).")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")
    p.add_argument("--topk", type=int, default=1000, help="Retrieval depth / MAP cutoff (default: 1000).")
    p.add_argument("--train-topics", type=int, default=50, help="How many initial topics to use as train (default: 50).")

    # Optuna controls
    p.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50).")
    p.add_argument("--timeout", type=int, default=None, help="Stop study after N seconds (optional).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL (e.g. sqlite:///results/bm25_optuna.db) to resume studies.",
    )
    p.add_argument("--study-name", default="bm25_optuna", help="Optuna study name (default: bm25_optuna).")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (default: 1). Use with --storage.")

    # Output
    p.add_argument("--out-json", required=True, help="Where to write best params JSON.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("rag.experiments.bm25_optuna")

    try:
        import optuna  # type: ignore
    except Exception as e:
        raise ImportError(
            "Optuna is required for this experiment.\n"
            "Install it into your environment, e.g.:\n"
            "  .venv/bin/python -m pip install optuna\n"
        ) from e

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")

    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    train_queries = _split_train(queries, train_topics=int(args.train_topics))
    train_queries = _filter_to_qrels_topics(train_queries, qrels)
    if not train_queries:
        raise ValueError("No train queries overlap qrels topics; cannot tune MAP.")

    log.info(
        "Tuning BM25 on %d train queries (train_topics=%d, MAP@%d).",
        len(train_queries),
        int(args.train_topics),
        int(args.topk),
    )

    # Keep searcher initialization outside objective; we default to n_jobs=1.
    # If user runs parallel trials, each job should have its own process and thus its own searcher.
    searcher = get_searcher(args.index)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, len(train_queries) // 5))

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=str(args.study_name),
        storage=str(args.storage) if args.storage else None,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: "optuna.Trial") -> float:
        # Standard BM25 ranges. Keep k1 > 0, b in [0,1].
        k1 = trial.suggest_float("k1", 0.1, 3.0)
        b = trial.suggest_float("b", 0.0, 1.0)
        return _compute_map_for_params(
            searcher=searcher,
            queries=train_queries,
            qrels=qrels,
            topk=int(args.topk),
            k1=float(k1),
            b=float(b),
            trial=trial,
        )

    log.info("Starting Optuna study: trials=%d timeout=%s n_jobs=%d", int(args.trials), str(args.timeout), int(args.n_jobs))
    study.optimize(objective, n_trials=int(args.trials), timeout=args.timeout, n_jobs=int(args.n_jobs))

    best = study.best_trial
    best_result = BestResult(
        k1=float(best.params.get("k1")),
        b=float(best.params.get("b")),
        map_at_k=float(best.value),
        k=int(args.topk),
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    payload: Dict[str, object] = {
        "approach": "bm25",
        "notes": "Optuna-tuned BM25 params for Approach1. No query expansion. No RM3.",
        "index": str(args.index),
        "topk": int(args.topk),
        "metric": {"name": f"MAP@{int(args.topk)}", "value": best_result.map_at_k},
        "params": {"k1": best_result.k1, "b": best_result.b},
        "how_to_run": f".venv/bin/python main.py --approach bm25 --split train --topk {int(args.topk)} --k1 {best_result.k1:.6f} --b {best_result.b:.6f} --qe none",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Best MAP@{int(args.topk)}: {best_result.map_at_k:.6f} (k1={best_result.k1:.6f}, b={best_result.b:.6f})")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

