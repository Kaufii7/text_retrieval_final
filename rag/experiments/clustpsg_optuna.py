"""Optuna tuning for Approach 2 (ClustPsg) on ROBUST04 train topics.

This optimizes a *restricted* set of hyperparameters (as requested):
- passage_retrieval.k1
- passage_retrieval.b
- passage_retrieval.lucene.rm3.fb_terms
- passage_retrieval.lucene.rm3.fb_docs
- passage_retrieval.lucene.rm3.original_query_weight
- clustering.threshold
- passage_rerank.beta
- rrf.k
- cluster_labeling.rr_k
- cluster_labeling.threshold_low
- cluster_labeling.threshold_high
- final.lambda_min
- final.lambda_max
- final.rr_k

All other knobs are kept fixed at `default_approach2_config()` defaults.

Explicitly NOT tuned here (budget/runtime knobs):
- doc_candidates_depth, doc_content_topk, clustering_max_passages, etc.

Example:
  .venv/bin/python -m rag.experiments.clustpsg_optuna \
    --trials 30 \
    --out-json results/clustpsg_optuna_best.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING

from rag.config import ApproachConfig, default_approach2_config
from rag.eval import average_precision
from rag.io import load_qrels, load_queries
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher
from rag.types import Query

if TYPE_CHECKING:  # pragma: no cover
    import optuna  # type: ignore


@dataclass(frozen=True)
class BestResult:
    map_at_k: float
    k: int
    config: ApproachConfig
    flat_params: Dict[str, object]


def _split_train(queries: List[Query], train_topics: int = 50) -> List[Query]:
    return queries[:train_topics]


def _filter_to_qrels_topics(queries: Sequence[Query], qrels: Mapping[int, Mapping[str, int]]) -> List[Query]:
    qrels_topics = set(qrels.keys())
    return [q for q in queries if q.topic_id in qrels_topics]


def _set_nested(d: Dict[str, object], path: Sequence[str], value: object) -> None:
    cur: Dict[str, object] = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def _atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    """Write JSON to path atomically (best-effort).

    This avoids partially-written JSON files if the process is interrupted.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _objective_config_from_trial(
    trial: "optuna.Trial",
    *,
    base: ApproachConfig,
    work_root: str,
) -> Tuple[ApproachConfig, bool]:
    """Build an ApproachConfig from an Optuna trial.

    Returns:
      (cfg, train_model_flag) where train_model_flag should be passed to clustpsg_run(train_model=...).
    """
    cfg = ApproachConfig(name=base.name, params=copy.deepcopy(base.params), candidates_depth=base.candidates_depth)
    p = cfg.params

    # -----------------------
    # ONLY the requested knobs
    # -----------------------
    _set_nested(p, ["passage_retrieval", "k1"], float(trial.suggest_float("passage_retrieval.k1", 0.1, 3.0)))
    _set_nested(p, ["passage_retrieval", "b"], float(trial.suggest_float("passage_retrieval.b", 0.0, 1.0)))
    _set_nested(p, ["passage_retrieval", "lucene", "rm3", "fb_terms"], int(trial.suggest_int("passage_retrieval.lucene.rm3.fb_terms", 10, 100)))
    _set_nested(p, ["passage_retrieval", "lucene", "rm3", "fb_docs"], int(trial.suggest_int("passage_retrieval.lucene.rm3.fb_docs", 10, 100)))
    _set_nested(p, ["passage_retrieval", "lucene", "rm3", "original_query_weight"], float(trial.suggest_float("passage_retrieval.lucene.rm3.original_query_weight", 0.0, 1.0)))
    _set_nested(p, ["clustering", "threshold"], float(trial.suggest_float("clustering.threshold", 0.0, 1.0)))
    _set_nested(p, ["passage_rerank", "beta"], float(trial.suggest_float("passage_rerank.beta", 0.0, 1.0)))
    _set_nested(p, ["rrf", "k"], int(trial.suggest_int("rrf.k", 0, 200)))
    # _set_nested(p, ["cluster_labeling", "rr_k"], int(trial.suggest_int("cluster_labeling.rr_k", 0, 200)))
    # thr_low = float(trial.suggest_float("cluster_labeling.threshold_low", 0.0, 0.5))
    # thr_high = float(trial.suggest_float("cluster_labeling.threshold_high", thr_low, 1.0))
    # _set_nested(p, ["cluster_labeling", "threshold_low"], thr_low)
    # _set_nested(p, ["cluster_labeling", "threshold_high"], thr_high)
    lam_min = float(trial.suggest_float("final.lambda_min", 0.0, 1.0))
    lam_max = float(trial.suggest_float("final.lambda_max", lam_min, 1.0))
    _set_nested(p, ["final", "lambda_min"], lam_min)
    _set_nested(p, ["final", "lambda_max"], lam_max)
    _set_nested(p, ["final", "rr_k"], int(trial.suggest_int("final.rr_k", 20, 200)))

    # Everything else remains as in default_approach2_config().
    # train_model = bool((p.get("final") or {}).get("use_svm_cluster_scores", True))
    train_model = False
    return cfg, train_model


def _build_payload(
    *,
    study: "optuna.Study",
    best_cfg: ApproachConfig,
    index: str,
    topk: int,
    last_trial: "optuna.trial.FrozenTrial | None",
) -> Dict[str, object]:
    best = study.best_trial
    payload: Dict[str, object] = {
        "approach": "clustpsg",
        "index": str(index),
        "topk": int(topk),
        "metric": {"name": f"MAP@{int(topk)}", "value": float(best.value)},
        "best_trial_number": int(best.number),
        "best_trial_params_flat": {k: v for k, v in best.params.items()},
        "best_config": {
            "name": best_cfg.name,
            "params": best_cfg.params,
            "candidates_depth": best_cfg.candidates_depth,
        },
        "updated_at_unix": int(time.time()),
    }
    if last_trial is not None:
        payload["last_trial_number"] = int(last_trial.number)
        payload["last_trial_value"] = float(last_trial.value) if last_trial.value is not None else None
        payload["last_trial_state"] = str(last_trial.state)
        payload["last_trial_params_flat"] = {k: v for k, v in last_trial.params.items()}
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna tuning for Approach2 (clustpsg) on train topics (MAP@K).")
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file (train topics).")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")
    p.add_argument("--topk", type=int, default=1000, help="Run depth / MAP cutoff (default: 1000).")
    p.add_argument("--train-topics", type=int, default=50, help="How many initial topics to use as train (default: 50).")

    # Optuna controls
    p.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30).")
    p.add_argument("--timeout", type=int, default=None, help="Stop study after N seconds (optional).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--storage",
        default=None,
        help=(
            "Optional Optuna storage URL. If omitted, a SQLite DB is created under --work-root so the study can resume. "
            "Example: sqlite:///results/clustpsg_optuna.db"
        ),
    )
    p.add_argument("--study-name", default="clustpsg_optuna", help="Optuna study name (default: clustpsg_optuna).")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (default: 1). Use with --storage.")

    # Work dir for per-trial artifacts (models, Lucene passage index cache)
    p.add_argument("--work-root", default="cache/optuna_clustpsg", help="Where to write trial artifacts/caches.")

    # Output
    p.add_argument("--out-json", required=True, help="Where to write best params JSON.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("rag.experiments.clustpsg_optuna")

    try:
        import optuna  # type: ignore
    except Exception as e:
        raise ImportError(
            "Optuna is required for this experiment.\n"
            "Install it into your environment, e.g.:\n"
            "  python -m pip install optuna\n"
        ) from e

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")

    os.makedirs(args.work_root, exist_ok=True)

    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    train_queries = _split_train(queries, train_topics=int(args.train_topics))
    train_queries = _filter_to_qrels_topics(train_queries, qrels)
    if not train_queries:
        raise ValueError("No train queries overlap qrels topics; cannot tune MAP.")

    log.info("Tuning clustpsg on %d train queries (MAP@%d).", len(train_queries), int(args.topk))

    # Keep searcher initialization outside objective (recommended: n_jobs=1).
    searcher = get_searcher(args.index)
    base = default_approach2_config()

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))

    # Persistent storage enables resuming after interruptions.
    # If user doesn't provide --storage, default to a SQLite DB under --work-root.
    storage_url = str(args.storage) if args.storage else None
    if not storage_url:
        db_path = os.path.abspath(os.path.join(str(args.work_root), "clustpsg_optuna.db"))
        storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=str(args.study_name),
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        # Lazy import to avoid importing Approach2 code unless the experiment runs.
        from rag.clustpsg.pipeline import clustpsg_run

        cfg, train_model = _objective_config_from_trial(trial, base=base, work_root=str(args.work_root))

        run = clustpsg_run(
            queries=train_queries,
            searcher=searcher,
            topk=int(args.topk),
            config=cfg,
            split="train",
            qrels_path=str(args.qrels),
            train_model=bool(train_model),
            precompute_only=False,
            logger=log,
        )

        ap_sum = 0.0
        n = 0
        for q in train_queries:
            pairs = run.get(q.topic_id, [])
            ranked_docids = [docid for docid, _score in pairs]
            ap = average_precision(qrels.get(q.topic_id, {}), ranked_docids, k=int(args.topk))
            ap_sum += ap
            n += 1
        return ap_sum / float(n) if n else 0.0

    def dump_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        # Write "best so far" after every completed trial.
        try:
            best_cfg, _ = _objective_config_from_trial(study.best_trial, base=base, work_root=str(args.work_root))
            payload = _build_payload(
                study=study,
                best_cfg=best_cfg,
                index=str(args.index),
                topk=int(args.topk),
                last_trial=trial,
            )
            _atomic_write_json(str(args.out_json), payload)
        except Exception as e:
            # Don't fail the whole study because snapshotting failed.
            log.warning("Failed to write per-trial snapshot to %s: %s", str(args.out_json), str(e))

    log.info("Starting Optuna study: trials=%d timeout=%s n_jobs=%d", int(args.trials), str(args.timeout), int(args.n_jobs))
    study.optimize(
        objective,
        n_trials=int(args.trials),
        timeout=args.timeout,
        n_jobs=int(args.n_jobs),
        callbacks=[dump_callback],
    )

    best = study.best_trial
    best_cfg, _ = _objective_config_from_trial(best, base=base, work_root=str(args.work_root))
    best_result = BestResult(
        map_at_k=float(best.value),
        k=int(args.topk),
        config=best_cfg,
        flat_params={k: v for k, v in best.params.items()},
    )

    payload = _build_payload(
        study=study,
        best_cfg=best_cfg,
        index=str(args.index),
        topk=int(args.topk),
        last_trial=best,
    )
    payload["notes"] = (
        "This Optuna search excludes budget knobs by design. "
        "main.py currently does not accept an approach2 config file; reproduce via clustpsg_run(config=...)."
    )
    _atomic_write_json(str(args.out_json), payload)

    print(f"Best MAP@{int(args.topk)}: {best_result.map_at_k:.6f}")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

