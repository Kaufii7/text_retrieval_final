"""PR A3-6: Tuning driver for Approach 3 (train MAP loop + structured logging).

This script runs Approach 3 on the first 50 topics (train split), evaluates MAP@k,
and writes a CSV/JSON summary for reproducible tuning.

Example:
  python3 -m rag.experiments.approach3_grid --out-csv results/a3_grid.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from rag.approaches.approach3 import approach3_retrieve
from rag.config import ApproachConfig, default_approach3_config
from rag.eval import mean_average_precision
from rag.io import load_qrels, load_queries
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher
from rag.types import Query


@dataclass(frozen=True)
class GridResult:
    candidates_depth: int
    backend: str
    metric: str
    ef: int | None
    rerank_enabled: bool
    rerank_topn: int
    alpha: float
    map_at_k: float


def _split_train(queries: List[Query], train_topics: int = 50) -> List[Query]:
    return queries[:train_topics]


def _parse_ints(values: Sequence[str]) -> List[int]:
    return [int(v) for v in values]


def _parse_floats(values: Sequence[str]) -> List[float]:
    return [float(v) for v in values]


def _results_to_ranked_docids(results_by_topic: Mapping[int, Sequence[Mapping[str, object]]]) -> Dict[int, List[str]]:
    ranked: Dict[int, List[str]] = {}
    for topic_id, entries in results_by_topic.items():
        ranked[topic_id] = [str(e["docid"]) for e in entries if "docid" in e]
    return ranked


def _with_overrides(
    base: ApproachConfig,
    *,
    candidates_depth: int,
    backend: str,
    metric: str,
    ef: int | None,
    rerank_enabled: bool,
    rerank_topn: int,
    alpha: float,
) -> ApproachConfig:
    params = copy.deepcopy(base.params or {})
    dense = params.setdefault("dense", {})
    if not isinstance(dense, dict):
        dense = {}
        params["dense"] = dense
    dense["backend"] = str(backend)
    dense["metric"] = str(metric)
    hnsw = dense.setdefault("hnsw", {})
    if not isinstance(hnsw, dict):
        hnsw = {}
        dense["hnsw"] = hnsw
    hnsw["ef"] = None if ef is None else int(ef)

    rerank = params.setdefault("rerank", {})
    if not isinstance(rerank, dict):
        rerank = {}
        params["rerank"] = rerank
    rerank["enabled"] = bool(rerank_enabled)
    rerank["topn"] = int(rerank_topn)

    fusion = params.setdefault("score_fusion", {})
    if not isinstance(fusion, dict):
        fusion = {}
        params["score_fusion"] = fusion
    fusion["alpha"] = float(alpha)

    return ApproachConfig(name=base.name, params=params, candidates_depth=int(candidates_depth))


def run_grid(
    *,
    queries: List[Query],
    qrels_path: str,
    index_name: str,
    topk: int,
    candidates_depth_values: Sequence[int],
    backends: Sequence[str],
    metrics: Sequence[str],
    ef_values: Sequence[int | None],
    rerank_enabled_values: Sequence[bool],
    rerank_topn_values: Sequence[int],
    alpha_values: Sequence[float],
    log: logging.Logger,
) -> List[GridResult]:
    qrels = load_qrels(qrels_path)
    searcher = get_searcher(index_name)
    base_cfg = default_approach3_config()

    # total combinations
    total = (
        len(candidates_depth_values)
        * len(backends)
        * len(metrics)
        * len(ef_values)
        * len(rerank_enabled_values)
        * len(rerank_topn_values)
        * len(alpha_values)
    )
    done = 0

    results: List[GridResult] = []
    for candidates_depth in candidates_depth_values:
        for backend in backends:
            for metric in metrics:
                for ef in ef_values:
                    for rerank_enabled in rerank_enabled_values:
                        for rerank_topn in rerank_topn_values:
                            for alpha in alpha_values:
                                done += 1
                                log.info(
                                    "Grid %d/%d: depth=%d backend=%s metric=%s ef=%s rerank=%s topn=%d alpha=%.3f",
                                    done,
                                    total,
                                    int(candidates_depth),
                                    str(backend),
                                    str(metric),
                                    str(ef),
                                    str(bool(rerank_enabled)),
                                    int(rerank_topn),
                                    float(alpha),
                                )
                                cfg = _with_overrides(
                                    base_cfg,
                                    candidates_depth=int(candidates_depth),
                                    backend=str(backend),
                                    metric=str(metric),
                                    ef=ef,
                                    rerank_enabled=bool(rerank_enabled),
                                    rerank_topn=int(rerank_topn),
                                    alpha=float(alpha),
                                )
                                retrieved = approach3_retrieve(
                                    queries=queries,
                                    searcher=searcher,
                                    topk=int(topk),
                                    config=cfg,
                                )
                                ranked = _results_to_ranked_docids(retrieved)
                                map_value, _ap_by_topic = mean_average_precision(qrels, ranked, k=int(topk))
                                results.append(
                                    GridResult(
                                        candidates_depth=int(candidates_depth),
                                        backend=str(backend),
                                        metric=str(metric),
                                        ef=None if ef is None else int(ef),
                                        rerank_enabled=bool(rerank_enabled),
                                        rerank_topn=int(rerank_topn),
                                        alpha=float(alpha),
                                        map_at_k=float(map_value),
                                    )
                                )

    results.sort(key=lambda r: (-r.map_at_k, r.candidates_depth, r.backend, r.metric, (r.ef or -1), r.rerank_topn, r.alpha))
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Approach 3 grid search on train (first 50 topics).")
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file.")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")
    p.add_argument("--topk", type=int, default=1000, help="Retrieval depth / MAP cutoff (default: 1000).")
    p.add_argument("--log-level", default="INFO", help="Logging level.")

    p.add_argument("--candidates-depth", nargs="+", default=["2000"], help="Stage-1 candidates depth values.")
    p.add_argument("--backend", nargs="+", default=["hnswlib"], help="Backends: hnswlib or exact.")
    p.add_argument("--metric", nargs="+", default=["cosine"], help="Metrics: cosine, ip, l2.")
    p.add_argument(
        "--ef",
        nargs="+",
        default=["200"],
        help="HNSW ef values; use 'none' to skip setting ef (default: 200).",
    )
    p.add_argument("--rerank-enabled", nargs="+", default=["false"], help="Rerank enabled flags: true/false.")
    p.add_argument("--rerank-topn", nargs="+", default=["100"], help="Rerank top-N values.")
    p.add_argument("--alpha", nargs="+", default=["1.0"], help="Score fusion alpha values.")

    p.add_argument("--out-csv", required=True, help="Where to write CSV results.")
    p.add_argument("--out-json", default=None, help="Optional path to write JSON summary.")
    return p


def _parse_bools(values: Sequence[str]) -> List[bool]:
    out: List[bool] = []
    for v in values:
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            out.append(True)
        elif s in ("0", "false", "f", "no", "n", "off"):
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {v!r}")
    return out


def _parse_optional_ints(values: Sequence[str]) -> List[int | None]:
    out: List[int | None] = []
    for v in values:
        s = str(v).strip().lower()
        if s in ("none", "null", ""):
            out.append(None)
        else:
            out.append(int(s))
    return out


def main() -> int:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("rag.experiments.approach3_grid")

    queries = load_queries(args.queries)
    train_queries = _split_train(queries, train_topics=50)
    log.info("Using %d train queries (first 50 topics).", len(train_queries))

    candidates_depth_values = _parse_ints(args.candidates_depth)
    backends = [str(x) for x in args.backend]
    metrics = [str(x) for x in args.metric]
    ef_values = _parse_optional_ints(args.ef)
    rerank_enabled_values = _parse_bools(args.rerank_enabled)
    rerank_topn_values = _parse_ints(args.rerank_topn)
    alpha_values = _parse_floats(args.alpha)

    results = run_grid(
        queries=train_queries,
        qrels_path=args.qrels,
        index_name=args.index,
        topk=int(args.topk),
        candidates_depth_values=candidates_depth_values,
        backends=backends,
        metrics=metrics,
        ef_values=ef_values,
        rerank_enabled_values=rerank_enabled_values,
        rerank_topn_values=rerank_topn_values,
        alpha_values=alpha_values,
        log=log,
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "candidates_depth",
                "backend",
                "metric",
                "ef",
                "rerank_enabled",
                "rerank_topn",
                "alpha",
                f"map@{args.topk}",
            ]
        )
        for r in results:
            w.writerow(
                [
                    str(r.candidates_depth),
                    r.backend,
                    r.metric,
                    "" if r.ef is None else str(r.ef),
                    "true" if r.rerank_enabled else "false",
                    str(r.rerank_topn),
                    f"{r.alpha:.6f}",
                    f"{r.map_at_k:.6f}",
                ]
            )

    best = results[0] if results else None
    if best is None:
        print("No results produced.")
        return 1

    print(
        f"Best MAP@{args.topk}: {best.map_at_k:.6f} "
        f"(depth={best.candidates_depth}, backend={best.backend}, metric={best.metric}, ef={best.ef}, "
        f"rerank={best.rerank_enabled}, topn={best.rerank_topn}, alpha={best.alpha:.3f})"
    )
    print(f"Wrote CSV: {args.out_csv}")

    if args.out_json:
        summary = {
            "topk": int(args.topk),
            "best": asdict(best),
            "results": [asdict(r) for r in results],
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

