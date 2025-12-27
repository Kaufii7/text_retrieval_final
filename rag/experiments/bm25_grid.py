"""BM25 parameter sweep on the first 50 topics (train split).

Writes a CSV of (k1, b, map@k) and prints the best setting.

Example:
  .venv/bin/python -m rag.experiments.bm25_grid --out-csv results/bm25_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from rag.approaches.approach1 import bm25_retrieve
from rag.eval import mean_average_precision
from rag.io import load_qrels, load_queries
from rag.types import Query
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher


@dataclass(frozen=True)
class GridResult:
    k1: float
    b: float
    map_at_k: float


def _parse_floats(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        out.append(float(v))
    return out


def _split_train(queries: List[Query], train_topics: int = 50) -> List[Query]:
    return queries[:train_topics]


def _results_to_ranked_docids(results_by_topic) -> dict:
    """Convert bm25_retrieve() output to {topic_id: [docid1, docid2, ...]}."""
    ranked = {}
    for topic_id, entries in results_by_topic.items():
        ranked[topic_id] = [e["docid"] for e in entries]
    return ranked


def run_grid(
    *,
    queries: List[Query],
    qrels_path: str,
    index_name: str,
    topk: int,
    k1_values: Sequence[float],
    b_values: Sequence[float],
    log: logging.Logger,
) -> List[GridResult]:
    qrels = load_qrels(qrels_path)
    searcher = get_searcher(index_name)

    results: List[GridResult] = []
    total = len(k1_values) * len(b_values)
    done = 0

    for k1 in k1_values:
        for b in b_values:
            done += 1
            log.info("Grid %d/%d: k1=%.3f b=%.3f", done, total, k1, b)
            retrieved = bm25_retrieve(queries=queries, searcher=searcher, topk=topk, k1=k1, b=b)
            ranked = _results_to_ranked_docids(retrieved)
            map_value, _ap_by_topic = mean_average_precision(qrels, ranked, k=topk)
            results.append(GridResult(k1=k1, b=b, map_at_k=map_value))

    results.sort(key=lambda r: (-r.map_at_k, r.k1, r.b))
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25 grid search on train (first 50 topics).")
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file.")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")
    p.add_argument("--topk", type=int, default=1000, help="Retrieval depth / MAP cutoff (default: 1000).")
    p.add_argument("--log-level", default="INFO", help="Logging level.")

    p.add_argument(
        "--k1",
        nargs="+",
        default=["0.6", "0.9", "1.2", "1.5", "1.8"],
        help="Space-separated list of k1 values (default: 0.6 0.9 1.2 1.5 1.8).",
    )
    p.add_argument(
        "--b",
        nargs="+",
        default=["0.1", "0.3", "0.5", "0.7", "0.9"],
        help="Space-separated list of b values (default: 0.1 0.3 0.5 0.7 0.9).",
    )

    p.add_argument("--out-csv", required=True, help="Where to write CSV results.")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("rag.experiments.bm25_grid")

    queries = load_queries(args.queries)
    train_queries = _split_train(queries, train_topics=50)
    log.info("Using %d train queries (first 50 topics).", len(train_queries))

    k1_values = _parse_floats(args.k1)
    b_values = _parse_floats(args.b)

    results = run_grid(
        queries=train_queries,
        qrels_path=args.qrels,
        index_name=args.index,
        topk=args.topk,
        k1_values=k1_values,
        b_values=b_values,
        log=log,
    )

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k1", "b", f"map@{args.topk}"])
        for r in results:
            w.writerow([f"{r.k1:.6f}", f"{r.b:.6f}", f"{r.map_at_k:.6f}"])

    best = results[0] if results else None
    if best is None:
        print("No results produced.")
        return 1

    print(f"Best MAP@{args.topk}: {best.map_at_k:.6f} (k1={best.k1:.3f}, b={best.b:.3f})")
    print(f"Wrote CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


