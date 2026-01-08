from __future__ import annotations

import argparse
import logging
from typing import List

from rag.approaches.approach1 import bm25_retrieve
from rag.approaches.approach3 import approach3_retrieve
from rag.clustpsg.pipeline import clustpsg_run
from rag.eval import load_trec_run, mean_average_precision
from rag.io import Query, load_queries
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher
from rag.runs import write_trec_run


def _split_queries(queries: List[Query], split: str, train_topics: int = 50) -> List[Query]:
    if split == "all":
        return queries
    if split == "train":
        return queries[:train_topics]
    if split == "test":
        return queries[train_topics:]
    raise ValueError("Unknown split: {0}".format(split))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ROBUST04 retrieval runner (Pyserini).")
    p.add_argument("--approach", choices=["bm25", "clustpsg", "approach3"], default="bm25")
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--output", required=True, help="Output run file path (TREC format).")
    p.add_argument("--run-tag", default="run1", help="Run tag (column 6).")
    p.add_argument("--topk", type=int, default=1000, help="Max docs per query (default: 1000).")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")

    # BM25 params
    p.add_argument("--k1", type=float, default=0.9)
    p.add_argument("--b", type=float, default=0.4)

    # Index
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")

    # Evaluation (optional)
    p.add_argument("--evaluate", action="store_true", help="Evaluate the produced run with qrels (MAP).")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file (train topics).")
    p.add_argument("--eval-k", type=int, default=1000, help="Evaluation cutoff depth (default: 1000).")
    p.add_argument("--per-topic", action="store_true", help="If evaluating, also print per-topic AP.")

    # clustpsg (Approach 2) training control
    p.add_argument("--train-model", action="store_true", help="If --approach clustpsg and split=train, train and save the SVM model.")
    p.add_argument(
        "--precompute-passages",
        action="store_true",
        help="If --approach clustpsg, precompute and cache ranked passages (train/inference stages) and exit (no model needed).",
    )

    # Approach 3 config (optional, for reproducibility)
    p.add_argument(
        "--approach3-config",
        default=None,
        help="Optional path to a JSON file describing ApproachConfig for --approach approach3.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("main")

    queries = load_queries(args.queries)
    queries = _split_queries(queries, args.split, train_topics=50)
    log.info("Loaded %d queries for split=%s", len(queries), args.split)

    searcher = get_searcher(args.index)
    if args.approach == "bm25":
        results_by_topic = bm25_retrieve(
            queries=queries,
            searcher=searcher,
            topk=args.topk,
            k1=args.k1,
            b=args.b,
        )
    elif args.approach == "clustpsg":
        # clustpsg returns (docid, score) tuples; convert to run-writer-compatible entries
        run = clustpsg_run(
            queries=queries,
            searcher=searcher,
            topk=args.topk,
            split=args.split,
            qrels_path=args.qrels,
            train_model=args.train_model,
            precompute_only=bool(args.precompute_passages),
            logger=log,
        )
        if args.precompute_passages:
            log.info("Precomputed ranked passages cache. Exiting as requested by --precompute-passages.")
            return 0
        results_by_topic = {tid: [{"docid": docid, "score": score} for docid, score in pairs] for tid, pairs in run.items()}
    elif args.approach == "approach3":
        cfg = None
        if args.approach3_config:
            from rag.config import load_approach_config_json

            cfg = load_approach_config_json(args.approach3_config)
        results_by_topic = approach3_retrieve(
            queries=queries,
            searcher=searcher,
            topk=args.topk,
            config=cfg,
        )
    else:
        raise ValueError("Unknown approach: {0}".format(args.approach))

    write_trec_run(
        results_by_topic=results_by_topic,
        output_path=args.output,
        run_tag=args.run_tag,
        topk=args.topk,
    )
    log.info("Wrote run file: %s", args.output)

    if args.evaluate:
        from rag.io import load_qrels

        qrels = load_qrels(args.qrels)
        run = load_trec_run(args.output, k=args.eval_k)
        map_value, ap_by_topic = mean_average_precision(qrels, run, k=args.eval_k)
        print(f"MAP@{args.eval_k}: {map_value:.6f}")
        if args.per_topic:
            for topic_id in sorted(ap_by_topic.keys()):
                print(f"{topic_id}\t{ap_by_topic[topic_id]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

