from __future__ import annotations

import argparse
import logging
from typing import List

from rag.approaches.bm25 import bm25_retrieve
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
    p.add_argument("--approach", choices=["bm25"], default="bm25")
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
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
    else:
        raise ValueError("Unknown approach: {0}".format(args.approach))

    write_trec_run(
        results_by_topic=results_by_topic,
        output_path=args.output,
        run_tag=args.run_tag,
        topk=args.topk,
    )
    log.info("Wrote run file: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

