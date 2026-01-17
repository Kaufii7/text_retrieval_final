from __future__ import annotations

import argparse
import logging
import os
from typing import List

from rag.approaches.approach1 import bm25_retrieve, bm25_rm3_retrieve
from rag.eval import average_precision, load_trec_run, mean_average_precision
from rag.io import Query, load_queries
from rag.logging_utils import configure_logging
from rag.lucene_backend import get_searcher
from rag.query_expansion import expand_query_text_reque_wordnet
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
    p.add_argument(
        "--stream-output",
        action="store_true",
        help="Write output incrementally (only supported for --approach approach3).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume incremental output from an existing partial run (only supported for --approach approach3).",
    )

    # BM25 params
    p.add_argument("--k1", type=float, default=0.9)
    p.add_argument("--b", type=float, default=0.4)

    # RM3 (Approach 1 optional; applies on top of BM25)
    p.add_argument("--rm3", action="store_true", help="Enable RM3 pseudo-relevance feedback for --approach bm25.")
    p.add_argument("--rm3-fb-terms", type=int, default=50, help="RM3 fb_terms (default: 50).")
    p.add_argument("--rm3-fb-docs", type=int, default=50, help="RM3 fb_docs (default: 50).")
    p.add_argument(
        "--rm3-orig-weight",
        type=float,
        default=0.2,
        help="RM3 original_query_weight in [0,1] (default: 0.2).",
    )

    # Query expansion (Approach 1)
    p.add_argument(
        "--qe",
        choices=["none", "reque_wordnet"],
        default="none",
        help="Optional query expansion for --approach bm25 (default: none).",
    )
    p.add_argument("--qe-topn", type=int, default=3, help="Top-N expansions per term for ReQue WordNet (default: 3).")
    p.add_argument(
        "--qe-replace",
        action="store_true",
        help="If set, allow replacement behavior in some expanders (default: append-only).",
    )

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

    if args.approach == "bm25":
        if args.qe != "none":
            if args.qe == "reque_wordnet":
                queries = [
                    Query(
                        id=q.id,
                        content=expand_query_text_reque_wordnet(
                            query_text=q.content,
                            topn=int(args.qe_topn),
                            replace=bool(args.qe_replace),
                        ),
                    )
                    for q in queries
                ]
            else:
                raise ValueError(f"Unknown --qe option: {args.qe}")

        searcher = get_searcher(args.index)
        if bool(args.rm3):
            results_by_topic = bm25_rm3_retrieve(
                queries=queries,
                searcher=searcher,
                topk=args.topk,
                k1=args.k1,
                b=args.b,
                rm3_fb_terms=int(args.rm3_fb_terms),
                rm3_fb_docs=int(args.rm3_fb_docs),
                rm3_original_query_weight=float(args.rm3_orig_weight),
            )
        else:
            results_by_topic = bm25_retrieve(
                queries=queries,
                searcher=searcher,
                topk=args.topk,
                k1=args.k1,
                b=args.b,
            )
    elif args.approach == "clustpsg":
        # Lazy import: keep `main.py` usable for BM25 runs even if Approach 2 deps/code change.
        from rag.clustpsg.pipeline import clustpsg_run

        searcher = get_searcher(args.index)
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
        # Lazy import: Approach 3 has heavy optional deps; don't import unless needed.
        from rag.approaches.approach3 import approach3_retrieve

        cfg = None
        if args.approach3_config:
            from rag.config import load_approach_config_json

            cfg = load_approach_config_json(args.approach3_config)
        else:
            # Default config has reranking disabled; this lets us run stage-1 dense retrieval
            # without initializing Pyserini/Java (avoids macOS libomp/JVM crashes).
            from rag.config import default_approach3_config

            cfg = default_approach3_config()

        # Only initialize Pyserini/Java searcher if reranking is enabled (needs doc text).
        params = cfg.params or {}
        rerank_cfg = params.get("rerank") if isinstance(params, dict) else None
        rerank_enabled = False
        if isinstance(rerank_cfg, dict):
            rerank_enabled = bool(rerank_cfg.get("enabled", False))

        searcher = get_searcher(args.index) if rerank_enabled else None
        if searcher is None:
            log.info("Approach3: rerank disabled -> running stage-1 dense retrieval without Pyserini/Java searcher.")
        # Streaming/resume mode: save results as we go and allow restart without losing progress.
        if bool(args.stream_output) or bool(args.resume):
            from rag.approaches.approach3 import approach3_retrieve_iter
            from rag.runs import append_trec_run_topic, load_trec_run_topic_ids

            partial_out = str(args.output) + ".partial"
            done_topics = set()
            if bool(args.resume) and os.path.exists(partial_out):
                done_topics = set(load_trec_run_topic_ids(partial_out))
                log.info("Resuming Approach 3 run: found %d completed topics in %s", len(done_topics), partial_out)
            else:
                # Fresh streaming run: truncate partial output if it exists.
                os.makedirs(os.path.dirname(partial_out) or ".", exist_ok=True)
                with open(partial_out, "w", encoding="utf-8") as f:
                    f.write("")

            for topic_id, entries in approach3_retrieve_iter(
                queries=queries,
                searcher=searcher,
                topk=int(args.topk),
                config=cfg,
                skip_topic_ids=sorted(done_topics),
            ):
                append_trec_run_topic(
                    output_path=partial_out,
                    topic_id=int(topic_id),
                    entries=entries,
                    run_tag=str(args.run_tag),
                    topk=int(args.topk),
                )
            os.replace(partial_out, str(args.output))
            log.info("Wrote run file: %s", args.output)
            results_by_topic = None
        else:
            results_by_topic = approach3_retrieve(
                queries=queries,
                searcher=searcher,
                topk=args.topk,
                config=cfg,
            )
    else:
        raise ValueError("Unknown approach: {0}".format(args.approach))

    if results_by_topic is not None:
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
        qrels_topics = set(qrels.keys())
        run_topics = set(run.keys())
        overlap = sorted(qrels_topics.intersection(run_topics))

        if not overlap:
            # This commonly happens when evaluating a test split run with a train-only qrels file.
            raise ValueError(
                "No topic overlap between run and qrels; MAP would be 0 by construction.\n"
                f"- run topics:  count={len(run_topics)}, min={min(run_topics) if run_topics else 'n/a'}, max={max(run_topics) if run_topics else 'n/a'}\n"
                f"- qrels topics: count={len(qrels_topics)}, min={min(qrels_topics) if qrels_topics else 'n/a'}, max={max(qrels_topics) if qrels_topics else 'n/a'}\n"
                "Fix: either run with --split train (to match qrels_50_Queries), or pass a qrels file that matches your evaluated topics via --qrels."
            )

        # Compute MAP over the overlapping topics only.
        ap_by_topic = {tid: average_precision(qrels[tid], run.get(tid, []), k=args.eval_k) for tid in overlap}
        map_value = sum(ap_by_topic.values()) / float(len(ap_by_topic)) if ap_by_topic else 0.0

        if len(overlap) < len(qrels_topics):
            log.warning(
                "Evaluating on topic intersection only: overlap=%d, qrels_topics=%d, run_topics=%d",
                len(overlap),
                len(qrels_topics),
                len(run_topics),
            )

        print(f"MAP@{args.eval_k}: {map_value:.6f}")
        if args.per_topic:
            for topic_id in overlap:
                print(f"{topic_id}\t{ap_by_topic[topic_id]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

