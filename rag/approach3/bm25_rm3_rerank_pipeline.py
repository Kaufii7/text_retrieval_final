"""BM25+RM3 -> (CE or MonoT5) rerank pipeline with k-fold ensemble support.

This script is meant to be the "production" pipeline:
- Stage 1: BM25 + RM3 retrieval (Pyserini)
  - Parameters loaded from an Optuna JSON artifact (e.g., results/bm25_rm3_optuna_best_5k.json)
- Stage 2: Rerank top-N with a neural reranker:
  - CE (sequence classification) or MonoT5 (seq2seq true/false)
  - Supports ensembling by repeating --ce-model / --monot5-model
- Output: TREC 6-column run file, with stream+resume support.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from rag.io import load_queries
from rag.lucene_backend import get_searcher, set_bm25, set_rm3
from rag.runs import append_trec_run_topic, load_trec_run_topic_ids, write_trec_run
from rag.types import Query

# Reuse the scoring logic from the existing pipeline (includes MonoT5+CE ensemble support).
from rag.approach3.bm25_ce_pipeline import _CorpusJsonlLookup, bm25_to_ce_topk  # noqa: E402


def _configure_logging(*, level: str) -> logging.Logger:
    log = logging.getLogger("rag.approach3.bm25_rm3_rerank_pipeline")
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


def _load_bm25_rm3_params(path: str) -> Tuple[float, float, int, int, float, int]:
    """Return (k1, b, fb_terms, fb_docs, orig_weight, topk_default)."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{path}: expected params dict")
    k1 = float(params.get("k1"))
    b = float(params.get("b"))
    rm3 = params.get("rm3") or {}
    if not isinstance(rm3, dict):
        raise ValueError(f"{path}: expected params.rm3 dict")
    fb_terms = int(rm3.get("fb_terms"))
    fb_docs = int(rm3.get("fb_docs"))
    orig_w = float(rm3.get("original_query_weight"))
    topk_default = int(obj.get("topk", 5000))
    return k1, b, fb_terms, fb_docs, orig_w, topk_default


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25+RM3 -> rerank -> write TREC run (with fold ensemble).")
    p.add_argument("--index", default="robust04")
    p.add_argument("--queries", default="queriesROBUST.txt")
    p.add_argument("--output-run", required=True, help="TREC run output path.")
    p.add_argument("--run-tag", default="bm25_rm3_rerank")
    p.add_argument("--log-level", default="INFO")

    p.add_argument(
        "--bm25-rm3-params-json",
        default="results/bm25_rm3_optuna_best_5k.json",
        help="Path to Optuna-tuned BM25+RM3 params JSON.",
    )
    p.add_argument("--bm25-topk", type=int, default=None, help="Override stage-1 retrieval depth (default from JSON topk).")

    # Output control
    p.add_argument("--final-topk", type=int, default=1000)
    p.add_argument("--rerank-depth", type=int, default=1000, help="How many of the top candidates to rerank (default: 1000).")
    p.add_argument("--alpha", type=float, default=0.2, help="Blend: final=(1-alpha)*lex + alpha*rerank (default: 0.2).")

    # Doc text source (recommended to avoid extra Java calls)
    p.add_argument("--corpus-jsonl", default="cache/approach3_dense/corpus_robust04.jsonl")
    p.add_argument("--corpus-index-db", default=None)

    # Reranker
    p.add_argument("--reranker-type", choices=["ce", "monot5"], default="ce")
    p.add_argument("--ce-model", action="append", default=[], help="Repeatable CE model paths/ids (ensemble).")
    p.add_argument("--monot5-model", action="append", default=[], help="Repeatable MonoT5 model paths/ids (ensemble).")
    p.add_argument("--monot5-torch-dtype", default="auto")
    p.add_argument("--ce-device", default="cpu")
    p.add_argument("--ce-batch-size", type=int, default=16)
    p.add_argument("--ce-max-length", type=int, default=256)
    p.add_argument("--ce-log-every-batches", type=int, default=10)

    # Streaming/resume
    p.add_argument("--stream-output", action="store_true", help="Append each topic as it finishes.")
    p.add_argument("--resume", action="store_true", help="Skip topics already present in output run file (implies stream).")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    log = _configure_logging(level=str(args.log_level))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    k1, b, fb_terms, fb_docs, orig_w, topk_default = _load_bm25_rm3_params(str(args.bm25_rm3_params_json))
    bm25_topk = int(args.bm25_topk) if args.bm25_topk is not None else int(topk_default)
    log.info(
        "BM25+RM3 params: k1=%.6f b=%.6f rm3_fb_terms=%d rm3_fb_docs=%d rm3_orig_weight=%.6f topk=%d",
        float(k1),
        float(b),
        int(fb_terms),
        int(fb_docs),
        float(orig_w),
        int(bm25_topk),
    )

    rr_type = str(args.reranker_type).lower().strip()
    if rr_type == "ce" and not args.ce_model:
        raise SystemExit("--ce-model is required when --reranker-type=ce")
    if rr_type == "monot5" and not args.monot5_model:
        raise SystemExit("--monot5-model is required when --reranker-type=monot5")

    # Searcher (BM25+RM3)
    searcher = get_searcher(str(args.index))
    set_bm25(searcher, k1=float(k1), b=float(b))
    set_rm3(
        searcher,
        fb_terms=int(fb_terms),
        fb_docs=int(fb_docs),
        original_query_weight=float(orig_w),
    )

    # Load queries
    qs: List[Query] = load_queries(str(args.queries))
    qs = sorted(qs, key=lambda q: int(q.topic_id))

    # Doc text lookup
    corpus_lookup: Optional[_CorpusJsonlLookup] = None
    if args.corpus_jsonl:
        db = str(args.corpus_index_db or os.path.join(os.path.dirname(str(args.corpus_jsonl)), "corpus_offsets.sqlite3"))
        corpus_lookup = _CorpusJsonlLookup(corpus_jsonl=str(args.corpus_jsonl), sqlite_path=db, logger=log)

    out_path = str(args.output_run)
    stream = bool(args.stream_output) or bool(args.resume)

    try:
        if stream:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            done = set()
            if bool(args.resume) and os.path.exists(out_path):
                done = set(load_trec_run_topic_ids(out_path))
                log.info("Resuming: found %d completed topics in %s", len(done), out_path)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("")

            for q in qs:
                topic_id = int(q.topic_id)
                if topic_id in done:
                    continue
                log.info("topic=%d start: bm25_topk=%d rerank_depth=%d final_topk=%d", topic_id, bm25_topk, int(args.rerank_depth), int(args.final_topk))
                top, _ = bm25_to_ce_topk(
                    query=q.text,
                    searcher=searcher,
                    bm25_topk=int(bm25_topk),
                    final_topk=int(args.final_topk),
                    rerank_depth=int(args.rerank_depth),
                    alpha=float(args.alpha),
                    docid_to_row=None,
                    embeddings=None,
                    reranker_type=str(args.reranker_type),
                    ce_model=args.ce_model,
                    monot5_model=args.monot5_model if args.monot5_model else "",
                    monot5_torch_dtype=str(args.monot5_torch_dtype),
                    ce_device=str(args.ce_device),
                    ce_batch_size=int(args.ce_batch_size),
                    ce_max_length=int(args.ce_max_length),
                    corpus_lookup=corpus_lookup,
                    log=log,
                    ce_log_every_batches=int(args.ce_log_every_batches),
                    ce_log_prefix=f"{'MonoT5' if rr_type == 'monot5' else 'CE'} topic={topic_id}",
                )
                append_trec_run_topic(
                    output_path=out_path,
                    topic_id=topic_id,
                    entries=[(c.docid, float(c.final_score)) for c in top],
                    run_tag=str(args.run_tag),
                    topk=int(args.final_topk),
                )
                log.info("topic=%d done: returned=%d (appended)", topic_id, len(top))
            log.info("Wrote run (streaming): %s", out_path)
            return 0

        results_by_topic: Dict[int, List[Tuple[str, float]]] = {}
        for q in qs:
            topic_id = int(q.topic_id)
            log.info("topic=%d start: bm25_topk=%d rerank_depth=%d final_topk=%d", topic_id, bm25_topk, int(args.rerank_depth), int(args.final_topk))
            top, _ = bm25_to_ce_topk(
                query=q.text,
                searcher=searcher,
                bm25_topk=int(bm25_topk),
                final_topk=int(args.final_topk),
                rerank_depth=int(args.rerank_depth),
                alpha=float(args.alpha),
                docid_to_row=None,
                embeddings=None,
                reranker_type=str(args.reranker_type),
                ce_model=args.ce_model,
                monot5_model=args.monot5_model if args.monot5_model else "",
                monot5_torch_dtype=str(args.monot5_torch_dtype),
                ce_device=str(args.ce_device),
                ce_batch_size=int(args.ce_batch_size),
                ce_max_length=int(args.ce_max_length),
                corpus_lookup=corpus_lookup,
                log=log,
                ce_log_every_batches=int(args.ce_log_every_batches),
                ce_log_prefix=f"{'MonoT5' if rr_type == 'monot5' else 'CE'} topic={topic_id}",
            )
            results_by_topic[topic_id] = [(c.docid, float(c.final_score)) for c in top]
            log.info("topic=%d done: returned=%d", topic_id, len(top))

        write_trec_run(
            results_by_topic=results_by_topic,
            output_path=out_path,
            run_tag=str(args.run_tag),
            topk=int(args.final_topk),
        )
        log.info("Wrote run: %s", out_path)
        return 0
    finally:
        if corpus_lookup is not None:
            corpus_lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())

