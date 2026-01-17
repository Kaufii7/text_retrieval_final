#!/usr/bin/env python3
"""Build a candidate-only corpus JSONL for the 50 judged (train) topics.

This script collects the union of BM25 top-k documents for the topics present in
`qrels_50_Queries` (50 topics), then writes a JSONL file:

  {"docid": "...", "text": "..."}

The resulting JSONL can be passed to:
  rag/approach3/evaluate_all_topics.py --corpus-jsonl <path>

to enable fast doc-text lookup via the built-in SQLite offset index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, Set

# Ensure repo root is on sys.path when running as a script (python scripts/..py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.io import load_qrels, load_queries
from rag.lucene_backend import fetch_doc_contents, get_searcher, search, set_bm25


def _iter_existing_docids(jsonl_path: str) -> Iterable[str]:
    """Best-effort iterator of docids already present in an existing JSONL."""
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            docid = obj.get("docid")
            if isinstance(docid, str) and docid:
                yield docid


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build corpus.jsonl from union of BM25 top-k docs on the 50 qrels topics."
    )
    p.add_argument("--queries", default="queriesROBUST.txt", help="Path to queries file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file (train topics).")
    p.add_argument("--index", default="robust04", help="Pyserini prebuilt index name.")
    p.add_argument("--output", default="corpus_top5000_train.jsonl", help="Output JSONL path.")
    p.add_argument("--topk", type=int, default=5000, help="BM25 depth per topic (default: 5000).")
    p.add_argument("--k1", type=float, default=0.9, help="BM25 k1 (default: 0.9).")
    p.add_argument("--b", type=float, default=0.4, help="BM25 b (default: 0.4).")
    p.add_argument(
        "--skip-empty-text",
        action="store_true",
        help="Skip writing docs whose fetched text is empty/whitespace.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, read existing docids and append only missing ones.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N newly-written docs (default: 1000).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if int(args.topk) <= 0:
        raise SystemExit("--topk must be > 0")

    qrels = load_qrels(str(args.qrels))
    train_topic_ids = sorted(int(t) for t in qrels.keys())
    train_topic_set = set(train_topic_ids)

    queries = load_queries(str(args.queries))
    train_queries = [q for q in queries if int(q.topic_id) in train_topic_set]
    train_queries.sort(key=lambda q: int(q.topic_id))

    if not train_queries:
        print("No train queries found after filtering by qrels topics.", file=sys.stderr)
        return 2

    out_path = str(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    seen: Set[str] = set()
    mode = "w"
    if bool(args.resume) and os.path.exists(out_path):
        mode = "a"
        for d in _iter_existing_docids(out_path):
            seen.add(d)
        print(f"Resume enabled: found {len(seen)} existing docids in {out_path}")

    searcher = get_searcher(str(args.index))
    set_bm25(searcher, k1=float(args.k1), b=float(args.b))

    t0 = time.perf_counter()
    wrote = 0
    skipped_empty = 0

    print(
        "Building candidate corpus:"
        f" topics={len(train_queries)} topk={int(args.topk)}"
        f" bm25(k1={float(args.k1)}, b={float(args.b)})"
        f" output={out_path} resume={bool(args.resume)}"
    )

    with open(out_path, mode, encoding="utf-8") as out:
        for qi, q in enumerate(train_queries, start=1):
            hits = search(searcher, q.text, topk=int(args.topk))
            for h in hits:
                docid = str(h.docid)
                if docid in seen:
                    continue
                seen.add(docid)

                text = fetch_doc_contents(searcher, docid)
                if bool(args.skip_empty_text) and not str(text).strip():
                    skipped_empty += 1
                    continue

                out.write(json.dumps({"docid": docid, "text": text}, ensure_ascii=False) + "\n")
                wrote += 1
                if int(args.log_every) > 0 and (wrote % int(args.log_every) == 0):
                    elapsed = time.perf_counter() - t0
                    rate = wrote / elapsed if elapsed > 1e-9 else 0.0
                    print(
                        f"Wrote {wrote} docs (unique_seen={len(seen)})"
                        f" queries={qi}/{len(train_queries)}"
                        f" elapsed={elapsed:.1f}s rate={rate:.1f} docs/s"
                    )

    elapsed = time.perf_counter() - t0
    print(
        f"Done. Wrote {wrote} docs to {out_path}."
        f" unique_seen={len(seen)} skipped_empty={skipped_empty} elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

