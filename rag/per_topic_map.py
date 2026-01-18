"""Per-topic AP / MAP evaluation for TREC run files.

This is a convenience wrapper around `rag.eval` that prints per-topic AP and an
overall MAP@k. It can optionally include some basic diagnostics per topic
(number of judged relevant docs, and how many were retrieved in the top-k).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from rag.eval import load_trec_run, mean_average_precision
from rag.io import load_qrels


@dataclass(frozen=True)
class TopicRow:
    topic_id: int
    ap: float
    rel_total: int
    rel_retrieved: int
    retrieved: int


def _count_rel_retrieved(qrels_for_topic: Mapping[str, int], ranked_docids: Sequence[str], k: int) -> Tuple[int, int, int]:
    rel = {docid for docid, r in qrels_for_topic.items() if int(r) > 0}
    top = list(ranked_docids[: int(k)])
    rel_hits = sum(1 for d in top if d in rel)
    return int(len(rel)), int(rel_hits), int(len(top))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print per-topic AP (and overall MAP) for a TREC run.")
    p.add_argument("--run", required=True, help="Path to TREC 6-column run file.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file.")
    p.add_argument("--k", type=int, default=1000, help="Cutoff depth (default: 1000).")
    p.add_argument(
        "--sort",
        choices=["topic", "ap"],
        default="topic",
        help="Sort output by topic id or AP (default: topic).",
    )
    p.add_argument(
        "--desc",
        action="store_true",
        help="If set, sort descending (only meaningful with --sort ap).",
    )
    p.add_argument(
        "--with-counts",
        action="store_true",
        help="Include basic counts per topic: rel_total, rel_retrieved@k, retrieved@k.",
    )
    p.add_argument("--output", default=None, help="Optional path to write TSV output.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    k = int(args.k)
    qrels = load_qrels(str(args.qrels))
    run = load_trec_run(str(args.run), k=k)
    map_value, ap_by_topic = mean_average_precision(qrels, run, k=k)

    rows: List[TopicRow] = []
    for topic_id in sorted(qrels.keys()):
        ap = float(ap_by_topic.get(int(topic_id), 0.0))
        rel_total, rel_retrieved, retrieved = _count_rel_retrieved(qrels[int(topic_id)], run.get(int(topic_id), []), k=k)
        rows.append(
            TopicRow(
                topic_id=int(topic_id),
                ap=float(ap),
                rel_total=int(rel_total),
                rel_retrieved=int(rel_retrieved),
                retrieved=int(retrieved),
            )
        )

    if str(args.sort) == "ap":
        rows.sort(key=lambda r: (r.ap, -r.topic_id), reverse=bool(args.desc))
    else:
        rows.sort(key=lambda r: r.topic_id)

    header = ["topic_id", f"ap@{k}"]
    if bool(args.with_counts):
        header.extend([f"rel_total", f"rel_retrieved@{k}", f"retrieved@{k}"])

    lines: List[str] = []
    lines.append("# " + "\t".join(header))
    for r in rows:
        parts = [str(int(r.topic_id)), f"{float(r.ap):.6f}"]
        if bool(args.with_counts):
            parts.extend([str(int(r.rel_total)), str(int(r.rel_retrieved)), str(int(r.retrieved))])
        lines.append("\t".join(parts))

    # Explicitly state the denominator (all topics present in qrels).
    n_qrels_topics = int(len(qrels))
    n_run_topics = int(len(run))
    print(f"MAP@{k} (all qrels topics={n_qrels_topics}): {map_value:.6f}")
    print(f"Run topics present: {n_run_topics}")
    for ln in lines:
        print(ln)

    if args.output:
        with open(str(args.output), "w", encoding="utf-8") as f:
            f.write(f"MAP@{k}\t{map_value:.6f}\tqrels_topics={n_qrels_topics}\trun_topics={n_run_topics}\n")
            for ln in lines:
                f.write(ln + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

