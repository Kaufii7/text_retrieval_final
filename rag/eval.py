"""Evaluation utilities (MAP) for TREC-style runs.

This module is intentionally lightweight and dependency-free.
It evaluates MAP on the training topics using `qrels_50_Queries`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rag.io import load_qrels


@dataclass(frozen=True)
class RunEntry:
    topic_id: int
    docid: str
    rank: int
    score: float


def load_trec_run(path: str, k: int = 1000) -> Dict[int, List[str]]:
    """Load a TREC 6-column run file.

    Returns:
      dict[topic_id] -> ranked list of docids (length <= k)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    by_topic: Dict[int, List[Tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"{path}:{line_no}: expected >= 6 columns 'topic_id Q0 docid rank score run_tag', got: {line!r}"
                )
            topic_str, _q0, docid, rank_str, score_str = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                topic_id = int(topic_str)
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: invalid topic_id {topic_str!r}") from e
            try:
                rank = int(rank_str)
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: invalid rank {rank_str!r}") from e
            try:
                _score = float(score_str)
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: invalid score {score_str!r}") from e

            by_topic.setdefault(topic_id, []).append((rank, docid))

    out: Dict[int, List[str]] = {}
    for topic_id, pairs in by_topic.items():
        pairs.sort(key=lambda x: x[0])  # sort by rank asc
        seen = set()
        docids: List[str] = []
        for _rank, docid in pairs:
            if docid in seen:
                continue
            seen.add(docid)
            docids.append(docid)
            if len(docids) >= k:
                break
        out[topic_id] = docids

    return out


def average_precision(qrels_for_topic: Mapping[str, int], ranked_docids: Sequence[str], k: int = 1000) -> float:
    """Compute Average Precision for a single topic.

    Relevance is treated as binary: rel > 0 is relevant.
    AP divides by the total number of relevant documents in qrels (if 0, returns 0.0).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    relevant = {docid for docid, rel in qrels_for_topic.items() if rel > 0}
    if not relevant:
        return 0.0

    num_rel_seen = 0
    sum_precisions = 0.0

    for i, docid in enumerate(ranked_docids[:k], start=1):
        if docid in relevant:
            num_rel_seen += 1
            sum_precisions += num_rel_seen / float(i)

    return sum_precisions / float(len(relevant))


def mean_average_precision(
    qrels: Mapping[int, Mapping[str, int]],
    run: Mapping[int, Sequence[str]],
    k: int = 1000,
) -> Tuple[float, Dict[int, float]]:
    """Compute MAP over the topics present in qrels.

    Returns:
      (map_value, ap_by_topic)
    """
    ap_by_topic: Dict[int, float] = {}
    topics = sorted(qrels.keys())
    if not topics:
        return 0.0, ap_by_topic

    for topic_id in topics:
        ap_by_topic[topic_id] = average_precision(qrels[topic_id], run.get(topic_id, []), k=k)

    map_value = sum(ap_by_topic.values()) / float(len(ap_by_topic)) if ap_by_topic else 0.0
    return map_value, ap_by_topic


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a TREC run file with qrels (MAP).")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file.")
    p.add_argument("--run", required=True, help="Path to TREC run file.")
    p.add_argument("--k", type=int, default=1000, help="Cutoff depth (default: 1000).")
    p.add_argument("--per-topic", action="store_true", help="Print per-topic AP.")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    qrels = load_qrels(args.qrels)
    run = load_trec_run(args.run, k=args.k)
    map_value, ap_by_topic = mean_average_precision(qrels, run, k=args.k)

    print(f"MAP@{args.k}: {map_value:.6f}")
    if args.per_topic:
        for topic_id in sorted(ap_by_topic.keys()):
            print(f"{topic_id}\t{ap_by_topic[topic_id]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


