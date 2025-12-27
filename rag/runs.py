"""Run file writing utilities (TREC 6-column format).

Format per line:
  topic_id Q0 docid rank score run_tag

Determinism requirements:
- Topics are written in ascending topic_id order.
- Per-topic results are sorted by score desc, then docid asc (tie-break).
- Rank starts at 1.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union


ResultLike = Union[Tuple[str, float], Mapping[str, object]]


def _normalize_entry(entry: ResultLike) -> Tuple[str, float]:
    """Normalize a result entry to (docid, score)."""
    if isinstance(entry, tuple) and len(entry) == 2:
        docid, score = entry
        docid = str(docid)
        score_f = float(score)
        return docid, score_f

    if isinstance(entry, Mapping):
        if "docid" not in entry or "score" not in entry:
            raise ValueError("Result dict must contain keys: 'docid' and 'score'")
        docid = str(entry["docid"])
        score_f = float(entry["score"])
        return docid, score_f

    raise TypeError("Result entry must be a (docid, score) tuple or a dict-like with keys 'docid'/'score'")


def write_trec_run(
    results_by_topic: Mapping[int, Sequence[ResultLike]],
    output_path: str,
    run_tag: str,
    topk: int = 1000,
) -> None:
    """Write a TREC-format run file.

    Args:
        results_by_topic: mapping of topic_id -> list of results. Each result is either:
          - (docid, score) tuple, or
          - dict-like with keys: {'docid': ..., 'score': ...}
        output_path: path to write the run file to.
        run_tag: string placed in column 6.
        topk: max number of documents per topic (default 1000).
    """
    if not isinstance(run_tag, str) or not run_tag.strip():
        raise ValueError("run_tag must be a non-empty string")
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("topk must be a positive integer")

    lines: List[str] = []

    for topic_id in sorted(results_by_topic.keys()):
        entries = results_by_topic.get(topic_id, [])
        normalized: List[Tuple[str, float]] = []
        for entry in entries:
            docid, score = _normalize_entry(entry)
            if not docid or any(ch.isspace() for ch in docid):
                raise ValueError(f"Invalid docid for topic_id={topic_id}: {docid!r}")
            normalized.append((docid, score))

        normalized.sort(key=lambda x: (-x[1], x[0]))
        normalized = normalized[:topk]

        for rank, (docid, score) in enumerate(normalized, start=1):
            # 6-column TREC run format
            # Use a stable float string (trec_eval accepts this fine).
            lines.append(f"{topic_id} Q0 {docid} {rank} {score:.6f} {run_tag}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


