"""I/O utilities for queries and qrels.

Expected inputs in this repo:
- queriesROBUST.txt: `topic_id<TAB>query_text` (one query per line)
- qrels_50_Queries: `topic_id 0 doc_id relevance` (whitespace separated)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from rag.types import Query


def _iter_nonempty_lines(path: str) -> Iterable[Tuple[int, str]]:
    """Yield (line_no, stripped_line) skipping empty/whitespace-only lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            yield i, line


def load_queries(path: str) -> List[Query]:
    """Load queries from `queriesROBUST.txt`.

    Format per line: `topic_id<TAB>query_text`
    Returns queries sorted by topic_id (deterministic).
    """
    queries: List[Query] = []
    seen_topics = set()

    for line_no, line in _iter_nonempty_lines(path):
        if "\t" not in line:
            raise ValueError(
                f"{path}:{line_no}: expected TAB-separated 'topic_id<TAB>query_text', got: {line!r}"
            )
        topic_str, query_text = line.split("\t", 1)
        topic_str = topic_str.strip()
        query_text = query_text.strip()
        if not topic_str:
            raise ValueError(f"{path}:{line_no}: empty topic_id")
        if not query_text:
            raise ValueError(f"{path}:{line_no}: empty query_text for topic_id={topic_str!r}")
        try:
            topic_id = int(topic_str)
        except ValueError as e:
            raise ValueError(f"{path}:{line_no}: invalid topic_id {topic_str!r}") from e

        if topic_id in seen_topics:
            raise ValueError(f"{path}:{line_no}: duplicate topic_id={topic_id}")
        seen_topics.add(topic_id)
        queries.append(Query(id=topic_id, content=query_text))

    queries.sort(key=lambda q: q.topic_id)
    return queries


def load_qrels(path: str) -> Dict[int, Dict[str, int]]:
    """Load qrels from `qrels_50_Queries`.

    Expected format (whitespace separated):
      topic_id 0 doc_id relevance

    Returns:
      dict[topic_id][doc_id] = relevance (int)

    Notes:
    - If duplicates appear for the same (topic_id, doc_id), keeps the max relevance.
    """
    qrels: Dict[int, Dict[str, int]] = {}

    for line_no, line in _iter_nonempty_lines(path):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"{path}:{line_no}: expected 4 columns 'topic_id 0 doc_id relevance', got {len(parts)}: {line!r}"
            )
        topic_str, _zero, doc_id, rel_str = parts
        try:
            topic_id = int(topic_str)
        except ValueError as e:
            raise ValueError(f"{path}:{line_no}: invalid topic_id {topic_str!r}") from e
        if not doc_id:
            raise ValueError(f"{path}:{line_no}: empty doc_id")
        try:
            rel = int(rel_str)
        except ValueError as e:
            raise ValueError(f"{path}:{line_no}: invalid relevance {rel_str!r}") from e

        topic_map = qrels.setdefault(topic_id, {})
        if doc_id in topic_map:
            topic_map[doc_id] = max(topic_map[doc_id], rel)
        else:
            topic_map[doc_id] = rel

    return qrels


