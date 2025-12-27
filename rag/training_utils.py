"""Shared training utilities across approaches."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from rag.types import Document


def augment_candidates_with_qrels(
    *,
    candidates_by_topic: Mapping[int, Sequence[Document]],
    qrels: Mapping[int, Mapping[str, int]],
) -> Dict[int, List[Document]]:
    """Augment retrieved candidates with *all* judged docids from qrels (train-time).

    This ensures training can use the full set of judged documents even if the
    retrieval run is capped at top-k (e.g., 1000).

    Rules:
    - Preserve the original candidate order.
    - Append missing qrels docids deterministically (sorted by docid).
    - For appended docs, score is set to None (unknown / not retrieved).
    - Content is left empty; downstream may fetch it if needed.
    """
    out: Dict[int, List[Document]] = {}
    for topic_id in sorted(set(candidates_by_topic.keys()) | set(qrels.keys())):
        existing = list(candidates_by_topic.get(topic_id, []))
        seen = {d.id for d in existing}
        to_add = [docid for docid in qrels.get(topic_id, {}).keys() if docid not in seen]
        to_add.sort()
        augmented = existing + [Document(id=docid, content="", score=None) for docid in to_add]
        out[topic_id] = augmented
    return out


