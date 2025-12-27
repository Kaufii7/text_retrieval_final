"""PR2: Passage extraction for clustpsg (sentence-window sliding).

Key requirement (per project spec):
- Do NOT truncate sentences. We define a *soft* cap and allow exceeding it to
  keep full sentences. Passages always end on sentence boundaries.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Sequence

from rag.config import ApproachConfig
from rag.types import Document, Passage


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split a document into sentences using a simple punctuation regex."""
    if not text:
        return []
    # Keep it simple/deterministic; downstream can replace with NLP segmenters if desired.
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s and s.strip()]


def extract_passages_from_document(
    text: str,
    *,
    stride_sentences: int,
    min_sentences: int,
    max_sentences: int,
    max_chars_per_sentence_soft: int,
) -> List[str]:
    """Split a document into overlapping passages using sentence windows.

    Notes:
    - No sentence truncation: if a sentence exceeds max_chars_per_sentence_soft,
      we keep it intact (assumption: very long sentences are rare).
    - Passages always end on sentence boundaries.
    """
    if not text:
        return []
    if min_sentences <= 0 or max_sentences <= 0:
        raise ValueError("min_sentences and max_sentences must be > 0")
    if min_sentences > max_sentences:
        raise ValueError("min_sentences must be <= max_sentences")
    if stride_sentences <= 0:
        raise ValueError("stride_sentences must be > 0")
    if max_chars_per_sentence_soft <= 0:
        raise ValueError("max_chars_per_sentence_soft must be > 0")

    sentences = split_sentences(text)
    if not sentences:
        return []

    # Soft cap: do not truncate; keep sentence whole.
    # We still filter out pathological whitespace-only content above.
    _ = max_chars_per_sentence_soft

    stride = min(stride_sentences, max_sentences)
    passages: List[str] = []
    current: List[str] = []

    for sentence in sentences:
        if current and len(current) >= max_sentences:
            if len(current) >= min_sentences:
                passages.append(" ".join(current))

            overlap = current[-stride:] if len(current) >= stride else current.copy()
            current = overlap + [sentence] if len(overlap) < max_sentences else [sentence]
        else:
            current.append(sentence)

    if current and len(current) >= min_sentences:
        passages.append(" ".join(current))

    return passages


def passages_for_documents(
    docs: Sequence[Document],
    *,
    cfg: ApproachConfig,
) -> List[Passage]:
    """Extract `Passage` objects for the given documents.

    Expects `doc.content` to be populated. (Fetching document content from the
    index/searcher can be added as a separate helper if needed.)
    """
    params = cfg.params or {}
    min_sentences = int(params.get("min_sentences", 3))
    max_sentences = int(params.get("max_sentences", 5))
    stride_sentences = int(params.get("stride_sentences", 2))
    max_chars_per_sentence_soft = int(params.get("max_chars_per_sentence_soft", 300))

    out: List[Passage] = []
    for doc in docs:
        if not doc.content:
            continue
        ptexts = extract_passages_from_document(
            doc.content,
            stride_sentences=stride_sentences,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_chars_per_sentence_soft=max_chars_per_sentence_soft,
        )
        for i, p in enumerate(ptexts):
            out.append(Passage(document_id=doc.id, index=i, content=p, score=None))
    return out


def passages_by_topic(
    docs_by_topic: Mapping[int, Sequence[Document]],
    *,
    cfg: ApproachConfig,
) -> Dict[int, List[Passage]]:
    """Extract passages for each topic_id -> list[Document]."""
    return {topic_id: passages_for_documents(docs, cfg=cfg) for topic_id, docs in docs_by_topic.items()}


