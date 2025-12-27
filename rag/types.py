"""Core reusable dataclasses for retrieval pipelines.

These types are intentionally lightweight and backend-agnostic (no Pyserini deps).
They provide a shared interchange format across:
- retrieval
- passage extraction
- clustering / reranking
- evaluation utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Query:
    """A query/topic."""

    id: int
    content: str

    # Backward-compatible aliases for older code that used topic_id/text naming.
    @property
    def topic_id(self) -> int:  # pragma: no cover
        return self.id

    @property
    def text(self) -> str:  # pragma: no cover
        return self.content


@dataclass(frozen=True)
class Document:
    """A document (optionally with a score from a retriever/reranker)."""

    id: str
    content: str
    score: Optional[float] = None

    # Common alias
    @property
    def docid(self) -> str:  # pragma: no cover
        return self.id


@dataclass(frozen=True)
class Passage:
    """A passage extracted from a source document."""

    document_id: str
    index: int
    content: str
    score: Optional[float] = None


