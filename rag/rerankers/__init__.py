"""Reranker modules (optional heavy dependencies, gated on use).

These modules are designed to keep the repo import-safe when ML dependencies
are not installed. Only import/use rerankers when you explicitly enable them.
"""

from __future__ import annotations

__all__ = [
    "CrossEncoderReranker",
    "MonoT5Reranker",
    "Reranked",
]

# Keep imports lazy-ish: these modules gate heavy deps internally.
from .cross_encoder import CrossEncoderReranker, Reranked  # noqa: E402
from .monot5 import MonoT5Reranker  # noqa: E402

