"""PR A3-4: Cross-encoder reranker wrapper (Stage 2; inference only).

This module intentionally **does not** import heavy dependencies at import time.
If you instantiate `CrossEncoderReranker` without the required packages, you get
a clear actionable error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def _require_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Cross-encoder reranking requires the optional dependency 'sentence-transformers'. "
            "Install it to enable reranking."
        ) from e
    return CrossEncoder


@dataclass(frozen=True)
class Reranked:
    docid: str
    score: float


class CrossEncoderReranker:
    """Thin wrapper around a cross-encoder model for (query, doc_text) scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: Optional[int] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string (e.g., 'cpu' or 'cuda')")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        CrossEncoder = _require_sentence_transformers()
        kwargs = {"device": device}
        if max_length is not None:
            kwargs["max_length"] = int(max_length)
        self._model = CrossEncoder(model_name, **kwargs)
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = None if max_length is None else int(max_length)

    def score_pairs(self, query: str, docs: Sequence[str]) -> List[float]:
        """Score a single query against a list of document texts."""
        if not isinstance(query, str) or not query.strip():
            return [0.0 for _ in docs]
        if not docs:
            return []

        pairs = [(query, d if isinstance(d, str) else str(d)) for d in docs]
        # sentence-transformers CrossEncoder exposes .predict(pairs)
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        # It may return a numpy array; normalize to list[float]
        try:
            return [float(x) for x in scores]
        except TypeError:
            return [float(x) for x in list(scores)]

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Reranked]:
        """Rerank candidates given as (docid, doc_text) pairs.

        Output is deterministic: score desc, docid asc tie-break.
        """
        if not candidates:
            return []
        docids = [str(docid) for docid, _ in candidates]
        texts = [text for _, text in candidates]
        scores = self.score_pairs(query, texts)
        out = [Reranked(docid=docids[i], score=float(scores[i])) for i in range(len(docids))]
        out.sort(key=lambda x: (-x.score, x.docid))
        return out


def _iter_docs_from_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.rstrip("\n")
            if not s.strip():
                continue
            yield s


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-encoder reranker smoke test (A3-4).")
    p.add_argument("--query", required=True, help="Query string.")
    p.add_argument("--doc", action="append", default=[], help="Document text (repeatable).")
    p.add_argument("--doc-file", default=None, help="Optional path to a text file with one document per line.")
    p.add_argument("--model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--topn", type=int, default=10)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    docs: List[str] = list(args.doc or [])
    if args.doc_file:
        docs.extend(list(_iter_docs_from_file(str(args.doc_file))))
    if not docs:
        raise SystemExit("Provide at least one --doc or --doc-file")

    r = CrossEncoderReranker(
        model_name=str(args.model_name),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_length=args.max_length if args.max_length is None else int(args.max_length),
    )
    candidates = [(f"d{i+1}", t) for i, t in enumerate(docs)]
    reranked = r.rerank(str(args.query), candidates)[: int(args.topn)]
    for i, x in enumerate(reranked, start=1):
        print(f"{i}\t{x.docid}\t{x.score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

