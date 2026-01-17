"""MonoT5 reranker (Seq2Seq) for query-document relevance scoring.

This follows the common MonoT5 pattern:
- Build an input prompt from (query, document)
- Run a single decoder step
- Use the log-probability of generating "true" (vs "false") as a relevance score

Heavy deps (torch/transformers) are imported only when the reranker is instantiated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def _require_torch_transformers():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "MonoT5 reranking requires optional deps: torch + transformers (AutoTokenizer/AutoModelForSeq2SeqLM)."
        ) from e


@dataclass(frozen=True)
class Reranked:
    docid: str
    score: float


class MonoT5Reranker:
    """MonoT5 reranker that scores (query, doc_text) pairs via "true/false" likelihood."""

    def __init__(
        self,
        model_name: str = "castorini/monot5-3b-msmarco-10k",
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        prompt_template: str = "Query: {query} Document: {doc} Relevant:",
        torch_dtype: Optional[str] = "auto",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string (e.g., 'cpu' or 'cuda')")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(prompt_template, str) or "{query}" not in prompt_template or "{doc}" not in prompt_template:
            raise ValueError("prompt_template must contain '{query}' and '{doc}' placeholders")

        _require_torch_transformers()
        import os
        from pathlib import Path

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = str(model_name)
        is_local = os.path.isdir(model_id)

        td_raw = None if torch_dtype is None else str(torch_dtype)
        if td_raw is None or td_raw.lower() == "none":
            torch_dtype_arg = None
        elif td_raw.lower() == "auto":
            torch_dtype_arg = "auto"
        else:
            torch_dtype_arg = getattr(torch, td_raw, None)
            if torch_dtype_arg is None:
                raise ValueError(f"Unknown torch_dtype='{td_raw}'. Use 'auto', 'float16', 'bfloat16', 'float32', or None.")

        if is_local:
            p = Path(model_id).resolve()
            self._tokenizer = AutoTokenizer.from_pretrained(p)
            if torch_dtype_arg is None:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(p)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(p, torch_dtype=torch_dtype_arg)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            if torch_dtype_arg is None:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype_arg)

        self._model.eval()
        self._device = torch.device(str(device))
        self._model.to(self._device)

        # Resolve token ids for "true"/"false" (SentencePiece tokens like ▁true / ▁false).
        # We only score the first generated token, which is the common MonoT5 heuristic.
        true_ids = self._tokenizer.encode("true", add_special_tokens=False)
        false_ids = self._tokenizer.encode("false", add_special_tokens=False)
        if not true_ids or not false_ids:
            raise RuntimeError("Tokenizer could not encode 'true'/'false' into token ids")
        self._true_token_id = int(true_ids[0])
        self._false_token_id = int(false_ids[0])

        self.model_name = model_name
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.prompt_template = str(prompt_template)
        self.torch_dtype = td_raw

    def _make_inputs(self, query: str, doc: str) -> str:
        return self.prompt_template.format(query=query, doc=doc)

    def score_pairs(self, query: str, docs: Sequence[str]) -> List[float]:
        """Score a single query against a list of document texts."""
        if not isinstance(query, str) or not query.strip():
            return [0.0 for _ in docs]
        if not docs:
            return []

        import torch

        tok = self._tokenizer
        model = self._model
        dev = self._device

        decoder_start_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
        if decoder_start_id is None:
            # T5 typically uses pad_token_id as decoder start.
            decoder_start_id = getattr(getattr(tok, "pad_token_id", None), "__int__", lambda: None)()
        if decoder_start_id is None:
            decoder_start_id = int(0)

        scores: List[float] = []
        bs = max(1, int(self.batch_size))

        with torch.no_grad():
            for i in range(0, len(docs), bs):
                batch_docs = docs[i : i + bs]
                inputs = [self._make_inputs(query, d if isinstance(d, str) else str(d)) for d in batch_docs]
                enc = tok(
                    inputs,
                    truncation=True,
                    padding=True,
                    max_length=int(self.max_length),
                    return_tensors="pt",
                )
                enc = {k: v.to(dev) for k, v in enc.items()}

                # One-step decode: logits for the *first generated* token.
                decoder_input_ids = torch.full(
                    (int(len(inputs)), 1),
                    int(decoder_start_id),
                    dtype=torch.long,
                    device=dev,
                )
                out = model(**enc, decoder_input_ids=decoder_input_ids)
                logits = out.logits  # [B, 1, V]
                step_logits = logits[:, 0, :]
                log_probs = torch.log_softmax(step_logits, dim=-1)

                # Score = logP(true) - logP(false) (monotonic w.r.t. logP(true)).
                s = log_probs[:, self._true_token_id] - log_probs[:, self._false_token_id]
                scores.extend([float(x) for x in s.detach().cpu().tolist()])
        return scores

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Reranked]:
        """Rerank candidates given as (docid, doc_text) pairs."""
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
    p = argparse.ArgumentParser(description="MonoT5 reranker smoke test.")
    p.add_argument("--query", required=True, help="Query string.")
    p.add_argument("--doc", action="append", default=[], help="Document text (repeatable).")
    p.add_argument("--doc-file", default=None, help="Optional path to a text file with one document per line.")
    p.add_argument("--model-name", default="castorini/monot5-3b-msmarco-10k")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--topn", type=int, default=10)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    docs: List[str] = list(args.doc or [])
    if args.doc_file:
        docs.extend(list(_iter_docs_from_file(str(args.doc_file))))
    if not docs:
        raise SystemExit("Provide at least one --doc or --doc-file")

    r = MonoT5Reranker(
        model_name=str(args.model_name),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    candidates = [(f"d{i+1}", t) for i, t in enumerate(docs)]
    reranked = r.rerank(str(args.query), candidates)[: int(args.topn)]
    for i, x in enumerate(reranked, start=1):
        print(f"{i}\t{x.docid}\t{x.score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

