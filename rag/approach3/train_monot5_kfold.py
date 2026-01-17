"""Train (fine-tune) MonoT5 with topic-level K-fold CV on ROBUST04 qrels.

Goal
----
Fine-tune a MonoT5 model (T5 seq2seq) for query-document relevance scoring using
your *doc-level* qrels (`qrels_50_Queries`) on the first 50 topics.

This script is designed for CPU training and matches this repo's inference logic:
we score relevance using a *single decoder step* and the logit difference between
the first generated token being "true" vs "false".

Key points
----------
- Topic-level K-fold CV (avoid leakage across topics).
- Negatives are BM25 hard negatives (cheap; relies on Pyserini doc fetching).
- Optional LoRA via PEFT; adapters are merged and saved as a plain HF checkpoint,
  so you can pass the saved directory directly to:
    `rag/approach3/evaluate_all_topics.py --reranker-type monot5 --monot5-model <dir>`

Dependencies (optional)
-----------------------
- torch
/- transformers
/- peft
/- pyserini (already used by this repo)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

# Optional progress bars (tqdm). If unavailable, fall back to plain iterators.
try:  # pragma: no cover
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

# Allow running as a script from any working directory:
#   python rag/approach3/train_monot5_kfold.py ...
# by ensuring the repo root is on sys.path for `import rag.*`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.io import load_qrels, load_queries
from rag.lucene_backend import fetch_doc_contents, get_searcher, search, set_bm25
from rag.types import Query


def _require_training_deps() -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "MonoT5 fine-tuning requires optional deps: torch + transformers + peft. "
            "Install them to run this script."
        ) from e


@dataclass(frozen=True)
class PairExample:
    topic_id: int
    query: str
    docid: str
    label: int  # 0/1 (0=false, 1=true)


class _DocTextCache:
    """Best-effort cache for docid->text to reduce repeated Lucene fetches."""

    def __init__(self, fetch: Callable[[str], str], *, max_items: int = 50_000):
        self._fetch = fetch
        self._max_items = int(max_items)
        self._cache: MutableMapping[str, str] = {}

    def get(self, docid: str) -> str:
        k = str(docid)
        if k in self._cache:
            return self._cache[k]
        txt = self._fetch(k)
        if len(self._cache) >= self._max_items:
            try:
                self._cache.pop(next(iter(self._cache)))
            except Exception:
                self._cache.clear()
        self._cache[k] = txt
        return txt


def _maybe_tqdm(it, *, enabled: bool, **kwargs):
    if not enabled or _tqdm is None:
        return it
    return _tqdm(it, **kwargs)


def _configure_logging(*, out_dir: str, level: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("rag.approach3.train_monot5_kfold")
    log.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def _split_topics_kfold(topics: Sequence[int], *, k: int, seed: int) -> List[List[int]]:
    t = list(sorted(set(int(x) for x in topics)))
    rng = random.Random(int(seed))
    rng.shuffle(t)
    folds: List[List[int]] = [[] for _ in range(int(k))]
    for i, topic_id in enumerate(t):
        folds[i % int(k)].append(int(topic_id))
    # Keep each fold deterministic in ascending order
    for f in folds:
        f.sort()
    return folds


def _topic_queries(queries: Sequence[Query]) -> Dict[int, Query]:
    return {int(q.id): q for q in queries}


def _build_examples_for_topics(
    *,
    topic_ids: Sequence[int],
    queries_by_id: Mapping[int, Query],
    qrels: Mapping[int, Mapping[str, int]],
    searcher,
    bm25_topk: int,
    label_rel_threshold: int,
    max_pos_per_topic: int,
    neg_per_pos: int,
    hard_neg_ratio: float,
    seed: int,
    logger: logging.Logger,
) -> List[PairExample]:
    """Build (query, docid, label) examples using BM25 negatives.

    Negatives:
    - "hard" negatives come from the top of BM25 hits that are not relevant by qrels
    - remaining negatives are sampled randomly from the non-relevant BM25 pool
    """
    rng = random.Random(int(seed))
    out: List[PairExample] = []

    n_neg = max(0, int(neg_per_pos))
    h = float(hard_neg_ratio)
    h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
    n_hard = max(0, min(n_neg, int(round(h * n_neg))))
    n_rand = n_neg - n_hard

    topic_ids = list(sorted(int(t) for t in topic_ids))
    t0 = time.perf_counter()
    for idx, topic_id in enumerate(topic_ids, start=1):
        q = queries_by_id.get(int(topic_id))
        if q is None:
            continue
        qrels_topic = qrels.get(int(topic_id), {}) or {}
        rel_docids = sorted([d for d, rel in qrels_topic.items() if int(rel) >= int(label_rel_threshold)])
        if not rel_docids:
            continue
        # Cap positives per topic for CPU-feasible training (large qrels topics can dominate runtime).
        mpt = int(max_pos_per_topic)
        if mpt > 0 and len(rel_docids) > mpt:
            rr = random.Random(int(seed) + 17_000 + int(topic_id))
            rel_docids = sorted(rr.sample(rel_docids, int(mpt)))

        hits = search(searcher, q.text, topk=int(bm25_topk))
        cand_docids = [h.docid for h in hits]
        rel_set = set(rel_docids)
        neg_pool = [d for d in cand_docids if d not in rel_set]
        if not neg_pool:
            continue

        hard_ptr = 0
        for pos_docid in rel_docids:
            out.append(PairExample(topic_id=int(topic_id), query=q.text, docid=str(pos_docid), label=1))

            used: set[str] = set()
            # Hard negatives (round-robin from top of BM25 non-relevant hits)
            for _ in range(n_hard):
                if hard_ptr >= len(neg_pool):
                    hard_ptr = 0
                nd = str(neg_pool[hard_ptr])
                hard_ptr += 1
                if nd in used:
                    continue
                used.add(nd)
                out.append(PairExample(topic_id=int(topic_id), query=q.text, docid=nd, label=0))

            # Random negatives from the remaining pool
            if n_rand > 0:
                remaining = [d for d in neg_pool if str(d) not in used]
                if remaining:
                    k = min(int(n_rand), len(remaining))
                    for nd in rng.sample(list(remaining), k):
                        out.append(PairExample(topic_id=int(topic_id), query=q.text, docid=str(nd), label=0))

        if idx % 5 == 0:
            logger.info(
                "Built examples: topics=%d/%d examples=%d (neg_per_pos=%d hard=%d rand=%d bm25_topk=%d)",
                int(idx),
                int(len(topic_ids)),
                int(len(out)),
                int(n_neg),
                int(n_hard),
                int(n_rand),
                int(bm25_topk),
            )

    logger.info(
        "Built examples done: topics=%d examples=%d elapsed=%.2fs",
        int(len(topic_ids)),
        int(len(out)),
        float(time.perf_counter() - t0),
    )
    return out


def _make_prompt(query: str, doc_text: str) -> str:
    # Must match inference prompt used in this repo.
    return f"Query: {query} Document: {doc_text} Relevant:"


def _iter_trainable_params(model) -> Iterable:
    for p in model.parameters():
        if getattr(p, "requires_grad", False):
            yield p


def _ceil_div(a: int, b: int) -> int:
    b = max(1, int(b))
    return (int(a) + b - 1) // b


def _epoch_permutation(n: int, *, seed: int, fold_idx: int, epoch: int) -> List[int]:
    """Deterministic permutation for resuming mid-epoch."""
    idxs = list(range(int(n)))
    rng = random.Random(int(seed) + 100_000 * int(fold_idx) + 1_000 * int(epoch))
    rng.shuffle(idxs)
    return idxs


def _train_one_fold(
    *,
    fold_idx: int,
    train_examples: Sequence[PairExample],
    dev_examples: Sequence[PairExample],
    output_dir: str,
    base_model: str,
    device: str,
    max_length: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    num_epochs: int,
    patience: int,
    seed: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
    doc_cache: _DocTextCache,
    logger: logging.Logger,
    show_progress: bool,
    resume: bool,
    save_every_steps: int,
    save_every_seconds: float,
) -> Dict[str, object]:
    _require_training_deps()
    import math

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(int(seed))

    tok = AutoTokenizer.from_pretrained(str(base_model))

    # ---- Resume / checkpoint paths ----
    resume_state_path = os.path.join(output_dir, "resume_state.pt")
    resume_adapter_dir = os.path.join(output_dir, "resume_adapter")  # LoRA only
    resume_model_state_path = os.path.join(output_dir, "resume_model.pt")  # non-LoRA only

    # ---- Build model (optionally from resume checkpoint) ----
    model = None
    if bool(resume) and os.path.exists(resume_state_path):
        logger.info("Fold %d: resuming from %s", int(fold_idx), resume_state_path)
        # Load state early so we know whether LoRA was used.
        _require_training_deps()
        import torch

        state = torch.load(resume_state_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid resume_state (expected dict): {resume_state_path}")
        resumed_use_lora = bool(state.get("use_lora", False))
        if resumed_use_lora:
            base = AutoModelForSeq2SeqLM.from_pretrained(str(base_model))
            # Load adapter into a trainable PEFT model.
            if not os.path.isdir(resume_adapter_dir):
                raise RuntimeError(f"Missing resume_adapter_dir: {resume_adapter_dir}")
            model = PeftModel.from_pretrained(base, resume_adapter_dir, is_trainable=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(base_model))
            if os.path.exists(resume_model_state_path):
                sd = torch.load(resume_model_state_path, map_location="cpu")
                model.load_state_dict(sd)
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(base_model))

    # Compute token ids for "true" / "false" once.
    true_ids = tok.encode("true", add_special_tokens=False)
    false_ids = tok.encode("false", add_special_tokens=False)
    if not true_ids or not false_ids:
        raise RuntimeError("Tokenizer could not encode 'true'/'false' into token ids")
    true_token_id = int(true_ids[0])
    false_token_id = int(false_ids[0])

    decoder_start_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = getattr(tok, "pad_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = 0

    if bool(use_lora) and not (bool(resume) and os.path.exists(resume_state_path)):
        targets = [str(x).strip() for x in lora_target_modules if str(x).strip()]
        if not targets:
            targets = ["q", "v"]
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(targets),
        )
        model = get_peft_model(model, lora_cfg)
        logger.info(
            "Fold %d: enabled LoRA (r=%d alpha=%d dropout=%.3f targets=%s)",
            int(fold_idx),
            int(lora_r),
            int(lora_alpha),
            float(lora_dropout),
            ",".join(targets),
        )

    dev = torch.device(str(device))
    model.to(dev)

    # Dataset is just a list; collate does tokenization + doc fetching.
    train_list = list(train_examples)
    dev_list = list(dev_examples)

    def _collate(exs: List[PairExample]):
        # Fetch texts lazily with caching.
        prompts: List[str] = []
        labels: List[int] = []
        for ex in exs:
            txt = doc_cache.get(ex.docid)
            prompts.append(_make_prompt(ex.query, txt))
            labels.append(int(ex.label))
        enc = tok(
            prompts,
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        batch = {k: v for k, v in enc.items()}
        batch["labels_bin"] = torch.tensor(labels, dtype=torch.long)
        return batch

    # Dev "loader": iterate in-order over dev_list with batching.
    dev_bs = max(1, int(batch_size))
    dev_batches = _ceil_div(len(dev_list), dev_bs)

    params = list(_iter_trainable_params(model))
    if not params:
        raise RuntimeError("No trainable parameters found (did you freeze everything?)")

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    def _forward_loss(batch) -> torch.Tensor:
        labels = batch.pop("labels_bin")
        batch = {k: v.to(dev) for k, v in batch.items()}
        labels = labels.to(dev)

        decoder_input_ids = torch.full(
            (int(labels.shape[0]), 1),
            int(decoder_start_id),
            dtype=torch.long,
            device=dev,
        )
        out = model(**batch, decoder_input_ids=decoder_input_ids)
        logits = out.logits  # [B,1,V]
        step_logits = logits[:, 0, :]  # [B,V]

        # Build 2-class logits: class0="false", class1="true"
        two = torch.stack(
            [step_logits[:, int(false_token_id)], step_logits[:, int(true_token_id)]],
            dim=1,
        )  # [B,2]
        loss = F.cross_entropy(two, labels.long())
        return loss

    def _eval_mean_loss() -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            it = range(int(dev_batches))
            it = _maybe_tqdm(
                it,
                enabled=bool(show_progress),
                total=int(dev_batches),
                desc=f"fold {int(fold_idx)} dev",
                leave=False,
            )
            for bi in it:
                start = int(bi) * int(dev_bs)
                end = min(len(dev_list), start + int(dev_bs))
                batch = _collate(list(dev_list[start:end]))
                loss = _forward_loss(batch)
                losses.append(float(loss.detach().cpu().item()))
        return float(sum(losses) / max(1, len(losses)))

    def _save_resume_state(
        *,
        epoch: int,
        batch_pos: int,
        perm: List[int],
        best_dev: float,
        best_epoch: int,
        bad_epochs: int,
        global_steps: int,
    ) -> None:
        """Save resumable training state.

        We only save at safe points (right after optimizer.step + zero_grad),
        so we don't need to serialize partial gradient accumulation buffers.
        """
        import torch

        # Save model weights
        if bool(use_lora):
            os.makedirs(resume_adapter_dir, exist_ok=True)
            model.save_pretrained(resume_adapter_dir)
            tok.save_pretrained(resume_adapter_dir)
        else:
            torch.save(model.state_dict(), resume_model_state_path)

        state = {
            "fold": int(fold_idx),
            "use_lora": bool(use_lora),
            "epoch": int(epoch),
            "batch_pos": int(batch_pos),
            "perm": list(int(i) for i in perm),
            "best_dev": float(best_dev),
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
            "global_steps": int(global_steps),
            "optimizer": optimizer.state_dict(),
            "py_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "time": float(time.time()),
        }
        torch.save(state, resume_state_path)

    # Training
    logger.info(
        "Fold %d train start: train_size=%d dev_size=%d base_model=%s device=%s bs=%d grad_accum=%d max_len=%d",
        int(fold_idx),
        int(len(train_list)),
        int(len(dev_list)),
        str(base_model),
        str(device),
        int(batch_size),
        int(grad_accum_steps),
        int(max_length),
    )

    best_dev = float("inf")
    best_epoch = -1
    bad_epochs = 0
    global_steps = 0
    t0 = time.perf_counter()
    best_adapter_dir = os.path.join(output_dir, "best_adapter") if bool(use_lora) else None
    last_resume_save_t = float(time.time())

    # Resume training counters (if any)
    start_epoch = 0
    start_batch_pos = 0
    start_perm: Optional[List[int]] = None
    if bool(resume) and os.path.exists(resume_state_path):
        import torch

        st = torch.load(resume_state_path, map_location="cpu")
        if isinstance(st, dict):
            start_epoch = int(st.get("epoch", 0))
            start_batch_pos = int(st.get("batch_pos", 0))
            start_perm = list(st.get("perm", []) or []) or None
            best_dev = float(st.get("best_dev", best_dev))
            best_epoch = int(st.get("best_epoch", best_epoch))
            bad_epochs = int(st.get("bad_epochs", bad_epochs))
            global_steps = int(st.get("global_steps", global_steps))
            try:
                optimizer.load_state_dict(cast(dict, st.get("optimizer", {})))
            except Exception as e:
                logger.warning("Fold %d: failed to restore optimizer state: %r", int(fold_idx), e)
            try:
                py_state = st.get("py_random_state")
                if py_state is not None:
                    random.setstate(py_state)
            except Exception:
                pass
            try:
                torch_state = st.get("torch_random_state")
                if torch_state is not None:
                    torch.set_rng_state(torch_state)
            except Exception:
                pass
            logger.info(
                "Fold %d: resume loaded (epoch=%d batch_pos=%d global_steps=%d best_dev=%.6f)",
                int(fold_idx),
                int(start_epoch),
                int(start_batch_pos),
                int(global_steps),
                float(best_dev),
            )

    for epoch in range(int(start_epoch), int(num_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0

        # Deterministic shuffle order so we can resume mid-epoch.
        perm = start_perm if (epoch == start_epoch and start_perm) else _epoch_permutation(len(train_list), seed=seed, fold_idx=fold_idx, epoch=epoch)
        # batch_pos is index of next batch to process (0-based)
        bs = max(1, int(batch_size))
        total_batches = _ceil_div(len(perm), bs)
        batch_pos = int(start_batch_pos) if epoch == start_epoch else 0
        if batch_pos < 0 or batch_pos > total_batches:
            batch_pos = 0

        pbar = _maybe_tqdm(
            range(int(batch_pos), int(total_batches)),
            enabled=bool(show_progress),
            total=int(total_batches),
            initial=int(batch_pos),
            desc=f"fold {int(fold_idx)} epoch {int(epoch+1)}/{int(num_epochs)} train",
            leave=False,
        )
        # `step_in_epoch` counts *batches* (not optimizer steps).
        step_in_epoch = int(batch_pos)
        for bi in pbar:
            step_in_epoch += 1
            start = int(bi) * int(bs)
            end = min(len(perm), start + int(bs))
            exs = [train_list[int(perm[j])] for j in range(int(start), int(end))]
            batch = _collate(exs)
            loss = _forward_loss(batch)
            # Gradient accumulation
            loss_scaled = loss / float(max(1, int(grad_accum_steps)))
            loss_scaled.backward()
            running += float(loss.detach().cpu().item())
            seen += 1
            if _tqdm is not None and hasattr(pbar, "set_postfix"):
                try:
                    pbar.set_postfix(loss=float(loss.detach().cpu().item()), opt_steps=int(global_steps))
                except Exception:
                    pass

            # Optimizer step every grad_accum_steps *batches*.
            if (step_in_epoch % int(max(1, grad_accum_steps))) == 0:
                if float(max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_steps += 1
                start_batch_pos = 0
                start_perm = None

                # Periodic resumable checkpoint (safe point).
                now = float(time.time())
                do_time = float(save_every_seconds) > 0 and (now - last_resume_save_t) >= float(save_every_seconds)
                do_step = int(save_every_steps) > 0 and (int(global_steps) % int(save_every_steps)) == 0
                if do_time or do_step:
                    # Next batch to process is bi+1
                    _save_resume_state(
                        epoch=int(epoch),
                        batch_pos=int(bi) + 1,
                        perm=perm,
                        best_dev=float(best_dev),
                        best_epoch=int(best_epoch),
                        bad_epochs=int(bad_epochs),
                        global_steps=int(global_steps),
                    )
                    last_resume_save_t = now

        train_loss = running / float(max(1, seen))
        # End-of-epoch checkpoint (resume from epoch boundary even if not on save interval).
        _save_resume_state(
            epoch=int(epoch) + 1,
            batch_pos=0,
            perm=[],
            best_dev=float(best_dev),
            best_epoch=int(best_epoch),
            bad_epochs=int(bad_epochs),
            global_steps=int(global_steps),
        )

        dev_loss = _eval_mean_loss()
        logger.info(
            "Fold %d epoch %d/%d: train_loss=%.6f dev_loss=%.6f steps=%d elapsed=%.1fs",
            int(fold_idx),
            int(epoch + 1),
            int(num_epochs),
            float(train_loss),
            float(dev_loss),
            int(global_steps),
            float(time.perf_counter() - t0),
        )

        improved = dev_loss + 1e-9 < best_dev
        if improved:
            best_dev = float(dev_loss)
            best_epoch = int(epoch)
            bad_epochs = 0

            # Save best checkpoint.
            # - For LoRA: save adapters only during training (merging mid-training would break further updates).
            # - For full fine-tune: save the model directly.
            if bool(use_lora):
                assert best_adapter_dir is not None
                os.makedirs(best_adapter_dir, exist_ok=True)
                model.save_pretrained(best_adapter_dir)
                tok.save_pretrained(best_adapter_dir)
                logger.info("Fold %d: saved best LoRA adapter to %s", int(fold_idx), str(best_adapter_dir))
            else:
                model.save_pretrained(output_dir)
                tok.save_pretrained(output_dir)
                logger.info("Fold %d: saved best checkpoint to %s", int(fold_idx), str(output_dir))
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                logger.info(
                    "Fold %d early stopping at epoch %d (best_epoch=%d best_dev_loss=%.6f)",
                    int(fold_idx),
                    int(epoch + 1),
                    int(best_epoch + 1),
                    float(best_dev),
                )
                break

    # If LoRA was used, merge the *best* adapter into the base model and save a plain HF checkpoint
    # at output_dir so existing evaluation code can load it with AutoModelForSeq2SeqLM.from_pretrained().
    if bool(use_lora):
        assert best_adapter_dir is not None
        if not os.path.isdir(best_adapter_dir):
            raise RuntimeError(f"Expected best_adapter_dir to exist, but not found: {best_adapter_dir}")
        logger.info("Fold %d: merging best LoRA adapter into base model for export.", int(fold_idx))
        base = AutoModelForSeq2SeqLM.from_pretrained(str(base_model))
        merged = PeftModel.from_pretrained(base, best_adapter_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
        logger.info("Fold %d: wrote merged HF checkpoint to %s", int(fold_idx), str(output_dir))

    return {
        "fold": int(fold_idx),
        "train_size": int(len(train_list)),
        "dev_size": int(len(dev_list)),
        "best_epoch": int(best_epoch + 1 if best_epoch >= 0 else 1),
        "best_dev_loss": float(best_dev),
        "global_optimizer_steps": int(global_steps),
        "elapsed_s": float(time.perf_counter() - t0),
        "checkpoint_dir": str(output_dir),
        "base_model": str(base_model),
        "use_lora": bool(use_lora),
        "lora": {
            "r": int(lora_r),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "target_modules": [str(x) for x in lora_target_modules],
        }
        if bool(use_lora)
        else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune MonoT5 with topic-level K-fold CV (qrels_50_Queries).")
    p.add_argument("--queries", default="queriesROBUST.txt")
    p.add_argument("--qrels", default="qrels_50_Queries")
    p.add_argument("--index", default="robust04")
    p.add_argument("--out-dir", default="models/approach3_monot5_kfold")
    p.add_argument("--log-level", default="INFO")

    # Data / sampling
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-rel-threshold", type=int, default=1)
    p.add_argument(
        "--bm25-topk",
        type=int,
        default=5000,
        help="BM25 depth used to sample negatives (default: 200). Lower is faster; higher gives harder negatives.",
    )
    p.add_argument("--bm25-k1", type=float, default=0.9)
    p.add_argument("--bm25-b", type=float, default=0.4)
    p.add_argument(
        "--max-pos-per-topic",
        type=int,
        default=0,
        help="Cap positives (relevant docids) per topic to keep CPU runtime manageable (default: 50; 0 disables cap).",
    )
    p.add_argument("--neg-per-pos", type=int, default=4)
    p.add_argument("--hard-neg-ratio", type=float, default=0.7)

    # Model / training
    p.add_argument("--base-model", default="castorini/monot5-base-msmarco")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)

    # LoRA (PEFT)
    p.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters (recommended on CPU).")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q,v",
        help="Comma-separated module name fragments to target with LoRA (default: q,v).",
    )

    # UX
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars (requires tqdm). If tqdm is missing, this is ignored.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume fold training from the latest saved state in each fold directory (if present).",
    )
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=50,
        help="Save resumable state every N optimizer steps (default: 50). Use 0 to disable.",
    )
    p.add_argument(
        "--save-every-seconds",
        type=float,
        default=600.0,
        help="Save resumable state every N seconds (default: 600). Use 0 to disable.",
    )

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    log = _configure_logging(out_dir=out_dir, level=str(args.log_level))
    log.info("cwd=%s out_dir=%s", os.getcwd(), out_dir)

    # CPU stability defaults (avoid oversubscription + tokenizer thread warnings).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    log.info("Loading queries=%s qrels=%s", str(args.queries), str(args.qrels))
    queries = load_queries(str(args.queries))
    qrels = load_qrels(str(args.qrels))
    topics = sorted(int(t) for t in qrels.keys())
    log.info("Loaded topics=%d (expected 50)", int(len(topics)))
    q_by_id = _topic_queries(queries)

    # Pyserini searcher for negative sampling and doc fetching.
    searcher = get_searcher(str(args.index))
    set_bm25(searcher, k1=float(args.bm25_k1), b=float(args.bm25_b))
    doc_cache = _DocTextCache(lambda d: fetch_doc_contents(searcher, d), max_items=50_000)

    folds = _split_topics_kfold(topics, k=int(args.folds), seed=int(args.seed))
    log.info("Prepared %d folds (topic-level).", int(len(folds)))

    # Parse LoRA target modules
    ltm = [s.strip() for s in str(args.lora_target_modules or "").split(",") if s.strip()]
    fold_reports: List[Dict[str, object]] = []

    for fold_idx in range(int(args.folds)):
        dev_topics = list(folds[fold_idx])
        train_topics = [t for i, f in enumerate(folds) if i != fold_idx for t in f]
        log.info(
            "Fold %d topics: train=%d dev=%d",
            int(fold_idx),
            int(len(train_topics)),
            int(len(dev_topics)),
        )

        # Build examples (BM25 negatives)
        train_ex = _build_examples_for_topics(
            topic_ids=train_topics,
            queries_by_id=q_by_id,
            qrels=qrels,
            searcher=searcher,
            bm25_topk=int(args.bm25_topk),
            label_rel_threshold=int(args.label_rel_threshold),
            max_pos_per_topic=int(args.max_pos_per_topic),
            neg_per_pos=int(args.neg_per_pos),
            hard_neg_ratio=float(args.hard_neg_ratio),
            seed=int(args.seed) + 1000 + int(fold_idx),
            logger=log,
        )
        dev_ex = _build_examples_for_topics(
            topic_ids=dev_topics,
            queries_by_id=q_by_id,
            qrels=qrels,
            searcher=searcher,
            bm25_topk=int(args.bm25_topk),
            label_rel_threshold=int(args.label_rel_threshold),
            max_pos_per_topic=int(args.max_pos_per_topic),
            neg_per_pos=int(args.neg_per_pos),
            hard_neg_ratio=float(args.hard_neg_ratio),
            seed=int(args.seed) + 2000 + int(fold_idx),
            logger=log,
        )

        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        report = _train_one_fold(
            fold_idx=int(fold_idx),
            train_examples=train_ex,
            dev_examples=dev_ex,
            output_dir=fold_dir,
            base_model=str(args.base_model),
            device=str(args.device),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            grad_accum_steps=int(args.grad_accum_steps),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            max_grad_norm=float(args.max_grad_norm),
            num_epochs=int(args.epochs),
            patience=int(args.patience),
            seed=int(args.seed) + int(fold_idx),
            use_lora=bool(args.use_lora),
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            lora_target_modules=ltm,
            doc_cache=doc_cache,
            logger=log,
            show_progress=bool(args.progress),
            resume=bool(args.resume),
            save_every_steps=int(args.save_every_steps),
            save_every_seconds=float(args.save_every_seconds),
        )
        report["topics"] = {"train": [int(t) for t in train_topics], "dev": [int(t) for t in dev_topics]}
        fold_reports.append(report)

        with open(os.path.join(fold_dir, "fold_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")

    summary = {
        "folds": int(args.folds),
        "seed": int(args.seed),
        "data": {"queries": str(args.queries), "qrels": str(args.qrels), "index": str(args.index)},
        "bm25": {
            "topk": int(args.bm25_topk),
            "k1": float(args.bm25_k1),
            "b": float(args.bm25_b),
        },
        "params": {
            "label_rel_threshold": int(args.label_rel_threshold),
            "max_pos_per_topic": int(args.max_pos_per_topic),
            "neg_per_pos": int(args.neg_per_pos),
            "hard_neg_ratio": float(args.hard_neg_ratio),
            "base_model": str(args.base_model),
            "device": str(args.device),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "use_lora": bool(args.use_lora),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target_modules": ltm,
        },
        "folds_report": fold_reports,
    }
    out_path = os.path.join(out_dir, "cv_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    log.info("Wrote CV summary: %s", out_path)
    print(f"Wrote CV summary: {out_path}")
    print(f"Fold checkpoints under: {out_dir}/fold_*")
    print(f"Logs: {os.path.join(out_dir, 'train.log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

