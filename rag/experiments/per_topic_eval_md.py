"""Generate an EVAL_PER_TOPIC_*.md report from one or more TREC run files.

This is a lightweight alternative to `rag.approach3.evaluate_all_topics`, intended for:
- Comparing already-generated run files
- Producing a Markdown summary similar to `EVAL_PER_TOPIC_V2*.md`

Example:
  .venv/bin/python -m rag.experiments.per_topic_eval_md \
    --qrels qrels_50_Queries \
    --out-md results/EVAL_PER_TOPIC_BM25_ONLY.md \
    --run "BM25:run_1_train.res:5000:1000"

Run spec format:
  --run "LABEL:PATH[:RECALL_K[:AP_K]]"
or:
  --run "LABEL,PATH[,RECALL_K[,AP_K]]"

Notes:
- Relevance is treated as binary (rel > 0).
- AP@K is used for MAP@K (mean AP over topics).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rag.eval import average_precision, load_trec_run
from rag.io import load_qrels
from rag.logging_utils import configure_logging


def recall_at_k(qrels_for_topic: Mapping[str, int], ranked_docids: Sequence[str], k: int) -> float:
    """Recall@k for a single topic (binary relevance: rel>0)."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = {docid for docid, rel in qrels_for_topic.items() if int(rel) > 0}
    if not relevant:
        return 0.0
    hit = 0
    for docid in ranked_docids[:k]:
        if docid in relevant:
            hit += 1
    return float(hit) / float(len(relevant))


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: str
    recall_k: int
    ap_k: int


def _parse_run_spec(raw: str, *, default_recall_k: int, default_ap_k: int) -> RunSpec:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty --run spec")

    # Allow either "label:path:..." or "label,path,..."
    parts: List[str]
    if "," in s and ":" not in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split(":")]

    if len(parts) < 2:
        raise ValueError(f"Invalid --run spec {raw!r}; expected at least LABEL and PATH.")

    label = parts[0].strip()
    path = parts[1].strip()
    if not label:
        raise ValueError(f"Invalid --run spec {raw!r}; empty LABEL.")
    if not path:
        raise ValueError(f"Invalid --run spec {raw!r}; empty PATH.")

    recall_k = int(parts[2]) if len(parts) >= 3 and parts[2] else int(default_recall_k)
    ap_k = int(parts[3]) if len(parts) >= 4 and parts[3] else int(default_ap_k)
    if recall_k <= 0 or ap_k <= 0:
        raise ValueError(f"Invalid --run spec {raw!r}; RECALL_K and AP_K must be > 0.")

    return RunSpec(label=label, path=path, recall_k=recall_k, ap_k=ap_k)


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs)) / float(len(xs)) if xs else 0.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate per-topic eval Markdown for one or more runs.")
    p.add_argument("--qrels", default="qrels_50_Queries", help="Path to qrels file (train topics).")
    p.add_argument("--out-md", required=True, help="Where to write the Markdown report.")
    p.add_argument("--title-suffix", default=None, help="Optional suffix appended to the main title line.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")

    p.add_argument("--default-recall-k", type=int, default=5000, help="Default Recall@K if not provided per run.")
    p.add_argument("--default-ap-k", type=int, default=1000, help="Default AP@K if not provided per run.")
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help='Run spec: "LABEL:PATH[:RECALL_K[:AP_K]]" (or comma-separated). Can be repeated.',
    )
    p.add_argument(
        "--delta",
        choices=["auto", "off"],
        default="auto",
        help="If exactly 2 runs, add ΔAP column (run2 - run1). Default: auto.",
    )
    return p


def _write_markdown(
    *,
    out_md: str,
    qrels_name: str,
    runs: Sequence[RunSpec],
    topics: Sequence[int],
    depth_by_run: Mapping[str, Tuple[int, int]],
    recall_by_run: Mapping[str, Mapping[int, float]],
    ap_by_run: Mapping[str, Mapping[int, float]],
    map_by_run: Mapping[str, float],
    delta_mode: str,
    title_suffix: Optional[str],
) -> None:
    lines: List[str] = []

    title = f"## Per-topic evaluation (ROBUST04 {qrels_name})"
    if title_suffix:
        title += f" — {title_suffix}"
    lines.append(title + "\n\n")

    lines.append("**What this doc contains**: per-topic Recall and AP (used for MAP) for:\n")
    for r in runs:
        lines.append(f"- **{r.label}**: Recall@{int(r.recall_k)} and AP@{int(r.ap_k)}\n")
    lines.append("\n")

    lines.append("**Important notes**:\n")
    lines.append("- “MAP per topic” is **AP**; MAP is the mean of AP over topics.\n")
    lines.append("- If a run file contains fewer than K documents for a topic, Recall/AP use the available depth.\n")
    for r in runs:
        mn, mx = depth_by_run.get(r.label, (0, 0))
        lines.append(f"- Run depth (**{r.label}**): min={int(mn)} max={int(mx)} docs/topic (over qrels topics)\n")
    lines.append("\n")

    lines.append(f"### Overall ({len(topics)} topics)\n\n")
    for r in runs:
        lines.append(f"- **{r.label} MAP@{int(r.ap_k)}**: {float(map_by_run[r.label]):.4f}\n")
    show_delta = (delta_mode == "auto" and len(runs) == 2)
    if show_delta:
        a, b = runs[0], runs[1]
        lines.append(f"- **Δ ({b.label} - {a.label})**: {float(map_by_run[b.label]) - float(map_by_run[a.label]):+.4f}\n")
    lines.append("\n")

    lines.append("### Per-topic table\n\n")
    header_cols: List[str] = ["Topic"]
    for r in runs:
        header_cols.append(f"{r.label} Recall@{int(r.recall_k)}")
        header_cols.append(f"{r.label} AP@{int(r.ap_k)}")
    if show_delta:
        header_cols.append("ΔAP")

    lines.append("| " + " | ".join(header_cols) + " |\n")
    lines.append("|" + "|".join(["---:" for _ in header_cols]) + "|\n")

    for t in topics:
        row: List[str] = [str(int(t))]
        for r in runs:
            row.append(f"{float(recall_by_run[r.label].get(int(t), 0.0)):.4f}")
            row.append(f"{float(ap_by_run[r.label].get(int(t), 0.0)):.4f}")
        if show_delta:
            a, b = runs[0], runs[1]
            da = float(ap_by_run[b.label].get(int(t), 0.0)) - float(ap_by_run[a.label].get(int(t), 0.0))
            row.append(f"{da:+.4f}")
        lines.append("| " + " | ".join(row) + " |\n")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> int:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("rag.experiments.per_topic_eval_md")

    if args.default_recall_k <= 0 or args.default_ap_k <= 0:
        raise ValueError("--default-recall-k and --default-ap-k must be > 0")

    qrels = load_qrels(args.qrels)
    topics = sorted(int(t) for t in qrels.keys())
    if not topics:
        raise ValueError("qrels contains no topics")

    runs = [
        _parse_run_spec(r, default_recall_k=int(args.default_recall_k), default_ap_k=int(args.default_ap_k))
        for r in (args.run or [])
    ]
    if not runs:
        raise ValueError("At least one --run spec is required")

    recall_by_run: Dict[str, Dict[int, float]] = {}
    ap_by_run: Dict[str, Dict[int, float]] = {}
    map_by_run: Dict[str, float] = {}
    depth_by_run: Dict[str, Tuple[int, int]] = {}

    for r in runs:
        need_k = max(int(r.recall_k), int(r.ap_k))
        log.info("Loading run: %s (%s) k=%d", r.label, r.path, need_k)
        run = load_trec_run(r.path, k=need_k)

        depths = [len(run.get(int(t), [])) for t in topics]
        depth_by_run[r.label] = (min(depths) if depths else 0, max(depths) if depths else 0)

        per_topic_recall: Dict[int, float] = {}
        per_topic_ap: Dict[int, float] = {}
        for t in topics:
            ranked = run.get(int(t), [])
            per_topic_recall[int(t)] = recall_at_k(qrels.get(int(t), {}), ranked, k=int(r.recall_k))
            per_topic_ap[int(t)] = float(average_precision(qrels.get(int(t), {}), ranked, k=int(r.ap_k)))

        recall_by_run[r.label] = per_topic_recall
        ap_by_run[r.label] = per_topic_ap
        map_by_run[r.label] = _mean([per_topic_ap[t] for t in topics])

    qrels_name = os.path.basename(str(args.qrels)) or "qrels"
    _write_markdown(
        out_md=str(args.out_md),
        qrels_name=qrels_name,
        runs=runs,
        topics=topics,
        depth_by_run=depth_by_run,
        recall_by_run=recall_by_run,
        ap_by_run=ap_by_run,
        map_by_run=map_by_run,
        delta_mode=str(args.delta),
        title_suffix=str(args.title_suffix) if args.title_suffix else None,
    )
    print(f"Wrote: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

