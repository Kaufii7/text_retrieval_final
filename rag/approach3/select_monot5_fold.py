"""Helper: pick the "best" MonoT5 fold from k-fold training artifacts.

By default, this ranks folds by lowest `best_dev_loss` from:
- models/approach3_monot5_kfold/cv_summary.json (preferred), or
- models/approach3_monot5_kfold/fold_*/fold_report.json (fallback)

This is a convenience tool to decide which fold directory to pass to:
  python -m rag.approach3.bm25_ce_pipeline --reranker-type monot5 --monot5-model <FOLD_DIR> ...
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FoldInfo:
    fold: int
    checkpoint_dir: str
    best_dev_loss: float
    best_epoch: int
    train_size: int
    dev_size: int


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def _from_cv_summary(path: str) -> List[FoldInfo]:
    obj = _read_json(path)
    fr = obj.get("folds_report")
    if not isinstance(fr, list) or not fr:
        raise ValueError(f"{path}: missing/invalid 'folds_report'")
    out: List[FoldInfo] = []
    for x in fr:
        if not isinstance(x, dict):
            continue
        fold = int(x.get("fold"))
        ckpt = str(x.get("checkpoint_dir") or "")
        best_dev_loss = float(x.get("best_dev_loss"))
        best_epoch = int(x.get("best_epoch"))
        train_size = int(x.get("train_size"))
        dev_size = int(x.get("dev_size"))
        out.append(
            FoldInfo(
                fold=fold,
                checkpoint_dir=ckpt,
                best_dev_loss=best_dev_loss,
                best_epoch=best_epoch,
                train_size=train_size,
                dev_size=dev_size,
            )
        )
    if not out:
        raise ValueError(f"{path}: no fold reports found")
    return out


def _from_fold_reports_dir(root: str) -> List[FoldInfo]:
    out: List[FoldInfo] = []
    for name in os.listdir(root):
        if not name.startswith("fold_"):
            continue
        fold_dir = os.path.join(root, name)
        if not os.path.isdir(fold_dir):
            continue
        report_path = os.path.join(fold_dir, "fold_report.json")
        if not os.path.exists(report_path):
            continue
        x = _read_json(report_path)
        out.append(
            FoldInfo(
                fold=int(x.get("fold", int(name.replace("fold_", "")))),
                checkpoint_dir=str(x.get("checkpoint_dir") or fold_dir),
                best_dev_loss=float(x.get("best_dev_loss")),
                best_epoch=int(x.get("best_epoch")),
                train_size=int(x.get("train_size")),
                dev_size=int(x.get("dev_size")),
            )
        )
    if not out:
        raise ValueError(f"{root}: no fold_*/fold_report.json found")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select best MonoT5 fold from k-fold training outputs.")
    p.add_argument(
        "--cv-summary",
        default="models/approach3_monot5_kfold/cv_summary.json",
        help="Path to cv_summary.json (preferred input).",
    )
    p.add_argument(
        "--folds-dir",
        default="models/approach3_monot5_kfold",
        help="Fallback: directory containing fold_*/fold_report.json.",
    )
    p.add_argument(
        "--metric",
        choices=["best_dev_loss"],
        default="best_dev_loss",
        help="Which metric to rank by (lower is better).",
    )
    p.add_argument("--top", type=int, default=5, help="How many folds to print (default: 5).")
    p.add_argument(
        "--print-best-path",
        action="store_true",
        help="Print only the best checkpoint_dir path (useful for scripting).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    folds: Optional[List[FoldInfo]] = None
    try:
        if args.cv_summary and os.path.exists(str(args.cv_summary)):
            folds = _from_cv_summary(str(args.cv_summary))
    except Exception:
        folds = None

    if folds is None:
        folds = _from_fold_reports_dir(str(args.folds_dir))

    # Rank by metric (currently only best_dev_loss).
    folds_sorted = sorted(folds, key=lambda f: (float(f.best_dev_loss), int(f.fold)))
    best = folds_sorted[0]

    if bool(args.print_best_path):
        print(str(best.checkpoint_dir))
        return 0

    topn = max(1, int(args.top))
    print("rank\tfold\tbest_dev_loss\tbest_epoch\ttrain_size\tdev_size\tcheckpoint_dir")
    for i, f in enumerate(folds_sorted[:topn], start=1):
        print(
            f"{i}\t{int(f.fold)}\t{float(f.best_dev_loss):.6f}\t{int(f.best_epoch)}\t{int(f.train_size)}\t{int(f.dev_size)}\t{str(f.checkpoint_dir)}"
        )
    print(f"\nBest fold by {args.metric}: fold_{int(best.fold)}  (best_dev_loss={float(best.best_dev_loss):.6f})")
    print(f"Use as: --monot5-model \"{best.checkpoint_dir}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

