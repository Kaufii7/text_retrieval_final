"""SVM^rank integration (Joachims).

This module formats training/inference data for SVM^rank and calls the external
CLI binaries:
- svm_rank_learn
- svm_rank_classify

Expected input format (per line):
  <label> qid:<topic_id> 1:<v1> 2:<v2> ... # <docid>

Notes:
- Feature indices are 1-based.
- Labels can be 0/1 for binary relevance.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Sequence

from rag.clustpsg.dataset import LTRInstance


def write_svmrank_file(
    *,
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    output_path: str,
    include_labels: bool,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ins in instances:
            label = int(ins.label) if include_labels else 0
            feats = []
            # 1-based indices; omit zeros for sparsity
            for idx, name in enumerate(feature_names, start=1):
                v = float(ins.features.get(name, 0.0))
                if v == 0.0:
                    continue
                feats.append(f"{idx}:{v:.6f}")
            feat_str = " ".join(feats)
            if feat_str:
                line = f"{label} qid:{ins.topic_id} {feat_str} # {ins.docid}\n"
            else:
                line = f"{label} qid:{ins.topic_id} # {ins.docid}\n"
            f.write(line)


def run_svmrank_learn(
    *,
    learn_bin: str,
    train_path: str,
    model_out_path: str,
    C: float = 1.0,
) -> None:
    os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
    cmd = [learn_bin, "-c", str(C), train_path, model_out_path]
    subprocess.run(cmd, check=True)


def run_svmrank_classify(
    *,
    classify_bin: str,
    test_path: str,
    model_path: str,
    predictions_out_path: str,
) -> None:
    os.makedirs(os.path.dirname(predictions_out_path) or ".", exist_ok=True)
    cmd = [classify_bin, test_path, model_path, predictions_out_path]
    subprocess.run(cmd, check=True)


def read_predictions(path: str) -> List[float]:
    preds: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            preds.append(float(s))
    return preds


