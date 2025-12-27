"""PR7: Learning-to-rank model for clustpsg (SVM).

This module trains a (linear) SVM on the doc-level training instances produced
by `rag.clustpsg.dataset`.

Design goals:
- deterministic and reproducible (fixed random_state where relevant)
- stable feature schema (feature_names drives vectorization order)
- simple persistence (pickle)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from rag.clustpsg.dataset import LTRInstance


@dataclass(frozen=True)
class TrainedModel:
    feature_names: List[str]
    model_type: str
    model: object


def vectorize_instances(instances: Sequence[LTRInstance], feature_names: Sequence[str]) -> Tuple["object", "object"]:
    """Vectorize instances into (X, y) using the provided feature order.

    Returns numpy arrays if numpy is available (via sklearn), otherwise lists.
    """
    X_rows: List[List[float]] = []
    y: List[int] = []
    for ins in instances:
        X_rows.append([float(ins.features.get(name, 0.0)) for name in feature_names])
        y.append(int(ins.label))

    # sklearn will convert lists to numpy arrays; keep dependency-light here.
    return X_rows, y


def train_linear_svm(
    *,
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 5000,
    random_state: int = 42,
) -> TrainedModel:
    """Train a linear SVM classifier and return a TrainedModel.

    Uses scikit-learn if available.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for PR7 SVM training. Install it in your venv (e.g., `pip install scikit-learn`)."
        ) from e

    X, y = vectorize_instances(instances, feature_names)

    # Standardize features for stable optimization.
    # with_mean=True is fine because inputs are dense numeric vectors.
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=float(C), class_weight=class_weight, max_iter=int(max_iter), random_state=int(random_state))),
        ]
    )
    clf.fit(X, y)

    return TrainedModel(feature_names=list(feature_names), model_type="LinearSVC", model=clf)


def save_model(m: TrainedModel, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(m, f)


def load_model(path: str) -> TrainedModel:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrainedModel):
        raise TypeError(f"Model at {path!r} is not a TrainedModel")
    return obj


def score_documents(
    *,
    model: TrainedModel,
    instances: Sequence[LTRInstance],
) -> Dict[Tuple[int, str], float]:
    """Score instances using the trained model.

    Returns:
      dict[(topic_id, docid)] -> score (higher is better)
    """
    X, _y = vectorize_instances(instances, model.feature_names)
    clf = model.model

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        raise TypeError("Model does not expose decision_function for scoring.")

    # scores is array-like
    out: Dict[Tuple[int, str], float] = {}
    for ins, s in zip(instances, scores):
        out[(ins.topic_id, ins.docid)] = float(s)
    return out


