"""Learning-to-rank model backends for clustpsg.

Supports:
- LinearSVC (pointwise classifier baseline)
- SVM^rank (Joachims) via external binaries (recommended for ranking)

Design goals:
- deterministic and reproducible (fixed random_state where relevant)
- stable feature schema (feature_names drives vectorization order)
- simple persistence (pickle)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from rag.clustpsg.dataset import LTRInstance
from rag.clustpsg.svm_rank import (
    read_predictions,
    run_svmrank_classify,
    run_svmrank_learn,
    write_svmrank_file,
)


@dataclass(frozen=True)
class TrainedModel:
    feature_names: List[str]
    model_type: str
    model: object


def _fit_feature_scaler(
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    *,
    mode: str,
) -> Dict[str, object]:
    """Fit a simple global feature scaler for RankSVM.

    This is intentionally dependency-free (no numpy/sklearn required).
    Modes:
      - "none": no scaling
      - "minmax": scale each feature to [0, 1] using global min/max
      - "zscore": standardize each feature to (x-mean)/std using global mean/std
    """
    mode = (mode or "none").lower()
    if mode in ("none", ""):
        return {"mode": "none"}

    # Collect per-feature values (including implicit zeros).
    # Note: RankSVM input is dense-by-feature-name, but we omit zeros for sparsity when writing.
    n = int(len(instances))
    if n <= 0:
        return {"mode": "none"}

    if mode == "minmax":
        mins: Dict[str, float] = {name: 0.0 for name in feature_names}
        maxs: Dict[str, float] = {name: 0.0 for name in feature_names}
        # Initialize mins/maxs from first instance to avoid biasing toward 0.0 if values are always positive/negative.
        first = instances[0]
        for name in feature_names:
            v0 = float(first.features.get(name, 0.0))
            mins[name] = v0
            maxs[name] = v0
        for ins in instances[1:]:
            for name in feature_names:
                v = float(ins.features.get(name, 0.0))
                if v < mins[name]:
                    mins[name] = v
                if v > maxs[name]:
                    maxs[name] = v
        return {"mode": "minmax", "min": mins, "max": maxs}

    if mode == "zscore":
        sums: Dict[str, float] = {name: 0.0 for name in feature_names}
        sums2: Dict[str, float] = {name: 0.0 for name in feature_names}
        for ins in instances:
            for name in feature_names:
                v = float(ins.features.get(name, 0.0))
                sums[name] += v
                sums2[name] += v * v
        means: Dict[str, float] = {name: (sums[name] / float(n)) for name in feature_names}
        stds: Dict[str, float] = {}
        for name in feature_names:
            mu = means[name]
            var = (sums2[name] / float(n)) - (mu * mu)
            if var < 0.0:
                var = 0.0
            std = var ** 0.5
            stds[name] = std
        return {"mode": "zscore", "mean": means, "std": stds}

    raise ValueError(f"Unknown RankSVM feature normalization mode: {mode!r} (use none|minmax|zscore)")


def _apply_feature_scaler_to_instances(
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    scaler: Mapping[str, object],
) -> List[LTRInstance]:
    """Return new instances with features transformed by the provided scaler."""
    mode = str(scaler.get("mode", "none")).lower()
    if mode in ("none", ""):
        return list(instances)

    out: List[LTRInstance] = []
    eps = 1e-12
    if mode == "minmax":
        mins = scaler.get("min", {})
        maxs = scaler.get("max", {})
        for ins in instances:
            feats: Dict[str, float] = {}
            for name in feature_names:
                x = float(ins.features.get(name, 0.0))
                mn = float(mins.get(name, 0.0))
                mx = float(maxs.get(name, 0.0))
                denom = (mx - mn)
                feats[name] = 0.0 if abs(denom) <= eps else ((x - mn) / denom)
            out.append(LTRInstance(topic_id=ins.topic_id, docid=ins.docid, label=int(ins.label), features=feats))
        return out

    if mode == "zscore":
        means = scaler.get("mean", {})
        stds = scaler.get("std", {})
        for ins in instances:
            feats = {}
            for name in feature_names:
                x = float(ins.features.get(name, 0.0))
                mu = float(means.get(name, 0.0))
                sd = float(stds.get(name, 0.0))
                feats[name] = 0.0 if sd <= eps else ((x - mu) / sd)
            out.append(LTRInstance(topic_id=ins.topic_id, docid=ins.docid, label=int(ins.label), features=feats))
        return out

    # Unknown -> treat as no scaling (forward compat).
    return list(instances)


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


def train_svm_rank(
    *,
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    learn_bin: str,
    classify_bin: str,
    C: float,
    work_dir: str,
    external_model_path: str,
    feature_norm: str = "none",
) -> TrainedModel:
    """Train SVM^rank and return a TrainedModel that references the external model file."""
    import os

    os.makedirs(work_dir, exist_ok=True)
    train_path = os.path.join(work_dir, "svmrank_train.dat")
    scaler = _fit_feature_scaler(instances, feature_names, mode=feature_norm)
    train_instances = _apply_feature_scaler_to_instances(instances, feature_names, scaler)
    write_svmrank_file(instances=train_instances, feature_names=feature_names, output_path=train_path, include_labels=True)

    run_svmrank_learn(learn_bin=learn_bin, train_path=train_path, model_out_path=external_model_path, C=C)

    meta = {
        "backend": "svm_rank",
        "learn_bin": learn_bin,
        "classify_bin": classify_bin,
        "external_model_path": external_model_path,
        "work_dir": work_dir,
        "feature_scaler": dict(scaler),
    }
    return TrainedModel(feature_names=list(feature_names), model_type="SVMrank", model=meta)


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
    if model.model_type == "LinearSVC":
        X, _y = vectorize_instances(instances, model.feature_names)
        clf = model.model
        if not hasattr(clf, "decision_function"):
            raise TypeError("LinearSVC pipeline does not expose decision_function for scoring.")
        scores = clf.decision_function(X)
        out: Dict[Tuple[int, str], float] = {}
        for ins, s in zip(instances, scores):
            out[(ins.topic_id, ins.docid)] = float(s)
        return out

    if model.model_type == "SVMrank":
        import os

        meta = model.model
        if not isinstance(meta, dict):
            raise TypeError("SVMrank model metadata is invalid.")
        classify_bin = str(meta["classify_bin"])
        external_model_path = str(meta["external_model_path"])
        work_dir = str(meta["work_dir"])
        test_path = os.path.join(work_dir, "svmrank_test.dat")
        pred_path = os.path.join(work_dir, "svmrank_pred.txt")
        scaler = meta.get("feature_scaler", {"mode": "none"})
        test_instances = _apply_feature_scaler_to_instances(instances, model.feature_names, scaler if isinstance(scaler, dict) else {"mode": "none"})
        write_svmrank_file(instances=test_instances, feature_names=model.feature_names, output_path=test_path, include_labels=False)
        run_svmrank_classify(
            classify_bin=classify_bin,
            test_path=test_path,
            model_path=external_model_path,
            predictions_out_path=pred_path,
        )
        preds = read_predictions(pred_path)
        if len(preds) != len(instances):
            raise RuntimeError(f"SVMrank predictions length mismatch: {len(preds)} != {len(instances)}")
        out: Dict[Tuple[int, str], float] = {}
        for ins, s in zip(instances, preds):
            out[(ins.topic_id, ins.docid)] = float(s)
        return out

    raise ValueError(f"Unknown model_type: {model.model_type!r}")


def train_model_from_config(
    *,
    instances: Sequence[LTRInstance],
    feature_names: Sequence[str],
    svm_cfg: Mapping[str, object],
) -> TrainedModel:
    """Train the configured backend."""
    backend = str(svm_cfg.get("backend", "svm_rank")).lower()
    if backend in ("linear_svc", "linearsvc", "linear"):
        return train_linear_svm(
            instances=instances,
            feature_names=feature_names,
            C=float(svm_cfg.get("C", 1.0)),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            max_iter=int(svm_cfg.get("max_iter", 5000)),
            random_state=int(svm_cfg.get("random_state", 42)),
        )
    if backend in ("svm_rank", "svmrank"):
        learn_bin = str(svm_cfg.get("svm_rank_learn_bin", "svm_rank_learn"))
        classify_bin = str(svm_cfg.get("svm_rank_classify_bin", "svm_rank_classify"))
        work_dir = str(svm_cfg.get("svm_rank_work_dir", "models/svmrank_work"))
        external_model_path = str(svm_cfg.get("svm_rank_model_path", "models/svmrank.model"))
        feature_norm = str(svm_cfg.get("svm_rank_feature_norm", "none"))
        return train_svm_rank(
            instances=instances,
            feature_names=feature_names,
            learn_bin=learn_bin,
            classify_bin=classify_bin,
            C=float(svm_cfg.get("C", 1.0)),
            work_dir=work_dir,
            external_model_path=external_model_path,
            feature_norm=feature_norm,
        )
    raise ValueError(f"Unknown SVM backend: {backend!r}")


