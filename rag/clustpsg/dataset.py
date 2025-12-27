"""PR6: Build labeled training data for clustpsg (doc-level LTR).

We are given doc-level qrels (binary relevance). This module produces a training
set of (X, y) at the *document* level, where X comes from aggregating cluster
features derived from passages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rag.config import ApproachConfig
from rag.clustpsg.cluster import Cluster
from rag.clustpsg.features import FeatureContext, compute_cluster_features, get_enabled_feature_names
from rag.types import Document, Passage, Query


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)


@dataclass(frozen=True)
class LTRInstance:
    topic_id: int
    docid: str
    label: int
    features: Dict[str, float]


def _rank_map_docs(docs: Sequence[Document]) -> Dict[str, int]:
    return {d.id: i for i, d in enumerate(docs, start=1)}


def _rank_map_passages(passages: Sequence[Passage]) -> Dict[tuple[str, int], int]:
    return {(p.document_id, p.index): i for i, p in enumerate(passages, start=1)}


def _agg_init(mode: str) -> float:
    if mode == "max":
        return float("-inf")
    if mode in ("sum", "mean"):
        return 0.0
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def _agg_update(mode: str, current: float, value: float) -> float:
    if mode == "max":
        return value if value > current else current
    if mode in ("sum", "mean"):
        return current + value
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def _agg_finalize(mode: str, total: float, count: int) -> float:
    if mode == "max":
        return 0.0 if total == float("-inf") else total
    if mode == "sum":
        return total
    if mode == "mean":
        return (total / float(count)) if count > 0 else 0.0
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def build_doc_level_training_set(
    *,
    queries: Sequence[Query],
    qrels: Mapping[int, Mapping[str, int]],
    doc_candidates_by_topic: Mapping[int, Sequence[Document]],
    ranked_passages_by_topic: Mapping[int, Sequence[Passage]],
    clusters_by_topic: Mapping[int, Sequence[Cluster]],
    config: ApproachConfig,
) -> Tuple[List[LTRInstance], List[str]]:
    """Build doc-level training instances.

    Inputs are expected to be aligned by topic:
    - doc_candidates_by_topic[topic] is the list of candidate docs to label/score
    - ranked_passages_by_topic[topic] is the ranked list of passages (PR3 output)
    - clusters_by_topic[topic] clusters refer to indices into the *passage list*
      used for clustering. In this simple implementation, we assume clustering
      was performed over the same ranked_passages_by_topic[topic] list (or a
      prefix of it) and indices match.
    """
    params = config.params or {}
    enabled = params.get("enabled_features", [])
    agg_mode = str(params.get("doc_feature_aggregation", "max")).lower()
    label_rel_threshold = int(params.get("label_rel_threshold", 1))
    max_pairwise = int(params.get("feature_max_pairwise_sim", 2000))
    similarity_cfg = params.get("clustering", {}) or {}

    feature_names = get_enabled_feature_names(enabled)

    instances: List[LTRInstance] = []

    # Build per-topic
    q_by_id = {q.id: q for q in queries}
    for topic_id in sorted(q_by_id.keys()):
        q = q_by_id[topic_id]
        docs = list(doc_candidates_by_topic.get(topic_id, []))
        ranked_passages = list(ranked_passages_by_topic.get(topic_id, []))
        clusters = list(clusters_by_topic.get(topic_id, []))

        doc_rank = _rank_map_docs(docs)
        passage_rank = _rank_map_passages(ranked_passages)

        # Prepare empty feature accumulators per doc
        doc_feat_totals: Dict[str, Dict[str, float]] = {d.id: {n: _agg_init(agg_mode) for n in feature_names} for d in docs}
        doc_feat_counts: Dict[str, int] = {d.id: 0 for d in docs}

        for cl in clusters:
            ctx = FeatureContext(
                query=q,
                cluster=cl,
                passages=ranked_passages,
                passage_rank=passage_rank,
                document_rank=doc_rank,
                stopwords=DEFAULT_STOPWORDS,
                similarity_cfg=similarity_cfg,
                max_pairwise=max_pairwise,
            )
            feats = compute_cluster_features(ctx=ctx, enabled_features=feature_names)

            # Assign this cluster's features to each document that appears in the cluster
            docs_in_cluster = {ranked_passages[i].document_id for i in cl.passage_indices if 0 <= i < len(ranked_passages)}
            for docid in docs_in_cluster:
                if docid not in doc_feat_totals:
                    continue  # ignore docs outside candidate set
                for name, value in feats.items():
                    doc_feat_totals[docid][name] = _agg_update(agg_mode, doc_feat_totals[docid][name], float(value))
                doc_feat_counts[docid] += 1

        # Finalize instances with labels from qrels (binary)
        qrels_topic = qrels.get(topic_id, {})
        for d in docs:
            rel = int(qrels_topic.get(d.id, 0))
            label = 1 if rel >= label_rel_threshold else 0
            finalized = {n: _agg_finalize(agg_mode, doc_feat_totals[d.id][n], doc_feat_counts[d.id]) for n in feature_names}
            instances.append(LTRInstance(topic_id=topic_id, docid=d.id, label=label, features=finalized))

    return instances, feature_names


def write_ltr_csv(instances: Sequence[LTRInstance], feature_names: Sequence[str], output_path: str) -> None:
    """Write LTR instances to CSV for debugging / offline training."""
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "docid", "label"] + list(feature_names))
        for ins in instances:
            w.writerow([ins.topic_id, ins.docid, ins.label] + [f"{ins.features.get(n, 0.0):.6f}" for n in feature_names])


