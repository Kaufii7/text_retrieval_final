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
    doc_rank_candidates_by_topic: Mapping[int, Sequence[Document]] | None = None,
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

        # Important: some features depend on document rank (retrieval order).
        # When training docs come from qrels (sorted docids), `docs` is NOT in retrieval order.
        # Allow passing a separate candidate list to define a consistent rank signal.
        rank_docs = list((doc_rank_candidates_by_topic or {}).get(topic_id, docs))
        # Ensure every doc we build instances for has a rank (missing => would become 0 RR).
        # We append unseen docs after the ranked list, preserving `docs` order.
        seen = {d.id for d in rank_docs}
        for d in docs:
            if d.id not in seen:
                rank_docs.append(d)
                seen.add(d.id)
        doc_rank = _rank_map_docs(rank_docs)
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


def build_cluster_level_training_set(
    *,
    queries: Sequence[Query],
    qrels: Mapping[int, Mapping[str, int]],
    doc_candidates_by_topic: Mapping[int, Sequence[Document]],
    doc_rank_candidates_by_topic: Mapping[int, Sequence[Document]] | None = None,
    ranked_passages_by_topic: Mapping[int, Sequence[Passage]],
    clusters_by_topic: Mapping[int, Sequence[Cluster]],
    config: ApproachConfig,
    pseudo_relevant_passages_by_topic: Mapping[int, "set[tuple[str, int]]"] | None = None,
) -> Tuple[List[LTRInstance], List[str]]:
    """Build *cluster*-level training instances.

    This supports the intended ClustPsg pipeline:
      docs -> passages -> clusters -> SVM (rank clusters) -> propagate to passages -> RRF -> docs

    Labels:
      Default ("any_relevant_doc"):
        cluster label = 1 iff the cluster contains at least one passage whose source document
        is relevant in qrels for this topic (rel >= label_rel_threshold).

      Optional ("top_weighted_density"):
        Compute a top-weighted relevance density of passages in the cluster:

          density = sum_{p in cluster, rel(doc(p))=1} w(rank(p)) / sum_{p in cluster} w(rank(p))
          w(r) = 1 / (rr_k + r)   where r is the global passage rank (1 = best)

        Then bucket to an integer label for SVM^rank:
          2 if density >= threshold_high
          1 if density >= threshold_low
          0 otherwise

        Config (under cfg.params):
          cluster_labeling.method: "any_relevant_doc" | "top_weighted_density"
          cluster_labeling.rr_k: int (default: 0)
          cluster_labeling.threshold_low: float (default: 0.10)
          cluster_labeling.threshold_high: float (default: 0.30)

      Optional ("best_evidence"):
        Label a cluster by the *best* (highest-ranked) evidence it contains:
        find best_rank = min global passage rank among passages in the cluster whose source doc is relevant.
        Then bucket by rank thresholds:
          2 if best_rank <= best_rank_high
          1 if best_rank <= best_rank_low
          0 otherwise (or if no relevant-doc passage exists in the cluster)

        Config (under cfg.params):
          cluster_labeling.method: "best_evidence"
          cluster_labeling.best_rank_low: int (default: 100)
          cluster_labeling.best_rank_high: int (default: 20)

      Optional ("pseudo_passage_overlap"):
        Training-only: label clusters using *pseudo-relevant passages* derived from qrels-relevant documents.

        Expected input:
          pseudo_relevant_passages_by_topic[topic_id] is a set of (docid, passage_index) pairs that were
          marked as pseudo-relevant (e.g., top-k passages per relevant doc by Lucene BM25+RM3).

        Labeling:
          overlap = |{(docid, idx) in cluster} ∩ pseudo_relevant_passages_by_topic[topic_id]|
          2 if overlap >= pseudo_overlap_high
          1 if overlap >= pseudo_overlap_low
          0 otherwise

        Config (under cfg.params.cluster_labeling):
          pseudo_overlap_low: int (default: 1)
          pseudo_overlap_high: int (default: 2)

    Notes:
    - We keep using `LTRInstance` for compatibility with existing SVM tooling. Here, `docid`
      is a stable cluster identifier string (e.g., "cl_17").
    """
    params = config.params or {}
    enabled = params.get("enabled_features", [])
    label_rel_threshold = int(params.get("label_rel_threshold", 1))
    max_pairwise = int(params.get("feature_max_pairwise_sim", 2000))
    similarity_cfg = params.get("clustering", {}) or {}

    feature_names = get_enabled_feature_names(enabled)
    instances: List[LTRInstance] = []

    q_by_id = {q.id: q for q in queries}
    for topic_id in sorted(q_by_id.keys()):
        q = q_by_id[topic_id]
        docs = list(doc_candidates_by_topic.get(topic_id, []))
        ranked_passages = list(ranked_passages_by_topic.get(topic_id, []))
        clusters = list(clusters_by_topic.get(topic_id, []))

        rank_docs = list((doc_rank_candidates_by_topic or {}).get(topic_id, docs))
        # Ensure every doc we might reference (via passages) has a rank signal.
        seen = {d.id for d in rank_docs}
        for d in docs:
            if d.id not in seen:
                rank_docs.append(d)
                seen.add(d.id)
        doc_rank = _rank_map_docs(rank_docs)
        passage_rank = _rank_map_passages(ranked_passages)

        qrels_topic = qrels.get(topic_id, {})

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

            docs_in_cluster = {
                ranked_passages[i].document_id
                for i in cl.passage_indices
                if 0 <= i < len(ranked_passages)
            }
            # --- Cluster labeling (training supervision) ---
            label = 0
            if qrels_topic and cl.passage_indices:
                labeling_cfg = params.get("cluster_labeling", {}) or {}
                method = str(labeling_cfg.get("method", "any_relevant_doc")).lower()

                if method in ("any_relevant_doc", "any", "binary"):
                    # Old behavior: if cluster touches any relevant doc -> positive.
                    if docs_in_cluster:
                        for docid in docs_in_cluster:
                            rel = int(qrels_topic.get(docid, 0))
                            if rel >= label_rel_threshold:
                                label = 1
                                break

                elif method in ("top_weighted_density", "weighted_density", "top_weighted"):
                    rr_k = int(labeling_cfg.get("rr_k", 0))
                    if rr_k < 0:
                        rr_k = 0
                    thr_low = float(labeling_cfg.get("threshold_low", 0.10))
                    thr_high = float(labeling_cfg.get("threshold_high", 0.30))
                    if thr_low < 0.0:
                        thr_low = 0.0
                    if thr_high < thr_low:
                        thr_high = thr_low

                    num = 0.0
                    den = 0.0
                    for pi in cl.passage_indices:
                        if not (0 <= pi < len(ranked_passages)):
                            continue
                        p = ranked_passages[pi]
                        # passage_rank should map every passage, but fall back to list position if needed.
                        r = int(passage_rank.get((p.document_id, p.index), pi + 1))
                        if r <= 0:
                            continue
                        w = 1.0 / float(rr_k + r)
                        den += w
                        rel = int(qrels_topic.get(p.document_id, 0))
                        if rel >= label_rel_threshold:
                            num += w

                    density = (num / den) if den > 0.0 else 0.0
                    if density >= thr_high:
                        label = 2
                    elif density >= thr_low:
                        label = 1
                    else:
                        label = 0

                elif method in ("best_evidence", "best", "max_evidence"):
                    best_rank_low = int(labeling_cfg.get("best_rank_low", 100))
                    best_rank_high = int(labeling_cfg.get("best_rank_high", 20))
                    if best_rank_low < 1:
                        best_rank_low = 1
                    if best_rank_high < 1:
                        best_rank_high = 1
                    # Ensure "high" is the stricter (smaller) threshold.
                    if best_rank_high > best_rank_low:
                        best_rank_high = best_rank_low

                    best_rank = None
                    for pi in cl.passage_indices:
                        if not (0 <= pi < len(ranked_passages)):
                            continue
                        p = ranked_passages[pi]
                        rel = int(qrels_topic.get(p.document_id, 0))
                        if rel < label_rel_threshold:
                            continue
                        r = int(passage_rank.get((p.document_id, p.index), pi + 1))
                        if r <= 0:
                            continue
                        if best_rank is None or r < best_rank:
                            best_rank = r

                    if best_rank is None:
                        label = 0
                    elif best_rank <= best_rank_high:
                        label = 2
                    elif best_rank <= best_rank_low:
                        label = 1
                    else:
                        label = 0

                elif method in ("pseudo_passage_overlap", "pseudo_overlap", "pseudo"):
                    pseudo = (pseudo_relevant_passages_by_topic or {}).get(topic_id, set())
                    low = int(labeling_cfg.get("pseudo_overlap_low", 1))
                    high = int(labeling_cfg.get("pseudo_overlap_high", 2))
                    if low < 1:
                        low = 1
                    if high < low:
                        high = low

                    overlap = 0
                    if pseudo:
                        for pi in cl.passage_indices:
                            if not (0 <= pi < len(ranked_passages)):
                                continue
                            p = ranked_passages[pi]
                            if (p.document_id, int(p.index)) in pseudo:
                                overlap += 1

                    if overlap >= high:
                        label = 2
                    elif overlap >= low:
                        label = 1
                    else:
                        label = 0

                else:
                    raise ValueError(
                        f"Unknown cluster_labeling.method: {method!r} "
                        f"(use 'any_relevant_doc', 'top_weighted_density', 'best_evidence', or 'pseudo_passage_overlap')"
                    )

            instances.append(
                LTRInstance(
                    topic_id=topic_id,
                    docid=f"cl_{cl.id}",
                    label=label,
                    features={n: float(feats.get(n, 0.0)) for n in feature_names},
                )
            )

    return instances, feature_names


def write_ltr_csv(instances: Sequence[LTRInstance], feature_names: Sequence[str], output_path: str) -> None:
    """Write LTR instances to CSV for debugging / offline training."""
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "docid", "label"] + list(feature_names))
        for ins in instances:
            w.writerow([ins.topic_id, ins.docid, ins.label] + [f"{ins.features.get(n, 0.0):.6f}" for n in feature_names])


