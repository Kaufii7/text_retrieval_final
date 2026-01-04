# SVM poor performance
_Exported on 1/3/2026 at 13:09:29 GMT+2 from Cursor (2.2.43)_

---

**User**

Our SVM damages the overall MAP score, and currently passage reranking is more accurate based on the simple scoring of clusters (without SVM).

What is the reason for the poor performance of the SVM?

---

**Cursor**

Got it. I’m going to (a) locate exactly how the SVM (and SVMRank) training data/labels/features are built, (b) check for train/test leakage or query/doc mismatch, and (c) compare that setup to the “simple cluster scoring” baseline to pinpoint why the SVM would hurt MAP.

Nice—there’s a dedicated `rag/clustpsg/svm_rank.py` plus `rag/clustpsg/model.py` that writes `svmrank_train.dat/test.dat` and reads `svmrank_pred.txt`. Next I’ll inspect how labels/features are constructed and whether the train/test split and query-grouping are correct—those are the most common reasons RankSVM tanks MAP.

Aha: your SVMRank path writes **raw, unnormalized numeric features** (unlike the `LinearSVC` path which standardizes). Next I’ll inspect how `LTRInstance.label` is created (qrels → doc/cluster labels) and what the feature ranges look like—those two together are the usual culprits when MAP drops.

### Why the SVM (SVMRank) hurts MAP in this pipeline

In your current implementation, the SVM is very likely learning a **misleading / noisy objective** from **mismatched training signals**, so its cluster scores are worse than the simple “seed rank” heuristic.

## 1) **Train–test distribution mismatch (big one)**
When `train_docs_source == "qrels"`, training passages/clusters are built **only from judged qrels docids** (a much smaller + very different candidate pool than inference):

```272:339:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/pipeline.py
if train_docs_source == "qrels":
    # Use judged documents only (both relevant and non-relevant).
    ...
    train_docs_by_topic[topic_id] = _docs_from_qrels(...)
```

But inference passages/clusters are built from **retrieved candidates**:

```341:397:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/pipeline.py
log.info("Stage(inference): retrieving doc candidates ...")
doc_candidates_by_topic = retrieve_doc_candidates(...)
...
ranked_passages_by_topic = rank_passages(... stage="inference")
```

This matters because several features depend on **relative ranks** (reciprocal rank of passage/doc rank). Those ranks are *only meaningful within the candidate set they were computed on*, so the model learns from one rank-distribution and is applied to another.

Example rank-based features:

```73:90:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/features.py
@register_feature("qdep_rr_cluster_passages_max")
def feat_rr_cluster_passages_max(ctx):
    ...
    r = ctx.passage_rank.get((p.document_id, p.index))

@register_feature("qdep_rr_cluster_docs_max")
def feat_rr_cluster_docs_max(ctx):
    ...
    r = ctx.document_rank.get(p.document_id)
```

So the SVM is effectively trained on a “qrels-only world” and deployed on a “retrieval world”.

## 2) **Very noisy cluster labels (weak supervision)**
Cluster labels are derived from **doc-level qrels**, but applied to **clusters** as:

> cluster label = 1 if it contains *any* passage from *any* relevant document

```190:275:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/dataset.py
# Labels:
#   cluster label = 1 iff the cluster contains at least one passage whose source document
#   is relevant in qrels ...
...
docs_in_cluster = { ranked_passages[i].document_id ... }
label = 0
for docid in docs_in_cluster:
    rel = int(qrels_topic.get(docid, 0))
    if rel >= label_rel_threshold:
        label = 1
        break
```

That creates **false-positive clusters** (clusters that “touch” a relevant doc but aren’t actually good evidence), and often **many positive clusters per query**, which gives SVMRank weak/ambiguous pairwise constraints—so it can learn to rank clusters in a way that doesn’t translate to better *document* MAP.

## 3) **No feature scaling for SVMRank**
Your `LinearSVC` path standardizes features, but `SVMrank` uses raw magnitudes:

```74:84:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/model.py
("scaler", StandardScaler(...)),
("svm", LinearSVC(...)),
```

Whereas SVMRank writes raw feature values directly:

```25:48:/Users/kaufi/Downloads/text_retrieval_final/rag/clustpsg/svm_rank.py
for idx, name in enumerate(feature_names, start=1):
    v = float(ins.features.get(name, 0.0))
    ...
    feats.append(f"{idx}:{v:.6f}")
```

If one feature has a larger numeric range (e.g., entropy / unique_docs) it can dominate the learned ranking and swamp the useful RR-based signals.

## 4) **Objective mismatch: you train “rank clusters”, but evaluate MAP on documents**
Even if cluster ranking were “locally reasonable,” your evaluation is document MAP after multiple nonlinear steps (cluster→passage rerank→RR aggregation→fusion). Optimizing a binary cluster label with SVMRank is not well-aligned with that final metric, so it’s easy for the model to overfit the proxy and degrade MAP.

---

### Most likely root cause (if you want one sentence)
**You train the SVM on clusters built from a different candidate distribution (often qrels-only) with noisy cluster labels, using rank-based features whose meaning changes between train and inference, and without feature scaling—so the learned weights don’t generalize and hurt MAP.**

```python
if train_docs_source == "qrels":
    # Use judged documents only (both relevant and non-relevant).
    ...
    train_docs_by_topic[topic_id] = _docs_from_qrels(...)
```

```python
log.info("Stage(inference): retrieving doc candidates ...")
doc_candidates_by_topic = retrieve_doc_candidates(...)
...
ranked_passages_by_topic = rank_passages(... stage="inference")
```

```python
@register_feature("qdep_rr_cluster_passages_max")
def feat_rr_cluster_passages_max(ctx):
    ...
    r = ctx.passage_rank.get((p.document_id, p.index))

@register_feature("qdep_rr_cluster_docs_max")
def feat_rr_cluster_docs_max(ctx):
    ...
    r = ctx.document_rank.get(p.document_id)
```

```python
# Labels:
#   cluster label = 1 iff the cluster contains at least one passage whose source document
#   is relevant in qrels ...
...
docs_in_cluster = { ranked_passages[i].document_id ... }
label = 0
for docid in docs_in_cluster:
    rel = int(qrels_topic.get(docid, 0))
    if rel >= label_rel_threshold:
        label = 1
        break
```

```python
("scaler", StandardScaler(...)),
("svm", LinearSVC(...)),
```

```python
for idx, name in enumerate(feature_names, start=1):
    v = float(ins.features.get(name, 0.0))
    ...
    feats.append(f"{idx}:{v:.6f}")
```

