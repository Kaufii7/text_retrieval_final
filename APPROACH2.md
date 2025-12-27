## Approach #2 — `clustpsg` (plan split into committable PRs)

Goal: **re-rank documents** by extracting and clustering passages, then learning feature weights with an **SVM** using the provided **binary document relevance labels** (qrels for first 50 queries).

Design principles:
- Each PR is **merge-safe** and usable on its own.
- PRs add **library code first**, then wire it into `main.py` behind a flag.
- Determinism: fix seeds, stable sorting, stable feature extraction order.

---

### PR A — `clustpsg` scaffolding (no behavior change)

- **Goal**: define the approach interface + config surface without changing current runs.
- **Changes**
  - Update `rag/approaches/approach2.py` to be the **`clustpsg`** approach module (template/skeleton).
  - Add/extend `rag/config.py` placeholders for `clustpsg` params (if needed).
  - Add `CLUSTPSG.md` (optional) with high-level architecture diagram / notes.
- **Acceptance**
  - Imports succeed; no changes to default `--approach bm25`.

---

### PR 1 — Retrieve and rank documents (LuceneSearcher baseline for clustpsg)

- **Goal**: produce a **candidate document ranking** for each query using `LuceneSearcher` with:
  - **BM25**, and
  - **QLD (query likelihood w/ Dirichlet smoothing)** with **`mu=1000`**
- **Changes**
  - Add a helper that returns top-`doc_topk` document candidates per query (e.g., 1000).
  - Add a config switch for doc retrieval model: `bm25` vs `qld(mu=1000)` (and optionally both for ablations).
  - Store per-query candidates in a normalized structure (topic_id → list of `{docid, score, rank}`).
- **Acceptance**
  - Given a query set, you can produce deterministic doc candidates identical to Approach1 settings.
  - (Optional) Write candidates to a debug file for inspection (flag-gated).

---

### PR 2 — Passage extraction (sliding window over sentences)

- **Goal**: from each candidate document, extract **passages** via sentence-based sliding window.
- **Changes**
  - Base the algorithm on the existing implementation in
    `q-a-rag-assignment-3-reichman-uni/rag_system/passages.py`:
    - sentence split via regex: `re.split(r"(?<=[.!?])\\s+", text)`
    - strip empty sentences
    - cap sentence length with `max_chars_per_sentence` (truncate and add `...`)
    - build overlapping windows using:
      - `min_sentences`
      - `max_sentences`
      - `stride_sentences` (overlap size)
  - Define a `Passage` object representation:
    - `document_id`, `index` (passage number), `content`, `score` (optional)
  - Add unit-like sanity checks on a few sample documents (even if lightweight).
- **Acceptance**
  - For a document text input, passage extraction yields deterministic passage boundaries/counts.
  - Empty/short docs handled gracefully.

---

### PR 3 — Passage retrieval / scoring (query → passages)

- **Goal**: assign **query-dependent scores/ranks** to passages so we can build clustering features later.
- **Changes**
  - Use a `LuceneSearcher` over a **passage index** and support:
    - **BM25**, and
    - **QLD with `mu=1000`**
  - Create/maintain a passage index (one-time build) from extracted passages so passage retrieval is efficient.
  - Produce per-query ranked passage list and (optionally) per-doc passage ranks.
- **Acceptance**
  - For a query, you can list top passages with stable ranks/scores.

---

### PR 4 — Cluster passages by similarity

- **Goal**: cluster passages into topical groups per query (or per doc → then merged).
- **Changes**
  - Choose similarity + clustering:
    - TF-IDF cosine + agglomerative / k-means, OR
    - embeddings + k-means / HDBSCAN (if allowed)
  - Define cluster container:
    - cluster_id → list of passages (or passage ids)
  - Add parameters: `n_clusters` or threshold, distance metric, random seed.
- **Acceptance**
  - Clustering produces stable cluster assignments given fixed seed.
  - Each passage belongs to exactly one cluster.

---

### PR 5 — Feature computation (cluster → feature vector)

- **Goal**: convert each cluster into a vector of features for learning-to-rank.
- **Features**
  - **Query-dependent features**
    - **reciprocal rank of cluster’s passages**
      - a cluster is more important if its passages are highly ranked by the passage scorer
    - **reciprocal rank of the passages’ source documents**
      - a cluster is stronger if it draws from highly ranked documents
  - **Cluster priors**
    - **entropy of term distributions**
      - coherent clusters should have lower entropy (penalize generic clusters)
    - **stopword ratio**
      - clusters dominated by stopwords are less informative
    - **inter-passage similarity**
      - strong clusters contain mutually supportive passages
    - **# unique documents contributing passages**
      - evidence from multiple docs is more reliable
- **Changes**
  - Add `rag/clustpsg/features.py`:
    - `compute_cluster_features(query, cluster, context) -> Dict[str, float]`
  - Make feature addition easy:
    - implement a **feature registry**: `FEATURES: Dict[str, FeatureFn]`
    - config selects enabled features by name
    - output is a stable, sorted feature vector schema for training/inference
- **Acceptance**
  - Feature extraction is deterministic and produces the same feature keys across queries.
  - Feature vectors can be serialized to CSV/NPZ for debugging.

---

### PR 6 — Label generation (clusters/documents) from qrels

- **Goal**: define supervised labels from the provided **doc-level binary relevance**.
- Changes (choose one training target; document-level is simplest):
  - **Document-level LTR**: learn a score per document using cluster-derived aggregated features, label = doc relevance.
  - (Optional) Cluster-level proxy labels derived from relevant docs contributing to the cluster.
- **Acceptance**
  - You can produce `(X, y)` for the first 50 topics reproducibly.

---

### PR 7 — “Learning to rank”: SVM training + inference

- **Goal**: train an SVM to predict a score used to rerank documents (or clusters → then docs).
- **Changes**
  - Add `rag/clustpsg/model.py`:
    - training: fit SVM on `(X_train, y_train)` (with scaling if needed)
    - inference: score new instances deterministically
    - persistence: save/load model artifacts (e.g., `models/clustpsg_svm.pkl`)
  - Decide SVM flavor:
    - linear SVM for interpretability / stability, or
    - kernel SVM if justified (but heavier)
- **Acceptance**
  - Training runs on first 50 topics and produces a saved model.
  - Inference produces scores for test queries with the same feature schema.

---

### PR 8 — End-to-end `clustpsg` wiring in `main.py` (flag-gated)

- **Goal**: generate `run_2.res` using `--approach clustpsg`.
- **Changes**
  - Add `--approach clustpsg` to `main.py` (flag-gated).
  - `train` mode: optionally train the model and report MAP (or just evaluate a frozen model).
  - `test` mode: load model and produce final run.
- **Acceptance**
  - `main.py --approach clustpsg --split train --output run_2_train.res --run-tag run2 --evaluate` works.
  - Output is valid TREC format, deterministic.


