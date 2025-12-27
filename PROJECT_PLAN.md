## Project Plan — Text Retrieval Challenge (ROBUST04 / Pyserini)

Source requirements: [instructions.pdf](file:///Users/kaufi/Downloads/text_retrieval_final/instructions.pdf)

### Goal & deliverables

- **Goal**: maximize **MAP** on the (hidden) 199 test queries.
- **Train/validation data**: use **qrels for first 50 topics** for parameter tuning and model selection.
- **Outputs**: produce **3 run files** in standard **6-column TREC format**, each with **top 1000 docs per query**:

```txt
630 Q0 ZF08-175-870 1 0.7 run1
630 Q0 ZF08-306-044 2 0.5 run1
...
```

- **Submission**: zip the 3 run files: `run_1.res`, `run_2.res`, `run_3.res`.

### Data inputs (in this repo)

- **Queries**: `queriesROBUST.txt`
  - Expected format per line: `topic_id<TAB>query_text`
  - Example: `301    international organized crime`
- **Relevance judgments (train only)**: `qrels_50_Queries`
  - Expected columns: `topic_id 0 doc_id relevance`

### Index / retrieval backend

- Use the **prebuilt Pyserini index** `robust04`:
  - `IndexReader.from_prebuilt_index('robust04')`
  - `LuceneSearcher.from_prebuilt_index('robust04')`
- Note (from instructions): index built with **Porter stemming** and **no stopword removal**.

---

## Three retrieval approaches

### Approach 1 — BM25 baseline (strong lexical)

**What**: plain BM25 retrieval with parameter tuning.

- **Retriever**: `LuceneSearcher` BM25
- **Hyperparameters to tune on the first 50 topics**:
  - **k1** (e.g., 0.6–1.8)
  - **b** (e.g., 0.1–0.9)
- **Output**: `run_1.res`

**Why**: provides a robust baseline and a control run for comparison.

---

### Approach 2 — TBD (placeholder)

**What**: **TBD**. This run is reserved for an additional method beyond the BM25 baseline.

- **Implementation skeleton**:
  - **Inputs**: `(queries, lucene_searcher, topk, config)`
  - **Outputs**: a per-query ranked list of up to `topk` docids + scores, written as `run_2.res`
  - **Constraints**: deterministic / reproducible; scores must be non-increasing per query
- **Config placeholder** (`rag/config.py`):
  - `approach2.name` (string)
  - `approach2.params` (dict; TBD)
  - `approach2.candidates_depth` (int; optional)
- **Evaluation plan**:
  - Use first 50 topics to compare MAP vs Approach 1, once the method is selected.

---

### Approach 3 — TBD (placeholder; “advanced” slot)

**What**: **TBD**. This run is reserved for the “advanced / beyond-class” method.

- **Implementation skeleton**:
  - **Inputs**: `(queries, lucene_searcher, topk, config)`
  - **Outputs**: a per-query ranked list of up to `topk` docids + scores, written as `run_3.res`
  - **May include** (all TBD): multi-stage retrieval, fusion of multiple candidate lists, reranking
  - **Constraints**: deterministic / reproducible; scores must be non-increasing per query
- **Config placeholder** (`rag/config.py`):
  - `approach3.name` (string)
  - `approach3.params` (dict; TBD)
  - `approach3.candidates_depth` (int; optional)
  - `approach3.fusion` (dict; optional; TBD)

---

## Repository structure (proposed)

Keep `main.py` as the entrypoint; place approach implementations and utilities under `rag/`.

```txt
text_retrieval_final/
  main.py
  PROJECT_PLAN.md
  queriesROBUST.txt
  qrels_50_Queries
  rag/
    __init__.py
    types.py
    config.py
    io.py
    logging_utils.py
    lucene_backend.py
    eval.py
    runs.py
    approaches/
      __init__.py
      bm25.py
      approach2_tbd.py
      approach3_tbd.py
```

---

## `main.py` (entrypoint) requirements

### CLI behavior

- Select **approach**: `bm25`, `approach2`, `approach3`
- Select **split**:
  - `train`: first 50 topics only (for tuning + evaluation)
  - `test`: remaining topics (for producing official runs)
- Control output:
  - `--output run_1.res` (etc.)
  - `--run-tag run1` (printed in column 6)
- Common knobs:
  - `--topk 1000` (always 1000 for final outputs)
  - `--seed` (for reproducibility where relevant)
  - `--log-level`

### Expected flow

- Load queries → optionally split train/test → init search backend → run selected approach → write TREC run file.

---

## Shared utilities (core requirements)

### Core dataclasses (`rag/types.py`)

- Implement a small set of **generic, reusable dataclasses** used across all approaches:
  - `Query(id, content)`
  - `Document(id, content, score=None)`
  - `Passage(document_id, index, content, score=None)`
- **Goal**: standardize data interchange between modules (retrieval, reranking, fusion, evaluation) and reduce ad-hoc dict usage.
- **Guidelines**:
  - Keep them lightweight and immutable (`frozen=True`) where possible.
  - Avoid backend-specific dependencies (no Pyserini types here).
  - Make fields explicit and typed; prefer `Optional[float]` for scores.

### Logging (`rag/logging_utils.py`)

- Configure console logging with timestamps, log levels, and module name.
- Support `--log-level` and consistent formatting across scripts.

### CSV/TSV I/O (`rag/io.py`)

- **Load queries** from `queriesROBUST.txt`:
  - parse `topic_id` (int) and `query_text` (str)
- **Optionally write** train/test splits to CSV for experiment tracking:
  - `data/queries_train.csv`, `data/queries_test.csv` (optional)
- **Read qrels** from `qrels_50_Queries` for evaluation.

### Lucene index/searcher init (`rag/lucene_backend.py`)

- Centralize initialization of:
  - `IndexReader.from_prebuilt_index('robust04')`
  - `LuceneSearcher.from_prebuilt_index('robust04')`
- Provide helpers:
  - set BM25 parameters
  - execute search with `topk`
  - return a normalized internal result object: `{docid, score, rank}`

### Run file writing (`rag/runs.py`)

- Write **TREC run** format exactly:
  - `topic_id Q0 docid rank score run_tag`
- Enforce:
  - topK=1000
  - scores are non-increasing per query (sort by score desc, then docid as tie-breaker)

### Evaluation (`rag/eval.py`)

- Compute MAP on train topics using `qrels_50_Queries`.
- If available, use `trec_eval` or Pyserini evaluation utilities; otherwise implement MAP@1000 directly.
- Output:
  - MAP summary
  - per-topic AP (optional)

---

## Experiment plan (how to proceed)

### Phase 1 — Baseline sanity check

- Implement **core dataclasses** (`rag/types.py`) + query loading + Lucene searcher + TREC run writer.
- Generate a BM25 run for:
  - first 50 queries (train) → evaluate MAP
  - all 249 queries (full) → verify output formatting and top-1000 completeness

---

## Additional PR — Shared dataclasses for reuse across approaches

**What**: add `rag/types.py` implementing reusable dataclasses like `Document`, `Passage`, `Query` for consistent internal representations.

- **Why**: Approach 2/3 will likely involve multi-stage retrieval (candidates → rerank) and/or fusion, which benefits from shared, typed structures.
- **Acceptance**:
  - The dataclasses are imported and used by utilities/approaches instead of ad-hoc dicts where practical.
  - No approach behavior changes required; this is a refactor / foundation PR.

### Phase 2 — Approach 2 (TBD)

- Grid search BM25 (k1, b) on the first 50 topics.
- Implement Approach 2 once selected; tune its parameters on the first 50 topics.
- Keep a small results table (CSV) of params → MAP for reproducibility (once Approach 2 exists).

### Phase 3 — Approach 3 (TBD; “advanced”)

- Implement Approach 3 once selected; tune its parameters on the first 50 topics.
- Produce `run_3.res` for test topics once chosen.

---

## Reproducibility checklist

- Fix random seeds where relevant.
- Save:
  - chosen parameters for each run
  - software versions / environment notes (optional `requirements.txt`)
  - command lines used to generate the final `run_1.res`, `run_2.res`, `run_3.res`


