## PR Plan — Split `PROJECT_PLAN.md` into small, mergeable PRs

This document decomposes `PROJECT_PLAN.md` into **small, committable PRs**. Each PR is designed to be **independent (merge-safe on its own)**:

- No PR should require unmerged code from another PR to keep the repo usable.
- Later PRs may *enhance* functionality, but earlier PRs must not be prerequisites for correctness of already-merged behavior.
- If a PR adds a new module/feature, it must either:
  - be fully wired and working end-to-end, or
  - be clearly **optional** and not affect existing behavior.

---

### PR 1 — Repo skeleton + “how to run” docs (no functionality change)

- **Goal**: establish a stable project layout and documentation without changing behavior.
- **Changes**
  - Add `rag/` package skeleton and placeholder module docstrings.
  - Add `README.md` describing the expected workflow (train/test split, output run format, topk=1000).
  - Add `.gitignore` (Python caches, venv, local artifacts).
- **Files (suggested)**
  - `rag/__init__.py`
  - `rag/approaches/__init__.py`
  - `README.md`
  - `.gitignore`
- **Acceptance**
  - `python main.py` still runs (even if it does nothing meaningful yet).
  - Importing `rag` works: `python -c "import rag"`.

---

### PR 2 — Query + qrels I/O utilities (pure library; not wired to `main.py`)

- **Goal**: add robust parsing utilities with zero impact on runtime until used.
- **Changes**
  - Implement `rag/io.py`:
    - `load_queries(path) -> list[(topic_id, query_text)]` for `queriesROBUST.txt` (`topic_id<TAB>query_text`)
    - `load_qrels(path) -> dict[topic_id, dict[doc_id, rel]]` for `qrels_50_Queries`
  - Include basic validation (empty lines, bad ints) and deterministic ordering.
- **Files**
  - `rag/io.py`
- **Acceptance**
  - `python -c "from rag.io import load_queries; print(len(load_queries('queriesROBUST.txt')))"` works.
  - `python -c "from rag.io import load_qrels; q=load_qrels('qrels_50_Queries'); print(len(q))"` works.

---

### PR 3 — TREC run writer utility (pure library; not wired to `main.py`)

- **Goal**: generate correct 6-column TREC output deterministically.
- **Changes**
  - Implement `rag/runs.py`:
    - `write_trec_run(results_by_topic, output_path, run_tag, topk=1000)`
    - Enforce sorting: score desc, docid asc tie-break, rank starts at 1.
    - Ensure **≤ topk** lines per topic; stable output; newline at EOF.
- **Files**
  - `rag/runs.py`
- **Acceptance**
  - A tiny synthetic example writes the expected format:
    - `topic_id Q0 docid rank score run_tag`

---

### PR 4 — Logging utilities (pure library; not wired or optional wiring)

- **Goal**: consistent logging configuration (useful across scripts).
- **Changes**
  - Implement `rag/logging_utils.py`:
    - `configure_logging(level: str) -> None`
    - consistent format with timestamps / level / module
- **Files**
  - `rag/logging_utils.py`
- **Acceptance**
  - `python -c "from rag.logging_utils import configure_logging; configure_logging('INFO')"` works.

---

### PR 5 — Lucene backend wrapper (pure library; optional usage)

- **Goal**: centralize Pyserini initialization and searching behind one interface.
- **Changes**
  - Implement `rag/lucene_backend.py`:
    - `get_searcher(index_name='robust04')`
    - `set_bm25(searcher, k1, b)`
    - `search(searcher, query, topk) -> list[{docid, score, rank}]`
  - Keep it import-safe: if Pyserini is missing, raise a clear error message only when used.
- **Files**
  - `rag/lucene_backend.py`
- **Acceptance**
  - Importing the module does not crash the repo.
  - If Pyserini is installed: a simple search returns results.

---

### PR 6 — End-to-end BM25 run generation via `main.py` (first real vertical slice)

- **Goal**: one working approach end-to-end: load queries → search → write run.
- **Changes**
  - Expand `main.py` to support:
    - `--approach bm25`
    - `--split train|test|all` (train = first 50 topics, test = remaining, all = everything)
    - `--output`, `--run-tag`, `--topk` (default 1000), `--log-level`
    - BM25 params `--k1`, `--b`
  - Implement `rag/approaches/bm25.py` as the approach function used by `main.py`.
  - Ensure deterministic output ordering.
- **Files**
  - `main.py`
  - `rag/approaches/bm25.py`
  - (uses) `rag/io.py`, `rag/lucene_backend.py`, `rag/runs.py`, `rag/logging_utils.py`
- **Acceptance**
  - `python main.py --approach bm25 --split train --output run_1_train.res --run-tag run1`
  - Output file exists and has ≤ 1000 lines per topic; correct 6 columns.

---

### PR 7 — Evaluation (MAP on first 50 topics) as a standalone tool

- **Goal**: compute MAP on the training topics without forcing changes to run generation.
- **Changes**
  - Implement `rag/eval.py` with a small, testable API:
    - `mean_average_precision(qrels, run_results, k=1000) -> float`
  - Add an optional CLI tool (either `python -m rag.eval ...` or a `--evaluate` mode in `main.py`)
    - Must not change default behavior unless flag is provided.
- **Files**
  - `rag/eval.py`
  - (optional) `main.py` (flag-gated)
- **Acceptance**
  - You can evaluate an existing run file against `qrels_50_Queries` and print MAP.

---

### PR 8 — BM25 parameter sweep script (experiment driver; optional)

- **Goal**: make tuning reproducible without changing core retriever logic.
- **Changes**
  - Add a small grid-search driver (new script/module) that:
    - runs BM25 on train topics for each `(k1, b)`
    - evaluates MAP
    - writes a CSV summary
  - Keep it optional and separate from `main.py` default path.
- **Files (suggested)**
  - `rag/experiments/bm25_grid.py` (or `scripts/bm25_grid.py`)
  - `rag/experiments/__init__.py`
- **Acceptance**
  - One command produces a CSV table of params → MAP and identifies best config.

---

### PR 9 — Approach 2 (fully working, not a placeholder)

- **Goal**: implement a second, complete method that can generate `run_2.res`.
- **Constraint**: this PR must include everything needed for the approach to run end-to-end (no “TBD”).
- **Changes (examples, pick one)**
  - **Query expansion** (RM3 / pseudo-relevance feedback) using Pyserini utilities, OR
  - **Hybrid retrieval** (BM25 + dense) if allowed by constraints, OR
  - **BM25 variants** (e.g., fielded, doc2query expansions) if you have resources.
  - Add `--approach approach2` with flags gated to that approach.
- **Files**
  - `rag/approaches/approach2_*.py`
  - `main.py` (flag-gated approach selection)
  - `rag/config.py` (optional; only if used)
- **Acceptance**
  - `python main.py --approach approach2 --split train --output run_2_train.res --run-tag run2`
  - MAP can be computed; outputs are deterministic.

---

### PR 10 — Approach 3 (advanced; multi-stage or fusion; fully working)

- **Goal**: implement the “advanced” run as a complete method generating `run_3.res`.
- **Changes (examples)**
  - **Two-stage retrieval**: BM25 candidate generation → rerank with a cross-encoder (if allowed), OR
  - **Fusion**: combine multiple candidate lists (e.g., BM25 variants) with Reciprocal Rank Fusion (RRF), OR
  - **Learning-to-rank** (only if feasible within constraints).
- **Files**
  - `rag/approaches/approach3_*.py`
  - (optional) `rag/fusion.py`
  - `main.py` (new approach selection)
- **Acceptance**
  - `python main.py --approach approach3 --split test --output run_3.res --run-tag run3`
  - Output is valid TREC format and deterministic.

---

### PR 11 — Final “run packaging” + reproducibility checklist

- **Goal**: make final submission creation idiot-proof.
- **Changes**
  - Add a `Makefile` or `scripts/make_runs.sh` that:
    - generates `run_1.res`, `run_2.res`, `run_3.res` (test split, topk=1000)
    - zips them into `submission.zip`
  - Add a short `REPRODUCIBILITY.md` recording exact commands + chosen params.
- **Files (suggested)**
  - `scripts/make_runs.sh` (or `Makefile`)
  - `REPRODUCIBILITY.md`
- **Acceptance**
  - One command produces the three `.res` files and the final zip.


