## PR Plan — Approach 3 (Two-stage retrieval: Bi-encoder recall → Cross-Encoder re-ranking)

This document decomposes **Approach 3** into small, mergeable PRs. The target is a **two-stage retrieval pipeline**:

- **Stage 1 (recall)**: pretrained **bi-encoder** retrieval (dense embeddings + ANN index)
- **Stage 2 (precision)**: cross-encoder scores `(query, doc_text)` for top-N candidates, then reranks

### Constraints / design principles

- **Determinism**: stable output for the same inputs (stable sorting and tie-breaks; fixed seeds where relevant).
- **Performance**: cross-encoder runs only on top-N (e.g., 50–300) per query.
- **Merge safety**: heavy ML dependencies should be optional and isolated; errors should be clear and actionable when a component is explicitly enabled.
- **Compatibility**: output must match `rag.runs.write_trec_run` expectations (topic → list of `{docid, score}` or `(docid, score)`).

---

### PR A3-1 — Dense indexing foundations (corpus extraction + embedding cache)

**Goal**
- Create the *assets* needed for bi-encoder retrieval: a local/cached corpus representation and document embeddings.

**Changes**
- Add a small data builder that:
  - iterates all docids in the `robust04` index
  - fetches best-effort document text (prefer `contents()`, then `raw()` with JSON `"contents"` extraction)
  - writes a deterministic corpus file (e.g., JSONL: `{docid, text}`)
- Add an embedding builder that:
  - loads a pretrained bi-encoder (e.g., `sentence-transformers/all-mpnet-base-v2` or `multi-qa-mpnet-base-dot-v1`)
  - computes doc embeddings in batches
  - stores embeddings + docid mapping on disk (cache) for reuse
- Keep this PR as a **builder-only** vertical slice (no retrieval yet).

**Acceptance**
- One command produces corpus + embeddings cache artifacts deterministically (same docid order, stable outputs).
- Artifacts can be reused without recomputation (idempotent build).

**Rollback safety**
- Purely additive; does not change existing approaches.

---

### PR A3-2 — Centralize doc-text fetching (shared utility)

**Goal**
- Provide a single reusable function that fetches the best-effort raw text for a given `docid` from Lucene.

**Rationale**
- `rag/clustpsg/pipeline.py` already uses a robust pattern via `searcher.doc(docid)` and prefers `contents()` over `raw()`.
- Approach 3 reranking needs the same capability for `(query, doc_text)` scoring.

**Changes**
- Add a helper such as `rag.lucene_backend.fetch_doc_contents(searcher, docid) -> str`
  - try `doc.contents()` if available
  - else fall back to `doc.raw()` and extract `"contents"` if JSON
- Update call sites that need doc text (Approach 3 later; optionally refactor clustpsg to reuse it).

**Acceptance**
- For a sample of docids from the `robust04` index, `fetch_doc_contents` returns non-empty text for most docs.
- No behavior change unless a caller opts into using the new helper.

---

### PR A3-3 — ANN index + bi-encoder retrieval (Stage 1)

**Goal**
- Implement the actual **bi-encoder recall** stage: query embedding + ANN search over precomputed doc embeddings.

**Changes**
- Add an ANN backend (choose one and document it):
  - FAISS (fast; extra dep), or
  - HNSWlib (simpler; extra dep), or
  - a lightweight fallback (exact dot-product) for debugging only (not for final runs).
- Implement:
  - `build_index(embeddings) -> index`
  - `retrieve(query_embedding, topk) -> docids + scores`
- Add caching for the ANN index on disk.

**Acceptance**
- Given a query, Stage 1 returns top-K docids quickly using the cached index (no full rescoring pass).
- Results are deterministic (given fixed artifacts and model).

---

### PR A3-4 — Add cross-encoder reranker module (Stage 2; inference only; deps gated)

**Goal**
- Add a thin reranker wrapper that can score `(query, doc_text)` pairs and return reranked results.

**Changes**
- Add `rag/rerankers/cross_encoder.py`:
  - `CrossEncoderReranker(model_name, device, batch_size)` (wrapper around a cross-encoder)
  - `score_pairs(query, docs) -> list[float]` and/or `rerank(query, candidates) -> ranked`
- Dependency gating:
  - Only import heavy deps when reranking is enabled.
  - If deps aren’t available and rerank is enabled, raise a clear actionable error.

**Acceptance**
- With deps installed, reranking works on a small N (e.g., 10) for one query.
- Without deps installed, the repo still runs for approaches that don’t enable reranking.

---

### PR A3-5 — Integrate full Approach 3 (Bi-encoder Stage 1 → CE rerank top-N → output topk)

**Goal**
- Implement the full two-stage pipeline inside `approach3_retrieve`.

**Changes**
- Extend `default_approach3_config()` in `rag/config.py` with params:
  - **Bi-encoder**: `model_name`, `normalize_embeddings`, `batch_size`, `device`
  - **ANN**: backend + index params + paths to cached artifacts
  - **Candidates**: `candidates_depth` (e.g., 2000–5000)
  - **Rerank**: `enabled`, `model_name`, `topn`, `batch_size`, `device`
- Scoring policy (choose and document one):
  - **Simple**: rank by cross-encoder score within top-N; keep tail by bi-encoder score
  - **Safer**: blended score: `final = alpha * ce + (1 - alpha) * norm(bi_encoder)` for reranked docs
- Ensure stable ordering and tie-breaks (score desc, docid asc).

**Acceptance**
- Approach 3 runs end-to-end to produce a TREC run file for `--split train|test|all`.
- Stage 1 uses the ANN index (not Lucene BM25).
- Runtime is reasonable (topn small); output is deterministic.

---

### PR A3-6 — Tuning driver for Approach 3 (train MAP loop + structured logging)

**Goal**
- Make parameter/model selection reproducible on the first 50 topics.

**Changes**
- Add a small experiment driver (module or script), e.g. `rag/experiments/approach3_grid.py`, that:
  - runs Approach 3 on split=train
  - evaluates MAP using existing `rag.eval.mean_average_precision`
  - writes results to CSV/JSON (config → MAP)
- Log key knobs per run (model name, candidates_depth, topn, alpha, device).

**Acceptance**
- One command produces a table of configurations and MAP values and prints the best config.

---

### PR A3-7 (Optional) — Light fine-tuning of the cross-encoder on 50 labeled topics (CV + early stopping)

**Goal**
- Optional performance boost via light domain adaptation with minimal overfitting risk.

**Changes**
- Build training pairs from qrels:
  - **Positives**: (q, doc) where relevance ≥ 1
  - **Negatives**: sampled from bi-encoder candidates not labeled relevant for that topic
- Training regimen:
  - topic-level K-fold CV on the 50 topics (e.g., 5-fold)
  - early stopping
  - save best checkpoint under `models/approach3_ce/`
- Add a config flag to select pretrained vs fine-tuned model.

**Acceptance**
- Training runs end-to-end (optional path) and produces checkpoints.
- CV summary (mean/variance) is reported.

---

### PR A3-8 — Final run packaging + reproducibility notes

**Goal**
- Generate `run_3.res` reliably and document the exact settings used.

**Changes**
- Update `scripts/make_runs.sh` to include Approach 3 run generation.
- Update `REPRODUCIBILITY.md` with:
  - exact command lines
  - chosen config values (bi-encoder model name, ANN backend/params, candidates_depth, topn, alpha, cross-encoder model name)
  - environment notes (CPU/GPU, batch size)

**Acceptance**
- One command produces `run_1.res`, `run_2.res`, `run_3.res` and the final zip (if used).
- Approach 3 final configuration is explicit and reproducible.

---

### Suggested starting defaults (initial bi-encoder run)

- Bi-encoder model: `sentence-transformers/all-mpnet-base-v2` (strong default) or `multi-qa-mpnet-base-dot-v1` (retrieval-optimized)
- `candidates_depth`: 2000
- `rerank.topn`: 100
- `rerank.model_name`: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `score_fusion.alpha`: 0.8 (only if blending is implemented)

### Risk controls

- Keep all large artifacts cached (corpus JSONL, doc embeddings, ANN index).
- Keep reranking optional and confined to top-N to avoid runtime blowups.
- Use stable sorting and deterministic tie-breaks.


