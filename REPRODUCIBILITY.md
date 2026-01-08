## Reproducibility

This file records the exact commands and parameters used to generate the final submission artifacts.

### Environment

- **Python**: use the provided virtualenv: `.venv/bin/python` (Python 3.10.19)
- **Retrieval backend**: Pyserini prebuilt index `robust04`
- **Output format**: TREC 6-column run files, top 1000 docs/query

### Approach 3 (dense + rerank) artifacts

Approach 3 uses cached dense assets built from the `robust04` index.

Build the corpus + embeddings (A3-1):

```bash
.venv/bin/python -m rag.approach3.build_dense_assets \
  --index robust04 \
  --out-dir cache/approach3_dense \
  --model-name sentence-transformers/all-mpnet-base-v2 \
  --device cpu \
  --batch-size 64
```

Notes:
- If you use `hnswlib` for ANN, the HNSW index will be built on-demand when running Approach 3 (if enabled in config).
- Cross-encoder reranking is optional and controlled via the Approach 3 config JSON.

### Generate the submission artifacts

The submission packaging script generates:
- `run_1.res`
- `run_2.res`
- `run_3.res`
- `submission.zip` (containing the three `.res` files)

Command:

```bash
bash scripts/make_runs.sh
```

### Parameters used (edit to match your final run)

Defaults in `scripts/make_runs.sh` (override via env vars):

- **Split**: `SPLIT=test` (topics after the first 50)
- **TopK**: `TOPK=1000`
- **Run 1 (BM25)**: `RUN1_K1=0.9`, `RUN1_B=0.4`
- **Run 2 (BM25)**: `RUN2_K1=1.2`, `RUN2_B=0.4`
- **Run 3**:
  - BM25 mode (default): `RUN3_APPROACH=bm25` with `RUN3_K1=1.5`, `RUN3_B=0.4`
  - Approach 3 mode: `RUN3_APPROACH=approach3` with `A3_CONFIG=configs/approach3_run3.example.json`

Example override:

```bash
SPLIT=test TOPK=1000 RUN1_K1=0.9 RUN1_B=0.4 bash scripts/make_runs.sh
```

### Generating run_3.res with Approach 3

1) Build dense assets (see above)
2) Ensure your Approach 3 config is set (example provided):
   - `configs/approach3_run3.example.json`
3) Generate submission package with run_3 using Approach 3:

```bash
SPLIT=test TOPK=1000 RUN3_APPROACH=approach3 A3_CONFIG=configs/approach3_run3.example.json bash scripts/make_runs.sh
```

### Optional: fine-tune the cross-encoder (A3-7)

If you fine-tune a cross-encoder, run:

```bash
.venv/bin/python -m rag.approach3.finetune_cross_encoder --out-dir models/approach3_ce
```

Then update your Approach 3 config JSON (e.g. `configs/approach3_run3.example.json`) to:
- `"rerank": { "enabled": true, "use_finetuned": true, "finetuned_model_dir": "models/approach3_ce/fold_0" }`

### Training evaluation (MAP on first 50 topics)

If you generate a train run, evaluate with:

```bash
.venv/bin/python -m rag.eval --run run_1_train.res --qrels qrels_50_Queries --k 1000
```


