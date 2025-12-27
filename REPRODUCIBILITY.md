## Reproducibility

This file records the exact commands and parameters used to generate the final submission artifacts.

### Environment

- **Python**: use the provided virtualenv: `.venv/bin/python` (Python 3.10.19)
- **Retrieval backend**: Pyserini prebuilt index `robust04`
- **Output format**: TREC 6-column run files, top 1000 docs/query

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
- **Run 3 (BM25)**: `RUN3_K1=1.5`, `RUN3_B=0.4`

Example override:

```bash
SPLIT=test TOPK=1000 RUN1_K1=0.9 RUN1_B=0.4 bash scripts/make_runs.sh
```

### Training evaluation (MAP on first 50 topics)

If you generate a train run, evaluate with:

```bash
.venv/bin/python -m rag.eval --run run_1_train.res --qrels qrels_50_Queries --k 1000
```


