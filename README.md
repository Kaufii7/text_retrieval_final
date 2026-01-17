## Text Retrieval Challenge (ROBUST04 / Pyserini)

This repo contains code and notes for generating **TREC-format run files** for the ROBUST04 retrieval task.

### Data in this repo

- **Queries**: `queriesROBUST.txt` (expected format: `topic_id<TAB>query_text`)
- **Train qrels**: `qrels_50_Queries` (expected columns: `topic_id 0 doc_id relevance`)
- **Project notes / plan**: `PROJECT_PLAN.md`

### Code layout

- `main.py`: entrypoint CLI for generating run files
- `rag/`: library code (approaches, I/O, run writing, evaluation, etc.)

### Environment (use the provided virtualenv)

This project expects **Python 3.10.19** via the provided `.venv`.

```bash
.venv/bin/python main.py
```

### Expected workflow

- **Tune on train topics**: use the **first 50 topics** (those with qrels) to tune parameters / models.
- **Generate final runs on test topics**: use the remaining topics to produce the final submission run files.
- **Output requirement**: write **TREC 6-column** run files with **topk=1000** documents per query.

### Quickstart (BM25 baseline)

Generate a BM25 run file:

```bash
.venv/bin/python main.py \
  --approach bm25 \
  --split train \
  --queries queriesROBUST.txt \
  --output run_1_train.res \
  --run-tag run1 \
  --topk 1000 \
  --k1 0.9 \
  --b 0.4
```

Optionally evaluate with MAP@1000 on the training topics (requires qrels):

```bash
.venv/bin/python main.py \
  --approach bm25 \
  --split train \
  --output run_1_train.res \
  --run-tag run1 \
  --evaluate \
  --qrels qrels_50_Queries \
  --eval-k 1000
```

### Tune BM25 with Optuna (no QE, no RM3)

If you want an automated hyperparameter search for **Approach 1 (BM25)**, run:

```bash
.venv/bin/python -m rag.experiments.bm25_optuna \
  --trials 50 \
  --topk 1000 \
  --out-json results/bm25_optuna_best.json
```

This will tune only `(k1, b)` on the **first 50 topics** (those with qrels) and write the best params to JSON,
including a copy-pastable `main.py` command that keeps **query expansion disabled** (`--qe none`).

### Tune BM25+RM3 with Optuna

To optimize **BM25+RM3** (k1, b, fb_terms, fb_docs, original_query_weight) on the training topics:

```bash
.venv/bin/python -m rag.experiments.bm25_rm3_optuna \
  --trials 50 \
  --topk 1000 \
  --out-json results/bm25_rm3_optuna_best.json
```

To generate a BM25+RM3 run file with custom parameters:

```bash
.venv/bin/python main.py \
  --approach bm25 \
  --split train \
  --output run_rm3_train.res \
  --run-tag run_rm3 \
  --topk 1000 \
  --k1 0.9 \
  --b 0.4 \
  --rm3 \
  --rm3-fb-terms 50 \
  --rm3-fb-docs 50 \
  --rm3-orig-weight 0.2 \
  --qe none
```

### Generate an EVAL_PER_TOPIC-style Markdown report from run files

To produce a summary like `EVAL_PER_TOPIC_V2*.md` for one or more **already-generated** TREC run files:

```bash
.venv/bin/python -m rag.experiments.per_topic_eval_md \
  --qrels qrels_50_Queries \
  --out-md results/EVAL_PER_TOPIC_BM25_ONLY.md \
  --run "BM25:run_1_train.res:5000:1000"
```

You can pass multiple `--run` specs to compare systems side-by-side.

### Create submission artifacts

To generate the submission run files and package them:

```bash
bash scripts/make_runs.sh
```

### Run file format (TREC 6-column)

Each output run file should contain up to **1000 documents per query** in this format:

```txt
630 Q0 ZF08-175-870 1 0.7 run1
```

Columns: `topic_id Q0 docid rank score run_tag`


