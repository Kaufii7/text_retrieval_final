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


