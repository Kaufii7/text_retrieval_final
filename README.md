## Text Retrieval Challenge (ROBUST04 / Pyserini)

This repo contains code and notes for generating **TREC-format run files** for the ROBUST04 retrieval task.

### Data in this repo

- **Queries**: `queriesROBUST.txt` (expected format: `topic_id<TAB>query_text`)
- **Train qrels**: `qrels_50_Queries` (expected columns: `topic_id 0 doc_id relevance`)
- **Project notes / plan**: `PROJECT_PLAN.md`

### Code layout

- `main.py`: entrypoint (will be expanded into a CLI as approaches are implemented)
- `rag/`: library code (approaches, I/O, run writing, evaluation, etc.)

### Environment (use the provided virtualenv)

This project expects **Python 3.10.19** via the provided `.venv`.

```bash
.venv/bin/python main.py
```

### Run file format (TREC 6-column)

Each output run file should contain up to **1000 documents per query** in this format:

```txt
630 Q0 ZF08-175-870 1 0.7 run1
```

Columns: `topic_id Q0 docid rank score run_tag`


