#!/usr/bin/env bash
set -euo pipefail

# Generate the three TREC run files and package them for submission.
#
# Defaults:
# - uses provided venv at .venv/bin/python
# - uses test split (topics after the first 50)
# - writes run_1.res, run_2.res, run_3.res to repo root
# - zips them into submission.zip
#
# Note:
# Approach2/3 are currently templates. Until they are implemented, this script
# generates all three runs using BM25 with (optionally) different k1/b values.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT_DIR}/.venv/bin/python"
MAIN="${ROOT_DIR}/main.py"

TOPK="${TOPK:-1000}"
SPLIT="${SPLIT:-test}"
QUERIES="${QUERIES:-${ROOT_DIR}/queriesROBUST.txt}"

# Per-run BM25 parameters (override via env vars if you want)
RUN1_K1="${RUN1_K1:-0.9}"
RUN1_B="${RUN1_B:-0.4}"
RUN2_K1="${RUN2_K1:-1.2}"
RUN2_B="${RUN2_B:-0.4}"
RUN3_K1="${RUN3_K1:-1.5}"
RUN3_B="${RUN3_B:-0.4}"

OUT1="${OUT1:-${ROOT_DIR}/run_1.res}"
OUT2="${OUT2:-${ROOT_DIR}/run_2.res}"
OUT3="${OUT3:-${ROOT_DIR}/run_3.res}"
ZIP_OUT="${ZIP_OUT:-${ROOT_DIR}/submission.zip}"

echo "Using python: ${PY}"
echo "Split: ${SPLIT}  TOPK: ${TOPK}"
echo "Queries: ${QUERIES}"

echo "Generating run_1.res (BM25 k1=${RUN1_K1} b=${RUN1_B}) -> ${OUT1}"
"${PY}" "${MAIN}" --approach bm25 --split "${SPLIT}" --queries "${QUERIES}" --topk "${TOPK}" \
  --k1 "${RUN1_K1}" --b "${RUN1_B}" \
  --output "${OUT1}" --run-tag run1

echo "Generating run_2.res (BM25 k1=${RUN2_K1} b=${RUN2_B}) -> ${OUT2}"
"${PY}" "${MAIN}" --approach bm25 --split "${SPLIT}" --queries "${QUERIES}" --topk "${TOPK}" \
  --k1 "${RUN2_K1}" --b "${RUN2_B}" \
  --output "${OUT2}" --run-tag run2

echo "Generating run_3.res (BM25 k1=${RUN3_K1} b=${RUN3_B}) -> ${OUT3}"
"${PY}" "${MAIN}" --approach bm25 --split "${SPLIT}" --queries "${QUERIES}" --topk "${TOPK}" \
  --k1 "${RUN3_K1}" --b "${RUN3_B}" \
  --output "${OUT3}" --run-tag run3

echo "Packaging into ${ZIP_OUT}"
rm -f "${ZIP_OUT}"
(cd "${ROOT_DIR}" && zip -q "$(basename "${ZIP_OUT}")" "$(basename "${OUT1}")" "$(basename "${OUT2}")" "$(basename "${OUT3}")")

echo "Done."
echo "Created:"
echo "  ${OUT1}"
echo "  ${OUT2}"
echo "  ${OUT3}"
echo "  ${ZIP_OUT}"


