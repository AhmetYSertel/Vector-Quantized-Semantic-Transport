#!/usr/bin/env bash
# Download external datasets into data/. Run from repo root: bash scripts/download_data.sh
set -e
mkdir -p data/bfcl/possible_answer data/intent

GORILLA="https://raw.githubusercontent.com/ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data"
for cat in simple multiple; do
  curl -sL "$GORILLA/BFCL_v3_${cat}.json" -o "data/bfcl/BFCL_v3_${cat}.json"
  curl -sL "$GORILLA/possible_answer/BFCL_v3_${cat}.json" -o "data/bfcl/possible_answer/BFCL_v3_${cat}.json"
done

curl -sL "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json" \
  -o data/intent/clinc150.json
curl -sL "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv" \
  -o data/intent/banking77_train.csv
curl -sL "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv" \
  -o data/intent/banking77_test.csv

echo "done. datasets in data/"
