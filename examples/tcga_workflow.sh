#!/usr/bin/env bash
set -euo pipefail

CANCER="LUAD"
DATA_DIR="./e2m_data"
RESULT_DIR="results/luad"
MODEL_DIR="models/luad"

e2m download --cancer "$CANCER" --expression-dataset star_counts --transform raw \
  --data-dir "$DATA_DIR" --output "$RESULT_DIR/data"

e2m cv --cancer "$CANCER" --expression-dataset star_counts --transform raw \
  --data-dir "$DATA_DIR" --output "$RESULT_DIR/mutation_cv"

e2m tmb --cancer "$CANCER" --expression-dataset star_counts --transform raw \
  --data-dir "$DATA_DIR" --output "$RESULT_DIR/tmb_cv"

e2m train --cancer "$CANCER" --expression-dataset star_counts --transform raw \
  --data-dir "$DATA_DIR" --output "$MODEL_DIR"

e2m explain --model-dir "$MODEL_DIR" --target TP53 --method xgboost \
  --data-dir "$DATA_DIR" --output "$RESULT_DIR/shap_xgboost_tp53"

e2m embed --model-dir "$MODEL_DIR" --expression "$RESULT_DIR/data/expression.csv.gz" \
  --output "$RESULT_DIR/sample_embeddings.csv"

e2m head-weights --model-dir "$MODEL_DIR" --output "$RESULT_DIR/head_weights.csv"
