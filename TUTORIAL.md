# One-dataset tutorial: TCGA LUAD

This tutorial runs the full Expression-to-Mutation workflow for TCGA lung adenocarcinoma, abbreviated `LUAD`. It follows the manuscript setup and keeps evaluation separate from interpretation.

## 1. Install

From the repository root, create an environment and install the package. With conda:

```bash
conda create -n e2m python=3.10
conda activate e2m
pip install -e ".[interpretation]"
```

Or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[interpretation]"
```

## 2. Download and prepare LUAD

```bash
e2m download --cancer LUAD \
  --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/data
```

This step downloads:

- `TCGA-LUAD.star_counts.tsv.gz` from the Xena GDC hub.
- The LUAD MC3 nonsilent gene-level matrix from the Xena TCGA hub.
- GENCODE v36 gene annotation.

The loader converts the Xena log2 count values back to counts, applies a natural log (`log1p`) by default, keeps protein-coding genes, maps Ensembl identifiers to symbols, and averages duplicated symbols. To model linear counts instead, pass `--transform raw` to skip the log step. Expression and mutation tables are matched by TCGA sample identifier. Mutation targets must be present in at least 5 percent of the matched samples, and only the 400 most frequent targets are kept.

The exported files are:

- `results/luad/data/expression.csv.gz`
- `results/luad/data/mutations.csv.gz`
- `results/luad/data/samples.csv`

Downloaded files and prepared caches remain under `e2m_data`.

## 3. Evaluate mutation prediction

```bash
e2m cv --cancer LUAD \
  --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/mutation_cv
```

The model has a shared encoder with 512 and 256 units and one binary output per mutation target. It uses layer normalization, GELU, dropout 0.3, weighted binary cross-entropy, Adam, and an internal validation set for early stopping.

Five-fold cross-validation produces one held-out prediction for every sample. The main output is `metrics.csv`. It includes mutation prevalence, AUPRC, normalized AUPRC, ROC AUC, and threshold-based classification metrics.

Normalized AUPRC is calculated as:

```text
(AUPRC - prevalence) / (1 - prevalence)
```

It is 0 at the prevalence baseline and 1 for perfect ranking.

Other outputs are:

- `oof_probabilities.csv`
- `oof_predictions.csv`
- `fold_assignments.csv`
- `run_metadata.json`

These held-out results should be used when reporting prediction performance.

## 4. Evaluate TMB prediction

```bash
e2m tmb --cancer LUAD \
  --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/tmb_cv
```

TMB is counted from per-sample MC3 events in the coding classes listed in `default_config.yaml`. Samples without a coding MC3 event are dropped, matching the maintained manuscript workflow. XGBoost predicts `log2(TMB + 1)` with five-fold held-out predictions.

Outputs are:

- `oof_predictions.csv`: true and predicted TMB on raw and log2 scales.
- `summary.csv`: overall and cancer-specific Spearman, Pearson, MAE, and RMSE.
- `fold_metrics.csv`: fold-level metrics.
- `run_metadata.json`: target definition, model settings, and split method.

## 5. Train a reusable mutation model

```bash
e2m train --cancer LUAD \
  --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output models/luad
```

This trains on all matched LUAD samples. It is intended for prediction, embeddings, head weights, and direct neural SHAP. Do not use this full-cohort fit to report held-out performance.

## 6. Explain one mutation target with manuscript Tree SHAP

First confirm that `TP53` is present in `models/luad/targets.json`. Then run:

```bash
e2m explain --model-dir models/luad --target TP53 \
  --method xgboost --data-dir ./e2m_data \
  --output results/luad/shap_xgboost_tp53
```

This follows the manuscript interpretation step. It trains a separate XGBoost classifier for TP53 on all matched LUAD samples and applies `shap.TreeExplainer`. The XGBoost model is used only for attribution. It does not replace the multitask model used for held-out mutation performance.

`feature_summary.csv` ranks expression genes by mean absolute SHAP. `sample_shap_top_features.csv.gz` contains per-sample SHAP values and expression values for the top 20 features.

SHAP identifies features used by the model. Features can reflect downstream mutation effects, tumor subtype, co-mutation, immune or stromal composition, or other correlated biology. SHAP alone does not establish causality.

## 7. Explain the multitask neural output directly

```bash
e2m explain --model-dir models/luad --target TP53 \
  --method neural --data-dir ./e2m_data \
  --output results/luad/shap_neural_tp53
```

This uses SHAP GradientExplainer on the TP53 probability output of the trained multitask network. The background is sampled from LUAD. This answers a different question from the XGBoost analysis: it explains the deployed neural output itself.

The direct neural method can be slower. Results can also vary with the background set. Use the XGBoost method for strict alignment with the manuscript, and use neural SHAP when the goal is to inspect the multitask model directly.

## 8. Extract sample embeddings

```bash
e2m embed --model-dir models/luad \
  --expression results/luad/data/expression.csv.gz \
  --output results/luad/sample_embeddings.csv
```

The output contains the 256-dimensional shared representation for each sample. This representation can be used for clustering or visualization.

For a new cohort, provide a samples by genes CSV using the same expression quantity and transform. Input genes are aligned by symbol. The command stops if fewer than 80 percent of the training features overlap.

## 9. Extract output-head weights

```bash
e2m head-weights --model-dir models/luad \
  --output results/luad/head_weights.csv
```

Each row is one mutation target. Each column corresponds to one embedding dimension.

## 10. Predict a new expression matrix

```bash
e2m predict --model-dir models/luad \
  --expression my_luad_expression.csv \
  --output my_luad_mutation_probabilities.csv
```

The input must have sample identifiers in the first column and gene symbols in the header. Use STAR count values processed in the same way as the training data. The output contains one mutation probability per retained target.

## 11. The same workflow as objects

Every step above is also available through the Python objects, which return pandas
directly instead of writing files. A `Dataset` holds the aligned expression, mutation,
and TMB tables; an `E2MModel` and a `TmbModel` are fitted on a dataset and predict on any
dataset or samples-by-genes matrix.

```python
from e2m import Dataset, E2MModel, TmbModel

data = Dataset.from_tcga(["LUAD"], data_dir="./e2m_data")   # .expression / .mutations / .tmb / .cancer

train = data.subset(samples=data.samples[:400])
model = E2MModel().fit(train)                               # or E2MModel(epochs=50)
model.predict(data.subset(samples=data.samples[400:]))     # probabilities DataFrame
model.cross_validate(data)                                 # metrics DataFrame
model.save("models/luad")                                  # reload with E2MModel.load(...)

TmbModel(backend="lightgbm").cross_validate(data)          # xgboost/lightgbm/random_forest/gradient_boosting
```

See section 8 of [examples/LUAD_tutorial.ipynb](examples/LUAD_tutorial.ipynb) for a runnable
version, and [examples/external_transfer.py](examples/external_transfer.py) for predicting on an
external (GEO) cohort with a TCGA-trained model.

## Changing parameters

The command line exposes the main choices. Use a YAML configuration file for less common settings:

```bash
e2m cv --config my_config.yaml --cancer LUAD --output results/custom_cv
```

Start from `src/e2m/default_config.yaml`. Important settings include the expression dataset, Xena transform and offset, mutation prevalence threshold, target cap, network architecture, cross-validation folds, XGBoost settings, and SHAP sample limits.
