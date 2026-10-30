# E2M

`e2m` predicts somatic mutation status and tumor mutational burden (TMB) from RNA expression.

The mutation model is a multitask neural network: a shared encoder learns a sample representation, and one output predicts each mutation target. TMB is predicted with tree regression (XGBoost by default). SHAP explains one mutation target, either with a separate XGBoost classifier, as in the manuscript, or with the multitask network directly.

<img width="2246" height="759" alt="Overview of the Expression-to-Mutation workflow" src="https://github.com/user-attachments/assets/46bc627f-922a-4792-ae67-893da44edaaa" />

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/yizhak-lab-ccg/E2M.git
cd E2M

conda create -n e2m python=3.10
conda activate e2m
pip install -e ".[interpretation]"   # the extra adds SHAP
```

Alternatively, run `conda env create -f environment.yml`, then `conda activate e2m`. Use `pip install -e .` if you do not need SHAP.

For venv, use `python -m venv .venv`, then `source .venv/bin/activate` (Windows PowerShell: `.venv\Scripts\Activate.ps1`), followed by the same pip install.

For a pinned Linux GPU environment (Python 3.10, CUDA 12.6), build the [Dockerfile](Dockerfile) using the instructions at the top of that file. It installs the versions in [requirements-lock.txt](requirements-lock.txt).

To check the installation, run `e2m --help` and `python examples/synthetic_quickstart.py`.

## Tutorials

- [examples/LUAD_tutorial.ipynb](examples/LUAD_tutorial.ipynb) works through the full workflow on TCGA-LUAD, step by step.
- [examples/external_transfer.ipynb](examples/external_transfer.ipynb) trains on TCGA-LUAD, predicts on the external cohort GSE31210, and scores EGFR and KRAS against that cohort's reported alterations.

## Python API

A `Dataset` loads one or more TCGA cohorts and keeps the expression, mutation labels, TMB, and cancer labels aligned to the same samples. `E2MModel` is the multitask mutation network and `TmbModel` is the tree regressor for TMB. Fit either one on a `Dataset`, then predict on another `Dataset` or on a samples-by-genes DataFrame.

```python
from e2m import Dataset, E2MModel, TmbModel

data = Dataset.from_tcga(["LUAD"], data_dir="./e2m_data")
train = data.subset(samples=data.samples[:400])
test = data.subset(samples=data.samples[400:])

model = E2MModel().fit(train)          # or E2MModel(epochs=50, hidden_layers=[512, 256])
probabilities = model.predict(test)    # samples x targets DataFrame
model.predict(test.expression)        # DataFrame columns align by gene symbol
metrics = model.cross_validate(data, output="results/luad_cv")
embeddings = model.embed(data)
shap = model.explain(data, target="TP53", method="xgboost")
model.save("models/luad")              # E2MModel.load("models/luad")

tmb_summary = TmbModel().cross_validate(data)   # per-cancer and overall Spearman, Pearson, MAE, RMSE
```

Mutation outputs are ranking scores, not calibrated probabilities. Report held-out performance using cross-validation outputs. Normalized AUPRC is `(AUPRC - prevalence) / (1 - prevalence)`: 0 at the prevalence baseline and 1 for perfect ranking. Classification metrics use a fixed 0.5 threshold. By default, neural fitting holds out 15% of the supplied samples for early stopping, with no subsequent refit on all samples.

For external data, use the same expression units and preprocessing as the training data. DataFrames use sample IDs as the index and gene symbols as columns; `E2MModel` also accepts NumPy and sparse matrices, which must already follow the training feature order. Call `e2m.options()` to inspect supported choices and `e2m.set_verbose()` to print progress.

## Command line

| Command | Purpose |
|---|---|
| `download` | Download, preprocess, cache, and optionally export matched data. |
| `cv` | Cross-validated mutation probabilities and metrics. |
| `tmb` | Cross-validated predictions of `log2(coding TMB + 1)`. |
| `train` | Train a mutation model with an internal early-stopping split. |
| `predict` | Predict mutation probabilities for a samples-by-genes CSV. |
| `embed` | Extract the last shared encoder layer. |
| `head-weights` | Extract one output-head vector per mutation target. |
| `explain` | Explain one mutation target with SHAP. |
| `options` | List accepted expression datasets, transforms, TMB models, SHAP methods, and cancer codes. |

```bash
e2m cv --cancer LUAD --data-dir ./e2m_data --output results/luad/mutation_cv
e2m tmb --cancer LUAD --data-dir ./e2m_data --output results/luad/tmb_cv
e2m train --cancer LUAD --data-dir ./e2m_data --output models/luad
e2m explain --model-dir models/luad --target TP53 --method xgboost \
  --data-dir ./e2m_data --output results/luad/shap_tp53
```

Run `e2m <command> --help` for each command's options and add `--verbose` to print progress. All defaults are in [src/e2m/default_config.yaml](src/e2m/default_config.yaml).

The commands above use package defaults, including no CPM normalization. The exact manuscript TCGA configuration is copied in [configs/manuscript.yaml](configs/manuscript.yaml) from [E2M-analyses, Runs/TCGA/config/e2m.yaml](https://github.com/asafpinhasitechnion/E2M-analyses/blob/main/Runs/TCGA/config/e2m.yaml); the published runs used E2M commit `b201b22`. Pass `--config configs/manuscript.yaml` to `download`, `cv`, `train`, or `tmb` to use those settings (including GPU XGBoost). To enable only CPM, pass `--normalization cpm`; the package default remains `none`. The analysis repository's run scripts handle the manuscript cohorts and downstream analyses.

## Data

Expression is downloaded from the UCSC Xena GDC hub: `star_counts` (default, as in the manuscript), `star_tpm`, `star_fpkm`, or `star_fpkm_uq`. By default (`--transform log1p`), `e2m` inverts the downloaded log2 values and applies natural `log1p`. The inversion assumes `log2(value + offset)`, with `--log-offset 1` by default; set the offset to match the dataset metadata. Use `raw` for the recovered linear scale or `xena` to keep the downloaded values without inversion.

The loader keeps GENCODE v36 protein-coding genes and sums duplicate gene symbols. Mutation labels are the MC3 nonsilent gene-level calls. Samples without matching MC3 data are excluded; missing gene entries in the MC3 gene-level matrices are treated as no mutation. Targets are genes mutated in at least 5% of samples, up to the 400 most frequent.

Coding TMB is the number of MC3 events in the configured coding variant classes per sample; it is not normalized per megabase. Samples without a qualifying event are excluded from TMB analysis. The regression target is `log2(TMB + 1)`.

## SHAP methods

- `xgboost` (manuscript default) trains a separate XGBoost classifier on the full cohort for the selected target and applies Tree SHAP. `lightgbm`, `random_forest`, and `gradient_boosting` work the same way.
- `neural` applies SHAP GradientExplainer to the selected output of the trained multitask network, with a background sample from the same cohort.

Tree SHAP rankings are descriptive, in-sample attributions and may reflect subtype, co-mutation, immune/stromal composition, or other correlated biology; they do not establish causal effects.

The tree methods explain a separately fitted classifier; `neural` explains the mutation network itself. LightGBM requires the additional `lightgbm` extra: `pip install -e ".[interpretation,lightgbm]"`.

The CLI, or `model.explain(..., output=...)` in Python, writes `feature_summary.csv` (all features ranked by mean absolute SHAP), `sample_shap_top_features.csv.gz` (per-sample values for the top features), and `metadata.json`.

## Outputs

- **Mutation cross-validation** (`cv`): `oof_probabilities.csv` (scores), `oof_predictions.csv` (binary calls), `metrics.csv` (pooled metrics), `fold_metrics.csv`, and `fold_assignments.csv`.
- **TMB cross-validation** (`tmb`): `oof_predictions.csv` (true/predicted values on raw and log2 scales, with fold assignments), `summary.csv` (pooled and per-cancer metrics), and `fold_metrics.csv`. MAE, RMSE, and correlations are computed on the log2 scale.
- **Both CV workflows**: fold models (`folds/`, by default) and `run_metadata.json` (settings, package versions, device, and E2M commit).
- **Trained model** (`train`): network weights and input scaler (`model.pt`), feature order (`features.json`), target order (`targets.json`), feature means for missing genes, metadata, and training history.
- **Preprocessing**: `download --output`, `cv`, and `train` write `preprocessing_manifest.json` (data sources, checksums, and preprocessing settings); in Python it is `data.manifest`.

Prediction requires at least 80% of the training features by default; missing features are filled with their training means.

## Citation and license

If you use this package, please cite the accompanying Expression-to-Mutation manuscript. The repository is licensed under the MIT License.
