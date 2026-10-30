# E2M

**E2M (Expression to Mutation)** predicts somatic mutation status and tumor mutational burden (TMB) from RNA expression. It also provides tools to examine mutation-associated expression features and learned representations of samples and target genes.

This package accompanies *Expression to Mutation: Inferring somatic mutations and cellular perturbation states from RNA expression* (Pinhasi and Yizhak).

![Overview of the E2M workflow](https://github.com/user-attachments/assets/46bc627f-922a-4792-ae67-893da44edaaa)

Explore the results in the [interactive mutation viewer](https://expression-to-mutation.streamlit.app/).

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/yizhak-lab-ccg/E2M.git
cd E2M

conda create -n e2m python=3.10
conda activate e2m
pip install -e ".[interpretation]"
```

The `interpretation` extra installs SHAP. For a minimal installation, use `pip install -e .`. You can also use `conda env create -f environment.yml` or the provided [Dockerfile](Dockerfile) and [locked requirements](requirements-lock.txt).

Check the installation:

```bash
e2m --help
python examples/synthetic_quickstart.py
```

## Quick start

### Python

```python
from e2m import Dataset, E2MModel, TmbModel

data = Dataset.from_tcga(["LUAD"], data_dir="./e2m_data")
train = data.subset(samples=data.samples[:400])
test = data.subset(samples=data.samples[400:])

model = E2MModel().fit(train)
scores = model.predict(test)       # samples x mutation targets
embeddings = model.embed(test)     # sample representations
model.save("models/luad")

cv_metrics = E2MModel().cross_validate(data, output="results/luad_cv")
tmb_metrics = TmbModel().cross_validate(data, output="results/luad_tmb_cv")
```

Mutation outputs are scores between 0 and 1, **not calibrated probabilities**. Use held-out predictions to assess performance.

### Command line

```bash
e2m cv --cancer LUAD --data-dir ./e2m_data --output results/luad/mutation_cv
e2m tmb --cancer LUAD --data-dir ./e2m_data --output results/luad/tmb_cv
e2m train --cancer LUAD --data-dir ./e2m_data --output models/luad
e2m explain --model-dir models/luad --target TP53 --method xgboost \
  --data-dir ./e2m_data --output results/luad/shap_tp53
```

These commands use **package defaults**, which differ from the manuscript settings (notably, CPM normalization is off by default). To use the manuscript's TCGA configuration, add `--config configs/manuscript.yaml` to `download`, `cv`, `train`, or `tmb`.

Run `e2m <command> --help` for additional options. Package defaults are defined in [`src/e2m/default_config.yaml`](src/e2m/default_config.yaml).

## Tutorials

- [TCGA-LUAD tutorial](examples/LUAD_tutorial.ipynb): data loading, training, evaluation, and interpretation.
- [External cohort transfer](examples/external_transfer.ipynb): train on TCGA-LUAD and evaluate EGFR and KRAS predictions in GSE31210.

## Reproducing the manuscript

**This repository provides the E2M software.** The complete analyses reported in the paper, including external cohort validation, single-cell experiments, and figure generation, are in [E2M-analyses](https://github.com/asafpinhasitechnion/E2M-analyses).

- **Analysis code and configurations:** [E2M-analyses](https://github.com/asafpinhasitechnion/E2M-analyses)
- **Manuscript configuration:** [`configs/manuscript.yaml`](configs/manuscript.yaml), copied from the analysis repository
- **E2M version used for the reported analyses:** commit `b201b22`
- **Trained models, predictions, and analysis outputs:** Zenodo archive (**DOI/link to be added**)

The archive is intended to allow the reported results to be checked without retraining the models. For a pinned Linux GPU environment, see the [Dockerfile](Dockerfile) and [requirements-lock.txt](requirements-lock.txt). Training and prediction times are in [RUNTIME.md](RUNTIME.md).

## Methods and data

**Mutation prediction:** A multitask neural network with a shared encoder (512 and 256 units) and one output per mutation target. Training uses weighted binary cross-entropy and early stopping. Predictions are evaluated using five-fold cross-validation and prevalence-normalized AUPRC:

`(AUPRC - mutation prevalence) / (1 - mutation prevalence)`

**TMB prediction:** XGBoost regression predicts `log2(TMB + 1)`. TMB is the number of qualifying coding mutation events, **not** mutations per megabase. Performance is evaluated using held-out Spearman correlation.

**Interpretation:** SHAP analysis uses a *separate* XGBoost classifier for each selected cancer-gene task, trained on the full cohort. These attributions are descriptive and do not represent held-out validation or establish causality. The package also supports SHAP explanations of the neural model.

**Embeddings:** The neural network's final shared layer provides a 256-dimensional representation of each sample. Output-head weights provide a vector for each mutation target.

**TCGA inputs:** RNA expression is retrieved from the UCSC Xena GDC hub, with mutation labels from MC3. The loader retains GENCODE v36 protein-coding genes, maps Ensembl IDs to gene symbols, and sums duplicate gene symbols. Samples without matching mutation data are excluded. Samples without qualifying coding MC3 events are excluded from TMB analysis.

**Preprocessing:** The manuscript's TCGA workflow converts Xena STAR-count values back to counts, applies CPM normalization and `log1p`, then standardizes expression using training-set statistics. For most external cohorts, the manuscript uses within-sample gene ranks for both training and external samples, permitting independent prediction; Morrison and ComBat comparisons use cohort-dependent processing. See the [analysis repository](https://github.com/asafpinhasitechnion/E2M-analyses) for the exact external-cohort workflows.

For your own data, match the training model's gene symbols, expression scale, normalization, and feature order. DataFrames use samples as rows and genes as columns. See `e2m options` for supported dataset and transformation settings.

## Saved outputs

- **Cross-validation:** out-of-fold scores, metrics, fold assignments, and fitted fold models.
- **Trained models:** weights, scaler, feature and target order, configuration, and metadata.
- **Interpretation:** SHAP feature summaries, per-sample SHAP values, and metadata.
- **Preprocessing:** source and preprocessing manifests where supported.

See the generated output directories for the complete file manifests.

## Citation and license

If you use E2M, please cite the accompanying manuscript (citation/DOI to be added). Released under the [MIT License](LICENSE).
