# e2m

`e2m` predicts somatic mutation status and tumor mutational burden from RNA expression.

The main mutation model is a multitask neural network. A shared encoder learns a sample representation and one output predicts each retained mutation target. TMB prediction uses XGBoost regression. SHAP can explain one mutation output with either a separate XGBoost classifier, as used in the manuscript, or the trained multitask neural network directly.

RNA expression can contain signals from tumor cells, stromal cells, immune cells, cancer subtype, and co-occurring mutations. A prediction or SHAP association should not automatically be interpreted as a direct effect of a mutation.

## Installation

Python 3.10 or newer is required. Clone the repository and create an isolated environment with either Conda or venv. `pip install -e .` is an editable install from the clone.

### Conda

```bash
git clone https://github.com/asafpinhasitechnion/E2M.git
cd E2M

conda create -n e2m python=3.10
conda activate e2m
pip install -e .
```

`pip install -e .` installs mutation prediction and TMB prediction. To also use SHAP interpretation, install the optional extra:

```bash
pip install -e ".[interpretation]"
```

The `interpretation` extra adds SHAP. XGBoost is a core dependency, so both TMB prediction and the XGBoost SHAP workflow work without the extra.

### venv

```bash
git clone https://github.com/asafpinhasitechnion/E2M.git
cd E2M

python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

For SHAP interpretation, install `pip install -e ".[interpretation]"`.

### Conda environment file

```bash
git clone https://github.com/asafpinhasitechnion/E2M.git
cd E2M

conda env create -f environment.yml
conda activate e2m
```

The environment file creates the Conda environment and installs the package, with the interpretation extra, through pip.

### Verify the installation

Check that the command-line interface is available:

```bash
e2m --help
```

Run the lightweight synthetic check:

```bash
python examples/synthetic_quickstart.py
```

When the interpretation dependencies are installed, run the full runtime check:

```bash
python examples/full_runtime_check.py
```

The full check covers neural-network training, model persistence, prediction, embeddings, output-head weights, XGBoost Tree SHAP, neural-network SHAP, and TMB cross-validation.

## One cancer example

The following commands use TCGA lung adenocarcinoma, `LUAD`, and the manuscript preprocessing choices.

```bash
# Download Xena STAR counts and MC3 mutation data, then export prepared tables.
e2m download --cancer LUAD --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/data

# Evaluate gene mutation prediction with held-out five-fold predictions.
e2m cv --cancer LUAD --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/mutation_cv

# Evaluate coding TMB prediction with held-out five-fold predictions.
e2m tmb --cancer LUAD --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output results/luad/tmb_cv

# Train a deployable multitask model on all matched LUAD samples.
e2m train --cancer LUAD --expression-dataset star_counts --transform log1p \
  --data-dir ./e2m_data --output models/luad

# Explain TP53 with the manuscript XGBoost Tree SHAP approach.
e2m explain --model-dir models/luad --target TP53 --method xgboost \
  --data-dir ./e2m_data --output results/luad/shap_xgboost_tp53

# Explain the TP53 output of the multitask network directly.
e2m explain --model-dir models/luad --target TP53 --method neural \
  --data-dir ./e2m_data --output results/luad/shap_neural_tp53

# Extract LUAD sample embeddings and target head weights.
e2m embed --model-dir models/luad \
  --expression results/luad/data/expression.csv.gz \
  --output results/luad/sample_embeddings.csv
e2m head-weights --model-dir models/luad \
  --output results/luad/head_weights.csv
```

See [TUTORIAL.md](TUTORIAL.md) for the complete explanation of each step and output. A clean notebook version is available at [examples/LUAD_tutorial.ipynb](examples/LUAD_tutorial.ipynb).

## Xena expression choices

`e2m` downloads cohort-level files from the UCSC Xena GDC hub.

| Option | Xena file pattern | Meaning |
|---|---|---|
| `star_counts` | `TCGA-{cancer}.star_counts.tsv.gz` | STAR gene counts. This is the manuscript default. |
| `star_tpm` | `TCGA-{cancer}.star_tpm.tsv.gz` | STAR TPM. |
| `star_fpkm` | `TCGA-{cancer}.star_fpkm.tsv.gz` | STAR FPKM. |
| `star_fpkm_uq` | `TCGA-{cancer}.star_fpkm-uq.tsv.gz` | STAR upper-quartile FPKM. |

Xena RNA expression values are downloaded on a log2 scale. `e2m` treats the underlying quantity as counts (or TPM/FPKM) and log-transforms it by default. Choose one transform:

| Option | Action |
|---|---|
| `auto` | Use `log1p` for STAR counts and `xena` for TPM or FPKM. This is the package default. |
| `log1p` | Recover counts by inverting `log2(value + offset)`, then apply natural `log1p`. Default for STAR counts. |
| `raw` | Recover counts by inverting `log2(value + offset)`, with no log transform. Use this to opt out of the log transform. The default offset is 1. |
| `xena` | Keep the downloaded Xena values unchanged (already log2). A sensible choice for TPM or FPKM data. |

The offset is configurable as `data.xena_log_offset` in `default_config.yaml`. Check the Xena dataset metadata before changing expression datasets because Xena datasets can use different small offsets.

By default, the loader retains GENCODE v36 protein-coding genes, maps Ensembl identifiers to gene symbols, and averages duplicate symbols. Mutation labels come from the Xena MC3 nonsilent gene-level matrices. Targets must occur in at least 5 percent of matched samples, and at most the 400 most frequent targets are retained.

## Main commands

| Command | Purpose |
|---|---|
| `download` | Download, preprocess, cache, and optionally export matched data. |
| `cv` | Produce held-out mutation probabilities and normalized AUPRC metrics. |
| `tmb` | Produce held-out predictions for `log2(coding TMB + 1)`. |
| `train` | Train a full-cohort model bundle for reuse. |
| `predict` | Predict mutation probabilities for a samples by genes CSV. |
| `embed` | Extract the last shared encoder layer. |
| `head-weights` | Extract one output-head vector per mutation target. |
| `explain` | Explain one selected mutation output with SHAP. |

Use `e2m <command> --help` for command-specific options.

## SHAP methods

`--method xgboost` follows the manuscript interpretation workflow. It trains a separate full-cohort XGBoost classifier for the selected mutation target and applies Tree SHAP. This model is for interpretation and is not used as the primary cross-validated mutation predictor.

`--method neural` applies SHAP GradientExplainer directly to the selected probability output of the trained multitask network. It uses a sampled background from the same cohort. This method explains the neural model itself, but can be slower and more sensitive to the background sample than Tree SHAP.

Each explanation writes:

- `feature_summary.csv`: all expression features ranked by mean absolute SHAP.
- `sample_shap_top_features.csv.gz`: per-sample SHAP and expression values for the leading features.
- `metadata.json`: method, output scale, and sample counts.

## Model bundle

`e2m train` writes:

- `model.pt`: network weights, scaler, and model settings.
- `features.json`: required expression feature order.
- `feature_means.npy`: safe fill values for missing input features.
- `targets.json`: mutation output order.
- `model_metadata.json`: cancers, data choices, and resolved training configuration.
- `training_history.csv`: full-cohort training loss.

Prediction requires at least 80 percent feature overlap by default. Missing training features are filled with their training means, which become zero after standardization.

## Repository layout

```text
src/e2m/
  cli.py
  config.py
  data.py
  interpretation.py
  metrics.py
  models.py
  splits.py
  tmb.py
  training.py
  workflow.py
  default_config.yaml
examples/
tests/
TUTORIAL.md
```

## Citation and license

If you use this package, cite the accompanying Expression-to-Mutation manuscript. The repository is licensed under the MIT License.
