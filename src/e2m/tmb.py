from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import make_regressor
from .evaluation import tmb_metrics
from .logging_utils import logger
from .splits import make_folds


def tmb_ml_model_params(config: dict, ml_model: str | None = None):
    """Resolve the TMB regression ML model and its parameters from a config."""
    ml_model = ml_model or config["model"].get("tmb_ml_model", "xgboost")
    section = "xgboost" if ml_model == "xgboost" else ml_model
    return ml_model, dict(config["model"].get(section, {}))


def run_tmb_cv(expression, tmb, cancer, config, ml_model=None, params=None):
    """Core out-of-fold TMB regression. Returns (summary, predictions, fold_metrics, metadata)."""
    resolved_ml_model, resolved_params = tmb_ml_model_params(config, ml_model)
    if params is not None:
        resolved_params = dict(params)
    folds = int(config["evaluation"].get("cv_folds", 5))
    random_state = int(config["preprocessing"].get("random_state", 42))
    split_iterator, split_method = make_folds(cancer, folds, random_state)
    truth = tmb["TMB_log2"].to_numpy()
    out_of_fold = np.full(len(expression), np.nan)
    fold_ids = np.full(len(expression), -1, dtype=int)
    fold_rows = []
    logger.info(
        "TMB cross-validation (%s): %d folds, %d samples",
        resolved_ml_model, folds, len(expression),
    )
    for fold, (train_index, test_index) in enumerate(split_iterator, start=1):
        logger.info("  fold %d/%d: train %d, evaluate %d", fold, folds, len(train_index), len(test_index))
        model = make_regressor(resolved_ml_model, resolved_params)
        model.fit(expression.iloc[train_index], truth[train_index])
        prediction = model.predict(expression.iloc[test_index])
        out_of_fold[test_index] = prediction
        fold_ids[test_index] = fold
        fold_rows.append({"fold": fold, **tmb_metrics(truth[test_index], prediction)})

    predictions = pd.DataFrame(
        {
            "sample_id": expression.index,
            "cancer": cancer.reindex(expression.index).values,
            "fold": fold_ids,
            "TMB_log2_true": truth,
            "TMB_log2_pred": out_of_fold,
            "TMB_true": np.power(2, truth) - 1,
            "TMB_pred": np.maximum(np.power(2, out_of_fold) - 1, 0),
        }
    )
    summaries = [{"cancer": "__OVERALL__", **tmb_metrics(truth, out_of_fold)}]
    for cancer_code, subset in predictions.groupby("cancer"):
        summaries.append({"cancer": cancer_code, **tmb_metrics(subset["TMB_log2_true"], subset["TMB_log2_pred"])})
    summary = pd.DataFrame(summaries)
    fold_metrics = pd.DataFrame(fold_rows)
    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "folds": folds,
        "split_method": split_method,
        "target": "log2(coding TMB + 1)",
        "samples_without_coding_mutation_events": "dropped",
        "model": {"ml_model": resolved_ml_model, **resolved_params},
    }
    return summary, predictions, fold_metrics, metadata


def write_tmb(output_dir, summary, predictions, fold_metrics, metadata) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def run_tmb_cross_validation(expression, tmb, cancer, config, output_dir) -> dict:
    """Functional entry point used by the CLI: run TMB CV, write files, return metadata."""
    summary, predictions, fold_metrics, metadata = run_tmb_cv(expression, tmb, cancer, config)
    write_tmb(output_dir, summary, predictions, fold_metrics, metadata)
    return metadata
