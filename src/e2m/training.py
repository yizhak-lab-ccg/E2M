from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import logger
from .metrics import evaluate_predictions
from .models import MultitaskModel
from .splits import make_folds


def model_settings(config: dict, overrides: dict | None = None) -> dict:
    settings = dict(config["model"]["multitask_nn"])
    settings["standardize_inputs"] = config["preprocessing"].get("standardize_inputs", True)
    settings["random_state"] = config["preprocessing"].get("random_state", 42)
    settings.update(overrides or {})
    return settings


def run_cross_validation(expression, mutations, cancer, config, model_overrides=None):
    """Core out-of-fold multitask cross-validation.

    Returns ``(metrics, probabilities, predictions, fold_assignments, metadata)`` in
    memory. Nothing is written; callers decide what to keep or persist.
    """
    folds = int(config["evaluation"].get("cv_folds", 5))
    random_state = int(config["preprocessing"].get("random_state", 42))
    if len(expression) < folds:
        raise ValueError(f"Cannot run {folds} folds with only {len(expression)} samples.")
    cancer = cancer.reindex(expression.index)
    splits, split_method = make_folds(cancer, folds, random_state)

    probabilities = np.zeros(mutations.shape, dtype=np.float32)
    predictions = np.zeros(mutations.shape, dtype=np.int8)
    fold_number = np.zeros(len(expression), dtype=np.int16)
    settings = model_settings(config, model_overrides)
    logger.info(
        "Mutation cross-validation: %d folds, %d samples, %d targets",
        folds, len(expression), mutations.shape[1],
    )
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        logger.info("  fold %d/%d: train %d, evaluate %d", fold, folds, len(train_index), len(test_index))
        model = MultitaskModel(expression.shape[1], mutations.shape[1], **settings)
        model.fit(expression.iloc[train_index].values, mutations.iloc[train_index].values)
        fold_predictions, fold_probabilities = model.predict(expression.iloc[test_index].values)
        predictions[test_index] = fold_predictions
        probabilities[test_index] = fold_probabilities
        fold_number[test_index] = fold

    prob_frame = pd.DataFrame(probabilities, index=expression.index, columns=mutations.columns)
    pred_frame = pd.DataFrame(predictions, index=expression.index, columns=mutations.columns)
    metrics = evaluate_predictions(mutations.values, predictions, probabilities, mutations.columns)
    fold_frame = pd.DataFrame({"sample_id": expression.index, "fold": fold_number, "cancer": cancer.values})
    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "folds": folds,
        "split_method": split_method,
        "model": settings,
    }
    return metrics, prob_frame, pred_frame, fold_frame, metadata


def write_cross_validation(output_dir, metrics, prob_frame, pred_frame, fold_frame, metadata) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prob_frame.to_csv(output / "oof_probabilities.csv")
    pred_frame.to_csv(output / "oof_predictions.csv")
    metrics.to_csv(output / "metrics.csv")
    fold_frame.to_csv(output / "fold_assignments.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def cross_validate(expression, mutations, cancer, config, output_dir, model_overrides=None) -> dict:
    """Functional entry point used by the CLI: run CV, write files, return metadata."""
    metrics, prob_frame, pred_frame, fold_frame, metadata = run_cross_validation(
        expression, mutations, cancer, config, model_overrides
    )
    write_cross_validation(output_dir, metrics, prob_frame, pred_frame, fold_frame, metadata)
    return metadata


def train_full(expression, mutations, config, output_dir, metadata, model_overrides=None) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = model_settings(config, model_overrides)
    logger.info(
        "Training full-cohort model: %d samples, %d targets, %s epochs",
        len(expression), mutations.shape[1], settings.get("epochs", "default"),
    )
    model = MultitaskModel(expression.shape[1], mutations.shape[1], **settings)
    model.fit_full(expression.values, mutations.values)
    model.save(output / "model.pt")
    np.save(output / "feature_means.npy", expression.mean(axis=0).to_numpy(dtype=np.float32))
    (output / "features.json").write_text(json.dumps(expression.columns.tolist()), encoding="utf-8")
    (output / "targets.json").write_text(json.dumps(mutations.columns.tolist()), encoding="utf-8")
    bundle_metadata = {
        **metadata,
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "model": settings,
    }
    (output / "model_metadata.json").write_text(json.dumps(bundle_metadata, indent=2), encoding="utf-8")
    pd.DataFrame(model.history).to_csv(output / "training_history.csv", index=False)
    return bundle_metadata
