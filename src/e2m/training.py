from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .metrics import evaluate_predictions
from .models import MultitaskModel
from .splits import make_folds


def model_settings(config: dict, overrides: dict | None = None) -> dict:
    settings = dict(config["model"]["multitask_nn"])
    settings["standardize_inputs"] = config["preprocessing"].get("standardize_inputs", True)
    settings["random_state"] = config["preprocessing"].get("random_state", 42)
    settings.update(overrides or {})
    return settings


def cross_validate(
    expression: pd.DataFrame,
    mutations: pd.DataFrame,
    cancer: pd.Series,
    config: dict,
    output_dir: str | Path,
    model_overrides: dict | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        model = MultitaskModel(expression.shape[1], mutations.shape[1], **settings)
        model.fit(expression.iloc[train_index].values, mutations.iloc[train_index].values)
        fold_predictions, fold_probabilities = model.predict(expression.iloc[test_index].values)
        predictions[test_index] = fold_predictions
        probabilities[test_index] = fold_probabilities
        fold_number[test_index] = fold

    probability_frame = pd.DataFrame(probabilities, index=expression.index, columns=mutations.columns)
    prediction_frame = pd.DataFrame(predictions, index=expression.index, columns=mutations.columns)
    metrics = evaluate_predictions(mutations.values, predictions, probabilities, mutations.columns)
    probability_frame.to_csv(output / "oof_probabilities.csv")
    prediction_frame.to_csv(output / "oof_predictions.csv")
    metrics.to_csv(output / "metrics.csv")
    pd.DataFrame({"sample_id": expression.index, "fold": fold_number, "cancer": cancer.values}).to_csv(
        output / "fold_assignments.csv", index=False
    )
    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "folds": folds,
        "split_method": split_method,
        "model": settings,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_full(
    expression: pd.DataFrame,
    mutations: pd.DataFrame,
    config: dict,
    output_dir: str | Path,
    metadata: dict,
    model_overrides: dict | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = model_settings(config, model_overrides)
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
