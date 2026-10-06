from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .backends import make_regressor
from .evaluation import evaluate_predictions, tmb_metrics
from .logging_utils import logger
from .models import MultitaskModel


def model_settings(config: dict, overrides: dict | None = None) -> dict:
    return {
        **config["model"]["multitask_nn"],
        "standardize_inputs": config["preprocessing"].get(
            "standardize_inputs", True
        ),
        "random_state": config["preprocessing"].get("random_state", 42),
        **(overrides or {}),
    }


def tmb_ml_model_params(config: dict, ml_model: str | None = None):
    ml_model = ml_model or config["model"].get("tmb_ml_model", "xgboost")
    return ml_model, dict(config["model"].get(ml_model, {}))


def cross_validate(cancer: pd.Series, config: dict, fit_predict, dtype):
    """Run cross-validation and return out-of-fold predictions, fold IDs, and split method."""
    n_splits = int(config["evaluation"].get("cv_folds", 5))
    random_state = int(config["preprocessing"].get("random_state", 42))

    if len(cancer) < n_splits:
        raise ValueError(
            f"Cannot run {n_splits} folds with only {len(cancer)} samples."
        )

    stratify = (
        cancer.nunique() > 1
        and cancer.value_counts().min() >= n_splits
    )

    if stratify:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(np.arange(len(cancer)), cancer)
        split_method = "stratified_by_cancer"
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(np.arange(len(cancer)))
        split_method = "shuffled_kfold"

    out_of_fold = None
    fold_ids = np.zeros(len(cancer), dtype=np.int16)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        logger.info(
            "  fold %d/%d: train %d, evaluate %d",
            fold,
            n_splits,
            len(train_idx),
            len(test_idx),
        )

        prediction = fit_predict(train_idx, test_idx)

        if out_of_fold is None:
            out_of_fold = np.full(
                (len(cancer), *np.shape(prediction)[1:]),
                np.nan,
                dtype=dtype,
            )

        out_of_fold[test_idx] = prediction
        fold_ids[test_idx] = fold

    return out_of_fold, fold_ids, split_method


def cross_validate_mutations(
    expression,
    mutations,
    cancer,
    config,
    model_overrides=None,
    output=None,
):
    """Out-of-fold multitask mutation prediction."""
    cancer = cancer.reindex(expression.index)
    settings = model_settings(config, model_overrides)

    logger.info(
        "Mutation cross-validation: %d samples, %d targets",
        len(expression),
        mutations.shape[1],
    )

    def fit_predict(train_idx, test_idx):
        model = MultitaskModel(
            expression.shape[1],
            mutations.shape[1],
            **settings,
        )
        model.fit(
            expression.iloc[train_idx].values,
            mutations.iloc[train_idx].values,
        )
        return model.predict(expression.iloc[test_idx].values)[1]

    probabilities, fold_ids, split_method = cross_validate(
        cancer,
        config,
        fit_predict,
        np.float32,
    )

    predictions = (probabilities >= 0.5).astype(np.int8)

    metrics = evaluate_predictions(
        mutations.values,
        predictions,
        probabilities,
        mutations.columns,
    )

    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "folds": int(fold_ids.max()),
        "split_method": split_method,
        "classification_threshold": 0.5,
        "score_note": (
            "Sigmoid outputs from weighted BCE are ranking scores, "
            "not calibrated probabilities."
        ),
        "model": settings,
    }

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            probabilities,
            index=expression.index,
            columns=mutations.columns,
        ).to_csv(output / "oof_probabilities.csv")

        pd.DataFrame(
            predictions,
            index=expression.index,
            columns=mutations.columns,
        ).to_csv(output / "oof_predictions.csv")

        metrics.to_csv(output / "metrics.csv")

        pd.DataFrame(
            {
                "sample_id": expression.index,
                "fold": fold_ids,
                "cancer": cancer.values,
            }
        ).to_csv(output / "fold_assignments.csv", index=False)

        (output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return metrics, metadata


def cross_validate_tmb(
    expression,
    tmb,
    cancer,
    config,
    ml_model=None,
    params=None,
    output=None,
):
    """Out-of-fold TMB regression."""
    ml_model, resolved_params = tmb_ml_model_params(config, ml_model)
    if params is not None:
        resolved_params = dict(params)

    cancer = cancer.reindex(expression.index)
    truth = tmb["TMB_log2"].to_numpy()

    logger.info(
        "TMB cross-validation (%s): %d samples",
        ml_model,
        len(expression),
    )

    def fit_predict(train_idx, test_idx):
        model = make_regressor(ml_model, resolved_params)
        model.fit(
            expression.iloc[train_idx],
            truth[train_idx],
        )
        return model.predict(expression.iloc[test_idx])

    out_of_fold, fold_ids, split_method = cross_validate(
        cancer,
        config,
        fit_predict,
        np.float64,
    )

    predictions = pd.DataFrame(
        {
            "sample_id": expression.index,
            "cancer": cancer.values,
            "fold": fold_ids,
            "TMB_log2_true": truth,
            "TMB_log2_pred": out_of_fold,
            "TMB_true": np.power(2, truth) - 1,
            "TMB_pred": np.maximum(np.power(2, out_of_fold) - 1, 0),
        }
    )

    summaries = [
        {
            "cancer": "__OVERALL__",
            **tmb_metrics(truth, out_of_fold),
        }
    ]

    for cancer_code, subset in predictions.groupby("cancer"):
        summaries.append(
            {
                "cancer": cancer_code,
                **tmb_metrics(
                    subset["TMB_log2_true"],
                    subset["TMB_log2_pred"],
                ),
            }
        )

    summary = pd.DataFrame(summaries)

    folds = range(1, int(fold_ids.max()) + 1)

    fold_metrics = pd.DataFrame(
        [
            {
                "fold": fold,
                **tmb_metrics(
                    truth[fold_ids == fold],
                    out_of_fold[fold_ids == fold],
                ),
            }
            for fold in folds
        ]
    )

    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "folds": int(fold_ids.max()),
        "split_method": split_method,
        "target": "log2(coding TMB + 1)",
        "samples_without_coding_mutation_events": "dropped",
        "model": {
            "ml_model": ml_model,
            **resolved_params,
        },
    }

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        predictions.to_csv(
            output / "oof_predictions.csv",
            index=False,
        )
        summary.to_csv(
            output / "summary.csv",
            index=False,
        )
        fold_metrics.to_csv(
            output / "fold_metrics.csv",
            index=False,
        )
        (output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return summary, metadata


def train_full(
    expression,
    mutations,
    config,
    output_dir,
    metadata,
    model_overrides=None,
) -> dict:
    """Train and save the multitask model on the full cohort."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    settings = model_settings(config, model_overrides)

    logger.info(
        "Training full-cohort model: %d samples, %d targets, %s epochs",
        len(expression),
        mutations.shape[1],
        settings.get("epochs", "default"),
    )

    model = MultitaskModel(
        expression.shape[1],
        mutations.shape[1],
        **settings,
    )
    model.fit_full(expression.values, mutations.values)
    model.save(output / "model.pt")

    np.save(
        output / "feature_means.npy",
        expression.mean(axis=0).to_numpy(dtype=np.float32),
    )

    (output / "features.json").write_text(
        json.dumps(expression.columns.tolist()),
        encoding="utf-8",
    )

    (output / "targets.json").write_text(
        json.dumps(mutations.columns.tolist()),
        encoding="utf-8",
    )

    bundle_metadata = {
        **metadata,
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "model": settings,
    }

    (output / "model_metadata.json").write_text(
        json.dumps(bundle_metadata, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(model.history).to_csv(
        output / "training_history.csv",
        index=False,
    )

    return bundle_metadata
