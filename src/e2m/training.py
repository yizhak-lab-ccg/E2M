from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .config import logger
from .evaluation import evaluate_predictions, tmb_metrics
from .models import E2MModel, TmbModel, model_settings, tmb_ml_model_params

PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost", "lightgbm", "shap")


def make_splits(cancer: pd.Series, config: dict, groups=None, folds=None):
    """Return the (train_idx, test_idx) pairs and the split method name."""
    if folds is not None:
        labels = np.asarray(folds)
        if pd.isna(labels).any():
            raise ValueError("folds= must assign every sample to a fold.")
        splits = [
            (np.flatnonzero(labels != label), np.flatnonzero(labels == label))
            for label in np.unique(labels)
        ]
        return splits, "predefined"

    n_splits = int(config["evaluation"].get("cv_folds", 5))
    random_state = int(config["preprocessing"].get("random_state", 42))
    split = config["evaluation"].get("split", "stratified")
    index = np.arange(len(cancer))

    if len(cancer) < n_splits:
        raise ValueError(
            f"Cannot run {n_splits} folds with only {len(cancer)} samples."
        )

    if split == "grouped":
        if groups is None:
            raise ValueError('split "grouped" needs groups= (one group label per sample).')
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(index, groups=np.asarray(groups))), "grouped"

    if split not in ("stratified", "kfold"):
        raise ValueError("evaluation.split must be stratified, kfold, or grouped.")

    stratify = (
        split == "stratified"
        and cancer.nunique() > 1
        and cancer.value_counts().min() >= n_splits
    )

    if stratify:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(index, cancer)), "stratified_by_cancer"

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(index)), "shuffled_kfold"


def cross_validate(cancer: pd.Series, config: dict, fit_predict, dtype, groups=None, folds=None):
    """Run fit_predict(fold, train_idx, test_idx) per fold; return out-of-fold predictions, fold IDs, split method."""
    splits, split_method = make_splits(cancer, config, groups, folds)

    out_of_fold = None
    fold_ids = np.zeros(len(cancer), dtype=np.int16)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        logger.info(
            "  fold %d/%d: train %d, evaluate %d",
            fold,
            len(splits),
            len(train_idx),
            len(test_idx),
        )

        prediction = fit_predict(fold, train_idx, test_idx)

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
    groups=None,
    folds=None,
):
    """Out-of-fold multitask mutation prediction; `groups`/`folds` give each sample's group or fold."""
    started = time.perf_counter()
    cancer = cancer.reindex(expression.index)
    settings = model_settings(config, model_overrides)
    fold_dir = _fold_dir(output, config)

    logger.info(
        "Mutation cross-validation: %d samples, %d targets",
        len(expression),
        mutations.shape[1],
    )

    def fit_predict(fold, train_idx, test_idx):
        model = E2MModel(config, **(model_overrides or {}))
        model.fit(
            expression.iloc[train_idx],
            mutations.iloc[train_idx],
        )
        if fold_dir is not None:
            model.save(fold_dir / f"fold_{fold}")
        return model.predict(expression.iloc[test_idx]).to_numpy()

    probabilities, fold_ids, split_method = cross_validate(
        cancer,
        config,
        fit_predict,
        np.float32,
        _aligned(groups, expression.index),
        _aligned(folds, expression.index),
    )

    predictions = (probabilities >= 0.5).astype(np.int8)

    metrics = evaluate_predictions(
        mutations.values,
        predictions,
        probabilities,
        mutations.columns,
    )

    fold_metrics = []
    for fold in range(1, int(fold_ids.max()) + 1):
        in_fold = fold_ids == fold
        table = evaluate_predictions(
            mutations.values[in_fold],
            predictions[in_fold],
            probabilities[in_fold],
            mutations.columns,
        ).reset_index()
        table.insert(0, "fold", fold)
        fold_metrics.append(table)
    fold_metrics = pd.concat(fold_metrics, ignore_index=True)

    metadata = {
        "n_samples": len(expression),
        "n_features": expression.shape[1],
        "n_targets": mutations.shape[1],
        "folds": int(fold_ids.max()),
        "split_method": split_method,
        "headline_metrics": "pooled out-of-fold (metrics.csv); per-fold values in fold_metrics.csv",
        "classification_threshold": 0.5,
        "score_note": (
            "Sigmoid outputs from weighted BCE are ranking scores, "
            "not calibrated probabilities."
        ),
        "model": settings,
        "run": run_info(started),
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
        fold_metrics.to_csv(output / "fold_metrics.csv", index=False)

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
    groups=None,
    folds=None,
):
    """Out-of-fold TMB regression; `groups`/`folds` give each sample's group or fold."""
    started = time.perf_counter()
    ml_model, resolved_params = tmb_ml_model_params(config, ml_model)
    if params is not None:
        resolved_params = dict(params)

    cancer = cancer.reindex(expression.index)
    truth = tmb["TMB_log2"].to_numpy()
    fold_dir = _fold_dir(output, config)

    logger.info(
        "TMB cross-validation (%s): %d samples",
        ml_model,
        len(expression),
    )

    def fit_predict(fold, train_idx, test_idx):
        model = TmbModel(ml_model, config=config, **resolved_params)
        model.fit(
            expression.iloc[train_idx],
            tmb.iloc[train_idx],
        )
        if fold_dir is not None:
            model.save(fold_dir / f"fold_{fold}")
        return model.predict(expression.iloc[test_idx])["TMB_log2_pred"].to_numpy()

    out_of_fold, fold_ids, split_method = cross_validate(
        cancer,
        config,
        fit_predict,
        np.float64,
        _aligned(groups, expression.index),
        _aligned(folds, expression.index),
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
        "headline_metrics": "pooled out-of-fold (__OVERALL__ row of summary.csv); per-fold values in fold_metrics.csv",
        "target": "log2(coding TMB + 1)",
        "samples_without_coding_mutation_events": "dropped",
        "model": {
            "ml_model": ml_model,
            **resolved_params,
        },
        "run": run_info(started),
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


def run_info(started: float) -> dict:
    """Timing, hardware, package versions, and E2M version/commit for run_metadata.json."""
    import torch

    from . import __version__

    cuda = torch.cuda.is_available()
    commit, dirty = _commit()
    return {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1),
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "process_peak_memory_mb": _peak_memory_mb(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "e2m_version": __version__,
        "e2m_commit": commit,
        "e2m_uncommitted_changes": dirty,
    }


def _fold_dir(output, config: dict) -> Path | None:
    if output is None or not config["evaluation"].get("save_fold_models", True):
        return None
    return Path(output) / "folds"


def _aligned(values, index: pd.Index):
    """A Series is matched by sample ID; an array is taken in sample order."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return values.reindex(index)
    return pd.Series(np.asarray(values), index=index)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _peak_memory_mb() -> float | None:
    """Peak resident memory of this process so far (Linux/macOS via resource, Windows via psutil if installed)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if platform.system() == "Darwin" else 2**10), 1)
    except ImportError:
        pass
    try:
        import psutil

        return round(psutil.Process().memory_info().peak_wset / 2**20, 1)
    except (ImportError, AttributeError):
        return None


def _commit() -> tuple[str | None, bool | None]:
    """The E2M git commit (or $E2M_COMMIT, e.g. set in a container) and whether the tree had changes."""
    if os.environ.get("E2M_COMMIT"):
        return os.environ["E2M_COMMIT"], None
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        if head.returncode:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return head.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None
