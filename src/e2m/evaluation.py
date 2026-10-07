from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalized_auprc(auprc: float, prevalence: float) -> float:
    if not np.isfinite(auprc) or not np.isfinite(prevalence) or prevalence >= 1:
        return np.nan
    return (auprc - prevalence) / (1.0 - prevalence)


def evaluate_predictions(labels, predictions, probabilities, targets) -> pd.DataFrame:
    """Per-target classification metrics; a NaN label means "not profiled" and that sample is skipped for that target."""
    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    profiled_mask = ~np.isnan(y_true)
    if not profiled_mask.all():
        warnings.warn(
            f"{int((~profiled_mask).sum())} mutation labels are missing (not profiled) across "
            f"{int((~profiled_mask).any(axis=0).sum())} targets; metrics use profiled samples only.",
            stacklevel=2,
        )
    rows = []
    for index, target in enumerate(targets):
        profiled = profiled_mask[:, index]
        truth = y_true[profiled, index].astype(int)
        predicted = y_pred[profiled, index]
        probability = y_prob[profiled, index]
        n_positive = int(truth.sum())
        n_negative = len(truth) - n_positive
        evaluable = n_positive > 0 and n_negative > 0
        prevalence = float(truth.mean()) if len(truth) else np.nan
        auprc = float(average_precision_score(truth, probability)) if evaluable else np.nan
        roc_auc = float(roc_auc_score(truth, probability)) if evaluable else np.nan
        true_negative = int(((truth == 0) & (predicted == 0)).sum())
        rows.append(
            {
                "target": target,
                "n_samples": len(truth),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_not_profiled": int((~profiled).sum()),
                "evaluable": evaluable,
                "prevalence": prevalence,
                "auprc": auprc,
                "normalized_auprc": normalized_auprc(auprc, prevalence),
                "roc_auc": roc_auc,
                "accuracy": accuracy_score(truth, predicted) if len(truth) else np.nan,
                "precision": precision_score(truth, predicted, zero_division=0) if len(truth) else np.nan,
                "recall": recall_score(truth, predicted, zero_division=0) if len(truth) else np.nan,
                "specificity": true_negative / n_negative if n_negative else np.nan,
                "f1": f1_score(truth, predicted, zero_division=0) if len(truth) else np.nan,
                "mcc": matthews_corrcoef(truth, predicted) if len(truth) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("target")


def tmb_metrics(y_true, y_pred, min_n: int = 3) -> dict:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[keep], prediction[keep]
    result = {
        "n": len(truth),
        "mae": np.nan,
        "rmse": np.nan,
        "spearman": np.nan,
        "spearman_p": np.nan,
        "pearson": np.nan,
        "pearson_p": np.nan,
    }
    if not len(truth):
        return result
    result["mae"] = mean_absolute_error(truth, prediction)
    result["rmse"] = np.sqrt(mean_squared_error(truth, prediction))
    if len(truth) >= min_n and np.std(truth) > 0 and np.std(prediction) > 0:
        spearman = spearmanr(truth, prediction)
        pearson = pearsonr(truth, prediction)
        result.update(
            {
                "spearman": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "pearson": pearson.statistic,
                "pearson_p": pearson.pvalue,
            }
        )
    return result
