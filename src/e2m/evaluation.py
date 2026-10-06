from __future__ import annotations

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
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    rows = []
    for index, target in enumerate(targets):
        truth = y_true[:, index]
        predicted = y_pred[:, index]
        probability = y_prob[:, index]
        prevalence = float(truth.mean())
        auprc = float(average_precision_score(truth, probability)) if len(np.unique(truth)) > 1 else np.nan
        roc_auc = float(roc_auc_score(truth, probability)) if len(np.unique(truth)) > 1 else np.nan
        rows.append(
            {
                "target": target,
                "n_samples": len(truth),
                "n_positive": int(truth.sum()),
                "prevalence": prevalence,
                "auprc": auprc,
                "normalized_auprc": normalized_auprc(auprc, prevalence),
                "roc_auc": roc_auc,
                "accuracy": accuracy_score(truth, predicted),
                "precision": precision_score(truth, predicted, zero_division=0),
                "recall": recall_score(truth, predicted, zero_division=0),
                "f1": f1_score(truth, predicted, zero_division=0),
                "mcc": matthews_corrcoef(truth, predicted),
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
