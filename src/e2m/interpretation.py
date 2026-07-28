from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_single_target_explanation(
    model,
    expression: pd.DataFrame,
    labels: pd.Series,
    target: str,
    target_index: int,
    method: str,
    config: dict,
):
    """Compute SHAP for one target in memory.

    Returns ``(summary, sample_values, metadata)`` and writes nothing; callers decide
    what to keep or persist (see :func:`write_explanation`).
    """
    settings = config["interpretation"]
    random_state = int(config["preprocessing"].get("random_state", 42))
    rng = np.random.default_rng(random_state)
    max_samples = int(settings.get("max_samples", 500) or len(expression))
    if len(expression) > max_samples:
        selected = np.sort(rng.choice(len(expression), size=max_samples, replace=False))
        explained_expression = expression.iloc[selected]
    else:
        explained_expression = expression

    from .backends import TREE_MODELS

    method = method.lower()
    if method in TREE_MODELS:
        section = "xgboost" if method == "xgboost" else method
        shap_values, output_scale = _explain_tree(
            expression,
            labels,
            explained_expression,
            method,
            config["model"].get(section, {}),
        )
    elif method == "neural":
        shap_values, output_scale = _explain_neural(
            model,
            expression,
            explained_expression,
            target_index,
            settings,
            rng,
        )
    else:
        raise ValueError(
            "SHAP method must be one of: " + ", ".join(TREE_MODELS) + ", neural."
        )

    shap_values = _two_dimensional(shap_values)
    if shap_values.shape != explained_expression.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; expected {explained_expression.shape}."
        )
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    summary = pd.DataFrame(
        {
            "target": target,
            "method": method,
            "feature": explained_expression.columns,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
        }
    ).iloc[order]
    summary.insert(2, "rank", np.arange(1, len(summary) + 1))

    top_n = min(int(settings.get("top_features", 20)), expression.shape[1])
    top_indices = order[:top_n]
    sample_rows = []
    raw_values = explained_expression.values
    sample_ids = explained_expression.index.astype(str).to_numpy()
    for feature_index in top_indices:
        sample_rows.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "target": target,
                    "method": method,
                    "feature": explained_expression.columns[feature_index],
                    "shap_value": shap_values[:, feature_index],
                    "expression_value": raw_values[:, feature_index],
                }
            )
        )
    sample_values = pd.concat(sample_rows, ignore_index=True)
    metadata = {
        "target": target,
        "method": method,
        "output_scale": output_scale,
        "n_training_samples": len(expression),
        "n_explained_samples": len(explained_expression),
        "n_features": expression.shape[1],
        "top_features_saved_per_sample": top_n,
    }
    return summary, sample_values, metadata


def write_explanation(output_dir, summary, sample_values, metadata) -> Path:
    """Write the three SHAP outputs produced by :func:`run_single_target_explanation`."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "feature_summary.csv", index=False)
    sample_values.to_csv(output / "sample_shap_top_features.csv.gz", index=False, compression="gzip")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def explain_single_target(
    model,
    expression: pd.DataFrame,
    labels: pd.Series,
    target: str,
    target_index: int,
    method: str,
    output_dir: str | Path,
    config: dict,
) -> dict:
    """Functional entry point used by the CLI: compute SHAP, write files, return metadata."""
    summary, sample_values, metadata = run_single_target_explanation(
        model, expression, labels, target, target_index, method, config
    )
    write_explanation(output_dir, summary, sample_values, metadata)
    return metadata


def _require_shap():
    try:
        import shap

        return shap
    except ImportError as exc:
        raise ImportError('This SHAP method requires: pip install -e ".[interpretation]"') from exc


def _explain_tree(expression, labels, explained_expression, ml_model, settings):
    """Train a tree classifier for one target and explain it with Tree SHAP.

    ``ml_model`` is one of :data:`e2m.backends.TREE_MODELS`. Class imbalance is handled
    per model: ``scale_pos_weight`` for the boosting models, ``class_weight`` for the
    random forest, and balanced per-sample weights for scikit-learn gradient boosting.

    XGBoost is explained with its own native ``pred_contribs`` (the exact Tree SHAP that
    ``shap`` delegates to for XGBoost), which needs no ``shap`` install and is robust across
    XGBoost serialization versions. The other models use ``shap.TreeExplainer``.
    """
    from .backends import make_classifier

    y = labels.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("The selected mutation target has only one class.")
    positives = int(y.sum())
    negatives = len(y) - positives

    parameters = dict(settings)
    fit_kwargs: dict = {}
    if ml_model in ("xgboost", "lightgbm"):
        parameters.setdefault("scale_pos_weight", negatives / max(positives, 1))
    elif ml_model == "random_forest":
        parameters.setdefault("class_weight", "balanced")
    else:  # gradient_boosting has no class weighting; weight the samples instead
        weight_positive = len(y) / (2.0 * max(positives, 1))
        weight_negative = len(y) / (2.0 * max(negatives, 1))
        fit_kwargs["sample_weight"] = np.where(y == 1, weight_positive, weight_negative)

    classifier = make_classifier(ml_model, parameters)
    classifier.fit(expression.values, y, **fit_kwargs)

    if ml_model == "xgboost":
        import xgboost as xgb

        booster = classifier.get_booster()
        contributions = booster.predict(
            xgb.DMatrix(explained_expression.values), pred_contribs=True
        )
        # pred_contribs returns per-feature SHAP values plus a trailing bias column.
        return contributions[:, :-1], "xgboost_raw_margin"

    shap = _require_shap()
    explainer = shap.TreeExplainer(classifier)
    return explainer.shap_values(explained_expression.values), f"{ml_model}_tree_shap"


def _explain_neural(model, expression, explained_expression, target_index, settings, rng):
    shap = _require_shap()
    import torch
    import torch.nn as nn

    class SingleOutput(nn.Module):
        def __init__(self, network, output_index):
            super().__init__()
            self.network = network
            self.output_index = output_index

        def forward(self, values):
            logits = self.network(values)[:, self.output_index : self.output_index + 1]
            return torch.sigmoid(logits)

    transformed_all = model.transform(expression.values)
    transformed_explained = model.transform(explained_expression.values)
    background_size = min(int(settings.get("background_samples", 100)), len(expression))
    background_index = np.sort(rng.choice(len(expression), size=background_size, replace=False))
    wrapper = SingleOutput(model.network, target_index).to(model.device).eval()
    background = torch.from_numpy(transformed_all[background_index]).to(model.device)
    explained = torch.from_numpy(transformed_explained).to(model.device)
    explainer_name = str(settings.get("neural_explainer", "gradient")).lower()
    if explainer_name == "deep":
        explainer = shap.DeepExplainer(wrapper, background)
    elif explainer_name == "gradient":
        explainer = shap.GradientExplainer(wrapper, background)
    else:
        raise ValueError("neural_explainer must be gradient or deep.")
    return explainer.shap_values(explained), "mutation_probability"


def _two_dimensional(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]  # per-class list (e.g. LightGBM): keep the positive class
    array = np.asarray(values)
    if array.ndim == 3:
        # (samples, features, outputs): a single neural output, or per-class tree
        # contributions. Take the last column, i.e. the positive class.
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional SHAP matrix, received shape {array.shape}.")
    return array
