"""The :class:`E2MModel` — a multitask model you fit on a :class:`Dataset`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import resolve_config
from .models import MultitaskModel as _MultitaskNetwork
from .training import model_settings, run_cross_validation, write_cross_validation


def _expression(data) -> pd.DataFrame:
    """Accept a Dataset (duck-typed) or a samples-by-genes DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    return data.expression


def _expression_mutations(data, mutations):
    if isinstance(data, pd.DataFrame):
        return data, mutations
    return data.expression, data.mutations if mutations is None else mutations


class E2MModel:
    """Predict gene mutation status from RNA expression with a multitask network.

    Fit it on a :class:`Dataset` (or expression/mutation frames), then predict on
    any dataset or samples-by-genes matrix; input genes are aligned to the training
    features by symbol. Model settings come from ``config`` (the packaged defaults by
    default); pass keyword hyperparameters to override, e.g. ``E2MModel(epochs=50)``.
    """

    def __init__(self, config: Optional[dict] = None, **hyperparameters):
        self.config = config if config is not None else resolve_config()
        self.hyperparameters = hyperparameters
        self.network: Optional[_MultitaskNetwork] = None
        self.features: Optional[list] = None
        self.targets: Optional[list] = None
        self.feature_means: Optional[np.ndarray] = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, data, mutations=None) -> "E2MModel":
        """Train on a :class:`Dataset` (or ``expression`` + ``mutations`` frames)."""
        expression, mutations = _expression_mutations(data, mutations)
        if mutations is None:
            raise ValueError("E2MModel.fit needs mutation labels (a Dataset with mutations, or mutations=).")
        settings = model_settings(self.config, self.hyperparameters)
        network = _MultitaskNetwork(expression.shape[1], mutations.shape[1], **settings)
        network.fit_full(expression.values, mutations.values)
        self.network = network
        self.features = expression.columns.tolist()
        self.targets = mutations.columns.tolist()
        self.feature_means = expression.mean(axis=0).to_numpy(dtype=np.float32)
        self.metadata = {
            "n_samples": len(expression),
            "n_features": expression.shape[1],
            "n_targets": mutations.shape[1],
            "model": settings,
        }
        # Record the training cohorts so a saved bundle is self-describing (the CLI's
        # explain re-loads them). Available whenever fit gets a Dataset with cancer labels.
        cancer = getattr(data, "cancer", None)
        if cancer is not None:
            self.metadata["cancers"] = sorted(pd.Series(cancer).dropna().unique().tolist())
        return self

    # -------------------------------------------------------------- predict
    def _align(self, data, min_feature_overlap):
        from .workflow import align_expression

        expression = _expression(data)
        default = self.config["data"].get("min_feature_overlap", 0.8)
        overlap = default if min_feature_overlap is None else min_feature_overlap
        aligned, _ = align_expression(expression, self.features, self.feature_means, overlap)
        return aligned

    def predict(self, data, min_feature_overlap=None) -> pd.DataFrame:
        """Mutation probabilities (samples x targets) for a dataset or expression matrix."""
        self._check_fitted()
        aligned = self._align(data, min_feature_overlap)
        _, probabilities = self.network.predict(aligned.values)
        return pd.DataFrame(probabilities, index=aligned.index, columns=self.targets)

    def embed(self, data, min_feature_overlap=None) -> pd.DataFrame:
        """Shared-encoder embeddings (samples x embedding dims)."""
        self._check_fitted()
        aligned = self._align(data, min_feature_overlap)
        values = self.network.embeddings(aligned.values)
        columns = [f"embedding_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=aligned.index, columns=columns)

    def head_weights(self) -> pd.DataFrame:
        """One output-head weight vector per mutation target."""
        self._check_fitted()
        values = self.network.head_weights()
        columns = [f"weight_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=self.targets, columns=columns)

    # ------------------------------------------------------------ evaluate
    def cross_validate(self, data, mutations=None, cancer=None, output=None) -> pd.DataFrame:
        """Held-out cross-validation of this model's settings; returns the metrics table.

        Writes the out-of-fold probabilities/predictions to ``output`` if given.
        """
        if isinstance(data, pd.DataFrame):
            expression = data
        else:
            expression, mutations, cancer = data.expression, (mutations or data.mutations), (
                cancer if cancer is not None else data.cancer
            )
        if mutations is None:
            raise ValueError("cross_validate needs mutation labels (a Dataset with mutations, or mutations=).")
        if cancer is None:
            cancer = pd.Series("cohort", index=expression.index, name="cancer")
        metrics, prob_frame, pred_frame, fold_frame, metadata = run_cross_validation(
            expression, mutations, cancer, self.config, self.hyperparameters
        )
        if output is not None:
            write_cross_validation(output, metrics, prob_frame, pred_frame, fold_frame, metadata)
        return metrics

    # ----------------------------------------------------------- persistence
    def save(self, output_dir) -> Path:
        """Write the model bundle (same layout the CLI's ``predict``/``embed`` read)."""
        self._check_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.network.save(output / "model.pt")
        np.save(output / "feature_means.npy", self.feature_means)
        (output / "features.json").write_text(json.dumps(self.features), encoding="utf-8")
        (output / "targets.json").write_text(json.dumps(self.targets), encoding="utf-8")
        bundle_metadata = {**self.metadata, "training_config": self.config}
        (output / "model_metadata.json").write_text(json.dumps(bundle_metadata, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, model_dir) -> "E2MModel":
        """Load a model bundle previously written by :meth:`save` (or the CLI ``train``)."""
        directory = Path(model_dir)
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        model = cls(config=metadata.get("training_config"))
        model.network = _MultitaskNetwork.load(directory / "model.pt")
        model.features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
        model.targets = json.loads((directory / "targets.json").read_text(encoding="utf-8"))
        model.feature_means = np.load(directory / "feature_means.npy")
        model.metadata = metadata
        return model

    def _check_fitted(self) -> None:
        if self.network is None:
            raise ValueError("Model is not fitted. Call fit(dataset) or E2MModel.load(dir).")

    def __repr__(self) -> str:
        state = "fitted" if self.network is not None else "unfitted"
        targets = 0 if self.targets is None else len(self.targets)
        return f"E2MModel({state}, targets={targets})"


def _expression_tmb(data, tmb):
    if isinstance(data, pd.DataFrame):
        return data, tmb
    return data.expression, (data.tmb if tmb is None else tmb)


class TmbModel:
    """Predict tumor mutational burden — ``log2(TMB + 1)`` — from expression.

    Fit it on a :class:`Dataset` (which carries ``tmb``) or on ``expression`` + a
    ``tmb`` table with a ``TMB_log2`` column. The tree ``ml_model`` is one of
    ``xgboost`` (default), ``lightgbm``, ``random_forest``, or ``gradient_boosting``;
    each reads its parameter block from ``config['model']``. Samples without a TMB
    label are dropped. ``predict`` returns ``TMB_log2_pred`` and ``TMB_pred``.
    """

    def __init__(self, ml_model: Optional[str] = None, config: Optional[dict] = None, **params):
        self.config = config if config is not None else resolve_config()
        self.ml_model = ml_model or self.config["model"].get("tmb_ml_model", "xgboost")
        self.params = params
        self.regressor = None
        self.features: Optional[list] = None
        self.feature_means: Optional[np.ndarray] = None

    def _resolved_params(self) -> dict:
        from .tmb import tmb_ml_model_params

        _, base = tmb_ml_model_params(self.config, self.ml_model)
        base.update(self.params)
        return base

    def fit(self, data, tmb=None) -> "TmbModel":
        from .backends import make_regressor

        expression, labels = _expression_tmb(data, tmb)
        if labels is None:
            raise ValueError("TmbModel.fit needs TMB labels (a Dataset with tmb, or tmb=).")
        index = labels.index[labels["TMB_log2"].notna()]
        expression = expression.loc[index]
        self.regressor = make_regressor(self.ml_model, self._resolved_params())
        self.regressor.fit(expression.values, labels.loc[index, "TMB_log2"].to_numpy())
        self.features = expression.columns.tolist()
        self.feature_means = expression.mean(axis=0).to_numpy(dtype=np.float32)
        return self

    def predict(self, data) -> pd.DataFrame:
        if self.regressor is None:
            raise ValueError("Model is not fitted. Call fit(dataset).")
        expression = _expression(data)
        aligned = expression.reindex(columns=self.features)
        aligned = aligned.fillna(pd.Series(self.feature_means, index=self.features)).astype(np.float32)
        predicted = self.regressor.predict(aligned.values)
        return pd.DataFrame(
            {"TMB_log2_pred": predicted, "TMB_pred": np.maximum(np.power(2.0, predicted) - 1.0, 0.0)},
            index=aligned.index,
        )

    def cross_validate(self, data, tmb=None, cancer=None, output=None) -> pd.DataFrame:
        """Held-out TMB regression; returns the per-cancer + overall summary table."""
        from .tmb import run_tmb_cv, write_tmb

        expression, labels = _expression_tmb(data, tmb)
        if labels is None:
            raise ValueError("cross_validate needs TMB labels (a Dataset with tmb, or tmb=).")
        if cancer is None:
            cancer = getattr(data, "cancer", None)
        if cancer is None:
            cancer = pd.Series("cohort", index=expression.index, name="cancer")
        index = labels.index[labels["TMB_log2"].notna()]
        summary, predictions, fold_metrics, metadata = run_tmb_cv(
            expression.loc[index],
            labels.loc[index],
            cancer.loc[index],
            self.config,
            ml_model=self.ml_model,
            params=self._resolved_params(),
        )
        if output is not None:
            write_tmb(output, summary, predictions, fold_metrics, metadata)
        return summary

    def __repr__(self) -> str:
        state = "fitted" if self.regressor is not None else "unfitted"
        return f"TmbModel(ml_model={self.ml_model!r}, {state})"
