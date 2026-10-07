from __future__ import annotations

import copy
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .data import Dataset, align_expression

# Deterministic cuBLAS needs this before CUDA starts; an existing value is kept.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

TREE_MODELS = ("xgboost", "lightgbm", "random_forest", "gradient_boosting")
_TREE_CLASSES = {
    "xgboost": ("xgboost", "XGB"),
    "lightgbm": ("lightgbm", "LGBM"),
    "random_forest": ("sklearn.ensemble", "RandomForest"),
    "gradient_boosting": ("sklearn.ensemble", "GradientBoosting"),
}


def make_tree_model(ml_model: str, kind: str, params: dict | None = None):
    """Build an unfitted tree model; `kind` is "classifier" or "regressor"."""
    if ml_model not in TREE_MODELS:
        raise ValueError(f"ml_model must be one of {TREE_MODELS}; got {ml_model!r}.")
    module_name, prefix = _TREE_CLASSES[ml_model]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f'{ml_model} needs {module_name} (LightGBM: pip install -e ".[lightgbm]").') from exc
    return getattr(module, prefix + kind.capitalize())(**dict(params or {}))


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


def _as_dataset(data, mutations=None, tmb=None, cancer=None) -> Dataset:
    """Wrap a samples-by-genes frame as a Dataset; explicit labels replace the Dataset's own."""
    if isinstance(data, Dataset):
        mutations = data.mutations if mutations is None else mutations
        tmb = data.tmb if tmb is None else tmb
        cancer = data.cancer if cancer is None else cancer
        data = data.expression
    return Dataset(data, mutations, tmb, cancer)


class MultitaskNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int], dropout_rate: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for hidden in hidden_layers:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.GELU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(previous, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(values))


class E2MModel:
    """Multitask network predicting mutation status from expression; settings come from `config` plus keyword overrides."""

    def __init__(self, config: Optional[dict] = None, **hyperparameters):
        self.config = config if config is not None else load_config()
        self.hyperparameters = hyperparameters
        self.settings = model_settings(self.config, hyperparameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[MultitaskNetwork] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: list[dict] = []
        self.features: Optional[list] = None
        self.targets: Optional[list] = None
        self.feature_means: Optional[np.ndarray] = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, data, mutations=None) -> "E2MModel":
        """Train; with `early_stopping` (default) a validation split picks the best epoch."""
        dataset = _as_dataset(data, mutations)
        if dataset.mutations is None:
            raise ValueError("E2MModel.fit needs mutation labels (a Dataset with mutations, or mutations=).")
        missing = int(dataset.mutations.isna().to_numpy().sum())
        if missing:
            raise ValueError(
                f"{missing} mutation labels are missing (not profiled) in "
                f"{int(dataset.mutations.isna().any().sum())} targets; training needs complete labels."
            )
        x = _float_array(dataset.expression)
        y = _float_array(dataset.mutations)
        if self.settings["standardize_inputs"]:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x).astype(np.float32)
        if self.settings.get("early_stopping", True):
            train_idx, validation_idx = train_test_split(
                np.arange(len(x)),
                test_size=self.settings["validation_split"],
                random_state=self.settings["random_state"],
                shuffle=True,
            )
            self._train(
                self._loader(x[train_idx], y[train_idx], shuffle=True),
                y[train_idx],
                self._loader(x[validation_idx], y[validation_idx], shuffle=False),
            )
        else:
            self._train(self._loader(x, y, shuffle=True), y, validation_loader=None)

        self.features = dataset.expression.columns.tolist()
        self.targets = dataset.mutations.columns.tolist()
        self.feature_means = dataset.expression.mean(axis=0).to_numpy(dtype=np.float32)
        self.metadata = {
            "n_samples": len(x),
            "n_features": x.shape[1],
            "n_targets": y.shape[1],
            "model": self.settings,
            "device": str(self.device),
        }
        # Recorded so a saved bundle knows its cohorts, in loading order (the CLI's explain reloads them).
        if isinstance(data, Dataset):
            self.metadata["cancers"] = data.cancer.dropna().unique().tolist()
        self.metadata["training_config"] = self.config
        return self

    def _loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=int(self.settings["batch_size"]), shuffle=shuffle)

    def _new_network(self, input_size: int, output_size: int) -> MultitaskNetwork:
        return MultitaskNetwork(
            input_size,
            output_size,
            list(self.settings["hidden_layers"]),
            float(self.settings["dropout_rate"]),
        )

    def _train(self, train_loader: DataLoader, train_labels: np.ndarray, validation_loader: DataLoader | None):
        settings = self.settings
        torch.use_deterministic_algorithms(bool(settings.get("deterministic", True)))
        torch.manual_seed(settings["random_state"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings["random_state"])
        self.network = self._new_network(train_loader.dataset.tensors[0].shape[1], train_labels.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, int(settings["patience"]) // 3),
        )
        pos_weight = None
        if settings["use_pos_weight"]:
            positives = torch.from_numpy(train_labels).sum(dim=0)
            negatives = len(train_labels) - positives
            pos_weight = torch.where(positives > 0, negatives / positives, torch.ones_like(positives))
            pos_weight = pos_weight.clamp(min=1.0, max=1e6).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_loss = float("inf")
        best_state = None
        stalled = 0
        self.history = []
        for epoch in range(int(settings["epochs"])):
            train_loss = self._epoch(train_loader, criterion, optimizer, settings["gradient_clip"])
            row = {"epoch": epoch + 1, "train_loss": train_loss}
            if validation_loader is not None:
                validation_loss = self._epoch(validation_loader, criterion, None)
                scheduler.step(validation_loss)
                row["validation_loss"] = validation_loss
                row["learning_rate"] = optimizer.param_groups[0]["lr"]
                if validation_loss + 1e-5 < best_loss:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.network.state_dict())
                    stalled = 0
                else:
                    stalled += 1
            self.history.append(row)
            if validation_loader is not None and stalled >= int(settings["patience"]):
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()

    def _epoch(self, loader: DataLoader, criterion, optimizer, gradient_clip=None) -> float:
        training = optimizer is not None
        self.network.train(training)
        total = 0.0
        count = 0
        with torch.set_grad_enabled(training):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                if training:
                    optimizer.zero_grad()
                loss = criterion(self.network(x_batch), y_batch)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(gradient_clip))
                    optimizer.step()
                total += float(loss.item()) * len(x_batch)
                count += len(x_batch)
        return total / max(count, 1)

    # -------------------------------------------------------------- predict
    def transform(self, expression) -> np.ndarray:
        """Training-scaled float32 array of an already-aligned expression matrix."""
        x = _float_array(expression)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        return x

    def _align(self, data, min_feature_overlap) -> pd.DataFrame:
        default = self.config["data"].get("min_feature_overlap", 0.8)
        overlap = default if min_feature_overlap is None else min_feature_overlap
        aligned, _ = align_expression(_as_dataset(data).expression, self.features, self.feature_means, overlap)
        return aligned

    def predict(self, data, min_feature_overlap=None) -> pd.DataFrame:
        """Mutation probabilities (samples x targets)."""
        self._check_fitted()
        aligned = self._align(data, min_feature_overlap)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.from_numpy(self.transform(aligned.values)).to(self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return pd.DataFrame(probabilities, index=aligned.index, columns=self.targets)

    def embed(self, data, min_feature_overlap=None) -> pd.DataFrame:
        """Shared-encoder embeddings (samples x embedding dims)."""
        self._check_fitted()
        aligned = self._align(data, min_feature_overlap)
        self.network.eval()
        with torch.no_grad():
            values = self.network.encoder(torch.from_numpy(self.transform(aligned.values)).to(self.device)).cpu().numpy()
        columns = [f"embedding_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=aligned.index, columns=columns)

    def head_weights(self) -> pd.DataFrame:
        """One output-head weight vector per mutation target."""
        self._check_fitted()
        values = self.network.output.weight.detach().cpu().numpy()
        columns = [f"weight_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=self.targets, columns=columns)

    # ----------------------------------------------------------- interpret
    def explain(self, data, target, method="xgboost", output=None, min_feature_overlap=1.0) -> pd.DataFrame:
        """SHAP feature summary for one target on in-memory `data` (needs its mutation labels)."""
        self._check_fitted()
        from .interpretation import explain_single_target

        if self.targets is None or target not in self.targets:
            raise ValueError(f"Target {target!r} is not in this model's targets.")
        mutations = _as_dataset(data).mutations
        if mutations is None or target not in mutations.columns:
            raise ValueError(f"explain needs mutation labels for {target!r} (pass a Dataset with mutations).")
        aligned = self._align(data, min_feature_overlap)
        labels = mutations[target].reindex(aligned.index)
        summary, _ = explain_single_target(
            self, aligned, labels, target, self.targets.index(target), method, self.config, output
        )
        return summary

    # ------------------------------------------------------------ evaluate
    def cross_validate(self, data, mutations=None, cancer=None, output=None, groups=None, folds=None) -> pd.DataFrame:
        """Pooled out-of-fold metrics; writes predictions, per-fold metrics, and fold models to `output` if given."""
        from .training import cross_validate_mutations

        dataset = _as_dataset(data, mutations, cancer=cancer)
        if dataset.mutations is None:
            raise ValueError("cross_validate needs mutation labels (a Dataset with mutations, or mutations=).")
        metrics, _ = cross_validate_mutations(
            dataset.expression,
            dataset.mutations,
            dataset.cancer,
            self.config,
            self.hyperparameters,
            output,
            groups,
            folds,
        )
        return metrics

    # ----------------------------------------------------------- persistence
    def save(self, output_dir) -> Path:
        """Write the model bundle: weights and scaler, feature and target order, feature means, metadata, history."""
        self._check_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        scaler = None
        if self.scaler is not None:
            scaler = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(self.scaler.n_features_in_),
            }
        torch.save(
            {
                "input_size": len(self.features),
                "output_size": len(self.targets),
                "settings": self.settings,
                "network_state": self.network.state_dict(),
                "scaler": scaler,
                "history": self.history,
            },
            output / "model.pt",
        )
        np.save(output / "feature_means.npy", self.feature_means)
        (output / "features.json").write_text(json.dumps(self.features), encoding="utf-8")
        (output / "targets.json").write_text(json.dumps(self.targets), encoding="utf-8")
        (output / "model_metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        pd.DataFrame(self.history).to_csv(output / "training_history.csv", index=False)
        return output

    @classmethod
    def load(cls, model_dir) -> "E2MModel":
        """Load a bundle written by `save` or the CLI `train`."""
        directory = Path(model_dir)
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        model = cls(config=metadata.get("training_config"))
        checkpoint = torch.load(directory / "model.pt", map_location="cpu")
        model.settings = checkpoint["settings"]
        model.network = model._new_network(checkpoint["input_size"], checkpoint["output_size"])
        model.network.load_state_dict(checkpoint["network_state"])
        model.network.to(model.device).eval()
        if checkpoint.get("scaler") is not None:
            state = checkpoint["scaler"]
            model.scaler = StandardScaler()
            model.scaler.mean_ = np.asarray(state["mean"])
            model.scaler.scale_ = np.asarray(state["scale"])
            model.scaler.var_ = np.asarray(state["var"])
            model.scaler.n_features_in_ = int(state["n_features_in"])
        model.history = checkpoint.get("history", [])
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


class TmbModel:
    """Tree regressor predicting log2(TMB + 1) from expression; samples without a TMB label are dropped."""

    def __init__(self, ml_model: Optional[str] = None, config: Optional[dict] = None, **params):
        self.config = config if config is not None else load_config()
        self.ml_model = ml_model or self.config["model"].get("tmb_ml_model", "xgboost")
        self.params = params
        self.regressor = None
        self.features: Optional[list] = None
        self.feature_means: Optional[np.ndarray] = None
        self.metadata: dict = {}

    def _resolved_params(self) -> dict:
        _, base = tmb_ml_model_params(self.config, self.ml_model)
        base.update(self.params)
        return base

    def fit(self, data, tmb=None) -> "TmbModel":
        dataset = _as_dataset(data, tmb=tmb)
        if dataset.tmb is None:
            raise ValueError("TmbModel.fit needs TMB labels (a Dataset with tmb, or tmb=).")
        index = dataset.tmb.index[dataset.tmb["TMB_log2"].notna()]
        expression = dataset.expression.loc[index]
        self.regressor = make_tree_model(self.ml_model, "regressor", self._resolved_params())
        self.regressor.fit(expression.values, dataset.tmb.loc[index, "TMB_log2"].to_numpy())
        self.features = expression.columns.tolist()
        self.feature_means = expression.mean(axis=0).to_numpy(dtype=np.float32)
        self.metadata = {
            "n_samples": len(expression),
            "n_features": expression.shape[1],
            "target": "log2(coding TMB + 1)",
            "ml_model": self.ml_model,
            "params": self._resolved_params(),
            "training_config": self.config,
        }
        return self

    def save(self, output_dir) -> Path:
        """Write the regressor (XGBoost JSON or joblib), feature order, feature means, and metadata."""
        if self.regressor is None:
            raise ValueError("Model is not fitted. Call fit(dataset).")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.ml_model == "xgboost":
            self.regressor.save_model(output / "model.json")
        else:
            joblib.dump(self.regressor, output / "model.joblib")
        np.save(output / "feature_means.npy", self.feature_means)
        (output / "features.json").write_text(json.dumps(self.features), encoding="utf-8")
        (output / "model_metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, model_dir) -> "TmbModel":
        """Load a model written by `save`."""
        directory = Path(model_dir)
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        model = cls(metadata["ml_model"], config=metadata.get("training_config"), **metadata["params"])
        if model.ml_model == "xgboost":
            model.regressor = make_tree_model("xgboost", "regressor", metadata["params"])
            model.regressor.load_model(directory / "model.json")
        else:
            model.regressor = joblib.load(directory / "model.joblib")
        model.features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
        model.feature_means = np.load(directory / "feature_means.npy")
        model.metadata = metadata
        return model

    def predict(self, data, min_feature_overlap=None) -> pd.DataFrame:
        """Predicted TMB_log2_pred and TMB_pred per sample."""
        if self.regressor is None:
            raise ValueError("Model is not fitted. Call fit(dataset).")
        threshold = self.config["data"].get("min_feature_overlap", 0.8) if min_feature_overlap is None else min_feature_overlap
        aligned, _ = align_expression(_as_dataset(data).expression, self.features, self.feature_means, threshold)
        predicted = self.regressor.predict(aligned.values)
        return pd.DataFrame(
            {"TMB_log2_pred": predicted, "TMB_pred": np.maximum(np.power(2.0, predicted) - 1.0, 0.0)},
            index=aligned.index,
        )

    def cross_validate(self, data, tmb=None, cancer=None, output=None, groups=None, folds=None) -> pd.DataFrame:
        """Out-of-fold TMB regression; returns the per-cancer and overall (pooled) summary."""
        from .training import _aligned, cross_validate_tmb

        dataset = _as_dataset(data, tmb=tmb, cancer=cancer)
        if dataset.tmb is None:
            raise ValueError("cross_validate needs TMB labels (a Dataset with tmb, or tmb=).")
        index = dataset.tmb.index[dataset.tmb["TMB_log2"].notna()]
        groups = _aligned(groups, dataset.samples)
        folds = _aligned(folds, dataset.samples)
        summary, _ = cross_validate_tmb(
            dataset.expression.loc[index],
            dataset.tmb.loc[index],
            dataset.cancer.loc[index],
            self.config,
            ml_model=self.ml_model,
            params=self._resolved_params(),
            output=output,
            groups=None if groups is None else groups.loc[index],
            folds=None if folds is None else folds.loc[index],
        )
        return summary

    def __repr__(self) -> str:
        state = "fitted" if self.regressor is not None else "unfitted"
        return f"TmbModel(ml_model={self.ml_model!r}, {state})"


def _float_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Input contains missing or infinite values.")
    return array
