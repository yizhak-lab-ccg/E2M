from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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


class MultitaskModel:
    def __init__(self, input_size: int, output_size: int, **settings):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.settings = {
            "hidden_layers": list(settings.get("hidden_layers", [512, 256])),
            "dropout_rate": float(settings.get("dropout_rate", 0.3)),
            "learning_rate": float(settings.get("learning_rate", 0.0005)),
            "weight_decay": float(settings.get("weight_decay", 0.0003)),
            "batch_size": int(settings.get("batch_size", 64)),
            "epochs": int(settings.get("epochs", 100)),
            "patience": int(settings.get("patience", 6)),
            "validation_split": float(settings.get("validation_split", 0.15)),
            "gradient_clip": float(settings.get("gradient_clip", 1.0)),
            "use_pos_weight": bool(settings.get("use_pos_weight", True)),
            "standardize_inputs": bool(settings.get("standardize_inputs", True)),
            "random_state": int(settings.get("random_state", 42)),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self._new_network().to(self.device)
        self.scaler: StandardScaler | None = None
        self.history: list[dict] = []

    def _new_network(self) -> MultitaskNetwork:
        return MultitaskNetwork(
            self.input_size,
            self.output_size,
            self.settings["hidden_layers"],
            self.settings["dropout_rate"],
        )

    def fit(self, expression, labels) -> "MultitaskModel":
        x = _float_array(expression)
        y = _float_array(labels)
        if self.settings["standardize_inputs"]:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x).astype(np.float32)
        indices = np.arange(len(x))
        train_idx, validation_idx = train_test_split(
            indices,
            test_size=self.settings["validation_split"],
            random_state=self.settings["random_state"],
            shuffle=True,
        )
        train_loader = self._loader(x[train_idx], y[train_idx], shuffle=True)
        validation_loader = self._loader(x[validation_idx], y[validation_idx], shuffle=False)
        self._train(train_loader, y[train_idx], validation_loader)
        return self

    def fit_full(self, expression, labels) -> "MultitaskModel":
        x = _float_array(expression)
        y = _float_array(labels)
        if self.settings["standardize_inputs"]:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x).astype(np.float32)
        loader = self._loader(x, y, shuffle=True)
        self._train(loader, y, validation_loader=None)
        return self

    def _loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=self.settings["batch_size"], shuffle=shuffle)

    def _train(self, train_loader: DataLoader, train_labels: np.ndarray, validation_loader: DataLoader | None):
        torch.manual_seed(self.settings["random_state"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings["random_state"])
        self.network = self._new_network().to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.settings["learning_rate"],
            weight_decay=self.settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, self.settings["patience"] // 3),
        )
        pos_weight = None
        if self.settings["use_pos_weight"]:
            positives = torch.from_numpy(train_labels).sum(dim=0)
            negatives = len(train_labels) - positives
            pos_weight = torch.where(positives > 0, negatives / positives, torch.ones_like(positives))
            pos_weight = pos_weight.clamp(min=1.0, max=1e6).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_loss = float("inf")
        best_state = None
        stalled = 0
        self.history = []
        for epoch in range(self.settings["epochs"]):
            train_loss = self._epoch(train_loader, criterion, optimizer)
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
            if validation_loader is not None and stalled >= self.settings["patience"]:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()

    def _epoch(self, loader: DataLoader, criterion, optimizer) -> float:
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
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.settings["gradient_clip"])
                    optimizer.step()
                total += float(loss.item()) * len(x_batch)
                count += len(x_batch)
        return total / max(count, 1)

    def transform(self, expression) -> np.ndarray:
        x = _float_array(expression)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        return x

    def predict(self, expression) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform(expression)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.from_numpy(x).to(self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return (probabilities >= 0.5).astype(np.int8), probabilities

    def embeddings(self, expression) -> np.ndarray:
        x = self.transform(expression)
        self.network.eval()
        with torch.no_grad():
            return self.network.encoder(torch.from_numpy(x).to(self.device)).cpu().numpy()

    def head_weights(self) -> np.ndarray:
        return self.network.output.weight.detach().cpu().numpy()

    def save(self, path: str | Path) -> None:
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
                "input_size": self.input_size,
                "output_size": self.output_size,
                "settings": self.settings,
                "network_state": self.network.state_dict(),
                "scaler": scaler,
                "history": self.history,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultitaskModel":
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls(checkpoint["input_size"], checkpoint["output_size"], **checkpoint["settings"])
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
        return model


def _float_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Input contains missing or infinite values.")
    return array
