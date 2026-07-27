from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from e2m import Dataset, MutationModel


def _synthetic():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(12)]
    samples = [f"S{i}" for i in range(45)]
    expression = pd.DataFrame(rng.normal(size=(45, 12)), index=samples, columns=genes)
    signal = expression["G0"] - 0.5 * expression["G1"]
    mutations = pd.DataFrame(
        {
            "TP53": (signal > signal.median()).astype(int),
            "KRAS": (rng.random(45) < 0.4).astype(int),
            "EGFR": (rng.random(45) < 0.3).astype(int),
        },
        index=samples,
    )
    return expression, mutations


def _fast(model: MutationModel) -> MutationModel:
    model.hyperparameters.update({"epochs": 2, "batch_size": 16, "hidden_layers": [8]})
    model.config["evaluation"]["cv_folds"] = 3
    return model


def test_dataset_holds_and_subsets_frames():
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    assert data.expression.shape == (45, 12)
    assert list(data.targets) == ["TP53", "KRAS", "EGFR"]
    sub = data.subset(samples=expression.index[:20], targets=["TP53", "KRAS"])
    assert sub.expression.shape == (20, 12)
    assert list(sub.targets) == ["TP53", "KRAS"]
    assert sub.mutations.shape == (20, 2)


def test_model_receives_dataset_fit_and_predict():
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    train = data.subset(samples=expression.index[:35])
    model = _fast(MutationModel()).fit(train)          # receives a Dataset
    probs = model.predict(data.subset(samples=expression.index[35:]))
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (10, 3)
    assert list(probs.columns) == ["TP53", "KRAS", "EGFR"]
    # predicting on a bare expression matrix works too
    assert model.predict(expression.iloc[:4]).shape == (4, 3)


def test_cross_validate_returns_metrics_frame(tmp_path):
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    metrics = _fast(MutationModel()).cross_validate(data)
    assert isinstance(metrics, pd.DataFrame)
    assert {"auprc", "normalized_auprc", "roc_auc"}.issubset(metrics.columns)
    assert list(metrics.index) == ["TP53", "KRAS", "EGFR"]
    # with an output path it also writes the out-of-fold tables
    _fast(MutationModel()).cross_validate(data, output=tmp_path / "cv")
    assert (tmp_path / "cv" / "metrics.csv").exists()
    assert (tmp_path / "cv" / "oof_probabilities.csv").exists()


def test_model_save_load_roundtrip(tmp_path):
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    model = _fast(MutationModel()).fit(data)
    model.save(tmp_path / "model")
    assert (tmp_path / "model" / "model.pt").exists()
    reloaded = MutationModel.load(tmp_path / "model")
    a = model.predict(expression.iloc[:5]).values
    b = reloaded.predict(expression.iloc[:5]).values
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
