from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from e2m import Dataset, E2MModel, TmbModel


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


def _fast(model: E2MModel) -> E2MModel:
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
    model = _fast(E2MModel()).fit(train)          # receives a Dataset
    probs = model.predict(data.subset(samples=expression.index[35:]))
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (10, 3)
    assert list(probs.columns) == ["TP53", "KRAS", "EGFR"]
    # predicting on a bare expression matrix works too
    assert model.predict(expression.iloc[:4]).shape == (4, 3)


def test_cross_validate_returns_metrics_frame(tmp_path):
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    metrics = _fast(E2MModel()).cross_validate(data)
    assert isinstance(metrics, pd.DataFrame)
    assert {"auprc", "normalized_auprc", "roc_auc"}.issubset(metrics.columns)
    assert list(metrics.index) == ["TP53", "KRAS", "EGFR"]
    # with an output path it also writes the out-of-fold tables
    _fast(E2MModel()).cross_validate(data, output=tmp_path / "cv")
    assert (tmp_path / "cv" / "metrics.csv").exists()
    assert (tmp_path / "cv" / "oof_probabilities.csv").exists()


def test_model_explain_runs_on_in_memory_data(tmp_path):
    pytest.importorskip("xgboost")  # xgboost Tree SHAP uses native pred_contribs, no shap needed
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    model = _fast(E2MModel()).fit(data)
    summary = model.explain(data, target="TP53", method="xgboost", output=tmp_path / "shap")
    assert isinstance(summary, pd.DataFrame)
    assert {"feature", "mean_abs_shap", "rank"}.issubset(summary.columns)
    assert len(summary) == expression.shape[1]
    # TP53 is driven by G0/G1 in the synthetic data, so one of them should rank at the top
    assert summary.iloc[0]["feature"] in {"G0", "G1"}
    assert (tmp_path / "shap" / "feature_summary.csv").exists()


def test_cross_validate_accepts_explicit_mutation_labels():
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    # Passing mutations= explicitly (a DataFrame) must not raise "truth value is ambiguous".
    metrics = _fast(E2MModel()).cross_validate(data, mutations=mutations)
    assert isinstance(metrics, pd.DataFrame)
    assert list(metrics.index) == ["TP53", "KRAS", "EGFR"]


def test_model_save_load_roundtrip(tmp_path):
    expression, mutations = _synthetic()
    data = Dataset(expression=expression, mutations=mutations)
    model = _fast(E2MModel()).fit(data)
    model.save(tmp_path / "model")
    assert (tmp_path / "model" / "model.pt").exists()
    reloaded = E2MModel.load(tmp_path / "model")
    a = model.predict(expression.iloc[:5]).values
    b = reloaded.predict(expression.iloc[:5]).values
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def _dataset_with_tmb():
    expression, mutations = _synthetic()
    rng = np.random.default_rng(1)
    raw = rng.integers(1, 60, size=len(expression))
    tmb = pd.DataFrame({"TMB": raw, "TMB_log2": np.log2(raw + 1.0)}, index=expression.index)
    return Dataset(expression=expression, mutations=mutations, tmb=tmb)


@pytest.mark.parametrize("ml_model", ["random_forest", "gradient_boosting"])
def test_tmb_model_ml_models_fit_predict_cv(ml_model, tmp_path):
    data = _dataset_with_tmb()
    model = TmbModel(ml_model=ml_model, n_estimators=5, random_state=0)
    model.config["evaluation"]["cv_folds"] = 3
    model.fit(data)
    pred = model.predict(data)
    assert list(pred.columns) == ["TMB_log2_pred", "TMB_pred"]
    assert pred.shape[0] == len(data.expression)
    summary = model.cross_validate(data, output=tmp_path / f"tmb_{ml_model}")
    assert (summary["cancer"] == "__OVERALL__").any()
    assert (tmp_path / f"tmb_{ml_model}" / "summary.csv").exists()


def test_tmb_default_ml_model_is_xgboost():
    pytest.importorskip("xgboost")
    data = _dataset_with_tmb()
    model = TmbModel(n_estimators=3, max_depth=2, n_jobs=1)
    assert model.ml_model == "xgboost"
    model.fit(data)
    assert model.predict(data).shape[0] == len(data.expression)


def test_tmb_predict_enforces_feature_overlap():
    data = _dataset_with_tmb()
    model = TmbModel(ml_model="random_forest", n_estimators=5, random_state=0).fit(data)
    disjoint = data.expression.rename(columns=lambda name: name + "_other")
    with pytest.raises(ValueError, match="feature overlap"):
        model.predict(disjoint)
