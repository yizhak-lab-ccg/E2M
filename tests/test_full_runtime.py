from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("shap")
pytest.importorskip("xgboost")

from e2m.config import load_config
from e2m.interpretation import explain_single_target
from e2m.models import MultitaskModel


@pytest.fixture()
def synthetic_model_and_data():
    rng = np.random.default_rng(12)
    features = [f"GENE_{index}" for index in range(8)]
    samples = [f"SAMPLE_{index}" for index in range(48)]
    expression = pd.DataFrame(rng.normal(size=(48, 8)), index=samples, columns=features)
    logits = expression["GENE_0"] - 0.5 * expression["GENE_1"]
    labels = (logits > logits.median()).astype(int).rename("TP53")
    model = MultitaskModel(
        input_size=8,
        output_size=1,
        hidden_layers=[12, 6],
        dropout_rate=0.0,
        epochs=3,
        batch_size=16,
        patience=2,
        random_state=12,
    )
    model.fit_full(expression.values, labels.to_numpy()[:, None])
    return model, expression, labels


def _small_interpretation_config():
    config = load_config()
    config["interpretation"].update(
        {
            "top_features": 3,
            "max_samples": 12,
            "background_samples": 8,
            "neural_explainer": "gradient",
        }
    )
    config["model"]["xgboost"].update({"n_estimators": 4, "max_depth": 2, "n_jobs": 1})
    return config


@pytest.mark.parametrize("method", ["xgboost", "neural"])
def test_single_target_shap_methods(synthetic_model_and_data, tmp_path, method):
    model, expression, labels = synthetic_model_and_data
    output = tmp_path / method
    metadata = explain_single_target(
        model=model,
        expression=expression,
        labels=labels,
        target="TP53",
        target_index=0,
        method=method,
        output_dir=output,
        config=_small_interpretation_config(),
    )
    assert metadata["method"] == method
    assert metadata["n_explained_samples"] == 12
    summary = pd.read_csv(output / "feature_summary.csv")
    sample_values = pd.read_csv(output / "sample_shap_top_features.csv.gz")
    stored_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(summary) == expression.shape[1]
    assert len(sample_values) == 12 * 3
    assert stored_metadata["target"] == "TP53"
