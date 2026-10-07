"""Run a small offline check with all optional e2m dependencies."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e2m.config import load_config
from e2m.interpretation import explain_single_target
from e2m.models import E2MModel
from e2m.training import cross_validate_tmb


def main() -> None:
    base = os.environ.get("E2M_RUNTIME_DIR")
    if base:
        root = Path(base)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="e2m_runtime_"))
    rng = np.random.default_rng(21)
    features = [f"GENE_{index}" for index in range(8)]
    samples = [f"SAMPLE_{index}" for index in range(48)]
    expression = pd.DataFrame(rng.normal(size=(48, 8)), index=samples, columns=features)
    signal = expression["GENE_0"] - 0.5 * expression["GENE_1"]
    labels = (signal > signal.median()).astype(int).rename("TP53")

    model = E2MModel(hidden_layers=[12, 6], dropout_rate=0.0, epochs=3, batch_size=16, patience=2, random_state=21)
    model.fit(expression, labels.to_frame())
    probabilities = model.predict(expression)
    assert probabilities.shape == (48, 1)
    assert model.embed(expression).shape == (48, 6)
    assert model.head_weights().shape == (1, 6)

    restored = E2MModel.load(model.save(root / "model"))
    np.testing.assert_allclose(probabilities, restored.predict(expression), rtol=1e-6, atol=1e-6)

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
    for method in ("xgboost", "neural"):
        output = root / f"shap_{method}"
        _, metadata = explain_single_target(
            model=model,
            expression=expression,
            labels=labels,
            target="TP53",
            target_index=0,
            method=method,
            output=output,
            config=config,
        )
        assert metadata["n_explained_samples"] == 12
        assert len(pd.read_csv(output / "feature_summary.csv")) == 8
        assert len(pd.read_csv(output / "sample_shap_top_features.csv.gz")) == 36

    raw_tmb = np.arange(1, 49)
    tmb = pd.DataFrame({"TMB": raw_tmb, "TMB_log2": np.log2(raw_tmb + 1)}, index=samples)
    cancer = pd.Series(["LUAD"] * len(samples), index=samples)
    config["evaluation"]["cv_folds"] = 3
    _, tmb_metadata = cross_validate_tmb(expression, tmb, cancer, config, output=root / "tmb")
    assert tmb_metadata["n_samples"] == 48
    assert len(pd.read_csv(root / "tmb/oof_predictions.csv")) == 48

    result = {
        "status": "passed",
        "device": str(model.device),
        "samples": len(expression),
        "features": expression.shape[1],
        "shap_methods": ["xgboost", "neural"],
        "tmb_folds": 3,
        "output_dir": str(root),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
