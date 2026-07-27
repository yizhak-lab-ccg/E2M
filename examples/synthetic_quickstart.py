"""Small offline check of training, prediction, embeddings, and head weights."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e2m.models import MultitaskModel
from e2m.workflow import embed, head_weights, predict


def main() -> None:
    rng = np.random.default_rng(4)
    features = [f"GENE_{index}" for index in range(30)]
    targets = ["MUT_A", "MUT_B", "MUT_C"]
    samples = [f"TRAIN_{index}" for index in range(80)]
    expression = pd.DataFrame(rng.normal(size=(80, 30)), index=samples, columns=features)
    mutations = pd.DataFrame(rng.random((80, 3)) < 0.25, index=samples, columns=targets).astype(int)

    configured_root = os.environ.get("E2M_RUNTIME_DIR")
    root = Path(configured_root) if configured_root else Path(tempfile.mkdtemp(prefix="e2m_example_"))
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "model"
    model_dir.mkdir()
    model = MultitaskModel(30, 3, hidden_layers=[16, 8], epochs=3, batch_size=16)
    model.fit_full(expression.values, mutations.values)
    model.save(model_dir / "model.pt")
    (model_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    (model_dir / "targets.json").write_text(json.dumps(targets), encoding="utf-8")
    np.save(model_dir / "feature_means.npy", expression.mean(axis=0).to_numpy(dtype=np.float32))
    metadata = {
        "cancers": ["SYNTHETIC"],
        "training_config": {"data": {"min_feature_overlap": 0.8}},
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    new_expression = expression.iloc[:5].copy()
    expression_path = root / "expression.csv"
    new_expression.to_csv(expression_path)
    print(predict(model_dir, expression_path, root / "probabilities.csv"))
    print(embed(model_dir, expression_path, root / "embeddings.csv"))
    print(head_weights(model_dir, root / "head_weights.csv"))
    print(f"Outputs: {root}")


if __name__ == "__main__":
    main()
