"""Small offline check of training, prediction, embeddings, and head weights."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e2m.models import E2MModel


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
    model_dir = E2MModel(hidden_layers=[16, 8], epochs=3, batch_size=16).fit(expression, mutations).save(root / "model")

    model = E2MModel.load(model_dir)
    new_expression = expression.iloc[:5]
    print(model.predict(new_expression))
    print(model.embed(new_expression))
    print(model.head_weights())
    print(f"Model bundle: {model_dir}")


if __name__ == "__main__":
    main()
