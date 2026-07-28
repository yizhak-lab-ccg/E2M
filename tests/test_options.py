from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from e2m import format_options, options, set_verbose
from e2m.backends import TREE_BACKENDS, make_classifier


def test_options_lists_the_main_choices():
    reference = options()
    assert set(reference) >= {
        "expression_dataset",
        "expression_transform",
        "tmb_backend",
        "shap_method",
        "cancers",
    }
    assert tuple(reference["expression_transform"]) == ("auto", "raw", "log1p", "xena")
    assert tuple(reference["tmb_backend"]) == TREE_BACKENDS
    assert tuple(reference["shap_method"]) == (*TREE_BACKENDS, "neural")
    assert "LUAD" in reference["cancers"]


def test_format_options_is_printable_text():
    text = format_options()
    assert isinstance(text, str)
    assert "star_counts" in text and "neural" in text


def test_set_verbose_adds_and_removes_one_handler():
    log = logging.getLogger("e2m")
    set_verbose(True)
    tagged = [h for h in log.handlers if getattr(h, "_e2m_stream", False)]
    assert len(tagged) == 1
    set_verbose(True)  # idempotent: still exactly one
    assert len([h for h in log.handlers if getattr(h, "_e2m_stream", False)]) == 1
    set_verbose(False)
    assert not any(getattr(h, "_e2m_stream", False) for h in log.handlers)


@pytest.mark.parametrize("backend", ["random_forest", "gradient_boosting"])
def test_make_classifier_sklearn_backends(backend):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(30, 5))
    labels = (features[:, 0] > 0).astype(int)
    classifier = make_classifier(backend, {"n_estimators": 10, "random_state": 0})
    classifier.fit(features, labels)
    assert classifier.predict(features).shape == (30,)


def test_make_classifier_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_classifier("not_a_backend")


@pytest.mark.parametrize("method", ["xgboost", "random_forest", "gradient_boosting"])
def test_tree_shap_supports_multiple_backends(method, tmp_path):
    pytest.importorskip("shap")
    from e2m.config import resolve_config
    from e2m.interpretation import explain_single_target

    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(8)]
    samples = [f"S{i}" for i in range(60)]
    expression = pd.DataFrame(rng.normal(size=(60, 8)), index=samples, columns=genes)
    labels = pd.Series((expression["G0"] > 0).astype(int).to_numpy(), index=samples)

    config = resolve_config()
    # model is unused for the tree methods, so a fitted network is not required here.
    metadata = explain_single_target(
        None, expression, labels, "G0_target", 0, method, tmp_path / method, config
    )
    assert metadata["method"] == method
    summary = pd.read_csv(tmp_path / method / "feature_summary.csv")
    assert len(summary) == expression.shape[1]
    # the driver feature should rank near the top for every backend
    assert "G0" in set(summary.sort_values("mean_abs_shap", ascending=False).head(3)["feature"])
