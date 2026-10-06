from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import _read_yaml, deep_update, load_config
from .data import XenaTCGALoader, align_expression, normalize_cancers


def download(
    cancers,
    config_path=None,
    data_dir=None,
    data_overrides=None,
    use_cache=True,
    output_dir=None,
) -> dict:
    config, selected = _data_config(config_path, cancers, data_overrides)
    data = XenaTCGALoader(config, data_dir=data_dir, use_cache=use_cache).prepare(selected)
    summary = _data_summary(data)
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        data.expression.to_csv(output / "expression.csv.gz", compression="gzip")
        data.mutations.to_csv(output / "mutations.csv.gz", compression="gzip")
        data.cancer.rename("cancer").to_csv(output / "samples.csv")
        summary["output_dir"] = str(output)
    return summary


def cross_validate(
    cancers,
    output_dir,
    config_path=None,
    data_dir=None,
    data_overrides=None,
    model_overrides=None,
    use_cache=True,
) -> dict:
    from .training import cross_validate_mutations

    config, selected = _data_config(config_path, cancers, data_overrides)
    data = XenaTCGALoader(config, data_dir=data_dir, use_cache=use_cache).prepare(selected)
    _, metadata = cross_validate_mutations(
        data.expression, data.mutations, data.cancer, config, model_overrides, output_dir
    )
    return metadata


def train(
    cancers,
    output_dir,
    config_path=None,
    data_dir=None,
    data_overrides=None,
    model_overrides=None,
    use_cache=True,
) -> dict:
    from .training import train_full

    config, selected = _data_config(config_path, cancers, data_overrides)
    loader = XenaTCGALoader(config, data_dir=data_dir, use_cache=use_cache)
    data = loader.prepare(selected)
    metadata = {
        "cancers": selected,
        "data_dir": str(loader.data_dir),
        "training_config": config,
    }
    return train_full(
        data.expression,
        data.mutations,
        config,
        output_dir,
        metadata,
        model_overrides=model_overrides,
    )


def predict_tmb(
    cancers,
    output_dir,
    config_path=None,
    data_dir=None,
    data_overrides=None,
) -> dict:
    from .training import cross_validate_tmb

    config, selected = _data_config(config_path, cancers, data_overrides)
    loader = XenaTCGALoader(config, data_dir=data_dir, use_cache=False)
    expression, tmb, cancer = loader.prepare_tmb(selected)
    return cross_validate_tmb(expression, tmb, cancer, config, output=output_dir)[1]


def predict(model_dir, expression_path, output_path, min_feature_overlap=None) -> dict:
    model, features, targets, means, metadata = _load_bundle(model_dir)
    expression = _read_expression(expression_path)
    aligned, overlap = align_expression(
        expression,
        features,
        means,
        _bundle_overlap(metadata) if min_feature_overlap is None else min_feature_overlap,
    )
    _, probabilities = model.predict(aligned.values)
    output = pd.DataFrame(probabilities, index=aligned.index, columns=targets)
    _write_frame(output, output_path)
    return {"n_samples": len(output), "n_targets": len(targets), "feature_overlap": overlap, "output": str(output_path)}


def embed(model_dir, expression_path, output_path, min_feature_overlap=None) -> dict:
    model, features, _, means, metadata = _load_bundle(model_dir)
    expression = _read_expression(expression_path)
    aligned, overlap = align_expression(
        expression,
        features,
        means,
        _bundle_overlap(metadata) if min_feature_overlap is None else min_feature_overlap,
    )
    values = model.embeddings(aligned.values)
    output = pd.DataFrame(values, index=aligned.index, columns=[f"embedding_{i}" for i in range(values.shape[1])])
    _write_frame(output, output_path)
    return {"n_samples": len(output), "embedding_size": values.shape[1], "feature_overlap": overlap, "output": str(output_path)}


def head_weights(model_dir, output_path) -> dict:
    model, _, targets, _, _ = _load_bundle(model_dir)
    values = model.head_weights()
    output = pd.DataFrame(values, index=targets, columns=[f"weight_{i}" for i in range(values.shape[1])])
    _write_frame(output, output_path)
    return {"n_targets": len(targets), "embedding_size": values.shape[1], "output": str(output_path)}


def explain(
    model_dir,
    target,
    method,
    output_dir,
    config_path=None,
    data_dir=None,
    use_cache=True,
) -> dict:
    from .interpretation import explain_single_target

    model, features, targets, means, metadata = _load_bundle(model_dir)
    if target not in targets:
        raise ValueError(f"Target {target!r} is not in this model. Available targets are stored in targets.json.")
    config = metadata.get("training_config") or load_config()
    if config_path:
        config = deep_update(config, _read_yaml(Path(config_path)))
    cancers = metadata.get("cancers")
    if not cancers:
        raise ValueError(
            "This model bundle does not record its training cohorts, so SHAP cannot re-load the "
            "expression data. Train it with cancer-labelled data (Dataset.from_tcga or the CLI "
            "`e2m train`), or pass a config that supplies data.cancers."
        )
    loader = XenaTCGALoader(config, data_dir=data_dir or metadata.get("data_dir"), use_cache=use_cache)
    data = loader.prepare(cancers)
    expression, _ = align_expression(data.expression, features, means, min_overlap=1.0)
    labels = data.mutations[target].reindex(expression.index)
    return explain_single_target(
        model,
        expression,
        labels,
        target,
        targets.index(target),
        method,
        output_dir,
        config,
    )


def _data_config(config_path, cancers, data_overrides) -> tuple[dict[str, Any], list[str]]:
    config = load_config(config_path, {"data": data_overrides or {}})
    selected = normalize_cancers(cancers or config["data"].get("cancers"))
    config["data"]["cancers"] = selected
    return config, selected


def _load_bundle(model_dir):
    from .models import MultitaskModel

    directory = Path(model_dir)
    model = MultitaskModel.load(directory / "model.pt")
    features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    targets = json.loads((directory / "targets.json").read_text(encoding="utf-8"))
    means = np.load(directory / "feature_means.npy")
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    return model, features, targets, means, metadata


def _read_expression(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.index.duplicated().any():
        raise ValueError("Expression sample identifiers must be unique.")
    return frame


def _write_frame(frame: pd.DataFrame, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination)


def _bundle_overlap(metadata: dict) -> float:
    return float(metadata["training_config"]["data"].get("min_feature_overlap", 0.8))


def _data_summary(data) -> dict:
    return {
        "n_samples": len(data.expression),
        "n_expression_features": data.expression.shape[1],
        "n_mutation_targets": data.mutations.shape[1],
        "cancers": sorted(data.cancer.unique().tolist()),
    }
