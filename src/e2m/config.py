from __future__ import annotations

import logging
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("e2m")
logger.addHandler(logging.NullHandler())

# The 33 TCGA cohort codes on the UCSC Xena GDC hub: a reference list, not a restriction.
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC",
    "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV",
    "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM",
    "UCEC", "UCS", "UVM",
)

_HELP = {
    "expression_dataset": {
        "star_counts": "STAR gene counts (manuscript default)",
        "star_tpm": "STAR TPM",
        "star_fpkm": "STAR FPKM",
        "star_fpkm_uq": "STAR upper-quartile FPKM",
    },
    "expression_transform": {
        "log1p": "invert Xena's log2 to counts, then natural log1p (default)",
        "raw": "invert Xena's log2 to linear counts, no log",
        "xena": "keep the downloaded Xena log2 unchanged (no inversion, no offset)",
    },
    "tmb_ml_model": {
        "xgboost": "gradient-boosted trees (default)",
        "lightgbm": "LightGBM gradient boosting (needs the lightgbm extra)",
        "random_forest": "scikit-learn RandomForestRegressor",
        "gradient_boosting": "scikit-learn GradientBoostingRegressor",
    },
    "shap_method": {
        "xgboost": "Tree SHAP on an XGBoost classifier (manuscript default)",
        "lightgbm": "Tree SHAP on a LightGBM classifier (needs the lightgbm extra)",
        "random_forest": "Tree SHAP on a scikit-learn RandomForestClassifier",
        "gradient_boosting": "Tree SHAP on a scikit-learn GradientBoostingClassifier",
        "neural": "Gradient SHAP on the multitask network output itself",
    },
}


def set_verbose(enabled: bool = True, level: int = logging.INFO) -> None:
    """Print e2m progress messages (prefixed [e2m]) to stderr, or silence them."""
    for handler in list(logger.handlers):
        if getattr(handler, "_e2m_stream", False):
            logger.removeHandler(handler)
    if enabled:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[e2m] %(message)s"))
        handler._e2m_stream = True  # tag so a later call can find and replace it
        logger.addHandler(handler)
        logger.setLevel(level)
    logger.propagate = False


def options() -> dict:
    """Accepted values, with one-line descriptions, for the main data and model options."""
    from .data import EXPRESSION_DATASETS, EXPRESSION_TRANSFORMS
    from .models import TREE_MODELS

    accepted = {
        "expression_dataset": EXPRESSION_DATASETS,
        "expression_transform": EXPRESSION_TRANSFORMS,
        "tmb_ml_model": TREE_MODELS,
        "shap_method": (*TREE_MODELS, "neural"),
    }
    reference = {key: {name: _HELP[key].get(name, "") for name in names} for key, names in accepted.items()}
    reference["cancers"] = list(TCGA_CANCERS)
    return reference


def format_options() -> str:
    reference = options()
    lines: list[str] = []
    for key in ("expression_dataset", "expression_transform", "tmb_ml_model", "shap_method"):
        lines.append(f"{key}:")
        for value, description in reference[key].items():
            lines.append(f"  {value:<16} {description}")
        lines.append("")
    cancers = reference["cancers"]
    lines.append(f"cancers ({len(cancers)} TCGA cohorts; any Xena-hosted code also works):")
    for start in range(0, len(cancers), 10):
        lines.append("  " + " ".join(cancers[start:start + 10]))
    return "\n".join(lines)


def _read_yaml(source) -> dict[str, Any]:
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {source}")
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Packaged defaults, updated by the YAML at `path`, then by `overrides`."""
    config = _read_yaml(files("e2m").joinpath("default_config.yaml"))
    if path:
        config = deep_update(config, _read_yaml(Path(path)))
    return deep_update(config, overrides)
