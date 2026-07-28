"""Discover the values the main options accept (:func:`options`).

A small reference so you do not have to read the config to build ``data_overrides``
or pick a TMB backend::

    import e2m
    e2m.options()                            # the whole reference as a dict
    e2m.options()["expression_transform"]    # {"auto": ..., "log1p": ..., ...}
    e2m.options()["cancers"]                 # ["ACC", "BLCA", ...]
    print(e2m.format_options())              # a printable summary
"""

from __future__ import annotations

from .backends import TREE_BACKENDS
from .data import EXPRESSION_DATASETS, EXPRESSION_TRANSFORMS

# The 33 TCGA cohort codes hosted on the UCSC Xena GDC hub. Any code Xena serves
# works; this is a reference list, not a restriction the tool enforces.
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC",
    "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV",
    "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM",
    "UCEC", "UCS", "UVM",
)

_EXPRESSION_DATASET_HELP = {
    "star_counts": "STAR gene counts (manuscript default)",
    "star_tpm": "STAR TPM",
    "star_fpkm": "STAR FPKM",
    "star_fpkm_uq": "STAR upper-quartile FPKM",
}
_EXPRESSION_TRANSFORM_HELP = {
    "auto": "log1p for STAR counts, else keep Xena log2 (default)",
    "log1p": "invert the Xena log2, then natural log1p",
    "raw": "invert the Xena log2 to linear counts, no log",
    "xena": "keep the downloaded Xena log2 values unchanged",
}
_TMB_BACKEND_HELP = {
    "xgboost": "gradient-boosted trees (default)",
    "lightgbm": "LightGBM gradient boosting (needs the lightgbm extra)",
    "random_forest": "scikit-learn RandomForestRegressor",
    "gradient_boosting": "scikit-learn GradientBoostingRegressor",
}
_SHAP_METHOD_HELP = {
    "xgboost": "Tree SHAP on an XGBoost classifier (manuscript default)",
    "lightgbm": "Tree SHAP on a LightGBM classifier (needs the lightgbm extra)",
    "random_forest": "Tree SHAP on a scikit-learn RandomForestClassifier",
    "gradient_boosting": "Tree SHAP on a scikit-learn GradientBoostingClassifier",
    "neural": "Gradient SHAP on the multitask network output itself",
}


def options() -> dict:
    """Return the accepted values for the main data and model options.

    The keys mirror the ``data_overrides`` fields and the ``TmbModel`` backend; each
    value maps an accepted option to a one-line description, except ``cancers`` which
    is the reference list of TCGA cohort codes.
    """
    return {
        "expression_dataset": {name: _EXPRESSION_DATASET_HELP.get(name, "") for name in EXPRESSION_DATASETS},
        "expression_transform": {name: _EXPRESSION_TRANSFORM_HELP.get(name, "") for name in EXPRESSION_TRANSFORMS},
        "tmb_backend": {name: _TMB_BACKEND_HELP.get(name, "") for name in TREE_BACKENDS},
        "shap_method": {name: _SHAP_METHOD_HELP.get(name, "") for name in (*TREE_BACKENDS, "neural")},
        "cancers": list(TCGA_CANCERS),
    }


def format_options() -> str:
    """A printable, human-readable summary of :func:`options`."""
    reference = options()
    lines: list[str] = []
    for key in ("expression_dataset", "expression_transform", "tmb_backend", "shap_method"):
        lines.append(f"{key}:")
        for value, description in reference[key].items():
            lines.append(f"  {value:<16} {description}")
        lines.append("")
    cancers = reference["cancers"]
    lines.append(f"cancers ({len(cancers)} TCGA cohorts; any Xena-hosted code also works):")
    for start in range(0, len(cancers), 10):
        lines.append("  " + " ".join(cancers[start:start + 10]))
    return "\n".join(lines)
