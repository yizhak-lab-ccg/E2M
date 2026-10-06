"""Predict somatic mutation status and TMB from RNA expression."""

from .data import Dataset
from .logging_utils import set_verbose
from .model import E2MModel, TmbModel
from .options import format_options, options
from .workflow import cross_validate, download, embed, explain, head_weights, predict, predict_tmb, train

__all__ = [
    # objects
    "Dataset",
    "E2MModel",
    "TmbModel",
    # functional API
    "download",
    "cross_validate",
    "train",
    "predict_tmb",
    "predict",
    "embed",
    "head_weights",
    "explain",
    # discoverability / progress
    "options",
    "format_options",
    "set_verbose",
]
__version__ = "0.2.0"
