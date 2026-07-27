"""Expression-to-mutation prediction from RNA expression.

Two ways to use the package:

* Objects — a :class:`Dataset` holds aligned expression / mutations / TMB, and a
  :class:`MutationModel` is fitted on a dataset and predicts on any dataset or
  expression matrix. Methods return plain pandas objects::

      from e2m import Dataset, E2MModel
      data  = Dataset.from_tcga(["LUAD"], data_dir="./e2m_data")
      model = E2MModel().fit(data.subset(samples=train_ids))
      probs = model.predict(data.subset(samples=test_ids))   # DataFrame
      metrics = model.cross_validate(data)                   # DataFrame

  TMB prediction uses :class:`TmbModel` (pluggable tree backend).

* The stateless functional API (also what the ``e2m`` CLI calls):
  :func:`download`, :func:`cross_validate`, :func:`train`, :func:`predict_tmb`,
  :func:`predict`, :func:`embed`, :func:`head_weights`, :func:`explain`.
"""

from .data import PreparedData
from .dataset import Dataset
from .model import E2MModel, TmbModel
from .workflow import cross_validate, download, embed, explain, head_weights, predict, predict_tmb, train

__all__ = [
    # objects
    "Dataset",
    "E2MModel",
    "TmbModel",
    "PreparedData",
    # functional API
    "download",
    "cross_validate",
    "train",
    "predict_tmb",
    "predict",
    "embed",
    "head_weights",
    "explain",
]
__version__ = "0.2.0"
