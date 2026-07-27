"""The :class:`Dataset` — aligned expression / mutations / TMB you build models on."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .config import resolve_config
from .data import XenaTCGALoader, normalize_cancers


class Dataset:
    """Aligned expression, mutation labels, and TMB for one or more cohorts.

    Build one from TCGA with :meth:`from_tcga`, or wrap your own frames::

        data = Dataset.from_tcga(["LUAD"], data_dir="./e2m_data")
        data = Dataset(expression=expr_df, mutations=mut_df, tmb=tmb_df)

    ``expression`` is samples-by-genes, ``mutations`` is samples-by-targets (0/1),
    ``tmb`` is a samples-by-{TMB, TMB_log2} table, and ``cancer`` labels each sample;
    all are aligned to ``expression.index``. Use :meth:`subset` to slice samples or
    targets, then pass the dataset to a model (e.g. ``MutationModel().fit(data)``).
    """

    def __init__(self, expression: pd.DataFrame, mutations=None, tmb=None, cancer=None):
        self.expression = expression
        self.mutations = mutations.reindex(expression.index) if mutations is not None else None
        self.tmb = tmb.reindex(expression.index) if tmb is not None else None
        if cancer is None:
            cancer = pd.Series("cohort", index=expression.index, name="cancer")
        self.cancer = pd.Series(cancer, name="cancer").reindex(expression.index)

    @classmethod
    def from_tcga(
        cls,
        cancers=None,
        data_dir=None,
        config=None,
        data_overrides=None,
        use_cache: bool = True,
        with_tmb: bool = True,
    ) -> "Dataset":
        """Download, preprocess, and align TCGA expression, mutation labels, and TMB."""
        cfg = resolve_config(config, {"data": data_overrides or {}})
        selected = normalize_cancers(cancers or cfg["data"].get("cancers"))
        cfg["data"]["cancers"] = selected
        loader = XenaTCGALoader(cfg, data_dir=data_dir, use_cache=use_cache)
        prepared = loader.prepare(selected)
        tmb = None
        if with_tmb:
            _, tmb_labels, _ = loader.prepare_tmb(selected)
            tmb = tmb_labels
        return cls(
            expression=prepared.expression,
            mutations=prepared.mutations,
            tmb=tmb,
            cancer=prepared.cancer,
        )

    @property
    def samples(self) -> pd.Index:
        return self.expression.index

    @property
    def genes(self) -> pd.Index:
        return self.expression.columns

    @property
    def targets(self) -> Optional[pd.Index]:
        return None if self.mutations is None else self.mutations.columns

    def subset(self, samples: Optional[Iterable] = None, targets: Optional[Iterable] = None) -> "Dataset":
        """Return a new :class:`Dataset` restricted to the given samples and/or targets."""
        expression = self.expression
        cancer = self.cancer
        mutations = self.mutations
        tmb = self.tmb
        if samples is not None:
            index = expression.index.intersection(pd.Index(samples))
            expression = expression.loc[index]
            cancer = cancer.loc[index]
            mutations = mutations.loc[index] if mutations is not None else None
            tmb = tmb.loc[index] if tmb is not None else None
        if targets is not None and mutations is not None:
            mutations = mutations[list(targets)]
        return Dataset(expression=expression, mutations=mutations, tmb=tmb, cancer=cancer)

    def __repr__(self) -> str:
        targets = 0 if self.mutations is None else self.mutations.shape[1]
        cohorts = sorted(self.cancer.dropna().unique().tolist())
        return (
            f"Dataset(samples={len(self.expression)}, genes={self.expression.shape[1]}, "
            f"targets={targets}, tmb={self.tmb is not None}, cohorts={cohorts})"
        )
