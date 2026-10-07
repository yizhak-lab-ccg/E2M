from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import requests

from .config import load_config, logger


EXPRESSION_DATASETS = ("star_counts", "star_tpm", "star_fpkm", "star_fpkm_uq")
EXPRESSION_TRANSFORMS = ("log1p", "raw", "xena")


class Dataset:
    """Samples-by-genes expression with aligned mutation labels, TMB, and cancer labels."""

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
        cfg = load_config(config, {"data": data_overrides or {}})
        selected = normalize_cancers(cancers or cfg["data"].get("cancers"))
        cfg["data"]["cancers"] = selected
        loader = XenaTCGALoader(cfg, data_dir=data_dir, use_cache=use_cache)
        data = loader.prepare(selected)
        tmb = pd.concat(loader._load_tmb(cancer) for cancer in selected) if with_tmb else None
        return cls(data.expression, data.mutations, tmb, data.cancer)

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


def align_expression(
    expression: pd.DataFrame,
    features: list[str],
    fill_values: np.ndarray,
    min_overlap: float,
) -> tuple[pd.DataFrame, float]:
    """Reorder columns to the training features; fill missing ones with training means."""
    if expression.columns.duplicated().any():
        expression = expression.T.groupby(level=0, sort=False).mean().T
    overlap_count = len(set(features).intersection(expression.columns))
    overlap = overlap_count / max(len(features), 1)
    if overlap < min_overlap:
        raise ValueError(
            f"Expression feature overlap is {overlap:.1%}; at least {min_overlap:.1%} is required. "
            "Use the same expression quantity, transform, and gene symbols used for training."
        )
    aligned = expression.reindex(columns=features)
    aligned = aligned.fillna(pd.Series(fill_values, index=features))
    return aligned.astype(np.float32), overlap


def normalize_cancers(values: Iterable[str] | None) -> list[str]:
    cancers: list[str] = []
    for value in values or []:
        cancers.extend(part.strip().upper() for part in str(value).split(",") if part.strip())
    if not cancers:
        raise ValueError("Select at least one TCGA cancer code, for example LUAD.")
    return list(dict.fromkeys(cancers))


def canonical_sample_id(value: str) -> str:
    parts = str(value).split("-")
    if len(parts) < 4 or parts[0] != "TCGA":
        return str(value)
    sample = parts[3][:3]
    if len(sample) == 2:
        sample += "A"
    return "-".join([parts[0], parts[1], parts[2], sample])


class XenaTCGALoader:
    """Download and prepare cohort-level TCGA data from UCSC Xena."""

    def __init__(self, config: dict, data_dir: str | Path | None = None, use_cache: bool | None = None):
        self.config = config
        self.data_cfg = config["data"]
        self.xena_cfg = config["xena"]
        self.data_dir = Path(data_dir or self.data_cfg["data_dir"]).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = self.data_cfg.get("use_cache", True) if use_cache is None else use_cache

    def prepare(self, cancers: list[str]) -> Dataset:
        cache_path = self._cache_path(cancers)
        if self.use_cache and cache_path.exists():
            logger.info("Loading prepared data for %s from cache", ", ".join(cancers))
            return joblib.load(cache_path)

        logger.info("Preparing %d cohort(s): %s", len(cancers), ", ".join(cancers))
        annotation = self._load_annotation()
        expression_parts: list[pd.DataFrame] = []
        mutation_parts: list[pd.DataFrame] = []
        cancer_parts: list[pd.Series] = []
        for cancer in cancers:
            expression = self._load_expression(cancer, annotation)
            mutations = self._load_mutations(cancer, annotation)
            common = expression.index.intersection(mutations.index)
            if common.empty:
                raise ValueError(f"No matched expression and mutation samples were found for {cancer}.")
            expression_parts.append(expression.loc[common])
            mutation_parts.append(mutations.loc[common])
            cancer_parts.append(pd.Series(cancer, index=common, name="cancer"))

        expression = pd.concat(expression_parts, axis=0, join="inner")
        mutations = pd.concat(mutation_parts, axis=0, join="outer").fillna(0).astype(np.int8)
        cancer = pd.concat(cancer_parts).reindex(expression.index)
        mutations = mutations.reindex(expression.index).fillna(0).astype(np.int8)
        mutations = self._select_targets(mutations)
        if mutations.empty:
            raise ValueError("No mutation targets passed the prevalence threshold.")

        logger.info(
            "Prepared %d samples x %d genes, %d mutation targets",
            len(expression), expression.shape[1], mutations.shape[1],
        )
        prepared = Dataset(expression, mutations, cancer=cancer)
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(prepared, cache_path)
        return prepared

    def prepare_tmb(self, cancers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        logger.info("Preparing coding TMB for %d cohort(s): %s", len(cancers), ", ".join(cancers))
        annotation = self._load_annotation()
        expression_parts = []
        tmb_parts = []
        cancer_parts = []
        for cancer_code in cancers:
            expression = self._load_expression(cancer_code, annotation)
            tmb = self._load_tmb(cancer_code)
            common = expression.index.intersection(tmb.index)
            expression_parts.append(expression.loc[common])
            tmb_parts.append(tmb.loc[common])
            cancer_parts.append(pd.Series(cancer_code, index=common, name="cancer"))
        expression = pd.concat(expression_parts, axis=0, join="inner")
        tmb = pd.concat(tmb_parts).reindex(expression.index)
        cancer = pd.concat(cancer_parts).reindex(expression.index)
        if expression.empty:
            raise ValueError("No matched expression and coding TMB samples were found.")
        logger.info("Prepared coding TMB for %d samples", len(expression))
        return expression, tmb, cancer

    def _load_tmb(self, cancer: str) -> pd.DataFrame:
        filename = self.xena_cfg["mutation_event_file"].format(cancer=cancer)
        url = self.xena_cfg["mutation_event_url_template"].format(cancer=cancer)
        path = self._ensure_file(Path("mutation_events") / filename, url)
        events = pd.read_csv(path, sep="\t", low_memory=False)
        sample_column = "sample" if "sample" in events.columns else "Tumor_Sample_Barcode"
        class_column = "effect" if "effect" in events.columns else "Variant_Classification"
        if sample_column not in events.columns or class_column not in events.columns:
            raise ValueError(f"Missing sample or mutation class columns in {path}.")
        events = events.loc[events[class_column].isin(self.config["tmb"]["coding_variant_classes"]), [sample_column]]
        events["sample_id"] = events[sample_column].map(canonical_sample_id)
        tmb = events.groupby("sample_id").size().rename("TMB").to_frame()
        tmb["TMB_log2"] = np.log2(tmb["TMB"].astype(float) + 1.0)
        return tmb

    def _load_expression(self, cancer: str, annotation: pd.DataFrame) -> pd.DataFrame:
        dataset = self.data_cfg.get("expression_dataset", "star_counts")
        if dataset not in EXPRESSION_DATASETS:
            raise ValueError(f"expression_dataset must be one of: {', '.join(EXPRESSION_DATASETS)}")
        filename = self.xena_cfg["expression_files"][dataset].format(cancer=cancer)
        url = self.xena_cfg["expression_url_template"].format(filename=filename)
        path = self._ensure_file(Path("expression") / filename, url)
        matrix = _read_numeric(path).T
        matrix.index = matrix.index.map(canonical_sample_id)
        matrix = matrix[~matrix.index.duplicated(keep="first")]
        matrix = self._transform_expression(matrix)

        base_ids = pd.Index(matrix.columns.astype(str).str.split(".").str[0], name="gene_id")
        symbols = annotation["symbol"].reindex(base_ids).to_numpy()
        biotypes = annotation["biotype"].reindex(base_ids).to_numpy()
        keep = pd.notna(symbols)
        if self.data_cfg.get("protein_coding_only", True):
            keep &= biotypes == "protein_coding"
        matrix = matrix.loc[:, keep]
        matrix.columns = symbols[keep].astype(str)
        collapse = self.data_cfg.get("collapse_duplicate_symbols", "mean")
        if collapse == "mean":
            matrix = matrix.T.groupby(level=0, sort=False).mean().T
        elif collapse == "sum":
            matrix = matrix.T.groupby(level=0, sort=False).sum().T
        elif collapse != "first":
            raise ValueError("collapse_duplicate_symbols must be mean, sum, or first.")
        else:
            matrix = matrix.loc[:, ~matrix.columns.duplicated(keep="first")]
        matrix.index.name = "sample_id"
        return matrix.astype(np.float32)

    def _transform_expression(self, matrix: pd.DataFrame) -> pd.DataFrame:
        # Xena serves STAR values as log2(x + offset): undo it, then log1p (or keep linear / keep Xena's log2).
        transform = self.data_cfg["expression_transform"]
        if transform not in EXPRESSION_TRANSFORMS:
            raise ValueError(f"expression_transform must be one of: {', '.join(EXPRESSION_TRANSFORMS)}")
        if transform == "xena":
            return matrix
        counts = (2.0 ** matrix - float(self.data_cfg["xena_log_offset"])).clip(lower=0)
        return np.log1p(counts) if transform == "log1p" else counts

    def _load_mutations(self, cancer: str, annotation: pd.DataFrame) -> pd.DataFrame:
        local_name = self.xena_cfg["mutation_file"].format(cancer=cancer)
        url = self.xena_cfg["mutation_url_template"].format(cancer=cancer)
        path = self._ensure_file(Path("mutations") / local_name, url)
        matrix = _read_numeric(path)
        index_samples = matrix.index.astype(str).str.startswith("TCGA-").mean()
        column_samples = matrix.columns.astype(str).str.startswith("TCGA-").mean()
        if column_samples > index_samples:
            matrix = matrix.T
        matrix.index = matrix.index.map(canonical_sample_id)
        matrix = matrix[~matrix.index.duplicated(keep="first")]
        matrix = matrix.fillna(0)
        matrix.columns = matrix.columns.astype(str)
        if self.data_cfg.get("protein_coding_mutation_targets", True):
            protein_symbols = set(annotation.loc[annotation["biotype"].eq("protein_coding"), "symbol"])
            matrix = matrix.loc[:, matrix.columns.isin(protein_symbols)]
        matrix.index.name = "sample_id"
        return (matrix > 0).astype(np.int8)

    def _select_targets(self, mutations: pd.DataFrame) -> pd.DataFrame:
        prevalence = mutations.mean(axis=0)
        threshold = float(self.data_cfg.get("min_mutation_prevalence", 0.05))
        keep = prevalence[prevalence >= threshold].sort_values(ascending=False)
        cap = int(self.data_cfg.get("max_mutation_targets", 400) or 0)
        if cap > 0:
            keep = keep.head(cap)
        return mutations.loc[:, keep.index]

    def _load_annotation(self) -> pd.DataFrame:
        filename = self.xena_cfg["gencode_file"]
        path = self._ensure_file(Path("annotation") / filename, self.xena_cfg["gencode_url"])
        cache = path.with_suffix(path.suffix + ".joblib")
        if cache.exists():
            return joblib.load(cache)
        rows: list[tuple[str, str, str]] = []
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                attributes = fields[8]
                gene_id = _gtf_value(attributes, "gene_id").split(".")[0]
                symbol = _gtf_value(attributes, "gene_name")
                biotype = _gtf_value(attributes, "gene_type") or _gtf_value(attributes, "gene_biotype")
                if gene_id and symbol:
                    rows.append((gene_id, symbol, biotype))
        annotation = pd.DataFrame(rows, columns=["gene_id", "symbol", "biotype"]).drop_duplicates("gene_id")
        annotation = annotation.set_index("gene_id")
        joblib.dump(annotation, cache)
        return annotation

    def _ensure_file(self, relative_path: Path, url: str) -> Path:
        path = self.data_dir / relative_path
        if path.exists():
            return path
        if not self.data_cfg.get("download_missing", True):
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", relative_path.as_posix())
        partial = path.with_name(path.name + ".part")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        partial.replace(path)
        return path

    def _cache_path(self, cancers: list[str]) -> Path:
        key_data = {
            "cancers": cancers,
            "dataset": self.data_cfg.get("expression_dataset"),
            "transform": self.data_cfg["expression_transform"],
            "offset": self.data_cfg.get("xena_log_offset"),
            "protein": self.data_cfg.get("protein_coding_only"),
            "protein_mutations": self.data_cfg.get("protein_coding_mutation_targets"),
            "collapse": self.data_cfg.get("collapse_duplicate_symbols"),
            "prevalence": self.data_cfg.get("min_mutation_prevalence"),
            "target_cap": self.data_cfg.get("max_mutation_targets"),
            # Source provenance: changing any download URL/filename or the annotation
            # must invalidate the cache rather than silently reuse stale prepared data.
            "gencode": self.xena_cfg.get("gencode_file"),
            "gencode_url": self.xena_cfg.get("gencode_url"),
            "expression_files": self.xena_cfg.get("expression_files"),
            "expression_url": self.xena_cfg.get("expression_url_template"),
            "mutation_file": self.xena_cfg.get("mutation_file"),
            "mutation_url": self.xena_cfg.get("mutation_url_template"),
        }
        digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.data_dir / "cache" / f"dataset_{digest}.joblib"


def _read_numeric(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", index_col=0)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError(f"Non-numeric values in {path}.")
    return matrix


def _gtf_value(attributes: str, key: str) -> str:
    match = re.search(rf'{re.escape(key)} "([^"]+)"', attributes)
    return match.group(1) if match else ""
