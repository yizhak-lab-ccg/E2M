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
# Data settings that change the prepared data: recorded in the manifest and part of the cache key.
MANIFEST_SETTINGS = (
    "expression_dataset", "expression_transform", "xena_log_offset", "normalization",
    "protein_coding_only", "protein_coding_mutation_targets", "collapse_duplicate_symbols",
    "min_mutation_prevalence", "min_mutation_positives", "max_mutation_targets", "targets",
)


class Dataset:
    """Samples-by-genes expression with aligned mutation labels, TMB, and cancer labels."""

    def __init__(self, expression: pd.DataFrame, mutations=None, tmb=None, cancer=None, manifest=None):
        self.expression = expression
        self.mutations = mutations.reindex(expression.index) if mutations is not None else None
        self.tmb = tmb.reindex(expression.index) if tmb is not None else None
        if cancer is None:
            cancer = pd.Series("cohort", index=expression.index, name="cancer")
        self.cancer = pd.Series(cancer, name="cancer").reindex(expression.index)
        self.manifest = manifest

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
        tmb = None
        if with_tmb:
            events = pd.concat(loader._load_tmb(cancer) for cancer in selected)
            tmb = events.reindex(data.samples.map(sample_id)).set_axis(data.samples)
        return cls(data.expression, data.mutations, tmb, data.cancer, data.manifest)

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


def sample_id(value: str, vial: bool = False) -> str:
    """TCGA barcode cut to patient + sample type (TCGA-XX-XXXX-01), the MC3 form; with `vial`, TCGA-XX-XXXX-01A."""
    parts = str(value).split("-")
    if len(parts) < 4 or parts[0] != "TCGA":
        return str(value)
    sample = parts[3][:3].ljust(3, "A") if vial else parts[3][:2]
    return "-".join([parts[0], parts[1], parts[2], sample])


def _match_mc3(expression: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair expression rows with MC3 rows by the vial-free ID, one vial per tumor (earliest letter); keep expression IDs."""
    order = np.argsort(expression.index.to_numpy(dtype=str), kind="stable")
    keys = expression.index[order].map(sample_id)
    expression = expression.iloc[np.sort(order[~keys.duplicated() & keys.isin(table.index)])]
    keys = expression.index.map(sample_id)
    return expression, table.loc[keys].set_axis(expression.index)


class XenaTCGALoader:
    """Download and prepare cohort-level TCGA data from UCSC Xena."""

    def __init__(self, config: dict, data_dir: str | Path | None = None, use_cache: bool | None = None):
        self.config = config
        self.data_cfg = config["data"]
        self.xena_cfg = config["xena"]
        self.data_dir = Path(data_dir or self.data_cfg["data_dir"]).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = self.data_cfg.get("use_cache", True) if use_cache is None else use_cache
        self._sources: dict[str, Path] = {}

    def prepare(self, cancers: list[str]) -> Dataset:
        cache_path = self._cache_path(cancers)
        if self.use_cache and cache_path.exists():
            logger.info("Loading prepared data for %s from cache", ", ".join(cancers))
            return joblib.load(cache_path)

        logger.info("Preparing %d cohort(s): %s", len(cancers), ", ".join(cancers))
        self._sources = {}
        annotation = self._load_annotation()
        expression_parts: list[pd.DataFrame] = []
        mutation_parts: list[pd.DataFrame] = []
        cancer_parts: list[pd.Series] = []
        cohorts = {}
        for cancer in cancers:
            stats = {}
            expression = self._load_expression(cancer, annotation, stats)
            mutations = self._load_mutations(cancer, annotation)
            stats.update(expression_samples=len(expression), mc3_samples=len(mutations))
            expression, mutations = _match_mc3(expression, mutations)
            if expression.empty:
                raise ValueError(f"No matched expression and mutation samples were found for {cancer}.")
            stats.update(
                matched_samples=len(expression),
                matched_from_vial_b_or_later=int((expression.index.str[15:16] > "A").sum()),
            )
            cohorts[cancer] = stats
            expression_parts.append(expression)
            mutation_parts.append(mutations)
            cancer_parts.append(pd.Series(cancer, index=expression.index, name="cancer"))

        expression = pd.concat(expression_parts, axis=0, join="inner")
        mutations = pd.concat(mutation_parts, axis=0, join="outer").fillna(0).astype(np.int8)
        cancer = pd.concat(cancer_parts).reindex(expression.index)
        mutations = mutations.reindex(expression.index).fillna(0).astype(np.int8)
        candidate_targets = mutations.shape[1]
        mutations = self._select_targets(mutations)
        if mutations.empty:
            raise ValueError("No mutation targets passed the prevalence and minimum-positives rules.")

        logger.info(
            "Prepared %d samples x %d genes, %d mutation targets",
            len(expression), expression.shape[1], mutations.shape[1],
        )
        manifest = {
            "settings": {key: value for key, value in self.data_cfg.items() if key in MANIFEST_SETTINGS},
            "cohorts": cohorts,
            "combined": {
                "samples": len(expression),
                "genes": expression.shape[1],
                "candidate_targets": candidate_targets,
                "targets": mutations.shape[1],
            },
            "sources": {
                name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for name, path in sorted(self._sources.items())
            },
        }
        prepared = Dataset(expression, mutations, cancer=cancer, manifest=manifest)
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
            expression, tmb = _match_mc3(self._load_expression(cancer_code, annotation), self._load_tmb(cancer_code))
            expression_parts.append(expression)
            tmb_parts.append(tmb)
            cancer_parts.append(pd.Series(cancer_code, index=expression.index, name="cancer"))
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
        events["sample_id"] = events[sample_column].map(sample_id)
        tmb = events.groupby("sample_id").size().rename("TMB").to_frame()
        tmb["TMB_log2"] = np.log2(tmb["TMB"].astype(float) + 1.0)
        return tmb

    def _load_expression(self, cancer: str, annotation: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
        stats = {} if stats is None else stats
        dataset = self.data_cfg.get("expression_dataset", "star_counts")
        if dataset not in EXPRESSION_DATASETS:
            raise ValueError(f"expression_dataset must be one of: {', '.join(EXPRESSION_DATASETS)}")
        filename = self.xena_cfg["expression_files"][dataset].format(cancer=cancer)
        url = self.xena_cfg["expression_url_template"].format(filename=filename)
        path = self._ensure_file(Path("expression") / filename, url)
        matrix = _read_numeric(path).T
        matrix.index = matrix.index.map(lambda value: sample_id(value, vial=True))
        matrix = self._linear(matrix)

        base_ids = pd.Index(matrix.columns.astype(str).str.split(".").str[0], name="gene_id")
        symbols = annotation["symbol"].reindex(base_ids).to_numpy()
        biotypes = annotation["biotype"].reindex(base_ids).to_numpy()
        keep = pd.notna(symbols)
        if self.data_cfg.get("protein_coding_only", True):
            keep &= biotypes == "protein_coding"
        matrix = matrix.loc[:, keep]
        matrix.columns = symbols[keep].astype(str)
        stats["duplicate_symbols"] = int(matrix.columns[matrix.columns.duplicated()].nunique())
        matrix = self._collapse(matrix)
        if self.data_cfg["expression_transform"] == "log1p":
            matrix = np.log1p(matrix)
        matrix.index.name = "sample_id"
        return matrix.astype(np.float32)

    def _linear(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Undo Xena's log2(x + offset), then CPM if configured; `xena` keeps the downloaded log2 values."""
        transform = self.data_cfg["expression_transform"]
        if transform not in EXPRESSION_TRANSFORMS:
            raise ValueError(f"expression_transform must be one of: {', '.join(EXPRESSION_TRANSFORMS)}")
        normalization = self.data_cfg.get("normalization", "none")
        if normalization not in ("none", "cpm"):
            raise ValueError("normalization must be none or cpm.")
        if transform == "xena":
            if normalization != "none":
                raise ValueError("normalization: cpm needs expression_transform log1p or raw.")
            return matrix
        counts = (2.0 ** matrix - float(self.data_cfg["xena_log_offset"])).clip(lower=0)
        if normalization == "cpm":
            counts = counts.div(counts.sum(axis=1), axis=0) * 1e6
        return counts

    def _collapse(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Merge columns sharing a symbol; xena log2 values are averaged since summing logs is meaningless."""
        collapse = self.data_cfg.get("collapse_duplicate_symbols", "sum")
        if collapse == "sum" and self.data_cfg["expression_transform"] == "xena":
            collapse = "mean"
        if collapse == "first":
            return matrix.loc[:, ~matrix.columns.duplicated(keep="first")]
        if collapse not in ("sum", "mean"):
            raise ValueError("collapse_duplicate_symbols must be sum, mean, or first.")
        return matrix.T.groupby(level=0, sort=False).agg(collapse).T

    def _load_mutations(self, cancer: str, annotation: pd.DataFrame) -> pd.DataFrame:
        local_name = self.xena_cfg["mutation_file"].format(cancer=cancer)
        url = self.xena_cfg["mutation_url_template"].format(cancer=cancer)
        path = self._ensure_file(Path("mutations") / local_name, url)
        matrix = _read_numeric(path)
        index_samples = matrix.index.astype(str).str.startswith("TCGA-").mean()
        column_samples = matrix.columns.astype(str).str.startswith("TCGA-").mean()
        if column_samples > index_samples:
            matrix = matrix.T
        matrix.index = matrix.index.map(sample_id)
        matrix = matrix[~matrix.index.duplicated(keep="first")].fillna(0)
        matrix.columns = matrix.columns.astype(str)
        if self.data_cfg.get("protein_coding_mutation_targets", True):
            protein_symbols = set(annotation.loc[annotation["biotype"].eq("protein_coding"), "symbol"])
            matrix = matrix.loc[:, matrix.columns.isin(protein_symbols)]
        matrix.index.name = "sample_id"
        return (matrix > 0).astype(np.int8)

    def _select_targets(self, mutations: pd.DataFrame) -> pd.DataFrame:
        """The fixed `targets` list, or targets passing prevalence and minimum positives (on matched samples)."""
        fixed = self.data_cfg.get("targets")
        if fixed:
            missing = [target for target in fixed if target not in mutations.columns]
            if missing:
                raise ValueError(f"Fixed targets missing from the mutation data: {', '.join(missing)}")
            return mutations.loc[:, list(fixed)]
        prevalence = mutations.mean(axis=0)
        threshold = float(self.data_cfg.get("min_mutation_prevalence", 0.05))
        min_positives = int(self.data_cfg.get("min_mutation_positives", 0) or 0)
        keep = prevalence[(prevalence >= threshold) & (mutations.sum(axis=0) >= min_positives)]
        # By prevalence, ties by gene name, so the order is reproducible.
        keep = keep.sort_index().sort_values(ascending=False, kind="stable")
        cap = int(self.data_cfg.get("max_mutation_targets", 400) or 0)
        return mutations.loc[:, (keep.head(cap) if cap > 0 else keep).index]

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
        self._sources[relative_path.as_posix()] = path
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
        from . import __version__

        # Every preprocessing setting and source file/URL, so a change never reuses stale data.
        key_data = {
            "e2m_version": __version__,
            "cancers": cancers,
            "settings": {key: value for key, value in self.data_cfg.items() if key in MANIFEST_SETTINGS},
            "xena": self.xena_cfg,
        }
        digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.data_dir / "cache" / f"dataset_{digest}.joblib"


def _read_numeric(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", index_col=0)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError(f"Non-numeric values in {path}.")
    return matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gtf_value(attributes: str, key: str) -> str:
    match = re.search(rf'{re.escape(key)} "([^"]+)"', attributes)
    return match.group(1) if match else ""
