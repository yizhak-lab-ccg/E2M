"""Predict mutation status on an external GEO cohort with a TCGA-trained model.

This trains an E2M mutation model on TCGA lung adenocarcinoma (``LUAD``) and applies
it to GSE31210 (Okayama et al.), an independent lung-adenocarcinoma microarray cohort
the model has never seen, to demonstrate cross-cohort prediction end to end.

Requirements: network access, and GEOparse for the GEO download::

    pip install GEOparse

Caveat: GSE31210 is Affymetrix microarray data, on a different measurement scale from
the TCGA STAR counts the model is trained on. This script shows the *mechanics* of
external prediction — align genes by symbol, then predict. For quantitative transfer,
the manuscript batch-corrects each external cohort against its TCGA training set
(ComBat / rank normalization); that correction is not performed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


def _symbol_column(annotation_table: pd.DataFrame) -> str:
    for name in ("Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol", "gene_assignment"):
        if name in annotation_table.columns:
            return name
    raise SystemExit(f"Could not find a gene-symbol column in the platform annotation: {list(annotation_table.columns)}")


def load_gse31210(cache_dir: Path) -> pd.DataFrame:
    """Download GSE31210, map probes to gene symbols, return a samples-by-genes frame."""
    try:
        import GEOparse
    except ImportError as exc:  # pragma: no cover - example-only dependency
        raise SystemExit('This example needs GEOparse. Install it with: pip install GEOparse') from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(geo="GSE31210", destdir=str(cache_dir), silent=True)
    probes = gse.pivot_samples("VALUE")  # probes x samples

    platform = list(gse.gpls.values())[0]
    annotation = platform.table.set_index("ID")
    symbols = annotation[_symbol_column(annotation)].reindex(probes.index).astype(str)
    symbols = symbols.str.split(r"\s*///\s*").str[0].str.strip()

    keep = symbols.notna() & ~symbols.isin({"", "nan", "---"})
    probes = probes.loc[keep]
    probes.index = symbols[keep].values
    expression = probes.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean().T
    expression.index.name = "sample"
    return expression


def main() -> None:
    from e2m import Dataset, E2MModel

    cache = Path("./e2m_data")

    print("Training an E2M mutation model on TCGA-LUAD (downloads TCGA on first run)...")
    tcga = Dataset.from_tcga(["LUAD"], data_dir=cache, with_tmb=False)
    model = E2MModel().fit(tcga)
    print(f"  trained on {len(tcga.expression)} samples, {len(model.targets)} mutation targets")

    print("Downloading external cohort GSE31210...")
    external = load_gse31210(cache / "geo")
    print(f"  external cohort: {external.shape[0]} samples x {external.shape[1]} genes")

    # Align to the model's training features by gene symbol; allow partial overlap
    # because a microarray covers only part of the protein-coding transcriptome.
    probabilities = model.predict(external, min_feature_overlap=0.0)
    print(f"  predicted {probabilities.shape[1]} targets for {probabilities.shape[0]} external samples")

    top = probabilities.mean().sort_values(ascending=False).head(10)
    print("\nTop predicted mutation targets by mean probability:")
    print(top.to_string())

    output = Path("results/external_gse31210_probabilities.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(output)
    print(f"\nWrote per-sample probabilities to {output}")


if __name__ == "__main__":
    main()
