from __future__ import annotations

import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e2m.cli import build_parser
from e2m.config import load_config
from e2m.data import XenaTCGALoader, canonical_sample_id
from e2m.metrics import normalized_auprc
from e2m.workflow import align_expression


def test_default_config_is_packaged_and_named_clearly():
    config = load_config()
    assert config["data"]["expression_dataset"] == "star_counts"
    assert config["data"]["expression_transform"] == "auto"
    assert "source" not in config["data"]


def test_normalized_auprc_baseline_and_perfect():
    assert normalized_auprc(0.2, 0.2) == pytest.approx(0.0)
    assert normalized_auprc(1.0, 0.2) == pytest.approx(1.0)


def test_alignment_uses_training_means_and_checks_overlap():
    expression = pd.DataFrame([[4.0, 8.0]], index=["sample"], columns=["A", "B"])
    aligned, overlap = align_expression(expression, ["A", "B", "C"], np.array([1.0, 2.0, 3.0]), 0.5)
    assert overlap == pytest.approx(2 / 3)
    assert aligned.loc["sample", "C"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="feature overlap"):
        align_expression(expression, ["A", "B", "C"], np.array([1.0, 2.0, 3.0]), 0.8)


def test_xena_transforms_are_explicit(tmp_path):
    downloaded = pd.DataFrame([[0.0, 2.0]])  # Xena log2(count + 1) -> counts [0, 3]

    # Default (auto) for STAR counts recovers counts and log-transforms them.
    default_loader = XenaTCGALoader(load_config(), data_dir=tmp_path)
    defaulted = default_loader._transform_expression(downloaded)
    assert defaulted.iloc[0].tolist() == pytest.approx(np.log1p([0.0, 3.0]).tolist())

    # raw opts out of the log transform and returns linear counts.
    raw_config = load_config()
    raw_config["data"]["expression_transform"] = "raw"
    raw = XenaTCGALoader(raw_config, data_dir=tmp_path)._transform_expression(downloaded)
    assert raw.iloc[0].tolist() == pytest.approx([0.0, 3.0])

    # xena keeps the downloaded log2 values unchanged.
    xena_config = load_config()
    xena_config["data"]["expression_transform"] = "xena"
    kept = XenaTCGALoader(xena_config, data_dir=tmp_path)._transform_expression(downloaded)
    assert kept.equals(downloaded)


def test_sample_ids_match_expression_and_mc3_forms():
    assert canonical_sample_id("TCGA-AB-1234-01") == "TCGA-AB-1234-01A"
    assert canonical_sample_id("TCGA-AB-1234-01A-01R") == "TCGA-AB-1234-01A"


def test_local_xena_loader_maps_collapses_and_aligns(tmp_path):
    config = load_config()
    config["data"].update(
        {
            "expression_dataset": "star_counts",
            "expression_transform": "raw",
            "min_mutation_prevalence": 0.5,
            "protein_coding_mutation_targets": False,
            "use_cache": False,
            "download_missing": False,
        }
    )
    expression_dir = tmp_path / "expression"
    mutation_dir = tmp_path / "mutations"
    annotation_dir = tmp_path / "annotation"
    expression_dir.mkdir()
    mutation_dir.mkdir()
    annotation_dir.mkdir()
    sample_a = "TCGA-AB-0001-01A"
    sample_b = "TCGA-AB-0002-01A"
    xena_expression = pd.DataFrame(
        {sample_a: [1.0, 2.0, 5.0], sample_b: [2.0, 3.0, 6.0]},
        index=["ENSG1.1", "ENSG2.1", "ENSG3.1"],
    )
    xena_expression.to_csv(expression_dir / "TCGA-LUAD.star_counts.tsv.gz", sep="\t", compression="gzip")
    mutation = pd.DataFrame(
        {"TCGA-AB-0001-01": [1, 0], "TCGA-AB-0002-01": [0, 1]},
        index=["TP53", "KRAS"],
    )
    mutation.to_csv(mutation_dir / "LUAD_mc3_gene_level.txt.gz", sep="\t", compression="gzip")
    gtf = (
        'chr1\ttest\tgene\t1\t2\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "A"; gene_type "protein_coding";\n'
        'chr1\ttest\tgene\t3\t4\t.\t+\t.\tgene_id "ENSG2.1"; gene_name "A"; gene_type "protein_coding";\n'
        'chr1\ttest\tgene\t5\t6\t.\t+\t.\tgene_id "ENSG3.1"; gene_name "B"; gene_type "lncRNA";\n'
    )
    with gzip.open(annotation_dir / "gencode.v36.annotation.gtf.gz", "wt", encoding="utf-8") as handle:
        handle.write(gtf)
    prepared = XenaTCGALoader(config, data_dir=tmp_path, use_cache=False).prepare(["LUAD"])
    assert prepared.expression.shape == (2, 1)
    assert prepared.expression.columns.tolist() == ["A"]
    assert prepared.expression.loc[sample_a, "A"] == pytest.approx(2.0)
    assert set(prepared.mutations.columns) == {"TP53", "KRAS"}


def test_cli_contains_mutation_tmb_and_shap_commands():
    parser = build_parser()
    help_text = parser.format_help()
    assert "tmb" in help_text
    assert "explain" in help_text
    assert "head-weights" in help_text


def test_tutorial_notebook_has_no_stored_outputs():
    notebook_path = Path(__file__).parents[1] / "examples" / "LUAD_tutorial.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in code_cells)


def test_tmb_uses_shared_folds_and_writes_outputs(tmp_path):
    pytest.importorskip("xgboost")
    from e2m.tmb import run_tmb_cross_validation

    rng = np.random.default_rng(5)
    samples = [f"S{index}" for index in range(30)]
    expression = pd.DataFrame(rng.normal(size=(30, 4)), index=samples)
    raw_tmb = np.arange(1, 31)
    tmb = pd.DataFrame({"TMB": raw_tmb, "TMB_log2": np.log2(raw_tmb + 1)}, index=samples)
    cancer = pd.Series(["LUAD"] * 30, index=samples)
    config = load_config()
    config["evaluation"]["cv_folds"] = 3
    config["model"]["xgboost"].update({"n_estimators": 2, "n_jobs": 1, "max_depth": 2})
    result = run_tmb_cross_validation(expression, tmb, cancer, config, tmp_path)
    assert result["split_method"] == "shuffled_kfold"
    assert (tmp_path / "oof_predictions.csv").exists()
    assert (tmp_path / "summary.csv").exists()
