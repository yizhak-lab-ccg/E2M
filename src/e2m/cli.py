from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import _read_yaml, deep_update, format_options, load_config, set_verbose
from .data import EXPRESSION_DATASETS, EXPRESSION_TRANSFORMS, XenaTCGALoader, align_expression, normalize_cancers
from .interpretation import explain_single_target
from .models import TREE_MODELS, E2MModel
from .training import cross_validate_mutations, cross_validate_tmb


def _data_arguments(parser, require_cancer=True, include_mutation_targets=True, include_cache=True):
    parser.add_argument("--config", help="Optional YAML configuration file.")
    parser.add_argument("--data-dir", help="Download and cache directory. Default: ./e2m_data")
    parser.add_argument(
        "--cancer",
        action="append",
        required=require_cancer,
        help="TCGA cancer code. Repeat the option or use commas for multiple cancers.",
    )
    parser.add_argument("--expression-dataset", choices=EXPRESSION_DATASETS)
    parser.add_argument("--transform", choices=EXPRESSION_TRANSFORMS)
    parser.add_argument(
        "--log-offset",
        type=float,
        default=None,
        help="Additive offset used when inverting Xena's log2 for raw/log1p (default: 1.0).",
    )
    parser.add_argument(
        "--protein-coding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict expression features and mutation targets to protein-coding genes (default: yes).",
    )
    if include_mutation_targets:
        parser.add_argument("--min-prevalence", type=float)
        parser.add_argument("--max-targets", type=int)
    if include_cache:
        parser.add_argument("--no-cache", action="store_true")


def _model_arguments(parser):
    parser.add_argument("--hidden-layers", help="Comma-separated encoder sizes, for example 512,256.")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--patience",
        type=int,
        help="Early-stopping patience for the internal validation split used in `cv`. "
        "Full-cohort `train` runs the fixed epoch count and does not early-stop.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e2m", description="Predict mutation status and TMB from RNA expression.")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and prepare TCGA expression and mutation data.")
    _data_arguments(download)
    download.add_argument("--output", help="Optional directory for prepared expression, mutation, and sample tables.")

    cv = commands.add_parser("cv", help="Run held-out multitask mutation prediction.")
    _data_arguments(cv)
    _model_arguments(cv)
    cv.add_argument("--output", required=True)

    train = commands.add_parser("train", help="Train a full-cohort model bundle.")
    _data_arguments(train)
    _model_arguments(train)
    train.add_argument("--output", required=True)

    tmb = commands.add_parser("tmb", help="Run held-out TMB prediction with XGBoost.")
    _data_arguments(tmb, include_mutation_targets=False, include_cache=False)
    tmb.add_argument("--output", required=True)

    predict = commands.add_parser("predict", help="Predict mutation probabilities for a samples by genes CSV.")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--expression", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--min-feature-overlap", type=float)

    embed = commands.add_parser("embed", help="Extract sample embeddings.")
    embed.add_argument("--model-dir", required=True)
    embed.add_argument("--expression", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--min-feature-overlap", type=float)

    weights = commands.add_parser("head-weights", help="Extract one prediction-head vector per target.")
    weights.add_argument("--model-dir", required=True)
    weights.add_argument("--output", required=True)

    explain = commands.add_parser("explain", help="Explain one mutation output with SHAP.")
    explain.add_argument("--model-dir", required=True)
    explain.add_argument("--target", required=True)
    explain.add_argument(
        "--method",
        choices=[*TREE_MODELS, "neural"],
        default="xgboost",
        help="Attribution model: a tree ML model for Tree SHAP (default xgboost), or neural.",
    )
    explain.add_argument("--output", required=True)
    explain.add_argument("--config")
    explain.add_argument("--data-dir")
    explain.add_argument("--no-cache", action="store_true")

    commands.add_parser(
        "options",
        help="List accepted expression datasets, transforms, TMB models, and TCGA cancer codes.",
    )

    for name, sub in commands.choices.items():
        sub.add_argument(
            "--verbose", "-v", action="store_true", help="Print progress messages to stderr."
        )
    return parser


def _data_overrides(args) -> dict:
    values = {}
    if getattr(args, "expression_dataset", None):
        values["expression_dataset"] = args.expression_dataset
    if getattr(args, "transform", None):
        values["expression_transform"] = args.transform
    if getattr(args, "log_offset", None) is not None:
        values["xena_log_offset"] = args.log_offset
    if getattr(args, "min_prevalence", None) is not None:
        values["min_mutation_prevalence"] = args.min_prevalence
    if getattr(args, "max_targets", None) is not None:
        values["max_mutation_targets"] = args.max_targets
    if getattr(args, "protein_coding", None) is not None:
        values["protein_coding_only"] = args.protein_coding
        values["protein_coding_mutation_targets"] = args.protein_coding
    return values


def _model_overrides(args) -> dict:
    values = {}
    if getattr(args, "hidden_layers", None):
        values["hidden_layers"] = [int(value) for value in args.hidden_layers.split(",")]
    mapping = {
        "dropout": "dropout_rate",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "patience": "patience",
    }
    for source, destination in mapping.items():
        value = getattr(args, source, None)
        if value is not None:
            values[destination] = value
    return values


def _tcga(args, use_cache: bool):
    """Resolved config, selected cancers, and a TCGA loader for a data command."""
    config = load_config(args.config, {"data": _data_overrides(args)})
    cancers = normalize_cancers(args.cancer or config["data"].get("cancers"))
    config["data"]["cancers"] = cancers
    return config, cancers, XenaTCGALoader(config, data_dir=args.data_dir, use_cache=use_cache)


def _download(args) -> dict:
    _, cancers, loader = _tcga(args, not args.no_cache)
    data = loader.prepare(cancers)
    summary = {
        "n_samples": len(data.expression),
        "n_expression_features": data.expression.shape[1],
        "n_mutation_targets": data.mutations.shape[1],
        "cancers": sorted(data.cancer.unique().tolist()),
    }
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        data.expression.to_csv(output / "expression.csv.gz", compression="gzip")
        data.mutations.to_csv(output / "mutations.csv.gz", compression="gzip")
        data.cancer.rename("cancer").to_csv(output / "samples.csv")
        summary["output_dir"] = str(output)
    return summary


def _cv(args) -> dict:
    config, cancers, loader = _tcga(args, not args.no_cache)
    data = loader.prepare(cancers)
    return cross_validate_mutations(
        data.expression, data.mutations, data.cancer, config, _model_overrides(args), args.output
    )[1]


def _train(args) -> dict:
    config, cancers, loader = _tcga(args, not args.no_cache)
    model = E2MModel(config, **_model_overrides(args)).fit(loader.prepare(cancers))
    model.metadata["data_dir"] = str(loader.data_dir)
    model.save(args.output)
    return model.metadata


def _tmb(args) -> dict:
    config, cancers, loader = _tcga(args, use_cache=False)
    expression, tmb, cancer = loader.prepare_tmb(cancers)
    return cross_validate_tmb(expression, tmb, cancer, config, output=args.output)[1]


def _predict(args) -> dict:
    model = E2MModel.load(args.model_dir)
    expression = _read_expression(args.expression)
    output = model.predict(expression, args.min_feature_overlap)
    _write_frame(output, args.output)
    return {
        "n_samples": len(output),
        "n_targets": len(model.targets),
        "feature_overlap": _overlap(model, expression),
        "output": str(args.output),
    }


def _embed(args) -> dict:
    model = E2MModel.load(args.model_dir)
    expression = _read_expression(args.expression)
    output = model.embed(expression, args.min_feature_overlap)
    _write_frame(output, args.output)
    return {
        "n_samples": len(output),
        "embedding_size": output.shape[1],
        "feature_overlap": _overlap(model, expression),
        "output": str(args.output),
    }


def _head_weights(args) -> dict:
    model = E2MModel.load(args.model_dir)
    output = model.head_weights()
    _write_frame(output, args.output)
    return {"n_targets": len(model.targets), "embedding_size": output.shape[1], "output": str(args.output)}


def _explain(args) -> dict:
    model = E2MModel.load(args.model_dir)
    if args.target not in model.targets:
        raise ValueError(f"Target {args.target!r} is not in this model. Available targets are stored in targets.json.")
    config = model.metadata.get("training_config") or load_config()
    if args.config:
        config = deep_update(config, _read_yaml(Path(args.config)))
    cancers = model.metadata.get("cancers")
    if not cancers:
        raise ValueError(
            "This model bundle does not record its training cohorts, so SHAP cannot re-load the "
            "expression data. Train it with cancer-labelled data (Dataset.from_tcga or the CLI "
            "`e2m train`)."
        )
    data_dir = args.data_dir or model.metadata.get("data_dir")
    data = XenaTCGALoader(config, data_dir=data_dir, use_cache=not args.no_cache).prepare(cancers)
    expression, _ = align_expression(data.expression, model.features, model.feature_means, min_overlap=1.0)
    labels = data.mutations[args.target].reindex(expression.index)
    return explain_single_target(
        model, expression, labels, args.target, model.targets.index(args.target), args.method, config, args.output
    )[1]


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


def _overlap(model: E2MModel, expression: pd.DataFrame) -> float:
    return len(set(model.features).intersection(expression.columns)) / max(len(model.features), 1)


COMMANDS = {
    "download": _download,
    "cv": _cv,
    "train": _train,
    "tmb": _tmb,
    "predict": _predict,
    "embed": _embed,
    "head-weights": _head_weights,
    "explain": _explain,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "verbose", False):
        set_verbose(True)
    if args.command == "options":
        print(format_options())
        return 0
    print(json.dumps(COMMANDS[args.command](args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
