from __future__ import annotations

import argparse
import json

from . import workflow
from .data import EXPRESSION_DATASETS, EXPRESSION_TRANSFORMS


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
    parser.add_argument("--patience", type=int)


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
    explain.add_argument("--method", choices=["xgboost", "neural"], default="xgboost")
    explain.add_argument("--output", required=True)
    explain.add_argument("--config")
    explain.add_argument("--data-dir")
    explain.add_argument("--no-cache", action="store_true")
    return parser


def _data_overrides(args) -> dict:
    values = {}
    if getattr(args, "expression_dataset", None):
        values["expression_dataset"] = args.expression_dataset
    if getattr(args, "transform", None):
        values["expression_transform"] = args.transform
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        result = workflow.download(
            args.cancer, args.config, args.data_dir, _data_overrides(args), not args.no_cache, args.output
        )
    elif args.command == "cv":
        result = workflow.cross_validate(
            args.cancer,
            args.output,
            args.config,
            args.data_dir,
            _data_overrides(args),
            _model_overrides(args),
            not args.no_cache,
        )
    elif args.command == "train":
        result = workflow.train(
            args.cancer,
            args.output,
            args.config,
            args.data_dir,
            _data_overrides(args),
            _model_overrides(args),
            not args.no_cache,
        )
    elif args.command == "tmb":
        result = workflow.predict_tmb(
            args.cancer, args.output, args.config, args.data_dir, _data_overrides(args)
        )
    elif args.command == "predict":
        result = workflow.predict(args.model_dir, args.expression, args.output, args.min_feature_overlap)
    elif args.command == "embed":
        result = workflow.embed(args.model_dir, args.expression, args.output, args.min_feature_overlap)
    elif args.command == "head-weights":
        result = workflow.head_weights(args.model_dir, args.output)
    elif args.command == "explain":
        result = workflow.explain(
            args.model_dir,
            args.target,
            args.method,
            args.output,
            args.config,
            args.data_dir,
            not args.no_cache,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
