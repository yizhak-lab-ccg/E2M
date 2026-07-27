from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def default_config_path():
    return files("e2m").joinpath("default_config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else default_config_path()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {source}")
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config()
    if path:
        config = deep_update(config, load_config(path))
    return deep_update(config, overrides)
