from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(source) -> dict[str, Any]:
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


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Packaged defaults, updated by the YAML at `path`, then by `overrides`."""
    config = _read_yaml(files("e2m").joinpath("default_config.yaml"))
    if path:
        config = deep_update(config, _read_yaml(Path(path)))
    return deep_update(config, overrides)
