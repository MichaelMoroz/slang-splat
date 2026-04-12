from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults.json"


def defaults_path() -> Path:
    return DEFAULTS_PATH


def load_defaults() -> dict[str, Any]:
    with DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_defaults(data: dict[str, Any]) -> None:
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def training_build_arg_defaults() -> dict[str, object]:
    return dict(load_defaults()["training_build_args"])


def renderer_defaults() -> dict[str, object]:
    return dict(load_defaults()["renderer"])


def viewer_defaults() -> dict[str, dict[str, object]]:
    viewer = load_defaults()["viewer"]
    return {key: dict(value) for key, value in viewer.items()}


def cli_defaults() -> dict[str, dict[str, object]]:
    cli = load_defaults()["cli"]
    return {key: dict(value) for key, value in cli.items()}


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
