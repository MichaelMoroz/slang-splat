from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults.json"
CONFIGS_PATH = Path(__file__).resolve().parents[1] / "configs"


def defaults_path() -> Path:
    return DEFAULTS_PATH


def configs_path() -> Path:
    return CONFIGS_PATH


def list_configs() -> tuple[Path, ...]:
    if not CONFIGS_PATH.exists():
        return ()
    return tuple(sorted(path for path in CONFIGS_PATH.glob("*.json") if path.is_file()))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return data


def deep_merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_defaults() -> dict[str, Any]:
    return load_json(DEFAULTS_PATH)


def write_defaults(data: dict[str, Any]) -> None:
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    base = load_defaults()
    if path is None:
        return base
    return deep_merge_config(base, load_json(path))


def write_config(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def training_build_arg_defaults() -> dict[str, object]:
    return dict(load_defaults()["training_build_args"])


def renderer_defaults() -> dict[str, object]:
    return dict(load_defaults()["renderer"])


def viewer_defaults() -> dict[str, dict[str, object]]:
    viewer = load_defaults()["viewer"]
    return {key: dict(value) for key, value in viewer.items()}


def json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
