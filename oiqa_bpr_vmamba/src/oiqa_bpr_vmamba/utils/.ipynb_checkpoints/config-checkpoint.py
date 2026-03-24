from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config at {path} must be a mapping.')
    base_key = cfg.pop('_base_', None)
    if base_key is None:
        return cfg
    base_path = (path.parent / base_key).resolve() if not Path(base_key).is_absolute() else Path(base_key)
    base_cfg = load_yaml_config(base_path)
    return _deep_update(base_cfg, cfg)
