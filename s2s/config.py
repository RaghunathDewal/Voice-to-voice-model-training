"""YAML configuration with inheritance (`base:` key) and command-line overrides.

A config is loaded into a `Config` object that allows both attribute access
(`cfg.thinker.model`) and dict access (`cfg["thinker"]["model"]`).
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access; nested dicts are converted recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"config has no key '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return {k: _unwrap(v) for k, v in self.items()}


def _wrap(v: Any) -> Any:
    if isinstance(v, Config):
        return v
    if isinstance(v, dict):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v: Any) -> Any:
    if isinstance(v, Config):
        return v.to_dict()
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return copy.deepcopy(v)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        data = deep_merge(_load_yaml_with_base(base_path), data)
    return data


def parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got '{item}'")
    key, raw = item.split("=", 1)
    return key.split("."), yaml.safe_load(raw)


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    data = _load_yaml_with_base(path)
    for item in overrides or []:
        keys, value = parse_override(item)
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return Config(data)


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
