#!/usr/bin/env python3
"""Shared I/O helpers for the all-method visualization experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_REGISTRY = EXPERIMENT_DIR / "configs" / "model_registry.json"


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return read_json(path)


def output_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return resolve(override if override is not None else config["output_root"])


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_output_tree(root: Path) -> None:
    for name in ("tables", "figures", "html", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
