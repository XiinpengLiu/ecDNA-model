"""Lean file IO helpers for the config-centered local ABC-SMC fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular file, dispatching by extension (csv/tsv/parquet/json)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table extension for {path}.")


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` to ``path``; extension selects csv vs parquet."""
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix in (".csv",):
        frame.to_csv(path, index=False)
    elif suffix == ".tsv":
        frame.to_csv(path, index=False, sep="\t")
    else:
        raise ValueError(f"Unsupported table extension for {path}.")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    return path


def write_yaml(payload: Any, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_markdown_report(text: str, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable.")
