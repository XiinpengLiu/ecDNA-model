"""File IO utilities for deterministic fit stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(resolved)
    if suffix == ".tsv":
        return pd.read_csv(resolved, sep="\t")
    if suffix == ".json":
        return pd.read_json(resolved)
    raise ValueError(f"Unsupported table extension for {resolved}")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(resolved, index=False)
    elif suffix == ".csv":
        df.to_csv(resolved, index=False)
    elif suffix == ".tsv":
        df.to_csv(resolved, index=False, sep="\t")
    else:
        raise ValueError(f"Unsupported table extension for {resolved}")
    return resolved


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")
    return resolved


def write_npz(path: str | Path, **arrays: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    np.savez(resolved, **arrays)
    return resolved


def write_markdown_report(path: str | Path, title: str, sections: list[tuple[str, str]]) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    resolved.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return resolved


def write_text_pdf(path: str | Path, title: str, lines: list[str]) -> Path:
    """Write a small deterministic PDF report using matplotlib."""

    resolved = Path(path)
    ensure_dir(resolved.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        fig.text(0.08, y, str(line), fontsize=9, va="top", wrap=True)
        y -= 0.035
        if y < 0.08:
            break
    fig.savefig(resolved, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return resolved


def write_dataset_netcdf(path: str | Path, variables: dict[str, Any], attrs: dict[str, Any] | None = None) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    import xarray as xr

    data_vars = {}
    for name, values in variables.items():
        arr = np.asarray(values)
        dims = tuple(f"{name}_dim_{idx}" for idx in range(arr.ndim))
        data_vars[name] = (dims, arr)
    dataset = xr.Dataset(data_vars=data_vars, attrs=_to_jsonable(attrs or {}))
    dataset.to_netcdf(resolved)
    return resolved


def require_paths(paths: list[str | Path], label: str) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: {', '.join(missing)}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
