"""Run manifest and schema lock stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_markdown_report, write_table
from fit.raw import load_raw_tables, validate_raw_tables


def build_run_manifest(
    raw_dir: str | Path,
    output_dir: str | Path,
    experiment_config: str | Path | None = None,
    model_schema: str | Path | None = None,
) -> dict[str, Path]:
    """Lock raw input files, dimensions, and schema before modeling."""

    tables = load_raw_tables(raw_dir)
    validate_raw_tables(tables)
    out = ensure_dir(output_dir)
    index = _analysis_index(tables)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "raw_dir": str(Path(raw_dir)),
        "experiment_config": str(experiment_config) if experiment_config else None,
        "model_schema": str(model_schema) if model_schema else None,
        "raw_files": {name: str(path) for name, path in _resolved_raw_paths(raw_dir).items()},
        "weeks": sorted(int(value) for value in index["week"].unique()),
        "conditions": sorted(str(value) for value in index["condition"].unique()),
        "replicates": sorted(str(value) for value in index["replicate"].unique()),
        "states": list(schemas.STATE_NAMES),
        "species": list(schemas.SPECIES),
        "n_analysis_rows": int(len(index)),
    }
    paths = {
        "run_manifest": out / "run_manifest.json",
        "analysis_index": out / "analysis_index.parquet",
        "schema_check_report": out / "schema_check_report.md",
    }
    write_json(paths["run_manifest"], manifest)
    write_table(index, paths["analysis_index"])
    write_markdown_report(
        paths["schema_check_report"],
        "Schema Check Report",
        [
            ("Scope", "Raw files, dimensions, state names, species names, and non-negative numeric fields were validated before modeling."),
            ("Dimensions", f"weeks={manifest['weeks']}; conditions={manifest['conditions']}; replicates={manifest['replicates']}"),
            ("Files", "\n".join(f"- {name}: {path}" for name, path in manifest["raw_files"].items())),
        ],
    )
    return paths


def load_run_manifest(manifest_path: str | Path) -> dict:
    manifest = read_json(manifest_path)
    required = {"raw_files", "weeks", "conditions", "replicates", "states", "species"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"run_manifest missing required fields: {missing}")
    return manifest


def _analysis_index(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_keys = (
        pd.concat(
            [
                tables["flow"][["week", "condition", "replicate"]].drop_duplicates(),
                tables["qpcdr"][["week", "condition", "replicate"]].drop_duplicates(),
                tables["ectag"][["week", "condition", "replicate"]].drop_duplicates(),
                tables["ddpcr"][["week", "condition", "replicate"]].drop_duplicates(),
                tables["cell_count"][["week", "condition", "replicate"]].drop_duplicates(),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["week", "condition", "replicate"])
    )
    states = pd.DataFrame({"state_gate": list(schemas.STATE_NAMES)})
    species = pd.DataFrame({"species": list(schemas.SPECIES)})
    index = base_keys.merge(states, how="cross").merge(species, how="cross")
    index["analysis_key"] = [
        schemas.stable_feature_id(
            "analysis",
            week=row.week,
            condition=row.condition,
            replicate=row.replicate,
            state_gate=row.state_gate,
            species=row.species,
        )
        for row in index.itertuples(index=False)
    ]
    return index.reset_index(drop=True)


def _resolved_raw_paths(raw_dir: str | Path) -> dict[str, Path]:
    from fit.raw import table_paths_from_raw_dir

    return table_paths_from_raw_dir(raw_dir)
