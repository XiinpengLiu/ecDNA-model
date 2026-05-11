"""Run manifest and data-mask lock stage for the bulk fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_markdown_report, write_table
from fit.raw import load_raw_tables, table_paths_from_raw_dir, validate_raw_tables


def build_run_manifest(
    raw_dir: str | Path,
    output_dir: str | Path,
    experiment_config: str | Path | None = None,
    model_schema: str | Path | None = None,
) -> dict[str, Path]:
    """Lock raw bulk inputs and explicitly close unavailable modalities."""

    tables = load_raw_tables(raw_dir)
    validate_raw_tables(tables)
    out = ensure_dir(output_dir)
    index = _analysis_index(tables)
    mask = {
        "ddpcr_bulk": True,
        "cell_count": True,
        "flow_3group_early": True,
        "flow_4state": False,
        "qpcdr_sorted": False,
        "ectag_single_cell": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method_source": "markdown/fit_method.md",
        "workflow": "00_manifest_bulk",
        "raw_dir": str(Path(raw_dir)),
        "experiment_config": str(experiment_config) if experiment_config else None,
        "model_schema": str(model_schema) if model_schema else None,
        "raw_files": {name: str(path) for name, path in table_paths_from_raw_dir(raw_dir).items() if path.exists()},
        "weeks": sorted(int(value) for value in index["week"].unique()),
        "conditions": sorted(str(value) for value in index["condition"].unique()),
        "replicates": sorted(str(value) for value in index["replicate"].unique()),
        "species": list(schemas.SPECIES),
        "flow3_groups": list(schemas.FLOW3_GROUPS),
        "available_data_mask": mask,
        "n_analysis_rows": int(len(index)),
    }
    paths = {
        "run_manifest": out / "run_manifest.json",
        "analysis_index": out / "analysis_index.parquet",
        "available_data_mask": out / "available_data_mask.json",
    }
    write_json(paths["run_manifest"], manifest)
    write_table(index, paths["analysis_index"])
    write_json(paths["available_data_mask"], mask)
    write_markdown_report(
        out / "schema_check_report.md",
        "Bulk Manifest Check Report",
        [
            ("Opened Channels", "ddPCR bulk, cell count, and early three-group flow are available."),
            ("Closed Channels", "qPCDR, ecTAG, flow4, and state-specific copy likelihoods are closed by data mask."),
            ("Dimensions", f"weeks={manifest['weeks']}; conditions={manifest['conditions']}; replicates={manifest['replicates']}"),
        ],
    )
    return paths


def load_run_manifest(manifest_path: str | Path) -> dict:
    manifest = read_json(manifest_path)
    required = {"raw_files", "weeks", "conditions", "replicates", "species", "available_data_mask"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"run_manifest missing required fields: {missing}")
    return manifest


def _analysis_index(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ddpcr = tables["ddpcr"][["week", "condition", "replicate", "species"]].drop_duplicates()
    cell = tables["cell_count"][["week", "condition", "replicate"]].drop_duplicates()
    flow = tables["flow"][["week", "condition", "replicate"]].drop_duplicates()
    base = pd.concat(
        [
            ddpcr[["week", "condition", "replicate"]],
            cell,
            flow,
        ],
        ignore_index=True,
    ).drop_duplicates()
    rows = []
    for row in base.itertuples(index=False):
        for species in schemas.SPECIES:
            rows.append(
                {
                    "week": int(row.week),
                    "condition": str(row.condition),
                    "replicate": str(row.replicate),
                    "species": species,
                    "ddpcr_available": bool(((ddpcr["week"] == row.week) & (ddpcr["condition"] == row.condition) & (ddpcr["replicate"] == row.replicate) & (ddpcr["species"] == species)).any()),
                    "cell_count_available": bool(((cell["week"] == row.week) & (cell["condition"] == row.condition) & (cell["replicate"] == row.replicate)).any()),
                    "flow3_available": bool(((flow["week"] == row.week) & (flow["condition"] == row.condition) & (flow["replicate"] == row.replicate)).any()),
                }
            )
    index = pd.DataFrame(rows).sort_values(["week", "condition", "replicate", "species"]).reset_index(drop=True)
    index["analysis_key"] = [
        schemas.stable_feature_id("analysis", week=row.week, condition=row.condition, replicate=row.replicate, species=row.species)
        for row in index.itertuples(index=False)
    ]
    return index
