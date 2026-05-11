"""Raw and clean data ingestion for the bulk-only fit method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_table, write_json, write_markdown_report, write_table


@dataclass(frozen=True)
class CleanDataPaths:
    ddpcr: Path
    cell_count: Path
    flow3: Path
    drug_metadata: Path
    qpcdr_unavailable: Path
    ectag_unavailable: Path
    report: Path


def table_paths_from_raw_dir(raw_dir: str | Path) -> dict[str, Path]:
    base = Path(raw_dir)
    return {
        "ddpcr": _find_raw_table(base, "ddpcr"),
        "cell_count": _find_raw_table(base, "cell_count"),
        "flow": _find_raw_table(base, "flow3") if _find_raw_table(base, "flow3").exists() else _find_raw_table(base, "flow"),
        "metadata": _find_raw_table(base, "metadata"),
    }


def load_raw_tables(raw_dir: str | Path | None = None, **paths: str | Path | None) -> dict[str, pd.DataFrame]:
    resolved = table_paths_from_raw_dir(raw_dir) if raw_dir is not None else {}
    for key, value in paths.items():
        if value is not None:
            resolved[key] = Path(value)
    required = ("ddpcr", "cell_count", "flow")
    missing = [key for key in required if key not in resolved or not resolved[key].exists()]
    if missing:
        raise ValueError(f"Missing raw bulk table paths for: {', '.join(missing)}")
    tables = {name: read_table(path) for name, path in resolved.items() if path.exists()}
    if "metadata" not in tables:
        tables["metadata"] = _metadata_from_conditions(tables["ddpcr"], tables["cell_count"])
    return tables


def validate_raw_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name in ("ddpcr", "cell_count", "flow"):
        if name not in tables:
            raise ValueError(f"Missing raw table: {name}")
    schemas.validate_required_columns(set(tables["ddpcr"].columns), schemas.RAW_TABLE_SCHEMAS["ddpcr"], "ddpcr")
    schemas.validate_required_columns(set(tables["cell_count"].columns), schemas.RAW_TABLE_SCHEMAS["cell_count"], "cell_count")
    schemas.validate_required_columns(set(tables["flow"].columns), schemas.RAW_TABLE_SCHEMAS["flow"], "flow")
    for name in ("ddpcr", "cell_count", "flow"):
        schemas.validate_weeks(tables[name]["week"], name)
    schemas.validate_species(tables["ddpcr"]["species"], "ddpcr")
    schemas.validate_nonnegative(tables["ddpcr"]["ddpcr_copy_number"], "ddpcr_copy_number", "ddpcr")
    schemas.validate_nonnegative(tables["cell_count"]["total_cell_count"], "total_cell_count", "cell_count")
    schemas.validate_nonnegative(tables["flow"]["fraction"], "fraction", "flow")
    if bool((tables["flow"]["fraction"].astype(float) > 1.0 + 1e-8).any()):
        raise ValueError("flow.fraction values must lie in [0, 1]")


def standardize_raw_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> CleanDataPaths:
    validate_raw_tables(tables)
    out = ensure_dir(output_dir)
    ddpcr = _clean_ddpcr(tables["ddpcr"])
    cell_count = _clean_cell_count(tables["cell_count"])
    flow3 = _clean_flow3(tables["flow"])
    metadata = _clean_metadata(tables.get("metadata", _metadata_from_conditions(ddpcr, cell_count)))

    paths = CleanDataPaths(
        ddpcr=out / "ddpcr_long.parquet",
        cell_count=out / "cell_count_long.parquet",
        flow3=out / "flow3_early_long.parquet",
        drug_metadata=out / "drug_metadata_long.parquet",
        qpcdr_unavailable=out / "qpcdr_unavailable.json",
        ectag_unavailable=out / "ectag_unavailable.json",
        report=out / "clean_qc_report.md",
    )
    write_table(ddpcr, paths.ddpcr)
    write_table(cell_count, paths.cell_count)
    write_table(flow3, paths.flow3)
    write_table(metadata, paths.drug_metadata)
    unavailable = {"available": False, "likelihood_weight": 0, "reason": "closed by bulk-only fit_method.md"}
    write_json(paths.qpcdr_unavailable, unavailable)
    write_json(paths.ectag_unavailable, unavailable)
    write_markdown_report(
        paths.report,
        "Clean QC Report",
        [
            ("Scope", "Standardized ddPCR, cell count, flow3, and drug metadata tables."),
            ("Closed Modalities", "qPCDR and ecTAG are marked unavailable and cannot enter likelihoods."),
            ("Rows", f"ddPCR={len(ddpcr)}; cell_count={len(cell_count)}; flow3={len(flow3)}; metadata={len(metadata)}"),
        ],
    )
    return paths


def ingest_raw_data(raw_dir: str | Path, output_dir: str | Path) -> CleanDataPaths:
    return standardize_raw_tables(load_raw_tables(raw_dir), output_dir)


def load_clean_tables(clean_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(clean_dir)
    return {
        "ddpcr": read_table(base / "ddpcr_long.parquet"),
        "cell_count": read_table(base / "cell_count_long.parquet"),
        "flow3": read_table(base / "flow3_early_long.parquet"),
        "drug_metadata": read_table(base / "drug_metadata_long.parquet"),
    }


def create_synthetic_raw_dataset(output_dir: str | Path, seed: int = 1) -> dict[str, Path]:
    """Create a deterministic 10-week bulk fixture."""

    rng = np.random.default_rng(seed)
    out = ensure_dir(output_dir)
    weeks = range(1, 11)
    conditions = ("ctrl", "drug_low")
    replicate = "r1"
    species_base = {"MYC": 18.0, "CDK4": 12.0, "PDGFRA": 9.0}
    ddpcr_rows: list[dict] = []
    count_rows: list[dict] = []
    flow_rows: list[dict] = []
    metadata_rows = [
        {"condition": "ctrl", "drug": "vehicle", "dose": 0.0, "dose_unit": "uM", "start_week": 1, "end_week": 10, "schedule": "continuous"},
        {"condition": "drug_low", "drug": "drug", "dose": 1.0, "dose_unit": "uM", "start_week": 1, "end_week": 10, "schedule": "continuous"},
    ]
    for condition in conditions:
        drug_effect = -0.08 if condition != "ctrl" else 0.0
        for week in weeks:
            growth = 0.13 + drug_effect
            total = 8000.0 * np.exp(growth * (week - 1))
            count_rows.append(
                {
                    "week": week,
                    "condition": condition,
                    "replicate": replicate,
                    "total_cell_count": float(max(1.0, rng.normal(total, 0.03 * total))),
                    "viability": 0.95,
                    "batch_id": "synthetic",
                }
            )
            flow = np.array([0.68, 0.22, 0.10]) + np.array([0.01, -0.005, -0.005]) * np.sin(week / 2.0)
            flow = schemas.normalize_probabilities(flow, name="synthetic flow3")
            for group, fraction in zip(schemas.FLOW3_GROUPS, flow):
                flow_rows.append(
                    {
                        "time_label": "early",
                        "week": week,
                        "condition": condition,
                        "replicate": replicate,
                        "group": group,
                        "fraction": float(fraction),
                        "fraction_or_count": float(fraction),
                        "total_events": 1000,
                        "batch_id": "synthetic",
                    }
                )
            for species, base in species_base.items():
                copy_velocity = 0.035 + (0.02 if species == "MYC" and condition != "ctrl" else 0.0)
                mean = base * np.exp(copy_velocity * (week - 1))
                ddpcr_rows.append(
                    {
                        "week": week,
                        "condition": condition,
                        "replicate": replicate,
                        "species": species,
                        "ddpcr_copy_number": float(max(0.01, rng.normal(mean, 0.04 * mean))),
                        "ddpcr_sd_or_ci": float(0.08 * mean),
                        "batch_id": "synthetic",
                    }
                )

    outputs = {
        "ddpcr": out / "ddpcr.csv",
        "cell_count": out / "cell_count.csv",
        "flow": out / "flow3.csv",
        "metadata": out / "metadata.csv",
    }
    pd.DataFrame(ddpcr_rows).to_csv(outputs["ddpcr"], index=False)
    pd.DataFrame(count_rows).to_csv(outputs["cell_count"], index=False)
    pd.DataFrame(flow_rows).to_csv(outputs["flow"], index=False)
    pd.DataFrame(metadata_rows).to_csv(outputs["metadata"], index=False)
    return outputs


def _find_raw_table(base: Path, name: str) -> Path:
    for direct_name in (f"{name}.csv", f"{name}.tsv", f"{name}.parquet"):
        direct = base / direct_name
        if direct.exists():
            return direct
    folder = base / name
    if folder.exists():
        matches = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".csv", ".tsv", ".parquet"})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Raw input folder {folder} contains multiple candidate tables")
    return base / f"{name}.csv"


def _clean_ddpcr(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["ddpcr_copy_number"] = clean["ddpcr_copy_number"].astype(float)
    clean["ddpcr_sd_or_ci"] = pd.to_numeric(clean["ddpcr_sd_or_ci"], errors="coerce")
    return clean.sort_values(["week", "condition", "replicate", "species"]).reset_index(drop=True)


def _clean_cell_count(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["total_cell_count"] = clean["total_cell_count"].astype(float)
    clean["viability"] = pd.to_numeric(clean["viability"], errors="coerce")
    return clean.sort_values(["week", "condition", "replicate"]).reset_index(drop=True)


def _clean_flow3(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    if "group" not in clean.columns:
        raise ValueError("flow3 input must include method flow3 group labels; four-state flow is closed")
    invalid = sorted(set(clean["group"].astype(str)).difference(schemas.FLOW3_GROUPS))
    if invalid:
        raise ValueError(f"flow3 contains invalid group labels: {invalid}")
    if "fraction_or_count" not in clean.columns:
        clean["fraction_or_count"] = clean["fraction"]
    if "total_events" not in clean.columns:
        clean["total_events"] = np.nan
    clean["week"] = clean["week"].astype(int)
    clean["fraction"] = clean["fraction"].astype(float)
    rows = []
    for key, group in clean.groupby(["week", "condition", "replicate"], dropna=False):
        pooled = group.groupby("group", as_index=False).agg(fraction=("fraction", "sum"), total_events=("total_events", "max"))
        values = pooled.set_index("group")["fraction"].reindex(schemas.FLOW3_GROUPS).fillna(0.0).to_numpy(dtype=float)
        values = schemas.normalize_probabilities(values + 1e-12, name="flow3 fractions")
        total_events = pooled["total_events"].dropna()
        n_eff = float(total_events.iloc[0]) if len(total_events) else 300.0
        for group_name, fraction in zip(schemas.FLOW3_GROUPS, values):
            rows.append(
                {
                    "time_label": "early",
                    "week": int(key[0]),
                    "condition": str(key[1]),
                    "replicate": str(key[2]),
                    "group": group_name,
                    "fraction_or_count": float(fraction),
                    "fraction": float(fraction),
                    "total_events": n_eff,
                    "batch_id": str(group["batch_id"].iloc[0]) if "batch_id" in group else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["week", "condition", "replicate", "group"]).reset_index(drop=True)


def _clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "drug", "dose", "dose_unit", "start_week", "end_week", "schedule"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
    clean = df.copy()
    clean["dose"] = clean["dose"].astype(float)
    clean["start_week"] = clean["start_week"].astype(int)
    clean["end_week"] = clean["end_week"].astype(int)
    return clean.sort_values("condition").reset_index(drop=True)


def _metadata_from_conditions(*tables: pd.DataFrame) -> pd.DataFrame:
    conditions = sorted({str(value) for table in tables for value in table["condition"].dropna().unique()})
    return pd.DataFrame(
        [
            {
                "condition": condition,
                "drug": "vehicle" if condition.lower() in {"ctrl", "control"} else condition,
                "dose": 0.0 if condition.lower() in {"ctrl", "control"} else 1.0,
                "dose_unit": "a.u.",
                "start_week": 1,
                "end_week": 10,
                "schedule": "unspecified",
            }
            for condition in conditions
        ]
    )
