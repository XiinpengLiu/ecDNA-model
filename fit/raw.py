"""Raw and clean data ingestion for the fit pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_table, write_json, write_markdown_report, write_table


@dataclass(frozen=True)
class CleanDataPaths:
    flow: Path
    qpcdr: Path
    ectag: Path
    ddpcr: Path
    cell_count: Path
    report: Path


def table_paths_from_raw_dir(raw_dir: str | Path) -> dict[str, Path]:
    base = Path(raw_dir)
    return {
        "flow": _find_raw_table(base, "flow"),
        "qpcdr": _find_raw_table(base, "qpcdr"),
        "ectag": _find_raw_table(base, "ectag"),
        "ddpcr": _find_raw_table(base, "ddpcr"),
        "cell_count": _find_raw_table(base, "cell_count"),
    }


def load_raw_tables(raw_dir: str | Path | None = None, **paths: str | Path | None) -> dict[str, pd.DataFrame]:
    resolved = table_paths_from_raw_dir(raw_dir) if raw_dir is not None else {}
    for key, value in paths.items():
        if value is not None:
            resolved[key] = Path(value)
    missing = [key for key in schemas.RAW_TABLE_SCHEMAS if key not in resolved]
    if missing:
        raise ValueError(f"Missing raw table paths for: {', '.join(missing)}")
    return {name: read_table(path) for name, path in resolved.items()}


def validate_raw_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, required in schemas.RAW_TABLE_SCHEMAS.items():
        if name not in tables:
            raise ValueError(f"Missing raw table: {name}")
        df = tables[name]
        schemas.validate_required_columns(set(df.columns), required, name)
        if "week" in df:
            schemas.validate_weeks(df["week"], name)
        if "state_gate" in df:
            schemas.validate_states(df["state_gate"], name)
        if "species" in df:
            schemas.validate_species(df["species"], name)

    schemas.validate_nonnegative(tables["flow"]["pre_sort_count"], "pre_sort_count", "flow")
    schemas.validate_nonnegative(tables["flow"]["post_sort_count"], "post_sort_count", "flow")
    schemas.validate_nonnegative(tables["flow"]["fraction"], "fraction", "flow")
    schemas.validate_nonnegative(tables["ectag"]["ectag_count"], "ectag_count", "ectag")
    schemas.validate_nonnegative(tables["ddpcr"]["ddpcr_copy_number"], "ddpcr_copy_number", "ddpcr")
    schemas.validate_nonnegative(tables["cell_count"]["total_cell_count"], "total_cell_count", "cell_count")

    bad_fraction = tables["flow"]["fraction"].astype(float) > 1.0 + 1e-8
    if bool(bad_fraction.any()):
        raise ValueError("flow.fraction values must lie in [0, 1]")

    flow_keys = ["week", "condition", "replicate"]
    state_counts = tables["flow"].groupby(flow_keys)["state_gate"].nunique()
    incomplete = state_counts[state_counts < len(schemas.STATE_NAMES)]
    if not incomplete.empty:
        raise ValueError("Each week-condition-replicate needs all four flow state gates")


def standardize_raw_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> CleanDataPaths:
    validate_raw_tables(tables)
    out = ensure_dir(output_dir)

    clean = {
        "flow": _clean_flow(tables["flow"]),
        "qpcdr": _clean_qpcdr(tables["qpcdr"]),
        "ectag": _clean_ectag(tables["ectag"]),
        "ddpcr": _clean_ddpcr(tables["ddpcr"]),
        "cell_count": _clean_cell_count(tables["cell_count"]),
    }

    paths = CleanDataPaths(
        flow=out / "flow_long.parquet",
        qpcdr=out / "qpcdr_long.parquet",
        ectag=out / "ectag_cell_long.parquet",
        ddpcr=out / "ddpcr_long.parquet",
        cell_count=out / "cell_count_long.parquet",
        report=out / "raw_data_qc_report.md",
    )
    write_table(clean["flow"], paths.flow)
    write_table(clean["qpcdr"], paths.qpcdr)
    write_table(clean["ectag"], paths.ectag)
    write_table(clean["ddpcr"], paths.ddpcr)
    write_table(clean["cell_count"], paths.cell_count)
    write_json(out / "clean_data_manifest.json", {name: str(getattr(paths, name)) for name in clean})

    report = [
        ("Scope", "Raw files were converted to long clean tables without biological inference."),
        (
            "Method Guards",
            "\n".join(
                [
                    "- ddPCR is stored only as a bulk pooled mean anchor.",
                    "- ecTAG records remain single-cell and species-specific.",
                    "- No code-level ecTAG detection ceiling is introduced during ingestion.",
                ]
            ),
        ),
        ("Rows", "\n".join(f"- {name}: {len(df)} rows" for name, df in clean.items())),
    ]
    write_markdown_report(paths.report, "Raw Data QC Report", report)
    return paths


def ingest_raw_data(raw_dir: str | Path, output_dir: str | Path) -> CleanDataPaths:
    return standardize_raw_tables(load_raw_tables(raw_dir), output_dir)


def load_clean_tables(clean_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(clean_dir)
    return {
        "flow": read_table(base / "flow_long.parquet"),
        "qpcdr": read_table(base / "qpcdr_long.parquet"),
        "ectag": read_table(base / "ectag_cell_long.parquet"),
        "ddpcr": read_table(base / "ddpcr_long.parquet"),
        "cell_count": read_table(base / "cell_count_long.parquet"),
    }


def create_synthetic_raw_dataset(output_dir: str | Path, seed: int = 1) -> dict[str, Path]:
    """Create a small deterministic fixture that exercises every fit channel."""

    rng = np.random.default_rng(seed)
    out = ensure_dir(output_dir)
    weeks = (1, 2, 3)
    condition = "ctrl"
    replicate = "r1"
    flow_fraction_by_week = {
        1: np.array([0.40, 0.30, 0.20, 0.10]),
        2: np.array([0.34, 0.34, 0.22, 0.10]),
        3: np.array([0.28, 0.37, 0.24, 0.11]),
    }
    base_mu = np.array(
        [
            [10.0, 24.0, 12.0],
            [14.0, 12.0, 27.0],
            [8.0, 10.0, 11.0],
            [6.0, 8.0, 10.0],
        ],
        dtype=float,
    )

    flow_rows = []
    qpcdr_rows = []
    ectag_rows = []
    ddpcr_rows = []
    cell_count_rows = []
    for week in weeks:
        fractions = flow_fraction_by_week[week]
        total_events = 1000
        for state_index, state in enumerate(schemas.STATE_NAMES):
            flow_rows.append(
                {
                    "week": week,
                    "condition": condition,
                    "replicate": replicate,
                    "state_gate": state,
                    "pre_sort_count": int(total_events * fractions[state_index]),
                    "post_sort_count": int(total_events * fractions[state_index] * 0.95),
                    "fraction": float(fractions[state_index]),
                    "sort_purity": np.nan,
                    "marker_panel": "synthetic-four-state",
                    "batch_id": "synthetic",
                }
            )
            for species_index, species in enumerate(schemas.SPECIES):
                mean_copy = float(base_mu[state_index, species_index] * (1.0 + 0.08 * (week - 1)))
                for technical_rep in (1, 2):
                    qpcdr_rows.append(
                        {
                            "week": week,
                            "condition": condition,
                            "replicate": replicate,
                            "state_gate": state,
                            "species": species,
                            "technical_rep": technical_rep,
                            "raw_Ct_or_Cq": np.nan,
                            "relative_copy_number": max(0.01, rng.normal(mean_copy, 0.06 * mean_copy)),
                            "plate_id": "synthetic-plate",
                            "batch_id": "synthetic",
                        }
                    )
                for cell_index in range(36):
                    zero = rng.random() < max(0.05, 0.35 - 0.05 * state_index)
                    value = 0 if zero else int(rng.negative_binomial(3, 3 / (3 + mean_copy)))
                    if rng.random() < 0.08:
                        value += int(rng.integers(20, 80))
                    ectag_rows.append(
                        {
                            "week": week,
                            "condition": condition,
                            "replicate": replicate,
                            "state_gate": state,
                            "cell_id": f"w{week}-{state}-cell{cell_index:03d}",
                            "species": species,
                            "ectag_count": int(value),
                            "image_qc_pass": True,
                            "batch_id": "synthetic",
                        }
                    )
        for species_index, species in enumerate(schemas.SPECIES):
            pooled = float(np.dot(fractions, base_mu[:, species_index] * (1.0 + 0.08 * (week - 1))))
            ddpcr_rows.append(
                {
                    "week": week,
                    "condition": condition,
                    "replicate": replicate,
                    "species": species,
                    "ddpcr_copy_number": max(0.01, rng.normal(pooled, 0.05 * pooled)),
                    "ddpcr_sd_or_ci": max(0.05, 0.08 * pooled),
                    "batch_id": "synthetic",
                }
            )
        cell_count_rows.append(
            {
                "week": week,
                "condition": condition,
                "replicate": replicate,
                "total_cell_count": int(8000 * (1.12 ** (week - 1))),
                "viability": 0.95,
                "passage_info": "synthetic",
                "batch_id": "synthetic",
            }
        )

    outputs = {
        "flow": out / "flow.csv",
        "qpcdr": out / "qpcdr.csv",
        "ectag": out / "ectag.csv",
        "ddpcr": out / "ddpcr.csv",
        "cell_count": out / "cell_count.csv",
    }
    pd.DataFrame(flow_rows).to_csv(outputs["flow"], index=False)
    pd.DataFrame(qpcdr_rows).to_csv(outputs["qpcdr"], index=False)
    pd.DataFrame(ectag_rows).to_csv(outputs["ectag"], index=False)
    pd.DataFrame(ddpcr_rows).to_csv(outputs["ddpcr"], index=False)
    pd.DataFrame(cell_count_rows).to_csv(outputs["cell_count"], index=False)
    return outputs


def _find_raw_table(base: Path, name: str) -> Path:
    direct = base / f"{name}.csv"
    if direct.exists():
        return direct
    folder = base / name
    if folder.exists():
        matches = sorted([path for path in folder.iterdir() if path.suffix.lower() in {".csv", ".tsv", ".parquet"}])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Raw input folder {folder} contains multiple candidate tables; pass explicit paths")
    return direct


def _clean_flow(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["fraction"] = clean["fraction"].astype(float)
    clean["pre_sort_count"] = clean["pre_sort_count"].astype(int)
    clean["post_sort_count"] = clean["post_sort_count"].astype(int)
    clean["sort_purity"] = pd.to_numeric(clean["sort_purity"], errors="coerce")
    return clean


def _clean_qpcdr(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["relative_copy_number"] = pd.to_numeric(clean["relative_copy_number"], errors="coerce")
    clean["raw_Ct_or_Cq"] = pd.to_numeric(clean["raw_Ct_or_Cq"], errors="coerce")
    has_value = clean["relative_copy_number"].notna() | clean["raw_Ct_or_Cq"].notna()
    if not bool(has_value.all()):
        raise ValueError("qPCDR rows need relative_copy_number or raw_Ct_or_Cq")
    return clean


def _clean_ectag(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean[clean["image_qc_pass"].astype(bool)].copy()
    clean["week"] = clean["week"].astype(int)
    clean["ectag_count"] = clean["ectag_count"].astype(int)
    return clean


def _clean_ddpcr(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["ddpcr_copy_number"] = clean["ddpcr_copy_number"].astype(float)
    clean["ddpcr_sd_or_ci"] = pd.to_numeric(clean["ddpcr_sd_or_ci"], errors="coerce")
    return clean


def _clean_cell_count(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["week"] = clean["week"].astype(int)
    clean["total_cell_count"] = clean["total_cell_count"].astype(float)
    clean["viability"] = pd.to_numeric(clean["viability"], errors="coerce")
    return clean
