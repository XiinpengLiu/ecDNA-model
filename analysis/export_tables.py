"""
R-friendly table exports for simulation results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as cfg
from core.simulation import SimulationResult


PLOT_TIMEPOINT_COUNT = 8
LEGACY_SIMULATION_FILES = (
    "summary.csv",
    "truth_snapshots.jsonl",
    "observations.jsonl",
    "snapshots.jsonl",
    "events.jsonl",
)


def write_simulation_tables(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write Parquet/CSV tables for downstream R plotting."""

    run_dir = Path(output_dir)
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_simulation_outputs(run_dir)

    outputs: dict[str, Path] = {}
    table_frames = {
        "time_summary": _time_summary_frame(result, condition=condition, seed=seed),
        "cell_snapshots": _cell_snapshot_frame(result, condition=condition, seed=seed),
        "events": _event_frame(result, condition=condition, seed=seed),
        "lineage_edges": _lineage_edge_frame(result, condition=condition, seed=seed),
        "observations": _observation_frame(result, condition=condition, seed=seed),
        "selected_plot_timepoints": _selected_plot_timepoints_frame(result, condition=condition, seed=seed),
    }

    for name, frame in table_frames.items():
        if name in {"cell_snapshots", "events", "lineage_edges"}:
            path = table_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
        else:
            path = table_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
        outputs[name] = path

    metadata_path = table_dir / "metadata.json"
    metadata_payload = dict(metadata or {})
    metadata_payload.update({"condition": condition, "seed": int(seed), "table_dir": str(table_dir)})
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    outputs["metadata"] = metadata_path

    manifest = _manifest_frame(table_frames, outputs)
    manifest_path = table_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    outputs["manifest"] = manifest_path
    return outputs


def _remove_legacy_simulation_outputs(run_dir: Path) -> None:
    for directory in (run_dir, run_dir / "simulation_data"):
        if not directory.exists():
            continue
        for file_name in LEGACY_SIMULATION_FILES:
            path = directory / file_name
            if path.is_file():
                try:
                    path.unlink()
                except PermissionError:
                    pass
        if directory.name == "simulation_data" and directory.exists() and not any(directory.iterdir()):
            try:
                directory.rmdir()
            except PermissionError:
                pass


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return token or "value"


def _sequence_value(values: Any, index: int, default: float | int | None = np.nan) -> Any:
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _run_columns(condition: str, seed: int) -> dict[str, Any]:
    return {"condition": condition, "seed": int(seed)}


def _time_summary_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, time in enumerate(result.times):
        row = {
            **_run_columns(condition, seed),
            "record_index": int(record_index),
            "time": float(time),
            "population_size": int(_sequence_value(result.population_sizes, record_index, 0)),
            "mean_stress_score": float(_sequence_value(result.mean_stress_scores, record_index)),
            "mean_survival_score": float(_sequence_value(result.mean_survival_scores, record_index)),
            "mean_division_hazard": float(_sequence_value(result.mean_division_hazard, record_index)),
            "mean_death_hazard": float(_sequence_value(result.mean_death_hazard, record_index)),
            "stop_time": result.stop_time,
            "stop_reason": result.stop_reason,
        }
        exposure = _sequence_value(result.exposures, record_index, {}) or {}
        row.update(
            {
                "D_C": float(exposure.get("D_C", np.nan)),
                "D_P": float(exposure.get("D_P", np.nan)),
                "a": float(exposure.get("a", np.nan)),
                "m": float(exposure.get("m", np.nan)),
            }
        )

        state_fractions = _sequence_value(result.soft_state_fractions, record_index, [])
        cycle_fractions = _sequence_value(result.cycle_fractions, record_index, [])
        bulk_means = _sequence_value(result.bulk_copy_means, record_index, [])
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            token = _safe_token(state_name)
            row[f"fraction_{token}"] = float(_sequence_value(state_fractions, state_index))
        for cycle_index, cycle_name in enumerate(cfg.CYCLE_NAMES):
            token = _safe_token(cycle_name)
            row[f"cycle_fraction_{token}"] = float(_sequence_value(cycle_fractions, cycle_index))
        for species_index, species_name in enumerate(cfg.SPECIES):
            row[f"mean_copy_{species_name}"] = float(_sequence_value(bulk_means, species_index))

        truth = _sequence_value(result.truth_snapshots, record_index, {}) or {}
        _add_state_species_summary(row, truth, "copy_means_by_gate", "state_mean_copy")
        _add_state_species_summary(row, truth, "copy_vars_by_gate", "state_var_copy")
        _add_state_species_summary(row, truth, "zero_fraction_by_gate", "state_zero_fraction")
        _add_state_species_summary(row, truth, "tail_fraction_by_gate", "state_tail_fraction")

        dominant_counts = truth.get("dominant_state_counts", [])
        dominant_fractions = truth.get("dominant_state_fractions", [])
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            token = _safe_token(state_name)
            row[f"dominant_count_{token}"] = int(_sequence_value(dominant_counts, state_index, 0))
            row[f"dominant_fraction_{token}"] = float(_sequence_value(dominant_fractions, state_index))
        rows.append(row)
    return pd.DataFrame(rows)


def _add_state_species_summary(row: dict[str, Any], snapshot: Mapping[str, Any], source_key: str, column_prefix: str) -> None:
    by_state = snapshot.get(source_key, {})
    for state_name in cfg.STATE_NAMES:
        state_values = by_state.get(state_name, []) if isinstance(by_state, Mapping) else []
        state_token = _safe_token(state_name)
        for species_index, species_name in enumerate(cfg.SPECIES):
            row[f"{column_prefix}_{state_token}_{species_name}"] = float(_sequence_value(state_values, species_index))


def _cell_snapshot_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, (time, snapshot) in enumerate(zip(result.times, result.cell_snapshots)):
        exposure = _sequence_value(result.exposures, record_index, {}) or {}
        for cell in snapshot:
            row = {
                **_run_columns(condition, seed),
                "record_index": int(record_index),
                "time": float(time),
                "D_C": float(exposure.get("D_C", np.nan)),
                "D_P": float(exposure.get("D_P", np.nan)),
                "a": float(exposure.get("a", np.nan)),
                "m": float(exposure.get("m", np.nan)),
                "division_hazard": float(cell.get("division_hazard", np.nan)),
                "death_hazard": float(cell.get("death_hazard", np.nan)),
                "last_D_C": float(cell.get("last_D_C", np.nan)),
                "last_D_P": float(cell.get("last_D_P", np.nan)),
            }
            row.update(_flatten_cell_state(cell))
            _add_transition_generator_columns(row, cell)
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_cell_state(cell: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    column_prefix = f"{prefix}_" if prefix else ""
    row: dict[str, Any] = {}
    for key in (
        "cell_id",
        "parent_id",
        "cycle_state",
        "cycle_index",
        "dominant_state",
        "dominant_state_index",
        "age",
        "stress_score",
        "survival_score",
        "last_update_time",
    ):
        row[f"{column_prefix}{key}"] = cell.get(key)

    copies = cell.get("copy_numbers", [])
    for species_index, species_name in enumerate(cfg.SPECIES):
        row[f"{column_prefix}copy_{species_name}"] = _sequence_value(copies, species_index)

    soft_state = cell.get("soft_state", [])
    for state_index, state_name in enumerate(cfg.STATE_NAMES):
        row[f"{column_prefix}soft_{_safe_token(state_name)}"] = _sequence_value(soft_state, state_index)

    latent_state = cell.get("latent_state", [])
    for latent_index in range(cfg.LATENT_DIM):
        row[f"{column_prefix}latent_{latent_index + 1}"] = _sequence_value(latent_state, latent_index)
    return row


def _add_transition_generator_columns(row: dict[str, Any], cell: Mapping[str, Any]) -> None:
    derived = cell.get("derived_report_only", {})
    matrix = derived.get("local_transition_generator", []) if isinstance(derived, Mapping) else []
    for source_index, source_state in enumerate(cfg.STATE_NAMES):
        source_token = _safe_token(source_state)
        source_values = _sequence_value(matrix, source_index, [])
        for target_index, target_state in enumerate(cfg.STATE_NAMES):
            target_token = _safe_token(target_state)
            row[f"transition_{source_token}_to_{target_token}"] = _sequence_value(source_values, target_index)


def _event_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_index, (time, event_type, cell_id, details) in enumerate(result.events):
        row: dict[str, Any] = {
            **_run_columns(condition, seed),
            "event_index": int(event_index),
            "time": float(time),
            "event_type": str(event_type),
            "cell_id": int(cell_id),
        }
        if isinstance(details, Mapping):
            if isinstance(details.get("state_pre"), Mapping):
                row.update(_flatten_cell_state(details["state_pre"], prefix="pre"))
            if isinstance(details.get("state_post"), Mapping):
                row.update(_flatten_cell_state(details["state_post"], prefix="post"))
            for daughter_key in ("daughter_one", "daughter_two"):
                if isinstance(details.get(daughter_key), Mapping):
                    row.update(_flatten_cell_state(details[daughter_key], prefix=daughter_key))
        rows.append(row)
    return pd.DataFrame(rows)


def _lineage_edge_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_index, (time, event_type, cell_id, details) in enumerate(result.events):
        if event_type != "division" or not isinstance(details, Mapping):
            continue
        parent_state = details.get("state_pre", {})
        for daughter_index, daughter_key in enumerate(("daughter_one", "daughter_two"), start=1):
            child_state = details.get(daughter_key, {})
            if not isinstance(child_state, Mapping):
                continue
            row: dict[str, Any] = {
                **_run_columns(condition, seed),
                "event_index": int(event_index),
                "division_time": float(time),
                "parent_id": int(cell_id),
                "child_id": child_state.get("cell_id"),
                "daughter_index": int(daughter_index),
            }
            if isinstance(parent_state, Mapping):
                row.update(_flatten_cell_state(parent_state, prefix="parent"))
            row.update(_flatten_cell_state(child_state, prefix="child"))
            rows.append(row)
    return pd.DataFrame(rows)


def _observation_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, (time, observation) in enumerate(zip(result.times, result.observations)):
        row: dict[str, Any] = {
            **_run_columns(condition, seed),
            "record_index": int(record_index),
            "time": float(time),
            "observed_count": observation.get("observed_count"),
        }
        for state_index, state_name in enumerate(cfg.STATE_NAMES):
            token = _safe_token(state_name)
            row[f"latent_gate_count_{token}"] = _sequence_value(observation.get("latent_gate_counts", []), state_index, 0)
            row[f"latent_gate_fraction_{token}"] = _sequence_value(observation.get("latent_gate_fractions", []), state_index)
            row[f"flow_count_{token}"] = _sequence_value(observation.get("flow_counts", []), state_index, 0)
            row[f"flow_fraction_{token}"] = _sequence_value(observation.get("flow_fractions", []), state_index)
            row[f"sorted_state_count_{token}"] = (observation.get("sorted_state_counts", {}) or {}).get(state_name, 0)
        for species_index, species_name in enumerate(cfg.SPECIES):
            row[f"pooled_qpcdr_mean_{species_name}"] = _sequence_value(observation.get("pooled_qpcdr_means", []), species_index)
            row[f"pooled_ecTAG_mean_{species_name}"] = _sequence_value(observation.get("pooled_ecTAG_means", []), species_index)
        _add_state_species_summary(row, observation, "sorted_bulk_copy_means", "sorted_mean_copy")
        _add_state_species_summary(row, observation.get("sorted_qpcdr", {}) or {}, "means", "sorted_qpcdr_mean")
        _add_state_species_summary(row, observation.get("sorted_ecTAG", {}) or {}, "means", "sorted_ecTAG_mean")
        rows.append(row)
    return pd.DataFrame(rows)


def _selected_plot_timepoints_frame(result: SimulationResult, *, condition: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = set(_terminal_aligned_indices(len(result.times), PLOT_TIMEPOINT_COUNT))
    for record_index, time in enumerate(result.times):
        if record_index in selected:
            rows.append(
                {
                    **_run_columns(condition, seed),
                    "record_index": int(record_index),
                    "time": float(time),
                    "plot_timepoint_index": len(rows),
                }
            )
    return pd.DataFrame(rows)


def _terminal_aligned_indices(length: int, target_count: int) -> list[int]:
    if length <= 0:
        return []
    if length <= target_count:
        return list(range(length))
    raw_positions = np.rint(np.linspace(0, length - 1, num=target_count)).astype(int)
    return sorted(set(int(position) for position in raw_positions.tolist()))


def _manifest_frame(table_frames: Mapping[str, pd.DataFrame], outputs: Mapping[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in table_frames.items():
        path = outputs[name]
        rows.append(
            {
                "table": name,
                "file": path.name,
                "format": path.suffix.lstrip("."),
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            }
        )
    return pd.DataFrame(rows)
