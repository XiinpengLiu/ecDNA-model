"""
Simulation output data-package exports.

The public entry point, ``write_simulation_tables``, writes the complete
``markdown/export.md`` package without changing core simulator objects.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as cfg
from core.simulation import SimulationResult


DEFAULT_ENSEMBLE_ID = "ENS_000001"
MODEL_VARIANT = "full"
INITIAL_CONDITION_ID = "parental"
REPLICATE_ID = "REP001"
PARAMETER_SET_ID = "PARAM_000001"
SIMULATOR_VERSION = "v4"
OUTPUT_SCHEMA_VERSION = "t_internal_v1"
HIGH_COPY_THRESHOLD = 10

LEGACY_SIMULATION_FILES = (
    "summary.csv",
    "truth_snapshots.jsonl",
    "observations.jsonl",
    "snapshots.jsonl",
    "events.jsonl",
)

FORBIDDEN_COLUMN_TOKENS = {"week", "day", "tau", "time_day", "simulation_time"}

COPY_BINS: tuple[tuple[str, float, float | None, str], ...] = (
    ("bin_0", 0.0, 0.0, "0"),
    ("bin_1_15", 1.0, 15.0, "1-15"),
    ("bin_16_30", 16.0, 30.0, "16-30"),
    ("bin_31_60", 31.0, 60.0, "31-60"),
    ("bin_61_120", 61.0, 120.0, "61-120"),
    ("bin_gt120", 121.0, None, ">120"),
)

OBSERVABLES_LONG_COLUMNS = (
    "sim_id",
    "ensemble_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "assay",
    "species",
    "state_id",
    "state_compartment",
    "value_true",
    "value_noisy",
    "n_cells_alive",
    "n_effective",
    "unit",
    "notes",
)

COPY_VECTOR_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "myc_copy_mean",
    "cdk4_copy_mean",
    "pdgfra_copy_mean",
    "log2_myc_copy",
    "log2_cdk4_copy",
    "log2_pdgfra_copy",
    "total_copy_mean",
    "cell_count",
    "olig2_high_fraction",
    "ac_like_fraction",
    "mes_like_fraction",
)

CELL_REGISTRY_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "cell_id",
    "cell_uid",
    "founder_id",
    "founder_uid",
    "parent_id",
    "parent_uid",
    "birth_event_id",
    "death_event_id",
    "birth_t",
    "death_t",
    "final_t",
    "final_status",
    "lineage_depth",
    "cell_weight",
    "is_founder",
    "k_myc_birth",
    "k_cdk4_birth",
    "k_pdgfra_birth",
    "k_myc_final",
    "k_cdk4_final",
    "k_pdgfra_final",
    "total_burden_birth",
    "total_burden_final",
    "hard_state_birth",
    "hard_state_final",
    "coarse_state_birth",
    "coarse_state_final",
    "x_npc_birth",
    "x_opc_birth",
    "x_ac_birth",
    "x_mes_birth",
    "x_npc_final",
    "x_opc_final",
    "x_ac_final",
    "x_mes_final",
    "r_stress_birth",
    "r_stress_final",
    "v_survival_birth",
    "v_survival_final",
    "n_divisions_as_parent",
    "n_children_total",
)

CELL_SNAPSHOT_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "cell_id",
    "cell_uid",
    "founder_id",
    "founder_uid",
    "parent_id",
    "parent_uid",
    "lineage_depth",
    "cell_weight",
    "birth_t",
    "age_t",
    "alive",
    "hard_state",
    "coarse_state",
    "u1",
    "u2",
    "u3",
    "x_npc",
    "x_opc",
    "x_ac",
    "x_mes",
    "olig2_high_score",
    "k_myc",
    "k_cdk4",
    "k_pdgfra",
    "total_burden",
    "log1p_myc",
    "log1p_cdk4",
    "log1p_pdgfra",
    "r_stress",
    "v_survival",
    "cell_cycle_state",
    "division_hazard",
    "death_hazard",
    "copy_selection_score",
    "state_growth_score",
    "drug_effect_score",
)

CELL_TERMINAL_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "cell_id",
    "cell_uid",
    "founder_id",
    "terminal_t",
    "terminal_reason",
    "terminal_event_id",
    "k_myc_terminal",
    "k_cdk4_terminal",
    "k_pdgfra_terminal",
    "hard_state_terminal",
    "coarse_state_terminal",
    "x_npc_terminal",
    "x_opc_terminal",
    "x_ac_terminal",
    "x_mes_terminal",
    "lineage_depth",
    "lifetime_t",
)

EVENT_LOG_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "event_id",
    "event_order",
    "t",
    "event_type",
    "cell_id",
    "cell_uid",
    "founder_id",
    "founder_uid",
    "parent_id",
    "species",
    "k_myc_before",
    "k_cdk4_before",
    "k_pdgfra_before",
    "k_myc_after",
    "k_cdk4_after",
    "k_pdgfra_after",
    "hard_state_before",
    "hard_state_after",
    "coarse_state_before",
    "coarse_state_after",
    "x_npc_before",
    "x_opc_before",
    "x_ac_before",
    "x_mes_before",
    "x_npc_after",
    "x_opc_after",
    "x_ac_after",
    "x_mes_after",
    "r_stress_before",
    "r_stress_after",
    "v_survival_before",
    "v_survival_after",
    "daughter1_id",
    "daughter2_id",
    "daughter1_uid",
    "daughter2_uid",
    "event_rate",
    "accepted_by_thinning",
    "notes",
)

LINEAGE_EDGE_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "division_event_id",
    "t_birth",
    "parent_id",
    "parent_uid",
    "child_id",
    "child_uid",
    "child_order",
    "founder_id",
    "founder_uid",
    "parent_lineage_depth",
    "child_lineage_depth",
    "parent_k_myc_before_division",
    "parent_k_cdk4_before_division",
    "parent_k_pdgfra_before_division",
    "child_k_myc_birth",
    "child_k_cdk4_birth",
    "child_k_pdgfra_birth",
    "parent_state_before_division",
    "child_state_birth",
    "parent_total_burden",
    "child_total_burden",
)

DIVISION_INHERITANCE_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "division_event_id",
    "t",
    "parent_id",
    "daughter1_id",
    "daughter2_id",
    "founder_id",
    "parent_k_myc",
    "parent_k_cdk4",
    "parent_k_pdgfra",
    "amplification_myc",
    "amplification_cdk4",
    "amplification_pdgfra",
    "segregation_pool_myc",
    "segregation_pool_cdk4",
    "segregation_pool_pdgfra",
    "daughter1_k_myc",
    "daughter1_k_cdk4",
    "daughter1_k_pdgfra",
    "daughter2_k_myc",
    "daughter2_k_cdk4",
    "daughter2_k_pdgfra",
    "post_loss_d1_myc",
    "post_loss_d1_cdk4",
    "post_loss_d1_pdgfra",
    "post_loss_d2_myc",
    "post_loss_d2_cdk4",
    "post_loss_d2_pdgfra",
    "imbalance_myc",
    "imbalance_cdk4",
    "imbalance_pdgfra",
    "parent_state",
    "daughter1_state",
    "daughter2_state",
)

VIRTUAL_ASSAY_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "t",
    "assay",
    "virtual_sample_id",
    "cell_id",
    "cell_uid",
    "species",
    "state_compartment",
    "true_value",
    "observed_value",
    "measurement_noise_model",
    "n_sampled_cells",
)

POPULATION_SUMMARY_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "n_alive_cells",
    "n_cells_ever_born",
    "n_dead_cumulative",
    "n_divisions_cumulative",
    "mean_myc",
    "mean_cdk4",
    "mean_pdgfra",
    "mean_total_burden",
    "median_total_burden",
    "olig2_high_fraction",
    "ac_like_fraction",
    "mes_like_fraction",
    "npc_fraction",
    "opc_fraction",
    "ac_fraction",
    "mes_fraction",
    "mean_r_stress",
    "mean_v_survival",
)

STATE_COPY_SUMMARY_COLUMNS = (
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "state_compartment",
    "state_level",
    "species",
    "weighted_cell_count",
    "state_fraction",
    "mean_copy",
    "median_copy",
    "q05_copy",
    "q25_copy",
    "q75_copy",
    "q95_copy",
    "zero_fraction",
    "copy_bin_0_fraction",
    "copy_bin_1_15_fraction",
    "copy_bin_16_30_fraction",
    "copy_bin_31_60_fraction",
    "copy_bin_61_120_fraction",
    "copy_bin_gt120_fraction",
)

FOUNDER_T_SUMMARY_COLUMNS = (
    "sim_id",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "t",
    "t_index",
    "founder_id",
    "founder_uid",
    "descendant_count_alive",
    "weighted_descendant_count",
    "myc_copy_sum",
    "cdk4_copy_sum",
    "pdgfra_copy_sum",
    "total_burden_sum",
    "myc_copy_mean",
    "cdk4_copy_mean",
    "pdgfra_copy_mean",
    "olig2_high_count",
    "ac_like_count",
    "mes_like_count",
    "npc_count",
    "opc_count",
    "ac_count",
    "mes_count",
    "division_count_cumulative",
    "death_count_cumulative",
    "gain_myc_count_cumulative",
    "loss_myc_count_cumulative",
    "gain_cdk4_count_cumulative",
    "loss_cdk4_count_cumulative",
    "gain_pdgfra_count_cumulative",
    "loss_pdgfra_count_cumulative",
)

COPY_DISTRIBUTION_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "t",
    "species",
    "copy_bin_id",
    "copy_bin_label",
    "n_cells",
    "fraction",
    "mean_copy_in_bin",
    "state_compartment",
)

EVENT_SUMMARY_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "t_start",
    "t_end",
    "event_type",
    "species",
    "state_compartment",
    "event_count",
    "event_rate_per_cell",
    "event_rate_per_t",
    "founder_group",
)

LINEAGE_FAMILY_SUMMARY_COLUMNS = (
    "sim_id",
    "condition_id",
    "replicate_id",
    "founder_id",
    "founder_uid",
    "founder_state",
    "founder_coarse_state",
    "founder_k_myc",
    "founder_k_cdk4",
    "founder_k_pdgfra",
    "final_descendant_count",
    "final_myc_copy_sum",
    "final_cdk4_copy_sum",
    "final_pdgfra_copy_sum",
    "final_total_burden_sum",
    "max_descendant_count",
    "max_cdk4_copy_sum",
    "survival_status",
    "dominant_final_state",
    "n_divisions_total",
    "n_deaths_total",
)

PARAMETER_TABLE_COLUMNS = (
    "sim_id",
    "parameter_set_id",
    "model_variant",
    "condition_id",
    "parameter_block",
    "parameter_name",
    "species",
    "state_id",
    "from_state",
    "to_state",
    "dose_value",
    "value",
    "unit",
    "fixed_or_free",
    "description",
)

PARAMETER_BLOCK_COLUMNS = ("parameter_block", "enabled", "disabled_in_variant", "description")

RUN_INDEX_COLUMNS = (
    "ensemble_id",
    "sim_id",
    "model_variant",
    "condition_id",
    "initial_condition_id",
    "replicate_id",
    "seed",
    "parameter_set_id",
    "t_min",
    "t_max",
    "snapshot_grid_id",
    "dense_output",
    "output_path",
    "status",
)

REQUIRED_RUN_FILES = (
    "manifest.json",
    "parameters/parameter_table.parquet",
    "parameters/parameter_blocks.parquet",
    "root/observables_long.parquet",
    "root/copy_vector.parquet",
    "root/cell_registry.parquet",
    "root/cell_snapshot",
    "root/cell_terminal_state.parquet",
    "root/event_log.parquet",
    "root/lineage_edges.parquet",
    "root/division_inheritance.parquet",
    "root/virtual_assay_draws.parquet",
    "cache/population_summary.parquet",
    "cache/state_copy_summary.parquet",
    "cache/founder_t_summary.parquet",
    "cache/copy_distribution_summary.parquet",
    "cache/event_summary.parquet",
    "cache/lineage_family_summary.parquet",
    "qc/output_integrity_report.json",
    "qc/id_consistency_report.parquet",
)

REQUIRED_ENSEMBLE_FILES = (
    "ensemble_manifest.json",
    "run_index.parquet",
    "metadata/conditions.parquet",
    "metadata/model_variants.parquet",
    "metadata/initial_conditions.parquet",
    "metadata/t_grid.parquet",
    "metadata/species.parquet",
    "metadata/state_definitions.parquet",
    "metadata/assay_definitions.parquet",
    "metadata/copy_bins.parquet",
    "metadata/event_type_definitions.parquet",
)


@dataclass(frozen=True)
class ExportContext:
    ensemble_id: str
    sim_id: str
    condition: str
    condition_id: str
    seed: int
    replicate_id: str
    model_variant: str
    initial_condition_id: str
    parameter_set_id: str
    t_values: tuple[float, ...]
    t_index_by_key: Mapping[float, int]
    t_min: float
    t_max: float
    snapshot_grid_id: str
    dense_output: bool


def write_simulation_tables(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a complete ensemble/run package for one simulation result."""

    metadata_payload = dict(metadata or {})
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_simulation_outputs(base_dir)

    ctx = _export_context(result, condition=condition, seed=seed, metadata=metadata_payload)
    ensemble_dir = _ensemble_dir(base_dir, ctx.ensemble_id)
    run_dir = ensemble_dir / "runs" / f"sim_id={ctx.sim_id}"

    _replace_output_path(run_dir)
    for directory in (
        ensemble_dir / "metadata",
        run_dir / "parameters",
        run_dir / "root",
        run_dir / "cache",
        run_dir / "qc",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    tables = _build_run_tables(result, ctx, metadata_payload)
    outputs: dict[str, Path] = {}

    _write_ensemble_files(ensemble_dir, ctx)
    outputs.update(_ensemble_output_paths(ensemble_dir))

    _write_run_files(run_dir, ctx, result, metadata_payload, tables)
    outputs.update(_run_output_paths(run_dir))

    _update_run_index(ensemble_dir, run_dir, ctx)
    outputs["run_index"] = ensemble_dir / "run_index.parquet"

    return outputs


def _export_context(
    result: SimulationResult,
    *,
    condition: str,
    seed: int,
    metadata: Mapping[str, Any],
) -> ExportContext:
    ensemble_id = str(metadata.get("ensemble_id", DEFAULT_ENSEMBLE_ID))
    condition_id = _condition_id(condition)
    replicate_id = str(metadata.get("replicate_id", REPLICATE_ID))
    model_variant = str(metadata.get("model_variant", MODEL_VARIANT))
    initial_condition_id = str(metadata.get("initial_condition_id", INITIAL_CONDITION_ID))
    parameter_set_id = str(metadata.get("parameter_set_id", PARAMETER_SET_ID))
    sim_id = str(
        metadata.get(
            "sim_id",
            f"SIM_{_safe_token(model_variant).upper()}_{_safe_token(condition).upper()}_{replicate_id}",
        )
    )
    times = tuple(float(t) for t in result.times)
    if times:
        t_values = tuple(sorted(dict.fromkeys(times)))
    else:
        simulation = metadata.get("simulation", {})
        record_times = simulation.get("record_times", []) if isinstance(simulation, Mapping) else []
        t_values = tuple(float(t) for t in record_times)
    t_min = 0.0 if not t_values else min(0.0, float(t_values[0]))
    t_max = _metadata_t_max(metadata, result)
    if t_values and abs(float(t_values[-1]) - t_max) > 1e-8:
        t_max = float(t_values[-1])
    t_index_by_key = {_t_key(t): index for index, t in enumerate(t_values)}
    intervals = np.diff(np.asarray(t_values, dtype=float)) if len(t_values) > 1 else np.array([], dtype=float)
    dense_output = bool(intervals.size and float(np.nanmedian(intervals)) < 1.0)
    snapshot_grid_id = _snapshot_grid_id(t_values)
    return ExportContext(
        ensemble_id=ensemble_id,
        sim_id=sim_id,
        condition=condition,
        condition_id=condition_id,
        seed=int(seed),
        replicate_id=replicate_id,
        model_variant=model_variant,
        initial_condition_id=initial_condition_id,
        parameter_set_id=parameter_set_id,
        t_values=t_values,
        t_index_by_key=t_index_by_key,
        t_min=float(t_min),
        t_max=float(t_max),
        snapshot_grid_id=snapshot_grid_id,
        dense_output=dense_output,
    )


def _metadata_t_max(metadata: Mapping[str, Any], result: SimulationResult) -> float:
    simulation = metadata.get("simulation", {})
    if isinstance(simulation, Mapping) and simulation.get("t_max") is not None:
        return float(simulation["t_max"])
    if result.stop_time is not None:
        return float(result.stop_time)
    if result.times:
        return float(result.times[-1])
    return 12.0


def _snapshot_grid_id(t_values: Sequence[float]) -> str:
    if tuple(t_values) == tuple(float(t) for t in range(13)):
        return "main_0_12_by1"
    if not t_values:
        return "empty_grid"
    start = _format_t(t_values[0])
    end = _format_t(t_values[-1])
    if len(t_values) > 1:
        step = _format_t(float(np.nanmedian(np.diff(np.asarray(t_values, dtype=float)))))
        return f"custom_{start}_{end}_by{step}"
    return f"custom_{start}"


def _ensemble_dir(base_dir: Path, ensemble_id: str) -> Path:
    if base_dir.name == f"ensemble_id={ensemble_id}":
        return base_dir
    return base_dir / f"ensemble_id={ensemble_id}"


def _write_ensemble_files(ensemble_dir: Path, ctx: ExportContext) -> None:
    manifest = {
        "ensemble_id": ctx.ensemble_id,
        "project": "ecDNA_state_simulation",
        "time_variable": "t",
        "t_min": ctx.t_min,
        "t_max": ctx.t_max,
        "uses_real_time": False,
        "uses_week_labels": False,
        "records_all_cells_ever_born": True,
        "records_all_alive_cells_at_snapshots": True,
        "records_all_events": True,
        "records_lineage_edges": True,
        "simulator_version": SIMULATOR_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "notes": "Internal simulation time only. Scores are calculated downstream in R.",
    }
    (ensemble_dir / "ensemble_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2),
        encoding="utf-8",
    )
    metadata_frames = {
        "conditions": _conditions_frame(ctx.condition),
        "model_variants": _model_variants_frame(),
        "initial_conditions": _initial_conditions_frame(),
        "t_grid": _t_grid_frame(ctx),
        "species": _species_frame(),
        "state_definitions": _state_definitions_frame(),
        "assay_definitions": _assay_definitions_frame(),
        "copy_bins": _copy_bins_frame(),
        "event_type_definitions": _event_type_definitions_frame(),
    }
    metadata_dir = ensemble_dir / "metadata"
    for name, frame in metadata_frames.items():
        frame.to_parquet(metadata_dir / f"{name}.parquet", index=False)


def _write_run_files(
    run_dir: Path,
    ctx: ExportContext,
    result: SimulationResult,
    metadata: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> None:
    parameter_dir = run_dir / "parameters"
    root_dir = run_dir / "root"
    cache_dir = run_dir / "cache"
    qc_dir = run_dir / "qc"

    tables["parameter_table"].to_parquet(parameter_dir / "parameter_table.parquet", index=False)
    tables["parameter_blocks"].to_parquet(parameter_dir / "parameter_blocks.parquet", index=False)
    for name in (
        "observables_long",
        "copy_vector",
        "cell_registry",
        "cell_terminal_state",
        "event_log",
        "lineage_edges",
        "division_inheritance",
        "virtual_assay_draws",
    ):
        tables[name].to_parquet(root_dir / f"{name}.parquet", index=False)
    _write_partitioned_cell_snapshot(tables["cell_snapshot"], root_dir / "cell_snapshot")
    for name in (
        "population_summary",
        "state_copy_summary",
        "founder_t_summary",
        "copy_distribution_summary",
        "event_summary",
        "lineage_family_summary",
    ):
        tables[name].to_parquet(cache_dir / f"{name}.parquet", index=False)

    # Placeholder QC files make the existence audit cover the final paths.
    pd.DataFrame(columns=["check_name", "passed", "n_violations", "notes"]).to_parquet(
        qc_dir / "id_consistency_report.parquet",
        index=False,
    )
    (qc_dir / "output_integrity_report.json").write_text("{}", encoding="utf-8")

    manifest = _run_manifest(ctx, result, metadata, tables)
    (run_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    id_report, integrity = _qc_reports(run_dir, tables)
    id_report.to_parquet(qc_dir / "id_consistency_report.parquet", index=False)
    (qc_dir / "output_integrity_report.json").write_text(
        json.dumps(_json_ready(integrity), indent=2),
        encoding="utf-8",
    )


def _build_run_tables(
    result: SimulationResult,
    ctx: ExportContext,
    metadata: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    event_records = _event_records(result)
    birth_records = _birth_records_from_result(result, ctx, event_records)
    parent_map = {cell_id: _optional_int(record.get("parent_id")) for cell_id, record in birth_records.items()}
    founder_map = _founder_map(parent_map)
    death_map = _death_map(event_records)
    division_terminal_map = _division_terminal_map(event_records)
    final_records = _final_records_from_result(result, birth_records, event_records)

    cell_snapshot = _cell_snapshot_frame(result, ctx, birth_records, parent_map, founder_map, death_map, division_terminal_map)
    event_log = _event_log_frame(ctx, event_records, parent_map, founder_map)
    lineage_edges = _lineage_edges_frame(ctx, event_records, parent_map, founder_map)
    cell_registry = _cell_registry_frame(ctx, birth_records, final_records, parent_map, founder_map, death_map, division_terminal_map, lineage_edges)
    cell_terminal_state = _cell_terminal_state_frame(ctx, cell_registry, death_map, division_terminal_map)
    division_inheritance = _division_inheritance_frame(ctx, event_records, founder_map)
    observables = _observables_long_frame(ctx, cell_snapshot)
    copy_vector = _copy_vector_frame(observables)
    parameter_table = _parameter_table_frame(ctx, metadata)
    population_summary = _population_summary_frame(ctx, cell_snapshot, cell_registry, event_log)
    state_copy_summary = _state_copy_summary_frame(ctx, cell_snapshot)
    founder_t_summary = _founder_t_summary_frame(ctx, cell_snapshot, event_log)
    copy_distribution_summary = _copy_distribution_summary_frame(ctx, cell_snapshot)
    event_summary = _event_summary_frame(ctx, event_log, population_summary)
    lineage_family_summary = _lineage_family_summary_frame(ctx, cell_registry, founder_t_summary, event_log)

    return {
        "parameter_table": _enforce_schema(parameter_table, PARAMETER_TABLE_COLUMNS),
        "parameter_blocks": _enforce_schema(_parameter_blocks_frame(), PARAMETER_BLOCK_COLUMNS),
        "observables_long": _enforce_schema(observables, OBSERVABLES_LONG_COLUMNS),
        "copy_vector": _enforce_schema(copy_vector, COPY_VECTOR_COLUMNS),
        "cell_registry": _enforce_schema(cell_registry, CELL_REGISTRY_COLUMNS),
        "cell_snapshot": _enforce_schema(cell_snapshot, CELL_SNAPSHOT_COLUMNS),
        "cell_terminal_state": _enforce_schema(cell_terminal_state, CELL_TERMINAL_COLUMNS),
        "event_log": _enforce_schema(event_log, EVENT_LOG_COLUMNS),
        "lineage_edges": _enforce_schema(lineage_edges, LINEAGE_EDGE_COLUMNS),
        "division_inheritance": _enforce_schema(division_inheritance, DIVISION_INHERITANCE_COLUMNS),
        "virtual_assay_draws": _enforce_schema(pd.DataFrame(), VIRTUAL_ASSAY_COLUMNS),
        "population_summary": _enforce_schema(population_summary, POPULATION_SUMMARY_COLUMNS),
        "state_copy_summary": _enforce_schema(state_copy_summary, STATE_COPY_SUMMARY_COLUMNS),
        "founder_t_summary": _enforce_schema(founder_t_summary, FOUNDER_T_SUMMARY_COLUMNS),
        "copy_distribution_summary": _enforce_schema(copy_distribution_summary, COPY_DISTRIBUTION_COLUMNS),
        "event_summary": _enforce_schema(event_summary, EVENT_SUMMARY_COLUMNS),
        "lineage_family_summary": _enforce_schema(lineage_family_summary, LINEAGE_FAMILY_SUMMARY_COLUMNS),
    }


def _event_records(result: SimulationResult) -> list[dict[str, Any]]:
    rows = []
    for order, (t, event_type, cell_id, details) in enumerate(result.events):
        rows.append(
            {
                "event_id": f"EVT_{order:08d}",
                "event_order": int(order),
                "t": float(t),
                "raw_event_type": str(event_type),
                "cell_id": int(cell_id),
                "details": details if isinstance(details, Mapping) else {},
            }
        )
    return rows


def _birth_records_from_result(
    result: SimulationResult,
    ctx: ExportContext,
    event_records: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for t, snapshot in zip(result.times, result.cell_snapshots):
        for cell in snapshot:
            cell_id = _optional_int(cell.get("cell_id"))
            if cell_id is None:
                continue
            parent_id = _optional_int(cell.get("parent_id"))
            inferred_birth_t = max(0.0, float(t) - float(cell.get("age", 0.0)))
            if parent_id is None and _t_key(float(t)) == _t_key(ctx.t_min):
                inferred_birth_t = ctx.t_min
            records.setdefault(
                cell_id,
                {
                    "cell_id": cell_id,
                    "parent_id": parent_id,
                    "birth_event_id": f"INIT_{cell_id:08d}" if parent_id is None else None,
                    "birth_t": float(inferred_birth_t),
                    "birth_state": dict(cell),
                },
            )
    for event in event_records:
        if event["raw_event_type"] != "division":
            continue
        details = event["details"]
        parent_id = int(event["cell_id"])
        for daughter_key in ("daughter_one", "daughter_two"):
            daughter = details.get(daughter_key, {}) if isinstance(details, Mapping) else {}
            if not isinstance(daughter, Mapping):
                continue
            child_id = _optional_int(daughter.get("cell_id"))
            if child_id is None:
                continue
            records[child_id] = {
                "cell_id": child_id,
                "parent_id": parent_id,
                "birth_event_id": event["event_id"],
                "birth_t": float(event["t"]),
                "birth_state": dict(daughter),
            }
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        if isinstance(pre, Mapping) and parent_id not in records:
            records[parent_id] = {
                "cell_id": parent_id,
                "parent_id": _optional_int(pre.get("parent_id")),
                "birth_event_id": None,
                "birth_t": max(0.0, float(event["t"]) - float(pre.get("age", 0.0))),
                "birth_state": dict(pre),
            }
    for event in event_records:
        details = event["details"]
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        cell_id = int(event["cell_id"])
        if isinstance(pre, Mapping) and cell_id not in records:
            records[cell_id] = {
                "cell_id": cell_id,
                "parent_id": _optional_int(pre.get("parent_id")),
                "birth_event_id": None,
                "birth_t": max(0.0, float(event["t"]) - float(pre.get("age", 0.0))),
                "birth_state": dict(pre),
            }
    return records


def _death_map(event_records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    deaths = {}
    for event in event_records:
        if event["raw_event_type"] == "death":
            deaths[int(event["cell_id"])] = {"death_t": float(event["t"]), "death_event_id": event["event_id"]}
    return deaths


def _division_terminal_map(event_records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    terminals = {}
    for event in event_records:
        if event["raw_event_type"] == "division":
            terminals[int(event["cell_id"])] = {"division_t": float(event["t"]), "division_event_id": event["event_id"]}
    return terminals


def _final_records_from_result(
    result: SimulationResult,
    birth_records: Mapping[int, Mapping[str, Any]],
    event_records: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    final_records = {cell_id: dict(record.get("birth_state", {})) for cell_id, record in birth_records.items()}
    final_t = {cell_id: float(record.get("birth_t", 0.0)) for cell_id, record in birth_records.items()}
    for t, snapshot in zip(result.times, result.cell_snapshots):
        for cell in snapshot:
            cell_id = _optional_int(cell.get("cell_id"))
            if cell_id is not None and float(t) >= final_t.get(cell_id, -np.inf):
                final_records[cell_id] = dict(cell)
                final_t[cell_id] = float(t)
    for event in event_records:
        details = event["details"]
        event_t = float(event["t"])
        if event["raw_event_type"] == "death":
            pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
            if isinstance(pre, Mapping):
                final_records[int(event["cell_id"])] = dict(pre)
                final_t[int(event["cell_id"])] = event_t
        elif event["raw_event_type"] == "division":
            pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
            if isinstance(pre, Mapping):
                final_records[int(event["cell_id"])] = dict(pre)
                final_t[int(event["cell_id"])] = event_t
        else:
            post = details.get("state_post", {}) if isinstance(details, Mapping) else {}
            if isinstance(post, Mapping):
                final_records[int(event["cell_id"])] = dict(post)
                final_t[int(event["cell_id"])] = event_t
    return final_records


def _cell_snapshot_frame(
    result: SimulationResult,
    ctx: ExportContext,
    birth_records: Mapping[int, Mapping[str, Any]],
    parent_map: Mapping[int, int | None],
    founder_map: Mapping[int, int],
    death_map: Mapping[int, Mapping[str, Any]],
    division_terminal_map: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for t, snapshot in zip(result.times, result.cell_snapshots):
        t = float(t)
        t_index = _t_index(ctx, t)
        for cell in snapshot:
            cell_id = _optional_int(cell.get("cell_id"))
            if cell_id is None:
                continue
            parent_id = parent_map.get(cell_id, _optional_int(cell.get("parent_id")))
            founder_id = founder_map.get(cell_id, cell_id)
            birth = birth_records.get(cell_id, {})
            birth_t = float(birth.get("birth_t", max(0.0, t - float(cell.get("age", 0.0)))))
            state = _cell_state_columns(cell)
            rows.append(
                {
                    **_run_keys(ctx),
                    "t": t,
                    "t_index": t_index,
                    "cell_id": cell_id,
                    "cell_uid": _cell_uid(ctx, cell_id),
                    "founder_id": founder_id,
                    "founder_uid": _cell_uid(ctx, founder_id),
                    "parent_id": parent_id,
                    "parent_uid": _cell_uid(ctx, parent_id),
                    "lineage_depth": _lineage_depth(cell_id, parent_map),
                    "cell_weight": 1.0,
                    "birth_t": birth_t,
                    "age_t": max(0.0, t - birth_t),
                    "alive": True,
                    **state,
                    "copy_selection_score": np.nan,
                    "state_growth_score": np.nan,
                    "drug_effect_score": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Drop any impossible stale rows if a stopped run contains a terminal record at the same t.
        frame = frame[
            frame.apply(
                lambda row: _snapshot_alive(row, death_map, division_terminal_map),
                axis=1,
            )
        ].reset_index(drop=True)
    return frame


def _snapshot_alive(
    row: pd.Series,
    death_map: Mapping[int, Mapping[str, Any]],
    division_terminal_map: Mapping[int, Mapping[str, Any]],
) -> bool:
    cell_id = _optional_int(row.get("cell_id"))
    if cell_id is None:
        return False
    t = float(row.get("t", np.nan))
    death_t = death_map.get(cell_id, {}).get("death_t", np.nan)
    division_t = division_terminal_map.get(cell_id, {}).get("division_t", np.nan)
    if pd.notna(death_t) and t >= float(death_t) - 1e-10:
        return False
    if pd.notna(division_t) and t >= float(division_t) - 1e-10:
        return False
    return True


def _event_log_frame(
    ctx: ExportContext,
    event_records: Sequence[Mapping[str, Any]],
    parent_map: Mapping[int, int | None],
    founder_map: Mapping[int, int],
) -> pd.DataFrame:
    rows = []
    for event in event_records:
        details = event["details"]
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        post = details.get("state_post", {}) if isinstance(details, Mapping) else {}
        if event["raw_event_type"] == "division":
            post = {}
        cell_id = int(event["cell_id"])
        daughter_one = details.get("daughter_one", {}) if isinstance(details, Mapping) else {}
        daughter_two = details.get("daughter_two", {}) if isinstance(details, Mapping) else {}
        event_type, species = _normalise_event_type(str(event["raw_event_type"]))
        founder_id = founder_map.get(cell_id, cell_id)
        row = {
            **_run_keys(ctx),
            "event_id": event["event_id"],
            "event_order": int(event["event_order"]),
            "t": float(event["t"]),
            "event_type": event_type,
            "cell_id": cell_id,
            "cell_uid": _cell_uid(ctx, cell_id),
            "founder_id": founder_id,
            "founder_uid": _cell_uid(ctx, founder_id),
            "parent_id": parent_map.get(cell_id, _optional_int(pre.get("parent_id")) if isinstance(pre, Mapping) else None),
            "species": species,
            "daughter1_id": _optional_int(daughter_one.get("cell_id")) if isinstance(daughter_one, Mapping) else None,
            "daughter2_id": _optional_int(daughter_two.get("cell_id")) if isinstance(daughter_two, Mapping) else None,
            "daughter1_uid": _cell_uid(ctx, _optional_int(daughter_one.get("cell_id"))) if isinstance(daughter_one, Mapping) else None,
            "daughter2_uid": _cell_uid(ctx, _optional_int(daughter_two.get("cell_id"))) if isinstance(daughter_two, Mapping) else None,
            "event_rate": np.nan,
            "accepted_by_thinning": True,
            "notes": None,
        }
        row.update(_before_after_columns(pre, post))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=PARAMETER_TABLE_COLUMNS)
    for column, value in (
        ("sim_id", ctx.sim_id),
        ("parameter_set_id", ctx.parameter_set_id),
        ("model_variant", ctx.model_variant),
        ("condition_id", ctx.condition_id),
    ):
        frame[column] = frame[column].fillna(value) if column in frame else value
    dose_value = float(cfg.T87_CONDITION_TREATMENTS.get(ctx.condition, ("", 0.0))[1])
    frame["dose_value"] = frame["dose_value"].fillna(dose_value) if "dose_value" in frame else dose_value
    return frame


def _before_after_columns(pre: Mapping[str, Any], post: Mapping[str, Any]) -> dict[str, Any]:
    before = _cell_state_columns(pre) if isinstance(pre, Mapping) else {}
    after = _cell_state_columns(post) if isinstance(post, Mapping) and post else {}
    row = {}
    for species in ("myc", "cdk4", "pdgfra"):
        row[f"k_{species}_before"] = before.get(f"k_{species}")
        row[f"k_{species}_after"] = after.get(f"k_{species}")
    row["hard_state_before"] = before.get("hard_state")
    row["hard_state_after"] = after.get("hard_state")
    row["coarse_state_before"] = before.get("coarse_state")
    row["coarse_state_after"] = after.get("coarse_state")
    for state in ("npc", "opc", "ac", "mes"):
        row[f"x_{state}_before"] = before.get(f"x_{state}")
        row[f"x_{state}_after"] = after.get(f"x_{state}")
    row["r_stress_before"] = before.get("r_stress")
    row["r_stress_after"] = after.get("r_stress")
    row["v_survival_before"] = before.get("v_survival")
    row["v_survival_after"] = after.get("v_survival")
    return row


def _lineage_edges_frame(
    ctx: ExportContext,
    event_records: Sequence[Mapping[str, Any]],
    parent_map: Mapping[int, int | None],
    founder_map: Mapping[int, int],
) -> pd.DataFrame:
    rows = []
    for event in event_records:
        if event["raw_event_type"] != "division":
            continue
        details = event["details"]
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        parent_id = int(event["cell_id"])
        parent_state = _cell_state_columns(pre) if isinstance(pre, Mapping) else {}
        for child_order, daughter_key in enumerate(("daughter_one", "daughter_two"), start=1):
            daughter = details.get(daughter_key, {}) if isinstance(details, Mapping) else {}
            if not isinstance(daughter, Mapping):
                continue
            child_id = _optional_int(daughter.get("cell_id"))
            if child_id is None:
                continue
            child_state = _cell_state_columns(daughter)
            founder_id = founder_map.get(child_id, founder_map.get(parent_id, parent_id))
            rows.append(
                {
                    "sim_id": ctx.sim_id,
                    "condition_id": ctx.condition_id,
                    "replicate_id": ctx.replicate_id,
                    "division_event_id": event["event_id"],
                    "t_birth": float(event["t"]),
                    "parent_id": parent_id,
                    "parent_uid": _cell_uid(ctx, parent_id),
                    "child_id": child_id,
                    "child_uid": _cell_uid(ctx, child_id),
                    "child_order": child_order,
                    "founder_id": founder_id,
                    "founder_uid": _cell_uid(ctx, founder_id),
                    "parent_lineage_depth": _lineage_depth(parent_id, parent_map),
                    "child_lineage_depth": _lineage_depth(child_id, parent_map),
                    "parent_k_myc_before_division": parent_state.get("k_myc"),
                    "parent_k_cdk4_before_division": parent_state.get("k_cdk4"),
                    "parent_k_pdgfra_before_division": parent_state.get("k_pdgfra"),
                    "child_k_myc_birth": child_state.get("k_myc"),
                    "child_k_cdk4_birth": child_state.get("k_cdk4"),
                    "child_k_pdgfra_birth": child_state.get("k_pdgfra"),
                    "parent_state_before_division": parent_state.get("hard_state"),
                    "child_state_birth": child_state.get("hard_state"),
                    "parent_total_burden": parent_state.get("total_burden"),
                    "child_total_burden": child_state.get("total_burden"),
                }
            )
    return pd.DataFrame(rows)


def _division_inheritance_frame(
    ctx: ExportContext,
    event_records: Sequence[Mapping[str, Any]],
    founder_map: Mapping[int, int],
) -> pd.DataFrame:
    rows = []
    for event in event_records:
        if event["raw_event_type"] != "division":
            continue
        details = event["details"]
        pre = details.get("state_pre", {}) if isinstance(details, Mapping) else {}
        d1 = details.get("daughter_one", {}) if isinstance(details, Mapping) else {}
        d2 = details.get("daughter_two", {}) if isinstance(details, Mapping) else {}
        if not all(isinstance(item, Mapping) for item in (pre, d1, d2)):
            continue
        parent_id = int(event["cell_id"])
        parent = _cell_state_columns(pre)
        daughter1 = _cell_state_columns(d1)
        daughter2 = _cell_state_columns(d2)
        row = {
            "sim_id": ctx.sim_id,
            "condition_id": ctx.condition_id,
            "replicate_id": ctx.replicate_id,
            "division_event_id": event["event_id"],
            "t": float(event["t"]),
            "parent_id": parent_id,
            "daughter1_id": _optional_int(d1.get("cell_id")),
            "daughter2_id": _optional_int(d2.get("cell_id")),
            "founder_id": founder_map.get(parent_id, parent_id),
            "parent_state": parent.get("hard_state"),
            "daughter1_state": daughter1.get("hard_state"),
            "daughter2_state": daughter2.get("hard_state"),
        }
        for species in ("myc", "cdk4", "pdgfra"):
            parent_k = _safe_float(parent.get(f"k_{species}"))
            d1_k = _safe_float(daughter1.get(f"k_{species}"))
            d2_k = _safe_float(daughter2.get(f"k_{species}"))
            pool = d1_k + d2_k if np.isfinite(d1_k) and np.isfinite(d2_k) else np.nan
            row[f"parent_k_{species}"] = parent_k
            row[f"amplification_{species}"] = pool - 2.0 * parent_k if np.isfinite(pool) and np.isfinite(parent_k) else np.nan
            row[f"segregation_pool_{species}"] = pool
            row[f"daughter1_k_{species}"] = d1_k
            row[f"daughter2_k_{species}"] = d2_k
            row[f"post_loss_d1_{species}"] = np.nan
            row[f"post_loss_d2_{species}"] = np.nan
            row[f"imbalance_{species}"] = abs(d1_k - d2_k) if np.isfinite(d1_k) and np.isfinite(d2_k) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_registry_frame(
    ctx: ExportContext,
    birth_records: Mapping[int, Mapping[str, Any]],
    final_records: Mapping[int, Mapping[str, Any]],
    parent_map: Mapping[int, int | None],
    founder_map: Mapping[int, int],
    death_map: Mapping[int, Mapping[str, Any]],
    division_terminal_map: Mapping[int, Mapping[str, Any]],
    lineage_edges: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    children_counts = lineage_edges.groupby("parent_id").size().to_dict() if not lineage_edges.empty else {}
    division_counts = lineage_edges.groupby("parent_id")["division_event_id"].nunique().to_dict() if not lineage_edges.empty else {}
    for cell_id in sorted(birth_records):
        birth = birth_records[cell_id]
        birth_state = _cell_state_columns(birth.get("birth_state", {}))
        final_state = _cell_state_columns(final_records.get(cell_id, birth.get("birth_state", {})))
        parent_id = parent_map.get(cell_id)
        founder_id = founder_map.get(cell_id, cell_id)
        death = death_map.get(cell_id, {})
        division = division_terminal_map.get(cell_id, {})
        if death:
            final_t = float(death["death_t"])
            final_status = "dead"
            terminal_event_id = death.get("death_event_id")
        elif division:
            final_t = float(division["division_t"])
            final_status = "divided"
            terminal_event_id = division.get("division_event_id")
        else:
            latest_observed_t = ctx.t_max
            final_t = latest_observed_t
            final_status = "alive_at_tmax" if abs(latest_observed_t - ctx.t_max) <= 1e-8 else "censored"
            terminal_event_id = None
        rows.append(
            {
                **_run_keys(ctx),
                "cell_id": cell_id,
                "cell_uid": _cell_uid(ctx, cell_id),
                "founder_id": founder_id,
                "founder_uid": _cell_uid(ctx, founder_id),
                "parent_id": parent_id,
                "parent_uid": _cell_uid(ctx, parent_id),
                "birth_event_id": birth.get("birth_event_id"),
                "death_event_id": death.get("death_event_id"),
                "birth_t": float(birth.get("birth_t", 0.0)),
                "death_t": death.get("death_t"),
                "final_t": final_t,
                "final_status": final_status,
                "lineage_depth": _lineage_depth(cell_id, parent_map),
                "cell_weight": 1.0,
                "is_founder": parent_id is None,
                "k_myc_birth": birth_state.get("k_myc"),
                "k_cdk4_birth": birth_state.get("k_cdk4"),
                "k_pdgfra_birth": birth_state.get("k_pdgfra"),
                "k_myc_final": final_state.get("k_myc"),
                "k_cdk4_final": final_state.get("k_cdk4"),
                "k_pdgfra_final": final_state.get("k_pdgfra"),
                "total_burden_birth": birth_state.get("total_burden"),
                "total_burden_final": final_state.get("total_burden"),
                "hard_state_birth": birth_state.get("hard_state"),
                "hard_state_final": final_state.get("hard_state"),
                "coarse_state_birth": birth_state.get("coarse_state"),
                "coarse_state_final": final_state.get("coarse_state"),
                "x_npc_birth": birth_state.get("x_npc"),
                "x_opc_birth": birth_state.get("x_opc"),
                "x_ac_birth": birth_state.get("x_ac"),
                "x_mes_birth": birth_state.get("x_mes"),
                "x_npc_final": final_state.get("x_npc"),
                "x_opc_final": final_state.get("x_opc"),
                "x_ac_final": final_state.get("x_ac"),
                "x_mes_final": final_state.get("x_mes"),
                "r_stress_birth": birth_state.get("r_stress"),
                "r_stress_final": final_state.get("r_stress"),
                "v_survival_birth": birth_state.get("v_survival"),
                "v_survival_final": final_state.get("v_survival"),
                "n_divisions_as_parent": int(division_counts.get(cell_id, 0)),
                "n_children_total": int(children_counts.get(cell_id, 0)),
            }
        )
    return pd.DataFrame(rows)


def _cell_terminal_state_frame(
    ctx: ExportContext,
    registry: pd.DataFrame,
    death_map: Mapping[int, Mapping[str, Any]],
    division_terminal_map: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for row in registry.to_dict(orient="records"):
        rows.append(
            {
                "sim_id": ctx.sim_id,
                "condition_id": ctx.condition_id,
                "replicate_id": ctx.replicate_id,
                "cell_id": row.get("cell_id"),
                "cell_uid": row.get("cell_uid"),
                "founder_id": row.get("founder_id"),
                "terminal_t": row.get("final_t"),
                "terminal_reason": row.get("final_status"),
                "terminal_event_id": _terminal_event_id(row, death_map, division_terminal_map),
                "k_myc_terminal": row.get("k_myc_final"),
                "k_cdk4_terminal": row.get("k_cdk4_final"),
                "k_pdgfra_terminal": row.get("k_pdgfra_final"),
                "hard_state_terminal": row.get("hard_state_final"),
                "coarse_state_terminal": row.get("coarse_state_final"),
                "x_npc_terminal": row.get("x_npc_final"),
                "x_opc_terminal": row.get("x_opc_final"),
                "x_ac_terminal": row.get("x_ac_final"),
                "x_mes_terminal": row.get("x_mes_final"),
                "lineage_depth": row.get("lineage_depth"),
                "lifetime_t": _safe_float(row.get("final_t")) - _safe_float(row.get("birth_t")),
            }
        )
    return pd.DataFrame(rows)


def _terminal_event_id(
    row: Mapping[str, Any],
    death_map: Mapping[int, Mapping[str, Any]],
    division_terminal_map: Mapping[int, Mapping[str, Any]],
) -> str | None:
    cell_id = _optional_int(row.get("cell_id"))
    if cell_id is None:
        return None
    if row.get("final_status") == "dead":
        return death_map.get(cell_id, {}).get("death_event_id")
    if row.get("final_status") == "divided":
        return division_terminal_map.get(cell_id, {}).get("division_event_id")
    return None


def _observables_long_frame(ctx: ExportContext, cell_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, t_index), group in cell_snapshot.groupby(["t", "t_index"], dropna=False):
        n_cells = int(len(group))
        base = {
            "sim_id": ctx.sim_id,
            "ensemble_id": ctx.ensemble_id,
            "model_variant": ctx.model_variant,
            "condition_id": ctx.condition_id,
            "initial_condition_id": ctx.initial_condition_id,
            "replicate_id": ctx.replicate_id,
            "t": float(t),
            "t_index": int(t_index),
            "value_noisy": np.nan,
            "n_cells_alive": n_cells,
            "n_effective": n_cells,
            "notes": None,
        }
        rows.append(
            {
                **base,
                "assay": "cell_count",
                "species": None,
                "state_id": None,
                "state_compartment": None,
                "value_true": float(n_cells),
                "unit": "cells",
            }
        )
        for species, column in (("MYC", "k_myc"), ("CDK4", "k_cdk4"), ("PDGFRA", "k_pdgfra")):
            rows.append(
                {
                    **base,
                    "assay": "ddpcr",
                    "species": species,
                    "state_id": None,
                    "state_compartment": None,
                    "value_true": float(pd.to_numeric(group[column], errors="coerce").mean()) if n_cells else np.nan,
                    "unit": "copy_number",
                }
            )
        for compartment in ("OLIG2_high", "AC_like", "MES_like"):
            rows.append(
                {
                    **base,
                    "assay": "flow",
                    "species": None,
                    "state_id": None,
                    "state_compartment": compartment,
                    "value_true": float((group["coarse_state"] == compartment).mean()) if n_cells else np.nan,
                    "unit": "fraction",
                }
            )
    return pd.DataFrame(rows)


def _copy_vector_frame(observables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if observables.empty:
        return pd.DataFrame(columns=COPY_VECTOR_COLUMNS)
    for keys, group in observables.groupby(["sim_id", "model_variant", "condition_id", "initial_condition_id", "replicate_id", "t", "t_index"]):
        sim_id, model_variant, condition_id, initial_condition_id, replicate_id, t, t_index = keys
        row = {
            "sim_id": sim_id,
            "model_variant": model_variant,
            "condition_id": condition_id,
            "initial_condition_id": initial_condition_id,
            "replicate_id": replicate_id,
            "t": float(t),
            "t_index": int(t_index),
            "cell_count": np.nan,
            "olig2_high_fraction": np.nan,
            "ac_like_fraction": np.nan,
            "mes_like_fraction": np.nan,
        }
        for species in ("myc", "cdk4", "pdgfra"):
            row[f"{species}_copy_mean"] = np.nan
            row[f"log2_{species}_copy"] = np.nan
        for item in group.to_dict(orient="records"):
            if item["assay"] == "cell_count":
                row["cell_count"] = item["value_true"]
            elif item["assay"] == "ddpcr" and item["species"] in cfg.SPECIES:
                species = str(item["species"]).lower()
                value = float(item["value_true"])
                row[f"{species}_copy_mean"] = value
                row[f"log2_{species}_copy"] = float(np.log2(value + 1.0)) if np.isfinite(value) else np.nan
            elif item["assay"] == "flow":
                key = {
                    "OLIG2_high": "olig2_high_fraction",
                    "AC_like": "ac_like_fraction",
                    "MES_like": "mes_like_fraction",
                }.get(item["state_compartment"])
                if key:
                    row[key] = item["value_true"]
        row["total_copy_mean"] = sum(
            _safe_float(row[f"{species}_copy_mean"]) for species in ("myc", "cdk4", "pdgfra")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _population_summary_frame(
    ctx: ExportContext,
    cell_snapshot: pd.DataFrame,
    registry: pd.DataFrame,
    event_log: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    n_ever = int(registry.shape[0])
    for (t, t_index), group in cell_snapshot.groupby(["t", "t_index"], dropna=False):
        event_subset = event_log[event_log["t"] <= float(t)] if not event_log.empty else pd.DataFrame()
        deaths = int((event_subset["event_type"] == "death").sum()) if not event_subset.empty else 0
        divisions = int((event_subset["event_type"] == "division").sum()) if not event_subset.empty else 0
        n_cells = int(len(group))
        rows.append(
            {
                **_run_keys(ctx),
                "t": float(t),
                "t_index": int(t_index),
                "n_alive_cells": n_cells,
                "n_cells_ever_born": n_ever,
                "n_dead_cumulative": deaths,
                "n_divisions_cumulative": divisions,
                "mean_myc": _mean(group, "k_myc"),
                "mean_cdk4": _mean(group, "k_cdk4"),
                "mean_pdgfra": _mean(group, "k_pdgfra"),
                "mean_total_burden": _mean(group, "total_burden"),
                "median_total_burden": _median(group, "total_burden"),
                "olig2_high_fraction": _fraction(group, "coarse_state", "OLIG2_high"),
                "ac_like_fraction": _fraction(group, "coarse_state", "AC_like"),
                "mes_like_fraction": _fraction(group, "coarse_state", "MES_like"),
                "npc_fraction": _fraction(group, "hard_state", "NPC"),
                "opc_fraction": _fraction(group, "hard_state", "OPC"),
                "ac_fraction": _fraction(group, "hard_state", "AC"),
                "mes_fraction": _fraction(group, "hard_state", "MES"),
                "mean_r_stress": _mean(group, "r_stress"),
                "mean_v_survival": _mean(group, "v_survival"),
            }
        )
    return pd.DataFrame(rows)


def _state_copy_summary_frame(ctx: ExportContext, cell_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    compartments = (
        ("OLIG2_high", "coarse", lambda frame: frame["coarse_state"] == "OLIG2_high"),
        ("AC_like", "coarse", lambda frame: frame["coarse_state"] == "AC_like"),
        ("MES_like", "coarse", lambda frame: frame["coarse_state"] == "MES_like"),
        ("NPC", "latent_four_state", lambda frame: frame["hard_state"] == "NPC"),
        ("OPC", "latent_four_state", lambda frame: frame["hard_state"] == "OPC"),
        ("AC", "latent_four_state", lambda frame: frame["hard_state"] == "AC"),
        ("MES", "latent_four_state", lambda frame: frame["hard_state"] == "MES"),
    )
    species_columns = (("MYC", "k_myc"), ("CDK4", "k_cdk4"), ("PDGFRA", "k_pdgfra"), ("total_burden", "total_burden"))
    for (t, t_index), time_group in cell_snapshot.groupby(["t", "t_index"], dropna=False):
        total = float(len(time_group))
        for compartment, level, mask_fn in compartments:
            group = time_group[mask_fn(time_group)]
            for species, column in species_columns:
                values = pd.to_numeric(group[column], errors="coerce") if column in group else pd.Series(dtype=float)
                row = {
                    **_run_keys(ctx),
                    "t": float(t),
                    "t_index": int(t_index),
                    "state_compartment": compartment,
                    "state_level": level,
                    "species": species,
                    "weighted_cell_count": float(len(group)),
                    "state_fraction": float(len(group) / total) if total else np.nan,
                    "mean_copy": float(values.mean()) if len(values) else np.nan,
                    "median_copy": float(values.median()) if len(values) else np.nan,
                    "q05_copy": float(values.quantile(0.05)) if len(values) else np.nan,
                    "q25_copy": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q75_copy": float(values.quantile(0.75)) if len(values) else np.nan,
                    "q95_copy": float(values.quantile(0.95)) if len(values) else np.nan,
                    "zero_fraction": float((values == 0).mean()) if len(values) else np.nan,
                }
                row.update(_bin_fractions(values))
                rows.append(row)
    return pd.DataFrame(rows)


def _founder_t_summary_frame(ctx: ExportContext, cell_snapshot: pd.DataFrame, event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, t_index, founder_id), group in cell_snapshot.groupby(["t", "t_index", "founder_id"], dropna=False):
        founder_id = int(founder_id)
        events = event_log[(event_log["founder_id"] == founder_id) & (event_log["t"] <= float(t))] if not event_log.empty else pd.DataFrame()
        row = {
            "sim_id": ctx.sim_id,
            "condition_id": ctx.condition_id,
            "initial_condition_id": ctx.initial_condition_id,
            "replicate_id": ctx.replicate_id,
            "t": float(t),
            "t_index": int(t_index),
            "founder_id": founder_id,
            "founder_uid": _cell_uid(ctx, founder_id),
            "descendant_count_alive": int(len(group)),
            "weighted_descendant_count": float(len(group)),
            "myc_copy_sum": float(group["k_myc"].sum()),
            "cdk4_copy_sum": float(group["k_cdk4"].sum()),
            "pdgfra_copy_sum": float(group["k_pdgfra"].sum()),
            "total_burden_sum": float(group["total_burden"].sum()),
            "myc_copy_mean": _mean(group, "k_myc"),
            "cdk4_copy_mean": _mean(group, "k_cdk4"),
            "pdgfra_copy_mean": _mean(group, "k_pdgfra"),
            "olig2_high_count": int((group["coarse_state"] == "OLIG2_high").sum()),
            "ac_like_count": int((group["coarse_state"] == "AC_like").sum()),
            "mes_like_count": int((group["coarse_state"] == "MES_like").sum()),
            "npc_count": int((group["hard_state"] == "NPC").sum()),
            "opc_count": int((group["hard_state"] == "OPC").sum()),
            "ac_count": int((group["hard_state"] == "AC").sum()),
            "mes_count": int((group["hard_state"] == "MES").sum()),
            "division_count_cumulative": int((events["event_type"] == "division").sum()) if not events.empty else 0,
            "death_count_cumulative": int((events["event_type"] == "death").sum()) if not events.empty else 0,
        }
        for event, column in (
            ("ecDNA_gain", "gain"),
            ("ecDNA_loss", "loss"),
        ):
            for species in ("myc", "cdk4", "pdgfra"):
                row[f"{column}_{species}_count_cumulative"] = int(
                    ((events["event_type"] == event) & (events["species"].str.lower() == species)).sum()
                ) if not events.empty and "species" in events else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _copy_distribution_summary_frame(ctx: ExportContext, cell_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, _t_index), group in cell_snapshot.groupby(["t", "t_index"], dropna=False):
        total = float(len(group))
        for species, column in (("MYC", "k_myc"), ("CDK4", "k_cdk4"), ("PDGFRA", "k_pdgfra")):
            values = pd.to_numeric(group[column], errors="coerce")
            for bin_id, lower, upper, label in COPY_BINS:
                mask = values >= lower if upper is None else (values >= lower) & (values <= upper)
                subset = values[mask]
                rows.append(
                    {
                        "sim_id": ctx.sim_id,
                        "condition_id": ctx.condition_id,
                        "replicate_id": ctx.replicate_id,
                        "t": float(t),
                        "species": species,
                        "copy_bin_id": bin_id,
                        "copy_bin_label": label,
                        "n_cells": int(mask.sum()),
                        "fraction": float(mask.sum() / total) if total else np.nan,
                        "mean_copy_in_bin": float(subset.mean()) if len(subset) else np.nan,
                        "state_compartment": None,
                    }
                )
    return pd.DataFrame(rows)


def _event_summary_frame(ctx: ExportContext, event_log: pd.DataFrame, population_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not ctx.t_values:
        return pd.DataFrame(rows)
    starts = [ctx.t_values[0], *ctx.t_values[:-1]]
    ends = list(ctx.t_values)
    for start, end in zip(starts, ends):
        if start == end:
            subset = event_log[event_log["t"] <= end] if not event_log.empty else pd.DataFrame()
        else:
            subset = event_log[(event_log["t"] > start) & (event_log["t"] <= end)] if not event_log.empty else pd.DataFrame()
        pop_row = population_summary[population_summary["t"] == start] if not population_summary.empty else pd.DataFrame()
        n_cells = float(pop_row["n_alive_cells"].iloc[0]) if not pop_row.empty else np.nan
        interval = max(0.0, float(end) - float(start))
        if subset.empty:
            rows.append(_event_summary_row(ctx, start, end, None, None, 0, n_cells, interval))
            continue
        for (event_type, species), group in subset.groupby(["event_type", "species"], dropna=False):
            rows.append(_event_summary_row(ctx, start, end, event_type, species, int(len(group)), n_cells, interval))
    return pd.DataFrame(rows)


def _event_summary_row(
    ctx: ExportContext,
    start: float,
    end: float,
    event_type: str | None,
    species: str | None,
    count: int,
    n_cells: float,
    interval: float,
) -> dict[str, Any]:
    return {
        "sim_id": ctx.sim_id,
        "condition_id": ctx.condition_id,
        "replicate_id": ctx.replicate_id,
        "t_start": float(start),
        "t_end": float(end),
        "event_type": event_type,
        "species": species,
        "state_compartment": None,
        "event_count": int(count),
        "event_rate_per_cell": float(count / n_cells) if np.isfinite(n_cells) and n_cells > 0 else np.nan,
        "event_rate_per_t": float(count / interval) if interval > 0 else np.nan,
        "founder_group": None,
    }


def _lineage_family_summary_frame(
    ctx: ExportContext,
    registry: pd.DataFrame,
    founder_t: pd.DataFrame,
    event_log: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if registry.empty:
        return pd.DataFrame(rows)
    founders = registry[registry["is_founder"] == True]  # noqa: E712
    final_t = founder_t["t"].max() if not founder_t.empty else np.nan
    final_founder_t = founder_t[founder_t["t"] == final_t] if np.isfinite(final_t) else pd.DataFrame()
    for founder in founders.to_dict(orient="records"):
        founder_id = int(founder["cell_id"])
        final = final_founder_t[final_founder_t["founder_id"] == founder_id] if not final_founder_t.empty else pd.DataFrame()
        family_ts = founder_t[founder_t["founder_id"] == founder_id] if not founder_t.empty else pd.DataFrame()
        events = event_log[event_log["founder_id"] == founder_id] if not event_log.empty else pd.DataFrame()
        dominant_state = None
        if not final.empty:
            counts = {
                "OLIG2_high": final["olig2_high_count"].iloc[0],
                "AC_like": final["ac_like_count"].iloc[0],
                "MES_like": final["mes_like_count"].iloc[0],
            }
            dominant_state = max(counts, key=counts.get)
        rows.append(
            {
                "sim_id": ctx.sim_id,
                "condition_id": ctx.condition_id,
                "replicate_id": ctx.replicate_id,
                "founder_id": founder_id,
                "founder_uid": _cell_uid(ctx, founder_id),
                "founder_state": founder.get("hard_state_birth"),
                "founder_coarse_state": founder.get("coarse_state_birth"),
                "founder_k_myc": founder.get("k_myc_birth"),
                "founder_k_cdk4": founder.get("k_cdk4_birth"),
                "founder_k_pdgfra": founder.get("k_pdgfra_birth"),
                "final_descendant_count": int(final["descendant_count_alive"].iloc[0]) if not final.empty else 0,
                "final_myc_copy_sum": float(final["myc_copy_sum"].iloc[0]) if not final.empty else 0.0,
                "final_cdk4_copy_sum": float(final["cdk4_copy_sum"].iloc[0]) if not final.empty else 0.0,
                "final_pdgfra_copy_sum": float(final["pdgfra_copy_sum"].iloc[0]) if not final.empty else 0.0,
                "final_total_burden_sum": float(final["total_burden_sum"].iloc[0]) if not final.empty else 0.0,
                "max_descendant_count": int(family_ts["descendant_count_alive"].max()) if not family_ts.empty else 0,
                "max_cdk4_copy_sum": float(family_ts["cdk4_copy_sum"].max()) if not family_ts.empty else 0.0,
                "survival_status": "surviving" if not final.empty and int(final["descendant_count_alive"].iloc[0]) > 0 else "extinct",
                "dominant_final_state": dominant_state,
                "n_divisions_total": int((events["event_type"] == "division").sum()) if not events.empty else 0,
                "n_deaths_total": int((events["event_type"] == "death").sum()) if not events.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _parameter_table_frame(ctx: ExportContext, metadata: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _append_parameter_rows(rows, cfg.DEFAULT_MODEL_PARAMETERS, block=None, path=())
    simulation = metadata.get("simulation", {})
    if isinstance(simulation, Mapping):
        for key, value in simulation.items():
            if _is_scalar(value):
                rows.append(
                    _parameter_row(
                        ctx,
                        block="simulation",
                        name=str(key),
                        value=value,
                        description="simulation metadata",
                    )
                )
    return pd.DataFrame(rows)


def _append_parameter_rows(
    rows: list[dict[str, Any]],
    value: Any,
    *,
    block: str | None,
    path: tuple[str, ...],
    species: str | None = None,
) -> None:
    if is_dataclass(value):
        for field in fields(value):
            child = getattr(value, field.name)
            child_block = field.name if block is None else block
            _append_parameter_rows(rows, child, block=child_block, path=(*path, field.name), species=species)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in cfg.SPECIES:
                _append_parameter_rows(rows, child, block=block, path=path, species=str(key))
            elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(item, int) for item in key):
                rows.append(
                    _parameter_row(
                        None,
                        block=block or "model",
                        name=".".join(path),
                        value=child,
                        species=species,
                        from_state=_compact_state(cfg.STATE_NAMES[int(key[0])]),
                        to_state=_compact_state(cfg.STATE_NAMES[int(key[1])]),
                        description="state transition edge parameter",
                    )
                )
            else:
                _append_parameter_rows(rows, child, block=block, path=(*path, _safe_token(str(key))), species=species)
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 1 and array.shape[0] == cfg.N_SPECIES:
            for idx, species_name in enumerate(cfg.SPECIES):
                rows.append(_parameter_row(None, block=block or "model", name=".".join(path), value=array[idx], species=species_name))
        elif array.ndim == 1 and array.shape[0] == cfg.N_STATES:
            for idx, state_name in enumerate(cfg.STATE_NAMES):
                rows.append(
                    _parameter_row(
                        None,
                        block=block or "model",
                        name=".".join(path),
                        value=array[idx],
                        species=species,
                        state_id=_compact_state(state_name),
                    )
                )
        elif array.ndim == 2 and array.shape == (cfg.N_STATES, cfg.N_STATES):
            for i, from_state in enumerate(cfg.STATE_NAMES):
                for j, to_state in enumerate(cfg.STATE_NAMES):
                    rows.append(
                        _parameter_row(
                            None,
                            block=block or "model",
                            name=".".join(path),
                            value=array[i, j],
                            species=species,
                            from_state=_compact_state(from_state),
                            to_state=_compact_state(to_state),
                        )
                    )
        return
    if _is_scalar(value):
        rows.append(_parameter_row(None, block=block or "model", name=".".join(path), value=value, species=species))


def _parameter_row(
    ctx: ExportContext | None,
    *,
    block: str,
    name: str,
    value: Any,
    species: str | None = None,
    state_id: str | None = None,
    from_state: str | None = None,
    to_state: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "sim_id": ctx.sim_id if ctx is not None else None,
        "parameter_set_id": ctx.parameter_set_id if ctx is not None else PARAMETER_SET_ID,
        "model_variant": ctx.model_variant if ctx is not None else MODEL_VARIANT,
        "condition_id": ctx.condition_id if ctx is not None else None,
        "parameter_block": block,
        "parameter_name": name,
        "species": species,
        "state_id": state_id,
        "from_state": from_state,
        "to_state": to_state,
        "dose_value": None,
        "value": _numeric_or_nan(value),
        "unit": _parameter_unit(block, name),
        "fixed_or_free": "fixed",
        "description": description,
    }


def _parameter_blocks_frame() -> pd.DataFrame:
    blocks = (
        ("growth", True, None, "Cell-cycle and division growth terms."),
        ("death", True, None, "Cell death hazard terms."),
        ("copy_selection", True, None, "Copy-number-dependent selection terms."),
        ("turnover", True, None, "ecDNA gain/loss turnover terms."),
        ("state_transition", True, None, "Latent state transition terms."),
        ("drug_effect", True, None, "Drug exposure and response terms."),
        ("inheritance", True, None, "Division inheritance terms."),
        ("stress", True, None, "Latent stress terms."),
        ("survival", True, None, "Latent survival reserve terms."),
        ("simulation", True, None, "Simulation controls."),
    )
    return pd.DataFrame(
        [
            {
                "parameter_block": block,
                "enabled": enabled,
                "disabled_in_variant": disabled,
                "description": description,
            }
            for block, enabled, disabled, description in blocks
        ]
    )


def _conditions_frame(active_condition: str) -> pd.DataFrame:
    rows = []
    for idx, condition in enumerate(cfg.T87_CONDITION_TREATMENTS):
        rows.append(_condition_row(condition, idx))
    if active_condition not in cfg.T87_CONDITION_TREATMENTS:
        rows.append(_condition_row(active_condition, len(rows)))
    return pd.DataFrame(rows)


def _condition_row(condition: str, plot_order: int) -> dict[str, Any]:
    drug, dose = cfg.T87_CONDITION_TREATMENTS.get(condition, ("vehicle", 0.0))
    return {
        "condition_id": _condition_id(condition),
        "treatment_family": _treatment_family(condition),
        "drug_name": drug,
        "dose_value": float(dose),
        "dose_unit": "nM",
        "target_species": _target_species(condition),
        "treatment_start_t": 0.0,
        "treatment_end_t": 12.0,
        "plot_order": int(plot_order),
        "label": _condition_label(condition),
    }


def _model_variants_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_variant": MODEL_VARIANT,
                "variant_label": "Full model",
                "variant_type": "full",
                "disabled_mechanism": None,
                "description": "Full ecDNA-state simulation model.",
                "plot_order": 0,
            }
        ]
    )


def _initial_conditions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "initial_condition_id": INITIAL_CONDITION_ID,
                "description": "Parental mixed initial population.",
                "initial_olig2_high_fraction": np.nan,
                "initial_ac_fraction": np.nan,
                "initial_mes_fraction": np.nan,
                "initial_npc_fraction": np.nan,
                "initial_opc_fraction": np.nan,
                "plot_order": 0,
            }
        ]
    )


def _t_grid_frame(ctx: ExportContext) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t_grid_id": ctx.snapshot_grid_id,
                "t": float(t),
                "t_index": _t_index(ctx, t),
                "is_snapshot_t": True,
                "is_observable_t": True,
                "is_dense_t": ctx.dense_output,
                "label": f"t={_format_t(t)}",
            }
            for t in ctx.t_values
        ]
    )


def _species_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "species": species,
                "species_index": idx + 1,
                "ecDNA_label": species,
                "targetable": species in {"CDK4", "PDGFRA"},
                "plot_order": idx,
            }
            for idx, species in enumerate(cfg.SPECIES)
        ]
    )


def _state_definitions_frame() -> pd.DataFrame:
    rows = []
    for idx, state_name in enumerate(cfg.STATE_NAMES):
        state_id = _compact_state(state_name)
        rows.append(
            {
                "state_id": state_id,
                "state_level": "latent_four_state",
                "definition": state_name,
                "maps_to_coarse_state": _coarse_state(state_id),
                "plot_order": idx,
            }
        )
    for idx, state_id in enumerate(("OLIG2_high", "AC_like", "MES_like"), start=len(rows)):
        rows.append(
            {
                "state_id": state_id,
                "state_level": "coarse_observed",
                "definition": "NPC+OPC collapsed" if state_id == "OLIG2_high" else state_id,
                "maps_to_coarse_state": state_id,
                "plot_order": idx,
            }
        )
    return pd.DataFrame(rows)


def _assay_definitions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("cell_count", "bulk", False, False, "Alive cell count."),
            ("ddpcr", "bulk", True, False, "Bulk ecDNA copy number."),
            ("flow", "state_fraction", False, True, "Coarse state fraction."),
            ("qpcdr", "single_cell_virtual", True, True, "Reserved virtual qPCDR-like assay."),
            ("ectag_like", "single_cell_virtual", True, True, "Reserved virtual ecTAG-like assay."),
        ],
        columns=("assay", "assay_level", "species_required", "state_required", "description"),
    )


def _copy_bins_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "copy_bin_id": bin_id,
                "lower": lower,
                "upper": upper,
                "label": label,
                "plot_order": idx,
            }
            for idx, (bin_id, lower, upper, label) in enumerate(COPY_BINS)
        ]
    )


def _event_type_definitions_frame() -> pd.DataFrame:
    event_types = (
        ("initialization", "Initial cells."),
        ("division", "Cell division."),
        ("death", "Cell death."),
        ("ecDNA_gain", "Single-copy ecDNA gain."),
        ("ecDNA_loss", "Single-copy ecDNA loss."),
        ("ecDNA_amplification", "Division-coupled amplification."),
        ("post_segregation_loss", "Post-segregation copy loss."),
        ("state_jump", "Discrete state jump if present."),
        ("cell_cycle_transition", "Cell-cycle transition."),
        ("drug_change", "Treatment schedule change."),
        ("bottleneck", "Population bottleneck."),
        ("sampling", "Virtual assay sampling."),
    )
    return pd.DataFrame(
        [
            {
                "event_type": event_type,
                "description": description,
                "plot_order": idx,
            }
            for idx, (event_type, description) in enumerate(event_types)
        ]
    )


def _run_manifest(
    ctx: ExportContext,
    result: SimulationResult,
    metadata: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "sim_id": ctx.sim_id,
        "ensemble_id": ctx.ensemble_id,
        "model_variant": ctx.model_variant,
        "condition_id": ctx.condition_id,
        "initial_condition_id": ctx.initial_condition_id,
        "replicate_id": ctx.replicate_id,
        "seed": ctx.seed,
        "parameter_set_id": ctx.parameter_set_id,
        "time_variable": "t",
        "t_min": ctx.t_min,
        "t_max": ctx.t_max,
        "snapshot_grid_id": ctx.snapshot_grid_id,
        "dense_output": ctx.dense_output,
        "records_all_cells_ever_born": True,
        "records_all_alive_cells_at_snapshots": True,
        "records_all_events": True,
        "uses_cell_weights": False,
        "initial_n_cells": _initial_n_cells(tables["cell_snapshot"]),
        "final_n_cells": int(result.population_sizes[-1]) if result.population_sizes else 0,
        "n_cells_ever_born": int(tables["cell_registry"].shape[0]),
        "n_events": int(tables["event_log"].shape[0]),
        "stop_reason": result.stop_reason,
        "stop_t": result.stop_time,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata": _json_ready(dict(metadata)),
    }


def _update_run_index(ensemble_dir: Path, run_dir: Path, ctx: ExportContext) -> None:
    path = ensemble_dir / "run_index.parquet"
    row = {
        "ensemble_id": ctx.ensemble_id,
        "sim_id": ctx.sim_id,
        "model_variant": ctx.model_variant,
        "condition_id": ctx.condition_id,
        "initial_condition_id": ctx.initial_condition_id,
        "replicate_id": ctx.replicate_id,
        "seed": ctx.seed,
        "parameter_set_id": ctx.parameter_set_id,
        "t_min": ctx.t_min,
        "t_max": ctx.t_max,
        "snapshot_grid_id": ctx.snapshot_grid_id,
        "dense_output": ctx.dense_output,
        "output_path": str(run_dir.relative_to(ensemble_dir)),
        "status": "completed",
    }
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=RUN_INDEX_COLUMNS)
    if not existing.empty and "sim_id" in existing:
        existing = existing[existing["sim_id"] != ctx.sim_id]
    updated = pd.DataFrame([row]) if existing.empty else pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    _enforce_schema(updated, RUN_INDEX_COLUMNS).to_parquet(path, index=False)


def _qc_reports(run_dir: Path, tables: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = []

    registry = tables["cell_registry"]
    snapshot = tables["cell_snapshot"]
    event_log = tables["event_log"]
    lineage = tables["lineage_edges"]
    observables = tables["observables_long"]

    required_files = [run_dir / path for path in REQUIRED_RUN_FILES]
    all_required = all(path.exists() for path in required_files)
    checks.append(_check_row("all_required_files_exist", all_required, 0 if all_required else sum(not path.exists() for path in required_files), None))

    duplicate_cell_uid = int(registry["cell_uid"].duplicated().sum()) if not registry.empty else 0
    checks.append(_check_row("unique_cell_uid", duplicate_cell_uid == 0, duplicate_cell_uid, None))

    cell_ids = set(_dropna_ints(registry["cell_id"])) if not registry.empty else set()
    parent_ids = set(_dropna_ints(registry["parent_id"])) if not registry.empty else set()
    missing_parents = parent_ids - cell_ids
    checks.append(_check_row("parent_exists", not missing_parents, len(missing_parents), ",".join(map(str, sorted(missing_parents))) or None))

    founder_ids = set(_dropna_ints(registry["founder_id"])) if not registry.empty else set()
    missing_founders = founder_ids - cell_ids
    checks.append(_check_row("founder_exists", not missing_founders, len(missing_founders), ",".join(map(str, sorted(missing_founders))) or None))

    child_ids = set(_dropna_ints(lineage["child_id"])) if not lineage.empty else set()
    missing_children = child_ids - cell_ids
    checks.append(_check_row("child_ids_in_registry", not missing_children, len(missing_children), ",".join(map(str, sorted(missing_children))) or None))

    event_cell_ids = set(_dropna_ints(event_log["cell_id"])) if not event_log.empty else set()
    event_daughter_ids = set(_dropna_ints(event_log["daughter1_id"])).union(set(_dropna_ints(event_log["daughter2_id"]))) if not event_log.empty else set()
    missing_events = (event_cell_ids | event_daughter_ids) - cell_ids
    checks.append(_check_row("event_cells_in_registry", not missing_events, len(missing_events), ",".join(map(str, sorted(missing_events))) or None))

    alive_violations = _snapshot_alive_violations(snapshot, registry)
    checks.append(_check_row("snapshot_alive_only", alive_violations == 0, alive_violations, None))

    negative_copy = _negative_copy_violations(tables)
    checks.append(_check_row("no_negative_copy", negative_copy == 0, negative_copy, None))

    state_sum_violations = _state_sum_violations(snapshot)
    checks.append(_check_row("state_compositions_sum_to_one", state_sum_violations == 0, state_sum_violations, None))

    state_violations = _state_label_violations(snapshot)
    checks.append(_check_row("state_composition_legal", state_violations == 0, state_violations, None))

    observable_violations = _observable_snapshot_violations(snapshot, observables)
    checks.append(_check_row("observables_match_snapshots", observable_violations == 0, observable_violations, None))

    forbidden = _forbidden_column_violations(tables)
    checks.append(_check_row("no_forbidden_time_columns", forbidden == 0, forbidden, None))

    report = pd.DataFrame(checks)
    integrity = {
        "all_required_files_exist": bool(report.loc[report["check_name"] == "all_required_files_exist", "passed"].iloc[0]),
        "no_duplicate_cell_uid": bool(report.loc[report["check_name"] == "unique_cell_uid", "passed"].iloc[0]),
        "all_child_ids_in_registry": bool(report.loc[report["check_name"] == "child_ids_in_registry", "passed"].iloc[0]),
        "all_parent_ids_in_registry_or_na": bool(report.loc[report["check_name"] == "parent_exists", "passed"].iloc[0]),
        "all_event_cell_ids_in_registry": bool(report.loc[report["check_name"] == "event_cells_in_registry", "passed"].iloc[0]),
        "all_snapshot_cells_alive": bool(report.loc[report["check_name"] == "snapshot_alive_only", "passed"].iloc[0]),
        "no_negative_copy_numbers": bool(report.loc[report["check_name"] == "no_negative_copy", "passed"].iloc[0]),
        "state_compositions_sum_to_one": bool(report.loc[report["check_name"] == "state_compositions_sum_to_one", "passed"].iloc[0]),
        "observables_match_snapshots": bool(report.loc[report["check_name"] == "observables_match_snapshots", "passed"].iloc[0]),
        "no_forbidden_time_columns": bool(report.loc[report["check_name"] == "no_forbidden_time_columns", "passed"].iloc[0]),
        "all_checks_passed": bool(report["passed"].all()),
    }
    return report, integrity


def _check_row(check_name: str, passed: bool, n_violations: int, notes: str | None) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "passed": bool(passed),
        "n_violations": int(n_violations),
        "notes": notes,
    }


def _snapshot_alive_violations(snapshot: pd.DataFrame, registry: pd.DataFrame) -> int:
    if snapshot.empty or registry.empty:
        return 0
    life = registry.set_index("cell_id")[["birth_t", "death_t", "final_t", "final_status"]].to_dict(orient="index")
    violations = 0
    for row in snapshot.to_dict(orient="records"):
        cell_id = _optional_int(row.get("cell_id"))
        if cell_id not in life or row.get("alive") is not True:
            violations += 1
            continue
        t = float(row["t"])
        birth_t = _safe_float(life[cell_id].get("birth_t"))
        death_t = life[cell_id].get("death_t")
        final_t = _safe_float(life[cell_id].get("final_t"))
        status = life[cell_id].get("final_status")
        if t + 1e-9 < birth_t:
            violations += 1
        if pd.notna(death_t) and t >= float(death_t) - 1e-9:
            violations += 1
        if status == "divided" and t >= final_t - 1e-9:
            violations += 1
    return violations


def _negative_copy_violations(tables: Mapping[str, pd.DataFrame]) -> int:
    violations = 0
    for frame in tables.values():
        for column in frame.columns:
            lower = column.lower()
            if lower.startswith("k_") or lower.endswith("_copy") or lower.endswith("_copy_mean") or lower.endswith("_burden"):
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                violations += int((values < 0).sum())
    return violations


def _state_sum_violations(snapshot: pd.DataFrame) -> int:
    if snapshot.empty:
        return 0
    sums = snapshot[["x_npc", "x_opc", "x_ac", "x_mes"]].sum(axis=1)
    return int((np.abs(sums - 1.0) > 1e-6).sum())


def _state_label_violations(snapshot: pd.DataFrame) -> int:
    if snapshot.empty:
        return 0
    allowed_hard = {"NPC", "OPC", "AC", "MES"}
    allowed_coarse = {"OLIG2_high", "AC_like", "MES_like"}
    return int((~snapshot["hard_state"].isin(allowed_hard)).sum() + (~snapshot["coarse_state"].isin(allowed_coarse)).sum())


def _observable_snapshot_violations(snapshot: pd.DataFrame, observables: pd.DataFrame) -> int:
    if snapshot.empty:
        return 0
    violations = 0
    obs = observables.set_index(["t", "assay", "species", "state_compartment"], drop=False)
    for t, group in snapshot.groupby("t", dropna=False):
        checks = [
            (("cell_count", None, None), float(len(group))),
            (("ddpcr", "MYC", None), _mean(group, "k_myc")),
            (("ddpcr", "CDK4", None), _mean(group, "k_cdk4")),
            (("ddpcr", "PDGFRA", None), _mean(group, "k_pdgfra")),
            (("flow", None, "OLIG2_high"), _fraction(group, "coarse_state", "OLIG2_high")),
            (("flow", None, "AC_like"), _fraction(group, "coarse_state", "AC_like")),
            (("flow", None, "MES_like"), _fraction(group, "coarse_state", "MES_like")),
        ]
        for (assay, species, state_compartment), expected in checks:
            subset = observables[
                (observables["t"] == t)
                & (observables["assay"] == assay)
                & (observables["species"].isna() if species is None else observables["species"].eq(species))
                & (observables["state_compartment"].isna() if state_compartment is None else observables["state_compartment"].eq(state_compartment))
            ]
            if subset.empty or abs(float(subset["value_true"].iloc[0]) - float(expected)) > 1e-6:
                violations += 1
    return violations


def _forbidden_column_violations(tables: Mapping[str, pd.DataFrame]) -> int:
    violations = 0
    for frame in tables.values():
        for column in frame.columns:
            parts = set(str(column).lower().split("_"))
            lower = str(column).lower()
            if parts & FORBIDDEN_COLUMN_TOKENS or lower in FORBIDDEN_COLUMN_TOKENS:
                violations += 1
    return violations


def _write_partitioned_cell_snapshot(frame: pd.DataFrame, path: Path) -> None:
    _replace_output_path(path)
    path.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame.to_parquet(path / "part-000.parquet", index=False)
        return
    partition_cols = ["condition_id", "replicate_id", "t_index"]
    for keys, group in frame.groupby(partition_cols, dropna=False):
        condition_id, replicate_id, t_index = keys
        partition_dir = (
            path
            / f"condition_id={condition_id}"
            / f"replicate_id={replicate_id}"
            / f"t_index={int(t_index)}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=partition_cols).to_parquet(partition_dir / "part-000.parquet", index=False)


def _ensemble_output_paths(ensemble_dir: Path) -> dict[str, Path]:
    paths = {
        "ensemble_manifest": ensemble_dir / "ensemble_manifest.json",
        "run_index": ensemble_dir / "run_index.parquet",
    }
    for name in (
        "conditions",
        "model_variants",
        "initial_conditions",
        "t_grid",
        "species",
        "state_definitions",
        "assay_definitions",
        "copy_bins",
        "event_type_definitions",
    ):
        paths[f"metadata_{name}"] = ensemble_dir / "metadata" / f"{name}.parquet"
    return paths


def _run_output_paths(run_dir: Path) -> dict[str, Path]:
    paths = {"run_manifest": run_dir / "manifest.json", "cell_snapshot": run_dir / "root" / "cell_snapshot"}
    for name in ("parameter_table", "parameter_blocks"):
        paths[name] = run_dir / "parameters" / f"{name}.parquet"
    for name in (
        "observables_long",
        "copy_vector",
        "cell_registry",
        "cell_terminal_state",
        "event_log",
        "lineage_edges",
        "division_inheritance",
        "virtual_assay_draws",
    ):
        paths[name] = run_dir / "root" / f"{name}.parquet"
    for name in (
        "population_summary",
        "state_copy_summary",
        "founder_t_summary",
        "copy_distribution_summary",
        "event_summary",
        "lineage_family_summary",
    ):
        paths[name] = run_dir / "cache" / f"{name}.parquet"
    paths["output_integrity_report"] = run_dir / "qc" / "output_integrity_report.json"
    paths["id_consistency_report"] = run_dir / "qc" / "id_consistency_report.parquet"
    return paths


def _run_keys(ctx: ExportContext) -> dict[str, Any]:
    return {
        "sim_id": ctx.sim_id,
        "model_variant": ctx.model_variant,
        "condition_id": ctx.condition_id,
        "initial_condition_id": ctx.initial_condition_id,
        "replicate_id": ctx.replicate_id,
    }


def _cell_state_columns(cell: Mapping[str, Any]) -> dict[str, Any]:
    copies = list(cell.get("copy_numbers", [])) if isinstance(cell, Mapping) else []
    soft = list(cell.get("soft_state", [])) if isinstance(cell, Mapping) else []
    latent = list(cell.get("latent_state", [])) if isinstance(cell, Mapping) else []
    hard_state = _compact_state(cell.get("dominant_state")) if isinstance(cell, Mapping) else None
    if hard_state is None and len(soft) == cfg.N_STATES:
        hard_state = _compact_state(cfg.STATE_NAMES[int(np.argmax(np.asarray(soft, dtype=float)))])
    row = {
        "hard_state": hard_state,
        "coarse_state": _coarse_state(hard_state),
        "u1": _sequence_value(latent, 0),
        "u2": _sequence_value(latent, 1),
        "u3": _sequence_value(latent, 2),
        "x_npc": _sequence_value(soft, cfg.NPC),
        "x_opc": _sequence_value(soft, cfg.OPC),
        "x_ac": _sequence_value(soft, cfg.AC),
        "x_mes": _sequence_value(soft, cfg.MES),
        "k_myc": _sequence_value(copies, cfg.MYC),
        "k_cdk4": _sequence_value(copies, cfg.CDK4),
        "k_pdgfra": _sequence_value(copies, cfg.PDGFRA),
        "r_stress": cell.get("stress_score") if isinstance(cell, Mapping) else np.nan,
        "v_survival": cell.get("survival_score") if isinstance(cell, Mapping) else np.nan,
        "cell_cycle_state": cell.get("cycle_state") if isinstance(cell, Mapping) else None,
        "division_hazard": cell.get("division_hazard") if isinstance(cell, Mapping) else np.nan,
        "death_hazard": cell.get("death_hazard") if isinstance(cell, Mapping) else np.nan,
    }
    row["olig2_high_score"] = _safe_float(row["x_npc"]) + _safe_float(row["x_opc"])
    row["total_burden"] = _safe_float(row["k_myc"]) + _safe_float(row["k_cdk4"]) + _safe_float(row["k_pdgfra"])
    row["log1p_myc"] = float(np.log1p(row["k_myc"])) if pd.notna(row["k_myc"]) else np.nan
    row["log1p_cdk4"] = float(np.log1p(row["k_cdk4"])) if pd.notna(row["k_cdk4"]) else np.nan
    row["log1p_pdgfra"] = float(np.log1p(row["k_pdgfra"])) if pd.notna(row["k_pdgfra"]) else np.nan
    return row


def _normalise_event_type(event_type: str) -> tuple[str, str | None]:
    if event_type == "division":
        return "division", None
    if event_type == "death":
        return "death", None
    if event_type.startswith("gain_"):
        return "ecDNA_gain", event_type.split("_", 1)[1]
    if event_type.startswith("loss_"):
        return "ecDNA_loss", event_type.split("_", 1)[1]
    return "cell_cycle_transition", None


def _condition_id(condition: str) -> str:
    if condition == "ctrl":
        return "CTRL"
    drug, dose = cfg.T87_CONDITION_TREATMENTS.get(condition, (condition, 0.0))
    dose_token = int(dose) if float(dose).is_integer() else dose
    if drug == "Palbociclib":
        return f"CDK4i_{dose_token}nM"
    if drug == "Ripretinib":
        return f"PDGFRAi_{dose_token}nM"
    return _safe_token(condition).upper()


def _condition_label(condition: str) -> str:
    if condition == "ctrl":
        return "Control"
    drug, dose = cfg.T87_CONDITION_TREATMENTS.get(condition, (condition, 0.0))
    dose_token = int(dose) if float(dose).is_integer() else dose
    target = "CDK4i" if drug == "Palbociclib" else "PDGFRAi" if drug == "Ripretinib" else drug
    return f"{target} {dose_token} nM"


def _treatment_family(condition: str) -> str:
    drug, _dose = cfg.T87_CONDITION_TREATMENTS.get(condition, ("vehicle", 0.0))
    if condition == "ctrl":
        return "control"
    if drug == "Palbociclib":
        return "cdk4_inhibitor"
    if drug == "Ripretinib":
        return "pdgfra_inhibitor"
    return _safe_token(drug).lower()


def _target_species(condition: str) -> str:
    family = _treatment_family(condition)
    if family == "cdk4_inhibitor":
        return "CDK4"
    if family == "pdgfra_inhibitor":
        return "PDGFRA"
    return "none"


def _compact_state(state_name: Any) -> str | None:
    if state_name is None or pd.isna(state_name):
        return None
    text = str(state_name)
    return {
        "NPC-like": "NPC",
        "OPC-like": "OPC",
        "AC-like": "AC",
        "MES-like": "MES",
    }.get(text, text.replace("-like", ""))


def _coarse_state(hard_state: Any) -> str | None:
    if hard_state in {"NPC", "OPC"}:
        return "OLIG2_high"
    if hard_state == "AC":
        return "AC_like"
    if hard_state == "MES":
        return "MES_like"
    if hard_state in {"AC_like", "MES_like", "OLIG2_high"}:
        return str(hard_state)
    return None


def _founder_map(parent_map: Mapping[int, int | None]) -> dict[int, int]:
    founders: dict[int, int] = {}
    for cell_id in parent_map:
        current = int(cell_id)
        seen = set()
        while True:
            parent = parent_map.get(current)
            if parent is None or parent in seen:
                founders[int(cell_id)] = current
                break
            seen.add(current)
            current = int(parent)
    return founders


def _lineage_depth(cell_id: int, parent_map: Mapping[int, int | None]) -> int:
    depth = 0
    current = int(cell_id)
    seen = set()
    while True:
        parent = parent_map.get(current)
        if parent is None or parent in seen:
            return depth
        seen.add(current)
        depth += 1
        current = int(parent)


def _cell_uid(ctx: ExportContext, cell_id: Any) -> str | None:
    parsed = _optional_int(cell_id)
    if parsed is None:
        return None
    return f"{ctx.sim_id}:cell_{parsed:08d}"


def _t_index(ctx: ExportContext, t: float) -> int:
    return int(ctx.t_index_by_key.get(_t_key(t), 0))


def _t_key(t: float) -> float:
    return round(float(t), 9)


def _format_t(t: float) -> str:
    return f"{float(t):.6g}"


def _sequence_value(values: Any, index: int, default: Any = np.nan) -> Any:
    try:
        if values is None or len(values) <= index:
            return default
        return values[index]
    except (TypeError, KeyError, IndexError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _numeric_or_nan(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return np.nan


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None


def _parameter_unit(block: str, name: str) -> str | None:
    if block in {"growth", "death", "turnover", "simulation"} or "rate" in name:
        return "per_t"
    return "dimensionless"


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean()) if len(frame) and column in frame else np.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").median()) if len(frame) and column in frame else np.nan


def _fraction(frame: pd.DataFrame, column: str, value: Any) -> float:
    return float((frame[column] == value).mean()) if len(frame) and column in frame else np.nan


def _bin_fractions(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {f"{bin_id}_fraction" if bin_id != "bin_gt120" else "copy_bin_gt120_fraction": np.nan for bin_id, *_ in COPY_BINS}
    row = {}
    for bin_id, lower, upper, _label in COPY_BINS:
        mask = values >= lower if upper is None else (values >= lower) & (values <= upper)
        column = f"copy_{bin_id}_fraction" if bin_id != "bin_gt120" else "copy_bin_gt120_fraction"
        row[column] = float(mask.mean())
    return {
        "copy_bin_0_fraction": row.get("copy_bin_0_fraction", row.get("copy_bin_0_fraction", np.nan)),
        "copy_bin_1_15_fraction": row.get("copy_bin_1_15_fraction", np.nan),
        "copy_bin_16_30_fraction": row.get("copy_bin_16_30_fraction", np.nan),
        "copy_bin_31_60_fraction": row.get("copy_bin_31_60_fraction", np.nan),
        "copy_bin_61_120_fraction": row.get("copy_bin_61_120_fraction", np.nan),
        "copy_bin_gt120_fraction": row.get("copy_bin_gt120_fraction", np.nan),
    }


def _initial_n_cells(snapshot: pd.DataFrame) -> int:
    if snapshot.empty:
        return 0
    first_t = snapshot["t"].min()
    return int((snapshot["t"] == first_t).sum())


def _dropna_ints(values: pd.Series) -> list[int]:
    return [int(value) for value in values.dropna().tolist()]


def _enforce_schema(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, list(columns)]


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "value"


def _replace_output_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, onexc=_chmod_and_retry)
    elif path.exists():
        path.unlink()


def _chmod_and_retry(function: Any, path: str, _excinfo: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _remove_legacy_simulation_outputs(run_dir: Path) -> None:
    for legacy_dir in (run_dir / "tables", run_dir / "simulation_data"):
        if legacy_dir.exists():
            _replace_output_path(legacy_dir)
    for directory in (run_dir,):
        for filename in LEGACY_SIMULATION_FILES:
            path = directory / filename
            if path.exists():
                _replace_output_path(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value
