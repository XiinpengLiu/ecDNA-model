"""Shared schemas and deterministic helpers for the bulk-only fit pipeline.

The constants here define file contracts from ``markdown/fit_method.md``.
They are not experimental facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config as cfg

STATE_NAMES: tuple[str, ...] = tuple(cfg.STATE_NAMES)
SPECIES: tuple[str, ...] = tuple(cfg.SPECIES)
FLOW3_GROUPS: tuple[str, ...] = ("OLIG2-high", "AC", "MES")
PHASES: tuple[int, ...] = (1, 2, 3)

RAW_TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "ddpcr": ("week", "condition", "replicate", "species", "ddpcr_copy_number", "ddpcr_sd_or_ci", "batch_id"),
    "cell_count": ("week", "condition", "replicate", "total_cell_count", "viability", "batch_id"),
    "flow": ("week", "condition", "replicate", "group", "fraction", "batch_id"),
}

MANIFEST_OUTPUTS: tuple[str, ...] = (
    "run_manifest.json",
    "analysis_index.parquet",
    "available_data_mask.json",
)

CLEAN_OUTPUTS: tuple[str, ...] = (
    "ddpcr_long.parquet",
    "cell_count_long.parquet",
    "flow3_early_long.parquet",
    "drug_metadata_long.parquet",
    "qpcdr_unavailable.json",
    "ectag_unavailable.json",
    "clean_qc_report.md",
)

OBSERVATION_OUTPUTS: tuple[str, ...] = (
    "obs_params_for_bulk_lite.json",
    "obs_params_for_full.json",
    "flow3_projection_matrix.npy",
    "flow3_steady_target.json",
    "observation_qc_report.md",
)

LITE_OUTPUTS: tuple[str, ...] = (
    "BULK_LITE_final_fit.nc",
    "BULK_LITE_ddpcr_trajectories.parquet",
    "BULK_LITE_cell_count_trajectories.parquet",
    "BULK_LITE_growth_velocity.parquet",
    "BULK_LITE_copy_velocity.parquet",
    "BULK_LITE_flow3_steady.json",
    "BULK_LITE_hidden_4state_initializer.parquet",
    "BULK_LITE_initial_population_sampler.json",
    "BULK_LITE_to_FULL_prior_scales.json",
    "BULK_LITE_to_FULL_fit_mask.json",
    "BULK_LITE_unavailable_modalities.json",
    "BULK_LITE_ppc_report.pdf",
)

PARAMETER_REGISTRY_OUTPUTS: tuple[str, ...] = (
    "PARAMETER_registry_resolved.yaml",
    "PARAMETER_active_blocks.json",
    "PARAMETER_nuisance_blocks.json",
    "PARAMETER_hard_bounds.json",
    "PARAMETER_interpretability_prior_table.csv",
)

PRIOR_GATE_OUTPUTS: tuple[str, ...] = (
    "PRIOR_predictive_gate_report.pdf",
    "PRIOR_predictive_accepted_region.parquet",
    "PRIOR_predictive_rejection_reasons.csv",
)

MOMENT_OUTPUTS: tuple[str, ...] = (
    "MOMENT_candidate_parameters.parquet",
    "MOMENT_scores.parquet",
    "MOMENT_keep_top_particles.parquet",
    "MOMENT_prescreen_report.pdf",
)

FULL_INIT_OUTPUTS: tuple[str, ...] = (
    "FULL_initial_population.zarr",
    "FULL_initial_population_summary.parquet",
    "FULL_initial_parameter_particles.parquet",
    "FULL_initialization_report.pdf",
)

FULL_OUTPUTS: tuple[str, ...] = (
    "FULL_coarse_particles.parquet",
    "FULL_coarse_scores.parquet",
    "FULL_particles_final.zarr",
    "FULL_particle_parameters.parquet",
    "FULL_particle_weights.parquet",
    "FULL_particle_scores.parquet",
    "FULL_smc_adaptation_log.parquet",
    "FULL_early_rejection_log.parquet",
    "FULL_monte_carlo_noise_report.csv",
    "FULL_replay_histories.zarr",
)

VALIDATION_OUTPUTS: tuple[str, ...] = (
    "FULL_ddpcr_ppc.parquet",
    "FULL_cellcount_ppc.parquet",
    "FULL_flow3steady_ppc.parquet",
    "FULL_ppc_report.pdf",
    "FULL_identifiability_report.csv",
    "FULL_boundary_forcing_report.csv",
    "FULL_ridge_report.csv",
    "FULL_holdout_validation_report.pdf",
)

FINAL_OUTPUTS: tuple[str, ...] = (
    "FINAL_bulkfit_main_report.pdf",
    "FINAL_data_constrained_results.csv",
    "FINAL_latent_model_dependent_results.csv",
    "FINAL_parameter_interpretability_table.csv",
    "FINAL_scenario_summary.pdf",
    "FINAL_method_manifest.json",
    "FULL_latent_history_samples.zarr",
    "FULL_hidden_4state_summary.parquet",
    "FULL_hidden_copy_distribution_summary.parquet",
    "FULL_event_summary.parquet",
    "FULL_scenario_classes.parquet",
    "FULL_scenario_summary.pdf",
)

FIT_MASK: dict[str, bool] = {
    "use_ddpcr_bulk": True,
    "use_cell_count": True,
    "use_flow3_steady": True,
    "use_flow4": False,
    "use_qpcdr": False,
    "use_ectag": False,
    "use_state_specific_copy": False,
    "use_zero_tail_summary": False,
    "use_lite_summary_in_final_score": False,
}


@dataclass(frozen=True)
class ResultLayout:
    """Directory layout required by the method document."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "00_manifest"

    @property
    def clean_data(self) -> Path:
        return self.root / "01_clean_data"

    @property
    def observation(self) -> Path:
        return self.root / "02_observation_model"

    @property
    def lite(self) -> Path:
        return self.root / "03_v4_lite_bulk"

    @property
    def parameter_registry(self) -> Path:
        return self.root / "04_parameter_registry"

    @property
    def prior_predictive(self) -> Path:
        return self.root / "05_prior_predictive"

    @property
    def moment_prescreen(self) -> Path:
        return self.root / "06_moment_prescreen"

    @property
    def full_init(self) -> Path:
        return self.root / "07_full_initialization"

    @property
    def full_smc(self) -> Path:
        return self.root / "08_full_smc"

    @property
    def validation(self) -> Path:
        return self.root / "09_validation"

    @property
    def final_report(self) -> Path:
        return self.root / "10_final_report"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stable_feature_id(channel: str, **parts: Any) -> str:
    ordered = "|".join(f"{key}={parts[key]}" for key in sorted(parts) if parts[key] is not None)
    return f"{channel}|{ordered}" if ordered else channel


def phase_for_week(week: int | float) -> int:
    week_i = int(week)
    if week_i <= 3:
        return 1
    if week_i <= 6:
        return 2
    return 3


def validate_required_columns(columns: set[str], required: tuple[str, ...], table_name: str) -> None:
    missing = [column for column in required if column not in columns]
    require(not missing, f"{table_name} is missing required columns: {', '.join(missing)}")


def validate_species(values: Any, table_name: str) -> None:
    invalid = sorted(set(str(value) for value in values if str(value) not in SPECIES))
    require(not invalid, f"{table_name} contains invalid species values: {invalid}")


def validate_nonnegative(values: Any, field: str, table_name: str) -> None:
    arr = np.asarray(values, dtype=float)
    bad = arr[np.isfinite(arr) & (arr < 0)]
    require(bad.size == 0, f"{table_name}.{field} contains negative values")


def validate_weeks(values: Any, table_name: str) -> None:
    arr = np.asarray(values, dtype=float)
    require(bool(np.all(np.isfinite(arr))), f"{table_name}.week contains non-finite values")
    require(bool(np.all(arr >= 0)), f"{table_name}.week contains negative values")


def normalize_probabilities(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    require(arr.ndim == 1 and arr.size > 0, f"{name} must be a non-empty vector")
    require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    require(total > 0.0, f"{name} must have positive total mass")
    return arr / total


def softplus(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)
