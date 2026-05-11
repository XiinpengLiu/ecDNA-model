"""Fit pipeline public API aligned to ``markdown/fit_method.md``."""

from fit.final_report import build_final_report_layer, materialize_method_layout, validate_final_artifacts
from fit.full_smc import aggregate_accepted_histories, create_full_initial_particles, run_full_reconstruction, run_moment_prescreen
from fit.manifest import build_run_manifest, load_run_manifest
from fit.observation import calculate_ddpcr_pooled_mean, fit_observation_model, load_observation_params, validate_observation_params
from fit.parameter_registry import build_parameter_registry, run_prior_predictive_gate
from fit.raw import create_synthetic_raw_dataset, ingest_raw_data, load_clean_tables, load_raw_tables, standardize_raw_tables, validate_raw_tables
from fit.stage_runner import run_pipeline_from_raw
from fit.validation import build_validation_reports, validate_full_artifacts, validate_method_contracts
from fit.v4_lite import fit_v4_lite_summary_posterior, load_lite_artifacts, validate_lite_artifacts

__all__ = [
    "aggregate_accepted_histories",
    "build_final_report_layer",
    "build_parameter_registry",
    "build_run_manifest",
    "build_validation_reports",
    "calculate_ddpcr_pooled_mean",
    "create_full_initial_particles",
    "create_synthetic_raw_dataset",
    "fit_observation_model",
    "fit_v4_lite_summary_posterior",
    "ingest_raw_data",
    "load_clean_tables",
    "load_lite_artifacts",
    "load_observation_params",
    "load_raw_tables",
    "load_run_manifest",
    "materialize_method_layout",
    "run_full_reconstruction",
    "run_moment_prescreen",
    "run_pipeline_from_raw",
    "run_prior_predictive_gate",
    "standardize_raw_tables",
    "validate_full_artifacts",
    "validate_final_artifacts",
    "validate_lite_artifacts",
    "validate_method_contracts",
    "validate_observation_params",
    "validate_raw_tables",
]
