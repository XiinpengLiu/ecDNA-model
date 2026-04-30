"""Fit pipeline public API aligned to markdown/fit_method.md."""

from fit.empirical import build_empirical_summaries
from fit.full_smc import aggregate_accepted_histories, create_full_initial_particles, run_full_reconstruction
from fit.observation import calculate_ddpcr_pooled_mean, fit_observation_model, load_observation_params, validate_observation_params
from fit.objective import score_particle_summary, score_particles_from_files
from fit.raw import create_synthetic_raw_dataset, ingest_raw_data, load_clean_tables, load_raw_tables, standardize_raw_tables, validate_raw_tables
from fit.scenarios import classify_scenarios, classify_scenarios_from_files
from fit.stage_runner import run_pipeline_from_raw
from fit.validation import validate_full_artifacts, validate_method_contracts
from fit.v4_lite import fit_v4_lite_summary_posterior, load_lite_artifacts, validate_lite_artifacts

__all__ = [
    "aggregate_accepted_histories",
    "build_empirical_summaries",
    "calculate_ddpcr_pooled_mean",
    "classify_scenarios",
    "classify_scenarios_from_files",
    "create_full_initial_particles",
    "create_synthetic_raw_dataset",
    "fit_observation_model",
    "fit_v4_lite_summary_posterior",
    "ingest_raw_data",
    "load_clean_tables",
    "load_lite_artifacts",
    "load_observation_params",
    "load_raw_tables",
    "run_full_reconstruction",
    "run_pipeline_from_raw",
    "score_particle_summary",
    "score_particles_from_files",
    "standardize_raw_tables",
    "validate_full_artifacts",
    "validate_lite_artifacts",
    "validate_method_contracts",
    "validate_observation_params",
    "validate_raw_tables",
]
