"""Stage orchestration helpers for the method-defined 00-10 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fit.final_report import build_final_report_layer
from fit.full_smc import METHOD_MOMENT_MIN_TOP, create_full_initial_particles, run_full_reconstruction, run_moment_prescreen
from fit.manifest import build_run_manifest
from fit.observation import fit_observation_model
from fit.parameter_registry import build_parameter_registry, run_prior_predictive_gate
from fit.raw import ingest_raw_data
from fit.schemas import ResultLayout
from fit.validation import build_validation_reports
from fit.v4_lite import fit_v4_lite_summary_posterior


def _progress(message: str) -> None:
    print(f"[fit] {message}", flush=True)


@dataclass(frozen=True)
class SmokeRunResult:
    output_root: Path
    observation_dir: Path
    lite_dir: Path
    parameter_registry_dir: Path
    prior_predictive_dir: Path
    moment_prescreen_dir: Path
    full_init_dir: Path
    full_dir: Path
    validation_dir: Path
    final_dir: Path


def run_pipeline_from_raw(
    raw_dir: str | Path,
    output_root: str | Path,
    seed: int = 1,
    posterior_draws: int = 64,
    particles: int = 3000,
    cells: int = 10000,
    workers: int = 1,
) -> SmokeRunResult:
    layout = ResultLayout(Path(output_root))
    _progress(
        f"run-all start: raw_dir={Path(raw_dir)}, output={layout.root}, "
        f"seed={seed}, particles={particles}, cells={cells}, workers={workers}"
    )
    _progress("stage 00 manifest start")
    build_run_manifest(raw_dir, layout.manifest)
    _progress(f"stage 00 manifest done: output={layout.manifest}")
    _progress("stage 01 clean data start")
    ingest_raw_data(raw_dir, layout.clean_data)
    _progress(f"stage 01 clean data done: output={layout.clean_data}")
    _progress("stage 02 observation model start")
    fit_observation_model(layout.clean_data, layout.observation, seed=seed)
    _progress(f"stage 02 observation model done: output={layout.observation}")
    _progress("stage 03 bulk-lite summaries start")
    fit_v4_lite_summary_posterior(layout.clean_data, layout.observation / "obs_params_for_bulk_lite.json", layout.lite, seed=seed, posterior_draws=posterior_draws)
    _progress(f"stage 03 bulk-lite summaries done: output={layout.lite}")
    _progress("stage 04 parameter registry start")
    build_parameter_registry(layout.lite, layout.parameter_registry)
    _progress(f"stage 04 parameter registry done: output={layout.parameter_registry}")
    _progress("stage 05 prior predictive gate start")
    run_prior_predictive_gate(layout.parameter_registry, layout.lite, layout.observation / "obs_params_for_full.json", layout.prior_predictive, seed=seed)
    _progress(f"stage 05 prior predictive gate done: output={layout.prior_predictive}")
    _progress("stage 06 moment prescreen start")
    run_moment_prescreen(
        layout.lite,
        layout.prior_predictive,
        layout.moment_prescreen,
        seed=seed,
        n_candidates=max(METHOD_MOMENT_MIN_TOP, particles * 100),
        keep_top=max(METHOD_MOMENT_MIN_TOP, particles * 20),
        workers=workers,
    )
    _progress(f"stage 06 moment prescreen done: output={layout.moment_prescreen}")
    _progress("stage 07 full initialization start")
    create_full_initial_particles(layout.lite, layout.full_init, particles=particles, cells=cells, seed=seed, moment_dir=layout.moment_prescreen)
    _progress(f"stage 07 full initialization done: output={layout.full_init}")
    _progress("stage 08 full SMC start")
    run_full_reconstruction(layout.lite, layout.observation / "obs_params_for_full.json", layout.full_smc, particles=particles, cells=cells, seed=seed, smc_steps=4, moment_dir=layout.moment_prescreen, workers=workers)
    _progress(f"stage 08 full SMC done: output={layout.full_smc}")
    _progress("stage 09 validation start")
    build_validation_reports(layout.lite, layout.full_smc, layout.parameter_registry, layout.validation, layout.observation / "obs_params_for_full.json")
    _progress(f"stage 09 validation done: output={layout.validation}")
    _progress("stage 10 final report start")
    build_final_report_layer(layout.observation, layout.lite, layout.full_smc, layout.final_report, validation_dir=layout.validation)
    _progress(f"stage 10 final report done: output={layout.final_report}")
    _progress(f"run-all done: output={layout.root}")
    return SmokeRunResult(
        Path(output_root),
        layout.observation,
        layout.lite,
        layout.parameter_registry,
        layout.prior_predictive,
        layout.moment_prescreen,
        layout.full_init,
        layout.full_smc,
        layout.validation,
        layout.final_report,
    )


__all__ = ["run_pipeline_from_raw"]
