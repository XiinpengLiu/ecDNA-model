"""Stage orchestration helpers for workflow systems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fit.empirical import build_empirical_summaries
from fit.full_smc import create_full_initial_particles, run_full_reconstruction
from fit.observation import fit_observation_model
from fit.raw import ingest_raw_data
from fit.schemas import ResultLayout
from fit.v4_lite import fit_v4_lite_summary_posterior


@dataclass(frozen=True)
class SmokeRunResult:
    output_root: Path
    observation_dir: Path
    empirical_dir: Path
    lite_dir: Path
    full_dir: Path


def run_pipeline_from_raw(
    raw_dir: str | Path,
    output_root: str | Path,
    seed: int = 1,
    posterior_draws: int = 64,
    particles: int = 32,
    cells: int = 300,
) -> SmokeRunResult:
    layout = ResultLayout(Path(output_root))
    ingest_raw_data(raw_dir, layout.clean_data)
    fit_observation_model(layout.clean_data, layout.observation, seed=seed)
    build_empirical_summaries(layout.clean_data, layout.observation / "obs_params_for_lite.json", layout.empirical)
    fit_v4_lite_summary_posterior(
        layout.empirical,
        layout.observation / "obs_params_for_lite.json",
        layout.lite,
        seed=seed,
        posterior_draws=posterior_draws,
    )
    create_full_initial_particles(layout.lite, layout.full_init, particles=particles, cells=cells, seed=seed)
    run_full_reconstruction(
        layout.lite,
        layout.observation / "obs_params_for_full.json",
        layout.full_smc,
        particles=particles,
        cells=cells,
        seed=seed,
    )
    return SmokeRunResult(Path(output_root), layout.observation, layout.empirical, layout.lite, layout.full_smc)


__all__ = ["run_pipeline_from_raw"]
