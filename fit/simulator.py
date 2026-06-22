"""Run the simulator per candidate and score against observed ddPCR.

The ABC distance is the root mean squared error between observed and simulated
ddPCR copy numbers after log2 transformation (fit_method.md equation for
D_ddPCR).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from core.simulation import SimulationResult, run_simulation  # noqa: E402

from .parameters import params_from_phi
from .proposal import Candidate, candidate_row


def build_simulation_params(
    base: cfg.ModelParameters,
    record_times: tuple[float, ...],
    *,
    n_init: int,
    target_population_size: int | None,
    max_pop_size: int,
    seed: int,
) -> cfg.ModelParameters:
    """Apply the fitting overrides to SimulationParameters.

    Per fit_method.md: candidates are simulated at the ddPCR observation times
    with free-running growth (target_population_size disabled) and
    fitting_mode=False so the run reaches t_max.
    """
    from dataclasses import replace

    simulation = replace(
        base.simulation,
        time_unit="t",
        t_max=float(max(record_times)),
        record_times=record_times,
        n_init=int(n_init),
        target_population_size=None if target_population_size is None else int(target_population_size),
        max_pop_size=int(max_pop_size),
        random_seed=int(seed),
        fitting_mode=False,
    )
    resolved = replace(base, simulation=simulation)
    cfg.validate_model_parameters(resolved)
    return resolved


def _prediction_value(result: SimulationResult, sim_time: float, species: str) -> float:
    """Simulated bulk copy mean at ``sim_time`` for one species.

    Uses the exact recorded time when present; on extinction at or before the
    requested time the copy number is zero; otherwise falls back to the nearest
    recorded time.
    """
    if not result.times or not result.bulk_copy_means:
        return 0.0
    species_idx = cfg.SPECIES_INDEX[species]
    times = np.asarray(result.times, dtype=float)
    exact = np.where(np.isclose(times, float(sim_time), rtol=0.0, atol=1e-8))[0]
    if exact.size:
        return float(np.asarray(result.bulk_copy_means[int(exact[-1])], dtype=float)[species_idx])
    if float(result.stop_time or 0.0) <= float(sim_time) and result.stop_reason == "population_extinction":
        return 0.0
    nearest_idx = int(np.argmin(np.abs(times - float(sim_time))))
    return float(np.asarray(result.bulk_copy_means[nearest_idx], dtype=float)[species_idx])


def _run_condition_predictions(
    *,
    condition: str,
    params: cfg.ModelParameters,
    targets: pd.DataFrame,
    raw_dir: Path,
    seed: int,
    rows_per_state: int,
    verbose: bool,
) -> pd.DataFrame:
    initialization = cfg.build_t87_initialization_parameters(
        condition,
        ddpcr_path=Path(raw_dir) / "ddpcr.csv",
        seed=int(seed),
        rows_per_state=int(rows_per_state),
    )
    result = run_simulation(
        params=params,
        initialization=initialization,
        input_schedules=cfg.t87_input_schedules_for_condition(condition),
        seed=int(seed),
        verbose=verbose,
    )
    rows = []
    for target in targets.itertuples(index=False):
        sim = _prediction_value(result, float(target.sim_time), str(target.species))
        obs = float(target.ddpcr_obs)
        log2_obs = float(np.log2(obs + 1.0))
        log2_sim = float(np.log2(max(0.0, sim) + 1.0))
        rows.append(
            {
                "condition": condition,
                "week": int(target.week),
                "day": float(target.day),
                "sim_time": float(target.sim_time),
                "species": str(target.species),
                "ddpcr_obs": obs,
                "ddpcr_sim": float(sim),
                "log2_obs": log2_obs,
                "log2_sim": log2_sim,
                "residual": log2_sim - log2_obs,
                "stop_reason": result.stop_reason,
                "stop_time": float(result.stop_time) if result.stop_time is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate(
    *,
    candidate: Candidate,
    base_params: cfg.ModelParameters,
    phi0: np.ndarray,
    targets: pd.DataFrame,
    raw_dir: Path,
    rows_per_state: int,
    condition_offsets: dict[str, int],
    verbose: bool,
) -> tuple[dict, pd.DataFrame]:
    """Decode phi, run all conditions, and return (candidate_row, predictions).

    The candidate row is ``candidate_row`` enriched with ``rmse_ddpcr``.
    Predictions carry the per-point log2 residuals used by the ABC distance.
    """
    params = params_from_phi(base_params, candidate.phi)
    parts = []
    for condition, group in targets.groupby("condition", sort=False):
        condition_seed = int(candidate.seed) + int(condition_offsets[str(condition)])
        parts.append(
            _run_condition_predictions(
                condition=str(condition),
                params=params,
                targets=group,
                raw_dir=raw_dir,
                seed=condition_seed,
                rows_per_state=rows_per_state,
                verbose=verbose,
            )
        )
    predictions = pd.concat(parts, ignore_index=True)
    # D_ddPCR = sqrt(mean(residual^2)) over all matched (condition, time, species) points.
    rmse = float(np.sqrt(np.mean(np.square(predictions["residual"].astype(float)))))

    row = candidate_row(candidate, phi0)
    row["rmse_ddpcr"] = rmse

    predictions.insert(0, "global_id", int(candidate.global_id))
    predictions.insert(0, "candidate_id", int(candidate.candidate_id))
    predictions.insert(0, "generation", int(candidate.generation))
    predictions.insert(3, "accepted", False)
    return row, predictions
