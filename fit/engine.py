"""Config-centered local ABC-SMC fit driver (fit_method.md Steps 0-4).

Runs ``generations`` SMC generations of ``n_per_generation`` candidates each,
retains the top ``accepted_count`` by ddPCR log2-RMSE per generation, and treats
the final generation's accepted set as the empirical ABC posterior.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as cfg  # noqa: E402

from .io_utils import ensure_dir
from .outputs import (
    accepted_particles_frame,
    generation_summary_row,
    write_all_outputs,
    write_baseline_prediction,
    write_fit_config,
)
from .parameters import reference_params, reference_phi
from .proposal import Candidate, generate_candidates
from .simulator import build_simulation_params, evaluate_candidate
from .targets import load_ddpcr_targets, record_times_from_targets


@dataclass
class FitConfig:
    """All knobs for one local ABC-SMC fit run."""

    raw_dir: Path
    output_dir: Path
    conditions: tuple[str, ...]
    generations: int = 4
    n_per_generation: int = 50
    accepted_count: int = 20
    seed: int | None = None
    n_init: int | None = None
    rows_per_state: int = 512
    target_population_size: int | None = None
    max_pop_size: int | None = None
    verbose: bool = False


def _rank_generation(candidate_frame: pd.DataFrame, accepted_count: int) -> pd.DataFrame:
    """Rank primary by rmse_ddpcr, secondary by distance_to_config (fit_method.md 70-77).

    The generation threshold epsilon_g is the largest ddPCR distance among the
    accepted particles.
    """
    ranked = candidate_frame.sort_values(["rmse_ddpcr", "distance_to_config", "global_id"]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    cutoff = min(int(accepted_count), len(ranked))
    epsilon = float(ranked.iloc[cutoff - 1]["rmse_ddpcr"]) if cutoff >= 1 else float("nan")
    ranked["epsilon_generation"] = epsilon
    ranked["accepted"] = ranked["rank"] <= int(accepted_count)
    return ranked


def _evaluate_all(candidates, **kwargs) -> tuple[list[dict], list[pd.DataFrame]]:
    """Evaluate every candidate sequentially and collect rows + prediction frames."""
    rows = []
    prediction_parts = []
    for candidate in candidates:
        row, predictions = evaluate_candidate(candidate=candidate, **kwargs)
        rows.append(row)
        prediction_parts.append(predictions)
    return rows, prediction_parts


def _run_baseline(
    output_dir: Path,
    *,
    base_params: cfg.ModelParameters,
    phi0: np.ndarray,
    targets: pd.DataFrame,
    raw_dir: Path,
    rows_per_state: int,
    condition_offsets: dict[str, int],
    verbose: bool,
) -> None:
    """Step 0 anchor: simulate the reference parameterization and write baseline outputs."""
    anchor = Candidate(
        generation=-1,
        candidate_id=0,
        global_id=-1,
        proposal_type="anchor",
        parent_generation=-1,
        parent_candidate_id=-1,
        seed=int(base_params.simulation.random_seed),
        phi=phi0.copy(),
    )
    row, prediction = evaluate_candidate(
        candidate=anchor,
        base_params=base_params,
        phi0=phi0,
        targets=targets,
        raw_dir=raw_dir,
        rows_per_state=rows_per_state,
        condition_offsets=condition_offsets,
        verbose=verbose,
    )
    write_baseline_prediction(output_dir, prediction, float(row["rmse_ddpcr"]))


def run_local_abc_fit(config: FitConfig) -> Path:
    """Run the full monolithic local ABC-SMC fit and write all nine outputs."""
    output_dir = ensure_dir(Path(config.output_dir))
    raw_dir = Path(config.raw_dir)

    root_params = reference_params()
    targets = load_ddpcr_targets(raw_dir, config.conditions)
    cfg.require(not targets.empty, "No ddPCR targets were loaded.")
    record_times = record_times_from_targets(targets)

    seed = int(config.seed) if config.seed is not None else int(root_params.simulation.random_seed)
    n_init = int(config.n_init) if config.n_init is not None else int(root_params.simulation.n_init)
    target_pop = config.target_population_size
    max_pop = int(config.max_pop_size) if config.max_pop_size is not None else int(root_params.simulation.max_pop_size)

    accepted_count = min(int(config.accepted_count), int(config.n_per_generation))
    cfg.require(accepted_count >= 1, "accepted must be at least 1.")

    base_params = build_simulation_params(
        root_params,
        record_times,
        n_init=n_init,
        target_population_size=target_pop,
        max_pop_size=max_pop,
        seed=seed,
    )
    phi0 = reference_phi()
    condition_offsets = {condition: 100_003 * idx for idx, condition in enumerate(config.conditions)}
    rng = np.random.default_rng(seed)

    write_fit_config(
        output_dir,
        conditions=config.conditions,
        generations=config.generations,
        n_per_generation=config.n_per_generation,
        accepted_count=config.accepted_count,
        seed=seed,
        n_init=n_init,
        rows_per_state=int(config.rows_per_state),
        target_population_size=target_pop,
        max_pop_size=max_pop,
        targets=targets,
    )
    _run_baseline(
        output_dir,
        base_params=base_params,
        phi0=phi0,
        targets=targets,
        raw_dir=raw_dir,
        rows_per_state=int(config.rows_per_state),
        condition_offsets=condition_offsets,
        verbose=bool(config.verbose),
    )

    eval_kwargs = dict(
        base_params=base_params,
        phi0=phi0,
        targets=targets,
        raw_dir=raw_dir,
        rows_per_state=int(config.rows_per_state),
        condition_offsets=condition_offsets,
        verbose=bool(config.verbose),
    )

    all_candidate_frames = []
    all_prediction_frames = []
    accepted_frames = []
    summary_rows = []
    previous_accepted: pd.DataFrame | None = None
    global_start = 0
    for generation in range(int(config.generations)):
        print(f"Generation {generation}: evaluating {int(config.n_per_generation)} candidates...")
        candidates = generate_candidates(
            generation=generation,
            global_start=global_start,
            n_per_generation=int(config.n_per_generation),
            base_seed=seed,
            rng=rng,
            phi0=phi0,
            previous_accepted=previous_accepted,
        )
        rows, prediction_parts = _evaluate_all(candidates, **eval_kwargs)
        ranked = _rank_generation(pd.DataFrame(rows), accepted_count)
        accepted = accepted_particles_frame(ranked)

        prediction_frame = pd.concat(prediction_parts, ignore_index=True)
        accepted_ids = set(ranked.loc[ranked["accepted"], "global_id"].astype(int))
        prediction_frame["accepted"] = prediction_frame["global_id"].astype(int).isin(accepted_ids)

        all_candidate_frames.append(ranked)
        all_prediction_frames.append(prediction_frame)
        accepted_frames.append(accepted)
        summary_rows.append(generation_summary_row(ranked))
        previous_accepted = accepted
        global_start += int(config.n_per_generation)
        print(
            f"Generation {generation}: epsilon={float(accepted['rmse_ddpcr'].max()):.4g}, "
            f"best_rmse={float(ranked['rmse_ddpcr'].min()):.4g}, "
            f"accepted_anchor={bool((accepted['proposal_type'] == 'anchor').any())}"
        )

    candidates_all = pd.concat(all_candidate_frames, ignore_index=True)
    accepted_all = pd.concat(accepted_frames, ignore_index=True)
    predictions_all = pd.concat(all_prediction_frames, ignore_index=True)
    generation_summary = pd.DataFrame(summary_rows)
    final_generation = int(config.generations) - 1
    final_accepted = accepted_all[accepted_all["generation"] == final_generation].copy()
    cfg.require(not final_accepted.empty, f"Generation {final_generation} accepted particles are missing.")
    final_ids = set(final_accepted["global_id"].astype(int))
    final_predictions = predictions_all[predictions_all["global_id"].astype(int).isin(final_ids)].copy()

    write_all_outputs(
        output_dir,
        candidates_all=candidates_all,
        accepted_all=accepted_all,
        predictions_all=predictions_all,
        generation_summary=generation_summary,
        final_accepted=final_accepted,
        final_predictions=final_predictions,
        phi0=phi0,
    )

    from .outputs import validate_outputs

    validate_outputs(output_dir)
    print(f"Local ABC-SMC fit outputs written to: {Path(output_dir).resolve()}")
    return Path(output_dir)
