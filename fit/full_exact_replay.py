"""Exact v4 event-queue replay for accepted full particles."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.cell import Cell, CellPopulation
from core.simulation import HybridOgataSimulator, _clone_model_parameters_with_overrides
from fit import schemas
from fit.full_raw_ppc import generate_full_raw_table_ppc
from fit.full_smc import _event_summary, _features_from_snapshots, _history_row, _summarize_population
from fit.io_utils import ensure_dir, read_json, write_json, write_markdown_report, write_table
from fit.objective import score_particle_summary
from fit.v4_lite import load_lite_artifacts


EXACT_REPLAY_OUTPUTS: tuple[str, ...] = (
    "FULL_exact_replay_histories.parquet",
    "FULL_exact_replay_snapshot_summaries.parquet",
    "FULL_exact_replay_event_log.parquet",
    "FULL_exact_replay_event_summaries.parquet",
    "FULL_exact_replay_scores.parquet",
    "FULL_exact_replay_particle_weights.parquet",
    "FULL_exact_replay_report.md",
    "FULL_exact_replay_manifest.json",
)


def run_full_exact_replay(
    full_dir: str | Path,
    lite_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 1,
    acceptance_quantile: float = 0.5,
) -> dict[str, Path]:
    """Replay accepted full particles through the core exact event queue."""

    base_full = Path(full_dir)
    out = ensure_dir(base_full if output_dir is None else output_dir)
    artifacts = load_lite_artifacts(lite_dir)
    obs_params = read_json(obs_params_path)
    if not bool(obs_params.get("locked_for_full")):
        raise ValueError("Exact replay requires locked obs_params_for_full.json")

    histories = pd.read_parquet(base_full / "FULL_single_cell_history_samples.parquet")
    particle_params = pd.read_parquet(base_full / "FULL_particle_parameters.parquet")
    full_weights = pd.read_parquet(base_full / "FULL_particle_weights.parquet")
    accepted_weights = _accepted_weights(full_weights)
    histories = histories[histories["particle_id"].astype(int).isin(set(accepted_weights["particle_id"].astype(int)))].copy()
    if histories.empty:
        raise ValueError("Exact replay requires accepted histories from full reconstruction")

    rng = np.random.default_rng(seed)
    target_weeks = sorted(int(value) for value in artifacts["target"]["week"].dropna().unique())
    parameter_by_particle = {
        int(row.particle_id): row._asdict()
        for row in particle_params.itertuples(index=False)
        if int(row.particle_id) in set(accepted_weights["particle_id"].astype(int))
    }
    history_rows: list[dict] = []
    snapshot_frames: list[pd.DataFrame] = []
    event_log_frames: list[pd.DataFrame] = []
    event_summary_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    score_rows: list[dict] = []

    for particle_id, particle_history in histories.groupby("particle_id", sort=True):
        particle_id = int(particle_id)
        params = parameter_by_particle.get(particle_id, {"particle_id": particle_id})
        particle_snapshots: list[pd.DataFrame] = []
        particle_event_rows: list[dict] = []
        for (condition, replicate), stratum_history in particle_history.groupby(["condition", "replicate"], dropna=False):
            initial_week = int(stratum_history["week"].astype(int).min())
            initial = stratum_history[stratum_history["week"].astype(int) == initial_week].copy()
            replay = _simulate_exact_stratum(
                particle_id,
                str(condition),
                str(replicate),
                initial_week,
                target_weeks,
                initial,
                params,
                artifacts["sampler"],
                rng,
            )
            history_rows.extend(replay["histories"])
            particle_snapshots.append(replay["snapshots"])
            particle_event_rows.extend(replay["event_rows"])
            event_log_frames.append(replay["event_log"])
        particle_events = _event_summary(particle_id, particle_event_rows, target_weeks)
        particle_snapshot = pd.concat(particle_snapshots, ignore_index=True) if particle_snapshots else pd.DataFrame()
        particle_features = _features_from_snapshots(particle_snapshot, artifacts["target"], artifacts["sampler"], particle_events)
        particle_features["particle_id"] = particle_id
        score = score_particle_summary(particle_features, artifacts["target"], artifacts["distance_weights"], params)
        score_rows.append(
            {
                "particle_id": particle_id,
                "score": float(score["score"]),
                **score["contributions"],
                **_particle_coverage_columns(particle_features, artifacts["target"]),
            }
        )
        snapshot_frames.append(particle_snapshot)
        event_summary_frames.append(particle_events)
        feature_frames.append(particle_features)

    history_df = pd.DataFrame(history_rows)
    snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    event_log = pd.concat(event_log_frames, ignore_index=True) if event_log_frames else _empty_event_log()
    event_summaries = pd.concat(event_summary_frames, ignore_index=True) if event_summary_frames else pd.DataFrame()
    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    scores = pd.DataFrame(score_rows)
    exact_weights = _exact_replay_weights(scores, accepted_weights, float(acceptance_quantile))

    write_table(history_df, out / "FULL_exact_replay_histories.parquet")
    write_table(snapshots, out / "FULL_exact_replay_snapshot_summaries.parquet")
    write_table(event_log, out / "FULL_exact_replay_event_log.parquet")
    write_table(event_summaries, out / "FULL_exact_replay_event_summaries.parquet")
    write_table(features, out / "FULL_exact_replay_particle_features.parquet")
    write_table(scores, out / "FULL_exact_replay_scores.parquet")
    write_table(exact_weights, out / "FULL_exact_replay_particle_weights.parquet")
    generate_full_raw_table_ppc(out, obs_params_path, lite_dir, out, seed=seed)

    accepted_count = int(exact_weights["accepted"].astype(bool).sum())
    write_json(
        out / "FULL_exact_replay_manifest.json",
        {
            "schema_version": 1,
            "method_source": "markdown/fit_method.md",
            "history_source": str(base_full / "FULL_single_cell_history_samples.parquet"),
            "accepted_particle_controls_source": str(base_full / "FULL_particle_parameters.parquet"),
            "simulator": "core.simulation.HybridOgataSimulator",
            "event_log_policy": "division/death/ecDNA gain/ecDNA loss/cell-cycle events plus state drift checkpoints",
            "exact_replay_particles": int(exact_weights["particle_id"].nunique()),
            "exact_replay_accepted_particles": accepted_count,
            "acceptance_quantile": float(acceptance_quantile),
        },
    )
    write_markdown_report(
        out / "FULL_exact_replay_report.md",
        "Full Exact Replay Report",
        [
            ("Scope", "Accepted full particles were replayed with the core HybridOgataSimulator exact event queue."),
            ("Outputs", ", ".join(EXACT_REPLAY_OUTPUTS)),
            ("Selection", f"exact_replay_accepted_particles={accepted_count}; acceptance_quantile={float(acceptance_quantile):.3f}"),
            ("Event Log", "Discrete events and state drift checkpoints are written to FULL_exact_replay_event_log.parquet."),
        ],
    )
    return {name: out / name for name in EXACT_REPLAY_OUTPUTS}


def _simulate_exact_stratum(
    particle_id: int,
    condition: str,
    replicate: str,
    initial_week: int,
    target_weeks: list[int],
    initial: pd.DataFrame,
    particle_params: dict,
    sampler: dict,
    rng: np.random.Generator,
) -> dict:
    weeks = [week for week in target_weeks if week >= int(initial_week)]
    if not weeks:
        weeks = [int(initial_week)]
    relative_times = tuple(float(week - initial_week) for week in weeks)
    t_max = float(max(relative_times))
    replay_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    model_params = _particle_model_parameters(
        particle_params,
        t_max=t_max,
        n_init=len(initial),
        record_times=relative_times,
        seed=replay_seed,
    )
    event_seed, observation_seed = np.random.SeedSequence(replay_seed).spawn(2)
    population = _population_from_initial_history(initial, model_params, np.random.default_rng(event_seed))
    simulator = HybridOgataSimulator(
        params=model_params,
        observation_params=cfg.DEFAULT_OBSERVATION_PARAMETERS,
        seed=replay_seed,
        event_rng=np.random.default_rng(event_seed),
        observation_rng=np.random.default_rng(observation_seed),
    )
    result = simulator.simulate(population, verbose=False)
    histories: list[dict] = []
    snapshots: list[pd.DataFrame] = []
    event_rows: list[dict] = []
    state_checkpoint_rows: list[dict] = []
    previous_states: dict[int, str] = {}
    population_weight = _stratum_population_weight(initial)

    for relative_time, cell_snapshot in zip(result.times, result.cell_snapshots):
        absolute_week = int(round(float(relative_time) + int(initial_week)))
        fit_population = [
            _snapshot_cell_to_fit_dict(row, particle_id, condition, replicate, population_weight)
            for row in cell_snapshot
        ]
        for cell in fit_population:
            histories.append(_history_row(particle_id, absolute_week, cell))
        snapshots.append(_summarize_population(particle_id, absolute_week, fit_population, sampler, condition, replicate))
        checkpoint_events, previous_states = _state_checkpoint_events(
            particle_id,
            absolute_week,
            condition,
            replicate,
            fit_population,
            previous_states,
        )
        event_rows.extend(checkpoint_events)
        state_checkpoint_rows.extend(checkpoint_events)

    event_log = _event_log_dataframe(
        result.events,
        particle_id,
        initial_week,
        condition,
        replicate,
    )
    event_rows.extend(_event_summary_rows_from_log(event_log))
    if state_checkpoint_rows:
        event_log = pd.concat([event_log, pd.DataFrame(state_checkpoint_rows)], ignore_index=True, sort=False)
    return {
        "histories": histories,
        "snapshots": pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame(),
        "event_log": event_log if not event_log.empty else _empty_event_log(),
        "event_rows": event_rows,
    }


def _particle_model_parameters(
    particle_params: dict,
    *,
    t_max: float,
    n_init: int,
    record_times: tuple[float, ...],
    seed: int,
) -> cfg.ModelParameters:
    max_pop_size = int(max(1000, min(100_000, max(4, n_init) * 8)))
    params = _clone_model_parameters_with_overrides(
        cfg.DEFAULT_MODEL_PARAMETERS,
        t_max=t_max,
        n_init=n_init,
        record_times=record_times,
        record_interval=None,
        target_population_size=None,
        max_pop_size=max_pop_size,
        seed=seed,
    )
    params = _apply_particle_controls(params, particle_params)
    simulation = replace(
        params.simulation,
        fitting_mode=True,
        record_full_snapshots=True,
        record_events=True,
        max_cells_saved_per_snapshot=max_pop_size,
    )
    return replace(params, simulation=simulation)


def _apply_particle_controls(params: cfg.ModelParameters, particle_params: dict) -> cfg.ModelParameters:
    gain_factor = 1.0 + max(0.0, float(particle_params.get("copy_gain_rate", 0.0)))
    loss_factor = 1.0 + max(0.0, float(particle_params.get("copy_loss_rate", 0.0)))
    transition_factor = 1.0 + max(0.0, float(particle_params.get("state_transition_rate", 0.0)))
    division_factor = 1.0 + max(0.0, float(particle_params.get("division_rate", 0.0)))
    death_factor = 1.0 + max(0.0, float(particle_params.get("death_rate", 0.0)))
    segregation = float(np.clip(particle_params.get("segregation_strength", 0.0), 0.0, 1.0))

    turnover = {
        species: replace(item, gain_ceiling=float(item.gain_ceiling) * gain_factor, loss_ceiling=float(item.loss_ceiling) * loss_factor)
        for species, item in params.turnover.items()
    }
    hazard = replace(
        params.hazard,
        lambda_div_ceiling=float(params.hazard.lambda_div_ceiling) * division_factor,
        lambda_death_ceiling=float(params.hazard.lambda_death_ceiling) * death_factor,
    )
    generator = replace(
        params.generator,
        base_edges={edge: float(value) * transition_factor for edge, value in params.generator.base_edges.items()},
    )
    division = replace(
        params.division,
        rho_U=float(np.clip(params.division.rho_U + 0.1 * segregation, 0.0, 0.99)),
        rho_R=float(np.clip(params.division.rho_R + 0.1 * segregation, 0.0, 0.99)),
        rho_V=float(np.clip(params.division.rho_V + 0.1 * segregation, 0.0, 0.99)),
    )
    return replace(params, turnover=turnover, hazard=hazard, generator=generator, division=division)


def _population_from_initial_history(initial: pd.DataFrame, params: cfg.ModelParameters, rng: np.random.Generator) -> CellPopulation:
    population = CellPopulation(params, cfg.DEFAULT_INITIALIZATION_PARAMETERS, rng)
    cells = [_history_row_to_cell(row) for row in initial.itertuples(index=False)]
    population.cells = cells
    population.next_id = max((int(cell.cell_id) for cell in cells), default=-1) + 1
    return population


def _history_row_to_cell(row) -> Cell:
    soft = np.asarray([float(getattr(row, f"X_{state}", 0.0)) for state in schemas.STATE_NAMES], dtype=float)
    soft = schemas.normalize_probabilities(soft + 1e-9, name="exact replay initial soft state")
    latent = np.asarray([float(getattr(row, f"latent_U_{idx}", 0.0)) for idx in (1, 2, 3)], dtype=float)
    if latent.shape != (cfg.LATENT_DIM,) or not np.isfinite(latent).all():
        latent = cfg.ilr(soft)
    copies = np.asarray([int(max(0, getattr(row, f"K_{species}", 0))) for species in schemas.SPECIES], dtype=int)
    parent_id = int(getattr(row, "parent_id", -1))
    return Cell(
        cycle_state=int(cfg.CYCLE_INDEX.get(str(getattr(row, "cycle_state", "G1")), cfg.G1)),
        copy_numbers=copies,
        latent_state=latent,
        soft_state=soft,
        stress_score=float(getattr(row, "latent_R_raw", 0.0)) + 1.0,
        survival_score=float(getattr(row, "V", 0.0)),
        age=max(0.0, float(getattr(row, "age", 0.0))),
        cell_id=int(getattr(row, "cell_id", 0)),
        parent_id=None if parent_id < 0 else parent_id,
    )


def _snapshot_cell_to_fit_dict(row: dict, particle_id: int, condition: str, replicate: str, population_weight: float) -> dict:
    copies = {species: int(value) for species, value in zip(schemas.SPECIES, row["copy_numbers"])}
    soft = schemas.normalize_probabilities(np.asarray(row["soft_state"], dtype=float) + 1e-9, name="exact replay snapshot soft state")
    return {
        "particle_id": int(particle_id),
        "condition": condition,
        "replicate": replicate,
        "cell_id": int(row["cell_id"]),
        "parent_id": -1,
        "state_gate": str(row["dominant_state"]),
        "soft_state": dict(zip(schemas.STATE_NAMES, soft.tolist())),
        "latent_state": cfg.ilr(soft).tolist(),
        "copies": copies,
        "cycle_state": str(row["cycle_state"]),
        "age": float(row["age"]),
        "stress_score": float(row["stress_score"]),
        "survival_score": float(row["survival_score"]),
        "population_weight": float(population_weight),
        "alive": True,
    }


def _event_log_dataframe(
    events: list[tuple[float, str, int, dict]],
    particle_id: int,
    initial_week: int,
    condition: str,
    replicate: str,
) -> pd.DataFrame:
    rows = []
    for time, event_name, cell_id, details in events:
        absolute_time = float(time) + float(initial_week)
        summary = _classify_exact_event(str(event_name), details)
        rows.append(
            {
                "particle_id": int(particle_id),
                "time": absolute_time,
                "week": int(np.floor(absolute_time)),
                "condition": condition,
                "replicate": replicate,
                "cell_id": int(cell_id),
                "event_name": str(event_name),
                "event_type": summary["event_type"],
                "species": summary["species"],
                "from_state": summary["from_state"],
                "to_state": summary["to_state"],
                "count": int(summary["count"]),
                "details_json": json.dumps(_json_safe(details), sort_keys=True),
            }
        )
        rows.extend(_division_segregation_rows(particle_id, absolute_time, condition, replicate, int(cell_id), details))
    return pd.DataFrame(rows) if rows else _empty_event_log()


def _classify_exact_event(event_name: str, details: dict) -> dict:
    if event_name.startswith("gain_"):
        return {"event_type": "gain", "species": event_name.split("_", 1)[1], "from_state": "", "to_state": "", "count": 1}
    if event_name.startswith("loss_"):
        return {"event_type": "loss", "species": event_name.split("_", 1)[1], "from_state": "", "to_state": "", "count": 1}
    if event_name in {"G1_to_S", "G1_to_Q", "Q_to_G1", "S_to_G2M"}:
        source, target = event_name.split("_to_", 1)
        return {"event_type": "cycle_transition", "species": "", "from_state": source, "to_state": target, "count": 1}
    if event_name == "division":
        return {"event_type": "division", "species": "", "from_state": "", "to_state": "", "count": 1}
    if event_name == "death":
        state = str((details or {}).get("state_pre", {}).get("dominant_state", ""))
        return {"event_type": "death", "species": "", "from_state": state, "to_state": "", "count": 1}
    return {"event_type": event_name, "species": "", "from_state": "", "to_state": "", "count": 1}


def _division_segregation_rows(
    particle_id: int,
    absolute_time: float,
    condition: str,
    replicate: str,
    cell_id: int,
    details: dict,
) -> list[dict]:
    if not isinstance(details, dict) or "daughter_one" not in details or "daughter_two" not in details:
        return []
    one = details["daughter_one"].get("copy_numbers", [])
    two = details["daughter_two"].get("copy_numbers", [])
    if len(one) != len(schemas.SPECIES) or len(two) != len(schemas.SPECIES):
        return []
    rows = []
    for species, left, right in zip(schemas.SPECIES, one, two):
        delta = int(abs(int(left) - int(right)))
        if delta <= 0:
            continue
        rows.append(
            {
                "particle_id": int(particle_id),
                "time": float(absolute_time),
                "week": int(np.floor(float(absolute_time))),
                "condition": condition,
                "replicate": replicate,
                "cell_id": int(cell_id),
                "event_name": "segregation",
                "event_type": "segregation",
                "species": species,
                "from_state": "",
                "to_state": "",
                "count": delta,
                "details_json": json.dumps({"daughter_copy_delta": delta}, sort_keys=True),
            }
        )
    return rows


def _state_checkpoint_events(
    particle_id: int,
    week: int,
    condition: str,
    replicate: str,
    population: list[dict],
    previous_states: dict[int, str],
) -> tuple[list[dict], dict[int, str]]:
    rows = []
    current_states = {}
    for cell in population:
        cell_id = int(cell["cell_id"])
        state = str(cell["state_gate"])
        previous = previous_states.get(cell_id)
        current_states[cell_id] = state
        rows.append(
            {
                "particle_id": int(particle_id),
                "time": float(week),
                "week": int(week),
                "condition": condition,
                "replicate": replicate,
                "cell_id": cell_id,
                "event_name": "state_drift_checkpoint",
                "event_type": "state_checkpoint",
                "species": "",
                "from_state": previous or state,
                "to_state": state,
                "count": 1,
                "details_json": "{}",
            }
        )
        if previous is not None and previous != state:
            rows.append(
                {
                    "particle_id": int(particle_id),
                    "time": float(week),
                    "week": int(week),
                    "condition": condition,
                    "replicate": replicate,
                    "cell_id": cell_id,
                    "event_name": "state_drift_transition",
                    "event_type": "transition",
                    "species": "",
                    "from_state": previous,
                    "to_state": state,
                    "count": 1,
                    "details_json": "{}",
                }
            )
    return rows, current_states


def _event_summary_rows_from_log(event_log: pd.DataFrame) -> list[dict]:
    if event_log.empty:
        return []
    columns = ["particle_id", "week", "condition", "replicate", "event_type", "species", "from_state", "to_state", "count"]
    return event_log[columns].to_dict(orient="records")


def _exact_replay_weights(scores: pd.DataFrame, accepted_weights: pd.DataFrame, acceptance_quantile: float) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["particle_id", "score", "weight", "accepted"])
    merged = scores.merge(accepted_weights, on="particle_id", how="left")
    merged["accepted_weight"] = merged["accepted_weight"].fillna(0.0).astype(float)
    shifted = merged["score"].astype(float) - float(merged["score"].astype(float).min())
    raw = merged["accepted_weight"].to_numpy(dtype=float) * np.exp(-0.5 * shifted.to_numpy(dtype=float))
    if raw.sum() <= 0.0 or not np.isfinite(raw).all():
        raw = np.ones(len(merged), dtype=float)
    merged["weight"] = raw / raw.sum()
    cutoff = float(merged["score"].astype(float).quantile(float(np.clip(acceptance_quantile, 0.0, 1.0))))
    merged["accepted"] = merged["score"].astype(float) <= cutoff
    if not bool(merged["accepted"].any()):
        merged.loc[merged["score"].astype(float).idxmin(), "accepted"] = True
    columns = ["particle_id", "score", "weight", "accepted", "accepted_weight"]
    coverage_cols = [column for column in merged.columns if column.startswith("coverage_")]
    return merged[columns + coverage_cols].sort_values("particle_id").reset_index(drop=True)


def _particle_coverage_columns(features: pd.DataFrame, target: pd.DataFrame) -> dict:
    merged = target.merge(features[["feature_id", "value"]], on="feature_id", how="left")
    merged["value"] = merged["value"].fillna(0.0)
    merged["covered"] = (merged["value"].astype(float) - merged["target"].astype(float)).abs() <= 2.0 * np.sqrt(merged["variance"].astype(float).clip(lower=1e-9))
    return {f"coverage_{channel}": float(group["covered"].astype(float).mean()) for channel, group in merged.groupby("channel")}


def _accepted_weights(weights: pd.DataFrame) -> pd.DataFrame:
    accepted = weights[weights["accepted"].astype(bool)].copy() if "accepted" in weights else weights.copy()
    if accepted.empty:
        accepted = weights.copy()
    total = float(accepted["weight"].astype(float).sum())
    accepted["accepted_weight"] = accepted["weight"].astype(float) / total if total > 0.0 and np.isfinite(total) else 1.0 / max(1, len(accepted))
    return accepted[["particle_id", "accepted_weight"]].copy()


def _stratum_population_weight(initial: pd.DataFrame) -> float:
    if "population_weight" not in initial:
        return 1.0
    values = initial["population_weight"].astype(float)
    median = float(values.median()) if len(values) else 1.0
    return median if np.isfinite(median) and median > 0.0 else 1.0


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return copy.deepcopy(value)


def _empty_event_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "particle_id",
            "time",
            "week",
            "condition",
            "replicate",
            "cell_id",
            "event_name",
            "event_type",
            "species",
            "from_state",
            "to_state",
            "count",
            "details_json",
        ]
    )
