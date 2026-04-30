"""Full conditional single-cell history reconstruction with particle scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_jsonl, write_markdown_report, write_table
from fit.objective import SCORE_COMPONENTS, score_particle_summary
from fit.scenarios import classify_scenarios
from fit.v4_lite import load_lite_artifacts


ALLOWED_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("NPC-like", "OPC-like"),
    ("OPC-like", "NPC-like"),
    ("OPC-like", "AC-like"),
    ("AC-like", "OPC-like"),
    ("AC-like", "MES-like"),
    ("MES-like", "AC-like"),
    ("NPC-like", "AC-like"),
    ("AC-like", "NPC-like"),
)


@dataclass
class ParticleResult:
    particle_id: int
    parameters: dict
    snapshots: pd.DataFrame
    features: pd.DataFrame
    events: pd.DataFrame
    histories: list[dict]
    score: dict


def create_full_initial_particles(
    lite_dir: str | Path,
    output_dir: str | Path,
    particles: int = 16,
    cells: int = 200,
    seed: int = 1,
) -> dict[str, Path]:
    artifacts = load_lite_artifacts(lite_dir)
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for particle_id in range(int(particles)):
        population = _sample_initial_population(artifacts["sampler"], int(cells), rng, particle_id)
        for cell in population:
            cell["particle_id"] = particle_id
            rows.append(cell)
    path = out / "initial_particles.parquet"
    write_table(pd.DataFrame(rows), path)
    write_json(
        out / "initial_particles_manifest.json",
        {"particles": int(particles), "cells_per_particle": int(cells), "source": str(Path(lite_dir))},
    )
    return {"initial_particles": path}


def run_full_reconstruction(
    lite_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
    particles: int = 32,
    cells: int = 300,
    seed: int = 1,
    acceptance_quantile: float = 0.5,
) -> dict[str, Path]:
    """Run a deterministic-seed SMC-ABC style particle reconstruction.

    Each particle stores an explicit representative single-cell history,
    summary features, score, posterior weight, and scenario label.
    """

    artifacts = load_lite_artifacts(lite_dir)
    obs_params = read_json(obs_params_path)
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    weeks = sorted(int(value) for value in artifacts["target"]["week"].dropna().unique())
    results: list[ParticleResult] = []
    for particle_id in range(int(particles)):
        params = _sample_particle_parameters(artifacts["prior_scales"], rng, particle_id)
        result = _simulate_and_score_particle(
            particle_id,
            params,
            artifacts,
            obs_params,
            weeks,
            int(cells),
            rng,
        )
        results.append(result)

    parameter_rows = []
    score_rows = []
    snapshot_rows = []
    feature_rows = []
    event_rows = []
    history_rows = []
    for result in results:
        parameter_rows.append(result.parameters)
        score_rows.append({"particle_id": result.particle_id, "score": result.score["score"], **result.score["contributions"]})
        snapshot_rows.append(result.snapshots)
        feature_rows.append(result.features.assign(particle_id=result.particle_id))
        event_rows.append(result.events)
        history_rows.extend(result.histories)

    scores = pd.DataFrame(score_rows)
    weights = _posterior_weights(scores)
    score_cutoff = float(scores["score"].quantile(float(acceptance_quantile)))
    weights["accepted"] = weights["score"] <= score_cutoff
    accepted_ids = set(weights.loc[weights["accepted"], "particle_id"].astype(int))

    snapshots = pd.concat(snapshot_rows, ignore_index=True)
    features = pd.concat(feature_rows, ignore_index=True)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    accepted_histories = [row for row in history_rows if int(row["particle_id"]) in accepted_ids]
    scenario_classes = classify_scenarios(events, snapshots, weights)

    write_jsonl(out / "accepted_histories.jsonl", accepted_histories)
    write_table(pd.DataFrame(parameter_rows), out / "particle_parameters.parquet")
    write_table(weights, out / "particle_weights.parquet")
    write_table(snapshots, out / "full_snapshot_summaries.parquet")
    write_table(events, out / "event_summaries.parquet")
    write_table(scenario_classes, out / "scenario_classes.parquet")
    write_table(features, out / "particle_summary_features.parquet")
    _write_ppc_report(out, weights, artifacts["target"], features)
    write_json(
        out / "full_reconstruction_manifest.json",
        {
            "schema_version": 1,
            "method_source": "markdown/fit_method.md",
            "mode": "conditional_single_cell_history_particle_ensemble",
            "particles": int(particles),
            "cells_per_particle": int(cells),
            "accepted_particles": sorted(int(pid) for pid in accepted_ids),
            "obs_params_locked": bool(obs_params.get("locked_for_full")),
        },
    )
    return {name: out / name for name in schemas.FULL_OUTPUTS}


def aggregate_accepted_histories(full_dir: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    base = Path(full_dir)
    weights = pd.read_parquet(base / "particle_weights.parquet")
    scenarios = pd.read_parquet(base / "scenario_classes.parquet")
    accepted = weights[weights["accepted"]].merge(scenarios, on="particle_id", how="left")
    summary = (
        accepted.groupby("scenario_class", as_index=False)
        .agg(posterior_weight=("weight", "sum"), particles=("particle_id", "nunique"), median_score=("score", "median"))
        .sort_values("posterior_weight", ascending=False)
    )
    if output_dir is not None:
        out = ensure_dir(output_dir)
        write_table(summary, out / "accepted_history_summary.parquet")
    return summary


def _simulate_and_score_particle(
    particle_id: int,
    params: dict,
    artifacts: dict,
    obs_params: dict,
    weeks: list[int],
    cells: int,
    rng: np.random.Generator,
) -> ParticleResult:
    del obs_params
    population = _sample_initial_population(artifacts["sampler"], cells, rng, particle_id)
    snapshots = []
    history_rows = []
    event_rows = []
    biology_penalty = 0.0
    last_week = weeks[0]
    for week in weeks:
        if week != last_week:
            population, interval_events, penalty = _advance_population(population, last_week, week, params, rng)
            biology_penalty += penalty
            event_rows.extend(interval_events)
        for cell in population:
            history_rows.append(_history_row(particle_id, week, cell))
        snapshots.append(_summarize_population(particle_id, week, population, artifacts["sampler"]))
        last_week = week
    snapshot_df = pd.concat(snapshots, ignore_index=True)
    features = _features_from_snapshots(snapshot_df, artifacts["target"], artifacts["sampler"])
    score = score_particle_summary(features, artifacts["target"], artifacts["distance_weights"], params, biology_penalty)
    events = _event_summary(particle_id, event_rows, weeks)
    return ParticleResult(particle_id, params, snapshot_df, features, events, history_rows, score)


def _sample_particle_parameters(prior_scales: dict, rng: np.random.Generator, particle_id: int) -> dict:
    transition = abs(rng.normal(0.06, float(prior_scales.get("state_transition_scale", 0.05))))
    gain = abs(rng.normal(0.04, 0.5 * float(prior_scales.get("copy_gain_scale", 0.1))))
    loss = abs(rng.normal(0.03, 0.5 * float(prior_scales.get("copy_loss_scale", 0.1))))
    return {
        "particle_id": int(particle_id),
        "state_transition_rate": float(min(0.5, transition)),
        "copy_gain_rate": float(min(0.5, gain)),
        "copy_loss_rate": float(min(0.5, loss)),
        "division_rate": float(min(0.2, abs(rng.normal(0.03, float(prior_scales.get("division_scale", 0.05)))))),
        "death_rate": float(min(0.2, abs(rng.normal(0.015, float(prior_scales.get("death_scale", 0.02)))))),
        "segregation_strength": float(min(1.0, abs(rng.normal(0.1, float(prior_scales.get("segregation_scale", 0.1)))))),
    }


def _sample_initial_population(sampler: dict, cells: int, rng: np.random.Generator, particle_id: int) -> list[dict]:
    states = list(sampler["states"])
    state_probs = np.asarray([sampler["state_probabilities"][state] for state in states], dtype=float)
    state_probs = schemas.normalize_probabilities(state_probs, name="state_probabilities")
    population = []
    for cell_id in range(int(cells)):
        state = str(rng.choice(states, p=state_probs))
        copies = {}
        for species in sampler["species"]:
            dist = sampler["state_species_copy_distributions"][state][species]
            labels = list(dist["bin_labels"])
            probs = schemas.normalize_probabilities(dist["probabilities"], name=f"{state}-{species}")
            label = str(rng.choice(labels, p=probs))
            copies[species] = _sample_copy_from_label(label, sampler["copy_number_bins"], rng)
        soft = np.full(len(states), 0.05 / (len(states) - 1), dtype=float)
        soft[states.index(state)] = 0.95
        population.append(
            {
                "particle_id": int(particle_id),
                "cell_id": int(cell_id),
                "parent_id": -1,
                "state_gate": state,
                "soft_state": dict(zip(states, soft.tolist())),
                "copies": copies,
                "alive": True,
            }
        )
    return population


def _sample_copy_from_label(label: str, bins: list[dict], rng: np.random.Generator) -> int:
    item = next(item for item in bins if str(item["label"]) == str(label))
    low = int(item["low"])
    high = item["high"]
    if high is None:
        return int(low + rng.geometric(0.35) - 1)
    if low == int(high):
        return low
    return int(rng.integers(low, int(high) + 1))


def _advance_population(population: list[dict], start_week: int, end_week: int, params: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict], float]:
    steps = max(1, int(end_week - start_week))
    events = []
    penalty = 0.0
    next_id = max(int(cell["cell_id"]) for cell in population) + 1 if population else 0
    for step in range(steps):
        current_week = start_week + step
        new_population = []
        for cell in population:
            if not cell.get("alive", True):
                continue
            if rng.random() < params["death_rate"]:
                events.append(_event_row(params["particle_id"], current_week, "death", None, 1))
                continue
            working = _copy_cell(cell)
            if rng.random() < params["state_transition_rate"]:
                old_state = working["state_gate"]
                candidates = [to_state for from_state, to_state in ALLOWED_TRANSITIONS if from_state == old_state]
                if candidates:
                    working["state_gate"] = str(rng.choice(candidates))
                    events.append(_event_row(params["particle_id"], current_week, "transition", None, 1, old_state, working["state_gate"]))
            for species in schemas.SPECIES:
                if rng.random() < params["copy_gain_rate"]:
                    increment = int(max(1, rng.poisson(2)))
                    working["copies"][species] += increment
                    events.append(_event_row(params["particle_id"], current_week, "gain", species, increment))
                if rng.random() < params["copy_loss_rate"]:
                    if working["copies"][species] > 0:
                        working["copies"][species] -= 1
                        events.append(_event_row(params["particle_id"], current_week, "loss", species, 1))
                    else:
                        penalty += 1.0
            new_population.append(working)
            if rng.random() < params["division_rate"]:
                daughter = _copy_cell(working)
                daughter["parent_id"] = int(working["cell_id"])
                daughter["cell_id"] = int(next_id)
                next_id += 1
                for species in schemas.SPECIES:
                    if rng.random() < params["segregation_strength"]:
                        delta = int(rng.choice([-1, 1]))
                        daughter["copies"][species] = max(0, daughter["copies"][species] + delta)
                new_population.append(daughter)
                events.append(_event_row(params["particle_id"], current_week, "division", None, 1))
        if not new_population:
            penalty += 100.0
            new_population = population[:1]
        if len(new_population) > 4 * max(1, len(population)):
            penalty += float(len(new_population))
            new_population = new_population[: 4 * max(1, len(population))]
        population = new_population
    return population, events, penalty


def _copy_cell(cell: dict) -> dict:
    return {
        "particle_id": int(cell["particle_id"]),
        "cell_id": int(cell["cell_id"]),
        "parent_id": int(cell.get("parent_id", -1)),
        "state_gate": str(cell["state_gate"]),
        "soft_state": dict(cell["soft_state"]),
        "copies": {species: int(cell["copies"][species]) for species in schemas.SPECIES},
        "alive": bool(cell.get("alive", True)),
    }


def _summarize_population(particle_id: int, week: int, population: list[dict], sampler: dict) -> pd.DataFrame:
    condition = "ctrl"
    replicate = "r1"
    rows = []
    pop_size = max(1, len(population))
    for state in schemas.STATE_NAMES:
        state_cells = [cell for cell in population if cell["state_gate"] == state]
        fraction = len(state_cells) / pop_size
        for species in schemas.SPECIES:
            values = np.asarray([cell["copies"][species] for cell in state_cells], dtype=float)
            if values.size == 0:
                values = np.asarray([0.0])
            bins = sampler["copy_number_bins"]
            labels = [schemas.assign_copy_bin(value, bins) for value in values]
            top_label = str(bins[-1]["label"])
            rows.append(
                {
                    "particle_id": particle_id,
                    "week": int(week),
                    "condition": condition,
                    "replicate": replicate,
                    "state_gate": state,
                    "species": species,
                    "flow_fraction": float(fraction),
                    "copy_mean": float(np.mean(values)),
                    "copy_variance": float(np.var(values)),
                    "zero_fraction": float(np.mean(values == 0)),
                    "tail_fraction": float(np.mean(np.asarray(labels) == top_label)),
                    "n_cells": int(len(state_cells)),
                    "population_size": int(pop_size),
                }
            )
    return pd.DataFrame(rows)


def _features_from_snapshots(snapshot: pd.DataFrame, target: pd.DataFrame, sampler: dict) -> pd.DataFrame:
    feature_rows = []
    snapshot_index = {
        (row.week, row.condition, row.replicate, row.state_gate, row.species): row
        for row in snapshot.itertuples(index=False)
    }
    for row in target.itertuples(index=False):
        channel = str(row.channel)
        variable = str(row.variable)
        if channel == "flow":
            matches = snapshot[
                (snapshot["week"] == row.week)
                & (snapshot["condition"] == row.condition)
                & (snapshot["replicate"] == row.replicate)
                & (snapshot["state_gate"] == row.state_gate)
            ]
            value = float(matches["flow_fraction"].iloc[0]) if not matches.empty else 0.0
        elif channel == "ectag":
            key = (row.week, row.condition, row.replicate, row.state_gate, row.species)
            snap = snapshot_index.get(key)
            if snap is None:
                value = 0.0
            else:
                values = _synthetic_values_for_feature(snap, sampler["copy_number_bins"], row.bin_label)
                value = float(values)
        elif channel in {"qpcdr", "ddpcr", "lite_summary"}:
            if channel == "ddpcr":
                value = _pooled_mean_from_snapshot(snapshot, row.week, row.condition, row.replicate, row.species)
            else:
                key = (row.week, row.condition, row.replicate, row.state_gate, row.species)
                snap = snapshot_index.get(key)
                value = float(getattr(snap, _snapshot_column(variable), 0.0)) if snap is not None else 0.0
        else:
            value = 0.0
        feature_rows.append({"feature_id": row.feature_id, "value": float(value)})
    return pd.DataFrame(feature_rows)


def _synthetic_values_for_feature(snap, bins: list[dict], target_label: str) -> float:
    if str(target_label) == "0":
        return float(snap.zero_fraction)
    if str(target_label) == str(bins[-1]["label"]):
        return float(snap.tail_fraction)
    # Distribute non-zero, non-tail mass across middle bins for summary-level particles.
    middle_bins = max(1, len(bins) - 2)
    middle_mass = max(0.0, 1.0 - float(snap.zero_fraction) - float(snap.tail_fraction))
    return float(middle_mass / middle_bins)


def _pooled_mean_from_snapshot(snapshot: pd.DataFrame, week, condition, replicate, species) -> float:
    subset = snapshot[
        (snapshot["week"] == week)
        & (snapshot["condition"] == condition)
        & (snapshot["replicate"] == replicate)
        & (snapshot["species"] == species)
    ]
    if subset.empty:
        return 0.0
    return float(np.sum(subset["flow_fraction"].astype(float) * subset["copy_mean"].astype(float)))


def _snapshot_column(variable: str) -> str:
    if variable == "state_species_mean":
        return "copy_mean"
    return variable


def _event_row(particle_id: int, week: int, event_type: str, species: str | None, count: int, from_state: str | None = None, to_state: str | None = None) -> dict:
    return {
        "particle_id": int(particle_id),
        "week": int(week),
        "event_type": event_type,
        "species": species or "",
        "count": int(count),
        "from_state": from_state or "",
        "to_state": to_state or "",
    }


def _event_summary(particle_id: int, event_rows: list[dict], weeks: list[int]) -> pd.DataFrame:
    if not event_rows:
        return pd.DataFrame([{"particle_id": particle_id, "week": week, "event_type": "none", "species": "", "count": 0} for week in weeks])
    rows = pd.DataFrame(event_rows)
    return rows.groupby(["particle_id", "week", "event_type", "species", "from_state", "to_state"], as_index=False)["count"].sum()


def _history_row(particle_id: int, week: int, cell: dict) -> dict:
    row = {
        "particle_id": int(particle_id),
        "week": int(week),
        "cell_id": int(cell["cell_id"]),
        "parent_id": int(cell.get("parent_id", -1)),
        "state_gate": str(cell["state_gate"]),
    }
    for species in schemas.SPECIES:
        row[f"K_{species}"] = int(cell["copies"][species])
    return row


def _posterior_weights(scores: pd.DataFrame) -> pd.DataFrame:
    shifted = scores["score"].astype(float) - float(scores["score"].min())
    raw = np.exp(-0.5 * shifted.to_numpy(dtype=float))
    total = float(np.sum(raw))
    weights = raw / total if total > 0.0 else np.ones(len(raw), dtype=float) / max(1, len(raw))
    result = scores.copy()
    result["weight"] = weights
    return result


def _write_ppc_report(out: Path, weights: pd.DataFrame, target: pd.DataFrame, features: pd.DataFrame) -> None:
    merged = features.merge(weights[["particle_id", "weight"]], on="particle_id", how="left")
    merged["weighted_value"] = merged["value"].astype(float) * merged["weight"].astype(float)
    weighted = merged.groupby("feature_id", as_index=False)["weighted_value"].sum().rename(columns={"weighted_value": "posterior_mean"})
    report = target[["feature_id", "channel", "target"]].merge(weighted, on="feature_id", how="left")
    report["abs_error"] = (report["posterior_mean"] - report["target"]).abs()
    channel = report.groupby("channel", as_index=False)["abs_error"].mean()
    write_table(channel, out / "full_ppc_channel_errors.parquet")
    write_json(
        out / "full_ppc_report.json",
        {
            "score_components": list(SCORE_COMPONENTS),
            "mean_abs_error_by_channel": dict(zip(channel["channel"], channel["abs_error"])),
            "history_ensemble_not_single_best_parameter": True,
        },
    )
    write_markdown_report(
        out / "full_ppc_report.md",
        "Full Posterior Predictive Check",
        [
            ("Scope", "Compared weighted particle summaries against raw/lite target features."),
            ("Channels", ", ".join(str(channel_name) for channel_name in channel["channel"])),
            ("Interpretation", "Accepted output is a weighted history ensemble, not a unique best-fit full parameter."),
        ],
    )
