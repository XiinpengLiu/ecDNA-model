"""Full conditional single-cell history reconstruction with particle scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core import dynamics as dyn
from core.cell import Cell
from core.simulation import HybridOgataSimulator
from fit import schemas
from fit.full_raw_ppc import generate_full_raw_table_ppc
from fit.io_utils import ensure_dir, read_json, write_json, write_jsonl, write_markdown_report, write_table, write_text_pdf
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

METHOD_N_SIM_FIT = 10_000
METHOD_N_SIM_REPLAY = 50_000


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
    cells: int = METHOD_N_SIM_FIT,
    seed: int = 1,
) -> dict[str, Path]:
    artifacts = load_lite_artifacts(lite_dir)
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for particle_id in range(int(particles)):
        simulator = _make_v4_simulator(rng)
        for stratum_index, (condition, replicate) in enumerate(_sampler_strata(artifacts["sampler"])):
            population_weight = _initial_population_weight(artifacts["target"], condition, replicate, artifacts["sampler"]["initial_week"], int(cells))
            population = _sample_initial_population(
                artifacts["sampler"],
                int(cells),
                rng,
                particle_id,
                condition,
                replicate,
                stratum_index * int(cells),
                population_weight=population_weight,
                simulator=simulator,
            )
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
    cells: int = METHOD_N_SIM_FIT,
    seed: int = 1,
    acceptance_quantile: float = 0.5,
    smc_steps: int = 3,
) -> dict[str, Path]:
    """Run a deterministic-seed SMC-ABC style particle reconstruction.

    Each particle stores an explicit representative single-cell history,
    summary features, score, posterior weight, and scenario label.
    """

    artifacts = load_lite_artifacts(lite_dir)
    obs_params = read_json(obs_params_path)
    if not bool(obs_params.get("locked_for_full")):
        raise ValueError("Full reconstruction requires locked obs_params_for_full.json")
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    weeks = sorted(int(value) for value in artifacts["target"]["week"].dropna().unique())
    results: list[ParticleResult] = []
    proposal_bank: list[dict] = []
    for smc_round in range(max(1, int(smc_steps))):
        round_results: list[ParticleResult] = []
        for local_id in range(int(particles)):
            particle_id = smc_round * int(particles) + local_id
            params = (
                _sample_particle_parameters(artifacts["prior_scales"], rng, particle_id)
                if not proposal_bank
                else _perturb_particle_parameters(proposal_bank[int(rng.integers(0, len(proposal_bank)))], rng, particle_id)
            )
            result = _simulate_and_score_particle(
                particle_id,
                params,
                artifacts,
                obs_params,
                weeks,
                int(cells),
                rng,
            )
            result.parameters["smc_round"] = int(smc_round)
            round_results.append(result)
        scores = pd.DataFrame({"particle_id": [item.particle_id for item in round_results], "score": [item.score["score"] for item in round_results]})
        tolerance = float(scores["score"].quantile(max(0.1, 0.6 - 0.15 * smc_round)))
        accepted = [item for item in round_results if item.score["score"] <= tolerance]
        if not accepted:
            accepted = [min(round_results, key=lambda item: item.score["score"])]
        proposal_bank = [dict(item.parameters) for item in accepted]
        for item in round_results:
            item.parameters["smc_tolerance"] = tolerance
        results.extend(round_results)

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
    final_round = int(weights["particle_id"].max() // max(1, int(particles)))
    final_scores = scores[scores["particle_id"] >= final_round * int(particles)]
    score_cutoff = float(final_scores["score"].quantile(float(acceptance_quantile)))
    weights["accepted"] = (weights["particle_id"] >= final_round * int(particles)) & (weights["score"] <= score_cutoff)
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
    _write_method_full_outputs(out, accepted_histories, parameter_rows, weights, snapshots, events, features, artifacts["target"])
    ppc_payload = _write_ppc_report(out, weights, artifacts["target"], features)
    generate_full_raw_table_ppc(out, obs_params_path, lite_dir, out, seed=seed)
    full_diagnostics = _full_continue_diagnostics(weights, ppc_payload, scenario_classes)
    if not full_diagnostics["continue_gate_passed"]:
        _write_incompatibility_report(out, full_diagnostics)
    write_json(
        out / "full_reconstruction_manifest.json",
        {
            "schema_version": 1,
            "method_source": "markdown/fit_method.md",
            "mode": "conditional_single_cell_history_particle_ensemble",
            "particles": int(particles),
            "cells_per_particle": int(cells),
            "method_n_sim_fit": METHOD_N_SIM_FIT,
            "method_n_sim_replay": METHOD_N_SIM_REPLAY,
            "representative_cell_weighting": "simulated cells carry population_weight when real cell count exceeds N_sim_fit",
            "smc_steps": int(smc_steps),
            "accepted_particles": sorted(int(pid) for pid in accepted_ids),
            "obs_params_locked": bool(obs_params.get("locked_for_full")),
            "full_v4_chain_policy": "cell X/U/R/V and event-rate proposals are refreshed through HybridOgataSimulator/core.dynamics",
            "full_continue_diagnostics": full_diagnostics,
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


def _sampler_strata(sampler: dict) -> list[tuple[str, str]]:
    keys = sampler.get("state_probabilities_by_stratum", {})
    if not keys:
        return [("ctrl", "r1")]
    strata = []
    for key in sorted(keys):
        condition, replicate = str(key).split("|", 1)
        strata.append((condition, replicate))
    return strata


def _target_strata(target: pd.DataFrame, sampler: dict) -> list[tuple[str, str]]:
    if {"condition", "replicate"}.issubset(target.columns):
        subset = target[["condition", "replicate"]].dropna().drop_duplicates()
        if not subset.empty:
            return [(str(row.condition), str(row.replicate)) for row in subset.itertuples(index=False)]
    return _sampler_strata(sampler)


def _initial_population_weight(target: pd.DataFrame, condition: str, replicate: str, initial_week: int, cells: int) -> float:
    if "channel" not in target:
        return 1.0
    subset = target[
        (target["channel"] == "cell_count")
        & (target["variable"] == "total_cell_count")
        & (target["week"] == initial_week)
        & (target["condition"].astype(str) == str(condition))
        & (target["replicate"].astype(str) == str(replicate))
    ]
    if subset.empty:
        return 1.0
    total = float(subset["target"].astype(float).median())
    return float(max(1.0, total / max(1, int(cells))))


def _make_v4_simulator(rng: np.random.Generator) -> HybridOgataSimulator:
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    event_seed, observation_seed = np.random.SeedSequence(seed).spawn(2)
    return HybridOgataSimulator(
        params=cfg.DEFAULT_MODEL_PARAMETERS,
        observation_params=cfg.DEFAULT_OBSERVATION_PARAMETERS,
        seed=seed,
        event_rng=np.random.default_rng(event_seed),
        observation_rng=np.random.default_rng(observation_seed),
    )


def _simulate_and_score_particle(
    particle_id: int,
    params: dict,
    artifacts: dict,
    obs_params: dict,
    weeks: list[int],
    cells: int,
    rng: np.random.Generator,
) -> ParticleResult:
    if not bool(obs_params.get("locked_for_full")):
        raise ValueError("Full particle scoring requires locked observation parameters")
    snapshots = []
    history_rows = []
    event_rows = []
    biology_penalty = 0.0
    simulator = _make_v4_simulator(rng)
    for stratum_index, (condition, replicate) in enumerate(_target_strata(artifacts["target"], artifacts["sampler"])):
        population_weight = _initial_population_weight(artifacts["target"], condition, replicate, weeks[0], cells)
        population = _sample_initial_population(
            artifacts["sampler"],
            cells,
            rng,
            particle_id,
            condition,
            replicate,
            stratum_index * int(cells) * 1000,
            population_weight=population_weight,
            simulator=simulator,
        )
        last_week = weeks[0]
        for week in weeks:
            if week != last_week:
                population, interval_events, penalty = _advance_population(population, last_week, week, params, rng, simulator)
                biology_penalty += penalty
                event_rows.extend(interval_events)
            for cell in population:
                history_rows.append(_history_row(particle_id, week, cell))
            snapshots.append(_summarize_population(particle_id, week, population, artifacts["sampler"], condition, replicate))
            last_week = week
    events = _event_summary(particle_id, event_rows, weeks)
    snapshot_df = pd.concat(snapshots, ignore_index=True)
    features = _features_from_snapshots(snapshot_df, artifacts["target"], artifacts["sampler"], events)
    score = score_particle_summary(features, artifacts["target"], artifacts["distance_weights"], params, biology_penalty)
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
        "cycle_transition_scale": float(min(1.0, abs(rng.normal(0.25, 0.10)))),
    }


def _perturb_particle_parameters(parent: dict, rng: np.random.Generator, particle_id: int) -> dict:
    perturbed: dict[str, float | int] = {"particle_id": int(particle_id)}
    for key in ("state_transition_rate", "copy_gain_rate", "copy_loss_rate", "division_rate", "death_rate", "segregation_strength", "cycle_transition_scale"):
        base = float(parent.get(key, 0.05))
        scale = 0.25 * max(base, 0.02)
        upper = 1.0 if key == "segregation_strength" else 0.5
        perturbed[key] = float(np.clip(rng.normal(base, scale), 0.0, upper))
    return perturbed


def _sample_initial_population(
    sampler: dict,
    cells: int,
    rng: np.random.Generator,
    particle_id: int,
    condition: str = "ctrl",
    replicate: str = "r1",
    cell_id_offset: int = 0,
    population_weight: float = 1.0,
    simulator: HybridOgataSimulator | None = None,
) -> list[dict]:
    states = list(sampler["states"])
    stratum_key = f"{condition}|{replicate}"
    state_prob_source = sampler.get("state_probabilities_by_stratum", {}).get(stratum_key, sampler["state_probabilities"])
    state_probs = np.asarray([state_prob_source[state] for state in states], dtype=float)
    state_probs = schemas.normalize_probabilities(state_probs, name="state_probabilities")
    population = []
    for cell_id in range(int(cells)):
        state = str(rng.choice(states, p=state_probs))
        species_uniforms = _species_uniforms(sampler, state, rng)
        copies = {}
        for species, uniform in zip(sampler["species"], species_uniforms):
            dist = sampler["state_species_copy_distributions"][state][species]
            labels = list(dist["bin_labels"])
            probs = schemas.normalize_probabilities(dist["probabilities"], name=f"{state}-{species}")
            label = _sample_label_by_uniform(labels, probs, float(uniform))
            tail_mean = sampler.get("state_species_tail_means", {}).get(state, {}).get(species)
            copies[species] = _sample_copy_from_label(label, sampler["copy_number_bins"], rng, tail_mean)
        soft = _soft_state_for_gate(state, rng)
        soft_values = np.asarray([soft[state_name] for state_name in schemas.STATE_NAMES], dtype=float)
        cell = {
            "particle_id": int(particle_id),
            "condition": condition,
            "replicate": replicate,
            "cell_id": int(cell_id_offset + cell_id),
            "parent_id": -1,
            "state_gate": state,
            "soft_state": soft,
            "latent_state": cfg.ilr(soft_values).tolist(),
            "copies": copies,
            "cycle_state": cfg.CYCLE_NAMES[cfg.sample_initial_cycle_state(rng, cfg.DEFAULT_INITIALIZATION_PARAMETERS)],
            "age": cfg.sample_initial_age(rng, cfg.DEFAULT_INITIALIZATION_PARAMETERS),
            "stress_score": 0.0,
            "survival_score": 0.0,
            "population_weight": float(population_weight),
            "alive": True,
        }
        _refresh_full_chain_state(cell, 0.0, rng, duration=0.0, simulator=simulator)
        population.append(cell)
    return population


def _species_uniforms(sampler: dict, state: str, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import norm

    species = list(sampler["species"])
    corr = np.asarray(sampler.get("species_correlation_by_state", {}).get(state, np.eye(len(species))), dtype=float)
    if corr.shape != (len(species), len(species)):
        corr = np.eye(len(species), dtype=float)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    values = rng.multivariate_normal(np.zeros(len(species)), corr, check_valid="warn")
    return norm.cdf(values)


def _sample_label_by_uniform(labels: list[str], probabilities: np.ndarray, uniform: float) -> str:
    cumulative = np.cumsum(probabilities)
    index = int(np.searchsorted(cumulative, min(max(uniform, 0.0), 1.0), side="right"))
    index = min(index, len(labels) - 1)
    return str(labels[index])


def _sample_copy_from_label(label: str, bins: list[dict], rng: np.random.Generator, tail_mean: float | None = None) -> int:
    item = next(item for item in bins if str(item["label"]) == str(label))
    low = int(item["low"])
    high = item["high"]
    if high is None:
        target_mean = max(float(low), float(tail_mean) if tail_mean is not None else float(low * 1.5))
        probability = float(np.clip(1.0 / max(1.0, target_mean - low + 1.0), 1e-6, 1.0))
        return int(low + rng.geometric(probability) - 1)
    if low == int(high):
        return low
    return int(rng.integers(low, int(high) + 1))


def _advance_population(
    population: list[dict],
    start_week: int,
    end_week: int,
    params: dict,
    rng: np.random.Generator,
    simulator: HybridOgataSimulator | None = None,
) -> tuple[list[dict], list[dict], float]:
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
            working = _copy_cell(cell)
            chain = _refresh_full_chain_state(working, float(current_week), rng, duration=1.0, simulator=simulator)
            event_rates = chain["event_rates"]
            death_probability = float(np.clip(params["death_rate"] + event_rates.get("death", 0.0), 0.0, 0.95))
            if rng.random() < death_probability:
                events.append(_event_row(params["particle_id"], current_week, "death", None, 1, condition=cell.get("condition", ""), replicate=cell.get("replicate", "")))
                continue
            _maybe_apply_cycle_transition(working, event_rates, params, rng)
            generator = chain["transition_generator"]
            from_index = list(schemas.STATE_NAMES).index(working["state_gate"])
            transition_rates = np.clip(generator[from_index, :], 0.0, None)
            transition_probability = float(np.clip(params["state_transition_rate"] * max(1.0, transition_rates.sum()), 0.0, 0.75))
            if rng.random() < transition_probability:
                old_state = working["state_gate"]
                candidates = [to_state for from_state, to_state in ALLOWED_TRANSITIONS if from_state == old_state]
                if candidates:
                    candidate_indices = [list(schemas.STATE_NAMES).index(to_state) for to_state in candidates]
                    weights = transition_rates[candidate_indices]
                    if float(weights.sum()) <= 0.0:
                        weights = np.ones(len(candidate_indices), dtype=float)
                    weights = weights / weights.sum()
                    working["state_gate"] = str(rng.choice(candidates, p=weights))
                    working["soft_state"] = _soft_state_for_gate(working["state_gate"], rng)
                    soft_values = np.asarray([working["soft_state"][state] for state in schemas.STATE_NAMES], dtype=float)
                    working["latent_state"] = cfg.ilr(soft_values).tolist()
                    events.append(
                        _event_row(
                            params["particle_id"],
                            current_week,
                            "transition",
                            None,
                            1,
                            old_state,
                            working["state_gate"],
                            condition=working.get("condition", ""),
                            replicate=working.get("replicate", ""),
                        )
                    )
            for species in schemas.SPECIES:
                gain_probability = float(np.clip(params["copy_gain_rate"] + event_rates.get(f"gain_{species}", 0.0), 0.0, 0.95))
                if rng.random() < gain_probability:
                    increment = int(max(1, rng.poisson(2)))
                    working["copies"][species] += increment
                    events.append(_event_row(params["particle_id"], current_week, "gain", species, increment, condition=working.get("condition", ""), replicate=working.get("replicate", "")))
                loss_probability = float(np.clip(params["copy_loss_rate"] + event_rates.get(f"loss_{species}", 0.0), 0.0, 0.95))
                if working["copies"][species] > 0 and rng.random() < loss_probability:
                    working["copies"][species] -= 1
                    events.append(_event_row(params["particle_id"], current_week, "loss", species, 1, condition=working.get("condition", ""), replicate=working.get("replicate", "")))
            new_population.append(working)
            division_probability = float(np.clip(params["division_rate"] + event_rates.get("division", 0.0), 0.0, 0.75))
            if rng.random() < division_probability:
                daughter = _copy_cell(working)
                daughter["parent_id"] = int(working["cell_id"])
                daughter["cell_id"] = int(next_id)
                daughter["age"] = 0.0
                working["age"] = 0.0
                next_id += 1
                for species in schemas.SPECIES:
                    if rng.random() < params["segregation_strength"]:
                        delta = int(rng.choice([-1, 1]))
                        daughter["copies"][species] = max(0, daughter["copies"][species] + delta)
                _refresh_full_chain_state(working, float(current_week), rng, duration=0.0, simulator=simulator)
                _refresh_full_chain_state(daughter, float(current_week), rng, duration=0.0, simulator=simulator)
                new_population.append(daughter)
                events.append(_event_row(params["particle_id"], current_week, "division", None, 1, condition=working.get("condition", ""), replicate=working.get("replicate", "")))
        if not new_population:
            penalty += 100.0
            new_population = population[:1]
        if len(new_population) > 4 * max(1, len(population)):
            penalty += float(len(new_population))
            new_population = new_population[: 4 * max(1, len(population))]
        population = new_population
    return population, events, penalty


def _refresh_full_chain_state(
    cell: dict,
    week: float,
    rng: np.random.Generator,
    *,
    duration: float,
    simulator: HybridOgataSimulator | None = None,
) -> dict:
    """Refresh X/U/R/V and event-rate proposals through the v4 simulator kernel."""

    active_simulator = simulator or _make_v4_simulator(rng)
    core_cell = _to_core_cell(cell)
    core_cell.last_update_time = float(week)
    core_cell.last_D_C = float(cell.get("last_D_C", active_simulator.params.exposure.D_C0))
    core_cell.last_D_P = float(cell.get("last_D_P", active_simulator.params.exposure.D_P0))
    if duration > 0.0:
        context = active_simulator.advance_cell_to_time(core_cell, float(week) + float(duration), rng)
    else:
        context = active_simulator.build_context(float(week), core_cell.last_D_C, core_cell.last_D_P)
    derived = dyn.compute_derived_quantities(core_cell, context, active_simulator.params)
    if duration == 0.0:
        core_cell.stress_score = float(dyn.compute_stress_attractor(core_cell, derived, context, active_simulator.params))
        core_cell.survival_score = float(dyn.compute_survival_attractor(core_cell, derived, context, active_simulator.params))
    event_rates = dyn.compute_all_event_rates(core_cell, derived, context, active_simulator.params)
    transition_generator = dyn.compute_local_transition_generator(derived.logits, active_simulator.params)
    cell["soft_state"] = dict(zip(schemas.STATE_NAMES, core_cell.soft_state.astype(float).tolist()))
    cell["latent_state"] = core_cell.latent_state.astype(float).tolist()
    cell["stress_score"] = float(core_cell.stress_score)
    cell["survival_score"] = float(core_cell.survival_score)
    cell["cycle_state"] = cfg.CYCLE_NAMES[int(core_cell.cycle_state)]
    cell["age"] = float(core_cell.age)
    cell["last_D_C"] = float(core_cell.last_D_C)
    cell["last_D_P"] = float(core_cell.last_D_P)
    cell["full_chain_policy"] = "HybridOgataSimulator"
    return {"event_rates": event_rates, "transition_generator": transition_generator}


def _to_core_cell(cell: dict) -> Cell:
    soft = np.asarray([cell.get("soft_state", {}).get(state, 0.0) for state in schemas.STATE_NAMES], dtype=float)
    soft = schemas.normalize_probabilities(soft + 1e-9, name="soft_state")
    latent = np.asarray(cell.get("latent_state", cfg.ilr(soft).tolist()), dtype=float)
    if latent.shape != (cfg.LATENT_DIM,):
        latent = cfg.ilr(soft)
    copy_numbers = np.asarray([int(cell["copies"][species]) for species in schemas.SPECIES], dtype=int)
    return Cell(
        cycle_state=_cycle_index(str(cell.get("cycle_state", "G1"))),
        copy_numbers=copy_numbers,
        latent_state=latent,
        soft_state=soft,
        stress_score=float(cell.get("stress_score", 0.0)),
        survival_score=float(cell.get("survival_score", 0.0)),
        age=float(cell.get("age", 0.0)),
        cell_id=int(cell.get("cell_id", 0)),
        parent_id=None if int(cell.get("parent_id", -1)) < 0 else int(cell.get("parent_id", -1)),
    )


def _maybe_apply_cycle_transition(cell: dict, event_rates: dict[str, float], params: dict, rng: np.random.Generator) -> None:
    cycle_edges = {
        "G1_to_S": "S",
        "G1_to_Q": "Q",
        "Q_to_G1": "G1",
        "S_to_G2M": "G2M",
    }
    for event_name, to_cycle in cycle_edges.items():
        probability = float(np.clip(event_rates.get(event_name, 0.0) * (1.0 + params.get("cycle_transition_scale", 0.0)), 0.0, 0.75))
        if rng.random() < probability:
            cell["cycle_state"] = to_cycle
            return


def _cycle_index(cycle_state: str) -> int:
    if cycle_state in cfg.CYCLE_INDEX:
        return int(cfg.CYCLE_INDEX[cycle_state])
    return int(cfg.G1)


def _copy_cell(cell: dict) -> dict:
    return {
        "particle_id": int(cell["particle_id"]),
        "condition": str(cell.get("condition", "ctrl")),
        "replicate": str(cell.get("replicate", "r1")),
        "cell_id": int(cell["cell_id"]),
        "parent_id": int(cell.get("parent_id", -1)),
        "state_gate": str(cell["state_gate"]),
        "soft_state": dict(cell["soft_state"]),
        "latent_state": list(cell.get("latent_state", [0.0, 0.0, 0.0])),
        "copies": {species: int(cell["copies"][species]) for species in schemas.SPECIES},
        "cycle_state": str(cell.get("cycle_state", "G1")),
        "age": float(cell.get("age", 0.0)) + 1.0,
        "stress_score": float(cell.get("stress_score", 0.0)),
        "survival_score": float(cell.get("survival_score", 1.0)),
        "population_weight": float(cell.get("population_weight", 1.0)),
        "last_D_C": float(cell.get("last_D_C", 0.0)),
        "last_D_P": float(cell.get("last_D_P", 0.0)),
        "alive": bool(cell.get("alive", True)),
    }


def _soft_state_for_gate(state: str, rng: np.random.Generator | None = None) -> dict[str, float]:
    if rng is None:
        values = np.full(len(schemas.STATE_NAMES), 0.05 / (len(schemas.STATE_NAMES) - 1), dtype=float)
        values[list(schemas.STATE_NAMES).index(state)] = 0.95
    else:
        concentration = np.ones(len(schemas.STATE_NAMES), dtype=float)
        concentration[list(schemas.STATE_NAMES).index(state)] = 35.0
        values = rng.dirichlet(concentration)
    return dict(zip(schemas.STATE_NAMES, values.tolist()))


def _summarize_population(particle_id: int, week: int, population: list[dict], sampler: dict, condition: str, replicate: str) -> pd.DataFrame:
    rows = []
    pop_size = max(1, len(population))
    population_weights = np.asarray([float(cell.get("population_weight", 1.0)) for cell in population], dtype=float)
    represented_pop_size = float(np.sum(population_weights)) if population_weights.size else float(pop_size)
    for state in schemas.STATE_NAMES:
        state_cells = [cell for cell in population if cell["state_gate"] == state]
        state_weights = np.asarray([float(cell.get("population_weight", 1.0)) for cell in state_cells], dtype=float)
        state_weight_total = float(np.sum(state_weights))
        fraction = state_weight_total / max(1e-9, represented_pop_size)
        for species in schemas.SPECIES:
            values = np.asarray([cell["copies"][species] for cell in state_cells], dtype=float)
            if values.size == 0:
                values = np.asarray([0.0])
                weighted_values = np.asarray([1.0], dtype=float)
                state_weight_total_for_species = 1.0
                n_state_cells = 0
            else:
                weighted_values = state_weights
                state_weight_total_for_species = max(1e-9, state_weight_total)
                n_state_cells = len(state_cells)
            bins = sampler["copy_number_bins"]
            labels = [schemas.assign_copy_bin(value, bins) for value in values]
            top_label = str(bins[-1]["label"])
            copy_mean = float(np.average(values, weights=weighted_values))
            copy_variance = float(np.average((values - copy_mean) ** 2, weights=weighted_values))
            row = {
                "particle_id": particle_id,
                "week": int(week),
                "condition": condition,
                "replicate": replicate,
                "state_gate": state,
                "species": species,
                "flow_fraction": float(fraction),
                "copy_mean": copy_mean,
                "copy_variance": copy_variance,
                "zero_fraction": float(np.sum(weighted_values[values == 0]) / state_weight_total_for_species),
                "tail_fraction": float(np.sum(weighted_values[np.asarray(labels) == top_label]) / state_weight_total_for_species),
                "n_cells": int(n_state_cells),
                "population_size": int(pop_size),
                "represented_population_size": represented_pop_size,
            }
            label_values = np.asarray(labels, dtype=str)
            for item in bins:
                label = str(item["label"])
                row[_bin_probability_column(label)] = float(np.sum(weighted_values[label_values == label]) / state_weight_total_for_species)
            rows.append(row)
    return pd.DataFrame(rows)


def _features_from_snapshots(snapshot: pd.DataFrame, target: pd.DataFrame, sampler: dict, events: pd.DataFrame | None = None) -> pd.DataFrame:
    feature_rows = []
    event_table = pd.DataFrame() if events is None else events
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
                value = _histogram_probability_from_snapshot(snap, row.bin_label)
        elif channel in {"qpcdr", "ddpcr", "lite_summary"}:
            if channel == "ddpcr":
                value = _pooled_mean_from_snapshot(snapshot, row.week, row.condition, row.replicate, row.species)
            elif channel == "lite_summary" and variable == "transition_probability":
                value = _transition_probability_from_events(snapshot, event_table, row)
            elif channel == "lite_summary" and variable == "growth_summary":
                value = _growth_summary_from_snapshots(snapshot, row)
            else:
                key = (row.week, row.condition, row.replicate, row.state_gate, row.species)
                snap = snapshot_index.get(key)
                value = float(getattr(snap, _snapshot_column(variable), 0.0)) if snap is not None else 0.0
        elif channel == "cell_count":
            matches = snapshot[
                (snapshot["week"] == row.week)
                & (snapshot["condition"] == row.condition)
                & (snapshot["replicate"] == row.replicate)
            ]
            if matches.empty:
                value = 0.0
            elif "represented_population_size" in matches:
                value = float(matches["represented_population_size"].astype(float).max())
            else:
                value = float(matches["population_size"].astype(float).max())
        else:
            value = 0.0
        feature_rows.append({"feature_id": row.feature_id, "value": float(value)})
    return pd.DataFrame(feature_rows)


def _transition_probability_from_events(snapshot: pd.DataFrame, events: pd.DataFrame, row) -> float:
    from_state = str(row.from_state)
    to_state = str(row.to_state)
    state_snapshot = snapshot[
        (snapshot["week"] == row.week)
        & (snapshot["condition"] == row.condition)
        & (snapshot["replicate"] == row.replicate)
        & (snapshot["state_gate"] == from_state)
    ]
    n_from = float(state_snapshot["n_cells"].max()) if not state_snapshot.empty else 0.0
    if n_from <= 0.0:
        return 1.0 if from_state == to_state else 0.0
    if events.empty:
        return 1.0 if from_state == to_state else 0.0
    transitions = events[
        (events["week"] == row.week)
        & (events["condition"] == row.condition)
        & (events["replicate"] == row.replicate)
        & (events["event_type"] == "transition")
        & (events["from_state"] == from_state)
    ]
    if from_state == to_state:
        moved = float(transitions["count"].sum()) if not transitions.empty else 0.0
        return float(np.clip(1.0 - moved / n_from, 0.0, 1.0))
    count = float(transitions.loc[transitions["to_state"] == to_state, "count"].sum()) if not transitions.empty else 0.0
    return float(np.clip(count / n_from, 0.0, 1.0))


def _growth_summary_from_snapshots(snapshot: pd.DataFrame, row) -> float:
    state = str(row.state_gate)
    current = snapshot[
        (snapshot["week"] == row.week)
        & (snapshot["condition"] == row.condition)
        & (snapshot["replicate"] == row.replicate)
        & (snapshot["state_gate"] == state)
    ]
    if current.empty:
        return 0.0
    later_weeks = sorted(
        int(value)
        for value in snapshot.loc[
            (snapshot["week"] > row.week) & (snapshot["condition"] == row.condition) & (snapshot["replicate"] == row.replicate),
            "week",
        ].unique()
    )
    if not later_weeks:
        return 0.0
    nxt = snapshot[
        (snapshot["week"] == later_weeks[0])
        & (snapshot["condition"] == row.condition)
        & (snapshot["replicate"] == row.replicate)
        & (snapshot["state_gate"] == state)
    ]
    if nxt.empty:
        return 0.0
    f0 = float(current["flow_fraction"].mean())
    f1 = float(nxt["flow_fraction"].mean())
    return float(np.log((f1 + 1e-9) / (f0 + 1e-9)))


def _histogram_probability_from_snapshot(snap, target_label: str) -> float:
    column = _bin_probability_column(str(target_label))
    if hasattr(snap, column):
        return float(getattr(snap, column))
    if str(target_label) == "0":
        return float(snap.zero_fraction)
    return 0.0


def _bin_probability_column(label: str) -> str:
    safe = str(label).replace("+", "plus").replace("-", "_")
    return f"bin_probability__{safe}"


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


def _event_row(
    particle_id: int,
    week: int,
    event_type: str,
    species: str | None,
    count: int,
    from_state: str | None = None,
    to_state: str | None = None,
    condition: str = "",
    replicate: str = "",
) -> dict:
    return {
        "particle_id": int(particle_id),
        "week": int(week),
        "condition": condition,
        "replicate": replicate,
        "event_type": event_type,
        "species": species or "",
        "count": int(count),
        "from_state": from_state or "",
        "to_state": to_state or "",
    }


def _event_summary(particle_id: int, event_rows: list[dict], weeks: list[int]) -> pd.DataFrame:
    if not event_rows:
        return pd.DataFrame([{"particle_id": particle_id, "week": week, "condition": "", "replicate": "", "event_type": "none", "species": "", "from_state": "", "to_state": "", "count": 0} for week in weeks])
    rows = pd.DataFrame(event_rows)
    return rows.groupby(["particle_id", "week", "condition", "replicate", "event_type", "species", "from_state", "to_state"], as_index=False)["count"].sum()


def _history_row(particle_id: int, week: int, cell: dict) -> dict:
    total_copy = sum(int(cell["copies"][species]) for species in schemas.SPECIES)
    stress = float(cell.get("stress_score", np.log1p(total_copy)))
    survival = float(cell.get("survival_score", 1.0 / (1.0 + np.log1p(total_copy))))
    latent = np.asarray(cell.get("latent_state", [0.0, 0.0, 0.0]), dtype=float)
    if latent.shape != (cfg.LATENT_DIM,):
        soft = np.asarray([cell.get("soft_state", {}).get(state, 0.0) for state in schemas.STATE_NAMES], dtype=float)
        latent = cfg.ilr(schemas.normalize_probabilities(soft + 1e-9, name="history soft_state"))
    row = {
        "particle_id": int(particle_id),
        "week": int(week),
        "condition": str(cell.get("condition", "ctrl")),
        "replicate": str(cell.get("replicate", "r1")),
        "cell_id": int(cell["cell_id"]),
        "parent_id": int(cell.get("parent_id", -1)),
        "state_gate": str(cell["state_gate"]),
        "cycle_state": str(cell.get("cycle_state", "G1")),
        "age": float(cell.get("age", week)),
        "latent_R_raw": float(stress - 1.0),
        "latent_V_raw": float(1.0 - stress),
        "R": float(schemas.softplus([stress - 1.0])[0]),
        "V": float(schemas.softplus([1.0 - stress])[0]),
        "A": float(cell.get("age", week)),
        "population_weight": float(cell.get("population_weight", 1.0)),
        "latent_U_1": float(latent[0]),
        "latent_U_2": float(latent[1]),
        "latent_U_3": float(latent[2]),
    }
    for state, value in cell.get("soft_state", {}).items():
        row[f"X_{state}"] = float(value)
    for species in schemas.SPECIES:
        row[f"K_{species}"] = int(cell["copies"][species])
    return row


def _write_method_full_outputs(
    out: Path,
    accepted_histories: list[dict],
    parameter_rows: list[dict],
    weights: pd.DataFrame,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    write_table(pd.DataFrame(parameter_rows), out / "FULL_particle_parameters.parquet")
    write_table(weights, out / "FULL_particle_weights.parquet")
    write_table(snapshots, out / "FULL_snapshot_summaries.parquet")
    write_table(events, out / "FULL_event_summaries.parquet")
    history_df = pd.DataFrame(accepted_histories)
    write_table(history_df, out / "FULL_single_cell_history_samples.parquet")
    derived = _derived_q_from_snapshots(snapshots)
    write_table(derived, out / "FULL_derived_Q.parquet")
    accepted_weights = _accepted_normalized_weights(weights)
    raw_like = features.merge(accepted_weights[["particle_id", "accepted_weight"]], on="particle_id", how="inner")
    target_meta = target.rename(columns={"weight": "target_weight"})
    raw_like = raw_like.merge(
        target_meta.drop(columns=[column for column in ("value",) if column in target_meta.columns]),
        on="feature_id",
        how="left",
        validate="many_to_one",
    )
    raw_like = raw_like.rename(columns={"accepted_weight": "posterior_weight"})
    write_table(raw_like, out / "FULL_ppc_raw_observables.parquet")
    _write_zarr_history_ensemble(out / "FULL_particles_final.zarr", history_df, weights, events)
    write_text_pdf(
        out / "FULL_history_reconstruction_report.pdf",
        "Full History Reconstruction Report",
        [
            "Output is a weighted conditional single-cell history ensemble.",
            f"particles={weights['particle_id'].nunique()}, accepted={int(weights['accepted'].sum())}",
            "Parameters are latent controls and are not reported as unique biological truths.",
        ],
    )


def _write_zarr_history_ensemble(path: Path, history: pd.DataFrame, weights: pd.DataFrame, events: pd.DataFrame) -> None:
    """Persist accepted histories as a real zarr group with typed arrays."""

    import zarr
    from zarr.storage import ZipStore

    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"{path} is an older directory zarr store; choose a clean output directory or remove it before rerun")
    store = ZipStore(str(path), mode="w")
    try:
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update(
            {
                "artifact": "FULL_particles_final",
                "store": "zarr.ZipStore",
                "method_source": "markdown/fit_method.md",
                "role": "accepted conditional single-cell history ensemble",
                "species": list(schemas.SPECIES),
                "states": list(schemas.STATE_NAMES),
                "history_rows": int(len(history)),
                "particles": int(weights["particle_id"].nunique()) if "particle_id" in weights else 0,
            }
        )

        history_group = root.create_group("history")
        if history.empty:
            _zarr_dataset(history_group, "particle_id", np.asarray([], dtype=np.int64))
            _zarr_dataset(history_group, "week", np.asarray([], dtype=np.int16))
            _zarr_dataset(history_group, "condition_code", np.asarray([], dtype=np.int16))
            _zarr_dataset(history_group, "replicate_code", np.asarray([], dtype=np.int16))
            _zarr_dataset(history_group, "cell_id", np.asarray([], dtype=np.int64))
            _zarr_dataset(history_group, "parent_id", np.asarray([], dtype=np.int64))
            _zarr_dataset(history_group, "state_code", np.asarray([], dtype=np.int16))
            _zarr_dataset(history_group, "K", np.empty((0, len(schemas.SPECIES)), dtype=np.int32))
            _zarr_dataset(history_group, "X", np.empty((0, len(schemas.STATE_NAMES)), dtype=np.float32))
            _zarr_dataset(history_group, "RVA", np.empty((0, 3), dtype=np.float32))
            _zarr_dataset(history_group, "latent_U", np.empty((0, 3), dtype=np.float32))
            _zarr_dataset(history_group, "population_weight", np.asarray([], dtype=np.float32))
        else:
            state_codes = _encode_categories(history["state_gate"], schemas.STATE_NAMES)
            condition_categories = sorted(str(value) for value in history["condition"].dropna().unique())
            replicate_categories = sorted(str(value) for value in history["replicate"].dropna().unique())
            _zarr_dataset(history_group, "particle_id", history["particle_id"].to_numpy(dtype=np.int64))
            _zarr_dataset(history_group, "week", history["week"].to_numpy(dtype=np.int16))
            _zarr_dataset(history_group, "condition_code", _encode_categories(history["condition"], condition_categories))
            _zarr_dataset(history_group, "replicate_code", _encode_categories(history["replicate"], replicate_categories))
            _zarr_dataset(history_group, "cell_id", history["cell_id"].to_numpy(dtype=np.int64))
            _zarr_dataset(history_group, "parent_id", history["parent_id"].to_numpy(dtype=np.int64))
            _zarr_dataset(history_group, "state_code", state_codes)
            copy_matrix = np.column_stack([history[f"K_{species}"].to_numpy(dtype=np.int32) for species in schemas.SPECIES])
            soft_matrix = np.column_stack([history[f"X_{state}"].to_numpy(dtype=np.float32) for state in schemas.STATE_NAMES])
            rva_matrix = history[["R", "V", "A"]].to_numpy(dtype=np.float32)
            latent_matrix = history[["latent_U_1", "latent_U_2", "latent_U_3"]].to_numpy(dtype=np.float32)
            _zarr_dataset(history_group, "population_weight", history["population_weight"].to_numpy(dtype=np.float32))
            _zarr_dataset(history_group, "K", copy_matrix)
            _zarr_dataset(history_group, "X", soft_matrix)
            _zarr_dataset(history_group, "RVA", rva_matrix)
            _zarr_dataset(history_group, "latent_U", latent_matrix)
        history_group.attrs.update(
            {
                "state_code_categories": list(schemas.STATE_NAMES),
                "condition_code_categories": [] if history.empty else condition_categories,
                "replicate_code_categories": [] if history.empty else replicate_categories,
                "K_columns": list(schemas.SPECIES),
                "X_columns": [f"X_{state}" for state in schemas.STATE_NAMES],
                "RVA_columns": ["R", "V", "A"],
                "latent_U_columns": ["latent_U_1", "latent_U_2", "latent_U_3"],
            }
        )

        weight_group = root.create_group("weights")
        _zarr_dataset(weight_group, "particle_id", weights["particle_id"].to_numpy(dtype=np.int64))
        _zarr_dataset(weight_group, "score", weights["score"].to_numpy(dtype=np.float64))
        _zarr_dataset(weight_group, "weight", weights["weight"].to_numpy(dtype=np.float64))
        _zarr_dataset(weight_group, "accepted", weights["accepted"].to_numpy(dtype=bool))

        event_group = root.create_group("events")
        if events.empty:
            _zarr_dataset(event_group, "particle_id", np.asarray([], dtype=np.int64))
            _zarr_dataset(event_group, "week", np.asarray([], dtype=np.int16))
            _zarr_dataset(event_group, "event_type_code", np.asarray([], dtype=np.int16))
            _zarr_dataset(event_group, "species_code", np.asarray([], dtype=np.int16))
            _zarr_dataset(event_group, "count", np.asarray([], dtype=np.int32))
            event_types: list[str] = []
            species_labels: list[str] = [""]
        else:
            event_types = sorted(str(value) for value in events["event_type"].dropna().unique())
            species_labels = [""] + [species for species in schemas.SPECIES if species in set(events["species"].astype(str))]
            _zarr_dataset(event_group, "particle_id", events["particle_id"].to_numpy(dtype=np.int64))
            _zarr_dataset(event_group, "week", events["week"].to_numpy(dtype=np.int16))
            _zarr_dataset(event_group, "event_type_code", _encode_categories(events["event_type"], event_types))
            _zarr_dataset(event_group, "species_code", _encode_categories(events["species"].fillna(""), species_labels))
            _zarr_dataset(event_group, "count", events["count"].to_numpy(dtype=np.int32))
        event_group.attrs.update({"event_type_code_categories": event_types, "species_code_categories": species_labels})
    finally:
        store.close()


def _zarr_dataset(group, name: str, data: np.ndarray) -> None:
    chunks = _zarr_chunks(data)
    if chunks:
        group.create_dataset(name, data=data, chunks=chunks, compressor=None)
    else:
        group.create_dataset(name, data=data, compressor=None)


def _zarr_chunks(data: np.ndarray) -> tuple[int, ...] | None:
    if data.ndim == 0:
        return None
    if data.shape[0] == 0:
        return data.shape
    first = min(1024, int(data.shape[0]))
    return (first, *data.shape[1:])


def _encode_categories(values: pd.Series, categories: tuple[str, ...] | list[str]) -> np.ndarray:
    mapping = {str(value): idx for idx, value in enumerate(categories)}
    encoded = []
    for value in values.astype(str):
        if value not in mapping:
            raise ValueError(f"Cannot encode unknown category {value!r}; expected {list(categories)}")
        encoded.append(mapping[value])
    return np.asarray(encoded, dtype=np.int16)


def _derived_q_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in snapshots.itertuples(index=False):
        q_value = float(row.zero_fraction) - float(row.tail_fraction)
        rows.append(
            {
                "particle_id": int(row.particle_id),
                "week": int(row.week),
                "condition": row.condition,
                "replicate": row.replicate,
                "state_gate": row.state_gate,
                "species": row.species,
                "Q": q_value,
                "B": float(row.copy_mean),
                "P": float(row.tail_fraction),
            }
        )
    return pd.DataFrame(rows)


def _posterior_weights(scores: pd.DataFrame) -> pd.DataFrame:
    shifted = scores["score"].astype(float) - float(scores["score"].min())
    raw = np.exp(-0.5 * shifted.to_numpy(dtype=float))
    total = float(np.sum(raw))
    weights = raw / total if total > 0.0 else np.ones(len(raw), dtype=float) / max(1, len(raw))
    result = scores.copy()
    result["weight"] = weights
    return result


def _write_ppc_report(out: Path, weights: pd.DataFrame, target: pd.DataFrame, features: pd.DataFrame) -> dict:
    accepted_weights = _accepted_normalized_weights(weights)
    merged = features.merge(accepted_weights[["particle_id", "accepted_weight"]], on="particle_id", how="inner")
    merged["weighted_value"] = merged["value"].astype(float) * merged["accepted_weight"].astype(float)
    weighted = merged.groupby("feature_id", as_index=False)["weighted_value"].sum().rename(columns={"weighted_value": "posterior_mean"})
    report = target[["feature_id", "channel", "target", "variance"]].merge(weighted, on="feature_id", how="left")
    report["abs_error"] = (report["posterior_mean"] - report["target"]).abs()
    report["covered_by_two_sigma"] = report["abs_error"] <= 2.0 * np.sqrt(report["variance"].astype(float).clip(lower=1e-9))
    channel = report.groupby("channel", as_index=False)["abs_error"].mean()
    coverage = report.groupby("channel", as_index=False)["covered_by_two_sigma"].mean().rename(columns={"covered_by_two_sigma": "coverage"})
    write_table(channel, out / "full_ppc_channel_errors.parquet")
    write_table(coverage, out / "full_ppc_channel_coverage.parquet")
    ess = _effective_sample_size(accepted_weights["accepted_weight"].astype(float).to_numpy())
    payload = {
        "score_components": list(SCORE_COMPONENTS),
        "mean_abs_error_by_channel": dict(zip(channel["channel"], channel["abs_error"])),
        "coverage_by_channel": dict(zip(coverage["channel"], coverage["coverage"].astype(float))),
        "history_ensemble_not_single_best_parameter": True,
        "ppc_particle_scope": "accepted_particles_only_with_renormalized_weights",
        "accepted_particle_ess": ess,
    }
    write_json(
        out / "full_ppc_report.json",
        payload,
    )
    write_markdown_report(
        out / "full_ppc_report.md",
        "Full Posterior Predictive Check",
        [
            ("Scope", "Compared weighted particle summaries against raw/lite target features."),
            ("Channels", ", ".join(str(channel_name) for channel_name in channel["channel"])),
            ("Particle Scope", "Only accepted particles are used, with posterior weights renormalized within the accepted ensemble."),
            ("Interpretation", "Accepted output is a weighted history ensemble, not a unique best-fit full parameter."),
        ],
    )
    return payload


def _full_continue_diagnostics(weights: pd.DataFrame, ppc_payload: dict, scenario_classes: pd.DataFrame) -> dict:
    accepted = _accepted_normalized_weights(weights)
    retained = int(accepted["particle_id"].nunique())
    ess = _effective_sample_size(accepted["accepted_weight"].astype(float).to_numpy())
    coverage = {str(key): float(value) for key, value in ppc_payload.get("coverage_by_channel", {}).items()}
    thresholds = {
        "particle_ess_ge_20pct_retained": bool(ess >= 0.2 * max(1, retained)),
        "flow_ppc_ge_0.85": bool(coverage.get("flow", 1.0) >= 0.85),
        "ectag_ppc_ge_0.80": bool(coverage.get("ectag", 1.0) >= 0.80),
        "qpcdr_ppc_ge_0.85": bool(coverage.get("qpcdr", 1.0) >= 0.85),
        "ddpcr_ppc_ge_0.90": bool(coverage.get("ddpcr", 1.0) >= 0.90),
        "cell_count_ppc_ge_0.85": bool(coverage.get("cell_count", 1.0) >= 0.85),
        "lite_summary_ppc_ge_0.80": bool(coverage.get("lite_summary", 1.0) >= 0.80),
    }
    sensitivity = _prior_sensitivity_summary(weights, scenario_classes)
    thresholds["prior_bounds_sensitivity_20pct_passed"] = bool(sensitivity["primary_scenario_retained"])
    accepted_scenarios = scenario_classes[scenario_classes["accepted"]] if "accepted" in scenario_classes else scenario_classes
    thresholds["scenario_diversity_not_single_history"] = bool(retained > 1 and accepted_scenarios["scenario_class"].nunique() >= 1)
    return {
        "retained_particles": retained,
        "accepted_particle_ess": ess,
        "coverage_by_channel": coverage,
        "prior_sensitivity_20pct": sensitivity,
        "continue_thresholds": thresholds,
        "continue_gate_passed": bool(all(thresholds.values())),
    }


def _prior_sensitivity_summary(weights: pd.DataFrame, scenario_classes: pd.DataFrame) -> dict:
    if scenario_classes.empty:
        return {"primary_scenario": "", "primary_scenario_retained": False, "scenario_weight_by_scale": {}}
    merged = scenario_classes.merge(weights[["particle_id", "score", "accepted"]], on=["particle_id", "accepted"], how="left")
    accepted = merged[merged["accepted"]].copy()
    if accepted.empty:
        accepted = merged.copy()
    scenario_weight_by_scale: dict[str, dict[str, float]] = {}
    primary_scenario = ""
    for scale in (0.8, 1.0, 1.2):
        shifted = accepted["score"].astype(float) - float(accepted["score"].astype(float).min())
        raw = accepted["posterior_weight"].astype(float).to_numpy() * np.exp(-0.5 * (float(scale) - 1.0) * shifted.to_numpy(dtype=float))
        total = float(np.sum(raw))
        normalized = raw / total if total > 0.0 and np.isfinite(total) else np.ones(len(raw), dtype=float) / max(1, len(raw))
        temp = accepted[["scenario_class"]].copy()
        temp["weight"] = normalized
        summary = temp.groupby("scenario_class")["weight"].sum().sort_values(ascending=False)
        scenario_weight_by_scale[f"{scale:.1f}"] = {str(key): float(value) for key, value in summary.items()}
        if scale == 1.0 and not summary.empty:
            primary_scenario = str(summary.index[0])
    retained = all(weights_by_scenario.get(primary_scenario, 0.0) > 0.0 for weights_by_scenario in scenario_weight_by_scale.values()) if primary_scenario else False
    return {
        "policy": "posterior scenario weights reweighted after +/-20% prior-score scale perturbation",
        "primary_scenario": primary_scenario,
        "primary_scenario_retained": bool(retained),
        "scenario_weight_by_scale": scenario_weight_by_scale,
    }


def _write_incompatibility_report(out: Path, diagnostics: dict) -> None:
    failed = [name for name, passed in diagnostics.get("continue_thresholds", {}).items() if not passed]
    write_markdown_report(
        out / "FULL_model_incompatibility_report.md",
        "FULL Model Incompatibility Report",
        [
            ("Conclusion", "Under the current v4 full structure and locked observation model, the run did not pass every stability gate."),
            ("Failed Gates", ", ".join(failed) if failed else "none"),
            (
                "Allowed Recovery Order",
                "Increase particle count; relax summary tolerance without changing observation calibration; identify the conflicting data source; rerun before changing biological interpretation.",
            ),
        ],
    )


def _accepted_normalized_weights(weights: pd.DataFrame) -> pd.DataFrame:
    accepted = weights[weights["accepted"]].copy()
    if accepted.empty:
        accepted = weights.copy()
    total = float(accepted["weight"].astype(float).sum())
    if total <= 0.0 or not np.isfinite(total):
        accepted["accepted_weight"] = 1.0 / max(1, len(accepted))
    else:
        accepted["accepted_weight"] = accepted["weight"].astype(float) / total
    return accepted


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    normalized = values / total
    denom = float(np.sum(normalized * normalized))
    return float(1.0 / denom) if denom > 0.0 else 0.0
