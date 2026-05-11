"""Bulk-visible full-model wrapper and adaptive SMC artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from core.simulation import run_simulation
from fit import schemas
from fit.io_utils import ensure_dir, read_json, write_json, write_table, write_text_pdf
from fit.objective import score_bulk_predictions
from fit.observation import load_observation_params
from fit.v4_lite import load_lite_artifacts

METHOD_N_SIM_FIT = 10_000
METHOD_N_SIM_REPLAY = 50_000
METHOD_MOMENT_CANDIDATES = 200_000
METHOD_MOMENT_MAX_CANDIDATES = 500_000
METHOD_MOMENT_KEEP_TOP = 10_000
METHOD_MOMENT_MIN_TOP = 5_000
METHOD_FULL_PARTICLES_INITIAL = 3_000
METHOD_FULL_PARTICLES_FINAL = 1_000
METHOD_COARSE_PARTICLES = 10_000
METHOD_COARSE_CELLS = 1_000


def _progress(message: str) -> None:
    print(f"[fit] {message}", flush=True)


def _fmt(value: float) -> str:
    return f"{float(value):.4g}" if np.isfinite(float(value)) else str(value)


def _resolve_workers(workers: int | None, task_count: int) -> int:
    requested = 1 if workers is None else int(workers)
    if requested < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return min(requested, max(1, int(task_count)))


def _parallel_map(func, tasks: list, workers: int | None) -> list:
    if not tasks:
        return []
    worker_count = _resolve_workers(workers, len(tasks))
    if worker_count == 1:
        return [func(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(func, tasks))


def _dataframe_chunks(df: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
    if df.empty:
        return []
    chunk_count = min(max(1, int(parts)), len(df))
    chunk_size = int(np.ceil(len(df) / chunk_count))
    return [df.iloc[start : start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def run_moment_prescreen(
    lite_dir: str | Path,
    prior_dir: str | Path,
    output_dir: str | Path,
    seed: int = 1,
    n_candidates: int = METHOD_MOMENT_CANDIDATES,
    keep_top: int = METHOD_MOMENT_KEEP_TOP,
    workers: int = 1,
) -> dict[str, Path]:
    """Cheap moment approximation pre-screen for active effective controls."""

    artifacts = load_lite_artifacts(lite_dir)
    prior = pd.read_parquet(Path(prior_dir) / "PRIOR_predictive_accepted_region.parquet")
    out = ensure_dir(output_dir)
    worker_count = _resolve_workers(workers, max(1, int(n_candidates)))
    rng = np.random.default_rng(seed)
    method_min_keep = METHOD_MOMENT_MIN_TOP
    target_keep = max(method_min_keep, int(keep_top))
    candidate_count = max(1, int(n_candidates))
    _progress(
        f"moment prescreen start: candidates={candidate_count}, keep_top={target_keep}, "
        f"method_min_keep={method_min_keep}, workers={worker_count}"
    )
    candidates = _candidate_parameter_table(artifacts, prior, rng, candidate_count)
    scores = _score_candidate_moments(candidates, artifacts, workers=worker_count)
    expanded = False
    if len(scores) < method_min_keep and candidate_count < METHOD_MOMENT_MAX_CANDIDATES:
        _progress(f"moment prescreen expanding: scored={len(scores)} < {method_min_keep}; candidates={METHOD_MOMENT_MAX_CANDIDATES}")
        candidate_count = METHOD_MOMENT_MAX_CANDIDATES
        candidates = _candidate_parameter_table(artifacts, prior, rng, candidate_count)
        scores = _score_candidate_moments(candidates, artifacts, workers=worker_count)
        expanded = True
    if len(scores) < method_min_keep:
        write_table(candidates, out / "MOMENT_candidate_parameters.parquet")
        write_table(scores, out / "MOMENT_scores.parquet")
        report = out / "MOMENT_prescreen_incompatible_report.md"
        report.write_text(
            "# Moment Prescreen Incompatible\n\n"
            f"Required at least {method_min_keep} top candidates under the method continue criteria.\n\n"
            f"Available candidates after expansion: {len(scores)}.\n",
            encoding="utf-8",
        )
        _progress(f"moment prescreen failed: scored={len(scores)}, report={report}")
        raise RuntimeError(f"moment prescreen produced fewer than {method_min_keep} candidates; wrote {report}")
    keep_n = min(target_keep, len(scores))
    keep_scores = scores.nsmallest(keep_n, "D_moment").drop(columns=[column for column in ("D_prior", "D_biology") if column in scores.columns])
    keep = keep_scores.merge(candidates, on="particle_id", how="left")
    _progress(
        "moment prescreen result: "
        f"scored={len(scores)}, keep={len(keep)}, "
        f"best_D={_fmt(scores['D_moment'].min())}, median_D={_fmt(scores['D_moment'].median())}, "
        f"median_D_prior={_fmt(scores['D_prior'].median()) if 'D_prior' in scores else 'NA'}"
    )
    write_table(candidates, out / "MOMENT_candidate_parameters.parquet")
    write_table(scores, out / "MOMENT_scores.parquet")
    write_table(keep, out / "MOMENT_keep_top_particles.parquet")
    write_text_pdf(
        out / "MOMENT_prescreen_report.pdf",
        "Moment Prescreen Report",
        [
            f"candidates={len(candidates)}; keep_top={len(keep)}; method_min_top={method_min_keep}",
            f"expanded_to_500000={expanded}",
            f"workers={worker_count}",
            "Moment score uses ddPCR bulk velocity, cell-count growth, flow3 steady, and prior penalty.",
            "No single-cell ecTAG/qPCDR likelihood is used.",
        ],
    )
    _progress(f"moment prescreen done: output={out}")
    return {name: out / name for name in schemas.MOMENT_OUTPUTS}


def create_full_initial_particles(
    lite_dir: str | Path,
    output_dir: str | Path,
    particles: int = 16,
    cells: int = METHOD_N_SIM_FIT,
    seed: int = 1,
    moment_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Create full initialization inputs without modifying the full model."""

    rng = np.random.default_rng(seed)
    artifacts = load_lite_artifacts(lite_dir)
    out = ensure_dir(output_dir)
    moment = _load_moment_particles(moment_dir, artifacts, rng, particles)
    sampler = artifacts["sampler"]
    first_counts = pd.DataFrame(sampler["cell_count_anchor"])
    first_ddpcr = pd.DataFrame(sampler["ddpcr_bulk_anchor"])
    flow_init = artifacts["initializer"]
    _progress(f"full initialization start: particles={int(particles)}, cells_per_stratum={int(cells)}, strata={len(first_counts)}")
    pop_rows = []
    summary_rows = []
    for row in first_counts.itertuples(index=False):
        weight = float(row.total_cell_count) / max(1, int(cells))
        mean_vector = _anchor_mean_vector(first_ddpcr, str(row.condition), str(row.replicate))
        for particle_id in range(int(particles)):
            rho = float(rng.beta(2, 2))
            state_fractions = _state_fractions_from_flow3(artifacts["flow3"], rho)
            for state, fraction in state_fractions.items():
                n_state = int(round(int(cells) * fraction))
                state_copies = _mean_matched_zinb_pool(rng, mean_vector, max(1, n_state))
                summary_rows.append(
                    {
                        "particle_id": particle_id,
                        "condition": str(row.condition),
                        "replicate": str(row.replicate),
                        "state_gate": state,
                        "sim_cells": n_state,
                        "population_weight": weight,
                        "projected_flow3_group": _state_to_flow3(state),
                        "mean_MYC_copy": float(state_copies[:, 0].mean()),
                        "mean_CDK4_copy": float(state_copies[:, 1].mean()),
                        "mean_PDGFRA_copy": float(state_copies[:, 2].mean()),
                    }
                )
                for local_id in range(n_state):
                    copies = state_copies[local_id]
                    pop_rows.append(
                        {
                            "particle_id": particle_id,
                            "condition": str(row.condition),
                            "replicate": str(row.replicate),
                            "cell_id": len(pop_rows),
                            "state_gate": state,
                            "population_weight": weight,
                            "rho": rho,
                            "MYC_copy": int(copies[0]),
                            "CDK4_copy": int(copies[1]),
                            "PDGFRA_copy": int(copies[2]),
                        }
                    )
    params = moment.head(int(particles)).copy()
    if "particle_id" in params:
        params["particle_id"] = range(len(params))
    write_table(pd.DataFrame(summary_rows), out / "FULL_initial_population_summary.parquet")
    write_table(params, out / "FULL_initial_parameter_particles.parquet")
    _write_population_zarr(out / "FULL_initial_population.zarr", pd.DataFrame(pop_rows), first_ddpcr, flow_init)
    write_text_pdf(
        out / "FULL_initialization_report.pdf",
        "Full Initialization Report",
        [
            "Initial N matches week-1 cell count through representative cell weights.",
            "Four-state initialization uses Beta(2,2) hidden NPC/OPC split under the flow3 projection.",
            "Copy-number initialization is mean-matched to week-1 ddPCR; single-cell distribution shape is prior-only.",
        ],
    )
    _progress(
        f"full initialization done: population_rows={len(pop_rows)}, "
        f"summary_rows={len(summary_rows)}, output={out}"
    )
    return {name: out / name for name in schemas.FULL_INIT_OUTPUTS}


def run_full_reconstruction(
    lite_dir: str | Path,
    obs_params_path: str | Path,
    output_dir: str | Path,
    particles: int = METHOD_FULL_PARTICLES_INITIAL,
    cells: int = METHOD_N_SIM_FIT,
    seed: int = 1,
    acceptance_quantile: float = 0.5,
    smc_steps: int = 4,
    moment_dir: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Path]:
    """Run adaptive partial-observation SMC over bulk-visible controls only."""

    artifacts = load_lite_artifacts(lite_dir)
    obs = load_observation_params(obs_params_path)
    out = ensure_dir(output_dir)
    worker_count = _resolve_workers(workers, max(1, int(particles)))
    rng = np.random.default_rng(seed)
    current = _with_prior_distance(_load_moment_particles(moment_dir, artifacts, rng, max(int(particles), 1)), artifacts)
    centers = _centers(artifacts)
    growth_controls = sum(1 for name in centers if name.startswith("r__"))
    copy_controls = sum(1 for name in centers if name.startswith("v__"))
    flow_controls = sum(1 for name in centers if name.startswith("zeta_flow3__"))
    smc_rounds = max(1, int(smc_steps))
    _progress(
        f"full SMC start: particles={int(particles)}, cells={int(cells)}, smc_steps={smc_rounds}, "
        f"workers={worker_count}, active_controls={len(centers)} "
        f"(growth={growth_controls}, copy={copy_controls}, flow3={flow_controls})"
    )
    adaptation_rows = []
    all_params = []
    all_scores = []
    early_rows = []
    mc_rows = []
    proposal_scale = {"growth": 1.0, "copy_MYC": 1.0, "copy_CDK4": 1.0, "copy_PDGFRA": 1.0, "flow3": 1.0}
    epsilon = float("inf")
    quantiles = [0.70, 0.50, 0.30, 0.20]
    next_round_cells = min(int(cells), METHOD_COARSE_CELLS)
    for smc_round in range(smc_rounds):
        round_cells = max(1, int(next_round_cells))
        _progress(
            f"SMC round {smc_round + 1}/{smc_rounds} start: "
            f"proposals={int(particles)}, sim_cells={round_cells}, epsilon_prev={_fmt(epsilon)}"
        )
        proposed = _propose_round(current, artifacts, rng, int(particles), smc_round, proposal_scale)
        score_rows = []
        score_tasks = [(param.to_dict(), artifacts, obs, epsilon, round_cells, seed + smc_round, smc_round) for _, param in proposed.iterrows()]
        for row, screen_rows in _parallel_map(_score_proposed_particle, score_tasks, worker_count):
            early_rows.extend(screen_rows)
            score_rows.append(row)
        scores = pd.DataFrame(score_rows)
        q = quantiles[min(smc_round, len(quantiles) - 1)]
        candidate_epsilon = float(scores["score"].quantile(q))
        if np.isfinite(epsilon):
            lower = epsilon * 0.50
            upper = epsilon * 0.95
            candidate_epsilon = float(min(upper, max(lower, candidate_epsilon)))
        epsilon = candidate_epsilon
        prior_limit = float(max(current["D_prior"].astype(float).quantile(0.99), current["D_prior"].astype(float).min())) if "D_prior" in current else float("inf")
        scores_for_acceptance = scores.drop(columns=["round", "D_prior"], errors="ignore")
        accepted = proposed.merge(scores_for_acceptance, on="particle_id", how="left")
        accepted = accepted[(accepted["score"] <= epsilon) & (accepted["D_prior"].astype(float) <= prior_limit) & (accepted["D_biology"].astype(float) == 0.0) & (~accepted["early_rejected"].fillna(False).astype(bool))].copy()
        if accepted.empty:
            feasible = proposed.merge(scores_for_acceptance, on="particle_id", how="left")
            feasible = feasible[(feasible["D_prior"].astype(float) <= prior_limit) & (feasible["D_biology"].astype(float) == 0.0) & (~feasible["early_rejected"].fillna(False).astype(bool))]
            report = out / "FULL_smc_incompatible_under_current_tolerance.md"
            report.write_text(
                "# FULL SMC Incompatible Under Current Tolerance\n\n"
                f"round={smc_round}\n\n"
                f"epsilon={epsilon}\n\n"
                f"prior_limit={prior_limit}\n\n"
                f"feasible_particles={len(feasible)}\n\n"
                "No fallback particle was accepted because the method requires data, prior, and biological gates to pass simultaneously.\n",
                encoding="utf-8",
            )
            _progress(
                f"SMC round {smc_round + 1}/{smc_rounds} failed: "
                f"epsilon={_fmt(epsilon)}, prior_limit={_fmt(prior_limit)}, feasible_particles={len(feasible)}, report={report}"
            )
            raise RuntimeError(f"no accepted particles under method gates; wrote {report}")
        acceptance_rate = len(accepted) / max(1, len(proposed))
        ess = _effective_sample_size(np.ones(len(accepted)) / max(1, len(accepted)))
        entropy = float(np.log(max(1, len(accepted))))
        _adapt_proposal_scale(proposal_scale, acceptance_rate)
        round_weights = accepted[["particle_id", "score"]].copy()
        round_weights["weight"] = 1.0 / max(1, len(round_weights))
        round_weights["accepted"] = True
        round_noise = _mc_noise_report(accepted, round_weights, artifacts, obs, round_cells, seed + smc_round * 10_003, workers=worker_count)
        if not round_noise.empty:
            round_noise["round"] = smc_round
            mc_rows.append(round_noise)
            next_round_cells = int(round_noise["n_sim_cells_next"].astype(int).max())
        else:
            next_round_cells = int(cells)
        _progress(
            f"SMC round {smc_round + 1}/{smc_rounds} result: "
            f"accepted={len(accepted)}/{len(proposed)} ({acceptance_rate:.1%}), "
            f"epsilon={_fmt(epsilon)}, median_score={_fmt(scores['score'].median())}, "
            f"median_D_ddPCR={_fmt(scores['D_ddpcr'].median())}, "
            f"median_D_cellcount={_fmt(scores['D_cell_count'].median())}, "
            f"median_D_flow3={_fmt(scores['D_flow3'].median())}, "
            f"median_D_prior={_fmt(scores['D_prior'].median())}, next_cells={next_round_cells}"
        )
        adaptation_rows.append(
            {
                "round": smc_round,
                "n_sim_cells": round_cells,
                "n_sim_cells_next": next_round_cells,
                "epsilon": epsilon,
                "acceptance_rate": acceptance_rate,
                "ESS": ess,
                "particle_entropy": entropy,
                "proposal_scale_growth": proposal_scale["growth"],
                "proposal_scale_copy_MYC": proposal_scale["copy_MYC"],
                "proposal_scale_copy_CDK4": proposal_scale["copy_CDK4"],
                "proposal_scale_copy_PDGFRA": proposal_scale["copy_PDGFRA"],
                "proposal_scale_flow3": proposal_scale["flow3"],
                "median_distance": float(scores["score"].median()),
                "median_D_ddPCR": float(scores["D_ddpcr"].median()),
                "median_D_cellcount": float(scores["D_cell_count"].median()),
                "median_D_flow3": float(scores["D_flow3"].median()),
                "median_D_prior": float(scores["D_prior"].median()),
            }
        )
        all_params.append(proposed)
        all_scores.append(scores)
        current = _with_prior_distance(accepted.drop(columns=[column for column in ("round", "score", "D_ddpcr", "D_cell_count", "D_flow3", "D_prior", "D_biology", "early_rejected", "simulated_full") if column in accepted.columns]), artifacts)

    params = pd.concat(all_params, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    final_round = int(scores["round"].max())
    final_scores = scores[scores["round"] == final_round]
    cutoff = float(final_scores["score"].quantile(float(acceptance_quantile)))
    final_gate_ids = set(current["particle_id"].astype(int)) if "particle_id" in current else set()
    weights = _weights_from_scores(scores, final_round, cutoff, final_gate_ids)
    particle_scores = scores.merge(weights[["particle_id", "weight", "accepted"]], on="particle_id", how="left")
    accepted_scores = weights.loc[weights["accepted"], "score"].astype(float)
    _progress(
        f"full SMC final weights: accepted={int(weights['accepted'].sum())}, "
        f"cutoff={_fmt(cutoff)}, median_accepted_score={_fmt(accepted_scores.median())}, "
        f"best_accepted_score={_fmt(accepted_scores.min())}"
    )
    coarse_scores = scores[scores["round"] == 0].copy()
    coarse_particles = params[params["round"] == 0].copy()
    write_table(coarse_particles, out / "FULL_coarse_particles.parquet")
    write_table(coarse_scores, out / "FULL_coarse_scores.parquet")
    write_table(params, out / "FULL_particle_parameters.parquet")
    write_table(weights, out / "FULL_particle_weights.parquet")
    write_table(particle_scores, out / "FULL_particle_scores.parquet")
    write_table(pd.DataFrame(adaptation_rows), out / "FULL_smc_adaptation_log.parquet")
    write_table(pd.DataFrame(early_rows), out / "FULL_early_rejection_log.parquet")
    final_noise = _mc_noise_report(params, weights, artifacts, obs, int(cells), seed, workers=worker_count)
    final_noise["round"] = "final"
    if mc_rows:
        write_table(pd.concat([*mc_rows, final_noise], ignore_index=True), out / "FULL_monte_carlo_noise_report.csv")
    else:
        write_table(final_noise, out / "FULL_monte_carlo_noise_report.csv")
    _progress(f"full SMC writing accepted particle histories: output={out / 'FULL_particles_final.zarr'}")
    _write_particle_zarr(out / "FULL_particles_final.zarr", params, weights, artifacts, obs, accepted_only=True, cells=int(cells), seed=seed, workers=worker_count)
    _progress(f"full SMC writing replay histories: output={out / 'FULL_replay_histories.zarr'}")
    _write_particle_zarr(
        out / "FULL_replay_histories.zarr",
        params,
        weights,
        artifacts,
        obs,
        accepted_only=True,
        replay=True,
        cells=_replay_cell_count(int(cells)),
        seed=seed + 10_000,
        replay_repetitions=_replay_repetitions(int(cells)),
        workers=worker_count,
    )
    write_json(
        out / "full_reconstruction_manifest.json",
        {
            "method_source": "markdown/fit_method.md",
            "fit_mask": schemas.FIT_MASK,
            "method_defaults": {
                "moment_n_candidates": METHOD_MOMENT_CANDIDATES,
                "moment_keep_top": METHOD_MOMENT_KEEP_TOP,
                "coarse_n_particles": METHOD_COARSE_PARTICLES,
                "coarse_n_sim_cells": METHOD_COARSE_CELLS,
                "standard_n_particles_initial": METHOD_FULL_PARTICLES_INITIAL,
                "standard_n_particles_final": METHOD_FULL_PARTICLES_FINAL,
                "n_sim_cells_fit": METHOD_N_SIM_FIT,
                "n_sim_cells_replay": METHOD_N_SIM_REPLAY,
                "fitting_simulator_dt": 0.20,
                "workers": worker_count,
            },
            "smc_features": ["moment_prescreen", "coarse_full", "adaptive_tolerance", "adaptive_proposal", "early_rejection", "blockwise_update", "common_random_numbers", "monte_carlo_noise_controller", "hard_gate_no_fallback", "core_full_simulator_replay", "threaded_particle_evaluation"],
            "active_controls_only": ["net_growth_rate", "bulk_copy_velocity", "flow3_projection_bias"],
            "nuisance_not_data_identified": ["division_death_turnover", "ecDNA_gain_loss_turnover", "hidden_npc_opc_split"],
            "initial_copy_distribution": "mean-matched ZINB prior; current data do not directly identify single-cell copy-number shape",
        },
    )
    _progress(f"full SMC done: output={out}")
    return {name: out / name for name in schemas.FULL_OUTPUTS}


def aggregate_accepted_histories(full_dir: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    weights = pd.read_parquet(Path(full_dir) / "FULL_particle_weights.parquet")
    summary = pd.DataFrame(
        {
            "scenario_class": ["bulk-compatible-history-ensemble"],
            "posterior_weight": [float(weights.loc[weights["accepted"], "weight"].sum())],
            "particles": [int(weights["accepted"].sum())],
            "median_score": [float(weights.loc[weights["accepted"], "score"].median()) if weights["accepted"].any() else float(weights["score"].median())],
        }
    )
    if output_dir is not None:
        write_table(summary, ensure_dir(output_dir) / "accepted_history_summary.parquet")
    return summary


def _candidate_parameter_table(artifacts: dict, prior: pd.DataFrame, rng: np.random.Generator, n_candidates: int) -> pd.DataFrame:
    centers = _centers(artifacts)
    rows = []
    for particle_id in range(int(n_candidates)):
        prior_row = prior.iloc[int(rng.integers(0, len(prior)))] if len(prior) else {}
        row = {"particle_id": particle_id, "random_stream_id": particle_id}
        for name, center in centers.items():
            scale = 0.10 if name.startswith("r__") else 0.08
            lo, hi = _active_bounds(name)
            row[name] = float(np.clip(center + rng.normal(0.0, scale), lo, hi))
        row["division_death_turnover"] = float(getattr(prior_row, "division_death_turnover", rng.lognormal(0.0, 0.75)))
        row["ecDNA_gain_loss_turnover"] = float(getattr(prior_row, "ecDNA_gain_loss_turnover", rng.lognormal(0.0, 0.75)))
        row["hidden_npc_opc_split"] = float(rng.beta(2, 2))
        row["D_prior"] = _prior_distance(row, artifacts)
        rows.append(row)
    return pd.DataFrame(rows)


def _score_candidate_moments(candidates: pd.DataFrame, artifacts: dict, workers: int = 1) -> pd.DataFrame:
    chunks = _dataframe_chunks(candidates, _resolve_workers(workers, len(candidates)))
    scored = _parallel_map(lambda chunk: _score_candidate_moments_chunk(chunk, artifacts), chunks, workers)
    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame(columns=["particle_id", "D_moment", "D_count", "D_ddPCR", "D_flow3", "D_prior", "D_biology"])


def _score_candidate_moments_chunk(candidates: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    rows = []
    centers = _centers(artifacts)
    for _, row in candidates.iterrows():
        d_count = 0.0
        d_copy = 0.0
        for name, center in centers.items():
            value = float(row.get(name, center))
            if name.startswith("r__"):
                d_count += ((value - center) / 0.25) ** 2
            elif name.startswith("v__"):
                d_copy += ((value - center) / 0.20) ** 2
        row_dict = row.to_dict()
        d_prior = _prior_distance(row_dict, artifacts)
        d_biology = _biology_distance(row_dict)
        rows.append({"particle_id": int(row["particle_id"]), "D_moment": d_count + d_copy + d_prior + d_biology, "D_count": d_count, "D_ddPCR": d_copy, "D_flow3": 0.0, "D_prior": d_prior, "D_biology": d_biology})
    return pd.DataFrame(rows)


def _load_moment_particles(moment_dir: str | Path | None, artifacts: dict, rng: np.random.Generator, particles: int) -> pd.DataFrame:
    if moment_dir is not None and (Path(moment_dir) / "MOMENT_keep_top_particles.parquet").exists():
        table = pd.read_parquet(Path(moment_dir) / "MOMENT_keep_top_particles.parquet")
        if len(table):
            return table.head(int(particles)).copy()
    prior = pd.DataFrame({"particle_id": range(int(particles)), "D_prior": np.zeros(int(particles))})
    return _candidate_parameter_table(artifacts, prior, rng, particles)


def _centers(artifacts: dict) -> dict[str, float]:
    result = {}
    growth = artifacts["growth_velocity"]
    for (condition, phase), group in growth.groupby(["condition", "phase"], dropna=False):
        result[f"r__{condition}__p{int(phase)}"] = float(group["r_center"].astype(float).median())
    copy = artifacts["copy_velocity"]
    for (condition, species, phase), group in copy.groupby(["condition", "species", "phase"], dropna=False):
        result[f"v__{condition}__{species}__p{int(phase)}"] = float(group["v_center"].astype(float).median())
    for phase in schemas.PHASES:
        result[f"zeta_flow3__p{phase}"] = 0.0
    return result


def _propose_round(current: pd.DataFrame, artifacts: dict, rng: np.random.Generator, particles: int, smc_round: int, proposal_scale: dict[str, float]) -> pd.DataFrame:
    centers = _centers(artifacts)
    rows = []
    block = _block_for_round(smc_round)
    for local_id in range(int(particles)):
        parent = current.iloc[int(rng.integers(0, len(current)))] if len(current) else pd.Series(dtype=float)
        row = {
            "particle_id": smc_round * int(particles) + local_id,
            "round": smc_round,
            "random_stream_id": int(parent.get("random_stream_id", parent.get("particle_id", local_id))),
            "updated_block": block,
        }
        for name, center in centers.items():
            base = float(parent.get(name, center))
            lo, hi = _active_bounds(name)
            if _name_in_block(name, block):
                scale = 0.05 * _scale_for_name(name, proposal_scale)
                row[name] = float(np.clip(base + rng.standard_t(5) * scale, lo, hi))
            else:
                row[name] = float(np.clip(base, lo, hi))
        row["division_death_turnover"] = float(parent.get("division_death_turnover", rng.lognormal(0.0, 0.75)))
        row["ecDNA_gain_loss_turnover"] = float(parent.get("ecDNA_gain_loss_turnover", rng.lognormal(0.0, 0.75)))
        row["hidden_npc_opc_split"] = float(np.clip(parent.get("hidden_npc_opc_split", rng.beta(2, 2)), 0.0, 1.0))
        row["D_prior"] = _prior_distance(row, artifacts)
        rows.append(row)
    return pd.DataFrame(rows)


def _with_prior_distance(params: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    result = params.copy()
    if result.empty:
        return result
    result["D_prior"] = [_prior_distance(row.to_dict(), artifacts) for _, row in result.iterrows()]
    if "random_stream_id" not in result:
        result["random_stream_id"] = result["particle_id"].astype(int) if "particle_id" in result else range(len(result))
    return result


def _score_proposed_particle(task: tuple[dict, dict, dict, float, int, int, int]) -> tuple[dict, list[dict]]:
    param_dict, artifacts, obs, epsilon, cells, seed, smc_round = task
    param = pd.Series(param_dict)
    score_params = dict(param_dict)
    score_params["D_prior"] = float(score_params.get("D_prior", _prior_distance(score_params, artifacts)))
    score_params["D_biology"] = _biology_distance(score_params)
    screen_rows, screen_score = _early_rejection_screen(param, artifacts, obs, epsilon, score_params)
    if screen_score is not None:
        return (
            {
                "particle_id": int(param_dict["particle_id"]),
                "round": int(smc_round),
                "score": screen_score["score"],
                "early_rejected": True,
                "simulated_full": False,
                **{f"D_{k}": v for k, v in screen_score["contributions"].items()},
            },
            screen_rows,
        )
    predictions = _predict_from_particle(param, artifacts, obs, cells=cells, seed=seed)
    score = score_bulk_predictions(predictions, obs, score_params)
    return (
        {
            "particle_id": int(param_dict["particle_id"]),
            "round": int(smc_round),
            "score": score["score"],
            "early_rejected": False,
            "simulated_full": True,
            **{f"D_{k}": v for k, v in score["contributions"].items()},
        },
        screen_rows,
    )


def _prior_distance(row: dict | pd.Series, artifacts: dict) -> float:
    centers = _centers(artifacts)
    prior_scales = artifacts.get("prior_scales", {})
    r_sd = max(0.05, float(prior_scales.get("r_center_sd", 0.25)))
    v_sd = max(0.05, float(prior_scales.get("v_center_sd", 0.20)))
    flow_sd = max(0.02, float(prior_scales.get("flow3_bias_sd", 0.05)))
    total = 0.0
    for name, center in centers.items():
        value = float(row.get(name, center))
        if name.startswith("r__"):
            scale = r_sd
        elif name.startswith("v__"):
            scale = v_sd
        else:
            scale = flow_sd
        total += ((value - float(center)) / scale) ** 2
    tau_n = max(1e-9, float(row.get("division_death_turnover", 1.0)))
    tau_k = max(1e-9, float(row.get("ecDNA_gain_loss_turnover", 1.0)))
    rho = float(row.get("hidden_npc_opc_split", 0.5))
    rho_sd = np.sqrt(2.0 * 2.0 / ((2.0 + 2.0) ** 2 * (2.0 + 2.0 + 1.0)))
    total += (np.log(tau_n) / 0.75) ** 2
    total += (np.log(tau_k) / 0.75) ** 2
    total += ((rho - 0.5) / rho_sd) ** 2
    return float(total)


def _block_for_round(smc_round: int) -> str:
    blocks = ("growth", "copy_MYC", "copy_CDK4", "copy_PDGFRA", "flow3")
    return blocks[int(smc_round) % len(blocks)]


def _name_in_block(name: str, block: str) -> bool:
    if block == "growth":
        return name.startswith("r__")
    if block == "flow3":
        return name.startswith("zeta_flow3__")
    if block.startswith("copy_"):
        species = block.removeprefix("copy_")
        return name.startswith("v__") and f"__{species}__" in name
    return False


def _active_bounds(name: str) -> tuple[float, float]:
    if name.startswith("r__"):
        return -3.0, 3.0
    if name.startswith("v__"):
        return -1.5, 1.5
    if name.startswith("zeta_flow3__"):
        return -0.25, 0.25
    return -np.inf, np.inf


def _biology_distance(row: dict | pd.Series) -> float:
    tau_n = float(row.get("division_death_turnover", 0.0))
    tau_k = float(row.get("ecDNA_gain_loss_turnover", 0.0))
    rho = float(row.get("hidden_npc_opc_split", 0.5))
    r_values = [abs(float(value)) for key, value in row.items() if str(key).startswith("r__")]
    v_values = [abs(float(value)) for key, value in row.items() if str(key).startswith("v__")]
    if tau_n < 0.0 or tau_n > 8.0:
        return float("inf")
    if tau_k < 0.0 or tau_k > 10.0:
        return float("inf")
    if r_values and tau_n < max(r_values):
        return float("inf")
    if v_values and tau_k < max(v_values):
        return float("inf")
    if rho < 0.0 or rho > 1.0:
        return float("inf")
    return 0.0


def _predict_from_particle(param: pd.Series, artifacts: dict, obs: dict, *, cells: int, seed: int) -> dict[str, pd.DataFrame]:
    return _simulate_particle_full(param, artifacts, obs, cells=cells, seed=seed)["predictions"]


def _early_rejection_screen(param: pd.Series, artifacts: dict, obs: dict, epsilon: float, score_params: dict) -> tuple[list[dict], dict | None]:
    if not np.isfinite(epsilon):
        return [], None
    predictions = _predict_moment_predictions(param, artifacts, obs)
    rows = []
    cumulative = {"ddpcr": 0.0, "cell_count": 0.0, "flow3": 0.0}
    data_distance = 0.0
    weeks = sorted(set(predictions["cell_count"]["week"].astype(int)).union(set(predictions["ddpcr"]["week"].astype(int))))
    for week in weeks:
        subset = {name: table[table["week"].astype(int) == week].copy() for name, table in predictions.items()}
        partial = score_bulk_predictions(subset, obs, {"D_prior": 0.0, "D_biology": 0.0})
        for key in cumulative:
            cumulative[key] += float(partial["contributions"][key])
        data_distance += float(partial["score"])
        total = data_distance + float(score_params["D_prior"]) + float(score_params["D_biology"])
        rejected = total > float(epsilon)
        rows.append(
            {
                "particle_id": int(param.particle_id),
                "round": int(param.get("round", -1)),
                "week": int(week),
                "partial_data_distance": data_distance,
                "partial_total_distance": total,
                "epsilon": float(epsilon),
                "early_rejected": bool(rejected),
                "simulated_full": False,
                "screen_stage": "moment_pre_full",
            }
        )
        if rejected:
            return rows, {
                "score": float(total),
                "contributions": {
                    "ddpcr": cumulative["ddpcr"],
                    "cell_count": cumulative["cell_count"],
                    "flow3": cumulative["flow3"],
                    "prior": float(score_params["D_prior"]),
                    "biology": float(score_params["D_biology"]),
                },
            }
    return rows, None


def _predict_moment_predictions(param: pd.Series, artifacts: dict, obs: dict) -> dict[str, pd.DataFrame]:
    dd_rows = []
    for target in artifacts["ddpcr"].itertuples(index=False):
        dd_rows.append({**target._asdict(), "observed_bulk_mean": float(target.bulk_mean), "predicted_bulk_mean": _moment_ddpcr_value(param, artifacts, target)})
    cc_rows = []
    for target in artifacts["cell_count"].itertuples(index=False):
        cc_rows.append({**target._asdict(), "observed_cell_count": float(target.total_cell_count), "predicted_cell_count": _moment_count_value(param, artifacts, target)})
    flow_target = obs["flow3"]["target"]["fractions"]
    strata = artifacts["cell_count"][["week", "condition", "replicate"]].drop_duplicates()
    flow_rows = []
    for row in strata.itertuples(index=False):
        phase = schemas.phase_for_week(int(row.week))
        bias = float(param.get(f"zeta_flow3__p{phase}", 0.0))
        raw = np.asarray([flow_target[group] for group in schemas.FLOW3_GROUPS], dtype=float)
        raw[0] += bias
        raw[1:] -= bias / 2.0
        projected = schemas.normalize_probabilities(np.clip(raw, 1e-6, None), name="moment flow3 projection")
        for group, fraction in zip(schemas.FLOW3_GROUPS, projected):
            flow_rows.append({"week": int(row.week), "condition": row.condition, "replicate": row.replicate, "group": group, "target_fraction": float(flow_target[group]), "predicted_fraction": float(fraction)})
    return {"ddpcr": pd.DataFrame(dd_rows), "cell_count": pd.DataFrame(cc_rows), "flow3": pd.DataFrame(flow_rows)}


def _moment_ddpcr_value(param: pd.Series, artifacts: dict, target) -> float:
    group = artifacts["ddpcr"][(artifacts["ddpcr"]["condition"] == target.condition) & (artifacts["ddpcr"]["replicate"] == target.replicate) & (artifacts["ddpcr"]["species"] == target.species)].sort_values("week")
    current = float(group["bulk_mean"].iloc[0])
    first_week = int(group["week"].iloc[0])
    for week in range(first_week + 1, int(target.week) + 1):
        current *= np.exp(float(param.get(f"v__{target.condition}__{target.species}__p{schemas.phase_for_week(week - 1)}", 0.0)))
    return float(current)


def _moment_count_value(param: pd.Series, artifacts: dict, target) -> float:
    group = artifacts["cell_count"][(artifacts["cell_count"]["condition"] == target.condition) & (artifacts["cell_count"]["replicate"] == target.replicate)].sort_values("week")
    current = float(group["total_cell_count"].iloc[0])
    first_week = int(group["week"].iloc[0])
    for week in range(first_week + 1, int(target.week) + 1):
        current *= np.exp(float(param.get(f"r__{target.condition}__p{schemas.phase_for_week(week - 1)}", 0.0)))
    return float(current)


def _simulate_particle_full(param: pd.Series, artifacts: dict, obs: dict, *, cells: int, seed: int) -> dict[str, dict | pd.DataFrame]:
    dd = artifacts["ddpcr"].copy()
    cc = artifacts["cell_count"].copy()
    flow_target = obs["flow3"]["target"]["fractions"]
    pred_dd: list[dict] = []
    pred_cc: list[dict] = []
    flow_rows: list[dict] = []
    history_rows: list[dict] = []
    event_rows: list[dict] = []

    strata = cc[["condition", "replicate"]].drop_duplicates()
    for srow in strata.itertuples(index=False):
        condition = str(srow.condition)
        replicate = str(srow.replicate)
        cc_group = cc[(cc["condition"] == condition) & (cc["replicate"] == replicate)].sort_values("week")
        dd_group = dd[(dd["condition"] == condition) & (dd["replicate"] == replicate)].sort_values(["species", "week"])
        if cc_group.empty or dd_group.empty:
            continue
        first_week = int(min(cc_group["week"].min(), dd_group["week"].min()))
        weeks = sorted(set(cc_group["week"].astype(int)).union(set(dd_group["week"].astype(int))))
        record_times = tuple(float(week - first_week) for week in weeks)
        stream_id = int(param.get("random_stream_id", param.get("particle_id", 0)))
        sim_seed = int(seed + stream_id * 1009 + _stable_seed_offset(condition, replicate))
        model_params = _core_model_parameters(param, condition, max(record_times), record_times, max(1, int(cells)), sim_seed)
        initialization = _core_initialization(param, dd_group, flow_target, first_week, max(1, int(cells)), sim_seed)
        result = run_simulation(
            params=model_params,
            observation_params=cfg.DEFAULT_OBSERVATION_PARAMETERS,
            initialization=initialization,
            t_max=max(record_times),
            n_init=max(1, int(cells)),
            record_times=record_times,
            seed=sim_seed,
            verbose=False,
        )
        snapshots = _snapshots_by_week(result, first_week)
        initial_count = float(cc_group.loc[cc_group["week"].astype(int) == first_week, "total_cell_count"].median())
        initial_pop = max(1.0, float(_snapshot_at_week(snapshots, first_week)["population_size"]))
        population_weight = initial_count / initial_pop
        for row in cc_group.itertuples(index=False):
            snap = _snapshot_at_week(snapshots, int(row.week))
            pred_cc.append({**row._asdict(), "observed_cell_count": float(row.total_cell_count), "predicted_cell_count": float(snap["population_size"]) * population_weight})
        for row in dd_group.itertuples(index=False):
            snap = _snapshot_at_week(snapshots, int(row.week))
            species_idx = schemas.SPECIES.index(str(row.species))
            pred_dd.append({**row._asdict(), "observed_bulk_mean": float(row.bulk_mean), "predicted_bulk_mean": float(snap["bulk_copy_means"][species_idx])})
        for week, snap in snapshots.items():
            projected = _project_flow3(np.asarray(snap["soft_state_fractions"], dtype=float), float(param.get(f"zeta_flow3__p{schemas.phase_for_week(week)}", 0.0)))
            for group, fraction in zip(schemas.FLOW3_GROUPS, projected):
                flow_rows.append({"week": int(week), "condition": condition, "replicate": replicate, "group": group, "target_fraction": float(flow_target[group]), "predicted_fraction": float(fraction)})
            history_rows.append(
                {
                    "particle_id": int(param.get("particle_id", 0)),
                    "week": int(week),
                    "condition": condition,
                    "replicate": replicate,
                    "population_size": int(snap["population_size"]),
                    "mean_division_hazard": float(snap["mean_division_hazard"]),
                    "mean_death_hazard": float(snap["mean_death_hazard"]),
                    "bulk_MYC": float(snap["bulk_copy_means"][0]),
                    "bulk_CDK4": float(snap["bulk_copy_means"][1]),
                    "bulk_PDGFRA": float(snap["bulk_copy_means"][2]),
                    "flow3_OLIG2_high": float(projected[0]),
                    "flow3_AC": float(projected[1]),
                    "flow3_MES": float(projected[2]),
                    "metadata": "current data do not directly identify this quantity",
                }
            )
        for event_time, event_type, cell_id, details in result.events:
            event_rows.append(
                {
                    "particle_id": int(param.get("particle_id", 0)),
                    "condition": condition,
                    "replicate": replicate,
                    "week_time": float(first_week + event_time),
                    "event_type": str(event_type),
                    "cell_id": int(cell_id),
                    "details": str(details),
                    "metadata": "current data do not directly identify this quantity",
                }
            )
    return {
        "predictions": {"ddpcr": pd.DataFrame(pred_dd), "cell_count": pd.DataFrame(pred_cc), "flow3": pd.DataFrame(flow_rows)},
        "history": pd.DataFrame(history_rows),
        "events": pd.DataFrame(event_rows),
    }


def _core_model_parameters(param: pd.Series, condition: str, t_max: float, record_times: tuple[float, ...], cells: int, seed: int) -> cfg.ModelParameters:
    base = cfg.DEFAULT_MODEL_PARAMETERS
    r_value = _mean_control(param, f"r__{condition}__p")
    tau_n = max(abs(r_value) + 1e-6, float(param.get("division_death_turnover", abs(r_value) + 1.0)))
    hazard = replace(base.hazard, lambda_div_ceiling=float((tau_n + r_value) / 2.0), lambda_death_ceiling=float((tau_n - r_value) / 2.0))
    tau_k = float(param.get("ecDNA_gain_loss_turnover", 1.0))
    turnover: dict[str, cfg.TurnoverSpeciesParameters] = {}
    for species in schemas.SPECIES:
        v_value = _mean_control(param, f"v__{condition}__{species}__p")
        total = max(abs(v_value) + 1e-6, tau_k)
        species_params = base.turnover[species]
        turnover[species] = replace(species_params, gain_ceiling=float((total + v_value) / 2.0), loss_ceiling=float((total - v_value) / 2.0))
    simulation = replace(
        base.simulation,
        dt=0.20,
        t_max=float(t_max),
        record_times=record_times,
        n_init=int(cells),
        target_population_size=None,
        max_pop_size=max(int(cells) * 50, int(cells) + 10),
        random_seed=int(seed),
        fitting_mode=True,
        record_full_snapshots=False,
        record_events=True,
    )
    return replace(base, hazard=hazard, turnover=turnover, simulation=simulation)


def _core_initialization(param: pd.Series, dd_group: pd.DataFrame, flow_target: dict, first_week: int, cells: int, seed: int) -> cfg.InitializationParameters:
    rng = np.random.default_rng(seed)
    first = dd_group[dd_group["week"].astype(int) == int(first_week)]
    mean_vector = _anchor_mean_vector(first if not first.empty else dd_group, str(dd_group["condition"].iloc[0]), str(dd_group["replicate"].iloc[0]))
    rho = float(np.clip(param.get("hidden_npc_opc_split", 0.5), 0.0, 1.0))
    flow = np.asarray([rho * float(flow_target["OLIG2-high"]), (1.0 - rho) * float(flow_target["OLIG2-high"]), float(flow_target["AC"]), float(flow_target["MES"])], dtype=float)
    flow = schemas.normalize_probabilities(flow, name="core initialization flow")
    rows_per_state = max(4, min(64, int(cells)))
    distributions = {}
    for state in cfg.STATE_NAMES:
        distributions[state] = _mean_matched_zinb_pool(rng, mean_vector, rows_per_state)
    return replace(
        cfg.DEFAULT_INITIALIZATION_PARAMETERS,
        mode=cfg.EMPIRICAL_WEEK1,
        empirical_flow_fractions=flow,
        empirical_sorted_copy_distributions=distributions,
        empirical_soft_state_concentration=25.0,
    )


def _mean_control(param: pd.Series, prefix: str) -> float:
    values = [float(param.get(f"{prefix}{phase}", 0.0)) for phase in schemas.PHASES if f"{prefix}{phase}" in param]
    return float(np.mean(values)) if values else 0.0


def _anchor_mean_vector(ddpcr_anchor: pd.DataFrame, condition: str, replicate: str) -> np.ndarray:
    subset = ddpcr_anchor[(ddpcr_anchor["condition"].astype(str) == condition) & (ddpcr_anchor["replicate"].astype(str) == replicate)]
    if subset.empty:
        subset = ddpcr_anchor
    means = []
    value_column = "bulk_mean" if "bulk_mean" in subset.columns else "ddpcr_copy_number"
    for species in schemas.SPECIES:
        species_rows = subset[subset["species"].astype(str) == species]
        value = float(species_rows[value_column].median()) if not species_rows.empty else float(subset[value_column].median())
        means.append(max(0.0, value))
    return np.asarray(means, dtype=float)


def _mean_matched_zinb_pool(rng: np.random.Generator, mean_vector: np.ndarray, rows: int) -> np.ndarray:
    n_rows = max(1, int(rows))
    means = np.clip(np.asarray(mean_vector, dtype=float), 0.0, None)
    matrix = np.zeros((n_rows, len(schemas.SPECIES)), dtype=int)
    pi0 = rng.beta(1.5, 1.5, size=len(schemas.SPECIES))
    phi = rng.lognormal(np.log(2.0), 0.75, size=len(schemas.SPECIES))
    for idx, mean in enumerate(means):
        if mean <= 0.0:
            continue
        positive_mean = mean / max(1e-6, 1.0 - float(pi0[idx]))
        gamma_rate = rng.gamma(shape=float(phi[idx]), scale=positive_mean / max(1e-6, float(phi[idx])), size=n_rows)
        values = rng.poisson(gamma_rate).astype(int)
        values[rng.random(n_rows) < float(pi0[idx])] = 0
        matrix[:, idx] = values
    return matrix


def _snapshots_by_week(result, first_week: int) -> dict[int, dict]:
    snapshots: dict[int, dict] = {}
    for time, snapshot in zip(result.times, result.truth_snapshots):
        snapshots[int(round(float(first_week) + float(time)))] = snapshot
    return snapshots


def _snapshot_at_week(snapshots: dict[int, dict], week: int) -> dict:
    if week in snapshots:
        return snapshots[week]
    earlier = [item for item in snapshots if item <= week]
    return snapshots[max(earlier)] if earlier else snapshots[min(snapshots)]


def _project_flow3(soft_state_fractions: np.ndarray, bias: float) -> np.ndarray:
    raw = np.asarray([soft_state_fractions[cfg.NPC] + soft_state_fractions[cfg.OPC], soft_state_fractions[cfg.AC], soft_state_fractions[cfg.MES]], dtype=float)
    raw[0] += float(bias)
    raw[1:] -= float(bias) / 2.0
    return schemas.normalize_probabilities(np.clip(raw, 1e-6, None), name="core projected flow3")


def _stable_seed_offset(*parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 1_000_000)


def _early_rejection_rows(param: pd.Series, predictions: dict, obs: dict, epsilon: float) -> list[dict]:
    rows = []
    if not np.isfinite(epsilon):
        return rows
    partial = 0.0
    weeks = sorted(predictions["cell_count"]["week"].astype(int).unique())
    for week in weeks:
        subset = {name: table[table["week"].astype(int) == week].copy() for name, table in predictions.items()}
        partial += score_bulk_predictions(subset, obs, param.to_dict())["score"]
        rejected = partial > epsilon
        rows.append({"particle_id": int(param.particle_id), "week": week, "partial_distance": partial, "epsilon": epsilon, "early_rejected": bool(rejected)})
        if rejected:
            break
    return rows


def _weights_from_scores(scores: pd.DataFrame, final_round: int, cutoff: float, gate_particle_ids: set[int] | None = None) -> pd.DataFrame:
    result = scores.copy()
    final_mask = result["round"] == int(final_round)
    early_rejected = result["early_rejected"].fillna(False).astype(bool) if "early_rejected" in result else pd.Series(False, index=result.index)
    gated = result["particle_id"].astype(int).isin(gate_particle_ids) if gate_particle_ids is not None else pd.Series(True, index=result.index)
    eligible = final_mask & gated & (~early_rejected)
    result["accepted"] = eligible & (result["score"] <= cutoff)
    if not bool(result["accepted"].any()):
        if not bool(eligible.any()):
            raise RuntimeError("no final particles passed data, prior, and biological gates")
        best_score = float(result.loc[eligible, "score"].astype(float).min())
        raise RuntimeError(
            "no final particles passed the final score cutoff after data, prior, and biological gates; "
            f"cutoff={float(cutoff):.6g}, best_eligible_score={best_score:.6g}"
        )
    raw = np.zeros(len(result), dtype=float)
    accepted_scores = result.loc[result["accepted"], "score"].astype(float)
    shifted = accepted_scores - float(accepted_scores.min())
    raw[result["accepted"].to_numpy()] = np.exp(-0.5 * shifted.to_numpy())
    total = float(raw.sum())
    if total <= 0.0:
        raise RuntimeError("accepted final particles produced non-positive posterior weights")
    result["weight"] = raw / total
    return result[["particle_id", "round", "score", "weight", "accepted"]]


def _write_population_zarr(path: Path, population: pd.DataFrame, ddpcr_anchor: pd.DataFrame, flow_init: pd.DataFrame) -> None:
    import zarr
    from zarr.storage import ZipStore

    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a clean output directory")
    store = ZipStore(str(path), mode="w")
    try:
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update({"method_source": "markdown/fit_method.md", "role": "FULL_initial_population", "metadata": "current data do not directly identify single-cell latent quantities"})
        root.create_dataset("particle_id", data=population["particle_id"].to_numpy(dtype=np.int64), shape=(len(population),))
        root.create_dataset("cell_id", data=population["cell_id"].to_numpy(dtype=np.int64), shape=(len(population),))
        root.create_dataset("population_weight", data=population["population_weight"].to_numpy(dtype=np.float64), shape=(len(population),))
        if {"MYC_copy", "CDK4_copy", "PDGFRA_copy"}.issubset(population.columns):
            copy_matrix = population[["MYC_copy", "CDK4_copy", "PDGFRA_copy"]].to_numpy(dtype=np.int64)
            root.create_dataset("initial_copy_numbers", data=copy_matrix, shape=copy_matrix.shape)
        root.attrs["ddpcr_anchor_rows"] = int(len(ddpcr_anchor))
        root.attrs["flow_initializer_rows"] = int(len(flow_init))
    finally:
        store.close()


def _simulate_particle_for_zarr(task: tuple[dict, dict, dict, int, int, int]) -> dict[str, pd.DataFrame]:
    param_dict, artifacts, obs, cells, sim_seed, replay_id = task
    param = pd.Series(param_dict)
    particle_id = int(param.get("particle_id", 0))
    simulated = _simulate_particle_full(param, artifacts, obs, cells=cells, seed=sim_seed)
    predictions = simulated["predictions"]
    history = simulated["history"]
    events = simulated["events"]
    return {
        "ddpcr": predictions["ddpcr"].assign(particle_id=particle_id, replay_id=replay_id),
        "cell_count": predictions["cell_count"].assign(particle_id=particle_id, replay_id=replay_id),
        "flow3": predictions["flow3"].assign(particle_id=particle_id, replay_id=replay_id),
        "history": history.assign(replay_id=replay_id) if isinstance(history, pd.DataFrame) and not history.empty else pd.DataFrame(),
        "events": events.assign(replay_id=replay_id) if isinstance(events, pd.DataFrame) and not events.empty else pd.DataFrame(),
    }


def _write_particle_zarr(
    path: Path,
    params: pd.DataFrame,
    weights: pd.DataFrame,
    artifacts: dict,
    obs: dict,
    *,
    accepted_only: bool,
    cells: int,
    seed: int,
    replay: bool = False,
    replay_repetitions: int = 1,
    workers: int = 1,
) -> None:
    import zarr
    from zarr.storage import ZipStore

    selected_ids = set(weights.loc[weights["accepted"], "particle_id"].astype(int)) if accepted_only else set(weights["particle_id"].astype(int))
    selected = params[params["particle_id"].astype(int).isin(selected_ids)].copy()
    if selected.empty:
        selected = params.head(1).copy()
    dd_rows = []
    cc_rows = []
    fl_rows = []
    history_rows = []
    event_rows = []
    simulation_tasks = [
        (row.to_dict(), artifacts, obs, int(cells), seed + int(row.particle_id) + replay_id * 100_003, replay_id)
        for _, row in selected.iterrows()
        for replay_id in range(max(1, int(replay_repetitions)))
    ]
    for result in _parallel_map(_simulate_particle_for_zarr, simulation_tasks, workers):
        dd_rows.append(result["ddpcr"])
        cc_rows.append(result["cell_count"])
        fl_rows.append(result["flow3"])
        if not result["history"].empty:
            history_rows.append(result["history"])
        if not result["events"].empty:
            event_rows.append(result["events"])
    dd = pd.concat(dd_rows, ignore_index=True)
    cc = pd.concat(cc_rows, ignore_index=True)
    fl = pd.concat(fl_rows, ignore_index=True)
    history_table = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    event_table = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a clean output directory")
    store = ZipStore(str(path), mode="w")
    try:
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update(
            {
                "method_source": "markdown/fit_method.md",
                "role": "accepted bulk-compatible full histories" if not replay else "independent final replay histories",
                "metadata": "current data do not directly identify latent single-cell quantities",
                "disabled_likelihoods": ["qpcdr", "ectag", "flow4", "state_specific_copy"],
                "full_simulator": "core.simulation.run_simulation",
                "n_sim_cells": int(cells),
                "replay_repetitions": int(replay_repetitions),
            }
        )
        root.create_dataset("ddpcr_predicted_bulk_mean", data=dd["predicted_bulk_mean"].to_numpy(dtype=np.float64), shape=(len(dd),))
        root.create_dataset("cell_count_predicted", data=cc["predicted_cell_count"].to_numpy(dtype=np.float64), shape=(len(cc),))
        root.create_dataset("flow3_predicted_fraction", data=fl["predicted_fraction"].to_numpy(dtype=np.float64), shape=(len(fl),))
        root.create_dataset("particle_id", data=selected["particle_id"].to_numpy(dtype=np.int64), shape=(len(selected),))
        if not history_table.empty:
            root.create_dataset("history_population_size", data=history_table["population_size"].to_numpy(dtype=np.int64), shape=(len(history_table),))
            root.create_dataset("history_bulk_copy_means", data=history_table[["bulk_MYC", "bulk_CDK4", "bulk_PDGFRA"]].to_numpy(dtype=np.float64), shape=(len(history_table), 3))
            root.create_dataset("history_flow3_projection", data=history_table[["flow3_OLIG2_high", "flow3_AC", "flow3_MES"]].to_numpy(dtype=np.float64), shape=(len(history_table), 3))
            root.create_dataset("history_mean_hazards", data=history_table[["mean_division_hazard", "mean_death_hazard"]].to_numpy(dtype=np.float64), shape=(len(history_table), 2))
        root.create_dataset("event_count", data=np.asarray([len(event_table)], dtype=np.int64), shape=(1,))
    finally:
        store.close()
    stem = path.stem
    write_table(dd, path.with_name(f"{stem}_ddpcr_predictions.parquet"))
    write_table(cc, path.with_name(f"{stem}_cellcount_predictions.parquet"))
    write_table(fl, path.with_name(f"{stem}_flow3_predictions.parquet"))
    if not history_table.empty:
        write_table(history_table, path.with_name(f"{stem}_history_summary.parquet"))
    if not event_table.empty:
        write_table(event_table, path.with_name(f"{stem}_event_summary.parquet"))


def _state_fractions_from_flow3(flow3: dict, rho: float) -> dict[str, float]:
    g = flow3["fractions"]
    olig2 = float(g["OLIG2-high"])
    return {"NPC-like": float(rho * olig2), "OPC-like": float((1.0 - rho) * olig2), "AC-like": float(g["AC"]), "MES-like": float(g["MES"])}


def _state_to_flow3(state: str) -> str:
    return {"NPC-like": "OLIG2-high", "OPC-like": "OLIG2-high", "AC-like": "AC", "MES-like": "MES"}[state]


def _adapt_proposal_scale(scale: dict[str, float], acceptance_rate: float) -> None:
    if acceptance_rate < 0.10:
        factor = 0.50
    elif acceptance_rate < 0.15:
        factor = 0.70
    elif acceptance_rate <= 0.35:
        factor = 1.0
    else:
        factor = 1.25
    for key in scale:
        scale[key] = float(np.clip(scale[key] * factor, 0.05, 5.0))


def _scale_for_name(name: str, scale: dict[str, float]) -> float:
    if name.startswith("r__"):
        return scale["growth"]
    if name.startswith("zeta_flow3__"):
        return scale["flow3"]
    for species in schemas.SPECIES:
        if f"__{species}__" in name:
            return scale[f"copy_{species}"]
    return 1.0


def _mc_noise_report(params: pd.DataFrame, weights: pd.DataFrame, artifacts: dict, obs: dict, cells: int, seed: int, workers: int = 1) -> pd.DataFrame:
    accepted_ids = set(weights.loc[weights["accepted"], "particle_id"].astype(int))
    selected = params[params["particle_id"].astype(int).isin(accepted_ids)].head(3 if cells >= METHOD_N_SIM_FIT else 1)
    if selected.empty:
        selected = params.head(1)
    repeats = 5 if cells >= METHOD_N_SIM_FIT else 2
    rows = []
    tasks = [
        (param.to_dict(), artifacts, obs, max(1, int(cells)), seed + 50_000 + repeat * 997 + int(param.particle_id), repeat)
        for _, param in selected.iterrows()
        for repeat in range(repeats)
    ]
    for task_rows in _parallel_map(_mc_noise_task, tasks, workers):
        rows.extend(task_rows)
    raw = pd.DataFrame(rows)
    summary_rows = []
    for observable, group in raw.groupby("observable", dropna=False):
        if observable == "cell_count":
            threshold = 0.5 * float(obs["cell_count"]["log_sd"])
        else:
            species = str(observable).removeprefix("ddpcr_")
            threshold = 0.5 * float(obs["ddpcr"]["log_sd_by_species"].get(species, obs["ddpcr"]["default_log_sd"]))
        sd = float(group["log_value"].astype(float).std(ddof=0))
        if sd > threshold:
            next_cells = min(30_000, max(int(cells) * 3, int(cells) + 1))
        elif sd < threshold * 0.25:
            next_cells = max(1_000 if cells >= METHOD_N_SIM_FIT else 1, int(cells) // 2)
        else:
            next_cells = int(cells)
        summary_rows.append({"observable": observable, "estimated_mc_sd_log": sd, "threshold": threshold, "n_sim_cells_current": int(cells), "n_sim_cells_next": int(next_cells), "repeats": repeats, "particles_checked": int(selected["particle_id"].nunique())})
    return pd.DataFrame(summary_rows)


def _mc_noise_task(task: tuple[dict, dict, dict, int, int, int]) -> list[dict]:
    param_dict, artifacts, obs, cells, sim_seed, repeat = task
    param = pd.Series(param_dict)
    particle_id = int(param.get("particle_id", 0))
    simulated = _simulate_particle_full(param, artifacts, obs, cells=cells, seed=sim_seed)
    dd = simulated["predictions"]["ddpcr"]
    cc = simulated["predictions"]["cell_count"]
    rows = []
    for species, group in dd.groupby("species", dropna=False):
        rows.append({"particle_id": particle_id, "repeat": repeat, "observable": f"ddpcr_{species}", "log_value": float(np.log(group["predicted_bulk_mean"].astype(float).clip(lower=1e-9)).mean())})
    rows.append({"particle_id": particle_id, "repeat": repeat, "observable": "cell_count", "log_value": float(np.log(cc["predicted_cell_count"].astype(float).clip(lower=0.0) + 1.0).mean())})
    return rows


def _replay_cell_count(cells: int) -> int:
    if int(cells) >= METHOD_N_SIM_FIT:
        return METHOD_N_SIM_REPLAY
    return max(int(cells), int(cells) * 2)


def _replay_repetitions(cells: int) -> int:
    return 3 if int(cells) >= METHOD_N_SIM_FIT else 1


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    denom = float(np.sum(p * p))
    return float(1.0 / denom) if denom > 0 else 0.0
