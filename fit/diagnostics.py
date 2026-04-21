"""
Diagnostics for the staged synthetic-likelihood fitting shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ecdna_model import config as cfg
from ecdna_model.fit.objective import SyntheticLikelihoodArtifacts, SyntheticLikelihoodObjective
from ecdna_model.fit.parameter_registry import FitParameterSpec, ParameterBoundsError, ParameterRegistry
from ecdna_model.fit.summary import SummaryCollection, mean_summary_collection, summarize_simulation_runset


@dataclass(frozen=True)
class PriorPredictiveReport:
    n_draws: int
    pass_rate: float
    failures: dict[str, int]


@dataclass(frozen=True)
class PosteriorPredictiveReport:
    block_rmse: dict[str, float]
    block_relative_rmse: dict[str, float]
    block_max_abs_residual: dict[str, float]
    worst_relative_rmse: float


@dataclass(frozen=True)
class ProfilePoint:
    dimension_index: int
    offset: float
    objective_value: float


@dataclass(frozen=True)
class FakeDataRecoveryReport:
    recovered_objective: float
    normalized_error: float
    passed: bool


def run_prior_predictive(
    objective: SyntheticLikelihoodObjective,
    *,
    n_draws: int,
    seed: int,
) -> PriorPredictiveReport:
    cfg.require(n_draws > 0, "n_draws must be positive.")
    rng = np.random.default_rng(seed)
    default_vector = objective.adapter.default_vector()
    proposal_scales = objective.adapter.proposal_scales()
    failures = {
        "hard_bounds": 0,
        "population_extinction": 0,
        "population_explosion": 0,
        "state_jump": 0,
        "ectag_tail": 0,
        "qpcdr_range": 0,
    }
    n_pass = 0
    for _draw in range(n_draws):
        trial_vector = default_vector + rng.normal(scale=proposal_scales, size=proposal_scales.shape)
        try:
            bundle = objective.adapter.unpack_vector(trial_vector)
        except ParameterBoundsError:
            failures["hard_bounds"] += 1
            continue
        run_set = objective.runner.run_bundle(bundle, condition_names=objective.condition_names, seeds=objective.runner.config.seeds[:1])
        replicate_summary = summarize_simulation_runset(
            run_set,
            objective.dataset,
            dynamic_only=False,
            observed_layer=True,
        )[0]
        mean_summary = mean_summary_collection((replicate_summary,))
        failure_state = None
        for results in run_set.runs.values():
            result = results[0]
            if result.stop_reason == "population_extinction":
                failure_state = "population_extinction"
                break
            if result.stop_reason in {"max_pop_size", "target_population_size"}:
                failure_state = "population_explosion"
                break
            if result.population_sizes and max(result.population_sizes) >= int(0.95 * objective.runner.config.max_pop_size):
                failure_state = "population_explosion"
                break
        if failure_state is None and "flow_fraction" in mean_summary.blocks:
            values = mean_summary.blocks["flow_fraction"].values
            if values.size >= cfg.N_STATES * 2:
                reshaped = values.reshape(-1, cfg.N_STATES)
                jump = np.max(np.sum(np.abs(np.diff(reshaped, axis=0)), axis=1))
                if jump > 1.5:
                    failure_state = "state_jump"
        if failure_state is None and "ectag_moments" in mean_summary.blocks:
            tail_values = [
                value
                for key, value in mean_summary.blocks["ectag_moments"].as_mapping().items()
                if key.endswith("|tail_ge_16")
            ]
            if tail_values and max(tail_values) > 0.90:
                failure_state = "ectag_tail"
        if failure_state is None and "qpcdr" in mean_summary.blocks:
            max_abs_qpcdr = max(abs(value) for value in mean_summary.blocks["qpcdr"].values.tolist())
            if max_abs_qpcdr > 1e3:
                failure_state = "qpcdr_range"
        if failure_state is None:
            n_pass += 1
        else:
            failures[failure_state] += 1
    return PriorPredictiveReport(n_draws=n_draws, pass_rate=float(n_pass / n_draws), failures=failures)


def run_posterior_predictive(
    observed_summary: SummaryCollection,
    artifacts: SyntheticLikelihoodArtifacts,
) -> PosteriorPredictiveReport:
    mean_summary = artifacts.mean_summary.align_to(observed_summary)
    block_rmse: dict[str, float] = {}
    block_relative_rmse: dict[str, float] = {}
    block_max_abs_residual: dict[str, float] = {}
    for block_name in observed_summary.block_names():
        residual = mean_summary.blocks[block_name].values - observed_summary.blocks[block_name].values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        observed_scale = float(np.sqrt(np.mean(np.square(observed_summary.blocks[block_name].values))))
        observed_scale = max(observed_scale, 1e-6)
        block_rmse[block_name] = rmse
        block_relative_rmse[block_name] = float(rmse / observed_scale)
        block_max_abs_residual[block_name] = float(np.max(np.abs(residual)))
    return PosteriorPredictiveReport(
        block_rmse=block_rmse,
        block_relative_rmse=block_relative_rmse,
        block_max_abs_residual=block_max_abs_residual,
        worst_relative_rmse=max(block_relative_rmse.values(), default=0.0),
    )


def run_profile_likelihood(
    objective: SyntheticLikelihoodObjective,
    vector: np.ndarray,
    *,
    profile_scales: np.ndarray,
    n_points: int,
    max_dimensions: int,
) -> tuple[ProfilePoint, ...]:
    cfg.require(n_points >= 3, "profile likelihood requires at least 3 points.")
    cfg.require(max_dimensions > 0, "max_dimensions must be positive.")
    capped_dimensions = min(max_dimensions, vector.size)
    offsets = np.linspace(-1.0, 1.0, num=n_points, dtype=float)
    points: list[ProfilePoint] = []
    for dimension_index in range(capped_dimensions):
        for offset in offsets.tolist():
            trial = np.asarray(vector, dtype=float).copy()
            trial[dimension_index] += offset * profile_scales[dimension_index]
            objective_value = objective.evaluate_vector(trial).total_objective
            points.append(ProfilePoint(dimension_index=dimension_index, offset=float(offset), objective_value=float(objective_value)))
    return tuple(points)


def run_fake_data_recovery(
    objective: SyntheticLikelihoodObjective,
    truth_vector: np.ndarray,
    optimizer: Callable[[SyntheticLikelihoodObjective, np.ndarray, int], tuple[np.ndarray, float]],
    *,
    n_restarts: int,
) -> FakeDataRecoveryReport:
    truth_evaluation = objective.evaluate_vector(truth_vector, return_artifacts=True)
    cfg.require(truth_evaluation.artifacts is not None, "Truth evaluation must return artifacts for fake-data recovery.")
    synthetic_observed = truth_evaluation.artifacts.mean_summary
    recovery_objective = objective.with_observed_summary(synthetic_observed)
    recovered_vector, recovered_value = optimizer(recovery_objective, recovery_objective.adapter.default_vector(), n_restarts)
    scale = np.maximum(recovery_objective.adapter.proposal_scales(), 1e-6)
    normalized_error = float(np.linalg.norm((recovered_vector - truth_vector) / scale) / np.sqrt(max(1, truth_vector.size)))
    return FakeDataRecoveryReport(
        recovered_objective=float(recovered_value),
        normalized_error=normalized_error,
        passed=bool(normalized_error <= 1.5),
    )


def run_boundary_check(
    registry: ParameterRegistry,
    artifacts: SyntheticLikelihoodArtifacts,
    active_specs: tuple[FitParameterSpec, ...],
) -> list[dict[str, object]]:
    return registry.boundary_report(artifacts.bundle, active_specs)
