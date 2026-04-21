"""
Synthetic-likelihood objective for the staged fitting shell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecdna_model import config as cfg
from ecdna_model.fit.data import CanonicalFitDataset
from ecdna_model.fit.parameter_registry import FitParameterSpec, ParameterBoundsError, ParameterBundle, ParameterRegistry
from ecdna_model.fit.simulation_runner import FitSimulationRunner, SimulationRunSet
from ecdna_model.fit.summary import SummaryCollection, mean_summary_collection, summarize_simulation_runset
from ecdna_model.fit.transforms import ParameterVectorAdapter


INVALID_OBJECTIVE = 1e18


@dataclass(frozen=True)
class BlockLikelihoodResult:
    name: str
    dimension: int
    negative_log_likelihood: float
    residual_norm: float
    log_determinant: float


@dataclass(frozen=True)
class SyntheticLikelihoodArtifacts:
    bundle: ParameterBundle
    run_set: SimulationRunSet
    replicate_summaries: tuple[SummaryCollection, ...]
    mean_summary: SummaryCollection


@dataclass(frozen=True)
class SyntheticLikelihoodResult:
    total_objective: float
    data_negative_log_likelihood: float
    prior_penalty: float
    boundary_penalty: float
    block_results: tuple[BlockLikelihoodResult, ...]
    artifacts: SyntheticLikelihoodArtifacts | None = None


class SyntheticLikelihoodObjective:
    def __init__(
        self,
        *,
        dataset: CanonicalFitDataset,
        observed_summary: SummaryCollection,
        registry: ParameterRegistry,
        active_specs: tuple[FitParameterSpec, ...],
        runner: FitSimulationRunner,
        base_bundle: ParameterBundle,
        condition_names: tuple[str, ...],
    ):
        self.dataset = dataset
        self.observed_summary = observed_summary
        self.registry = registry
        self.active_specs = active_specs
        self.runner = runner
        self.base_bundle = base_bundle.deep_copy()
        self.condition_names = condition_names
        self.adapter = ParameterVectorAdapter(active_specs, base_bundle=self.base_bundle)

    def with_observed_summary(self, observed_summary: SummaryCollection) -> "SyntheticLikelihoodObjective":
        return SyntheticLikelihoodObjective(
            dataset=self.dataset,
            observed_summary=observed_summary,
            registry=self.registry,
            active_specs=self.active_specs,
            runner=self.runner,
            base_bundle=self.base_bundle,
            condition_names=self.condition_names,
        )

    def evaluate_vector(
        self,
        vector: np.ndarray,
        *,
        return_artifacts: bool = False,
    ) -> SyntheticLikelihoodResult:
        try:
            bundle = self.adapter.unpack_vector(vector)
        except ParameterBoundsError:
            return SyntheticLikelihoodResult(
                total_objective=INVALID_OBJECTIVE,
                data_negative_log_likelihood=INVALID_OBJECTIVE,
                prior_penalty=0.0,
                boundary_penalty=0.0,
                block_results=(),
                artifacts=None,
            )
        run_set = self.runner.run_bundle(bundle, condition_names=self.condition_names)
        replicate_summaries = summarize_simulation_runset(
            run_set,
            self.dataset,
            dynamic_only=True,
            reference=self.observed_summary,
            observed_layer=True,
        )
        mean_summary = mean_summary_collection(replicate_summaries)
        block_results: list[BlockLikelihoodResult] = []
        data_nll = 0.0
        for block_name in self.observed_summary.block_names():
            observed = self.observed_summary.blocks[block_name].values
            samples = np.stack([summary.blocks[block_name].values for summary in replicate_summaries], axis=0)
            result = _synthetic_block_likelihood(block_name, observed, samples)
            block_results.append(result)
            data_nll += result.negative_log_likelihood
        prior_penalty, boundary_penalty = self.registry.prior_penalty(bundle, self.active_specs)
        total = data_nll + prior_penalty + boundary_penalty
        artifacts = None
        if return_artifacts:
            artifacts = SyntheticLikelihoodArtifacts(
                bundle=bundle,
                run_set=run_set,
                replicate_summaries=replicate_summaries,
                mean_summary=mean_summary,
            )
        return SyntheticLikelihoodResult(
            total_objective=float(total),
            data_negative_log_likelihood=float(data_nll),
            prior_penalty=float(prior_penalty),
            boundary_penalty=float(boundary_penalty),
            block_results=tuple(block_results),
            artifacts=artifacts,
        )


def _synthetic_block_likelihood(
    block_name: str,
    observed: np.ndarray,
    samples: np.ndarray,
) -> BlockLikelihoodResult:
    cfg.require(samples.ndim == 2, f"Synthetic likelihood samples for {block_name} must be a 2D array.")
    cfg.require(samples.shape[1] == observed.size, f"Block {block_name} observed/sample dimension mismatch.")
    dimension = observed.size
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    if samples.shape[0] > 1:
        empirical_covariance = (centered.T @ centered) / float(samples.shape[0] - 1)
        empirical_variance = np.var(samples, axis=0, ddof=1)
    else:
        empirical_variance = np.maximum(1e-4, np.abs(mean) * 0.05)
        empirical_covariance = np.diag(empirical_variance)
    diagonal_target = np.diag(np.maximum(empirical_variance, 1e-4))
    shrinkage = 1.0 if samples.shape[0] <= 2 else min(0.75, dimension / max(10.0 * samples.shape[0], 1.0))
    covariance = shrinkage * diagonal_target + (1.0 - shrinkage) * empirical_covariance
    covariance = covariance + np.eye(dimension, dtype=float) * 1e-6
    chol = np.linalg.cholesky(covariance)
    diff = observed - mean
    solved = np.linalg.solve(chol, diff)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(chol))))
    negative_log_likelihood = 0.5 * (float(np.dot(solved, solved)) + log_determinant + dimension * np.log(2.0 * np.pi))
    negative_log_likelihood /= max(1, dimension)
    residual_norm = float(np.linalg.norm(diff) / np.sqrt(max(1, dimension)))
    return BlockLikelihoodResult(
        name=block_name,
        dimension=dimension,
        negative_log_likelihood=float(negative_log_likelihood),
        residual_norm=residual_norm,
        log_determinant=log_determinant,
    )
