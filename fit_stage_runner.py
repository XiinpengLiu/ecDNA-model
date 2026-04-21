"""
Stage orchestration for the fitting shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

import config as cfg
from fit_data import CanonicalFitDataset
from fit_diagnostics import (
    FakeDataRecoveryReport,
    PosteriorPredictiveReport,
    PriorPredictiveReport,
    ProfilePoint,
    run_boundary_check,
    run_fake_data_recovery,
    run_posterior_predictive,
    run_prior_predictive,
    run_profile_likelihood,
)
from fit_objective import SyntheticLikelihoodObjective
from fit_parameter_registry import CATEGORY_STAGE1, CATEGORY_STAGE2, ParameterBundle, ParameterRegistry
from fit_simulation_runner import FitRunnerConfig, FitSimulationRunner
from fit_summary import SummaryCollection, summarize_dataset


DEFAULT_CLASSIFICATION_PATH = Path("markdown/full_v4_parameter_classification.md")


@dataclass(frozen=True)
class StageDefinition:
    name: str
    unlock_blocks: tuple[str, ...]
    condition_mode: str
    description: str


@dataclass(frozen=True)
class OptimizationSettings:
    n_restarts: int = 4
    maxiter: int = 40
    random_start_scale: float = 1.0
    prior_predictive_draws: int = 12
    prior_predictive_seed: int = 17
    min_prior_predictive_pass_rate: float = 0.25
    profile_points: int = 5
    max_profile_dimensions: int = 8
    fake_recovery_restarts: int = 2
    min_objective_improvement: float = 1e-6
    max_posterior_predictive_relative_rmse: float = 2.0
    min_profile_objective_span: float = 1e-3
    reject_on_boundary_touch: bool = True
    require_fake_data_recovery_pass: bool = True
    seed: int = 123


@dataclass
class StageFitResult:
    stage_name: str
    active_blocks: tuple[str, ...]
    condition_names: tuple[str, ...]
    objective_before: float | None
    objective_after: float | None
    active_spec_names: tuple[str, ...]
    diagnostics: dict[str, object] = field(default_factory=dict)
    skipped_reason: str | None = None
    accepted: bool = False
    rejection_reasons: tuple[str, ...] = ()
    best_vector: np.ndarray | None = None
    best_bundle: ParameterBundle | None = None


@dataclass
class StagedFitResult:
    prior_predictive: PriorPredictiveReport | None
    stage_results: list[StageFitResult]
    final_bundle: ParameterBundle


STAGE_SEQUENCE = (
    StageDefinition(
        name="stage-1A",
        unlock_blocks=("observation_core",),
        condition_mode="all",
        description="Observation layer only.",
    ),
    StageDefinition(
        name="stage-1B",
        unlock_blocks=("baseline_state_core", "baseline_turnover_core", "baseline_hazard_core", "division_core"),
        condition_mode="baseline",
        description="Baseline untreated dynamics.",
    ),
    StageDefinition(
        name="stage-1C",
        unlock_blocks=("drug_core",),
        condition_mode="drug",
        description="Drug-response core.",
    ),
    StageDefinition(
        name="stage-1D",
        unlock_blocks=("cue_core",),
        condition_mode="cue",
        description="Exogenous cue effects.",
    ),
    StageDefinition(
        name="stage-1E",
        unlock_blocks=(),
        condition_mode="all",
        description="Joint stage-1 refinement.",
    ),
    StageDefinition(
        name="stage-2A",
        unlock_blocks=("stage2_weights",),
        condition_mode="all",
        description="Burden and proliferative weights.",
    ),
    StageDefinition(
        name="stage-2B",
        unlock_blocks=("stage2_myc_plasticity",),
        condition_mode="all",
        description="MYC landscape and latent plasticity.",
    ),
    StageDefinition(
        name="stage-2C",
        unlock_blocks=("stage2_stress_survival",),
        condition_mode="all",
        description="Stress and survival subsystem.",
    ),
    StageDefinition(
        name="stage-2D",
        unlock_blocks=("stage2_cycle",),
        condition_mode="all",
        description="Cell-cycle subsystem.",
    ),
    StageDefinition(
        name="stage-2E",
        unlock_blocks=("stage2_detailed_turnover",),
        condition_mode="all",
        description="Detailed turnover terms.",
    ),
    StageDefinition(
        name="stage-2F",
        unlock_blocks=("stage2_detailed_hazard",),
        condition_mode="all",
        description="Detailed hazard terms.",
    ),
    StageDefinition(
        name="stage-2G",
        unlock_blocks=("stage2_division_daughter",),
        condition_mode="all",
        description="Division amplification and daughter initialization.",
    ),
)


class StagedFitRunner:
    def __init__(
        self,
        dataset: CanonicalFitDataset,
        *,
        classification_path: str | Path = DEFAULT_CLASSIFICATION_PATH,
        base_model: cfg.ModelParameters | None = None,
        base_observation: cfg.ObservationParameters | None = None,
        runner_config: FitRunnerConfig | None = None,
        optimization_settings: OptimizationSettings | None = None,
    ):
        self.dataset = dataset
        self.registry = ParameterRegistry.from_markdown(classification_path, dataset=dataset)
        self.current_bundle = self.registry.default_bundle(model=base_model, observation=base_observation)
        self.runner = FitSimulationRunner(dataset, config=runner_config)
        self.settings = OptimizationSettings() if optimization_settings is None else optimization_settings
        self._observed_summary_cache: dict[tuple[tuple[str, ...], bool], SummaryCollection] = {}

    def run_all_stages(self) -> StagedFitResult:
        prior_predictive = self._run_pre_fit_prior_predictive()
        stage_results: list[StageFitResult] = []
        accepted_blocks: list[str] = []
        for stage in STAGE_SEQUENCE:
            candidate_blocks = tuple(accepted_blocks + list(stage.unlock_blocks))
            stage_result = self.run_stage(stage, candidate_blocks)
            if stage_result.accepted:
                for block_name in stage.unlock_blocks:
                    if block_name not in accepted_blocks:
                        accepted_blocks.append(block_name)
            stage_results.append(stage_result)
        return StagedFitResult(
            prior_predictive=prior_predictive,
            stage_results=stage_results,
            final_bundle=self.current_bundle.deep_copy(),
        )

    def run_stage(self, stage: StageDefinition, unlocked_blocks: tuple[str, ...]) -> StageFitResult:
        condition_names = self._condition_names_for_mode(stage.condition_mode)
        if not condition_names:
            return StageFitResult(
                stage_name=stage.name,
                active_blocks=unlocked_blocks,
                condition_names=condition_names,
                objective_before=None,
                objective_after=None,
                active_spec_names=(),
                skipped_reason=f"No conditions matched mode={stage.condition_mode}.",
            )

        active_specs = self._active_specs_for_blocks(unlocked_blocks)
        if not active_specs:
            return StageFitResult(
                stage_name=stage.name,
                active_blocks=unlocked_blocks,
                condition_names=condition_names,
                objective_before=None,
                objective_after=None,
                active_spec_names=(),
                skipped_reason="No supported parameters are active for this stage.",
            )

        observed_summary = self._observed_summary(condition_names)
        objective = SyntheticLikelihoodObjective(
            dataset=self.dataset,
            observed_summary=observed_summary,
            registry=self.registry,
            active_specs=active_specs,
            runner=self.runner,
            base_bundle=self.current_bundle,
            condition_names=condition_names,
        )
        initial_vector = objective.adapter.pack_bundle(self.current_bundle)
        before = objective.evaluate_vector(initial_vector).total_objective
        best_vector, best_value = self._optimize_objective(objective, initial_vector, self.settings.n_restarts)
        artifacts_result = objective.evaluate_vector(best_vector, return_artifacts=True)
        cfg.require(artifacts_result.artifacts is not None, "Stage optimization must return artifacts.")
        posterior_predictive = run_posterior_predictive(observed_summary, artifacts_result.artifacts)
        profile = run_profile_likelihood(
            objective,
            best_vector,
            profile_scales=np.maximum(objective.adapter.proposal_scales(), 1e-6),
            n_points=self.settings.profile_points,
            max_dimensions=self.settings.max_profile_dimensions,
        )
        fake_data_recovery = run_fake_data_recovery(
            objective,
            best_vector,
            self._optimize_objective,
            n_restarts=self.settings.fake_recovery_restarts,
        )
        boundary = run_boundary_check(self.registry, artifacts_result.artifacts, active_specs)
        diagnostics: dict[str, object] = {
            "posterior_predictive": posterior_predictive,
            "profile": profile,
            "fake_data_recovery": fake_data_recovery,
            "boundary": boundary,
        }
        accepted, rejection_reasons = self._assess_stage_acceptance(
            objective_before=float(before),
            objective_after=float(best_value),
            posterior_predictive=posterior_predictive,
            profile=profile,
            fake_data_recovery=fake_data_recovery,
            boundary=boundary,
            active_specs=active_specs,
        )
        if accepted:
            self.current_bundle = artifacts_result.artifacts.bundle.deep_copy()
        return StageFitResult(
            stage_name=stage.name,
            active_blocks=unlocked_blocks,
            condition_names=condition_names,
            objective_before=float(before),
            objective_after=float(best_value),
            active_spec_names=tuple(spec.name for spec in active_specs),
            diagnostics=diagnostics,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            best_vector=best_vector,
            best_bundle=self.current_bundle.deep_copy(),
        )

    def _run_pre_fit_prior_predictive(self) -> PriorPredictiveReport:
        active_specs = self._active_specs_for_blocks(
            tuple(
                sorted(
                    {
                        spec.block
                        for spec in self.registry.supported_specs(categories=(CATEGORY_STAGE1,))
                    }
                )
            )
        )
        objective = SyntheticLikelihoodObjective(
            dataset=self.dataset,
            observed_summary=self._observed_summary(self.dataset.condition_names()),
            registry=self.registry,
            active_specs=active_specs,
            runner=self.runner,
            base_bundle=self.current_bundle,
            condition_names=self.dataset.condition_names(),
        )
        report = run_prior_predictive(
            objective,
            n_draws=self.settings.prior_predictive_draws,
            seed=self.settings.prior_predictive_seed,
        )
        cfg.require(
            report.pass_rate >= self.settings.min_prior_predictive_pass_rate,
            f"Prior predictive pass rate {report.pass_rate:.3f} is below the required threshold {self.settings.min_prior_predictive_pass_rate:.3f}.",
        )
        return report

    def _assess_stage_acceptance(
        self,
        *,
        objective_before: float,
        objective_after: float,
        posterior_predictive: PosteriorPredictiveReport,
        profile: tuple[ProfilePoint, ...],
        fake_data_recovery: FakeDataRecoveryReport,
        boundary: list[dict[str, object]],
        active_specs: tuple,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        improvement = objective_before - objective_after
        if not np.isfinite(objective_after):
            reasons.append("objective is not finite")
        elif improvement <= self.settings.min_objective_improvement:
            reasons.append(
                f"objective improvement {improvement:.6g} did not exceed {self.settings.min_objective_improvement:.6g}"
            )
        if not np.isfinite(posterior_predictive.worst_relative_rmse):
            reasons.append("posterior predictive residuals are not finite")
        elif posterior_predictive.worst_relative_rmse > self.settings.max_posterior_predictive_relative_rmse:
            reasons.append(
                "posterior predictive relative RMSE "
                f"{posterior_predictive.worst_relative_rmse:.6g} exceeded "
                f"{self.settings.max_posterior_predictive_relative_rmse:.6g}"
            )
        if self.settings.require_fake_data_recovery_pass and not fake_data_recovery.passed:
            reasons.append(
                "fake-data recovery failed "
                f"(normalized_error={fake_data_recovery.normalized_error:.6g})"
            )
        if self.settings.reject_on_boundary_touch:
            specs_by_name = {spec.name: spec for spec in active_specs}
            touching: list[str] = []
            for row in boundary:
                if not bool(row["touching_boundary"]):
                    continue
                spec = specs_by_name.get(str(row["name"]))
                if spec is None:
                    touching.append(str(row["name"]))
                    continue
                prior_on_lower = np.any(np.isclose(spec.prior_center, spec.lower, atol=1e-10, rtol=0.0))
                prior_on_upper = np.any(np.isclose(spec.prior_center, spec.upper, atol=1e-10, rtol=0.0))
                if prior_on_lower or prior_on_upper:
                    continue
                touching.append(str(row["name"]))
            if touching:
                reasons.append(f"parameters touching hard boundary: {', '.join(touching)}")
        flat_dimensions = self._flat_profile_dimensions(profile)
        if flat_dimensions:
            reasons.append(f"profile likelihood is flat for dimensions {flat_dimensions}")
        return (not reasons), tuple(reasons)

    def _flat_profile_dimensions(self, profile: tuple[ProfilePoint, ...]) -> tuple[int, ...]:
        if not profile:
            return ()
        objective_by_dimension: dict[int, list[float]] = {}
        for point in profile:
            objective_by_dimension.setdefault(point.dimension_index, []).append(float(point.objective_value))
        flat: list[int] = []
        for dimension_index, values in objective_by_dimension.items():
            span = max(values) - min(values)
            if span < self.settings.min_profile_objective_span:
                flat.append(int(dimension_index))
        return tuple(sorted(flat))

    def _condition_names_for_mode(self, mode: str) -> tuple[str, ...]:
        if mode == "all":
            return self.dataset.condition_names()
        if mode == "baseline":
            return self.runner.baseline_conditions()
        if mode == "drug":
            return self.runner.drug_conditions()
        if mode == "cue":
            return self.runner.cue_conditions()
        raise ValueError(f"Unknown stage condition mode {mode}.")

    def _observed_summary(self, condition_names: Iterable[str]) -> SummaryCollection:
        key = (tuple(condition_names), True)
        if key not in self._observed_summary_cache:
            self._observed_summary_cache[key] = summarize_dataset(
                self.dataset,
                condition_names=condition_names,
                dynamic_only=True,
            )
        return self._observed_summary_cache[key]

    def _active_specs_for_blocks(self, blocks: tuple[str, ...]):
        active_blocks = set(blocks)
        if not self.runner.drug_conditions():
            active_blocks.discard("drug_core")
        if not self.runner.cue_conditions():
            active_blocks.discard("cue_core")
        return self.registry.supported_specs(
            categories=(CATEGORY_STAGE1, CATEGORY_STAGE2),
            blocks=tuple(sorted(active_blocks)),
        )

    def _optimize_objective(
        self,
        objective: SyntheticLikelihoodObjective,
        initial_vector: np.ndarray,
        n_restarts: int,
    ) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.settings.seed)
        proposal_scales = np.maximum(objective.adapter.proposal_scales(), 1e-6)
        starts = [np.asarray(initial_vector, dtype=float).copy()]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(initial_vector + rng.normal(scale=proposal_scales * self.settings.random_start_scale))

        best_vector = starts[0]
        best_value = objective.evaluate_vector(best_vector).total_objective

        for start_vector in starts:
            result = minimize(
                lambda trial: objective.evaluate_vector(trial).total_objective,
                np.asarray(start_vector, dtype=float),
                method="Powell",
                options={"maxiter": self.settings.maxiter, "xtol": 1e-3, "ftol": 1e-3},
            )
            if result.fun < best_value:
                best_vector = np.asarray(result.x, dtype=float).copy()
                best_value = float(result.fun)
        return best_vector, float(best_value)
