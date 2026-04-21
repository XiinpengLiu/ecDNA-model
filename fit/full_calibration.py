"""
Minimal full-to-lite calibration workflow.

This module keeps full-v4 calibration separate from the v4-lite MAP path.  The
stages below calibrate against coarse v4-lite targets; they are diagnostic
calibration stages, not a formal full-model Bayesian posterior.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize

import config as cfg
from core.simulation import SimulationResult, run_simulation
from fit.data import CanonicalFitDataset
from fit.parameter_registry import ParameterBundle
from fit.v4_lite import (
    FullToLiteProjection,
    V4LiteFitResult,
    V4LiteStructure,
    _projection_from_prediction,
    predict_v4_lite,
    project_full_to_lite,
    run_v4_lite_posterior_predictive,
    summarize_dataset_v4_lite,
)


@dataclass(frozen=True)
class FullCalibrationSettings:
    record_times: tuple[float, ...] | None = None
    seeds: tuple[int, ...] = (301,)
    max_pop_size: int = 200000
    n_init: int | None = None
    verbose: bool = False
    maxiter: int = 20
    formal_maxiter: int = 8
    run_formal_raw_refinement: bool = True
    optimizer_method: str = "Powell"


@dataclass(frozen=True)
class FullCalibrationStageResult:
    stage_name: str
    opened_parameters: tuple[str, ...]
    objective_before: float | None
    objective_after: float | None
    residuals: dict[str, object]
    accepted: bool
    skipped_reason: str | None = None
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class FullCalibrationResult:
    stage_results: tuple[FullCalibrationStageResult, ...]
    calibrated_bundle: ParameterBundle
    projection: FullToLiteProjection
    coarse_residual_report: dict[str, object]
    skipped_stages: dict[str, str]
    formal_inference_report: dict[str, object] = field(default_factory=dict)
    mode_label: str = "calibration_not_formal_full_bayesian_posterior"


@dataclass(frozen=True)
class _StageParameterSpec:
    name: str
    getter: Callable[[ParameterBundle], np.ndarray]
    setter: Callable[[ParameterBundle, np.ndarray], None]
    scale: np.ndarray
    transform: str = "identity"
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def raw_values(self, bundle: ParameterBundle) -> np.ndarray:
        return np.asarray(self.getter(bundle), dtype=float).reshape(-1)


def _default_bundle() -> ParameterBundle:
    return ParameterBundle(model=copy.deepcopy(cfg.DEFAULT_MODEL_PARAMETERS), observation=copy.deepcopy(cfg.DEFAULT_OBSERVATION_PARAMETERS))


def _set_model_array(bundle: ParameterBundle, container_name: str, field_name: str, values: np.ndarray) -> None:
    target = getattr(getattr(bundle.model, container_name), field_name)
    target[...] = np.asarray(values, dtype=float).reshape(target.shape)


def _set_model_scalar(bundle: ParameterBundle, container_name: str, field_name: str, value: float) -> None:
    object.__setattr__(getattr(bundle.model, container_name), field_name, float(value))


def _set_turnover_scalar(bundle: ParameterBundle, species_name: str, field_name: str, value: float) -> None:
    object.__setattr__(bundle.model.turnover[species_name], field_name, float(value))


def _positive_bounds(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(center, dtype=float).reshape(-1), 1e-8)
    return np.maximum(1e-8, 0.10 * values), np.maximum(values + 1.0, 5.0 * values)


def _identity_bounds(center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(center, dtype=float).reshape(-1)
    widths = np.maximum(0.25, 4.0 * np.asarray(scale, dtype=float).reshape(-1))
    return values - widths, values + widths


def _encode_parameter_delta(spec: _StageParameterSpec, center: np.ndarray, values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=float).reshape(-1)
    base = np.asarray(center, dtype=float).reshape(-1)
    scale = np.maximum(np.asarray(spec.scale, dtype=float).reshape(-1), 1e-8)
    if spec.transform == "log":
        return (np.log(np.clip(current, 1e-12, None)) - np.log(np.clip(base, 1e-12, None))) / scale
    return (current - base) / scale


def _decode_parameter_delta(spec: _StageParameterSpec, center: np.ndarray, delta: np.ndarray) -> np.ndarray:
    base = np.asarray(center, dtype=float).reshape(-1)
    scale = np.maximum(np.asarray(spec.scale, dtype=float).reshape(-1), 1e-8)
    values = np.asarray(delta, dtype=float).reshape(-1)
    if spec.transform == "log":
        raw = np.exp(np.log(np.clip(base, 1e-12, None)) + values * scale)
    else:
        raw = base + values * scale
    if spec.lower is not None:
        raw = np.maximum(raw, np.asarray(spec.lower, dtype=float).reshape(-1))
    if spec.upper is not None:
        raw = np.minimum(raw, np.asarray(spec.upper, dtype=float).reshape(-1))
    return raw


def _pack_stage_vector(bundle: ParameterBundle, specs: tuple[_StageParameterSpec, ...], centers: tuple[np.ndarray, ...]) -> np.ndarray:
    pieces = [_encode_parameter_delta(spec, center, spec.raw_values(bundle)) for spec, center in zip(specs, centers)]
    return np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=float)


def _apply_stage_vector(bundle: ParameterBundle, specs: tuple[_StageParameterSpec, ...], centers: tuple[np.ndarray, ...], vector: np.ndarray) -> None:
    flat = np.asarray(vector, dtype=float).reshape(-1)
    offset = 0
    for spec, center in zip(specs, centers):
        size = int(center.size)
        raw = _decode_parameter_delta(spec, center, flat[offset : offset + size])
        spec.setter(bundle, raw)
        offset += size
    cfg.validate_model_parameters(bundle.model)
    cfg.validate_observation_parameters(bundle.observation)


def _spec_array(
    name: str,
    getter: Callable[[ParameterBundle], np.ndarray],
    setter: Callable[[ParameterBundle, np.ndarray], None],
    center: np.ndarray,
    *,
    scale: float,
    transform: str = "identity",
) -> _StageParameterSpec:
    center_values = np.asarray(center, dtype=float).reshape(-1)
    scale_values = np.full(center_values.size, float(scale), dtype=float)
    lower, upper = _positive_bounds(center_values) if transform == "log" else _identity_bounds(center_values, scale_values)
    return _StageParameterSpec(name, getter, setter, scale_values, transform, lower, upper)


def _stage_parameter_specs(bundle: ParameterBundle, stage_name: str) -> tuple[_StageParameterSpec, ...]:
    if stage_name == "F1-state-landscape":
        alpha_center = bundle.model.landscape.alpha.astype(float).copy()
        gamma_c = np.array([bundle.model.landscape.gamma_C[cfg.NPC]], dtype=float)
        gamma_p = np.array([bundle.model.landscape.gamma_P[cfg.OPC]], dtype=float)
        sigma_m = np.array([bundle.model.landscape.sigma_M], dtype=float)
        return (
            _spec_array(
                "landscape.alpha",
                lambda current: current.model.landscape.alpha.astype(float).copy(),
                lambda current, values: _set_model_array(current, "landscape", "alpha", np.asarray(values, dtype=float)),
                alpha_center,
                scale=0.20,
            ),
            _spec_array(
                "landscape.gamma_C[NPC]",
                lambda current: np.array([current.model.landscape.gamma_C[cfg.NPC]], dtype=float),
                lambda current, values: _set_single_array_entry(current, "landscape", "gamma_C", cfg.NPC, float(values[0])),
                gamma_c,
                scale=0.20,
            ),
            _spec_array(
                "landscape.gamma_P[OPC]",
                lambda current: np.array([current.model.landscape.gamma_P[cfg.OPC]], dtype=float),
                lambda current, values: _set_single_array_entry(current, "landscape", "gamma_P", cfg.OPC, float(values[0])),
                gamma_p,
                scale=0.20,
            ),
            _spec_array(
                "landscape.sigma_M",
                lambda current: np.array([current.model.landscape.sigma_M], dtype=float),
                lambda current, values: _set_model_scalar(current, "landscape", "sigma_M", float(values[0])),
                sigma_m,
                scale=0.25,
                transform="log",
            ),
        )
    if stage_name == "F2-ecDNA-turnover":
        specs: list[_StageParameterSpec] = []
        for species_name in cfg.SPECIES:
            for field_name in ("gain_ceiling", "loss_ceiling"):
                center = np.array([getattr(bundle.model.turnover[species_name], field_name)], dtype=float)
                specs.append(
                    _spec_array(
                        f"turnover.{species_name}.{field_name}",
                        lambda current, species_name=species_name, field_name=field_name: np.array(
                            [getattr(current.model.turnover[species_name], field_name)],
                            dtype=float,
                        ),
                        lambda current, values, species_name=species_name, field_name=field_name: _set_turnover_scalar(
                            current,
                            species_name,
                            field_name,
                            float(values[0]),
                        ),
                        center,
                        scale=0.20,
                        transform="log",
                    )
                )
        for species_name, field_name in ((cfg.SPECIES[cfg.CDK4], "b_C"), (cfg.SPECIES[cfg.PDGFRA], "b_P")):
            center = np.array([getattr(bundle.model.turnover[species_name], field_name)], dtype=float)
            specs.append(
                _spec_array(
                    f"turnover.{species_name}.{field_name}",
                    lambda current, species_name=species_name, field_name=field_name: np.array(
                        [getattr(current.model.turnover[species_name], field_name)],
                        dtype=float,
                    ),
                    lambda current, values, species_name=species_name, field_name=field_name: _set_turnover_scalar(
                        current,
                        species_name,
                        field_name,
                        float(values[0]),
                    ),
                    center,
                    scale=0.20,
                )
            )
        return tuple(specs)
    if stage_name == "F3-hazard-net-growth":
        fields = (
            ("theta_P", "identity", 0.20),
            ("chi_C", "log", 0.20),
            ("chi_P", "log", 0.20),
            ("phi_B", "identity", 0.20),
        )
        specs = []
        for field_name, transform, scale in fields:
            center = np.array([getattr(bundle.model.hazard, field_name)], dtype=float)
            specs.append(
                _spec_array(
                    f"hazard.{field_name}",
                    lambda current, field_name=field_name: np.array([getattr(current.model.hazard, field_name)], dtype=float),
                    lambda current, values, field_name=field_name: _set_model_scalar(current, "hazard", field_name, float(values[0])),
                    center,
                    scale=scale,
                    transform=transform,
                )
            )
        return tuple(specs)
    return ()


def _set_single_array_entry(bundle: ParameterBundle, container_name: str, field_name: str, index: int, value: float) -> None:
    target = getattr(getattr(bundle.model, container_name), field_name)
    updated = np.asarray(target, dtype=float).copy()
    updated[int(index)] = float(value)
    _set_model_array(bundle, container_name, field_name, updated)


def _target_from_fit_result(result: V4LiteFitResult) -> FullToLiteProjection:
    if result.projection_targets is not None:
        return result.projection_targets
    prediction = predict_v4_lite(result.tensor, result.final_params)
    return _projection_from_prediction(prediction)


def _target_uncertainty_from_fit_result(result: V4LiteFitResult) -> dict[str, float]:
    payload = result.reports.posterior_predictive_report.get("block_rmse", {}) if result.reports is not None else {}
    if not isinstance(payload, dict):
        return {}
    flow_scale = max(float(payload.get("flow_count", 0.0)), float(payload.get("flow_fraction", 0.0)), 1e-8)
    count_scale = max(float(payload.get("count_total", 0.0)), float(payload.get("count_gate", 0.0)), 1e-8)
    tag_scale = max(float(payload.get("ectag_hist", 0.0)), float(payload.get("ectag_moments", 0.0)), float(payload.get("qpcdr", 0.0)), 1e-8)
    return {
        "state_abundance": max(flow_scale, count_scale),
        "copy_distribution": tag_scale,
        "transition_matrix": flow_scale,
        "growth_rate": count_scale,
        "copy_kernel": tag_scale,
    }


def _finite_rmse(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(flat))))


def _block_residual(block: str, observed: np.ndarray, predicted: np.ndarray, *, uncertainty: float | None = None, weight: float = 1.0) -> dict[str, object]:
    observed_values = np.asarray(observed, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    residual = predicted_values - observed_values
    rmse = _finite_rmse(residual)
    scale = max(float(uncertainty) if uncertainty is not None and np.isfinite(float(uncertainty)) else _finite_rmse(observed_values), 1e-8)
    return {
        "block": block,
        "rmse": rmse,
        "relative_rmse": float(rmse / scale) if np.isfinite(rmse) else float("nan"),
        "weight": float(weight),
        "weighted_relative_rmse": float(weight * rmse / scale) if np.isfinite(rmse) else float("nan"),
        "scale_source": "target_uncertainty" if uncertainty is not None else "target_rms",
        "n": int(residual.size),
    }


def coarse_residual_report(
    target: FullToLiteProjection,
    projection: FullToLiteProjection,
    *,
    target_uncertainty: dict[str, float] | None = None,
    block_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    uncertainty = {} if target_uncertainty is None else dict(target_uncertainty)
    weights = {} if block_weights is None else dict(block_weights)

    def block_weight(block_name: str) -> float:
        return float(weights.get(block_name, 1.0))

    target_week_to_index = {week: index for index, week in enumerate(target.weeks)}
    projection_week_to_index = {week: index for index, week in enumerate(projection.weeks)}
    common_weeks = tuple(week for week in target.weeks if week in projection_week_to_index)
    rows: list[dict[str, object]] = []
    if common_weeks:
        target_indices = [target_week_to_index[week] for week in common_weeks]
        projection_indices = [projection_week_to_index[week] for week in common_weeks]
        rows.append(
            _block_residual(
                "state_abundance",
                target.state_abundance[target_indices, :],
                projection.state_abundance[projection_indices, :],
                uncertainty=uncertainty.get("state_abundance"),
                weight=block_weight("state_abundance"),
            )
        )
        rows.append(
            _block_residual(
                "copy_distribution",
                target.copy_distributions[target_indices, :, :, :],
                projection.copy_distributions[projection_indices, :, :, :],
                uncertainty=uncertainty.get("copy_distribution"),
                weight=block_weight("copy_distribution"),
            )
        )

    interval_count = min(
        0 if target.transition_matrices is None else int(target.transition_matrices.shape[0]),
        0 if projection.transition_matrices is None else int(projection.transition_matrices.shape[0]),
    )
    if interval_count:
        rows.append(
            _block_residual(
                "transition_matrix",
                target.transition_matrices[:interval_count],
                projection.transition_matrices[:interval_count],
                uncertainty=uncertainty.get("transition_matrix"),
                weight=block_weight("transition_matrix"),
            )
        )
    growth_count = min(
        0 if target.growth_rates is None else int(target.growth_rates.shape[0]),
        0 if projection.growth_rates is None else int(projection.growth_rates.shape[0]),
    )
    if growth_count:
        rows.append(
            _block_residual(
                "growth_rate",
                target.growth_rates[:growth_count],
                projection.growth_rates[:growth_count],
                uncertainty=uncertainty.get("growth_rate"),
                weight=block_weight("growth_rate"),
            )
        )
    kernel_count = min(
        0 if target.copy_kernels is None else int(target.copy_kernels.shape[0]),
        0 if projection.copy_kernels is None else int(projection.copy_kernels.shape[0]),
    )
    if kernel_count:
        rows.append(
            _block_residual(
                "copy_kernel",
                target.copy_kernels[:kernel_count],
                projection.copy_kernels[:kernel_count],
                uncertainty=uncertainty.get("copy_kernel"),
                weight=block_weight("copy_kernel"),
            )
        )

    finite_relative = [float(row["weighted_relative_rmse"]) for row in rows if np.isfinite(float(row["weighted_relative_rmse"])) and float(row["weight"]) > 0.0]
    weighted_rmse = float(np.mean(finite_relative)) if finite_relative else float("nan")
    return {
        "mode_label": "coarse_calibration_residuals",
        "common_weeks": common_weeks,
        "weighted_relative_rmse": weighted_rmse,
        "rows": rows,
        "target_uncertainty": uncertainty,
        "block_weights": weights,
        "double_counting_policy": "calibration mode may use v4-lite coarse targets; formal inference mode must fit raw observations only.",
    }


def _report_objective(report: dict[str, object]) -> float | None:
    value = report.get("weighted_relative_rmse")
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _expanded_parameter_names(specs: tuple[_StageParameterSpec, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for spec in specs:
        size = int(np.asarray(spec.scale, dtype=float).size)
        if size == 1:
            names.append(spec.name)
        else:
            names.extend(f"{spec.name}[{index}]" for index in range(size))
    return tuple(names)


def _summary_distribution_residual_report(observed, simulated) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for block_name in observed.block_names():
        if block_name not in simulated.blocks:
            continue
        observed_block = observed.blocks[block_name]
        simulated_block = simulated.blocks[block_name]
        observed_mapping = observed_block.as_mapping()
        simulated_mapping = simulated_block.as_mapping()
        common_keys = tuple(key for key in observed_block.keys if key in simulated_mapping)
        if common_keys:
            observed_values = np.asarray([observed_mapping[key] for key in common_keys], dtype=float)
            simulated_values = np.asarray([simulated_mapping[key] for key in common_keys], dtype=float)
            matching = "key"
        else:
            observed_values = np.sort(np.asarray(observed_block.values, dtype=float).reshape(-1))
            simulated_values = np.sort(np.asarray(simulated_block.values, dtype=float).reshape(-1))
            n = min(observed_values.size, simulated_values.size)
            if n == 0:
                continue
            observed_values = observed_values[:n]
            simulated_values = simulated_values[:n]
            matching = "distribution_order_statistic"
        residual = simulated_values - observed_values
        rmse = _finite_rmse(residual)
        scale = max(_finite_rmse(observed_values), 1e-8)
        rows.append(
            {
                "block": block_name,
                "matching": matching,
                "rmse": rmse,
                "relative_rmse": float(rmse / scale) if np.isfinite(rmse) else float("nan"),
                "n": int(residual.size),
            }
        )
    finite = [float(row["relative_rmse"]) for row in rows if np.isfinite(float(row["relative_rmse"]))]
    return {
        "mode_label": "formal_raw_observation_summary_residuals",
        "weighted_relative_rmse": float(np.mean(finite)) if finite else float("nan"),
        "rows": rows,
        "observed_blocks": observed.block_names(),
        "simulated_blocks": simulated.block_names(),
    }


class FullCalibrationRunner:
    def __init__(
        self,
        dataset: CanonicalFitDataset,
        lite_targets: V4LiteFitResult | FullToLiteProjection,
        *,
        base_bundle: ParameterBundle | None = None,
        structure: V4LiteStructure | None = None,
        settings: FullCalibrationSettings | None = None,
    ) -> None:
        self.dataset = dataset
        self.target = _target_from_fit_result(lite_targets) if isinstance(lite_targets, V4LiteFitResult) else lite_targets
        self.target_uncertainty = _target_uncertainty_from_fit_result(lite_targets) if isinstance(lite_targets, V4LiteFitResult) else {}
        self.bundle = _default_bundle() if base_bundle is None else base_bundle.deep_copy()
        self.structure = V4LiteStructure.default() if structure is None else structure
        self.settings = FullCalibrationSettings() if settings is None else settings

    def run_all_stages(self, *, condition_name: str | None = None) -> FullCalibrationResult:
        selected_condition = self._resolve_condition(condition_name)
        bundle = self.bundle.deep_copy()
        simulation_result = self._run_full_simulation(condition_name=selected_condition, bundle=bundle)
        projection = project_full_to_lite(simulation_result, structure=self.structure, purity_matrix=self.dataset.purity_matrix)
        residuals = self._coarse_report(projection)
        objective = _report_objective(residuals)
        f0_diagnostics = {
            "source": "full_simulator",
            "has_N": projection.state_abundance.size > 0,
            "has_p": projection.copy_distributions.size > 0,
            "has_T": projection.transition_matrices is not None,
            "has_g": projection.growth_rates is not None,
            "has_G": projection.copy_kernels is not None,
            "projection_diagnostics": projection.diagnostics,
        }
        stage_results: list[FullCalibrationStageResult] = [
            FullCalibrationStageResult(
                "F0-skeleton",
                (),
                None,
                objective,
                residuals,
                accepted=bool(f0_diagnostics["has_N"] and f0_diagnostics["has_p"]),
                diagnostics=f0_diagnostics,
            )
        ]
        stage_plan = (
            ("F1-state-landscape", {"state_abundance": 1.0, "transition_matrix": 1.0, "copy_distribution": 0.0, "growth_rate": 0.0, "copy_kernel": 0.0}),
            ("F2-ecDNA-turnover", {"state_abundance": 0.0, "transition_matrix": 0.0, "copy_distribution": 1.0, "growth_rate": 0.0, "copy_kernel": 1.0}),
            ("F3-hazard-net-growth", {"state_abundance": 0.5, "transition_matrix": 0.0, "copy_distribution": 0.0, "growth_rate": 1.0, "copy_kernel": 0.0}),
        )
        for stage_name, weights in stage_plan:
            stage_result, bundle, projection = self._optimize_calibration_stage(bundle, selected_condition, stage_name, weights)
            stage_results.append(stage_result)

        formal_report: dict[str, object]
        if self.settings.run_formal_raw_refinement:
            formal_stage, bundle, projection, formal_report = self._run_formal_raw_refinement(bundle, selected_condition)
            stage_results.append(formal_stage)
        else:
            formal_report = {"mode": "direct_raw_observation_map_skipped_by_settings"}

        final_residuals = self._coarse_report(projection)
        skipped = self._skipped_stages()
        return FullCalibrationResult(
            stage_results=tuple(stage_results),
            calibrated_bundle=bundle.deep_copy(),
            projection=projection,
            coarse_residual_report={
                **final_residuals,
                "calibration_mode": "v4_lite_summary_target",
                "formal_inference_mode": formal_report.get("mode", "direct_raw_observation_map"),
                "posterior_label": "map_or_diagnostic_not_full_bayesian_posterior",
            },
            skipped_stages=skipped,
            formal_inference_report=formal_report,
        )

    def run_f0_from_simulation_result(self, simulation_result: SimulationResult) -> FullCalibrationResult:
        projection = project_full_to_lite(simulation_result, structure=self.structure, purity_matrix=self.dataset.purity_matrix)
        return self._build_result_from_projection(projection, f0_source="provided_simulation_result")

    def _record_times(self) -> tuple[float, ...]:
        if self.settings.record_times is not None:
            return tuple(float(value) for value in self.settings.record_times)
        return tuple(float(week - 1) for week in self.target.weeks)

    def _resolve_condition(self, condition_name: str | None) -> str:
        selected_condition = condition_name or next(iter(self.dataset.conditions))
        cfg.require(selected_condition in self.dataset.conditions, f"Unknown condition {selected_condition}.")
        return selected_condition

    def _prepared_bundle(self, bundle: ParameterBundle | None = None) -> ParameterBundle:
        source = self.bundle if bundle is None else bundle
        record_times = self._record_times()
        simulation_params = replace(
            source.model.simulation,
            record_times=record_times,
            t_max=float(record_times[-1]),
            max_pop_size=int(self.settings.max_pop_size),
            n_init=source.model.simulation.n_init if self.settings.n_init is None else int(self.settings.n_init),
            target_population_size=None,
            fitting_mode=True,
            record_full_snapshots=True,
            record_events=True,
        )
        model = replace(source.model, simulation=simulation_params)
        return ParameterBundle(model=copy.deepcopy(model), observation=copy.deepcopy(source.observation))

    def _run_full_simulation(self, *, condition_name: str | None, bundle: ParameterBundle | None = None) -> SimulationResult:
        selected_condition = self._resolve_condition(condition_name)
        prepared_bundle = self._prepared_bundle(bundle)
        record_times = self._record_times()
        seed = int(self.settings.seeds[0])
        return run_simulation(
            params=prepared_bundle.model,
            observation_params=prepared_bundle.observation,
            initialization=self.dataset.build_empirical_initialization(selected_condition),
            input_schedules=self.dataset.conditions[selected_condition].build_input_schedules(),
            seed=seed,
            record_times=record_times,
            t_max=float(record_times[-1]),
            n_init=self.settings.n_init,
            max_pop_size=self.settings.max_pop_size,
            verbose=self.settings.verbose,
        )

    def _coarse_report(self, projection: FullToLiteProjection, block_weights: dict[str, float] | None = None) -> dict[str, object]:
        return coarse_residual_report(
            self.target,
            projection,
            target_uncertainty=self.target_uncertainty,
            block_weights=block_weights,
        )

    def _project_bundle(self, bundle: ParameterBundle, condition_name: str) -> FullToLiteProjection:
        simulation_result = self._run_full_simulation(condition_name=condition_name, bundle=bundle)
        return project_full_to_lite(simulation_result, structure=self.structure, purity_matrix=self.dataset.purity_matrix)

    def _optimize_calibration_stage(
        self,
        bundle: ParameterBundle,
        condition_name: str,
        stage_name: str,
        block_weights: dict[str, float],
    ) -> tuple[FullCalibrationStageResult, ParameterBundle, FullToLiteProjection]:
        specs = _stage_parameter_specs(bundle, stage_name)
        opened = tuple(spec.name for spec in specs)
        if not specs:
            projection = self._project_bundle(bundle, condition_name)
            residuals = self._coarse_report(projection, block_weights)
            return (
                FullCalibrationStageResult(stage_name, opened, None, _report_objective(residuals), residuals, accepted=False, skipped_reason="No stage parameters configured."),
                bundle.deep_copy(),
                projection,
            )
        centers = tuple(spec.raw_values(bundle) for spec in specs)
        initial = _pack_stage_vector(bundle, specs, centers)
        before_projection = self._project_bundle(bundle, condition_name)
        before_report = self._coarse_report(before_projection, block_weights)
        before = _report_objective(before_report)

        def objective(vector: np.ndarray) -> float:
            trial = bundle.deep_copy()
            try:
                _apply_stage_vector(trial, specs, centers, vector)
                projection = self._project_bundle(trial, condition_name)
                report = self._coarse_report(projection, block_weights)
                score = _report_objective(report)
                if score is None:
                    return 1e12
                prior = 0.02 * float(np.mean(np.square(np.asarray(vector, dtype=float)))) if vector.size else 0.0
                return float(score + prior)
            except Exception:
                return 1e12

        result = minimize(
            objective,
            initial,
            method=self.settings.optimizer_method,
            options={"maxiter": int(self.settings.maxiter), "disp": False},
        )
        best_vector = np.asarray(result.x if result.success or np.isfinite(result.fun) else initial, dtype=float)
        best_bundle = bundle.deep_copy()
        _apply_stage_vector(best_bundle, specs, centers, best_vector)
        best_projection = self._project_bundle(best_bundle, condition_name)
        best_report = self._coarse_report(best_projection, block_weights)
        after = _report_objective(best_report)
        accepted = after is not None and (before is None or after <= before + 1e-8)
        if not accepted:
            best_bundle = bundle.deep_copy()
            best_projection = before_projection
            best_report = before_report
            after = before
        diagnostics = {
            "calibration_stage": True,
            "optimization": "Powell MAP over full simulator coarse residuals",
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_evaluations": int(getattr(result, "nfev", 0)),
            "parameter_deltas": {name: float(value) for name, value in zip(_expanded_parameter_names(specs), best_vector.tolist())},
            "double_counting_policy": "calibration objective uses v4-lite coarse targets only; formal raw objective is run separately.",
        }
        return (
            FullCalibrationStageResult(
                stage_name=stage_name,
                opened_parameters=opened,
                objective_before=before,
                objective_after=after,
                residuals=best_report,
                accepted=accepted,
                diagnostics=diagnostics,
            ),
            best_bundle,
            best_projection,
        )

    def _run_formal_raw_refinement(
        self,
        bundle: ParameterBundle,
        condition_name: str,
    ) -> tuple[FullCalibrationStageResult, ParameterBundle, FullToLiteProjection, dict[str, object]]:
        specs = tuple(
            spec
            for stage_name in ("F1-state-landscape", "F2-ecDNA-turnover", "F3-hazard-net-growth")
            for spec in _stage_parameter_specs(bundle, stage_name)
        )
        centers = tuple(spec.raw_values(bundle) for spec in specs)
        initial = _pack_stage_vector(bundle, specs, centers)
        before_projection = self._project_bundle(bundle, condition_name)
        before_report = self._formal_raw_observation_report(before_projection, condition_name, bundle=bundle)
        before = _report_objective(before_report)

        def objective(vector: np.ndarray) -> float:
            trial = bundle.deep_copy()
            try:
                _apply_stage_vector(trial, specs, centers, vector)
                projection = self._project_bundle(trial, condition_name)
                report = self._formal_raw_observation_report(projection, condition_name, bundle=trial)
                score = _report_objective(report)
                if score is None:
                    return 1e12
                prior = 0.05 * float(np.mean(np.square(np.asarray(vector, dtype=float)))) if vector.size else 0.0
                return float(score + prior)
            except Exception:
                return 1e12

        result = minimize(
            objective,
            initial,
            method=self.settings.optimizer_method,
            options={"maxiter": int(self.settings.formal_maxiter), "disp": False},
        )
        best_vector = np.asarray(result.x if result.success or np.isfinite(result.fun) else initial, dtype=float)
        best_bundle = bundle.deep_copy()
        _apply_stage_vector(best_bundle, specs, centers, best_vector)
        best_projection = self._project_bundle(best_bundle, condition_name)
        best_report = self._formal_raw_observation_report(best_projection, condition_name, bundle=best_bundle)
        after = _report_objective(best_report)
        accepted = after is not None and (before is None or after <= before + 1e-8)
        if not accepted:
            best_bundle = bundle.deep_copy()
            best_projection = before_projection
            best_report = before_report
            after = before
        stage = FullCalibrationStageResult(
            stage_name="F-formal-raw-MAP",
            opened_parameters=tuple(spec.name for spec in specs),
            objective_before=before,
            objective_after=after,
            residuals=best_report,
            accepted=accepted,
            diagnostics={
                "formal_inference_stage": True,
                "optimization": "direct raw-observation MAP approximation; no v4-lite target in objective",
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_evaluations": int(getattr(result, "nfev", 0)),
                "double_counting_policy": "v4-lite summaries are not used in this objective.",
            },
        )
        return stage, best_bundle, best_projection, best_report

    def _formal_raw_observation_report(self, projection: FullToLiteProjection, condition_name: str, *, bundle: ParameterBundle) -> dict[str, object]:
        simulation_result = self._run_full_simulation(condition_name=condition_name, bundle=bundle)
        simulated_dataset = CanonicalFitDataset.from_simulation_runs(
            {condition_name: (simulation_result,)},
            conditions={condition_name: self.dataset.conditions[condition_name]},
            ectag_hist_max=self.dataset.ectag_upper_bound(),
        )
        observed = summarize_dataset_v4_lite(self.dataset, condition_names=(condition_name,), binning=self.structure.binning)
        simulated = summarize_dataset_v4_lite(simulated_dataset, condition_names=(condition_name,), binning=self.structure.binning)
        report = _summary_distribution_residual_report(observed, simulated)
        report.update(
            {
                "mode": "direct_raw_observation_map",
                "projection_weeks": projection.weeks,
                "double_counting_policy": "direct raw observations only; v4-lite posterior targets excluded.",
            }
        )
        return report

    @staticmethod
    def _skipped_stages() -> dict[str, str]:
        return {
            "F4-RV": "skipped: requires independent R/V marker evidence.",
            "F5-co-segregation-daughter-memory": "skipped: requires same-cell lineage evidence.",
        }

    def _build_result_from_projection(self, projection: FullToLiteProjection, *, f0_source: str = "full_simulator") -> FullCalibrationResult:
        residuals = self._coarse_report(projection)
        objective = float(residuals["weighted_relative_rmse"]) if np.isfinite(float(residuals["weighted_relative_rmse"])) else None
        f0_diagnostics = {
            "source": f0_source,
            "has_N": projection.state_abundance.size > 0,
            "has_p": projection.copy_distributions.size > 0,
            "has_T": projection.transition_matrices is not None,
            "has_g": projection.growth_rates is not None,
            "has_G": projection.copy_kernels is not None,
            "projection_diagnostics": projection.diagnostics,
        }
        stages = [
            FullCalibrationStageResult(
                "F0-skeleton",
                (),
                None,
                objective,
                residuals,
                accepted=bool(f0_diagnostics["has_N"] and f0_diagnostics["has_p"]),
                diagnostics=f0_diagnostics,
            ),
            self._diagnostic_stage(
                "F1-state-landscape",
                ("landscape.alpha", "landscape.gamma_C[NPC]", "landscape.gamma_P[OPC]", "MYC mobility proxy"),
                residuals,
            ),
            self._diagnostic_stage(
                "F2-ecDNA-turnover",
                ("turnover.gain/loss baseline", "target drug loss effect"),
                residuals,
            ),
            self._diagnostic_stage(
                "F3-hazard-net-growth",
                ("hazard/death growth proxy",),
                residuals,
            ),
        ]
        skipped = self._skipped_stages()
        return FullCalibrationResult(
            stage_results=tuple(stages),
            calibrated_bundle=self.bundle.deep_copy(),
            projection=projection,
            coarse_residual_report={
                **residuals,
                "calibration_mode": "v4_lite_summary_target",
                "formal_inference_mode": "not_run",
                "posterior_label": "not_formal_full_bayesian_posterior",
            },
            skipped_stages=skipped,
            formal_inference_report={"mode": "not_run_for_provided_simulation_result"},
        )

    @staticmethod
    def _diagnostic_stage(stage_name: str, opened_parameters: tuple[str, ...], residuals: dict[str, object]) -> FullCalibrationStageResult:
        objective = float(residuals["weighted_relative_rmse"]) if np.isfinite(float(residuals["weighted_relative_rmse"])) else None
        return FullCalibrationStageResult(
            stage_name=stage_name,
            opened_parameters=opened_parameters,
            objective_before=objective,
            objective_after=objective,
            residuals=residuals,
            accepted=objective is not None,
            diagnostics={
                "calibration_stage": True,
                "optimization": "coarse residual diagnostic; no formal full posterior sampling",
                "double_counting_policy": "do not combine v4-lite posterior targets with raw-observation likelihood in formal inference mode",
            },
        )


def write_full_calibration_reports(output_dir: str | Path, result: FullCalibrationResult) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode_label": result.mode_label,
        "stage_results": [
            {
                "stage_name": stage.stage_name,
                "opened_parameters": stage.opened_parameters,
                "objective_before": stage.objective_before,
                "objective_after": stage.objective_after,
                "accepted": stage.accepted,
                "skipped_reason": stage.skipped_reason,
                "diagnostics": stage.diagnostics,
            }
            for stage in result.stage_results
        ],
        "coarse_residual_report": result.coarse_residual_report,
        "skipped_stages": result.skipped_stages,
        "formal_inference_mode": result.formal_inference_report.get("mode", "not_run"),
        "formal_inference_report": result.formal_inference_report,
        "double_counting_policy": "calibration reports may use v4-lite coarse targets; formal full inference must use raw observations only.",
    }
    (destination / "full_calibration_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with open(destination / "full_coarse_residuals.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = ("block", "rmse", "relative_rmse", "n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.coarse_residual_report.get("rows", ()):
            writer.writerow({name: row.get(name) for name in fieldnames})
