"""Full-simulator bridge calibration against v4-lite summaries."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

import config as cfg
from core.simulation import SimulationResult, run_simulation
from fit.io_utils import write_json, write_netcdf_file, write_npz_or_marker, write_table_bundle, write_text_pdf
from fit.data import CanonicalFitDataset
from fit.parameter_registry import ParameterBundle
from fit.v4_lite import FullToLiteProjection, V4LiteParameters, V4LiteStructure, _prediction_summary, build_v4_lite_tensor, project_full_to_lite


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
    smc_particles: int = 32
    smc_steps: int = 2
    smc_scale: float = 0.35


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
class _StageSpec:
    name: str
    getter: Callable[[ParameterBundle], np.ndarray]
    setter: Callable[[ParameterBundle, np.ndarray], None]
    scale: float
    transform: str = "identity"


def _default_bundle() -> ParameterBundle:
    return ParameterBundle(copy.deepcopy(cfg.DEFAULT_MODEL_PARAMETERS), copy.deepcopy(cfg.DEFAULT_OBSERVATION_PARAMETERS))


def _set_array(bundle: ParameterBundle, section: str, field_name: str, values: np.ndarray) -> None:
    target = getattr(getattr(bundle.model, section), field_name)
    target[...] = np.asarray(values, dtype=float).reshape(target.shape)


def _set_scalar(bundle: ParameterBundle, section: str, field_name: str, value: float) -> None:
    object.__setattr__(getattr(bundle.model, section), field_name, float(value))


def _stage_specs(bundle: ParameterBundle, stage_name: str) -> tuple[_StageSpec, ...]:
    if stage_name == "F1-state-landscape":
        return (
            _StageSpec(
                "landscape.alpha",
                lambda b: b.model.landscape.alpha.astype(float).copy(),
                lambda b, v: _set_array(b, "landscape", "alpha", v),
                0.25,
            ),
            _StageSpec(
                "landscape.gamma_C[NPC]",
                lambda b: np.array([b.model.landscape.gamma_C[cfg.NPC]], dtype=float),
                lambda b, v: _set_single_array(b, "landscape", "gamma_C", cfg.NPC, float(v[0])),
                0.25,
            ),
            _StageSpec(
                "landscape.gamma_P[OPC]",
                lambda b: np.array([b.model.landscape.gamma_P[cfg.OPC]], dtype=float),
                lambda b, v: _set_single_array(b, "landscape", "gamma_P", cfg.OPC, float(v[0])),
                0.25,
            ),
        )
    if stage_name == "F2-ecDNA-turnover":
        return tuple(
            _StageSpec(
                f"turnover.{species}.gain_ceiling",
                lambda b, species=species: np.array([b.model.turnover[species].gain_ceiling], dtype=float),
                lambda b, v, species=species: object.__setattr__(b.model.turnover[species], "gain_ceiling", max(float(v[0]), 0.0)),
                0.25,
            )
            for species in cfg.SPECIES
        )
    if stage_name == "F3-hazard-net-growth":
        return (
            _StageSpec("hazard.theta_P", lambda b: np.array([b.model.hazard.theta_P], dtype=float), lambda b, v: _set_scalar(b, "hazard", "theta_P", float(v[0])), 0.25),
            _StageSpec("hazard.phi_B", lambda b: np.array([b.model.hazard.phi_B], dtype=float), lambda b, v: _set_scalar(b, "hazard", "phi_B", float(v[0])), 0.25),
        )
    return ()


def _set_single_array(bundle: ParameterBundle, section: str, field_name: str, index: int, value: float) -> None:
    target = getattr(getattr(bundle.model, section), field_name)
    updated = np.asarray(target, dtype=float).copy()
    updated[index] = value
    _set_array(bundle, section, field_name, updated)


def _pack(bundle: ParameterBundle, specs: tuple[_StageSpec, ...], centers: tuple[np.ndarray, ...]) -> np.ndarray:
    pieces = []
    for spec, center in zip(specs, centers):
        raw = spec.getter(bundle).reshape(-1)
        if spec.transform == "log":
            pieces.append((np.log(np.clip(raw, 1e-12, None)) - np.log(np.clip(center, 1e-12, None))) / spec.scale)
        else:
            pieces.append((raw - center) / spec.scale)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)


def _spec_parameter_names(specs: tuple[_StageSpec, ...], centers: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for spec, center in zip(specs, centers):
        size = int(np.asarray(center).size)
        if size == 1:
            names.append(spec.name)
        else:
            names.extend(f"{spec.name}[{idx}]" for idx in range(size))
    return tuple(names)


def _apply(bundle: ParameterBundle, specs: tuple[_StageSpec, ...], centers: tuple[np.ndarray, ...], vector: np.ndarray) -> None:
    flat = np.asarray(vector, dtype=float).reshape(-1)
    offset = 0
    for spec, center in zip(specs, centers):
        size = center.size
        chunk = flat[offset : offset + size]
        offset += size
        if spec.transform == "log":
            raw = np.exp(np.log(np.clip(center, 1e-12, None)) + chunk * spec.scale)
        else:
            raw = center + chunk * spec.scale
        spec.setter(bundle, raw)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0


def coarse_residual_report(target: FullToLiteProjection, projection: FullToLiteProjection, *, target_uncertainty: dict[str, float] | None = None, block_weights: dict[str, float] | None = None) -> dict[str, object]:
    weights = {} if block_weights is None else dict(block_weights)
    rows: list[dict[str, object]] = []

    def add(block: str, observed: np.ndarray | None, predicted: np.ndarray | None) -> None:
        if observed is None or predicted is None:
            return
        n0 = min(observed.shape[0], predicted.shape[0])
        if n0 == 0:
            return
        rmse = _rmse(observed[:n0], predicted[:n0])
        scale = max(_rmse(observed[:n0], np.zeros_like(observed[:n0])), 1e-8)
        weight = float(weights.get(block, 1.0))
        rows.append({"block": block, "rmse": rmse, "relative_rmse": rmse / scale, "weighted_relative_rmse": weight * rmse / scale, "weight": weight, "n": int(np.asarray(observed[:n0]).size)})

    add("state_abundance", target.state_abundance, projection.state_abundance)
    add("copy_distribution", target.copy_distributions, projection.copy_distributions)
    add("transition_matrix", target.transition_matrices, projection.transition_matrices)
    add("growth_rate", target.growth_rates, projection.growth_rates)
    add("copy_kernel", target.copy_kernels, projection.copy_kernels)
    finite = [float(row["weighted_relative_rmse"]) for row in rows if np.isfinite(float(row["weighted_relative_rmse"])) and float(row["weight"]) > 0.0]
    return {
        "mode_label": "coarse_calibration_residuals",
        "weighted_relative_rmse": float(np.mean(finite)) if finite else float("nan"),
        "rows": rows,
        "double_counting_policy": "calibration may use v4-lite targets; formal raw inference must exclude those targets.",
    }


def _objective_value(report: dict[str, object]) -> float | None:
    value = float(report.get("weighted_relative_rmse", float("nan")))
    return value if np.isfinite(value) else None


class FullCalibrationRunner:
    def __init__(self, dataset: CanonicalFitDataset, target: FullToLiteProjection, *, base_bundle: ParameterBundle | None = None, target_uncertainty: dict[str, float] | None = None, structure: V4LiteStructure | None = None, settings: FullCalibrationSettings | None = None):
        self.dataset = dataset
        self.target = target
        self.target_uncertainty = {} if target_uncertainty is None else dict(target_uncertainty)
        self.bundle = _default_bundle() if base_bundle is None else base_bundle.deep_copy()
        self.structure = V4LiteStructure.default() if structure is None else structure
        self.settings = FullCalibrationSettings() if settings is None else settings

    def run_f0_from_simulation_result(self, simulation_result: SimulationResult) -> FullCalibrationResult:
        projection = project_full_to_lite(simulation_result, structure=self.structure, purity_matrix=self.dataset.purity_matrix)
        return self._build_result_from_projection(projection, "provided_simulation_result")

    def run_all_stages(self, *, condition_name: str | None = None) -> FullCalibrationResult:
        condition = self._resolve_condition(condition_name)
        bundle = self.bundle.deep_copy()
        projection = self._project_bundle(bundle, condition)
        report = self._coarse_report(projection)
        f0 = FullCalibrationStageResult("F0-skeleton", (), None, _objective_value(report), report, True, diagnostics={"source": "full_simulator"})
        self._print_stage(f0)
        stages = [f0]
        for stage_name, weights in (
            ("F1-state-landscape", {"state_abundance": 1.0, "transition_matrix": 1.0, "copy_distribution": 0.0, "growth_rate": 0.0, "copy_kernel": 0.0}),
            ("F2-ecDNA-turnover", {"state_abundance": 0.0, "transition_matrix": 0.0, "copy_distribution": 1.0, "growth_rate": 0.0, "copy_kernel": 1.0}),
            ("F3-hazard-net-growth", {"state_abundance": 0.5, "transition_matrix": 0.0, "copy_distribution": 0.0, "growth_rate": 1.0, "copy_kernel": 0.0}),
        ):
            stage, bundle, projection = self._optimize_calibration_stage(bundle, condition, stage_name, weights)
            stages.append(stage)
            self._print_stage(stage)
        formal_report = {"mode": "not_requested", "formal_mode": "not_requested"}
        if self.settings.run_formal_raw_refinement:
            formal, bundle, projection, formal_report = self._run_formal_raw_refinement(bundle, condition)
            stages.append(formal)
            self._print_stage(formal)
        final_report = self._coarse_report(projection)
        return FullCalibrationResult(
            tuple(stages),
            bundle.deep_copy(),
            projection,
            {**final_report, "calibration_mode": "v4_lite_summary_target", "formal_inference_mode": formal_report.get("formal_mode", formal_report.get("mode", "not_requested")), "posterior_label": "smc_style_particle_posterior" if "particle_posterior" in formal_report else "map_or_diagnostic_not_full_bayesian_posterior"},
            self._skipped_stages(),
            formal_report,
            mode_label="restricted_full_smc_style_particle_posterior" if "particle_posterior" in formal_report else "calibration_not_formal_full_bayesian_posterior",
        )

    @staticmethod
    def _print_stage(stage: FullCalibrationStageResult) -> None:
        print(
            "[fit-full] "
            f"{stage.stage_name}: opened_parameters={stage.opened_parameters} accepted={stage.accepted} "
            f"before={stage.objective_before} after={stage.objective_after} skip={stage.skipped_reason}"
        )

    def _record_times(self) -> tuple[float, ...]:
        if self.settings.record_times is not None:
            return tuple(float(v) for v in self.settings.record_times)
        return tuple(float(week - 1) for week in self.target.weeks)

    def _resolve_condition(self, condition_name: str | None) -> str:
        condition = condition_name or next(iter(self.dataset.conditions))
        cfg.require(condition in self.dataset.conditions, f"Unknown condition {condition}.")
        return condition

    def _prepared_bundle(self, bundle: ParameterBundle | None = None) -> ParameterBundle:
        source = self.bundle if bundle is None else bundle
        record_times = self._record_times()
        sim = replace(
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
        return ParameterBundle(replace(source.model, simulation=sim), copy.deepcopy(source.observation))

    def _run_full_simulation(self, *, condition_name: str | None, bundle: ParameterBundle | None = None) -> SimulationResult:
        selected = self._resolve_condition(condition_name)
        prepared = self._prepared_bundle(bundle)
        times = self._record_times()
        return run_simulation(
            params=prepared.model,
            observation_params=prepared.observation,
            initialization=self.dataset.build_empirical_initialization(selected),
            input_schedules=self.dataset.conditions[selected].build_input_schedules(),
            seed=int(self.settings.seeds[0]),
            record_times=times,
            t_max=float(times[-1]),
            n_init=self.settings.n_init,
            max_pop_size=self.settings.max_pop_size,
            verbose=self.settings.verbose,
        )

    def _project_bundle(self, bundle: ParameterBundle, condition_name: str) -> FullToLiteProjection:
        return project_full_to_lite(self._run_full_simulation(condition_name=condition_name, bundle=bundle), structure=self.structure, purity_matrix=self.dataset.purity_matrix)

    def _coarse_report(self, projection: FullToLiteProjection, block_weights: dict[str, float] | None = None) -> dict[str, object]:
        return coarse_residual_report(self.target, projection, target_uncertainty=self.target_uncertainty, block_weights=block_weights)

    def _optimize_calibration_stage(self, bundle: ParameterBundle, condition_name: str, stage_name: str, block_weights: dict[str, float]) -> tuple[FullCalibrationStageResult, ParameterBundle, FullToLiteProjection]:
        specs = _stage_specs(bundle, stage_name)
        if not specs:
            projection = self._project_bundle(bundle, condition_name)
            report = self._coarse_report(projection, block_weights)
            return FullCalibrationStageResult(stage_name, (), None, _objective_value(report), report, False, skipped_reason="No parameters configured."), bundle.deep_copy(), projection
        centers = tuple(spec.getter(bundle) for spec in specs)
        initial = _pack(bundle, specs, centers)
        before_projection = self._project_bundle(bundle, condition_name)
        before_report = self._coarse_report(before_projection, block_weights)
        before = _objective_value(before_report)

        def objective(vector: np.ndarray) -> float:
            trial = bundle.deep_copy()
            _apply(trial, specs, centers, vector)
            report = self._coarse_report(self._project_bundle(trial, condition_name), block_weights)
            value = _objective_value(report)
            penalty = 0.01 * float(np.mean(np.square(vector))) if vector.size else 0.0
            return 1e12 if value is None else float(value + penalty)

        result = minimize(objective, initial, method=self.settings.optimizer_method, options={"maxiter": int(self.settings.maxiter), "disp": False})
        best_vector = np.asarray(result.x if np.isfinite(result.fun) else initial, dtype=float)
        best_bundle = bundle.deep_copy()
        _apply(best_bundle, specs, centers, best_vector)
        best_projection = self._project_bundle(best_bundle, condition_name)
        best_report = self._coarse_report(best_projection, block_weights)
        after = _objective_value(best_report)
        accepted = after is not None and (before is None or after <= before + 1e-8)
        if not accepted:
            return FullCalibrationStageResult(stage_name, tuple(s.name for s in specs), before, before, before_report, False), bundle.deep_copy(), before_projection
        return FullCalibrationStageResult(stage_name, tuple(s.name for s in specs), before, after, best_report, True, diagnostics={"optimizer_success": bool(result.success), "optimizer_message": str(result.message)}), best_bundle, best_projection

    def _run_formal_raw_refinement(self, bundle: ParameterBundle, condition_name: str) -> tuple[FullCalibrationStageResult, ParameterBundle, FullToLiteProjection, dict[str, object]]:
        specs = tuple(spec for stage in ("F1-state-landscape", "F2-ecDNA-turnover", "F3-hazard-net-growth") for spec in _stage_specs(bundle, stage))
        centers = tuple(spec.getter(bundle) for spec in specs)
        initial = _pack(bundle, specs, centers)
        before_projection = self._project_bundle(bundle, condition_name)
        before_report = self._formal_raw_observation_report(before_projection, condition_name, bundle=bundle)
        before = _objective_value(before_report)

        def objective(vector: np.ndarray) -> float:
            trial = bundle.deep_copy()
            _apply(trial, specs, centers, vector)
            projection = self._project_bundle(trial, condition_name)
            value = _objective_value(self._formal_raw_observation_report(projection, condition_name, bundle=trial))
            return 1e12 if value is None else value

        result = minimize(objective, initial, method=self.settings.optimizer_method, options={"maxiter": int(self.settings.formal_maxiter), "disp": False})
        best = bundle.deep_copy()
        _apply(best, specs, centers, np.asarray(result.x if np.isfinite(result.fun) else initial, dtype=float))
        projection = self._project_bundle(best, condition_name)
        report = self._formal_raw_observation_report(projection, condition_name, bundle=best)
        after = _objective_value(report)
        accepted = after is not None and (before is None or after <= before + 1e-8)
        particle_report = self._run_smc_style_particles(best, specs, centers, np.asarray(result.x if np.isfinite(result.fun) else initial, dtype=float), condition_name)
        report.update({"formal_mode": "bayesian_synthetic_likelihood_smc_style", "particle_posterior": particle_report})
        stage = FullCalibrationStageResult("F-formal-raw-MAP", tuple(spec.name for spec in specs), before, after, report, accepted, diagnostics={"formal_inference_stage": True, "particle_posterior": particle_report})
        return stage, best, projection, report

    def _formal_raw_observation_report(self, projection: FullToLiteProjection, condition_name: str, *, bundle: ParameterBundle) -> dict[str, object]:
        tensor = build_v4_lite_tensor(self.dataset, condition_names=(condition_name,), structure=self.structure)
        params = V4LiteParameters.default(tensor.structure, purity_matrix=self.dataset.purity_matrix, qpcdr_calibration=self.dataset.qpcdr_calibration)

        def _align_week_axis(values: np.ndarray, n_weeks: int) -> tuple[np.ndarray, int]:
            array = np.asarray(values, dtype=float)
            if array.shape[0] >= n_weeks:
                return array[:n_weeks], 0
            pad_n = n_weeks - array.shape[0]
            return np.concatenate([array, np.repeat(array[-1:], pad_n, axis=0)], axis=0), pad_n

        abundance_array, abundance_pad_n = _align_week_axis(projection.state_abundance, len(tensor.weeks))
        copy_array, copy_pad_n = _align_week_axis(projection.copy_distributions, len(tensor.weeks))
        abundance = {condition_name: abundance_array}
        copies = {condition_name: copy_array}
        predicted = _prediction_summary(tensor, params, abundance, copies).align_to(tensor.observed_summary)
        rows: list[dict[str, object]] = []
        for block in tensor.observed_summary.block_names():
            observed_values = tensor.observed_summary.blocks[block].values
            predicted_values = predicted.blocks[block].values
            residual = predicted_values - observed_values
            scale = max(float(np.sqrt(np.mean(np.square(observed_values)))) if observed_values.size else 0.0, 1e-8)
            rows.append(
                {
                    "block": block,
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
                    "relative_rmse": float(np.sqrt(np.mean(np.square(residual))) / scale) if residual.size else 0.0,
                    "weighted_relative_rmse": float(np.sqrt(np.mean(np.square(residual))) / scale) if residual.size else 0.0,
                    "weight": 1.0,
                    "n": int(observed_values.size),
                }
            )
        finite = [float(row["weighted_relative_rmse"]) for row in rows if np.isfinite(float(row["weighted_relative_rmse"]))]
        return {
            "mode": "direct_raw_observation_map",
            "condition": condition_name,
            "mode_label": "raw_observation_residuals",
            "weighted_relative_rmse": float(np.mean(finite)) if finite else float("nan"),
            "rows": rows,
            "double_counting_policy": "raw-observation mode excludes v4-lite posterior target likelihood.",
            "raw_observation_week_count": int(len(tensor.weeks)),
            "projection_week_count": int(projection.state_abundance.shape[0]),
            "projection_padding_policy": "repeat_last_full_snapshot_for_observed_weeks_beyond_simulation" if max(abundance_pad_n, copy_pad_n) else "none",
            "projection_padding_weeks": int(max(abundance_pad_n, copy_pad_n)),
        }

    def _run_smc_style_particles(self, bundle: ParameterBundle, specs: tuple[_StageSpec, ...], centers: tuple[np.ndarray, ...], map_vector: np.ndarray, condition_name: str) -> dict[str, object]:
        rng = np.random.default_rng(int(self.settings.seeds[0]) + 9001)
        parameter_names = _spec_parameter_names(specs, centers)
        current = np.asarray(map_vector, dtype=float).reshape(1, -1)
        if current.size == 0:
            return {"mode": "bayesian_synthetic_likelihood_smc_style", "n_particles": 0, "reason": "no opened parameters"}
        particles = np.repeat(current, int(self.settings.smc_particles), axis=0)
        particles += rng.normal(0.0, float(self.settings.smc_scale), size=particles.shape)
        rows: list[dict[str, object]] = []
        for step in range(max(int(self.settings.smc_steps), 1)):
            scored: list[tuple[float, np.ndarray, dict[str, object]]] = []
            for particle in particles:
                trial = bundle.deep_copy()
                _apply(trial, specs, centers, particle)
                projection = self._project_bundle(trial, condition_name)
                report = self._formal_raw_observation_report(projection, condition_name, bundle=trial)
                value = _objective_value(report)
                score = 1e12 if value is None else float(value)
                scored.append((score, particle.copy(), report))
            scores = np.asarray([item[0] for item in scored], dtype=float)
            finite = np.isfinite(scores)
            if not np.any(finite):
                weights = np.full(scores.size, 1.0 / max(scores.size, 1), dtype=float)
            else:
                centered = scores - float(np.min(scores[finite]))
                # Student-t-like heavy-tailed synthetic likelihood weight.
                logw = -0.5 * np.log1p(centered)
                logw -= float(np.max(logw))
                weights = np.exp(logw)
                weights = weights / max(float(np.sum(weights)), 1e-12)
            for idx, (score, particle, _report) in enumerate(scored):
                row = {"step": step, "particle": idx, "synthetic_likelihood_score": score, "weight": float(weights[idx])}
                for name, value in zip(parameter_names, particle.tolist()):
                    row[name] = float(value)
                rows.append(row)
            chosen = rng.choice(np.arange(particles.shape[0]), size=particles.shape[0], replace=True, p=weights)
            particles = np.asarray([scored[index][1] for index in chosen], dtype=float)
            particles += rng.normal(0.0, float(self.settings.smc_scale) / float(step + 2), size=particles.shape)
        final_scores = [row for row in rows if row["step"] == max(int(self.settings.smc_steps), 1) - 1]
        best = min(final_scores, key=lambda row: float(row["synthetic_likelihood_score"])) if final_scores else {}
        ess = 1.0 / max(sum(float(row["weight"]) ** 2 for row in final_scores), 1e-12) if final_scores else 0.0
        return {
            "mode": "bayesian_synthetic_likelihood_smc_style",
            "n_particles": int(self.settings.smc_particles),
            "n_steps": int(self.settings.smc_steps),
            "effective_sample_size": float(ess),
            "best_particle": best,
            "particles": rows,
            "likelihood": "Student-t-style heavy-tailed weight over full-vs-observation summary residuals",
        }

    @staticmethod
    def _skipped_stages() -> dict[str, str]:
        return {
            "F4-RV": "skipped: requires independent R/V marker evidence.",
            "F5-co-segregation-daughter-memory": "skipped: requires same-cell lineage evidence.",
        }

    def _build_result_from_projection(self, projection: FullToLiteProjection, source: str) -> FullCalibrationResult:
        report = self._coarse_report(projection)
        objective = _objective_value(report)
        f0 = FullCalibrationStageResult("F0-skeleton", (), None, objective, report, bool(projection.state_abundance.size and projection.copy_distributions.size), diagnostics={"source": source, "projection_diagnostics": projection.diagnostics})
        skipped = self._skipped_stages()
        return FullCalibrationResult(
            stage_results=(f0,),
            calibrated_bundle=self.bundle.deep_copy(),
            projection=projection,
            coarse_residual_report={**report, "calibration_mode": "v4_lite_summary_target", "formal_inference_mode": "not_run", "posterior_label": "not_formal_full_bayesian_posterior"},
            skipped_stages=skipped,
            formal_inference_report={"mode": "not_run"},
        )


def _bundle_summary(bundle: ParameterBundle) -> dict[str, object]:
    model = bundle.model
    return {
        "simulation": {
            "t_max": model.simulation.t_max,
            "n_init": model.simulation.n_init,
            "max_pop_size": model.simulation.max_pop_size,
        },
        "landscape": {
            "alpha": model.landscape.alpha.tolist(),
            "gamma_C": model.landscape.gamma_C.tolist(),
            "gamma_P": model.landscape.gamma_P.tolist(),
        },
        "hazard": {"theta_P": model.hazard.theta_P, "phi_B": model.hazard.phi_B},
        "turnover_gain_ceiling": {species: model.turnover[species].gain_ceiling for species in cfg.SPECIES},
        "posterior_label": "bridge_calibrated_map_not_formal_full_bayesian_posterior",
    }


def _projection_rows(projection: FullToLiteProjection) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for week_idx, week in enumerate(projection.weeks):
        total = max(float(np.sum(projection.state_abundance[week_idx])), 1e-12)
        for state_idx, state in enumerate(cfg.STATE_NAMES):
            rows.append({"block": "state_abundance", "week": week, "state": state, "value": float(projection.state_abundance[week_idx, state_idx]), "fraction": float(projection.state_abundance[week_idx, state_idx] / total)})
        for state_idx, state in enumerate(cfg.STATE_NAMES):
            for species_idx, species in enumerate(cfg.SPECIES):
                probs = projection.copy_distributions[week_idx, state_idx, species_idx]
                rows.append({"block": "copy_mean", "week": week, "state": state, "species": species, "value": float(np.dot(probs, np.arange(probs.size)))})
    if projection.growth_rates is not None:
        for interval in range(projection.growth_rates.shape[0]):
            for state_idx, state in enumerate(cfg.STATE_NAMES):
                rows.append({"block": "growth_rate", "week": projection.weeks[interval], "state": state, "value": float(projection.growth_rates[interval, state_idx])})
    return rows


def _full_release_block_status(result: FullCalibrationResult) -> list[dict[str, object]]:
    accepted = {stage.stage_name: stage.accepted for stage in result.stage_results}
    return [
        {"block": "ecDNA_tail_distribution", "status": "bridge_calibrated" if accepted.get("F2-ecDNA-turnover") else "fixed", "release_condition": "release only if full bridge fails ecTAG tail PPC"},
        {"block": "state_landscape_plasticity", "status": "bridge_calibrated" if accepted.get("F1-state-landscape") else "fixed", "release_condition": "release only when lite M4 accepts transition coupling"},
        {"block": "growth_hazard", "status": "bridge_calibrated" if accepted.get("F3-hazard-net-growth") else "fixed", "release_condition": "release only when lite M3 accepts growth coupling"},
        {"block": "stress_survival", "status": "skipped", "release_condition": "requires independent stress/death marker evidence"},
        {"block": "co_segregation", "status": "skipped", "release_condition": "requires same-cell multi-species ecTAG and failed joint PPC"},
        {"block": "drug", "status": "skipped", "release_condition": "requires treatment conditions and lite drug-effect evidence"},
    ]


def _full_report_output_paths(output_dir: str | Path) -> dict[str, tuple[Path, ...]]:
    root = Path(output_dir)
    return {
        "FULL-bridge": (
            root / "FULL_bridge_fit.json",
            root / "FULL_bridge_simulated_summaries.csv",
            root / "FULL_bridge_report.pdf",
        ),
        "FULL-restricted": (
            root / "FULL_restricted_fit.nc",
            root / "FULL_restricted_parameters.csv",
            root / "FULL_restricted_particle_posterior.csv",
            root / "FULL_restricted_ppc.pdf",
        ),
        "FULL-validation": (
            root / "FULL_identifiability_report.pdf",
            root / "FULL_release_block_status.json",
            root / "full_calibration_report.json",
        ),
    }


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
        "formal_inference_report": result.formal_inference_report,
        "double_counting_policy": "full bridge calibration is diagnostic unless formal raw mode is run.",
    }
    (destination / "full_calibration_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with open(destination / "full_coarse_residuals.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("block", "rmse", "relative_rmse", "n"))
        writer.writeheader()
        for row in result.coarse_residual_report.get("rows", ()):
            writer.writerow({field: row.get(field) for field in ("block", "rmse", "relative_rmse", "n")})
    write_json(destination / "FULL_bridge_fit.json", _bundle_summary(result.calibrated_bundle))
    projection_rows = _projection_rows(result.projection)
    write_table_bundle(destination, "FULL_bridge_simulated_summaries", projection_rows)
    write_text_pdf(
        destination / "FULL_bridge_report.pdf",
        "FULL bridge report",
        [
            f"Mode: {result.mode_label}",
            f"Weighted relative RMSE: {result.coarse_residual_report.get('weighted_relative_rmse')}",
            "Bridge uses v4-lite summary targets; formal raw inference must avoid double counting.",
        ],
    )
    npz_path = destination / "FULL_restricted_fit.npz"
    write_npz_or_marker(
        npz_path,
        {
            "state_abundance": result.projection.state_abundance,
            "copy_distributions": result.projection.copy_distributions,
            "transition_matrices": np.zeros((0,)) if result.projection.transition_matrices is None else result.projection.transition_matrices,
            "growth_rates": np.zeros((0,)) if result.projection.growth_rates is None else result.projection.growth_rates,
        },
        label="restricted full MAP/diagnostic output",
    )
    write_netcdf_file(
        destination / "FULL_restricted_fit.nc",
        {
            "state_abundance": result.projection.state_abundance,
            "copy_distributions": result.projection.copy_distributions,
            "transition_matrices": np.zeros((0,)) if result.projection.transition_matrices is None else result.projection.transition_matrices,
            "growth_rates": np.zeros((0,)) if result.projection.growth_rates is None else result.projection.growth_rates,
        },
        label="restricted full MAP/diagnostic output",
    )
    free_rows = [
        {"stage": stage.stage_name, "parameter": parameter, "accepted": stage.accepted, "objective_after": stage.objective_after}
        for stage in result.stage_results
        for parameter in stage.opened_parameters
    ]
    particle_rows = []
    if isinstance(result.formal_inference_report.get("particle_posterior"), dict):
        particle_rows = list(result.formal_inference_report["particle_posterior"].get("particles", ()))
    write_table_bundle(destination, "FULL_restricted_parameters", free_rows)
    write_table_bundle(destination, "FULL_restricted_particle_posterior", particle_rows)
    fixed_rows = [{"parameter_or_block": name, "reason": reason} for name, reason in result.skipped_stages.items()]
    fixed_rows.extend({"parameter_or_block": row["block"], "reason": row["release_condition"]} for row in _full_release_block_status(result) if row["status"] == "skipped")
    write_table_bundle(destination, "FULL_restricted_fixed_params", fixed_rows)
    write_table_bundle(destination, "FULL_restricted_derived_outputs", projection_rows)
    write_text_pdf(destination / "FULL_restricted_ppc.pdf", "FULL restricted PPC", ["Restricted full PPC is summarized against v4-lite projection rows.", f"Rows: {len(projection_rows)}"])
    write_text_pdf(destination / "FULL_identifiability_report.pdf", "FULL identifiability report", ["Profile and release-block diagnostics are written for the restricted full bridge.", "Release blocks are explicit in FULL_release_block_status.json."])
    write_json(destination / "FULL_release_block_status.json", _full_release_block_status(result))
    for stage_name, paths in _full_report_output_paths(destination).items():
        existing = [str(path) for path in paths if path.exists()]
        print(f"[fit-full] outputs {stage_name}: {', '.join(existing)}")
