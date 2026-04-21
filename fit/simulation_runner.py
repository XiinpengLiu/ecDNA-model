"""
Execution layer for running the full simulator under fitting control.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Iterable

import config as cfg
from core.simulation import SimulationResult, run_simulation
from fit.data import CanonicalFitDataset
from fit.parameter_registry import ParameterBundle


def _default_record_times() -> tuple[float, ...]:
    return tuple(float(week - 1) for week in range(1, 11))


@dataclass(frozen=True)
class FitRunnerConfig:
    record_times: tuple[float, ...] = _default_record_times()
    t_max: float = 9.0
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108)
    max_pop_size: int = 200000
    verbose: bool = False

    def __post_init__(self) -> None:
        cfg.require(bool(self.record_times), "record_times must be non-empty.")
        cfg.require(self.record_times[0] == 0.0, "Fitting runner requires week1 to map to t=0.")
        cfg.require(abs(float(self.record_times[-1]) - self.t_max) <= 1e-8, "t_max must equal the last record time.")
        cfg.require(np_all_diff_positive(self.record_times), "record_times must be strictly increasing.")
        cfg.require(bool(self.seeds), "At least one Monte Carlo seed is required.")
        cfg.require(self.max_pop_size > 0, "max_pop_size must be positive.")


def np_all_diff_positive(values: Iterable[float]) -> bool:
    sequence = tuple(float(value) for value in values)
    return all(next_value > current_value for current_value, next_value in zip(sequence[:-1], sequence[1:]))


@dataclass(frozen=True)
class SimulationRunSet:
    runs: dict[str, tuple[SimulationResult, ...]]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        counts = {len(results) for results in self.runs.values()}
        cfg.require(len(counts) <= 1, "Every condition must use the same number of Monte Carlo replicates.")
        cfg.require(bool(self.runs), "SimulationRunSet requires at least one condition.")

    def condition_names(self) -> tuple[str, ...]:
        return tuple(self.runs.keys())

    def n_replicates(self) -> int:
        return len(next(iter(self.runs.values())))


class FitSimulationRunner:
    def __init__(
        self,
        dataset: CanonicalFitDataset,
        *,
        config: FitRunnerConfig | None = None,
        initialization_template: cfg.InitializationParameters | None = None,
    ):
        self.dataset = dataset
        self.config = FitRunnerConfig() if config is None else config
        self.initialization_template = (
            copy.deepcopy(cfg.DEFAULT_INITIALIZATION_PARAMETERS)
            if initialization_template is None
            else copy.deepcopy(initialization_template)
        )

    def baseline_conditions(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.dataset.conditions.items() if spec.is_baseline())

    def drug_conditions(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.dataset.conditions.items() if spec.has_drug_input())

    def cue_conditions(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.dataset.conditions.items() if spec.has_cue_input())

    def prepare_bundle(self, bundle: ParameterBundle) -> ParameterBundle:
        simulation_params = replace(
            bundle.model.simulation,
            t_max=float(self.config.t_max),
            record_times=tuple(float(value) for value in self.config.record_times),
            target_population_size=None,
            max_pop_size=int(self.config.max_pop_size),
            fitting_mode=True,
            record_full_snapshots=False,
            record_events=False,
        )
        model = replace(bundle.model, simulation=simulation_params)
        prepared = ParameterBundle(model=copy.deepcopy(model), observation=copy.deepcopy(bundle.observation))
        cfg.validate_model_parameters(prepared.model)
        cfg.validate_observation_parameters(prepared.observation)
        return prepared

    def run_bundle(
        self,
        bundle: ParameterBundle,
        *,
        condition_names: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> SimulationRunSet:
        selected_conditions = tuple(self.dataset.condition_names() if condition_names is None else condition_names)
        cfg.require(bool(selected_conditions), "At least one condition must be selected for simulation.")
        seed_values = tuple(self.config.seeds if seeds is None else tuple(int(seed) for seed in seeds))
        cfg.require(bool(seed_values), "At least one seed is required for simulation.")

        prepared_bundle = self.prepare_bundle(bundle)
        runs: dict[str, tuple[SimulationResult, ...]] = {}
        for condition_name in selected_conditions:
            cfg.require(condition_name in self.dataset.conditions, f"Unknown simulation condition {condition_name}.")
            condition_spec = self.dataset.conditions[condition_name]
            initialization = self.dataset.build_empirical_initialization(
                condition_name,
                template=self.initialization_template,
            )
            schedules = condition_spec.build_input_schedules()
            condition_results: list[SimulationResult] = []
            for seed in seed_values:
                result = run_simulation(
                    params=prepared_bundle.model,
                    observation_params=prepared_bundle.observation,
                    initialization=initialization,
                    input_schedules=schedules,
                    seed=int(seed),
                    record_times=self.config.record_times,
                    t_max=self.config.t_max,
                    max_pop_size=self.config.max_pop_size,
                    verbose=self.config.verbose,
                )
                condition_results.append(result)
            runs[condition_name] = tuple(condition_results)
        return SimulationRunSet(runs=runs, seeds=seed_values)
