"""Minimal full-simulation runner compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass

from core.simulation import SimulationResult, run_simulation
from fit.data import CanonicalFitDataset
from fit.parameter_registry import ParameterBundle


@dataclass(frozen=True)
class SimulationRunSet:
    runs: dict[str, tuple[SimulationResult, ...]]

    def condition_names(self) -> tuple[str, ...]:
        return tuple(self.runs)

    def n_replicates(self) -> int:
        return min((len(v) for v in self.runs.values()), default=0)


class FitSimulationRunner:
    def __init__(self, dataset: CanonicalFitDataset, seeds: tuple[int, ...] = (1,), verbose: bool = False):
        self.dataset = dataset
        self.seeds = tuple(seeds)
        self.verbose = verbose

    def run_bundle(self, bundle: ParameterBundle, condition_names: tuple[str, ...] | None = None) -> SimulationRunSet:
        selected = self.dataset.condition_names() if condition_names is None else condition_names
        runs: dict[str, tuple[SimulationResult, ...]] = {}
        for condition in selected:
            runs[condition] = tuple(
                run_simulation(
                    params=bundle.model,
                    observation_params=bundle.observation,
                    initialization=self.dataset.build_empirical_initialization(condition),
                    input_schedules=self.dataset.conditions[condition].build_input_schedules(),
                    seed=seed,
                    verbose=self.verbose,
                )
                for seed in self.seeds
            )
        return SimulationRunSet(runs)
