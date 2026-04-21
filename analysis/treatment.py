"""
Treatment protocols and summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ecdna_model.core.simulation import SimulationResult, run_simulation


@dataclass(frozen=True)
class TreatmentProtocol:
    name: str
    schedules: dict[str, Callable[[float], float]]
    duration: float = 10.0


def constant_input(value: float, start: float = 0.0, end: float = np.inf) -> Callable[[float], float]:
    def schedule(time: float) -> float:
        return value if start <= time < end else 0.0

    return schedule


PROTOCOLS = {
    "untreated": TreatmentProtocol(
        name="Untreated",
        schedules={"u_C": lambda _t: 0.0, "u_P": lambda _t: 0.0, "a": lambda _t: 0.0, "m": lambda _t: 0.0},
    ),
    "cdk4_inhibition": TreatmentProtocol(
        name="CDK4 inhibitor",
        schedules={"u_C": constant_input(1.0, start=2.0), "u_P": lambda _t: 0.0, "a": lambda _t: 0.0, "m": lambda _t: 0.1},
    ),
    "pdgfra_inhibition": TreatmentProtocol(
        name="PDGFRA inhibitor",
        schedules={"u_C": lambda _t: 0.0, "u_P": constant_input(1.0, start=2.0), "a": lambda _t: 0.0, "m": lambda _t: 0.1},
    ),
    "mesenchymal_pressure": TreatmentProtocol(
        name="Mesenchymal cue",
        schedules={"u_C": lambda _t: 0.0, "u_P": lambda _t: 0.0, "a": lambda _t: 0.0, "m": constant_input(0.8, start=0.0)},
    ),
    "astrocytic_plus_cdk4i": TreatmentProtocol(
        name="Astrocytic cue plus CDK4 inhibitor",
        schedules={"u_C": constant_input(1.0, start=2.0), "u_P": lambda _t: 0.0, "a": constant_input(0.8, start=0.0), "m": lambda _t: 0.0},
    ),
}


class InSilicoTrial:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results: dict[str, list[SimulationResult]] = {}

    def run_protocol(
        self,
        protocol: TreatmentProtocol,
        n_replicates: int = 1,
        n_init: int | None = None,
        verbose: bool = True,
    ) -> list[SimulationResult]:
        outputs: list[SimulationResult] = []
        for replicate in range(n_replicates):
            result = run_simulation(
                t_max=protocol.duration,
                n_init=n_init,
                input_schedules=protocol.schedules,
                seed=self.base_seed + replicate,
                record_interval=1.0,
                verbose=verbose,
            )
            outputs.append(result)
        self.results[protocol.name] = outputs
        return outputs


def compute_growth_rate(result: SimulationResult, window: float = 2.0) -> float:
    times = np.asarray(result.times, dtype=float)
    populations = np.asarray(result.population_sizes, dtype=float)
    if len(times) < 2:
        return 0.0
    mask = times >= max(times[-1] - window, times[0])
    if np.sum(mask) < 2:
        return 0.0
    coefficients = np.polyfit(times[mask], np.log(populations[mask] + 1.0), 1)
    return float(coefficients[0])


def compute_bulk_copy_trends(result: SimulationResult) -> dict[str, float]:
    if len(result.bulk_copy_means) < 2:
        return {name: 0.0 for name in ("MYC", "CDK4", "PDGFRA")}
    times = np.asarray(result.times, dtype=float)
    bulk = np.asarray(result.bulk_copy_means, dtype=float)
    trends = {}
    for idx, name in enumerate(("MYC", "CDK4", "PDGFRA")):
        coeff = np.polyfit(times, bulk[:, idx], 1)
        trends[name] = float(coeff[0])
    return trends


def compute_terminal_event_counts(result: SimulationResult) -> dict[str, int]:
    counts = {"division": 0, "death": 0}
    for _, event_type, _, _ in result.events:
        if event_type in counts:
            counts[event_type] += 1
    return counts
