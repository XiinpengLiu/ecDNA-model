"""
Single-cell state and population utilities for the ecDNA v4 model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import v4_config as cfg

if TYPE_CHECKING:
    import v4_dynamics as dyn


@dataclass(eq=False)
class Cell:
    cycle_state: int
    copy_numbers: np.ndarray
    latent_state: np.ndarray
    soft_state: np.ndarray
    stress: float
    survival: float
    age: float
    cell_id: int = 0
    parent_id: int | None = None
    last_update_time: float = 0.0
    last_D_C: float = 0.0
    last_D_P: float = 0.0
    _derived_cache: "dyn.DerivedQuantities | None" = field(default=None, init=False, repr=False)
    _derived_context_key: tuple[float, float, float, float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.copy_numbers = np.asarray(self.copy_numbers, dtype=int)
        self.latent_state = np.asarray(self.latent_state, dtype=float)
        self.soft_state = np.asarray(self.soft_state, dtype=float)
        self.validate()

    def validate(self) -> None:
        cfg.validate_cycle_state(self.cycle_state)
        cfg.validate_copy_vector(self.copy_numbers)
        cfg.require(self.latent_state.shape == (cfg.LATENT_DIM,), "Latent state must have dimension 3.")
        cfg.validate_simplex(self.soft_state)
        cfg.require(np.isfinite(self.stress), "Stress must be finite.")
        cfg.require(np.isfinite(self.survival), "Survival reserve must be finite.")
        cfg.require(self.age >= 0.0, "Cell age must be non-negative.")
        cfg.require(self.last_update_time >= 0.0, "Last update time must be non-negative.")
        cfg.require(self.last_D_C >= 0.0, "Stored D_C must be non-negative.")
        cfg.require(self.last_D_P >= 0.0, "Stored D_P must be non-negative.")

    def copy(self, validate: bool = False) -> "Cell":
        duplicate = object.__new__(Cell)
        duplicate.cycle_state = self.cycle_state
        duplicate.copy_numbers = self.copy_numbers.copy()
        duplicate.latent_state = self.latent_state.copy()
        duplicate.soft_state = self.soft_state.copy()
        duplicate.stress = float(self.stress)
        duplicate.survival = float(self.survival)
        duplicate.age = float(self.age)
        duplicate.cell_id = self.cell_id
        duplicate.parent_id = self.parent_id
        duplicate.last_update_time = float(self.last_update_time)
        duplicate.last_D_C = float(self.last_D_C)
        duplicate.last_D_P = float(self.last_D_P)
        duplicate._derived_cache = self._derived_cache
        duplicate._derived_context_key = self._derived_context_key
        if validate:
            duplicate.validate()
        return duplicate

    def overwrite_state_from(self, other: "Cell", validate: bool = True) -> None:
        self.cycle_state = other.cycle_state
        self.copy_numbers[...] = other.copy_numbers
        self.latent_state[...] = other.latent_state
        self.soft_state[...] = other.soft_state
        self.stress = float(other.stress)
        self.survival = float(other.survival)
        self.age = float(other.age)
        self.last_update_time = float(other.last_update_time)
        self.last_D_C = float(other.last_D_C)
        self.last_D_P = float(other.last_D_P)
        self._derived_cache = other._derived_cache
        self._derived_context_key = other._derived_context_key
        if validate:
            self.validate()

    def invalidate_derived_cache(self) -> None:
        self._derived_cache = None
        self._derived_context_key = None

    def get_cached_derived_quantities(self, context: "dyn.ReplicateContext") -> "dyn.DerivedQuantities | None":
        context_key = (context.D_C, context.D_P, context.astrocytic_cue, context.mesenchymal_cue)
        if self._derived_context_key != context_key:
            return None
        return self._derived_cache

    def cache_derived_quantities(self, context: "dyn.ReplicateContext", derived: "dyn.DerivedQuantities") -> None:
        self._derived_context_key = (context.D_C, context.D_P, context.astrocytic_cue, context.mesenchymal_cue)
        self._derived_cache = derived

    def get_state_dict(self) -> dict:
        return {
            "cycle_state": cfg.CYCLE_NAMES[self.cycle_state],
            "cycle_index": self.cycle_state,
            "copy_numbers": self.copy_numbers.tolist(),
            "soft_state": self.soft_state.tolist(),
            "latent_state": self.latent_state.tolist(),
            "stress": float(self.stress),
            "survival": float(self.survival),
            "age": float(self.age),
            "cell_id": self.cell_id,
            "parent_id": self.parent_id,
            "last_update_time": float(self.last_update_time),
        }


class CellPopulation:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng(cfg.PARAMS.simulation.random_seed)
        self.cells: list[Cell] = []
        self.next_id = 0
        self.events: list[tuple[float, str, int, dict]] = []

    def add_cell(self, cell: Cell) -> Cell:
        cell.cell_id = self.next_id
        self.next_id += 1
        cell.validate()
        self.cells.append(cell)
        return cell

    def remove_cell(self, cell: Cell) -> None:
        self.cells.remove(cell)

    def size(self) -> int:
        return len(self.cells)

    def initialize(self, n: int | None = None) -> None:
        import v4_dynamics as dyn

        target_n = cfg.PARAMS.simulation.n_init if n is None else n
        base_context = dyn.ReplicateContext(
            time=0.0,
            u_C=0.0,
            u_P=0.0,
            D_C=cfg.PARAMS.exposure.D_C0,
            D_P=cfg.PARAMS.exposure.D_P0,
            astrocytic_cue=0.0,
            mesenchymal_cue=0.0,
        )
        for _ in range(target_n):
            copy_numbers = cfg.sample_initial_copy_numbers(self.rng)
            soft_state = cfg.sample_initial_soft_state(self.rng)
            latent_state = cfg.ilr(soft_state)
            cycle_state = cfg.sample_initial_cycle_state(self.rng)
            initial_cell = Cell(
                cycle_state=cycle_state,
                copy_numbers=copy_numbers,
                latent_state=latent_state,
                soft_state=soft_state,
                stress=0.0,
                survival=0.0,
                age=cfg.sample_initial_age(self.rng),
                last_update_time=0.0,
                last_D_C=cfg.PARAMS.exposure.D_C0,
                last_D_P=cfg.PARAMS.exposure.D_P0,
            )
            derived = dyn.compute_derived_quantities(initial_cell, base_context)
            initial_cell.stress = float(dyn.compute_stress_attractor(initial_cell, derived, base_context))
            initial_cell.survival = float(dyn.compute_survival_attractor(initial_cell, derived, base_context))
            self.add_cell(initial_cell)

    def log_event(self, time: float, event_type: str, cell_id: int, details: dict | None = None) -> None:
        self.events.append((time, event_type, cell_id, details or {}))

    def summary(self, context: "dyn.ReplicateContext") -> dict:
        import v4_dynamics as dyn

        if not self.cells:
            return {
                "population_size": 0,
                "state_fractions": np.zeros(cfg.N_STATES),
                "cycle_fractions": np.zeros(cfg.N_CYCLE),
                "bulk_copy_means": np.zeros(cfg.N_SPECIES),
                "mean_stress": 0.0,
                "mean_survival": 0.0,
                "mean_division_hazard": 0.0,
                "mean_death_hazard": 0.0,
            }

        state_totals = np.zeros(cfg.N_STATES, dtype=float)
        cycle_counts = np.zeros(cfg.N_CYCLE, dtype=float)
        copy_totals = np.zeros(cfg.N_SPECIES, dtype=float)
        stress_total = 0.0
        survival_total = 0.0
        division_total = 0.0
        death_total = 0.0
        for cell in self.cells:
            state_totals += cell.soft_state
            cycle_counts[cell.cycle_state] += 1.0
            copy_totals += cell.copy_numbers
            stress_total += cell.stress
            survival_total += cell.survival
            derived = dyn.compute_derived_quantities(cell, context)
            division_total += dyn.compute_division_hazard(cell, derived, context)
            death_total += dyn.compute_death_hazard(cell, derived, context)

        count = float(len(self.cells))

        return {
            "population_size": len(self.cells),
            "state_fractions": state_totals / count,
            "cycle_fractions": cycle_counts / count,
            "bulk_copy_means": copy_totals / count,
            "mean_stress": float(stress_total / count),
            "mean_survival": float(survival_total / count),
            "mean_division_hazard": float(division_total / count),
            "mean_death_hazard": float(death_total / count),
        }
