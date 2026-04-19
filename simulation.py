"""
Approximate hybrid continuous-time simulation for the ecDNA v4 model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
import csv
import json

import numpy as np

import config as cfg
import dynamics as dyn
from cell import Cell, CellPopulation
from division import DivisionKernel


@dataclass
class SimulationResult:
    times: list[float] = field(default_factory=list)
    population_sizes: list[int] = field(default_factory=list)
    state_fractions: list[np.ndarray] = field(default_factory=list)
    cycle_fractions: list[np.ndarray] = field(default_factory=list)
    bulk_copy_means: list[np.ndarray] = field(default_factory=list)
    mean_stress: list[float] = field(default_factory=list)
    mean_survival: list[float] = field(default_factory=list)
    mean_division_hazard: list[float] = field(default_factory=list)
    mean_death_hazard: list[float] = field(default_factory=list)
    exposures: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    cell_snapshots: list[list[dict]] = field(default_factory=list)
    ecdna_distributions: list[np.ndarray] = field(default_factory=list)
    events: list[tuple[float, str, int, dict]] = field(default_factory=list)
    stop_time: float | None = None
    stop_reason: str = ""

    def save_as_csv(self, base_dir: str | Path) -> None:
        output_dir = Path(base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time",
                    "population_size",
                    "npc_fraction",
                    "opc_fraction",
                    "ac_fraction",
                    "mes_fraction",
                    "mean_myc",
                    "mean_cdk4",
                    "mean_pdgfra",
                    "mean_stress",
                    "mean_survival",
                    "mean_division_hazard",
                    "mean_death_hazard",
                    "D_C",
                    "D_P",
                    "a",
                    "m",
                ]
            )
            for idx, time in enumerate(self.times):
                writer.writerow(
                    [
                        time,
                        self.population_sizes[idx],
                        *self.state_fractions[idx].tolist(),
                        *self.bulk_copy_means[idx].tolist(),
                        self.mean_stress[idx],
                        self.mean_survival[idx],
                        self.mean_division_hazard[idx],
                        self.mean_death_hazard[idx],
                        self.exposures[idx]["D_C"],
                        self.exposures[idx]["D_P"],
                        self.exposures[idx]["a"],
                        self.exposures[idx]["m"],
                    ]
                )

        with open(output_dir / "snapshots.jsonl", "w", encoding="utf-8") as handle:
            for time, snapshot in zip(self.times, self.cell_snapshots):
                handle.write(json.dumps({"time": time, "cells": snapshot}) + "\n")

        with open(output_dir / "events.jsonl", "w", encoding="utf-8") as handle:
            for time, event_type, cell_id, details in self.events:
                handle.write(json.dumps({"time": time, "event_type": event_type, "cell_id": cell_id, "details": details}) + "\n")


class HybridOgataSimulator:
    def __init__(self, input_schedules: dict[str, callable] | None = None, seed: int | None = None):
        self.params = cfg.PARAMS
        self.seed = self.params.simulation.random_seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.input_schedules = dict(cfg.DEFAULT_INPUT_SCHEDULES)
        if input_schedules is not None:
            self.input_schedules.update(input_schedules)
        self.division_kernel = DivisionKernel(self.rng)
        self.r_bar = self.compute_dominating_bound()

    def compute_dominating_bound(self) -> float:
        cycle_bound = (
            self.params.cycle.qbar_G1S
            + self.params.cycle.qbar_G1Q
            + self.params.cycle.qbar_QG1
            + self.params.cycle.qbar_SG2M
        )
        turnover_bound = 0.0
        for species_params in self.params.turnover.values():
            turnover_bound += species_params.gain_ceiling + species_params.loss_ceiling
        hazard_bound = self.params.hazard.lambda_div_ceiling + self.params.hazard.lambda_death_ceiling
        bound = cycle_bound + turnover_bound + hazard_bound
        cfg.require(bound > 0.0, "Dominating bound must be strictly positive.")
        return float(bound)

    def build_context(self, time: float, D_C: float, D_P: float) -> dyn.ReplicateContext:
        return dyn.ReplicateContext(
            time=time,
            u_C=float(self.input_schedules["u_C"](time)),
            u_P=float(self.input_schedules["u_P"](time)),
            D_C=D_C,
            D_P=D_P,
            astrocytic_cue=float(self.input_schedules["a"](time)),
            mesenchymal_cue=float(self.input_schedules["m"](time)),
        )

    def print_monitor_line(self, time: float, summary: dict, event_counts: dict[str, int], label: str) -> None:
        state_text = ", ".join(
            f"{name}={summary['state_fractions'][idx]:.2f}"
            for idx, name in enumerate(cfg.STATE_NAMES)
        )
        cycle_text = ", ".join(
            f"{name}={summary['cycle_fractions'][idx]:.2f}"
            for idx, name in enumerate(cfg.CYCLE_NAMES)
        )
        copy_text = ", ".join(
            f"{name}={summary['bulk_copy_means'][idx]:.2f}"
            for idx, name in enumerate(cfg.SPECIES)
        )
        event_text = ", ".join(f"{name}={count}" for name, count in sorted(event_counts.items())) if event_counts else "none"
        print(
            f"[monitor:{label}] "
            f"t={time:.2f} "
            f"pop={summary['population_size']} "
            f"states[{state_text}] "
            f"cycle[{cycle_text}] "
            f"ecDNA[{copy_text}] "
            f"R={summary['mean_stress']:.3f} "
            f"V={summary['mean_survival']:.3f} "
            f"haz_div={summary['mean_division_hazard']:.3f} "
            f"haz_death={summary['mean_death_hazard']:.3f} "
            f"events[{event_text}]"
        )

    def exposure_step(self, current_exposure: float, dose: float, decay: float, conversion: float, dt: float) -> float:
        if decay <= 1e-12:
            next_value = current_exposure + conversion * dose * dt
        else:
            exp_decay = np.exp(-decay * dt)
            next_value = current_exposure * exp_decay + (conversion * dose / decay) * (1.0 - exp_decay)
        cfg.require(next_value >= -1e-10, "Integrated exposure must remain non-negative.")
        return max(0.0, float(next_value))

    def advance_cell_to_time(self, cell: Cell, target_time: float, rng: np.random.Generator | None = None) -> dyn.ReplicateContext:
        active_rng = self.rng if rng is None else rng
        cfg.require(target_time >= cell.last_update_time, "Cannot move a cell backwards in time.")

        current_time = cell.last_update_time
        current_D_C = cell.last_D_C
        current_D_P = cell.last_D_P

        while current_time < target_time - 1e-12:
            step = min(self.params.simulation.dt, target_time - current_time)
            midpoint = current_time + 0.5 * step
            u_C_mid = float(self.input_schedules["u_C"](midpoint))
            u_P_mid = float(self.input_schedules["u_P"](midpoint))
            a_mid = float(self.input_schedules["a"](midpoint))
            m_mid = float(self.input_schedules["m"](midpoint))

            next_D_C = self.exposure_step(current_D_C, u_C_mid, self.params.exposure.k_C, self.params.exposure.eta_C, step)
            next_D_P = self.exposure_step(current_D_P, u_P_mid, self.params.exposure.k_P, self.params.exposure.eta_P, step)

            midpoint_context = dyn.ReplicateContext(
                time=midpoint,
                u_C=u_C_mid,
                u_P=u_P_mid,
                D_C=0.5 * (current_D_C + next_D_C),
                D_P=0.5 * (current_D_P + next_D_P),
                astrocytic_cue=a_mid,
                mesenchymal_cue=m_mid,
            )
            dyn.update_continuous_state(cell, midpoint_context, step, active_rng)
            current_time += step
            current_D_C = next_D_C
            current_D_P = next_D_P

        cell.last_update_time = float(target_time)
        cell.last_D_C = float(current_D_C)
        cell.last_D_P = float(current_D_P)
        return self.build_context(target_time, current_D_C, current_D_P)

    def synchronize_population_to_time(self, population: CellPopulation, target_time: float) -> dyn.ReplicateContext:
        context = self.build_context(target_time, self.params.exposure.D_C0, self.params.exposure.D_P0)
        for cell in population.cells:
            context = self.advance_cell_to_time(cell, target_time, self.rng)
        if population.cells:
            context = self.build_context(target_time, population.cells[0].last_D_C, population.cells[0].last_D_P)
        return context

    def projection_rng(self, cell: Cell, target_time: float) -> np.random.Generator:
        time_bits = int(np.float64(target_time).view(np.uint64))
        last_bits = int(np.float64(cell.last_update_time).view(np.uint64))
        seed_value = (
            int(self.seed)
            ^ ((cell.cell_id + 1) * 0x9E3779B185EBCA87)
            ^ (time_bits * 0xC2B2AE3D27D4EB4F)
            ^ (last_bits * 0x165667B19E3779F9)
        ) & ((1 << 63) - 1)
        return np.random.default_rng(seed_value)

    def project_cell_for_observation(self, cell: Cell, target_time: float) -> tuple[Cell, dyn.ReplicateContext]:
        if target_time <= cell.last_update_time + 1e-12:
            return cell, self.build_context(target_time, cell.last_D_C, cell.last_D_P)
        projected_cell = cell.copy()
        projection_context = self.advance_cell_to_time(
            projected_cell,
            target_time,
            rng=self.projection_rng(cell, target_time),
        )
        return projected_cell, projection_context

    def sample_next_event(
        self,
        cell: Cell,
        t_start: float,
        t_max: float,
    ) -> tuple[float | None, str | None, Cell | None, dict | None]:
        current_t = t_start
        temp_cell = cell.copy()
        proposals = 0

        while current_t < t_max:
            delta = float(self.rng.exponential(1.0 / self.r_bar))
            candidate_t = current_t + delta
            proposals += 1

            if candidate_t >= t_max:
                return None, None, None, None

            context = self.advance_cell_to_time(temp_cell, candidate_t, self.rng)
            derived = dyn.compute_derived_quantities(temp_cell, context)
            rates = dyn.compute_all_event_rates(temp_cell, derived, context)
            total_rate = float(sum(rates.values()))
            accept_prob = total_rate / self.r_bar

            if total_rate > 0.0 and self.rng.random() < accept_prob:
                names = list(rates.keys())
                probabilities = np.array([rates[name] for name in names], dtype=float) / total_rate
                selected_event = str(self.rng.choice(names, p=probabilities))
                return (
                    candidate_t,
                    selected_event,
                    temp_cell,
                    {"proposals": proposals, "accept_prob": accept_prob, "total_rate": total_rate},
                )

            current_t = candidate_t

        return None, None, None, None

    def simulate(
        self,
        population: CellPopulation | None = None,
        t_max: float | None = None,
        record_interval: float | None = None,
        target_population_size: int | None = None,
        max_pop_size: int | None = None,
        verbose: bool = True,
    ) -> SimulationResult:
        final_time = self.params.simulation.t_max if t_max is None else t_max
        snapshot_interval = self.params.simulation.record_interval if record_interval is None else record_interval
        target_pop_size = (
            self.params.simulation.target_population_size if target_population_size is None else target_population_size
        )
        hard_pop_limit = self.params.simulation.max_pop_size if max_pop_size is None else max_pop_size
        cfg.require(final_time > 0.0, "Simulation t_max must be strictly positive.")
        cfg.require(snapshot_interval > 0.0, "Record interval must be strictly positive.")
        cfg.require(hard_pop_limit > 0, "Population hard limit must be strictly positive.")
        if target_pop_size is not None:
            cfg.require(target_pop_size > 0, "Target population size must be strictly positive.")
            cfg.require(
                target_pop_size <= hard_pop_limit,
                "Target population size cannot exceed the hard population limit.",
            )

        if population is None:
            population = CellPopulation(self.rng)
            population.initialize(self.params.simulation.n_init)

        for cell in population.cells:
            cell.last_update_time = 0.0
            cell.last_D_C = self.params.exposure.D_C0
            cell.last_D_P = self.params.exposure.D_P0

        result = SimulationResult()
        time = 0.0
        next_record = snapshot_interval
        initial_context = self.build_context(time, self.params.exposure.D_C0, self.params.exposure.D_P0)
        event_counts: dict[str, int] = {}
        if verbose:
            print(
                "Simulation start: "
                f"t_max={final_time:.2f}, "
                f"record_interval={snapshot_interval:.2f}, "
                f"target_population_size={target_pop_size}, "
                f"max_pop_size={hard_pop_limit}, "
                f"n_init={population.size()}"
            )
        initial_summary = self.record_state(result, population, time, fallback_context=initial_context)
        if verbose:
            self.print_monitor_line(time, initial_summary, event_counts, label="initial")

        event_heap: list[tuple[float, int, str, int, Cell, dict]] = []
        cell_versions: dict[int, int] = {}
        cell_lookup: dict[int, Cell] = {cell.cell_id: cell for cell in population.cells}

        def population_stop_reason() -> str | None:
            size = population.size()
            if target_pop_size is not None and size >= target_pop_size:
                return "target_population_size"
            if size >= hard_pop_limit:
                return "max_pop_size"
            return None

        def set_stop(reason: str, stop_time: float) -> None:
            result.stop_reason = reason
            result.stop_time = float(stop_time)

        def record_terminal_state(stop_time: float, label: str, fallback_context: dyn.ReplicateContext | None = None) -> None:
            if result.times and stop_time <= result.times[-1] + 1e-12:
                return
            summary = self.record_state(result, population, stop_time, fallback_context=fallback_context)
            if verbose:
                self.print_monitor_line(stop_time, summary, event_counts, label=label)

        initial_stop_reason = population_stop_reason()
        if initial_stop_reason is not None:
            set_stop(initial_stop_reason, time)
            if verbose:
                print(f"Stopping immediately at t={time:.2f}: {initial_stop_reason} reached.")
            result.events = population.events.copy()
            return result

        def schedule_cell_event(cell: Cell, from_time: float) -> None:
            event_time, event_name, flow_cell, thinning_stats = self.sample_next_event(cell, from_time, final_time)
            version = cell_versions.get(cell.cell_id, 0) + 1
            cell_versions[cell.cell_id] = version
            if event_time is not None and event_name is not None and flow_cell is not None:
                heapq.heappush(event_heap, (event_time, cell.cell_id, event_name, version, flow_cell, thinning_stats or {}))

        def resample_all_events(from_time: float) -> None:
            event_heap.clear()
            for live_cell in population.cells:
                schedule_cell_event(live_cell, from_time)

        resample_all_events(time)

        while time < final_time and population.size() > 0:
            stop_reason = population_stop_reason()
            if stop_reason is not None:
                set_stop(stop_reason, time)
                if population.size() > 0:
                    record_terminal_state(time, label="stop")
                if verbose:
                    print(f"Stopping at t={time:.2f}: {stop_reason} reached.")
                break

            next_event_time = final_time
            next_cell = None
            next_event_name = None
            next_flow_cell = None
            next_event_entry: tuple[float, int, str, int, Cell, dict] | None = None

            while event_heap:
                event_entry = heapq.heappop(event_heap)
                event_time, cell_id, event_name, version, flow_cell, _stats = event_entry
                if cell_versions.get(cell_id) != version:
                    continue
                live_cell = cell_lookup.get(cell_id)
                if live_cell is None:
                    continue
                next_event_entry = event_entry
                next_event_time = event_time
                next_cell = live_cell
                next_event_name = event_name
                next_flow_cell = flow_cell
                break

            if next_record <= min(next_event_time, final_time):
                if next_event_entry is not None:
                    heapq.heappush(event_heap, next_event_entry)
                summary = self.record_state(result, population, next_record)
                if verbose:
                    self.print_monitor_line(next_record, summary, event_counts, label="snapshot")
                next_record += snapshot_interval
                continue

            if next_cell is None or next_event_name is None or next_flow_cell is None:
                break

            next_cell.overwrite_state_from(next_flow_cell, validate=False)
            time = next_event_time
            event_context = self.build_context(time, next_cell.last_D_C, next_cell.last_D_P)
            state_pre = next_cell.get_state_dict()

            if next_event_name == "death":
                population.remove_cell(next_cell)
                event_counts["death"] = event_counts.get("death", 0) + 1
                population.log_event(time, "death", next_cell.cell_id, {"state_pre": state_pre})
                cell_lookup.pop(next_cell.cell_id, None)
                cell_versions.pop(next_cell.cell_id, None)
                if population.size() == 0:
                    record_terminal_state(time, label="extinction", fallback_context=event_context)
                continue

            if next_event_name == "division":
                daughter_one, daughter_two = self.division_kernel.divide(next_cell, event_context)
                daughter_one.last_update_time = time
                daughter_one.last_D_C = next_cell.last_D_C
                daughter_one.last_D_P = next_cell.last_D_P
                daughter_two.last_update_time = time
                daughter_two.last_D_C = next_cell.last_D_C
                daughter_two.last_D_P = next_cell.last_D_P

                population.remove_cell(next_cell)
                daughter_one = population.add_cell(daughter_one)
                daughter_two = population.add_cell(daughter_two)
                event_counts["division"] = event_counts.get("division", 0) + 1
                population.log_event(
                    time,
                    "division",
                    next_cell.cell_id,
                    {
                        "state_pre": state_pre,
                        "daughter_one": daughter_one.get_state_dict(),
                        "daughter_two": daughter_two.get_state_dict(),
                    },
                )

                cell_lookup.pop(next_cell.cell_id, None)
                cell_versions.pop(next_cell.cell_id, None)
                cell_lookup[daughter_one.cell_id] = daughter_one
                cell_lookup[daughter_two.cell_id] = daughter_two
                schedule_cell_event(daughter_one, time)
                schedule_cell_event(daughter_two, time)
                stop_reason = population_stop_reason()
                if stop_reason is not None:
                    set_stop(stop_reason, time)
                    record_terminal_state(time, label="stop", fallback_context=event_context)
                    if verbose:
                        print(f"Stopping at t={time:.2f}: {stop_reason} reached after division.")
                    break
                continue

            dyn.apply_nonterminal_event(next_cell, next_event_name)
            next_cell.last_update_time = time
            next_cell.last_D_C = next_flow_cell.last_D_C
            next_cell.last_D_P = next_flow_cell.last_D_P
            event_counts[next_event_name] = event_counts.get(next_event_name, 0) + 1
            population.log_event(
                time,
                next_event_name,
                next_cell.cell_id,
                {
                    "state_pre": state_pre,
                    "state_post": next_cell.get_state_dict(),
                },
            )
            cell_lookup[next_cell.cell_id] = next_cell
            schedule_cell_event(next_cell, time)

        result.events = population.events.copy()
        if result.stop_reason == "":
            if population.size() == 0:
                set_stop("population_extinction", time)
            else:
                set_stop("t_max", final_time)
        if population.size() > 0 and result.times and result.times[-1] < final_time - 1e-12 and result.stop_reason == "t_max":
            final_summary = self.record_state(result, population, final_time)
            if verbose:
                self.print_monitor_line(final_time, final_summary, event_counts, label="final")
        return result

    def summarize_observed_cells(
        self,
        observed_cells: list[tuple[Cell, dyn.DerivedQuantities]],
    ) -> dict:
        if not observed_cells:
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
        for observed_cell, derived in observed_cells:
            state_totals += observed_cell.soft_state
            cycle_counts[observed_cell.cycle_state] += 1.0
            copy_totals += observed_cell.copy_numbers
            stress_total += observed_cell.stress
            survival_total += observed_cell.survival
            cell_context = self.build_context(observed_cell.last_update_time, observed_cell.last_D_C, observed_cell.last_D_P)
            division_total += dyn.compute_division_hazard(observed_cell, derived, cell_context)
            death_total += dyn.compute_death_hazard(observed_cell, derived, cell_context)

        count = float(len(observed_cells))
        return {
            "population_size": int(count),
            "state_fractions": state_totals / count,
            "cycle_fractions": cycle_counts / count,
            "bulk_copy_means": copy_totals / count,
            "mean_stress": float(stress_total / count),
            "mean_survival": float(survival_total / count),
            "mean_division_hazard": float(division_total / count),
            "mean_death_hazard": float(death_total / count),
        }

    def record_state(
        self,
        result: SimulationResult,
        population: CellPopulation,
        time: float,
        fallback_context: dyn.ReplicateContext | None = None,
    ) -> dict:
        observed_cells: list[tuple[Cell, dyn.DerivedQuantities]] = []
        snapshot: list[dict] = []
        distribution_rows = []
        context = fallback_context
        for live_cell in population.cells:
            observed_cell, context = self.project_cell_for_observation(live_cell, time)
            derived = dyn.compute_derived_quantities(observed_cell, context)
            observed_cells.append((observed_cell, derived))
            snapshot.append(
                {
                    "cell_id": observed_cell.cell_id,
                    "cycle_state": cfg.CYCLE_NAMES[observed_cell.cycle_state],
                    "copy_numbers": observed_cell.copy_numbers.tolist(),
                    "soft_state": observed_cell.soft_state.tolist(),
                    "stress": float(observed_cell.stress),
                    "survival": float(observed_cell.survival),
                    "division_hazard": dyn.compute_division_hazard(observed_cell, derived, context),
                    "death_hazard": dyn.compute_death_hazard(observed_cell, derived, context),
                    "local_transition_generator": dyn.compute_local_transition_generator(derived.logits).tolist(),
                }
            )
            distribution_rows.append(observed_cell.copy_numbers.copy())

        if context is None:
            context = self.build_context(time, self.params.exposure.D_C0, self.params.exposure.D_P0)

        summary = self.summarize_observed_cells(observed_cells)
        result.times.append(float(time))
        result.population_sizes.append(int(summary["population_size"]))
        result.state_fractions.append(summary["state_fractions"])
        result.cycle_fractions.append(summary["cycle_fractions"])
        result.bulk_copy_means.append(summary["bulk_copy_means"])
        result.mean_stress.append(summary["mean_stress"])
        result.mean_survival.append(summary["mean_survival"])
        result.mean_division_hazard.append(summary["mean_division_hazard"])
        result.mean_death_hazard.append(summary["mean_death_hazard"])
        result.exposures.append({"D_C": context.D_C, "D_P": context.D_P, "a": context.astrocytic_cue, "m": context.mesenchymal_cue})
        result.observations.append(
            {
                "flow_fractions": summary["state_fractions"].tolist(),
                "bulk_copy_means": summary["bulk_copy_means"].tolist(),
                "count_prediction": int(summary["population_size"]),
            }
        )
        result.cell_snapshots.append(snapshot)
        result.ecdna_distributions.append(np.array(distribution_rows, dtype=int) if distribution_rows else np.zeros((0, cfg.N_SPECIES), dtype=int))
        return summary


def run_simulation(
    t_max: float | None = None,
    n_init: int | None = None,
    input_schedules: dict[str, callable] | None = None,
    seed: int | None = None,
    record_interval: float | None = None,
    target_population_size: int | None = None,
    max_pop_size: int | None = None,
    verbose: bool = True,
) -> SimulationResult:
    simulator = HybridOgataSimulator(input_schedules=input_schedules, seed=seed)
    population = CellPopulation(np.random.default_rng(cfg.PARAMS.simulation.random_seed if seed is None else seed))
    population.initialize(cfg.PARAMS.simulation.n_init if n_init is None else n_init)
    return simulator.simulate(
        population=population,
        t_max=cfg.PARAMS.simulation.t_max if t_max is None else t_max,
        record_interval=record_interval,
        target_population_size=target_population_size,
        max_pop_size=max_pop_size,
        verbose=verbose,
    )
