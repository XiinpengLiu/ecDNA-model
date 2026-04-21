"""
Hybrid continuous-time simulation for the ecDNA model.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field, replace
import heapq
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ecdna_model import config as cfg
from ecdna_model.core import dynamics as dyn
from ecdna_model.core import observation as obs
from ecdna_model.core.cell import Cell, CellPopulation
from ecdna_model.core.division import DivisionKernel


@dataclass
class SimulationResult:
    times: list[float] = field(default_factory=list)
    population_sizes: list[int] = field(default_factory=list)
    soft_state_fractions: list[np.ndarray] = field(default_factory=list)
    cycle_fractions: list[np.ndarray] = field(default_factory=list)
    bulk_copy_means: list[np.ndarray] = field(default_factory=list)
    mean_stress_scores: list[float] = field(default_factory=list)
    mean_survival_scores: list[float] = field(default_factory=list)
    mean_division_hazard: list[float] = field(default_factory=list)
    mean_death_hazard: list[float] = field(default_factory=list)
    exposures: list[dict] = field(default_factory=list)
    truth_snapshots: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    cell_snapshots: list[list[dict]] = field(default_factory=list)
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
                    "mean_stress_score",
                    "mean_survival_score",
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
                        *self.soft_state_fractions[idx].tolist(),
                        *self.bulk_copy_means[idx].tolist(),
                        self.mean_stress_scores[idx],
                        self.mean_survival_scores[idx],
                        self.mean_division_hazard[idx],
                        self.mean_death_hazard[idx],
                        self.exposures[idx]["D_C"],
                        self.exposures[idx]["D_P"],
                        self.exposures[idx]["a"],
                        self.exposures[idx]["m"],
                    ]
                )

        with open(output_dir / "truth_snapshots.jsonl", "w", encoding="utf-8") as handle:
            for time, snapshot in zip(self.times, self.truth_snapshots):
                handle.write(json.dumps({"time": time, "snapshot": snapshot}) + "\n")

        with open(output_dir / "observations.jsonl", "w", encoding="utf-8") as handle:
            for time, snapshot in zip(self.times, self.observations):
                handle.write(json.dumps({"time": time, "observation": snapshot}) + "\n")

        with open(output_dir / "snapshots.jsonl", "w", encoding="utf-8") as handle:
            for time, snapshot in zip(self.times, self.cell_snapshots):
                handle.write(json.dumps({"time": time, "cells": snapshot}) + "\n")

        with open(output_dir / "events.jsonl", "w", encoding="utf-8") as handle:
            for time, event_type, cell_id, details in self.events:
                handle.write(
                    json.dumps(
                        {"time": time, "event_type": event_type, "cell_id": cell_id, "details": details}
                    )
                    + "\n"
                )


class HybridOgataSimulator:
    def __init__(
        self,
        params: cfg.ModelParameters,
        observation_params: cfg.ObservationParameters,
        input_schedules: dict[str, callable] | None = None,
        seed: int | None = None,
        event_rng: np.random.Generator | None = None,
        observation_rng: np.random.Generator | None = None,
    ):
        self.params = copy.deepcopy(params)
        self.observation_params = copy.deepcopy(observation_params)
        cfg.validate_model_parameters(self.params)
        cfg.validate_observation_parameters(self.observation_params)
        self.seed = self.params.simulation.random_seed if seed is None else seed
        self.event_rng = np.random.default_rng(self.seed) if event_rng is None else event_rng
        self.observation_rng = np.random.default_rng(self.seed + 1) if observation_rng is None else observation_rng
        self.input_schedules = dict(cfg.DEFAULT_INPUT_SCHEDULES)
        if input_schedules is not None:
            self.input_schedules.update(input_schedules)
        self.division_kernel = DivisionKernel(self.params, self.event_rng)
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
            f"{name}={summary['soft_state_fractions'][idx]:.2f}"
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
            f"soft[{state_text}] "
            f"cycle[{cycle_text}] "
            f"ecDNA[{copy_text}] "
            f"R={summary['mean_stress_score']:.3f} "
            f"V={summary['mean_survival_score']:.3f} "
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
        active_rng = self.event_rng if rng is None else rng
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
            dyn.update_continuous_state(cell, midpoint_context, step, active_rng, self.params)
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
            context = self.advance_cell_to_time(cell, target_time, self.event_rng)
        if population.cells:
            context = self.build_context(target_time, population.cells[0].last_D_C, population.cells[0].last_D_P)
        return context

    def sample_next_event_or_horizon(
        self,
        cell: Cell,
        t_start: float,
        horizon: float,
    ) -> tuple[float, str | None, Cell, dict]:
        current_t = t_start
        temp_cell = cell.copy()
        proposals = 0

        while current_t < horizon - 1e-12:
            delta = float(self.event_rng.exponential(1.0 / self.r_bar))
            candidate_t = current_t + delta
            proposals += 1

            if candidate_t >= horizon:
                self.advance_cell_to_time(temp_cell, horizon, self.event_rng)
                return horizon, None, temp_cell, {"no_event": True, "proposals": proposals}

            context = self.advance_cell_to_time(temp_cell, candidate_t, self.event_rng)
            derived = dyn.compute_derived_quantities(temp_cell, context, self.params)
            rates = dyn.compute_all_event_rates(temp_cell, derived, context, self.params)
            total_rate = float(sum(rates.values()))
            if total_rate > self.r_bar * (1.0 + 1e-8):
                raise RuntimeError(
                    f"Dominating bound violated: total_rate={total_rate:.6f}, r_bar={self.r_bar:.6f}"
                )
            accept_prob = min(1.0, total_rate / self.r_bar)

            if total_rate > 0.0 and self.event_rng.random() < accept_prob:
                names = list(rates.keys())
                probabilities = np.array([rates[name] for name in names], dtype=float) / total_rate
                selected_event = str(self.event_rng.choice(names, p=probabilities))
                return (
                    candidate_t,
                    selected_event,
                    temp_cell,
                    {
                        "no_event": False,
                        "proposals": proposals,
                        "accept_prob": accept_prob,
                        "total_rate": total_rate,
                    },
                )

            current_t = candidate_t

        self.advance_cell_to_time(temp_cell, horizon, self.event_rng)
        return horizon, None, temp_cell, {"no_event": True, "proposals": proposals}

    def _sample_snapshot_cells(self, population: CellPopulation) -> list[Cell]:
        if not self.params.simulation.record_full_snapshots:
            return []
        if not population.cells:
            return []
        ordered = sorted(population.cells, key=lambda cell: cell.cell_id)
        max_cells = self.params.simulation.max_cells_saved_per_snapshot
        if len(ordered) <= max_cells:
            return ordered
        indices = np.rint(np.linspace(0, len(ordered) - 1, num=max_cells)).astype(int)
        unique_indices = sorted(set(int(index) for index in indices.tolist()))
        return [ordered[index] for index in unique_indices]

    def summarize_truth_population(self, population: CellPopulation, context: dyn.ReplicateContext) -> dict:
        if not population.cells:
            empty_matrices = {state_name: np.zeros((0, cfg.N_SPECIES), dtype=int) for state_name in cfg.STATE_NAMES}
            copy_summary = obs.summarize_copy_statistics(empty_matrices)
            return {
                "population_size": 0,
                "soft_state_fractions": np.zeros(cfg.N_STATES, dtype=float).tolist(),
                "cycle_fractions": np.zeros(cfg.N_CYCLE, dtype=float).tolist(),
                "bulk_copy_means": np.zeros(cfg.N_SPECIES, dtype=float).tolist(),
                "mean_stress_score": 0.0,
                "mean_survival_score": 0.0,
                "mean_division_hazard": 0.0,
                "mean_death_hazard": 0.0,
                "dominant_state_counts": np.zeros(cfg.N_STATES, dtype=int).tolist(),
                "dominant_state_fractions": np.zeros(cfg.N_STATES, dtype=float).tolist(),
                **copy_summary,
            }

        soft_state_totals = np.zeros(cfg.N_STATES, dtype=float)
        cycle_counts = np.zeros(cfg.N_CYCLE, dtype=float)
        copy_totals = np.zeros(cfg.N_SPECIES, dtype=float)
        dominant_counts = np.zeros(cfg.N_STATES, dtype=int)
        stress_total = 0.0
        survival_total = 0.0
        division_total = 0.0
        death_total = 0.0
        copy_matrices_by_gate: dict[str, list[np.ndarray]] = {state_name: [] for state_name in cfg.STATE_NAMES}

        for cell in population.cells:
            dominant_state = cell.dominant_state_index()
            soft_state_totals += cell.soft_state
            cycle_counts[cell.cycle_state] += 1.0
            copy_totals += cell.copy_numbers
            dominant_counts[dominant_state] += 1
            copy_matrices_by_gate[cfg.STATE_NAMES[dominant_state]].append(cell.copy_numbers.copy())
            stress_total += cell.stress_score
            survival_total += cell.survival_score
            derived = dyn.compute_derived_quantities(cell, context, self.params)
            division_total += dyn.compute_division_hazard(cell, derived, context, self.params)
            death_total += dyn.compute_death_hazard(cell, derived, context, self.params)

        stacked_matrices = {
            gate_name: (
                np.stack(values, axis=0).astype(int) if values else np.zeros((0, cfg.N_SPECIES), dtype=int)
            )
            for gate_name, values in copy_matrices_by_gate.items()
        }
        copy_summary = obs.summarize_copy_statistics(stacked_matrices)
        count = float(len(population.cells))
        return {
            "population_size": int(count),
            "soft_state_fractions": (soft_state_totals / count).astype(float).tolist(),
            "cycle_fractions": (cycle_counts / count).astype(float).tolist(),
            "bulk_copy_means": (copy_totals / count).astype(float).tolist(),
            "mean_stress_score": float(stress_total / count),
            "mean_survival_score": float(survival_total / count),
            "mean_division_hazard": float(division_total / count),
            "mean_death_hazard": float(death_total / count),
            "dominant_state_counts": dominant_counts.astype(int).tolist(),
            "dominant_state_fractions": (dominant_counts / count).astype(float).tolist(),
            **copy_summary,
        }

    def record_state(
        self,
        result: SimulationResult,
        population: CellPopulation,
        time: float,
        *,
        synchronize: bool,
        fallback_context: dyn.ReplicateContext | None = None,
    ) -> dict:
        context = self.synchronize_population_to_time(population, time) if synchronize else fallback_context
        if context is None:
            context = self.build_context(time, self.params.exposure.D_C0, self.params.exposure.D_P0)

        truth_snapshot = self.summarize_truth_population(population, context)
        observation_snapshot = obs.make_observation_snapshot(
            population.cells,
            truth_snapshot,
            self.observation_params,
            self.observation_rng,
        )

        sampled_cells = self._sample_snapshot_cells(population)
        snapshot_rows: list[dict] = []
        for cell in sampled_cells:
            derived = dyn.compute_derived_quantities(cell, context, self.params)
            snapshot_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "cycle_state": cfg.CYCLE_NAMES[cell.cycle_state],
                    "copy_numbers": cell.copy_numbers.tolist(),
                    "soft_state": cell.soft_state.tolist(),
                    "stress_score": float(cell.stress_score),
                    "survival_score": float(cell.survival_score),
                    "age": float(cell.age),
                    "dominant_state": cfg.STATE_NAMES[cell.dominant_state_index()],
                    "division_hazard": dyn.compute_division_hazard(cell, derived, context, self.params),
                    "death_hazard": dyn.compute_death_hazard(cell, derived, context, self.params),
                    "derived_report_only": {
                        "local_transition_generator": dyn.compute_local_transition_generator(
                            derived.logits,
                            self.params,
                        ).tolist()
                    },
                }
            )

        result.times.append(float(time))
        result.population_sizes.append(int(truth_snapshot["population_size"]))
        result.soft_state_fractions.append(np.asarray(truth_snapshot["soft_state_fractions"], dtype=float))
        result.cycle_fractions.append(np.asarray(truth_snapshot["cycle_fractions"], dtype=float))
        result.bulk_copy_means.append(np.asarray(truth_snapshot["bulk_copy_means"], dtype=float))
        result.mean_stress_scores.append(float(truth_snapshot["mean_stress_score"]))
        result.mean_survival_scores.append(float(truth_snapshot["mean_survival_score"]))
        result.mean_division_hazard.append(float(truth_snapshot["mean_division_hazard"]))
        result.mean_death_hazard.append(float(truth_snapshot["mean_death_hazard"]))
        result.exposures.append(
            {"D_C": context.D_C, "D_P": context.D_P, "a": context.astrocytic_cue, "m": context.mesenchymal_cue}
        )
        result.truth_snapshots.append(truth_snapshot)
        result.observations.append(observation_snapshot)
        result.cell_snapshots.append(snapshot_rows)
        return truth_snapshot

    def simulate(
        self,
        population: CellPopulation | None = None,
        verbose: bool = True,
    ) -> SimulationResult:
        if population is None:
            raise ValueError("Population must be constructed explicitly with params and initialization.")

        result = SimulationResult()
        time = 0.0
        event_counts: dict[str, int] = {}
        final_time = self.params.simulation.t_max
        record_times = tuple(float(value) for value in self.params.simulation.record_times)

        for cell in population.cells:
            cell.last_update_time = 0.0
            cell.last_D_C = self.params.exposure.D_C0
            cell.last_D_P = self.params.exposure.D_P0

        if verbose:
            print(
                "Simulation start: "
                f"time_unit={self.params.simulation.time_unit}, "
                f"t_max={final_time:.2f}, "
                f"record_times={record_times}, "
                f"target_population_size={self.params.simulation.target_population_size}, "
                f"max_pop_size={self.params.simulation.max_pop_size}, "
                f"n_init={population.size()}"
            )

        def population_stop_reason() -> str | None:
            size = population.size()
            if self.params.simulation.target_population_size is not None and size >= self.params.simulation.target_population_size:
                return "target_population_size"
            if size >= self.params.simulation.max_pop_size:
                return "max_pop_size"
            return None

        initial_stop_reason = population_stop_reason()
        if initial_stop_reason is not None:
            result.stop_reason = initial_stop_reason
            result.stop_time = 0.0
            summary = self.record_state(
                result,
                population,
                0.0,
                synchronize=True,
                fallback_context=self.build_context(0.0, self.params.exposure.D_C0, self.params.exposure.D_P0),
            )
            if verbose:
                self.print_monitor_line(0.0, summary, event_counts, label="stop")
            result.events = population.events.copy()
            return result

        stopped = False

        for boundary in record_times:
            event_heap: list[tuple[float, int, int, int, str, Cell, dict]] = []
            cell_versions: dict[int, int] = {}
            cell_lookup: dict[int, Cell] = {cell.cell_id: cell for cell in population.cells}

            def schedule_cell_event(cell: Cell, from_time: float) -> None:
                event_time, event_name, flow_cell, thinning_stats = self.sample_next_event_or_horizon(cell, from_time, boundary)
                version = cell_versions.get(cell.cell_id, 0) + 1
                cell_versions[cell.cell_id] = version
                heapq.heappush(
                    event_heap,
                    (
                        float(event_time),
                        int(cell.cell_id),
                        int(version),
                        1 if event_name is None else 0,
                        "" if event_name is None else event_name,
                        flow_cell,
                        thinning_stats,
                    ),
                )

            for live_cell in population.cells:
                schedule_cell_event(live_cell, time)

            while event_heap:
                event_time, cell_id, version, no_event_flag, event_name, flow_cell, _stats = heapq.heappop(event_heap)
                if cell_versions.get(cell_id) != version:
                    continue
                live_cell = cell_lookup.get(cell_id)
                if live_cell is None:
                    continue

                if no_event_flag == 1:
                    live_cell.overwrite_state_from(flow_cell, validate=False)
                    continue

                live_cell.overwrite_state_from(flow_cell, validate=False)
                time = float(event_time)
                event_context = self.build_context(time, live_cell.last_D_C, live_cell.last_D_P)
                state_pre = live_cell.get_state_dict()

                if event_name == "death":
                    population.remove_cell(live_cell)
                    event_counts["death"] = event_counts.get("death", 0) + 1
                    if self.params.simulation.record_events:
                        population.log_event(time, "death", live_cell.cell_id, {"state_pre": state_pre})
                    cell_lookup.pop(live_cell.cell_id, None)
                    cell_versions.pop(live_cell.cell_id, None)
                    if population.size() == 0:
                        result.stop_reason = "population_extinction"
                        result.stop_time = float(time)
                        summary = self.record_state(
                            result,
                            population,
                            time,
                            synchronize=False,
                            fallback_context=event_context,
                        )
                        if verbose:
                            self.print_monitor_line(time, summary, event_counts, label="extinction")
                        stopped = True
                        break
                    continue

                if event_name == "division":
                    daughter_one, daughter_two = self.division_kernel.divide(live_cell, event_context)
                    daughter_one.last_update_time = time
                    daughter_one.last_D_C = live_cell.last_D_C
                    daughter_one.last_D_P = live_cell.last_D_P
                    daughter_two.last_update_time = time
                    daughter_two.last_D_C = live_cell.last_D_C
                    daughter_two.last_D_P = live_cell.last_D_P

                    population.remove_cell(live_cell)
                    daughter_one = population.add_cell(daughter_one)
                    daughter_two = population.add_cell(daughter_two)
                    event_counts["division"] = event_counts.get("division", 0) + 1
                    if self.params.simulation.record_events:
                        population.log_event(
                            time,
                            "division",
                            live_cell.cell_id,
                            {
                                "state_pre": state_pre,
                                "daughter_one": daughter_one.get_state_dict(),
                                "daughter_two": daughter_two.get_state_dict(),
                            },
                        )
                    cell_lookup.pop(live_cell.cell_id, None)
                    cell_versions.pop(live_cell.cell_id, None)
                    cell_lookup[daughter_one.cell_id] = daughter_one
                    cell_lookup[daughter_two.cell_id] = daughter_two
                    schedule_cell_event(daughter_one, time)
                    schedule_cell_event(daughter_two, time)

                    stop_reason = population_stop_reason()
                    if stop_reason is not None:
                        result.stop_reason = stop_reason
                        result.stop_time = float(time)
                        summary = self.record_state(
                            result,
                            population,
                            time,
                            synchronize=True,
                            fallback_context=event_context,
                        )
                        if verbose:
                            self.print_monitor_line(time, summary, event_counts, label="stop")
                        stopped = True
                        break
                    continue

                dyn.apply_nonterminal_event(live_cell, event_name)
                live_cell.last_update_time = time
                live_cell.last_D_C = flow_cell.last_D_C
                live_cell.last_D_P = flow_cell.last_D_P
                event_counts[event_name] = event_counts.get(event_name, 0) + 1
                if self.params.simulation.record_events:
                    population.log_event(
                        time,
                        event_name,
                        live_cell.cell_id,
                        {
                            "state_pre": state_pre,
                            "state_post": live_cell.get_state_dict(),
                        },
                    )
                schedule_cell_event(live_cell, time)

            if stopped:
                break

            time = float(boundary)
            summary = self.record_state(
                result,
                population,
                boundary,
                synchronize=False,
                fallback_context=self.build_context(boundary, self.params.exposure.D_C0, self.params.exposure.D_P0)
                if not population.cells
                else self.build_context(boundary, population.cells[0].last_D_C, population.cells[0].last_D_P),
            )
            if verbose:
                self.print_monitor_line(boundary, summary, event_counts, label="snapshot")

        if not result.stop_reason:
            result.stop_reason = "t_max"
            result.stop_time = float(final_time)

        result.events = population.events.copy()
        return result


def _build_record_times(t_max: float, record_interval: float) -> tuple[float, ...]:
    cfg.require(record_interval > 0.0, "record_interval must be strictly positive.")
    values = np.arange(record_interval, t_max + 1e-12, record_interval, dtype=float)
    if values.size == 0 or abs(float(values[-1]) - t_max) > 1e-8:
        values = np.append(values, float(t_max))
    return tuple(float(value) for value in values.tolist())


def _clone_model_parameters_with_overrides(
    params: cfg.ModelParameters,
    *,
    t_max: float | None,
    n_init: int | None,
    record_times: Iterable[float] | None,
    record_interval: float | None,
    target_population_size: int | None,
    max_pop_size: int | None,
    seed: int | None,
) -> cfg.ModelParameters:
    cloned = copy.deepcopy(params)
    simulation = cloned.simulation

    resolved_t_max = simulation.t_max if t_max is None else float(t_max)
    if record_times is not None:
        resolved_record_times = tuple(float(value) for value in record_times)
    elif record_interval is not None:
        resolved_record_times = _build_record_times(resolved_t_max, float(record_interval))
    else:
        resolved_record_times = simulation.record_times
        cfg.require(
            abs(float(resolved_record_times[-1]) - resolved_t_max) <= 1e-8,
            "t_max override requires record_times or record_interval to keep the time grid explicit.",
        )

    resolved_simulation = replace(
        simulation,
        t_max=resolved_t_max,
        n_init=simulation.n_init if n_init is None else int(n_init),
        record_times=resolved_record_times,
        target_population_size=simulation.target_population_size if target_population_size is None else target_population_size,
        max_pop_size=simulation.max_pop_size if max_pop_size is None else int(max_pop_size),
        random_seed=simulation.random_seed if seed is None else int(seed),
    )
    return replace(cloned, simulation=resolved_simulation)


def run_simulation(
    *,
    params: cfg.ModelParameters | None = None,
    observation_params: cfg.ObservationParameters | None = None,
    initialization: cfg.InitializationParameters | None = None,
    t_max: float | None = None,
    n_init: int | None = None,
    input_schedules: dict[str, callable] | None = None,
    seed: int | None = None,
    record_times: Iterable[float] | None = None,
    record_interval: float | None = None,
    target_population_size: int | None = None,
    max_pop_size: int | None = None,
    verbose: bool = True,
) -> SimulationResult:
    base_params = cfg.DEFAULT_MODEL_PARAMETERS if params is None else params
    resolved_params = _clone_model_parameters_with_overrides(
        base_params,
        t_max=t_max,
        n_init=n_init,
        record_times=record_times,
        record_interval=record_interval,
        target_population_size=target_population_size,
        max_pop_size=max_pop_size,
        seed=seed,
    )
    resolved_observation = (
        copy.deepcopy(cfg.DEFAULT_OBSERVATION_PARAMETERS)
        if observation_params is None
        else copy.deepcopy(observation_params)
    )
    resolved_initialization = (
        copy.deepcopy(cfg.DEFAULT_INITIALIZATION_PARAMETERS)
        if initialization is None
        else copy.deepcopy(initialization)
    )

    cfg.validate_model_parameters(resolved_params)
    cfg.validate_observation_parameters(resolved_observation)
    cfg.validate_initialization_parameters(resolved_initialization)

    seed_value = resolved_params.simulation.random_seed if seed is None else int(seed)
    seed_sequence = np.random.SeedSequence(seed_value)
    init_seed, event_seed, observation_seed = seed_sequence.spawn(3)
    population = CellPopulation(resolved_params, resolved_initialization, np.random.default_rng(init_seed))
    population.initialize(resolved_params.simulation.n_init if n_init is None else n_init)
    simulator = HybridOgataSimulator(
        params=resolved_params,
        observation_params=resolved_observation,
        input_schedules=input_schedules,
        seed=seed_value,
        event_rng=np.random.default_rng(event_seed),
        observation_rng=np.random.default_rng(observation_seed),
    )
    return simulator.simulate(population=population, verbose=verbose)
