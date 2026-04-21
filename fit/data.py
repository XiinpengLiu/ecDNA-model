"""
Canonical fitting dataset objects and loaders for week1-10 calibration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

import config as cfg


WEEK1 = 1
WEEK10 = 10
DEFAULT_QPCDR_BATCH = "default"
DEFAULT_QPCDR_SCALE = "copy_number"
SUPPORTED_QPCDR_SCALES = {DEFAULT_QPCDR_SCALE, "ct"}


def _resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        cfg.require(base_dir is not None, f"Relative path {path} requires a base directory.")
        resolved = base_dir / resolved
    return resolved.resolve()


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(text)


def _piecewise_constant(points: Sequence[tuple[float, float]]) -> Callable[[float], float]:
    ordered = tuple(sorted((float(time), float(value)) for time, value in points))
    for time, _value in ordered:
        cfg.require(time >= 0.0, "Schedule times must be non-negative.")

    def schedule(query_time: float) -> float:
        current_value = 0.0
        for start_time, value in ordered:
            if query_time + 1e-12 < start_time:
                break
            current_value = value
        return float(current_value)

    return schedule


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    schedules: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    initialization_source: str | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.name), "Condition name must be non-empty.")
        object.__setattr__(self, "schedules", self._normalize_schedules(self.schedules))
        if self.initialization_source is not None:
            cfg.require(bool(self.initialization_source), "initialization_source must be non-empty when provided.")

    @staticmethod
    def _normalize_schedules(
        schedules: Mapping[str, Iterable[tuple[float, float]]] | None,
    ) -> dict[str, tuple[tuple[float, float], ...]]:
        result: dict[str, tuple[tuple[float, float], ...]] = {}
        if schedules is None:
            return result
        for schedule_name, points in schedules.items():
            cfg.require(
                schedule_name in cfg.DEFAULT_INPUT_SCHEDULES,
                f"Unknown schedule key {schedule_name}; expected one of {tuple(cfg.DEFAULT_INPUT_SCHEDULES.keys())}.",
            )
            normalized = tuple((float(time), float(value)) for time, value in points)
            if normalized:
                starts = np.array([time for time, _ in normalized], dtype=float)
                cfg.require(np.all(np.diff(starts) >= 0.0), f"Schedule {schedule_name} must be sorted by time.")
            result[schedule_name] = normalized
        return result

    def build_input_schedules(self) -> dict[str, Callable[[float], float]]:
        schedule_functions = dict(cfg.DEFAULT_INPUT_SCHEDULES)
        for schedule_name, points in self.schedules.items():
            schedule_functions[schedule_name] = _piecewise_constant(points)
        return schedule_functions

    def is_baseline(self) -> bool:
        return not self.has_drug_input() and not self.has_cue_input()

    def has_drug_input(self) -> bool:
        for schedule_name in ("u_C", "u_P"):
            if any(abs(value) > 1e-12 for _, value in self.schedules.get(schedule_name, ())):
                return True
        return False

    def has_cue_input(self) -> bool:
        for schedule_name in ("a", "m"):
            if any(abs(value) > 1e-12 for _, value in self.schedules.get(schedule_name, ())):
                return True
        return False


@dataclass(frozen=True)
class FlowRecord:
    condition: str
    week: int
    state: str
    count: int | None = None
    fraction: float | None = None
    total_events: int | None = None
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.condition), "FlowRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "FlowRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid flow state {self.state}.")
        cfg.require(self.count is not None or self.fraction is not None, "FlowRecord requires count or fraction.")
        if self.count is not None:
            cfg.require(self.count >= 0, "FlowRecord.count must be non-negative.")
        if self.fraction is not None:
            cfg.require(0.0 <= self.fraction <= 1.0, "FlowRecord.fraction must lie in [0, 1].")
        if self.total_events is not None:
            cfg.require(self.total_events >= 0, "FlowRecord.total_events must be non-negative.")


@dataclass(frozen=True)
class CountRecord:
    condition: str
    week: int
    value: float
    replicate_id: str | None = None
    gate: str | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.condition), "CountRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "CountRecord.week must be at least 1.")
        cfg.require(np.isfinite(self.value) and self.value >= 0.0, "CountRecord.value must be finite and non-negative.")
        if self.gate is not None:
            cfg.require(self.gate in cfg.STATE_NAMES, f"Invalid count gate {self.gate}.")


@dataclass(frozen=True)
class QPCDRRecord:
    condition: str
    week: int
    state: str
    species: str
    value: float
    replicate_id: str | None = None
    batch: str = DEFAULT_QPCDR_BATCH
    value_scale: str = DEFAULT_QPCDR_SCALE

    def __post_init__(self) -> None:
        cfg.require(bool(self.condition), "QPCDRRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "QPCDRRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid qPCDR state {self.state}.")
        cfg.require(self.species in cfg.SPECIES, f"Invalid qPCDR species {self.species}.")
        cfg.require(np.isfinite(self.value), "QPCDRRecord.value must be finite.")
        cfg.require(bool(self.batch), "QPCDRRecord.batch must be non-empty.")
        cfg.require(
            self.value_scale in SUPPORTED_QPCDR_SCALES,
            f"Unsupported qPCDR value scale {self.value_scale}; expected one of {SUPPORTED_QPCDR_SCALES}.",
        )


@dataclass(frozen=True)
class EcTAGRecord:
    condition: str
    week: int
    state: str
    species: str
    cell_id: str
    value: int
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.condition), "EcTAGRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "EcTAGRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid ecTAG state {self.state}.")
        cfg.require(self.species in cfg.SPECIES, f"Invalid ecTAG species {self.species}.")
        cfg.require(bool(self.cell_id), "EcTAGRecord.cell_id must be non-empty.")
        cfg.require(self.value >= 0, "EcTAGRecord.value must be non-negative.")


@dataclass(frozen=True)
class CanonicalFitDataset:
    conditions: dict[str, ConditionSpec]
    flow: tuple[FlowRecord, ...]
    counts: tuple[CountRecord, ...]
    qpcdr: tuple[QPCDRRecord, ...]
    ectag: tuple[EcTAGRecord, ...]
    week1_copy_distributions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    ectag_hist_max: int | None = None
    purity_matrix: np.ndarray | None = None
    purity_sensitivity: tuple[np.ndarray, ...] = ()
    qpcdr_calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    batch_column_policy: str = "as-provided"

    def __post_init__(self) -> None:
        normalized_conditions = {name: spec for name, spec in self.conditions.items()}
        cfg.require(bool(normalized_conditions), "CanonicalFitDataset requires at least one condition.")
        object.__setattr__(self, "conditions", normalized_conditions)
        object.__setattr__(self, "flow", tuple(self.flow))
        object.__setattr__(self, "counts", tuple(self.counts))
        object.__setattr__(self, "qpcdr", tuple(self.qpcdr))
        object.__setattr__(self, "ectag", tuple(self.ectag))
        object.__setattr__(self, "week1_copy_distributions", self._normalize_week1_copy_distributions(self.week1_copy_distributions))
        if self.purity_matrix is not None:
            object.__setattr__(self, "purity_matrix", self._normalize_purity_matrix(self.purity_matrix, "purity_matrix"))
        object.__setattr__(
            self,
            "purity_sensitivity",
            tuple(self._normalize_purity_matrix(matrix, f"purity_sensitivity[{index}]") for index, matrix in enumerate(self.purity_sensitivity)),
        )
        object.__setattr__(self, "qpcdr_calibration", dict(self.qpcdr_calibration))
        cfg.require(bool(self.batch_column_policy), "batch_column_policy must be non-empty.")
        self.validate()

    @staticmethod
    def _normalize_week1_copy_distributions(
        values: Mapping[str, Mapping[str, np.ndarray]] | None,
    ) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        if values is None:
            return result
        for condition, by_state in values.items():
            state_payload: dict[str, np.ndarray] = {}
            for state_name, matrix in by_state.items():
                array = np.asarray(matrix, dtype=int)
                cfg.require(
                    array.ndim == 2 and array.shape[1] == cfg.N_SPECIES,
                    f"week1 copy matrix for {condition}/{state_name} must have shape (n, {cfg.N_SPECIES}).",
                )
                cfg.require(array.shape[0] > 0, f"week1 copy matrix for {condition}/{state_name} must be non-empty.")
                cfg.require(np.all(array >= 0), f"week1 copy matrix for {condition}/{state_name} must be non-negative.")
                state_payload[state_name] = array.copy()
            result[condition] = state_payload
        return result

    @staticmethod
    def _normalize_purity_matrix(values: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        cfg.require(matrix.shape == (cfg.N_STATES, cfg.N_STATES), f"{name} must have shape ({cfg.N_STATES}, {cfg.N_STATES}).")
        cfg.require(np.all(np.isfinite(matrix)), f"{name} must be finite.")
        cfg.require(np.all(matrix >= 0.0), f"{name} must be non-negative.")
        column_sums = np.sum(matrix, axis=0)
        cfg.require(np.all(column_sums > 0.0), f"Every {name} column must have positive mass.")
        return matrix / column_sums

    def validate(self) -> None:
        referenced_conditions = set(self.conditions)
        for collection_name, records in (
            ("flow", self.flow),
            ("counts", self.counts),
            ("qpcdr", self.qpcdr),
            ("ectag", self.ectag),
        ):
            for record in records:
                condition = getattr(record, "condition")
                cfg.require(condition in referenced_conditions, f"{collection_name} record references unknown condition {condition}.")

        qpcdr_scales = {record.value_scale for record in self.qpcdr}
        cfg.require(len(qpcdr_scales) <= 1, "qPCDR records must use a single value scale across the dataset.")

        dynamic_weeks = self.dynamic_weeks()
        cfg.require(bool(dynamic_weeks), "CanonicalFitDataset requires at least one dynamic week in week2-10.")
        for condition_name in self.conditions:
            init_condition = self.resolve_initialization_condition(condition_name)
            self._validate_week1_flow(init_condition)
            self._validate_week1_copy_source(init_condition)

    def resolve_initialization_condition(self, condition_name: str) -> str:
        cfg.require(condition_name in self.conditions, f"Unknown condition {condition_name}.")
        init_source = self.conditions[condition_name].initialization_source
        resolved = condition_name if init_source is None else init_source
        cfg.require(resolved in self.conditions, f"Initialization condition {resolved} is not defined.")
        return resolved

    def _validate_week1_flow(self, condition_name: str) -> None:
        records = [record for record in self.flow if record.condition == condition_name and record.week == WEEK1]
        cfg.require(records, f"Missing week1 flow records for condition {condition_name}.")
        by_state = {record.state for record in records}
        cfg.require(set(cfg.STATE_NAMES).issubset(by_state), f"week1 flow records for {condition_name} must cover every state.")

    def _validate_week1_copy_source(self, condition_name: str) -> None:
        if condition_name in self.week1_copy_distributions:
            by_state = self.week1_copy_distributions[condition_name]
            cfg.require(set(by_state) == set(cfg.STATE_NAMES), f"week1 copy matrices for {condition_name} must cover every state.")
            return
        by_state = self._week1_ectag_grouped(condition_name)
        cfg.require(set(by_state) == set(cfg.STATE_NAMES), f"week1 ecTAG records for {condition_name} must cover every state.")
        for state_name, cell_map in by_state.items():
            cfg.require(cell_map, f"week1 ecTAG records for {condition_name}/{state_name} must be non-empty.")
            for cell_key, species_map in cell_map.items():
                missing = set(cfg.SPECIES) - set(species_map)
                cfg.require(
                    not missing,
                    f"week1 ecTAG cell {cell_key} in {condition_name}/{state_name} is missing species {sorted(missing)}; aligned cell-level week1 data is required.",
                )

    def qpcdr_scale(self) -> str:
        if not self.qpcdr:
            return DEFAULT_QPCDR_SCALE
        return self.qpcdr[0].value_scale

    def qpcdr_batch(self) -> str:
        if not self.qpcdr:
            return DEFAULT_QPCDR_BATCH
        return self.qpcdr[0].batch

    def qpcdr_batches(self) -> tuple[str, ...]:
        if not self.qpcdr:
            return (DEFAULT_QPCDR_BATCH,)
        return tuple(sorted({record.batch for record in self.qpcdr}))

    def condition_names(self) -> tuple[str, ...]:
        return tuple(self.conditions.keys())

    def dynamic_weeks(self) -> tuple[int, ...]:
        weeks = {
            record.week
            for collection in (self.flow, self.counts, self.qpcdr, self.ectag)
            for record in collection
            if record.week > WEEK1
        }
        return tuple(sorted(int(week) for week in weeks))

    def ectag_upper_bound(self) -> int:
        if self.ectag_hist_max is not None:
            return int(self.ectag_hist_max)
        if not self.ectag:
            return 0
        return int(max(record.value for record in self.ectag))

    def _week1_ectag_grouped(self, condition_name: str) -> dict[str, dict[str, dict[str, int]]]:
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for record in self.ectag:
            if record.condition != condition_name or record.week != WEEK1:
                continue
            cell_key = self._ectag_cell_key(record)
            grouped.setdefault(record.state, {}).setdefault(cell_key, {})[record.species] = int(record.value)
        return grouped

    @staticmethod
    def _ectag_cell_key(record: EcTAGRecord) -> str:
        replicate_token = "" if record.replicate_id is None else f"{record.replicate_id}|"
        return f"{replicate_token}{record.cell_id}"

    def build_empirical_initialization(
        self,
        condition_name: str,
        *,
        template: cfg.InitializationParameters | None = None,
    ) -> cfg.InitializationParameters:
        init_condition = self.resolve_initialization_condition(condition_name)
        base = cfg.DEFAULT_INITIALIZATION_PARAMETERS if template is None else template
        flow_fractions = self._build_week1_flow_fractions(init_condition)
        copy_distributions = self._build_week1_copy_distributions(init_condition)
        initialization = cfg.InitializationParameters(
            mode=cfg.EMPIRICAL_WEEK1,
            parametric_copy_number_mean=np.asarray(base.parametric_copy_number_mean, dtype=float).copy(),
            parametric_state_dirichlet_alpha=np.asarray(base.parametric_state_dirichlet_alpha, dtype=float).copy(),
            cycle_probabilities=np.asarray(base.cycle_probabilities, dtype=float).copy(),
            age_scale=float(base.age_scale),
            empirical_flow_fractions=flow_fractions,
            empirical_sorted_copy_distributions=copy_distributions,
            empirical_soft_state_concentration=float(base.empirical_soft_state_concentration),
        )
        cfg.validate_initialization_parameters(initialization)
        return initialization

    def _build_week1_flow_fractions(self, condition_name: str) -> np.ndarray:
        rows = [record for record in self.flow if record.condition == condition_name and record.week == WEEK1]
        state_totals = np.zeros(cfg.N_STATES, dtype=float)
        if any(record.count is not None for record in rows):
            for record in rows:
                if record.count is not None:
                    state_totals[cfg.STATE_INDEX[record.state]] += float(record.count)
                elif record.fraction is not None and record.total_events is not None:
                    state_totals[cfg.STATE_INDEX[record.state]] += float(record.fraction * record.total_events)
        else:
            replicate_maps: dict[str, np.ndarray] = {}
            for record in rows:
                replicate_key = record.replicate_id or "__aggregate__"
                replicate_maps.setdefault(replicate_key, np.zeros(cfg.N_STATES, dtype=float))
                cfg.require(record.fraction is not None, f"week1 flow record for {condition_name}/{record.state} is missing fraction.")
                replicate_maps[replicate_key][cfg.STATE_INDEX[record.state]] = float(record.fraction)
            cfg.require(bool(replicate_maps), f"No week1 flow replicates found for {condition_name}.")
            state_totals = np.mean(np.stack(tuple(replicate_maps.values()), axis=0), axis=0)

        total = float(np.sum(state_totals))
        cfg.require(total > 0.0, f"week1 flow totals for {condition_name} must be positive.")
        flow_fractions = state_totals / total
        cfg.validate_probability_vector(flow_fractions, name=f"{condition_name}.week1_flow_fractions", expected_shape=(cfg.N_STATES,))
        return flow_fractions.astype(float)

    def _build_week1_copy_distributions(self, condition_name: str) -> dict[str, np.ndarray]:
        if condition_name in self.week1_copy_distributions:
            return {
                state_name: np.asarray(matrix, dtype=int).copy()
                for state_name, matrix in self.week1_copy_distributions[condition_name].items()
            }

        grouped = self._week1_ectag_grouped(condition_name)
        copy_distributions: dict[str, np.ndarray] = {}
        for state_name in cfg.STATE_NAMES:
            cell_map = grouped.get(state_name, {})
            rows: list[list[int]] = []
            for _cell_key, species_map in sorted(cell_map.items()):
                rows.append([int(species_map[species]) for species in cfg.SPECIES])
            matrix = np.asarray(rows, dtype=int)
            cfg.require(
                matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] == cfg.N_SPECIES,
                f"week1 ecTAG-derived copy distribution for {condition_name}/{state_name} must have shape (n, {cfg.N_SPECIES}).",
            )
            copy_distributions[state_name] = matrix
        return copy_distributions

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "CanonicalFitDataset":
        manifest_file = _resolve_path(manifest_path)
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        base_dir = manifest_file.parent

        conditions = {
            name: ConditionSpec(
                name=name,
                schedules=spec_payload.get("schedules", {}),
                initialization_source=spec_payload.get("initialization_source"),
            )
            for name, spec_payload in payload["conditions"].items()
        }

        flow = load_flow_csv(_resolve_path(payload["files"]["flow"], base_dir=base_dir))
        counts = ()
        if payload["files"].get("counts"):
            counts = load_count_csv(_resolve_path(payload["files"]["counts"], base_dir=base_dir))
        qpcdr = ()
        if payload["files"].get("qpcdr"):
            qpcdr = load_qpcdr_csv(_resolve_path(payload["files"]["qpcdr"], base_dir=base_dir))
        ectag = ()
        if payload["files"].get("ectag"):
            ectag = load_ectag_csv(_resolve_path(payload["files"]["ectag"], base_dir=base_dir))

        week1_copy_distributions: dict[str, dict[str, np.ndarray]] = {}
        if payload["files"].get("week1_copy_distributions"):
            matrix_payload = json.loads(
                _resolve_path(payload["files"]["week1_copy_distributions"], base_dir=base_dir).read_text(encoding="utf-8")
            )
            week1_copy_distributions = {
                condition: {state_name: np.asarray(matrix, dtype=int) for state_name, matrix in by_state.items()}
                for condition, by_state in matrix_payload.items()
            }

        purity_matrix = payload.get("purity_matrix")
        purity_sensitivity = tuple(np.asarray(matrix, dtype=float) for matrix in payload.get("purity_sensitivity", ()))

        return cls(
            conditions=conditions,
            flow=flow,
            counts=counts,
            qpcdr=qpcdr,
            ectag=ectag,
            week1_copy_distributions=week1_copy_distributions,
            ectag_hist_max=payload.get("ectag_hist_max"),
            purity_matrix=None if purity_matrix is None else np.asarray(purity_matrix, dtype=float),
            purity_sensitivity=purity_sensitivity,
            qpcdr_calibration=payload.get("qpcdr_calibration", {}),
            batch_column_policy=payload.get("batch_column_policy", "as-provided"),
        )

    @classmethod
    def from_simulation_runs(
        cls,
        runs_by_condition: Mapping[str, Sequence["SimulationResult"] | "SimulationResult"],
        *,
        conditions: Mapping[str, ConditionSpec] | None = None,
        qpcdr_value_scale: str = DEFAULT_QPCDR_SCALE,
        ectag_hist_max: int | None = None,
    ) -> "CanonicalFitDataset":
        from core.simulation import SimulationResult

        normalized_runs: dict[str, tuple[SimulationResult, ...]] = {}
        for condition_name, payload in runs_by_condition.items():
            if isinstance(payload, SimulationResult):
                normalized_runs[condition_name] = (payload,)
            else:
                normalized_runs[condition_name] = tuple(payload)
            cfg.require(normalized_runs[condition_name], f"Simulation payload for {condition_name} must be non-empty.")

        condition_specs = (
            {name: ConditionSpec(name=name) for name in normalized_runs}
            if conditions is None
            else {name: spec for name, spec in conditions.items()}
        )

        flow: list[FlowRecord] = []
        counts: list[CountRecord] = []
        qpcdr: list[QPCDRRecord] = []
        ectag: list[EcTAGRecord] = []
        week1_copy_rows: dict[str, dict[str, list[list[int]]]] = {}

        for condition_name, results in normalized_runs.items():
            cfg.require(condition_name in condition_specs, f"Missing ConditionSpec for {condition_name}.")
            for replicate_index, result in enumerate(results):
                replicate_id = f"sim{replicate_index}"
                for time_value, truth_snapshot, observation_snapshot in zip(
                    result.times,
                    result.truth_snapshots,
                    result.observations,
                ):
                    week = int(round(float(time_value))) + 1
                    for state_index, state_name in enumerate(cfg.STATE_NAMES):
                        flow.append(
                            FlowRecord(
                                condition=condition_name,
                                week=week,
                                state=state_name,
                                count=int(observation_snapshot["flow_counts"][state_index]),
                                fraction=float(observation_snapshot["flow_fractions"][state_index]),
                                total_events=int(sum(observation_snapshot["flow_counts"])),
                                replicate_id=replicate_id,
                            )
                        )
                    counts.append(
                        CountRecord(
                            condition=condition_name,
                            week=week,
                            value=float(observation_snapshot["observed_count"]),
                            replicate_id=replicate_id,
                        )
                    )
                    sorted_state_counts = observation_snapshot.get("sorted_state_counts", {})
                    for state_name in cfg.STATE_NAMES:
                        if state_name in sorted_state_counts:
                            counts.append(
                                CountRecord(
                                    condition=condition_name,
                                    week=week,
                                    value=float(sorted_state_counts[state_name]),
                                    replicate_id=replicate_id,
                                    gate=state_name,
                                )
                            )

                    for state_name in cfg.STATE_NAMES:
                        qpcdr_payload = observation_snapshot["sorted_qpcdr"]["values"][state_name]
                        ectag_payload = observation_snapshot["sorted_ecTAG"]["values"][state_name]
                        per_species_lengths = [len(ectag_payload[species_name]) for species_name in cfg.SPECIES]
                        aligned_ectag = len(set(per_species_lengths)) == 1
                        for species_name in cfg.SPECIES:
                            for value_index, value in enumerate(qpcdr_payload[species_name]):
                                qpcdr.append(
                                    QPCDRRecord(
                                        condition=condition_name,
                                        week=week,
                                        state=state_name,
                                        species=species_name,
                                        value=float(value),
                                        replicate_id=replicate_id,
                                        batch=DEFAULT_QPCDR_BATCH,
                                        value_scale=qpcdr_value_scale,
                                    )
                                )
                            for value_index, value in enumerate(ectag_payload[species_name]):
                                cell_id = (
                                    f"{replicate_id}|{condition_name}|week{week}|{state_name}|cell{value_index}"
                                    if aligned_ectag
                                    else f"{replicate_id}|{condition_name}|week{week}|{state_name}|{species_name}|obs{value_index}"
                                )
                                ectag.append(
                                    EcTAGRecord(
                                        condition=condition_name,
                                        week=week,
                                        state=state_name,
                                        species=species_name,
                                        cell_id=cell_id,
                                        value=int(value),
                                        replicate_id=replicate_id,
                                    )
                                )

                        if week == WEEK1:
                            copy_payload = observation_snapshot["sorted_copy_distributions"][state_name]
                            week1_copy_rows.setdefault(condition_name, {}).setdefault(state_name, []).extend(copy_payload)

        week1_copy_distributions = {
            condition_name: {state_name: np.asarray(rows, dtype=int) for state_name, rows in by_state.items()}
            for condition_name, by_state in week1_copy_rows.items()
        }
        inferred_hist_max = ectag_hist_max
        if inferred_hist_max is None and ectag:
            inferred_hist_max = max(record.value for record in ectag)

        return cls(
            conditions=condition_specs,
            flow=tuple(flow),
            counts=tuple(counts),
            qpcdr=tuple(qpcdr),
            ectag=tuple(ectag),
            week1_copy_distributions=week1_copy_distributions,
            ectag_hist_max=inferred_hist_max,
            purity_matrix=None,
            purity_sensitivity=(),
            qpcdr_calibration={},
            batch_column_policy="as-provided",
        )


def load_flow_csv(path: str | Path) -> tuple[FlowRecord, ...]:
    resolved = _resolve_path(path)
    rows: list[FlowRecord] = []
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                FlowRecord(
                    condition=raw["condition"],
                    week=int(raw["week"]),
                    state=raw["state"],
                    count=_int_or_none(raw.get("count")),
                    fraction=_float_or_none(raw.get("fraction")),
                    total_events=_int_or_none(raw.get("total_events")),
                    replicate_id=raw.get("replicate_id") or None,
                )
            )
    return tuple(rows)


def load_count_csv(path: str | Path) -> tuple[CountRecord, ...]:
    resolved = _resolve_path(path)
    rows: list[CountRecord] = []
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                CountRecord(
                    condition=raw["condition"],
                    week=int(raw["week"]),
                    value=float(raw["count"]),
                    replicate_id=raw.get("replicate_id") or None,
                    gate=raw.get("gate") or None,
                )
            )
    return tuple(rows)


def load_qpcdr_csv(path: str | Path) -> tuple[QPCDRRecord, ...]:
    resolved = _resolve_path(path)
    rows: list[QPCDRRecord] = []
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                QPCDRRecord(
                    condition=raw["condition"],
                    week=int(raw["week"]),
                    state=raw["state"],
                    species=raw["species"],
                    value=float(raw["value"]),
                    replicate_id=raw.get("replicate_id") or None,
                    batch=(raw.get("batch") or DEFAULT_QPCDR_BATCH),
                    value_scale=(raw.get("value_scale") or DEFAULT_QPCDR_SCALE),
                )
            )
    return tuple(rows)


def load_ectag_csv(path: str | Path) -> tuple[EcTAGRecord, ...]:
    resolved = _resolve_path(path)
    rows: list[EcTAGRecord] = []
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                EcTAGRecord(
                    condition=raw["condition"],
                    week=int(raw["week"]),
                    state=raw["state"],
                    species=raw["species"],
                    cell_id=raw["cell_id"],
                    value=int(raw["value"]),
                    replicate_id=raw.get("replicate_id") or None,
                )
            )
    return tuple(rows)
