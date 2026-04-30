"""Canonical fitting inputs and CSV/manifest loaders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

import config as cfg
from fit.io_utils import write_json, write_table_bundle


WEEK1 = 1
DEFAULT_QPCDR_BATCH = "default"
DEFAULT_QPCDR_SCALE = "copy_number"
SUPPORTED_QPCDR_SCALES = {DEFAULT_QPCDR_SCALE, "ct"}
ECTAG_HIST_MAX_METADATA_FLAG = "ectag_hist_max_from_metadata"


def _resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        cfg.require(base_dir is not None, f"Relative path {path} requires a base directory.")
        resolved = base_dir / resolved
    return resolved.resolve()


def _float_or_none(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(float(value))


def _manifest_ectag_hist_max(payload: Mapping[str, object]) -> int | None:
    value = payload.get("ectag_hist_max")
    if value is None:
        return None
    cfg.require(
        payload.get(ECTAG_HIST_MAX_METADATA_FLAG) is True,
        f"manifest ectag_hist_max is only allowed when experimental metadata states an ecTAG observation limit; set {ECTAG_HIST_MAX_METADATA_FLAG}: true.",
    )
    maximum = int(float(value))
    cfg.require(maximum >= 0, "manifest ectag_hist_max must be non-negative.")
    return maximum


def _piecewise_constant(points: Sequence[tuple[float, float]]) -> Callable[[float], float]:
    ordered = tuple(sorted((float(time), float(value)) for time, value in points))

    def schedule(query_time: float) -> float:
        current = 0.0
        for start_time, value in ordered:
            if float(query_time) + 1e-12 < start_time:
                break
            current = value
        return float(current)

    return schedule


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    schedules: Mapping[str, Iterable[tuple[float, float]]] | None = None
    initialization_source: str | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.name), "ConditionSpec.name must be non-empty.")
        normalized: dict[str, tuple[tuple[float, float], ...]] = {}
        for key, points in (self.schedules or {}).items():
            cfg.require(key in cfg.DEFAULT_INPUT_SCHEDULES, f"Unknown input schedule {key}.")
            normalized[key] = tuple((float(t), float(v)) for t, v in points)
        object.__setattr__(self, "schedules", normalized)

    def build_input_schedules(self) -> dict[str, Callable[[float], float]]:
        schedules = dict(cfg.DEFAULT_INPUT_SCHEDULES)
        for name, points in self.schedules.items():
            schedules[name] = _piecewise_constant(points)
        return schedules

    def has_drug_input(self) -> bool:
        return any(abs(value) > 1e-12 for key in ("u_C", "u_P") for _time, value in self.schedules.get(key, ()))


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
        cfg.require(self.condition != "", "FlowRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "FlowRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid flow state {self.state}.")
        cfg.require(self.count is not None or self.fraction is not None, "FlowRecord requires count or fraction.")
        if self.count is not None:
            cfg.require(self.count >= 0, "FlowRecord.count must be non-negative.")
        if self.fraction is not None:
            cfg.require(0.0 <= self.fraction <= 1.0, "FlowRecord.fraction must be in [0, 1].")


@dataclass(frozen=True)
class CountRecord:
    condition: str
    week: int
    value: float
    replicate_id: str | None = None
    gate: str | None = None

    def __post_init__(self) -> None:
        cfg.require(self.condition != "", "CountRecord.condition must be non-empty.")
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
        cfg.require(self.condition != "", "QPCDRRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "QPCDRRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid qPCDR state {self.state}.")
        cfg.require(self.species in cfg.SPECIES, f"Invalid qPCDR species {self.species}.")
        cfg.require(np.isfinite(self.value), "QPCDRRecord.value must be finite.")
        cfg.require(self.value_scale in SUPPORTED_QPCDR_SCALES, f"Unsupported qPCDR scale {self.value_scale}.")


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
        cfg.require(self.condition != "", "EcTAGRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "EcTAGRecord.week must be at least 1.")
        cfg.require(self.state in cfg.STATE_NAMES, f"Invalid ecTAG state {self.state}.")
        cfg.require(self.species in cfg.SPECIES, f"Invalid ecTAG species {self.species}.")
        cfg.require(self.cell_id != "", "EcTAGRecord.cell_id must be non-empty.")
        cfg.require(self.value >= 0, "EcTAGRecord.value must be non-negative.")


@dataclass(frozen=True)
class DDPCRRecord:
    condition: str
    week: int
    species: str
    value: float
    lower: float | None = None
    upper: float | None = None
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        cfg.require(self.condition != "", "DDPCRRecord.condition must be non-empty.")
        cfg.require(self.week >= WEEK1, "DDPCRRecord.week must be at least 1.")
        cfg.require(self.species in cfg.SPECIES, f"Invalid ddPCR species {self.species}.")
        cfg.require(np.isfinite(self.value) and self.value >= 0.0, "DDPCRRecord.value must be finite and non-negative.")


@dataclass(frozen=True)
class CanonicalFitDataset:
    conditions: dict[str, ConditionSpec]
    flow: tuple[FlowRecord, ...] = ()
    counts: tuple[CountRecord, ...] = ()
    qpcdr: tuple[QPCDRRecord, ...] = ()
    ectag: tuple[EcTAGRecord, ...] = ()
    ddpcr: tuple[DDPCRRecord, ...] = ()
    week1_copy_distributions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    ectag_hist_max: int | None = None
    purity_matrix: np.ndarray | None = None
    purity_sensitivity: tuple[np.ndarray, ...] = ()
    qpcdr_calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    olig2_initial_ratio: float | None = None

    def __post_init__(self) -> None:
        cfg.require(bool(self.conditions), "CanonicalFitDataset requires at least one condition.")
        object.__setattr__(self, "flow", tuple(self.flow))
        object.__setattr__(self, "counts", tuple(self.counts))
        object.__setattr__(self, "qpcdr", tuple(self.qpcdr))
        object.__setattr__(self, "ectag", tuple(self.ectag))
        object.__setattr__(self, "ddpcr", tuple(self.ddpcr))
        object.__setattr__(self, "week1_copy_distributions", self._normalize_week1(self.week1_copy_distributions))
        if self.ectag_hist_max is not None:
            maximum = int(self.ectag_hist_max)
            cfg.require(maximum >= 0, "ectag_hist_max must be non-negative.")
            object.__setattr__(self, "ectag_hist_max", maximum)
        if self.purity_matrix is not None:
            object.__setattr__(self, "purity_matrix", self._normalize_purity(self.purity_matrix))
        object.__setattr__(self, "purity_sensitivity", tuple(self._normalize_purity(m) for m in self.purity_sensitivity))
        if self.olig2_initial_ratio is not None:
            cfg.require(np.isfinite(self.olig2_initial_ratio) and self.olig2_initial_ratio > 0.0, "OLIG2 initial ratio must be positive when provided.")
        self.validate()

    @staticmethod
    def _week1_matrix_from_marginals(payload: Mapping[str, Sequence[int] | np.ndarray]) -> np.ndarray:
        cfg.require(set(payload) == set(cfg.SPECIES), "species-specific week1 marginals must cover all ecDNA species.")
        columns: list[np.ndarray] = []
        n_rows = 0
        for species in cfg.SPECIES:
            values = np.asarray(payload[species], dtype=int).reshape(-1)
            cfg.require(values.size > 0, f"week1 marginal for {species} must be non-empty.")
            cfg.require(np.all(values >= 0), f"week1 marginal for {species} must be non-negative.")
            columns.append(values)
            n_rows = max(n_rows, int(values.size))
        return np.column_stack([np.resize(values, n_rows) for values in columns]).astype(int, copy=False)

    @staticmethod
    def _validate_week1_matrix(matrix: np.ndarray, *, label: str) -> np.ndarray:
        array = np.asarray(matrix, dtype=int)
        cfg.require(array.ndim == 2 and array.shape[1] == cfg.N_SPECIES, f"{label} week1 copy matrix must be n x species.")
        cfg.require(array.shape[0] > 0, f"{label} week1 copy matrix must be non-empty.")
        cfg.require(np.all(array >= 0), f"{label} week1 copy matrix must be non-negative.")
        return array.copy()

    @staticmethod
    def _normalize_week1(payload: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, np.ndarray]]:
        normalized: dict[str, dict[str, np.ndarray]] = {}
        for condition, by_state in payload.items():
            normalized[condition] = {}
            for state_name, raw in by_state.items():
                cfg.require(state_name in cfg.STATE_NAMES, f"Invalid week1 state {state_name}.")
                if isinstance(raw, Mapping):
                    normalized[condition][state_name] = CanonicalFitDataset._week1_matrix_from_marginals(raw)
                else:
                    normalized[condition][state_name] = CanonicalFitDataset._validate_week1_matrix(np.asarray(raw, dtype=int), label=f"{condition}/{state_name}")
        return normalized

    @staticmethod
    def _normalize_purity(matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        cfg.require(values.shape == (cfg.N_STATES, cfg.N_STATES), "purity matrix has invalid shape.")
        cfg.require(np.all(np.isfinite(values)) and np.all(values >= 0.0), "purity matrix must be finite and non-negative.")
        totals = np.sum(values, axis=0)
        cfg.require(np.all(totals > 0.0), "purity matrix columns must have positive mass.")
        return values / totals

    def validate(self) -> None:
        names = set(self.conditions)
        for label, rows in (("flow", self.flow), ("counts", self.counts), ("qpcdr", self.qpcdr), ("ectag", self.ectag), ("ddpcr", self.ddpcr)):
            for row in rows:
                cfg.require(getattr(row, "condition") in names, f"{label} record references unknown condition.")
        scales = {row.value_scale for row in self.qpcdr}
        cfg.require(len(scales) <= 1, "qPCDR records must use one scale per dataset.")
        cfg.require(bool(self.dynamic_weeks()), "CanonicalFitDataset requires dynamic data after week1.")
        for condition in self.conditions:
            init_condition = self.resolve_initialization_condition(condition)
            self._validate_week1_flow(init_condition)
            self._validate_week1_copy(init_condition)

    def condition_names(self) -> tuple[str, ...]:
        return tuple(self.conditions)

    def dynamic_weeks(self) -> tuple[int, ...]:
        weeks = {
            row.week
            for collection in (self.flow, self.counts, self.qpcdr, self.ectag, self.ddpcr)
            for row in collection
            if row.week > WEEK1
        }
        return tuple(sorted(weeks))

    def qpcdr_scale(self) -> str:
        return self.qpcdr[0].value_scale if self.qpcdr else DEFAULT_QPCDR_SCALE

    def qpcdr_batches(self) -> tuple[str, ...]:
        return tuple(sorted({row.batch for row in self.qpcdr})) if self.qpcdr else (DEFAULT_QPCDR_BATCH,)

    def ectag_upper_bound(self) -> int:
        values = [row.value for row in self.ectag]
        if self.ectag_hist_max is not None:
            values.append(int(self.ectag_hist_max))
        return max(values) if values else 0

    def resolve_initialization_condition(self, condition: str) -> str:
        cfg.require(condition in self.conditions, f"Unknown condition {condition}.")
        source = self.conditions[condition].initialization_source or condition
        cfg.require(source in self.conditions, f"Unknown initialization source {source}.")
        return source

    def _validate_week1_flow(self, condition: str) -> None:
        states = {row.state for row in self.flow if row.condition == condition and row.week == WEEK1}
        cfg.require(set(cfg.STATE_NAMES).issubset(states), f"week1 flow must cover every state for {condition}.")

    def _validate_week1_copy(self, condition: str) -> None:
        if condition in self.week1_copy_distributions:
            cfg.require(set(self.week1_copy_distributions[condition]) == set(cfg.STATE_NAMES), "week1 copy distributions must cover all states.")
            return
        grouped = self._week1_ectag_grouped(condition)
        cfg.require(set(grouped) == set(cfg.STATE_NAMES), f"week1 ecTAG must cover every state for {condition}.")
        for state_name, cells in grouped.items():
            cfg.require(bool(cells), f"week1 ecTAG has no cells for {condition}/{state_name}.")
            species_present = {species for species_values in cells.values() for species in species_values}
            missing = set(cfg.SPECIES) - species_present
            cfg.require(not missing, f"week1 ecTAG for {condition}/{state_name} is missing species marginals {sorted(missing)}.")

    def _week1_ectag_grouped(self, condition: str) -> dict[str, dict[str, dict[str, int]]]:
        grouped: dict[str, dict[str, dict[str, int]]] = {}
        for row in self.ectag:
            if row.condition == condition and row.week == WEEK1:
                key = self._cell_key(row)
                grouped.setdefault(row.state, {}).setdefault(key, {})[row.species] = int(row.value)
        return grouped

    @staticmethod
    def _cell_key(row: EcTAGRecord) -> str:
        return f"{row.replicate_id or ''}|{row.cell_id}"

    def build_empirical_initialization(self, condition: str, *, template: cfg.InitializationParameters | None = None) -> cfg.InitializationParameters:
        source = self.resolve_initialization_condition(condition)
        base = cfg.DEFAULT_INITIALIZATION_PARAMETERS if template is None else template
        flow = self._week1_flow_fractions(source)
        copies = self._week1_copy_distributions(source)
        init = cfg.InitializationParameters(
            mode=cfg.EMPIRICAL_WEEK1,
            parametric_copy_number_mean=np.asarray(base.parametric_copy_number_mean, dtype=float).copy(),
            parametric_state_dirichlet_alpha=np.asarray(base.parametric_state_dirichlet_alpha, dtype=float).copy(),
            cycle_probabilities=np.asarray(base.cycle_probabilities, dtype=float).copy(),
            age_scale=float(base.age_scale),
            empirical_flow_fractions=flow,
            empirical_sorted_copy_distributions=copies,
            empirical_soft_state_concentration=float(base.empirical_soft_state_concentration),
        )
        cfg.validate_initialization_parameters(init)
        return init

    def _week1_flow_fractions(self, condition: str) -> np.ndarray:
        rows = [row for row in self.flow if row.condition == condition and row.week == WEEK1]
        totals = np.zeros(cfg.N_STATES, dtype=float)
        uses_event_counts = any(row.count is not None or (row.fraction is not None and row.total_events is not None) for row in rows)
        if any(row.count is not None for row in rows):
            for row in rows:
                if row.count is not None:
                    totals[cfg.STATE_INDEX[row.state]] += float(row.count)
                elif row.fraction is not None and row.total_events is not None:
                    totals[cfg.STATE_INDEX[row.state]] += float(row.fraction * row.total_events)
        else:
            for row in rows:
                totals[cfg.STATE_INDEX[row.state]] += float(row.fraction or 0.0)
        total = float(np.sum(totals))
        cfg.require(total > 0.0, f"week1 flow total must be positive for {condition}.")
        olig2_prior = self._olig2_dirichlet_prior()
        if olig2_prior is not None:
            if not uses_event_counts:
                totals = totals * float(np.sum(olig2_prior))
            totals = totals + olig2_prior
            total = float(np.sum(totals))
        fractions = totals / total
        cfg.validate_probability_vector(fractions, name="week1 flow fractions", expected_shape=(cfg.N_STATES,))
        return fractions

    def _olig2_dirichlet_prior(self, *, strength: float = 40.0) -> np.ndarray | None:
        if self.olig2_initial_ratio is None:
            return None
        ratio = max(float(self.olig2_initial_ratio), 1e-8)
        high_mass = ratio / (1.0 + ratio)
        low_mass = 1.0 / (1.0 + ratio)
        npc_opc_split = np.array([0.34, 0.32], dtype=float)
        ac_mes_split = np.array([0.17, 0.16], dtype=float)
        fractions = np.array(
            [
                high_mass * npc_opc_split[0] / float(np.sum(npc_opc_split)),
                high_mass * npc_opc_split[1] / float(np.sum(npc_opc_split)),
                low_mass * ac_mes_split[0] / float(np.sum(ac_mes_split)),
                low_mass * ac_mes_split[1] / float(np.sum(ac_mes_split)),
            ],
            dtype=float,
        )
        return float(strength) * fractions

    def _week1_copy_distributions(self, condition: str) -> dict[str, np.ndarray]:
        if condition in self.week1_copy_distributions:
            return {state: matrix.copy() for state, matrix in self.week1_copy_distributions[condition].items()}
        grouped = self._week1_ectag_grouped(condition)
        matrices: dict[str, np.ndarray] = {}
        for state_name in cfg.STATE_NAMES:
            cells = grouped[state_name]
            complete_rows = [
                [species_values[species] for species in cfg.SPECIES]
                for _cell, species_values in sorted(cells.items())
                if set(species_values) == set(cfg.SPECIES)
            ]
            if complete_rows and len(complete_rows) == len(cells):
                matrices[state_name] = np.asarray(complete_rows, dtype=int)
                continue
            marginals = {
                species: np.asarray(
                    [species_values[species] for _cell, species_values in sorted(cells.items()) if species in species_values],
                    dtype=int,
                )
                for species in cfg.SPECIES
            }
            matrices[state_name] = self._week1_matrix_from_marginals(marginals)
        return matrices

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CanonicalFitDataset":
        manifest = _resolve_path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        base = manifest.parent
        ectag_hist_max = _manifest_ectag_hist_max(payload)
        conditions = {
            name: ConditionSpec(name, spec.get("schedules", {}), spec.get("initialization_source"))
            for name, spec in payload["conditions"].items()
        }
        files = payload.get("files", {})
        week1: dict[str, dict[str, np.ndarray]] = {}
        if files.get("week1_copy_distributions"):
            raw = json.loads(_resolve_path(files["week1_copy_distributions"], base_dir=base).read_text(encoding="utf-8"))
            week1 = {condition: dict(by_state) for condition, by_state in raw.items()}
        return cls(
            conditions=conditions,
            flow=load_flow_csv(_resolve_path(files["flow"], base_dir=base)) if files.get("flow") else (),
            counts=load_count_csv(_resolve_path(files["counts"], base_dir=base)) if files.get("counts") else (),
            qpcdr=load_qpcdr_csv(_resolve_path(files["qpcdr"], base_dir=base)) if files.get("qpcdr") else (),
            ectag=load_ectag_csv(_resolve_path(files["ectag"], base_dir=base)) if files.get("ectag") else (),
            ddpcr=load_ddpcr_csv(_resolve_path(files["ddpcr"], base_dir=base)) if files.get("ddpcr") else (),
            week1_copy_distributions=week1,
            ectag_hist_max=ectag_hist_max,
            purity_matrix=None if payload.get("purity_matrix") is None else np.asarray(payload["purity_matrix"], dtype=float),
            purity_sensitivity=tuple(np.asarray(m, dtype=float) for m in payload.get("purity_sensitivity", ())),
            qpcdr_calibration=payload.get("qpcdr_calibration", {}),
            olig2_initial_ratio=payload.get("olig2_initial_ratio"),
        )

    @classmethod
    def from_simulation_runs(
        cls,
        runs_by_condition: Mapping[str, Sequence[object] | object],
        *,
        conditions: Mapping[str, ConditionSpec] | None = None,
        qpcdr_value_scale: str = DEFAULT_QPCDR_SCALE,
        ectag_hist_max: int | None = None,
    ) -> "CanonicalFitDataset":
        normalized: dict[str, tuple[object, ...]] = {}
        for condition, payload in runs_by_condition.items():
            if isinstance(payload, tuple) or isinstance(payload, list):
                normalized[condition] = tuple(payload)
            else:
                normalized[condition] = (payload,)
        specs = {name: ConditionSpec(name) for name in normalized} if conditions is None else dict(conditions)
        flow: list[FlowRecord] = []
        counts: list[CountRecord] = []
        qpcdr: list[QPCDRRecord] = []
        ectag: list[EcTAGRecord] = []
        week1: dict[str, dict[str, list[list[int]]]] = {}
        for condition, runs in normalized.items():
            for rep_idx, result in enumerate(runs):
                replicate = f"sim{rep_idx}"
                observations = getattr(result, "observations", ())
                snapshots = getattr(result, "cell_snapshots", ())
                for time_idx, time_value in enumerate(getattr(result, "times", ())):
                    week = int(round(float(time_value))) + 1
                    obs = observations[time_idx] if time_idx < len(observations) else {}
                    if obs:
                        for state_idx, state in enumerate(cfg.STATE_NAMES):
                            flow.append(FlowRecord(condition, week, state, int(obs["flow_counts"][state_idx]), float(obs["flow_fractions"][state_idx]), int(sum(obs["flow_counts"])), replicate))
                        counts.append(CountRecord(condition, week, float(obs.get("observed_count", sum(obs["flow_counts"]))), replicate))
                        for state in cfg.STATE_NAMES:
                            for species in cfg.SPECIES:
                                for value in obs.get("sorted_qpcdr", {}).get("values", {}).get(state, {}).get(species, ()):
                                    qpcdr.append(QPCDRRecord(condition, week, state, species, float(value), replicate, value_scale=qpcdr_value_scale))
                                for idx, value in enumerate(obs.get("sorted_ecTAG", {}).get("values", {}).get(state, {}).get(species, ())):
                                    ectag.append(EcTAGRecord(condition, week, state, species, f"{state}|cell{idx}", int(value), replicate))
                    elif time_idx < len(snapshots):
                        cells = snapshots[time_idx]
                        hard_states = [int(np.argmax(cell["soft_state"])) for cell in cells]
                        for state_idx, state in enumerate(cfg.STATE_NAMES):
                            state_cells = [cell for cell, hard in zip(cells, hard_states) if hard == state_idx]
                            flow.append(FlowRecord(condition, week, state, len(state_cells), len(state_cells) / max(len(cells), 1), len(cells), replicate))
                            if week == WEEK1:
                                week1.setdefault(condition, {}).setdefault(state, []).extend(np.asarray([cell["copy_numbers"] for cell in state_cells], dtype=int).tolist())
        week1_arrays = {condition: {state: np.asarray(rows or [[0, 0, 0]], dtype=int) for state, rows in by_state.items()} for condition, by_state in week1.items()}
        for condition in specs:
            week1_arrays.setdefault(condition, {state: np.asarray([[0, 0, 0]], dtype=int) for state in cfg.STATE_NAMES})
            for state in cfg.STATE_NAMES:
                week1_arrays[condition].setdefault(state, np.asarray([[0, 0, 0]], dtype=int))
        return cls(specs, tuple(flow), tuple(counts), tuple(qpcdr), tuple(ectag), (), week1_arrays, ectag_hist_max)


def load_flow_csv(path: str | Path) -> tuple[FlowRecord, ...]:
    rows: list[FlowRecord] = []
    with open(_resolve_path(path), "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(FlowRecord(raw["condition"], int(raw["week"]), raw["state"], _int_or_none(raw.get("count")), _float_or_none(raw.get("fraction")), _int_or_none(raw.get("total_events")), raw.get("replicate_id") or None))
    return tuple(rows)


def load_count_csv(path: str | Path) -> tuple[CountRecord, ...]:
    rows: list[CountRecord] = []
    with open(_resolve_path(path), "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(CountRecord(raw["condition"], int(raw["week"]), float(raw["count"]), raw.get("replicate_id") or None, raw.get("gate") or None))
    return tuple(rows)


def load_qpcdr_csv(path: str | Path) -> tuple[QPCDRRecord, ...]:
    rows: list[QPCDRRecord] = []
    with open(_resolve_path(path), "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(QPCDRRecord(raw["condition"], int(raw["week"]), raw["state"], raw["species"], float(raw["value"]), raw.get("replicate_id") or None, raw.get("batch") or DEFAULT_QPCDR_BATCH, raw.get("value_scale") or DEFAULT_QPCDR_SCALE))
    return tuple(rows)


def load_ectag_csv(path: str | Path) -> tuple[EcTAGRecord, ...]:
    rows: list[EcTAGRecord] = []
    with open(_resolve_path(path), "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(EcTAGRecord(raw["condition"], int(raw["week"]), raw["state"], raw["species"], raw["cell_id"], int(float(raw["value"])), raw.get("replicate_id") or None))
    return tuple(rows)


def load_ddpcr_csv(path: str | Path) -> tuple[DDPCRRecord, ...]:
    rows: list[DDPCRRecord] = []
    with open(_resolve_path(path), "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("value") is not None:
                value = raw["value"]
            else:
                value = raw.get("Ratio") or raw.get("ddPCR_copy_number")
            if value is None or str(value).upper() == "NA" or not str(value).strip():
                continue
            species = raw.get("species") or raw.get("Target") or ""
            species = species.replace("ecMyc", "MYC").replace("ecCDK4", "CDK4").replace("ecPDGFRA", "PDGFRA")
            condition = raw.get("condition") or raw.get("treatment") or "unknown"
            week = raw.get("week") or raw.get("day")
            rows.append(DDPCRRecord(condition, int(float(week)), species, float(value), _float_or_none(raw.get("lower") or raw.get("PoissonRatioMin")), _float_or_none(raw.get("upper") or raw.get("PoissonRatioMax")), raw.get("replicate_id") or raw.get("Well")))
    return tuple(rows)


def build_raw_data_qc_report(dataset: CanonicalFitDataset) -> dict[str, object]:
    ectag_cells: dict[str, set[str]] = {}
    for row in dataset.ectag:
        key = f"{row.condition}|week{row.week}|state={row.state}"
        ectag_cells.setdefault(key, set()).add(f"{row.replicate_id or ''}|{row.cell_id}")
    low_ectag = [
        {"snapshot": key, "n_cells": len(cells), "policy": "mean-level only below 50 cells"}
        for key, cells in sorted(ectag_cells.items())
        if len(cells) < 50
    ]
    ddpcr_pairs = {(row.week, row.species) for row in dataset.ddpcr}
    qpcdr_pairs = {(row.week, row.state, row.species) for row in dataset.qpcdr}
    flow_pairs = {(row.week, row.state) for row in dataset.flow}
    has_required_modalities = bool(dataset.flow and dataset.qpcdr and dataset.ectag)
    return {
        "conditions": dataset.condition_names(),
        "dynamic_weeks": dataset.dynamic_weeks(),
        "row_counts": {
            "flow": len(dataset.flow),
            "count": len(dataset.counts),
            "qpcdr": len(dataset.qpcdr),
            "ectag": len(dataset.ectag),
            "ddpcr": len(dataset.ddpcr),
        },
        "coverage": {
            "flow_week_state_pairs": len(flow_pairs),
            "qpcdr_week_state_species_pairs": len(qpcdr_pairs),
            "ddpcr_week_species_pairs": len(ddpcr_pairs),
            "ectag_week_state_snapshots": len(ectag_cells),
        },
        "criteria": {
            "flow_present": bool(dataset.flow),
            "qpcdr_present": bool(dataset.qpcdr),
            "ectag_present": bool(dataset.ectag),
            "ddpcr_optional_bulk_anchor_present": bool(dataset.ddpcr),
            "ddpcr_four_timepoints_three_species": len(ddpcr_pairs) >= 12,
            "has_real_fit_modalities": has_required_modalities,
            "olig2_initial_ratio_present": dataset.olig2_initial_ratio is not None,
        },
        "low_ectag_histogram_cell_snapshots": low_ectag,
        "policies": {
            "ddpcr": "bulk pooled mean anchor only",
            "ectag": "species-specific single-cell histogram; no config.py max-observed censoring assumption",
            "low_ectag_cells": "snapshots with fewer than 50 cells should be down-weighted or inspected as mean-level evidence",
            "olig2": "optional weak initial state prior on (NPC+OPC)/(AC+MES)",
        },
        "olig2_initial_ratio": dataset.olig2_initial_ratio,
    }


def _flow_rows(dataset: CanonicalFitDataset) -> list[dict[str, object]]:
    return [
        {
            "condition": row.condition,
            "week": row.week,
            "state": row.state,
            "count": row.count,
            "fraction": row.fraction,
            "total_events": row.total_events,
            "replicate_id": row.replicate_id,
        }
        for row in dataset.flow
    ]


def _count_rows(dataset: CanonicalFitDataset) -> list[dict[str, object]]:
    return [
        {
            "condition": row.condition,
            "week": row.week,
            "count": row.value,
            "gate": row.gate,
            "replicate_id": row.replicate_id,
        }
        for row in dataset.counts
    ]


def _qpcdr_rows(dataset: CanonicalFitDataset) -> list[dict[str, object]]:
    return [
        {
            "condition": row.condition,
            "week": row.week,
            "state": row.state,
            "species": row.species,
            "value": row.value,
            "value_scale": row.value_scale,
            "batch": row.batch,
            "replicate_id": row.replicate_id,
        }
        for row in dataset.qpcdr
    ]


def _ectag_rows(dataset: CanonicalFitDataset) -> list[dict[str, object]]:
    return [
        {
            "condition": row.condition,
            "week": row.week,
            "state": row.state,
            "species": row.species,
            "cell_id": row.cell_id,
            "ecTAG_count": row.value,
            "replicate_id": row.replicate_id,
        }
        for row in dataset.ectag
    ]


def _ddpcr_rows(dataset: CanonicalFitDataset) -> list[dict[str, object]]:
    return [
        {
            "condition": row.condition,
            "week": row.week,
            "species": row.species,
            "ddPCR_copy_number": row.value,
            "lower": row.lower,
            "upper": row.upper,
            "replicate_id": row.replicate_id,
        }
        for row in dataset.ddpcr
    ]


def write_standardized_dataset(output_dir: str | Path, dataset: CanonicalFitDataset) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "flow_long": write_table_bundle(destination, "flow_long", _flow_rows(dataset)),
        "count_long": write_table_bundle(destination, "count_long", _count_rows(dataset)),
        "qpcdr_long": write_table_bundle(destination, "qpcdr_long", _qpcdr_rows(dataset)),
        "ectag_cell_long": write_table_bundle(destination, "ectag_cell_long", _ectag_rows(dataset)),
        "ddpcr_long": write_table_bundle(destination, "ddpcr_long", _ddpcr_rows(dataset)),
    }
    qc = build_raw_data_qc_report(dataset)
    write_json(destination / "raw_data_qc_report.json", qc)
    markdown = [
        "# Raw Data QC Report",
        "",
        f"- Conditions: {', '.join(dataset.condition_names())}",
        f"- Dynamic weeks: {list(dataset.dynamic_weeks())}",
        f"- Flow rows: {len(dataset.flow)}",
        f"- qPCDR rows: {len(dataset.qpcdr)}",
        f"- ecTAG rows: {len(dataset.ectag)}",
        f"- ddPCR rows: {len(dataset.ddpcr)}",
        "- ddPCR policy: bulk pooled mean anchor only.",
        "- ecTAG policy: species-specific histograms; no config.py max-observed censoring assumption.",
    ]
    if qc["low_ectag_histogram_cell_snapshots"]:
        markdown.append("- Low ecTAG cell snapshots are listed in raw_data_qc_report.json.")
    (destination / "raw_data_qc_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    index = {
        "standardized_tables": outputs,
        "qc_report": str(destination / "raw_data_qc_report.json"),
        "qc_report_markdown": str(destination / "raw_data_qc_report.md"),
        "ready_for_real_fit": bool(qc["criteria"]["has_real_fit_modalities"]),
    }
    write_json(destination / "analysis_index.json", index)
    return index
